"""
End-to-End HPO — 가변적 파이프라인 + 단일 Optuna objective

pipeline_config로 모든 스위치를 제어한다:
- input_level: 'die' | 'unit'
- run_clf / clf_output / clf_filter / clf_optuna
- run_fs / fs_optuna
- reg_level / reg_optuna

optuna=OFF인 단계는 1번만 실행하고 캐싱하여 trial당 시간을 절약한다.

확장 (LGBM-only baseline):
- run_e2e_optimization_with_pp / rerun_best_trial_with_pp
  → 전처리(cleaning + outlier + 집계 preset)까지 Optuna trial 안에서
    동시 탐색. 같은 전처리 파라미터 조합은 LRU 캐시로 재사용.
"""
import hashlib
import json
import sys
import warnings
from collections import OrderedDict
from datetime import datetime

import os
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, recall_score, precision_score,
)

from utils.config import (
    SEED, TARGET_COL, KEY_COL, POSITION_COL, PROJECT_ROOT,
)
from utils.evaluate import rmse, postprocess

# ── run_cleaning / run_outlier_treatment 지연 import 준비 ──
# 2_preprocessing은 python package가 아니므로 sys.path에 경로를 추가한 뒤
# 함수 호출 시점에 import 한다 (모듈 로드 시 불필요한 의존성을 피함).
import os as _os
_PP_PATH = _os.path.join(PROJECT_ROOT, "2_preprocessing")
if _PP_PATH not in sys.path:
    sys.path.insert(0, _PP_PATH)
from .model_zoo import create_model, fit_model, supports_early_stopping, get_default_params, DEVICE
from .search_space import (
    SEARCH_SPACES,
    preprocessing_space,
    split_pp_params,
    extract_pp_params_from_best,
    AGG_PRESETS,
)
from .aggregate import aggregate_die_to_unit
from .feature_select import select_top_k, remove_zero_variance


# ═════════════════════════════════════════════════════════════
# 기본 pipeline_config
# ═════════════════════════════════════════════════════════════
DEFAULT_CONFIG = dict(
    # 인풋
    input_level="die",          # 'die' | 'unit'

    # 분류
    run_clf=True,               # False → 분류 스킵
    clf_output="proba",         # 'proba' | 'binary'
    clf_filter=False,           # True → 0 예측 샘플 회귀 제외
    clf_optuna=True,            # False → 기본 파라미터

    # 피처 선택
    run_fs=True,                # False → FS 스킵
    fs_optuna=True,             # False → 고정 top_k

    # 회귀
    reg_level="unit",           # 'unit' | 'position'
    reg_optuna=True,            # False → 기본 파라미터
)


def _merge_config(user_config):
    """기본값에 사용자 설정을 덮어쓰기"""
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(user_config or {})

    # input_level='unit'이면 분류 자동 OFF
    if cfg["input_level"] == "unit":
        cfg["run_clf"] = False
        cfg["clf_optuna"] = False

    # 분류 OFF면 clf 관련 옵션 무시
    if not cfg["run_clf"]:
        cfg["clf_output"] = None
        cfg["clf_filter"] = False
        cfg["clf_optuna"] = False

    return cfg


# ═════════════════════════════════════════════════════════════
# 경량 분류 OOF (silent)
# ═════════════════════════════════════════════════════════════
def _run_clf_oof(pos_data, feat_cols, clf_params, model_name,
                 n_folds, early_stop, label_col, imbalance_method,
                 clf_output="proba"):
    """
    Position별 분류 OOF

    Parameters
    ----------
    clf_output : str
        'proba' → P(Y>0) 확률, 'binary' → 0/1 예측

    Returns
    -------
    clf_result : dict
        {position: {"train_proba": arr, "val_proba": arr, "test_proba": arr}}
    """
    clf_result = {}

    for pos in sorted(pos_data.keys()):
        d = pos_data[pos]
        X_tr = d["train"][feat_cols].values
        y_tr = d["train"][label_col].values
        X_val = d["val"][feat_cols].values
        X_test = d["test"][feat_cols].values

        # 불균형 처리
        actual_params = clf_params.copy()
        if imbalance_method == "scale_pos_weight":
            n_neg = (y_tr == 0).sum()
            n_pos = (y_tr == 1).sum()
            spw = n_neg / n_pos if n_pos > 0 else 1.0
            if model_name in ("lgbm", "xgb"):
                actual_params["scale_pos_weight"] = spw
            elif model_name == "catboost":
                actual_params["auto_class_weights"] = "Balanced"
            elif model_name in ("rf", "et"):
                actual_params["class_weight"] = "balanced"

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        oof_proba = np.zeros(len(X_tr))
        fold_models = []

        for tr_idx, val_idx in skf.split(X_tr, y_tr):
            clf = create_model(model_name, "clf", actual_params)
            if supports_early_stopping(model_name):
                # ES용 inner holdout을 tr_idx 내부에서 분리 (val_idx 누수 방지)
                X_tr_fold = X_tr[tr_idx]
                y_tr_fold = y_tr[tr_idx]
                if len(np.unique(y_tr_fold)) >= 2:
                    inner_tr, inner_es = train_test_split(
                        np.arange(len(tr_idx)),
                        test_size=0.15,
                        random_state=SEED,
                        stratify=y_tr_fold,
                    )
                    fit_model(
                        clf,
                        X_tr_fold[inner_tr], y_tr_fold[inner_tr],
                        X_tr_fold[inner_es], y_tr_fold[inner_es],
                        early_stop,
                    )
                else:
                    # stratify 불가 (단일 클래스) → ES 없이 학습
                    fit_model(clf, X_tr_fold, y_tr_fold)
            else:
                fit_model(clf, X_tr[tr_idx], y_tr[tr_idx])

            oof_proba[val_idx] = clf.predict_proba(X_tr[val_idx])[:, 1]
            fold_models.append(clf)

        val_proba = np.mean(
            [m.predict_proba(X_val)[:, 1] for m in fold_models], axis=0
        )
        test_proba = np.mean(
            [m.predict_proba(X_test)[:, 1] for m in fold_models], axis=0
        )

        # binary 모드: 확률 → 0/1
        if clf_output == "binary":
            oof_proba = (oof_proba > 0.5).astype(float)
            val_proba = (val_proba > 0.5).astype(float)
            test_proba = (test_proba > 0.5).astype(float)

        clf_result[pos] = {
            "train_proba": oof_proba,
            "val_proba": val_proba,
            "test_proba": test_proba,
        }

    return clf_result


# ═════════════════════════════════════════════════════════════
# 단일 모델 분류 (rerun mode='single')
# ═════════════════════════════════════════════════════════════
def _run_clf_single(pos_data, feat_cols, clf_params, model_name,
                    early_stop, label_col, imbalance_method,
                    clf_output="proba", es_holdout=0.1):
    """
    Position별 분류 단일 학습 (KFold 없음, train 100% 사용)

    - ES용 holdout만 train에서 내부 분리
    - train 예측은 in-sample (진단용)
    - val/test는 단일 모델 예측
    """
    clf_result = {}

    for pos in sorted(pos_data.keys()):
        d = pos_data[pos]
        X_tr_full = d["train"][feat_cols].values
        y_tr_full = d["train"][label_col].values
        X_val = d["val"][feat_cols].values
        X_test = d["test"][feat_cols].values

        # 불균형 처리
        actual_params = clf_params.copy()
        if imbalance_method == "scale_pos_weight":
            n_neg = (y_tr_full == 0).sum()
            n_pos = (y_tr_full == 1).sum()
            spw = n_neg / n_pos if n_pos > 0 else 1.0
            if model_name in ("lgbm", "xgb"):
                actual_params["scale_pos_weight"] = spw
            elif model_name == "catboost":
                actual_params["auto_class_weights"] = "Balanced"
            elif model_name in ("rf", "et"):
                actual_params["class_weight"] = "balanced"

        # ES용 holdout 분리 (stratified)
        use_es = supports_early_stopping(model_name) and es_holdout > 0
        if use_es and len(np.unique(y_tr_full)) >= 2:
            X_fit, X_es, y_fit, y_es = train_test_split(
                X_tr_full, y_tr_full,
                test_size=es_holdout,
                random_state=SEED,
                stratify=y_tr_full,
            )
        else:
            X_fit, y_fit = X_tr_full, y_tr_full
            X_es, y_es = None, None

        clf = create_model(model_name, "clf", actual_params)
        if use_es and X_es is not None:
            fit_model(clf, X_fit, y_fit, X_es, y_es, early_stop)
        else:
            fit_model(clf, X_fit, y_fit)

        train_proba = clf.predict_proba(X_tr_full)[:, 1]  # in-sample
        val_proba = clf.predict_proba(X_val)[:, 1]
        test_proba = clf.predict_proba(X_test)[:, 1]

        if clf_output == "binary":
            train_proba = (train_proba > 0.5).astype(float)
            val_proba = (val_proba > 0.5).astype(float)
            test_proba = (test_proba > 0.5).astype(float)

        clf_result[pos] = {
            "train_proba": train_proba,
            "val_proba": val_proba,
            "test_proba": test_proba,
        }

    return clf_result


# ═════════════════════════════════════════════════════════════
# 단일 모델 회귀 (rerun mode='single')
# ═════════════════════════════════════════════════════════════
def _run_reg_single(unit_data, feat_cols, reg_params, model_name,
                    early_stop, use_clf=True, clf_filter=False,
                    es_holdout=0.1):
    """
    Unit-level 회귀 단일 학습 (KFold 없음, train 100% 사용)
    """
    X_train = unit_data["train"][feat_cols].values
    y_train = unit_data["train"][TARGET_COL].values
    X_val = unit_data["val"][feat_cols].values
    X_test = unit_data["test"][feat_cols].values

    # 학습 샘플 선택
    if use_clf and not clf_filter:
        # two-stage: Y>0인 샘플만
        mask = y_train > 0
    elif clf_filter and "clf_proba_mean" in unit_data["train"].columns:
        clf_pred = unit_data["train"]["clf_proba_mean"].values
        mask = clf_pred > 0.5
    else:
        mask = np.ones(len(y_train), dtype=bool)

    reg = None  # fallback: 학습 실패 시 None
    if mask.sum() == 0:
        oof_reg = np.zeros(len(X_train))
        val_reg = np.zeros(len(X_val))
        test_reg = np.zeros(len(X_test))
    else:
        X_fit_pool = X_train[mask]
        y_fit_pool = y_train[mask]

        use_es = (
            supports_early_stopping(model_name)
            and es_holdout > 0
            and len(X_fit_pool) >= 10
        )
        if use_es:
            X_fit, X_es, y_fit, y_es = train_test_split(
                X_fit_pool, y_fit_pool,
                test_size=es_holdout,
                random_state=SEED,
            )
        else:
            X_fit, y_fit = X_fit_pool, y_fit_pool
            X_es, y_es = None, None

        reg = create_model(model_name, "reg", reg_params)
        if use_es and X_es is not None:
            fit_model(reg, X_fit, y_fit, X_es, y_es, early_stop)
        else:
            fit_model(reg, X_fit, y_fit)

        oof_reg = reg.predict(X_train)  # in-sample
        val_reg = reg.predict(X_val)
        test_reg = reg.predict(X_test)

    # two-stage 곱셈
    if use_clf and "clf_proba_mean" in unit_data["train"].columns:
        proba_train = unit_data["train"]["clf_proba_mean"].values
        proba_val = unit_data["val"]["clf_proba_mean"].values
        proba_test = unit_data["test"]["clf_proba_mean"].values
        oof_pred = postprocess(proba_train * oof_reg)
        val_pred = postprocess(proba_val * val_reg)
        test_pred = postprocess(proba_test * test_reg)
    else:
        oof_pred = postprocess(oof_reg)
        val_pred = postprocess(val_reg)
        test_pred = postprocess(test_reg)

    return {
        "val_pred": val_pred,
        "test_pred": test_pred,
        "train_pred_insample": oof_pred,  # in-sample 예측 (OOF 아님, 진단/val/test만 안전)
        "train_rmse": rmse(y_train, oof_pred),
        "model": reg,
    }


# ═════════════════════════════════════════════════════════════
# 경량 회귀 OOF (silent)
# ═════════════════════════════════════════════════════════════
def _run_reg_oof(unit_data, feat_cols, reg_params, model_name,
                 n_folds, early_stop, use_clf=True, clf_filter=False):
    """
    Unit-level 회귀 OOF

    Parameters
    ----------
    use_clf : bool
        True → clf_proba_mean을 곱해서 최종 예측 (two-stage)
        False → 단순 회귀
    clf_filter : bool
        True → clf가 0으로 예측한 샘플을 학습에서 제외
    """
    X_train = unit_data["train"][feat_cols].values
    y_train = unit_data["train"][TARGET_COL].values
    X_val = unit_data["val"][feat_cols].values
    X_test = unit_data["test"][feat_cols].values

    # clf_filter: 분류기가 0으로 예측한 샘플 제외
    if clf_filter and "clf_proba_mean" in unit_data["train"].columns:
        clf_pred = unit_data["train"]["clf_proba_mean"].values
        train_mask = clf_pred > 0.5  # binary: 1인 샘플만
    else:
        train_mask = None

    # Fallback: fold skip 시 사용할 보수적 예측값 (Y>0 평균)
    # two-stage에서 회귀 모델이 학습하는 분포가 y>0이므로 y>0 평균이 합리적
    pos_mask_all = y_train > 0
    fallback_val = float(y_train[pos_mask_all].mean()) if pos_mask_all.any() else 0.0

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof_reg = np.full(len(X_train), fallback_val, dtype=float)  # 0 대신 fallback 초기화
    fold_models = []
    skipped_folds = 0

    for tr_idx, va_idx in kf.split(X_train):
        # ─── 1단계: X_fit, y_fit 결정 ───
        if use_clf and not clf_filter:
            # two-stage: Y>0인 샘플만 학습
            pos_tr = y_train[tr_idx] > 0
            if pos_tr.sum() == 0:
                skipped_folds += 1
                continue
            X_fit, y_fit = X_train[tr_idx][pos_tr], y_train[tr_idx][pos_tr]
        elif clf_filter and train_mask is not None:
            # clf_filter: 분류기가 1로 예측한 샘플만 학습
            tr_keep = train_mask[tr_idx]
            if tr_keep.sum() == 0:
                skipped_folds += 1
                continue
            X_fit, y_fit = X_train[tr_idx][tr_keep], y_train[tr_idx][tr_keep]
        else:
            # 단순 회귀: 전체 학습
            X_fit, y_fit = X_train[tr_idx], y_train[tr_idx]

        # ─── 2단계: inner holdout 분리 (ES용, va_idx 누수 방지) ───
        reg = create_model(model_name, "reg", reg_params)
        if supports_early_stopping(model_name) and len(X_fit) >= 10:
            inner_tr, inner_es = train_test_split(
                np.arange(len(X_fit)),
                test_size=0.15,
                random_state=SEED,
            )
            fit_model(
                reg,
                X_fit[inner_tr], y_fit[inner_tr],
                X_fit[inner_es], y_fit[inner_es],
                early_stop,
            )
        else:
            fit_model(reg, X_fit, y_fit)

        oof_reg[va_idx] = reg.predict(X_train[va_idx])
        fold_models.append(reg)

    # fold skip 경고 (일부 fold만 스킵된 경우)
    if skipped_folds > 0:
        warnings.warn(
            f"[_run_reg_oof] {skipped_folds}/{n_folds} fold(s) skipped due to "
            f"empty training samples (pos_tr or tr_keep == 0). "
            f"OOF for skipped folds filled with fallback value "
            f"({fallback_val:.6f} = mean of y>0).",
            stacklevel=2,
        )

    if len(fold_models) == 0:
        warnings.warn(
            f"[_run_reg_oof] All {n_folds} folds skipped. "
            f"val/test predictions filled with fallback value ({fallback_val:.6f}).",
            stacklevel=2,
        )
        val_reg = np.full(len(X_val), fallback_val, dtype=float)
        test_reg = np.full(len(X_test), fallback_val, dtype=float)
    else:
        val_reg = np.mean([m.predict(X_val) for m in fold_models], axis=0)
        test_reg = np.mean([m.predict(X_test) for m in fold_models], axis=0)

    # two-stage: proba * reg_pred
    if use_clf and "clf_proba_mean" in unit_data["train"].columns:
        proba_train = unit_data["train"]["clf_proba_mean"].values
        proba_val = unit_data["val"]["clf_proba_mean"].values
        proba_test = unit_data["test"]["clf_proba_mean"].values
        oof_pred = postprocess(proba_train * oof_reg)
        val_pred = postprocess(proba_val * val_reg)
        test_pred = postprocess(proba_test * test_reg)
    else:
        oof_pred = postprocess(oof_reg)
        val_pred = postprocess(val_reg)
        test_pred = postprocess(test_reg)

    return {
        "val_pred": val_pred,
        "test_pred": test_pred,
        "oof_pred": oof_pred,
        "train_rmse": rmse(y_train, oof_pred),
        "fold_models": fold_models,
    }


# ═════════════════════════════════════════════════════════════
# Die 예측 → Unit 집계 (reg_level='position' 전용)
# ═════════════════════════════════════════════════════════════
def _die_pred_to_unit(unit_data, reg_result):
    """
    reg_level='position'에서 die-level 예측을 unit-level로 집계.

    train 예측 키는 'oof_pred' (kfold) 또는 'train_pred_insample' (single) 둘 중 하나.
    어느 쪽이든 자동 감지하여 동일 키로 반환한다.

    Returns
    -------
    agg_result : dict
        val_pred, test_pred, <train_key> (unit-level), train_rmse
    agg_unit_data : dict
        {"train": df, "val": df, "test": df} — KEY_COL + TARGET_COL만 포함
    """
    # train 키 자동 감지
    if "oof_pred" in reg_result:
        train_key = "oof_pred"
    elif "train_pred_insample" in reg_result:
        train_key = "train_pred_insample"
    else:
        raise KeyError(
            "reg_result에 'oof_pred' 또는 'train_pred_insample' 키가 없습니다."
        )

    agg_result = {}
    agg_unit_data = {}

    for split_name, pred_key in [
        ("train", train_key), ("val", "val_pred"), ("test", "test_pred"),
    ]:
        df = unit_data[split_name][[KEY_COL, TARGET_COL]].copy()
        df["_pred"] = reg_result[pred_key]
        grp = df.groupby(KEY_COL, sort=False).agg(
            {TARGET_COL: "first", "_pred": "mean"}
        ).reset_index()
        agg_result[pred_key] = grp["_pred"].values
        agg_unit_data[split_name] = grp[[KEY_COL, TARGET_COL]]

    y_train = agg_unit_data["train"][TARGET_COL].values
    agg_result["train_rmse"] = rmse(y_train, agg_result[train_key])

    return agg_result, agg_unit_data


# ═════════════════════════════════════════════════════════════
# 집계 or position 피처 처리
# ═════════════════════════════════════════════════════════════
def _prepare_unit_data(pos_data, feat_cols, clf_result, cfg, agg_funcs):
    """
    reg_level에 따라 unit 데이터 준비

    - 'unit': die→unit 집계 (기존 방식)
    - 'position': 집계 없이 position을 정수 피처로 포함
    """
    if cfg["reg_level"] == "position":
        # 집계 안 함 — position을 피처로 포함
        # NOTE: 이 모드에서 예측은 die 레벨. 최종 RMSE/CSV 출력 시
        #       unit별 평균 집계가 필요함 (rerun에서 처리)
        unit_data = {}
        for split_name in ["train", "val", "test"]:
            frames = []
            for pos in sorted(pos_data.keys()):
                df = pos_data[pos][split_name].copy()
                if clf_result is not None:
                    df["clf_proba_mean"] = clf_result[pos][f"{split_name}_proba"]
                frames.append(df)
            unit_data[split_name] = pd.concat(frames, ignore_index=True)

        # position을 피처에 포함 (clf_proba_mean은 곱셈용이므로 feature 제외)
        unit_feat_cols = feat_cols + [POSITION_COL]

        return unit_data, unit_feat_cols

    else:  # 'unit' — 기존 집계
        if clf_result is not None:
            return aggregate_die_to_unit(
                pos_data, feat_cols, clf_result, agg_funcs=agg_funcs,
            )
        else:
            # clf 없이 집계 — dummy proba (전부 1)
            dummy_clf = {}
            for pos in sorted(pos_data.keys()):
                n_tr = len(pos_data[pos]["train"])
                n_val = len(pos_data[pos]["val"])
                n_test = len(pos_data[pos]["test"])
                dummy_clf[pos] = {
                    "train_proba": np.ones(n_tr),
                    "val_proba": np.ones(n_val),
                    "test_proba": np.ones(n_test),
                }
            return aggregate_die_to_unit(
                pos_data, feat_cols, dummy_clf, agg_funcs=agg_funcs,
            )


# ═════════════════════════════════════════════════════════════
# CLF 성능 지표 계산 (trial attrs용)
# ═════════════════════════════════════════════════════════════
def _compute_clf_metrics(clf_result, pos_data, label_col):
    """
    clf_result의 val/train proba와 pos_data의 실제 라벨로 분류 성능 지표 계산.

    Returns
    -------
    dict : clf_val_auc, clf_val_ap, clf_train_oof_auc,
           clf_val_f1, clf_val_recall, clf_val_precision
    """
    positions = sorted(clf_result)
    val_proba   = np.concatenate([clf_result[p]["val_proba"]   for p in positions])
    train_proba = np.concatenate([clf_result[p]["train_proba"] for p in positions])
    y_val   = np.concatenate([pos_data[p]["val"][label_col].values   for p in positions])
    y_train = np.concatenate([pos_data[p]["train"][label_col].values for p in positions])
    val_bin = (val_proba >= 0.5).astype(int)
    return {
        "clf_val_auc":       float(roc_auc_score(y_val, val_proba)),
        "clf_val_ap":        float(average_precision_score(y_val, val_proba)),
        "clf_train_oof_auc": float(roc_auc_score(y_train, train_proba)),
        "clf_val_f1":        float(f1_score(y_val, val_bin, zero_division=0)),
        "clf_val_recall":    float(recall_score(y_val, val_bin, zero_division=0)),
        "clf_val_precision": float(precision_score(y_val, val_bin, zero_division=0)),
    }


# ═════════════════════════════════════════════════════════════
# 트라이얼별 CSV 콜백
# ═════════════════════════════════════════════════════════════
def _make_trial_csv_callback(csv_path, exp_id):
    """
    Optuna trial 완료마다 CSV에 한 줄 append하는 콜백.
    컬럼: exp_id, trial_number, saved_at, reg/clf 성능 지표
    """
    _CSV_COLS = [
        "exp_id", "trial_number", "saved_at",
        "reg_val_rmse", "reg_train_rmse",
        "n_feat_clean", "n_feat_selected",
        "clf_val_auc", "clf_val_ap", "clf_train_oof_auc",
        "clf_val_f1", "clf_val_recall", "clf_val_precision",
    ]

    def callback(study, trial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        attrs = trial.user_attrs
        row = {
            "exp_id":           exp_id or "",
            "trial_number":     trial.number,
            "saved_at":         attrs.get("saved_at", ""),
            "reg_val_rmse":     attrs.get("val_rmse"),
            "reg_train_rmse":   attrs.get("train_rmse"),
            "n_feat_clean":     attrs.get("n_feat_clean"),
            "n_feat_selected":  attrs.get("n_feat_selected"),
            "clf_val_auc":      attrs.get("clf_val_auc"),
            "clf_val_ap":       attrs.get("clf_val_ap"),
            "clf_train_oof_auc": attrs.get("clf_train_oof_auc"),
            "clf_val_f1":       attrs.get("clf_val_f1"),
            "clf_val_recall":   attrs.get("clf_val_recall"),
            "clf_val_precision": attrs.get("clf_val_precision"),
        }
        df_row = pd.DataFrame([row], columns=_CSV_COLS)
        write_header = not os.path.exists(csv_path)
        df_row.to_csv(csv_path, mode="a", header=write_header, index=False)

    return callback


# ═════════════════════════════════════════════════════════════
# E2E Optimization (메인)
# ═════════════════════════════════════════════════════════════
def run_e2e_optimization(
    pos_data,
    feat_cols,
    pipeline_config=None,
    clf_model="lgbm",
    reg_model="lgbm",
    n_trials=100,
    n_folds=3,
    clf_early_stop=50,
    reg_early_stop=50,
    label_col="label_bin",
    imbalance_method="scale_pos_weight",
    agg_funcs=None,
    top_k_range=(50, 500),
    top_k_fixed=200,
    clf_fixed=None,
    reg_fixed=None,
    unit_data_input=None,
    unit_feat_cols_input=None,
):
    """
    End-to-End Optuna 최적화 (가변 파이프라인)

    Parameters
    ----------
    pos_data : dict or None
        {position: {"train": df, "val": df, "test": df}}
        input_level='die'일 때 필수
    feat_cols : list
        die-level 피처 컬럼명
    pipeline_config : dict
        파이프라인 스위치 (DEFAULT_CONFIG 참조)
    clf_model, reg_model : str
    n_trials, n_folds : int
    clf_early_stop, reg_early_stop : int
    label_col, imbalance_method : str
    agg_funcs : list, optional
    top_k_range : tuple
        fs_optuna=True일 때 탐색 범위
    top_k_fixed : int
        fs_optuna=False일 때 고정 top_k
    clf_fixed, reg_fixed : dict, optional
    unit_data_input : dict, optional
        input_level='unit'일 때 외부에서 주입
    unit_feat_cols_input : list, optional
        input_level='unit'일 때 피처 컬럼명

    Returns
    -------
    dict : best_params, best_value, study, cached (사전 실행 결과)
    """
    cfg = _merge_config(pipeline_config)
    clf_fixed = clf_fixed or {}
    reg_fixed = reg_fixed or {}
    clf_prefix = "clf_"
    reg_prefix = "reg_"

    # ─── 사전 실행 (optuna=OFF 단계 캐싱) ───────────────────
    cached = {}

    # 1) CLF 사전 실행 (clf_optuna=False이고 run_clf=True)
    if cfg["run_clf"] and not cfg["clf_optuna"]:
        print("[캐싱] CLF: 기본 파라미터로 1회 실행...")
        default_clf = get_default_params(clf_model, "clf")
        default_clf.update(clf_fixed)
        cached["clf_result"] = _run_clf_oof(
            pos_data, feat_cols, default_clf, clf_model,
            n_folds, clf_early_stop, label_col, imbalance_method,
            clf_output=cfg["clf_output"],
        )
        print("[캐싱] CLF 완료")

    # 2) 집계 사전 실행 (CLF가 캐싱되었거나 CLF OFF)
    clf_cached = cached.get("clf_result")
    if not cfg["clf_optuna"]:
        # CLF 결과가 고정 → 집계도 고정
        if cfg["input_level"] == "unit":
            cached["unit_data"] = unit_data_input
            cached["unit_feat_cols"] = unit_feat_cols_input
            print("[캐싱] Unit 데이터 외부 주입")
        else:
            print("[캐싱] 집계 실행...")
            ud, ufc = _prepare_unit_data(
                pos_data, feat_cols, clf_cached, cfg, agg_funcs,
            )
            cached["unit_data"] = ud
            cached["unit_feat_cols"] = ufc
            print(f"[캐싱] 집계 완료 (피처: {len(ufc)}개)")

    # 3) FS 사전 실행 (fs_optuna=False이고 run_fs=True이고 집계가 캐싱됨)
    if cfg["run_fs"] and not cfg["fs_optuna"] and "unit_data" in cached:
        print(f"[캐싱] FS: top_k={top_k_fixed}로 1회 실행...")
        ufc = cached["unit_feat_cols"]
        X_tr = cached["unit_data"]["train"][ufc].values
        y_tr = cached["unit_data"]["train"][TARGET_COL].values
        nz_cols, nz_mask = remove_zero_variance(X_tr, ufc)
        X_tr_nz = X_tr[:, nz_mask]
        sel, _ = select_top_k(X_tr_nz, y_tr, nz_cols, top_k_fixed)
        cached["selected_cols"] = sel
        print(f"[캐싱] FS 완료 ({len(sel)}개 선택)")

    # ─── Optuna 필요 여부 확인 ──────────────────────────────
    any_optuna = cfg["clf_optuna"] or cfg["fs_optuna"] or cfg["reg_optuna"]

    if not any_optuna:
        # 옵튜나 전혀 안 씀 → 사전 실행 결과로 바로 회귀
        print("[No Optuna] 기본 파라미터로 회귀 실행...")
        sel = cached.get("selected_cols", cached.get("unit_feat_cols"))
        default_reg = get_default_params(reg_model, "reg")
        default_reg.update(reg_fixed)
        reg_result = _run_reg_oof(
            cached["unit_data"], sel, default_reg, reg_model,
            n_folds, reg_early_stop,
            use_clf=cfg["run_clf"], clf_filter=cfg["clf_filter"],
        )
        if cfg["reg_level"] == "position":
            reg_result, agg_ud = _die_pred_to_unit(cached["unit_data"], reg_result)
            y_val = agg_ud["val"][TARGET_COL].values
        else:
            y_val = cached["unit_data"]["val"][TARGET_COL].values
        val_rmse_score = rmse(y_val, reg_result["val_pred"])
        print(f"Val RMSE: {val_rmse_score:.6f}")

        return {
            "best_params": {},
            "best_value": val_rmse_score,
            "best_trial": None,
            "study": None,
            "cached": cached,
            "reg_result": reg_result,
        }

    # ─── Optuna objective ───────────────────────────────────
    def objective(trial):
        # ── ① CLF ──
        if cfg["clf_optuna"]:
            clf_params = SEARCH_SPACES[clf_model](trial, prefix=clf_prefix)
            clf_params.update(clf_fixed)
            clf_result = _run_clf_oof(
                pos_data, feat_cols, clf_params, clf_model,
                n_folds, clf_early_stop, label_col, imbalance_method,
                clf_output=cfg["clf_output"],
            )
        else:
            clf_result = cached.get("clf_result")

        # ── ② 집계 ──
        if cfg["clf_optuna"]:
            # CLF 결과가 매번 바뀜 → 매번 집계
            unit_data, unit_feat_cols = _prepare_unit_data(
                pos_data, feat_cols, clf_result, cfg, agg_funcs,
            )
        else:
            unit_data = cached["unit_data"]
            unit_feat_cols = cached["unit_feat_cols"]

        # ── ③ FS ──
        if cfg["run_fs"]:
            if cfg["fs_optuna"]:
                top_k = trial.suggest_int("top_k", top_k_range[0], top_k_range[1])
            else:
                top_k = top_k_fixed

            # CLF optuna=ON이면 집계 결과가 매번 달라서 FS도 매번 실행
            if cfg["clf_optuna"] or cfg["fs_optuna"] or "selected_cols" not in cached:
                X_tr = unit_data["train"][unit_feat_cols].values
                y_tr = unit_data["train"][TARGET_COL].values
                nz_cols, nz_mask = remove_zero_variance(X_tr, unit_feat_cols)
                X_tr_nz = X_tr[:, nz_mask]
                selected_cols, _ = select_top_k(X_tr_nz, y_tr, nz_cols, top_k)
            else:
                selected_cols = cached["selected_cols"]

            if len(selected_cols) < 10:
                return float("inf")
        else:
            selected_cols = unit_feat_cols

        # ── ④ REG ──
        if cfg["reg_optuna"]:
            reg_params = SEARCH_SPACES[reg_model](trial, prefix=reg_prefix)
            reg_params.update(reg_fixed)
        else:
            reg_params = get_default_params(reg_model, "reg")
            reg_params.update(reg_fixed)

        reg_result = _run_reg_oof(
            unit_data, selected_cols, reg_params, reg_model,
            n_folds, reg_early_stop,
            use_clf=cfg["run_clf"], clf_filter=cfg["clf_filter"],
        )

        # ── ⑤ RMSE ──
        if cfg["reg_level"] == "position":
            reg_result, _ = _die_pred_to_unit(unit_data, reg_result)

        y_val = unit_data["val"][TARGET_COL].values
        if cfg["reg_level"] == "position":
            # unit-level y (중복 제거)
            y_val = (
                unit_data["val"]
                .groupby(KEY_COL, sort=False)[TARGET_COL]
                .first()
                .values
            )
        val_rmse_score = rmse(y_val, reg_result["val_pred"])

        trial.set_user_attr("train_rmse", reg_result["train_rmse"])
        trial.set_user_attr("val_rmse", val_rmse_score)
        trial.set_user_attr("n_features", len(selected_cols))

        return val_rmse_score

    # ─── Study 실행 ─────────────────────────────────────────
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),  # 시드 제거: 매 실행마다 다른 trial 시퀀스
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    # ─── 결과 요약 ──────────────────────────────────────────
    best = study.best_trial
    active = [s for s in ["clf", "fs", "reg"] if cfg[f"{s}_optuna"]]

    print(f"\n{'=' * 60}")
    print(f"E2E HPO 완료 ({n_trials} trials)")
    print(f"Optuna ON: {', '.join(active)}")
    print(f"Best Val RMSE : {best.value:.6f}")
    print(f"Train RMSE    : {best.user_attrs['train_rmse']:.6f}")
    print(f"N Features    : {best.user_attrs['n_features']}")
    if "top_k" in best.params:
        print(f"Top-K         : {best.params['top_k']}")
    print(f"{'=' * 60}")

    return {
        "best_params": best.params,
        "best_value": best.value,
        "best_trial": best,
        "study": study,
        "cached": cached,
        "pipeline_config": cfg,
    }


# ═════════════════════════════════════════════════════════════
# Best Trial 재실행
# ═════════════════════════════════════════════════════════════
def rerun_best_trial(
    pos_data,
    feat_cols,
    best_params,
    pipeline_config=None,
    clf_model="lgbm",
    reg_model="lgbm",
    mode="single",
    n_folds=5,
    es_holdout=0.1,
    clf_early_stop=100,
    reg_early_stop=100,
    label_col="label_bin",
    imbalance_method="scale_pos_weight",
    agg_funcs=None,
    top_k_fixed=200,
    clf_fixed=None,
    reg_fixed=None,
    unit_data_input=None,
    unit_feat_cols_input=None,
):
    """
    Best trial 파라미터로 최종 예측 재실행

    Parameters
    ----------
    mode : str
        'single' → train 100%로 단일 모델 학습 (빠름, 데이터 최대 활용)
        'kfold'  → n_folds CV로 fold별 모델 학습 후 예측 평균 (앙상블 효과)
    n_folds : int
        mode='kfold'일 때만 사용
    es_holdout : float
        mode='single'일 때 early stopping용 train 내부 holdout 비율
    """
    if mode not in ("single", "kfold"):
        raise ValueError(f"mode must be 'single' or 'kfold', got '{mode}'")
    cfg = _merge_config(pipeline_config)
    clf_fixed = clf_fixed or {}
    reg_fixed = reg_fixed or {}
    clf_prefix = "clf_"
    reg_prefix = "reg_"

    # ── ① CLF ──
    if cfg["run_clf"]:
        if cfg["clf_optuna"]:
            clf_params = _build_params_from_best(
                best_params, clf_prefix, clf_model, clf_fixed
            )
        else:
            clf_params = get_default_params(clf_model, "clf")
            clf_params.update(clf_fixed)

        if mode == "single":
            print(f"Rerun CLF: {clf_model}, mode=single (es_holdout={es_holdout})")
            clf_result = _run_clf_single(
                pos_data, feat_cols, clf_params, clf_model,
                clf_early_stop, label_col, imbalance_method,
                clf_output=cfg["clf_output"], es_holdout=es_holdout,
            )
        else:
            print(f"Rerun CLF: {clf_model}, mode=kfold (folds={n_folds})")
            clf_result = _run_clf_oof(
                pos_data, feat_cols, clf_params, clf_model,
                n_folds, clf_early_stop, label_col, imbalance_method,
                clf_output=cfg["clf_output"],
            )
    else:
        clf_result = None

    # ── ② 집계 ──
    if cfg["input_level"] == "unit":
        unit_data = unit_data_input
        unit_feat_cols = unit_feat_cols_input
    else:
        unit_data, unit_feat_cols = _prepare_unit_data(
            pos_data, feat_cols, clf_result, cfg, agg_funcs,
        )

    # ── ③ FS ──
    if cfg["run_fs"]:
        if cfg["fs_optuna"] and "top_k" in best_params:
            top_k = best_params["top_k"]
        else:
            top_k = top_k_fixed

        X_tr = unit_data["train"][unit_feat_cols].values
        y_tr = unit_data["train"][TARGET_COL].values
        nz_cols, nz_mask = remove_zero_variance(X_tr, unit_feat_cols)
        X_tr_nz = X_tr[:, nz_mask]
        selected_cols, all_importances = select_top_k(
            X_tr_nz, y_tr, nz_cols, top_k,
        )
        imp_dict = dict(zip(nz_cols, all_importances))
        sel_importances = {c: imp_dict[c] for c in selected_cols}
    else:
        selected_cols = unit_feat_cols
        sel_importances = None

    # ── ④ REG ──
    if cfg["reg_optuna"]:
        reg_params = _build_params_from_best(
            best_params, reg_prefix, reg_model, reg_fixed
        )
    else:
        reg_params = get_default_params(reg_model, "reg")
        reg_params.update(reg_fixed)

    if mode == "single":
        print(f"Rerun REG: {reg_model}, features={len(selected_cols)}, mode=single (es_holdout={es_holdout})")
        reg_result = _run_reg_single(
            unit_data, selected_cols, reg_params, reg_model,
            reg_early_stop,
            use_clf=cfg["run_clf"], clf_filter=cfg["clf_filter"],
            es_holdout=es_holdout,
        )
    else:
        print(f"Rerun REG: {reg_model}, features={len(selected_cols)}, mode=kfold (folds={n_folds})")
        reg_result = _run_reg_oof(
            unit_data, selected_cols, reg_params, reg_model,
            n_folds, reg_early_stop,
            use_clf=cfg["run_clf"], clf_filter=cfg["clf_filter"],
        )

    # position 모드: die → unit 집계
    if cfg["reg_level"] == "position":
        reg_result, agg_unit_data = _die_pred_to_unit(unit_data, reg_result)
        y_val = agg_unit_data["val"][TARGET_COL].values
        # CSV용으로 unit_data를 unit-level로 교체
        unit_data = agg_unit_data
    else:
        y_val = unit_data["val"][TARGET_COL].values

    val_rmse_score = rmse(y_val, reg_result["val_pred"])
    print(f"Rerun Val RMSE: {val_rmse_score:.6f}")

    result = {
        "unit_data": unit_data,
        "selected_cols": selected_cols,
        "val_pred": reg_result["val_pred"],
        "test_pred": reg_result["test_pred"],
        "val_rmse": val_rmse_score,
        "clf_result": clf_result,
        "importances": sel_importances,
        "pipeline_config": cfg,
    }
    # mode='kfold'만 진짜 OOF. single은 in-sample이므로 별도 키로 분리.
    if mode == "kfold":
        result["oof_pred"] = reg_result["oof_pred"]
    else:  # single
        result["train_pred_insample"] = reg_result["train_pred_insample"]
    return result


# ═════════════════════════════════════════════════════════════
# 유틸
# ═════════════════════════════════════════════════════════════
def _build_params_from_best(best_params, prefix, model_name, fixed):
    """Optuna best_params에서 prefix 파라미터 추출 + 고정값 추가"""

    params = {}
    for k, v in best_params.items():
        if k == "top_k":
            continue
        if k.startswith(prefix):
            params[k[len(prefix):]] = v

    # 모델별 고정값
    if model_name == "lgbm":
        params.setdefault("random_state", SEED)
        params.setdefault("n_jobs", -1)
        params.setdefault("verbose", -1)
        params.setdefault("device", DEVICE)
    elif model_name == "xgb":
        params.setdefault("tree_method", "hist")
        params.setdefault("device", "cuda" if DEVICE == "gpu" else "cpu")
        params.setdefault("random_state", SEED)
        params.setdefault("n_jobs", -1)
        params.setdefault("verbosity", 0)
    elif model_name == "catboost":
        # task_type='CPU' 강제: Colab T4 GPU에서 LGBM과 GPU 메모리 경합으로 OOM 발생하여 CPU로 고정
        params.setdefault("task_type", "CPU")
        params.setdefault("random_seed", SEED)
        params.setdefault("verbose", 0)
        # subsample을 쓰려면 Bernoulli 필요 (기본 Bayesian은 subsample 미지원)
        params.setdefault("bootstrap_type", "Bernoulli")
    elif model_name in ("rf", "et"):
        params.setdefault("random_state", SEED)
        params.setdefault("n_jobs", -1)

    params.update(fixed)
    return params


# ═════════════════════════════════════════════════════════════
# LGBM-only baseline: 전처리 통합 HPO
# ═════════════════════════════════════════════════════════════
#
# run_e2e_optimization_with_pp / rerun_best_trial_with_pp
#
# - raw 데이터(xs, xs_dict, ys, feat_cols)를 인자로 받아서
#   매 trial마다 (cleaning + outlier + die→unit 집계) 실행
# - 전처리 결과는 (cleaning_params, outlier_params) hash로 LRU 캐시
# - 기존 _run_clf_oof / _run_reg_oof / _prepare_unit_data /
#   _die_pred_to_unit / _build_params_from_best 그대로 재사용
# ═════════════════════════════════════════════════════════════


def _pp_hash(pp_params):
    """전처리 파라미터 dict → 고정 hash 키 (순서 무관)"""
    s = json.dumps(pp_params, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:12]


def _lru_put(cache, key, value, max_size):
    """OrderedDict 기반 LRU 삽입"""
    if key in cache:
        cache.move_to_end(key)
        return
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_size:
        cache.popitem(last=False)


def _lru_get(cache, key):
    """OrderedDict 기반 LRU 조회"""
    if key not in cache:
        return None
    cache.move_to_end(key)
    return cache[key]


def _build_pos_data(xs_train, xs_val, xs_test, ys, label_col,
                    use_sampling=False, sample_frac=1.0, silent=True):
    """
    cleaning/outlier가 끝난 xs_*를 받아서:
      1) ys와 merge (die-level)
      2) label_col (0 vs >0) 생성
      3) (옵션) unit 기준 샘플링
      4) position별 분리

    Returns
    -------
    pos_data : dict
        {position: {"train": df, "val": df, "test": df}}
    """
    die_train = xs_train.merge(ys["train"], on=KEY_COL, how="left")
    die_val = xs_val.merge(ys["validation"], on=KEY_COL, how="left")
    die_test = xs_test.merge(ys["test"], on=KEY_COL, how="left")

    if die_train[TARGET_COL].isna().any():
        raise ValueError("train health NaN — merge 실패")
    if die_val[TARGET_COL].isna().any():
        raise ValueError("val health NaN — merge 실패")
    if die_test[TARGET_COL].isna().any():
        raise ValueError("test health NaN — merge 실패")

    if use_sampling and sample_frac < 1.0:
        all_units = die_train[KEY_COL].drop_duplicates()
        sampled = all_units.sample(frac=sample_frac, random_state=SEED)
        die_train = die_train[die_train[KEY_COL].isin(sampled)].reset_index(drop=True)

    for df in (die_train, die_val, die_test):
        df[label_col] = (df[TARGET_COL] > 0).astype(int)

    positions = sorted(die_train[POSITION_COL].unique())
    pos_data = {}
    for pos in positions:
        pos_data[pos] = {
            "train": die_train[die_train[POSITION_COL] == pos].reset_index(drop=True),
            "val":   die_val[die_val[POSITION_COL] == pos].reset_index(drop=True),
            "test":  die_test[die_test[POSITION_COL] == pos].reset_index(drop=True),
        }

    if not silent:
        print(f"[_build_pos_data] positions={positions}, "
              f"train_units={die_train[KEY_COL].nunique()}")

    return pos_data


def _run_preprocessing(xs, xs_dict, ys, feat_cols,
                       cleaning_args, outlier_args,
                       label_col, exclude_cols,
                       use_sampling, sample_frac):
    """
    cleaning + outlier + (옵션) EXCLUDE_COLS 필터 + pos_data 빌드 1회 실행.

    Returns
    -------
    pos_data, feat_cols_clean
    """
    # 지연 import (모듈 로드 시점엔 불필요)
    from cleaning import run_cleaning  # noqa: E402
    from outlier import run_outlier_treatment  # noqa: E402

    ys_train = ys["train"]

    # --- 1) cleaning ---
    xs_train, xs_val, xs_test, clean_cols, _ = run_cleaning(
        xs, feat_cols, xs_dict,
        ys_train=ys_train,
        **cleaning_args,
    )

    # --- 2) outlier ---
    xs_train, xs_val, xs_test, _ = run_outlier_treatment(
        xs_train, xs_val, xs_test, clean_cols,
        **outlier_args,
    )

    # --- 3) exclude_cols 필터 ---
    if exclude_cols:
        clean_cols = [c for c in clean_cols if c not in set(exclude_cols)]

    # --- 4) pos_data 빌드 ---
    pos_data = _build_pos_data(
        xs_train, xs_val, xs_test, ys, label_col,
        use_sampling=use_sampling, sample_frac=sample_frac,
        silent=True,
    )

    return pos_data, clean_cols


def _get_or_run_pp(cache, cache_key, xs, xs_dict, ys, feat_cols,
                   cleaning_args, outlier_args,
                   label_col, exclude_cols,
                   use_sampling, sample_frac, max_size):
    """
    전처리 캐시 조회 + 없으면 실행 + 저장.
    """
    hit = _lru_get(cache, cache_key)
    if hit is not None:
        return hit

    # 캐시 miss → 전처리 실행 (print silencing)
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pos_data, feat_cols_clean = _run_preprocessing(
            xs, xs_dict, ys, feat_cols,
            cleaning_args, outlier_args,
            label_col, exclude_cols,
            use_sampling, sample_frac,
        )

    value = {
        "pos_data": pos_data,
        "feat_cols": feat_cols_clean,
        "n_feat": len(feat_cols_clean),
    }
    _lru_put(cache, cache_key, value, max_size)
    return value


# ═════════════════════════════════════════════════════════════
# 전처리 통합 E2E Optimization
# ═════════════════════════════════════════════════════════════
def run_e2e_optimization_with_pp(
    xs,
    xs_dict,
    ys,
    feat_cols,
    pipeline_config=None,
    clf_model="lgbm",
    reg_model="lgbm",
    n_trials=30,
    n_folds=3,
    clf_early_stop=50,
    reg_early_stop=50,
    label_col="label_bin",
    imbalance_method="scale_pos_weight",
    top_k_range=(50, 500),
    top_k_fixed=200,
    clf_fixed=None,
    reg_fixed=None,
    use_sampling=False,
    sample_frac=1.0,
    exclude_cols=None,
    pp_cache_size=10,
    # SQLite / warm start / CSV 로그
    exp_id=None,
    db_path=None,
    csv_path=None,
    study_user_attrs=None,
    trial_callbacks=None,
    warm_start_top_k=0,
    warm_start_enabled=False,
):
    """
    전처리(cleaning/outlier/집계) + CLF + FS + REG 을 모두 포함한
    단일 Optuna study.

    1 trial = (전처리 파라미터 → cleaning → outlier → pos_data 빌드
              → CLF OOF → die→unit 집계 → FS → REG OOF → val RMSE)

    동일 전처리 파라미터 조합은 LRU 캐시로 재사용 (pp_cache_size 상한).

    Parameters
    ----------
    xs : DataFrame
        원본 die-level 피처 (split 컬럼 포함)
    xs_dict : dict
        {"train": df, "validation": df, "test": df} — utils.data.split_xs 결과
    ys : dict
        {"train": df, "validation": df, "test": df} — utils.data.load_all의 ys
    feat_cols : list
        die-level 원본 피처 컬럼 (X0~X1086)
    pipeline_config : dict
        기존 e2e와 동일한 스위치
    clf_model, reg_model : str
        기본 'lgbm'. baseline은 둘 다 lgbm 고정 권장
    n_trials, n_folds : int
    clf_early_stop, reg_early_stop : int
    label_col, imbalance_method : str
    top_k_range, top_k_fixed : FS 탐색 범위 / 고정 값
    clf_fixed, reg_fixed : dict, optional
        탐색에서 제외하고 고정할 파라미터
    use_sampling, sample_frac : bool, float
        unit 기준 샘플링 (빠른 탐색용)
    exclude_cols : list, optional
        웨이퍼맵 필터 등에서 사전 제외할 feature
    pp_cache_size : int
        전처리 LRU 캐시 상한 (default 10)

    Returns
    -------
    dict : best_params, best_value, study, pp_cache, pipeline_config
    """
    cfg = _merge_config(pipeline_config)
    clf_fixed = clf_fixed or {}
    reg_fixed = reg_fixed or {}
    clf_prefix = "clf_"
    reg_prefix = "reg_"

    # input_level='unit' 모드는 지원 안 함 (전처리부터 돌리므로 die 고정)
    if cfg["input_level"] != "die":
        raise ValueError(
            "run_e2e_optimization_with_pp는 input_level='die'만 지원."
        )

    # ── 전처리 LRU 캐시 ──
    pp_cache = OrderedDict()

    # ── Optuna objective ──
    def objective(trial):
        # ── ⓪ 전처리 (캐싱) ──
        pp_params = preprocessing_space(trial)
        cleaning_args, outlier_args, agg_funcs = split_pp_params(pp_params)
        cache_key = _pp_hash(pp_params)

        pp_value = _get_or_run_pp(
            pp_cache, cache_key, xs, xs_dict, ys, feat_cols,
            cleaning_args, outlier_args,
            label_col, exclude_cols,
            use_sampling, sample_frac, pp_cache_size,
        )
        pos_data = pp_value["pos_data"]
        feat_cols_clean = pp_value["feat_cols"]

        if len(feat_cols_clean) < 10:
            return float("inf")

        # ── ① CLF OOF ──
        if cfg["run_clf"]:
            if cfg["clf_optuna"]:
                clf_params = SEARCH_SPACES[clf_model](trial, prefix=clf_prefix)
                clf_params.update(clf_fixed)
            else:
                clf_params = get_default_params(clf_model, "clf")
                clf_params.update(clf_fixed)

            clf_result = _run_clf_oof(
                pos_data, feat_cols_clean, clf_params, clf_model,
                n_folds, clf_early_stop, label_col, imbalance_method,
                clf_output=cfg["clf_output"],
            )
        else:
            clf_result = None

        # ── ② die→unit 집계 ──
        unit_data, unit_feat_cols = _prepare_unit_data(
            pos_data, feat_cols_clean, clf_result, cfg, agg_funcs,
        )

        # ── ③ FS ──
        if cfg["run_fs"]:
            if cfg["fs_optuna"]:
                top_k = trial.suggest_int("top_k", top_k_range[0], top_k_range[1])
            else:
                top_k = top_k_fixed

            X_tr = unit_data["train"][unit_feat_cols].values
            y_tr = unit_data["train"][TARGET_COL].values
            nz_cols, nz_mask = remove_zero_variance(X_tr, unit_feat_cols)
            X_tr_nz = X_tr[:, nz_mask]
            selected_cols, _ = select_top_k(X_tr_nz, y_tr, nz_cols, top_k)

            if len(selected_cols) < 10:
                return float("inf")
        else:
            selected_cols = unit_feat_cols

        # ── ④ REG OOF ──
        if cfg["reg_optuna"]:
            reg_params = SEARCH_SPACES[reg_model](trial, prefix=reg_prefix)
            reg_params.update(reg_fixed)
        else:
            reg_params = get_default_params(reg_model, "reg")
            reg_params.update(reg_fixed)

        reg_result = _run_reg_oof(
            unit_data, selected_cols, reg_params, reg_model,
            n_folds, reg_early_stop,
            use_clf=cfg["run_clf"], clf_filter=cfg["clf_filter"],
        )

        # ── ⑤ RMSE ──
        if cfg["reg_level"] == "position":
            reg_result, _ = _die_pred_to_unit(unit_data, reg_result)
            y_val = (
                unit_data["val"]
                .groupby(KEY_COL, sort=False)[TARGET_COL]
                .first()
                .values
            )
        else:
            y_val = unit_data["val"][TARGET_COL].values

        val_rmse_score = rmse(y_val, reg_result["val_pred"])

        trial.set_user_attr("train_rmse", reg_result["train_rmse"])
        trial.set_user_attr("val_rmse", val_rmse_score)
        trial.set_user_attr("n_feat_clean", len(feat_cols_clean))
        trial.set_user_attr("n_feat_selected", len(selected_cols))
        trial.set_user_attr("agg_funcs", agg_funcs)
        trial.set_user_attr("saved_at", datetime.now().strftime("%Y-%m-%d %H:%M"))
        # 재현성 확보용: resolved 전처리 인자 + 선택된 컬럼 저장
        trial.set_user_attr("cleaning_args", cleaning_args)
        trial.set_user_attr("outlier_args", outlier_args)
        trial.set_user_attr("selected_cols", list(selected_cols))
        # CLF 성능 지표
        if clf_result is not None:
            for k, v in _compute_clf_metrics(clf_result, pos_data, label_col).items():
                trial.set_user_attr(k, v)

        return val_rmse_score

    # ── Study 실행 ──
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # SQLite storage (매 trial 자동 저장)
    if db_path:
        import os as _os_db
        _os_db.makedirs(_os_db.path.dirname(db_path), exist_ok=True)
        storage = f"sqlite:///{db_path}"
    else:
        storage = None
    study_name = exp_id if db_path else None

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        load_if_exists=True,
    )

    # study-level user attrs
    if study_user_attrs:
        for k, v in study_user_attrs.items():
            study.set_user_attr(k, v)

    # warm start: 기존 완료 trial 상위 K개 enqueue
    if warm_start_enabled and warm_start_top_k > 0:
        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if completed:
            completed.sort(key=lambda t: t.value)
            n_enqueue = min(warm_start_top_k, len(completed))
            for t in completed[:n_enqueue]:
                study.enqueue_trial(t.params)
            print(f"[Warm Start] 기존 {len(completed)}개 완료 trial 중 "
                  f"상위 {n_enqueue}개 enqueue")

    n_existing = len([
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ])

    all_callbacks = list(trial_callbacks or [])
    if csv_path:
        os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
        all_callbacks.append(_make_trial_csv_callback(csv_path, exp_id))

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
        callbacks=all_callbacks,
    )

    # ── 결과 요약 ──
    all_completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    best = study.best_trial
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    print(f"\n{'=' * 60}")
    print(f"E2E + Preprocessing HPO 완료 ({n_trials} trials)  [{now_str}]")
    if db_path:
        print(f"Storage       : {db_path}")
        print(f"Study         : {study_name}")
        print(f"Total trials  : {len(all_completed)} "
              f"(기존 {n_existing} + 신규 {len(all_completed) - n_existing})")
    print(f"PP cache      : {len(pp_cache)}/{pp_cache_size} entries (unique combos)")
    print(f"Best Val RMSE : {best.value:.6f}")
    print(f"Train RMSE    : {best.user_attrs.get('train_rmse', float('nan')):.6f}")
    print(f"N Features    : clean={best.user_attrs.get('n_feat_clean')}  "
          f"selected={best.user_attrs.get('n_feat_selected')}")
    print(f"Best agg_funcs: {best.user_attrs.get('agg_funcs')}")
    if "top_k" in best.params:
        print(f"Top-K         : {best.params['top_k']}")
    best_saved = best.user_attrs.get('saved_at', '?')
    print(f"Best trial at : {best_saved}")
    print(f"{'=' * 60}")

    return {
        "best_params": best.params,
        "best_value": best.value,
        "best_trial": best,
        "study": study,
        "pp_cache": pp_cache,
        "pipeline_config": cfg,
    }


# ═════════════════════════════════════════════════════════════
# 전처리 통합 Best Trial 재실행
# ═════════════════════════════════════════════════════════════
def rerun_best_trial_with_pp(
    xs,
    xs_dict,
    ys,
    feat_cols,
    best_params,
    pipeline_config=None,
    clf_model="lgbm",
    reg_model="lgbm",
    mode="single",
    n_folds=5,
    es_holdout=0.1,
    clf_early_stop=100,
    reg_early_stop=100,
    label_col="label_bin",
    imbalance_method="scale_pos_weight",
    top_k_fixed=200,
    clf_fixed=None,
    reg_fixed=None,
    use_sampling=False,
    sample_frac=1.0,
    exclude_cols=None,
):
    """
    best_params에서 pp_*, clf_*, reg_*, top_k 를 복원하여
    best 전처리 + best 모델로 최종 예측 재실행.

    기본 흐름은 기존 rerun_best_trial과 동일하되,
    맨 앞에 "best_pp_params → cleaning/outlier/pos_data" 단계가 추가됨.
    """
    if mode not in ("single", "kfold"):
        raise ValueError(f"mode must be 'single' or 'kfold', got '{mode}'")
    cfg = _merge_config(pipeline_config)
    clf_fixed = clf_fixed or {}
    reg_fixed = reg_fixed or {}
    clf_prefix = "clf_"
    reg_prefix = "reg_"

    if cfg["input_level"] != "die":
        raise ValueError(
            "rerun_best_trial_with_pp는 input_level='die'만 지원."
        )

    # ── ⓪ 전처리 (best pp_params 복원) ──
    pp_params = extract_pp_params_from_best(best_params)
    cleaning_args, outlier_args, agg_funcs = split_pp_params(pp_params)

    print(f"Rerun preprocessing: cleaning={len(cleaning_args)} args, "
          f"outlier method={outlier_args.get('method')}, "
          f"agg_funcs={agg_funcs}")
    pos_data, feat_cols_clean = _run_preprocessing(
        xs, xs_dict, ys, feat_cols,
        cleaning_args, outlier_args,
        label_col, exclude_cols,
        use_sampling, sample_frac,
    )
    print(f"Rerun preprocessing 완료: feat_cols={len(feat_cols_clean)}")

    # ── ① CLF ──
    if cfg["run_clf"]:
        if cfg["clf_optuna"]:
            clf_params = _build_params_from_best(
                best_params, clf_prefix, clf_model, clf_fixed
            )
        else:
            clf_params = get_default_params(clf_model, "clf")
            clf_params.update(clf_fixed)

        if mode == "single":
            print(f"Rerun CLF: {clf_model}, mode=single (es_holdout={es_holdout})")
            clf_result = _run_clf_single(
                pos_data, feat_cols_clean, clf_params, clf_model,
                clf_early_stop, label_col, imbalance_method,
                clf_output=cfg["clf_output"], es_holdout=es_holdout,
            )
        else:
            print(f"Rerun CLF: {clf_model}, mode=kfold (folds={n_folds})")
            clf_result = _run_clf_oof(
                pos_data, feat_cols_clean, clf_params, clf_model,
                n_folds, clf_early_stop, label_col, imbalance_method,
                clf_output=cfg["clf_output"],
            )
    else:
        clf_result = None

    # ── ② 집계 ──
    unit_data, unit_feat_cols = _prepare_unit_data(
        pos_data, feat_cols_clean, clf_result, cfg, agg_funcs,
    )

    # ── ③ FS ──
    if cfg["run_fs"]:
        if cfg["fs_optuna"] and "top_k" in best_params:
            top_k = best_params["top_k"]
        else:
            top_k = top_k_fixed

        X_tr = unit_data["train"][unit_feat_cols].values
        y_tr = unit_data["train"][TARGET_COL].values
        nz_cols, nz_mask = remove_zero_variance(X_tr, unit_feat_cols)
        X_tr_nz = X_tr[:, nz_mask]
        selected_cols, all_importances = select_top_k(
            X_tr_nz, y_tr, nz_cols, top_k,
        )
        imp_dict = dict(zip(nz_cols, all_importances))
        sel_importances = {c: imp_dict[c] for c in selected_cols}
    else:
        selected_cols = unit_feat_cols
        sel_importances = None

    # ── ④ REG ──
    if cfg["reg_optuna"]:
        reg_params = _build_params_from_best(
            best_params, reg_prefix, reg_model, reg_fixed
        )
    else:
        reg_params = get_default_params(reg_model, "reg")
        reg_params.update(reg_fixed)

    if mode == "single":
        print(f"Rerun REG: {reg_model}, features={len(selected_cols)}, mode=single (es_holdout={es_holdout})")
        reg_result = _run_reg_single(
            unit_data, selected_cols, reg_params, reg_model,
            reg_early_stop,
            use_clf=cfg["run_clf"], clf_filter=cfg["clf_filter"],
            es_holdout=es_holdout,
        )
    else:
        print(f"Rerun REG: {reg_model}, features={len(selected_cols)}, mode=kfold (folds={n_folds})")
        reg_result = _run_reg_oof(
            unit_data, selected_cols, reg_params, reg_model,
            n_folds, reg_early_stop,
            use_clf=cfg["run_clf"], clf_filter=cfg["clf_filter"],
        )

    # position 모드: die → unit 집계
    if cfg["reg_level"] == "position":
        reg_result, agg_unit_data = _die_pred_to_unit(unit_data, reg_result)
        y_val = agg_unit_data["val"][TARGET_COL].values
        unit_data = agg_unit_data
    else:
        y_val = unit_data["val"][TARGET_COL].values

    val_rmse_score = rmse(y_val, reg_result["val_pred"])
    print(f"Rerun Val RMSE: {val_rmse_score:.6f}")

    # 모델 객체 추출 (SHAP / feature importance 사후 분석용)
    if mode == "kfold":
        reg_models = reg_result.get("fold_models", [])
    else:
        _single_model = reg_result.get("model")
        reg_models = [_single_model] if _single_model is not None else []

    result = {
        "unit_data": unit_data,
        "selected_cols": selected_cols,
        "feat_cols_clean": feat_cols_clean,
        "val_pred": reg_result["val_pred"],
        "test_pred": reg_result["test_pred"],
        "val_rmse": val_rmse_score,
        "clf_result": clf_result,
        "importances": sel_importances,
        "pipeline_config": cfg,
        "best_pp_params": pp_params,
        "cleaning_args": cleaning_args,
        "outlier_args": outlier_args,
        "agg_funcs": agg_funcs,
        "reg_models": reg_models,
    }
    if mode == "kfold":
        result["oof_pred"] = reg_result["oof_pred"]
    else:
        result["train_pred_insample"] = reg_result["train_pred_insample"]
    return result
