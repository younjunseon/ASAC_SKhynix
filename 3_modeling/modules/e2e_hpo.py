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
from sklearn.model_selection import (
    StratifiedKFold, KFold, GroupKFold, StratifiedGroupKFold,
    GroupShuffleSplit, train_test_split,
)
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, recall_score, precision_score,
)

from utils.config import (
    SEED, TARGET_COL, KEY_COL, POSITION_COL, PROJECT_ROOT, ENV,
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
    PP_SCALE_CONFIG,        # ★ 2차: hybrid_scale 고정 설정
    PP_BINARIZE_CONFIG,     # ★ 2차: binarize_degenerate 고정 설정
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

    # 후처리
    zero_clip=False,            # True → 회귀 예측값 < threshold → 0 으로 clip
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
    4-position concat 분류 OOF (★ Speed A: fit 횟수 1/4)

    기존: pos별 독립 학습 (4pos × n_folds = 4N fits)
    변경: 4pos concat → 1회 학습 (n_folds fits) + position feature 추가
          GroupKFold → 같은 unit의 die가 train/val에 섞이지 않음 (leakage 방지)

    Parameters
    ----------
    clf_output : str
        'proba' → P(Y>0) 확률, 'binary' → 0/1 예측

    Returns
    -------
    clf_result : dict
        {position: {"train_proba": arr, "val_proba": arr, "test_proba": arr}}
        기존과 동일한 포맷 (하위 소비자 15개 호환)
    """
    positions = sorted(pos_data.keys())
    OHE_POSITIONS = [1, 2, 3, 4]

    # ── 1. 4-position concat + position feature 추가 ──
    def _concat_with_pos(split_name):
        frames_X, frames_y, frames_group = [], [], []
        pos_lengths = {}
        for pos in positions:
            df = pos_data[pos][split_name]
            X = df[feat_cols].values
            # position features: ordinal + OHE
            pos_ord = np.full((len(X), 1), pos, dtype=np.float32)
            pos_ohe = np.zeros((len(X), len(OHE_POSITIONS)), dtype=np.int8)
            pos_ohe[:, OHE_POSITIONS.index(pos)] = 1
            X_aug = np.hstack([X, pos_ord, pos_ohe])
            frames_X.append(X_aug)
            frames_y.append(df[label_col].values)
            frames_group.append(df[KEY_COL].values)
            pos_lengths[pos] = len(X)
        return (np.vstack(frames_X),
                np.concatenate(frames_y),
                np.concatenate(frames_group),
                pos_lengths)

    X_tr_all, y_tr_all, groups_all, pos_len_tr = _concat_with_pos("train")
    X_val_all, _, _, pos_len_val = _concat_with_pos("val")
    X_test_all, _, _, pos_len_test = _concat_with_pos("test")

    # ── 2. 불균형 처리 (concat 전체 기준 1회) ──
    actual_params = clf_params.copy()
    if imbalance_method == "scale_pos_weight":
        n_neg = (y_tr_all == 0).sum()
        n_pos = (y_tr_all == 1).sum()
        spw = n_neg / n_pos if n_pos > 0 else 1.0
        if model_name in ("lgbm", "xgb"):
            actual_params["scale_pos_weight"] = spw
        elif model_name == "catboost":
            actual_params["auto_class_weights"] = "Balanced"
        elif model_name in ("rf", "et"):
            actual_params["class_weight"] = "balanced"
        elif model_name == "logreg_enet":
            actual_params["class_weight"] = "balanced"

    # ── 3. StratifiedGroupKFold (unit 단위 분할 + 클래스 균형) ──
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof_proba = np.zeros(len(X_tr_all))
    fold_models = []

    for tr_idx, val_idx in sgkf.split(X_tr_all, y_tr_all, groups=groups_all):
        clf = create_model(model_name, "clf", actual_params)
        if supports_early_stopping(model_name):
            X_tr_fold = X_tr_all[tr_idx]
            y_tr_fold = y_tr_all[tr_idx]
            groups_fold = groups_all[tr_idx]
            if len(np.unique(y_tr_fold)) >= 2:
                # ES inner holdout도 group-aware
                gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
                inner_tr, inner_es = next(gss.split(X_tr_fold, y_tr_fold, groups=groups_fold))
                fit_model(
                    clf,
                    X_tr_fold[inner_tr], y_tr_fold[inner_tr],
                    X_tr_fold[inner_es], y_tr_fold[inner_es],
                    early_stop,
                )
            else:
                fit_model(clf, X_tr_fold, y_tr_fold)
        else:
            fit_model(clf, X_tr_all[tr_idx], y_tr_all[tr_idx])

        oof_proba[val_idx] = clf.predict_proba(X_tr_all[val_idx])[:, 1]
        fold_models.append(clf)

    # ── 4. val/test predict (fold 평균) ──
    val_proba_all = np.mean(
        [m.predict_proba(X_val_all)[:, 1] for m in fold_models], axis=0
    )
    test_proba_all = np.mean(
        [m.predict_proba(X_test_all)[:, 1] for m in fold_models], axis=0
    )

    # binary 모드
    if clf_output == "binary":
        oof_proba = (oof_proba > 0.5).astype(float)
        val_proba_all = (val_proba_all > 0.5).astype(float)
        test_proba_all = (test_proba_all > 0.5).astype(float)

    # ── 5. position별 split-back (기존 반환 포맷 유지) ──
    clf_result = {}
    tr_off, val_off, test_off = 0, 0, 0
    for pos in positions:
        n_tr = pos_len_tr[pos]
        n_val = pos_len_val[pos]
        n_test = pos_len_test[pos]
        clf_result[pos] = {
            "train_proba": oof_proba[tr_off:tr_off + n_tr],
            "val_proba":   val_proba_all[val_off:val_off + n_val],
            "test_proba":  test_proba_all[test_off:test_off + n_test],
        }
        tr_off += n_tr
        val_off += n_val
        test_off += n_test

    return clf_result


# ═════════════════════════════════════════════════════════════
# Isotonic calibration (★ 2차 신규)
# ═════════════════════════════════════════════════════════════
def _apply_isotonic_calibration(clf_result_single, pos_data, label_col):
    """
    Isotonic regression으로 position별 train OOF proba 보정 후
    모든 split에 transform.

    train_proba는 OOF(out-of-fold) 예측이라 Isotonic을 in-sample fit 해도
    누수 없음. val/test는 학습된 monotonic mapping을 그대로 적용.

    Parameters
    ----------
    clf_result_single : dict
        _run_clf_oof 반환 {pos: {'train_proba','val_proba','test_proba'}}
    pos_data : dict
        train label 조회용
    label_col : str
        pos_data[pos]['train']의 label 컬럼명

    Returns
    -------
    calibrated : dict  (clf_result_single과 동일 구조)
    """
    from sklearn.isotonic import IsotonicRegression  # 지연 import

    calibrated = {}
    for pos, d in clf_result_single.items():
        y_train = pos_data[pos]["train"][label_col].values
        p_train = d["train_proba"]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_train, y_train)
        calibrated[pos] = {
            "train_proba": iso.transform(p_train),
            "val_proba":   iso.transform(d["val_proba"]),
            "test_proba":  iso.transform(d["test_proba"]),
        }
    return calibrated


# ═════════════════════════════════════════════════════════════
# Multi-model 분류 OOF + Calibration + Soft voting (★ 2차 신규)
# ═════════════════════════════════════════════════════════════
def _run_clf_oof_multi(pos_data, feat_cols, clf_params_by_model, clf_models,
                       n_folds, early_stop, label_col, imbalance_method,
                       calibration=None, clf_output="proba"):
    """
    여러 분류 모델의 OOF proba를 생성하고, (옵션) Isotonic calibration을
    LGBM/ET에만 적용한 뒤 position별로 soft voting.

    Parameters
    ----------
    pos_data : dict {position: {'train','val','test'}}
    feat_cols : list
    clf_params_by_model : dict {model_name: params_dict}
        모델별 HP — Optuna에서 prefix 분리로 샘플링된 값
    clf_models : list or tuple
        학습할 모델 이름 순서
    calibration : dict or None
        {'method': 'isotonic', 'models': ['lgbm', 'et']}
        'models'에 포함된 모델에만 Isotonic 적용. LogReg 등은 자체 probabilistic
        이라 제외.
    clf_output : 'proba' | 'binary'

    Returns
    -------
    clf_result : dict {pos: {'train_proba','val_proba','test_proba'}}
        soft-voted 합본 (기존 _run_clf_oof 반환 형식과 호환 — 하위 _prepare_unit_data
        가 그대로 받음).
    per_model_clf_result : dict {model: {pos: {...}}}
        모델별 독립 OOF (calibration 적용 후 기준) — OOF CSV 저장/진단용.
    """
    calib_models = set()
    if calibration and calibration.get("method") == "isotonic":
        calib_models = set(calibration.get("models", []))

    per_model_results = {}
    for clf_m in clf_models:
        r = _run_clf_oof(
            pos_data, feat_cols, clf_params_by_model[clf_m], clf_m,
            n_folds, early_stop, label_col, imbalance_method,
            clf_output=clf_output,
        )
        # Isotonic calibration (LGBM/ET만)
        if clf_m in calib_models:
            r = _apply_isotonic_calibration(r, pos_data, label_col)
        per_model_results[clf_m] = r

    # Soft voting (position별 단순 평균)
    positions = sorted(per_model_results[clf_models[0]].keys())
    clf_result = {}
    for pos in positions:
        clf_result[pos] = {
            "train_proba": np.mean(
                [per_model_results[m][pos]["train_proba"] for m in clf_models], axis=0),
            "val_proba":   np.mean(
                [per_model_results[m][pos]["val_proba"]   for m in clf_models], axis=0),
            "test_proba":  np.mean(
                [per_model_results[m][pos]["test_proba"]  for m in clf_models], axis=0),
        }

    return clf_result, per_model_results


# ═════════════════════════════════════════════════════════════
# 단일 모델 분류 (rerun mode='single')
# ═════════════════════════════════════════════════════════════
def _run_clf_single(pos_data, feat_cols, clf_params, model_name,
                    early_stop, label_col, imbalance_method,
                    clf_output="proba", es_holdout=0.1):
    """
    4-position concat 분류 단일 학습 (★ Speed A: fit 1회)

    기존: pos별 독립 학습 (4 fits)
    변경: 4pos concat → 1회 학습 + position feature
          ES holdout도 GroupShuffleSplit (unit leakage 방지)

    Returns
    -------
    clf_result : dict  (기존과 동일 포맷)
    """
    positions = sorted(pos_data.keys())
    OHE_POSITIONS = [1, 2, 3, 4]

    def _concat_with_pos(split_name):
        frames_X, frames_y, frames_group = [], [], []
        pos_lengths = {}
        for pos in positions:
            df = pos_data[pos][split_name]
            X = df[feat_cols].values
            pos_ord = np.full((len(X), 1), pos, dtype=np.float32)
            pos_ohe = np.zeros((len(X), len(OHE_POSITIONS)), dtype=np.int8)
            pos_ohe[:, OHE_POSITIONS.index(pos)] = 1
            X_aug = np.hstack([X, pos_ord, pos_ohe])
            frames_X.append(X_aug)
            frames_y.append(df[label_col].values)
            frames_group.append(df[KEY_COL].values)
            pos_lengths[pos] = len(X)
        return (np.vstack(frames_X),
                np.concatenate(frames_y),
                np.concatenate(frames_group),
                pos_lengths)

    X_tr_all, y_tr_all, groups_all, pos_len_tr = _concat_with_pos("train")
    X_val_all, _, _, pos_len_val = _concat_with_pos("val")
    X_test_all, _, _, pos_len_test = _concat_with_pos("test")

    # 불균형 처리
    actual_params = clf_params.copy()
    if imbalance_method == "scale_pos_weight":
        n_neg = (y_tr_all == 0).sum()
        n_pos = (y_tr_all == 1).sum()
        spw = n_neg / n_pos if n_pos > 0 else 1.0
        if model_name in ("lgbm", "xgb"):
            actual_params["scale_pos_weight"] = spw
        elif model_name == "catboost":
            actual_params["auto_class_weights"] = "Balanced"
        elif model_name in ("rf", "et"):
            actual_params["class_weight"] = "balanced"
        elif model_name == "logreg_enet":
            actual_params["class_weight"] = "balanced"

    # ES holdout (group-aware)
    use_es = supports_early_stopping(model_name) and es_holdout > 0
    if use_es and len(np.unique(y_tr_all)) >= 2:
        gss = GroupShuffleSplit(n_splits=1, test_size=es_holdout, random_state=SEED)
        fit_idx, es_idx = next(gss.split(X_tr_all, y_tr_all, groups=groups_all))
        X_fit, y_fit = X_tr_all[fit_idx], y_tr_all[fit_idx]
        X_es, y_es = X_tr_all[es_idx], y_tr_all[es_idx]
    else:
        X_fit, y_fit = X_tr_all, y_tr_all
        X_es, y_es = None, None

    clf = create_model(model_name, "clf", actual_params)
    if use_es and X_es is not None:
        fit_model(clf, X_fit, y_fit, X_es, y_es, early_stop)
    else:
        fit_model(clf, X_fit, y_fit)

    train_proba = clf.predict_proba(X_tr_all)[:, 1]
    val_proba = clf.predict_proba(X_val_all)[:, 1]
    test_proba = clf.predict_proba(X_test_all)[:, 1]

    if clf_output == "binary":
        train_proba = (train_proba > 0.5).astype(float)
        val_proba = (val_proba > 0.5).astype(float)
        test_proba = (test_proba > 0.5).astype(float)

    # position별 split-back
    clf_result = {}
    tr_off, val_off, test_off = 0, 0, 0
    for pos in positions:
        n_tr = pos_len_tr[pos]
        n_val = pos_len_val[pos]
        n_test = pos_len_test[pos]
        clf_result[pos] = {
            "train_proba": train_proba[tr_off:tr_off + n_tr],
            "val_proba":   val_proba[val_off:val_off + n_val],
            "test_proba":  test_proba[test_off:test_off + n_test],
        }
        tr_off += n_tr
        val_off += n_val
        test_off += n_test

    return clf_result


# ═════════════════════════════════════════════════════════════
# 단일 모델 회귀 (rerun mode='single')
# ═════════════════════════════════════════════════════════════
def _run_reg_single(unit_data, feat_cols, reg_params, model_name,
                    early_stop, use_clf=True, clf_filter=False,
                    clf_filter_threshold=0.5, es_holdout=0.1,
                    target_transform_fn=None, target_inverse_fn=None):
    """
    Unit-level 회귀 단일 학습 (KFold 없음, train 100% 사용)

    clf_filter_threshold : float
        clf_filter=True일 때 회귀 학습에서 제외할 proba 임계값
        (clf_proba_mean <= threshold → 제외)
    target_transform_fn : callable, optional
        모델 학습 직전에 y에 적용할 변환 함수 (예: np.log1p, yeo transform).
        None이면 변환 없음. 변환/역변환은 **내부에서만** 일어나고
        unit_data의 y, 반환되는 예측값은 항상 **원본 스케일**이다.
    target_inverse_fn : callable, optional
        모델 예측 직후에 적용할 역변환 함수. target_transform_fn을 쓸 때 반드시 같이 지정.
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
        mask = clf_pred > clf_filter_threshold
    else:
        mask = np.ones(len(y_train), dtype=bool)

    reg = None  # fallback: 학습 실패 시 None
    if mask.sum() < 2:  # LightGBM 최소 2샘플 요구
        # fallback은 원본 스케일의 0 (이후 raw 스케일로 흘러감)
        oof_reg = np.zeros(len(X_train))
        val_reg = np.zeros(len(X_val))
        test_reg = np.zeros(len(X_test))
    else:
        X_fit_pool = X_train[mask]
        y_fit_pool = y_train[mask]

        # ── 타깃 변환: 모델 학습용으로만 변환 (y_train 원본은 건드리지 않음) ──
        if target_transform_fn is not None:
            y_fit_pool = target_transform_fn(y_fit_pool)

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

        oof_reg = reg.predict(X_train)  # in-sample (변환 스케일)
        val_reg = reg.predict(X_val)
        test_reg = reg.predict(X_test)

        # ── 예측 즉시 원본 스케일로 역변환 ──
        if target_inverse_fn is not None:
            oof_reg = target_inverse_fn(oof_reg)
            val_reg = target_inverse_fn(val_reg)
            test_reg = target_inverse_fn(test_reg)

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
                 n_folds, early_stop, use_clf=True, clf_filter=False,
                 clf_filter_threshold=0.5,
                 target_transform_fn=None, target_inverse_fn=None,
                 sample_weight=None):
    """
    Unit-level 회귀 OOF

    Parameters
    ----------
    use_clf : bool
        True → clf_proba_mean을 곱해서 최종 예측 (two-stage)
        False → 단순 회귀
    clf_filter : bool
        True → clf가 0으로 예측한 샘플을 학습에서 제외
    clf_filter_threshold : float
        clf_filter=True일 때 회귀 학습에서 제외할 proba 임계값
        (clf_proba_mean <= threshold → 제외)
    target_transform_fn : callable, optional
        모델 학습 직전에 y_fit에 적용할 변환 함수 (예: np.log1p, yeo transform).
        None이면 변환 없음. 변환/역변환은 **내부에서만** 일어나고
        unit_data의 y, fallback_val, 반환되는 예측값은 항상 **원본 스케일**이다.
    target_inverse_fn : callable, optional
        모델 예측 직후에 적용할 역변환 함수. target_transform_fn을 쓸 때 반드시 같이 지정.
    sample_weight : array-like or None (★ 2차 신규)
        X_train과 같은 길이의 sample weight (LDS 가중치 등).
        fold tr_idx + (pos_tr / tr_keep) + inner_tr 인덱싱을 거쳐 fit에 전달.
        None이면 균등 가중치.
    """
    X_train = unit_data["train"][feat_cols].values
    y_train = unit_data["train"][TARGET_COL].values
    groups = unit_data["train"][KEY_COL].values if KEY_COL in unit_data["train"].columns else None
    X_val = unit_data["val"][feat_cols].values
    X_test = unit_data["test"][feat_cols].values

    # clf_filter: 분류기가 0으로 예측한 샘플 제외
    if clf_filter and "clf_proba_mean" in unit_data["train"].columns:
        clf_pred = unit_data["train"]["clf_proba_mean"].values
        train_mask = clf_pred > clf_filter_threshold  # threshold 초과만 학습
    else:
        train_mask = None

    # Fallback: fold skip 시 사용할 보수적 예측값 (Y>0 평균)
    # two-stage에서 회귀 모델이 학습하는 분포가 y>0이므로 y>0 평균이 합리적
    pos_mask_all = y_train > 0
    fallback_val = float(y_train[pos_mask_all].mean()) if pos_mask_all.any() else 0.0

    if groups is not None and len(np.unique(groups)) >= n_folds:
        kf = GroupKFold(n_splits=n_folds)
        split_iter = kf.split(X_train, groups=groups)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        split_iter = kf.split(X_train)
    oof_reg = np.full(len(X_train), fallback_val, dtype=float)  # 0 대신 fallback 초기화
    fold_models = []
    skipped_folds = 0

    for tr_idx, va_idx in split_iter:
        # ─── 1단계: X_fit, y_fit, w_fit 결정 ───
        if use_clf and not clf_filter:
            # two-stage: Y>0인 샘플만 학습
            pos_tr = y_train[tr_idx] > 0
            if pos_tr.sum() < 2:  # LightGBM 최소 2샘플 요구
                skipped_folds += 1
                continue
            X_fit, y_fit = X_train[tr_idx][pos_tr], y_train[tr_idx][pos_tr]
            w_fit = sample_weight[tr_idx][pos_tr] if sample_weight is not None else None
        elif clf_filter and train_mask is not None:
            # clf_filter: 분류기가 1로 예측한 샘플만 학습
            tr_keep = train_mask[tr_idx]
            if tr_keep.sum() < 2:  # LightGBM 최소 2샘플 요구
                skipped_folds += 1
                continue
            X_fit, y_fit = X_train[tr_idx][tr_keep], y_train[tr_idx][tr_keep]
            w_fit = sample_weight[tr_idx][tr_keep] if sample_weight is not None else None
        else:
            # 단순 회귀: 전체 학습
            X_fit, y_fit = X_train[tr_idx], y_train[tr_idx]
            w_fit = sample_weight[tr_idx] if sample_weight is not None else None

        # ── 타깃 변환: 모델 학습용으로만 변환 ──
        if target_transform_fn is not None:
            y_fit = target_transform_fn(y_fit)

        # ─── 2단계: inner holdout 분리 (ES용, va_idx 누수 방지) ───
        reg = create_model(model_name, "reg", reg_params)
        if supports_early_stopping(model_name) and len(X_fit) >= 10:
            inner_tr, inner_es = train_test_split(
                np.arange(len(X_fit)),
                test_size=0.15,
                random_state=SEED,
            )
            w_inner = w_fit[inner_tr] if w_fit is not None else None
            fit_model(
                reg,
                X_fit[inner_tr], y_fit[inner_tr],
                X_fit[inner_es], y_fit[inner_es],
                early_stop,
                sample_weight=w_inner,
            )
        else:
            fit_model(reg, X_fit, y_fit, sample_weight=w_fit)

        # 예측 즉시 원본 스케일로 역변환
        pred_va = reg.predict(X_train[va_idx])
        if target_inverse_fn is not None:
            pred_va = target_inverse_fn(pred_va)
        oof_reg[va_idx] = pred_va
        fold_models.append(reg)

    # fold skip 경고 (일부 fold만 스킵된 경우)
    if skipped_folds > 0:
        warnings.warn(
            f"[_run_reg_oof] {skipped_folds}/{n_folds} fold(s) skipped due to "
            f"insufficient training samples (pos_tr or tr_keep < 2). "
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
        # fallback_val은 이미 원본 스케일이므로 역변환 없음
        val_reg = np.full(len(X_val), fallback_val, dtype=float)
        test_reg = np.full(len(X_test), fallback_val, dtype=float)
    else:
        val_reg = np.mean([m.predict(X_val) for m in fold_models], axis=0)
        test_reg = np.mean([m.predict(X_test) for m in fold_models], axis=0)
        # 앙상블 평균 후 원본 스케일로 역변환
        if target_inverse_fn is not None:
            val_reg = target_inverse_fn(val_reg)
            test_reg = target_inverse_fn(test_reg)

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
# Multi-model 회귀 OOF (★ 2차 신규) — LDS sample_weight + A4 주입 + 단순 평균 앙상블
# ═════════════════════════════════════════════════════════════
def _run_reg_oof_multi(unit_data, feat_cols, reg_params_by_model, reg_models,
                       n_folds, early_stop,
                       use_clf=True, clf_filter=False, clf_filter_threshold=0.5,
                       target_transform_fn=None, target_inverse_fn=None,
                       sample_weight=None,
                       add_clf_proba_to_reg=False):
    """
    여러 회귀 모델의 OOF 예측을 만들고 단순 평균으로 앙상블.

    Parameters
    ----------
    unit_data : dict
        _prepare_unit_data 반환 (reg_level='position'이면 die-level concat,
        'unit'이면 unit-level aggregated)
    feat_cols : list
        unit_feat_cols — reg_level='position'이면 POSITION_COL + p_1~p_4 포함
    reg_params_by_model : dict {model_name: params_dict}
    reg_models : list or tuple
        학습할 회귀 모델 이름 순서
    sample_weight : array-like or None
        X_train과 같은 길이. LDS 가중치 주입용. None이면 균등.
    add_clf_proba_to_reg : bool
        True면 unit_data에 있는 clf_proba_mean을 feature로 추가 (A4, 논문 2-1).
        use_clf=True(곱셈)와 독립적 — 둘 다 True면 feature + 곱셈 동시 적용 가능.
    use_clf / clf_filter / clf_filter_threshold / target_*_fn
        _run_reg_oof로 그대로 전달.

    Returns
    -------
    ensemble_reg : dict {'oof_pred','val_pred','test_pred','train_rmse'}
        모델별 예측을 단순 평균한 앙상블 결과 (fold_models는 per_model에 있음)
    per_model_reg : dict {model: dict}
        각 모델의 _run_reg_oof 원본 반환값 (fold_models 포함)
    """
    # A4: clf_proba_mean을 feature로 추가 (unit_data에 해당 컬럼이 있을 때만)
    effective_feat_cols = list(feat_cols)
    if add_clf_proba_to_reg and "clf_proba_mean" in unit_data["train"].columns:
        if "clf_proba_mean" not in effective_feat_cols:
            effective_feat_cols.append("clf_proba_mean")

    per_model_reg = {}
    for reg_m in reg_models:
        r = _run_reg_oof(
            unit_data, effective_feat_cols, reg_params_by_model[reg_m], reg_m,
            n_folds, early_stop,
            use_clf=use_clf, clf_filter=clf_filter,
            clf_filter_threshold=clf_filter_threshold,
            target_transform_fn=target_transform_fn,
            target_inverse_fn=target_inverse_fn,
            sample_weight=sample_weight,
        )
        per_model_reg[reg_m] = r

    # 단순 평균 (각 모델 1/len) — 가중 평균 튜닝은 후처리 Cell 8/9에서
    ensemble_reg = {
        "oof_pred":  np.mean([per_model_reg[m]["oof_pred"]  for m in reg_models], axis=0),
        "val_pred":  np.mean([per_model_reg[m]["val_pred"]  for m in reg_models], axis=0),
        "test_pred": np.mean([per_model_reg[m]["test_pred"] for m in reg_models], axis=0),
    }
    y_train = unit_data["train"][TARGET_COL].values
    ensemble_reg["train_rmse"] = rmse(y_train, ensemble_reg["oof_pred"])

    return ensemble_reg, per_model_reg


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
# 후처리: 예측값 zero-clip
# ═════════════════════════════════════════════════════════════
def _apply_zero_clip(reg_result, y_train, threshold):
    """
    회귀 예측값에서 threshold 미만을 0으로 치환하고 train_rmse 재계산.

    train(oof or insample), val, test 예측 모두 동일 threshold 적용.

    Parameters
    ----------
    reg_result : dict
        _run_reg_oof / _run_reg_single / _die_pred_to_unit 의 결과
    y_train : np.ndarray
        train_rmse 재계산용 정답값 (level 일치 필요)
    threshold : float
        0.0 또는 음수면 no-op (clip 안 함)

    Returns
    -------
    dict : 클립 적용된 새 reg_result (얕은 복사)
    """
    if threshold is None or threshold <= 0:
        return reg_result

    out = dict(reg_result)

    # train 예측 키 자동 감지 (kfold='oof_pred', single='train_pred_insample')
    if "oof_pred" in out:
        train_key = "oof_pred"
    elif "train_pred_insample" in out:
        train_key = "train_pred_insample"
    else:
        train_key = None

    if train_key is not None:
        out[train_key] = np.where(out[train_key] < threshold, 0.0, out[train_key])
        out["train_rmse"] = rmse(y_train, out[train_key])

    if "val_pred" in out:
        out["val_pred"] = np.where(out["val_pred"] < threshold, 0.0, out["val_pred"])
    if "test_pred" in out:
        out["test_pred"] = np.where(out["test_pred"] < threshold, 0.0, out["test_pred"])

    return out


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
        #
        # ★ 2차: Ordinal(POSITION_COL) + OHE(p_1~p_4) 병행.
        #   - 트리 모델(LGBM/ET): POSITION_COL 분할이 주력
        #   - 선형 모델(ElasticNet/LogReg-enet): p_1~p_4 독립 coef 활용
        #   두 인코딩 모두 feature로 제공하여 모델 타입 불문 대응.
        OHE_POSITIONS = [1, 2, 3, 4]
        unit_data = {}
        for split_name in ["train", "val", "test"]:
            frames = []
            for pos in sorted(pos_data.keys()):
                df = pos_data[pos][split_name].copy()
                # Position OHE 4컬럼 (int8로 메모리 절약)
                for p in OHE_POSITIONS:
                    df[f"p_{p}"] = np.int8(1 if pos == p else 0)
                if clf_result is not None:
                    df["clf_proba_mean"] = clf_result[pos][f"{split_name}_proba"]
                frames.append(df)
            unit_data[split_name] = pd.concat(frames, ignore_index=True)

        # Ordinal(POSITION_COL) + OHE(p_1~p_4) 모두 feature에 포함
        # clf_proba_mean은 곱셈용 기본, A4(add_clf_proba_to_reg)에서만 feature로 추가됨
        ohe_cols = [f"p_{p}" for p in OHE_POSITIONS]
        unit_feat_cols = feat_cols + [POSITION_COL] + ohe_cols

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
# Colab용: in-memory study ↔ SQLite 동기화
# ═════════════════════════════════════════════════════════════
def _sync_inmemory_to_sqlite(in_mem_storage, db_path, exp_id, verbose=True):
    """
    In-memory storage에 있는 study를 SQLite 파일로 통째로 덮어쓴다.
    (delete_study + copy_study 방식 — 일관성 보장)
    """
    storage_url = f"sqlite:///{db_path}"
    try:
        try:
            optuna.delete_study(study_name=exp_id, storage=storage_url)
        except KeyError:
            pass
        except Exception:
            pass
        optuna.copy_study(
            from_study_name=exp_id,
            from_storage=in_mem_storage,
            to_storage=storage_url,
            to_study_name=exp_id,
        )
        if verbose:
            n = len(optuna.load_study(study_name=exp_id, storage=storage_url).trials)
            print(f"[DB Sync] {n} trials → {db_path}")
    except Exception as e:
        print(f"[DB Sync Warning] {e}")


def _make_db_sync_callback(in_mem_storage, db_path, exp_id, sync_every=10):
    """sync_every trial마다 in-memory → SQLite로 sync하는 콜백."""
    state = {"last_synced": 0}

    def callback(study, trial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        completed = sum(
            1 for t in study.get_trials(deepcopy=False)
            if t.state == optuna.trial.TrialState.COMPLETE
        )
        if completed - state["last_synced"] >= sync_every:
            _sync_inmemory_to_sqlite(in_mem_storage, db_path, exp_id, verbose=True)
            state["last_synced"] = completed

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
    clf_filter_threshold_range=(0.05, 0.5),
    clf_filter_threshold_fixed=0.5,
    zero_clip_threshold_range=(0.0, 0.02),
    zero_clip_threshold_fixed=0.0,
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
    clf_filter_threshold_range : tuple
        clf_filter=True일 때 threshold 탐색 범위 (Optuna)
    clf_filter_threshold_fixed : float
        clf_filter=True이지만 탐색을 끄고 고정 임계값을 쓸 때 사용
        (현재는 reg_optuna=False & clf_filter=True 조합의 폴백 값으로 사용)
    zero_clip_threshold_range : tuple
        zero_clip=True일 때 threshold 탐색 범위 (Optuna)
        예측값 < threshold → 0 으로 치환
    zero_clip_threshold_fixed : float
        zero_clip=True이지만 탐색을 끄고 고정값을 쓸 때 사용 (no-optuna 폴백)
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
            clf_filter_threshold=clf_filter_threshold_fixed,
        )
        if cfg["reg_level"] == "position":
            reg_result, agg_ud = _die_pred_to_unit(cached["unit_data"], reg_result)
            y_train_for_clip = agg_ud["train"][TARGET_COL].values
            y_val = agg_ud["val"][TARGET_COL].values
        else:
            y_train_for_clip = cached["unit_data"]["train"][TARGET_COL].values
            y_val = cached["unit_data"]["val"][TARGET_COL].values

        # zero_clip: no-optuna 경로 → 고정값 사용
        if cfg["zero_clip"]:
            reg_result = _apply_zero_clip(
                reg_result, y_train_for_clip, zero_clip_threshold_fixed
            )

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

        # clf_filter_threshold: clf_filter=True일 때만 탐색
        if cfg["clf_filter"]:
            clf_filter_threshold = trial.suggest_float(
                "clf_filter_threshold",
                clf_filter_threshold_range[0],
                clf_filter_threshold_range[1],
                step=0.05,
            )
        else:
            clf_filter_threshold = clf_filter_threshold_fixed

        reg_result = _run_reg_oof(
            unit_data, selected_cols, reg_params, reg_model,
            n_folds, reg_early_stop,
            use_clf=cfg["run_clf"], clf_filter=cfg["clf_filter"],
            clf_filter_threshold=clf_filter_threshold,
        )

        # ── ⑤ RMSE ──
        if cfg["reg_level"] == "position":
            reg_result, agg_ud_for_clip = _die_pred_to_unit(unit_data, reg_result)
            y_train_for_clip = agg_ud_for_clip["train"][TARGET_COL].values
        else:
            y_train_for_clip = unit_data["train"][TARGET_COL].values

        # zero_clip: zero_clip=True일 때만 탐색
        if cfg["zero_clip"]:
            zero_clip_threshold = trial.suggest_float(
                "zero_clip_threshold",
                zero_clip_threshold_range[0],
                zero_clip_threshold_range[1],
                step=0.001,
            )
            reg_result = _apply_zero_clip(
                reg_result, y_train_for_clip, zero_clip_threshold
            )

        oof_rmse_score = reg_result["train_rmse"]
        trial.set_user_attr("train_rmse", oof_rmse_score)
        trial.set_user_attr("n_features", len(selected_cols))

        # val RMSE는 참고용으로만 기록 (Y가 있을 때)
        try:
            y_val = unit_data["val"][TARGET_COL].values
            if cfg["reg_level"] == "position":
                y_val = (
                    unit_data["val"]
                    .groupby(KEY_COL, sort=False)[TARGET_COL]
                    .first()
                    .values
                )
            trial.set_user_attr("val_rmse", rmse(y_val, reg_result["val_pred"]))
        except Exception:
            pass

        return oof_rmse_score

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
    print(f"Best OOF RMSE : {best.value:.6f}")
    val_rmse_ref = best.user_attrs.get('val_rmse')
    if val_rmse_ref is not None:
        print(f"Val RMSE (ref): {val_rmse_ref:.6f}")
    print(f"N Features    : {best.user_attrs['n_features']}")
    if "top_k" in best.params:
        print(f"Top-K         : {best.params['top_k']}")
    if "clf_filter_threshold" in best.params:
        print(f"CLF filter th : {best.params['clf_filter_threshold']:.3f}")
    if "zero_clip_threshold" in best.params:
        print(f"Zero clip th  : {best.params['zero_clip_threshold']:.4f}")
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
    clf_filter_threshold_fixed=0.5,
    zero_clip_threshold_fixed=0.0,
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

    # clf_filter_threshold 복원: best_params에 있으면 우선, 없으면 fixed
    clf_filter_threshold = best_params.get(
        "clf_filter_threshold", clf_filter_threshold_fixed
    )
    if cfg["clf_filter"]:
        print(f"Rerun clf_filter_threshold: {clf_filter_threshold}")

    if mode == "single":
        print(f"Rerun REG: {reg_model}, features={len(selected_cols)}, mode=single (es_holdout={es_holdout})")
        reg_result = _run_reg_single(
            unit_data, selected_cols, reg_params, reg_model,
            reg_early_stop,
            use_clf=cfg["run_clf"], clf_filter=cfg["clf_filter"],
            clf_filter_threshold=clf_filter_threshold,
            es_holdout=es_holdout,
        )
    else:
        print(f"Rerun REG: {reg_model}, features={len(selected_cols)}, mode=kfold (folds={n_folds})")
        reg_result = _run_reg_oof(
            unit_data, selected_cols, reg_params, reg_model,
            n_folds, reg_early_stop,
            use_clf=cfg["run_clf"], clf_filter=cfg["clf_filter"],
            clf_filter_threshold=clf_filter_threshold,
        )

    # 모델 객체 추출 (SHAP / feature importance 사후 분석용)
    # _die_pred_to_unit / _apply_zero_clip이 reg_result를 교체하기 전에 먼저 보관.
    if mode == "kfold":
        reg_models = reg_result.get("fold_models", [])
    else:
        _single_model = reg_result.get("model")
        reg_models = [_single_model] if _single_model is not None else []

    # position 모드: die → unit 집계
    # die_data: SHAP/Permutation Importance 등 사후 분석용으로 die-level
    # feature matrix (val/test) 를 slim copy 로 보관. agg 이후 unit_data는
    # [KEY_COL, TARGET_COL] 만 남아 feature 컬럼이 사라지므로 미리 떠 둔다.
    die_data = None
    if cfg["reg_level"] == "position":
        _keep_cols = [KEY_COL, TARGET_COL] + list(selected_cols)
        die_data = {}
        for _split in ("val", "test"):
            _cols = [c for c in _keep_cols if c in unit_data[_split].columns]
            die_data[_split] = unit_data[_split][_cols].copy()
        reg_result, agg_unit_data = _die_pred_to_unit(unit_data, reg_result)
        # CSV용으로 unit_data를 unit-level로 교체
        unit_data = agg_unit_data

    # zero_clip: best_params에 있으면 우선, 없으면 fixed
    if cfg["zero_clip"]:
        zero_clip_threshold = best_params.get(
            "zero_clip_threshold", zero_clip_threshold_fixed
        )
        print(f"Rerun zero_clip_threshold: {zero_clip_threshold}")
        y_train_for_clip = unit_data["train"][TARGET_COL].values
        reg_result = _apply_zero_clip(
            reg_result, y_train_for_clip, zero_clip_threshold
        )

    oof_rmse_score = reg_result["train_rmse"]
    print(f"Rerun OOF RMSE: {oof_rmse_score:.6f}")

    # val RMSE는 참고용
    val_rmse_score = None
    try:
        if cfg["reg_level"] == "position":
            y_val = unit_data["val"].groupby(KEY_COL, sort=False)[TARGET_COL].first().values
        else:
            y_val = unit_data["val"][TARGET_COL].values
        val_rmse_score = rmse(y_val, reg_result["val_pred"])
        print(f"Rerun Val RMSE: {val_rmse_score:.6f}  (참고용)")
    except Exception:
        pass

    result = {
        "unit_data": unit_data,
        "die_data": die_data,
        "selected_cols": selected_cols,
        "val_pred": reg_result["val_pred"],
        "test_pred": reg_result["test_pred"],
        "oof_rmse": oof_rmse_score,
        "val_rmse": val_rmse_score,
        "clf_result": clf_result,
        "importances": sel_importances,
        "pipeline_config": cfg,
        "reg_models": reg_models,
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

    # ★ LGBM objective 카테고리 파싱 (lgbm_space 내부 변환 재현)
    # search space: ['regression', 'poisson', 'tweedie_1.2', 'tweedie_1.5']
    # trial.params에는 'tweedie_1.2' 문자열로 저장되어 rerun 경로에서
    # LGBM이 이해 못하므로, 'tweedie' + tweedie_variance_power 로 분해.
    if model_name == "lgbm":
        obj = params.pop("objective", None)
        if obj == "poisson":
            params["objective"] = "poisson"
        elif isinstance(obj, str) and obj.startswith("tweedie"):
            params["objective"] = "tweedie"
            params["tweedie_variance_power"] = float(obj.split("_")[1])
        # "regression" 또는 None → LGBM 기본(MSE), objective 주입 생략

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
    # val/test Y는 비공개일 수 있으므로 있을 때만 검증
    if TARGET_COL in die_val.columns and die_val[TARGET_COL].notna().any():
        if die_val[TARGET_COL].isna().any():
            raise ValueError("val health NaN — merge 실패")
    if TARGET_COL in die_test.columns and die_test[TARGET_COL].notna().any():
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
                       cleaning_args, outlier_args, binarize_args,
                       iso_args, lds_args,
                       label_col, exclude_cols,
                       use_sampling, sample_frac):
    """
    2차 funnel 전처리 파이프라인 — 1회 실행.

    실행 순서 (strategy_2nd_preprocessing.md §7.1):
      1) cleaning
      1.5) binarize_degenerate (binarize_args['apply']=True일 때)
      2) outlier
      3) multivariate_anomaly_score (iso_args; iso_enabled=False면 스킵)
      4) hybrid_scale (고정: PP_SCALE_CONFIG, skew_threshold=5.0)
      5) exclude_cols 필터
      6) pos_data 빌드 (die-level, position 분리)
      7) compute_lds_weights (lds_args; expand_to_die=True → die-level weight)

    Parameters
    ----------
    binarize_args : dict
        {'apply', 'top_value_threshold', 'max_unique'}. None/빈 dict면
        PP_BINARIZE_CONFIG 폴백 (하위호환). split_pp_params가 만든 dict 그대로.

    Returns
    -------
    pos_data         : dict {position: {'train':df, 'val':df, 'test':df}}
    feat_cols_clean  : list — cleaning 이후 + iso_anomaly_score (iso_enabled=True일 때) + exclude 반영
    sample_weight    : np.ndarray (die-level) or None — train 전용, 길이 = 합산 die 수
    scaler           : HybridScaler (fitted) — 대시보드/예측 파이프라인 재사용
    """
    # 지연 import
    from cleaning import run_cleaning, binarize_degenerate   # noqa: E402
    from outlier import run_outlier_treatment, multivariate_anomaly_score   # noqa: E402
    from scaling import hybrid_scale   # noqa: E402
    from sample_weight import compute_lds_weights   # noqa: E402

    # binarize_args 폴백 (None/빈 dict → PP_BINARIZE_CONFIG 기본값)
    _ba = binarize_args or {}
    _apply     = _ba.get("apply", PP_BINARIZE_CONFIG.get("apply", False))
    _top_val   = _ba.get("top_value_threshold", PP_BINARIZE_CONFIG["top_value_threshold"])
    _max_uniq  = _ba.get("max_unique", PP_BINARIZE_CONFIG["max_unique"])

    ys_train = ys["train"]

    # --- 1) cleaning ---
    xs_train, xs_val, xs_test, clean_cols, _ = run_cleaning(
        xs, feat_cols, xs_dict,
        ys_train=ys_train,
        **cleaning_args,
    )

    # --- 1.5) binarize_degenerate (★ 2차 신규, trial args) ---
    if _apply:
        xs_train, xs_val, xs_test, _ = binarize_degenerate(
            xs_train, xs_val, xs_test, clean_cols,
            top_value_threshold=_top_val,
            max_unique=_max_uniq,
        )

    # --- 2) outlier ---
    xs_train, xs_val, xs_test, _ = run_outlier_treatment(
        xs_train, xs_val, xs_test, clean_cols,
        **outlier_args,
    )

    # --- 3) IsoForest anomaly score (★ 2차 신규, 옵션) ---
    if iso_args.get("iso_enabled", False):
        xs_train, xs_val, xs_test, clean_cols, _ = multivariate_anomaly_score(
            xs_train, xs_val, xs_test, clean_cols,
            contamination=iso_args.get("iso_contamination", "auto"),
            n_estimators=iso_args.get("iso_n_estimators", 200),
            random_state=SEED,
        )

    # --- 4) hybrid_scale (★ 2차 신규, 고정: skew_threshold=5.0) ---
    xs_train, xs_val, xs_test, scaler = hybrid_scale(
        xs_train, xs_val, xs_test, clean_cols,
        skew_threshold=PP_SCALE_CONFIG["skew_threshold"],
    )

    # --- 5) exclude_cols 필터 ---
    if exclude_cols:
        clean_cols = [c for c in clean_cols if c not in set(exclude_cols)]

    # --- 6) pos_data 빌드 ---
    pos_data = _build_pos_data(
        xs_train, xs_val, xs_test, ys, label_col,
        use_sampling=use_sampling, sample_frac=sample_frac,
        silent=True,
    )

    # --- 7) LDS sample_weight (★ 2차 신규, die-level 확장) ---
    if lds_args.get("lds_enabled", False):
        y_unit_train = ys_train[TARGET_COL].values
        sample_weight, _ = compute_lds_weights(
            y_unit_train,
            sigma=lds_args.get("lds_sigma", 0.01),
            max_weight=lds_args.get("lds_max_weight", 10.0),
            expand_to_die=True,
            ys_train_df=ys_train,
            pos_data=pos_data,
            key_col=KEY_COL,
        )
    else:
        sample_weight = None

    return pos_data, clean_cols, sample_weight, scaler


def _get_or_run_pp(cache, cache_key, xs, xs_dict, ys, feat_cols,
                   cleaning_args, outlier_args, binarize_args,
                   iso_args, lds_args,
                   label_col, exclude_cols,
                   use_sampling, sample_frac, max_size):
    """
    전처리 캐시 조회 + 없으면 실행 + 저장.

    캐시 value에 sample_weight(die-level) 및 scaler(HybridScaler) 추가.
    """
    hit = _lru_get(cache, cache_key)
    if hit is not None:
        return hit

    # 캐시 miss → 전처리 실행 (print silencing)
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pos_data, feat_cols_clean, sample_weight, scaler = _run_preprocessing(
            xs, xs_dict, ys, feat_cols,
            cleaning_args, outlier_args, binarize_args,
            iso_args, lds_args,
            label_col, exclude_cols,
            use_sampling, sample_frac,
        )

    value = {
        "pos_data": pos_data,
        "feat_cols": feat_cols_clean,
        "n_feat": len(feat_cols_clean),
        "sample_weight": sample_weight,   # ★ 2차: die-level train 가중치 or None
        "scaler": scaler,                 # ★ 2차: HybridScaler (fitted)
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
    clf_filter_threshold_range=(0.05, 0.5),
    clf_filter_threshold_fixed=0.5,
    zero_clip_threshold_range=(0.0, 0.02),
    zero_clip_threshold_fixed=0.0,
    clf_fixed=None,
    reg_fixed=None,
    use_sampling=False,
    sample_frac=1.0,
    exclude_cols=None,
    pp_cache_size=10,
    target_transform_fn=None,
    target_inverse_fn=None,
    # SQLite / warm start / CSV 로그
    exp_id=None,
    db_path=None,
    csv_path=None,
    study_user_attrs=None,
    trial_callbacks=None,
    warm_start_top_k=0,
    warm_start_enabled=False,
    # ★ 2차 신규
    clf_models=None,               # list/tuple. None이면 (clf_model,)로 폴백
    reg_models=None,               # list/tuple. None이면 (reg_model,)로 폴백
    calibration=None,              # dict {'method':'isotonic','models':[...]}
    add_clf_proba_to_reg=False,    # A4: clf_proba_mean을 reg feature로 추가
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
    clf_filter_threshold_range : tuple
        clf_filter=True일 때 threshold 탐색 범위 (Optuna)
    clf_filter_threshold_fixed : float
        clf_filter=False일 때 무시. 폴백용 고정값
    zero_clip_threshold_range : tuple
        zero_clip=True일 때 후처리 threshold 탐색 범위
    zero_clip_threshold_fixed : float
        zero_clip=True이지만 탐색을 끄고 고정값을 쓸 때 사용
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

    # ★ 2차: multi-model 지원 — list/tuple로 정규화
    clf_models_eff = tuple(clf_models) if clf_models else (clf_model,)
    reg_models_eff = tuple(reg_models) if reg_models else (reg_model,)

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
        cleaning_args, outlier_args, binarize_args, iso_args, lds_args, agg_funcs = split_pp_params(pp_params)
        cache_key = _pp_hash(pp_params)

        pp_value = _get_or_run_pp(
            pp_cache, cache_key, xs, xs_dict, ys, feat_cols,
            cleaning_args, outlier_args, binarize_args,
            iso_args, lds_args,
            label_col, exclude_cols,
            use_sampling, sample_frac, pp_cache_size,
        )
        pos_data = pp_value["pos_data"]
        feat_cols_clean = pp_value["feat_cols"]
        sample_weight = pp_value["sample_weight"]    # ★ 2차: die-level or None
        scaler = pp_value["scaler"]                   # ★ 2차: HybridScaler (fitted)

        if len(feat_cols_clean) < 10:
            return float("inf")

        # ── ① CLF OOF (★ 2차: multi-model + Isotonic + soft voting) ──
        if cfg["run_clf"]:
            # 모델별 HP를 prefix 분리로 샘플링
            clf_params_by_model = {}
            for m in clf_models_eff:
                if cfg["clf_optuna"]:
                    # multi 모드에선 모델 이름까지 prefix에 포함해 파라미터 충돌 방지
                    model_prefix = f"clf_{m}_" if len(clf_models_eff) > 1 else clf_prefix
                    p = SEARCH_SPACES[m](trial, prefix=model_prefix)
                else:
                    p = get_default_params(m, "clf")
                p.update(clf_fixed)
                clf_params_by_model[m] = p

            clf_result, per_model_clf = _run_clf_oof_multi(
                pos_data, feat_cols_clean, clf_params_by_model, clf_models_eff,
                n_folds, clf_early_stop, label_col, imbalance_method,
                calibration=calibration, clf_output=cfg["clf_output"],
            )
        else:
            clf_result = None
            per_model_clf = None

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

        # ── ④ REG OOF (★ 2차: multi-model + sample_weight + A4) ──
        reg_params_by_model = {}
        for m in reg_models_eff:
            if cfg["reg_optuna"]:
                model_prefix = f"reg_{m}_" if len(reg_models_eff) > 1 else reg_prefix
                p = SEARCH_SPACES[m](trial, prefix=model_prefix)
            else:
                p = get_default_params(m, "reg")
            p.update(reg_fixed)
            reg_params_by_model[m] = p

        # clf_filter_threshold: clf_filter=True일 때만 탐색
        if cfg["clf_filter"]:
            clf_filter_threshold = trial.suggest_float(
                "clf_filter_threshold",
                clf_filter_threshold_range[0],
                clf_filter_threshold_range[1],
                step=0.05,
            )
        else:
            clf_filter_threshold = clf_filter_threshold_fixed

        reg_result, per_model_reg = _run_reg_oof_multi(
            unit_data, selected_cols, reg_params_by_model, reg_models_eff,
            n_folds, reg_early_stop,
            use_clf=cfg["run_clf"], clf_filter=cfg["clf_filter"],
            clf_filter_threshold=clf_filter_threshold,
            target_transform_fn=target_transform_fn,
            target_inverse_fn=target_inverse_fn,
            sample_weight=sample_weight,
            add_clf_proba_to_reg=add_clf_proba_to_reg,
        )

        # ── ⑤ RMSE ──
        if cfg["reg_level"] == "position":
            reg_result, agg_ud_for_clip = _die_pred_to_unit(unit_data, reg_result)
            y_train_for_clip = agg_ud_for_clip["train"][TARGET_COL].values
        else:
            y_train_for_clip = unit_data["train"][TARGET_COL].values

        # zero_clip: zero_clip=True일 때만 탐색
        if cfg["zero_clip"]:
            zero_clip_threshold = trial.suggest_float(
                "zero_clip_threshold",
                zero_clip_threshold_range[0],
                zero_clip_threshold_range[1],
                step=0.001,
            )
            reg_result = _apply_zero_clip(
                reg_result, y_train_for_clip, zero_clip_threshold
            )

        oof_rmse_score = reg_result["train_rmse"]
        trial.set_user_attr("train_rmse", oof_rmse_score)
        trial.set_user_attr("n_feat_clean", len(feat_cols_clean))
        trial.set_user_attr("n_feat_selected", len(selected_cols))
        trial.set_user_attr("agg_funcs", agg_funcs)
        trial.set_user_attr("saved_at", datetime.now().strftime("%Y-%m-%d %H:%M"))
        # 재현성 확보용: resolved 전처리 인자 + 선택된 컬럼 저장
        trial.set_user_attr("cleaning_args", cleaning_args)
        trial.set_user_attr("outlier_args", outlier_args)
        trial.set_user_attr("selected_cols", list(selected_cols))
        # ★ 2차: IsoForest / LDS on/off 기록 (marginal 효과 분석용)
        trial.set_user_attr("iso_enabled", bool(iso_args.get("iso_enabled", False)))
        trial.set_user_attr("lds_enabled", bool(lds_args.get("lds_enabled", False)))
        # ★ 2차: 모델별 단독 OOF RMSE (앙상블 대비 단독 성능 비교용)
        trial.set_user_attr(
            "per_model_oof_rmse",
            {m: float(per_model_reg[m]["train_rmse"]) for m in reg_models_eff},
        )

        # val RMSE는 참고용으로만 기록 (Y가 있을 때)
        try:
            if cfg["reg_level"] == "position":
                y_val = (
                    unit_data["val"]
                    .groupby(KEY_COL, sort=False)[TARGET_COL]
                    .first()
                    .values
                )
            else:
                y_val = unit_data["val"][TARGET_COL].values
            trial.set_user_attr("val_rmse", rmse(y_val, reg_result["val_pred"]))
        except Exception:
            pass

        # CLF 성능 지표 (참고용)
        if clf_result is not None:
            try:
                for k, v in _compute_clf_metrics(clf_result, pos_data, label_col).items():
                    trial.set_user_attr(k, v)
            except Exception:
                pass

        return oof_rmse_score

    # ── Study 실행 ──
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Storage: Local은 매 trial 자동 SQLite 저장, Colab은 in-memory로만 돌리고 최종에 한 번 저장
    db_sync_callback = None
    inmem_storage_ref = None
    if db_path:
        import os as _os_db
        _db_dir = _os_db.path.dirname(db_path)
        if _db_dir:
            _os_db.makedirs(_db_dir, exist_ok=True)

        if ENV == "colab" and exp_id:
            # Colab: in-memory에서만 학습, 중간 sync 없음 (study.optimize 종료 후 1회 저장)
            inmem_storage_ref = optuna.storages.InMemoryStorage()
            storage = inmem_storage_ref
            # 기존 SQLite가 있으면 in-memory로 미리 로드 (warm start 데이터 보존)
            if _os_db.path.exists(db_path):
                _disk_url = f"sqlite:///{db_path}"
                try:
                    optuna.copy_study(
                        from_study_name=exp_id,
                        from_storage=_disk_url,
                        to_storage=inmem_storage_ref,
                        to_study_name=exp_id,
                    )
                    print(f"[Colab] 기존 DB에서 trial 로드 완료: {db_path}")
                except (KeyError, Exception) as _e:
                    pass
        else:
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

    # [Warm Start] enqueue_trial로 완료 trial을 재실행하던 구 로직은 compute 낭비였음.
    # TPESampler는 load_if_exists=True + SQLite 재로드로 기존 trial을 prior에
    # 자동 반영하므로, 별도 enqueue 없이도 사실상 warm-start됨.
    # warm_start_top_k / warm_start_enabled 인자는 하위호환 위해 남겨두지만 no-op.
    if warm_start_enabled and warm_start_top_k > 0:
        n_prior = sum(
            1 for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        )
        if n_prior > 0:
            print(f"[Warm Start] 기존 {n_prior}개 완료 trial을 TPE posterior로 재활용 "
                  f"(enqueue 재실행 X, warm_start_top_k={warm_start_top_k}는 deprecated no-op)")

    n_existing = len([
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ])

    all_callbacks = list(trial_callbacks or [])
    if csv_path:
        os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
        all_callbacks.append(_make_trial_csv_callback(csv_path, exp_id))
    if db_sync_callback is not None:
        all_callbacks.append(db_sync_callback)

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
        callbacks=all_callbacks,
    )

    # Colab: 마지막 sync 이후 잔여 trial을 디스크로 최종 동기화
    if inmem_storage_ref is not None and db_path and exp_id:
        _sync_inmemory_to_sqlite(inmem_storage_ref, db_path, exp_id, verbose=True)

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
    print(f"Best OOF RMSE : {best.value:.6f}")
    val_rmse_ref = best.user_attrs.get('val_rmse')
    if val_rmse_ref is not None:
        print(f"Val RMSE (ref): {val_rmse_ref:.6f}")
    print(f"N Features    : clean={best.user_attrs.get('n_feat_clean')}  "
          f"selected={best.user_attrs.get('n_feat_selected')}")
    print(f"Best agg_funcs: {best.user_attrs.get('agg_funcs')}")
    if "top_k" in best.params:
        print(f"Top-K         : {best.params['top_k']}")
    if "clf_filter_threshold" in best.params:
        print(f"CLF filter th : {best.params['clf_filter_threshold']:.3f}")
    if "zero_clip_threshold" in best.params:
        print(f"Zero clip th  : {best.params['zero_clip_threshold']:.4f}")
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
    clf_filter_threshold_fixed=0.5,
    zero_clip_threshold_fixed=0.0,
    clf_fixed=None,
    reg_fixed=None,
    target_transform_fn=None,
    target_inverse_fn=None,
    use_sampling=False,
    sample_frac=1.0,
    exclude_cols=None,
    # ★ 2차 신규
    clf_models=None,                # list/tuple. None이면 (clf_model,)로 폴백
    reg_models=None,                # list/tuple. None이면 (reg_model,)로 폴백
    calibration=None,               # dict {'method':'isotonic','models':[...]}
    add_clf_proba_to_reg=False,     # A4: clf_proba_mean을 reg feature로 추가
    save_per_model_oof=False,       # True면 die-level OOF CSV 7개 저장
    oof_dir=None,                   # save_per_model_oof=True일 때 필수
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

    # ★ 2차: multi-model 정규화
    clf_models_eff = tuple(clf_models) if clf_models else (clf_model,)
    reg_models_eff = tuple(reg_models) if reg_models else (reg_model,)
    # multi-model은 kfold 모드에서만 지원 (single은 1개만 허용)
    if mode == "single" and (len(clf_models_eff) > 1 or len(reg_models_eff) > 1):
        raise ValueError(
            "multi-model (len>1) rerun은 mode='kfold'에서만 지원. "
            "single 모드에선 clf_models/reg_models를 단일 원소 또는 None으로 주세요."
        )
    if save_per_model_oof and oof_dir is None:
        raise ValueError("save_per_model_oof=True면 oof_dir 필수.")

    # ── ⓪ 전처리 (best pp_params 복원) ──
    pp_params = extract_pp_params_from_best(best_params)
    cleaning_args, outlier_args, binarize_args, iso_args, lds_args, agg_funcs = split_pp_params(pp_params)

    print(f"Rerun preprocessing: cleaning={len(cleaning_args)} args, "
          f"outlier method={outlier_args.get('method')}, "
          f"binarize_apply={binarize_args.get('apply')}, "
          f"iso_enabled={iso_args.get('iso_enabled')}, "
          f"lds_enabled={lds_args.get('lds_enabled')}, "
          f"agg_funcs={agg_funcs}")
    pos_data, feat_cols_clean, sample_weight, scaler = _run_preprocessing(
        xs, xs_dict, ys, feat_cols,
        cleaning_args, outlier_args, binarize_args,
        iso_args, lds_args,
        label_col, exclude_cols,
        use_sampling, sample_frac,
    )
    print(f"Rerun preprocessing 완료: feat_cols={len(feat_cols_clean)}, "
          f"sample_weight={'die-level len=' + str(len(sample_weight)) if sample_weight is not None else 'None'}")

    # ── ① CLF (★ 2차: kfold에선 multi-model + Isotonic) ──
    per_model_clf = None    # kfold + multi에서만 채워짐
    if cfg["run_clf"]:
        if mode == "single":
            # single 모드는 단일 모델만 (multi는 위에서 이미 ValueError)
            single_clf = clf_models_eff[0]
            if cfg["clf_optuna"]:
                clf_params = _build_params_from_best(
                    best_params, clf_prefix, single_clf, clf_fixed
                )
            else:
                clf_params = get_default_params(single_clf, "clf")
                clf_params.update(clf_fixed)
            print(f"Rerun CLF: {single_clf}, mode=single (es_holdout={es_holdout})")
            clf_result = _run_clf_single(
                pos_data, feat_cols_clean, clf_params, single_clf,
                clf_early_stop, label_col, imbalance_method,
                clf_output=cfg["clf_output"], es_holdout=es_holdout,
            )
        else:
            # kfold: multi-model + Isotonic + soft voting
            clf_params_by_model = {}
            for m in clf_models_eff:
                if cfg["clf_optuna"]:
                    model_prefix = f"clf_{m}_" if len(clf_models_eff) > 1 else clf_prefix
                    p = _build_params_from_best(
                        best_params, model_prefix, m, clf_fixed
                    )
                else:
                    p = get_default_params(m, "clf")
                    p.update(clf_fixed)
                clf_params_by_model[m] = p

            print(f"Rerun CLF: {list(clf_models_eff)}, mode=kfold (folds={n_folds}), "
                  f"calibration={calibration.get('method') if calibration else None}")
            clf_result, per_model_clf = _run_clf_oof_multi(
                pos_data, feat_cols_clean, clf_params_by_model, clf_models_eff,
                n_folds, clf_early_stop, label_col, imbalance_method,
                calibration=calibration, clf_output=cfg["clf_output"],
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

    # ── ④ REG (★ 2차: kfold에선 multi-model + sample_weight + A4) ──
    per_model_reg = None
    # clf_filter_threshold 복원: best_params에 있으면 우선, 없으면 fixed
    clf_filter_threshold = best_params.get(
        "clf_filter_threshold", clf_filter_threshold_fixed
    )
    if cfg["clf_filter"]:
        print(f"Rerun clf_filter_threshold: {clf_filter_threshold}")

    if mode == "single":
        single_reg = reg_models_eff[0]
        if cfg["reg_optuna"]:
            reg_params = _build_params_from_best(
                best_params, reg_prefix, single_reg, reg_fixed
            )
        else:
            reg_params = get_default_params(single_reg, "reg")
            reg_params.update(reg_fixed)
        print(f"Rerun REG: {single_reg}, features={len(selected_cols)}, mode=single (es_holdout={es_holdout})")
        reg_result = _run_reg_single(
            unit_data, selected_cols, reg_params, single_reg,
            reg_early_stop,
            use_clf=cfg["run_clf"], clf_filter=cfg["clf_filter"],
            clf_filter_threshold=clf_filter_threshold,
            es_holdout=es_holdout,
            target_transform_fn=target_transform_fn,
            target_inverse_fn=target_inverse_fn,
        )
    else:
        # kfold: multi-model + sample_weight + A4
        reg_params_by_model = {}
        for m in reg_models_eff:
            if cfg["reg_optuna"]:
                model_prefix = f"reg_{m}_" if len(reg_models_eff) > 1 else reg_prefix
                p = _build_params_from_best(
                    best_params, model_prefix, m, reg_fixed
                )
            else:
                p = get_default_params(m, "reg")
                p.update(reg_fixed)
            reg_params_by_model[m] = p

        print(f"Rerun REG: {list(reg_models_eff)}, features={len(selected_cols)}, "
              f"mode=kfold (folds={n_folds}), "
              f"sample_weight={'die-level len=' + str(len(sample_weight)) if sample_weight is not None else 'None'}, "
              f"add_clf_proba_to_reg={add_clf_proba_to_reg}")
        reg_result, per_model_reg = _run_reg_oof_multi(
            unit_data, selected_cols, reg_params_by_model, reg_models_eff,
            n_folds, reg_early_stop,
            use_clf=cfg["run_clf"], clf_filter=cfg["clf_filter"],
            clf_filter_threshold=clf_filter_threshold,
            target_transform_fn=target_transform_fn,
            target_inverse_fn=target_inverse_fn,
            sample_weight=sample_weight,
            add_clf_proba_to_reg=add_clf_proba_to_reg,
        )

    # 모델 객체 추출 (SHAP / feature importance 사후 분석용)
    # _die_pred_to_unit / _apply_zero_clip이 reg_result를 교체하기 전에 먼저 보관.
    if mode == "kfold":
        if per_model_reg is not None:
            # multi-model: 각 모델의 fold_models 보관
            per_model_fold_models = {
                m: per_model_reg[m].get("fold_models", []) for m in reg_models_eff
            }
            # 앙상블 기준 fold_models는 없음. 대신 리스트를 flatten해서 기존 "reg_models" 키에 채움
            reg_fold_models_flat = []
            for m in reg_models_eff:
                reg_fold_models_flat.extend(per_model_reg[m].get("fold_models", []))
        else:
            # single-model fallback
            reg_fold_models_flat = reg_result.get("fold_models", [])
            per_model_fold_models = {reg_models_eff[0]: reg_fold_models_flat}
    else:
        _single_model = reg_result.get("model")
        reg_fold_models_flat = [_single_model] if _single_model is not None else []
        per_model_fold_models = {reg_models_eff[0]: reg_fold_models_flat}

    # position 모드: die → unit 집계
    # die_data: SHAP/Permutation Importance 등 사후 분석용으로 die-level
    # feature matrix (val/test) 를 slim copy 로 보관. agg 이후 unit_data는
    # [KEY_COL, TARGET_COL] 만 남아 feature 컬럼이 사라지므로 미리 떠 둔다.
    die_data = None
    # ★ 2차: per-model unit-level val/test RMSE 계산 (die→unit 집계 전에 per_model도 집계)
    per_model_val_rmse = {}
    per_model_test_rmse = {}
    # OOF 저장용으로 unit_data 집계 전 die-level 형태를 보관
    unit_data_die = unit_data if cfg["reg_level"] == "position" else None

    if cfg["reg_level"] == "position":
        _keep_cols = [KEY_COL, TARGET_COL] + list(selected_cols)
        die_data = {}
        for _split in ("val", "test"):
            _cols = [c for c in _keep_cols if c in unit_data[_split].columns]
            die_data[_split] = unit_data[_split][_cols].copy()

        # per-model RMSE를 집계 전에 계산 (각 모델의 die-level pred → unit mean)
        if per_model_reg is not None and mode == "kfold":
            for m in reg_models_eff:
                r_m = per_model_reg[m]
                _fake = {
                    "oof_pred":  r_m["oof_pred"],
                    "val_pred":  r_m["val_pred"],
                    "test_pred": r_m["test_pred"],
                    "train_rmse": r_m["train_rmse"],
                }
                _unit_pred_m, _agg_ud_m = _die_pred_to_unit(unit_data, _fake)
                try:
                    _y_val_m = _agg_ud_m["val"].groupby(KEY_COL, sort=False)[TARGET_COL].first().values
                    per_model_val_rmse[m] = float(rmse(_y_val_m, _unit_pred_m["val_pred"]))
                except Exception:
                    per_model_val_rmse[m] = None
                try:
                    _y_test_m = _agg_ud_m["test"].groupby(KEY_COL, sort=False)[TARGET_COL].first().values
                    per_model_test_rmse[m] = float(rmse(_y_test_m, _unit_pred_m["test_pred"]))
                except Exception:
                    per_model_test_rmse[m] = None

        reg_result, agg_unit_data = _die_pred_to_unit(unit_data, reg_result)
        unit_data = agg_unit_data
    else:
        # reg_level='unit'이면 per_model val/test RMSE는 unit-level 그대로
        if per_model_reg is not None and mode == "kfold":
            try:
                _y_val = unit_data["val"][TARGET_COL].values
                for m in reg_models_eff:
                    per_model_val_rmse[m] = float(rmse(_y_val, per_model_reg[m]["val_pred"]))
            except Exception:
                for m in reg_models_eff:
                    per_model_val_rmse[m] = None
            try:
                _y_test = unit_data["test"][TARGET_COL].values
                for m in reg_models_eff:
                    per_model_test_rmse[m] = float(rmse(_y_test, per_model_reg[m]["test_pred"]))
            except Exception:
                for m in reg_models_eff:
                    per_model_test_rmse[m] = None

    # zero_clip: best_params에 있으면 우선, 없으면 fixed
    if cfg["zero_clip"]:
        zero_clip_threshold = best_params.get(
            "zero_clip_threshold", zero_clip_threshold_fixed
        )
        print(f"Rerun zero_clip_threshold: {zero_clip_threshold}")
        y_train_for_clip = unit_data["train"][TARGET_COL].values
        reg_result = _apply_zero_clip(
            reg_result, y_train_for_clip, zero_clip_threshold
        )

    oof_rmse_score = reg_result["train_rmse"]
    print(f"Rerun OOF RMSE: {oof_rmse_score:.6f}")

    # val RMSE는 참고용
    val_rmse_score = None
    try:
        if cfg["reg_level"] == "position":
            y_val = unit_data["val"].groupby(KEY_COL, sort=False)[TARGET_COL].first().values
        else:
            y_val = unit_data["val"][TARGET_COL].values
        val_rmse_score = rmse(y_val, reg_result["val_pred"])
        print(f"Rerun Val RMSE: {val_rmse_score:.6f}  (참고용)")
    except Exception:
        pass

    # ── ★ 2차: OOF CSV 7개 저장 (save_per_model_oof=True + kfold + multi) ──
    oof_files = []
    if save_per_model_oof and mode == "kfold" and per_model_reg is not None:
        print(f"[save_per_model_oof] → {oof_dir}")
        oof_files = _save_per_model_oof(
            per_model_clf=per_model_clf,
            per_model_reg=per_model_reg,
            clf_result_soft=clf_result,
            sample_weight=sample_weight,
            pos_data=pos_data,
            unit_data=unit_data_die,    # die-level concat (집계 전)
            oof_dir=oof_dir,
            clf_models=list(clf_models_eff),
            reg_models=list(reg_models_eff),
        )
        print(f"  saved {len(oof_files)} files")

    result = {
        "unit_data": unit_data,
        "unit_data_die": unit_data_die,   # die-level concat (reg_level='position'용, 집계 전)
        "die_data": die_data,
        "selected_cols": selected_cols,
        "feat_cols_clean": feat_cols_clean,
        "val_pred": reg_result["val_pred"],
        "test_pred": reg_result["test_pred"],
        "oof_rmse": oof_rmse_score,
        "val_rmse": val_rmse_score,
        "clf_result": clf_result,
        "importances": sel_importances,
        "pipeline_config": cfg,
        "best_pp_params": pp_params,
        "cleaning_args": cleaning_args,
        "outlier_args": outlier_args,
        "agg_funcs": agg_funcs,
        "reg_models": reg_fold_models_flat,   # 앙상블 flat 리스트 (하위호환)
        # ★ 2차 신규 필드
        "per_model_fold_models": per_model_fold_models,
        "per_model_clf_fold_models": None,  # TODO: CLF fold_models 보관 원하면 _run_clf_oof 확장 필요
        "per_model_val_rmse": per_model_val_rmse,
        "per_model_test_rmse": per_model_test_rmse,
        "per_model_reg": per_model_reg,
        "per_model_clf": per_model_clf,
        "scaler": scaler,
        "sample_weight": sample_weight,
        "iso_args": iso_args,
        "lds_args": lds_args,
        "oof_files": oof_files,
    }
    if mode == "kfold":
        result["oof_pred"] = reg_result["oof_pred"]
    else:
        result["train_pred_insample"] = reg_result["train_pred_insample"]
    return result


# ═════════════════════════════════════════════════════════════
# OOF CSV 저장 (★ 2차 신규) — 7개 파일 (meta 1 + clf 3 + reg 3)
# ═════════════════════════════════════════════════════════════
def _save_per_model_oof(per_model_clf, per_model_reg, clf_result_soft,
                        sample_weight, pos_data, unit_data,
                        oof_dir, clf_models, reg_models):
    """
    die-level OOF 결과를 CSV로 저장 (총 7개).

    - oof_meta.csv           : ufs_serial, position, split, y_true, clf_proba_mean, lds_weight
    - oof_clf_{model}.csv × N: ufs_serial, position, split, y_true, clf_proba       (N=len(clf_models))
    - oof_reg_{model}.csv × N: ufs_serial, position, split, y_true, reg_pred_die    (N=len(reg_models))

    Parameters
    ----------
    per_model_clf : dict or None
        {model: {pos: {'train_proba','val_proba','test_proba'}}}
        _run_clf_oof_multi의 per_model 반환값. None이면 (run_clf=False) clf CSV 생성 스킵.
    per_model_reg : dict {model: {'oof_pred','val_pred','test_pred', ...}}
        _run_reg_oof_multi의 per_model 반환값 (reg_level='position' die-level concat 순서)
    clf_result_soft : dict or None
        {pos: {'train_proba','val_proba','test_proba'}}
        _run_clf_oof_multi의 soft-voted 반환값. None이면 meta의 clf_proba_mean은 NaN.
    sample_weight : np.ndarray or None
        train die-level 가중치 (LDS). None이면 meta의 lds_weight는 전부 NaN.
    pos_data : dict {pos: {'train','val','test'}}
        die-level, position별 분리 (train 순서 결정용)
    unit_data : dict {'train','val','test'}
        reg_level='position'이면 die-level concat (reg CSV 행 순서 기준)
    oof_dir : str
    clf_models, reg_models : list

    Returns
    -------
    saved_files : list[str] — 저장된 7개 CSV 경로
    """
    os.makedirs(oof_dir, exist_ok=True)
    saved_files = []

    # ── oof_meta.csv ──
    # train 구간의 sample_weight는 pos 1→2→3→4 순으로 concat된 위치에 매칭.
    train_pos_offsets = {}
    offset = 0
    for p in sorted(pos_data.keys()):
        train_pos_offsets[p] = offset
        offset += len(pos_data[p]["train"])

    meta_rows = []
    for split in ["train", "val", "test"]:
        for pos in sorted(pos_data.keys()):
            df = pos_data[pos][split]
            n = len(df)
            # LDS weight: train만 기록, 해당 position 구간 슬라이싱
            if split == "train" and sample_weight is not None:
                start = train_pos_offsets[pos]
                w_slice = sample_weight[start:start + n]
            else:
                w_slice = np.full(n, np.nan)

            clf_proba = (clf_result_soft[pos][f"{split}_proba"]
                         if clf_result_soft is not None
                         else np.full(n, np.nan))
            meta_rows.append(pd.DataFrame({
                KEY_COL:          df[KEY_COL].values,
                POSITION_COL:     pos,
                "split":          split,
                TARGET_COL:       df[TARGET_COL].values,
                "clf_proba_mean": clf_proba,
                "lds_weight":     w_slice,
            }))
    meta_df = pd.concat(meta_rows, ignore_index=True)
    meta_path = os.path.join(oof_dir, "oof_meta.csv")
    meta_df.to_csv(meta_path, index=False)
    saved_files.append(meta_path)

    # ── oof_clf_{model}.csv × N (run_clf=False면 스킵) ──
    if per_model_clf is not None:
        for m in clf_models:
            rows = []
            for split in ["train", "val", "test"]:
                for pos in sorted(pos_data.keys()):
                    df = pos_data[pos][split]
                    rows.append(pd.DataFrame({
                        KEY_COL:      df[KEY_COL].values,
                        POSITION_COL: pos,
                        "split":      split,
                        TARGET_COL:   df[TARGET_COL].values,
                        "clf_proba":  per_model_clf[m][pos][f"{split}_proba"],
                    }))
            clf_path = os.path.join(oof_dir, f"oof_clf_{m}.csv")
            pd.concat(rows, ignore_index=True).to_csv(clf_path, index=False)
            saved_files.append(clf_path)

    # ── oof_reg_{model}.csv × N ──
    # unit_data의 split별 row 순서 = per_model_reg[m][*_pred] 순서와 동일
    # (reg_level='position'에서 _prepare_unit_data가 position concat하여 만듦)
    for m in reg_models:
        r = per_model_reg[m]
        parts = []
        for split_name, pred_key in [("train", "oof_pred"),
                                     ("val",   "val_pred"),
                                     ("test",  "test_pred")]:
            d = unit_data[split_name]
            parts.append(pd.DataFrame({
                KEY_COL:        d[KEY_COL].values,
                POSITION_COL:   d[POSITION_COL].values,
                "split":        split_name,
                TARGET_COL:     d[TARGET_COL].values,
                "reg_pred_die": r[pred_key],
            }))
        reg_path = os.path.join(oof_dir, f"oof_reg_{m}.csv")
        pd.concat(parts, ignore_index=True).to_csv(reg_path, index=False)
        saved_files.append(reg_path)

    return saved_files


# ═════════════════════════════════════════════════════════════
# Rerun → Study append (★ 2차 신규) — DB 정공
# ═════════════════════════════════════════════════════════════
def add_rerun_to_study(study, best_params, rerun_value, user_attrs=None):
    """
    Rerun 결과를 study에 FrozenTrial로 추가 (같은 DB에 append).

    Parameters
    ----------
    study : optuna.Study
        main study (SQLite backed)
    best_params : dict
        main study의 best_params — 기존 trial distribution에 맞는 key만 사용
    rerun_value : float
        rerun 결과 objective (예: val RMSE)
    user_attrs : dict or None
        기본 {'is_rerun': True}. 추가 키 전달 가능.

    Notes
    -----
    - main study에 trial이 하나 이상 완료돼 있어야 distribution 추출 가능
    - best_params에 distribution에 없는 key가 있으면 조용히 무시 (안전)
    - Optuna viewer에서 `is_rerun=True` user_attr로 필터 가능
    """
    if len(study.trials) == 0:
        raise ValueError("study has no trials; cannot extract distributions for add_trial.")

    # 가장 최근 trial의 distribution을 재사용
    distributions = study.trials[-1].distributions
    valid_params = {k: v for k, v in best_params.items() if k in distributions}

    trial = optuna.trial.create_trial(
        params=valid_params,
        distributions={k: distributions[k] for k in valid_params},
        value=rerun_value,
        user_attrs=user_attrs or {"is_rerun": True},
    )
    study.add_trial(trial)
    print(f"[add_rerun_to_study] trial 추가 완료 "
          f"(총 {len(study.trials)} trials, value={rerun_value:.6f})")
