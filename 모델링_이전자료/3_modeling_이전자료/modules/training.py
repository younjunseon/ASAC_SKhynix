"""
학습 파이프라인 모듈

- run_classification: Position별 이진분류 (OOF)
- run_twostage_regression: Y>0 서브셋 회귀 → proba * reg_pred
- run_single_regression: 전체 데이터 회귀
- run_multi_model_comparison: 다중 모델 비교 (Stage 7-A)

사용법:
    from modules.training import run_classification, run_twostage_regression

    clf_result = run_classification(pos_data, feat_cols, clf_params,
                                     model_name="lgbm")
    reg_result = run_twostage_regression(unit_data, unit_feat_cols, reg_params,
                                          model_name="lgbm")
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

from utils.config import SEED, TARGET_COL, KEY_COL
from modules.model_zoo import (
    create_model, fit_model, get_best_iteration, supports_early_stopping,
)
from utils.evaluate import evaluate, postprocess


# ─── Stage 1: Position별 이진분류 (OOF) ──────────────────────
def run_classification(
    pos_data,
    feat_cols,
    clf_params,
    model_name="lgbm",
    n_folds=5,
    early_stop=50,
    label_col="label_bin",
    clf_threshold=0.5,
    imbalance_method="none",
    imbalance_params=None,
):
    """
    Position별 이진분류 OOF → P(Y>0) 확률 반환

    Parameters
    ----------
    pos_data : dict
        {position: {"train": df, "val": df, "test": df}}
    feat_cols : list
        피처 컬럼명
    clf_params : dict
        분류기 하이퍼파라미터
    model_name : str
        모델명 ("lgbm", "xgb", "catboost", "rf", "et")
    n_folds : int
        OOF fold 수
    early_stop : int
        early stopping patience
    label_col : str
        이진 라벨 컬럼명
    clf_threshold : float
        accuracy 리포트용 임계값
    imbalance_method : str
        클래스 불균형 처리 방법:
        - "none": 처리 안 함
        - "scale_pos_weight": 양성 클래스 가중치 자동 계산
        - "smote": SMOTE 오버샘플링
    imbalance_params : dict, optional
        불균형 처리 추가 파라미터 (smote_ratio 등)

    Returns
    -------
    dict : {position: {"train_proba", "val_proba", "test_proba"}}
    """
    all_results = {}

    for pos in sorted(pos_data.keys()):
        d = pos_data[pos]
        X_tr = d["train"][feat_cols].values
        y_tr = d["train"][label_col].values
        X_val = d["val"][feat_cols].values
        X_test = d["test"][feat_cols].values

        # --- 클래스 불균형 처리: scale_pos_weight ---
        actual_clf_params = clf_params.copy()
        if imbalance_method == "scale_pos_weight":
            n_neg = (y_tr == 0).sum()
            n_pos = (y_tr == 1).sum()
            spw = n_neg / n_pos if n_pos > 0 else 1.0

            if model_name == "lgbm":
                actual_clf_params["scale_pos_weight"] = spw
            elif model_name == "xgb":
                actual_clf_params["scale_pos_weight"] = spw
            elif model_name == "catboost":
                actual_clf_params["auto_class_weights"] = "Balanced"
            elif model_name in ("rf", "et"):
                actual_clf_params["class_weight"] = "balanced"

            if pos == sorted(pos_data.keys())[0]:
                print(f"  [불균형 처리] scale_pos_weight={spw:.3f} "
                      f"(neg={n_neg:,}, pos={n_pos:,})")

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        oof_proba = np.zeros(len(X_tr))
        fold_models = []

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr, y_tr)):
            X_fold_tr, y_fold_tr = X_tr[tr_idx], y_tr[tr_idx]
            X_fold_val, y_fold_val = X_tr[val_idx], y_tr[val_idx]

            # --- SMOTE ---
            if imbalance_method == "smote":
                try:
                    from imblearn.over_sampling import SMOTE
                except ImportError:
                    raise ImportError(
                        "SMOTE requires imbalanced-learn. "
                        "Install: pip install imbalanced-learn"
                    )
                smote_kw = imbalance_params or {}
                smote = SMOTE(random_state=SEED, **smote_kw)
                X_fold_tr, y_fold_tr = smote.fit_resample(X_fold_tr, y_fold_tr)

            clf = create_model(model_name, "clf", actual_clf_params)

            if supports_early_stopping(model_name):
                fit_model(clf, X_fold_tr, y_fold_tr,
                          X_fold_val, y_fold_val, early_stop)
            else:
                fit_model(clf, X_fold_tr, y_fold_tr)

            oof_proba[val_idx] = clf.predict_proba(X_tr[val_idx])[:, 1]
            fold_models.append(clf)

        # --- 전체 fold 평균으로 val/test 예측 ---
        train_proba = oof_proba
        val_proba = np.mean(
            [m.predict_proba(X_val)[:, 1] for m in fold_models], axis=0
        )
        test_proba = np.mean(
            [m.predict_proba(X_test)[:, 1] for m in fold_models], axis=0
        )

        train_acc = ((train_proba > clf_threshold).astype(int) == y_tr).mean()
        val_acc = ((val_proba > clf_threshold).astype(int)
                   == d["val"][label_col].values).mean()
        best_iter = get_best_iteration(fold_models[-1])
        iter_info = f", best_iter={best_iter}" if best_iter else ""
        print(f"  Position {pos}: train_acc={train_acc:.4f}, "
              f"val_acc={val_acc:.4f}{iter_info}")

        all_results[pos] = {
            "train_proba": train_proba,
            "val_proba": val_proba,
            "test_proba": test_proba,
            "models": fold_models,
        }

    return all_results


# ─── Stage 2-A: True Two-Stage (Y>0만 회귀) ──────────────────
def run_twostage_regression(
    unit_data,
    feat_cols,
    reg_params,
    model_name="lgbm",
    n_folds=5,
    early_stop=50,
):
    """
    True Two-Stage: Y>0 서브셋에서 회귀 학습 → proba * reg_pred

    Parameters
    ----------
    unit_data : dict
        {"train": df, "val": df, "test": df}
    feat_cols : list
        피처 컬럼명 (clf_proba_mean 포함)
    reg_params : dict
        회귀기 하이퍼파라미터
    model_name : str
    n_folds : int
    early_stop : int

    Returns
    -------
    dict : {"val_pred", "test_pred", "oof_pred", "models"}
    """
    X_train = unit_data["train"][feat_cols].values
    y_train = unit_data["train"][TARGET_COL].values
    X_val = unit_data["val"][feat_cols].values
    X_test = unit_data["test"][feat_cols].values

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof_reg = np.zeros(len(X_train))
    fold_models = []

    print(f"  Train: {len(X_train):,}건 (Y>0: {(y_train > 0).sum():,}건)")

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
        pos_tr = y_train[tr_idx] > 0
        pos_va = y_train[va_idx] > 0

        reg = create_model(model_name, "reg", reg_params)

        if supports_early_stopping(model_name) and pos_va.sum() > 0:
            fit_model(reg, X_train[tr_idx][pos_tr], y_train[tr_idx][pos_tr],
                      X_train[va_idx][pos_va], y_train[va_idx][pos_va],
                      early_stop)
        else:
            fit_model(reg, X_train[tr_idx][pos_tr], y_train[tr_idx][pos_tr])

        oof_reg[va_idx] = reg.predict(X_train[va_idx])
        fold_models.append(reg)

        best_iter = get_best_iteration(reg)
        iter_info = f", best_iter={best_iter}" if best_iter else ""
        print(f"  Fold {fold+1}: train_pos={pos_tr.sum():,}, "
              f"val_pos={pos_va.sum():,}{iter_info}")

    val_reg = np.mean([m.predict(X_val) for m in fold_models], axis=0)
    test_reg = np.mean([m.predict(X_test) for m in fold_models], axis=0)

    proba_train = unit_data["train"]["clf_proba_mean"].values
    proba_val = unit_data["val"]["clf_proba_mean"].values
    proba_test = unit_data["test"]["clf_proba_mean"].values

    return {
        "val_pred": postprocess(proba_val * val_reg),
        "test_pred": postprocess(proba_test * test_reg),
        "oof_pred": postprocess(proba_train * oof_reg),
        "models": fold_models,
        "val_reg_raw": val_reg,
        "test_reg_raw": test_reg,
        "oof_reg_raw": oof_reg,
    }


# ─── Stage 2-B: Single Regression (전체 데이터) ──────────────
def run_single_regression(
    unit_data,
    feat_cols,
    reg_params,
    model_name="lgbm",
    n_folds=5,
    early_stop=50,
):
    """
    Single regression: 전체 데이터로 학습, clf_proba_mean을 feature로 포함

    Parameters
    ----------
    (run_twostage_regression과 동일)

    Returns
    -------
    dict : {"val_pred", "test_pred", "oof_pred", "models"}
    """
    X_train = unit_data["train"][feat_cols].values
    y_train = unit_data["train"][TARGET_COL].values
    X_val = unit_data["val"][feat_cols].values
    X_test = unit_data["test"][feat_cols].values

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof_pred = np.zeros(len(X_train))
    fold_models = []

    print(f"  Train: {len(X_train):,}건")

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
        reg = create_model(model_name, "reg", reg_params)

        if supports_early_stopping(model_name):
            fit_model(reg, X_train[tr_idx], y_train[tr_idx],
                      X_train[va_idx], y_train[va_idx], early_stop)
        else:
            fit_model(reg, X_train[tr_idx], y_train[tr_idx])

        oof_pred[va_idx] = reg.predict(X_train[va_idx])
        fold_models.append(reg)

        best_iter = get_best_iteration(reg)
        iter_info = f", best_iter={best_iter}" if best_iter else ""
        print(f"  Fold {fold+1}{iter_info}")

    val_pred = postprocess(
        np.mean([m.predict(X_val) for m in fold_models], axis=0)
    )
    test_pred = postprocess(
        np.mean([m.predict(X_test) for m in fold_models], axis=0)
    )

    return {
        "val_pred": val_pred,
        "test_pred": test_pred,
        "oof_pred": postprocess(oof_pred),
        "models": fold_models,
    }


# ─── 다중 모델 비교 (Stage 7-A) ──────────────────────────────
def run_multi_model_comparison(
    unit_data,
    feat_cols,
    model_configs,
    mode="twostage",
    n_folds=5,
    early_stop=50,
    pos_data=None,
    clf_configs=None,
    agg_func=None,
    clf_threshold=0.5,
    imbalance_method="none",
    imbalance_params=None,
    label_col="label_bin",
):
    """
    다중 모델을 순회하며 Stage 2 결과를 비교

    Parameters
    ----------
    unit_data : dict
        {"train": df, "val": df, "test": df} — 이미 집계된 unit-level 데이터
    feat_cols : list
        unit-level 피처 컬럼명
    model_configs : dict
        {모델명: {"params": dict}} — Stage 2 회귀 모델 설정
        예: {"lgbm": {"params": lgbm_reg_params}, "xgb": {"params": xgb_reg_params}}
    mode : str
        "twostage" 또는 "single"
    n_folds, early_stop : int
    pos_data, clf_configs, agg_func : optional
        Stage 1부터 다시 돌릴 경우. None이면 unit_data에 이미 clf_proba_mean이 있다고 가정
    clf_threshold, imbalance_method, imbalance_params, label_col :
        Stage 1 파라미터 (pos_data가 있을 때만 사용)

    Returns
    -------
    results : dict
        {모델명: {"val_pred", "test_pred", "oof_pred", "models", "val_rmse"}}
    comparison_df : DataFrame
        모델별 val RMSE 정렬 표
    """
    y_val = unit_data["val"][TARGET_COL].values
    results = {}

    for name, config in model_configs.items():
        reg_params = config["params"]
        print(f"\n--- {name} ---")

        if mode == "twostage":
            result = run_twostage_regression(
                unit_data, feat_cols, reg_params,
                model_name=name, n_folds=n_folds, early_stop=early_stop,
            )
        else:
            result = run_single_regression(
                unit_data, feat_cols, reg_params,
                model_name=name, n_folds=n_folds, early_stop=early_stop,
            )

        val_rmse = evaluate(y_val, result["val_pred"], label=f"{name} val")
        result["val_rmse"] = val_rmse
        results[name] = result

    # --- 비교표 ---
    rows = [{"model": n, "val_rmse": r["val_rmse"]} for n, r in results.items()]
    comparison_df = pd.DataFrame(rows).sort_values("val_rmse").reset_index(drop=True)
    comparison_df.index += 1

    print(f"\n{'=' * 50}")
    print("다중 모델 비교 결과 (val RMSE)")
    print("=" * 50)
    print(comparison_df.to_string())

    return results, comparison_df