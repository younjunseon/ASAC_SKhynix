"""
앙상블 모듈 (보팅 + 스태킹)

사용법:
    from modules.ensemble import run_voting, run_stacking

    voting_result = run_voting(multi_results, y_val, y_test,
                                optimize_weights=True)
    stacking_result = run_stacking(multi_results, unit_data, unit_feat_cols,
                                    y_val, y_test)
"""
import numpy as np
import optuna
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from utils.config import SEED
from utils.evaluate import rmse, postprocess, evaluate


# ─── 보팅 (가중 평균) ────────────────────────────────────────
def run_voting(
    multi_results,
    y_val,
    y_test,
    optimize_weights=True,
    n_trials=200,
    voting_params=None,
):
    """
    다중 모델 예측값의 가중 평균

    Parameters
    ----------
    multi_results : dict
        {모델명: {"val_pred": array, "test_pred": array, ...}}
        run_multi_model_comparison()의 반환값
    y_val, y_test : array-like
        실제 target
    optimize_weights : bool
        True: Optuna로 최적 가중치 탐색
        False: val RMSE 역수 비례 가중치
    n_trials : int
        Optuna 탐색 횟수
    voting_params : dict, optional
        추가 파라미터 (향후 확장용)

    Returns
    -------
    dict : {"val_pred", "test_pred", "val_rmse", "test_rmse", "weights"}
    """
    model_names = list(multi_results.keys())
    val_preds = np.array([multi_results[m]["val_pred"] for m in model_names])
    test_preds = np.array([multi_results[m]["test_pred"] for m in model_names])

    if optimize_weights:
        weights = _optimize_voting_weights(val_preds, y_val, model_names, n_trials)
    else:
        # val RMSE 역수 비례
        val_rmses = np.array([multi_results[m]["val_rmse"] for m in model_names])
        inv_rmse = 1.0 / val_rmses
        weights = inv_rmse / inv_rmse.sum()

    # 가중 평균 예측
    val_pred = postprocess(np.average(val_preds, axis=0, weights=weights))
    test_pred = postprocess(np.average(test_preds, axis=0, weights=weights))

    val_rmse_val = evaluate(np.asarray(y_val), val_pred, label="Voting val")
    test_rmse_val = evaluate(np.asarray(y_test), test_pred, label="Voting test")

    # 가중치 출력
    print("\n보팅 가중치:")
    for name, w in zip(model_names, weights):
        print(f"  {name}: {w:.4f}")

    return {
        "val_pred": val_pred,
        "test_pred": test_pred,
        "val_rmse": val_rmse_val,
        "test_rmse": test_rmse_val,
        "weights": dict(zip(model_names, weights)),
        "method": "voting",
    }


def _optimize_voting_weights(val_preds, y_val, model_names, n_trials=200):
    """Optuna로 보팅 가중치 최적화"""
    y_val = np.asarray(y_val)

    def objective(trial):
        raw_weights = [
            trial.suggest_float(f"w_{name}", 0.0, 1.0)
            for name in model_names
        ]
        total = sum(raw_weights)
        if total < 1e-8:
            return float("inf")
        weights = np.array(raw_weights) / total
        pred = postprocess(np.average(val_preds, axis=0, weights=weights))
        return rmse(y_val, pred)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    raw = [study.best_params[f"w_{name}"] for name in model_names]
    total = sum(raw)
    weights = np.array(raw) / total

    print(f"Optuna 보팅 최적화: RMSE={study.best_value:.6f} ({n_trials} trials)")
    return weights


# ─── 스태킹 ──────────────────────────────────────────────────
def run_stacking(
    multi_results,
    unit_data,
    unit_feat_cols,
    y_val,
    y_test,
    stacking_params=None,
):
    """
    Level 0: 다중 모델의 OOF 예측 + unit feature
    Level 1: Ridge meta-learner

    Parameters
    ----------
    multi_results : dict
        {모델명: {"oof_pred": array, "val_pred": array, "test_pred": array, ...}}
    unit_data : dict
        {"train": df, "val": df, "test": df}
    unit_feat_cols : list
    y_val, y_test : array-like
    stacking_params : dict, optional
        {"meta_model": "ridge", "meta_alpha": 1.0, "use_features": True,
         "n_folds": 5}

    Returns
    -------
    dict : {"val_pred", "test_pred", "val_rmse", "test_rmse"}
    """
    params = stacking_params or {}
    meta_alpha = params.get("meta_alpha", 1.0)
    use_features = params.get("use_features", False)
    n_folds = params.get("n_folds", 5)

    model_names = list(multi_results.keys())
    y_val = np.asarray(y_val)
    y_test = np.asarray(y_test)

    # --- Level 0 OOF 예측값 행렬 ---
    # train: OOF 예측
    oof_matrix_train = np.column_stack([
        multi_results[m]["oof_pred"] for m in model_names
    ])
    # val/test: 각 모델의 예측
    oof_matrix_val = np.column_stack([
        multi_results[m]["val_pred"] for m in model_names
    ])
    oof_matrix_test = np.column_stack([
        multi_results[m]["test_pred"] for m in model_names
    ])

    # --- unit feature 결합 (선택) ---
    if use_features:
        from utils.config import TARGET_COL, KEY_COL
        train_feats = unit_data["train"][unit_feat_cols].values
        val_feats = unit_data["val"][unit_feat_cols].values
        test_feats = unit_data["test"][unit_feat_cols].values

        X_train_stack = np.hstack([oof_matrix_train, train_feats])
        X_val_stack = np.hstack([oof_matrix_val, val_feats])
        X_test_stack = np.hstack([oof_matrix_test, test_feats])
    else:
        X_train_stack = oof_matrix_train
        X_val_stack = oof_matrix_val
        X_test_stack = oof_matrix_test

    from utils.config import TARGET_COL
    y_train = unit_data["train"][TARGET_COL].values

    print(f"스태킹 Level 1 입력: {X_train_stack.shape[1]}개 "
          f"(모델 {len(model_names)}개"
          f"{' + feature ' + str(len(unit_feat_cols)) if use_features else ''})")

    # --- Level 1: Ridge K-Fold ---
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof_meta = np.zeros(len(y_train))
    val_preds_list = []
    test_preds_list = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train_stack)):
        meta = Ridge(alpha=meta_alpha, random_state=SEED)
        meta.fit(X_train_stack[tr_idx], y_train[tr_idx])
        oof_meta[va_idx] = meta.predict(X_train_stack[va_idx])
        val_preds_list.append(meta.predict(X_val_stack))
        test_preds_list.append(meta.predict(X_test_stack))

    val_pred = postprocess(np.mean(val_preds_list, axis=0))
    test_pred = postprocess(np.mean(test_preds_list, axis=0))

    val_rmse_val = evaluate(y_val, val_pred, label="Stacking val")
    test_rmse_val = evaluate(y_test, test_pred, label="Stacking test")

    return {
        "val_pred": val_pred,
        "test_pred": test_pred,
        "val_rmse": val_rmse_val,
        "test_rmse": test_rmse_val,
        "method": "stacking",
    }