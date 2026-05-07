"""
HPO (하이퍼파라미터 최적화) 모듈 — Optuna 기반

- optimize_clf: Stage 1 분류기 HPO
- optimize_reg: Stage 2 회귀기 HPO
- optimize_fs: Feature Selection 파라미터 HPO

사용법:
    from modules.hpo import optimize_clf, optimize_reg

    best_clf_params = optimize_clf(
        pos_data, feat_cols, model_name="lgbm", n_trials=50,
    )
    best_reg_params = optimize_reg(
        unit_data, feat_cols, model_name="lgbm", n_trials=100,
    )
"""
import numpy as np
import optuna
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, KFold

from utils.config import SEED, TARGET_COL, KEY_COL
from utils.evaluate import rmse, postprocess
from modules.model_zoo import create_model, fit_model, supports_early_stopping, DEVICE


# ─── 모델별 탐색 공간 정의 ────────────────────────────────────
def _lgbm_search_space(trial, task):
    """LightGBM 탐색 공간"""
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 3000),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        num_leaves=trial.suggest_int("num_leaves", 8, 256),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 300),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.1, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        min_split_gain=trial.suggest_float("min_split_gain", 1e-8, 1.0, log=True),
        path_smooth=trial.suggest_float("path_smooth", 0.0, 50.0),
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
        device=DEVICE,
    )
    return params


def _xgb_search_space(trial, task):
    """XGBoost 탐색 공간"""
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 3000),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 300),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.1, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        gamma=trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        tree_method="hist",
        device="cuda" if DEVICE == "gpu" else "cpu",
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )
    return params


def _catboost_search_space(trial, task):
    """CatBoost 탐색 공간

    task_type='CPU' 강제: Colab T4 GPU에서 LGBM과 GPU 메모리 경합으로 OOM 발생하여 CPU로 고정.
    """
    params = dict(
        iterations=trial.suggest_int("iterations", 100, 3000),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        depth=trial.suggest_int("depth", 3, 10),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 5, 300),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bylevel=trial.suggest_float("colsample_bylevel", 0.1, 1.0),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
        task_type="CPU",
        random_seed=SEED,
        verbose=0,
    )
    return params


def _rf_search_space(trial, task):
    """RandomForest 탐색 공간"""
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 1000),
        max_depth=trial.suggest_int("max_depth", 5, 30),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 50),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 50),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]),
        random_state=SEED,
        n_jobs=-1,
    )
    return params


def _et_search_space(trial, task):
    """ExtraTrees 탐색 공간"""
    return _rf_search_space(trial, task)  # 동일 공간


SEARCH_SPACES = {
    "lgbm": _lgbm_search_space,
    "xgb": _xgb_search_space,
    "catboost": _catboost_search_space,
    "rf": _rf_search_space,
    "et": _et_search_space,
}


# ─── Stage 2 회귀기 HPO ──────────────────────────────────────
def optimize_reg(
    unit_data,
    feat_cols,
    model_name="lgbm",
    mode="twostage",
    n_trials=100,
    n_folds=5,
    early_stop=50,
    hpo_params=None,
):
    """
    Stage 2 회귀기 HPO

    Parameters
    ----------
    unit_data : dict
        {"train": df, "val": df, "test": df}
    feat_cols : list
    model_name : str
    mode : str
        "twostage" (Y>0만 학습) 또는 "single" (전체 학습)
    n_trials : int
    n_folds : int
    early_stop : int
    hpo_params : dict, optional
        추가 파라미터 (objective 등 고정값)

    Returns
    -------
    best_params : dict
        최적 파라미터
    study : optuna.Study
    """
    hpo_params = hpo_params or {}

    X_train = unit_data["train"][feat_cols].values
    y_train = unit_data["train"][TARGET_COL].values

    def objective(trial):
        params = SEARCH_SPACES[model_name](trial, "reg")
        # 고정 파라미터 오버라이드
        params.update(hpo_params)

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        oof_pred = np.zeros(len(X_train))

        for tr_idx, va_idx in kf.split(X_train):
            if mode == "twostage":
                pos_tr = y_train[tr_idx] > 0
                pos_va = y_train[va_idx] > 0
                if pos_tr.sum() == 0 or pos_va.sum() == 0:
                    continue
                X_fit, y_fit = X_train[tr_idx][pos_tr], y_train[tr_idx][pos_tr]
                X_es, y_es = X_train[va_idx][pos_va], y_train[va_idx][pos_va]
            else:
                X_fit, y_fit = X_train[tr_idx], y_train[tr_idx]
                X_es, y_es = X_train[va_idx], y_train[va_idx]

            model = create_model(model_name, "reg", params)
            if supports_early_stopping(model_name):
                fit_model(model, X_fit, y_fit, X_es, y_es, early_stop)
            else:
                fit_model(model, X_fit, y_fit)

            oof_pred[va_idx] = model.predict(X_train[va_idx])

        if mode == "twostage":
            proba = unit_data["train"]["clf_proba_mean"].values
            pred = postprocess(proba * oof_pred)
        else:
            pred = postprocess(oof_pred)

        return rmse(y_train, pred)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),  # 시드 제거: 매 실행마다 다른 trial 시퀀스
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = SEARCH_SPACES[model_name](study.best_trial, "reg")
    best_params.update(hpo_params)

    print(f"\n{'=' * 50}")
    print(f"REG HPO 완료 ({model_name}, {n_trials} trials)")
    print(f"Best OOF RMSE: {study.best_value:.6f}")
    print(f"{'=' * 50}")

    return best_params, study


# ─── Stage 1 분류기 HPO ──────────────────────────────────────
def optimize_clf(
    pos_data,
    feat_cols,
    model_name="lgbm",
    n_trials=50,
    n_folds=5,
    early_stop=50,
    label_col="label_bin",
    hpo_params=None,
):
    """
    Stage 1 분류기 HPO

    Parameters
    ----------
    pos_data : dict
    feat_cols : list
    model_name : str
    n_trials, n_folds, early_stop : int
    label_col : str
    hpo_params : dict, optional

    Returns
    -------
    best_params : dict
    study : optuna.Study
    """
    hpo_params = hpo_params or {}

    # 모든 position 데이터를 합쳐서 HPO (position별로 하면 너무 오래 걸림)
    X_all = np.concatenate([
        pos_data[pos]["train"][feat_cols].values
        for pos in sorted(pos_data.keys())
    ])
    y_all = np.concatenate([
        pos_data[pos]["train"][label_col].values
        for pos in sorted(pos_data.keys())
    ])

    def objective(trial):
        params = SEARCH_SPACES[model_name](trial, "clf")
        params.update(hpo_params)

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        oof_proba = np.zeros(len(X_all))

        for tr_idx, va_idx in skf.split(X_all, y_all):
            model = create_model(model_name, "clf", params)
            if supports_early_stopping(model_name):
                fit_model(model, X_all[tr_idx], y_all[tr_idx],
                          X_all[va_idx], y_all[va_idx], early_stop)
            else:
                fit_model(model, X_all[tr_idx], y_all[tr_idx])

            oof_proba[va_idx] = model.predict_proba(X_all[va_idx])[:, 1]

        # 분류 정확도 (val RMSE가 아닌 accuracy 기준)
        from sklearn.metrics import log_loss
        return log_loss(y_all, oof_proba)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),  # 시드 제거: 매 실행마다 다른 trial 시퀀스
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = SEARCH_SPACES[model_name](study.best_trial, "clf")
    best_params.update(hpo_params)

    print(f"\n{'=' * 50}")
    print(f"CLF HPO 완료 ({model_name}, {n_trials} trials)")
    print(f"Best Log Loss: {study.best_value:.6f}")
    print(f"{'=' * 50}")

    return best_params, study


# ─── Feature Selection 파라미터 HPO ──────────────────────────
def optimize_fs(
    unit_data,
    all_feat_cols,
    base_reg_params,
    model_name="lgbm",
    n_trials=30,
    n_folds=3,
    early_stop=50,
    fs_hpo_params=None,
):
    """
    Feature Selection 파라미터 최적화

    min_votes, boruta_perc 등을 Optuna로 탐색하여
    최종 RMSE가 가장 낮은 FS 설정을 찾는다.

    Parameters
    ----------
    unit_data : dict
    all_feat_cols : list
        FS 전 전체 피처 목록
    base_reg_params : dict
        회귀기 파라미터 (고정)
    model_name : str
    n_trials : int
    n_folds : int
    early_stop : int
    fs_hpo_params : dict, optional

    Returns
    -------
    best_fs_params : dict
        최적 FS 파라미터
    study : optuna.Study
    """
    from feature_selection import run_feature_selection

    X_train = unit_data["train"]
    y_train = unit_data["train"][TARGET_COL]

    def objective(trial):
        fs_params = dict(
            methods=["boruta", "lgbm_importance", "null_importance", "permutation"],
            min_votes=trial.suggest_int("min_votes", 2, 4),
            sample_n=None,
            boruta_params=dict(
                max_iter=trial.suggest_int("boruta_max_iter", 50, 200),
                max_depth=trial.suggest_int("boruta_max_depth", 3, 8),
                perc=trial.suggest_int("boruta_perc", 60, 100),
            ),
            lgbm_params=dict(
                threshold=trial.suggest_float("lgbm_threshold", 0, 20),
            ),
            null_params=dict(
                n_runs=trial.suggest_int("null_n_runs", 5, 20),
                threshold=trial.suggest_float("null_threshold", 1.0, 3.0),
            ),
            perm_params=dict(
                threshold=trial.suggest_float("perm_threshold", 0, 20),
                n_repeats=5,
            ),
            mi_params=dict(threshold=0),
        )

        selected, _ = run_feature_selection(
            X_train=X_train, y_train=y_train,
            feat_cols=all_feat_cols, **fs_params,
        )

        if len(selected) < 10:
            return float("inf")

        # 간단한 CV로 RMSE 측정
        X = X_train[selected].values
        y = y_train.values
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        oof = np.zeros(len(y))

        for tr_idx, va_idx in kf.split(X):
            model = create_model(model_name, "reg", base_reg_params)
            if supports_early_stopping(model_name):
                fit_model(model, X[tr_idx], y[tr_idx],
                          X[va_idx], y[va_idx], early_stop)
            else:
                fit_model(model, X[tr_idx], y[tr_idx])
            oof[va_idx] = model.predict(X[va_idx])

        return rmse(y, postprocess(oof))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),  # 시드 제거: 매 실행마다 다른 trial 시퀀스
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n{'=' * 50}")
    print(f"FS HPO 완료 ({n_trials} trials)")
    print(f"Best RMSE: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")
    print(f"{'=' * 50}")

    return study.best_params, study