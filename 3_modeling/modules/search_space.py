"""
Optuna Search Space 정의 — 모델별 하이퍼파라미터 탐색 공간

기존 hpo.py에서 search space만 추출.
"""
from utils.config import SEED
from .model_zoo import DEVICE


def lgbm_space(trial, prefix=""):
    """LightGBM 탐색 공간"""
    p = prefix
    return dict(
        n_estimators=trial.suggest_int(f"{p}n_estimators", 100, 3000),
        learning_rate=trial.suggest_float(f"{p}learning_rate", 0.005, 0.3, log=True),
        num_leaves=trial.suggest_int(f"{p}num_leaves", 8, 256),
        max_depth=trial.suggest_int(f"{p}max_depth", 3, 12),
        min_child_samples=trial.suggest_int(f"{p}min_child_samples", 5, 300),
        subsample=trial.suggest_float(f"{p}subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float(f"{p}colsample_bytree", 0.1, 1.0),
        reg_alpha=trial.suggest_float(f"{p}reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float(f"{p}reg_lambda", 1e-8, 10.0, log=True),
        min_split_gain=trial.suggest_float(f"{p}min_split_gain", 1e-8, 1.0, log=True),
        path_smooth=trial.suggest_float(f"{p}path_smooth", 0.0, 50.0),
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
        device=DEVICE,
    )


def xgb_space(trial, prefix=""):
    """XGBoost 탐색 공간"""
    p = prefix
    return dict(
        n_estimators=trial.suggest_int(f"{p}n_estimators", 100, 3000),
        learning_rate=trial.suggest_float(f"{p}learning_rate", 0.005, 0.3, log=True),
        max_depth=trial.suggest_int(f"{p}max_depth", 3, 12),
        min_child_weight=trial.suggest_int(f"{p}min_child_weight", 1, 300),
        subsample=trial.suggest_float(f"{p}subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float(f"{p}colsample_bytree", 0.1, 1.0),
        reg_alpha=trial.suggest_float(f"{p}reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float(f"{p}reg_lambda", 1e-8, 10.0, log=True),
        gamma=trial.suggest_float(f"{p}gamma", 1e-8, 5.0, log=True),
        tree_method="hist",
        device="cuda" if DEVICE == "gpu" else "cpu",
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )


def catboost_space(trial, prefix=""):
    """CatBoost 탐색 공간

    Note: 기본 bootstrap_type=Bayesian은 subsample을 지원하지 않으므로
    subsample을 탐색하기 위해 Bernoulli로 고정한다.
    task_type='CPU' 강제: Colab T4 GPU에서 LGBM과 GPU 메모리 경합으로 OOM 발생하여 CPU로 고정.
    """
    p = prefix
    return dict(
        iterations=trial.suggest_int(f"{p}iterations", 100, 3000),
        learning_rate=trial.suggest_float(f"{p}learning_rate", 0.005, 0.3, log=True),
        depth=trial.suggest_int(f"{p}depth", 3, 10),
        min_data_in_leaf=trial.suggest_int(f"{p}min_data_in_leaf", 5, 300),
        bootstrap_type="Bernoulli",
        subsample=trial.suggest_float(f"{p}subsample", 0.5, 1.0),
        colsample_bylevel=trial.suggest_float(f"{p}colsample_bylevel", 0.1, 1.0),
        l2_leaf_reg=trial.suggest_float(f"{p}l2_leaf_reg", 1e-8, 10.0, log=True),
        task_type="CPU",
        random_seed=SEED,
        verbose=0,
    )


def rf_space(trial, prefix=""):
    """RandomForest / ExtraTrees 탐색 공간"""
    p = prefix
    return dict(
        n_estimators=trial.suggest_int(f"{p}n_estimators", 100, 1000),
        max_depth=trial.suggest_int(f"{p}max_depth", 5, 30),
        min_samples_leaf=trial.suggest_int(f"{p}min_samples_leaf", 2, 50),
        min_samples_split=trial.suggest_int(f"{p}min_samples_split", 2, 50),
        max_features=trial.suggest_categorical(
            f"{p}max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]
        ),
        random_state=SEED,
        n_jobs=-1,
    )


SEARCH_SPACES = {
    "lgbm": lgbm_space,
    "xgb": xgb_space,
    "catboost": catboost_space,
    "rf": rf_space,
    "et": rf_space,
}
