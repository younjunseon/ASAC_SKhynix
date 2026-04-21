"""
Final 파이프라인 — 모델 레지스트리 + Optuna Search Space

회귀 전용 6종:
  xgb / catboost / lgbm / et / enet / zitboost

각 모델은 sklearn-like API (`fit`, `predict`) 을 제공한다.
- zitboost: ZITboostRegressor (회귀 + 내부 π 분류 담당, 곱셈 구조로 기본 `predict` 반환)
- enet: 스케일링 필요 (scaler.maybe_scale 경유)
- 트리 4종(xgb/catboost/lgbm/et): 스케일링 불필요

사용법
------
    from final.modules import models

    space_fn = models.get_search_space("lgbm")
    params = space_fn(trial)         # Optuna trial 객체로 HP 샘플
    model = models.create_regressor("lgbm", params)
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
"""
import numpy as np
import lightgbm as lgb
import xgboost as xgb_lib
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import ElasticNet

from utils.config import SEED
from .zit import ZITboostRegressor


# ─── LightGBM GPU 자동 감지 ────────────────────────────────────
def _detect_lgbm_device():
    """LightGBM GPU 지원 여부 감지. GPU 빌드가 아니거나 실패하면 'cpu'."""
    try:
        ds = lgb.Dataset(np.zeros((10, 2)), label=np.zeros(10))
        lgb.train({"device": "gpu", "verbose": -1, "objective": "regression"},
                  ds, num_boost_round=1)
        return "gpu"
    except Exception:
        return "cpu"


DEVICE = "cpu"  # 노트북에서 원하면 `models.DEVICE = 'gpu'` 로 override


# ─── 모델 레지스트리 ──────────────────────────────────────────
MODEL_REGISTRY = {
    "lgbm":     lgb.LGBMRegressor,
    "xgb":      xgb_lib.XGBRegressor,
    "catboost": CatBoostRegressor,
    "et":       ExtraTreesRegressor,
    "enet":     ElasticNet,
    "zitboost": ZITboostRegressor,
}

AVAILABLE_MODELS = list(MODEL_REGISTRY)


def create_regressor(name, params):
    """모델 이름 + HP dict → 초기화된 regressor 인스턴스."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {AVAILABLE_MODELS}")
    cls = MODEL_REGISTRY[name]
    return cls(**params)


# ═════════════════════════════════════════════════════════════
# Search Space (Optuna trial → HP dict)
# ═════════════════════════════════════════════════════════════

def lgbm_space(trial):
    """LightGBM 회귀 탐색 공간. objective는 regression/poisson/tweedie 4종."""
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 3000),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        num_leaves=trial.suggest_int("num_leaves", 8, 384),
        max_depth=trial.suggest_int("max_depth", 3, 14),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 400),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=1,  # ← 없으면 subsample 무시됨 (LGBM 기본값 0)
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.1, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 30.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        min_split_gain=trial.suggest_float("min_split_gain", 1e-9, 1.0, log=True),
        path_smooth=trial.suggest_float("path_smooth", 0.0, 50.0),
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
        device=DEVICE,
    )
    obj_choice = trial.suggest_categorical(
        "objective", ["regression", "poisson", "tweedie_1.2", "tweedie_1.5"]
    )
    if obj_choice == "poisson":
        params["objective"] = "poisson"
    elif obj_choice.startswith("tweedie"):
        params["objective"] = "tweedie"
        params["tweedie_variance_power"] = float(obj_choice.split("_")[1])
    # 'regression' 선택 시 기본값(MSE) 유지
    return params


def xgb_space(trial):
    """XGBoost 회귀 탐색 공간. objective는 squarederror/tweedie 3종."""
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 3000),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        min_child_weight=trial.suggest_float("min_child_weight", 0.5, 30.0, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        gamma=trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
    )
    obj_choice = trial.suggest_categorical(
        "objective", ["reg:squarederror", "reg:tweedie_1.2", "reg:tweedie_1.5"]
    )
    if obj_choice == "reg:squarederror":
        params["objective"] = "reg:squarederror"
    else:
        params["objective"] = "reg:tweedie"
        params["tweedie_variance_power"] = float(obj_choice.split("_")[1])
    return params


def catboost_space(trial):
    """CatBoost 회귀 탐색 공간. loss_function은 RMSE/Tweedie 3종."""
    params = dict(
        iterations=trial.suggest_int("iterations", 100, 3000),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        depth=trial.suggest_int("depth", 3, 10),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        random_strength=trial.suggest_float("random_strength", 0.1, 10.0, log=True),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
        border_count=trial.suggest_int("border_count", 32, 254),
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
    )
    loss_choice = trial.suggest_categorical(
        "loss_function", ["RMSE", "Tweedie_1.2", "Tweedie_1.5"]
    )
    if loss_choice == "RMSE":
        params["loss_function"] = "RMSE"
    else:
        power = float(loss_choice.split("_")[1])
        params["loss_function"] = f"Tweedie:variance_power={power}"
    return params


def et_space(trial):
    """ExtraTrees 회귀 탐색 공간."""
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 300, 1500),
        max_depth=trial.suggest_int("max_depth", 6, 30),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 30),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 40),
        max_features=trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]
        ),
        bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
        random_state=SEED,
        n_jobs=-1,
    )


def enet_space(trial):
    """ElasticNet 회귀 탐색 공간. 스케일링 필수 (scaler.maybe_scale 경유)."""
    return dict(
        alpha=trial.suggest_float("alpha", 1e-5, 1.0, log=True),
        l1_ratio=trial.suggest_float("l1_ratio", 0.1, 0.9),
        max_iter=trial.suggest_int("max_iter", 2000, 8000, step=1000),
        tol=1e-4,
        random_state=SEED,
    )


def zitboost_space(trial):
    """ZI-Tweedie + LightGBM EM 탐색 공간 (21개 HP).

    μ(핵심 회귀): 9 / π(분류): 5 / φ(분산): 5 / ZIT 전용: 2
    """
    params = dict(
        zeta=trial.suggest_float("zeta", 1.1, 1.9),
        n_em_iters=trial.suggest_int("n_em_iters", 3, 20),
    )
    # μ 모델
    params.update(
        mu_n_estimators=trial.suggest_int("mu_n_estimators", 100, 2000),
        mu_learning_rate=trial.suggest_float("mu_learning_rate", 0.005, 0.1, log=True),
        mu_num_leaves=trial.suggest_int("mu_num_leaves", 32, 256),
        mu_max_depth=trial.suggest_int("mu_max_depth", 5, 12),
        mu_min_child_samples=trial.suggest_int("mu_min_child_samples", 5, 100),
        mu_subsample=trial.suggest_float("mu_subsample", 0.5, 1.0),
        mu_colsample_bytree=trial.suggest_float("mu_colsample_bytree", 0.3, 1.0),
        mu_reg_alpha=trial.suggest_float("mu_reg_alpha", 1e-8, 10.0, log=True),
        mu_reg_lambda=trial.suggest_float("mu_reg_lambda", 1e-8, 10.0, log=True),
    )
    # π 모델
    params.update(
        pi_n_estimators=trial.suggest_int("pi_n_estimators", 50, 500),
        pi_learning_rate=trial.suggest_float("pi_learning_rate", 0.01, 0.1, log=True),
        pi_num_leaves=trial.suggest_int("pi_num_leaves", 16, 128),
        pi_max_depth=trial.suggest_int("pi_max_depth", 3, 8),
        pi_min_child_samples=trial.suggest_int("pi_min_child_samples", 10, 100),
    )
    # φ 모델
    params.update(
        phi_n_estimators=trial.suggest_int("phi_n_estimators", 50, 500),
        phi_learning_rate=trial.suggest_float("phi_learning_rate", 0.01, 0.1, log=True),
        phi_num_leaves=trial.suggest_int("phi_num_leaves", 16, 128),
        phi_max_depth=trial.suggest_int("phi_max_depth", 3, 8),
        phi_min_child_samples=trial.suggest_int("phi_min_child_samples", 10, 100),
    )
    params.update(random_state=SEED, n_jobs=-1, verbose=-1, device=DEVICE)
    return params


SEARCH_SPACES = {
    "lgbm":     lgbm_space,
    "xgb":      xgb_space,
    "catboost": catboost_space,
    "et":       et_space,
    "enet":     enet_space,
    "zitboost": zitboost_space,
}


def get_search_space(name):
    """모델 이름 → search space 함수."""
    if name not in SEARCH_SPACES:
        raise KeyError(f"No search space for {name!r}. Available: {list(SEARCH_SPACES)}")
    return SEARCH_SPACES[name]
