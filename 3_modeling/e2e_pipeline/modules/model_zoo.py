"""
모델 팩토리 + 공통 fit 래퍼

지원 모델 (트리 5종):
- lgbm: LightGBM
- xgb: XGBoost
- catboost: CatBoost
- rf: RandomForest
- et: ExtraTrees

사용법:
    from modules.model_zoo import create_model, fit_model, get_default_params

    model = create_model("lgbm", "clf", params)
    fit_model(model, X_tr, y_tr, X_val, y_val, early_stop=50)
    proba = model.predict_proba(X_test)[:, 1]
"""
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
)

from utils.config import SEED


# ─── 모델 레지스트리 ──────────────────────────────────────────
MODEL_REGISTRY = {
    "lgbm": {
        "clf": lgb.LGBMClassifier,
        "reg": lgb.LGBMRegressor,
        "supports_early_stopping": True,
    },
    "xgb": {
        "clf": xgb.XGBClassifier,
        "reg": xgb.XGBRegressor,
        "supports_early_stopping": True,
    },
    "catboost": {
        "clf": CatBoostClassifier,
        "reg": CatBoostRegressor,
        "supports_early_stopping": True,
    },
    "rf": {
        "clf": RandomForestClassifier,
        "reg": RandomForestRegressor,
        "supports_early_stopping": False,
    },
    "et": {
        "clf": ExtraTreesClassifier,
        "reg": ExtraTreesRegressor,
        "supports_early_stopping": False,
    },
}

AVAILABLE_MODELS = list(MODEL_REGISTRY.keys())


def _detect_device():
    """GPU 사용 가능 여부 감지"""
    try:
        import torch
        if torch.cuda.is_available():
            return "gpu"
    except ImportError:
        pass
    return "cpu"


DEVICE = _detect_device()


# ─── 기본 파라미터 ────────────────────────────────────────────
def get_default_params(name, task, device=None):
    """
    모델별 기본 파라미터 반환

    Parameters
    ----------
    name : str
        모델명 ("lgbm", "xgb", "catboost", "rf", "et")
    task : str
        "clf" (분류) 또는 "reg" (회귀)
    device : str, optional
        "cpu" 또는 "gpu". None이면 자동 감지

    Returns
    -------
    dict : 기본 파라미터
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {AVAILABLE_MODELS}")
    if device is None:
        device = DEVICE

    if name == "lgbm":
        params = dict(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
            device=device,
        )
    elif name == "xgb":
        params = dict(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=20,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device="cuda" if device == "gpu" else "cpu",
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
        )
    elif name == "catboost":
        params = dict(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            min_data_in_leaf=20,
            subsample=0.8,
            colsample_bylevel=0.8,
            task_type="GPU" if device == "gpu" else "CPU",
            random_seed=SEED,
            verbose=0,
        )
    elif name == "rf":
        params = dict(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=SEED,
            n_jobs=-1,
        )
    elif name == "et":
        params = dict(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=SEED,
            n_jobs=-1,
        )

    return params


# ─── 모델 생성 ────────────────────────────────────────────────
def create_model(name, task, params=None):
    """
    모델 인스턴스 생성

    Parameters
    ----------
    name : str
        모델명
    task : str
        "clf" 또는 "reg"
    params : dict, optional
        커스텀 파라미터. None이면 기본값 사용

    Returns
    -------
    model : sklearn-compatible 모델 인스턴스
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {AVAILABLE_MODELS}")
    if task not in ("clf", "reg"):
        raise ValueError(f"task must be 'clf' or 'reg', got '{task}'")

    cls = MODEL_REGISTRY[name][task]
    if params is None:
        params = get_default_params(name, task)

    return cls(**params)


# ─── 공통 fit 래퍼 ────────────────────────────────────────────
def fit_model(model, X_train, y_train, X_val=None, y_val=None, early_stop=50):
    """
    모델 타입에 맞는 early stopping으로 학습

    Parameters
    ----------
    model : sklearn-compatible 모델
    X_train, y_train : 학습 데이터
    X_val, y_val : 검증 데이터 (early stopping용. None이면 early stopping 없이 학습)
    early_stop : int
        early stopping patience

    Returns
    -------
    model : 학습된 모델
    """
    model_cls = type(model).__name__.lower()

    # --- LightGBM ---
    if "lgbm" in model_cls:
        fit_kwargs = {}
        if X_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(early_stop, verbose=False),
                lgb.log_evaluation(0),
            ]
        model.fit(X_train, y_train, **fit_kwargs)

    # --- XGBoost ---
    elif "xgb" in model_cls:
        fit_kwargs = {}
        if X_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = False
        model.set_params(early_stopping_rounds=early_stop if X_val is not None else None)
        model.fit(X_train, y_train, **fit_kwargs)

    # --- CatBoost ---
    elif "catboost" in model_cls:
        fit_kwargs = {}
        if X_val is not None:
            fit_kwargs["eval_set"] = (X_val, y_val)
            fit_kwargs["early_stopping_rounds"] = early_stop
        model.fit(X_train, y_train, **fit_kwargs)

    # --- RF / ExtraTrees (no early stopping) ---
    else:
        model.fit(X_train, y_train)

    return model


def get_best_iteration(model):
    """학습된 모델의 best iteration 반환 (없으면 None)"""
    model_cls = type(model).__name__.lower()

    if "lgbm" in model_cls:
        return getattr(model, "best_iteration_", None)
    elif "xgb" in model_cls:
        return getattr(model, "best_iteration", None)
    elif "catboost" in model_cls:
        return getattr(model, "best_iteration_", None)
    return None


def supports_early_stopping(name):
    """해당 모델이 early stopping을 지원하는지 여부"""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {AVAILABLE_MODELS}")
    return MODEL_REGISTRY[name]["supports_early_stopping"]