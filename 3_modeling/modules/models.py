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
    """LightGBM 회귀 탐색 공간 (strategy.md §7.1 Evidence-driven Wide).

    - objective 3종: regression / poisson / tweedie
    - tweedie_variance_power: suggest_float(1.05, 1.95) — strategy.md §6
    """
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 400, 2500),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        num_leaves=trial.suggest_int("num_leaves", 64, 384),
        max_depth=trial.suggest_int("max_depth", 7, 14),
        min_child_samples=trial.suggest_int("min_child_samples", 50, 380),
        subsample=trial.suggest_float("subsample", 0.60, 0.95),
        subsample_freq=trial.suggest_int("subsample_freq", 0, 5),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.20, 0.80),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-7, 1e-1, log=True),
        min_split_gain=trial.suggest_float("min_split_gain", 1e-8, 1e-3, log=True),
        path_smooth=trial.suggest_float("path_smooth", 0.0, 50.0),
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
        device=DEVICE,
    )
    obj_choice = trial.suggest_categorical(
        "objective", ["regression", "poisson", "tweedie"]
    )
    if obj_choice == "poisson":
        params["objective"] = "poisson"
    elif obj_choice == "tweedie":
        params["objective"] = "tweedie"
        params["tweedie_variance_power"] = trial.suggest_float(
            "tweedie_variance_power", 1.05, 1.95
        )
    # 'regression' 선택 시 기본값(MSE) 유지
    return params


def xgb_space(trial):
    """XGBoost 회귀 탐색 공간 (strategy.md §7.5 Anchor ±50% Narrow).

    - objective 3종: reg:squarederror / count:poisson / reg:tweedie  (strategy.md §6)
    - tweedie_variance_power: suggest_float(1.05, 1.95)
    """
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 700, 2200),
        learning_rate=trial.suggest_float("learning_rate", 0.018, 0.073, log=True),
        max_depth=trial.suggest_int("max_depth", 7, 14),
        min_child_weight=trial.suggest_float("min_child_weight", 0.31, 1.24, log=True),
        subsample=trial.suggest_float("subsample", 0.55, 0.95),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.45, 0.95),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0084, 0.0336, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1.95e-6, 7.78e-6, log=True),
        gamma=trial.suggest_float("gamma", 1.92e-6, 7.67e-6, log=True),
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
    )
    obj_choice = trial.suggest_categorical(
        "objective", ["reg:squarederror", "count:poisson", "reg:tweedie"]
    )
    if obj_choice == "reg:tweedie":
        params["objective"] = "reg:tweedie"
        params["tweedie_variance_power"] = trial.suggest_float(
            "tweedie_variance_power", 1.05, 1.95
        )
    else:
        # reg:squarederror / count:poisson 그대로
        params["objective"] = obj_choice
    return params


def catboost_space(trial):
    """CatBoost 회귀 탐색 공간 (strategy.md §7.2 Evidence-driven Wide).

    - loss_function 3종: RMSE / Poisson / Tweedie  (strategy.md §6)
    - Tweedie variance_power: suggest_float(1.05, 1.95)
    """
    params = dict(
        iterations=trial.suggest_int("iterations", 800, 2000),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        depth=trial.suggest_int("depth", 6, 10),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 5.0, 30.0, log=True),
        random_strength=trial.suggest_float("random_strength", 0.1, 5.0, log=True),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
        border_count=trial.suggest_int("border_count", 64, 254),
        rsm=trial.suggest_float("rsm", 0.4, 1.0),
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
    )
    loss_choice = trial.suggest_categorical(
        "loss_function", ["RMSE", "Poisson", "Tweedie"]
    )
    if loss_choice == "RMSE":
        params["loss_function"] = "RMSE"
    elif loss_choice == "Poisson":
        params["loss_function"] = "Poisson"
    elif loss_choice == "Tweedie":
        power = trial.suggest_float("tweedie_variance_power", 1.05, 1.95)
        params["loss_function"] = f"Tweedie:variance_power={power}"
    return params


def et_space(trial):
    """ExtraTrees 회귀 탐색 공간 (strategy.md §7.3 Evidence-driven Wide)."""
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 300, 800),
        max_depth=trial.suggest_int("max_depth", 15, 25),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 20),
        min_samples_split=trial.suggest_int("min_samples_split", 15, 50),
        max_features=trial.suggest_categorical(
            "max_features", ["sqrt"]
        ),
        bootstrap=True,
        random_state=SEED,
        n_jobs=-1,
    )


def enet_space(trial):
    """ElasticNet 회귀 탐색 공간. 스케일링 필수 (scaler.maybe_scale 경유)."""
    return dict(
        alpha=trial.suggest_float("alpha", 1e-7, 1e-3, log=True),
        l1_ratio=trial.suggest_float("l1_ratio", 0.4, 0.99),
        max_iter=trial.suggest_int("max_iter", 8000, 15000, step=1000),
        tol=1e-6,
        selection="random",
        precompute=True,
        random_state=SEED,
    )


def zitboost_space(trial):
    """ZI-Tweedie + LightGBM EM 탐색 공간 (21개 HP).

    μ(핵심 회귀): 9 / π(분류): 5 / φ(분산): 5 / ZIT 전용: 2
    """
    params = dict(
        zeta=trial.suggest_float("zeta", 1.1, 1.4),
        n_em_iters=trial.suggest_int("n_em_iters", 10, 30),
    )
    # μ 모델
    params.update(
        mu_n_estimators=trial.suggest_int("mu_n_estimators", 50, 400),
        mu_learning_rate=trial.suggest_float("mu_learning_rate", 0.003, 0.02, log=True),
        mu_num_leaves=trial.suggest_int("mu_num_leaves", 100, 200),
        mu_max_depth=trial.suggest_int("mu_max_depth", 3, 8),
        mu_min_child_samples=trial.suggest_int("mu_min_child_samples", 30, 200),
        mu_subsample=trial.suggest_float("mu_subsample", 0.55, 0.80),
        mu_colsample_bytree=trial.suggest_float("mu_colsample_bytree", 0.2, 0.5),
        mu_reg_alpha=trial.suggest_float("mu_reg_alpha", 1e-6, 1e-2, log=True),
        mu_reg_lambda=trial.suggest_float("mu_reg_lambda", 0.01, 1.0, log=True),
    )
    # π 모델
    params.update(
        pi_n_estimators=trial.suggest_int("pi_n_estimators", 100, 250),
        pi_learning_rate=trial.suggest_float("pi_learning_rate", 0.05, 0.2, log=True),
        pi_num_leaves=trial.suggest_int("pi_num_leaves", 50, 100),
        pi_max_depth=trial.suggest_int("pi_max_depth", 5, 14),
        pi_min_child_samples=trial.suggest_int("pi_min_child_samples", 20, 60),
    )
    # φ 모델
    params.update(
        phi_n_estimators=trial.suggest_int("phi_n_estimators", 20, 250),
        phi_learning_rate=trial.suggest_float("phi_learning_rate", 0.005, 0.03, log=True),
        phi_num_leaves=trial.suggest_int("phi_num_leaves", 50, 120),
        phi_max_depth=trial.suggest_int("phi_max_depth", 4, 6),
        phi_min_child_samples=trial.suggest_int("phi_min_child_samples", 20, 100),
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


# ═════════════════════════════════════════════════════════════
# ZITREG (경로 B) 전용 Search Space — 03_zit_plus_reg.ipynb
# ═════════════════════════════════════════════════════════════
# 사용 시점: 03 노트북에서 hpo.run_hpo(..., space_variant='zitreg').
# 근거: zitreg-lgbm-002 study top 10 trial 분포 (2026-04-26 분석)
#   - num_leaves: 384(cap) 도달, max_depth: 13~14 cluster, min_split_gain: floor 군집
#   - objective: 10/10 trial 모두 tweedie_1.5 → regression/poisson 제거
#   - n_estimators: 2252~2773 cluster (3000 cap 근접 → 5000 으로 여유)
# xgb/catboost/et/enet 은 default 와 동일 (사용자 지시: "나머지는 그냥 복사").

def lgbm_zitreg_space(trial):
    """LightGBM 회귀 zitreg 전용. zitreg-lgbm-002 boundary hit 반영 확장."""
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 5000),                # 3000 → 5000
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        num_leaves=trial.suggest_int("num_leaves", 8, 768),                       # 384 → 768 (cap hit)
        max_depth=trial.suggest_int("max_depth", 3, 20),                          # 14 → 20 (cap hit)
        min_child_samples=trial.suggest_int("min_child_samples", 5, 400),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.1, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 30.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        min_split_gain=trial.suggest_float("min_split_gain", 1e-12, 1.0, log=True),  # 1e-9 → 1e-12 (floor cluster)
        path_smooth=trial.suggest_float("path_smooth", 0.0, 50.0),
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
        device=DEVICE,
    )
    # tweedie 4단 (1.2/1.3/1.5/1.7) — 1.5 선호 + 인접값 추가 탐색
    obj_choice = trial.suggest_categorical(
        "objective", ["tweedie_1.2", "tweedie_1.3", "tweedie_1.5", "tweedie_1.7"]
    )
    params["objective"] = "tweedie"
    params["tweedie_variance_power"] = float(obj_choice.split("_")[1])
    return params


# 나머지 4종 — default 와 동일한 범위 (사용자 지시: 복사만)
def xgb_zitreg_space(trial):
    return xgb_space(trial)


def catboost_zitreg_space(trial):
    return catboost_space(trial)


def et_zitreg_space(trial):
    return et_space(trial)


def enet_zitreg_space(trial):
    return enet_space(trial)


SEARCH_SPACES_ZITREG = {
    "lgbm":     lgbm_zitreg_space,
    "xgb":      xgb_zitreg_space,
    "catboost": catboost_zitreg_space,
    "et":       et_zitreg_space,
    "enet":     enet_zitreg_space,
}


def get_search_space(name, variant="default"):
    """모델 이름 + variant → search space 함수.

    variant
    -------
    'default' : 1차 baseline (lgbm/xgb/catboost/et/enet/zitboost)
    'zitreg'  : 03_zit_plus_reg 전용 (lgbm 확장, 나머지 4개는 default 복제)
    """
    if variant == "default":
        if name not in SEARCH_SPACES:
            raise KeyError(
                f"No search space for {name!r} (variant=default). "
                f"Available: {list(SEARCH_SPACES)}"
            )
        return SEARCH_SPACES[name]
    elif variant == "zitreg":
        if name not in SEARCH_SPACES_ZITREG:
            raise KeyError(
                f"No zitreg search space for {name!r}. "
                f"Available: {list(SEARCH_SPACES_ZITREG)}"
            )
        return SEARCH_SPACES_ZITREG[name]
    else:
        raise ValueError(f"Unknown variant {variant!r} (allowed: 'default', 'zitreg')")


# ═════════════════════════════════════════════════════════════
# 분류기 (Stage 1) — Two-Stage 03c 노트북 전용
# ═════════════════════════════════════════════════════════════
# binary 분류: y > 0 unit 식별 → die-level prob OOF 생성.
# 사용처: Two-Stage 곱셈 구조의 P(Y>0) 추정. 03c가 prob csv 저장 → 03e가 reg와 곱.
#
# 모델 추가 방법:
#   1) CLF_REGISTRY 에 한 줄 등록
#   2) {model}_clf_space 함수 작성
#   3) CLF_SEARCH_SPACES 에 한 줄 등록
# 끝.
#
# imbalance 대응 (scale_pos_weight / class_weight) 는 hpo 측에서 fold-local 로
# 자동 계산하므로 search space 에는 두지 않는다.

from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier

CLF_REGISTRY = {
    "lgbm":     lgb.LGBMClassifier,
    "xgb":      xgb_lib.XGBClassifier,
    "catboost": CatBoostClassifier,
    "et":       ExtraTreesClassifier,
}

CLF_AVAILABLE_MODELS = list(CLF_REGISTRY)


def create_classifier(name, params):
    """분류기 이름 + HP dict → 초기화된 classifier 인스턴스."""
    if name not in CLF_REGISTRY:
        raise KeyError(
            f"Unknown clf model {name!r}. Available: {CLF_AVAILABLE_MODELS}"
        )
    cls = CLF_REGISTRY[name]
    return cls(**params)


# ─── Search Space (분류, binary 고정) ───
# objective/loss_function 은 binary 고정이라 search 없음.
# 손실함수 다양화는 reg 쪽에서만 의미 있음.

def lgbm_clf_space(trial):
    """LightGBM 분류 탐색 공간. objective='binary' 고정.

    분류 친화 조정 (vs lgbm_space):
    - lr 상한 0.3 → 0.15 (calibration 보존)
    - n_est 3000 → 2000 (over-fit prob 방지)
    - num_leaves 384 → 320 (작은 트리 안정)
    - min_child_samples floor 5 → 30 (29% pos imbalance 안전)
    - scale_pos_weight categorical 탐색 (RMSE-aligned objective는 calibrated prob 선호)
    """
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 2000),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        num_leaves=trial.suggest_int("num_leaves", 8, 320),
        max_depth=trial.suggest_int("max_depth", 3, 14),
        min_child_samples=trial.suggest_int("min_child_samples", 30, 400),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.1, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 30.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        min_split_gain=trial.suggest_float("min_split_gain", 1e-9, 1.0, log=True),
        path_smooth=trial.suggest_float("path_smooth", 0.0, 50.0),
        scale_pos_weight=trial.suggest_categorical(
            "scale_pos_weight", [1.0, 1.5, 2.43, 3.5]
        ),
        objective="binary",
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
        device=DEVICE,
    )


def xgb_clf_space(trial):
    """XGBoost 분류 탐색 공간. objective='binary:logistic' 고정.

    분류 친화 조정: lr 상한 0.15, scale_pos_weight categorical 탐색.
    """
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 1500),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        min_child_weight=trial.suggest_float("min_child_weight", 0.5, 30.0, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        gamma=trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        scale_pos_weight=trial.suggest_categorical(
            "scale_pos_weight", [1.0, 1.5, 2.43, 3.5]
        ),
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
    )


def catboost_clf_space(trial):
    """CatBoost 분류 탐색 공간. loss_function='Logloss' 고정.

    분류 친화 조정: lr 상한 0.15, depth 상한 10, auto_class_weights categorical.
    """
    p = dict(
        iterations=trial.suggest_int("iterations", 100, 2500),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        depth=trial.suggest_int("depth", 3, 10),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        random_strength=trial.suggest_float("random_strength", 0.1, 10.0, log=True),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
        border_count=trial.suggest_int("border_count", 32, 192),
        rsm=trial.suggest_float("rsm", 0.3, 1.0),
        loss_function="Logloss",
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
    )
    acw = trial.suggest_categorical("auto_class_weights", ["None", "Balanced"])
    p["auto_class_weights"] = None if acw == "None" else acw
    return p


def et_clf_space(trial):
    """ExtraTrees 분류 탐색 공간.

    분류 친화 조정: n_estimators 상한 800, class_weight categorical.
    """
    p = dict(
        n_estimators=trial.suggest_int("n_estimators", 300, 800),
        max_depth=trial.suggest_int("max_depth", 10, 22),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 40),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 40),
        max_features=trial.suggest_categorical("max_features", ["sqrt"]),
        bootstrap=True,
        random_state=SEED,
        n_jobs=-1,
    )
    cw = trial.suggest_categorical("class_weight", ["None", "balanced"])
    p["class_weight"] = None if cw == "None" else cw
    return p


CLF_SEARCH_SPACES = {
    "lgbm":     lgbm_clf_space,
    "xgb":      xgb_clf_space,
    "catboost": catboost_clf_space,
    "et":       et_clf_space,
}


def get_clf_search_space(name):
    """모델 이름 → clf search space 함수."""
    if name not in CLF_SEARCH_SPACES:
        raise KeyError(
            f"No clf search space for {name!r}. "
            f"Available: {list(CLF_SEARCH_SPACES)}"
        )
    return CLF_SEARCH_SPACES[name]


def resolve_clf_imbalance(model_name, params, y_train_bin):
    """모델별 imbalance 옵션을 fold-local 로 자동 추가하여 새 dict 반환.

    y_train_bin: 0/1 array (이번 fold 학습 타깃). scale_pos_weight = n_neg / n_pos.

    원본 params 는 변경하지 않는다.
    """
    p = dict(params)
    n_pos = int((np.asarray(y_train_bin) == 1).sum())
    n_neg = int((np.asarray(y_train_bin) == 0).sum())
    spw = (n_neg / n_pos) if n_pos > 0 else 1.0

    if model_name in {"lgbm", "xgb"}:
        p.setdefault("scale_pos_weight", spw)
    elif model_name == "catboost":
        p.setdefault("auto_class_weights", "Balanced")
    elif model_name == "et":
        p.setdefault("class_weight", "balanced")
    return p
