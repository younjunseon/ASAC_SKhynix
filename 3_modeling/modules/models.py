"""
모델 레지스트리 + Optuna 탐색 공간.

회귀 모델(MODEL_REGISTRY): xgb / catboost / lgbm / et / enet / zitboost — 모두 sklearn 스타일 fit/predict.
  - enet : 스케일링 필요 (HPO 경로에서는 hpo.py가 fold별로 RobustScaler를 직접 적용)
  - 트리 4종 : 스케일링 불필요
  - zitboost : 회귀 + 내부에 π 분류를 품은 곱셈 구조 (predict가 곧 E[Y]=(1-π)μ)

분류 모델(CLF_REGISTRY): lgbm / xgb / catboost / et — Two-Stage의 Stage 1(y>0 식별)용.

탐색 공간:
  - SEARCH_SPACES        : 회귀 기본 (변종 'default')
  - SEARCH_SPACES_ZITREG : zitreg 경로 전용 (lgbm만 범위 확장, 나머지는 default 복제)
  - CLF_SEARCH_SPACES    : 분류용 (objective는 binary 고정, scale_pos_weight 등은 따로 둠)
get_search_space / get_clf_search_space로 (모델명 → trial→HP dict 함수)를 얻어 쓴다.

사용법
------
    from modules import models

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
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet

from utils.config import SEED
from .zit import ZITboostRegressor


DEVICE = "cpu"  # 노트북에서 `models.DEVICE = 'gpu'` 로 바꾸면 lgbm 계열·zitboost가 GPU로 학습됨 (xgb는 tree_method 고정, catboost/et는 이 값을 안 씀)


# 모델 이름 → 클래스 매핑 (회귀)
MODEL_REGISTRY = {
    "lgbm":     lgb.LGBMRegressor,
    "xgb":      xgb_lib.XGBRegressor,
    "catboost": CatBoostRegressor,
    "et":       ExtraTreesRegressor,
    "rf":       RandomForestRegressor,
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


# ============================================================
# 회귀 탐색 공간 (Optuna trial → HP dict)
# ============================================================

def lgbm_space(trial):
    """LightGBM 회귀 탐색 공간 (넓은 범위).

    objective는 regression(MSE) / poisson / tweedie 중 탐색. tweedie면 variance_power도 같이 탐색.
    """
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 4000),
        learning_rate=trial.suggest_float("learning_rate", 0.003, 0.08, log=True),
        num_leaves=trial.suggest_int("num_leaves", 64, 600),
        max_depth=trial.suggest_int("max_depth", 7, 18),
        min_child_samples=trial.suggest_int("min_child_samples", 50, 380),
        subsample=trial.suggest_float("subsample", 0.50, 1.0),
        subsample_freq=trial.suggest_int("subsample_freq", 0, 5),   # 0이면 subsample 무시됨
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
    # objective는 categorical로 고르고, tweedie면 variance_power(1~2)도 추가 탐색
    obj_choice = trial.suggest_categorical(
        "objective", ["regression", "poisson", "tweedie"]
    )
    if obj_choice == "poisson":
        params["objective"] = "poisson"
    elif obj_choice == "tweedie":
        params["objective"] = "tweedie"
        params["tweedie_variance_power"] = trial.suggest_float(
            "tweedie_variance_power", 1.05, 1.99
        )
    # 'regression'이면 별도 설정 없음 → LGBM 기본 objective(MSE) 사용
    return params


def xgb_space(trial):
    """XGBoost 회귀 탐색 공간.

    objective: reg:squarederror / count:poisson / reg:tweedie 중 탐색. tweedie면 variance_power도 함께.
    """
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 400, 3000),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.10, log=True),
        max_depth=trial.suggest_int("max_depth", 7, 18),
        min_child_weight=trial.suggest_float("min_child_weight", 0.1, 5.0, log=True),
        subsample=trial.suggest_float("subsample", 0.55, 0.95),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.30, 0.95),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 0.1, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-7, 1e-4, log=True),
        gamma=trial.suggest_float("gamma", 1e-7, 1e-4, log=True),
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
            "tweedie_variance_power", 1.05, 1.99
        )
        # reg:tweedie는 log-space 예측이라 zero-inflated·극소 target에서 leaf step이 폭주 → exp() overflow → NaN.
        # XGBoost는 count:poisson에는 max_delta_step=0.7을 자동으로 넣지만 reg:tweedie에는 안 넣어줘서 직접 켠다 (발산 가드).
        params["max_delta_step"] = 0.7
    else:
        params["objective"] = obj_choice   # squarederror / count:poisson 그대로
    return params


def catboost_space(trial):
    """CatBoost 회귀 탐색 공간.

    loss_function: RMSE / Poisson / Tweedie 중 탐색. Tweedie면 "Tweedie:variance_power=..." 문자열로 조립.
    """
    params = dict(
        iterations=trial.suggest_int("iterations", 800, 3500),
        learning_rate=trial.suggest_float("learning_rate", 0.003, 0.2, log=True),
        depth=trial.suggest_int("depth", 6, 12),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 60.0, log=True),
        random_strength=trial.suggest_float("random_strength", 0.05, 5.0, log=True),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
        border_count=trial.suggest_int("border_count", 64, 254),
        rsm=trial.suggest_float("rsm", 0.4, 1.0),
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,   # 학습 중 디스크에 로그 파일 안 쓰게
    )
    loss_choice = trial.suggest_categorical(
        "loss_function", ["RMSE", "Poisson", "Tweedie"]
    )
    if loss_choice == "RMSE":
        params["loss_function"] = "RMSE"
    elif loss_choice == "Poisson":
        params["loss_function"] = "Poisson"
    elif loss_choice == "Tweedie":
        # CatBoost는 Tweedie variance_power를 loss_function 문자열에 끼워 넣는 형식
        power = trial.suggest_float("tweedie_variance_power", 1.05, 1.99)
        params["loss_function"] = f"Tweedie:variance_power={power}"
    return params


def et_space(trial):
    """ExtraTrees 회귀 탐색 공간."""
    # max_features: 'sqrt' 고정 vs 비율(frac) 탐색 중 선택 — categorical로 분기
    mf_kind = trial.suggest_categorical("max_features_kind", ["sqrt", "frac"])
    if mf_kind == "sqrt":
        max_features_val = "sqrt"
    else:
        max_features_val = trial.suggest_float("max_features_frac", 0.03, 0.5)
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 300, 1500),
        max_depth=trial.suggest_int("max_depth", 10, 40),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 60),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 100),
        max_features=max_features_val,
        bootstrap=True,
        random_state=SEED,
        n_jobs=-1,
    )


def enet_space(trial):
    """ElasticNet 회귀 탐색 공간. 스케일링 필수 — HPO는 fold마다 train fold 기준 RobustScaler를 hpo.py에서 적용한다."""
    return dict(
        alpha=trial.suggest_float("alpha", 1e-7, 1e-3, log=True),       # 전체 정규화 강도
        l1_ratio=trial.suggest_float("l1_ratio", 0.4, 0.99),           # L1 비중 (1.0=Lasso) — 순수 Lasso는 위험해서 0.99까지만
        max_iter=trial.suggest_int("max_iter", 8000, 15000, step=1000),
        tol=1e-6,
        selection="random",   # 좌표하강 시 좌표를 무작위 순서로 — 수렴 빠름
        precompute=True,      # Gram 행렬 미리 계산 (n>>p일 때 빠름)
        random_state=SEED,
    )


def zitboost_space(trial):
    """ZITboost(ZI-Tweedie + LightGBM EM) 탐색 공간 — HP 21개.

    μ(핵심 회귀) 9개 / π(zero 분류) 5개 / φ(분산) 5개 / ZIT 전용(zeta, n_em_iters) 2개.
    """
    params = dict(
        zeta=trial.suggest_float("zeta", 1.1, 1.4),          # Tweedie power (1<ζ<2)
        n_em_iters=trial.suggest_int("n_em_iters", 10, 30),  # EM 반복 횟수 상한
    )
    # μ 모델 (Tweedie mean — 예측의 핵심)
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
    # π 모델 (structural zero 확률)
    params.update(
        pi_n_estimators=trial.suggest_int("pi_n_estimators", 100, 250),
        pi_learning_rate=trial.suggest_float("pi_learning_rate", 0.05, 0.2, log=True),
        pi_num_leaves=trial.suggest_int("pi_num_leaves", 50, 100),
        pi_max_depth=trial.suggest_int("pi_max_depth", 5, 14),
        pi_min_child_samples=trial.suggest_int("pi_min_child_samples", 20, 60),
    )
    # φ 모델 (dispersion)
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


# ============================================================
# zitreg 경로 전용 탐색 공간
# ============================================================
# "zit → reg" 경로에서 쓰는 lgbm 전용 확장 공간. 그 경로의 이전 실험에서 num_leaves/max_depth/n_estimators가
# 상한에 닿고 min_split_gain이 하한에 몰리는 현상이 보여서 범위를 넓혔다. objective는 전부 tweedie를 선호해서
# 후보를 tweedie만(power 1.2/1.3/1.5/1.7) 남겼다. 나머지 4개 모델은 default와 동일하다 (복제만).

def lgbm_zitreg_space(trial):
    """LightGBM 회귀 — zitreg 경로 전용. boundary hit 반영해 범위 확장한 버전."""
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 5000),                # default 3000 → 5000
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        num_leaves=trial.suggest_int("num_leaves", 8, 768),                       # 384 cap에 닿아서 768로
        max_depth=trial.suggest_int("max_depth", 3, 20),                          # 14 cap에 닿아서 20으로
        min_child_samples=trial.suggest_int("min_child_samples", 5, 400),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.1, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 30.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        min_split_gain=trial.suggest_float("min_split_gain", 1e-12, 1.0, log=True),  # 하한 cluster라 1e-12까지
        path_smooth=trial.suggest_float("path_smooth", 0.0, 50.0),
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
        device=DEVICE,
    )
    # tweedie variance_power만 4단계로 (이전 실험에서 1.5 선호 + 인접값 추가 탐색)
    obj_choice = trial.suggest_categorical(
        "objective", ["tweedie_1.2", "tweedie_1.3", "tweedie_1.5", "tweedie_1.7"]
    )
    params["objective"] = "tweedie"
    params["tweedie_variance_power"] = float(obj_choice.split("_")[1])   # "tweedie_1.5" → 1.5
    return params


# 나머지 4종은 default와 동일 (이름만 따로 둬서 zitreg 레지스트리에 등록)
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
    'default' : 일반 회귀 (lgbm/xgb/catboost/et/enet/zitboost)
    'zitreg'  : zit→reg 경로 전용 (lgbm만 범위 확장, 나머지 4개는 default 복제)
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


# ============================================================
# 분류기 (Two-Stage Stage 1)
# ============================================================
# binary 분류로 "y>0인 unit"을 식별 → die-level 확률 OOF 생성. combine 단계에서 reg 예측과 곱해 P(Y>0)×E[Y|Y>0].
# imbalance 대응(scale_pos_weight / class_weight)은 일부는 search space에서 탐색하고,
# 나머지는 hpo 쪽에서 fold마다 클래스 비율로 자동 계산한다(resolve_clf_imbalance).

from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

CLF_REGISTRY = {
    "lgbm":     lgb.LGBMClassifier,
    "xgb":      xgb_lib.XGBClassifier,
    "catboost": CatBoostClassifier,
    "et":       ExtraTreesClassifier,
    "rf":       RandomForestClassifier,
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


# --- 분류 탐색 공간 (objective/loss는 binary 고정 — 손실 다양화는 회귀 쪽에서만 의미 있음) ---

def lgbm_clf_space(trial):
    """LightGBM 분류 탐색 공간. objective='binary' 고정.

    회귀용(lgbm_space)과 비교: n_estimators 상한은 4000→2000으로 줄여 과적합을 억제하고,
    learning_rate 상한은 0.08→0.15로 넓혔으며, num_leaves·max_depth·정규화 항(reg_alpha/lambda)의 범위는 더 넓게 잡았다.
    클래스 불균형 대응으로 scale_pos_weight를 후보 4개 중에서 탐색.
    """
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 2000),
        learning_rate=trial.suggest_float("learning_rate", 0.003, 0.15, log=True),
        num_leaves=trial.suggest_int("num_leaves", 8, 512),
        max_depth=trial.suggest_int("max_depth", 3, 18),
        min_child_samples=trial.suggest_int("min_child_samples", 30, 400),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.1, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 30.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        min_split_gain=trial.suggest_float("min_split_gain", 1e-9, 1.0, log=True),
        path_smooth=trial.suggest_float("path_smooth", 0.0, 50.0),
        scale_pos_weight=trial.suggest_categorical(
            "scale_pos_weight", [1.0, 1.5, 2.43, 3.5]   # 1.0=가중치 없음, 2.43≈음/양 비율
        ),
        objective="binary",
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
        device=DEVICE,
    )


def xgb_clf_space(trial):
    """XGBoost 분류 탐색 공간. objective='binary:logistic' 고정 (lr 상한 0.15, scale_pos_weight 탐색)."""
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 1500),
        learning_rate=trial.suggest_float("learning_rate", 0.003, 0.15, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 14),
        min_child_weight=trial.suggest_float("min_child_weight", 0.5, 100.0, log=True),
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
    """CatBoost 분류 탐색 공간. loss_function='Logloss' 고정 (depth 상한 12, auto_class_weights 탐색)."""
    p = dict(
        iterations=trial.suggest_int("iterations", 100, 3000),
        learning_rate=trial.suggest_float("learning_rate", 0.003, 0.15, log=True),
        depth=trial.suggest_int("depth", 3, 12),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 60.0, log=True),
        random_strength=trial.suggest_float("random_strength", 0.05, 10.0, log=True),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
        border_count=trial.suggest_int("border_count", 32, 192),
        rsm=trial.suggest_float("rsm", 0.3, 1.0),
        loss_function="Logloss",
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
    )
    # "None" 문자열은 categorical 후보로만 쓰고 실제로는 파이썬 None으로 변환
    acw = trial.suggest_categorical("auto_class_weights", ["None", "Balanced"])
    p["auto_class_weights"] = None if acw == "None" else acw
    return p


def et_clf_space(trial):
    """ExtraTrees 분류 탐색 공간 (n_estimators 상한 1500, class_weight 탐색)."""
    mf_kind = trial.suggest_categorical("max_features_kind", ["sqrt", "frac"])
    if mf_kind == "sqrt":
        max_features_val = "sqrt"
    else:
        max_features_val = trial.suggest_float("max_features_frac", 0.03, 0.5)
    p = dict(
        n_estimators=trial.suggest_int("n_estimators", 300, 1500),
        max_depth=trial.suggest_int("max_depth", 10, 40),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 40),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 40),
        max_features=max_features_val,
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
    """모델별 imbalance 옵션을 이번 fold 클래스 비율로 자동 채워 새 dict 반환 (원본 params 불변).

    scale_pos_weight = n_neg / n_pos (음성이 더 많을수록 양성 가중치를 키움).
    catboost/et는 동등한 효과의 auto_class_weights / class_weight 사용.
    setdefault라 search space에서 이미 정해진 값이 있으면 그걸 우선한다.
    """
    p = dict(params)
    n_pos = int((np.asarray(y_train_bin) == 1).sum())
    n_neg = int((np.asarray(y_train_bin) == 0).sum())
    spw = (n_neg / n_pos) if n_pos > 0 else 1.0

    if model_name in {"lgbm", "xgb"}:
        p.setdefault("scale_pos_weight", spw)
    elif model_name == "catboost":
        p.setdefault("auto_class_weights", "Balanced")
        if p.get("auto_class_weights") == "None":   # "None" 문자열이 남아 있으면 진짜 None으로
            p["auto_class_weights"] = None
    elif model_name == "et":
        p.setdefault("class_weight", "balanced")
        if p.get("class_weight") == "None":
            p["class_weight"] = None
    return p
