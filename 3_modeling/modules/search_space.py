"""
Optuna Search Space 정의 — 모델별 하이퍼파라미터 + 전처리 탐색 공간

- 모델별 하이퍼파라미터: lgbm_space, xgb_space, catboost_space, rf_space
- 전처리/이상치/집계 통합: preprocessing_space (LGBM-only baseline용)

preprocessing_space의 후보 list는 모듈 상수(PP_*_CANDIDATES)로 노출되어
노트북에서 보고 바로 좁히거나 원소를 지워 탐색 공간을 축소할 수 있다.
"""
from utils.config import SEED
from .model_zoo import DEVICE


def lgbm_space(trial, prefix=""):
    """LightGBM 탐색 공간 — prefix에 따라 clf/reg 다른 범위 사용

    범위는 study `3-199-005` (LGBM Two-Stage, 501 trials, best RMSE=0.005658)
    상위 100 trial 분포 기반으로 재조정. clf/reg가 정반대 영역을 선호하므로
    prefix로 분기. 양봉(bimodal)인 영역은 보존.
    """
    p = prefix

    if p == "clf_":
        # Stage1 분류기: 깊고 큰 트리 + 약한 정규화 선호
        # (top100: n_est 2135~2954, num_leaves 187~256, max_depth 10~12)
        return dict(
            n_estimators=trial.suggest_int(f"{p}n_estimators", 1500, 3000),
            learning_rate=trial.suggest_float(f"{p}learning_rate", 0.008, 0.05, log=True),
            num_leaves=trial.suggest_int(f"{p}num_leaves", 128, 320),
            max_depth=trial.suggest_int(f"{p}max_depth", 8, 14),
            min_child_samples=trial.suggest_int(f"{p}min_child_samples", 5, 50),
            subsample=trial.suggest_float(f"{p}subsample", 0.75, 1.0),
            colsample_bytree=trial.suggest_float(f"{p}colsample_bytree", 0.7, 1.0),
            reg_alpha=trial.suggest_float(f"{p}reg_alpha", 1e-6, 1e-2, log=True),
            reg_lambda=trial.suggest_float(f"{p}reg_lambda", 1e-4, 1.0, log=True),
            min_split_gain=trial.suggest_float(f"{p}min_split_gain", 1e-4, 0.2, log=True),
            path_smooth=trial.suggest_float(f"{p}path_smooth", 0.0, 25.0),
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
            device=DEVICE,
        )

    if p == "reg_":
        # Stage2 회귀기: 더 큰 num_leaves + 강한 정규화(reg_alpha) 선호
        # 상한 포화 → 확장: num_leaves 256→384, reg_alpha 10→30, min_child_samples 300→400
        # 양봉 → 보존: colsample_bytree(0.2~0.3 vs 0.9~1.0), path_smooth(0~5 vs 30~38)
        return dict(
            n_estimators=trial.suggest_int(f"{p}n_estimators", 100, 1500),
            learning_rate=trial.suggest_float(f"{p}learning_rate", 0.005, 0.1, log=True),
            num_leaves=trial.suggest_int(f"{p}num_leaves", 64, 384),
            max_depth=trial.suggest_int(f"{p}max_depth", 5, 14),
            min_child_samples=trial.suggest_int(f"{p}min_child_samples", 5, 400),
            subsample=trial.suggest_float(f"{p}subsample", 0.6, 0.9),
            colsample_bytree=trial.suggest_float(f"{p}colsample_bytree", 0.1, 1.0),
            reg_alpha=trial.suggest_float(f"{p}reg_alpha", 1e-3, 30.0, log=True),
            reg_lambda=trial.suggest_float(f"{p}reg_lambda", 1e-7, 1e-2, log=True),
            min_split_gain=trial.suggest_float(f"{p}min_split_gain", 1e-9, 1e-5, log=True),
            path_smooth=trial.suggest_float(f"{p}path_smooth", 0.0, 50.0),
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
            device=DEVICE,
        )

    # 단일 모델(non-Two-Stage) 또는 알 수 없는 prefix: 기본 wide 범위
    return dict(
        n_estimators=trial.suggest_int(f"{p}n_estimators", 100, 3000),
        learning_rate=trial.suggest_float(f"{p}learning_rate", 0.005, 0.3, log=True),
        num_leaves=trial.suggest_int(f"{p}num_leaves", 8, 384),
        max_depth=trial.suggest_int(f"{p}max_depth", 3, 14),
        min_child_samples=trial.suggest_int(f"{p}min_child_samples", 5, 400),
        subsample=trial.suggest_float(f"{p}subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float(f"{p}colsample_bytree", 0.1, 1.0),
        reg_alpha=trial.suggest_float(f"{p}reg_alpha", 1e-8, 30.0, log=True),
        reg_lambda=trial.suggest_float(f"{p}reg_lambda", 1e-8, 10.0, log=True),
        min_split_gain=trial.suggest_float(f"{p}min_split_gain", 1e-9, 1.0, log=True),
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


# ═════════════════════════════════════════════════════════════
# 전처리 + 이상치 + 집계 통합 search space
# ═════════════════════════════════════════════════════════════
#
# 아래 PP_*_CANDIDATES는 Optuna trial에서 선택될 후보 list.
# 노트북에서 import 해서 확인하거나, 특정 후보만 남기고 싶을 때
# 리스트 원소를 지우는 방식으로 탐색 공간을 축소할 수 있다.
#
# e.g. 노트북에서 범위 좁히기:
#   from modules.search_space import PP_CLEAN_CANDIDATES
#   PP_CLEAN_CANDIDATES['imputation_method'] = ['median']  # spatial/knn 제외
#
# ═════════════════════════════════════════════════════════════

# ── 클리닝 후보 ───────────────────────────────────────────────
PP_CLEAN_CANDIDATES = {
    "const_threshold":            [1e-6],
    "missing_threshold":          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "remove_duplicates":          [True],
    "corr_threshold":             [0.90, 0.94, 0.98],
    "corr_keep_by":               ["std", "target_corr"],
    "add_indicator":              [True, False],
    "indicator_threshold":        [0.01, 0.05, 0.10, 0.15, 0.20, 0.25],
    "imputation_method":          ["spatial"],
    "knn_neighbors":              [3, 5, 10],
    "spatial_max_dist":           [1.0, 2.0, 3.0, 4.0, 5.0],
    # imputation 후 2차 corr 제거 (인공 상관 제거 목적, 보수적)
    "post_impute_corr_threshold": [0.97, 0.98, 0.99],
    "post_impute_corr_keep_by":   ["std", "target_corr"],
}

# ── 이상치 후보 ───────────────────────────────────────────────
PP_OUTLIER_CANDIDATES = {
    "method":         ["winsorize"], #,"grubbs", "lot_local", "none", "iqr_clip"],
    "lower_pct":      [0.0],
    "upper_pct":      [0.99],
    "iqr_multiplier": [1.5, 3.0, 5.0],
}

# ── 집계 후보 (presets) ────────────────────────────────────────
# aggregate_die_to_unit이 네이티브 지원하는 조합만 등록.
# (cv, range는 특별 처리. mean/std/min/max/median은 pandas groupby 지원)
AGG_PRESETS = [
    # 0: 현재 기본 6종
    ["mean", "std", "range", "min", "max", "median"],
    # 1: 핵심 2종 (빠른 탐색용)
    ["mean", "std"],
    # 2: 중심경향 + 편차 위주 (극값 제외)
    ["mean", "std","median", "range"],
    # 3: 편차 중심 (mean 배제, 상관 약함 대응)
    ["std", "range", "median"],
]

# ── 집계 preset 인덱스 후보 ───────────────────────────────────
PP_AGG_PRESET_IDX_CANDIDATES = list(range(len(AGG_PRESETS)))


# ── run_cleaning / run_outlier_treatment에 실제 전달할 인자 키 ──
# (preprocessing_space 반환 dict에서 해당 함수로 분배할 때 사용)
CLEANING_KEYS = [
    "const_threshold",
    "missing_threshold",
    "remove_duplicates",
    "corr_threshold",
    "corr_keep_by",
    "add_indicator",
    "indicator_threshold",
    "imputation_method",
    "knn_neighbors",
    "spatial_max_dist",
    "post_impute_corr_threshold",
    "post_impute_corr_keep_by",
]

OUTLIER_KEYS = [
    "method",
    "lower_pct",
    "upper_pct",
    "iqr_multiplier",
]


def preprocessing_space(trial):
    """
    전처리 + 이상치 + 집계 통합 search space.

    Returns
    -------
    dict
        {
            # cleaning (CLEANING_KEYS)
            "const_threshold": ...,
            ...
            # outlier (OUTLIER_KEYS, key에 'outlier__' prefix)
            "outlier__method": ...,
            "outlier__lower_pct": ...,
            ...
            # 집계
            "agg_preset_idx": int,
        }

    trial param 이름은 모두 'pp_' prefix로 통일 (clf_/reg_와 구분).
    """
    params = {}

    # ── 클리닝 ──
    for key, candidates in PP_CLEAN_CANDIDATES.items():
        params[key] = trial.suggest_categorical(f"pp_{key}", candidates)

    # ── 이상치 (key에 'outlier__' prefix 유지, run_outlier 호출 시 strip) ──
    for key, candidates in PP_OUTLIER_CANDIDATES.items():
        params[f"outlier__{key}"] = trial.suggest_categorical(
            f"pp_outlier_{key}", candidates
        )

    # ── 집계 preset ──
    params["agg_preset_idx"] = trial.suggest_categorical(
        "pp_agg_preset_idx", PP_AGG_PRESET_IDX_CANDIDATES
    )

    return params


def split_pp_params(pp_params):
    """
    preprocessing_space 반환 dict를 run_cleaning / run_outlier_treatment /
    aggregate 함수의 인자로 분배.

    Returns
    -------
    cleaning_args : dict — run_cleaning에 넘길 kwargs
    outlier_args  : dict — run_outlier_treatment에 넘길 kwargs
    agg_funcs     : list — aggregate_die_to_unit에 넘길 집계 함수 list
    """
    cleaning_args = {k: v for k, v in pp_params.items() if k in CLEANING_KEYS}

    outlier_args = {}
    for k, v in pp_params.items():
        if k.startswith("outlier__"):
            outlier_args[k[len("outlier__"):]] = v

    agg_idx = pp_params.get("agg_preset_idx", 0)
    agg_funcs = AGG_PRESETS[agg_idx]

    return cleaning_args, outlier_args, agg_funcs


def extract_pp_params_from_best(best_params):
    """
    Optuna study.best_params에서 'pp_' prefix 항목만 추출해
    preprocessing_space가 반환하는 형식으로 복원.

    trial param 이름 'pp_outlier_method' → dict key 'outlier__method' 로 변환.
    trial param 'pp_agg_preset_idx' → dict key 'agg_preset_idx'.
    """
    out = {}
    for k, v in best_params.items():
        if not k.startswith("pp_"):
            continue
        inner = k[len("pp_"):]
        if inner.startswith("outlier_"):
            # pp_outlier_method → outlier__method
            out[f"outlier__{inner[len('outlier_'):]}"] = v
        else:
            out[inner] = v
    return out
