"""
Optuna Search Space 정의 — 모델별 하이퍼파라미터 + 전처리 탐색 공간

- 모델별 하이퍼파라미터: lgbm_space, rf_space
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
        params = dict(
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
        # ★ 2차 신규: objective 탐색 (논문 2-2/2-5/2-6/2-7 근거)
        # zero-inflated + 비음수 타깃엔 Tweedie/Poisson이 MSE보다 우위일 수 있음.
        obj_choice = trial.suggest_categorical(
            f"{p}objective",
            ['regression', 'poisson', 'tweedie_1.2', 'tweedie_1.5'],
        )
        if obj_choice == 'poisson':
            params['objective'] = 'poisson'
        elif obj_choice.startswith('tweedie'):
            params['objective'] = 'tweedie'
            params['tweedie_variance_power'] = float(obj_choice.split('_')[1])
        # 'regression' (MSE, LGBM default) 선택 시 주입 없음
        return params

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


def rf_space(trial, prefix=""):
    """RandomForest 탐색 공간 (ET는 et_space 사용)"""
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


# ═════════════════════════════════════════════════════════════
# 2차 funnel 신규: ExtraTrees / LogReg-enet / ElasticNet space
# ═════════════════════════════════════════════════════════════

def et_space(trial, prefix=""):
    """ExtraTrees 탐색 공간 (분류·회귀 공통).

    RandomForest 대비 특징:
    - bootstrap 기본 False (ET 이론) — 탐색 축으로 열어둠
    - split threshold가 random이라 n_estimators를 RF보다 넉넉히 (300~)
    """
    p = prefix
    return dict(
        n_estimators=trial.suggest_int(f"{p}n_estimators", 300, 1500),
        max_depth=trial.suggest_int(f"{p}max_depth", 6, 30),
        min_samples_leaf=trial.suggest_int(f"{p}min_samples_leaf", 1, 30),
        min_samples_split=trial.suggest_int(f"{p}min_samples_split", 2, 40),
        max_features=trial.suggest_categorical(
            f"{p}max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]
        ),
        bootstrap=trial.suggest_categorical(f"{p}bootstrap", [True, False]),
        random_state=SEED,
        n_jobs=-1,
    )


def logreg_enet_space(trial, prefix=""):
    """LogisticRegression(penalty='elasticnet', solver='saga') 탐색 공간.

    - saga는 scaled data 전제 → hybrid_scale 적용 후 입력 필수
    - l1_ratio 0/1 끝단 제외 (순수 L1/L2는 엣지 케이스)
    """
    p = prefix
    return dict(
        penalty='elasticnet',
        solver='saga',
        C=trial.suggest_float(f"{p}C", 1e-3, 100.0, log=True),
        l1_ratio=trial.suggest_float(f"{p}l1_ratio", 0.1, 0.9),
        max_iter=trial.suggest_int(f"{p}max_iter", 1000, 4000, step=500),
        tol=1e-3,
        random_state=SEED,
        n_jobs=-1,
    )


def enet_space(trial, prefix=""):
    """ElasticNet 회귀 탐색 공간.

    약신호(EDA max|r|=0.037) + 다중공선성 47쌍 → 순수 Lasso(l1_ratio=1) 위험.
    Optuna가 고른 best l1_ratio가 쏠리면 데이터의 L1/L2 선호 진단치로 활용.
    """
    p = prefix
    return dict(
        alpha=trial.suggest_float(f"{p}alpha", 1e-5, 1.0, log=True),
        l1_ratio=trial.suggest_float(f"{p}l1_ratio", 0.1, 0.9),
        max_iter=trial.suggest_int(f"{p}max_iter", 2000, 8000, step=1000),
        tol=1e-4,
        random_state=SEED,
    )


def zitboost_space(trial, prefix=""):
    """ZI-Tweedie + LightGBM EM 탐색 공간 (21개 HP).

    μ(핵심 회귀): 9개 full HP
    π(zero 확률): 5개 medium HP
    φ(분산):      5개 medium HP
    ZIT 전용:     zeta, n_em_iters
    """
    p = prefix

    # ── ZIT 전용 (2개) ──
    params = dict(
        zeta=trial.suggest_float(f"{p}zeta", 1.1, 1.9),
        n_em_iters=trial.suggest_int(f"{p}n_em_iters", 3, 20),
    )

    # ── μ 모델: 핵심 회귀 (9개) ──
    params.update(
        mu_n_estimators=trial.suggest_int(f"{p}mu_n_estimators", 100, 2000),
        mu_learning_rate=trial.suggest_float(f"{p}mu_learning_rate", 0.005, 0.1, log=True),
        mu_num_leaves=trial.suggest_int(f"{p}mu_num_leaves", 32, 256),
        mu_max_depth=trial.suggest_int(f"{p}mu_max_depth", 5, 12),
        mu_min_child_samples=trial.suggest_int(f"{p}mu_min_child_samples", 5, 100),
        mu_subsample=trial.suggest_float(f"{p}mu_subsample", 0.5, 1.0),
        mu_colsample_bytree=trial.suggest_float(f"{p}mu_colsample_bytree", 0.3, 1.0),
        mu_reg_alpha=trial.suggest_float(f"{p}mu_reg_alpha", 1e-8, 10.0, log=True),
        mu_reg_lambda=trial.suggest_float(f"{p}mu_reg_lambda", 1e-8, 10.0, log=True),
    )

    # ── π 모델: zero 확률 분류 (5개) ──
    params.update(
        pi_n_estimators=trial.suggest_int(f"{p}pi_n_estimators", 50, 500),
        pi_learning_rate=trial.suggest_float(f"{p}pi_learning_rate", 0.01, 0.1, log=True),
        pi_num_leaves=trial.suggest_int(f"{p}pi_num_leaves", 16, 128),
        pi_max_depth=trial.suggest_int(f"{p}pi_max_depth", 3, 8),
        pi_min_child_samples=trial.suggest_int(f"{p}pi_min_child_samples", 10, 100),
    )

    # ── φ 모델: 분산 (5개) ──
    params.update(
        phi_n_estimators=trial.suggest_int(f"{p}phi_n_estimators", 50, 500),
        phi_learning_rate=trial.suggest_float(f"{p}phi_learning_rate", 0.01, 0.1, log=True),
        phi_num_leaves=trial.suggest_int(f"{p}phi_num_leaves", 16, 128),
        phi_max_depth=trial.suggest_int(f"{p}phi_max_depth", 3, 8),
        phi_min_child_samples=trial.suggest_int(f"{p}phi_min_child_samples", 10, 100),
    )

    # ── 공통 ──
    params.update(
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
        device=DEVICE,
    )

    return params


SEARCH_SPACES = {
    "lgbm":        lgbm_space,
    "rf":          rf_space,
    "et":          et_space,             # ★ 2차: rf_space → et_space 교체 (bootstrap 축 추가)
    "logreg_enet": logreg_enet_space,    # ★ 2차 신규
    "enet":        enet_space,           # ★ 2차 신규
    "zitboost":    zitboost_space,       # ★ ZITboost (ZI-Tweedie + EM)
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


# ═════════════════════════════════════════════════════════════
# 2차 funnel 신규 PP 후보 (IsoForest / LDS)
#
# Scaling(HybridScaler)과 Binarize는 옵튜나 탐색 X — 고정값(PP_SCALE_CONFIG /
# PP_BINARIZE_CONFIG)으로만 사용. test.ipynb 4종 비교 결과 skew_threshold=5.0
# 정규근사율 68%로 최고 확정.
# ═════════════════════════════════════════════════════════════

# ── IsoForest anomaly score 후보 (컬럼 추가, 제거 아님) ──
PP_ISO_ANOMALY_CANDIDATES = {
    "iso_enabled":       [True, False],           # on/off marginal 측정
    "iso_contamination": [0.05, 0.1, 'auto'],
    "iso_n_estimators":  [100, 200],
}

# ── LDS 가중치 후보 (Y>0 long-tail 대응) ──
PP_LDS_CANDIDATES = {
    "lds_enabled":    [True, False],
    "lds_sigma":      [0.005, 0.01, 0.02],        # y>0 평균 0.0087 기준
    "lds_max_weight": [5.0, 10.0],                # soft cap (재정규화 후 초과 가능)
}

# ── HybridScaler 고정 설정 (옵튜나 탐색 X) ──
# skew_threshold: test.ipynb 확정 (|skew|>5 → quantile, 나머지 → power)
# binary_passthrough: nunique≤2는 변환 없음 (binarize 결과 보존)
PP_SCALE_CONFIG = {
    "transform":         "hybrid",
    "skew_threshold":    5.0,
    "binary_passthrough": True,
}

# ── Binarize 고정 설정 (cleaning 직후) ──
# top%>0.95 OR nunique≤5 → 0/1 (int8)
# (zitboost_experiment/ensemble_2nd에서 재현성 로그용으로 계속 참조)
PP_BINARIZE_CONFIG = {
    "apply":               True,
    "top_value_threshold": 0.95,
    "max_unique":          5,
}

# ── Binarize 후보 (baseline 전용 Optuna 탐색) ──
# 기본값 apply=[False]로 두어 기존 노트북(zitboost/ensemble_2nd)의 Binarize OFF
# 동작을 보존한다. baseline.ipynb에서 dict 뮤테이트로만 override해 ON/OFF 탐색:
#   PP_BINARIZE_CANDIDATES['apply'] = [True, False]
PP_BINARIZE_CANDIDATES = {
    "apply":               [False],
    "top_value_threshold": [0.95],
    "max_unique":          [5],
}


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
    전처리 + 이상치 + IsoForest + LDS + 집계 통합 search space.

    Returns
    -------
    dict
        {
            # cleaning (CLEANING_KEYS, flat)
            "const_threshold": ...,
            ...
            # outlier (OUTLIER_KEYS, key에 'outlier__' prefix)
            "outlier__method": ..., "outlier__lower_pct": ..., ...
            # ★ 2차 신규: IsoForest anomaly score ('iso__' prefix)
            "iso__iso_enabled": ..., "iso__iso_contamination": ..., ...
            # ★ 2차 신규: LDS weight ('lds__' prefix)
            "lds__lds_enabled": ..., "lds__lds_sigma": ..., ...
            # 집계
            "agg_preset_idx": int,
        }

    Scaling(hybrid)과 Binarize는 여기서 탐색 X — PP_SCALE_CONFIG /
    PP_BINARIZE_CONFIG 고정값을 _run_preprocessing에서 직접 사용.

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

    # ── ★ Binarize (2차 신규, baseline 전용 탐색 — 기본값 apply=[False]) ──
    for key, candidates in PP_BINARIZE_CANDIDATES.items():
        params[f"binarize__{key}"] = trial.suggest_categorical(
            f"pp_binarize_{key}", candidates
        )

    # ── ★ IsoForest (2차 신규) ──
    for key, candidates in PP_ISO_ANOMALY_CANDIDATES.items():
        params[f"iso__{key}"] = trial.suggest_categorical(
            f"pp_iso_{key}", candidates
        )

    # ── ★ LDS (2차 신규) ──
    for key, candidates in PP_LDS_CANDIDATES.items():
        params[f"lds__{key}"] = trial.suggest_categorical(
            f"pp_lds_{key}", candidates
        )

    # ── 집계 preset ──
    params["agg_preset_idx"] = trial.suggest_categorical(
        "pp_agg_preset_idx", PP_AGG_PRESET_IDX_CANDIDATES
    )

    return params


def split_pp_params(pp_params):
    """
    preprocessing_space 반환 dict를 전처리 함수들의 인자로 분배.

    ★ 2차부터 6-tuple 반환 — binarize_args, iso_args, lds_args 추가.
    HybridScaler는 여전히 탐색 대상 아님 (PP_SCALE_CONFIG 고정).

    Returns
    -------
    cleaning_args : dict — run_cleaning에 넘길 kwargs
    outlier_args  : dict — run_outlier_treatment에 넘길 kwargs
    binarize_args : dict — binarize_degenerate 호출 제어
                   ({'apply', 'top_value_threshold', 'max_unique'})
    iso_args      : dict — multivariate_anomaly_score에 넘길 kwargs
                   ({'iso_enabled', 'iso_contamination', 'iso_n_estimators'})
    lds_args      : dict — compute_lds_weights에 넘길 kwargs
                   ({'lds_enabled', 'lds_sigma', 'lds_max_weight'})
    agg_funcs     : list — aggregate_die_to_unit에 넘길 집계 함수 list
    """
    cleaning_args = {k: v for k, v in pp_params.items() if k in CLEANING_KEYS}

    outlier_args = {k[len("outlier__"):]: v
                    for k, v in pp_params.items() if k.startswith("outlier__")}

    binarize_args = {k[len("binarize__"):]: v
                     for k, v in pp_params.items() if k.startswith("binarize__")}

    iso_args = {k[len("iso__"):]: v
                for k, v in pp_params.items() if k.startswith("iso__")}

    lds_args = {k[len("lds__"):]: v
                for k, v in pp_params.items() if k.startswith("lds__")}

    agg_idx = pp_params.get("agg_preset_idx", 0)
    agg_funcs = AGG_PRESETS[agg_idx]

    return cleaning_args, outlier_args, binarize_args, iso_args, lds_args, agg_funcs


def extract_pp_params_from_best(best_params):
    """
    Optuna study.best_params에서 'pp_' prefix 항목만 추출해
    preprocessing_space가 반환하는 형식으로 복원.

    trial param 이름 → dict key 변환:
      'pp_outlier_method'       → 'outlier__method'
      'pp_binarize_apply'       → 'binarize__apply'
      'pp_iso_iso_enabled'      → 'iso__iso_enabled'
      'pp_lds_lds_sigma'        → 'lds__lds_sigma'
      'pp_agg_preset_idx'       → 'agg_preset_idx'
      그 외                     → 'pp_' 떼고 그대로
    """
    out = {}
    for k, v in best_params.items():
        if not k.startswith("pp_"):
            continue
        inner = k[len("pp_"):]
        if inner.startswith("outlier_"):
            out[f"outlier__{inner[len('outlier_'):]}"] = v
        elif inner.startswith("binarize_"):
            out[f"binarize__{inner[len('binarize_'):]}"] = v
        elif inner.startswith("iso_"):
            out[f"iso__{inner[len('iso_'):]}"] = v
        elif inner.startswith("lds_"):
            out[f"lds__{inner[len('lds_'):]}"] = v
        else:
            out[inner] = v
    return out
