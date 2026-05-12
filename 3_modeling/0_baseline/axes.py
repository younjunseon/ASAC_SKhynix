"""
0_baseline ablation 실험의 축 정의 + grid 생성 + 단일 entrypoint.

두 노트북(01_oat.ipynb, 02_group_study.ipynb)이 공통으로 사용하는 모듈.

핵심:
- REFERENCE / AXES / AGG_PRESET_LIB / SEEDS — strategy.md §4 그대로
- generate_oat_grid() — OAT 셀 (axis, option, seed) 튜플 생성
- run_one(cfg, seed) — 단일 셀 학습. baseline.ipynb의 rerun_best_trial_with_pp 호출

데이터는 노트북에서 한 번 로드 후 axes.set_data(...)로 모듈 캐시에 저장 →
run_one 호출 시 자동 사용.
"""
from __future__ import annotations

import time
from copy import deepcopy

import numpy as np
import pandas as pd

# 0_baseline 격리 모듈 (3_modeling_이전자료/modules/ 복사 + 수정본).
# 노트북에서 sys.path에 0_baseline 디렉토리를 등록하면 _modules는 sub-package로 발견됨.
from _modules.e2e_hpo import rerun_best_trial_with_pp
from _modules import search_space as ss
from _modules.aggregate import aggregate_die_to_unit  # noqa: F401  (등록 확인용)


# ═════════════════════════════════════════════════════════════
# 1. baseline.ipynb 정합 — EXCLUDE_COLS / PP pin 값
# ═════════════════════════════════════════════════════════════

# baseline.ipynb cell 2 — 웨이퍼맵 사전 제외 (54개)
EXCLUDE_COLS = [
    "X124", "X300", "X301",
    # X441~X464
    "X441", "X442", "X443", "X444", "X445", "X446", "X447", "X448",
    "X449", "X450", "X451", "X452", "X453", "X454", "X455", "X456",
    "X457", "X458", "X459", "X460", "X461", "X462", "X463", "X464",
    # X499~X506
    "X499", "X500", "X501", "X502", "X503", "X504", "X505", "X506",
    # X658~X687 일부
    "X658", "X659", "X671", "X672",
    "X674", "X675", "X676", "X677",
    "X680", "X681",
    "X683", "X684", "X685", "X686", "X687",
    # 개별
    "X1041", "X1074", "X1078",
    "X1086",
]


# baseline.ipynb cell 6 — PP pin (OAT 흔드는 축 외 모두 고정)
# strategy.md §3.3 "PP base" 와 동일.
PP_PIN_CLEANING = {
    "const_threshold":            1e-6,
    "missing_threshold":          0.4,
    "remove_duplicates":          True,
    "corr_threshold":             0.9,
    "corr_keep_by":               "std",
    "add_indicator":              True,
    "indicator_threshold":        0.05,
    "imputation_method":          "spatial",   # OAT 'impute' 축에서 override
    "knn_neighbors":              5,
    "spatial_max_dist":           5.0,
    "post_impute_corr_threshold": 0.98,
    "post_impute_corr_keep_by":   "std",
}

PP_PIN_OUTLIER = {
    "method":         "winsorize",   # OAT 'outlier' 축에서 override
    "lower_pct":      0.0,
    "upper_pct":      0.99,
    "iqr_multiplier": 1.5,
}

PP_PIN_BINARIZE = {
    "apply":               False,    # OAT 'binarize' 축에서 override
    "top_value_threshold": 0.95,
    "max_unique":          5,
}

PP_PIN_ISO = {
    "iso_enabled":       False,      # OAT 'iso_anomaly' 축에서 override
    "iso_contamination": 0.05,
    "iso_n_estimators":  100,
}

PP_PIN_LDS = {
    "lds_enabled":    False,         # OAT 'lds' 축에서 override
    "lds_sigma":      0.01,
    "lds_max_weight": 5.0,
}

PP_PIN_GE = {
    "use_encoder":      False,
    "alpha":            20.0,
    "n_folds":          5,
    "group_specs_name": "default",
}


# ═════════════════════════════════════════════════════════════
# 2. agg_preset 12종 (strategy.md §4.3) + ss.AGG_PRESETS 확장
# ═════════════════════════════════════════════════════════════

AGG_PRESET_LIB = {
    # baseline.ipynb 원본 4종 (idx 0~3 보존)
    "P00_full6":     ["mean", "std", "range", "min", "max", "median"],
    "P01_meanstd":   ["mean", "std"],
    "P02_basic4":    ["mean", "std", "median", "range"],
    "P03_disp3":     ["std", "range", "median"],
    # 단일 함수 4종
    "P04_mean":      ["mean"],
    "P05_median":    ["median"],
    "P06_max":       ["max"],
    "P07_min":       ["min"],
    # 극단/분위수
    "P08_extremes":  ["max", "min", "range"],
    "P09_quartiles": ["Q25", "median", "Q75"],
    # 풀세트
    "P10_full8":     ["mean", "std", "range", "min", "max", "median", "Q25", "Q75"],
    "P11_full9_cv":  ["mean", "std", "range", "min", "max", "median", "Q25", "Q75", "cv"],
}

# ss.AGG_PRESETS 모듈 리스트를 12종으로 확장 (idx 매핑 보존).
# split_pp_params가 'agg_preset_idx'로 ss.AGG_PRESETS[idx]를 가져가므로
# 이 확장은 _modules에 정의된 baseline 격리 search_space 모듈 한정으로 적용된다.
ss.AGG_PRESETS.clear()
ss.AGG_PRESETS.extend(AGG_PRESET_LIB.values())

AGG_PRESET_KEYS = list(AGG_PRESET_LIB.keys())
AGG_PRESET_KEY_TO_IDX = {k: i for i, k in enumerate(AGG_PRESET_KEYS)}


# ═════════════════════════════════════════════════════════════
# 3. Reference 1점 + 11축 + Seed 5개 (strategy.md §4.1, §4.2)
# ═════════════════════════════════════════════════════════════

REFERENCE = {
    "CLF":              "off",          # run_clf=False
    "reg_level":        "position",
    "TARGET_TRANSFORM": "yeo-johnson",
    "CLIP_Y_EXTREME":   True,
    "loss":             "regression",   # LGBM default = mse
    "impute":           "spatial",
    "outlier":          "winsorize",
    "agg_preset":       "P00_full6",
    "binarize":         False,
    "iso_anomaly":      False,
    "lds":              False,
}

AXES = {
    "CLF":              ["off", "proba", "proba+filter", "binary"],
    "reg_level":        ["unit", "position"],
    "TARGET_TRANSFORM": ["none", "log1p", "yeo-johnson"],
    "CLIP_Y_EXTREME":   [True, False],
    "loss":             ["regression", "poisson", "tweedie", "huber"],
    "impute":           ["spatial", "median", "knn"],
    "outlier":          ["none", "winsorize", "iqr_clip", "grubbs", "lot_local"],
    "agg_preset":       AGG_PRESET_KEYS,
    "binarize":         [True, False],
    "iso_anomaly":      [True, False],
    "lds":              [True, False],
}

# baseline은 LGBM default 고정(sampling 없어 결정론적) + run_one이 seed를 모델에 안 넘김 →
# seed를 늘려도 결과가 비트 단위로 동일하다 (master.csv val_std 전 행 0.0 실측).
# strategy_common §7 "모델 random_state=42 고정" 정책과도 일치하므로 1개만 쓴다.
SEEDS = [42]


# ═════════════════════════════════════════════════════════════
# 4. OAT grid 생성
# ═════════════════════════════════════════════════════════════

def generate_oat_grid():
    """OAT cell list 생성. reference 1개 + 각 축의 non-reference 옵션.

    Returns
    -------
    list of (axis_name, option, seed, cfg_dict, is_reference)
        - axis_name : 'reference' (참조 셀) 또는 흔든 축 이름
        - option : 그 축의 옵션 값
        - seed : SEEDS 중 1개 (baseline은 [42] 1개)
        - cfg_dict : REFERENCE 사본에 해당 axis만 option으로 override.
                     단 axis=='agg_preset'이면 reg_level도 'unit'으로 함께 override
                     (position에선 agg_preset이 no-op이라 unit에서 12개 전부 비교).
        - is_reference : reference 셀 여부 (master.csv 기록용)
    """
    grid = []

    # reference × seed들
    for seed in SEEDS:
        grid.append(("reference", REFERENCE[next(iter(REFERENCE))], seed,
                     deepcopy(REFERENCE), True))

    # 각 축의 non-reference 옵션 × seed들
    for axis, options in AXES.items():
        ref_val = REFERENCE[axis]
        for opt in options:
            # agg_preset은 reg_level='position'(reference)에서는 die→unit 집계를 안 써서 no-op이다
            # (_modules/e2e_hpo._prepare_unit_data: position 모드는 agg_funcs 무시).
            # 집계가 실제로 쓰이는 reg_level='unit'으로 바꿔 12개 프리셋 전부 흔든다.
            # reference가 position이라 P00_full6도 비교 대상에 포함 — 이 축은 unit 셀들로 자기완결.
            if axis == "agg_preset":
                cfg = deepcopy(REFERENCE)
                cfg["reg_level"] = "unit"
                cfg["agg_preset"] = opt
                for seed in SEEDS:
                    grid.append((axis, opt, seed, cfg, False))
                continue
            if opt == ref_val:
                continue
            cfg = deepcopy(REFERENCE)
            cfg[axis] = opt
            for seed in SEEDS:
                grid.append((axis, opt, seed, cfg, False))

    return grid


# ═════════════════════════════════════════════════════════════
# 5. cfg → e2e 인자 매핑 helpers
# ═════════════════════════════════════════════════════════════

def _cfg_to_pp_params(cfg: dict) -> dict:
    """cfg → split_pp_params가 받을 dict 형식.

    - cleaning 키: 그대로 (CLEANING_KEYS에 매칭)
    - outlier_args: 'outlier__method' 등
    - binarize/iso/lds: 'binarize__apply', 'iso__iso_enabled', 'lds__lds_enabled' 등
    - agg_preset_idx: AGG_PRESETS list 인덱스 (0~11)

    pin 값에 cfg 축으로 override 적용.
    """
    pp = {}

    # cleaning (impute 축 override)
    cleaning = dict(PP_PIN_CLEANING)
    cleaning["imputation_method"] = cfg["impute"]
    pp.update(cleaning)

    # outlier (outlier 축 override)
    outlier = dict(PP_PIN_OUTLIER)
    outlier["method"] = cfg["outlier"]
    for k, v in outlier.items():
        pp[f"outlier__{k}"] = v

    # binarize (binarize 축 override)
    binarize = dict(PP_PIN_BINARIZE)
    binarize["apply"] = cfg["binarize"]
    for k, v in binarize.items():
        pp[f"binarize__{k}"] = v

    # iso_anomaly (iso 축 override)
    iso = dict(PP_PIN_ISO)
    iso["iso_enabled"] = cfg["iso_anomaly"]
    for k, v in iso.items():
        pp[f"iso__{k}"] = v

    # lds (lds 축 override)
    lds = dict(PP_PIN_LDS)
    lds["lds_enabled"] = cfg["lds"]
    for k, v in lds.items():
        pp[f"lds__{k}"] = v

    # ge (group encoder — baseline은 OFF 고정)
    for k, v in PP_PIN_GE.items():
        pp[f"ge__{k}"] = v

    # agg_preset_idx
    pp["agg_preset_idx"] = AGG_PRESET_KEY_TO_IDX[cfg["agg_preset"]]

    return pp


def _cfg_to_pipeline_config(cfg: dict) -> dict:
    """cfg → e2e_hpo의 pipeline_config dict.

    CLF 옵션 4종 매핑:
      'off'           → run_clf=False
      'proba'         → run_clf=True, clf_output='proba', clf_filter=False
      'proba+filter'  → run_clf=True, clf_output='proba', clf_filter=True
      'binary'        → run_clf=True, clf_output='binary', clf_filter=False
    """
    clf_map = {
        "off":          dict(run_clf=False, clf_output="proba", clf_filter=False),
        "proba":        dict(run_clf=True,  clf_output="proba", clf_filter=False),
        "proba+filter": dict(run_clf=True,  clf_output="proba", clf_filter=True),
        "binary":       dict(run_clf=True,  clf_output="binary", clf_filter=False),
    }
    clf_cfg = clf_map[cfg["CLF"]]

    return dict(
        input_level="die",
        run_clf=clf_cfg["run_clf"],
        clf_output=clf_cfg["clf_output"],
        clf_filter=clf_cfg["clf_filter"],
        clf_optuna=False,                # OAT/Group은 default HP 고정
        run_fs=False,                    # baseline은 FS 안 함
        fs_optuna=False,
        reg_level=cfg["reg_level"],
        reg_optuna=False,                # OAT/Group은 default HP 고정
        zero_clip=False,                 # 사용자 지시로 후처리 비적용
    )


def _cfg_to_reg_fixed(cfg: dict) -> dict:
    """cfg['loss'] → reg_fixed = {'objective': loss}.

    LGBM objective:
      regression / poisson / tweedie / huber
      tweedie의 경우 default tweedie_variance_power=1.5 (LGBM 기본)
    """
    loss = cfg["loss"]
    out = {"objective": loss}
    if loss == "tweedie":
        out["tweedie_variance_power"] = 1.5
    return out


TWEEDIE_LOSSES = {"tweedie"}   # 02_reg_single/strategy.md §4 룰: tweedie 시 target_transform 자동 OFF


def _effective_target_transform(cfg: dict) -> str:
    """학습에 실제 적용될 target_transform.

    cfg['loss']가 tweedie 계열이면 자동 'none'으로 override (log1p/yeo-johnson 충돌 방지).
    poisson은 그대로 (LGBM poisson objective는 log1p 호환).
    """
    if cfg["loss"] in TWEEDIE_LOSSES:
        return "none"
    return cfg["TARGET_TRANSFORM"]


def _make_target_transformers(cfg: dict, ys: dict, target_col: str = "health"):
    """effective_target_transform에 따라 (target_transform_fn, target_inverse_fn, factory) 반환.

    yeo-johnson은 fold-train y에만 fit해야 leakage가 없으므로 pre-fit하지 않고
    factory(y_fold_train)를 반환. _run_reg_oof / _run_reg_single이 fold 안에서
    factory를 호출해 fold-local fwd/inv를 만들어 사용한다.

    Returns
    -------
    (fwd, inv, factory) :
      - fwd, inv : pre-fit 변환 함수 (stateless transform 전용). yeo-johnson은 None.
      - factory  : callable(y_array) -> (fwd_local, inv_local) 또는 None.
                   factory가 not None이면 fwd/inv는 무시된다.

    baseline.ipynb cell 7과 동일 로직 + tweedie 자동 OFF 분기.
    """
    tt = _effective_target_transform(cfg)
    if tt == "none":
        return None, None, None

    if tt == "log1p":
        return (
            lambda y: np.log1p(np.asarray(y)),
            lambda y: np.clip(np.expm1(np.asarray(y)), 0.0, None),
            None,
        )

    if tt == "yeo-johnson":
        from sklearn.preprocessing import PowerTransformer

        def factory(y_fit_array):
            """fold-train y에만 fit한 PowerTransformer로 (fwd, inv) 빌드. (leakage 차단)"""
            pt = PowerTransformer(method="yeo-johnson", standardize=False)
            y_arr = np.asarray(y_fit_array).reshape(-1, 1)
            pt.fit(y_arr)
            y_trans = pt.transform(y_arr).ravel()
            tt_min = float(y_trans.min())
            tt_max = float(y_trans.max())

            def fwd_local(y):
                return pt.transform(np.asarray(y).reshape(-1, 1)).ravel()

            def inv_local(y):
                y_clip = np.clip(np.asarray(y), tt_min, tt_max)
                out = pt.inverse_transform(y_clip.reshape(-1, 1)).ravel()
                out = np.clip(out, 0.0, None)
                return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

            return fwd_local, inv_local

        return None, None, factory

    raise ValueError(f"Unknown TARGET_TRANSFORM: {tt}")


def _prepare_ys_input(cfg: dict, ys: dict, target_col: str = "health") -> dict:
    """CLIP_Y_EXTREME=True면 train y의 max=1.0을 두 번째로 큰 값으로 clip."""
    ys_input = {k: v.copy() for k, v in ys.items()}
    if cfg["CLIP_Y_EXTREME"]:
        train_y = ys_input["train"][target_col]
        second_max = train_y[train_y < train_y.max()].max()
        ys_input["train"][target_col] = train_y.clip(upper=second_max)
    return ys_input


def _make_default_best_params(n_estimators: int = 100) -> dict:
    """LGBM default HP를 best_params 형식으로.

    rerun_best_trial_with_pp는 best_params에서 'reg_*', 'clf_*' 키를 추출해
    _build_params_from_best 가 모델 인자로 변환한다. clf_optuna=False / reg_optuna=False
    이면 default get_default_params가 사용되므로 best_params는 비워도 된다.
    n_estimators만 명시 가능 (sklearn default 100, baseline strategy §11 결정).
    """
    return {}


# ═════════════════════════════════════════════════════════════
# 6. 데이터 모듈 캐시 (노트북에서 set_data로 등록)
# ═════════════════════════════════════════════════════════════

_DATA = {}


def set_data(xs, xs_dict, ys, feat_cols):
    """노트북 시작 시 1회 호출. 이후 run_one은 _DATA에서 자동 로드."""
    _DATA["xs"] = xs
    _DATA["xs_dict"] = xs_dict
    _DATA["ys"] = ys
    _DATA["feat_cols"] = feat_cols


def get_data():
    if not _DATA:
        raise RuntimeError(
            "axes.set_data(xs, xs_dict, ys, feat_cols)를 먼저 호출하세요."
        )
    return _DATA


# ═════════════════════════════════════════════════════════════
# 7. 단일 entrypoint — run_one(cfg, seed)
# ═════════════════════════════════════════════════════════════

def run_one(cfg: dict, seed: int, n_jobs: int = 7,
            n_estimators: int = 100, target_col: str = "health") -> dict:
    """단일 cell 학습 + 5-fold OOF + val/test RMSE 반환.

    Parameters
    ----------
    cfg : dict
        AXES 키들을 포함한 설정 dict (REFERENCE 형식)
    seed : int
        모델 random_state. KFold seed는 strategy_common §6/§7 따라 42 고정 (override X)
    n_jobs : int
        LGBM n_jobs (strategy_common §8, default 7 = 14코어 2 노트북 병렬)
    n_estimators : int
        LGBM n_estimators (strategy.md §11 결정 default 100)

    Returns
    -------
    dict : {
        'val_rmse': float,   # 5-fold val(holdout) 평균 RMSE
        'oof_rmse': float,   # 5-fold OOF RMSE (train)
        'test_rmse': float,  # test RMSE
        'fold_rmses': list,  # fold별 OOF RMSE (5개)
        'elapsed_sec': float,
    }
    """
    data = get_data()
    xs, xs_dict, ys, feat_cols = (
        data["xs"], data["xs_dict"], data["ys"], data["feat_cols"]
    )

    t0 = time.time()

    # 1) PP params dict 구성
    pp_params = _cfg_to_pp_params(cfg)

    # 2) pipeline_config / reg_fixed
    pipeline_config = _cfg_to_pipeline_config(cfg)
    reg_fixed = _cfg_to_reg_fixed(cfg)
    clf_fixed = {}   # baseline은 clf default

    # 3) target transform / clip
    target_fwd, target_inv, target_factory = _make_target_transformers(cfg, ys, target_col)
    ys_input = _prepare_ys_input(cfg, ys, target_col)

    # 4) best_params (LGBM default — 빈 dict로 default get_default_params 활용)
    best_params = _make_default_best_params(n_estimators)
    if n_estimators != 100:
        # default 외 값을 박고 싶으면 reg_n_estimators 명시
        best_params["reg_n_estimators"] = n_estimators

    # 5) e2e 호출 — 5-fold KFold + val/test 예측 (best_params=defaults라 사실상 단일 fit)
    final = rerun_best_trial_with_pp(
        xs=xs,
        xs_dict=xs_dict,
        ys=ys_input,
        feat_cols=feat_cols,
        best_params=best_params,
        best_pp_params_resolved=pp_params,
        pipeline_config=pipeline_config,
        clf_model="lgbm",
        reg_model="lgbm",
        mode="kfold",
        n_folds=5,
        clf_early_stop=100,
        reg_early_stop=100,
        label_col="label_bin",
        imbalance_method="scale_pos_weight",
        top_k_fixed=200,                       # run_fs=False라 미사용
        clf_filter_threshold_fixed=0.5,
        zero_clip_threshold_fixed=0.0,
        clf_fixed=clf_fixed,
        reg_fixed=reg_fixed,
        target_transform_fn=target_fwd,
        target_inverse_fn=target_inv,
        target_transformer_factory=target_factory,
        exclude_cols=EXCLUDE_COLS,
    )

    # 6) RMSE 추출 — final dict는 'oof_rmse', 'val_rmse', 'test_rmse' 또는 *_pred 보유
    from utils.evaluate import rmse

    out = {
        "elapsed_sec": time.time() - t0,
        "oof_rmse":  None,
        "val_rmse":  None,
        "test_rmse": None,
        "fold_rmses": None,
        "effective_target_transform": _effective_target_transform(cfg),  # tweedie 시 'none' override 추적
    }

    # OOF
    if "oof_pred" in final:
        y_train = ys["train"][target_col].values
        out["oof_rmse"] = float(rmse(y_train, final["oof_pred"]))

    # val
    if "val_pred" in final:
        y_val = ys["validation"][target_col].values
        out["val_rmse"] = float(rmse(y_val, final["val_pred"]))

    # test
    if "test_pred" in final:
        y_test = ys["test"][target_col].values
        out["test_rmse"] = float(rmse(y_test, final["test_pred"]))

    return out
