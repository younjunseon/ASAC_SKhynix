"""
Final 파이프라인 — 전처리 오케스트레이션

고정 파이프라인: Stage 0(웨이퍼맵 제외) → Cleaning → Outlier winsorize
- Cleaning 파라미터: DEFAULT_PARAMS + 사용자 override
- Outlier: winsorize(lower=0.0, upper=0.99) 하드코딩 (변경 불가)
- Imputation: 'spatial' 하드코딩 (변경 불가)

사용법
------
    from final.modules import preprocess

    result = preprocess.run(
        xs, ys, feat_cols, xs_dict,
        params={'corr_threshold': 0.98},   # 원하는 것만 override
    )
    xs_train = result['xs_train']
    xs_val   = result['xs_val']
    xs_test  = result['xs_test']
    clean_feat_cols = result['feat_cols']
    effective = result['effective_params']   # 재현성 로깅용
"""
import copy

from .cleaning import run_cleaning
from .outlier import run_outlier_treatment


# ─── 수정 가능한 기본값 (노트북에서 params dict로 override) ─────
DEFAULT_PARAMS = {
    "const_threshold":            1e-6,          # std ≤ 이 값 → 제거
    "missing_threshold":          0.5,           # 결측률 ≥ 이 값 → 제거
    "corr_threshold":             0.94,          # 1차 |r| > 이 값 → 제거
    # corr_keep_by: 'std' 기본. 'target_corr' 는 KFold 바깥에서 전체 train
    # target 으로 feature 를 고르기 때문에 OOF 평가가 낙관적으로 편향될 수
    # 있어 옵트인으로만 허용.
    "corr_keep_by":               "std",         # 'target_corr' | 'std'
    "corr_winsorize_pct":         0.0,           # std 계산 전 분위수 clip
    "add_indicator":              False,         # 결측 indicator 컬럼 추가
    "indicator_threshold":        0.05,          # indicator 생성 결측률 기준
    "spatial_max_dist":           2.0,           # spatial 보간 거리
    "post_impute_corr_threshold": 0.98,          # 2차 |r| 제거 임계값
    "post_impute_corr_keep_by":   "std",         # 2차 동률 기준
}

# ─── 고정값 (변경 불가, 설계 상 확정) ──────────────────────────
_FIXED = {
    "remove_duplicates":  True,
    "imputation_method":  "spatial",
    "outlier_method":     "winsorize",
    "outlier_lower_pct":  0.0,
    "outlier_upper_pct":  0.99,
}

# ─── 웨이퍼맵 사전 제외 리스트 (수동 분류) ──────────────────────
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
]


def _merge_params(params):
    """DEFAULT_PARAMS + 사용자 params 병합. None/누락 키는 기본값 사용."""
    effective = copy.deepcopy(DEFAULT_PARAMS)
    if params:
        for k, v in params.items():
            if k not in DEFAULT_PARAMS:
                raise KeyError(
                    f"Unknown param {k!r}. Allowed: {list(DEFAULT_PARAMS)}"
                )
            if v is not None:
                effective[k] = v
    return effective


def run(xs, ys, feat_cols, xs_dict, params=None, exclude_cols=None):
    """전처리 실행: Stage 0 제외 → Cleaning → Outlier winsorize.

    Parameters
    ----------
    xs : DataFrame
        원본 전체 xs (split 컬럼 포함).
    ys : dict
        {'train': df, 'validation': df, 'test': df}
    feat_cols : list[str]
        전처리 대상 feature 컬럼 (일반적으로 X0~X1086).
    xs_dict : dict
        {'train': df, 'validation': df, 'test': df}
    params : dict, optional
        DEFAULT_PARAMS 중 override할 값. None이면 전부 기본값.
    exclude_cols : list[str], optional
        None이면 모듈 상수 EXCLUDE_COLS 사용. [] 를 주면 사전 제외 스킵.

    Returns
    -------
    dict
        xs_train, xs_val, xs_test : cleaned DataFrame
        feat_cols : list (cleaning 이후 남은 feature + indicator)
        effective_params : dict (실제 적용된 전처리 파라미터 전체)
        report : dict (cleaning 내부 단계별 제거 내역)
    """
    effective = _merge_params(params)
    excl = EXCLUDE_COLS if exclude_cols is None else list(exclude_cols)

    # ── Stage 0: 웨이퍼맵 수동 제외 ──
    pre_n = len(feat_cols)
    feat_cols_after_excl = [c for c in feat_cols if c not in excl]
    print(f"[Stage 0] 웨이퍼맵 사전 제외: "
          f"{pre_n} → {len(feat_cols_after_excl)} "
          f"({pre_n - len(feat_cols_after_excl)}개 제거)")

    # ── Cleaning ──
    xs_train, xs_val, xs_test, clean_feat_cols, report = run_cleaning(
        xs, feat_cols_after_excl, xs_dict,
        const_threshold=effective["const_threshold"],
        missing_threshold=effective["missing_threshold"],
        remove_duplicates=_FIXED["remove_duplicates"],
        corr_threshold=effective["corr_threshold"],
        corr_keep_by=effective["corr_keep_by"],
        corr_winsorize_pct=effective["corr_winsorize_pct"],
        ys_train=ys.get("train"),
        add_indicator=effective["add_indicator"],
        indicator_threshold=effective["indicator_threshold"],
        imputation_method=_FIXED["imputation_method"],
        spatial_max_dist=effective["spatial_max_dist"],
        post_impute_corr_threshold=effective["post_impute_corr_threshold"],
        post_impute_corr_keep_by=effective["post_impute_corr_keep_by"],
    )

    # ── Outlier: winsorize(0.0, 0.99) 고정 ──
    xs_train, xs_val, xs_test, outlier_report = run_outlier_treatment(
        xs_train, xs_val, xs_test, clean_feat_cols,
        method=_FIXED["outlier_method"],
        lower_pct=_FIXED["outlier_lower_pct"],
        upper_pct=_FIXED["outlier_upper_pct"],
    )
    report["outlier"] = outlier_report

    # effective에 고정값도 같이 기록 (재현성)
    effective_full = {**effective, **{f"_fixed_{k}": v for k, v in _FIXED.items()}}
    effective_full["_exclude_cols_n"] = pre_n - len(feat_cols_after_excl)

    return {
        "xs_train":         xs_train,
        "xs_val":           xs_val,
        "xs_test":          xs_test,
        "feat_cols":        clean_feat_cols,
        "effective_params": effective_full,
        "report":           report,
    }