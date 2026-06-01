"""
전처리 오케스트레이션.

고정 파이프라인: Stage 0(웨이퍼맵 수동 제외) → Cleaning → Outlier(현재 'none'=스킵).
- Cleaning 세부 파라미터(상수/결측/상관 임계값 등): DEFAULT_PARAMS + 노트북에서 넘긴 params로 override
- 일부 항목은 설계상 고정(_FIXED): 중복 제거 ON, imputation='spatial', outlier='none'(이상치 처리 안 함)
- run()이 cleaning+outlier를 한 번에 돌려 주고, 실제 적용된 파라미터 전체(effective_params)를 같이 반환해 재현성을 남긴다.

사용법
------
    from modules import preprocess

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

from cleaning import run_cleaning
from outlier import run_outlier_treatment


# 노트북에서 params dict로 덮어쓸 수 있는 기본값들
DEFAULT_PARAMS = {
    "const_threshold":            1e-6,          # std ≤ 이 값 → 상수/극저분산으로 제거
    "missing_threshold":          0.30,          # 결측률 ≥ 이 값 → 제거
    "corr_threshold":             0.90,          # 1차 고상관 제거: |r| > 이 값인 쌍에서 한쪽 제거
    # corr_keep_by: 기본 'std'. 'target_corr'는 KFold 바깥에서 train 전체 target으로 feature를 고르므로
    #               OOF 평가가 낙관적으로 편향될 수 있음 → 의도적으로 쓸 때만(opt-in) 허용.
    "corr_keep_by":               "std",         # 'target_corr' | 'std' — 고상관 쌍에서 남길 쪽 기준
    "corr_winsorize_pct":         0.0,           # corr_keep_by='std'일 때 std 계산 전 분위수 clip 비율
    "add_indicator":              True,          # 결측 indicator 컬럼 추가 여부
    "indicator_threshold":        0.05,          # indicator 생성 기준 결측률
    "spatial_max_dist":           6.0,           # spatial imputation에서 이웃으로 볼 최대 거리
    "post_impute_corr_threshold": 0.96,          # imputation 후 2차 고상관 제거 임계값
    "post_impute_corr_keep_by":   "std",         # 2차 고상관 제거에서 남길 쪽 기준
}

# 설계상 고정(노트북에서 못 바꿈) — 파이프라인 일관성 유지
_FIXED = {
    "remove_duplicates":  True,         # 완전 중복 컬럼은 항상 제거
    "imputation_method":  "spatial",    # 결측은 공간 보간 → lot 중앙값 → 전체 중앙값
    "outlier_method":     "none",       # X 피처 이상치 처리 안 함 (winsorize 전체 비활성화)
    "outlier_lower_pct":  0.0,          # (outlier_method='none'이라 미사용)
    "outlier_upper_pct":  0.99,         # (outlier_method='none'이라 미사용)
}

# 웨이퍼맵 육안 판정 결과 "target과 무관/유해"로 분류된 feature — cleaning 이전에 통째로 제외.
# (Colab에서 1_eda/ 폴더 미동기화 가능성 때문에 디렉토리 스캔이 아니라 리스트 하드코딩 — 재현성 우선)
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
    # X1086 (날짜값 — feature 부적합)
    "X1086",
    # 위치기반 추가 제외
    "X1056",   # Ring (타원 고리) 패턴 — 기여 모호
    "X1072",   # Radial gradient — X708과 r=0.997 (사실상 중복)
]


def _merge_params(params):
    """DEFAULT_PARAMS 위에 사용자 params를 덮어씀. 모르는 키는 에러, 값이 None인 키는 기본값 유지."""
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
    """전처리 실행: Stage 0 제외 → Cleaning → Outlier(현재 'none'=스킵).

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
    excl = EXCLUDE_COLS if exclude_cols is None else list(exclude_cols)   # None=기본 리스트, []=제외 안 함

    # --- Stage 0: 웨이퍼맵 수동 제외 (cleaning 자동 로직보다 먼저) ---
    pre_n = len(feat_cols)
    feat_cols_after_excl = [c for c in feat_cols if c not in excl]
    print(f"[Stage 0] 웨이퍼맵 사전 제외: "
          f"{pre_n} → {len(feat_cols_after_excl)} "
          f"({pre_n - len(feat_cols_after_excl)}개 제거)")

    # --- Cleaning: 상수/결측/중복/고상관 제거 + spatial imputation (+2차 고상관) ---
    xs_train, xs_val, xs_test, clean_feat_cols, report = run_cleaning(
        xs, feat_cols_after_excl, xs_dict,
        const_threshold=effective["const_threshold"],
        missing_threshold=effective["missing_threshold"],
        remove_duplicates=_FIXED["remove_duplicates"],
        corr_threshold=effective["corr_threshold"],
        corr_keep_by=effective["corr_keep_by"],
        corr_winsorize_pct=effective["corr_winsorize_pct"],
        ys_train=ys.get("train"),                       # corr_keep_by='target_corr'일 때만 실제로 쓰임
        add_indicator=effective["add_indicator"],
        indicator_threshold=effective["indicator_threshold"],
        imputation_method=_FIXED["imputation_method"],
        spatial_max_dist=effective["spatial_max_dist"],
        post_impute_corr_threshold=effective["post_impute_corr_threshold"],
        post_impute_corr_keep_by=effective["post_impute_corr_keep_by"],
    )

    # --- Outlier: method='none' 고정 — X 피처 이상치 처리 안 함 (winsorize 비활성화) ---
    xs_train, xs_val, xs_test, outlier_report = run_outlier_treatment(
        xs_train, xs_val, xs_test, clean_feat_cols,
        method=_FIXED["outlier_method"],
        lower_pct=_FIXED["outlier_lower_pct"],
        upper_pct=_FIXED["outlier_upper_pct"],
    )
    report["outlier"] = outlier_report

    # 실제 적용된 파라미터(override 결과 + 고정값 + 제외 개수)를 한 dict로 — 재현성 로깅용
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
