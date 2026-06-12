"""
전처리 오케스트레이션.

고정 파이프라인: Cleaning → Outlier(현재 'none'=스킵).
- Cleaning 세부 파라미터(상수/결측/상관 임계값 등): DEFAULT_PARAMS + 노트북에서 넘긴 params로 override
- 일부 항목은 설계상 고정(_FIXED): imputation='spatial', outlier='none'(이상치 처리 안 함)
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
    # 아래 두 개는 no-op (2차 고상관 제거 로직 삭제됨). 호환을 위해 키만 유지 — 외부에서 넘겨도 무시됨.
    "post_impute_corr_threshold": 0.96,
    "post_impute_corr_keep_by":   "std",
}

# 설계상 고정(노트북에서 못 바꿈) — 파이프라인 일관성 유지
_FIXED = {
    "imputation_method":  "spatial",    # 결측은 공간 보간 → lot+xy 중앙값 → train xy 중앙값
    "outlier_method":     "none",       # X 피처 이상치 처리 안 함 (winsorize 전체 비활성화)
    "outlier_lower_pct":  0.0,          # (outlier_method='none'이라 미사용)
    "outlier_upper_pct":  0.99,         # (outlier_method='none'이라 미사용)
}

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


def run(xs, ys, feat_cols, xs_dict, params=None):
    """전처리 실행: Cleaning → Outlier(현재 'none'=스킵).

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

    Returns
    -------
    dict
        xs_train, xs_val, xs_test : cleaned DataFrame
        feat_cols : list (cleaning 이후 남은 feature + indicator)
        effective_params : dict (실제 적용된 전처리 파라미터 전체)
        report : dict (cleaning 내부 단계별 제거 내역)
    """
    effective = _merge_params(params)

    # --- Cleaning: 상수/결측/고상관(풀스캔, 1차만) 제거 + spatial imputation ---
    xs_train, xs_val, xs_test, clean_feat_cols, report = run_cleaning(
        xs, feat_cols, xs_dict,
        const_threshold=effective["const_threshold"],
        missing_threshold=effective["missing_threshold"],
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

    # 실제 적용된 파라미터(override 결과 + 고정값)를 한 dict로 — 재현성 로깅용
    effective_full = {**effective, **{f"_fixed_{k}": v for k, v in _FIXED.items()}}

    return {
        "xs_train":         xs_train,
        "xs_val":           xs_val,
        "xs_test":          xs_test,
        "feat_cols":        clean_feat_cols,
        "effective_params": effective_full,
        "report":           report,
    }
