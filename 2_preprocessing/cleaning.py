"""
데이터 클리닝 모듈
- 상수 feature 제거 (std=0)
- 극저분산 feature 제거 (std < threshold)
- 고결측 feature 제거 (결측률 >= threshold)
- 중복 컬럼 제거
- 고상관 feature 제거 (|r|>0.95 쌍에서 한쪽 제거)
- 결측치 imputation (train 기준 median) + 결측 indicator 옵션

EDA 결과 기반:
- 상수 feature: 97개
- 극저분산 (std<1e-6): 105개
- 고결측 (≥50%): 17개
- 고상관 쌍 (|r|>0.95): 47개
- 전체 1,087개 feature에 결측 존재 (대부분 0.23%)
- 결측률 1~50%: 약 11개 → indicator 컬럼 추가 고려
"""
import pandas as pd
import numpy as np


def remove_constant_features(xs, feat_cols, threshold=1e-6):
    """
    분산이 threshold 이하인 feature 제거

    Parameters
    ----------
    xs : DataFrame
    feat_cols : list
    threshold : float
        std가 이 값 이하이면 제거. 0이면 완전 상수만 제거

    Returns
    -------
    keep_cols : list (남은 feature 컬럼)
    removed_cols : list (제거된 feature 컬럼)
    """
    stds = xs[feat_cols].std()
    removed = stds[stds <= threshold].index.tolist()
    keep = [c for c in feat_cols if c not in removed]

    print(f"[상수/극저분산 제거] threshold={threshold}")
    print(f"  제거: {len(removed)}개, 잔여: {len(keep)}개")
    return keep, removed


def remove_high_missing_features(xs, feat_cols, threshold=0.5):
    """
    결측률이 threshold 이상인 feature 제거

    Parameters
    ----------
    xs : DataFrame
    feat_cols : list
    threshold : float
        결측 비율 (0~1). 기본 0.5 = 50%

    Returns
    -------
    keep_cols : list
    removed_cols : list
    """
    missing_pct = xs[feat_cols].isnull().mean()
    removed = missing_pct[missing_pct >= threshold].index.tolist()
    keep = [c for c in feat_cols if c not in removed]

    print(f"[고결측 제거] threshold={threshold*100:.0f}%")
    print(f"  제거: {len(removed)}개, 잔여: {len(keep)}개")
    return keep, removed


def remove_duplicate_features(xs, feat_cols, sample_n=5000, seed=42):
    """
    완전 중복 컬럼 쌍에서 하나씩 제거 (샘플 기반)

    Parameters
    ----------
    xs : DataFrame
    feat_cols : list
    sample_n : int
    seed : int

    Returns
    -------
    keep_cols : list
    removed_cols : list
    """
    sample = xs[feat_cols].sample(n=min(sample_n, len(xs)), random_state=seed)
    removed = set()

    for i in range(len(feat_cols)):
        if feat_cols[i] in removed:
            continue
        for j in range(i + 1, len(feat_cols)):
            if feat_cols[j] in removed:
                continue
            if sample[feat_cols[i]].equals(sample[feat_cols[j]]):
                removed.add(feat_cols[j])

    removed = list(removed)
    keep = [c for c in feat_cols if c not in removed]

    print(f"[중복 컬럼 제거] sample_n={sample_n}")
    print(f"  제거: {len(removed)}개, 잔여: {len(keep)}개")
    return keep, removed


def remove_high_corr_features(xs, feat_cols, threshold=0.95, sample_n=10000, seed=42):
    """
    피처 간 |상관계수| > threshold인 쌍에서 한쪽 제거 (다중공선성 감소)

    EDA Phase 11: |r|>0.95 쌍 47개 발견
    (X234↔X235↔X236↔X237, X254↔X256↔X257, X1↔X3 등)

    제거 기준: 쌍 중 target 상관이 낮은 쪽 제거 (target_corr 제공 시)
              미제공 시 뒤에 나오는 컬럼 제거

    Parameters
    ----------
    xs : DataFrame
    feat_cols : list
    threshold : float
        |r|이 이 값 초과인 쌍에서 한쪽 제거. 기본 0.95
    sample_n : int
        상관 계산 시 샘플 수 (속도)
    seed : int

    Returns
    -------
    keep_cols : list
    removed_cols : list
    """
    sample = xs[feat_cols].sample(n=min(sample_n, len(xs)), random_state=seed)
    corr_matrix = sample.corr().abs()

    # 상삼각 행렬만 사용
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    )

    removed = set()
    for col in feat_cols:
        if col in removed:
            continue
        # 이 컬럼과 threshold 초과 상관인 컬럼들
        high_corr = upper.index[upper[col] > threshold].tolist()
        for hc in high_corr:
            if hc not in removed:
                removed.add(hc)

    removed = list(removed)
    keep = [c for c in feat_cols if c not in removed]

    print(f"[고상관 제거] threshold={threshold}")
    print(f"  제거: {len(removed)}개, 잔여: {len(keep)}개")
    return keep, removed


def impute_missing(xs_train, xs_val, xs_test, feat_cols, add_indicator=False,
                   indicator_threshold=0.01, method="median", knn_neighbors=5):
    """
    결측치 imputation (data leakage 방지: fit on train, transform all)

    Parameters
    ----------
    xs_train, xs_val, xs_test : DataFrame
    feat_cols : list
    add_indicator : bool
        True면 결측률이 indicator_threshold 이상인 feature에 대해
        결측 여부 indicator 컬럼({feat}_missing) 추가
        EDA Phase 5: 결측률 1~50% feature 약 11개에 대해 정보 보존
    indicator_threshold : float
        이 비율 이상 결측인 feature에만 indicator 추가. 기본 0.01 (1%)
    method : str
        "median" — train median으로 채움 (빠름, 기본)
        "knn" — KNNImputer (논문 1-3 근거, 지역 구조 보존)
    knn_neighbors : int
        method="knn"일 때 이웃 수. 기본 5

    Returns
    -------
    xs_train, xs_val, xs_test : DataFrame (결측 채워진 복사본)
    imputer_info : dict (method별 정보)
    indicator_cols : list (추가된 indicator 컬럼명, add_indicator=False면 빈 리스트)
    """
    xs_train = xs_train.copy()
    xs_val = xs_val.copy()
    xs_test = xs_test.copy()

    # 결측 indicator 컬럼 추가 (imputation 전에 생성해야 함)
    indicator_cols = []
    if add_indicator:
        missing_pct = xs_train[feat_cols].isnull().mean()
        indicator_feats = missing_pct[missing_pct >= indicator_threshold].index.tolist()
        for col in indicator_feats:
            ind_col = f"{col}_missing"
            xs_train[ind_col] = xs_train[col].isnull().astype(int)
            xs_val[ind_col] = xs_val[col].isnull().astype(int)
            xs_test[ind_col] = xs_test[col].isnull().astype(int)
            indicator_cols.append(ind_col)
        if indicator_cols:
            print(f"[결측 indicator] {len(indicator_cols)}개 컬럼 추가 "
                  f"(결측률 >= {indicator_threshold*100:.0f}%)")

    imputer_info = {"method": method}

    if method == "median":
        medians = xs_train[feat_cols].median()
        xs_train[feat_cols] = xs_train[feat_cols].fillna(medians)
        xs_val[feat_cols] = xs_val[feat_cols].fillna(medians)
        xs_test[feat_cols] = xs_test[feat_cols].fillna(medians)
        imputer_info["medians"] = medians
        print(f"[결측 imputation] method=median")

    elif method == "knn":
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=knn_neighbors, weights="uniform")
        print(f"[결측 imputation] method=knn (n_neighbors={knn_neighbors})")
        print(f"  fitting on train ({len(xs_train):,} rows)...")
        xs_train[feat_cols] = imputer.fit_transform(xs_train[feat_cols])
        xs_val[feat_cols] = imputer.transform(xs_val[feat_cols])
        xs_test[feat_cols] = imputer.transform(xs_test[feat_cols])
        imputer_info["knn_neighbors"] = knn_neighbors

    else:
        raise ValueError(f"Unknown imputation method: {method}. Use 'median' or 'knn'")

    remaining = xs_train[feat_cols].isnull().sum().sum()
    print(f"  imputation 후 잔여 결측: {remaining}")
    return xs_train, xs_val, xs_test, imputer_info, indicator_cols


def run_cleaning(xs, feat_cols, xs_dict,
                 const_threshold=1e-6,
                 missing_threshold=0.5,
                 remove_duplicates=True,
                 corr_threshold=0.95,
                 add_indicator=False,
                 indicator_threshold=0.01,
                 imputation_method="median",
                 knn_neighbors=5):
    """
    클리닝 파이프라인 전체 실행

    Parameters
    ----------
    xs : DataFrame (전체)
    feat_cols : list
    xs_dict : dict (split별 DataFrame)
    const_threshold : float
    missing_threshold : float
    remove_duplicates : bool
    corr_threshold : float or None
        |r| > 이 값인 피처 쌍에서 한쪽 제거. None이면 스킵
    add_indicator : bool
        결측 indicator 컬럼 추가 여부
    indicator_threshold : float
        indicator 추가 기준 결측률
    imputation_method : str
        "median" 또는 "knn"
    knn_neighbors : int
        knn일 때 이웃 수

    Returns
    -------
    xs_train, xs_val, xs_test : DataFrame (클리닝 완료)
    clean_feat_cols : list (남은 feature 컬럼, indicator 컬럼 포함)
    report : dict (각 단계별 제거 내역)
    """
    print("=" * 60)
    print("클리닝 파이프라인 시작")
    print(f"원본 feature 수: {len(feat_cols)}")
    print("=" * 60)

    report = {}
    original_count = len(feat_cols)
    current_cols = feat_cols.copy()

    # split별 DataFrame 복사 (원본 보호)
    from utils.config import META_COLS
    xs_train = xs_dict["train"].copy()
    xs_val = xs_dict["validation"].copy()
    xs_test = xs_dict["test"].copy()

    def _drop_from_all(cols_to_drop):
        """3개 DataFrame에서 컬럼 일괄 drop"""
        nonlocal xs_train, xs_val, xs_test
        existing = [c for c in cols_to_drop if c in xs_train.columns]
        if existing:
            xs_train = xs_train.drop(columns=existing)
            xs_val = xs_val.drop(columns=existing)
            xs_test = xs_test.drop(columns=existing)

    # 1. 상수/극저분산 제거
    before = len(current_cols)
    current_cols, removed = remove_constant_features(xs_train, current_cols, const_threshold)
    _drop_from_all(removed)
    report["constant"] = removed
    print(f"    컬럼: {before} → {len(current_cols)} ({before - len(current_cols)}개 제거)")
    print(f"    DataFrame: {xs_train.shape}\n")

    # 2. 고결측 제거
    before = len(current_cols)
    current_cols, removed = remove_high_missing_features(xs_train, current_cols, missing_threshold)
    _drop_from_all(removed)
    report["high_missing"] = removed
    print(f"    컬럼: {before} → {len(current_cols)} ({before - len(current_cols)}개 제거)")
    print(f"    DataFrame: {xs_train.shape}\n")

    # 3. 중복 제거
    if remove_duplicates:
        before = len(current_cols)
        current_cols, removed = remove_duplicate_features(xs_train, current_cols)
        _drop_from_all(removed)
        report["duplicate"] = removed
        print(f"    컬럼: {before} → {len(current_cols)} ({before - len(current_cols)}개 제거)")
        print(f"    DataFrame: {xs_train.shape}\n")

    # 4. 고상관 피처 제거 (다중공선성 감소)
    if corr_threshold is not None:
        before = len(current_cols)
        current_cols, removed = remove_high_corr_features(
            xs_train, current_cols, threshold=corr_threshold
        )
        _drop_from_all(removed)
        report["high_corr"] = removed
        print(f"    컬럼: {before} → {len(current_cols)} ({before - len(current_cols)}개 제거)")
        print(f"    DataFrame: {xs_train.shape}\n")

    # 5. 결측 imputation + indicator
    xs_train, xs_val, xs_test, imputer_info, indicator_cols = impute_missing(
        xs_train, xs_val, xs_test, current_cols,
        add_indicator=add_indicator,
        indicator_threshold=indicator_threshold,
        method=imputation_method,
        knn_neighbors=knn_neighbors,
    )
    report["imputer_info"] = imputer_info
    report["indicator_cols"] = indicator_cols

    # indicator 컬럼도 feature 목록에 추가
    all_feat_cols = current_cols + indicator_cols

    print(f"\n{'=' * 60}")
    print(f"클리닝 완료: {original_count} → {len(current_cols)} features "
          f"({original_count - len(current_cols)}개 제거)")
    if indicator_cols:
        print(f"  + indicator 컬럼: {len(indicator_cols)}개 → 총 {len(all_feat_cols)}개")
    print(f"  train: {xs_train.shape}")
    print(f"  val:   {xs_val.shape}")
    print(f"  test:  {xs_test.shape}")
    print("=" * 60)

    return xs_train, xs_val, xs_test, all_feat_cols, report
