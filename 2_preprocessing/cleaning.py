"""
데이터 클리닝 모듈
- 상수 feature 제거 (std=0)
- 극저분산 feature 제거 (std < threshold)
- 고결측 feature 제거 (결측률 >= threshold)
- 중복 컬럼 제거
- 결측치 imputation (train 기준 median)

EDA 결과 기반:
- 상수 feature: 98개
- 극저분산 (std<1e-6): 105개
- 고결측 (≥50%): 17개
- 전체 1,087개 feature에 결측 존재 (대부분 0.23%)
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


def impute_missing(xs_train, xs_val, xs_test, feat_cols):
    """
    결측치를 train의 median으로 imputation (data leakage 방지)

    Parameters
    ----------
    xs_train, xs_val, xs_test : DataFrame
    feat_cols : list

    Returns
    -------
    xs_train, xs_val, xs_test : DataFrame (결측 채워진 복사본)
    medians : Series (train 기준 median 값)
    """
    medians = xs_train[feat_cols].median()

    xs_train = xs_train.copy()
    xs_val = xs_val.copy()
    xs_test = xs_test.copy()

    xs_train[feat_cols] = xs_train[feat_cols].fillna(medians)
    xs_val[feat_cols] = xs_val[feat_cols].fillna(medians)
    xs_test[feat_cols] = xs_test[feat_cols].fillna(medians)

    remaining = xs_train[feat_cols].isnull().sum().sum()
    print(f"[결측 imputation] train median 기준")
    print(f"  imputation 후 잔여 결측: {remaining}")
    return xs_train, xs_val, xs_test, medians


def run_cleaning(xs, feat_cols, xs_dict,
                 const_threshold=1e-6,
                 missing_threshold=0.5,
                 remove_duplicates=True):
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

    Returns
    -------
    xs_train, xs_val, xs_test : DataFrame (클리닝 완료)
    clean_feat_cols : list (남은 feature 컬럼)
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

    # 4. 결측 imputation
    xs_train, xs_val, xs_test, medians = impute_missing(
        xs_train, xs_val, xs_test, current_cols
    )
    report["medians"] = medians

    print(f"\n{'=' * 60}")
    print(f"클리닝 완료: {original_count} → {len(current_cols)} features ({original_count - len(current_cols)}개 제거)")
    print(f"  train: {xs_train.shape}")
    print(f"  val:   {xs_val.shape}")
    print(f"  test:  {xs_test.shape}")
    print("=" * 60)

    return xs_train, xs_val, xs_test, current_cols, report
