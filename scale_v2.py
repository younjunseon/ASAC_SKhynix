"""
스케일링 모듈 v2
- Z-score 표준화 (mean=0, std=1)
- Min-Max 표준화 (0~1 범위)
- Log scale 표준화 (min-shift → log1p)
- Robust 표준화 (median=0, IQR=1)        ← NEW
- Auto 스케일링 (왜도 기반 자동 선택)    ← NEW
- scale() 통합 함수 (transform 파라미터로 선택) ← NEW

scale.py와 완전히 동일한 인터페이스 유지.
기존 3개 함수는 stds.replace(0,1) → clip(lower=1e-8) 버그 수정만 적용.

사용법:
    from scale_v2 import zscore_scale, minmax_scale, log_scale
    from scale_v2 import robust_scale, auto_scale
    from scale_v2 import scale   # 통합 함수
"""
import numpy as np
import pandas as pd


def zscore_scale(xs, feat_cols, train_mask=None):
    """
    Z-score 표준화 (mean=0, std=1)

    Parameters
    ----------
    xs : DataFrame
    feat_cols : list
    train_mask : Series[bool], optional
        mean/std를 train에서만 계산할 마스크.
        None이면 전체 데이터 기준으로 계산

    Returns
    -------
    xs : DataFrame (inplace 수정)
    stats : DataFrame (컬럼별 mean, std)
    """
    ref = xs.loc[train_mask, feat_cols] if train_mask is not None else xs[feat_cols]
    means = ref.mean()
    stds = ref.std()

    stds_safe = stds.clip(lower=1e-8)  # replace(0,1) 대신 clip 사용 (near-zero float 처리)

    xs[feat_cols] = (xs[feat_cols] - means) / stds_safe

    stats = pd.DataFrame({'mean': means, 'std': stds})
    zero_std = (stds < 1e-8).sum()

    print(f"[Z-score 스케일링] train 기준 = {train_mask is not None}")
    print(f"  대상 feature: {len(feat_cols)}개")
    if zero_std > 0:
        print(f"  std=0 컬럼 (스킵): {zero_std}개")

    return xs, stats


def minmax_scale(xs, feat_cols, train_mask=None):
    """
    Min-Max 표준화 (0~1 범위)

    Parameters
    ----------
    xs : DataFrame
    feat_cols : list
    train_mask : Series[bool], optional
        min/max를 train에서만 계산할 마스크.
        None이면 전체 데이터 기준으로 계산

    Returns
    -------
    xs : DataFrame (inplace 수정)
    stats : DataFrame (컬럼별 min, max)
    """
    ref = xs.loc[train_mask, feat_cols] if train_mask is not None else xs[feat_cols]
    mins = ref.min()
    maxs = ref.max()

    ranges = maxs - mins
    ranges_safe = ranges.clip(lower=1e-8)  # replace(0,1) 대신 clip 사용

    xs[feat_cols] = (xs[feat_cols] - mins) / ranges_safe

    stats = pd.DataFrame({'min': mins, 'max': maxs})
    zero_range = (ranges < 1e-8).sum()

    print(f"[Min-Max 스케일링] train 기준 = {train_mask is not None}")
    print(f"  대상 feature: {len(feat_cols)}개")
    if zero_range > 0:
        print(f"  range=0 컬럼 (스킵): {zero_range}개")

    return xs, stats


def log_scale(xs, feat_cols, train_mask=None):
    """
    Log scale 표준화 (min-shift → log1p)

    각 컬럼의 최솟값을 0으로 맞춘 뒤 log1p(x) 적용.
    음수가 포함된 feature도 안전하게 처리됨.

    Parameters
    ----------
    xs : DataFrame
    feat_cols : list
    train_mask : Series[bool], optional
        min을 train에서만 계산할 마스크.
        None이면 전체 데이터 기준으로 계산

    Returns
    -------
    xs : DataFrame (inplace 수정)
    stats : DataFrame (컬럼별 min_shift 값)
    """
    ref = xs.loc[train_mask, feat_cols] if train_mask is not None else xs[feat_cols]
    mins = ref.min()

    xs[feat_cols] = xs[feat_cols] - mins
    xs[feat_cols] = np.log1p(xs[feat_cols])

    stats = pd.DataFrame({'min_shift': -mins})
    neg_shifted = (mins < 0).sum()

    print(f"[Log 스케일링] train 기준 = {train_mask is not None}")
    print(f"  대상 feature: {len(feat_cols)}개")
    if neg_shifted > 0:
        print(f"  음수 → shift 적용: {neg_shifted}개")

    return xs, stats


def robust_scale(xs, feat_cols, train_mask=None):
    """
    Robust 표준화 (median=0, IQR=1)

    이상치에 강건한 스케일링.
    X_scaled = (X - median) / IQR

    Parameters
    ----------
    xs : DataFrame
    feat_cols : list
    train_mask : Series[bool], optional
        median/IQR을 train에서만 계산할 마스크.
        None이면 전체 데이터 기준으로 계산

    Returns
    -------
    xs : DataFrame (inplace 수정)
    stats : DataFrame (컬럼별 median, iqr)
    """
    ref = xs.loc[train_mask, feat_cols] if train_mask is not None else xs[feat_cols]
    medians = ref.median()
    iqr = ref.quantile(0.75) - ref.quantile(0.25)

    iqr_safe = iqr.clip(lower=1e-8)  # IQR=0인 상수 컬럼 안전처리

    xs[feat_cols] = (xs[feat_cols] - medians) / iqr_safe

    stats = pd.DataFrame({'median': medians, 'iqr': iqr})
    zero_iqr = (iqr < 1e-8).sum()

    print(f"[Robust 스케일링] train 기준 = {train_mask is not None}")
    print(f"  대상 feature: {len(feat_cols)}개")
    if zero_iqr > 0:
        print(f"  IQR=0 컬럼 (스킵): {zero_iqr}개")

    return xs, stats


def auto_scale(xs, feat_cols, train_mask=None,
               skew_threshold=2.0, low_skew='robust'):
    """
    왜도 기반 자동 스케일링

    feature별로 왜도를 계산하여 스케일링 방법을 자동 선택:
      |skew| > skew_threshold  → log_scale  (분포 비대칭이 심한 경우)
      |skew| <= skew_threshold → robust 또는 zscore  (비교적 대칭인 경우)

    Parameters
    ----------
    xs : DataFrame
    feat_cols : list
    train_mask : Series[bool], optional
        통계량을 train에서만 계산할 마스크.
        None이면 전체 데이터 기준으로 계산
    skew_threshold : float
        log vs low_skew 분기 기준 왜도 절대값. 기본 2.0
    low_skew : str
        왜도가 낮은 feature에 적용할 방법: 'robust' 또는 'zscore'. 기본 'robust'

    Returns
    -------
    xs : DataFrame (inplace 수정)
    stats : dict  {'log': DataFrame, low_skew: DataFrame}
        각 방법별 통계량
    transform_map : dict  {feature명: 'log' or 'robust' or 'zscore'}
        feature별 실제 적용된 스케일링 방법 기록
    """
    assert low_skew in ('robust', 'zscore'), \
        f"low_skew must be 'robust' or 'zscore', got '{low_skew}'"

    ref = xs.loc[train_mask, feat_cols] if train_mask is not None else xs[feat_cols]
    skewness = ref.skew()

    log_cols = skewness[skewness.abs() > skew_threshold].index.tolist()
    low_cols = skewness[skewness.abs() <= skew_threshold].index.tolist()

    stats = {}
    transform_map = {}

    # 왜도 높은 feature → log
    if log_cols:
        xs, s = log_scale(xs, log_cols, train_mask=train_mask)
        stats['log'] = s
    for col in log_cols:
        transform_map[col] = 'log'

    # 왜도 낮은 feature → robust or zscore
    if low_cols:
        if low_skew == 'robust':
            xs, s = robust_scale(xs, low_cols, train_mask=train_mask)
        else:
            xs, s = zscore_scale(xs, low_cols, train_mask=train_mask)
        stats[low_skew] = s
    for col in low_cols:
        transform_map[col] = low_skew

    print(f"[Auto 스케일링] threshold={skew_threshold}, low_skew='{low_skew}'")
    print(f"  log 적용: {len(log_cols)}개, {low_skew} 적용: {len(low_cols)}개")

    return xs, stats, transform_map


def scale(xs, feat_cols, train_mask=None,
          transform='auto', skew_threshold=2.0, auto_low_skew='robust'):
    """
    통합 스케일링 함수 — transform 파라미터로 방법 선택

    Parameters
    ----------
    xs : DataFrame
    feat_cols : list
    train_mask : Series[bool], optional
        통계량을 train에서만 계산할 마스크.
        None이면 전체 데이터 기준으로 계산
    transform : str or None
        'zscore'  — Z-score 표준화
        'minmax'  — Min-Max 스케일링 (0~1)
        'log'     — log1p 변환 (min-shift)
        'robust'  — Robust 표준화 (median/IQR)
        'auto'    — 왜도 기반 자동 선택 (기본)
        None      — 스케일링 안 함
    skew_threshold : float
        transform='auto'일 때 log vs auto_low_skew 분기 기준. 기본 2.0
    auto_low_skew : str
        transform='auto'일 때 왜도 낮은 feature에 적용: 'robust' | 'zscore'. 기본 'robust'

    Returns
    -------
    xs : DataFrame (inplace 수정)
    stats : DataFrame or dict
        각 방법별 통계량 (auto는 dict, 나머지는 DataFrame)
    transform_map : dict  {feature명: 적용된 방법}
    """
    VALID = ('zscore', 'minmax', 'log', 'robust', 'auto', None)
    assert transform in VALID, f"transform must be one of {VALID}, got '{transform}'"
    assert auto_low_skew in ('robust', 'zscore'), \
        f"auto_low_skew must be 'robust' or 'zscore', got '{auto_low_skew}'"

    if transform == 'zscore':
        xs, stats = zscore_scale(xs, feat_cols, train_mask)
        transform_map = {col: 'zscore' for col in feat_cols}

    elif transform == 'minmax':
        xs, stats = minmax_scale(xs, feat_cols, train_mask)
        transform_map = {col: 'minmax' for col in feat_cols}

    elif transform == 'log':
        xs, stats = log_scale(xs, feat_cols, train_mask)
        transform_map = {col: 'log' for col in feat_cols}

    elif transform == 'robust':
        xs, stats = robust_scale(xs, feat_cols, train_mask)
        transform_map = {col: 'robust' for col in feat_cols}

    elif transform == 'auto':
        xs, stats, transform_map = auto_scale(
            xs, feat_cols, train_mask,
            skew_threshold=skew_threshold, low_skew=auto_low_skew
        )

    elif transform is None:
        print("[스케일링] 스킵")
        stats = {}
        transform_map = {col: None for col in feat_cols}

    return xs, stats, transform_map
