"""
스케일링 모듈 v2
- Robust 표준화 (median=0, IQR=1)
- Auto 스케일링 (왜도 기반 자동 선택: 왜도 높으면 log, 낮으면 robust)
- scale() 통합 함수 (transform 파라미터로 선택)
"""
import numpy as np
import pandas as pd


def _log_scale(xs, feat_cols, train_mask=None):
    ref = xs.loc[train_mask, feat_cols] if train_mask is not None else xs[feat_cols]
    mins = ref.min()
    xs[feat_cols] = np.log1p(xs[feat_cols] - mins)
    return xs, pd.DataFrame({'min_shift': -mins})


def robust_scale(xs, feat_cols, train_mask=None):
    """
    Robust 표준화 (median=0, IQR=1)

    Parameters
    ----------
    xs : DataFrame
    feat_cols : list
    train_mask : Series[bool], optional

    Returns
    -------
    xs : DataFrame (inplace 수정)
    stats : DataFrame (컬럼별 median, iqr)
    """
    ref = xs.loc[train_mask, feat_cols] if train_mask is not None else xs[feat_cols]
    medians = ref.median()
    iqr = ref.quantile(0.75) - ref.quantile(0.25)

    iqr_safe = iqr.clip(lower=1e-8)

    xs[feat_cols] = (xs[feat_cols] - medians) / iqr_safe

    stats = pd.DataFrame({'median': medians, 'iqr': iqr})
    zero_iqr = (iqr < 1e-8).sum()

    print(f"[Robust 스케일링] train 기준 = {train_mask is not None}")
    print(f"  대상 feature: {len(feat_cols)}개")
    if zero_iqr > 0:
        print(f"  IQR=0 컬럼 (스킵): {zero_iqr}개")

    return xs, stats


def auto_scale(xs, feat_cols, train_mask=None, skew_threshold=2.0):
    """
    왜도 기반 자동 스케일링

    |skew| > skew_threshold  → log (분포 비대칭이 심한 경우)
    |skew| <= skew_threshold → robust (비교적 대칭인 경우)

    Parameters
    ----------
    xs : DataFrame
    feat_cols : list
    train_mask : Series[bool], optional
    skew_threshold : float
        기본 2.0

    Returns
    -------
    xs : DataFrame (inplace 수정)
    stats : dict  {'log': DataFrame, 'robust': DataFrame}
    transform_map : dict  {feature명: 'log' or 'robust'}
    """
    ref = xs.loc[train_mask, feat_cols] if train_mask is not None else xs[feat_cols]
    skewness = ref.skew()

    log_cols = skewness[skewness.abs() > skew_threshold].index.tolist()
    robust_cols = skewness[skewness.abs() <= skew_threshold].index.tolist()

    stats = {}
    transform_map = {}

    if log_cols:
        xs, s = _log_scale(xs, log_cols, train_mask=train_mask)
        stats['log'] = s
    for col in log_cols:
        transform_map[col] = 'log'

    if robust_cols:
        xs, s = robust_scale(xs, robust_cols, train_mask=train_mask)
        stats['robust'] = s
    for col in robust_cols:
        transform_map[col] = 'robust'

    print(f"[Auto 스케일링] threshold={skew_threshold}")
    print(f"  log 적용: {len(log_cols)}개, robust 적용: {len(robust_cols)}개")

    return xs, stats, transform_map


def scale(xs, feat_cols, train_mask=None, transform='auto', skew_threshold=2.0):
    """
    통합 스케일링 함수

    Parameters
    ----------
    xs : DataFrame
    feat_cols : list
    train_mask : Series[bool], optional
    transform : str or None
        'robust'  — Robust 표준화 (median/IQR)
        'auto'    — 왜도 기반 자동 선택 (기본)
        None      — 스케일링 안 함
    skew_threshold : float
        transform='auto'일 때 log vs robust 분기 기준. 기본 2.0

    Returns
    -------
    xs : DataFrame (inplace 수정)
    stats : DataFrame or dict
    transform_map : dict  {feature명: 적용된 방법}
    """
    VALID = ('robust', 'auto', None)
    assert transform in VALID, f"transform must be one of {VALID}, got '{transform}'"

    if transform == 'robust':
        xs, stats = robust_scale(xs, feat_cols, train_mask)
        transform_map = {col: 'robust' for col in feat_cols}

    elif transform == 'auto':
        xs, stats, transform_map = auto_scale(
            xs, feat_cols, train_mask, skew_threshold=skew_threshold)

    elif transform is None:
        print("[스케일링] 스킵")
        stats = {}
        transform_map = {col: None for col in feat_cols}

    return xs, stats, transform_map
