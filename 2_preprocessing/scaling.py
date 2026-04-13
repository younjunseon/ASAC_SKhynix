"""
스케일링 모듈 v3
- Robust 표준화 (median=0, IQR=1)     → 트리 모델 / 시각화용
- Power 변환 (Yeo-Johnson 고정)       → 선형 모델용
- scale() 통합 함수 (transform 파라미터로 선택)

v2 대비 변경사항:
- zscore, minmax, log, auto 제거
- power_scale() 신규 추가 (sklearn PowerTransformer, Yeo-Johnson 고정)
- scale() 통합 함수 유지 (반환값 동일: xs, stats, transform_map)

사용법:
    from scale_v3 import scale

    # 트리 모델 / 시각화용
    xs, stats, tmap = scale(xs, feat_cols, train_mask, transform='robust')

    # 선형 모델용 (Yeo-Johnson)
    xs, stats, tmap = scale(xs, feat_cols, train_mask, transform='power')
"""
import pandas as pd
from sklearn.preprocessing import PowerTransformer


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


def power_scale(xs, feat_cols, train_mask=None):
    """
    Power 변환 스케일링 (선형 모델용, Yeo-Johnson 고정)

    sklearn PowerTransformer로 feature별 최적 lambda를 자동 탐색하여 변환.
    음수/0 포함 데이터 모두 처리 가능.
    변환 후 standardize=True로 mean=0, std=1 정규화까지 적용.

    Parameters
    ----------
    xs : DataFrame
    feat_cols : list
    train_mask : Series[bool], optional
        lambda를 train에서만 계산할 마스크.
        None이면 전체 데이터 기준으로 계산

    Returns
    -------
    xs : DataFrame (inplace 수정)
    stats : DataFrame (컬럼별 lambda)
    """
    ref = xs.loc[train_mask, feat_cols] if train_mask is not None else xs[feat_cols]

    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    pt.fit(ref)
    xs[feat_cols] = pt.transform(xs[feat_cols])

    stats = pd.DataFrame({'lambda': pt.lambdas_}, index=feat_cols)

    print(f"[Power 스케일링] method='yeo-johnson', train 기준 = {train_mask is not None}")
    print(f"  대상 feature: {len(feat_cols)}개")

    return xs, stats


def scale(xs, feat_cols, train_mask=None, transform='robust'):
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
        'robust'  — Robust 표준화 (median/IQR) [트리 모델 / 시각화용]
        'power'   — Power 변환 Yeo-Johnson [선형 모델용]
        None      — 스케일링 안 함

    Returns
    -------
    xs : DataFrame (inplace 수정)
    stats : DataFrame or dict
    transform_map : dict  {feature명: 적용된 방법}
    """
    VALID = ('robust', 'power', None)
    assert transform in VALID, f"transform must be one of {VALID}, got '{transform}'"

    if transform == 'robust':
        xs, stats = robust_scale(xs, feat_cols, train_mask)
        transform_map = {col: 'robust' for col in feat_cols}

    elif transform == 'power':
        xs, stats = power_scale(xs, feat_cols, train_mask)
        transform_map = {col: 'power' for col in feat_cols}

    elif transform is None:
        print("[스케일링] 스킵")
        stats = {}
        transform_map = {col: None for col in feat_cols}

    return xs, stats, transform_map
