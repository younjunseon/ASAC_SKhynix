"""
스케일링 모듈 v4
- Robust 표준화 (median=0, IQR=1)     → 트리 모델 / 시각화용
- Power 변환 (Yeo-Johnson 고정)       → 선형 모델용
- HybridScaler (2차 funnel, sklearn fit/transform 패턴)
  → |skew|>threshold = quantile, 나머지 = power (Yeo-Johnson)
- scale() 통합 함수 (transform 파라미터로 선택)

v3 대비 변경사항 (2026-04-16, strategy_2nd_preprocessing.md §3):
- HybridScaler 클래스 신규 (sklearn 표준 fit/transform 패턴)
- hybrid_scale() 편의 함수 신규 (train fit → train/val/test transform)

사용법:
    # 트리 모델 / 시각화용
    xs, stats, tmap = scale(xs, feat_cols, train_mask, transform='robust')

    # 선형 모델용 (Yeo-Johnson)
    xs, stats, tmap = scale(xs, feat_cols, train_mask, transform='power')

    # 2차 funnel (앙상블 전용, train fit → 모든 split transform)
    from scaling import hybrid_scale
    xs_train, xs_val, xs_test, scaler = hybrid_scale(
        xs_train, xs_val, xs_test, feat_cols, skew_threshold=10.0)
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, QuantileTransformer


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
    transformed = pt.transform(xs[feat_cols])
    # PowerTransformer는 내부에서 float64로 승격 → 입력 dtype으로 복구해
    # float32 다운캐스트된 파이프라인의 dtype 일관성 유지
    in_dtype = xs[feat_cols].dtypes.iloc[0]
    if in_dtype == np.float32:
        transformed = transformed.astype('float32', copy=False)
    xs[feat_cols] = transformed

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


# ============================================================
# 2차 funnel — HybridScaler (sklearn fit/transform 패턴)
# ============================================================

class HybridScaler:
    """
    Skew 기반 하이브리드 스케일러 (2차 funnel 앙상블 전용)

    3-그룹 분기 (2026-04-16 업데이트):
      1) Binary passthrough   : nunique ≤ 2 (변환 없음, 선형 모델 친화)
      2) Quantile 변환         : |skew| > threshold (rank 기반, heavy-tail 평탄화)
      3) Power(Yeo-Johnson)    : 나머지 (standardize=True, mean=0/std=1)

    Binary passthrough 그룹 추가 근거:
      - Sparse binary (top% > 95%) feature를 standardize하면 범위 ~10까지 늘어
        ElasticNet L1/L2 penalty가 sparse 신호를 과도 억제함
      - 0/1 원본 유지가 선형 모델 해석/수렴에 유리
      - 트리 모델에도 무영향 (분기 1개로 처리)

    sklearn 표준 fit/transform 패턴 — 객체 보관 후 대시보드/예측 파이프라인 재사용 가능.
    """

    def __init__(self, skew_threshold=10.0, n_quantiles=1000,
                 quantile_output='normal', random_state=42,
                 binary_passthrough=True):
        self.skew_threshold = skew_threshold
        self.n_quantiles = n_quantiles
        self.quantile_output = quantile_output
        self.random_state = random_state
        self.binary_passthrough = binary_passthrough

    def fit(self, X, feat_cols=None):
        """
        Parameters
        ----------
        X : DataFrame (train only)
        feat_cols : list, optional
            스케일링 대상. None이면 X의 모든 컬럼

        Returns
        -------
        self
        """
        if feat_cols is None:
            feat_cols = list(X.columns)
        self.feat_cols_ = list(feat_cols)

        # 1) Binary 분리 (nunique ≤ 2) — binary_passthrough=True일 때만
        if self.binary_passthrough:
            nuniq = X[self.feat_cols_].nunique()
            self.binary_cols_ = nuniq[nuniq <= 2].index.tolist()
        else:
            self.binary_cols_ = []
        remaining = [c for c in self.feat_cols_ if c not in set(self.binary_cols_)]

        # 2) remaining에서 Skew 기준 두 그룹 분기 (train 기준)
        if remaining:
            skew_vals = X[remaining].skew().abs()
        else:
            skew_vals = pd.Series(dtype=float)
        self.skew_vals_ = skew_vals
        self.quantile_cols_ = skew_vals[skew_vals > self.skew_threshold].index.tolist()
        self.power_cols_ = [c for c in remaining if c not in set(self.quantile_cols_)]

        # 3) 각 그룹에 대해 train fit
        self.qt_ = None
        if self.quantile_cols_:
            n_q = min(self.n_quantiles, len(X))
            self.qt_ = QuantileTransformer(
                n_quantiles=n_q,
                output_distribution=self.quantile_output,
                subsample=int(1e6),
                random_state=self.random_state,
            )
            self.qt_.fit(X[self.quantile_cols_])

        self.pt_ = None
        if self.power_cols_:
            self.pt_ = PowerTransformer(method='yeo-johnson', standardize=True)
            self.pt_.fit(X[self.power_cols_])

        print(f"[HybridScaler.fit] skew_threshold={self.skew_threshold}")
        if self.binary_passthrough:
            print(f"  Binary passthrough: {len(self.binary_cols_)}개 (nunique ≤ 2, 변환 없음)")
        print(f"  Quantile 적용     : {len(self.quantile_cols_)}개 (|skew| > {self.skew_threshold})")
        print(f"  Power 적용        : {len(self.power_cols_)}개 (Yeo-Johnson + standardize)")
        return self

    def transform(self, X, inplace=True):
        """
        Parameters
        ----------
        X : DataFrame (train/val/test 중 하나)
        inplace : bool, default True
            True면 X를 직접 수정(동일 객체 반환), False면 복사본 반환

        Returns
        -------
        X_transformed : DataFrame

        Notes
        -----
        binary_cols_는 건드리지 않음 (passthrough). quantile/power 그룹만 변환.
        """
        if not inplace:
            X = X.copy()

        # Quantile 그룹
        if self.quantile_cols_ and self.qt_ is not None:
            arr = self.qt_.transform(X[self.quantile_cols_])
            in_dtype = X[self.quantile_cols_].dtypes.iloc[0]
            if in_dtype == np.float32:
                arr = arr.astype('float32', copy=False)
            X[self.quantile_cols_] = arr

        # Power 그룹
        if self.power_cols_ and self.pt_ is not None:
            arr = self.pt_.transform(X[self.power_cols_])
            in_dtype = X[self.power_cols_].dtypes.iloc[0]
            if in_dtype == np.float32:
                arr = arr.astype('float32', copy=False)
            X[self.power_cols_] = arr

        # Binary 그룹은 건드리지 않음 (passthrough)
        return X

    @property
    def transform_map_(self):
        """컬럼별 적용된 변환 종류 {feature: 'binary'|'quantile'|'power'}"""
        return {
            **{c: 'binary'   for c in self.binary_cols_},
            **{c: 'quantile' for c in self.quantile_cols_},
            **{c: 'power'    for c in self.power_cols_},
        }


def hybrid_scale(xs_train, xs_val, xs_test, feat_cols, skew_threshold=10.0,
                 n_quantiles=1000, quantile_output='normal', random_state=42):
    """
    Hybrid 스케일링 편의 함수 — fit-on-train + transform-all.

    _run_preprocessing에서 이 함수 1번 호출하면 3개 split이 모두 스케일링되고,
    scaler 객체까지 함께 반환되어 대시보드/예측 파이프라인에서 pickle 재사용 가능.

    Parameters
    ----------
    xs_train, xs_val, xs_test : DataFrame
    feat_cols : list
    skew_threshold : float, default 10.0
        |skew| > threshold → Quantile, 나머지 → Power(Yeo-Johnson)
    n_quantiles : int, default 1000
    quantile_output : {'normal', 'uniform'}, default 'normal'
    random_state : int, default 42

    Returns
    -------
    xs_train, xs_val, xs_test : DataFrame (in-place 수정)
    scaler : HybridScaler (fitted, joblib/pickle 저장 가능)
    """
    scaler = HybridScaler(
        skew_threshold=skew_threshold,
        n_quantiles=n_quantiles,
        quantile_output=quantile_output,
        random_state=random_state,
    ).fit(xs_train, feat_cols)

    scaler.transform(xs_train, inplace=True)
    scaler.transform(xs_val,   inplace=True)
    scaler.transform(xs_test,  inplace=True)

    return xs_train, xs_val, xs_test, scaler
