"""
스케일링 모듈.

WT feature는 스케일이 천차만별이고 skew가 큰 컬럼도 많다. 트리 모델은 스케일에 둔감하지만
선형 모델·일부 딥러닝은 민감하므로 용도에 따라 다른 변환을 쓴다:
  - robust_scale : (X - median) / IQR — 이상치에 강건. 트리/시각화용 (혹은 그냥 통일용)
  - power_scale  : Yeo-Johnson PowerTransformer + standardize — 선형 모델용 (skew 보정 + 표준화)
  - scale()      : 위 둘을 transform='robust'|'power'|None 로 골라 쓰는 통합 함수
  - HybridScaler / hybrid_scale : skew 크기에 따라 컬럼별로 다른 변환을 섞어 적용 (앙상블/선형용)
      · nunique ≤ 2  → 그대로 둠 (binary passthrough)
      · |skew| > threshold → QuantileTransformer (rank 기반, heavy-tail 평탄화)
      · 나머지       → Yeo-Johnson + standardize

공통: 통계량(median/IQR/lambda/quantile)은 train에서만 fit하고 val/test엔 transform만 적용 (leakage 방지).

사용법:
    # 트리 모델 / 시각화용
    xs, stats, tmap = scale(xs, feat_cols, train_mask, transform='robust')

    # 선형 모델용 (Yeo-Johnson)
    xs, stats, tmap = scale(xs, feat_cols, train_mask, transform='power')

    # 앙상블/선형용 하이브리드 (train fit → 모든 split transform)
    from scaling import hybrid_scale
    xs_train, xs_val, xs_test, scaler = hybrid_scale(
        xs_train, xs_val, xs_test, feat_cols, skew_threshold=10.0)
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
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
    # 통계량 계산 대상: train_mask가 있으면 train 행만, 없으면 전체
    ref = xs.loc[train_mask, feat_cols] if train_mask is not None else xs[feat_cols]
    medians = ref.median()
    iqr = ref.quantile(0.75) - ref.quantile(0.25)

    iqr_safe = iqr.clip(lower=1e-8)  # IQR=0인 (사실상 상수) 컬럼에서 0 나눗셈 방지

    xs[feat_cols] = (xs[feat_cols] - medians) / iqr_safe   # (X - median) / IQR

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
    # lambda(=변환 강도)는 train에서만 추정 → 전체에 transform
    ref = xs.loc[train_mask, feat_cols] if train_mask is not None else xs[feat_cols]

    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    pt.fit(ref)
    transformed = pt.transform(xs[feat_cols])
    # PowerTransformer는 결과를 float64로 내놓음 → 입력이 float32였다면 다시 float32로 (파이프라인 dtype 일관성)
    in_dtype = xs[feat_cols].dtypes.iloc[0]
    if in_dtype == np.float32:
        transformed = transformed.astype('float32', copy=False)
    xs[feat_cols] = transformed

    stats = pd.DataFrame({'lambda': pt.lambdas_}, index=feat_cols)   # 컬럼별 추정 lambda

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


# ------------------------------------------------------------
# HybridScaler — skew 크기에 따라 컬럼별로 다른 변환 (sklearn fit/transform 패턴)
# ------------------------------------------------------------

class HybridScaler(BaseEstimator, TransformerMixin):
    """
    Skew 기반 하이브리드 스케일러.

    feature를 3그룹으로 나눠 다르게 처리:
      1) Binary passthrough   : nunique ≤ 2 → 변환 없음 (0/1 원본 유지)
      2) Quantile 변환         : |skew| > threshold → QuantileTransformer (rank 기반, heavy-tail 평탄화)
      3) Power(Yeo-Johnson)    : 나머지 → PowerTransformer(standardize=True) (mean=0/std=1)

    Binary를 그대로 두는 이유:
      - sparse한 binary feature를 standardize하면 값 범위가 ~10까지 늘어,
        ElasticNet의 L1/L2 penalty가 그 sparse 신호를 과도하게 억제함
      - 0/1 원본이 선형 모델의 해석·수렴에 유리하고, 트리 모델엔 영향 없음

    sklearn 표준 fit/transform 패턴(BaseEstimator + TransformerMixin) → fit_transform 자동 제공.
    fitted 객체를 pickle로 저장해 대시보드/예측 파이프라인에서 재사용 가능.

    입력 타입:
      - DataFrame  → DataFrame 반환 (inplace 옵션 유효)
      - numpy 2D   → numpy 반환 (sklearn CV 파이프라인 호환)
    """

    def __init__(self, skew_threshold=10.0, n_quantiles=1000,
                 quantile_output='normal', random_state=42,
                 binary_passthrough=True):
        # 하이퍼파라미터만 보관 (sklearn 규약: __init__에선 검증/연산 안 함)
        self.skew_threshold = skew_threshold
        self.n_quantiles = n_quantiles
        self.quantile_output = quantile_output
        self.random_state = random_state
        self.binary_passthrough = binary_passthrough

    def fit(self, X, y=None, feat_cols=None):
        """
        Parameters
        ----------
        X : DataFrame or ndarray (train only)
            DataFrame이면 컬럼명 사용, numpy면 자동으로 임시 컬럼명 부여
        y : ignored
            sklearn 호환용(fit_transform이 y=None을 forward). 사용하지 않음
        feat_cols : list, optional
            스케일링 대상. None이면 X의 모든 컬럼.
            DataFrame 입력 시 부분 컬럼만 fit하려면 명시.

        Returns
        -------
        self
        """
        # numpy 입력은 임시 DataFrame으로 승격해서 이후 처리를 컬럼명 기반으로 통일
        if isinstance(X, np.ndarray):
            if X.ndim != 2:
                raise ValueError(f"HybridScaler: numpy 입력은 2D여야 함 (got ndim={X.ndim})")
            if feat_cols is None:
                feat_cols = [f'_col_{i}' for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=list(feat_cols))
        elif feat_cols is None:
            feat_cols = list(X.columns)
        self.feat_cols_ = list(feat_cols)

        # (1) Binary 그룹 분리 — nunique ≤ 2. binary_passthrough=False면 비활성
        if self.binary_passthrough:
            nuniq = X[self.feat_cols_].nunique()
            self.binary_cols_ = nuniq[nuniq <= 2].index.tolist()
        else:
            self.binary_cols_ = []
        remaining = [c for c in self.feat_cols_ if c not in set(self.binary_cols_)]

        # (2) 나머지를 train의 |skew| 기준으로 Quantile / Power 그룹으로 나눔
        if remaining:
            skew_vals = X[remaining].skew().abs()
        else:
            skew_vals = pd.Series(dtype=float)
        self.skew_vals_ = skew_vals
        self.quantile_cols_ = skew_vals[skew_vals > self.skew_threshold].index.tolist()
        self.power_cols_ = [c for c in remaining if c not in set(self.quantile_cols_)]

        # (3) 각 그룹의 변환기를 train에서 fit (binary는 변환기 없음)
        self.qt_ = None
        if self.quantile_cols_:
            n_q = min(self.n_quantiles, len(X))   # 샘플 수보다 많은 quantile은 불가
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
        X : DataFrame or ndarray (train/val/test 중 하나)
            numpy 입력 시 fit과 동일한 컬럼 수여야 함
        inplace : bool, default True
            DataFrame 입력에만 의미. True면 X 직접 수정, False면 복사본 반환.
            numpy 입력은 항상 새 ndarray 반환

        Returns
        -------
        X_transformed : DataFrame (입력이 DataFrame이면) / ndarray (입력이 numpy면)

        Notes
        -----
        binary_cols_는 건드리지 않음 (passthrough). quantile/power 그룹만 변환.
        """
        # numpy 입력: 임시 DataFrame으로 변환 → 처리 → 다시 numpy로 반환
        is_numpy = isinstance(X, np.ndarray)
        in_dtype_np = X.dtype if is_numpy else None
        if is_numpy:
            if X.ndim != 2:
                raise ValueError(f"HybridScaler: numpy 입력은 2D여야 함 (got ndim={X.ndim})")
            if X.shape[1] != len(self.feat_cols_):
                raise ValueError(
                    f"HybridScaler: 입력 컬럼 수 불일치 (expected {len(self.feat_cols_)}, got {X.shape[1]})"
                )
            X = pd.DataFrame(X, columns=self.feat_cols_, copy=True)
            inplace = True  # 어차피 방금 만든 임시 df라 inplace 안전
        elif not inplace:
            X = X.copy()

        # Quantile 그룹 변환 (결과 float64 → 입력이 float32였으면 되돌림)
        if self.quantile_cols_ and self.qt_ is not None:
            arr = self.qt_.transform(X[self.quantile_cols_])
            in_dtype = X[self.quantile_cols_].dtypes.iloc[0]
            if in_dtype == np.float32:
                arr = arr.astype('float32', copy=False)
            X[self.quantile_cols_] = arr

        # Power 그룹 변환 (마찬가지로 dtype 복구)
        if self.power_cols_ and self.pt_ is not None:
            arr = self.pt_.transform(X[self.power_cols_])
            in_dtype = X[self.power_cols_].dtypes.iloc[0]
            if in_dtype == np.float32:
                arr = arr.astype('float32', copy=False)
            X[self.power_cols_] = arr

        # Binary 그룹은 손대지 않음 (passthrough)

        if is_numpy:
            out = X.values
            if in_dtype_np == np.float32:
                out = out.astype('float32', copy=False)
            return out
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

    이 함수 1번 호출이면 3개 split이 모두 스케일링되고, fitted scaler 객체까지
    반환되어 대시보드/예측 파이프라인에서 pickle로 재사용 가능.

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
    # fit은 train에서만. feat_cols를 키워드로 명시해 fit(X, y=None) 시그니처와 충돌 방지
    scaler = HybridScaler(
        skew_threshold=skew_threshold,
        n_quantiles=n_quantiles,
        quantile_output=quantile_output,
        random_state=random_state,
    ).fit(xs_train, feat_cols=feat_cols)

    # 같은 scaler로 3 split 모두 transform (val/test는 train에서 배운 변환을 그대로 적용)
    scaler.transform(xs_train, inplace=True)
    scaler.transform(xs_val,   inplace=True)
    scaler.transform(xs_test,  inplace=True)

    return xs_train, xs_val, xs_test, scaler
