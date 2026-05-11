"""
스케일링 분기 유틸.

정책:
- enet: RobustScaler 효과 — (X - median) / IQR. train 기준으로 통계량 fit, train/val/test 전부 transform
- xgb / catboost / lgbm / et / zitboost 등 트리 계열: pass-through (스케일링이 결과를 안 바꿈)

호출 측은 보통 maybe_scale()만 부르면 모델 이름에 따라 알아서 분기된다.

사용법
------
    from modules import scaler

    xs_train, xs_val, xs_test, stats = scaler.maybe_scale(
        xs_train, xs_val, xs_test, feat_cols, model_name,
    )
    # 트리 모델이면 stats=None, DataFrame은 원본 그대로 반환
"""
import numpy as np
import pandas as pd


# 스케일링이 필요한 모델 이름 집합 (현재 ElasticNet만 해당)
_SCALING_REQUIRED = {"enet"}


def needs_scaling(model_name):
    """이 모델 이름이 스케일링을 필요로 하는지 여부."""
    return model_name in _SCALING_REQUIRED


def fit_transform(xs_train, xs_val, xs_test, feat_cols):
    """Train 기준 RobustScaler fit → train/val/test 전부 transform.

    원본 DataFrame을 건드리지 않고 복사본 반환.

    Parameters
    ----------
    xs_train, xs_val, xs_test : DataFrame
    feat_cols : list[str]

    Returns
    -------
    xs_train_s, xs_val_s, xs_test_s : DataFrame (스케일링된 복사본)
    stats : DataFrame  (index=feat_cols, columns=['median', 'iqr'])
    """
    # 통계량은 train에서만 (leakage 방지)
    ref = xs_train[feat_cols]
    medians = ref.median()
    iqr = ref.quantile(0.75) - ref.quantile(0.25)
    iqr_safe = iqr.clip(lower=1e-8)   # IQR=0인 사실상 상수 컬럼에서 0 나눗셈 방지

    out = []
    for df in (xs_train, xs_val, xs_test):
        df_s = df.copy()                                       # 원본 보호
        df_s[feat_cols] = (df_s[feat_cols] - medians) / iqr_safe   # 3 split 모두 train 기준 값으로 변환
        out.append(df_s)

    stats = pd.DataFrame({"median": medians, "iqr": iqr})
    zero_iqr = int((iqr < 1e-8).sum())
    print(f"[RobustScaler fit_transform] feat={len(feat_cols)}개, "
          f"IQR=0 컬럼(스킵)={zero_iqr}개")

    return (*out, stats)


def maybe_scale(xs_train, xs_val, xs_test, feat_cols, model_name):
    """모델 이름에 따라 자동 분기. enet이면 fit_transform, 아니면 pass-through.

    Returns
    -------
    xs_train, xs_val, xs_test : DataFrame (enet이면 스케일된 복사본,
                                           아니면 원본 객체 그대로)
    stats : DataFrame or None
    """
    if needs_scaling(model_name):
        return fit_transform(xs_train, xs_val, xs_test, feat_cols)
    return xs_train, xs_val, xs_test, None   # 트리 계열 — 그대로 통과 (복사조차 안 함)
