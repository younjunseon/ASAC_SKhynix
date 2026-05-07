"""
Final 파이프라인 — 스케일링 분기 유틸

정책:
- enet: RobustScaler (median=0, IQR=1) — train 기준 fit, 전 split transform
- xgb / catboost / lgbm / et / zitboost: pass-through (스케일링 없음)

사용법
------
    from final.modules import scaler

    if scaler.needs_scaling(model_name):
        xs_train, xs_val, xs_test, stats = scaler.fit_transform(
            xs_train, xs_val, xs_test, feat_cols,
        )
    # 트리 모델이면 원본 그대로 사용
"""
import numpy as np
import pandas as pd


# 스케일링이 필요한 모델 이름 (ElasticNet만 해당)
_SCALING_REQUIRED = {"enet"}


def needs_scaling(model_name):
    """모델 이름 기준으로 스케일링 필요 여부 판정."""
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
    ref = xs_train[feat_cols]
    medians = ref.median()
    iqr = ref.quantile(0.75) - ref.quantile(0.25)
    iqr_safe = iqr.clip(lower=1e-8)

    out = []
    for df in (xs_train, xs_val, xs_test):
        df_s = df.copy()
        df_s[feat_cols] = (df_s[feat_cols] - medians) / iqr_safe
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
    return xs_train, xs_val, xs_test, None
