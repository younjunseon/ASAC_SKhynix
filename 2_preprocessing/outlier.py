"""
이상치 처리 모듈
- IQR 기반 탐지
- Winsorization (분위수 경계로 클리핑)

EDA 결과 기반:
- IQR 기준 이상치 5% 초과: 167개 feature
- X393(45.7%), X988(40.4%) 등 극단적 이상치 비율
- Feature 스케일 극도로 불균일 (mean 범위: -2,293 ~ 20,201,109)
"""
import pandas as pd
import numpy as np


def detect_outliers_iqr(xs, feat_cols, multiplier=1.5):
    """
    IQR 기반 이상치 탐지 (통계만 반환, 데이터 수정 안 함)

    Parameters
    ----------
    xs : DataFrame
    feat_cols : list
    multiplier : float
        IQR 배수. 기본 1.5

    Returns
    -------
    outlier_stats : DataFrame
        feature별 이상치 수, 비율
    """
    Q1 = xs[feat_cols].quantile(0.25)
    Q3 = xs[feat_cols].quantile(0.75)
    IQR = Q3 - Q1

    low = xs[feat_cols] < (Q1 - multiplier * IQR)
    high = xs[feat_cols] > (Q3 + multiplier * IQR)
    outlier_count = (low | high).sum()
    outlier_pct = (outlier_count / len(xs) * 100).round(2)

    stats = pd.DataFrame({
        "outlier_count": outlier_count,
        "outlier_pct": outlier_pct,
    }).sort_values("outlier_pct", ascending=False)

    print(f"[이상치 탐지] IQR × {multiplier}")
    print(f"  이상치 > 5%: {(stats['outlier_pct'] > 5).sum()}개")
    print(f"  이상치 > 10%: {(stats['outlier_pct'] > 10).sum()}개")
    return stats


def winsorize(xs_train, xs_val, xs_test, feat_cols,
              lower_pct=0.01, upper_pct=0.99):
    """
    Winsorization: train 기준 분위수로 클리핑 (data leakage 방지)

    Parameters
    ----------
    xs_train, xs_val, xs_test : DataFrame
    feat_cols : list
    lower_pct : float
        하위 분위수 (기본 1%)
    upper_pct : float
        상위 분위수 (기본 99%)

    Returns
    -------
    xs_train, xs_val, xs_test : DataFrame (클리핑된 복사본)
    bounds : DataFrame (feature별 lower/upper 경계)
    """
    lower = xs_train[feat_cols].quantile(lower_pct)
    upper = xs_train[feat_cols].quantile(upper_pct)

    xs_train = xs_train.copy()
    xs_val = xs_val.copy()
    xs_test = xs_test.copy()

    xs_train[feat_cols] = xs_train[feat_cols].clip(lower, upper, axis=1)
    xs_val[feat_cols] = xs_val[feat_cols].clip(lower, upper, axis=1)
    xs_test[feat_cols] = xs_test[feat_cols].clip(lower, upper, axis=1)

    bounds = pd.DataFrame({"lower": lower, "upper": upper})

    print(f"[Winsorization] lower={lower_pct*100:.0f}%, upper={upper_pct*100:.0f}%")
    print(f"  적용 feature: {len(feat_cols)}개")
    return xs_train, xs_val, xs_test, bounds


def run_outlier_treatment(xs_train, xs_val, xs_test, feat_cols,
                          lower_pct=0.01, upper_pct=0.99):
    """
    이상치 처리 파이프라인 전체 실행

    Parameters
    ----------
    xs_train, xs_val, xs_test : DataFrame
    feat_cols : list
    lower_pct, upper_pct : float

    Returns
    -------
    xs_train, xs_val, xs_test : DataFrame (처리 완료)
    report : dict
    """
    print("=" * 60)
    print("이상치 처리 파이프라인 시작")
    print("=" * 60)

    report = {}

    # 1. 처리 전 이상치 현황
    stats_before = detect_outliers_iqr(xs_train, feat_cols)
    report["before"] = stats_before

    # 2. Winsorization
    xs_train, xs_val, xs_test, bounds = winsorize(
        xs_train, xs_val, xs_test, feat_cols, lower_pct, upper_pct
    )
    report["bounds"] = bounds

    # 3. 처리 후 이상치 현황
    stats_after = detect_outliers_iqr(xs_train, feat_cols)
    report["after"] = stats_after

    # 전후 비교
    before_5pct = (stats_before["outlier_pct"] > 5).sum()
    after_5pct = (stats_after["outlier_pct"] > 5).sum()
    before_10pct = (stats_before["outlier_pct"] > 10).sum()
    after_10pct = (stats_after["outlier_pct"] > 10).sum()

    print(f"\n{'=' * 60}")
    print("이상치 처리 완료")
    print(f"  이상치 >5%  feature: {before_5pct} → {after_5pct} ({before_5pct - after_5pct}개 감소)")
    print(f"  이상치 >10% feature: {before_10pct} → {after_10pct} ({before_10pct - after_10pct}개 감소)")
    print(f"  train: {xs_train.shape}")
    print("=" * 60)

    return xs_train, xs_val, xs_test, report
