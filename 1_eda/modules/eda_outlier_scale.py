"""
EDA 모듈 4: 이상치 & 스케일 분석 + EDA 요약
노트북에서 import eda_outlier_scale as out 로 사용
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.config import SEED


def plot_outlier_ratio(xs, feat_cols):
    """IQR 기반 이상치 비율 분포 + 상위 6개 boxplot"""
    Q1 = xs[feat_cols].quantile(0.25)
    Q3 = xs[feat_cols].quantile(0.75)
    IQR = Q3 - Q1

    outlier_low = xs[feat_cols] < (Q1 - 1.5 * IQR)
    outlier_high = xs[feat_cols] > (Q3 + 1.5 * IQR)
    outlier_pct = ((outlier_low | outlier_high).sum() / len(xs) * 100).sort_values(ascending=False)

    print(f"이상치 비율 > 5% Feature: {(outlier_pct > 5).sum()}개")
    print(f"이상치 비율 > 10% Feature: {(outlier_pct > 10).sum()}개")
    print(f"\n이상치 비율 상위 20개:")
    print(outlier_pct.head(20).round(2))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(outlier_pct.values, bins=50, edgecolor='black', color='mediumpurple')
    axes[0].set_title('Feature별 이상치 비율 분포')
    axes[0].set_xlabel('이상치 비율 (%)')
    axes[0].set_ylabel('Feature 수')

    top_outlier_feats = outlier_pct.head(6).index.tolist()
    sample = xs[top_outlier_feats].sample(n=min(5000, len(xs)), random_state=SEED)
    sample_melted = sample.melt(var_name='Feature', value_name='Value')
    sns.boxplot(data=sample_melted, x='Feature', y='Value', ax=axes[1], fliersize=1)
    axes[1].set_title('이상치 비율 상위 6개 Feature Boxplot')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def plot_scale_analysis(xs, feat_cols):
    """Feature 스케일 비교 (mean/range/skewness 분포)"""
    feat_stats = xs[feat_cols].describe().T[['mean', 'std', 'min', 'max']]
    feat_stats['range'] = feat_stats['max'] - feat_stats['min']
    feat_stats['skew'] = xs[feat_cols].skew()

    print(f"mean 범위:  [{feat_stats['mean'].min():.4f}, {feat_stats['mean'].max():.4f}]")
    print(f"std 범위:   [{feat_stats['std'].min():.4f}, {feat_stats['std'].max():.4f}]")
    print(f"range 범위: [{feat_stats['range'].min():.4f}, {feat_stats['range'].max():.4f}]")
    print(f"\n왜도(skewness):")
    print(f"  |skew| > 2: {(feat_stats['skew'].abs() > 2).sum()}개")
    print(f"  |skew| > 5: {(feat_stats['skew'].abs() > 5).sum()}개")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(feat_stats['mean'].values, bins=100, edgecolor='black', color='steelblue')
    axes[0].set_title('Feature Mean 분포')
    axes[0].set_xlabel('Mean')

    axes[1].hist(feat_stats['range'].values, bins=100, edgecolor='black', color='coral')
    axes[1].set_title('Feature Range(max-min) 분포')
    axes[1].set_xlabel('Range')

    axes[2].hist(feat_stats['skew'].dropna().values, bins=100, edgecolor='black', color='mediumpurple')
    axes[2].set_title('Feature Skewness 분포')
    axes[2].set_xlabel('Skewness')
    axes[2].axvline(x=0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def print_summary(const_feats, missing_df, discrete_feats, continuous_feats):
    """EDA 요약 출력"""
    print(f"상수 feature: {len(const_feats)}개")
    print(f"결측 있는 feature: {len(missing_df)}개")
    print(f"이산형 feature: {len(discrete_feats)}개")
    print(f"연속형 feature: {len(continuous_feats)}개")