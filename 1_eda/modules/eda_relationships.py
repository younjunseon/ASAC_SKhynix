"""
EDA 모듈 3: Feature-Target & Feature 간 관계 분석
노트북에서 import eda_relationships as rel 로 사용
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.config import KEY_COL, TARGET_COL, SEED


def compute_correlation(xs_dict, ys_train, feat_cols):
    """die→unit 평균 집계 → target 상관계수 계산. (train_merged, corr_with_target, corr_abs) 반환."""
    xs_train = xs_dict["train"]
    xs_unit_mean = xs_train.groupby(KEY_COL)[feat_cols].mean()

    train_merged = xs_unit_mean.merge(ys_train, left_index=True, right_on=KEY_COL, how='inner')
    print(f"Train merged shape: {train_merged.shape}")

    corr_with_target = train_merged[feat_cols].corrwith(train_merged[TARGET_COL]).dropna()
    corr_abs = corr_with_target.abs().sort_values(ascending=False)

    print(f"\n상관계수 상위 20개 Feature:")
    top20_corr = corr_abs.head(20)
    for feat_name in top20_corr.index:
        print(f"  {feat_name:>8}: {corr_with_target[feat_name]:+.4f}")

    return train_merged, corr_with_target, corr_abs


def plot_corr_with_target(corr_with_target, corr_abs):
    """상관계수 분포 히스토그램 + 상위 20개 bar chart"""
    top20_corr = corr_abs.head(20)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].hist(corr_with_target.values, bins=100, edgecolor='black', color='steelblue')
    axes[0].set_title('Feature-Target 상관계수 분포')
    axes[0].set_xlabel('Pearson r')
    axes[0].set_ylabel('Feature 수')
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)

    colors = ['coral' if corr_with_target[f] < 0 else 'steelblue' for f in top20_corr.index]
    axes[1].barh(range(len(top20_corr)), [corr_with_target[f] for f in top20_corr.index],
                 color=colors, edgecolor='black')
    axes[1].set_yticks(range(len(top20_corr)))
    axes[1].set_yticklabels(top20_corr.index, fontsize=9)
    axes[1].set_xlabel('Pearson r')
    axes[1].set_title('상관계수 상위 20개 Feature')
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.show()


def plot_top_scatter(train_merged, corr_with_target, corr_abs, n=6):
    """상위 N개 feature vs health scatter plot"""
    top_feats = corr_abs.head(n).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i, col in enumerate(top_feats):
        sample = train_merged[[col, TARGET_COL]].dropna().sample(
            n=min(3000, len(train_merged)), random_state=SEED)
        axes[i].scatter(sample[col], sample[TARGET_COL], alpha=0.2, s=5, color='steelblue')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('health')
        axes[i].set_title(f'{col} vs health (r={corr_with_target[col]:+.3f})')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f'상관계수 상위 {n}개 Feature vs Target', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_feature_heatmap(train_merged, corr_abs, n=30):
    """상위 N개 feature 간 상관 히트맵 + 고상관 쌍 출력"""
    top_feats = corr_abs.head(n).index.tolist()
    corr_matrix = train_merged[top_feats].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr_matrix, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                annot=False, square=True, linewidths=0.5, ax=ax,
                xticklabels=True, yticklabels=True)
    ax.set_title(f'상위 {n}개 Feature 간 상관관계 히트맵', fontsize=14)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.show()

    # 높은 상관관계 쌍 탐지 (|r| > 0.95)
    high_corr_pairs = []
    for i in range(len(top_feats)):
        for j in range(i + 1, len(top_feats)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.95:
                high_corr_pairs.append((top_feats[i], top_feats[j], r))

    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    print(f"|r| > 0.95인 Feature 쌍: {len(high_corr_pairs)}개")
    for a, b, r in high_corr_pairs[:15]:
        print(f"  {a:>8} ↔ {b:<8}: r={r:+.4f}")


def plot_discrete_vs_target(xs_dict, ys_train, discrete_feats):
    """이산형 feature 값별 target 평균 bar chart"""
    if not discrete_feats:
        print("이산형 feature가 없습니다.")
        return

    sample_disc_feats = discrete_feats[:min(8, len(discrete_feats))]
    xs_tr = xs_dict["train"]

    xs_train_disc = xs_tr.groupby(KEY_COL)[sample_disc_feats].median()
    disc_merged = xs_train_disc.merge(ys_train, left_index=True, right_on=KEY_COL, how='inner')

    n_feats = len(sample_disc_feats)
    n_rows = (n_feats + 3) // 4
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(sample_disc_feats):
        group_mean = disc_merged.groupby(col)[TARGET_COL].mean()
        group_mean.plot(kind='bar', ax=axes[i], color='steelblue', edgecolor='black')
        axes[i].set_title(f'{col}별 health 평균')
        axes[i].set_ylabel('health 평균')
        axes[i].tick_params(axis='x', rotation=0)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('이산형 Feature 값별 Target 평균', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()
