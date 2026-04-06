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
    """
    Train set의 die→unit 평균 집계 후, 각 feature와 target의 Pearson 상관계수 계산

    Parameters
    ----------
    xs_dict : dict
        {"train": DataFrame, ...} split별 die-level 데이터
    ys_train : DataFrame
        train Y 데이터 (ufs_serial, health 컬럼)
    feat_cols : list of str

    Returns
    -------
    train_merged : DataFrame
        unit-level로 집계된 X + Y 합본
    corr_with_target : Series
        각 feature의 상관계수 (부호 포함)
    corr_abs : Series
        절대값 기준 내림차순 정렬
    """
    xs_train = xs_dict["train"]
    xs_unit_mean = xs_dict['train_unit_mean'] if 'train_unit_mean' in xs_dict else xs_train.groupby(KEY_COL)[feat_cols].mean()

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
    """
    Feature-Target 상관계수 분포 히스토그램 + 상위 20개 bar chart

    Parameters
    ----------
    corr_with_target : Series
        compute_correlation()에서 반환된 상관계수 (부호 포함)
    corr_abs : Series
        compute_correlation()에서 반환된 절대값 기준 정렬
    """
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
    """
    상관계수 상위 N개 feature vs health scatter plot

    Parameters
    ----------
    train_merged : DataFrame
        compute_correlation()에서 반환된 unit-level X+Y 합본
    corr_with_target : Series
        상관계수 (부호 포함)
    corr_abs : Series
        절대값 기준 정렬
    n : int
        시각화할 feature 수. 기본 6
    """
    top_feats = corr_abs.head(n).index.tolist()
    target = train_merged[TARGET_COL]

    # 전체 scatter
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes = np.array(axes).flatten()
    sample = train_merged.sample(n=min(3000, len(train_merged)), random_state=SEED)
    for i, col in enumerate(top_feats):
        axes[i].scatter(sample[col], sample[TARGET_COL], alpha=0.2, s=5, color='steelblue')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('health')
        axes[i].set_title(f'{col} vs health (r={corr_with_target[col]:+.3f})')
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(f'상관계수 상위 {n}개 Feature vs Target (전체)', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()

    # Y>0만 scatter
    pos_mask = target > 0
    merged_pos = train_merged[pos_mask]
    target_pos = target[pos_mask]
    sample_pos = merged_pos.sample(n=min(3000, len(merged_pos)), random_state=SEED)

    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes2 = np.array(axes2).flatten()
    for i, col in enumerate(top_feats):
        r_pos = sample_pos[col].corr(sample_pos[TARGET_COL])
        axes2[i].scatter(sample_pos[col], sample_pos[TARGET_COL],
                         alpha=0.2, s=8, color='coral')
        axes2[i].set_xlabel(col)
        axes2[i].set_ylabel('health')
        axes2[i].set_title(f'{col} vs health\nr(Y>0)={r_pos:+.3f}')
    for j in range(i + 1, len(axes2)):
        axes2[j].set_visible(False)
    plt.suptitle(f'상관계수 상위 {n}개 Feature vs Target (Y>0만, n={pos_mask.sum():,})',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()

    # Y>0 + 이상치 제거 scatter
    upper = target_pos.quantile(0.99)
    clip_mask = target_pos <= upper
    merged_clip = merged_pos[clip_mask]
    target_clip = target_pos[clip_mask]
    sample_clip = merged_clip.sample(n=min(3000, len(merged_clip)), random_state=SEED)

    fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes3 = np.array(axes3).flatten()
    for i, col in enumerate(top_feats):
        r_clip = sample_clip[col].corr(sample_clip[TARGET_COL])
        axes3[i].scatter(sample_clip[col], sample_clip[TARGET_COL],
                         alpha=0.2, s=8, color='mediumseagreen')
        axes3[i].set_xlabel(col)
        axes3[i].set_ylabel('health')
        axes3[i].set_title(f'{col} vs health\nr(Y>0,clip99)={r_clip:+.3f}')
    for j in range(i + 1, len(axes3)):
        axes3[j].set_visible(False)
    plt.suptitle(f'상관계수 상위 {n}개 Feature vs Target '
                 f'(Y>0 + 상위1% 제거, n={clip_mask.sum():,})',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()

    # 비교 테이블
    print(f"\n전체 vs Y>0 vs Y>0+clip99 상관계수 비교")
    print(f"  (전체: {len(train_merged):,}, Y>0: {pos_mask.sum():,}, "
          f"Y>0+clip99: {clip_mask.sum():,}, 상위1% 기준: {upper:.4f})")
    print(f"  {'Feature':>10}  {'r(전체)':>10}  {'r(Y>0)':>10}  {'r(clip99)':>10}")
    print("-" * 50)
    for col in top_feats:
        r_all = corr_with_target[col]
        r_p = merged_pos[col].corr(target_pos)
        r_c = merged_clip[col].corr(target_clip)
        print(f"  {col:>10}  {r_all:>+10.4f}  {r_p:>+10.4f}  {r_c:>+10.4f}")


def plot_feature_heatmap(train_merged, corr_abs, n=30):
    """
    Target 상관 상위 N개 feature 간 상관관계 히트맵 + |r|>0.95인 고상관 쌍 출력

    Parameters
    ----------
    train_merged : DataFrame
        compute_correlation()에서 반환된 unit-level X+Y 합본
    corr_abs : Series
        절대값 기준 정렬
    n : int
        히트맵에 포함할 feature 수. 기본 30
    """
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


def compute_vif(train_merged, feat_cols, top_n=30):
    """
    전체 feature의 VIF(Variance Inflation Factor)를 상관행렬 역행렬로 빠르게 계산하고,
    상위 top_n개를 시각화한다.

    VIF_i = diag(R^{-1})_i  (R = 상관행렬)
    상수/극저분산 feature 및 특이행렬 문제를 자동 처리한다.

    Parameters
    ----------
    train_merged : DataFrame
        compute_correlation()에서 반환된 unit-level X+Y 합본
    feat_cols : list of str
        VIF 계산 대상 feature 컬럼 리스트 (전체 X feature)
    top_n : int
        시각화 및 상세 출력할 상위 feature 수. 기본 30

    Returns
    -------
    vif_df : DataFrame
        feature, VIF 컬럼을 가진 DataFrame (VIF 내림차순, 전체)
    """
    # 1) 결측률 50% 이상 feature 제거
    missing_rate = train_merged[feat_cols].isnull().mean()
    high_missing = (missing_rate >= 0.5).sum()
    low_missing_feats = missing_rate[missing_rate < 0.5].index.tolist()

    # 2) 상수/극저분산 feature 제거 (VIF 계산 불가)
    valid_feats = [c for c in low_missing_feats
                   if train_merged[c].std() > 1e-10]
    low_var = len(low_missing_feats) - len(valid_feats)

    # 3) 나머지 결측은 median 대체
    X = train_merged[valid_feats].fillna(train_merged[valid_feats].median())

    print(f"VIF 계산: 전체 {len(feat_cols)}개 → {len(valid_feats)}개 대상")
    print(f"  제외: 결측>=50% {high_missing}개, 상수/극저분산 {low_var}개"
          f" | 나머지 결측은 median 대체 | {len(X):,}행")

    # 상관행렬 역행렬의 대각원소 = VIF
    corr_matrix = np.corrcoef(X.values, rowvar=False)
    # 특이행렬 대비 pseudo-inverse 사용
    try:
        corr_inv = np.linalg.inv(corr_matrix)
    except np.linalg.LinAlgError:
        corr_inv = np.linalg.pinv(corr_matrix)
        print("  (특이행렬 → pseudo-inverse 사용)")

    vif_values = np.diag(corr_inv)
    vif_df = pd.DataFrame({"feature": valid_feats, "VIF": vif_values})
    vif_df = vif_df.sort_values("VIF", ascending=False).reset_index(drop=True)

    # 요약 통계
    n_severe = (vif_df["VIF"] > 10).sum()
    n_moderate = ((vif_df["VIF"] > 5) & (vif_df["VIF"] <= 10)).sum()
    n_ok = (vif_df["VIF"] <= 5).sum()
    print(f"\n  VIF > 10 (심각한 다중공선성): {n_severe}개")
    print(f"  VIF 5~10 (주의 필요):       {n_moderate}개")
    print(f"  VIF <= 5 (양호):            {n_ok}개")

    # 상위 top_n 상세 출력
    show_n = min(top_n, len(vif_df))
    print(f"\nVIF 상위 {show_n}개:")
    for _, row in vif_df.head(show_n).iterrows():
        marker = " ★" if row["VIF"] > 10 else " ▲" if row["VIF"] > 5 else ""
        print(f"  {row['feature']:>8}: {row['VIF']:.2f}{marker}")

    # 시각화: 상위 top_n개
    plot_df = vif_df.head(show_n).iloc[::-1]  # 역순 (큰 값이 위)
    fig, ax = plt.subplots(figsize=(10, max(6, show_n * 0.3)))
    colors = ["#d62728" if v > 10 else "#ff7f0e" if v > 5 else "steelblue"
              for v in plot_df["VIF"].values]
    ax.barh(range(len(plot_df)), plot_df["VIF"].values, color=colors, edgecolor="black")
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["feature"].values, fontsize=9)
    ax.set_xlabel("VIF")
    ax.set_title(f"VIF 상위 {show_n}개 / 전체 {len(valid_feats)}개"
                 f" (★ >10: {n_severe}개, ▲ 5~10: {n_moderate}개)")
    ax.axvline(x=5, color="orange", linestyle="--", alpha=0.7, label="VIF=5")
    ax.axvline(x=10, color="red", linestyle="--", alpha=0.7, label="VIF=10")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return vif_df


def plot_discrete_vs_target(xs_dict, ys_train, discrete_feats):
    """
    이산형 feature의 각 값(카테고리)별 target 평균을 bar chart로 비교

    Parameters
    ----------
    xs_dict : dict
        {"train": DataFrame, ...} split별 die-level 데이터
    ys_train : DataFrame
        train Y 데이터 (ufs_serial, health 컬럼)
    discrete_feats : list of str
        이산형 feature 컬럼명 리스트 (최대 8개 시각화)
    """
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
        axes[i].tick_params(axis='x', rotation=90)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('이산형 Feature 값별 Target 평균', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()
