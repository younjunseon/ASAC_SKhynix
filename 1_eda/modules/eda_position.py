"""
EDA 모듈: Position별 분석
- unit 내 die position(1~4)은 물리적 위치를 나타냄
- position별 feature 평균 차이 → 특정 position이 불량에 더 기여하는지
- position별로 따로 집계한 feature가 target과 더 높은 상관을 보이는지
- position 간 편차 자체가 유용한 feature가 될 수 있는지
- 노트북에서 import eda_position as pos 로 사용
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats
from utils.config import KEY_COL, POSITION_COL, TARGET_COL, SEED


def position_overview(xs_dict, ys_train, feat_cols, n_feats=10):
    """
    Position별 기본 통계: feature 평균 차이, Kruskal-Wallis 검정

    Parameters
    ----------
    xs_dict : dict
    ys_train : DataFrame
    feat_cols : list of str
    n_feats : int  - position 간 차이가 큰 feature 출력 수

    Returns
    -------
    pos_diff_df : DataFrame  (feature별 position 간 차이 통계)
    """
    xs_train = xs_dict["train"]

    # position별 feature 평균
    pos_means = xs_train.groupby(POSITION_COL)[feat_cols].mean()

    print("=" * 60)
    print(f"Position 분포: {xs_train[POSITION_COL].value_counts().sort_index().to_dict()}")
    print("=" * 60)

    # feature별 position 간 최대 차이 (range of means)
    results = []
    for col in feat_cols:
        means = pos_means[col]
        overall_std = xs_train[col].std()
        if overall_std == 0 or pd.isna(overall_std):
            continue

        # position 간 평균 range / 전체 std → effect size
        pos_range = means.max() - means.min()
        effect = pos_range / overall_std

        # Kruskal-Wallis: position별 분포 차이
        groups = [xs_train.loc[xs_train[POSITION_COL] == p, col].dropna().values
                  for p in sorted(xs_train[POSITION_COL].unique())]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            h_stat, p_val = sp_stats.kruskal(*groups)
        else:
            h_stat, p_val = 0, 1.0

        results.append({
            "feature": col,
            "pos_range": pos_range,
            "effect": effect,
            "kw_h": h_stat,
            "kw_p": p_val,
            **{f"mean_pos{p}": means[p] for p in sorted(pos_means.index)},
        })

    pos_diff_df = pd.DataFrame(results)
    pos_diff_df = pos_diff_df.sort_values("effect", ascending=False).reset_index(drop=True)

    n_sig = (pos_diff_df["kw_p"] < 0.05).sum()
    print(f"\nKruskal-Wallis p<0.05인 feature: {n_sig}/{len(pos_diff_df)}개")
    print(f"\nPosition 간 차이 큰 상위 {n_feats}개 Feature:")
    print(f"  {'Feature':>8}  {'effect':>7}  {'KW p':>9}  " +
          "  ".join([f"{'pos'+str(p):>8}" for p in sorted(pos_means.index)]))
    print("-" * 60)
    for _, row in pos_diff_df.head(n_feats).iterrows():
        pos_str = "  ".join([f"{row[f'mean_pos{p}']:>8.4f}" for p in sorted(pos_means.index)])
        print(f"  {row['feature']:>8}  {row['effect']:>7.4f}  {row['kw_p']:>9.2e}  {pos_str}")

    return pos_diff_df


def plot_position_top_features(xs_dict, pos_diff_df, n=8):
    """
    Position 간 차이가 큰 상위 feature의 position별 분포 boxplot

    Parameters
    ----------
    xs_dict : dict
    pos_diff_df : DataFrame
    n : int  - 시각화할 feature 수
    """
    xs_train = xs_dict["train"]
    top_feats = pos_diff_df.head(n)["feature"].tolist()

    n_rows = (n + 3) // 4
    fig, axes = plt.subplots(n_rows, 4, figsize=(18, 4.5 * n_rows))
    axes = np.array(axes).flatten()

    for i, feat in enumerate(top_feats):
        # 샘플링
        sample = xs_train[[POSITION_COL, feat]].dropna()
        if len(sample) > 20000:
            sample = sample.sample(20000, random_state=SEED)

        sns.boxplot(x=POSITION_COL, y=feat, data=sample, ax=axes[i],
                    palette="Set2", fliersize=1)
        effect = pos_diff_df.loc[pos_diff_df["feature"] == feat, "effect"].values[0]
        axes[i].set_title(f"{feat} (effect={effect:.3f})", fontsize=10)
        axes[i].set_xlabel("Position")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Position별 Feature 분포 (차이 상위 {n}개)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


def position_corr_with_target(xs_dict, ys_train, feat_cols, n_feats=15):
    """
    Position별로 따로 집계한 feature의 target 상관 vs mean 집계 상관 비교
    → "position X의 feature가 mean보다 더 강한 상관을 보이는가?"

    Parameters
    ----------
    xs_dict : dict
    ys_train : DataFrame
    feat_cols : list of str
    n_feats : int  - 출력할 feature 수

    Returns
    -------
    pos_corr_df : DataFrame  (feature × position별 상관계수)
    """
    xs_train = xs_dict["train"]

    positions = sorted(xs_train[POSITION_COL].unique())
    corr_results = {}

    # mean 집계 상관 (기준선)
    xs_mean = xs_train.groupby(KEY_COL)[feat_cols].mean()
    merged_mean = xs_mean.merge(ys_train, left_index=True, right_on=KEY_COL, how="inner")
    corr_mean = merged_mean[feat_cols].corrwith(merged_mean[TARGET_COL])
    corr_results["mean"] = corr_mean

    # position별 상관
    for p in positions:
        xs_pos = xs_train[xs_train[POSITION_COL] == p].set_index(KEY_COL)[feat_cols]
        merged_pos = xs_pos.merge(ys_train, left_index=True, right_on=KEY_COL, how="inner")
        corr_pos = merged_pos[feat_cols].corrwith(merged_pos[TARGET_COL])
        corr_results[f"pos{p}"] = corr_pos

    pos_corr_df = pd.DataFrame(corr_results)
    pos_corr_df.index.name = "feature"

    # 각 feature별 최적 position 찾기
    abs_df = pos_corr_df.abs()
    best_col = abs_df.idxmax(axis=1)
    max_r = abs_df.max(axis=1)
    mean_r = abs_df["mean"]
    gain = max_r - mean_r

    summary = pd.DataFrame({
        "feature": pos_corr_df.index,
        "best_source": best_col.values,
        "best_r": max_r.values,
        "mean_r": mean_r.values,
        "gain": gain.values,
    }).sort_values("gain", ascending=False).reset_index(drop=True)

    # mean이 아닌 position이 더 좋은 feature
    non_mean = summary[summary["best_source"] != "mean"]

    print("=" * 65)
    print("Position별 집계 vs Mean 집계: Target 상관 비교")
    print("=" * 65)

    # 각 source별 최대 |r| 요약
    print(f"\n  {'Source':>8}  {'max|r|':>8}  {'mean|r|':>9}")
    print("-" * 30)
    for col in pos_corr_df.columns:
        vals = pos_corr_df[col].abs()
        print(f"  {col:>8}  {vals.max():>8.4f}  {vals.mean():>9.4f}")

    print(f"\nPosition 집계가 mean보다 좋은 상위 {n_feats}개:")
    print(f"  {'Feature':>8}  {'best':>6}  {'best|r|':>8}  {'mean|r|':>8}  {'향상':>8}")
    print("-" * 50)
    for _, row in non_mean.head(n_feats).iterrows():
        print(f"  {row['feature']:>8}  {row['best_source']:>6}  {row['best_r']:>8.4f}  "
              f"{row['mean_r']:>8.4f}  {row['gain']:>+8.4f}")

    return pos_corr_df


def plot_position_corr_heatmap(pos_corr_df, n=20):
    """
    상위 feature의 position별 상관 히트맵

    Parameters
    ----------
    pos_corr_df : DataFrame
    n : int  - 히트맵에 표시할 feature 수
    """
    max_abs = pos_corr_df.abs().max(axis=1).sort_values(ascending=False)
    top_feats = max_abs.head(n).index

    fig, ax = plt.subplots(figsize=(8, max(6, n * 0.35)))
    sns.heatmap(pos_corr_df.loc[top_feats], cmap="RdBu_r", center=0,
                annot=True, fmt=".3f", linewidths=0.5, ax=ax,
                cbar_kws={"label": "Pearson r"})
    ax.set_title(f"상위 {n} Feature × Position별 Target 상관", fontsize=12)
    ax.set_ylabel("")
    plt.tight_layout()
    plt.show()


def position_deviation_corr(xs_dict, ys_train, feat_cols, n_feats=15):
    """
    Position 간 편차(std, range) 자체를 feature로 만들어 target과 상관 계산
    → "die 간 불균일성이 큰 unit이 불량이 높은가?"

    Parameters
    ----------
    xs_dict : dict
    ys_train : DataFrame
    feat_cols : list of str
    n_feats : int  - 출력할 feature 수

    Returns
    -------
    dev_corr : DataFrame  (feature × 편차 집계별 상관)
    """
    xs_train = xs_dict["train"]

    # unit별 position 간 std, range
    xs_std = xs_train.groupby(KEY_COL)[feat_cols].std()
    xs_range = xs_train.groupby(KEY_COL)[feat_cols].apply(lambda x: x.max() - x.min())

    # target merge
    merged_std = xs_std.merge(ys_train, left_index=True, right_on=KEY_COL, how="inner")
    merged_range = xs_range.merge(ys_train, left_index=True, right_on=KEY_COL, how="inner")

    corr_std = merged_std[feat_cols].corrwith(merged_std[TARGET_COL]).dropna()
    corr_range = merged_range[feat_cols].corrwith(merged_range[TARGET_COL]).dropna()

    dev_corr = pd.DataFrame({
        "std_corr": corr_std,
        "range_corr": corr_range,
    })
    dev_corr["max_abs"] = dev_corr.abs().max(axis=1)
    dev_corr = dev_corr.sort_values("max_abs", ascending=False)

    print("=" * 60)
    print("Position 간 편차 → Target 상관 (불균일성 지표)")
    print("=" * 60)
    print(f"  std  상관: max|r|={corr_std.abs().max():.4f}, mean|r|={corr_std.abs().mean():.4f}")
    print(f"  range 상관: max|r|={corr_range.abs().max():.4f}, mean|r|={corr_range.abs().mean():.4f}")
    print(f"\n편차-Target 상관 상위 {n_feats}개:")
    print(f"  {'Feature':>8}  {'std r':>8}  {'range r':>9}")
    print("-" * 35)
    for feat, row in dev_corr.head(n_feats).iterrows():
        print(f"  {feat:>8}  {row['std_corr']:>+8.4f}  {row['range_corr']:>+9.4f}")

    return dev_corr.drop(columns=["max_abs"])