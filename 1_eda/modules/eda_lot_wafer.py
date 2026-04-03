"""
EDA 모듈: Lot/Wafer 레벨 분석
- lot 28개, wafer 432장 간 품질 편차 분석
- lot별 평균 health·불량률 비교 → 특정 lot에 불량 집중?
- lot 간 feature 분포 차이 → 공정 조건 변동 탐지
- lot 내 wafer 번호 트렌드 → 공정 drift 확인
- Stage 4.5 (메타피처 설계) 근거
- 노트북에서 import eda_lot_wafer as lw 로 사용
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats
from utils.config import KEY_COL, DIE_KEY_COL, TARGET_COL, SEED


def _parse_and_merge(xs_dict, ys_train):
    """
    Train die에서 lot/wafer 파싱 후 target merge (unit level)

    Returns
    -------
    die_merged : DataFrame  (die-level, lot/wafer/health 포함)
    unit_merged : DataFrame  (unit-level, lot/wafer/health 포함)
    """
    xs_train = xs_dict["train"]

    # lot, wafer 파싱
    parts = xs_train[DIE_KEY_COL].str.split("_", expand=True)
    die_df = xs_train[[KEY_COL, DIE_KEY_COL]].copy()
    die_df["lot"] = parts[0]
    die_df["wafer_no"] = parts[1].astype(int)
    die_df["wafer_id"] = die_df["lot"] + "_" + parts[1]

    # die-level에 target merge
    die_merged = die_df.merge(ys_train[[KEY_COL, TARGET_COL]], on=KEY_COL, how="inner")

    # unit-level 집계
    unit_merged = die_merged.groupby(KEY_COL).agg(
        lot=("lot", "first"),
        wafer_no=("wafer_no", "first"),
        wafer_id=("wafer_id", "first"),
        health=(TARGET_COL, "first"),
    ).reset_index()

    return die_merged, unit_merged


def lot_overview(xs_dict, ys_train):
    """
    Lot별 기본 통계 + 불량률 bar chart

    Parameters
    ----------
    xs_dict : dict
    ys_train : DataFrame

    Returns
    -------
    lot_stats : DataFrame  (lot별 통계)
    die_merged : DataFrame
    unit_merged : DataFrame
    """
    die_merged, unit_merged = _parse_and_merge(xs_dict, ys_train)

    # lot별 통계
    lot_stats = unit_merged.groupby("lot").agg(
        n_units=("health", "size"),
        n_wafers=("wafer_id", "nunique"),
        mean_health=("health", "mean"),
        defect_rate=("health", lambda x: (x > 0).mean() * 100),
        mean_health_pos=("health", lambda x: x[x > 0].mean() if (x > 0).any() else 0),
    ).reset_index()
    lot_stats = lot_stats.sort_values("defect_rate", ascending=False).reset_index(drop=True)

    print("=" * 70)
    print(f"Lot 레벨 개요 (총 {lot_stats['lot'].nunique()}개 lot)")
    print("=" * 70)
    print(f"  {'Lot':>10}  {'Units':>6}  {'Wafers':>7}  {'불량률%':>7}  "
          f"{'mean(health)':>13}  {'mean(Y>0)':>10}")
    print("-" * 70)
    for _, row in lot_stats.iterrows():
        print(f"  {row['lot']:>10}  {int(row['n_units']):>6}  {int(row['n_wafers']):>7}  "
              f"{row['defect_rate']:>7.1f}  {row['mean_health']:>13.6f}  "
              f"{row['mean_health_pos']:>10.6f}")

    return lot_stats, die_merged, unit_merged


def plot_lot_quality(lot_stats, top_n=15):
    """
    Lot별 불량률 + 평균 health bar chart

    Parameters
    ----------
    lot_stats : DataFrame
    top_n : int  - 표시할 lot 수
    """
    display_stats = lot_stats.head(top_n)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # 1) 불량률
    axes[0].bar(range(len(display_stats)), display_stats["defect_rate"],
                color="coral", edgecolor="black", alpha=0.8)
    axes[0].set_xticks(range(len(display_stats)))
    axes[0].set_xticklabels(display_stats["lot"], rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("불량률 (%)")
    axes[0].set_title(f"Lot별 불량률 (상위 {top_n})")

    # 2) 평균 health
    axes[1].bar(range(len(display_stats)), display_stats["mean_health"],
                color="steelblue", edgecolor="black", alpha=0.8)
    axes[1].set_xticks(range(len(display_stats)))
    axes[1].set_xticklabels(display_stats["lot"], rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("평균 health")
    axes[1].set_title(f"Lot별 평균 health (상위 {top_n})")

    # 3) unit 수
    axes[2].bar(range(len(display_stats)), display_stats["n_units"],
                color="mediumseagreen", edgecolor="black", alpha=0.8)
    axes[2].set_xticks(range(len(display_stats)))
    axes[2].set_xticklabels(display_stats["lot"], rotation=45, ha="right", fontsize=8)
    axes[2].set_ylabel("Unit 수")
    axes[2].set_title(f"Lot별 Unit 수 (상위 {top_n})")

    plt.suptitle("Lot 품질 비교", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def lot_quality_test(unit_merged):
    """
    Kruskal-Wallis H-test: lot 간 health 분포에 유의미한 차이가 있는지 검정
    + lot별 health boxplot

    Parameters
    ----------
    unit_merged : DataFrame

    Returns
    -------
    h_stat : float
    p_value : float
    """
    groups = [grp["health"].values for _, grp in unit_merged.groupby("lot")]
    h_stat, p_value = sp_stats.kruskal(*groups)

    print("=" * 60)
    print("Kruskal-Wallis H-test (Lot 간 health 차이)")
    print("=" * 60)
    print(f"  H-statistic: {h_stat:.4f}")
    print(f"  p-value: {p_value:.2e}")
    if p_value < 0.05:
        print("  → Lot 간 health 분포에 유의미한 차이 있음 (p < 0.05)")
        print("  → Lot 정보를 메타피처로 활용할 가치 있음")
    else:
        print("  → Lot 간 유의미한 차이 없음")

    # boxplot
    fig, ax = plt.subplots(figsize=(16, 5))
    lot_order = unit_merged.groupby("lot")["health"].mean().sort_values(ascending=False).index
    sns.boxplot(x="lot", y="health", data=unit_merged, order=lot_order,
                ax=ax, palette="Set3", fliersize=2)
    ax.set_title("Lot별 health 분포 (Kruskal-Wallis)")
    ax.set_xlabel("Lot")
    ax.set_ylabel("health")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    plt.tight_layout()
    plt.show()

    return h_stat, p_value


def wafer_trend_in_lot(unit_merged, top_n_lots=6):
    """
    Lot 내에서 wafer 번호에 따른 health 트렌드 확인 (공정 drift)

    Parameters
    ----------
    unit_merged : DataFrame
    top_n_lots : int  - 표시할 lot 수 (불량률 높은 순)
    """
    lot_order = (unit_merged.groupby("lot")["health"].mean()
                 .sort_values(ascending=False).index[:top_n_lots])

    n_cols = min(3, top_n_lots)
    n_rows = (top_n_lots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes = np.array(axes).flatten()

    for i, lot_id in enumerate(lot_order):
        lot_data = unit_merged[unit_merged["lot"] == lot_id]
        wafer_stats = lot_data.groupby("wafer_no").agg(
            mean_health=("health", "mean"),
            defect_rate=("health", lambda x: (x > 0).mean() * 100),
            n_units=("health", "size"),
        ).reset_index().sort_values("wafer_no")

        ax = axes[i]
        ax2 = ax.twinx()

        ax.bar(wafer_stats["wafer_no"], wafer_stats["defect_rate"],
               color="coral", alpha=0.6, label="불량률(%)")
        ax2.plot(wafer_stats["wafer_no"], wafer_stats["mean_health"],
                 "o-", color="steelblue", markersize=4, label="mean health")

        ax.set_xlabel("Wafer 번호")
        ax.set_ylabel("불량률 (%)", color="coral")
        ax2.set_ylabel("mean health", color="steelblue")
        ax.set_title(f"Lot {lot_id}", fontsize=10)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Lot 내 Wafer 번호별 품질 트렌드 (상위 {top_n_lots} Lot)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()


def lot_feature_comparison(xs_dict, ys_train, feat_cols, top_n_lots=6, top_n_feats=6):
    """
    불량률 상/하위 lot 간 주요 feature 분포 비교
    → 공정 조건 변동 탐지 (어떤 WT feature가 lot 간에 크게 다른지)

    Parameters
    ----------
    xs_dict : dict
    ys_train : DataFrame
    feat_cols : list of str
    top_n_lots : int  - 비교할 lot 수 (상위 + 하위)
    top_n_feats : int  - 비교할 feature 수
    """
    xs_train = xs_dict["train"]

    # lot 파싱
    parts = xs_train[DIE_KEY_COL].str.split("_", expand=True)
    xs_with_lot = xs_train[[KEY_COL, DIE_KEY_COL] + feat_cols].copy()
    xs_with_lot["lot"] = parts[0]

    # unit-level로 집계
    unit_data = xs_with_lot.groupby(KEY_COL).agg(
        lot=("lot", "first"),
        **{f: (f, "mean") for f in feat_cols}
    ).reset_index()

    unit_data = unit_data.merge(ys_train[[KEY_COL, TARGET_COL]], on=KEY_COL, how="inner")

    # 불량률 기준 상위/하위 lot
    lot_defect = unit_data.groupby("lot")[TARGET_COL].apply(lambda x: (x > 0).mean()).sort_values()
    worst_lots = lot_defect.tail(top_n_lots // 2).index.tolist()
    best_lots = lot_defect.head(top_n_lots // 2).index.tolist()

    # lot 간 feature 차이 큰 것 찾기 (worst vs best 평균 차이)
    worst_data = unit_data[unit_data["lot"].isin(worst_lots)]
    best_data = unit_data[unit_data["lot"].isin(best_lots)]

    mean_diff = (worst_data[feat_cols].mean() - best_data[feat_cols].mean()).abs()
    std_pool = unit_data[feat_cols].std()
    effect = (mean_diff / std_pool).dropna().sort_values(ascending=False)
    top_feats = effect.head(top_n_feats).index.tolist()

    print("=" * 60)
    print(f"불량 상위 Lot {worst_lots} vs 하위 Lot {best_lots}")
    print("=" * 60)
    print(f"Feature 차이 상위 {top_n_feats}개 (|mean_diff / pooled_std|):")
    for f in top_feats:
        print(f"  {f:>8}: effect={effect[f]:.4f}  "
              f"worst_mean={worst_data[f].mean():.4f}  best_mean={best_data[f].mean():.4f}")

    # 시각화
    n_rows = (top_n_feats + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 4.5 * n_rows))
    axes = np.array(axes).flatten()

    for i, feat in enumerate(top_feats):
        data = pd.DataFrame({
            "value": pd.concat([worst_data[feat], best_data[feat]], ignore_index=True),
            "group": ["불량 상위 Lot"] * len(worst_data) + ["불량 하위 Lot"] * len(best_data),
        })
        sns.violinplot(x="group", y="value", data=data, ax=axes[i],
                       palette={"불량 상위 Lot": "salmon", "불량 하위 Lot": "skyblue"},
                       inner="quartile", cut=0)
        axes[i].set_title(f"{feat} (effect={effect[feat]:.3f})", fontsize=10)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("불량 상위 vs 하위 Lot: Feature 분포 비교", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()