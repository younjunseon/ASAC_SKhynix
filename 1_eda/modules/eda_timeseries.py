"""
EDA 모듈: 생산 순서별 불량 패턴 (시계열 분석)
- X1086(생산날짜) → run_id(로트) → wafer_no(웨이퍼) → ufs_serial 순 정렬
- 생산 시점에 따른 health 변동 패턴 시각화
- 불량이 점진 증가 → 급락하는 반복 패턴 확인
- 노트북에서 import eda_timeseries as tss 로 사용
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.config import KEY_COL, TARGET_COL, DIE_KEY_COL


def _prepare_timeseries(xs, ys_train):
    """
    die→unit 집계 + 생산 순서 정렬

    Parameters
    ----------
    xs : DataFrame  (전체 die-level, split 통합)
    ys_train : DataFrame

    Returns
    -------
    unit_df : DataFrame  (unit-level, 생산순 정렬, prod_order 포함)
    """
    xs_tmp = xs.copy()
    _split = xs_tmp[DIE_KEY_COL].str.split("_", expand=True)
    xs_tmp["run_id"] = _split[0]
    xs_tmp["wafer_no"] = _split[1].astype(int)

    # X1086 → 생산일자
    xs_tmp["prod_date"] = xs_tmp["X1086"]

    # die → unit 집계 (정렬용 메타 정보)
    unit_meta = xs_tmp.groupby(KEY_COL).agg(
        prod_date=("prod_date", lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
        run_id=("run_id", "first"),
        wafer_no=("wafer_no", "first"),
    ).reset_index()

    # train Y와 병합
    unit_df = unit_meta.merge(ys_train, on=KEY_COL, how="inner")

    # 생산순 정렬: 날짜 → 로트 → 웨이퍼 → 시리얼
    unit_df = unit_df.sort_values(
        ["prod_date", "run_id", "wafer_no", KEY_COL]
    ).reset_index(drop=True)

    unit_df["prod_order"] = range(len(unit_df))

    print(f"Train units: {len(unit_df):,}")
    print(f"Y > 0: {(unit_df[TARGET_COL] > 0).sum():,} "
          f"({(unit_df[TARGET_COL] > 0).mean() * 100:.1f}%)")
    print(f"\n생산일자 분포:")
    print(unit_df["prod_date"].value_counts().sort_index())
    print(f"\n로트(run_id) 수: {unit_df['run_id'].nunique()}")
    print(f"일자별 로트 수:")
    print(unit_df.groupby("prod_date")["run_id"].nunique())

    return unit_df


def plot_timeseries_scatter(unit_df, upper_quantile=0.99):
    """
    생산 순서별 health scatter (전체 데이터, 극단값만 제거)

    상단: scatter (Y=0 포함, 상위 1% 제거) + rolling mean
    하단: 구간별 불량 발생 비율 (Y > 0 비율)

    Parameters
    ----------
    unit_df : DataFrame  (_prepare_timeseries 반환값)
    upper_quantile : float  (극단값 제거 기준, 기본 99%)
    """
    # 극단 이상치 제거 (상위 1% — health=1.0 등)
    upper_cut = unit_df[TARGET_COL].quantile(upper_quantile)
    df_clean = unit_df[unit_df[TARGET_COL] <= upper_cut].copy()

    n_removed = len(unit_df) - len(df_clean)
    print(f"전체 unit: {len(unit_df):,}")
    print(f"상위 {(1 - upper_quantile) * 100:.0f}% 기준값: {upper_cut:.4f}")
    print(f"극단값 제거 후: {len(df_clean):,} ({n_removed}개 제거)")

    # ─── Scatter Plot ───
    fig, axes = plt.subplots(2, 1, figsize=(18, 10),
                             gridspec_kw={"height_ratios": [3, 1]})

    # (상) Scatter: 생산순서 vs health (극단값 제거)
    ax1 = axes[0]
    scatter = ax1.scatter(
        df_clean["prod_order"], df_clean[TARGET_COL],
        c=df_clean["prod_date"], cmap="tab10",
        s=8, alpha=0.4, edgecolors="none",
    )

    # rolling mean (Y>0만으로 추세선)
    pos_sorted = df_clean[df_clean[TARGET_COL] > 0].sort_values("prod_order")
    rolling_mean = pos_sorted[TARGET_COL].rolling(
        window=200, min_periods=50, center=True
    ).mean()
    ax1.plot(pos_sorted["prod_order"], rolling_mean,
             color="red", linewidth=2, label="Rolling Mean (Y>0, w=200)")

    # 날짜 경계선
    date_boundaries = unit_df.groupby("prod_date")["prod_order"].agg(["min", "max"])
    for date_val, row in date_boundaries.iterrows():
        ax1.axvline(x=row["min"], color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax1.text(
            row["min"] + (row["max"] - row["min"]) / 2, ax1.get_ylim()[0],
            f"{int(date_val)}", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax1.set_xlabel("생산 순서 (날짜 → 로트 → 웨이퍼 → 시리얼)", fontsize=12)
    ax1.set_ylabel(f"{TARGET_COL} (상위 {(1 - upper_quantile) * 100:.0f}% 제거)", fontsize=12)
    ax1.set_title("생산 순서별 health — 극단값 제거", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(axis="y", alpha=0.3)

    # (하) Line: 구간별 Y>0 비율
    ax2 = axes[1]
    n_bins = 100
    order_bin = pd.cut(unit_df["prod_order"], bins=n_bins, labels=False)
    defect_rate = unit_df.groupby(order_bin)[TARGET_COL].apply(
        lambda x: (x > 0).mean()
    )
    bin_centers = unit_df.groupby(order_bin)["prod_order"].mean()
    ax2.plot(bin_centers.values, defect_rate.values,
             color="steelblue", linewidth=1.5, alpha=0.8)
    ax2.fill_between(bin_centers.values, defect_rate.values,
                     color="steelblue", alpha=0.15)

    # 날짜 경계선
    for date_val, row in date_boundaries.iterrows():
        ax2.axvline(x=row["min"], color="gray", linestyle="--", alpha=0.5, linewidth=1)

    ax2.set_xlabel("생산 순서", fontsize=12)
    ax2.set_ylabel("Y > 0 비율", fontsize=12)
    ax2.set_title("구간별 불량 발생 비율 (Y > 0 비율)", fontsize=14)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_lot_defect_pattern(unit_df):
    """
    로트별 불량 패턴: 생산순 bar chart + 로트 통계

    Parameters
    ----------
    unit_df : DataFrame  (_prepare_timeseries 반환값)
    """
    # 로트별 통계
    lot_stats = unit_df.groupby("run_id").agg(
        n_units=(TARGET_COL, "count"),
        defect_rate=(TARGET_COL, lambda x: (x > 0).mean()),
        mean_health=(TARGET_COL, "mean"),
        max_health=(TARGET_COL, "max"),
        prod_date=("prod_date", "first"),
        order_start=("prod_order", "min"),
    ).sort_values("order_start")

    print(f"로트 수: {len(lot_stats)}")
    print(f"\n로트별 통계 (생산순):")
    print(lot_stats.to_string())

    # ─── 로트별 불량률 bar chart (생산순) ───
    fig, ax = plt.subplots(figsize=(18, 5))

    colors = plt.cm.tab10(
        lot_stats["prod_date"].rank(method="dense").astype(int) - 1
    )
    ax.bar(range(len(lot_stats)), lot_stats["defect_rate"],
           color=colors, alpha=0.8, edgecolor="white")

    ax.set_xticks(range(len(lot_stats)))
    ax.set_xticklabels(lot_stats.index, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("로트 (생산순)", fontsize=12)
    ax.set_ylabel("불량 발생률 (Y > 0 비율)", fontsize=12)
    ax.set_title("로트별 불량 발생률 — 생산 순서대로 (색상 = 날짜)", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    # 날짜 레이블
    for date_val in lot_stats["prod_date"].unique():
        mask = lot_stats["prod_date"] == date_val
        idx = [i for i, m in enumerate(mask) if m]
        mid = idx[len(idx) // 2]
        ax.text(mid, ax.get_ylim()[1] * 0.95, f"{int(date_val)}",
                ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.show()