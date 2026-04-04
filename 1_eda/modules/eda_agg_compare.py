"""
EDA 모듈: Die→Unit 집계 방식별 Target 상관 비교
- 7종 집계 함수(mean, std, min, max, range, median, skew)로 die→unit 변환 후
  각각 target과의 Pearson 상관을 계산하여 어떤 집계가 가장 유용한지 파악
- 전처리 Stage 4 (die→unit 집계) 전략 설계의 핵심 근거
- 노트북에서 import eda_agg_compare as agg 로 사용
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.config import KEY_COL, TARGET_COL


# ─── 집계 함수 정의 ─────────────────────────────────────────
AGG_FUNCS = {
    "mean": "mean",
    "std": "std",
    "min": "min",
    "max": "max",
    "range": lambda x: x.max() - x.min(),
    "median": "median",
    "skew": "skew",
}


def compute_agg_correlations(xs_dict, ys_train, feat_cols):
    """
    7종 집계 방식 각각에 대해 die→unit 변환 후 target과의 Pearson 상관 계산

    반도체 맥락:
    - mean: die 4개의 중심 경향 → 기본 품질 수준
    - std: die 간 편차 → unit 내 불균일성 (공정 균일도 지표)
    - min/max: 최악/최선 die → 극단 die가 unit 불량에 미치는 영향
    - range: max-min → die 간 산포 폭
    - median: 이상치에 강건한 중심 경향
    - skew: die 분포 비대칭 → 특정 die만 이탈했는지

    Parameters
    ----------
    xs_dict : dict  - {"train": DataFrame, ...}
    ys_train : DataFrame  - train Y (ufs_serial, health)
    feat_cols : list of str

    Returns
    -------
    corr_by_agg : dict  - {agg_name: Series(feature→상관계수)}
    summary_df : DataFrame  - feature × agg_method 상관계수 매트릭스
    """
    xs_train = xs_dict["train"]

    # ── 1) groupby 1회로 built-in 집계 일괄 처리 ──
    # skew·range 제외한 5개를 한번에 계산 (skew는 pandas groupby에서 극도로 느림)
    grouped = xs_train.groupby(KEY_COL)[feat_cols]
    multi_agg = grouped.agg(["mean", "std", "min", "max", "median"])

    # ── 2) target을 index 기준으로 정렬 (merge 대신 reindex로 빠르게) ──
    target_series = ys_train.set_index(KEY_COL)[TARGET_COL].reindex(multi_agg.index)
    valid_mask = target_series.notna()
    target_vals = target_series[valid_mask].values

    # ── 3) 각 집계별 상관계수 계산 (numpy vectorized) ──
    corr_by_agg = {}

    # MultiIndex 컬럼 선택: [(feat, agg), ...] 튜플 리스트로 명시적 지정
    for agg_name in ["mean", "std", "min", "max", "median"]:
        cols = [(f, agg_name) for f in feat_cols]
        agg_vals = multi_agg.loc[valid_mask, cols].values  # (n_units, n_feats)
        corr_by_agg[agg_name] = _fast_corrwith(agg_vals, target_vals, feat_cols)

    # ── 4) range = max - min (이미 계산된 값 재활용, groupby 추가 없음) ──
    max_cols = [(f, "max") for f in feat_cols]
    min_cols = [(f, "min") for f in feat_cols]
    range_vals = (
        multi_agg.loc[valid_mask, max_cols].values
        - multi_agg.loc[valid_mask, min_cols].values
    )
    corr_by_agg["range"] = _fast_corrwith(range_vals, target_vals, feat_cols)

    # ── 5) skew: numpy로 직접 계산 (pandas groupby.skew 대비 ~10x 빠름) ──
    # 각 unit의 die 4개에 대한 skew = n*m3/m2^1.5 (biased estimator, scipy 호환)
    mean_cols = [(f, "mean") for f in feat_cols]
    mean_vals_full = multi_agg.loc[:, mean_cols]  # unit × feat
    # die-level에서 각 unit의 mean을 빼고 3차/2차 모멘트 계산
    keys = xs_train[KEY_COL].values
    feat_matrix = xs_train[feat_cols].values  # (n_dies, n_feats)

    # unit index mapping
    unique_keys = multi_agg.index.values
    key_to_idx = {k: i for i, k in enumerate(unique_keys)}
    die_unit_idx = np.array([key_to_idx[k] for k in keys])

    unit_means = mean_vals_full.values  # (n_units, n_feats)
    deviations = feat_matrix - unit_means[die_unit_idx]  # (n_dies, n_feats)

    # m2, m3 by unit (sum then divide by n)
    n_per_unit = np.bincount(die_unit_idx).astype(np.float64)  # (n_units,)
    m2 = np.zeros_like(unit_means)
    m3 = np.zeros_like(unit_means)
    np.add.at(m2, die_unit_idx, deviations ** 2)
    np.add.at(m3, die_unit_idx, deviations ** 3)
    m2 /= n_per_unit[:, None]
    m3 /= n_per_unit[:, None]

    with np.errstate(divide="ignore", invalid="ignore"):
        skew_vals = np.where(m2 > 0, m3 / (m2 ** 1.5), 0.0)

    skew_valid = skew_vals[valid_mask.values]
    corr_by_agg["skew"] = _fast_corrwith(skew_valid, target_vals, feat_cols)

    # ── 6) feature × agg 매트릭스 구성 ──
    # AGG_FUNCS 순서 유지
    corr_by_agg = {k: corr_by_agg[k] for k in AGG_FUNCS.keys()}
    summary_df = pd.DataFrame(corr_by_agg)
    summary_df.index.name = "feature"

    return corr_by_agg, summary_df


def _fast_corrwith(X, y, feat_cols):
    """
    numpy vectorized Pearson correlation: 각 컬럼과 y의 상관계수
    X: (n, p) array, y: (n,) array → Series(feat_cols → r)
    NaN이 포함된 컬럼은 해당 행을 제외하고 per-column으로 계산
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_feats = X.shape[1]
    r = np.full(n_feats, np.nan)

    # NaN이 전혀 없으면 vectorized 계산
    has_nan = np.isnan(X).any()
    if not has_nan:
        X_centered = X - X.mean(axis=0, keepdims=True)
        y_centered = y - y.mean()
        numerator = (X_centered * y_centered[:, None]).sum(axis=0)
        denom_x = np.sqrt((X_centered ** 2).sum(axis=0))
        denom_y = np.sqrt((y_centered ** 2).sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(denom_x > 0, numerator / (denom_x * denom_y), np.nan)
    else:
        # NaN이 있으면 컬럼별로 유효 행만 사용
        y_mean = np.nanmean(y)
        for j in range(n_feats):
            col = X[:, j]
            valid = ~(np.isnan(col) | np.isnan(y))
            if valid.sum() < 3:
                continue
            xv, yv = col[valid], y[valid]
            xc = xv - xv.mean()
            yc = yv - yv.mean()
            dx = np.sqrt((xc ** 2).sum())
            dy = np.sqrt((yc ** 2).sum())
            if dx > 0 and dy > 0:
                r[j] = (xc * yc).sum() / (dx * dy)

    return pd.Series(r, index=feat_cols)


def print_agg_summary(corr_by_agg, summary_df):
    """
    집계 방식별 상관계수 요약 통계 출력

    Parameters
    ----------
    corr_by_agg : dict
    summary_df : DataFrame
    """
    print("=" * 70)
    print("집계 방식별 Target 상관계수 요약")
    print("=" * 70)
    print(f"  {'집계':>8}  {'max|r|':>8}  {'mean|r|':>9}  {'|r|>0.03':>9}  {'|r|>0.05':>9}  {'top1 feature':>14}")
    print("-" * 70)

    for agg_name in AGG_FUNCS.keys():
        corr = corr_by_agg[agg_name]
        abs_corr = corr.abs()
        max_r = abs_corr.max()
        mean_r = abs_corr.mean()
        n_03 = (abs_corr > 0.03).sum()
        n_05 = (abs_corr > 0.05).sum()
        top_feat = abs_corr.idxmax()
        print(f"  {agg_name:>8}  {max_r:>8.4f}  {mean_r:>9.4f}  {n_03:>9d}  {n_05:>9d}  {top_feat:>14}")

    # 각 feature별로 어떤 집계가 가장 높은 상관을 주는지
    best_agg = summary_df.abs().idxmax(axis=1)
    best_agg_counts = best_agg.value_counts()

    print()
    print("Feature별 최적 집계 방식 분포 (|r| 최대인 집계):")
    for agg_name, count in best_agg_counts.items():
        print(f"  {agg_name:>8}: {count:,}개 ({count/len(best_agg)*100:.1f}%)")

    # mean 대비 다른 집계가 더 나은 feature 수
    mean_abs = summary_df["mean"].abs()
    for agg_name in ["std", "range", "min", "max", "skew"]:
        other_abs = summary_df[agg_name].abs()
        n_better = (other_abs > mean_abs).sum()
        print(f"\n  {agg_name}이 mean보다 |r|이 높은 feature: {n_better:,}개 ({n_better/len(mean_abs)*100:.1f}%)")


def plot_agg_comparison(corr_by_agg, summary_df):
    """
    집계 방식별 상관계수 분포 비교 시각화
    1) 집계별 |r| 분포 boxplot
    2) 집계별 |r| 분포 violin plot
    3) 상위 20개 feature의 집계별 상관 히트맵

    Parameters
    ----------
    corr_by_agg : dict
    summary_df : DataFrame
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # ── 1) 집계별 |r| boxplot ──
    abs_data = {name: corr.abs().values for name, corr in corr_by_agg.items()}
    box_df = pd.DataFrame({k: pd.Series(v) for k, v in abs_data.items()})
    box_df_melted = box_df.melt(var_name="집계 방식", value_name="|r|")

    sns.boxplot(x="집계 방식", y="|r|", data=box_df_melted, ax=axes[0],
                palette="Set2", fliersize=2)
    axes[0].set_title("집계 방식별 |Pearson r| 분포")
    axes[0].set_ylabel("|r| with target")
    axes[0].tick_params(axis="x", rotation=30)

    # ── 2) 집계별 |r| violin plot ──
    sns.violinplot(x="집계 방식", y="|r|", data=box_df_melted, ax=axes[1],
                   palette="Set2", inner="quartile", cut=0)
    axes[1].set_title("집계 방식별 |Pearson r| Violin")
    axes[1].set_ylabel("|r| with target")
    axes[1].tick_params(axis="x", rotation=30)

    # ── 3) 상위 20개 feature 히트맵 ──
    # 모든 집계 통틀어 |r| 최대값이 큰 feature 20개
    max_abs_per_feat = summary_df.abs().max(axis=1).sort_values(ascending=False)
    top20_feats = max_abs_per_feat.head(20).index.tolist()

    heatmap_data = summary_df.loc[top20_feats]
    sns.heatmap(heatmap_data, cmap="RdBu_r", center=0, annot=True, fmt=".3f",
                linewidths=0.5, ax=axes[2], cbar_kws={"label": "Pearson r"})
    axes[2].set_title("상위 20 Feature × 집계 방식 상관 히트맵")
    axes[2].set_ylabel("")
    axes[2].tick_params(axis="y", labelsize=9)

    plt.tight_layout()
    plt.show()


def plot_mean_vs_others(summary_df):
    """
    mean 집계 |r| vs 다른 집계 |r| scatter plot
    대각선 위 = 해당 집계가 mean보다 높은 상관, 아래 = mean이 더 좋음

    Parameters
    ----------
    summary_df : DataFrame
    """
    others = ["std", "min", "max", "range", "skew"]
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))

    mean_abs = summary_df["mean"].abs()

    for i, agg_name in enumerate(others):
        other_abs = summary_df[agg_name].abs()

        # 유효한 (NaN 아닌) 값만으로 scatter
        valid = mean_abs.notna() & other_abs.notna()
        mx, ox = mean_abs[valid], other_abs[valid]
        axes[i].scatter(mx, ox, alpha=0.3, s=8, color="steelblue")

        # 대각선 (유효값이 없거나 모두 0이면 기본 범위 사용)
        raw_lim = max(mx.max(), ox.max()) if len(mx) > 0 else 0.0
        lim = raw_lim * 1.05 if raw_lim > 0 and np.isfinite(raw_lim) else 0.05
        axes[i].plot([0, lim], [0, lim], "r--", alpha=0.5, linewidth=1)

        # 대각선 위 비율
        n_better = (ox > mx).sum()
        pct = n_better / len(mx) * 100 if len(mx) > 0 else 0.0

        axes[i].set_xlabel("mean |r|")
        axes[i].set_ylabel(f"{agg_name} |r|")
        axes[i].set_title(f"mean vs {agg_name}\n({agg_name} 우위: {pct:.0f}%)")
        axes[i].set_xlim(0, lim)
        axes[i].set_ylim(0, lim)
        axes[i].set_aspect("equal")

    plt.suptitle("mean 집계 대비 다른 집계의 Target 상관 비교", fontsize=13, y=1.03)
    plt.tight_layout()
    plt.show()


def find_best_agg_features(summary_df, n=15):
    """
    mean보다 다른 집계가 뚜렷이 더 나은 feature 탐색 및 출력
    → 전처리에서 mean 외 집계를 반드시 포함해야 할 feature 목록

    Parameters
    ----------
    summary_df : DataFrame
    n : int  - 출력할 feature 수

    Returns
    -------
    gain_df : DataFrame  - feature별 최적 집계, mean 대비 |r| 향상량
    """
    abs_df = summary_df.abs()
    mean_abs = abs_df["mean"]

    records = []
    for feat in summary_df.index:
        row = abs_df.loc[feat]
        best_agg = row.idxmax()
        best_r = row.max()
        mean_r = mean_abs[feat]
        gain = best_r - mean_r

        records.append({
            "feature": feat,
            "best_agg": best_agg,
            "best_r": best_r,
            "mean_r": mean_r,
            "gain_over_mean": gain,
        })

    gain_df = pd.DataFrame(records)
    # NaN gain은 -inf로 처리하여 정렬 시 맨 뒤로
    gain_df = gain_df.sort_values("gain_over_mean", ascending=False, na_position="last").reset_index(drop=True)

    # mean이 아닌 최적 집계인 feature만 (NaN 제외)
    non_mean = gain_df[(gain_df["best_agg"] != "mean") & gain_df["gain_over_mean"].notna()]

    print("=" * 70)
    print(f"mean 외 집계가 더 좋은 상위 {n}개 Feature")
    print("=" * 70)
    if len(non_mean) == 0:
        print("  (유효한 결과 없음 — mean 상관계수가 모두 NaN이면 비교 불가)")
    else:
        print(f"  {'Feature':>8}  {'최적집계':>8}  {'best|r|':>8}  {'mean|r|':>8}  {'향상':>8}")
        print("-" * 55)
        for _, row in non_mean.head(n).iterrows():
            print(f"  {row['feature']:>8}  {row['best_agg']:>8}  {row['best_r']:>8.4f}  "
                  f"{row['mean_r']:>8.4f}  {row['gain_over_mean']:>+8.4f}")

    return gain_df
