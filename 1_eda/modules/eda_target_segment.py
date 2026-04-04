"""
EDA 모듈: Target Segmentation 분석
- Health 값 구간별(segment) feature 분포 차이 분석
- ANOVA/Kruskal-Wallis 기반 segment 간 feature 차별력 평가
- 극단 불량 unit(상위 top_pct%)의 feature 프로파일 분석
- PCA를 통한 segment별 분리 가능성 시각화
- 노트북에서 import eda_target_segment as ts 로 사용
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils.config import KEY_COL, TARGET_COL, SEED


# ═══════════════════════════════════════════════════════════════
# Private Helper
# ═══════════════════════════════════════════════════════════════

def _prepare_data(xs_dict, ys_train, feat_cols):
    """
    Train die-level → unit-level 평균 집계 후 target merge

    Parameters
    ----------
    xs_dict : dict
        {"train": DataFrame, ...} split별 die-level 데이터
    ys_train : DataFrame
        train Y 데이터 (KEY_COL, TARGET_COL 컬럼 필요)
    feat_cols : list of str
        feature 컬럼명 리스트 (X0~X1086)

    Returns
    -------
    merged : DataFrame
        unit-level로 집계된 X features + target (health) 합본
    """
    xs_train = xs_dict["train"]
    xs_unit = xs_dict['train_unit_mean'] if 'train_unit_mean' in xs_dict else xs_train.groupby(KEY_COL)[feat_cols].mean()
    merged = xs_unit.merge(ys_train, left_index=True, right_on=KEY_COL, how="inner")
    print(f"Die→Unit 집계 완료: {len(xs_train):,} dies → {len(merged):,} units")
    return merged


# ═══════════════════════════════════════════════════════════════
# 1. Segment Overview
# ═══════════════════════════════════════════════════════════════

def segment_overview(xs_dict, ys_train, feat_cols):
    """
    Target 값을 4개 segment로 나누어 기초통계 및 분포 시각화

    Segments:
        - Y=0         : health가 정확히 0인 unit
        - Y∈(0, Q50]  : health > 0 중 하위 50% (low)
        - Y∈(Q50, Q90]: health > 0 중 50~90% (mid)
        - Y>Q90       : health > 0 중 상위 10% (high)
    Q50, Q90은 Y>0 서브셋 기준 분위수

    Parameters
    ----------
    xs_dict : dict
        {"train": DataFrame, ...} split별 die-level 데이터
    ys_train : DataFrame
        train Y 데이터 (KEY_COL, TARGET_COL 컬럼 필요)
    feat_cols : list of str
        feature 컬럼명 리스트

    Returns
    -------
    merged : DataFrame
        unit-level X+Y + "segment" 컬럼 추가
    segment_stats : DataFrame
        segment별 count, mean, std, min, median, max 통계
    """
    merged = _prepare_data(xs_dict, ys_train, feat_cols)

    # ── segment 구간 설정 (Y>0 서브셋 기준 분위수) ──
    pos_vals = merged.loc[merged[TARGET_COL] > 0, TARGET_COL]
    q50 = pos_vals.quantile(0.50)
    q90 = pos_vals.quantile(0.90)

    print(f"\nSegment 기준 (Y>0 분위수):")
    print(f"  Q50 = {q50:.6f}")
    print(f"  Q90 = {q90:.6f}")

    # ── segment 라벨 부여 ──
    conditions = [
        merged[TARGET_COL] == 0,
        (merged[TARGET_COL] > 0) & (merged[TARGET_COL] <= q50),
        (merged[TARGET_COL] > q50) & (merged[TARGET_COL] <= q90),
        merged[TARGET_COL] > q90,
    ]
    labels = ["Y=0", "Low (0,Q50]", "Mid (Q50,Q90]", "High >Q90"]
    merged["segment"] = np.select(conditions, labels, default="Unknown")

    # ── segment별 통계 ──
    segment_stats = merged.groupby("segment")[TARGET_COL].agg(
        ["count", "mean", "std", "min", "median", "max"]
    ).reindex(labels)
    segment_stats["pct"] = (segment_stats["count"] / len(merged) * 100).round(1)

    print("\n" + "=" * 70)
    print("Segment별 Health 통계")
    print("=" * 70)
    print(segment_stats.to_string())
    print()

    # ── 시각화 ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # 1) Segment 크기 bar chart
    colors_bar = ["#4A90D9", "#7BC67E", "#F5A623", "#D0021B"]
    axes[0].bar(labels, segment_stats["count"].values,
                color=colors_bar, edgecolor="black")
    axes[0].set_title("Segment별 Unit 수")
    axes[0].set_xlabel("Segment")
    axes[0].set_ylabel("Unit 수")
    for i, (lbl, cnt) in enumerate(zip(labels, segment_stats["count"].values)):
        axes[0].text(i, cnt + len(merged) * 0.005, f"{cnt:,.0f}\n({segment_stats['pct'].values[i]}%)",
                     ha="center", va="bottom", fontsize=9)

    # 2) Segment별 health boxplot (Y=0 제외, 0은 점 하나이므로)
    seg_order = ["Low (0,Q50]", "Mid (Q50,Q90]", "High >Q90"]
    plot_data = merged[merged["segment"] != "Y=0"]

    bp = axes[1].boxplot(
        [plot_data.loc[plot_data["segment"] == seg, TARGET_COL].values for seg in seg_order],
        labels=seg_order, patch_artist=True, showfliers=True,
        flierprops=dict(marker=".", markersize=2, alpha=0.3),
    )
    for patch, color in zip(bp["boxes"], colors_bar[1:]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_title("Segment별 Health 분포 (Y>0)")
    axes[1].set_ylabel("health")

    plt.suptitle("Target Segmentation Overview", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()

    return merged, segment_stats


# ═══════════════════════════════════════════════════════════════
# 2. Segment Feature Comparison (ANOVA / Kruskal-Wallis)
# ═══════════════════════════════════════════════════════════════

def segment_feature_comparison(merged, feat_cols, top_n_feats=12):
    """
    각 feature에 대해 4개 segment 간 분포 차이를 Kruskal-Wallis H-test로 검정

    Kruskal-Wallis는 비모수 검정으로, 정규성 가정이 불필요하여
    zero-inflated + 비대칭 분포에 적합하다.

    Parameters
    ----------
    merged : DataFrame
        segment_overview()에서 반환된 DataFrame ("segment" 컬럼 필요)
    feat_cols : list of str
        feature 컬럼명 리스트
    top_n_feats : int
        시각화 및 출력할 상위 feature 수 (기본 12)

    Returns
    -------
    segment_test_df : DataFrame
        columns = [feature, f_stat, p_value], f_stat 내림차순 정렬
    """
    seg_labels = ["Y=0", "Low (0,Q50]", "Mid (Q50,Q90]", "High >Q90"]
    groups = {seg: merged[merged["segment"] == seg] for seg in seg_labels}

    results = []
    for col in feat_cols:
        # 각 segment별 값 추출 (NaN 제거)
        seg_vals = [groups[seg][col].dropna().values for seg in seg_labels]

        # 모든 그룹에 최소 2개 값이 있어야 검정 가능
        if any(len(v) < 2 for v in seg_vals):
            continue

        # 전체 값이 동일하면 검정 불가 (상수 feature)
        all_vals = np.concatenate(seg_vals)
        if np.ptp(all_vals) == 0:
            continue

        # Kruskal-Wallis H-test (비모수 ANOVA)
        h_stat, p_val = stats.kruskal(*seg_vals)
        results.append({
            "feature": col,
            "f_stat": h_stat,
            "p_value": p_val,
        })

    segment_test_df = pd.DataFrame(results)
    segment_test_df = segment_test_df.sort_values("f_stat", ascending=False).reset_index(drop=True)

    # ── 요약 출력 ──
    n_sig_001 = (segment_test_df["p_value"] < 0.001).sum()
    n_sig_005 = (segment_test_df["p_value"] < 0.05).sum()

    print("=" * 60)
    print("Kruskal-Wallis H-test: Segment 간 Feature 차이 검정")
    print("=" * 60)
    print(f"검정 대상 feature: {len(segment_test_df):,}개")
    print(f"유의 (p<0.05) : {n_sig_005:,}개 ({n_sig_005/len(segment_test_df)*100:.1f}%)")
    print(f"유의 (p<0.001): {n_sig_001:,}개 ({n_sig_001/len(segment_test_df)*100:.1f}%)")
    print()

    print(f"H-statistic 상위 {top_n_feats}개 Feature:")
    print(f"  {'Feature':>8}  {'H-stat':>10}  {'p-value':>12}")
    print("-" * 40)
    for _, row in segment_test_df.head(top_n_feats).iterrows():
        print(f"  {row['feature']:>8}  {row['f_stat']:>10.2f}  {row['p_value']:>12.2e}")

    # ── 시각화: 상위 feature들의 violin plot ──
    top_feats = segment_test_df.head(top_n_feats)["feature"].tolist()
    n_rows = (top_n_feats + 3) // 4
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 4.5 * n_rows))
    axes = np.array(axes).flatten()

    palette = {"Y=0": "#4A90D9", "Low (0,Q50]": "#7BC67E",
               "Mid (Q50,Q90]": "#F5A623", "High >Q90": "#D0021B"}

    for i, feat in enumerate(top_feats):
        # 시각화 속도를 위한 샘플링 (segment별 최대 2000)
        plot_rows = []
        for seg in seg_labels:
            seg_data = merged.loc[merged["segment"] == seg, [feat, "segment"]].dropna(subset=[feat])
            if len(seg_data) > 2000:
                seg_data = seg_data.sample(2000, random_state=SEED)
            plot_rows.append(seg_data)
        plot_df = pd.concat(plot_rows, ignore_index=True)

        sns.violinplot(
            x="segment", y=feat, data=plot_df, ax=axes[i],
            order=seg_labels, palette=palette,
            inner="quartile", cut=0, scale="width",
        )

        h_val = segment_test_df.loc[segment_test_df["feature"] == feat, "f_stat"].values[0]
        axes[i].set_title(f"{feat} (H={h_val:.1f})", fontsize=10)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        axes[i].tick_params(axis="x", rotation=30, labelsize=8)

    for j in range(len(top_feats), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Segment 간 Feature 분포 비교 (H-stat 상위 {top_n_feats}개)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()

    return segment_test_df


# ═══════════════════════════════════════════════════════════════
# 3. Extreme Profile Analysis
# ═══════════════════════════════════════════════════════════════

def extreme_profile(xs_dict, ys_train, feat_cols, top_pct=0.01, n_feats=15):
    """
    극단 불량 unit(상위 top_pct% health) vs 나머지의 feature 프로파일 비교

    Cohen's d로 두 그룹 간 각 feature의 효과 크기를 측정하고,
    표준화된 평균값을 bar chart로 비교하여 극단 불량의 패턴을 파악한다.

    Parameters
    ----------
    xs_dict : dict
        {"train": DataFrame, ...} split별 die-level 데이터
    ys_train : DataFrame
        train Y 데이터 (KEY_COL, TARGET_COL 컬럼 필요)
    feat_cols : list of str
        feature 컬럼명 리스트
    top_pct : float
        극단 불량 기준 상위 비율 (기본 0.01 = 상위 1%)
    n_feats : int
        출력 및 시각화할 상위 feature 수 (기본 15)

    Returns
    -------
    extreme_df : DataFrame
        columns = [feature, cohens_d, mean_extreme, mean_rest]
        |cohens_d| 내림차순 정렬
    """
    merged = _prepare_data(xs_dict, ys_train, feat_cols)

    # ── 극단 불량 그룹 정의 ──
    threshold = merged[TARGET_COL].quantile(1.0 - top_pct)
    grp_extreme = merged[merged[TARGET_COL] >= threshold]
    grp_rest = merged[merged[TARGET_COL] < threshold]

    print(f"\n극단 불량 기준: 상위 {top_pct*100:.1f}% (health >= {threshold:.6f})")
    print(f"극단 그룹: {len(grp_extreme):,} units")
    print(f"나머지   : {len(grp_rest):,} units")
    print(f"극단 그룹 health: mean={grp_extreme[TARGET_COL].mean():.6f}, "
          f"median={grp_extreme[TARGET_COL].median():.6f}")
    print()

    # ── 각 feature별 Cohen's d 계산 ──
    results = []
    for col in feat_cols:
        vals_ext = grp_extreme[col].dropna().values
        vals_rest = grp_rest[col].dropna().values

        if len(vals_ext) < 2 or len(vals_rest) < 2:
            continue

        n_e, n_r = len(vals_ext), len(vals_rest)
        m_e, m_r = vals_ext.mean(), vals_rest.mean()
        s_e, s_r = vals_ext.std(ddof=1), vals_rest.std(ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n_e - 1) * s_e**2 + (n_r - 1) * s_r**2) / (n_e + n_r - 2))

        if pooled_std > 0:
            d = (m_e - m_r) / pooled_std
        else:
            d = 0.0

        results.append({
            "feature": col,
            "cohens_d": d,
            "mean_extreme": m_e,
            "mean_rest": m_r,
        })

    extreme_df = pd.DataFrame(results)
    extreme_df["abs_d"] = extreme_df["cohens_d"].abs()
    extreme_df = extreme_df.sort_values("abs_d", ascending=False).reset_index(drop=True)

    # ── 요약 출력 ──
    n_large = (extreme_df["abs_d"] >= 0.8).sum()
    n_medium = (extreme_df["abs_d"] >= 0.5).sum()
    n_small = (extreme_df["abs_d"] >= 0.2).sum()

    print("=" * 60)
    print(f"극단 불량 (상위 {top_pct*100:.1f}%) vs 나머지: Cohen's d 요약")
    print("=" * 60)
    print(f"검정 대상 feature: {len(extreme_df):,}개")
    print(f"|d| >= 0.2 (small) : {n_small:,}개")
    print(f"|d| >= 0.5 (medium): {n_medium:,}개")
    print(f"|d| >= 0.8 (large) : {n_large:,}개")
    print()

    print(f"|Cohen's d| 상위 {n_feats}개 Feature:")
    print(f"  {'Feature':>8}  {'Cohen d':>9}  {'mean(극단)':>12}  {'mean(나머지)':>12}")
    print("-" * 55)
    for _, row in extreme_df.head(n_feats).iterrows():
        print(f"  {row['feature']:>8}  {row['cohens_d']:>+9.4f}  "
              f"{row['mean_extreme']:>12.4f}  {row['mean_rest']:>12.4f}")

    # ── 시각화: 표준화된 평균 비교 bar chart ──
    top_df = extreme_df.head(n_feats).copy()

    # 전체 데이터 기준 표준화 (mean=0, std=1)
    all_feats_data = merged[top_df["feature"].tolist()]
    feat_mean = all_feats_data.mean()
    feat_std = all_feats_data.std()
    feat_std = feat_std.replace(0, 1)  # 0-division 방지

    top_df["z_extreme"] = top_df.apply(
        lambda r: (r["mean_extreme"] - feat_mean[r["feature"]]) / feat_std[r["feature"]], axis=1
    )
    top_df["z_rest"] = top_df.apply(
        lambda r: (r["mean_rest"] - feat_mean[r["feature"]]) / feat_std[r["feature"]], axis=1
    )

    fig, ax = plt.subplots(figsize=(12, max(6, n_feats * 0.45)))

    y_pos = np.arange(n_feats)
    bar_height = 0.35

    ax.barh(y_pos - bar_height / 2, top_df["z_extreme"].values,
            height=bar_height, color="#D0021B", edgecolor="black",
            alpha=0.8, label=f"Extreme (top {top_pct*100:.1f}%)")
    ax.barh(y_pos + bar_height / 2, top_df["z_rest"].values,
            height=bar_height, color="#4A90D9", edgecolor="black",
            alpha=0.8, label="Rest")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_df["feature"].values, fontsize=9)
    ax.set_xlabel("Standardized Mean (z-score)")
    ax.set_title(f"Extreme (top {top_pct*100:.1f}%) vs Rest: 표준화 평균 비교 (|d| 상위 {n_feats}개)")
    ax.axvline(x=0, color="gray", linestyle="-", alpha=0.4)
    ax.legend(loc="lower right", fontsize=10)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.show()

    return extreme_df.drop(columns=["abs_d"])


# ═══════════════════════════════════════════════════════════════
# 4. Segment PCA Visualization
# ═══════════════════════════════════════════════════════════════

def plot_segment_pca(merged, feat_cols, n_components=2):
    """
    PCA 차원축소 후 segment별 색 구분 scatter plot

    StandardScaler + NaN median imputation을 적용한 뒤,
    PCA로 n_components 차원으로 축소하여 시각화한다.

    Parameters
    ----------
    merged : DataFrame
        segment_overview()에서 반환된 DataFrame ("segment" 컬럼 필요)
    feat_cols : list of str
        feature 컬럼명 리스트
    n_components : int
        PCA 차원 수 (기본 2)
    """
    df = merged.copy()

    # ── NaN median imputation ──
    for col in feat_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # ── 분산 0인 feature 제외 ──
    valid_feats = [c for c in feat_cols if df[c].std() > 0]
    print(f"PCA 입력: {len(valid_feats):,}개 feature (분산>0)")

    # ── StandardScaler + PCA ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[valid_feats].values)

    pca = PCA(n_components=min(n_components, len(valid_feats)), random_state=SEED)
    X_pca = pca.fit_transform(X_scaled)

    var_explained = pca.explained_variance_ratio_
    print(f"PC1: {var_explained[0]*100:.2f}%, PC2: {var_explained[1]*100:.2f}% "
          f"(합: {sum(var_explained)*100:.2f}%)")

    # ── Scatter plot (segment별 색 구분) ──
    seg_labels = ["Y=0", "Low (0,Q50]", "Mid (Q50,Q90]", "High >Q90"]
    seg_colors = {"Y=0": "#4A90D9", "Low (0,Q50]": "#7BC67E",
                  "Mid (Q50,Q90]": "#F5A623", "High >Q90": "#D0021B"}
    seg_alphas = {"Y=0": 0.08, "Low (0,Q50]": 0.15,
                  "Mid (Q50,Q90]": 0.25, "High >Q90": 0.5}
    seg_sizes = {"Y=0": 2, "Low (0,Q50]": 4,
                 "Mid (Q50,Q90]": 6, "High >Q90": 10}

    fig, ax = plt.subplots(figsize=(12, 9))

    # Y=0을 먼저 그려서 배경으로, 이후 작은 segment를 위에 겹침
    for seg in seg_labels:
        mask = df["segment"].values == seg
        n_seg = mask.sum()
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            alpha=seg_alphas[seg], s=seg_sizes[seg],
            color=seg_colors[seg], label=f"{seg} (n={n_seg:,})",
            edgecolors="none",
        )

    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}%)")
    ax.set_title("PCA: Segment별 분포 (unit-level mean features)")
    ax.legend(markerscale=4, fontsize=10, loc="best", framealpha=0.8)

    plt.tight_layout()
    plt.show()
