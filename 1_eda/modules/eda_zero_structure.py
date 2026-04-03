"""
EDA 모듈: Y=0 내부 이질성 분석
- Zero-Inflated 이론(Lambert 1992): 구조적 zero vs 우연적 zero
- Y=0 unit 내에서 WT feature 기반 클러스터링으로 하위 그룹 탐색
- 클러스터별 feature 프로파일 비교 → 잠재 불량 unit 식별 가능성 확인
- Two-Stage 모델 Stage 1(분류) 설계 근거
- 노트북에서 import eda_zero_structure as zs 로 사용
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from utils.config import KEY_COL, TARGET_COL, SEED


def _prepare_zero_data(xs_dict, ys_train, feat_cols):
    """
    Y=0 unit만 추출: die→unit mean 집계 + NaN imputation + 분산 0 제외

    Returns
    -------
    zero_df : DataFrame  (Y=0 unit의 feature)
    valid_feats : list  (분산 > 0인 feature)
    """
    xs_train = xs_dict["train"]
    xs_unit = xs_train.groupby(KEY_COL)[feat_cols].mean()
    merged = xs_unit.merge(ys_train, left_index=True, right_on=KEY_COL, how="inner")

    zero_df = merged[merged[TARGET_COL] == 0].copy()

    # NaN median imputation
    for col in feat_cols:
        if zero_df[col].isna().any():
            zero_df[col] = zero_df[col].fillna(zero_df[col].median())

    # 분산 0 feature 제외
    valid_feats = [c for c in feat_cols if zero_df[c].std() > 0]

    print(f"Y=0 unit: {len(zero_df):,}개, 유효 feature: {len(valid_feats):,}개")

    return zero_df, valid_feats


def zero_cluster_analysis(xs_dict, ys_train, feat_cols, n_clusters=3):
    """
    Y=0 unit에 대해 K-Means 클러스터링 → 하위 그룹 발견

    ZIP 이론 배경:
    - Y=0 중 일부는 "진짜 양품(구조적 zero)" → 결함 불가
    - Y=0 중 일부는 "잠재 불량(우연적 zero)" → 아직 드러나지 않았을 뿐
    - 클러스터링으로 이 두 그룹을 분리할 수 있다면 Two-Stage Stage 1에 유리

    Parameters
    ----------
    xs_dict : dict
    ys_train : DataFrame
    feat_cols : list of str
    n_clusters : int  - 클러스터 수 (기본 3)

    Returns
    -------
    zero_df : DataFrame  ('cluster' 컬럼 추가)
    cluster_stats_df : DataFrame  (클러스터별 통계)
    valid_feats : list
    """
    zero_df, valid_feats = _prepare_zero_data(xs_dict, ys_train, feat_cols)

    # 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(zero_df[valid_feats].values)

    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    zero_df["cluster"] = kmeans.fit_predict(X_scaled)

    # Silhouette score (샘플링)
    n_sample = min(5000, len(X_scaled))
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(X_scaled), n_sample, replace=False)
    sil = silhouette_score(X_scaled[idx], zero_df["cluster"].values[idx])

    # 클러스터별 통계
    cluster_stats = zero_df.groupby("cluster").agg(
        n_units=(TARGET_COL, "size"),
    ).reset_index()
    cluster_stats["pct"] = (cluster_stats["n_units"] / len(zero_df) * 100).round(1)

    print("=" * 60)
    print(f"Y=0 클러스터링 결과 (K={n_clusters})")
    print("=" * 60)
    print(f"Silhouette Score: {sil:.4f}")
    print()
    for _, row in cluster_stats.iterrows():
        print(f"  Cluster {int(row['cluster'])}: {int(row['n_units']):,} units ({row['pct']}%)")

    # ANOVA로 클러스터 간 차이 큰 feature 찾기
    f_results = []
    for col in valid_feats:
        groups = [zero_df.loc[zero_df["cluster"] == c, col].values
                  for c in range(n_clusters)]
        if all(len(g) > 1 for g in groups):
            f_stat, p_val = sp_stats.f_oneway(*groups)
            f_results.append({"feature": col, "f_stat": f_stat, "p_value": p_val})

    f_df = pd.DataFrame(f_results).sort_values("f_stat", ascending=False)
    cluster_stats_df = f_df.reset_index(drop=True)

    n_sig = (cluster_stats_df["p_value"] < 0.001).sum()
    print(f"\nANOVA p<0.001인 feature: {n_sig:,}개 "
          f"({n_sig/len(cluster_stats_df)*100:.1f}%)")

    print(f"\n클러스터 간 차이 큰 상위 10개 Feature:")
    print(f"  {'Feature':>8}  {'F-stat':>10}  {'p-value':>12}")
    print("-" * 40)
    for _, row in cluster_stats_df.head(10).iterrows():
        print(f"  {row['feature']:>8}  {row['f_stat']:>10.1f}  {row['p_value']:>12.2e}")

    return zero_df, cluster_stats_df, valid_feats


def plot_zero_clusters(zero_df, feat_cols, cluster_stats_df, n_feats=6):
    """
    Y=0 클러스터 시각화: PCA scatter + 클러스터 크기 + 상위 feature violin

    Parameters
    ----------
    zero_df : DataFrame  ('cluster' 컬럼 필요)
    feat_cols : list of str
    cluster_stats_df : DataFrame
    n_feats : int  - violin plot feature 수
    """
    valid_feats = [c for c in feat_cols if c in zero_df.columns and zero_df[c].std() > 0]
    n_clusters = zero_df["cluster"].nunique()

    # ── PCA ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(zero_df[valid_feats].values)
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 1) PCA scatter
    colors = ["#4A90D9", "#7BC67E", "#D0021B", "#F5A623", "#9B59B6"]
    for c in range(n_clusters):
        mask = zero_df["cluster"].values == c
        n_c = mask.sum()
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        alpha=0.15, s=5, color=colors[c % len(colors)],
                        label=f"Cluster {c} (n={n_c:,})")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    axes[0].set_title("Y=0 클러스터 (PCA)")
    axes[0].legend(markerscale=4, fontsize=9)

    # 2) 클러스터 크기
    sizes = zero_df["cluster"].value_counts().sort_index()
    axes[1].bar(sizes.index, sizes.values, color=colors[:n_clusters],
                edgecolor="black", alpha=0.8)
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Unit 수")
    axes[1].set_title("클러스터별 Unit 수")
    for i, (c, v) in enumerate(sizes.items()):
        axes[1].text(c, v + len(zero_df) * 0.005,
                     f"{v:,}\n({v/len(zero_df)*100:.1f}%)",
                     ha="center", fontsize=9)

    # 3) 상위 feature의 클러스터별 평균 bar chart
    top_feats = cluster_stats_df.head(n_feats)["feature"].tolist()
    cluster_means = zero_df.groupby("cluster")[top_feats].mean()
    # 표준화하여 비교
    overall = zero_df[top_feats].mean()
    overall_std = zero_df[top_feats].std()
    overall_std = overall_std.replace(0, 1)
    z_means = (cluster_means - overall) / overall_std

    z_means.T.plot(kind="barh", ax=axes[2], color=colors[:n_clusters],
                   edgecolor="black", alpha=0.8)
    axes[2].set_xlabel("Z-score (표준화 평균)")
    axes[2].set_title(f"클러스터별 Feature 프로파일 (상위 {n_feats})")
    axes[2].axvline(x=0, color="gray", linestyle="-", alpha=0.4)
    axes[2].legend(title="Cluster", fontsize=8)

    plt.suptitle("Y=0 내부 클러스터 분석", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def compare_zero_clusters_with_positive(xs_dict, ys_train, feat_cols, zero_df):
    """
    각 Y=0 클러스터와 Y>0 그룹 간 Cohen's d 비교
    → 어떤 클러스터가 Y>0과 가장 유사한지 식별 (잠재 불량 후보)

    Parameters
    ----------
    xs_dict : dict
    ys_train : DataFrame
    feat_cols : list of str
    zero_df : DataFrame  ('cluster' 컬럼 필요)

    Returns
    -------
    comparison_df : DataFrame  (cluster별 mean |Cohen's d|)
    """
    # Y>0 데이터 준비
    xs_train = xs_dict["train"]
    xs_unit = xs_train.groupby(KEY_COL)[feat_cols].mean()
    merged = xs_unit.merge(ys_train, left_index=True, right_on=KEY_COL, how="inner")
    pos_df = merged[merged[TARGET_COL] > 0]

    valid_feats = [c for c in feat_cols if c in zero_df.columns and zero_df[c].std() > 0]
    n_clusters = zero_df["cluster"].nunique()

    # 클러스터별 Y>0과의 Cohen's d
    results = []
    for c in range(n_clusters):
        cluster_data = zero_df[zero_df["cluster"] == c]
        d_values = []

        for col in valid_feats:
            v_c = cluster_data[col].dropna().values
            v_p = pos_df[col].dropna().values
            if len(v_c) < 2 or len(v_p) < 2:
                continue

            n_c, n_p = len(v_c), len(v_p)
            m_c, m_p = v_c.mean(), v_p.mean()
            s_c, s_p = v_c.std(ddof=1), v_p.std(ddof=1)
            pooled = np.sqrt(((n_c - 1) * s_c**2 + (n_p - 1) * s_p**2) / (n_c + n_p - 2))
            d = abs((m_c - m_p) / pooled) if pooled > 0 else 0
            d_values.append(d)

        mean_d = np.mean(d_values) if d_values else 0
        median_d = np.median(d_values) if d_values else 0

        results.append({
            "cluster": c,
            "n_units": len(cluster_data),
            "mean_abs_d": mean_d,
            "median_abs_d": median_d,
            "d_small_pct": np.mean([d >= 0.2 for d in d_values]) * 100 if d_values else 0,
        })

    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values("mean_abs_d").reset_index(drop=True)

    # 가장 유사한 클러스터 = mean |d|가 가장 작은 것
    suspect = comparison_df.iloc[0]

    print("=" * 65)
    print("Y=0 클러스터 vs Y>0 유사도 비교 (Cohen's d)")
    print("=" * 65)
    print("  mean |d|가 작을수록 Y>0과 유사 → 잠재 불량 가능성")
    print()
    print(f"  {'Cluster':>8}  {'Units':>7}  {'mean|d|':>8}  {'median|d|':>9}  {'|d|≥0.2%':>9}")
    print("-" * 55)
    for _, row in comparison_df.iterrows():
        print(f"  {int(row['cluster']):>8}  {int(row['n_units']):>7}  "
              f"{row['mean_abs_d']:>8.4f}  {row['median_abs_d']:>9.4f}  "
              f"{row['d_small_pct']:>8.1f}%")

    print(f"\n  → Cluster {int(suspect['cluster'])} ({int(suspect['n_units']):,} units)이 "
          f"Y>0과 가장 유사 (mean|d|={suspect['mean_abs_d']:.4f})")
    print("    이 클러스터 = 잠재 불량(우연적 zero) 후보")

    # 시각화
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4A90D9", "#7BC67E", "#D0021B", "#F5A623", "#9B59B6"]
    bars = ax.bar(comparison_df["cluster"].astype(str), comparison_df["mean_abs_d"],
                  color=[colors[int(c) % len(colors)] for c in comparison_df["cluster"]],
                  edgecolor="black", alpha=0.8)

    # 가장 유사한 클러스터 하이라이트
    bars[0].set_edgecolor("red")
    bars[0].set_linewidth(3)

    ax.set_xlabel("Y=0 Cluster")
    ax.set_ylabel("Mean |Cohen's d| vs Y>0")
    ax.set_title("Y=0 클러스터와 Y>0 간 유사도\n(낮을수록 잠재 불량 가능성)")
    for i, row in comparison_df.iterrows():
        ax.text(i, row["mean_abs_d"] + 0.002,
                f"{row['mean_abs_d']:.4f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.show()

    return comparison_df


def boundary_analysis(xs_dict, ys_train, feat_cols, n_feats=8):
    """
    Y=0 unit 중 Y>0과 가까운 "경계선 unit" 분석

    Mahalanobis 거리로 Y>0 중심으로부터의 거리를 계산하여
    Y=0을 "확실한 양품" vs "경계선 양품"으로 분리

    Parameters
    ----------
    xs_dict : dict
    ys_train : DataFrame
    feat_cols : list of str
    n_feats : int  - 시각화할 feature 수

    Returns
    -------
    zero_scored : DataFrame  ('dist_score' 컬럼 추가)
    """
    xs_train = xs_dict["train"]
    xs_unit = xs_train.groupby(KEY_COL)[feat_cols].mean()
    merged = xs_unit.merge(ys_train, left_index=True, right_on=KEY_COL, how="inner")

    zero_df = merged[merged[TARGET_COL] == 0].copy()
    pos_df = merged[merged[TARGET_COL] > 0].copy()

    # NaN imputation + 분산 0 제외
    valid_feats = []
    for col in feat_cols:
        if zero_df[col].isna().any():
            med = merged[col].median()
            zero_df[col] = zero_df[col].fillna(med)
            pos_df[col] = pos_df[col].fillna(med)
        if zero_df[col].std() > 0:
            valid_feats.append(col)

    # target 상관 상위 feature로 축소 (차원 축소 대체)
    corr = merged[valid_feats].corrwith(merged[TARGET_COL]).abs().sort_values(ascending=False)
    use_feats = corr.head(min(100, len(corr))).index.tolist()

    # Y>0 중심(centroid)과 공분산 계산
    pos_vals = pos_df[use_feats].values
    pos_mean = pos_vals.mean(axis=0)

    # 유사도 = Y>0 centroid까지의 유클리드 거리 (표준화 후)
    scaler = StandardScaler()
    scaler.fit(merged[use_feats].values)

    zero_scaled = scaler.transform(zero_df[use_feats].values)
    pos_mean_scaled = scaler.transform(pos_mean.reshape(1, -1))[0]

    distances = np.sqrt(np.sum((zero_scaled - pos_mean_scaled) ** 2, axis=1))
    zero_df["dist_score"] = distances

    # 하위 20% = 경계선 (Y>0과 가까움), 상위 20% = 확실한 양품
    q20 = np.percentile(distances, 20)
    q80 = np.percentile(distances, 80)

    boundary = zero_df[zero_df["dist_score"] <= q20]
    confident = zero_df[zero_df["dist_score"] >= q80]

    print("=" * 65)
    print("Y=0 경계선 분석 (Y>0 centroid까지의 거리)")
    print("=" * 65)
    print(f"Y=0 전체: {len(zero_df):,} units")
    print(f"경계선 (하위 20%, dist<={q20:.2f}): {len(boundary):,} units ← Y>0과 유사")
    print(f"확실한 양품 (상위 20%, dist>={q80:.2f}): {len(confident):,} units")
    print()

    # 두 그룹 간 feature 차이
    diff_results = []
    for col in use_feats[:50]:
        v_b = boundary[col].values
        v_c = confident[col].values
        if len(v_b) < 2 or len(v_c) < 2:
            continue
        n_b, n_c = len(v_b), len(v_c)
        m_b, m_c = v_b.mean(), v_c.mean()
        s_b, s_c = v_b.std(ddof=1), v_c.std(ddof=1)
        pooled = np.sqrt(((n_b - 1) * s_b**2 + (n_c - 1) * s_c**2) / (n_b + n_c - 2))
        d = (m_b - m_c) / pooled if pooled > 0 else 0
        diff_results.append({"feature": col, "cohens_d": d, "abs_d": abs(d)})

    diff_df = pd.DataFrame(diff_results).sort_values("abs_d", ascending=False)

    print("경계선 vs 확실한 양품: Cohen's d 상위 10개:")
    print(f"  {'Feature':>8}  {'Cohen d':>9}")
    print("-" * 25)
    for _, row in diff_df.head(10).iterrows():
        print(f"  {row['feature']:>8}  {row['cohens_d']:>+9.4f}")

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # 1) 거리 분포
    axes[0].hist(distances, bins=60, edgecolor="black", color="steelblue", alpha=0.8)
    axes[0].axvline(x=q20, color="red", linestyle="--", label=f"하위 20% ({q20:.1f})")
    axes[0].axvline(x=q80, color="green", linestyle="--", label=f"상위 20% ({q80:.1f})")
    axes[0].set_xlabel("Y>0 centroid까지 거리")
    axes[0].set_ylabel("Unit 수")
    axes[0].set_title("Y=0 unit의 Y>0 유사도 분포")
    axes[0].legend(fontsize=9)

    # 2) PCA scatter (경계선 vs 확실한 양품)
    pca = PCA(n_components=2, random_state=SEED)
    X_all = scaler.transform(zero_df[use_feats].values)
    X_pca = pca.fit_transform(X_all)

    mask_b = zero_df["dist_score"].values <= q20
    mask_c = zero_df["dist_score"].values >= q80
    mask_m = ~mask_b & ~mask_c

    axes[1].scatter(X_pca[mask_m, 0], X_pca[mask_m, 1],
                    alpha=0.05, s=3, color="gray", label="중간")
    axes[1].scatter(X_pca[mask_c, 0], X_pca[mask_c, 1],
                    alpha=0.15, s=5, color="#4A90D9", label="확실한 양품")
    axes[1].scatter(X_pca[mask_b, 0], X_pca[mask_b, 1],
                    alpha=0.15, s=5, color="#D0021B", label="경계선 (잠재 불량)")
    axes[1].set_title("Y=0 경계선 분류 (PCA)")
    axes[1].legend(markerscale=4, fontsize=9)

    # 3) 상위 feature violin
    top_diff_feats = diff_df.head(min(n_feats, 4))["feature"].tolist()
    if top_diff_feats:
        plot_data = pd.concat([
            boundary[top_diff_feats].assign(group="경계선"),
            confident[top_diff_feats].assign(group="확실한 양품"),
        ])
        plot_melted = plot_data.melt(id_vars="group", var_name="feature", value_name="value")

        # 샘플링
        if len(plot_melted) > 50000:
            plot_melted = plot_melted.sample(50000, random_state=SEED)

        sns.boxplot(x="feature", y="value", hue="group", data=plot_melted,
                    ax=axes[2], palette={"경계선": "salmon", "확실한 양품": "skyblue"})
        axes[2].set_title("경계선 vs 확실한 양품: Feature 비교")
        axes[2].tick_params(axis="x", rotation=30, labelsize=9)
        axes[2].legend(fontsize=9)

    plt.suptitle("Y=0 경계선 분석 (잠재 불량 탐색)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    return zero_df
