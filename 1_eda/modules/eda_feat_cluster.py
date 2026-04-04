"""
EDA 모듈: Feature Group / Cluster Discovery
- 1,087개 WT feature 사이의 자연스러운 상관 그룹(클러스터)을 탐색
- 계층적 군집분석(Agglomerative Clustering)으로 유사 feature 묶음
- 클러스터별 intra-correlation heatmap, dendrogram 시각화
- 클러스터-Target 상관 분석으로 어떤 feature 그룹이 health 예측에 유용한지 파악
- 노트북에서 import eda_feat_cluster as fc 로 사용
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from utils.config import KEY_COL, TARGET_COL, SEED


def compute_feature_correlation(xs_dict, feat_cols, sample_n=5000):
    """
    Train set에서 샘플링하여 feature-feature Pearson 상관행렬 계산

    대규모 feature(~1,087개) 간 상관행렬은 연산량이 크므로,
    die-level 행을 sample_n개만 추출하여 빠르게 근사한다.

    Parameters
    ----------
    xs_dict : dict
        {"train": DataFrame, ...} split별 die-level 데이터
    feat_cols : list of str
        상관행렬 계산 대상 feature 컬럼명 리스트 (e.g. X0~X1086 중 유효 feature)
    sample_n : int
        속도를 위해 train에서 샘플링할 행 수. 기본 5,000

    Returns
    -------
    corr_matrix : DataFrame
        (n_features x n_features) Pearson 상관행렬
    """
    xs_train = xs_dict["train"]

    # 샘플링
    n_rows = len(xs_train)
    if n_rows > sample_n:
        xs_sample = xs_train[feat_cols].sample(n=sample_n, random_state=SEED)
        print(f"Train {n_rows:,}행 중 {sample_n:,}행 샘플링하여 상관행렬 계산")
    else:
        xs_sample = xs_train[feat_cols]
        print(f"Train 전체 {n_rows:,}행으로 상관행렬 계산")

    corr_matrix = xs_sample.corr(method="pearson")

    # NaN 처리 (상수 feature 등으로 발생 가능)
    nan_count = corr_matrix.isna().sum().sum()
    if nan_count > 0:
        print(f"  상관행렬 내 NaN: {nan_count}개 → 0으로 대체")
        corr_matrix = corr_matrix.fillna(0)

    print(f"  상관행렬 shape: {corr_matrix.shape}")
    print(f"  |r| > 0.9 쌍 수: {((corr_matrix.abs() > 0.9).sum().sum() - len(feat_cols)) // 2}")
    print(f"  |r| > 0.95 쌍 수: {((corr_matrix.abs() > 0.95).sum().sum() - len(feat_cols)) // 2}")

    return corr_matrix


def cluster_features(corr_matrix, n_clusters=20, method="average"):
    """
    상관행렬 기반 계층적 군집분석으로 feature를 그룹화

    거리 행렬로 1 - |corr|을 사용하여, 상관이 높은 feature가 가까운 거리로 묶이도록 한다.

    Parameters
    ----------
    corr_matrix : DataFrame
        compute_feature_correlation()에서 반환된 상관행렬
    n_clusters : int
        최종 클러스터 수. 기본 20
    method : str
        scipy linkage method. 기본 "average"
        (1-|corr| 거리는 비유클리드이므로 "ward" 부적합, "average"/"complete" 권장)

    Returns
    -------
    linkage_matrix : ndarray
        scipy linkage 결과 (dendrogram 그리기용)
    cluster_labels : Series
        index=feature명, value=cluster_id (0-based)
    cluster_df : DataFrame
        columns = [feature, cluster, mean_abs_corr_within]
        각 feature가 속한 클러스터와 클러스터 내 평균 |r|
    """
    features = corr_matrix.columns.tolist()

    # 거리 행렬: 1 - |r|  (|r|이 클수록 거리가 가까움)
    dist_matrix = 1 - corr_matrix.abs().values

    # 대각선을 0으로 보정 (부동소수점 오차 방지)
    np.fill_diagonal(dist_matrix, 0)

    # condensed distance matrix (상삼각 → 1D)
    from scipy.spatial.distance import squareform
    condensed_dist = squareform(dist_matrix, checks=False)

    # 계층적 군집분석
    linkage_matrix = linkage(condensed_dist, method=method)

    # n_clusters개로 절단
    labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
    # fcluster는 1-based → 0-based로 변환
    labels = labels - 1

    cluster_labels = pd.Series(labels, index=features, name="cluster")

    # 클러스터별 요약
    cluster_records = []
    for feat in features:
        cid = cluster_labels[feat]
        # 같은 클러스터 내 feature들
        members = cluster_labels[cluster_labels == cid].index.tolist()
        if len(members) > 1:
            # 해당 feature와 같은 클러스터 내 다른 feature들의 |r| 평균
            intra_corrs = corr_matrix.loc[feat, members].abs()
            # 자기 자신(r=1) 제외
            mean_abs_corr = intra_corrs.drop(feat).mean()
        else:
            mean_abs_corr = np.nan  # 클러스터에 feature가 1개뿐이면 계산 불가

        cluster_records.append({
            "feature": feat,
            "cluster": cid,
            "mean_abs_corr_within": mean_abs_corr,
        })

    cluster_df = pd.DataFrame(cluster_records)

    # 클러스터 크기 요약 출력
    cluster_sizes = cluster_labels.value_counts().sort_index()
    print(f"Feature 클러스터링 완료: {len(features)}개 feature → {n_clusters}개 클러스터")
    print(f"  method: {method}")
    print(f"\n클러스터 크기 분포:")
    print(f"  {'Cluster':>8}  {'Size':>6}  {'비율':>8}")
    print("  " + "-" * 28)
    for cid in cluster_sizes.index:
        size = cluster_sizes[cid]
        pct = size / len(features) * 100
        print(f"  {cid:>8}  {size:>6}  {pct:>7.1f}%")

    print(f"\n  최대 클러스터: {cluster_sizes.max()}개, "
          f"최소: {cluster_sizes.min()}개, "
          f"중앙값: {cluster_sizes.median():.0f}개")

    # 클러스터 내 평균 |r| 요약
    cluster_intra = cluster_df.groupby("cluster")["mean_abs_corr_within"].mean()
    print(f"\n클러스터 내 평균 |r|:")
    print(f"  전체 평균: {cluster_intra.mean():.3f}")
    print(f"  최대 (가장 응집): cluster {cluster_intra.idxmax()} "
          f"(|r|={cluster_intra.max():.3f})")
    print(f"  최소 (가장 느슨): cluster {cluster_intra.idxmin()} "
          f"(|r|={cluster_intra.min():.3f})")

    return linkage_matrix, cluster_labels, cluster_df


def plot_dendrogram(linkage_matrix, n_clusters=20, truncate_p=30):
    """
    계층적 군집분석 dendrogram 시각화 (truncated)

    Parameters
    ----------
    linkage_matrix : ndarray
        cluster_features()에서 반환된 linkage 행렬
    n_clusters : int
        클러스터 수 (color_threshold 결정에 사용). 기본 20
    truncate_p : int
        dendrogram에 표시할 leaf 수 (truncate). 기본 30
    """
    # color_threshold: n_clusters개로 나뉘는 거리를 자동 산출
    # fcluster maxclust와 일치하도록 linkage의 적절한 높이를 찾음
    if len(linkage_matrix) >= n_clusters:
        # linkage_matrix의 거리 컬럼(index 2)에서 n_clusters-1번째 병합 거리
        sorted_dists = linkage_matrix[:, 2]
        # 상위에서 n_clusters번째 병합 = 아래에서 (total - n_clusters)번째
        threshold_idx = len(sorted_dists) - n_clusters
        color_threshold = sorted_dists[threshold_idx]
    else:
        color_threshold = None

    fig, ax = plt.subplots(figsize=(18, 6))

    dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        p=truncate_p,
        color_threshold=color_threshold,
        leaf_rotation=90,
        leaf_font_size=8,
        ax=ax,
    )

    ax.set_title(f"Feature Dendrogram (truncated to {truncate_p} leaves, "
                 f"{n_clusters} clusters)", fontsize=14)
    ax.set_xlabel("Feature (or cluster size)")
    ax.set_ylabel("Distance (1 - |r|)")

    if color_threshold is not None:
        ax.axhline(y=color_threshold, color="red", linestyle="--", alpha=0.7,
                    label=f"cut threshold = {color_threshold:.3f}")
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_cluster_heatmap(corr_matrix, cluster_labels, top_n_clusters=5,
                         max_feats_per_cluster=15):
    """
    상위 N개 클러스터의 intra-cluster 상관행렬 히트맵

    가장 큰 클러스터부터 순서대로, 클러스터 내 feature 간 상관을 시각화한다.
    feature가 많으면 max_feats_per_cluster개로 제한하여 가독성 확보.

    Parameters
    ----------
    corr_matrix : DataFrame
        compute_feature_correlation()에서 반환된 상관행렬
    cluster_labels : Series
        cluster_features()에서 반환된 feature→cluster_id 매핑
    top_n_clusters : int
        시각화할 클러스터 수 (크기 순). 기본 5
    max_feats_per_cluster : int
        클러스터당 최대 표시 feature 수. 기본 15
    """
    cluster_sizes = cluster_labels.value_counts().sort_values(ascending=False)
    top_clusters = cluster_sizes.head(top_n_clusters).index.tolist()

    n_cols = min(top_n_clusters, 3)
    n_rows = (top_n_clusters + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5.5 * n_rows))
    axes = np.array(axes).flatten()

    for i, cid in enumerate(top_clusters):
        members = cluster_labels[cluster_labels == cid].index.tolist()
        n_members = len(members)

        # feature가 많으면 상관 합이 큰 순서대로 상위만 표시
        if n_members > max_feats_per_cluster:
            # 클러스터 내 평균 |r|이 높은 feature 우선
            mean_corrs = corr_matrix.loc[members, members].abs().mean()
            members = mean_corrs.nlargest(max_feats_per_cluster).index.tolist()
            suffix = f" (상위 {max_feats_per_cluster}/{n_members}개)"
        else:
            suffix = ""

        sub_corr = corr_matrix.loc[members, members]

        sns.heatmap(
            sub_corr,
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            annot=len(members) <= 10,
            fmt=".2f" if len(members) <= 10 else "",
            square=True,
            linewidths=0.3,
            ax=axes[i],
            xticklabels=True,
            yticklabels=True,
            cbar_kws={"shrink": 0.7},
        )
        axes[i].set_title(f"Cluster {cid} ({n_members}개 feature){suffix}",
                          fontsize=11)
        axes[i].tick_params(labelsize=7, axis="both")

    # 사용하지 않는 subplot 숨기기
    for j in range(len(top_clusters), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"상위 {top_n_clusters}개 클러스터 Intra-Correlation Heatmap",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def cluster_target_correlation(xs_dict, ys_train, feat_cols, cluster_labels,
                               n_top=10):
    """
    클러스터별 Target(health) 상관 분석

    각 클러스터에 속한 feature들의 target 상관계수 평균을 계산하여,
    어떤 feature 그룹이 health 예측에 가장 유용한지 파악한다.
    die->unit 평균 집계 후 Pearson 상관을 계산한다.

    Parameters
    ----------
    xs_dict : dict
        {"train": DataFrame, ...} split별 die-level 데이터
    ys_train : DataFrame
        train Y 데이터 (KEY_COL, TARGET_COL 컬럼)
    feat_cols : list of str
        상관 계산 대상 feature 컬럼명 리스트
    cluster_labels : Series
        cluster_features()에서 반환된 feature->cluster_id 매핑
    n_top : int
        상위 출력/시각화할 클러스터 수. 기본 10

    Returns
    -------
    cluster_summary : DataFrame
        columns = [cluster_id, n_features, mean_abs_corr, top_feature, top_feature_corr]
        mean_abs_corr 기준 내림차순 정렬
    """
    xs_train = xs_dict["train"]

    # die -> unit 평균 집계
    # feat_cols 중 cluster_labels에 포함된 것만 사용
    valid_feats = [f for f in feat_cols if f in cluster_labels.index]
    _cached = xs_dict.get('train_unit_mean')
    xs_unit = _cached[valid_feats] if _cached is not None else xs_train.groupby(KEY_COL)[valid_feats].mean()
    merged = xs_unit.merge(ys_train, left_index=True, right_on=KEY_COL, how="inner")

    print(f"Unit-level merged: {merged.shape[0]:,} units, {len(valid_feats)} features")

    # 전체 feature-target 상관계수
    feat_target_corr = merged[valid_feats].corrwith(merged[TARGET_COL]).dropna()

    # 클러스터별 요약
    summary_records = []
    unique_clusters = sorted(cluster_labels[valid_feats].unique())

    for cid in unique_clusters:
        members = cluster_labels[cluster_labels == cid].index.tolist()
        # feat_cols에 포함된 member만
        members_valid = [m for m in members if m in feat_target_corr.index]

        if not members_valid:
            continue

        abs_corrs = feat_target_corr[members_valid].abs()
        mean_abs_corr = abs_corrs.mean()

        # 클러스터 내 최고 상관 feature
        top_feat = abs_corrs.idxmax()
        top_corr = feat_target_corr[top_feat]

        summary_records.append({
            "cluster_id": cid,
            "n_features": len(members_valid),
            "mean_abs_corr": mean_abs_corr,
            "max_abs_corr": abs_corrs.max(),
            "min_abs_corr": abs_corrs.min(),
            "top_feature": top_feat,
            "top_feature_corr": top_corr,
        })

    cluster_summary = pd.DataFrame(summary_records)
    cluster_summary = cluster_summary.sort_values("mean_abs_corr", ascending=False)
    cluster_summary = cluster_summary.reset_index(drop=True)

    # 출력
    print(f"\n{'='*80}")
    print(f"클러스터별 Target 상관 요약 (상위 {n_top}개)")
    print(f"{'='*80}")
    print(f"  {'Cluster':>8}  {'N_feat':>6}  {'mean|r|':>8}  {'max|r|':>8}  "
          f"{'Top Feature':>12}  {'Top r':>10}")
    print("  " + "-" * 65)

    for _, row in cluster_summary.head(n_top).iterrows():
        print(f"  {int(row['cluster_id']):>8}  {int(row['n_features']):>6}  "
              f"{row['mean_abs_corr']:>8.4f}  {row['max_abs_corr']:>8.4f}  "
              f"{row['top_feature']:>12}  {row['top_feature_corr']:>+10.4f}")

    # 시각화: 클러스터별 mean|r| bar chart
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # ── 1) 전체 클러스터 mean|r| 바 차트 (내림차순) ──
    plot_data = cluster_summary.copy()
    x_labels = [f"C{int(c)}" for c in plot_data["cluster_id"]]
    colors = plt.cm.YlOrRd(
        plot_data["mean_abs_corr"] / plot_data["mean_abs_corr"].max()
    )

    axes[0].bar(range(len(plot_data)), plot_data["mean_abs_corr"].values,
                color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(range(len(plot_data)))
    axes[0].set_xticklabels(x_labels, rotation=45, fontsize=8, ha="right")
    axes[0].set_xlabel("Cluster")
    axes[0].set_ylabel("Mean |Pearson r| with Target")
    axes[0].set_title("클러스터별 Target 평균 상관 (내림차순)")

    # ── 2) 상위 n_top 클러스터: feature 수 vs mean|r| scatter ──
    top_data = cluster_summary.head(n_top)
    scatter = axes[1].scatter(
        top_data["n_features"],
        top_data["mean_abs_corr"],
        s=top_data["n_features"] * 5,
        c=top_data["mean_abs_corr"],
        cmap="YlOrRd",
        edgecolors="black",
        linewidth=0.5,
        alpha=0.8,
    )
    for _, row in top_data.iterrows():
        axes[1].annotate(
            f"C{int(row['cluster_id'])}",
            (row["n_features"], row["mean_abs_corr"]),
            fontsize=8,
            ha="left",
            va="bottom",
        )
    axes[1].set_xlabel("Cluster Size (feature 수)")
    axes[1].set_ylabel("Mean |Pearson r| with Target")
    axes[1].set_title(f"상위 {n_top}개 클러스터: 크기 vs Target 상관")
    plt.colorbar(scatter, ax=axes[1], label="Mean |r|", shrink=0.8)

    plt.suptitle("Feature Cluster - Target Correlation Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    return cluster_summary