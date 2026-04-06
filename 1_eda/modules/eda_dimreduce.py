"""
EDA 모듈: PCA / UMAP 차원축소 시각화
- PCA: 주성분 분석으로 분산 구조 파악, scree plot, 2D/3D scatter
- UMAP: 비선형 차원축소로 데이터의 국소 구조 시각화
- Y=0 vs Y>0 색 구분으로 분리 가능성 시각적 확인
- 노트북에서 import eda_dimreduce as dr 로 사용
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils.config import KEY_COL, TARGET_COL, SEED


def _prepare_unit_data(xs_dict, ys_train, feat_cols):
    """
    Train die→unit 평균 집계 + target merge + NaN imputation + StandardScaler

    Returns
    -------
    X_scaled : ndarray  (unit × feature, scaled)
    y : ndarray  (target values)
    valid_feats : list
    merged : DataFrame
    """
    xs_train = xs_dict["train"]
    xs_unit = xs_dict['train_unit_mean'] if 'train_unit_mean' in xs_dict else xs_train.groupby(KEY_COL)[feat_cols].mean()
    merged = xs_unit.merge(ys_train, left_index=True, right_on=KEY_COL, how="inner")

    # NaN median imputation
    for col in feat_cols:
        if merged[col].isna().any():
            merged[col] = merged[col].fillna(merged[col].median())

    # 분산 0 feature 제외
    valid_feats = [c for c in feat_cols if merged[c].std() > 0]

    # StandardScaler (PCA는 스케일 민감)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(merged[valid_feats].values)
    y = merged[TARGET_COL].values

    return X_scaled, y, valid_feats, merged


def run_pca(xs_dict, ys_train, feat_cols, n_components=50):
    """
    PCA 수행 및 결과 반환

    Parameters
    ----------
    xs_dict : dict
    ys_train : DataFrame
    feat_cols : list of str
    n_components : int  - 계산할 주성분 수 (기본 50)

    Returns
    -------
    pca : fitted PCA object
    X_pca : ndarray  (unit × n_components)
    y : ndarray
    valid_feats : list
    merged : DataFrame
    """
    X_scaled, y, valid_feats, merged = _prepare_unit_data(xs_dict, ys_train, feat_cols)

    pca = PCA(n_components=min(n_components, len(valid_feats)), random_state=SEED)
    X_pca = pca.fit_transform(X_scaled)

    total_var = pca.explained_variance_ratio_.sum() * 100
    print(f"PCA 완료: {pca.n_components_}개 주성분 → 누적 분산 {total_var:.1f}%")
    print(f"  PC1: {pca.explained_variance_ratio_[0]*100:.2f}%")
    print(f"  PC1~PC5: {pca.explained_variance_ratio_[:5].sum()*100:.2f}%")
    print(f"  PC1~PC10: {pca.explained_variance_ratio_[:10].sum()*100:.2f}%")
    print(f"  PC1~PC20: {pca.explained_variance_ratio_[:20].sum()*100:.2f}%")

    return pca, X_pca, y, valid_feats, merged


def plot_scree(pca, top_n=30):
    """
    Scree plot: 개별 분산 설명률 + 누적 분산 설명률

    Parameters
    ----------
    pca : fitted PCA object
    top_n : int  - 표시할 주성분 수
    """
    var_ratio = pca.explained_variance_ratio_[:top_n]
    cum_var = np.cumsum(var_ratio)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # 개별 분산
    axes[0].bar(range(1, len(var_ratio) + 1), var_ratio * 100,
                color="steelblue", edgecolor="black", alpha=0.8)
    axes[0].set_xlabel("주성분 (PC)")
    axes[0].set_ylabel("분산 설명률 (%)")
    axes[0].set_title("개별 주성분 분산 설명률")
    axes[0].set_xticks(range(1, len(var_ratio) + 1, 2))

    # 누적 분산
    axes[1].plot(range(1, len(cum_var) + 1), cum_var * 100,
                 "o-", color="coral", markersize=4)
    axes[1].set_xlabel("주성분 수")
    axes[1].set_ylabel("누적 분산 설명률 (%)")
    axes[1].set_title("누적 분산 설명률")
    axes[1].axhline(y=80, color="gray", linestyle="--", alpha=0.5, label="80%")
    axes[1].axhline(y=90, color="gray", linestyle=":", alpha=0.5, label="90%")
    axes[1].legend()
    axes[1].set_xticks(range(1, len(cum_var) + 1, 2))

    # 80%, 90% 달성 주성분 수 표시
    cum_pct = np.cumsum(pca.explained_variance_ratio_) * 100
    n_80 = int(np.argmax(cum_pct >= 80) + 1) if np.any(cum_pct >= 80) else None
    n_90 = int(np.argmax(cum_pct >= 90) + 1) if np.any(cum_pct >= 90) else None

    if n_80 is not None:
        axes[1].axvline(x=n_80, color="blue", linestyle="--", alpha=0.4)
        axes[1].text(n_80 + 0.5, 75, f"80%: PC{n_80}", fontsize=9, color="blue")
    if n_90 is not None:
        axes[1].axvline(x=n_90, color="red", linestyle="--", alpha=0.4)
        axes[1].text(n_90 + 0.5, 85, f"90%: PC{n_90}", fontsize=9, color="red")

    plt.tight_layout()
    plt.show()

    total_var = cum_pct[-1]
    print(f"계산된 {len(cum_pct)}개 주성분의 누적 분산: {total_var:.1f}%")
    if n_80 is not None:
        print(f"80% 분산 달성: PC{n_80}개 필요")
    else:
        print(f"80% 분산 미달성: {len(cum_pct)}개 PC로 {total_var:.1f}% (더 많은 주성분 필요)")
    if n_90 is not None:
        print(f"90% 분산 달성: PC{n_90}개 필요")
    else:
        print(f"90% 분산 미달성: {len(cum_pct)}개 PC로 {total_var:.1f}% (더 많은 주성분 필요)")


def plot_pca_scatter(X_pca, y):
    """
    PC1 vs PC2, PC1 vs PC3 scatter (Y=0 vs Y>0 색 구분)
    + health 값 연속 컬러맵 버전
    PC가 3개 미만이면 PC3 관련 subplot은 안내 메시지로 대체.

    Parameters
    ----------
    X_pca : ndarray
    y : ndarray
    """
    is_zero = y == 0
    is_pos = y > 0
    n_pc = X_pca.shape[1]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # ── 1) PC1 vs PC2: Y=0 vs Y>0 ──
    axes[0, 0].scatter(X_pca[is_zero, 0], X_pca[is_zero, 1],
                       alpha=0.1, s=3, color="skyblue", label="Y=0")
    axes[0, 0].scatter(X_pca[is_pos, 0], X_pca[is_pos, 1],
                       alpha=0.2, s=3, color="red", label="Y>0")
    axes[0, 0].set_xlabel("PC1")
    axes[0, 0].set_ylabel("PC2")
    axes[0, 0].set_title("PC1 vs PC2 (Y=0 vs Y>0)")
    axes[0, 0].legend(markerscale=5, fontsize=10)

    # ── 2) PC1 vs PC3: Y=0 vs Y>0 ──
    if n_pc >= 3:
        axes[0, 1].scatter(X_pca[is_zero, 0], X_pca[is_zero, 2],
                           alpha=0.1, s=3, color="skyblue", label="Y=0")
        axes[0, 1].scatter(X_pca[is_pos, 0], X_pca[is_pos, 2],
                           alpha=0.2, s=3, color="red", label="Y>0")
        axes[0, 1].set_xlabel("PC1")
        axes[0, 1].set_ylabel("PC3")
        axes[0, 1].set_title("PC1 vs PC3 (Y=0 vs Y>0)")
        axes[0, 1].legend(markerscale=5, fontsize=10)
    else:
        axes[0, 1].text(0.5, 0.5, f"PC3 없음 (n_components={n_pc})",
                         transform=axes[0, 1].transAxes, ha="center", fontsize=12)
        axes[0, 1].set_title("PC1 vs PC3 (N/A)")

    # ── 3) PC1 vs PC2: health 연속 컬러 ──
    # Y>0만 컬러로 표시 (Y=0은 회색 배경)
    axes[1, 0].scatter(X_pca[is_zero, 0], X_pca[is_zero, 1],
                       alpha=0.05, s=2, color="lightgray")
    sc = axes[1, 0].scatter(X_pca[is_pos, 0], X_pca[is_pos, 1],
                            c=y[is_pos], cmap="YlOrRd", alpha=0.3, s=5,
                            vmin=0, vmax=np.percentile(y[is_pos], 95))
    plt.colorbar(sc, ax=axes[1, 0], label="health", shrink=0.8)
    axes[1, 0].set_xlabel("PC1")
    axes[1, 0].set_ylabel("PC2")
    axes[1, 0].set_title("PC1 vs PC2 (health 값 컬러)")

    # ── 4) PC2 vs PC3: Y=0 vs Y>0 ──
    if n_pc >= 3:
        axes[1, 1].scatter(X_pca[is_zero, 1], X_pca[is_zero, 2],
                           alpha=0.1, s=3, color="skyblue", label="Y=0")
        axes[1, 1].scatter(X_pca[is_pos, 1], X_pca[is_pos, 2],
                           alpha=0.2, s=3, color="red", label="Y>0")
        axes[1, 1].set_xlabel("PC2")
        axes[1, 1].set_ylabel("PC3")
        axes[1, 1].set_title("PC2 vs PC3 (Y=0 vs Y>0)")
        axes[1, 1].legend(markerscale=5, fontsize=10)
    else:
        axes[1, 1].text(0.5, 0.5, f"PC3 없음 (n_components={n_pc})",
                         transform=axes[1, 1].transAxes, ha="center", fontsize=12)
        axes[1, 1].set_title("PC2 vs PC3 (N/A)")

    plt.suptitle("PCA 차원축소 시각화", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_pca_loadings(pca, valid_feats, n_pc=3, n_feat=15):
    """
    상위 주성분의 loading (어떤 feature가 해당 PC를 구성하는지)

    Parameters
    ----------
    pca : fitted PCA object
    valid_feats : list of str
    n_pc : int  - 표시할 주성분 수 (기본 3)
    n_feat : int  - PC당 표시할 feature 수 (기본 15)
    """
    fig, axes = plt.subplots(1, n_pc, figsize=(6 * n_pc, 6))
    if n_pc == 1:
        axes = [axes]

    for pc_idx in range(n_pc):
        loadings = pd.Series(pca.components_[pc_idx], index=valid_feats)
        abs_loadings = loadings.abs().sort_values(ascending=False)
        top_feats = abs_loadings.head(n_feat).index

        vals = loadings[top_feats]
        colors = ["coral" if v < 0 else "steelblue" for v in vals]

        axes[pc_idx].barh(range(len(top_feats)), vals.values,
                          color=colors, edgecolor="black")
        axes[pc_idx].set_yticks(range(len(top_feats)))
        axes[pc_idx].set_yticklabels(top_feats, fontsize=8)
        axes[pc_idx].set_xlabel("Loading")
        axes[pc_idx].set_title(f"PC{pc_idx+1} ({pca.explained_variance_ratio_[pc_idx]*100:.1f}%)")
        axes[pc_idx].invert_yaxis()
        axes[pc_idx].axvline(x=0, color="gray", linestyle="-", alpha=0.3)

    plt.suptitle(f"PCA Loading (상위 {n_feat}개 Feature)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


def run_umap_and_plot(X_pca, y, n_components_pca=20):
    """
    UMAP 2D 매핑 및 시각화 (PCA 사전축소 후 적용)
    UMAP 미설치 시 t-SNE fallback

    Parameters
    ----------
    X_pca : ndarray  - PCA 결과 (전체 component)
    y : ndarray  - target values
    n_components_pca : int  - UMAP 입력으로 사용할 PCA 차원 수
    """
    X_input = X_pca[:, :n_components_pca]

    try:
        import umap
        print(f"UMAP 실행 중 (입력: PC1~PC{n_components_pca}, {len(y):,} units)...")
        reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                            metric="euclidean", random_state=SEED)
        X_2d = reducer.fit_transform(X_input)
        method_name = "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE
        print(f"UMAP 미설치 → t-SNE fallback (입력: PC1~PC{n_components_pca}, {len(y):,} units)")
        tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, max_iter=1000)
        X_2d = tsne.fit_transform(X_input)
        method_name = "t-SNE"

    print(f"{method_name} 완료!")

    is_zero = y == 0
    is_pos = y > 0

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # ── 1) Y=0 vs Y>0 ──
    axes[0].scatter(X_2d[is_zero, 0], X_2d[is_zero, 1],
                    alpha=0.1, s=3, color="skyblue", label="Y=0")
    axes[0].scatter(X_2d[is_pos, 0], X_2d[is_pos, 1],
                    alpha=0.25, s=3, color="red", label="Y>0")
    axes[0].set_title(f"{method_name}: Y=0 vs Y>0")
    axes[0].legend(markerscale=5, fontsize=11)
    axes[0].set_xlabel(f"{method_name}-1")
    axes[0].set_ylabel(f"{method_name}-2")

    # ── 2) health 연속 컬러 ──
    axes[1].scatter(X_2d[is_zero, 0], X_2d[is_zero, 1],
                    alpha=0.05, s=2, color="lightgray")
    sc = axes[1].scatter(X_2d[is_pos, 0], X_2d[is_pos, 1],
                         c=y[is_pos], cmap="YlOrRd", alpha=0.35, s=5,
                         vmin=0, vmax=np.percentile(y[is_pos], 95))
    plt.colorbar(sc, ax=axes[1], label="health", shrink=0.8)
    axes[1].set_title(f"{method_name}: health 값 컬러")
    axes[1].set_xlabel(f"{method_name}-1")
    axes[1].set_ylabel(f"{method_name}-2")

    plt.suptitle(f"{method_name} 2D 차원축소 (PCA {n_components_pca}D → 2D)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()
