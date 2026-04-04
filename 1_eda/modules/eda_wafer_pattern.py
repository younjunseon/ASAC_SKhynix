"""
EDA 모듈: 웨이퍼 불량 패턴 분류
- 개별 웨이퍼의 불량 die 공간 분포를 분석하여 패턴 유형 분류
- 6가지 패턴: Center, Edge, Scratch, Local, Random, None
- 논문 근거: 3-1 (Radon+CNN), 3-2 (59개 hand-crafted 피처), 1-2 (Kang 2015, SK Hynix)
- 기존 eda_spatial (전역 통계)과 달리, 개별 웨이퍼 단위로 패턴 분류
- 노트북에서 import eda_wafer_pattern as wp 로 사용
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, Patch
from matplotlib.collections import PatchCollection
from utils.config import KEY_COL, TARGET_COL, SEED

# ─── 패턴 유형 정의 ──────────────────────────────────────
PATTERN_TYPES = ["None", "Center", "Edge", "Scratch", "Local", "Random"]

PATTERN_COLORS = {
    "None": "#cccccc",
    "Center": "#e74c3c",
    "Edge": "#3498db",
    "Scratch": "#f39c12",
    "Local": "#9b59b6",
    "Random": "#2ecc71",
}

PATTERN_DESC = {
    "None": "불량 die 거의 없음 (defect_rate < 5%)",
    "Center": "불량이 웨이퍼 중심부에 집중",
    "Edge": "불량이 웨이퍼 가장자리에 집중",
    "Scratch": "불량이 직선/대각선 형태로 분포 (PCA elongation 높음)",
    "Local": "불량이 특정 영역에 밀집 클러스터 (HDBSCAN)",
    "Random": "불량이 무작위 분산 (뚜렷한 패턴 없음)",
}


# ─── 1. 웨이퍼별 공간 피처 추출 ─────────────────────────
def extract_wafer_features(xs_parsed, ys_train):
    """
    전체 웨이퍼에 대해 불량 die의 공간 통계 피처를 추출한다.

    Parameters
    ----------
    xs_parsed : DataFrame
        parse_wafer_coords() 결과 (ufs_serial, wafer_id, die_x, die_y 등)
    ys_train : DataFrame
        train Y 데이터 (ufs_serial, health)

    Returns
    -------
    wafer_features : DataFrame
        웨이퍼별 공간 피처 (wafer_id, n_total, n_defect, defect_rate, ...)
    """
    # die에 health merge
    merged = xs_parsed.merge(ys_train[[KEY_COL, TARGET_COL]], on=KEY_COL, how="inner")

    wafer_ids = merged["wafer_id"].unique()
    results = []

    for wid in wafer_ids:
        wf = merged[merged["wafer_id"] == wid]
        n_total = len(wf)
        defect_mask = wf[TARGET_COL] > 0
        n_defect = defect_mask.sum()
        defect_rate = n_defect / n_total

        # 웨이퍼 중심 & 반지름 (전체 die 기준, x/y 독립)
        all_x = wf["die_x"].values.astype(float)
        all_y = wf["die_y"].values.astype(float)
        cx = (all_x.max() + all_x.min()) / 2
        cy = (all_y.max() + all_y.min()) / 2
        Rx = (all_x.max() - all_x.min()) / 2
        Ry = (all_y.max() - all_y.min()) / 2
        if Rx == 0:
            Rx = 1.0
        if Ry == 0:
            Ry = 1.0

        row = {
            "wafer_id": wid,
            "n_total": n_total,
            "n_defect": n_defect,
            "defect_rate": defect_rate,
            "mean_health": wf[TARGET_COL].mean(),
        }

        if n_defect < 3:
            # 불량 die가 너무 적으면 공간 피처 의미 없음
            row.update({
                "radial_mean": np.nan, "radial_std": np.nan,
                "ring_concentration": np.nan, "center_concentration": np.nan,
                "elongation": np.nan, "pca_angle": np.nan,
                "n_clusters": 0, "largest_cluster_ratio": np.nan,
                "noise_ratio": np.nan, "dispersion": np.nan,
            })
            results.append(row)
            continue

        # 불량 die 좌표
        dx = wf.loc[defect_mask, "die_x"].values.astype(float)
        dy = wf.loc[defect_mask, "die_y"].values.astype(float)

        # --- A) Radial 분포 (면적 보정 밀도 비율) ---
        # 타원 정규화: x, y 반지름이 다르므로 각각 정규화
        r_norm = np.sqrt(((dx - cx) / Rx)**2 + ((dy - cy) / Ry)**2)

        row["radial_mean"] = r_norm.mean()
        row["radial_std"] = r_norm.std()

        # 면적 보정 ring_concentration: 외곽 밀도 / 내부 밀도
        # 외곽 링(r > 0.7): 면적 비율 = 1 - 0.7² = 0.51
        # 내부(r ≤ 0.7): 면적 비율 = 0.7² = 0.49
        n_edge = (r_norm > 0.7).sum()
        n_inner = (r_norm <= 0.7).sum()
        edge_density = n_edge / 0.51 if n_edge > 0 else 0
        inner_density = n_inner / 0.49 if n_inner > 0 else 1e-10
        row["ring_concentration"] = edge_density / max(inner_density, 1e-10)

        # center_concentration: 중심 밀도 / 외부 밀도
        n_center = (r_norm < 0.4).sum()
        n_outer = (r_norm >= 0.4).sum()
        center_density = n_center / 0.16 if n_center > 0 else 0  # 0.4² = 0.16
        outer_density = n_outer / 0.84 if n_outer > 0 else 1e-10
        row["center_concentration"] = center_density / max(outer_density, 1e-10)

        # --- B) PCA elongation (정규화 좌표에서 계산) ---
        # x/y 범위가 다르므로 정규화하여 PCA → 실제 형상 반영
        coords = np.column_stack([(dx - cx) / Rx, (dy - cy) / Ry])
        if coords.shape[0] >= 2:
            cov = np.cov(coords.T)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)[::-1]  # 내림차순
            row["elongation"] = eigvals[0] / max(eigvals[1], 1e-10)

            # 주축 각도
            _, eigvecs = np.linalg.eigh(cov)
            v1 = eigvecs[:, -1]  # 최대 고유값에 대응하는 벡터
            row["pca_angle"] = np.degrees(np.arctan2(v1[1], v1[0]))
        else:
            row["elongation"] = np.nan
            row["pca_angle"] = np.nan

        # --- C) HDBSCAN 클러스터링 ---
        row.update(_compute_cluster_features(dx, dy))

        # --- D) 분산 정도 (정규화 좌표 기반) ---
        from scipy.spatial.distance import pdist
        norm_coords = np.column_stack([(dx - cx) / Rx, (dy - cy) / Ry])
        if n_defect <= 200:
            pairwise = pdist(norm_coords)
        else:
            rng = np.random.RandomState(SEED)
            idx = rng.choice(n_defect, 200, replace=False)
            pairwise = pdist(norm_coords[idx])
        row["dispersion"] = pairwise.mean() if len(pairwise) > 0 else np.nan

        results.append(row)

    wafer_features = pd.DataFrame(results)

    print(f"피처 추출 완료: {len(wafer_features)} wafers")
    print(f"  불량 die ≥ 3인 wafer: {wafer_features['n_defect'].ge(3).sum()}")
    print(f"  평균 defect_rate: {wafer_features['defect_rate'].mean():.1%}")

    return wafer_features


def _compute_cluster_features(dx, dy):
    """HDBSCAN 클러스터링 후 피처 반환"""
    n_defect = len(dx)
    result = {"n_clusters": 0, "largest_cluster_ratio": np.nan, "noise_ratio": np.nan}

    if n_defect < 5:
        return result

    try:
        from hdbscan import HDBSCAN
        min_size = max(5, int(n_defect * 0.05))
        clusterer = HDBSCAN(min_cluster_size=min_size, min_samples=3)
        labels = clusterer.fit_predict(np.column_stack([dx, dy]))
    except ImportError:
        # hdbscan 미설치 시 sklearn DBSCAN fallback
        from sklearn.cluster import DBSCAN
        # step size 추정 (인접 die 간격)
        dx_sorted = np.sort(np.unique(dx))
        eps = np.min(np.diff(dx_sorted)) * 1.5 if len(dx_sorted) > 1 else 2.0
        clusterer = DBSCAN(eps=eps, min_samples=3)
        labels = clusterer.fit_predict(np.column_stack([dx, dy]))

    unique_labels = set(labels)
    unique_labels.discard(-1)
    n_clusters = len(unique_labels)

    result["n_clusters"] = n_clusters
    result["noise_ratio"] = (labels == -1).mean()

    if n_clusters > 0:
        from collections import Counter
        counts = Counter(labels)
        counts.pop(-1, None)
        largest = max(counts.values())
        result["largest_cluster_ratio"] = largest / n_defect

    return result


# ─── 2. 규칙 기반 패턴 분류 ──────────────────────────────
def classify_pattern(row):
    """
    단일 웨이퍼의 피처로 패턴 유형을 판별한다.
    우선순위 기반 의사결정 트리.

    Parameters
    ----------
    row : dict or Series
        extract_wafer_features()의 한 행

    Returns
    -------
    str : 패턴 유형 (PATTERN_TYPES 중 하나)
    """
    # 1) 불량 거의 없음
    if row["defect_rate"] < 0.05 or row["n_defect"] < 5:
        return "None"

    # 2) Edge: 외곽 링 밀도가 내부 대비 높음 (면적 보정)
    ring_c = row.get("ring_concentration", np.nan)
    if not np.isnan(ring_c) and ring_c > 1.5:
        return "Edge"

    # 3) Center: 중심부 밀도가 외부 대비 높음 (면적 보정)
    center_c = row.get("center_concentration", np.nan)
    if not np.isnan(center_c) and center_c > 1.5:
        return "Center"

    # 4) Scratch: PCA elongation이 매우 높음 (정규화 좌표 기준)
    elongation = row.get("elongation", np.nan)
    if not np.isnan(elongation) and elongation > 4.0:
        return "Scratch"

    # 5) Local: 소수 클러스터에 불량이 집중
    n_clusters = row.get("n_clusters", 0)
    lcr = row.get("largest_cluster_ratio", np.nan)
    if n_clusters >= 1 and not np.isnan(lcr) and lcr > 0.5 and n_clusters <= 3:
        return "Local"

    # 6) 나머지 = Random
    return "Random"


def classify_all_wafers(xs_parsed, ys_train):
    """
    전체 웨이퍼에 대해 패턴 분류를 수행한다.

    Parameters
    ----------
    xs_parsed : DataFrame
        parse_wafer_coords() 결과
    ys_train : DataFrame
        train Y 데이터

    Returns
    -------
    wafer_features : DataFrame
        웨이퍼별 피처 + pattern 컬럼 추가
    """
    wafer_features = extract_wafer_features(xs_parsed, ys_train)
    wafer_features["pattern"] = wafer_features.apply(classify_pattern, axis=1)

    # 분류 결과 요약
    print("\n" + "=" * 60)
    print("웨이퍼 불량 패턴 분류 결과")
    print("=" * 60)

    counts = wafer_features["pattern"].value_counts()
    total = len(wafer_features)
    for ptype in PATTERN_TYPES:
        n = counts.get(ptype, 0)
        pct = n / total * 100
        print(f"  {ptype:>8}: {n:>4}장 ({pct:>5.1f}%)  — {PATTERN_DESC[ptype]}")
    print(f"  {'합계':>8}: {total:>4}장")

    return wafer_features


# ─── 3. 패턴 분포 통계 ───────────────────────────────────
def print_pattern_stats(wafer_features):
    """
    패턴 유형별 상세 통계 출력

    Parameters
    ----------
    wafer_features : DataFrame
        classify_all_wafers() 결과
    """
    print("\n" + "=" * 75)
    print("패턴 유형별 상세 통계")
    print("=" * 75)
    print(f"  {'패턴':>8}  {'n':>4}  {'불량률':>8}  {'mean_health':>12}  "
          f"{'ring_conc':>10}  {'center_conc':>12}  {'elongation':>11}")
    print("-" * 85)

    for ptype in PATTERN_TYPES:
        subset = wafer_features[wafer_features["pattern"] == ptype]
        if len(subset) == 0:
            continue
        print(f"  {ptype:>8}  {len(subset):>4}  "
              f"{subset['defect_rate'].mean():>8.1%}  "
              f"{subset['mean_health'].mean():>12.6f}  "
              f"{subset['ring_concentration'].mean():>10.2f}  "
              f"{subset['center_concentration'].mean():>12.2f}  "
              f"{subset['elongation'].mean():>11.1f}")


# ─── 4. 시각화 ───────────────────────────────────────────
def plot_pattern_distribution(wafer_features):
    """
    패턴 분포 파이차트 + 패턴별 평균 health/불량률 막대그래프

    Parameters
    ----------
    wafer_features : DataFrame
        classify_all_wafers() 결과
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    counts = wafer_features["pattern"].value_counts()
    # PATTERN_TYPES 순서로 정렬 (존재하는 것만)
    ordered = [p for p in PATTERN_TYPES if p in counts.index]
    counts = counts[ordered]
    colors = [PATTERN_COLORS[p] for p in ordered]

    # 1) 파이차트
    axes[0].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=colors, startangle=90, textprops={"fontsize": 10})
    axes[0].set_title("패턴 유형 분포", fontsize=12, fontweight="bold")

    # 2) 패턴별 평균 불량률
    pattern_stats = wafer_features.groupby("pattern").agg(
        mean_defect_rate=("defect_rate", "mean"),
        mean_health=("mean_health", "mean"),
    )
    pattern_stats = pattern_stats.reindex([p for p in PATTERN_TYPES if p in pattern_stats.index])

    bar_colors = [PATTERN_COLORS[p] for p in pattern_stats.index]
    axes[1].bar(range(len(pattern_stats)), pattern_stats["mean_defect_rate"] * 100,
                color=bar_colors, edgecolor="black", alpha=0.85)
    axes[1].set_xticks(range(len(pattern_stats)))
    axes[1].set_xticklabels(pattern_stats.index, fontsize=9)
    axes[1].set_ylabel("평균 불량률 (%)")
    axes[1].set_title("패턴별 평균 불량률", fontsize=12, fontweight="bold")

    # 3) 패턴별 평균 health
    axes[2].bar(range(len(pattern_stats)), pattern_stats["mean_health"],
                color=bar_colors, edgecolor="black", alpha=0.85)
    axes[2].set_xticks(range(len(pattern_stats)))
    axes[2].set_xticklabels(pattern_stats.index, fontsize=9)
    axes[2].set_ylabel("평균 health")
    axes[2].set_title("패턴별 평균 health", fontsize=12, fontweight="bold")

    plt.suptitle("웨이퍼 불량 패턴 분류 — 분포 및 통계",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_pattern_examples(xs_parsed, ys_train, wafer_features, n_per_pattern=2):
    """
    패턴 유형별 대표 웨이퍼를 시각화한다.
    각 패턴에서 해당 패턴 특성이 가장 뚜렷한 웨이퍼를 선택.

    Parameters
    ----------
    xs_parsed : DataFrame
        parse_wafer_coords() 결과
    ys_train : DataFrame
        train Y 데이터
    wafer_features : DataFrame
        classify_all_wafers() 결과
    n_per_pattern : int
        패턴당 시각화할 웨이퍼 수 (기본 2)
    """
    merged = xs_parsed.merge(ys_train[[KEY_COL, TARGET_COL]], on=KEY_COL, how="inner")

    # 패턴별 대표 웨이퍼 선택 (None 제외)
    display_patterns = [p for p in PATTERN_TYPES if p != "None"]
    selected = []
    for ptype in display_patterns:
        subset = wafer_features[wafer_features["pattern"] == ptype].copy()
        if len(subset) == 0:
            continue

        # 각 패턴의 대표성 점수로 정렬
        if ptype == "Center":
            subset = subset.sort_values("center_concentration", ascending=False)
        elif ptype == "Edge":
            subset = subset.sort_values("ring_concentration", ascending=False)
        elif ptype == "Scratch":
            subset = subset.sort_values("elongation", ascending=False)
        elif ptype == "Local":
            subset = subset.sort_values("largest_cluster_ratio", ascending=False)
        elif ptype == "Random":
            subset = subset.sort_values("defect_rate", ascending=False)

        for wid in subset["wafer_id"].head(n_per_pattern):
            selected.append((ptype, wid))

    if len(selected) == 0:
        print("시각화할 패턴이 없습니다.")
        return

    n = len(selected)
    n_cols = min(4, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    axes = np.array(axes).flatten()

    for idx, (ptype, wid) in enumerate(selected):
        wf = merged[merged["wafer_id"] == wid].copy()
        ax = axes[idx]
        _draw_pattern_wafer(ax, wf, wid, ptype)

    for j in range(len(selected), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("웨이퍼 불량 패턴 분류 — 유형별 대표 웨이퍼",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


def _draw_pattern_wafer(ax, wf, wafer_id, pattern_type):
    """
    패턴 분류 결과를 포함한 웨이퍼 시각화 (사각형 타일)
    Ellipse + aspect='auto'로 x/y 범위 달라도 원형 표시.

    Parameters
    ----------
    ax : matplotlib Axes
    wf : DataFrame (die_x, die_y, health)
    wafer_id : str
    pattern_type : str
    """
    from matplotlib.patches import Ellipse

    # die 간격 계산
    dx_sorted = np.sort(wf["die_x"].unique())
    dy_sorted = np.sort(wf["die_y"].unique())
    step_x = np.min(np.diff(dx_sorted)) if len(dx_sorted) > 1 else 1
    step_y = np.min(np.diff(dy_sorted)) if len(dy_sorted) > 1 else 1
    gap = 0.06
    die_w = step_x * (1 - gap)
    die_h = step_y * (1 - gap)

    # 웨이퍼 중심 / x·y 반지름 (독립)
    cx = (wf["die_x"].max() + wf["die_x"].min()) / 2
    cy = (wf["die_y"].max() + wf["die_y"].min()) / 2
    rx = (wf["die_x"].max() - wf["die_x"].min()) / 2 + step_x / 2
    ry = (wf["die_y"].max() - wf["die_y"].min()) / 2 + step_y / 2
    pad = 1.08

    # 배경 (Ellipse)
    ax.set_facecolor("#e8e8e8")
    wafer_bg = Ellipse((cx, cy), rx * 2 * pad, ry * 2 * pad,
                        fill=True, facecolor="#f8f8f8",
                        edgecolor="#444444", linewidth=2.5, zorder=0)
    ax.add_patch(wafer_bg)

    # 정상/불량 분리
    zero_mask = wf[TARGET_COL] == 0
    wf_zero = wf[zero_mask]
    wf_defect = wf[~zero_mask]

    # 정상 die (연한 회색 — 불량 패턴 색과 확실히 구분)
    normal_patches = []
    for _, row in wf_zero.iterrows():
        rect = Rectangle((row["die_x"] - die_w / 2, row["die_y"] - die_h / 2),
                          die_w, die_h)
        normal_patches.append(rect)
    if normal_patches:
        pc_normal = PatchCollection(normal_patches, facecolor="#d0d0d0",
                                    edgecolor="#bbbbbb", linewidth=0.3,
                                    alpha=0.7, zorder=2)
        ax.add_collection(pc_normal)

    # 불량 die — 패턴 색상
    pcolor = PATTERN_COLORS.get(pattern_type, "#e74c3c")
    if len(wf_defect) > 0:
        defect_patches = []
        for _, row in wf_defect.iterrows():
            rect = Rectangle((row["die_x"] - die_w / 2, row["die_y"] - die_h / 2),
                              die_w, die_h)
            defect_patches.append(rect)
        pc_defect = PatchCollection(defect_patches, facecolor=pcolor,
                                    edgecolor="#444444", linewidth=0.4,
                                    alpha=0.9, zorder=3)
        ax.add_collection(pc_defect)

    # 웨이퍼 윤곽
    wafer_outline = Ellipse((cx, cy), rx * 2 * pad, ry * 2 * pad,
                            fill=False, edgecolor="#333333", linewidth=2.5, zorder=5)
    ax.add_patch(wafer_outline)

    # 웨이퍼 원 밖의 die 클리핑 — 원 밖 포인트 제거
    clip_ellipse = Ellipse((cx, cy), rx * 2 * pad, ry * 2 * pad,
                            transform=ax.transData)
    for coll in ax.collections:
        coll.set_clip_path(clip_ellipse)

    # 축 설정
    mx = rx * 0.08
    my = ry * 0.08
    ax.set_xlim(cx - rx * pad - mx, cx + rx * pad + mx)
    ax.set_ylim(cy - ry * pad - my, cy + ry * pad + my)
    ax.set_aspect("auto")
    ax.set_xlabel("die_x", fontsize=9)
    ax.set_ylabel("die_y", fontsize=9)

    # 제목
    lot, wf_no = wafer_id.rsplit("_", 1)
    defect_rate = (~zero_mask).mean() * 100
    n_defect = (~zero_mask).sum()
    ax.set_title(f"[{pattern_type}] Lot {lot} / Wafer {wf_no}\n"
                 f"불량률 {defect_rate:.1f}% ({n_defect}/{len(wf)})",
                 fontsize=10, fontweight="bold", pad=10,
                 color=pcolor)

    # 범례
    legend_elements = [
        Patch(facecolor="#d0d0d0", edgecolor="#bbbbbb", label="정상 (Y=0)"),
        Patch(facecolor=pcolor, edgecolor="#444444",
              label=f"불량 [{pattern_type}] (n={n_defect})"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="upper left",
              framealpha=0.9, edgecolor="#cccccc")


def plot_pattern_feature_space(wafer_features):
    """
    패턴 분류에 사용된 핵심 피처 공간에서 웨이퍼들의 분포를 scatter로 표시.
    분류 결과가 합리적인지 시각적으로 검증.

    Parameters
    ----------
    wafer_features : DataFrame
        classify_all_wafers() 결과
    """
    # 분석 대상: defect >= 3인 웨이퍼만
    df = wafer_features[wafer_features["n_defect"] >= 3].copy()
    if len(df) == 0:
        print("분석 대상 웨이퍼 없음")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) ring_concentration vs center_concentration
    for ptype in PATTERN_TYPES:
        mask = df["pattern"] == ptype
        if mask.sum() == 0:
            continue
        axes[0].scatter(df.loc[mask, "ring_concentration"],
                        df.loc[mask, "center_concentration"],
                        c=PATTERN_COLORS[ptype], label=ptype, alpha=0.6, s=15,
                        edgecolors="none")
    axes[0].set_xlabel("ring_concentration (외곽밀도/내부밀도)")
    axes[0].set_ylabel("center_concentration (중심밀도/외부밀도)")
    axes[0].set_title("Edge vs Center 밀도 비율")
    axes[0].axvline(1.5, color="blue", linestyle="--", alpha=0.4, label="Edge thr=1.5")
    axes[0].axhline(1.5, color="red", linestyle="--", alpha=0.4, label="Center thr=1.5")
    axes[0].legend(fontsize=7, markerscale=2)

    # 2) elongation vs defect_rate
    for ptype in PATTERN_TYPES:
        mask = df["pattern"] == ptype
        if mask.sum() == 0:
            continue
        axes[1].scatter(df.loc[mask, "elongation"], df.loc[mask, "defect_rate"] * 100,
                        c=PATTERN_COLORS[ptype], label=ptype, alpha=0.6, s=15,
                        edgecolors="none")
    axes[1].set_xlabel("PCA elongation")
    axes[1].set_ylabel("defect_rate (%)")
    axes[1].set_title("Elongation (Scratch 지표)")
    axes[1].axvline(4.0, color="orange", linestyle="--", alpha=0.5, label="threshold=4.0")
    axes[1].legend(fontsize=8, markerscale=2)

    # 3) largest_cluster_ratio vs n_clusters
    for ptype in PATTERN_TYPES:
        mask = df["pattern"] == ptype
        if mask.sum() == 0:
            continue
        axes[2].scatter(df.loc[mask, "largest_cluster_ratio"],
                        df.loc[mask, "n_clusters"],
                        c=PATTERN_COLORS[ptype], label=ptype, alpha=0.6, s=15,
                        edgecolors="none")
    axes[2].set_xlabel("largest_cluster_ratio")
    axes[2].set_ylabel("n_clusters")
    axes[2].set_title("클러스터 피처 (Local 지표)")
    axes[2].legend(fontsize=8, markerscale=2)

    plt.suptitle("패턴 분류 피처 공간 — 분류 결과 검증",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()
