"""
EDA 모듈: Wafer 공간 패턴 심화 분석
- radial distance(웨이퍼 중심~edge) vs health → edge effect 정량화
- 웨이퍼를 ring/quadrant로 나눠 구역별 불량률 차이
- 공간 자기상관(Join Count) → 불량이 공간적으로 클러스터링되는지
- Stage 4.5 (공간 피처: radial_dist, is_edge 등) 설계 근거
- 노트북에서 import eda_spatial as sp 로 사용
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats
from utils.config import KEY_COL, DIE_KEY_COL, TARGET_COL, SPLIT_COL, SEED


def _parse_coords(xs_dict, ys_train):
    """
    Train die에서 좌표 파싱 + radial distance 계산 + target merge

    Returns
    -------
    die_df : DataFrame  (die-level, 좌표+health)
    unit_df : DataFrame  (unit-level, 좌표 집계+health)
    """
    xs_train = xs_dict["train"]
    parts = xs_train[DIE_KEY_COL].str.split("_", expand=True)

    die_df = xs_train[[KEY_COL, DIE_KEY_COL]].copy()
    die_df["lot"] = parts[0]
    die_df["wafer_no"] = parts[1].astype(int)
    die_df["wafer_id"] = die_df["lot"] + "_" + parts[1]
    die_df["die_x"] = parts[2].astype(int)
    die_df["die_y"] = parts[3].astype(int)

    # 웨이퍼별 중심 계산 후 radial distance
    wafer_centers = die_df.groupby("wafer_id").agg(
        cx=("die_x", "mean"), cy=("die_y", "mean")
    )
    die_df = die_df.merge(wafer_centers, on="wafer_id", how="left")
    die_df["radial_dist"] = np.sqrt(
        (die_df["die_x"] - die_df["cx"])**2 +
        (die_df["die_y"] - die_df["cy"])**2
    )

    # target merge
    die_df = die_df.merge(ys_train[[KEY_COL, TARGET_COL]], on=KEY_COL, how="inner")

    # unit-level 집계
    unit_df = die_df.groupby(KEY_COL).agg(
        radial_mean=("radial_dist", "mean"),
        radial_max=("radial_dist", "max"),
        radial_std=("radial_dist", "std"),
        die_x_mean=("die_x", "mean"),
        die_y_mean=("die_y", "mean"),
        health=(TARGET_COL, "first"),
        wafer_id=("wafer_id", "first"),
    ).reset_index()

    return die_df, unit_df


def radial_analysis(xs_dict, ys_train, n_bins=10):
    """
    Radial distance vs health 분석
    - radial distance 구간별 불량률/평균 health
    - radial distance와 health의 상관

    Parameters
    ----------
    xs_dict : dict
    ys_train : DataFrame
    n_bins : int  - radial distance 구간 수

    Returns
    -------
    die_df : DataFrame
    unit_df : DataFrame
    """
    die_df, unit_df = _parse_coords(xs_dict, ys_train)

    # unit-level 상관
    r_mean, p_mean = sp_stats.pearsonr(unit_df["radial_mean"], unit_df["health"])
    r_max, p_max = sp_stats.pearsonr(unit_df["radial_max"], unit_df["health"])
    rho_mean, _ = sp_stats.spearmanr(unit_df["radial_mean"], unit_df["health"])

    print("=" * 60)
    print("Radial Distance → Target 상관")
    print("=" * 60)
    print(f"  radial_mean vs health: Pearson r={r_mean:+.4f} (p={p_mean:.2e}), Spearman ρ={rho_mean:+.4f}")
    print(f"  radial_max  vs health: Pearson r={r_max:+.4f} (p={p_max:.2e})")

    # 구간별 불량률
    die_df["radial_bin"] = pd.cut(die_df["radial_dist"], bins=n_bins, labels=False)
    bin_stats = die_df.groupby("radial_bin").agg(
        n_dies=("radial_dist", "size"),
        mean_radial=("radial_dist", "mean"),
        defect_rate=(TARGET_COL, lambda x: (x > 0).mean() * 100),
        mean_health=(TARGET_COL, "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) 구간별 불량률
    axes[0].bar(bin_stats["mean_radial"], bin_stats["defect_rate"],
                width=bin_stats["mean_radial"].diff().median() * 0.8,
                color="coral", edgecolor="black", alpha=0.8)
    axes[0].set_xlabel("Radial Distance (웨이퍼 중심으로부터)")
    axes[0].set_ylabel("불량률 (%)")
    axes[0].set_title(f"Radial Distance 구간별 불량률 ({n_bins}구간)")

    # 2) 구간별 평균 health
    axes[1].bar(bin_stats["mean_radial"], bin_stats["mean_health"],
                width=bin_stats["mean_radial"].diff().median() * 0.8,
                color="steelblue", edgecolor="black", alpha=0.8)
    axes[1].set_xlabel("Radial Distance")
    axes[1].set_ylabel("평균 health")
    axes[1].set_title("Radial Distance 구간별 평균 health")

    # 3) scatter (unit-level)
    sample = unit_df.sample(min(5000, len(unit_df)), random_state=SEED)
    axes[2].scatter(sample["radial_mean"], sample["health"],
                    alpha=0.15, s=5, color="steelblue")
    axes[2].set_xlabel("unit 평균 Radial Distance")
    axes[2].set_ylabel("health")
    axes[2].set_title(f"Radial Distance vs Health (r={r_mean:+.4f})")

    plt.suptitle("Radial Distance (Edge Effect) 분석", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    return die_df, unit_df


def zone_analysis(die_df, n_rings=3, n_quadrants=4):
    """
    웨이퍼를 ring(동심원) × quadrant(사분면)로 나눠 구역별 불량률 비교

    Parameters
    ----------
    die_df : DataFrame  (radial_analysis 반환값)
    n_rings : int  - 동심원 구역 수 (center, middle, edge)
    n_quadrants : int  - 사분면 수
    """
    # Ring 구분 (radial distance 분위수 기반)
    die_df = die_df.copy()
    ring_labels = ["center", "middle", "edge"] if n_rings == 3 else \
                  [f"ring_{i}" for i in range(n_rings)]
    die_df["ring"] = pd.qcut(die_df["radial_dist"], q=n_rings, labels=ring_labels, duplicates="drop")

    # Quadrant 구분 (각도 기반)
    die_df["angle"] = np.arctan2(
        die_df["die_y"] - die_df["cy"],
        die_df["die_x"] - die_df["cx"]
    )
    quad_labels = ["Q1(↗)", "Q2(↖)", "Q3(↙)", "Q4(↘)"]
    die_df["quadrant"] = pd.cut(die_df["angle"], bins=n_quadrants,
                                labels=quad_labels[:n_quadrants])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) Ring별 불량률
    ring_stats = die_df.groupby("ring", observed=True).agg(
        defect_rate=(TARGET_COL, lambda x: (x > 0).mean() * 100),
        mean_health=(TARGET_COL, "mean"),
        n_dies=(TARGET_COL, "size"),
    )
    ring_stats["defect_rate"].plot(kind="bar", ax=axes[0], color="coral",
                                   edgecolor="black", alpha=0.8)
    axes[0].set_title("Ring(동심원) 구역별 불량률")
    axes[0].set_ylabel("불량률 (%)")
    axes[0].tick_params(axis="x", rotation=0)

    # 2) Quadrant별 불량률
    quad_stats = die_df.groupby("quadrant", observed=True).agg(
        defect_rate=(TARGET_COL, lambda x: (x > 0).mean() * 100),
    )
    quad_stats["defect_rate"].plot(kind="bar", ax=axes[1], color="mediumseagreen",
                                   edgecolor="black", alpha=0.8)
    axes[1].set_title("Quadrant(사분면)별 불량률")
    axes[1].set_ylabel("불량률 (%)")
    axes[1].tick_params(axis="x", rotation=0)

    # 3) Ring × Quadrant 히트맵
    cross = die_df.groupby(["ring", "quadrant"], observed=True)[TARGET_COL].apply(
        lambda x: (x > 0).mean() * 100
    ).unstack(fill_value=0)
    sns.heatmap(cross, annot=True, fmt=".1f", cmap="YlOrRd", ax=axes[2],
                cbar_kws={"label": "불량률(%)"})
    axes[2].set_title("Ring × Quadrant 불량률 히트맵")

    plt.suptitle("웨이퍼 구역별 불량률 분석", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    # 텍스트 요약
    print("\nRing별 통계:")
    print(ring_stats.to_string())
    print("\nQuadrant별 통계:")
    print(quad_stats.to_string())


def spatial_autocorrelation(die_df, n_wafers=6):
    """
    Join Count 기반 공간 자기상관 분석
    - 인접 die가 같은 상태(정상-정상 or 불량-불량)인 비율 vs 랜덤 기대값
    - 불량이 공간적으로 클러스터링되면 → 공간 피처 가치 높음

    Parameters
    ----------
    die_df : DataFrame
    n_wafers : int  - 분석할 wafer 수 (불량률 높은 순)
    """
    die_df = die_df.copy()
    die_df["is_defect"] = (die_df[TARGET_COL] > 0).astype(int)

    # 불량률 높은 wafer 선택
    wafer_defect = die_df.groupby("wafer_id")["is_defect"].mean().sort_values(ascending=False)
    top_wafers = wafer_defect.head(n_wafers).index

    results = []
    for wid in top_wafers:
        wf = die_df[die_df["wafer_id"] == wid].copy()
        n_total = len(wf)
        n_defect = wf["is_defect"].sum()
        if n_defect == 0 or n_defect == n_total:
            continue

        # 인접 die 쌍 찾기 (Manhattan distance == 1)
        coords = wf[["die_x", "die_y"]].values
        defects = wf["is_defect"].values

        n_pairs = 0
        n_same = 0
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist = abs(coords[i, 0] - coords[j, 0]) + abs(coords[i, 1] - coords[j, 1])
                if dist <= 2:  # 인접 (가까운 die)
                    n_pairs += 1
                    if defects[i] == defects[j]:
                        n_same += 1

        if n_pairs == 0:
            continue

        observed_ratio = n_same / n_pairs

        # 랜덤 기대값: P(같은 상태) = P(둘 다 정상) + P(둘 다 불량)
        p_defect = n_defect / n_total
        p_normal = 1 - p_defect
        expected_ratio = p_normal**2 + p_defect**2

        results.append({
            "wafer_id": wid,
            "n_dies": n_total,
            "defect_rate": p_defect * 100,
            "n_pairs": n_pairs,
            "observed_same": observed_ratio,
            "expected_same": expected_ratio,
            "clustering_ratio": observed_ratio / expected_ratio if expected_ratio > 0 else 1.0,
        })

    if not results:
        print("분석 가능한 wafer가 없습니다.")
        return

    res_df = pd.DataFrame(results)

    print("=" * 70)
    print("공간 자기상관 분석 (Join Count)")
    print("=" * 70)
    print("  clustering_ratio > 1 → 불량이 공간적으로 클러스터링")
    print("  clustering_ratio ≈ 1 → 불량이 랜덤 분포")
    print()
    print(f"  {'Wafer':>12}  {'Dies':>5}  {'불량%':>6}  {'Pairs':>6}  "
          f"{'관측':>6}  {'기대':>6}  {'C.Ratio':>8}")
    print("-" * 65)
    for _, row in res_df.iterrows():
        print(f"  {row['wafer_id']:>12}  {int(row['n_dies']):>5}  {row['defect_rate']:>6.1f}  "
              f"{int(row['n_pairs']):>6}  {row['observed_same']:>6.3f}  "
              f"{row['expected_same']:>6.3f}  {row['clustering_ratio']:>8.3f}")

    mean_cr = res_df["clustering_ratio"].mean()
    print(f"\n  평균 Clustering Ratio: {mean_cr:.3f}")
    if mean_cr > 1.05:
        print("  → 불량이 공간적으로 클러스터링되는 경향 → 공간 피처 가치 높음")
    else:
        print("  → 불량이 거의 랜덤 분포 → 공간 피처 효과 제한적일 수 있음")


def nnr_analysis(die_df, sigma=3.0, n_wafers=10):
    """
    NNR(Nearest Neighbor Residual) 분석: health 값의 공간 잔차

    die의 health에서 인접 die의 가우시안 가중 평균 health를 뺀 잔차를 계산.
    잔차가 크면 = 주변과 다른 불량 패턴 → 공간 피처의 추가 정보량 확인.
    논문 5-3 (NNR), 5-4 (GPR) 근거.

    Parameters
    ----------
    die_df : DataFrame  - radial_analysis() 반환값 (die_x, die_y, wafer_id, health 포함)
    sigma : float  - 가우시안 가중치 bandwidth (기본 3.0)
    n_wafers : int  - 분석할 wafer 수 (불량률 높은 순, 기본 10)
    """
    die_df = die_df.copy()
    die_df["is_defect"] = (die_df[TARGET_COL] > 0).astype(int)

    # 불량률 높은 wafer 선택
    wafer_defect = die_df.groupby("wafer_id")["is_defect"].mean().sort_values(ascending=False)
    top_wafers = wafer_defect.head(n_wafers).index

    all_residuals = []

    for wid in top_wafers:
        wf = die_df[die_df["wafer_id"] == wid].copy()
        coords = wf[["die_x", "die_y"]].values.astype(float)
        health_vals = wf[TARGET_COL].values.astype(float)
        n = len(coords)
        residuals = np.full(n, np.nan)

        for i in range(n):
            dists = np.sqrt(np.sum((coords - coords[i]) ** 2, axis=1))
            mask = (dists > 0) & (dists <= max(3, 3 * sigma)) & ~np.isnan(health_vals)

            if mask.sum() == 0:
                residuals[i] = np.nan
                continue

            weights = np.exp(-dists[mask] ** 2 / (2 * sigma ** 2))
            weighted_avg = np.average(health_vals[mask], weights=weights)
            residuals[i] = health_vals[i] - weighted_avg

        wf["nnr_residual"] = residuals
        all_residuals.append(wf)

    nnr_df = pd.concat(all_residuals, ignore_index=True).dropna(subset=["nnr_residual"])

    # 통계
    abs_resid = nnr_df["nnr_residual"].abs()
    r_val, p_val = sp_stats.pearsonr(abs_resid, nnr_df[TARGET_COL])
    rho, _ = sp_stats.spearmanr(abs_resid, nnr_df[TARGET_COL])

    print("=" * 60)
    print(f"NNR(Nearest Neighbor Residual) 분석 (sigma={sigma})")
    print("=" * 60)
    print(f"분석 wafer: {n_wafers}개, die: {len(nnr_df):,}개")
    print(f"잔차 mean: {nnr_df['nnr_residual'].mean():.6f}")
    print(f"잔차 std : {nnr_df['nnr_residual'].std():.6f}")
    print(f"|잔차| vs health: Pearson r={r_val:+.4f} (p={p_val:.2e}), Spearman ρ={rho:+.4f}")
    if abs(r_val) > 0.05:
        print("→ 공간 잔차가 불량과 관련 → NNR 기반 피처 가치 있음")
    else:
        print("→ 공간 잔차와 불량 관계 약함 (health 자체가 unit level이라 die level에서 약할 수 있음)")

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) 잔차 vs health scatter
    sample = nnr_df.sample(min(5000, len(nnr_df)), random_state=SEED)
    axes[0].scatter(sample["nnr_residual"].abs(), sample[TARGET_COL],
                    alpha=0.15, s=5, color="steelblue")
    axes[0].set_xlabel("|NNR Residual|")
    axes[0].set_ylabel("health")
    axes[0].set_title(f"|NNR Residual| vs Health (r={r_val:+.4f})")

    # 2) 잔차 분포 by Y=0 vs Y>0
    zero_resid = nnr_df.loc[nnr_df[TARGET_COL] == 0, "nnr_residual"].abs()
    pos_resid = nnr_df.loc[nnr_df[TARGET_COL] > 0, "nnr_residual"].abs()
    axes[1].hist(zero_resid, bins=40, alpha=0.6, color="skyblue",
                 label=f"Y=0 (n={len(zero_resid):,})", density=True)
    axes[1].hist(pos_resid, bins=40, alpha=0.6, color="salmon",
                 label=f"Y>0 (n={len(pos_resid):,})", density=True)
    axes[1].set_xlabel("|NNR Residual|")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Y=0 vs Y>0: NNR 잔차 크기 분포")
    axes[1].legend(fontsize=9)

    # 3) 잔차 구간별 불량률
    nnr_df["resid_bin"] = pd.qcut(nnr_df["nnr_residual"].abs(), q=10,
                                   labels=False, duplicates="drop")
    bin_stats = nnr_df.groupby("resid_bin").agg(
        mean_resid=("nnr_residual", lambda x: x.abs().mean()),
        defect_rate=(TARGET_COL, lambda x: (x > 0).mean() * 100),
    ).reset_index()
    bar_width = bin_stats["mean_resid"].diff().median() * 0.8
    if pd.isna(bar_width) or bar_width <= 0:
        bar_width = 0.001
    axes[2].bar(bin_stats["mean_resid"], bin_stats["defect_rate"],
                width=bar_width,
                color="coral", edgecolor="black", alpha=0.8)
    axes[2].set_xlabel("|NNR Residual| 구간 평균")
    axes[2].set_ylabel("불량률 (%)")
    axes[2].set_title("|NNR Residual| 구간별 불량률")

    plt.suptitle("NNR 공간 잔차 분석", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()