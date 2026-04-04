"""
EDA 모듈: 인접 Die 불량 연관성 분석
- 같은 웨이퍼 내 인접 die의 WT 측정값 유사도 분석
- 인접 die의 불량률(health)이 중심 die 불량에 미치는 영향
- 공간 자기상관(Moran's I)으로 불량 클러스터링 정량화
- 근거: Kang & Cho (2015) SK Hynix 공동 논문 — 인접 die 불량률이 예측 피처로 유효
- 노트북에서 import eda_neighbor_die as nd 로 사용
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from scipy.spatial.distance import cdist
from utils.config import KEY_COL, DIE_KEY_COL, TARGET_COL, SEED


def _parse_wafer_data(xs_dict, ys_train, feat_cols, n_feats=20):
    """
    Train die에서 좌표 파싱 + target merge + 상위 feature 선택

    Returns
    -------
    die_df : DataFrame  (die-level, 좌표 + feature + health)
    selected_feats : list
    """
    xs_train = xs_dict["train"]

    parts = xs_train[DIE_KEY_COL].str.split("_", expand=True)
    die_df = xs_train[[KEY_COL, DIE_KEY_COL]].copy()
    die_df["wafer_id"] = parts[0] + "_" + parts[1]
    die_df["die_x"] = parts[2].astype(int)
    die_df["die_y"] = parts[3].astype(int)

    # target 상관 상위 feature 선택
    xs_unit = xs_dict['train_unit_mean'] if 'train_unit_mean' in xs_dict else xs_train.groupby(KEY_COL)[feat_cols].mean()
    merged_tmp = xs_unit.merge(ys_train, left_index=True, right_on=KEY_COL, how="inner")
    corr = merged_tmp[feat_cols].corrwith(merged_tmp[TARGET_COL]).abs().sort_values(ascending=False)
    selected_feats = corr.head(n_feats).index.tolist()

    for f in selected_feats:
        die_df[f] = xs_train[f].values
    die_df = die_df.merge(ys_train[[KEY_COL, TARGET_COL]], on=KEY_COL, how="inner")

    print(f"파싱 완료: {len(die_df):,} dies, {die_df['wafer_id'].nunique()} wafers")
    print(f"선택 feature: {len(selected_feats)}개 (target |r| 상위)")

    return die_df, selected_feats


def _auto_radius(die_df):
    """
    die 좌표의 최소 간격을 자동 감지하여 인접 판정 radius를 반환한다.
    최소 step × 1.5 → 직접 인접 die + 대각선 이웃 포함.
    """
    dx_sorted = np.sort(die_df["die_x"].unique())
    dy_sorted = np.sort(die_df["die_y"].unique())
    step_x = np.min(np.diff(dx_sorted)) if len(dx_sorted) > 1 else 1
    step_y = np.min(np.diff(dy_sorted)) if len(dy_sorted) > 1 else 1
    step = min(step_x, step_y)
    return step * 1.5


def neighbor_similarity(xs_dict, ys_train, feat_cols, n_feats=20, radius=None):
    """
    인접 die 간 WT 측정값 유사도 분석

    같은 웨이퍼 내 유클리드 거리 ≤ radius인 die 쌍의 feature 차이를 계산하고,
    Y=0 unit과 Y>0 unit에서 인접 die 유사도가 다른지 비교한다.

    Parameters
    ----------
    xs_dict : dict
    ys_train : DataFrame
    feat_cols : list of str
    n_feats : int  - 분석할 feature 수
    radius : float  - 인접 die 판정 거리

    Returns
    -------
    similarity_df : DataFrame
        (feature, y0_mean_diff, ypos_mean_diff, ratio, ttest_pval)
    """
    die_df, selected_feats = _parse_wafer_data(xs_dict, ys_train, feat_cols, n_feats)
    wafer_ids = die_df["wafer_id"].unique()

    # radius 자동 감지
    if radius is None:
        radius = _auto_radius(die_df)
        print(f"인접 판정 radius 자동 감지: {radius:.1f}")

    # 웨이퍼별로 인접 die 쌍의 absolute difference 계산
    results = {f: {"y0_diffs": [], "ypos_diffs": []} for f in selected_feats}

    n_wafers_sample = min(len(wafer_ids), 200)
    rng = np.random.RandomState(SEED)
    sampled_wafers = rng.choice(wafer_ids, n_wafers_sample, replace=False)

    for wi, wid in enumerate(sampled_wafers):
        if (wi + 1) % 50 == 0:
            print(f"  웨이퍼 {wi+1}/{n_wafers_sample} 처리 중...", end="\r")

        wf = die_df[die_df["wafer_id"] == wid]
        if len(wf) < 2:
            continue

        coords = wf[["die_x", "die_y"]].values.astype(float)
        dists = cdist(coords, coords)

        # 인접 die 쌍 (상삼각만)
        i_idx, j_idx = np.where((dists > 0) & (dists <= radius) & np.triu(np.ones_like(dists, dtype=bool)))

        if len(i_idx) == 0:
            continue

        health_vals = wf[TARGET_COL].values

        # 쌍 중 하나라도 Y>0이면 ypos, 둘 다 Y=0이면 y0
        pair_health = np.maximum(health_vals[i_idx], health_vals[j_idx])
        y0_mask = pair_health == 0
        ypos_mask = pair_health > 0

        for feat in selected_feats:
            vals = wf[feat].values.astype(float)
            abs_diffs = np.abs(vals[i_idx] - vals[j_idx])

            # NaN diff 제거 (feature 결측치 전파 방지)
            valid_d = ~np.isnan(abs_diffs)

            if y0_mask.any():
                keep = valid_d & y0_mask
                if keep.any():
                    results[feat]["y0_diffs"].extend(abs_diffs[keep].tolist())
            if ypos_mask.any():
                keep = valid_d & ypos_mask
                if keep.any():
                    results[feat]["ypos_diffs"].extend(abs_diffs[keep].tolist())

    print(" " * 60)

    # 결과 요약
    rows = []
    for feat in selected_feats:
        y0_d = np.array(results[feat]["y0_diffs"])
        yp_d = np.array(results[feat]["ypos_diffs"])

        if len(y0_d) == 0 or len(yp_d) == 0:
            continue

        y0_mean = y0_d.mean()
        yp_mean = yp_d.mean()
        ratio = yp_mean / y0_mean if y0_mean > 0 else np.nan

        # Mann-Whitney U test
        _, pval = sp_stats.mannwhitneyu(y0_d, yp_d, alternative="two-sided")

        rows.append({
            "feature": feat,
            "y0_mean_diff": y0_mean,
            "ypos_mean_diff": yp_mean,
            "ratio": ratio,
            "mannwhitney_pval": pval,
            "y0_pairs": len(y0_d),
            "ypos_pairs": len(yp_d),
        })

    similarity_df = pd.DataFrame(rows).sort_values("ratio", ascending=False).reset_index(drop=True)

    # 요약 출력
    print("=" * 75)
    print(f"인접 Die 유사도 분석 (radius={radius})")
    print("=" * 75)
    if len(similarity_df) == 0:
        print("분석 결과 없음 (유효한 feature 쌍 없음)")
        return similarity_df
    n_sig = (similarity_df["mannwhitney_pval"] < 0.05).sum()
    print(f"분석 feature: {len(similarity_df)}개")
    print(f"유의미한 차이 (p<0.05): {n_sig}개 ({n_sig/len(similarity_df)*100:.1f}%)")
    print()
    print(f"  {'Feature':>10}  {'Y=0 diff':>10}  {'Y>0 diff':>10}  {'비율':>6}  {'p-value':>10}")
    print("-" * 60)
    for _, row in similarity_df.head(10).iterrows():
        sig = "***" if row["mannwhitney_pval"] < 0.001 else "**" if row["mannwhitney_pval"] < 0.01 else "*" if row["mannwhitney_pval"] < 0.05 else ""
        print(f"  {row['feature']:>10}  {row['y0_mean_diff']:>10.4f}  {row['ypos_mean_diff']:>10.4f}  "
              f"{row['ratio']:>6.2f}  {row['mannwhitney_pval']:>10.4g} {sig}")

    return similarity_df


def plot_neighbor_similarity(similarity_df, n=15):
    """
    인접 die 유사도 Y=0 vs Y>0 비교 시각화

    Parameters
    ----------
    similarity_df : DataFrame  - neighbor_similarity() 결과
    n : int  - 시각화할 feature 수
    """
    top = similarity_df.head(n)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1) Bar chart: Y=0 vs Y>0 mean difference
    x = np.arange(len(top))
    w = 0.35
    axes[0].barh(x - w/2, top["y0_mean_diff"].values,
                 height=w, color="skyblue", edgecolor="black", alpha=0.8, label="Y=0 쌍")
    axes[0].barh(x + w/2, top["ypos_mean_diff"].values,
                 height=w, color="salmon", edgecolor="black", alpha=0.8, label="Y>0 포함 쌍")
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(top["feature"].values, fontsize=9)
    axes[0].set_xlabel("인접 die 간 평균 |차이|")
    axes[0].set_title("인접 Die 유사도: Y=0 vs Y>0")
    axes[0].legend(fontsize=9)
    axes[0].invert_yaxis()

    # 2) Ratio bar
    colors = ["salmon" if r > 1.05 else "skyblue" if r < 0.95 else "gray"
              for r in top["ratio"].values]
    axes[1].barh(x, top["ratio"].values, color=colors, edgecolor="black", alpha=0.8)
    axes[1].axvline(1.0, color="red", linestyle="--", alpha=0.5)
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(top["feature"].values, fontsize=9)
    axes[1].set_xlabel("비율 (Y>0 diff / Y=0 diff)")
    axes[1].set_title("Y>0일 때 인접 die 차이 비율\n(>1: 불량 unit의 die 간 이질성 높음)")
    axes[1].invert_yaxis()

    plt.suptitle("인접 Die 불량 연관성 분석", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def neighbor_defect_rate(xs_dict, ys_train, feat_cols, radius=None, n_bins=5):
    """
    인접 die의 건강 상태가 중심 die에 미치는 영향 분석

    각 die에 대해 인접 die들의 평균 health를 계산하고,
    이 값이 높을수록 해당 unit의 health도 높은지 검증한다.

    Parameters
    ----------
    xs_dict : dict
    ys_train : DataFrame
    feat_cols : list of str  (사용하지 않지만 인터페이스 통일)
    radius : float or None  - None이면 die 간격에서 자동 감지
    n_bins : int  - 분석용 bin 수

    Returns
    -------
    neighbor_health_df : DataFrame  (unit-level, neighbor_health_mean + actual health)
    """
    die_df, _ = _parse_wafer_data(xs_dict, ys_train, feat_cols, n_feats=5)
    wafer_ids = die_df["wafer_id"].unique()

    # radius 자동 감지
    if radius is None:
        radius = _auto_radius(die_df)
        print(f"인접 판정 radius 자동 감지: {radius:.1f}")

    # 각 die에 대해 인접 die의 평균 health 계산
    die_df["neighbor_health"] = np.nan

    for wi, wid in enumerate(wafer_ids):
        if (wi + 1) % 100 == 0:
            print(f"  웨이퍼 {wi+1}/{len(wafer_ids)} 처리 중...", end="\r")

        mask = die_df["wafer_id"] == wid
        wf = die_df.loc[mask]
        if len(wf) < 2:
            continue

        coords = wf[["die_x", "die_y"]].values.astype(float)
        health_vals = wf[TARGET_COL].values.astype(float)
        dists = cdist(coords, coords)

        n = len(coords)
        neighbor_h = np.full(n, np.nan)
        for i in range(n):
            nbr_mask = (dists[i] > 0) & (dists[i] <= radius)
            if nbr_mask.sum() > 0:
                neighbor_h[i] = health_vals[nbr_mask].mean()

        die_df.loc[mask, "neighbor_health"] = neighbor_h

    print(" " * 60)

    # unit-level 집계
    unit_df = die_df.groupby(KEY_COL).agg(
        neighbor_health_mean=("neighbor_health", "mean"),
        neighbor_health_max=("neighbor_health", "max"),
        actual_health=(TARGET_COL, "first"),
    ).reset_index().dropna()

    # 상관 분석
    r_mean = unit_df["neighbor_health_mean"].corr(unit_df["actual_health"])
    r_max = unit_df["neighbor_health_max"].corr(unit_df["actual_health"])

    print("=" * 60)
    print("인접 Die Health → 중심 Die Health 영향 분석")
    print("=" * 60)
    print(f"Unit 수: {len(unit_df):,}")
    print(f"인접 health mean ↔ actual health  r = {r_mean:.4f}")
    print(f"인접 health max  ↔ actual health  r = {r_max:.4f}")
    print()

    # 구간별 분석
    unit_df["nbr_bin"] = pd.qcut(unit_df["neighbor_health_mean"],
                                  q=n_bins, duplicates="drop")
    bin_stats = unit_df.groupby("nbr_bin", observed=True).agg(
        n=("actual_health", "count"),
        defect_rate=("actual_health", lambda x: (x > 0).mean()),
        mean_health=("actual_health", "mean"),
    )

    print("인접 Die Health 구간별 불량률:")
    print(f"  {'구간':>30}  {'n':>6}  {'불량률':>8}  {'평균 health':>12}")
    print("-" * 65)
    for idx, row in bin_stats.iterrows():
        print(f"  {str(idx):>30}  {int(row['n']):>6}  {row['defect_rate']:>8.1%}  {row['mean_health']:>12.6f}")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1) 구간별 불량률
    bin_stats["defect_rate"].plot(kind="bar", ax=axes[0], color="coral",
                                  edgecolor="black", alpha=0.8)
    axes[0].set_xlabel("인접 Die Health 구간")
    axes[0].set_ylabel("불량률 (Y>0 비율)")
    axes[0].set_title("인접 Die Health ↔ 불량률")
    axes[0].tick_params(axis="x", rotation=45)

    # 2) Scatter (downsampled)
    sample = unit_df.sample(min(5000, len(unit_df)), random_state=SEED)
    axes[1].scatter(sample["neighbor_health_mean"], sample["actual_health"],
                    alpha=0.2, s=5, color="steelblue")
    axes[1].set_xlabel("인접 Die 평균 Health")
    axes[1].set_ylabel("실제 Health")
    axes[1].set_title(f"인접 Die vs 실제 Health (r={r_mean:.4f})")

    plt.suptitle("인접 Die 불량 전파 분석", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    return unit_df


def spatial_autocorrelation_moran(xs_dict, ys_train, feat_cols, n_wafers=50):
    """
    Moran's I 공간 자기상관 분석

    웨이퍼 내 die health가 공간적으로 클러스터링되는지 정량화한다.
    Moran's I > 0: 양의 공간 자기상관(불량 die가 모여있음)
    Moran's I ≈ 0: 랜덤
    Moran's I < 0: 음의 공간 자기상관(불량 die가 분산)

    Parameters
    ----------
    xs_dict : dict
    ys_train : DataFrame
    feat_cols : list of str  (인터페이스 통일용)
    n_wafers : int  - 분석할 웨이퍼 수

    Returns
    -------
    moran_df : DataFrame  (wafer_id, moran_i, expected_i, z_score, p_value)
    """
    die_df, _ = _parse_wafer_data(xs_dict, ys_train, feat_cols, n_feats=5)
    wafer_ids = die_df["wafer_id"].unique()

    # 인접 판정 거리 자동 감지
    moran_radius = _auto_radius(die_df)
    print(f"Moran's I 인접 판정 radius: {moran_radius:.1f}")

    rng = np.random.RandomState(SEED)
    sampled = rng.choice(wafer_ids, min(n_wafers, len(wafer_ids)), replace=False)

    results = []
    for wid in sampled:
        wf = die_df[die_df["wafer_id"] == wid]
        if len(wf) < 10:
            continue

        coords = wf[["die_x", "die_y"]].values.astype(float)
        y = wf[TARGET_COL].values.astype(float)
        n = len(y)

        # 공간 가중 행렬 (inverse distance, threshold)
        dists = cdist(coords, coords)
        W = np.zeros_like(dists)
        mask = (dists > 0) & (dists <= moran_radius)
        W[mask] = 1.0 / dists[mask]

        # S0, S1, S2 — 원본 W 기준 (분자도 원본 W 사용하여 일관성 보장)
        S0 = W.sum()
        if S0 == 0:
            continue
        S1 = 0.5 * np.sum((W + W.T) ** 2)
        S2 = np.sum((W.sum(axis=0) + W.sum(axis=1)) ** 2)

        # Moran's I 계산 (원본 W를 분자에도 동일하게 사용)
        y_bar = y.mean()
        y_dev = y - y_bar
        numerator = np.sum(W * np.outer(y_dev, y_dev))
        denominator = np.sum(y_dev ** 2)

        if denominator == 0:
            continue

        I = (n / S0) * (numerator / denominator)

        # 기대값과 분산 (정규 근사)
        E_I = -1.0 / (n - 1)

        D = n * ((n**2 - 3*n + 3) * S1 - n*S2 + 3*S0**2)
        D -= ((n**2 - n) * S1 - 2*n*S2 + 6*S0**2)
        D /= ((n-1)*(n-2)*(n-3)*S0**2 + 1e-12)

        V_I = D - E_I**2
        if V_I > 0:
            z = (I - E_I) / np.sqrt(V_I)
            p_value = 2 * (1 - sp_stats.norm.cdf(abs(z)))
        else:
            z = np.nan
            p_value = np.nan

        results.append({
            "wafer_id": wid,
            "n_dies": n,
            "moran_i": I,
            "expected_i": E_I,
            "z_score": z,
            "p_value": p_value,
        })

    moran_df = pd.DataFrame(results)

    # 요약
    print("=" * 65)
    print(f"Moran's I 공간 자기상관 분석 ({len(moran_df)} wafers)")
    print("=" * 65)
    print(f"평균 Moran's I: {moran_df['moran_i'].mean():.4f}")
    print(f"유의미한 양의 자기상관 (I>0, p<0.05): "
          f"{((moran_df['moran_i'] > 0) & (moran_df['p_value'] < 0.05)).sum()}개 "
          f"({((moran_df['moran_i'] > 0) & (moran_df['p_value'] < 0.05)).mean()*100:.1f}%)")
    print(f"유의미한 음의 자기상관 (I<0, p<0.05): "
          f"{((moran_df['moran_i'] < 0) & (moran_df['p_value'] < 0.05)).sum()}개")
    print()

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1) Moran's I 분포
    axes[0].hist(moran_df["moran_i"], bins=30, color="steelblue",
                 edgecolor="black", alpha=0.8)
    axes[0].axvline(0, color="red", linestyle="--", alpha=0.5, label="I=0 (랜덤)")
    axes[0].axvline(moran_df["moran_i"].mean(), color="orange",
                    linestyle="-", alpha=0.8, label=f"평균={moran_df['moran_i'].mean():.4f}")
    axes[0].set_xlabel("Moran's I")
    axes[0].set_ylabel("빈도")
    axes[0].set_title("웨이퍼별 Moran's I 분포")
    axes[0].legend(fontsize=9)

    # 2) Z-score 분포
    valid_z = moran_df["z_score"].dropna()
    axes[1].hist(valid_z, bins=30, color="coral", edgecolor="black", alpha=0.8)
    axes[1].axvline(1.96, color="red", linestyle="--", alpha=0.5, label="z=1.96 (p=0.05)")
    axes[1].axvline(-1.96, color="red", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Z-score")
    axes[1].set_ylabel("빈도")
    axes[1].set_title("Moran's I Z-score 분포\n(|z|>1.96: 유의미한 공간 자기상관)")
    axes[1].legend(fontsize=9)

    plt.suptitle("Health 공간 자기상관 분석 (Moran's I)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    return moran_df
