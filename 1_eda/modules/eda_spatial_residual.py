"""
EDA 모듈: 공간 잔차(Spatial Residual) 피처 분석
- die WT 측정값에서 인접 die 가우시안 가중 평균을 뺀 잔차(NNR) 분석
- 공간 트렌드에서 벗어난 die가 불량 위험이 높은지 검증
- 논문 5-3 (NNR), 5-4 (GPR 잔차로 15.6% 추가 이상치 검출) 근거
- 노트북에서 import eda_spatial_residual as sr 로 사용
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from scipy.spatial.distance import cdist
from utils.config import KEY_COL, DIE_KEY_COL, TARGET_COL, SEED


def _parse_and_prepare(xs_dict, ys_train, feat_cols, n_feats=30):
    """
    Train die에서 좌표 파싱 + target 상관 상위 feature 선택 + target merge

    Returns
    -------
    die_df : DataFrame  (die-level, 좌표 + 선택 feature + health)
    selected_feats : list
    """
    xs_train = xs_dict["train"]

    # 좌표 파싱
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

    # feature 값 + target 추가
    for f in selected_feats:
        die_df[f] = xs_train[f].values
    die_df = die_df.merge(ys_train[[KEY_COL, TARGET_COL]], on=KEY_COL, how="inner")

    print(f"파싱 완료: {len(die_df):,} dies, {die_df['wafer_id'].nunique()} wafers")
    print(f"선택 feature: {len(selected_feats)}개 (target |r| 상위)")

    return die_df, selected_feats


def _compute_nnr_for_wafer(wf_data, feat, sigma=3.0):
    """
    단일 웨이퍼에서 NNR(Nearest Neighbor Residual) 계산 (벡터화)

    Parameters
    ----------
    wf_data : DataFrame  (die_x, die_y, feat 컬럼)
    feat : str
    sigma : float  - 가우시안 가중치 bandwidth

    Returns
    -------
    residuals : ndarray  (각 die의 잔차)
    """
    coords = wf_data[["die_x", "die_y"]].values.astype(float)
    values = wf_data[feat].values.astype(float)
    n = len(coords)

    # 거리 행렬 한번에 계산 (C-level, O(n²) → 단일 호출)
    dist_matrix = cdist(coords, coords)

    cutoff = max(3, 3 * sigma)
    valid = ~np.isnan(values)

    # 이웃 마스크: 자기 자신 제외, 거리 cutoff 이내, NaN 아닌 값
    neighbor_mask = (dist_matrix > 0) & (dist_matrix <= cutoff) & valid[np.newaxis, :]

    # 가우시안 가중치 (이웃이 아닌 곳은 0)
    gauss_weights = np.where(neighbor_mask, np.exp(-dist_matrix ** 2 / (2 * sigma ** 2)), 0.0)

    # 가중 평균 계산 (행렬 연산)
    weight_sums = gauss_weights.sum(axis=1)
    values_safe = np.where(valid, values, 0.0)
    weighted_avg = np.where(weight_sums > 0, gauss_weights @ values_safe / weight_sums, np.nan)

    residuals = np.where(valid & (weight_sums > 0), values - weighted_avg, np.nan)

    return residuals


def compute_spatial_residual(xs_dict, ys_train, feat_cols, n_feats=30, sigma=3.0):
    """
    NNR 잔차 계산 후 unit-level 집계, target 상관 비교

    각 die의 WT 값에서 가우시안 가중 이웃 평균을 빼 잔차 산출.
    잔차가 크면 = 공간 트렌드에서 벗어난 die = 불량 위험.

    Parameters
    ----------
    xs_dict : dict
    ys_train : DataFrame
    feat_cols : list of str
    n_feats : int  - 분석할 feature 수
    sigma : float  - 가우시안 bandwidth

    Returns
    -------
    residual_corr_df : DataFrame
        (feature, original_corr, resid_mean_corr, resid_std_corr, resid_absmax_corr, best_resid_corr)
    """
    die_df, selected_feats = _parse_and_prepare(xs_dict, ys_train, feat_cols, n_feats)

    wafer_ids = die_df["wafer_id"].unique()
    n_wafers = len(wafer_ids)

    results = []
    for fi, feat in enumerate(selected_feats):
        print(f"  [{fi+1}/{len(selected_feats)}] {feat} NNR 계산 중 ({n_wafers} wafers)...",
              end="\r")

        # 웨이퍼별 NNR (원본 인덱스 기반 대입으로 행 정렬 보장)
        die_df[f"{feat}_resid"] = np.nan
        for wid in wafer_ids:
            mask = die_df["wafer_id"] == wid
            wf = die_df[mask]
            resids = _compute_nnr_for_wafer(wf, feat, sigma)
            die_df.loc[mask, f"{feat}_resid"] = resids

        # unit-level 집계
        resid_col = f"{feat}_resid"
        unit_agg = die_df.groupby(KEY_COL).agg(
            resid_mean=(resid_col, "mean"),
            resid_std=(resid_col, "std"),
            resid_absmax=(resid_col, lambda x: x.abs().max()),
            original_mean=(feat, "mean"),
            health=(TARGET_COL, "first"),
        ).reset_index().dropna()

        # 상관계수
        orig_r = unit_agg["original_mean"].corr(unit_agg["health"])
        mean_r = unit_agg["resid_mean"].corr(unit_agg["health"])
        std_r = unit_agg["resid_std"].corr(unit_agg["health"])
        absmax_r = unit_agg["resid_absmax"].corr(unit_agg["health"])

        best_r = max(abs(mean_r), abs(std_r), abs(absmax_r))

        results.append({
            "feature": feat,
            "original_corr": orig_r,
            "resid_mean_corr": mean_r,
            "resid_std_corr": std_r,
            "resid_absmax_corr": absmax_r,
            "best_resid_corr": best_r,
            "improvement": best_r - abs(orig_r),
        })

    print(" " * 80)  # clear line

    residual_corr_df = pd.DataFrame(results)
    residual_corr_df = residual_corr_df.sort_values("improvement", ascending=False).reset_index(drop=True)

    # 요약
    n_improved = (residual_corr_df["improvement"] > 0).sum()

    print("=" * 75)
    print(f"NNR 공간 잔차 분석 결과 (sigma={sigma})")
    print("=" * 75)
    print(f"분석 feature: {len(residual_corr_df)}개")
    print(f"잔차 피처가 원본보다 |r| 높은 feature: {n_improved}개 "
          f"({n_improved/len(residual_corr_df)*100:.1f}%)")
    print(f"평균 개선: {residual_corr_df['improvement'].mean():+.6f}")
    print()

    print("상위 15개 Feature:")
    print(f"  {'Feature':>8}  {'원본|r|':>8}  {'잔차 best|r|':>12}  {'향상':>8}  {'best 집계':>10}")
    print("-" * 60)
    for _, row in residual_corr_df.head(15).iterrows():
        best_type = "mean"
        if abs(row["resid_std_corr"]) == row["best_resid_corr"]:
            best_type = "std"
        elif abs(row["resid_absmax_corr"]) == row["best_resid_corr"]:
            best_type = "absmax"
        print(f"  {row['feature']:>8}  {abs(row['original_corr']):>8.4f}  "
              f"{row['best_resid_corr']:>12.4f}  {row['improvement']:>+8.4f}  {best_type:>10}")

    return residual_corr_df


def plot_residual_vs_original(residual_corr_df, n=15):
    """
    원본 vs 잔차 피처 상관 비교 시각화

    Parameters
    ----------
    residual_corr_df : DataFrame
    n : int  - 시각화할 feature 수
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    orig_abs = residual_corr_df["original_corr"].abs()
    best_abs = residual_corr_df["best_resid_corr"]

    # 1) Scatter
    axes[0].scatter(orig_abs, best_abs, alpha=0.5, s=20, color="steelblue")
    lim = max(orig_abs.max(), best_abs.max()) * 1.05
    axes[0].plot([0, lim], [0, lim], "r--", alpha=0.5)
    axes[0].set_xlabel("원본 |r|")
    axes[0].set_ylabel("NNR 잔차 best |r|")
    axes[0].set_title("원본 vs NNR 잔차 Target 상관")
    n_above = (best_abs > orig_abs).sum()
    axes[0].text(0.05, 0.95, f"잔차 우위: {n_above}/{len(residual_corr_df)}",
                 transform=axes[0].transAxes, fontsize=10, va="top")

    # 2) Bar chart (상위 N)
    top = residual_corr_df.head(n)
    x = np.arange(len(top))
    w = 0.35
    axes[1].barh(x - w/2, top["original_corr"].abs().values,
                 height=w, color="steelblue", edgecolor="black", alpha=0.8, label="원본")
    axes[1].barh(x + w/2, top["best_resid_corr"].values,
                 height=w, color="coral", edgecolor="black", alpha=0.8, label="NNR 잔차")
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(top["feature"].values, fontsize=9)
    axes[1].set_xlabel("|r| with target")
    axes[1].set_title(f"원본 vs NNR 잔차 상관 (상위 {n})")
    axes[1].legend(fontsize=9)
    axes[1].invert_yaxis()

    plt.suptitle("NNR 공간 잔차 피처 효과", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_residual_distribution(xs_dict, ys_train, feat_cols, n_feats=6, sigma=3.0):
    """
    Y=0 vs Y>0 unit의 잔차 분포 비교 → 불량 unit이 잔차가 큰지 확인

    Parameters
    ----------
    xs_dict, ys_train, feat_cols : 데이터
    n_feats : int  - 시각화할 feature 수
    sigma : float  - NNR bandwidth
    """
    die_df, selected_feats = _parse_and_prepare(xs_dict, ys_train, feat_cols, n_feats)
    use_feats = selected_feats[:n_feats]
    wafer_ids = die_df["wafer_id"].unique()

    n_cols = min(3, n_feats)
    n_rows = (n_feats + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes = np.array(axes).flatten()

    for fi, feat in enumerate(use_feats):
        # NNR (원본 인덱스 기반 대입으로 행 정렬 보장)
        die_df[f"{feat}_resid"] = np.nan
        for wid in wafer_ids:
            mask = die_df["wafer_id"] == wid
            wf = die_df[mask]
            resids = _compute_nnr_for_wafer(wf, feat, sigma)
            die_df.loc[mask, f"{feat}_resid"] = resids

        # unit-level |mean residual|
        unit_resid = die_df.groupby(KEY_COL).agg(
            abs_resid_mean=(f"{feat}_resid", lambda x: x.abs().mean()),
            health=(TARGET_COL, "first"),
        ).reset_index().dropna()

        zero_vals = unit_resid.loc[unit_resid["health"] == 0, "abs_resid_mean"]
        pos_vals = unit_resid.loc[unit_resid["health"] > 0, "abs_resid_mean"]

        axes[fi].hist(zero_vals, bins=40, alpha=0.6, color="skyblue",
                      label=f"Y=0 (n={len(zero_vals):,})", density=True)
        axes[fi].hist(pos_vals, bins=40, alpha=0.6, color="salmon",
                      label=f"Y>0 (n={len(pos_vals):,})", density=True)
        axes[fi].set_xlabel("|mean NNR residual|")
        axes[fi].set_ylabel("Density")
        axes[fi].set_title(feat, fontsize=10)
        axes[fi].legend(fontsize=8)

    for j in range(len(use_feats), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Y=0 vs Y>0: NNR 잔차 크기 분포\n(Y>0이 오른쪽에 치우치면 공간 이탈 = 불량 신호)",
                 fontsize=13, y=1.03)
    plt.tight_layout()
    plt.show()
