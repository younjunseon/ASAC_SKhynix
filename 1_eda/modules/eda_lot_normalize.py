"""
EDA 모듈: 로트별 정규화 효과 분석
- run_wf_xy에서 lot(작업번호)를 파싱하여 lot별 feature 분포 차이 확인
- lot별 z-score 정규화 전후 target 상관 비교
- 논문 5-3 근거: Multi-Site 분리로 양품 폐기율 40%→8.9%
- 노트북에서 import eda_lot_normalize as ln 으로 사용
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.config import KEY_COL, DIE_KEY_COL, TARGET_COL, SEED


def _parse_lot(xs_dict):
    """
    Train Xs에서 lot(작업번호) 파싱

    Returns
    -------
    xs_train : DataFrame  (lot 컬럼 추가)
    """
    xs_train = xs_dict["train"].copy()
    xs_train["lot"] = xs_train[DIE_KEY_COL].str.split("_").str[0]
    return xs_train


def lot_feature_variance(xs_dict, feat_cols, n_feats=10):
    """
    Feature별 로트 간 분산 vs 로트 내 분산 비교 (F-ratio)

    F-ratio가 높을수록 로트 간 차이가 크다 → 정규화 효과가 클 feature.
    논문 5-3에서 Multi-Site(로트) 분리가 최대 성능 향상 요인이었음.

    Parameters
    ----------
    xs_dict : dict
    feat_cols : list of str
    n_feats : int  - 출력할 상위 feature 수

    Returns
    -------
    variance_df : DataFrame  (feature, f_ratio, between_var, within_var)
    """
    xs_train = _parse_lot(xs_dict)

    results = []
    for col in feat_cols:
        vals = xs_train[[col, "lot"]].dropna()
        if len(vals) < 10:
            continue

        overall_mean = vals[col].mean()

        # between-lot variance
        lot_means = vals.groupby("lot")[col].mean()
        lot_sizes = vals.groupby("lot")[col].size()
        between_var = ((lot_sizes * (lot_means - overall_mean) ** 2).sum()
                       / max(len(lot_means) - 1, 1))

        # within-lot variance
        within_var = vals.groupby("lot")[col].var().mean()

        if within_var > 0:
            f_ratio = between_var / within_var
        elif between_var > 0:
            f_ratio = np.inf
        else:
            f_ratio = 0.0

        results.append({
            "feature": col,
            "f_ratio": f_ratio,
            "between_var": between_var,
            "within_var": within_var,
        })

    variance_df = pd.DataFrame(results)
    variance_df = variance_df.sort_values("f_ratio", ascending=False).reset_index(drop=True)

    # 요약 출력
    print("=" * 65)
    print("로트 간 vs 로트 내 분산 비교 (F-ratio)")
    print("=" * 65)
    print(f"분석 대상: {len(variance_df):,}개 feature, {xs_train['lot'].nunique()}개 lot")
    print(f"F-ratio > 10: {(variance_df['f_ratio'] > 10).sum():,}개")
    print(f"F-ratio > 100: {(variance_df['f_ratio'] > 100).sum():,}개")
    print()

    print(f"F-ratio 상위 {n_feats}개 Feature:")
    print(f"  {'Feature':>8}  {'F-ratio':>10}  {'Between':>12}  {'Within':>12}")
    print("-" * 55)
    for _, row in variance_df.head(n_feats).iterrows():
        print(f"  {row['feature']:>8}  {row['f_ratio']:>10.2f}  "
              f"{row['between_var']:>12.4f}  {row['within_var']:>12.4f}")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # 1) F-ratio 분포 (log 스케일)
    f_vals = variance_df["f_ratio"].values
    f_vals_pos = f_vals[(f_vals > 0) & np.isfinite(f_vals)]
    if len(f_vals_pos) > 0:
        axes[0].hist(np.log10(f_vals_pos + 1e-10), bins=60,
                     edgecolor="black", color="steelblue", alpha=0.8)
    axes[0].set_xlabel("log10(F-ratio)")
    axes[0].set_ylabel("Feature 수")
    axes[0].set_title("로트 간/내 분산비 (F-ratio) 분포")
    axes[0].axvline(x=np.log10(10), color="red", linestyle="--",
                    alpha=0.7, label="F=10")
    axes[0].legend()

    # 2) 상위 feature bar chart
    top = variance_df.head(n_feats)
    axes[1].barh(range(len(top)), top["f_ratio"].values,
                 color="coral", edgecolor="black", alpha=0.8)
    axes[1].set_yticks(range(len(top)))
    axes[1].set_yticklabels(top["feature"].values, fontsize=9)
    axes[1].set_xlabel("F-ratio")
    axes[1].set_title(f"F-ratio 상위 {n_feats}개 Feature")
    axes[1].invert_yaxis()

    plt.suptitle("로트 간 분산 분석 (정규화 필요성)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()

    return variance_df


def normalize_and_compare(xs_dict, ys_train, feat_cols):
    """
    로트별 z-score 정규화 전후 target 상관 비교

    정규화: z = (x - lot_mean) / lot_std  (lot_std==0 → 원본 유지)
    die→unit mean 집계 후 target과 Pearson 상관 계산

    Parameters
    ----------
    xs_dict : dict
    ys_train : DataFrame
    feat_cols : list of str

    Returns
    -------
    compare_df : DataFrame  (feature, raw_corr, norm_corr, improvement)
    """
    xs_train = _parse_lot(xs_dict)

    # ── 1) Raw 상관 (정규화 전) ──
    xs_raw_unit = xs_train.groupby(KEY_COL)[feat_cols].mean()
    merged_raw = xs_raw_unit.merge(ys_train, left_index=True, right_on=KEY_COL, how="inner")
    raw_corr = merged_raw[feat_cols].corrwith(merged_raw[TARGET_COL]).dropna()

    # ── 2) 로트별 z-score 정규화 ──
    xs_norm = xs_train[[KEY_COL, "lot"] + feat_cols].copy()

    lot_stats = xs_norm.groupby("lot")[feat_cols].agg(["mean", "std"])

    for col in feat_cols:
        lot_mean = xs_norm["lot"].map(lot_stats[(col, "mean")])
        lot_std = xs_norm["lot"].map(lot_stats[(col, "std")])
        # std > 0: z-score 정규화, std == 0: 로트 내 편차 없음 → 0 설정
        mask = lot_std > 0
        xs_norm.loc[mask, col] = (
            (xs_norm.loc[mask, col] - lot_mean[mask]) / lot_std[mask]
        )
        xs_norm.loc[~mask, col] = 0

    # die→unit mean 집계
    xs_norm_unit = xs_norm.groupby(KEY_COL)[feat_cols].mean()
    merged_norm = xs_norm_unit.merge(ys_train, left_index=True, right_on=KEY_COL, how="inner")
    norm_corr = merged_norm[feat_cols].corrwith(merged_norm[TARGET_COL]).dropna()

    # ── 비교 ──
    common = raw_corr.index.intersection(norm_corr.index)
    compare_df = pd.DataFrame({
        "feature": common,
        "raw_corr": raw_corr[common].values,
        "norm_corr": norm_corr[common].values,
    })
    compare_df["raw_abs"] = compare_df["raw_corr"].abs()
    compare_df["norm_abs"] = compare_df["norm_corr"].abs()
    compare_df["improvement"] = compare_df["norm_abs"] - compare_df["raw_abs"]
    compare_df = compare_df.sort_values("improvement", ascending=False).reset_index(drop=True)

    n_improved = (compare_df["improvement"] > 0).sum()
    n_worsened = (compare_df["improvement"] < 0).sum()
    mean_imp = compare_df["improvement"].mean()
    max_imp = compare_df["improvement"].max()

    print("=" * 65)
    print("로트별 정규화 전후 Target 상관 비교")
    print("=" * 65)
    print(f"분석 대상: {len(compare_df):,}개 feature")
    print(f"상관 향상: {n_improved:,}개 ({n_improved/len(compare_df)*100:.1f}%)")
    print(f"상관 악화: {n_worsened:,}개 ({n_worsened/len(compare_df)*100:.1f}%)")
    print(f"평균 |r| 변화: {mean_imp:+.6f}")
    print(f"최대 |r| 향상: {max_imp:+.6f}")
    print()

    print(f"Raw 최대 |r|: {compare_df['raw_abs'].max():.4f} ({compare_df.loc[compare_df['raw_abs'].idxmax(), 'feature']})")
    print(f"Norm 최대 |r|: {compare_df['norm_abs'].max():.4f} ({compare_df.loc[compare_df['norm_abs'].idxmax(), 'feature']})")
    print()

    print("향상 상위 15개 Feature:")
    print(f"  {'Feature':>8}  {'Raw |r|':>8}  {'Norm |r|':>9}  {'향상':>9}")
    print("-" * 45)
    for _, row in compare_df.head(15).iterrows():
        print(f"  {row['feature']:>8}  {row['raw_abs']:>8.4f}  "
              f"{row['norm_abs']:>9.4f}  {row['improvement']:>+9.4f}")

    return compare_df.drop(columns=["raw_abs", "norm_abs"])


def plot_normalization_effect(compare_df, n=20):
    """
    정규화 전후 상관 비교 시각화

    Parameters
    ----------
    compare_df : DataFrame  - normalize_and_compare() 반환값
    n : int  - 시각화할 feature 수
    """
    raw_abs = compare_df["raw_corr"].abs()
    norm_abs = compare_df["norm_corr"].abs()
    improvement = compare_df["improvement"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 1) Scatter: raw vs norm
    axes[0].scatter(raw_abs, norm_abs, alpha=0.3, s=10, color="steelblue")
    lim = max(raw_abs.max(), norm_abs.max()) * 1.05
    axes[0].plot([0, lim], [0, lim], "r--", alpha=0.5, linewidth=1)
    axes[0].set_xlabel("Raw |r| (정규화 전)")
    axes[0].set_ylabel("Normalized |r| (정규화 후)")
    axes[0].set_title("정규화 전후 Target 상관 비교")
    axes[0].set_xlim(0, lim)
    axes[0].set_ylim(0, lim)
    axes[0].set_aspect("equal")

    n_above = (norm_abs > raw_abs).sum()
    pct = n_above / len(compare_df) * 100
    axes[0].text(0.05, 0.95, f"대각선 위(향상): {pct:.0f}%",
                 transform=axes[0].transAxes, fontsize=10, va="top")

    # 2) 향상 상위 bar chart
    top_improved = compare_df.nlargest(n, "improvement")
    axes[1].barh(range(len(top_improved)), top_improved["improvement"].values,
                 color="mediumseagreen", edgecolor="black", alpha=0.8)
    axes[1].set_yticks(range(len(top_improved)))
    axes[1].set_yticklabels(top_improved["feature"].values, fontsize=8)
    axes[1].set_xlabel("Δ|r| (정규화 후 - 전)")
    axes[1].set_title(f"|r| 향상 상위 {n}개 Feature")
    axes[1].invert_yaxis()

    # 3) 악화 상위 bar chart
    top_worsened = compare_df.nsmallest(n, "improvement")
    if (top_worsened["improvement"] < 0).any():
        show = top_worsened[top_worsened["improvement"] < 0]
        axes[2].barh(range(len(show)), show["improvement"].values,
                     color="salmon", edgecolor="black", alpha=0.8)
        axes[2].set_yticks(range(len(show)))
        axes[2].set_yticklabels(show["feature"].values, fontsize=8)
        axes[2].set_xlabel("Δ|r| (정규화 후 - 전)")
        axes[2].set_title(f"|r| 악화 상위 Feature")
        axes[2].invert_yaxis()
    else:
        axes[2].text(0.5, 0.5, "악화된 feature 없음",
                     transform=axes[2].transAxes, ha="center", fontsize=12)
        axes[2].set_title("|r| 악화 Feature")

    plt.suptitle("로트별 정규화 효과 분석", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def lot_distribution_shift(xs_dict, feat_cols, variance_df=None, n_feats=6):
    """
    F-ratio 상위 feature의 lot별 분포 시각화 → 정규화가 제거할 shift 확인

    Parameters
    ----------
    xs_dict : dict
    feat_cols : list of str
    variance_df : DataFrame  - lot_feature_variance() 반환값. None이면 내부 계산
    n_feats : int  - 시각화할 feature 수
    """
    xs_train = _parse_lot(xs_dict)

    if variance_df is None:
        variance_df = lot_feature_variance(xs_dict, feat_cols, n_feats=n_feats)

    top_feats = variance_df.head(n_feats)["feature"].tolist()

    n_cols = min(3, n_feats)
    n_rows = (n_feats + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes = np.array(axes).flatten()

    # lot 순서: 평균값 기준 정렬
    for i, feat in enumerate(top_feats):
        lot_order = (xs_train.groupby("lot")[feat].mean()
                     .sort_values(ascending=False).index)

        # 샘플링 (시각화 속도)
        sample = xs_train[["lot", feat]].dropna()
        if len(sample) > 30000:
            sample = sample.sample(30000, random_state=SEED)

        sns.boxplot(x="lot", y=feat, data=sample, order=lot_order,
                    ax=axes[i], palette="Set3", fliersize=1)

        f_val = variance_df.loc[variance_df["feature"] == feat, "f_ratio"].values[0]
        axes[i].set_title(f"{feat} (F={f_val:.1f})", fontsize=10)
        axes[i].set_xlabel("Lot")
        axes[i].tick_params(axis="x", rotation=45, labelsize=7)

    for j in range(len(top_feats), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("로트별 Feature 분포 (정규화가 제거할 shift)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()
