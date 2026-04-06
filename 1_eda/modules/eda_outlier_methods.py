"""
EDA 모듈: 이상치 처리 방법 비교
- IQR vs Winsorization vs AEC DPAT vs Modified Z-score 비교
- 각 방법이 탐지하는 이상치의 수와 겹침 분석
- 이상치 처리 전후 target 상관 변화 비교
- 논문 1-2 (DBSCAN), 5-3 (AEC DPAT), 5-4 (GPR) 근거
- 노트북에서 import eda_outlier_methods as om 으로 사용
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.config import KEY_COL, TARGET_COL, SEED


def _prepare_data(xs_dict, ys_train, feat_cols, n_feats=None):
    """
    Die→unit mean 집계 + target merge + 상위 feature 선택

    Returns
    -------
    merged : DataFrame
    selected_feats : list
    """
    xs_train = xs_dict["train"]
    xs_unit = xs_dict['train_unit_mean'] if 'train_unit_mean' in xs_dict else xs_train.groupby(KEY_COL)[feat_cols].mean()
    merged = xs_unit.merge(ys_train, left_index=True, right_on=KEY_COL, how="inner")

    # target 상관 상위 feature 선택
    corr = merged[feat_cols].corrwith(merged[TARGET_COL]).abs().sort_values(ascending=False)
    valid = [c for c in corr.index if merged[c].std() > 0]
    selected_feats = valid[:n_feats] if n_feats is not None else valid

    return merged, selected_feats


def _detect_iqr(vals):
    """IQR 방법: Q1-1.5*IQR ~ Q3+1.5*IQR 밖이면 이상치"""
    q1, q3 = np.nanpercentile(vals, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (vals < lower) | (vals > upper), lower, upper


def _detect_winsor(vals, pct=1):
    """Winsorization: p1/p99 밖이면 이상치"""
    lower = np.nanpercentile(vals, pct)
    upper = np.nanpercentile(vals, 100 - pct)
    return (vals < lower) | (vals > upper), lower, upper


def _detect_aec_dpat(vals, k=3.0):
    """AEC DPAT (논문 5-3): median 기반 강건 한계"""
    median = np.nanmedian(vals)
    p1 = np.nanpercentile(vals, 1)
    p99 = np.nanpercentile(vals, 99)
    lower = median - k * (median - p1) * 0.43
    upper = median + k * (p99 - median) * 0.43
    return (vals < lower) | (vals > upper), lower, upper


def _detect_mad_zscore(vals, threshold=3.5):
    """Modified Z-score (MAD 기반): MAD=0이면 전부 정상 처리"""
    median = np.nanmedian(vals)
    mad = np.nanmedian(np.abs(vals - median))
    if mad == 0:
        return np.zeros(len(vals), dtype=bool), -np.inf, np.inf
    modified_z = 0.6745 * (vals - median) / mad
    return np.abs(modified_z) > threshold, None, None


def detect_outliers_comparison(xs_dict, ys_train, feat_cols, n_feats=None):
    """
    4가지 이상치 탐지 방법 비교

    1. IQR (표준): Q1-1.5*IQR ~ Q3+1.5*IQR
    2. Winsorization (1%/99%): p1 ~ p99 밖
    3. AEC DPAT (논문 5-3): median + k*(p99-median)*0.43
    4. Modified Z-score (MAD 기반): |modified_z| > 3.5

    Parameters
    ----------
    xs_dict : dict
    ys_train : DataFrame
    feat_cols : list of str
    n_feats : int

    Returns
    -------
    outlier_counts_df : DataFrame  (feature × method → outlier count/pct)
    """
    merged, selected_feats = _prepare_data(xs_dict, ys_train, feat_cols, n_feats)

    methods = {
        "IQR": _detect_iqr,
        "Winsor_1%": lambda v: _detect_winsor(v, 1),
        "AEC_DPAT": _detect_aec_dpat,
        "MAD_Z": _detect_mad_zscore,
    }

    results = []
    for feat in selected_feats:
        vals = merged[feat].dropna().values
        n_total = len(vals)
        row = {"feature": feat, "n_total": n_total}

        masks = {}
        for method_name, method_func in methods.items():
            result = method_func(vals)
            mask = result[0]
            masks[method_name] = mask
            n_outlier = mask.sum()
            row[f"{method_name}_count"] = n_outlier
            row[f"{method_name}_pct"] = n_outlier / n_total * 100

        # 겹침: 모든 방법이 탐지한 이상치
        all_mask = masks["IQR"] & masks["Winsor_1%"] & masks["AEC_DPAT"] & masks["MAD_Z"]
        row["all_4_count"] = all_mask.sum()
        # 하나 이상이 탐지
        any_mask = masks["IQR"] | masks["Winsor_1%"] | masks["AEC_DPAT"] | masks["MAD_Z"]
        row["any_count"] = any_mask.sum()

        results.append(row)

    outlier_counts_df = pd.DataFrame(results)

    # 요약 출력
    print("=" * 75)
    print("이상치 탐지 방법 비교 (unit-level, 4가지 방법)")
    print("=" * 75)
    print(f"분석 feature: {len(selected_feats)}개")
    print()

    for method_name in methods:
        pct_col = f"{method_name}_pct"
        mean_pct = outlier_counts_df[pct_col].mean()
        max_pct = outlier_counts_df[pct_col].max()
        max_feat = outlier_counts_df.loc[outlier_counts_df[pct_col].idxmax(), "feature"]
        print(f"  {method_name:>12}: 평균 {mean_pct:.1f}%, 최대 {max_pct:.1f}% ({max_feat})")

    mean_all4 = outlier_counts_df["all_4_count"].mean()
    mean_any = outlier_counts_df["any_count"].mean()
    print(f"\n  4방법 모두 탐지 (평균): {mean_all4:.1f}개/feature")
    print(f"  1개 이상 탐지 (평균)  : {mean_any:.1f}개/feature")

    return outlier_counts_df


def plot_outlier_comparison(outlier_counts_df, n=12):
    """
    이상치 탐지 비교 시각화

    Parameters
    ----------
    outlier_counts_df : DataFrame
    n : int  - 시각화할 feature 수
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    methods = ["IQR", "Winsor_1%", "AEC_DPAT", "MAD_Z"]
    colors = ["steelblue", "coral", "mediumseagreen", "orchid"]

    # 1) 상위 N feature grouped bar chart
    # 이상치 비율 평균이 높은 feature 선택
    outlier_counts_df["avg_pct"] = outlier_counts_df[[f"{m}_pct" for m in methods]].mean(axis=1)
    top = outlier_counts_df.nlargest(n, "avg_pct")

    x = np.arange(len(top))
    w = 0.2
    for i, method in enumerate(methods):
        axes[0].barh(x + i * w, top[f"{method}_pct"].values, height=w,
                     color=colors[i], edgecolor="black", alpha=0.8, label=method)
    axes[0].set_yticks(x + w * 1.5)
    axes[0].set_yticklabels(top["feature"].values, fontsize=8)
    axes[0].set_xlabel("이상치 비율 (%)")
    axes[0].set_title(f"방법별 이상치 비율 (상위 {n})")
    axes[0].legend(fontsize=8)
    axes[0].invert_yaxis()

    # 2) 방법별 이상치 비율 분포 boxplot
    box_data = []
    for method in methods:
        vals = outlier_counts_df[f"{method}_pct"].values
        box_data.append(vals)

    bp = axes[1].boxplot(box_data, labels=methods, patch_artist=True, showfliers=True,
                         flierprops=dict(marker=".", markersize=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_ylabel("이상치 비율 (%)")
    axes[1].set_title("방법별 이상치 비율 분포")
    axes[1].tick_params(axis="x", rotation=20)

    # 3) 겹침 분석 bar
    overlap_labels = ["4방법 모두", "1개 이상", "IQR만", "AEC만"]
    overlap_vals = [
        outlier_counts_df["all_4_count"].mean(),
        outlier_counts_df["any_count"].mean(),
        (outlier_counts_df["IQR_count"] - outlier_counts_df["all_4_count"]).clip(lower=0).mean(),
        (outlier_counts_df["AEC_DPAT_count"] - outlier_counts_df["all_4_count"]).clip(lower=0).mean(),
    ]
    axes[2].bar(overlap_labels, overlap_vals, color=["gold", "tomato", "steelblue", "mediumseagreen"],
                edgecolor="black", alpha=0.8)
    axes[2].set_ylabel("평균 이상치 수 / feature")
    axes[2].set_title("방법 간 겹침 분석")
    axes[2].tick_params(axis="x", rotation=20)
    for i, v in enumerate(overlap_vals):
        axes[2].text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=10)

    plt.suptitle("이상치 탐지 방법 비교", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def outlier_impact_on_correlation(xs_dict, ys_train, feat_cols, n_feats=None):
    """
    이상치 처리 전후 target 상관 변화 비교

    Parameters
    ----------
    xs_dict : dict
    ys_train : DataFrame
    feat_cols : list of str
    n_feats : int

    Returns
    -------
    impact_df : DataFrame  (feature, raw_corr, iqr_corr, winsor_corr, aec_corr)
    """
    merged, selected_feats = _prepare_data(xs_dict, ys_train, feat_cols, n_feats)
    merged = merged.reset_index(drop=True)
    target = merged[TARGET_COL]

    results = []
    for feat in selected_feats:
        vals = merged[feat].values.copy()
        raw_r = pd.Series(vals).corr(target)

        # IQR clipping
        _, lo, hi = _detect_iqr(vals)
        clipped_iqr = np.clip(vals, lo, hi)
        iqr_r = pd.Series(clipped_iqr).corr(target)

        # Winsorization
        _, lo, hi = _detect_winsor(vals, 1)
        clipped_win = np.clip(vals, lo, hi)
        win_r = pd.Series(clipped_win).corr(target)

        # AEC DPAT
        _, lo, hi = _detect_aec_dpat(vals)
        clipped_aec = np.clip(vals, lo, hi)
        aec_r = pd.Series(clipped_aec).corr(target)

        raw_abs = abs(raw_r) if pd.notna(raw_r) else 0
        best_abs = max(abs(iqr_r) if pd.notna(iqr_r) else 0,
                       abs(win_r) if pd.notna(win_r) else 0,
                       abs(aec_r) if pd.notna(aec_r) else 0)

        results.append({
            "feature": feat,
            "raw_corr": raw_r,
            "iqr_corr": iqr_r,
            "winsor_corr": win_r,
            "aec_corr": aec_r,
            "best_treated": best_abs,
            "improvement": best_abs - raw_abs,
        })

    impact_df = pd.DataFrame(results)
    impact_df = impact_df.sort_values("improvement", ascending=False).reset_index(drop=True)

    # 요약
    n_improved = (impact_df["improvement"] > 0).sum()
    print("=" * 70)
    print("이상치 처리 전후 Target 상관 변화")
    print("=" * 70)
    print(f"분석 feature: {len(impact_df)}개")
    print(f"상관 향상: {n_improved}개 ({n_improved/len(impact_df)*100:.1f}%)")
    print()

    # 방법별 평균 |r|
    for method, col in [("Raw", "raw_corr"), ("IQR", "iqr_corr"),
                        ("Winsor", "winsor_corr"), ("AEC DPAT", "aec_corr")]:
        mean_r = impact_df[col].abs().mean()
        max_r = impact_df[col].abs().max()
        print(f"  {method:>10}: mean|r|={mean_r:.4f}, max|r|={max_r:.4f}")

    print(f"\n향상 상위 10개:")
    print(f"  {'Feature':>8}  {'Raw|r|':>7}  {'IQR|r|':>7}  {'Win|r|':>7}  {'AEC|r|':>7}  {'향상':>7}")
    print("-" * 55)
    for _, row in impact_df.head(10).iterrows():
        print(f"  {row['feature']:>8}  {abs(row['raw_corr']):>7.4f}  "
              f"{abs(row['iqr_corr']):>7.4f}  {abs(row['winsor_corr']):>7.4f}  "
              f"{abs(row['aec_corr']):>7.4f}  {row['improvement']:>+7.4f}")

    return impact_df


def plot_outlier_impact(impact_df, n=15):
    """
    이상치 처리 효과 시각화

    Parameters
    ----------
    impact_df : DataFrame
    n : int
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    top = impact_df.head(n)

    # 1) 상위 feature 방법별 |r|
    x = np.arange(len(top))
    w = 0.2
    methods = [("raw_corr", "Raw", "gray"), ("iqr_corr", "IQR", "steelblue"),
               ("winsor_corr", "Winsor", "coral"), ("aec_corr", "AEC", "mediumseagreen")]
    for i, (col, label, color) in enumerate(methods):
        axes[0].barh(x + i * w, top[col].abs().values, height=w,
                     color=color, edgecolor="black", alpha=0.8, label=label)
    axes[0].set_yticks(x + w * 1.5)
    axes[0].set_yticklabels(top["feature"].values, fontsize=8)
    axes[0].set_xlabel("|r| with target")
    axes[0].set_title(f"이상치 처리 방법별 |r| (향상 상위 {n})")
    axes[0].legend(fontsize=8)
    axes[0].invert_yaxis()

    # 2) Raw vs Best scatter
    raw_abs = impact_df["raw_corr"].abs()
    best_abs = impact_df["best_treated"]
    axes[1].scatter(raw_abs, best_abs, alpha=0.5, s=20, color="steelblue")
    lim = max(raw_abs.max(), best_abs.max()) * 1.05
    axes[1].plot([0, lim], [0, lim], "r--", alpha=0.5)
    axes[1].set_xlabel("Raw |r|")
    axes[1].set_ylabel("Best Treated |r|")
    axes[1].set_title("이상치 처리 전후 상관 비교")
    n_above = (best_abs > raw_abs).sum()
    axes[1].text(0.05, 0.95, f"향상: {n_above}/{len(impact_df)}",
                 transform=axes[1].transAxes, fontsize=10, va="top")

    plt.suptitle("이상치 처리 효과 분석", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def target_outlier_analysis(ys_train):
    """
    Target 변수(health) 자체의 이상치 분석

    health max=1.0 vs median~0.006 → 극단적 이상치.
    각 방법으로 Y>0 서브셋의 이상치 탐지 비교.

    Parameters
    ----------
    ys_train : DataFrame
    """
    health = ys_train[TARGET_COL].values
    pos_health = health[health > 0]

    print("=" * 60)
    print("Target (health) 이상치 분석")
    print("=" * 60)
    print(f"전체 unit: {len(health):,}")
    print(f"Y=0: {(health == 0).sum():,} ({(health == 0).mean()*100:.1f}%)")
    print(f"Y>0: {len(pos_health):,}")
    print(f"Y>0 통계: mean={pos_health.mean():.6f}, median={np.median(pos_health):.6f}, "
          f"max={pos_health.max():.6f}")
    print()

    # 각 방법으로 이상치 탐지 (Y>0만)
    methods = {
        "IQR": _detect_iqr(pos_health),
        "Winsor 1%": _detect_winsor(pos_health, 1),
        "AEC DPAT": _detect_aec_dpat(pos_health),
        "MAD Z-score": _detect_mad_zscore(pos_health),
    }

    boundaries = {}
    print("Y>0 health 이상치 탐지 결과:")
    for name, (mask, lo, hi) in methods.items():
        n_out = mask.sum()
        boundaries[name] = (lo, hi)
        lo_str = f"{lo:.6f}" if lo is not None else "N/A"
        hi_str = f"{hi:.6f}" if hi is not None else "N/A"
        print(f"  {name:>12}: {n_out:,}개 ({n_out/len(pos_health)*100:.1f}%)  "
              f"범위: [{lo_str}, {hi_str}]")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # 1) Y>0 분포 + 각 방법의 경계선
    axes[0].hist(pos_health, bins=80, edgecolor="black", color="steelblue", alpha=0.7)
    colors = {"IQR": "red", "Winsor 1%": "orange", "AEC DPAT": "green", "MAD Z-score": "purple"}
    for name, (lo, hi) in boundaries.items():
        if hi is not None and np.isfinite(hi):
            axes[0].axvline(x=hi, color=colors[name], linestyle="--",
                            linewidth=1.5, alpha=0.8, label=f"{name} upper")
    # 오른쪽 확대 영역 표시 (빨간 점선 박스)
    from matplotlib.patches import Rectangle
    q95_val = np.percentile(pos_health, 95)
    q999_val = np.percentile(pos_health, 99.9)
    xlim_lo, xlim_hi = q95_val * 0.5, q999_val * 2.5
    ylim = axes[0].get_ylim()
    rect = Rectangle((xlim_lo, 0), xlim_hi - xlim_lo, ylim[1] * 0.15,
                      linewidth=1.5, edgecolor="red", facecolor="none", linestyle="--")
    axes[0].add_patch(rect)
    axes[0].annotate("→ 오른쪽 확대", xy=(xlim_hi, ylim[1] * 0.075),
                     fontsize=8, color="red", va="center")
    axes[0].set_xlabel("health (Y>0)")
    axes[0].set_ylabel("Unit 수")
    axes[0].set_title("Y>0 health 분포 + 이상치 경계")
    axes[0].legend(fontsize=8)

    # 2) 극단값 확대 (상위 5%, x축 줌인)
    q95 = np.percentile(pos_health, 95)
    extreme = pos_health[pos_health >= q95]
    axes[1].hist(extreme, bins=40, edgecolor="black", color="lightsteelblue", alpha=0.7)
    for name, (lo, hi) in boundaries.items():
        if hi is not None and np.isfinite(hi) and hi >= q95:
            axes[1].axvline(x=hi, color=colors[name], linestyle="--",
                            linewidth=1.5, alpha=0.8, label=f"{name}")
    q999 = np.percentile(pos_health, 99.9)
    axes[1].set_xlim(q95 * 0.5, q999 * 2.5)
    axes[1].set_xlabel("health (상위 5%)")
    axes[1].set_ylabel("Unit 수")
    axes[1].set_title("극단값 확대 (상위 5%)")
    axes[1].legend(fontsize=8)

    plt.suptitle("Target 변수 이상치 분석", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
