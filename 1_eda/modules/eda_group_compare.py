"""
EDA 모듈: Y=0 vs Y>0 그룹 비교 분석
- 반도체 WT 데이터에서 정상(Y=0)과 불량(Y>0) unit 간 feature 분포 차이를 통계적으로 검정
- Two-stage 모델의 1단계(분류) 설계 근거 제공
- 노트북에서 import eda_group_compare as gc 로 사용
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from utils.config import KEY_COL, TARGET_COL, SEED


def _build_group_data(xs_dict, ys_train, feat_cols):
    """
    Train die-level → unit-level 평균 집계 후, Y=0 / Y>0 그룹으로 분리

    Returns
    -------
    merged : DataFrame  (unit-level X + Y)
    grp_zero : DataFrame  (Y=0인 unit)
    grp_pos  : DataFrame  (Y>0인 unit)
    """
    xs_train = xs_dict["train"]
    xs_unit = xs_dict['train_unit_mean'] if 'train_unit_mean' in xs_dict else xs_train.groupby(KEY_COL)[feat_cols].mean()
    merged = xs_unit.merge(ys_train, left_index=True, right_on=KEY_COL, how="inner")

    grp_zero = merged[merged[TARGET_COL] == 0]
    grp_pos = merged[merged[TARGET_COL] > 0]

    return merged, grp_zero, grp_pos


def group_overview(xs_dict, ys_train, feat_cols):
    """
    Y=0 vs Y>0 그룹 기초통계 비교 출력

    Parameters
    ----------
    xs_dict : dict  - {"train": DataFrame, ...}
    ys_train : DataFrame  - train Y (ufs_serial, health)
    feat_cols : list of str

    Returns
    -------
    merged, grp_zero, grp_pos : DataFrame
    """
    merged, grp_zero, grp_pos = _build_group_data(xs_dict, ys_train, feat_cols)

    n_total = len(merged)
    n_zero = len(grp_zero)
    n_pos = len(grp_pos)

    print("=" * 60)
    print("Y=0 vs Y>0 그룹 기초통계")
    print("=" * 60)
    print(f"전체 unit 수: {n_total:,}")
    print(f"Y=0 (정상) : {n_zero:,} ({n_zero/n_total*100:.1f}%)")
    print(f"Y>0 (불량) : {n_pos:,} ({n_pos/n_total*100:.1f}%)")
    print()

    # 그룹별 feature 평균 비교 (상위 10개 차이 큰 feature)
    mean_zero = grp_zero[feat_cols].mean()
    mean_pos = grp_pos[feat_cols].mean()
    std_zero = grp_zero[feat_cols].std()
    std_pos = grp_pos[feat_cols].std()

    diff = (mean_pos - mean_zero).abs()
    top_diff = diff.sort_values(ascending=False).head(10)

    print("그룹 평균 차이가 큰 상위 10개 Feature:")
    print(f"  {'Feature':>8}  {'mean(Y=0)':>10}  {'mean(Y>0)':>10}  {'|차이|':>10}")
    print("-" * 50)
    for feat in top_diff.index:
        print(f"  {feat:>8}  {mean_zero[feat]:>10.4f}  {mean_pos[feat]:>10.4f}  {diff[feat]:>10.4f}")

    return merged, grp_zero, grp_pos


def statistical_tests(grp_zero, grp_pos, feat_cols, alpha=0.05):
    """
    feature별 Mann-Whitney U test + KS test + Cohen's d 계산

    반도체 맥락:
    - Mann-Whitney U: 두 그룹의 중위 경향 차이 검정 (비모수, 분포 가정 불필요)
    - KS test: 두 그룹의 분포 형태 자체가 다른지 검정
    - Cohen's d: 통계적 유의성뿐 아니라 실질적 효과 크기 (|d|>0.2 small, >0.5 medium, >0.8 large)

    Parameters
    ----------
    grp_zero : DataFrame  - Y=0 그룹 unit-level
    grp_pos  : DataFrame  - Y>0 그룹 unit-level
    feat_cols : list of str
    alpha : float  - 유의수준 (기본 0.05)

    Returns
    -------
    test_df : DataFrame
        columns = [feature, mw_statistic, mw_pvalue, ks_statistic, ks_pvalue, cohens_d]
        cohens_d 절대값 기준 내림차순 정렬
    """
    results = []

    for col in feat_cols:
        vals_zero = grp_zero[col].dropna().values
        vals_pos = grp_pos[col].dropna().values

        # 둘 다 비어있거나 분산이 0이면 스킵
        if len(vals_zero) < 2 or len(vals_pos) < 2:
            continue

        # Mann-Whitney U test (비모수)
        mw_stat, mw_p = stats.mannwhitneyu(vals_zero, vals_pos, alternative="two-sided")

        # KS test (분포 차이)
        ks_stat, ks_p = stats.ks_2samp(vals_zero, vals_pos)

        # Cohen's d (효과 크기)
        n0, n1 = len(vals_zero), len(vals_pos)
        m0, m1 = vals_zero.mean(), vals_pos.mean()
        s0, s1 = vals_zero.std(ddof=1), vals_pos.std(ddof=1)
        # pooled std
        pooled_std = np.sqrt(((n0 - 1) * s0**2 + (n1 - 1) * s1**2) / (n0 + n1 - 2))
        if pooled_std > 0:
            d = (m1 - m0) / pooled_std
        else:
            d = 0.0

        results.append({
            "feature": col,
            "mw_statistic": mw_stat,
            "mw_pvalue": mw_p,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_p,
            "cohens_d": d,
        })

    test_df = pd.DataFrame(results)
    test_df["abs_d"] = test_df["cohens_d"].abs()
    test_df = test_df.sort_values("abs_d", ascending=False).reset_index(drop=True)

    # 요약 출력
    n_mw_sig = (test_df["mw_pvalue"] < alpha).sum()
    n_ks_sig = (test_df["ks_pvalue"] < alpha).sum()
    n_d_small = (test_df["abs_d"] >= 0.2).sum()
    n_d_medium = (test_df["abs_d"] >= 0.5).sum()

    print("=" * 60)
    print("통계 검정 요약")
    print("=" * 60)
    print(f"검정 대상 feature: {len(test_df):,}개")
    print(f"Mann-Whitney U 유의 (p<{alpha}): {n_mw_sig:,}개 ({n_mw_sig/len(test_df)*100:.1f}%)")
    print(f"KS test 유의 (p<{alpha})        : {n_ks_sig:,}개 ({n_ks_sig/len(test_df)*100:.1f}%)")
    print(f"|Cohen's d| >= 0.2 (small)     : {n_d_small:,}개")
    print(f"|Cohen's d| >= 0.5 (medium)    : {n_d_medium:,}개")
    print()

    # 상위 20개 출력
    print("Cohen's d 상위 20개 Feature:")
    print(f"  {'Feature':>8}  {'Cohen d':>9}  {'MW p-val':>10}  {'KS stat':>9}  {'KS p-val':>10}")
    print("-" * 60)
    for _, row in test_df.head(20).iterrows():
        print(f"  {row['feature']:>8}  {row['cohens_d']:>+9.4f}  {row['mw_pvalue']:>10.2e}  "
              f"{row['ks_statistic']:>9.4f}  {row['ks_pvalue']:>10.2e}")

    return test_df.drop(columns=["abs_d"])


def plot_test_summary(test_df):
    """
    통계 검정 결과 시각화: Cohen's d 분포 + KS statistic 분포 + 상위 20 bar chart

    Parameters
    ----------
    test_df : DataFrame  - statistical_tests()의 반환값
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) Cohen's d 분포
    axes[0].hist(test_df["cohens_d"].values, bins=80, edgecolor="black", color="steelblue", alpha=0.8)
    axes[0].axvline(x=0.2, color="orange", linestyle="--", alpha=0.7, label="|d|=0.2 (small)")
    axes[0].axvline(x=-0.2, color="orange", linestyle="--", alpha=0.7)
    axes[0].axvline(x=0, color="red", linestyle="-", alpha=0.5)
    axes[0].set_title("Cohen's d 분포 (Y=0 vs Y>0)")
    axes[0].set_xlabel("Cohen's d")
    axes[0].set_ylabel("Feature 수")
    axes[0].legend(fontsize=9)

    # 2) KS statistic 분포
    axes[1].hist(test_df["ks_statistic"].values, bins=80, edgecolor="black", color="coral", alpha=0.8)
    axes[1].set_title("KS Statistic 분포")
    axes[1].set_xlabel("KS Statistic")
    axes[1].set_ylabel("Feature 수")

    # 3) Cohen's d 상위 20개 bar chart
    top20 = test_df.nlargest(20, "cohens_d", keep="first")
    bot20 = test_df.nsmallest(20, "cohens_d", keep="first")
    extreme20 = pd.concat([top20, bot20]).drop_duplicates(subset="feature")
    extreme20 = extreme20.reindex(extreme20["cohens_d"].abs().sort_values(ascending=True).index)

    colors = ["coral" if d < 0 else "steelblue" for d in extreme20["cohens_d"]]
    axes[2].barh(range(len(extreme20)), extreme20["cohens_d"].values, color=colors, edgecolor="black")
    axes[2].set_yticks(range(len(extreme20)))
    axes[2].set_yticklabels(extreme20["feature"].values, fontsize=8)
    axes[2].set_xlabel("Cohen's d")
    axes[2].set_title("Cohen's d 상/하위 Feature (파랑=Y>0에서 큼)")
    axes[2].axvline(x=0, color="gray", linestyle="-", alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_group_distributions(grp_zero, grp_pos, test_df, n=8):
    """
    Cohen's d 상위 N개 feature의 Y=0 vs Y>0 분포를 violin plot으로 비교

    Parameters
    ----------
    grp_zero : DataFrame  - Y=0 그룹
    grp_pos  : DataFrame  - Y>0 그룹
    test_df  : DataFrame  - statistical_tests() 반환값
    n : int  - 시각화할 feature 수 (기본 8)
    """
    top_feats = test_df.head(n)["feature"].tolist()

    n_rows = (n + 3) // 4
    fig, axes = plt.subplots(n_rows, 4, figsize=(18, 4.5 * n_rows))
    axes = np.array(axes).flatten()

    for i, feat in enumerate(top_feats):
        vals_zero = grp_zero[feat].dropna()
        vals_pos = grp_pos[feat].dropna()

        # 샘플링 (시각화 속도)
        rng = np.random.RandomState(SEED)
        n_sample = 3000
        if len(vals_zero) > n_sample:
            vals_zero = vals_zero.sample(n_sample, random_state=SEED)
        if len(vals_pos) > n_sample:
            vals_pos = vals_pos.sample(n_sample, random_state=SEED)

        data = pd.DataFrame({
            "value": pd.concat([vals_zero, vals_pos], ignore_index=True),
            "group": ["Y=0"] * len(vals_zero) + ["Y>0"] * len(vals_pos),
        })

        sns.violinplot(x="group", y="value", data=data, ax=axes[i],
                       palette={"Y=0": "skyblue", "Y>0": "salmon"},
                       inner="quartile", cut=0)

        d_val = test_df.loc[test_df["feature"] == feat, "cohens_d"].values[0]
        axes[i].set_title(f"{feat} (d={d_val:+.3f})", fontsize=10)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Y=0 vs Y>0 분포 비교 (Cohen's d 상위 {n}개)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


def compare_within_positive(xs_dict, ys_train, feat_cols, test_df, q_low=0.1, q_high=0.9, n=8):
    """
    Y>0 내에서 상위 10% vs 하위 10% unit 간 feature 차이 분석
    → 회귀 모델(2단계)에서 어떤 feature가 health 크기를 구분하는지 단서 제공

    Parameters
    ----------
    xs_dict : dict
    ys_train : DataFrame
    feat_cols : list of str
    test_df : DataFrame  - 기존 통계검정 결과 (참고용)
    q_low : float  - 하위 분위수 (기본 0.1)
    q_high : float  - 상위 분위수 (기본 0.9)
    n : int  - 시각화할 feature 수

    Returns
    -------
    pos_test_df : DataFrame  - Y>0 내 상위 vs 하위 통계 검정 결과
    """
    xs_train = xs_dict["train"]
    xs_unit = xs_dict['train_unit_mean'] if 'train_unit_mean' in xs_dict else xs_train.groupby(KEY_COL)[feat_cols].mean()
    merged = xs_unit.merge(ys_train, left_index=True, right_on=KEY_COL, how="inner")

    pos_only = merged[merged[TARGET_COL] > 0].copy()
    threshold_low = pos_only[TARGET_COL].quantile(q_low)
    threshold_high = pos_only[TARGET_COL].quantile(q_high)

    grp_low = pos_only[pos_only[TARGET_COL] <= threshold_low]
    grp_high = pos_only[pos_only[TARGET_COL] >= threshold_high]

    print("=" * 60)
    print(f"Y>0 내부 비교: 하위 {q_low*100:.0f}% vs 상위 {q_high*100:.0f}%")
    print("=" * 60)
    print(f"Y>0 전체: {len(pos_only):,} units")
    print(f"하위 그룹 (health <= {threshold_low:.6f}): {len(grp_low):,} units")
    print(f"상위 그룹 (health >= {threshold_high:.6f}): {len(grp_high):,} units")
    print()

    # 각 feature에 대해 Mann-Whitney U + Cohen's d
    results = []
    for col in feat_cols:
        v_low = grp_low[col].dropna().values
        v_high = grp_high[col].dropna().values
        if len(v_low) < 2 or len(v_high) < 2:
            continue

        mw_stat, mw_p = stats.mannwhitneyu(v_low, v_high, alternative="two-sided")

        n0, n1 = len(v_low), len(v_high)
        m0, m1 = v_low.mean(), v_high.mean()
        s0, s1 = v_low.std(ddof=1), v_high.std(ddof=1)
        pooled = np.sqrt(((n0 - 1) * s0**2 + (n1 - 1) * s1**2) / (n0 + n1 - 2))
        d = (m1 - m0) / pooled if pooled > 0 else 0.0

        results.append({"feature": col, "mw_pvalue": mw_p, "cohens_d": d})

    pos_test_df = pd.DataFrame(results)
    pos_test_df["abs_d"] = pos_test_df["cohens_d"].abs()
    pos_test_df = pos_test_df.sort_values("abs_d", ascending=False).reset_index(drop=True)

    print(f"Cohen's d 상위 15개 Feature (Y>0 내 상위 vs 하위):")
    print(f"  {'Feature':>8}  {'Cohen d':>9}  {'MW p-val':>10}")
    print("-" * 40)
    for _, row in pos_test_df.head(15).iterrows():
        print(f"  {row['feature']:>8}  {row['cohens_d']:>+9.4f}  {row['mw_pvalue']:>10.2e}")

    # 시각화: 상위 N개 feature의 하위 vs 상위 violin
    top_feats = pos_test_df.head(n)["feature"].tolist()

    n_rows = (n + 3) // 4
    fig, axes = plt.subplots(n_rows, 4, figsize=(18, 4.5 * n_rows))
    axes = np.array(axes).flatten()

    for i, feat in enumerate(top_feats):
        v_low = grp_low[feat].dropna()
        v_high = grp_high[feat].dropna()

        rng = np.random.RandomState(SEED)
        n_sample = 2000
        if len(v_low) > n_sample:
            v_low = v_low.sample(n_sample, random_state=SEED)
        if len(v_high) > n_sample:
            v_high = v_high.sample(n_sample, random_state=SEED)

        data = pd.DataFrame({
            "value": pd.concat([v_low, v_high], ignore_index=True),
            "group": [f"하위 {q_low*100:.0f}%"] * len(v_low) + [f"상위 {q_high*100:.0f}%"] * len(v_high),
        })

        sns.violinplot(x="group", y="value", data=data, ax=axes[i],
                       palette={f"하위 {q_low*100:.0f}%": "lightgreen", f"상위 {q_high*100:.0f}%": "tomato"},
                       inner="quartile", cut=0)

        d_val = pos_test_df.loc[pos_test_df["feature"] == feat, "cohens_d"].values[0]
        axes[i].set_title(f"{feat} (d={d_val:+.3f})", fontsize=10)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Y>0 내부: 하위 {q_low*100:.0f}% vs 상위 {q_high*100:.0f}% 분포 비교", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()

    return pos_test_df.drop(columns=["abs_d"])