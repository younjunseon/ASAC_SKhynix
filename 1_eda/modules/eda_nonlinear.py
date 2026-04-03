"""
EDA 모듈: 비선형 상관 분석 (Mutual Information, Spearman)
- Pearson(선형)으로 max |r|=0.037이었던 데이터에서 비선형 의존성을 탐색
- MI(Mutual Information): 모든 형태의 통계적 의존성을 정보량으로 정량화
- Spearman: 단조(monotonic) 비선형 관계 포착
- 세 방법의 랭킹 비교로 "Pearson이 놓친 핵심 feature" 발굴
- 노트북에서 import eda_nonlinear as nl 로 사용
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from sklearn.feature_selection import mutual_info_regression
from utils.config import KEY_COL, TARGET_COL, SEED


def _prepare_data(xs_dict, ys_train, feat_cols):
    """
    Train die→unit 평균 집계 + target merge + NaN imputation (MI용)

    Returns
    -------
    merged : DataFrame  (unit-level, NaN imputed)
    valid_feats : list  (분산 > 0인 feature만)
    """
    xs_train = xs_dict["train"]
    xs_unit = xs_train.groupby(KEY_COL)[feat_cols].mean()
    merged = xs_unit.merge(ys_train, left_index=True, right_on=KEY_COL, how="inner")

    # MI는 NaN을 처리하지 못하므로 median imputation
    for col in feat_cols:
        if merged[col].isna().any():
            merged[col] = merged[col].fillna(merged[col].median())

    # 분산 0인 feature 제외 (MI 계산 오류 방지)
    valid_feats = [c for c in feat_cols if merged[c].std() > 0]

    return merged, valid_feats


def compute_nonlinear_corr(xs_dict, ys_train, feat_cols, n_neighbors=5):
    """
    Pearson, Spearman, Mutual Information을 한번에 계산

    반도체 맥락:
    - Pearson: 선형 관계만 측정 → WT feature와 health 간 극도로 약한 결과
    - Spearman: 순위 기반 → 단조 비선형 관계(예: feature 증가 시 불량률 단조 증가) 포착
    - MI: 모든 형태의 통계적 의존성 → feature 값이 health 예측에 주는 정보량
      (비단조, 계단형, U자형 등도 포착)

    Parameters
    ----------
    xs_dict : dict
    ys_train : DataFrame
    feat_cols : list of str
    n_neighbors : int  - MI 추정용 KNN 이웃 수 (기본 5, 클수록 smooth)

    Returns
    -------
    result_df : DataFrame
        columns = [feature, pearson, spearman, mi]
        MI 기준 내림차순 정렬
    """
    merged, valid_feats = _prepare_data(xs_dict, ys_train, feat_cols)

    X = merged[valid_feats].values
    y = merged[TARGET_COL].values

    # 1) Pearson
    print("Pearson 상관계수 계산 중...")
    pearson_corr = merged[valid_feats].corrwith(merged[TARGET_COL])

    # 2) Spearman
    print("Spearman 순위 상관계수 계산 중...")
    spearman_corr = {}
    for col in valid_feats:
        rho, _ = sp_stats.spearmanr(merged[col].values, y)
        spearman_corr[col] = rho
    spearman_corr = pd.Series(spearman_corr)

    # 3) Mutual Information
    print(f"Mutual Information 계산 중 (n_neighbors={n_neighbors})... (1~3분 소요)")
    mi_scores = mutual_info_regression(
        X, y, n_neighbors=n_neighbors, random_state=SEED
    )
    mi_series = pd.Series(mi_scores, index=valid_feats)

    # 결과 DataFrame
    result_df = pd.DataFrame({
        "feature": valid_feats,
        "pearson": pearson_corr[valid_feats].values,
        "spearman": spearman_corr[valid_feats].values,
        "mi": mi_series.values,
    })

    # NaN 처리 (상수 feature 등에서 spearmanr/corrwith가 NaN 반환 가능)
    result_df["pearson"] = result_df["pearson"].fillna(0.0)
    result_df["spearman"] = result_df["spearman"].fillna(0.0)
    result_df["mi"] = result_df["mi"].fillna(0.0)

    # 절대값 컬럼 추가
    result_df["abs_pearson"] = result_df["pearson"].abs()
    result_df["abs_spearman"] = result_df["spearman"].abs()

    # MI 기준 내림차순
    result_df = result_df.sort_values("mi", ascending=False).reset_index(drop=True)

    # 랭킹 컬럼 추가
    result_df["rank_pearson"] = result_df["abs_pearson"].rank(ascending=False).astype(int)
    result_df["rank_spearman"] = result_df["abs_spearman"].rank(ascending=False).astype(int)
    result_df["rank_mi"] = result_df["mi"].rank(ascending=False).astype(int)

    print("완료!")
    return result_df


def print_nonlinear_summary(result_df, n=20):
    """
    세 방법의 요약 통계 + 상위 feature 비교 출력

    Parameters
    ----------
    result_df : DataFrame  - compute_nonlinear_corr() 반환값
    n : int  - 상위 출력 수
    """
    print("=" * 75)
    print("비선형 상관 분석 요약")
    print("=" * 75)

    # 각 방법별 요약
    print(f"\n  {'방법':>10}  {'max':>8}  {'mean':>8}  {'median':>8}  {'std':>8}")
    print("-" * 50)
    for method in ["abs_pearson", "abs_spearman", "mi"]:
        label = method.replace("abs_", "")
        vals = result_df[method]
        print(f"  {label:>10}  {vals.max():>8.4f}  {vals.mean():>8.4f}  "
              f"{vals.median():>8.4f}  {vals.std():>8.4f}")

    # MI 상위 N개 출력
    print(f"\n{'='*75}")
    print(f"MI 상위 {n}개 Feature (+ Pearson/Spearman 랭킹 비교)")
    print(f"{'='*75}")
    print(f"  {'Feature':>8}  {'MI':>8}  {'MI순위':>6}  {'Pearson':>9}  {'P순위':>6}  "
          f"{'Spearman':>9}  {'S순위':>6}  {'순위차':>6}")
    print("-" * 75)

    for _, row in result_df.head(n).iterrows():
        rank_diff = int(row["rank_pearson"] - row["rank_mi"])
        print(f"  {row['feature']:>8}  {row['mi']:>8.4f}  {int(row['rank_mi']):>6}  "
              f"{row['pearson']:>+9.4f}  {int(row['rank_pearson']):>6}  "
              f"{row['spearman']:>+9.4f}  {int(row['rank_spearman']):>6}  "
              f"{rank_diff:>+6d}")

    # Pearson 랭킹과 MI 랭킹이 크게 다른 feature (MI가 잡고 Pearson이 놓친 것)
    result_df_copy = result_df.copy()
    result_df_copy["rank_jump"] = result_df_copy["rank_pearson"] - result_df_copy["rank_mi"]
    big_jumps = result_df_copy.nlargest(10, "rank_jump")

    print(f"\n{'='*75}")
    print("MI가 잡고 Pearson이 놓친 Feature (Pearson순위 - MI순위 = 양수)")
    print(f"{'='*75}")
    print(f"  {'Feature':>8}  {'MI순위':>6}  {'P순위':>6}  {'순위차':>6}  {'MI':>8}  {'Pearson':>9}")
    print("-" * 60)
    for _, row in big_jumps.iterrows():
        print(f"  {row['feature']:>8}  {int(row['rank_mi']):>6}  {int(row['rank_pearson']):>6}  "
              f"{int(row['rank_jump']):>+6d}  {row['mi']:>8.4f}  {row['pearson']:>+9.4f}")


def plot_rank_comparison(result_df):
    """
    Pearson vs MI, Pearson vs Spearman 랭킹 비교 시각화
    + MI 상위 30 vs Pearson 상위 30 겹침 분석

    Parameters
    ----------
    result_df : DataFrame
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # ── 1) |Pearson| vs MI scatter ──
    axes[0].scatter(result_df["abs_pearson"], result_df["mi"],
                    alpha=0.3, s=10, color="steelblue")
    axes[0].set_xlabel("|Pearson r|")
    axes[0].set_ylabel("Mutual Information")
    axes[0].set_title("|Pearson| vs MI")

    # MI 상위 10개 하이라이트
    top10_mi = result_df.head(10)
    axes[0].scatter(top10_mi["abs_pearson"], top10_mi["mi"],
                    color="red", s=40, zorder=5, edgecolors="black", linewidth=0.5)
    for _, row in top10_mi.iterrows():
        axes[0].annotate(row["feature"], (row["abs_pearson"], row["mi"]),
                         fontsize=7, ha="left", va="bottom")

    # ── 2) Pearson순위 vs MI순위 scatter ──
    axes[1].scatter(result_df["rank_pearson"], result_df["rank_mi"],
                    alpha=0.2, s=8, color="steelblue")
    lim = len(result_df) + 10
    axes[1].plot([0, lim], [0, lim], "r--", alpha=0.5, linewidth=1)
    axes[1].set_xlabel("Pearson 순위")
    axes[1].set_ylabel("MI 순위")
    axes[1].set_title("Pearson 순위 vs MI 순위")
    axes[1].set_xlim(0, lim)
    axes[1].set_ylim(0, lim)
    axes[1].invert_xaxis()
    axes[1].invert_yaxis()

    # ── 3) 상위 30개 겹침 분석 bar chart ──
    top30_mi = set(result_df.nsmallest(30, "rank_mi")["feature"])
    top30_pearson = set(result_df.nsmallest(30, "rank_pearson")["feature"])
    top30_spearman = set(result_df.nsmallest(30, "rank_spearman")["feature"])

    overlap_mp = len(top30_mi & top30_pearson)
    overlap_ms = len(top30_mi & top30_spearman)
    overlap_ps = len(top30_pearson & top30_spearman)
    overlap_all = len(top30_mi & top30_pearson & top30_spearman)
    mi_only = len(top30_mi - top30_pearson - top30_spearman)

    labels = ["MI∩Pearson", "MI∩Spearman", "Pearson∩Spearman", "3방법 공통", "MI만"]
    values = [overlap_mp, overlap_ms, overlap_ps, overlap_all, mi_only]
    colors = ["steelblue", "coral", "mediumseagreen", "gold", "orchid"]

    axes[2].bar(labels, values, color=colors, edgecolor="black")
    axes[2].set_title("상위 30개 Feature 겹침 (방법별)")
    axes[2].set_ylabel("Feature 수")
    axes[2].tick_params(axis="x", rotation=25, labelsize=9)
    for j, v in enumerate(values):
        axes[2].text(j, v + 0.3, str(v), ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.show()

    # 텍스트 요약
    print(f"\n상위 30개 겹침 분석:")
    print(f"  MI ∩ Pearson    : {overlap_mp}/30")
    print(f"  MI ∩ Spearman   : {overlap_ms}/30")
    print(f"  Pearson ∩ Spearman : {overlap_ps}/30")
    print(f"  3방법 공통       : {overlap_all}/30")
    print(f"  MI에서만 상위    : {mi_only}/30  ← Pearson이 놓친 비선형 feature")


def plot_mi_top_scatter(xs_dict, ys_train, feat_cols, result_df, n=8):
    """
    MI 상위 N개 feature vs health scatter plot
    → 비선형 패턴(계단형, U자형 등)이 실제로 보이는지 시각 확인

    Parameters
    ----------
    xs_dict : dict
    ys_train : DataFrame
    feat_cols : list of str
    result_df : DataFrame
    n : int  - 시각화할 feature 수 (기본 8)
    """
    merged, valid_feats = _prepare_data(xs_dict, ys_train, feat_cols)

    top_feats = result_df.head(n)

    n_rows = (n + 3) // 4
    fig, axes = plt.subplots(n_rows, 4, figsize=(18, 4.5 * n_rows))
    axes = np.array(axes).flatten()

    # 시각화용 샘플링
    sample = merged.sample(n=min(5000, len(merged)), random_state=SEED)

    for i, (_, row) in enumerate(top_feats.iterrows()):
        feat = row["feature"]
        mi_val = row["mi"]
        p_val = row["pearson"]

        axes[i].scatter(sample[feat], sample[TARGET_COL],
                        alpha=0.15, s=5, color="steelblue")
        axes[i].set_xlabel(feat)
        axes[i].set_ylabel("health")
        axes[i].set_title(f"{feat}\nMI={mi_val:.4f}, r={p_val:+.4f}", fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"MI 상위 {n}개 Feature vs Target (비선형 패턴 확인)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()
