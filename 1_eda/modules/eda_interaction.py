"""
EDA 모듈: Feature Interaction 탐색
- 단일 feature-target 상관이 max |r|=0.037으로 극도로 약한 상황에서
  feature 간 상호작용(ratio, product, difference)이 더 높은 상관을 보이는지 탐색
- Shallow Decision Tree로 자연스러운 feature split 조합 발굴
- 노트북에서 import eda_interaction as ia 로 사용
"""
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from utils.config import KEY_COL, TARGET_COL, SEED


def _prepare_unit_data(xs_dict, ys_train, feat_cols):
    """
    Train die -> unit 평균 집계 + target merge + NaN median imputation

    Parameters
    ----------
    xs_dict : dict
        {"train": DataFrame, ...} split별 die-level 데이터
    ys_train : DataFrame
        train Y 데이터 (ufs_serial, health 컬럼)
    feat_cols : list of str
        feature 컬럼명 리스트 (X0~X1086 등)

    Returns
    -------
    merged : DataFrame
        unit-level로 집계된 X + Y 합본 (NaN imputed)
    valid_feats : list of str
        분산 > 0인 feature 컬럼명 리스트
    """
    xs_train = xs_dict["train"]
    xs_unit = xs_train.groupby(KEY_COL)[feat_cols].mean()
    merged = xs_unit.merge(ys_train, left_index=True, right_on=KEY_COL, how="inner")

    # NaN median imputation
    for col in feat_cols:
        if merged[col].isna().any():
            merged[col] = merged[col].fillna(merged[col].median())

    # 분산 0인 feature 제외
    valid_feats = [c for c in feat_cols if merged[c].std() > 0]

    return merged, valid_feats


def pairwise_interaction_corr(xs_dict, ys_train, feat_cols,
                              top_n_feats=30, n_top_pairs=20):
    """
    상위 feature 쌍의 상호작용(ratio, product, difference)과 target 간 상관 분석

    반도체 맥락:
    - 개별 WT feature와 health의 선형 상관이 극도로 약함 (max |r|=0.037)
    - 두 feature의 비율(공정 균형), 곱(복합 효과), 차이(편차)가
      불량 예측에 더 강한 신호를 줄 수 있음

    Parameters
    ----------
    xs_dict : dict
        {"train": DataFrame, ...} split별 die-level 데이터
    ys_train : DataFrame
        train Y 데이터 (ufs_serial, health 컬럼)
    feat_cols : list of str
        feature 컬럼명 리스트
    top_n_feats : int
        target과 |Pearson r| 상위 N개 feature를 선정하여 조합 탐색 (기본 30)
    n_top_pairs : int
        출력할 상위 interaction feature 수 (기본 20)

    Returns
    -------
    interaction_df : DataFrame
        columns = [feat_a, feat_b, operation, corr, abs_corr,
                   best_single_corr, improvement_over_best_single]
        improvement_over_best_single 기준 내림차순 정렬
    """
    merged, valid_feats = _prepare_unit_data(xs_dict, ys_train, feat_cols)
    target = merged[TARGET_COL]

    # 1) 각 feature와 target의 Pearson 상관 계산
    print(f"단일 feature-target 상관 계산 중 ({len(valid_feats)}개)...")
    single_corr = {}
    for col in valid_feats:
        r = merged[col].corr(target)
        if pd.notna(r):
            single_corr[col] = r
    single_corr = pd.Series(single_corr)
    single_abs = single_corr.abs().sort_values(ascending=False)

    # 2) 상위 N개 feature 선정
    top_feats = single_abs.head(top_n_feats).index.tolist()
    n_pairs = len(list(combinations(top_feats, 2)))
    print(f"상위 {len(top_feats)}개 feature 선정 → {n_pairs}개 쌍 탐색")
    print(f"  단일 feature 최대 |r| = {single_abs.iloc[0]:.6f} ({single_abs.index[0]})")

    # 3) 모든 쌍에 대해 3가지 연산 수행
    results = []
    for feat_a, feat_b in combinations(top_feats, 2):
        a_vals = merged[feat_a].values
        b_vals = merged[feat_b].values
        best_single = max(abs(single_corr[feat_a]), abs(single_corr[feat_b]))

        # ratio: a / (b + epsilon)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            epsilon = 1e-8
            ratio_vals = a_vals / (b_vals + epsilon)
            # inf/nan 제거 후 상관 계산
            mask_ratio = np.isfinite(ratio_vals)
            if mask_ratio.sum() > 100:
                r_ratio = pd.Series(ratio_vals[mask_ratio]).corr(
                    pd.Series(target.values[mask_ratio]))
                if pd.notna(r_ratio) and abs(r_ratio) > best_single:
                    results.append({
                        "feat_a": feat_a, "feat_b": feat_b,
                        "operation": "ratio",
                        "corr": r_ratio,
                        "abs_corr": abs(r_ratio),
                        "best_single_corr": best_single,
                        "improvement_over_best_single": abs(r_ratio) - best_single,
                    })

        # product: a * b
        prod_vals = a_vals * b_vals
        mask_prod = np.isfinite(prod_vals)
        if mask_prod.sum() > 100:
            r_prod = pd.Series(prod_vals[mask_prod]).corr(
                pd.Series(target.values[mask_prod]))
            if pd.notna(r_prod) and abs(r_prod) > best_single:
                results.append({
                    "feat_a": feat_a, "feat_b": feat_b,
                    "operation": "product",
                    "corr": r_prod,
                    "abs_corr": abs(r_prod),
                    "best_single_corr": best_single,
                    "improvement_over_best_single": abs(r_prod) - best_single,
                })

        # difference: a - b
        diff_vals = a_vals - b_vals
        mask_diff = np.isfinite(diff_vals)
        if mask_diff.sum() > 100:
            r_diff = pd.Series(diff_vals[mask_diff]).corr(
                pd.Series(target.values[mask_diff]))
            if pd.notna(r_diff) and abs(r_diff) > best_single:
                results.append({
                    "feat_a": feat_a, "feat_b": feat_b,
                    "operation": "difference",
                    "corr": r_diff,
                    "abs_corr": abs(r_diff),
                    "best_single_corr": best_single,
                    "improvement_over_best_single": abs(r_diff) - best_single,
                })

    # 4) 결과 정리
    if not results:
        print("\n단일 feature보다 높은 |r|을 가진 interaction이 없습니다.")
        return pd.DataFrame(columns=[
            "feat_a", "feat_b", "operation", "corr", "abs_corr",
            "best_single_corr", "improvement_over_best_single",
        ])

    interaction_df = pd.DataFrame(results)
    interaction_df = interaction_df.sort_values(
        "improvement_over_best_single", ascending=False
    ).reset_index(drop=True)

    # 5) 상위 결과 출력
    print(f"\n{'='*80}")
    print(f"단일 feature보다 높은 |r|을 가진 Interaction: {len(interaction_df)}개")
    print(f"{'='*80}")

    op_symbols = {"ratio": "/", "product": "*", "difference": "-"}
    display_n = min(n_top_pairs, len(interaction_df))
    print(f"\n상위 {display_n}개 Interaction Feature:")
    print(f"  {'#':>3}  {'Interaction':>30}  {'|r|':>8}  "
          f"{'Best Single |r|':>15}  {'Improvement':>11}")
    print("-" * 80)

    for idx, row in interaction_df.head(display_n).iterrows():
        sym = op_symbols.get(row["operation"], row["operation"])
        formula = f"{row['feat_a']} {sym} {row['feat_b']}"
        print(f"  {idx+1:>3}  {formula:>30}  {row['abs_corr']:>8.5f}  "
              f"{row['best_single_corr']:>15.5f}  "
              f"{row['improvement_over_best_single']:>+11.5f}")

    # 연산별 통계
    print(f"\n연산별 발견 수:")
    for op in ["ratio", "product", "difference"]:
        subset = interaction_df[interaction_df["operation"] == op]
        if len(subset) > 0:
            print(f"  {op:>12}: {len(subset):>4}개  "
                  f"(최대 |r| = {subset['abs_corr'].max():.5f})")

    return interaction_df


def plot_top_interactions(xs_dict, ys_train, feat_cols, interaction_df, n=6):
    """
    상위 N개 interaction feature vs health scatter plot

    Parameters
    ----------
    xs_dict : dict
        {"train": DataFrame, ...} split별 die-level 데이터
    ys_train : DataFrame
        train Y 데이터 (ufs_serial, health 컬럼)
    feat_cols : list of str
        feature 컬럼명 리스트
    interaction_df : DataFrame
        pairwise_interaction_corr() 반환값
    n : int
        시각화할 상위 interaction 수 (기본 6)
    """
    if interaction_df.empty:
        print("시각화할 interaction이 없습니다.")
        return

    merged, _ = _prepare_unit_data(xs_dict, ys_train, feat_cols)
    target = merged[TARGET_COL]

    display_n = min(n, len(interaction_df))
    n_cols = 3
    n_rows = (display_n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5.5 * n_rows))
    axes = np.array(axes).flatten()

    # 시각화용 샘플링
    sample_idx = merged.sample(
        n=min(5000, len(merged)), random_state=SEED
    ).index

    op_symbols = {"ratio": "/", "product": "*", "difference": "-"}
    epsilon = 1e-8

    for i, (_, row) in enumerate(interaction_df.head(display_n).iterrows()):
        feat_a = row["feat_a"]
        feat_b = row["feat_b"]
        op = row["operation"]
        r_val = row["corr"]
        sym = op_symbols.get(op, op)
        formula = f"{feat_a} {sym} {feat_b}"

        a_vals = merged.loc[sample_idx, feat_a].values
        b_vals = merged.loc[sample_idx, feat_b].values
        y_vals = target.loc[sample_idx].values

        if op == "ratio":
            inter_vals = a_vals / (b_vals + epsilon)
        elif op == "product":
            inter_vals = a_vals * b_vals
        else:  # difference
            inter_vals = a_vals - b_vals

        # inf/nan 제거
        mask = np.isfinite(inter_vals)
        axes[i].scatter(inter_vals[mask], y_vals[mask],
                        alpha=0.15, s=5, color="steelblue")
        axes[i].set_xlabel(formula, fontsize=10)
        axes[i].set_ylabel("health")
        axes[i].set_title(f"{formula}\nr = {r_val:+.5f}", fontsize=10)

    for j in range(display_n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"상위 {display_n}개 Interaction Feature vs Target",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


def shallow_tree_analysis(xs_dict, ys_train, feat_cols,
                          max_depth=3, top_n_feats=50):
    """
    Shallow Decision Tree로 feature 간 자연스러운 split 조합 발굴

    반도체 맥락:
    - 얕은 트리는 가장 정보량이 높은 feature split을 자동으로 찾음
    - 트리의 분기 구조 = 자연스러운 feature interaction
    - 예: "X100 > 0.5 이고 X200 < 1.0이면 health 높음" → X100과 X200의 상호작용

    Parameters
    ----------
    xs_dict : dict
        {"train": DataFrame, ...} split별 die-level 데이터
    ys_train : DataFrame
        train Y 데이터 (ufs_serial, health 컬럼)
    feat_cols : list of str
        feature 컬럼명 리스트
    max_depth : int
        트리 최대 깊이 (기본 3). 작을수록 해석이 쉬움
    top_n_feats : int
        target 상관 상위 N개 feature만 사용 (기본 50). 전체 사용 시 과적합 위험

    Returns
    -------
    tree_model : DecisionTreeRegressor
        학습된 트리 모델
    importance_df : DataFrame
        columns = [feature, importance], importance 기준 내림차순 정렬
    """
    merged, valid_feats = _prepare_unit_data(xs_dict, ys_train, feat_cols)
    target = merged[TARGET_COL]

    # target 상관 상위 N개 feature 선정
    single_corr = merged[valid_feats].corrwith(target).abs().sort_values(ascending=False)
    use_feats = single_corr.head(top_n_feats).index.tolist()
    print(f"사용 feature: {len(use_feats)}개 (target |r| 상위 {top_n_feats}개)")

    X = merged[use_feats].values
    y = target.values

    # Shallow tree 학습
    tree_model = DecisionTreeRegressor(
        max_depth=max_depth, random_state=SEED, min_samples_leaf=50
    )
    tree_model.fit(X, y)

    # Feature importance
    importances = tree_model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": use_feats,
        "importance": importances,
    })
    importance_df = importance_df[importance_df["importance"] > 0]
    importance_df = importance_df.sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)

    # 트리 구조 출력
    print(f"\n{'='*70}")
    print(f"Shallow Decision Tree (max_depth={max_depth})")
    print(f"{'='*70}")
    tree_text = export_text(tree_model, feature_names=use_feats, max_depth=max_depth)
    print(tree_text)

    # Feature importance 출력
    print(f"\n{'='*70}")
    print(f"Feature Importance (트리가 사용한 feature: {len(importance_df)}개)")
    print(f"{'='*70}")
    print(f"  {'#':>3}  {'Feature':>10}  {'Importance':>12}  {'누적':>8}")
    print("-" * 45)
    cumsum = 0.0
    for idx, row in importance_df.iterrows():
        cumsum += row["importance"]
        print(f"  {idx+1:>3}  {row['feature']:>10}  "
              f"{row['importance']:>12.5f}  {cumsum:>8.3f}")

    # Feature importance bar chart
    plot_df = importance_df.head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(plot_df)), plot_df["importance"].values,
            color="steelblue", edgecolor="black")
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["feature"].values, fontsize=10)
    ax.set_xlabel("Feature Importance (Gini / Variance Reduction)")
    ax.set_title(f"Shallow Tree Feature Importance (depth={max_depth})", fontsize=13)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

    return tree_model, importance_df


def plot_tree_splits(tree_model, feature_names):
    """
    Decision Tree를 시각적으로 표시

    Parameters
    ----------
    tree_model : DecisionTreeRegressor
        shallow_tree_analysis()에서 반환된 트리 모델
    feature_names : list of str
        트리 학습에 사용된 feature 이름 리스트
        (shallow_tree_analysis()의 importance_df['feature']가 아닌,
         학습 시 사용한 전체 feature 리스트)
    """
    fig, ax = plt.subplots(figsize=(18, 8))
    plot_tree(
        tree_model,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
        impurity=True,
        proportion=False,
    )
    ax.set_title("Decision Tree Split Structure", fontsize=14)
    plt.tight_layout()
    plt.show()
