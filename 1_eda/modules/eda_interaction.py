"""
EDA 모듈: Feature Interaction 탐색
- 전체 feature 쌍의 상호작용(ratio, product, difference)이
  단일 feature보다 높은 Mutual Information을 보이는지 탐색
- MI 기반으로 비선형 의존성까지 포착 (Pearson/Spearman 대비 우위)
- Shallow Decision Tree로 자연스러운 feature split 조합 발굴
- 노트북에서 import eda_interaction as ia 로 사용
"""
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.feature_selection import mutual_info_regression
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
    xs_unit = xs_dict['train_unit_mean'] if 'train_unit_mean' in xs_dict else xs_train.groupby(KEY_COL)[feat_cols].mean()
    merged = xs_unit.merge(ys_train, left_index=True, right_on=KEY_COL, how="inner")

    # NaN median imputation
    for col in feat_cols:
        if merged[col].isna().any():
            merged[col] = merged[col].fillna(merged[col].median())

    # 분산 0인 feature 제외
    valid_feats = [c for c in feat_cols if merged[c].std() > 0]

    return merged, valid_feats


def pairwise_interaction_corr(xs_dict, ys_train, feat_cols,
                              top_n_feats=None, n_top_pairs=20,
                              batch_size=1000):
    """
    전체 feature 쌍의 상호작용(ratio, product, difference)과 target 간 MI 분석

    반도체 맥락:
    - 개별 WT feature와 health의 선형 상관이 극도로 약함 (max |r|=0.037)
    - Mutual Information은 비선형 의존성까지 포착하므로 interaction 평가에 적합
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
    top_n_feats : int or None
        None이면 전체 feature 사용, 정수면 단일 MI 상위 N개만 사용
    n_top_pairs : int
        출력할 상위 interaction feature 수 (기본 20)
    batch_size : int
        MI 계산 시 한 번에 처리할 pair 수 (기본 1000)

    Returns
    -------
    interaction_df : DataFrame
        columns = [feat_a, feat_b, operation, mi,
                   best_single_mi, improvement]
        improvement 기준 내림차순 정렬
    """
    merged, valid_feats = _prepare_unit_data(xs_dict, ys_train, feat_cols)
    target = merged[TARGET_COL].values

    # 1) 단일 feature MI 계산
    print(f"단일 feature-target MI 계산 중 ({len(valid_feats)}개)...")
    X_all = merged[valid_feats].values
    single_mi_vals = mutual_info_regression(
        X_all, target, random_state=SEED, n_jobs=-1
    )
    single_mi = dict(zip(valid_feats, single_mi_vals))
    single_mi_series = pd.Series(single_mi).sort_values(ascending=False)

    print(f"  단일 feature 최대 MI = {single_mi_series.iloc[0]:.6f} "
          f"({single_mi_series.index[0]})")

    # 2) feature 선정
    if top_n_feats is not None:
        use_feats = single_mi_series.head(top_n_feats).index.tolist()
        label = f"상위 {top_n_feats}개"
    else:
        use_feats = valid_feats
        label = "전체"

    n_feats = len(use_feats)
    all_pairs = list(combinations(use_feats, 2))
    n_pairs = len(all_pairs)
    print(f"{label} feature {n_feats}개 → {n_pairs:,}개 쌍 탐색")

    # 3) batch MI 계산
    results = []
    epsilon = 1e-8

    for batch_start in range(0, n_pairs, batch_size):
        batch_end = min(batch_start + batch_size, n_pairs)
        batch_pairs = all_pairs[batch_start:batch_end]

        interaction_cols = []
        interaction_meta = []

        for feat_a, feat_b in batch_pairs:
            a = merged[feat_a].values
            b = merged[feat_b].values
            best_single = max(single_mi[feat_a], single_mi[feat_b])

            # product: a * b
            prod = a * b
            prod = np.where(np.isfinite(prod), prod, np.nan)
            med = np.nanmedian(prod)
            prod = np.where(np.isnan(prod), med if np.isfinite(med) else 0.0, prod)
            interaction_cols.append(prod)
            interaction_meta.append((feat_a, feat_b, "product", best_single))

            # ratio: a / (b + epsilon)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ratio = a / (b + epsilon)
            ratio = np.where(np.isfinite(ratio), ratio, np.nan)
            med = np.nanmedian(ratio)
            ratio = np.where(np.isnan(ratio), med if np.isfinite(med) else 0.0, ratio)
            interaction_cols.append(ratio)
            interaction_meta.append((feat_a, feat_b, "ratio", best_single))

            # difference: a - b
            diff = a - b
            diff = np.where(np.isfinite(diff), diff, np.nan)
            med = np.nanmedian(diff)
            diff = np.where(np.isnan(diff), med if np.isfinite(med) else 0.0, diff)
            interaction_cols.append(diff)
            interaction_meta.append((feat_a, feat_b, "difference", best_single))

        X_batch = np.column_stack(interaction_cols)
        mi_batch = mutual_info_regression(
            X_batch, target, random_state=SEED, n_jobs=-1
        )

        for (feat_a, feat_b, op, best_single), mi_val in zip(
            interaction_meta, mi_batch
        ):
            if mi_val > best_single:
                results.append({
                    "feat_a": feat_a, "feat_b": feat_b,
                    "operation": op, "mi": mi_val,
                    "best_single_mi": best_single,
                    "improvement": mi_val - best_single,
                })

        pct = batch_end / n_pairs * 100
        print(f"\r  진행: {batch_end:,}/{n_pairs:,} ({pct:.1f}%)", end="", flush=True)

    print()  # 진행률 후 줄바꿈

    # 4) 결과 정리
    if not results:
        print("\n단일 feature보다 높은 MI를 가진 interaction이 없습니다.")
        return pd.DataFrame(columns=[
            "feat_a", "feat_b", "operation", "mi",
            "best_single_mi", "improvement",
        ])

    interaction_df = pd.DataFrame(results)
    interaction_df = interaction_df.sort_values(
        "improvement", ascending=False
    ).reset_index(drop=True)

    # 5) 상위 결과 출력
    print(f"\n{'='*80}")
    print(f"단일 feature보다 높은 MI를 가진 Interaction: {len(interaction_df)}개")
    print(f"{'='*80}")

    op_symbols = {"ratio": "/", "product": "*", "difference": "-"}
    display_n = min(n_top_pairs, len(interaction_df))
    print(f"\n상위 {display_n}개 Interaction Feature:")
    print(f"  {'#':>3}  {'Interaction':>30}  {'MI':>10}  "
          f"{'Best Single MI':>15}  {'Improvement':>12}")
    print("-" * 80)

    for idx, row in interaction_df.head(display_n).iterrows():
        sym = op_symbols.get(row["operation"], row["operation"])
        formula = f"{row['feat_a']} {sym} {row['feat_b']}"
        print(f"  {idx+1:>3}  {formula:>30}  {row['mi']:>10.6f}  "
              f"{row['best_single_mi']:>15.6f}  "
              f"{row['improvement']:>+12.6f}")

    # 연산별 통계
    print(f"\n연산별 발견 수:")
    for op in ["ratio", "product", "difference"]:
        subset = interaction_df[interaction_df["operation"] == op]
        if len(subset) > 0:
            print(f"  {op:>12}: {len(subset):>4}개  "
                  f"(최대 MI = {subset['mi'].max():.6f})")

    return interaction_df


def _compute_interaction_vals(a, b, op, epsilon=1e-8):
    """interaction 값 계산 + inf/nan → median 대체"""
    if op == "product":
        vals = a * b
    elif op == "ratio":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vals = a / (b + epsilon)
    else:  # difference
        vals = a - b
    vals = np.where(np.isfinite(vals), vals, np.nan)
    med = np.nanmedian(vals)
    vals = np.where(np.isnan(vals), med if np.isfinite(med) else 0.0, vals)
    return vals


def plot_top_interactions(xs_dict, ys_train, feat_cols, interaction_df, n=6):
    """
    상위 N개 interaction feature vs health scatter plot + MI 비교
    전체 / Y>0 / Y>0+clip99 세 조건에서 MI를 비교 출력

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

    # Y>0 서브셋
    pos_mask = target > 0
    merged_pos = merged[pos_mask]
    target_pos = target[pos_mask]

    # Y>0 + 이상치 제거 서브셋
    upper = target_pos.quantile(0.99)
    clip_mask = target_pos <= upper
    merged_clip = merged_pos[clip_mask]
    target_clip = target_pos[clip_mask]

    display_n = min(n, len(interaction_df))
    op_symbols = {"ratio": "/", "product": "*", "difference": "-"}

    # --- 전체 scatter ---
    n_cols = 3
    n_rows = (display_n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5.5 * n_rows))
    axes = np.array(axes).flatten()

    sample_idx = merged.sample(
        n=min(5000, len(merged)), random_state=SEED
    ).index

    for i, (_, row) in enumerate(interaction_df.head(display_n).iterrows()):
        feat_a, feat_b, op = row["feat_a"], row["feat_b"], row["operation"]
        mi_val = row["mi"]
        sym = op_symbols.get(op, op)
        formula = f"{feat_a} {sym} {feat_b}"

        inter_vals = _compute_interaction_vals(
            merged.loc[sample_idx, feat_a].values,
            merged.loc[sample_idx, feat_b].values, op
        )
        y_vals = target.loc[sample_idx].values
        mask = np.isfinite(inter_vals)

        axes[i].scatter(inter_vals[mask], y_vals[mask],
                        alpha=0.15, s=5, color="steelblue")
        axes[i].set_xlabel(formula, fontsize=10)
        axes[i].set_ylabel("health")
        axes[i].set_title(f"{formula}\nMI = {mi_val:.5f}", fontsize=10)

    for j in range(display_n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"상위 {display_n}개 Interaction Feature vs Target (전체)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()

    # --- Y>0 scatter ---
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(18, 5.5 * n_rows))
    axes2 = np.array(axes2).flatten()

    sample_pos_idx = merged_pos.sample(
        n=min(5000, len(merged_pos)), random_state=SEED
    ).index

    for i, (_, row) in enumerate(interaction_df.head(display_n).iterrows()):
        feat_a, feat_b, op = row["feat_a"], row["feat_b"], row["operation"]
        sym = op_symbols.get(op, op)
        formula = f"{feat_a} {sym} {feat_b}"

        inter_vals = _compute_interaction_vals(
            merged_pos.loc[sample_pos_idx, feat_a].values,
            merged_pos.loc[sample_pos_idx, feat_b].values, op
        )
        y_vals = target_pos.loc[sample_pos_idx].values
        mask = np.isfinite(inter_vals)

        mi_pos = mutual_info_regression(
            inter_vals[mask].reshape(-1, 1),
            y_vals[mask], random_state=SEED, n_jobs=-1
        )[0]

        axes2[i].scatter(inter_vals[mask], y_vals[mask],
                         alpha=0.2, s=8, color="coral")
        axes2[i].set_xlabel(formula, fontsize=10)
        axes2[i].set_ylabel("health")
        axes2[i].set_title(f"{formula}\nMI(Y>0) = {mi_pos:.5f}", fontsize=10)

    for j in range(display_n, len(axes2)):
        axes2[j].set_visible(False)

    plt.suptitle(f"상위 {display_n}개 Interaction Feature vs Target "
                 f"(Y>0만, n={pos_mask.sum():,})", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()

    # --- Y>0 + clip99 scatter ---
    sample_clip_idx = merged_clip.sample(
        n=min(5000, len(merged_clip)), random_state=SEED
    ).index

    fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(18, 5.5 * n_rows))
    axes3 = np.array(axes3).flatten()

    for i, (_, row) in enumerate(interaction_df.head(display_n).iterrows()):
        feat_a, feat_b, op = row["feat_a"], row["feat_b"], row["operation"]
        sym = op_symbols.get(op, op)
        formula = f"{feat_a} {sym} {feat_b}"

        inter_vals = _compute_interaction_vals(
            merged_clip.loc[sample_clip_idx, feat_a].values,
            merged_clip.loc[sample_clip_idx, feat_b].values, op
        )
        y_vals = target_clip.loc[sample_clip_idx].values
        mask = np.isfinite(inter_vals)

        mi_clip = mutual_info_regression(
            inter_vals[mask].reshape(-1, 1),
            y_vals[mask], random_state=SEED, n_jobs=-1
        )[0]

        axes3[i].scatter(inter_vals[mask], y_vals[mask],
                         alpha=0.2, s=8, color="mediumseagreen")
        axes3[i].set_xlabel(formula, fontsize=10)
        axes3[i].set_ylabel("health")
        axes3[i].set_title(f"{formula}\nMI(clip99) = {mi_clip:.5f}", fontsize=10)

    for j in range(display_n, len(axes3)):
        axes3[j].set_visible(False)

    plt.suptitle(f"상위 {display_n}개 Interaction Feature vs Target "
                 f"(Y>0 + 상위1% 제거, n={clip_mask.sum():,})",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()

    # --- 전체 vs Y>0 vs clip99 MI 비교 테이블 ---
    print(f"\n{'='*75}")
    print(f"전체 vs Y>0 vs Y>0+clip99 MI 비교")
    print(f"  (전체: {len(merged):,}, Y>0: {pos_mask.sum():,}, "
          f"Y>0+clip99: {clip_mask.sum():,}, 상위1% 기준: {upper:.4f})")
    print(f"{'='*75}")
    print(f"  {'Interaction':>30}  {'MI(전체)':>10}  {'MI(Y>0)':>10}  {'MI(clip99)':>10}")
    print("-" * 70)

    for _, row in interaction_df.head(display_n).iterrows():
        feat_a, feat_b, op = row["feat_a"], row["feat_b"], row["operation"]
        sym = op_symbols.get(op, op)
        formula = f"{feat_a} {sym} {feat_b}"
        mi_all = row["mi"]

        # Y>0 MI
        iv_p = _compute_interaction_vals(
            merged_pos[feat_a].values, merged_pos[feat_b].values, op
        )
        mi_p = mutual_info_regression(
            iv_p.reshape(-1, 1), target_pos.values,
            random_state=SEED, n_jobs=-1
        )[0]

        # clip99 MI
        iv_c = _compute_interaction_vals(
            merged_clip[feat_a].values, merged_clip[feat_b].values, op
        )
        mi_c = mutual_info_regression(
            iv_c.reshape(-1, 1), target_clip.values,
            random_state=SEED, n_jobs=-1
        )[0]

        print(f"  {formula:>30}  {mi_all:>10.5f}  {mi_p:>10.5f}  {mi_c:>10.5f}")


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
        학습된 트리 모델 (feature_names_in_ 속성 포함)
    importance_df : DataFrame
        columns = [feature, importance], importance 기준 내림차순 정렬
    """
    merged, valid_feats = _prepare_unit_data(xs_dict, ys_train, feat_cols)
    target = merged[TARGET_COL]

    # target 상관 상위 N개 feature 선정
    single_corr = merged[valid_feats].corrwith(target).abs().sort_values(ascending=False)
    use_feats = single_corr.head(top_n_feats).index.tolist()
    print(f"사용 feature: {len(use_feats)}개 (target |r| 상위 {top_n_feats}개)")

    X = merged[use_feats]  # DataFrame 유지 → feature_names_in_ 자동 설정
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


def _extract_tree_paths(tree, feature_names):
    """
    DecisionTree의 모든 root-to-leaf 경로에서 사용된 feature 조합 추출

    Parameters
    ----------
    tree : DecisionTreeRegressor (학습 완료)
    feature_names : list of str

    Returns
    -------
    paths : list of list of str  - 각 경로에서 사용된 feature 이름 리스트
    """
    tree_ = tree.tree_
    paths = []

    def recurse(node, current_path):
        if tree_.children_left[node] == tree_.children_right[node]:
            # leaf node
            if len(current_path) > 0:
                paths.append(list(current_path))
            return

        feat_idx = tree_.feature[node]
        appended = False
        if feat_idx >= 0 and feat_idx < len(feature_names):
            feat_name = feature_names[feat_idx]
            current_path.append(feat_name)
            appended = True

        recurse(tree_.children_left[node], current_path)
        recurse(tree_.children_right[node], current_path)

        if appended:
            current_path.pop()

    recurse(0, [])
    return paths


def multi_way_split_analysis(xs_dict, ys_train, feat_cols,
                             n_trees=50, max_depth=4, top_n_feats=50):
    """
    다수의 Shallow Tree에서 반복 등장하는 feature 조합(co-occurrence) 발굴

    단일 트리는 불안정하므로, feature/데이터를 랜덤 서브샘플링하여
    n_trees개 트리를 학습하고, 같은 경로에 함께 등장하는 feature 쌍의
    빈도를 카운트한다. 높은 빈도 = 견고한 상호작용.

    Parameters
    ----------
    xs_dict : dict
    ys_train : DataFrame
    feat_cols : list of str
    n_trees : int  - 학습할 트리 수 (기본 50)
    max_depth : int  - 트리 최대 깊이 (기본 4)
    top_n_feats : int  - target 상관 상위 N개 feature 사용 (기본 50)

    Returns
    -------
    cooccurrence_df : DataFrame  (feat_a, feat_b, count, pct_of_trees)
    """
    merged, valid_feats = _prepare_unit_data(xs_dict, ys_train, feat_cols)
    target = merged[TARGET_COL]

    # target 상관 상위 feature
    single_corr = merged[valid_feats].corrwith(target).abs().sort_values(ascending=False)
    use_feats = single_corr.head(top_n_feats).index.tolist()
    print(f"Multi-way Split 분석: {len(use_feats)}개 feature, {n_trees}개 tree (depth={max_depth})")

    from collections import Counter
    pair_counter = Counter()
    feat_counter = Counter()

    for i in range(n_trees):
        rng = np.random.RandomState(SEED + i)

        # 70% feature 서브샘플링
        n_feat_sample = max(int(len(use_feats) * 0.7), 3)
        feat_sample = rng.choice(use_feats, n_feat_sample, replace=False).tolist()

        # 80% 데이터 서브샘플링
        n_data_sample = int(len(merged) * 0.8)
        data_idx = rng.choice(len(merged), n_data_sample, replace=False)

        X = merged.iloc[data_idx][feat_sample].values
        y = target.iloc[data_idx].values

        tree = DecisionTreeRegressor(
            max_depth=max_depth, min_samples_leaf=30, random_state=SEED + i
        )
        tree.fit(X, y)

        # 경로 추출
        paths = _extract_tree_paths(tree, feat_sample)

        for path in paths:
            unique_feats = list(set(path))
            for f in unique_feats:
                feat_counter[f] += 1
            # 경로 내 모든 feature 쌍
            for fi in range(len(unique_feats)):
                for fj in range(fi + 1, len(unique_feats)):
                    pair = tuple(sorted([unique_feats[fi], unique_feats[fj]]))
                    pair_counter[pair] += 1

    # 결과 정리
    records = []
    for (fa, fb), cnt in pair_counter.most_common():
        records.append({
            "feat_a": fa,
            "feat_b": fb,
            "count": cnt,
            "pct_of_trees": cnt / n_trees * 100,
        })

    cooccurrence_df = pd.DataFrame(records)
    if cooccurrence_df.empty:
        print("공동 출현 쌍이 없습니다.")
        return cooccurrence_df

    # 출력
    print(f"\n{'='*75}")
    print(f"Multi-way Split 분석 결과 ({n_trees}개 tree)")
    print(f"{'='*75}")
    print(f"발견된 feature 쌍: {len(cooccurrence_df):,}개")
    print(f"트리에 사용된 고유 feature: {len(feat_counter)}개")

    display_n = min(20, len(cooccurrence_df))
    print(f"\n상위 {display_n}개 Feature 공동 출현 쌍:")
    print(f"  {'#':>3}  {'Feature A':>10}  {'Feature B':>10}  {'Count':>7}  {'% of Trees':>11}")
    print("-" * 55)
    for idx, row in cooccurrence_df.head(display_n).iterrows():
        print(f"  {idx+1:>3}  {row['feat_a']:>10}  {row['feat_b']:>10}  "
              f"{int(row['count']):>7}  {row['pct_of_trees']:>10.1f}%")

    # 히트맵: 상위 15개 feature
    top_feats_list = [f for f, _ in feat_counter.most_common(15)]
    n_hm = len(top_feats_list)
    heatmap = np.zeros((n_hm, n_hm))
    feat_to_idx = {f: i for i, f in enumerate(top_feats_list)}

    for (fa, fb), cnt in pair_counter.items():
        if fa in feat_to_idx and fb in feat_to_idx:
            i, j = feat_to_idx[fa], feat_to_idx[fb]
            heatmap[i, j] = cnt
            heatmap[j, i] = cnt

    fig, ax = plt.subplots(figsize=(10, 8))
    import seaborn as sns
    sns.heatmap(heatmap, xticklabels=top_feats_list, yticklabels=top_feats_list,
                cmap="YlOrRd", annot=True, fmt=".0f", ax=ax,
                cbar_kws={"label": "공동 출현 횟수"})
    ax.set_title(f"Feature 공동 출현 히트맵 (상위 {n_hm}개, {n_trees} trees)", fontsize=13)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    plt.tight_layout()
    plt.show()

    return cooccurrence_df


def plot_interaction_network(cooccurrence_df, min_count=5):
    """
    Feature 상호작용 네트워크 시각화 (원형 배치)

    Parameters
    ----------
    cooccurrence_df : DataFrame  - multi_way_split_analysis() 반환값
    min_count : int  - 최소 공동 출현 횟수 필터
    """
    if cooccurrence_df.empty:
        print("시각화할 데이터가 없습니다.")
        return

    filtered = cooccurrence_df[cooccurrence_df["count"] >= min_count].copy()
    if filtered.empty:
        print(f"count >= {min_count}인 쌍이 없습니다.")
        return

    # 고유 feature 추출 (상위 20개)
    all_feats = pd.concat([filtered["feat_a"], filtered["feat_b"]]).value_counts()
    top_feats = all_feats.head(20).index.tolist()
    feat_set = set(top_feats)

    # 해당 feature만 포함하는 쌍
    edges = filtered[
        filtered["feat_a"].isin(feat_set) & filtered["feat_b"].isin(feat_set)
    ]

    if edges.empty:
        print("시각화할 edge가 없습니다.")
        return

    # 원형 배치
    n = len(top_feats)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    positions = {f: (np.cos(a), np.sin(a)) for f, a in zip(top_feats, angles)}

    # 노드 크기 = 총 공동 출현
    node_sizes = {f: all_feats.get(f, 1) for f in top_feats}
    max_size = max(node_sizes.values())

    fig, ax = plt.subplots(figsize=(12, 12))

    # edge 그리기
    max_count = edges["count"].max()
    for _, row in edges.iterrows():
        fa, fb, cnt = row["feat_a"], row["feat_b"], row["count"]
        if fa not in positions or fb not in positions:
            continue
        x = [positions[fa][0], positions[fb][0]]
        y = [positions[fa][1], positions[fb][1]]
        lw = 0.5 + (cnt / max_count) * 4
        alpha = 0.2 + (cnt / max_count) * 0.6
        ax.plot(x, y, color="steelblue", linewidth=lw, alpha=alpha)

    # 노드 그리기
    for feat in top_feats:
        x, y = positions[feat]
        size = 100 + (node_sizes[feat] / max_size) * 600
        ax.scatter(x, y, s=size, color="coral", edgecolors="black",
                   linewidth=1.5, zorder=5)
        ax.annotate(feat, (x, y), fontsize=8, ha="center", va="center",
                    fontweight="bold", zorder=6)

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Feature Interaction Network (min_count={min_count})", fontsize=14)
    plt.tight_layout()
    plt.show()
