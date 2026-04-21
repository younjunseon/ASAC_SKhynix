"""
자동 페어 피처 엔지니어링 (featuretools 영감)

파이프라인
----------
1. LightGBM importance(gain)로 feat_cols 중 top-K feature 선별
2. C(K,2) × ops 로 die-level 페어 파생 생성
3. die→unit 집계(mean) 후 target과의 |corr| 계산
4. baseline (단일 feature max|corr|) × gain_ratio 보다 큰 것만 채택
5. 선별된 페어를 xs_train / xs_val / xs_test 에 die-level 컬럼으로 병합

주의
----
- 페어 선정은 train 전체 target을 사용 → 엄밀한 OOF 추정은 아님.
  페어 후보를 고르는 단계일 뿐이므로 최종 평가는 HPO OOF RMSE로 판정.
- 페어 피처는 원본과 상관이 높아질 수 있음. LGBM 은 robust 하지만
  선형 모델(ElasticNet/Ridge)에 넣기 전에는 post-impute corr 제거 재실행 권장.
"""
from itertools import combinations

import numpy as np
import pandas as pd

from utils.config import KEY_COL, TARGET_COL


# ─── 연산 ─────────────────────────────────────────────
def _op_mul(a, b):   return a * b
def _op_add(a, b):   return a + b
def _op_sub(a, b):   return a - b
def _op_ratio(a, b):
    # b ≈ 0 인 위치는 결과 0 (해석 일관성 + inf 방지)
    safe = np.abs(b) > 1e-10
    out = np.zeros_like(a, dtype=np.float64)
    np.divide(a, b, out=out, where=safe)
    return out


OPS = {
    "mul":   _op_mul,
    "add":   _op_add,
    "sub":   _op_sub,
    "ratio": _op_ratio,
}


# ─── die→unit 집계 (성능 최적화: bincount 기반) ─────
def _make_unit_aggregator(xs):
    """xs[KEY_COL] 순서가 고정인 한 재사용 가능한 집계 함수 생성.

    Returns
    -------
    agg_fn : callable(die_vals: np.ndarray) -> pd.Series
        die 배열을 받아 unit-level mean 을 반환. 인덱스는 unit id.
    unit_ids : np.ndarray
    """
    key_arr = xs[KEY_COL].values
    unit_ids, inv = np.unique(key_arr, return_inverse=True)
    counts = np.bincount(inv).astype(np.float64)

    def agg(die_vals):
        v = np.asarray(die_vals, dtype=np.float64)
        # NaN/Inf 를 0 으로 대체 (상관 계산에 유한값만 기여)
        if not np.isfinite(v).all():
            v = np.where(np.isfinite(v), v, 0.0)
        sums = np.bincount(inv, weights=v)
        means = sums / counts
        return pd.Series(means, index=unit_ids)

    return agg, unit_ids


# ─── 1단계: top-K 선별 ────────────────────────────
def select_top_features_by_importance(
    xs_train, ys_train_unit, feat_cols,
    k=50, n_estimators=200, seed=42,
):
    """LightGBM 으로 빠르게 fit → importance(gain) top-k feat 반환."""
    import lightgbm as lgb

    y_map = ys_train_unit.set_index(KEY_COL)[TARGET_COL]
    y_die = xs_train[KEY_COL].map(y_map).values.astype(float)

    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(xs_train[feat_cols].values, y_die)

    imp = pd.Series(
        model.feature_importances_,
        index=feat_cols, name="importance",
    ).sort_values(ascending=False)
    top = imp.head(k).index.tolist()
    return top, imp


# ─── 2단계: baseline 단일 |corr| ─────────────────
def baseline_single_abs_corr(xs_train, ys_train_unit, feat_cols):
    """단일 feature 의 unit-level max|target corr|."""
    agg_fn, unit_ids = _make_unit_aggregator(xs_train)
    y_unit = (
        ys_train_unit.set_index(KEY_COL)[TARGET_COL]
        .reindex(unit_ids)
    )
    vals = y_unit.values

    max_c = 0.0
    top_col = None
    for c in feat_cols:
        u = agg_fn(xs_train[c].values).values
        # pearson 수식 직접 (nan 방어)
        if np.std(u) < 1e-12:
            continue
        corr = np.corrcoef(u, vals)[0, 1]
        if np.isfinite(corr) and abs(corr) > max_c:
            max_c = float(abs(corr))
            top_col = c
    return max_c, top_col


# ─── 3단계: 페어 생성 + 스코어링 ─────────────────
def generate_and_score_pairs(
    xs_train, ys_train_unit, feat_cols_top,
    ops=("mul", "add", "sub", "ratio"),
    metric="pearson",
    verbose=True,
):
    """모든 페어 × ops 조합의 unit-level |target corr| 계산.

    Returns
    -------
    DataFrame  columns=['col_a','col_b','op','corr','sign']
        corr 내림차순 정렬. sign 은 원 상관 부호(+1/-1).
    """
    agg_fn, unit_ids = _make_unit_aggregator(xs_train)
    y_unit = (
        ys_train_unit.set_index(KEY_COL)[TARGET_COL]
        .reindex(unit_ids)
    )
    y_vals = y_unit.values.astype(np.float64)

    # spearman 사전 준비 (rank 변환 1회)
    if metric == "spearman":
        y_rank = pd.Series(y_vals).rank().values
    feat_idx = {c: i for i, c in enumerate(feat_cols_top)}
    X = xs_train[feat_cols_top].values.astype(np.float64)

    pairs = list(combinations(feat_cols_top, 2))
    total = len(pairs) * len(ops)
    if verbose:
        print(f"  페어 {len(pairs)}쌍 × ops {len(ops)}개 = {total}개 스코어링")

    results = []
    for ca, cb in pairs:
        a = X[:, feat_idx[ca]]
        b = X[:, feat_idx[cb]]
        for op_name in ops:
            op_fn = OPS[op_name]
            derived = op_fn(a, b)
            if not np.isfinite(derived).any():
                continue
            u = agg_fn(derived).values
            if np.std(u) < 1e-12:
                continue

            if metric == "pearson":
                c = np.corrcoef(u, y_vals)[0, 1]
            elif metric == "spearman":
                u_rank = pd.Series(u).rank().values
                c = np.corrcoef(u_rank, y_rank)[0, 1]
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if not np.isfinite(c):
                continue
            results.append({
                "col_a": ca, "col_b": cb, "op": op_name,
                "corr": float(abs(c)), "sign": int(np.sign(c)),
            })

    df = pd.DataFrame(results).sort_values("corr", ascending=False).reset_index(drop=True)
    return df


# ─── 4단계: 채택 ──────────────────────────────────
def select_derived(scores_df, baseline_abs_corr, gain_ratio=1.5, max_keep=200):
    threshold = baseline_abs_corr * gain_ratio
    sel = scores_df[scores_df["corr"] > threshold].copy()
    if max_keep is not None:
        sel = sel.head(max_keep)
    sel = sel.reset_index(drop=True)
    sel.attrs["threshold"] = threshold
    sel.attrs["baseline"] = baseline_abs_corr
    return sel


# ─── 5단계: split 에 적용 ─────────────────────────
def apply_pairs_to_splits(xs_train, xs_val, xs_test, selected_df):
    """선별된 페어 die-level 생성 → 3 split 에 컬럼 추가. 새 컬럼명 반환."""
    new_cols = []
    for _, row in selected_df.iterrows():
        ca = row["col_a"]; cb = row["col_b"]; op_name = row["op"]
        op_fn = OPS[op_name]
        name = f"auto_{op_name}__{ca}__{cb}"
        for xs_ in (xs_train, xs_val, xs_test):
            xs_[name] = op_fn(xs_[ca].values, xs_[cb].values)
        new_cols.append(name)
    return new_cols


# ─── 파이프라인 ──────────────────────────────────
def run_auto_feature_engineering(
    xs_train, xs_val, xs_test, ys_train_unit, feat_cols,
    k=50,
    ops=("mul", "add", "sub", "ratio"),
    metric="pearson",
    gain_ratio=1.5,
    max_keep=200,
    seed=42,
):
    """전체 파이프라인 실행 → (확장된 feat_cols, 선별 DataFrame) 반환."""
    print("[AutoFE] 1/4 LightGBM importance top-K")
    top, imp = select_top_features_by_importance(
        xs_train, ys_train_unit, feat_cols, k=k, seed=seed,
    )
    print(f"  top-{k} 중 상위 5개: {top[:5]}")

    print("[AutoFE] 2/4 baseline 단일 max|corr| 계산")
    baseline, top_single = baseline_single_abs_corr(xs_train, ys_train_unit, top)
    print(f"  baseline = {baseline:.4f}  ({top_single})")

    print(f"[AutoFE] 3/4 페어 × {ops} → {metric}")
    scores = generate_and_score_pairs(
        xs_train, ys_train_unit, top,
        ops=ops, metric=metric,
    )

    print("[AutoFE] 4/4 채택")
    selected = select_derived(scores, baseline, gain_ratio, max_keep)
    thr = selected.attrs["threshold"]
    print(f"  threshold = {baseline:.4f} × {gain_ratio} = {thr:.4f}")
    print(f"  채택 수 = {len(selected)} / {len(scores)} "
          f"(상위 {max_keep}개 제한)")
    if len(selected) > 0:
        print("  상위 5개:")
        print("    " + selected.head().to_string(index=False).replace("\n", "\n    "))

    if len(selected) == 0:
        print("  [경고] 채택된 페어 없음 → 원본 feat_cols 유지")
        return list(feat_cols), selected

    new_cols = apply_pairs_to_splits(xs_train, xs_val, xs_test, selected)
    feat_ext = list(feat_cols) + new_cols
    print(f"[AutoFE] 완료: feat_cols {len(feat_cols)} → {len(feat_ext)} (+{len(new_cols)})")
    return feat_ext, selected