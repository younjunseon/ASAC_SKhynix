"""
Die-level → Unit-level 집계
- 다양한 통계량 지원 (mean, std, min, max, range, median, skew)
- Position별 피벗 지원
"""
import pandas as pd
import numpy as np
from utils.config import KEY_COL, POSITION_COL, TARGET_COL
from utils.data import load_xs, load_ys, get_feat_cols


def aggregate_to_unit(xs, feat_cols=None, agg_funcs=None):
    """
    die-level → unit-level 집계

    Parameters
    ----------
    xs : DataFrame
        die-level 데이터 (split 무관, 전체 또는 일부)
    feat_cols : list, optional
        집계할 feature 컬럼. None이면 자동 추출
    agg_funcs : list of str, optional
        집계 함수 목록. 기본값: ["mean", "std", "min", "max", "range", "median"]
        지원: "mean", "std", "min", "max", "median", "skew", "range"
        EDA Phase 26: median이 max|r| 1위(0.0377), range도 유용

    Returns
    -------
    DataFrame
        unit-level 집계 결과. 컬럼명: {feature}_{agg_func}
    """
    if feat_cols is None:
        feat_cols = get_feat_cols(xs)
    if agg_funcs is None:
        agg_funcs = ["mean", "std", "min", "max", "range", "median"]

    # range는 직접 계산 필요
    builtin_funcs = [f for f in agg_funcs if f != "range"]
    need_range = "range" in agg_funcs

    parts = []

    if builtin_funcs:
        agg_result = xs.groupby(KEY_COL)[feat_cols].agg(builtin_funcs)
        # MultiIndex 컬럼 → flat
        agg_result.columns = [f"{col}_{func}" for col, func in agg_result.columns]
        parts.append(agg_result)

    if need_range:
        g = xs.groupby(KEY_COL)[feat_cols]
        range_df = g.max() - g.min()
        range_df.columns = [f"{col}_range" for col in range_df.columns]
        parts.append(range_df)

    result = pd.concat(parts, axis=1)
    print(f"집계 완료: {len(result):,} units × {result.shape[1]:,} features "
          f"(agg: {agg_funcs})")
    return result


def pivot_by_position(xs, feat_cols=None):
    """
    Position별로 피벗하여 unit-level feature 생성.
    컬럼명: {feature}_pos{position}

    Parameters
    ----------
    xs : DataFrame
    feat_cols : list, optional

    Returns
    -------
    DataFrame
        unit-level, 컬럼: {feature}_pos1, {feature}_pos2, ...
    """
    if feat_cols is None:
        feat_cols = get_feat_cols(xs)

    positions = sorted(xs[POSITION_COL].unique())
    parts = []
    for pos in positions:
        sub = xs[xs[POSITION_COL] == pos].set_index(KEY_COL)[feat_cols]
        sub.columns = [f"{col}_pos{pos}" for col in sub.columns]
        parts.append(sub)

    result = pd.concat(parts, axis=1)
    print(f"Position 피벗 완료: {len(result):,} units × {result.shape[1]:,} features "
          f"(positions: {positions})")
    return result



def merge_with_target(unit_features, split="train"):
    """
    unit-level feature에 target(health) merge

    Parameters
    ----------
    unit_features : DataFrame
        index가 ufs_serial인 unit-level feature
    split : str
        "train", "validation", "test", "all"

    Returns
    -------
    X : DataFrame, y : Series
    """
    ys = load_ys()
    target = ys[split]

    merged = unit_features.merge(target, left_index=True, right_on=KEY_COL, how="inner")
    y = merged[TARGET_COL]
    X = merged.drop(columns=[KEY_COL, TARGET_COL])

    print(f"Merge ({split}): X={X.shape}, y={y.shape}, y_mean={y.mean():.6f}")
    return X, y