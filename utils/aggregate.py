"""
Die-level → Unit-level 변환 도구.

X는 die 단위(1 unit = 4 die)인데 예측·평가는 unit 단위라, die feature를 unit 단위로 합쳐야 한다.
두 가지 방식을 제공한다:
  - aggregate_to_unit: 4 die를 통계량(mean/std/min/max/range/median/skew)으로 요약 → {feat}_{stat} 컬럼
  - pivot_by_position: 4 die를 position 1~4 컬럼으로 옆으로 펼침 → {feat}_pos1..4 컬럼
그리고 그렇게 만든 unit feature에 target(health)을 붙이는 merge_with_target.
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
        (지원 7종 중에서는 median이 target과의 상관 |r|이 가장 크고, range도 die 간 산포 지표로 유용)

    Returns
    -------
    DataFrame
        unit-level 집계 결과. 컬럼명: {feature}_{agg_func}
    """
    if feat_cols is None:
        feat_cols = get_feat_cols(xs)
    if agg_funcs is None:
        agg_funcs = ["mean", "std", "min", "max", "range", "median"]

    # 'range'(=max-min)는 pandas 기본 집계에 없어 직접 계산해야 한다 → 나머지와 분리
    builtin_funcs = [f for f in agg_funcs if f != "range"]
    need_range = "range" in agg_funcs

    parts = []   # 만들어진 unit-level 조각들을 모아 마지막에 옆으로 붙임

    if builtin_funcs:
        # groupby.agg([...])는 컬럼이 MultiIndex (feature, func)로 나옴
        agg_result = xs.groupby(KEY_COL)[feat_cols].agg(builtin_funcs)
        # MultiIndex → "X0_mean", "X0_std", ... 평탄한 단일 컬럼명으로 변환
        agg_result.columns = [f"{col}_{func}" for col, func in agg_result.columns]
        parts.append(agg_result)

    if need_range:
        g = xs.groupby(KEY_COL)[feat_cols]
        range_df = g.max() - g.min()                                  # unit별 (max - min)
        range_df.columns = [f"{col}_range" for col in range_df.columns]
        parts.append(range_df)

    result = pd.concat(parts, axis=1)   # index(ufs_serial) 기준 가로 결합
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

    positions = sorted(xs[POSITION_COL].unique())   # 보통 [1, 2, 3, 4]
    parts = []
    for pos in positions:
        # 해당 position의 die 행만 골라 ufs_serial을 index로 → feature를 "X0_pos1" 식으로 rename
        sub = xs[xs[POSITION_COL] == pos].set_index(KEY_COL)[feat_cols]
        sub.columns = [f"{col}_pos{pos}" for col in sub.columns]
        parts.append(sub)

    # position 4개 조각을 index(ufs_serial) 기준으로 가로 결합 → unit 1행 = 4 die의 값을 옆으로 나열
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
    target = ys[split]   # 해당 split의 [ufs_serial, health] DataFrame

    # feature는 index가 ufs_serial, target은 컬럼이 ufs_serial → 그 둘을 키로 inner join
    # (inner라 양쪽에 다 있는 unit만 남음 — 보통 둘 다 동일 unit 집합)
    merged = unit_features.merge(target, left_index=True, right_on=KEY_COL, how="inner")
    y = merged[TARGET_COL]
    X = merged.drop(columns=[KEY_COL, TARGET_COL])   # 키와 타깃을 빼면 순수 feature 행렬

    print(f"Merge ({split}): X={X.shape}, y={y.shape}, y_mean={y.mean():.6f}")
    return X, y
