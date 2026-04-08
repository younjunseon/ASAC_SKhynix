"""
Die → Unit 집계 (경량 silent 버전)

E2E HPO trial 내부에서 반복 호출되므로 print 없이 동작한다.
"""
import numpy as np
import pandas as pd

from utils.config import KEY_COL, TARGET_COL, POSITION_COL

DEFAULT_AGG_FUNCS = ["mean", "std", "cv", "range", "min", "max", "median"]


def aggregate_die_to_unit(
    pos_data,
    feat_cols,
    clf_proba,
    agg_funcs=None,
    include_position_pivot=True,
):
    """
    Die-level → Unit-level 집계

    Parameters
    ----------
    pos_data : dict
        {position: {"train": df, "val": df, "test": df}}
    feat_cols : list
        die-level 피처 컬럼명
    clf_proba : dict
        {position: {"train_proba": array, "val_proba": array, "test_proba": array}}
    agg_funcs : list, optional
        집계 함수 목록. None이면 DEFAULT_AGG_FUNCS 사용
    include_position_pivot : bool
        True면 position별 원본값을 피벗하여 추가

    Returns
    -------
    unit_data : dict
        {"train": df, "val": df, "test": df}
    unit_feat_cols : list
        생성된 unit-level 피처 컬럼명 (clf_proba_mean 포함)
    """
    if agg_funcs is None:
        agg_funcs = DEFAULT_AGG_FUNCS

    positions = sorted(pos_data.keys())
    unit_data = {}

    for split_name in ["train", "val", "test"]:
        # --- die 데이터 합치기 ---
        die_frames = []
        for pos in positions:
            df = pos_data[pos][split_name].copy()
            df["clf_proba"] = clf_proba[pos][f"{split_name}_proba"]
            die_frames.append(df)

        die_all = pd.concat(die_frames, ignore_index=True)

        # --- WT feature 집계 ---
        grp = die_all.groupby(KEY_COL)[feat_cols]
        agg_parts = []

        for func in agg_funcs:
            if func == "cv":
                cv_df = grp.std() / grp.mean().abs().clip(lower=1.0)
                cv_df.columns = [f"{c}_cv" for c in feat_cols]
                agg_parts.append(cv_df)
            elif func == "range":
                range_df = grp.max() - grp.min()
                range_df.columns = [f"{c}_range" for c in feat_cols]
                agg_parts.append(range_df)
            else:
                grp_df = grp.agg(func)
                grp_df.columns = [f"{c}_{func}" for c in feat_cols]
                agg_parts.append(grp_df)

        unit_features = pd.concat(agg_parts, axis=1)

        # --- Position별 피벗 ---
        if include_position_pivot:
            for pos in positions:
                pos_mask = die_all[POSITION_COL] == pos
                pos_df = die_all.loc[pos_mask].set_index(KEY_COL)[feat_cols]
                pos_df.columns = [f"{c}_pos{pos}" for c in feat_cols]
                unit_features = unit_features.join(pos_df)

        # --- 분류 확률 집계 ---
        proba_mean = die_all.groupby(KEY_COL)["clf_proba"].mean()
        proba_mean.name = "clf_proba_mean"
        unit_features = unit_features.join(proba_mean)

        # --- health (target) ---
        health = die_all.groupby(KEY_COL)[TARGET_COL].first()
        unit_features = unit_features.join(health)

        unit_features = unit_features.reset_index()
        unit_data[split_name] = unit_features

    # --- unit_feat_cols 목록 생성 ---
    meta_cols = {KEY_COL, TARGET_COL}
    unit_feat_cols = [
        c for c in unit_data["train"].columns if c not in meta_cols
    ]

    return unit_data, unit_feat_cols
