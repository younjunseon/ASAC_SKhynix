"""
zit_pp.py — ZIT 트랙 공용 전처리 로더 (4조합 공통).

zit_only / bag_zit 의 base·eql 4조합이 동일하게 쓰는 die-level 전처리를 한 곳에 모은다.
이전에는 각 비-mmap 워커(01_zit_only_parallel_hpo.py, 02_bag_zit_parallel_hpo.py,
02_bag_zit_eql_parallel_hpo.py)가 load_preprocessed_data를 각자 정의했고, bag 계열은
cleaning.impute_spatial을 2·3단계 median 사본(_impute_spatial_median)으로 in-process
monkeypatch했다.

이 모듈은 그 monkeypatch를 제거하고 **2_preprocessing/cleaning.py 정본**을 그대로 따른다
(모든 전처리 통일 — 고상관 풀스캔 + xy 규제 공간보간, 2·3단계 fallback은 cleaning.py 기본).
PP_FIXED · clip_y_extreme · add_meta_features(position_mode="raw", use_die_xy=True)는
기존 워커와 100% 동일.

소비처: precompute_pp.py(mmap pp.npy 사전계산) + 각 조합의 hpo.py·fit.ipynb.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "setup.py").exists() and (p / "utils").exists():
            return p
    raise RuntimeError(f"Project root not found from {start}")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
for _sub in ["2_preprocessing", "3_modeling"]:
    _p = PROJECT_ROOT / _sub
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from meta_features import add_meta_features  # noqa: E402
from modules import preprocess  # noqa: E402
from utils.config import KEY_COL, TARGET_COL  # noqa: E402
from utils.data import get_feat_cols, load_all, split_xs  # noqa: E402


# 4조합(zit_only / bag_zit × base / eql) 공통 고정 전처리 파라미터.
# (기존 3개 비-mmap 워커의 PP_FIXED가 byte 동일함을 확인하고 단일화)
PP_FIXED = {
    "missing_threshold": 0.30,
    "corr_threshold": 0.90,
    "corr_keep_by": "std",
    "add_indicator": True,
    "indicator_threshold": 0.05,
    "spatial_max_dist": 6.0,
    "post_impute_corr_threshold": 0.96,
    "post_impute_corr_keep_by": "std",
}


def load_preprocessed_data(clip_y_extreme: bool = True):
    """ZIT die-level 전처리 — cleaning.py 정본 그대로 (median monkeypatch 제거).

    기존 워커 load_preprocessed_data와 동일한 흐름:
      load_all → (clip_y_extreme) → preprocess.run(PP_FIXED) → add_meta_features(raw, die_xy)

    Parameters
    ----------
    clip_y_extreme : bool, default True
        train health의 극단값 1.0을 두 번째 최댓값으로 clip (기존 CLIP_Y_EXTREME 동작과 동일).

    Returns
    -------
    x_train : np.ndarray (n_dies, F) float64 — die-level feature 행렬
    uid_train_die : np.ndarray (n_dies,) — die별 ufs_serial (KEY_COL)
    y_train_unit_s : pd.Series — index=ufs_serial, value=health (unit-level)
    y_train_die : np.ndarray (n_dies,) — unit health를 die로 broadcast
    feat_cols_clean : list[str] — 전처리/메타 후 최종 feature 이름
    """
    out = _preprocess_all(clip_y_extreme)
    return (
        out["X_train"],
        out["uid_train_die"],
        out["y_train_unit_s"],
        out["y_train_die"],
        out["feat_cols"],
    )


def _preprocess_all(clip_y_extreme: bool = True) -> dict:
    """공용 PP를 1회 실행해 train/val/test 전체를 반환 (DataFrame + 배열 + uid + y).

    load_preprocessed_data(precompute용 train 배열)와 load_for_fit(fit용 전체)이 공유하는
    단일 PP 본체. cleaning.py 정본 흐름:
      load_all → (clip_y_extreme) → preprocess.run(PP_FIXED) → add_meta_features(raw, die_xy)
    """
    xs, ys = load_all()
    feat_cols = get_feat_cols(xs)
    xs_dict = split_xs(xs)

    ys_input = {k: v.copy() for k, v in ys.items()}
    if clip_y_extreme:
        y_raw = ys_input["train"][TARGET_COL]
        second_max = y_raw[y_raw < y_raw.max()].max()
        n_clipped = int((y_raw >= 1.0).sum())
        ys_input["train"][TARGET_COL] = y_raw.clip(upper=second_max)
        print(f"[CLIP_Y_EXTREME] 1.0 -> {second_max:.6f}, n={n_clipped}")

    pp = preprocess.run(xs, ys_input, feat_cols, xs_dict, params=PP_FIXED)
    xs_train, xs_val, xs_test = pp["xs_train"], pp["xs_val"], pp["xs_test"]
    feat_cols_clean = pp["feat_cols"]

    feat_cols_clean = add_meta_features(
        xs_train, xs_val, xs_test, feat_cols_clean,
        position_mode="raw", use_die_xy=True,
    )

    X_train = xs_train[feat_cols_clean].values.astype(np.float64)
    X_val = xs_val[feat_cols_clean].values.astype(np.float64)
    X_test = xs_test[feat_cols_clean].values.astype(np.float64)

    y_train_unit_s = ys_input["train"].set_index(KEY_COL)[TARGET_COL]
    y_val_unit_s = ys_input["validation"].set_index(KEY_COL)[TARGET_COL]
    y_test_unit_s = ys_input["test"].set_index(KEY_COL)[TARGET_COL]
    y_train_die = xs_train[KEY_COL].map(y_train_unit_s).values.astype(np.float64)

    print(f"[preprocess done] n_features={len(feat_cols_clean)}")
    print(f"  X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}, "
          f"units train={len(y_train_unit_s):,}/val={len(y_val_unit_s):,}/test={len(y_test_unit_s):,}")
    return {
        "xs_train": xs_train, "xs_val": xs_val, "xs_test": xs_test,
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "uid_train_die": xs_train[KEY_COL].values,
        "uid_val_die": xs_val[KEY_COL].values,
        "uid_test_die": xs_test[KEY_COL].values,
        "y_train_unit_s": y_train_unit_s,
        "y_val_unit_s": y_val_unit_s,
        "y_test_unit_s": y_test_unit_s,
        "y_train_die": y_train_die,
        # ys_input: clip 적용된 unit-level y DataFrame dict ({'train','validation','test'}).
        # fit의 후처리(tune_unit_postprocess_train_val)가 ys_input['train']/['validation']를
        # set_index(KEY_COL)[TARGET_COL] 형태로 쓰므로 그대로 노출한다.
        "ys_input": ys_input,
        "feat_cols": feat_cols_clean,
        "clip_y_extreme": clip_y_extreme,
    }


def load_for_fit(clip_y_extreme: bool = True) -> dict:
    """fit.ipynb 용 전체 데이터 로더 (cleaning.py 정본 PP, median 패치 없음).

    HPO 워커는 pp.npy mmap(train 배열)만 쓰지만, fit은 postprocess.aggregate(position 가중)
    때문에 train/val/test **DataFrame**과 val/test 예측용 배열까지 필요하다. 반환 dict 키:
      xs_train/xs_val/xs_test, X_train/X_val/X_test, uid_train_die/uid_val_die/uid_test_die,
      y_train_unit_s/y_val_unit_s/y_test_unit_s, y_train_die, feat_cols, clip_y_extreme
    """
    return _preprocess_all(clip_y_extreme)
