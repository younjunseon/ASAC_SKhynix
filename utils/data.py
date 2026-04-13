"""
데이터 로드 및 split 분리
- 한 번 로드 후 캐싱
- feat_cols 자동 추출
"""
import pandas as pd
from utils.config import (
    XS_PATH, YS_TRAIN_PATH, YS_VAL_PATH, YS_TEST_PATH,
    META_COLS, SPLIT_COL, KEY_COL, TARGET_COL,
)

# ─── 캐시 ──────────────────────────────────────────────────
_cache = {}


def load_xs(force=False, downcast=True):
    """
    Xs(die-level) 데이터 로드. 한 번 로드 후 메모리 캐싱

    Parameters
    ----------
    force : bool
        True면 캐시 무시하고 CSV에서 다시 로드
    downcast : bool
        True(기본)면 X* feature 컬럼을 float32로 다운캐스트.
        메모리 사용량을 float64 대비 절반으로 줄인다.
        메타 컬럼(ufs_serial, run_wf_xy, split, position)은 불변.
        주의: 캐시 키가 고정이므로 downcast 값을 도중에 바꾸려면 force=True 필요.

    Returns
    -------
    DataFrame
        (174,980 × 1,091) die-level 데이터
    """
    if "xs" not in _cache or force:
        df = pd.read_csv(XS_PATH)
        if downcast:
            feat_cols = [c for c in df.columns if c.startswith("X")]
            df[feat_cols] = df[feat_cols].astype("float32")
        _cache["xs"] = df
    return _cache["xs"]


def load_ys(force=False):
    """
    Ys(unit-level) 데이터 로드. train/val/test 개별 + 통합 반환

    Parameters
    ----------
    force : bool
        True면 캐시 무시하고 CSV에서 다시 로드

    Returns
    -------
    dict
        {"train": DataFrame, "validation": DataFrame,
         "test": DataFrame, "all": DataFrame(3개 합본, split 컬럼 포함)}
    """
    if "ys" not in _cache or force:
        ys_train = pd.read_csv(YS_TRAIN_PATH)
        ys_val = pd.read_csv(YS_VAL_PATH)
        ys_test = pd.read_csv(YS_TEST_PATH)
        ys_all = pd.concat([
            ys_train.assign(**{SPLIT_COL: "train"}),
            ys_val.assign(**{SPLIT_COL: "validation"}),
            ys_test.assign(**{SPLIT_COL: "test"}),
        ], ignore_index=True)
        _cache["ys"] = {
            "train": ys_train,
            "validation": ys_val,
            "test": ys_test,
            "all": ys_all,
        }
    return _cache["ys"]


def load_all(force=False):
    """
    Xs + Ys 한 번에 로드하고 shape 출력

    Parameters
    ----------
    force : bool
        True면 캐시 무시하고 다시 로드

    Returns
    -------
    xs : DataFrame (die-level)
    ys : dict (load_ys 반환값과 동일)
    """
    xs = load_xs(force)
    ys = load_ys(force)
    print(f"Xs: {xs.shape}  |  "
          f"Ys: train={len(ys['train']):,}, "
          f"val={len(ys['validation']):,}, "
          f"test={len(ys['test']):,}")
    return xs, ys


def get_feat_cols(xs=None):
    """
    Feature 컬럼명 리스트 반환 (X0 ~ X1086)

    Parameters
    ----------
    xs : DataFrame, optional
        None이면 내부에서 load_xs()로 로드

    Returns
    -------
    list of str
        "X"로 시작하는 컬럼명 리스트 (1,087개)
    """
    if xs is None:
        xs = load_xs()
    return [c for c in xs.columns if c.startswith("X")]


def split_xs(xs=None):
    """
    Xs를 split 컬럼 기준으로 train/validation/test로 분리

    Parameters
    ----------
    xs : DataFrame, optional
        None이면 내부에서 load_xs()로 로드

    Returns
    -------
    dict
        {"train": DataFrame, "validation": DataFrame, "test": DataFrame}
        각각 .copy()된 독립 복사본
    """
    if xs is None:
        xs = load_xs()
    return {
        "train": xs[xs[SPLIT_COL] == "train"].copy(),
        "validation": xs[xs[SPLIT_COL] == "validation"].copy(),
        "test": xs[xs[SPLIT_COL] == "test"].copy(),
    }
