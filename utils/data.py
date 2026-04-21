"""
데이터 로드 및 split 분리
- 한 번 로드 후 캐싱
- feat_cols 자동 추출
"""
import pandas as pd
from utils.config import (
    XS_PATH, YS_TRAIN_PATH, YS_VAL_PATH, YS_TEST_PATH,
    META_COLS, SPLIT_COL, KEY_COL, POSITION_COL, TARGET_COL,
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
        X1086은 날짜값(8자리 정수)이므로 int32로 별도 변환 (float32 정밀도 부족).
        메타 컬럼(ufs_serial, run_wf_xy, split, position)은 불변.
        주의: 캐시 키가 고정이므로 downcast 값을 도중에 바꾸려면 force=True 필요.

    Notes
    -----
    - Feature 1,087개 전부 NaN인 행(all-NaN die)은 로딩 시 자동 제거된다.
    - 4 position이 온전하지 않은 unit은 로딩 시 자동 제거된다 (all-NaN die 제거 이후).

    Returns
    -------
    DataFrame
        die-level 데이터 (all-NaN 행 + 불완전 unit 제거 후)
    """
    if "xs" not in _cache or force:
        df = pd.read_csv(XS_PATH)

        # ── all-NaN 행 제거 (feature 1,087개 전부 결측인 die) ──
        feat_cols = [c for c in df.columns if c.startswith("X")]
        all_nan = df[feat_cols].isnull().all(axis=1)
        if all_nan.any():
            n = int(all_nan.sum())
            df = df[~all_nan].reset_index(drop=True)
            print(f"[load_xs] all-NaN 행 {n}개 제거 → {len(df):,}행")

        # ── Unit × Position 무결성: 4 position 미만 unit 제거 ──
        # all-NaN die 제거로 깨진 unit, 또는 원본부터 불완전한 unit 모두 제거
        pos_cnt = df.groupby(KEY_COL)[POSITION_COL].nunique()
        incomplete = pos_cnt[pos_cnt != 4].index
        if len(incomplete) > 0:
            n_die_before = len(df)
            split_dist = (df[df[KEY_COL].isin(incomplete)]
                          .groupby(SPLIT_COL)[KEY_COL].nunique().to_dict())
            df = df[~df[KEY_COL].isin(incomplete)].reset_index(drop=True)
            print(f"[load_xs] 4 position 미만 unit {len(incomplete)}개 제거 "
                  f"(split별: {split_dist}) → die {n_die_before:,} → {len(df):,}")

        # ── dtype 다운캐스트 ──
        if downcast:
            # X1086: 날짜값(8자리 정수) → float32 유효숫자 7자리라 끝자리 변조
            # all-NaN 행 제거 후 X1086에 NaN 없음 → numpy int32 사용 가능
            safe_cols = [c for c in feat_cols if c != "X1086"]
            df[safe_cols] = df[safe_cols].astype("float32")
            if "X1086" in df.columns:
                df["X1086"] = df["X1086"].astype("int32")

        _cache["xs"] = df
    return _cache["xs"]


def load_ys(force=False):
    """
    Ys(unit-level) 데이터 로드. train/val/test 개별 + 통합 반환

    Parameters
    ----------
    force : bool
        True면 캐시 무시하고 CSV에서 다시 로드

    Notes
    -----
    - load_xs에서 all-NaN die 또는 4 position 미만 unit이 제거되면,
      해당 unit은 ys에서도 자동 제거된다 (xs/ys 세트 일관성 보장).
      제거된 unit은 피처 전원 NaN 또는 die 불완전이라 예측 불가능.

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

        # xs에 존재하는 unit으로 ys 필터링 (xs/ys 세트 일관성)
        xs_units = set(load_xs(force=force)[KEY_COL].unique())

        def _filter_ys(df, name):
            before = len(df)
            out = df[df[KEY_COL].isin(xs_units)].reset_index(drop=True)
            n_removed = before - len(out)
            if n_removed > 0:
                print(f"[load_ys] {name}: xs에 없는 unit {n_removed}개 제거 "
                      f"→ {len(out):,}")
            return out

        ys_train = _filter_ys(ys_train, "train")
        ys_val = _filter_ys(ys_val, "validation")
        ys_test = _filter_ys(ys_test, "test")

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
