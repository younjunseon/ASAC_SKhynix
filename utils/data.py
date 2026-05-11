"""
데이터 로드 및 split 분리.

- 원본 CSV(X die-level, y unit-level)를 읽어 들이되, 한 번 읽으면 모듈 레벨 캐시(_cache)에 담아
  같은 세션 안에서는 재로드하지 않는다 (CSV가 1GB 넘어 재읽기 비용이 큼).
- 로드 단계에서 "예측 불가능한 행"을 정리한다: feature 전부 NaN인 die, die 4개가 온전치 않은 unit.
  이렇게 정리한 결과에 맞춰 y쪽 unit도 같이 걸러 X/y 세트의 unit 집합을 항상 일치시킨다.
- get_feat_cols / split_xs 같은 보조 함수도 여기 모아 둠.
"""
import pandas as pd
from utils.config import (
    XS_PATH, YS_TRAIN_PATH, YS_VAL_PATH, YS_TEST_PATH,
    META_COLS, SPLIT_COL, KEY_COL, POSITION_COL, TARGET_COL,
)

# 로드 결과 캐시: {"xs": DataFrame, "ys": {...}} — force=True를 줘야 무시하고 다시 읽는다
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
    if "xs" not in _cache or force:        # 캐시에 없거나 강제 재로드면 CSV에서 읽음
        df = pd.read_csv(XS_PATH)

        # (1) feature가 전부 결측인 die 제거 — WT 측정값이 하나도 없으니 예측에 쓸 수 없음
        feat_cols = [c for c in df.columns if c.startswith("X")]
        all_nan = df[feat_cols].isnull().all(axis=1)
        if all_nan.any():
            n = int(all_nan.sum())
            df = df[~all_nan].reset_index(drop=True)
            print(f"[load_xs] all-NaN 행 {n}개 제거 → {len(df):,}행")

        # (2) unit × position 무결성: die가 정확히 4개가 아닌 unit을 통째로 제거
        #     (1)에서 die를 빼다 깨진 unit, 또는 원본부터 불완전한 unit 모두 대상.
        #     집계(die→unit)가 4 die 전제로 동작하므로 여기서 막아 둔다.
        pos_cnt = df.groupby(KEY_COL)[POSITION_COL].nunique()   # unit별 고유 position 수
        incomplete = pos_cnt[pos_cnt != 4].index                # 4가 아닌 unit ID들
        if len(incomplete) > 0:
            n_die_before = len(df)
            # 어느 split에서 몇 unit이 빠지는지 로그로 남김 (보통 0이지만 확인용)
            split_dist = (df[df[KEY_COL].isin(incomplete)]
                          .groupby(SPLIT_COL)[KEY_COL].nunique().to_dict())
            df = df[~df[KEY_COL].isin(incomplete)].reset_index(drop=True)
            print(f"[load_xs] 4 position 미만 unit {len(incomplete)}개 제거 "
                  f"(split별: {split_dist}) → die {n_die_before:,} → {len(df):,}")

        # (3) dtype 다운캐스트로 메모리 절감
        if downcast:
            # X1086만 예외: 8자리 날짜 정수라 float32(유효숫자 ~7자리)면 끝자리가 변조됨.
            # (1)에서 all-NaN 행을 뺀 뒤라 X1086에 NaN이 없으므로 int32로 안전하게 변환 가능.
            safe_cols = [c for c in feat_cols if c != "X1086"]
            df[safe_cols] = df[safe_cols].astype("float32")
            if "X1086" in df.columns:
                df["X1086"] = df["X1086"].astype("int32")

        _cache["xs"] = df          # 정리·다운캐스트 끝난 DataFrame을 캐시에 보관
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
        # split별로 분리 저장돼 있는 y CSV 3개를 읽음 (대회 제출 평가용이라 val/test 라벨은 원래 비공개 — 로컬 데이터엔 실제 값이 들어 있지만 split별 분포가 달라 주의)
        ys_train = pd.read_csv(YS_TRAIN_PATH)
        ys_val = pd.read_csv(YS_VAL_PATH)
        ys_test = pd.read_csv(YS_TEST_PATH)

        # load_xs가 정리한 뒤 살아남은 unit 집합 — 이 집합에 없는 y 행은 버린다 (X/y 일관성)
        xs_units = set(load_xs(force=force)[KEY_COL].unique())

        def _filter_ys(df, name):
            """xs_units에 들어 있는 unit만 남기고, 몇 개를 버렸는지 로그로 남긴다."""
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

        # 3개를 split 컬럼을 붙여 세로로 합친 합본도 같이 제공 (split별 비교/시각화에 편함)
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
    xs = load_xs(force)        # die-level (정리·다운캐스트 적용)
    ys = load_ys(force)        # unit-level dict (xs와 unit 집합 일치)
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
    # 컬럼명을 하드코딩하지 않고 접두사 'X'로 추출 — 전처리로 컬럼이 바뀌어도 그대로 동작
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
    # .copy()로 떼어 줘 호출 측에서 수정해도 캐시 원본(_cache["xs"])이 오염되지 않게 함
    return {
        "train": xs[xs[SPLIT_COL] == "train"].copy(),
        "validation": xs[xs[SPLIT_COL] == "validation"].copy(),
        "test": xs[xs[SPLIT_COL] == "test"].copy(),
    }
