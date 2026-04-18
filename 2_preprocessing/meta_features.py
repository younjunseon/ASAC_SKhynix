"""
메타 피처 생성 모듈
- run_wf_xy 파싱 → lot, wafer_no, die_x, die_y
- 로트별 WT 집계 통계 피처 (lot mean, lot std) → 로트 품질 지표
- 웨이퍼 패턴 분류 → One-Hot 메타 피처
- die 좌표 집계 피처

EDA 결과 기반:
- 로트 간 health 차이 극도로 유의 (p=1.69e-242, Phase 19)
- 로트별 일괄 z-score 정규화는 역효과 (76.9% 악화, Phase 21)
  → 집계 통계 피처만 생성
- 웨이퍼 패턴별 health 유의 차이 (Random 가장 심각, Phase 18-1)
- radial_dist/is_edge 단독 예측력 없음 (r=0.006, Phase 23) → 제외
- NNR 공간 잔차 비효과적 (0/30 우위, Phase 24) → 제외
"""
import pandas as pd
import numpy as np


def parse_run_wf_xy(xs, prefix="", inplace=False, verbose=True):
    """
    run_wf_xy 컬럼을 파싱하여 lot, wafer_no, die_x, die_y 생성

    run_wf_xy 형식: '{작업번호}_{웨이퍼번호}_{X좌표}_{Y좌표}'

    Parameters
    ----------
    xs : DataFrame (die-level, run_wf_xy 컬럼 필요)
    prefix : str
        생성 컬럼의 접두사. 기본 "" → lot, wafer_no, die_x, die_y.
        예: "_" → _lot, _wafer_no, _die_x, _die_y (임시 컬럼용)
    inplace : bool
        True면 xs에 직접 컬럼 추가 후 xs 반환 (copy 비용 절약).
        False(기본)면 copy에 추가 후 반환 — 하위 호환
    verbose : bool
        True(기본)면 요약 print. 내부 호출 시 False로 억제 가능

    Returns
    -------
    xs : DataFrame (prefix+lot/wafer_no/die_x/die_y 컬럼 추가)
    """
    from utils.config import DIE_KEY_COL

    if not inplace:
        xs = xs.copy()

    split = xs[DIE_KEY_COL].str.split("_", expand=True)
    lot_c = f"{prefix}lot"
    wf_c = f"{prefix}wafer_no"
    dx_c = f"{prefix}die_x"
    dy_c = f"{prefix}die_y"
    xs[lot_c] = split[0]
    xs[wf_c] = split[1]
    xs[dx_c] = split[2].astype(int)
    xs[dy_c] = split[3].astype(int)

    if verbose:
        print(f"[run_wf_xy 파싱] lot: {xs[lot_c].nunique()}개, "
              f"wafer: {xs[wf_c].nunique()}개, "
              f"die_x: {xs[dx_c].min()}~{xs[dx_c].max()}, "
              f"die_y: {xs[dy_c].min()}~{xs[dy_c].max()}")
    return xs


def create_lot_stats_features(xs_train, xs_val, xs_test, feat_cols, agg_funcs=None):
    """
    로트별 WT feature 집계 통계를 메타 피처로 생성 (data leakage 방지)

    EDA Phase 19: 로트 간 불량률 6.9%~43.4% (6배 차이), p=1.69e-242
    → 로트 품질 지표가 강력한 예측 신호

    주의: train의 로트 통계만 사용. val/test에 train에 없는 로트가 있으면
    전체 train 평균으로 대체.

    Parameters
    ----------
    xs_train, xs_val, xs_test : DataFrame
        parse_run_wf_xy()로 lot 컬럼이 추가된 상태
    feat_cols : list
        WT feature 컬럼 (lot 통계를 계산할 대상)
    agg_funcs : list of str
        로트별 집계 함수. 기본: ["mean", "std"]

    Returns
    -------
    xs_train, xs_val, xs_test : DataFrame (lot_stat 컬럼 추가)
    lot_stat_cols : list (추가된 컬럼명)
    """
    if agg_funcs is None:
        agg_funcs = ["mean", "std"]

    # train 기준 로트별 통계 계산
    lot_stats = xs_train.groupby("lot")[feat_cols].agg(agg_funcs)
    lot_stats.columns = [f"lot_{func}_{col}" for col, func in lot_stats.columns]
    lot_stat_cols = lot_stats.columns.tolist()

    # 입력 feat_cols와 dtype 일치 (float32 파이프라인에서 merge 시 upcast 방지)
    # X1086 파생 컬럼 제외: 날짜값(8자리 정수)은 float32 정밀도 부족
    try:
        in_dtype = xs_train[feat_cols].dtypes.iloc[0]
        if in_dtype == np.float32:
            safe_cols = [c for c in lot_stats.columns if "X1086" not in c]
            if safe_cols:
                lot_stats[safe_cols] = lot_stats[safe_cols].astype('float32')
    except (AttributeError, IndexError):
        pass

    # 전체 train 평균 (unknown lot fallback)
    global_means = lot_stats.mean()

    def _merge_lot_stats(xs_split, lot_stats_df, fallback):
        """lot 통계를 die-level에 merge"""
        merged = xs_split.merge(lot_stats_df, left_on="lot", right_index=True, how="left")
        # train에 없는 로트 → 전체 평균으로 대체
        for col in lot_stat_cols:
            if merged[col].isnull().any():
                merged[col] = merged[col].fillna(fallback[col])
        return merged

    xs_train = _merge_lot_stats(xs_train, lot_stats, global_means)
    xs_val = _merge_lot_stats(xs_val, lot_stats, global_means)
    xs_test = _merge_lot_stats(xs_test, lot_stats, global_means)

    print(f"[로트 메타 피처] {len(lot_stat_cols)}개 컬럼 추가 "
          f"(feat × agg = {len(feat_cols)} × {len(agg_funcs)})")
    print(f"  train lot: {xs_train['lot'].nunique()}개, "
          f"val lot: {xs_val['lot'].nunique()}개, "
          f"test lot: {xs_test['lot'].nunique()}개")

    return xs_train, xs_val, xs_test, lot_stat_cols


def create_wafer_pattern_features(xs_train, xs_val, xs_test, ys_train):
    """
    웨이퍼 패턴(Center/Edge/Random 등)을 One-Hot 메타 피처로 추가

    EDA Phase 18-1: Random(43.5%) > Edge(35%) > Center(14.1%)
    패턴별 health 유의 차이 → 메타 피처 가치 있음

    주의: 패턴 분류에 target(health)을 사용하므로 train만으로 분류하고,
    val/test는 wafer_id 매칭으로 패턴을 할당.

    Parameters
    ----------
    xs_train, xs_val, xs_test : DataFrame
        parse_run_wf_xy()로 lot, wafer_no, die_x, die_y 추가된 상태
    ys_train : DataFrame
        train target (ufs_serial, health)

    Returns
    -------
    xs_train, xs_val, xs_test : DataFrame (패턴 One-Hot 컬럼 추가)
    pattern_cols : list (추가된 컬럼명)
    """
    from utils.config import KEY_COL, TARGET_COL

    # wafer_id 생성 (lot + wafer_no)
    for df in [xs_train, xs_val, xs_test]:
        df["wafer_id"] = df["lot"] + "_" + df["wafer_no"]

    # train에서 패턴 분류 (target 사용)
    merged = xs_train.merge(ys_train[[KEY_COL, TARGET_COL]], on=KEY_COL, how="left")

    wafer_patterns = {}
    for wid, wf in merged.groupby("wafer_id"):
        n_total = len(wf)
        n_defect = (wf[TARGET_COL] > 0).sum()
        defect_rate = n_defect / max(n_total, 1)

        if defect_rate < 0.05 or n_defect < 5:
            pattern = "None"
        else:
            # 간소화된 분류: 중심/가장자리/랜덤
            cx = (wf["die_x"].max() + wf["die_x"].min()) / 2
            cy = (wf["die_y"].max() + wf["die_y"].min()) / 2
            Rx = max((wf["die_x"].max() - wf["die_x"].min()) / 2, 1)
            Ry = max((wf["die_y"].max() - wf["die_y"].min()) / 2, 1)

            defect_dies = wf[wf[TARGET_COL] > 0]
            if len(defect_dies) == 0:
                pattern = "None"
            else:
                # 정규화 거리 계산
                norm_r = np.sqrt(
                    ((defect_dies["die_x"] - cx) / Rx) ** 2
                    + ((defect_dies["die_y"] - cy) / Ry) ** 2
                )
                mean_r = norm_r.mean()

                if mean_r < 0.4:
                    pattern = "Center"
                elif mean_r > 0.7:
                    pattern = "Edge"
                else:
                    pattern = "Random"

        wafer_patterns[wid] = pattern

    pattern_df = pd.Series(wafer_patterns, name="wafer_pattern")

    # 분포 출력
    pattern_counts = pattern_df.value_counts()
    print(f"[웨이퍼 패턴] train 웨이퍼 {len(pattern_df)}장 분류:")
    for p, cnt in pattern_counts.items():
        print(f"  {p}: {cnt}장 ({cnt/len(pattern_df)*100:.1f}%)")

    # One-Hot 인코딩하여 die-level에 merge
    pattern_dummies = pd.get_dummies(pattern_df, prefix="wf_pattern").astype(int)

    pattern_cols = pattern_dummies.columns.tolist()

    def _merge_pattern(xs_split):
        merged = xs_split.merge(pattern_dummies, left_on="wafer_id",
                                right_index=True, how="left")
        # train에 없는 wafer → 0으로 채움
        for col in pattern_cols:
            if col not in merged.columns:
                merged[col] = 0
            merged[col] = merged[col].fillna(0).astype(int)
        return merged

    xs_train = _merge_pattern(xs_train)
    xs_val = _merge_pattern(xs_val)
    xs_test = _merge_pattern(xs_test)

    print(f"  One-Hot 컬럼: {pattern_cols}")

    return xs_train, xs_val, xs_test, pattern_cols


def create_die_coord_features(xs_train, xs_val, xs_test):
    """
    die 좌표(die_x, die_y)를 피처로 추가

    EDA Phase 23: radial_dist 단독 예측력 없음(r=0.006)이지만,
    die_x, die_y 자체는 집계 시 unit 내 die 공간 분포를 반영할 수 있음.
    집계(mean, std, range) 후 unit의 위치 특성이 됨.

    Parameters
    ----------
    xs_train, xs_val, xs_test : DataFrame
        parse_run_wf_xy()로 die_x, die_y 추가된 상태

    Returns
    -------
    coord_cols : list (["die_x", "die_y"])
    """
    coord_cols = ["die_x", "die_y"]

    # 이미 파싱 단계에서 추가됨 → 존재 확인만
    for df_name, df in [("train", xs_train), ("val", xs_val), ("test", xs_test)]:
        for col in coord_cols:
            assert col in df.columns, f"{df_name}에 {col} 컬럼 없음"

    print(f"[die 좌표 피처] {coord_cols} 확인 완료")
    return coord_cols


def run_meta_features(xs_train, xs_val, xs_test, feat_cols, ys_train,
                      lot_stats=True, lot_agg_funcs=None,
                      wafer_pattern=True,
                      die_coords=True):
    """
    메타 피처 생성 파이프라인 전체 실행

    실행 순서:
    1. run_wf_xy 파싱 (lot, wafer_no, die_x, die_y)
    2. 로트별 WT 집계 통계 피처
    3. 웨이퍼 패턴 One-Hot 피처
    4. die 좌표 피처

    Parameters
    ----------
    xs_train, xs_val, xs_test : DataFrame (die-level, 클리닝 완료)
    feat_cols : list (WT feature 컬럼)
    ys_train : DataFrame (train target)
    lot_stats : bool
        로트 집계 통계 피처 생성 여부
    lot_agg_funcs : list of str
        로트 집계 함수. 기본: ["mean", "std"]
    wafer_pattern : bool
        웨이퍼 패턴 메타 피처 생성 여부
    die_coords : bool
        die 좌표 피처 추가 여부

    Returns
    -------
    xs_train, xs_val, xs_test : DataFrame (메타 피처 추가)
    meta_cols : list (추가된 메타 피처 컬럼명 전체)
    """
    print("=" * 60)
    print("메타 피처 생성 시작")
    print("=" * 60)

    meta_cols = []

    # 1. run_wf_xy 파싱
    xs_train = parse_run_wf_xy(xs_train)
    xs_val = parse_run_wf_xy(xs_val)
    xs_test = parse_run_wf_xy(xs_test)

    # 2. 로트별 WT 집계 통계
    if lot_stats:
        xs_train, xs_val, xs_test, lot_cols = create_lot_stats_features(
            xs_train, xs_val, xs_test, feat_cols, agg_funcs=lot_agg_funcs
        )
        meta_cols.extend(lot_cols)
        print()

    # 3. 웨이퍼 패턴
    if wafer_pattern:
        xs_train, xs_val, xs_test, pattern_cols = create_wafer_pattern_features(
            xs_train, xs_val, xs_test, ys_train
        )
        meta_cols.extend(pattern_cols)
        print()

    # 4. die 좌표
    if die_coords:
        coord_cols = create_die_coord_features(xs_train, xs_val, xs_test)
        meta_cols.extend(coord_cols)
        print()

    # 임시 컬럼 정리 (lot, wafer_no, wafer_id는 집계에 불필요)
    temp_cols = ["lot", "wafer_no", "wafer_id"]
    for df in [xs_train, xs_val, xs_test]:
        for col in temp_cols:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    print(f"{'=' * 60}")
    print(f"메타 피처 생성 완료: {len(meta_cols)}개 컬럼 추가")
    print(f"  train: {xs_train.shape}")
    print(f"  val:   {xs_val.shape}")
    print(f"  test:  {xs_test.shape}")
    print("=" * 60)

    return xs_train, xs_val, xs_test, meta_cols