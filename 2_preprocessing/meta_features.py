"""
메타 피처 생성 모듈
- run_wf_xy 파싱 → lot, wafer_no, die_x, die_y
- add_meta_features: 노트북 공통 헬퍼 (2026-05-09 결정 반영)

2026-05-09 결정 (meta_features_strategy.md, enet_experiments.md):
- position: 모든 모델 사용 (enet은 OHE 4, 트리/ET는 raw int 1)
- die_x, die_y: 트리/ET만 사용 (continuous), ElasticNet 제외
- lot / wafer_no / lot_wafer: 전 모델 제외 (leak + production 일반화 약화)

이전 EDA 결과 (참고용 보존):
- 로트 간 health 차이 극도로 유의 (p=1.69e-242, Phase 19)
- 로트별 일괄 z-score 정규화는 역효과 (76.9% 악화, Phase 21)
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


def add_meta_features(xs_train, xs_val, xs_test, feat_cols,
                      position_mode, use_die_xy, verbose=True,
                      use_loc_x_ohe=False,
                      loc_x_required=("X1073",),
                      loc_x_optional=("X1059", "X1075", "X1076", "X1077")):
    """노트북 공통 메타피처 추가 헬퍼 (2026-05-09 결정 반영).

    적용 범위:
    - position: 모든 모델 (enet은 OHE 4, 트리/ET는 raw int 1)
    - die_x, die_y: 트리/ET True, ElasticNet False
    - loc_x OHE (X1073 등 위치기반 X): ElasticNet True, 트리/ET False
    - lot / wafer_no / lot_wafer: 전 모델 제외 (인자 없음)

    Parameters
    ----------
    xs_train, xs_val, xs_test : DataFrame or None
        in-place 수정 (컬럼 추가). None인 split은 스킵 (enet objective는 train만 처리).
    feat_cols : list[str]
        기존 feature 컬럼 리스트. 본 함수는 이 리스트를 수정하지 않고 새 리스트 반환.
    position_mode : 'ohe' | 'raw'
        - 'ohe': pos_1~pos_4 4컬럼 OHE → ElasticNet 권장
        - 'raw': position 컬럼을 그대로 numeric feature로 추가 → 트리/ET 권장
    use_die_xy : bool
        True면 die_x, die_y continuous 추가. ElasticNet은 False, 트리/ET는 True.
    verbose : bool
        True(기본)면 추가된 컬럼 요약 print.
    use_loc_x_ohe : bool
        True면 위치기반 X (X1073 등)를 OHE로 변환 (단계 2a/2b, ElasticNet 전용).
        train 카테고리 기준 dummy 생성, val/test 신규 카테고리는 모든 dummy=0 가드.
        원본 컬럼은 feat_cols에서 제거 (ordinal+OHE 동시 유지 시 multicollinearity).
    loc_x_required : tuple[str, ...]
        단계 2a 필수 OHE 컬럼 (기본: X1073 4분면 sector).
    loc_x_optional : tuple[str, ...]
        단계 2b A/B OHE 컬럼 (기본: X1059/X1075/X1076/X1077 좌표/strip).
        loc_x_required와 동일하게 처리됨. ordinal 유지하려면 빈 튜플 전달.

    Returns
    -------
    feat_cols : list[str]
        메타피처 컬럼이 추가된 새 리스트.

    Notes
    -----
    - position 컬럼은 xs DataFrame에 이미 존재 (1~4 정수). 'raw'면 그대로 등록,
      'ohe'면 dummy 4개 생성 후 등록 (원본 'position'은 등록 안 함).
    - die_x/die_y는 parse_run_wf_xy() 재사용 (lot/wafer_no도 컬럼은 추가되지만
      feat_cols에는 등록 안 해 모델 입력에서 제외).
    - loc_x_required/optional 컬럼이 feat_cols에 없으면 (cleaning에서 제거됐거나
      EXCLUDE_COLS에 있는 경우) 해당 컬럼 스킵. 에러 안 발생.
    """
    feat_cols = list(feat_cols)  # 원본 보호
    splits = [df for df in (xs_train, xs_val, xs_test) if df is not None]
    if not splits:
        raise ValueError("xs_train/xs_val/xs_test 중 하나 이상 DataFrame이 필요")

    # ── 1. position ──
    if position_mode == 'ohe':
        added_pos = []
        for p in (1, 2, 3, 4):
            col = f'pos_{p}'
            for split_df in splits:
                split_df[col] = (split_df['position'] == p).astype(np.int8)
            added_pos.append(col)
        feat_cols += added_pos
    elif position_mode == 'raw':
        if 'position' not in feat_cols:
            feat_cols.append('position')
        added_pos = ['position']
    else:
        raise ValueError(
            f"position_mode must be 'ohe' or 'raw', got {position_mode!r}"
        )

    # ── 2. die_x, die_y (트리/ET 전용) ──
    if use_die_xy:
        for split_df in splits:
            parse_run_wf_xy(split_df, prefix='', inplace=True, verbose=False)
        added_diexy = ['die_x', 'die_y']
        feat_cols += added_diexy
    else:
        added_diexy = []

    # ── 3. loc_x OHE (단계 2a/2b, ElasticNet 전용) ──
    # 위치기반 X 컬럼을 OHE로 변환. monotonic 가정이 부적절한 sector/strip ID.
    # train 카테고리 기준 dummy 생성, val/test 신규 카테고리는 모든 dummy=0 가드.
    added_loc, removed_loc = [], []
    if use_loc_x_ohe:
        if xs_train is None:
            raise ValueError(
                "use_loc_x_ohe=True 시 xs_train 필수 (train 카테고리 기준 가드)"
            )
        loc_cols = list(loc_x_required) + list(loc_x_optional)
        for col in loc_cols:
            if col not in feat_cols:
                continue  # cleaning/EXCLUDE_COLS에서 이미 제거된 경우 스킵
            cats = sorted(xs_train[col].dropna().unique().tolist())
            cat_cols = [f"{col}_eq{int(cat)}" for cat in cats]
            for split_df in splits:
                for cat, cat_col in zip(cats, cat_cols):
                    split_df[cat_col] = (split_df[col] == cat).astype(np.int8)
            feat_cols = [c for c in feat_cols if c != col] + cat_cols
            added_loc += cat_cols
            removed_loc.append(col)

    if verbose:
        msg = (f"[add_meta_features] position_mode={position_mode!r}, "
               f"use_die_xy={use_die_xy}, use_loc_x_ohe={use_loc_x_ohe} → "
               f"position={added_pos}, die_xy={added_diexy}")
        if use_loc_x_ohe:
            msg += f", loc_x: -{removed_loc} +{len(added_loc)} dummy"
        msg += f" (feat_cols: {len(feat_cols)})"
        print(msg)
    return feat_cols



