"""
메타 피처 생성 모듈.

원본 X0~X1086에는 없는 "공정 맥락" 정보는 run_wf_xy("Lot_Wafer_X_Y" 문자열) 안에 들어 있다.
이 모듈은:
  - parse_run_wf_xy : run_wf_xy를 lot / wafer_no / die_x / die_y 컬럼으로 쪼갬
  - add_meta_features : 노트북 공통 헬퍼 — 모델별로 약속된 메타피처 세트를 추가

모델별 메타피처 정책:
  - position : 모든 모델 사용 (ElasticNet은 OHE 4컬럼, 트리/ET는 raw 정수 1컬럼)
  - die_x, die_y : 트리/ET만 사용 (연속형), ElasticNet 제외
  - 위치기반 X(X1073 등)의 OHE : ElasticNet만 사용 (sector/strip ID라 순서 가정이 부적절), 트리/ET 제외
  - wafer_no : 선택적 (use_wafer_no=True 시 추가). 1~25 정수(카세트 슬롯). train/val/test 모두 1~25로 공유되어 leakage 없음
  - lot / lot_wafer : 전 모델 제외 (lot 단위 target leak 우려)
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
        xs = xs.copy()   # 원본 보호 (inplace=True면 호출 측이 복사 비용을 아끼겠다는 뜻)

    # "Lot_Wafer_X_Y"를 '_'로 4조각 분리 → 0:lot, 1:wafer, 2:die_x, 3:die_y
    split = xs[DIE_KEY_COL].str.split("_", expand=True)
    lot_c = f"{prefix}lot"
    wf_c = f"{prefix}wafer_no"
    dx_c = f"{prefix}die_x"
    dy_c = f"{prefix}die_y"
    xs[lot_c] = split[0]
    xs[wf_c] = split[1]
    xs[dx_c] = split[2].astype(int)   # 좌표는 정수
    xs[dy_c] = split[3].astype(int)

    if verbose:
        print(f"[run_wf_xy 파싱] lot: {xs[lot_c].nunique()}개, "
              f"wafer: {xs[wf_c].nunique()}개, "
              f"die_x: {xs[dx_c].min()}~{xs[dx_c].max()}, "
              f"die_y: {xs[dy_c].min()}~{xs[dy_c].max()}")
    return xs


def add_meta_features(xs_train, xs_val, xs_test, feat_cols,
                      position_mode, use_die_xy=True, use_wafer_no=True, verbose=True,
                      use_loc_x_ohe=False,
                      loc_x_required=("X1073",),
                      loc_x_optional=("X1059", "X1075", "X1076", "X1077")):
    """노트북 공통 메타피처 추가 헬퍼.

    적용 범위:
    - position: 모든 모델 (enet은 OHE 4, 트리/ET는 raw int 1)
    - die_x, die_y: 트리/ET True, ElasticNet False
    - loc_x OHE (X1073 등 위치기반 X): ElasticNet True, 트리/ET False
    - wafer_no: use_wafer_no=True 시 선택적 추가 (1~25 정수, 카세트 슬롯 번호)
    - lot / lot_wafer: 전 모델 제외

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
        True(기본)면 die_x, die_y continuous 추가. ElasticNet은 False로 명시.
    use_wafer_no : bool
        True(기본)면 wafer_no(1~25 정수) 1컬럼 추가.
        use_die_xy=True/False 여부와 무관하게 독립 동작.
    verbose : bool
        True(기본)면 추가된 컬럼 요약 print.
    use_loc_x_ohe : bool
        True면 위치기반 X (X1073 등)를 OHE로 변환 (ElasticNet 전용).
        train 카테고리 기준 dummy 생성, val/test 신규 카테고리는 모든 dummy=0 가드.
        원본 컬럼은 feat_cols에서 제거 (ordinal+OHE 동시 유지 시 multicollinearity).
    loc_x_required : tuple[str, ...]
        필수 OHE 대상 컬럼 (기본: X1073 — 4분면 sector).
    loc_x_optional : tuple[str, ...]
        추가 OHE 대상 컬럼 (기본: X1059/X1075/X1076/X1077 — 좌표/strip).
        loc_x_required와 동일하게 처리됨. ordinal 유지하려면 빈 튜플 전달.

    Returns
    -------
    feat_cols : list[str]
        메타피처 컬럼이 추가된 새 리스트.

    Notes
    -----
    - position 컬럼은 xs DataFrame에 이미 존재 (1~4 정수). 'raw'면 그대로 등록,
      'ohe'면 dummy 4개 생성 후 등록 (원본 'position'은 등록 안 함).
    - die_x/die_y는 parse_run_wf_xy() 재사용. wafer_no는 use_wafer_no=True 시에만
      feat_cols에 등록됨. lot은 항상 feat_cols 제외.
    - loc_x_required/optional 컬럼이 feat_cols에 없으면 (cleaning에서 제거됐거나
      EXCLUDE_COLS에 있는 경우) 해당 컬럼 스킵. 에러 안 발생.
    """
    feat_cols = list(feat_cols)  # 받은 리스트를 직접 수정하지 않게 복사
    splits = [df for df in (xs_train, xs_val, xs_test) if df is not None]   # None 아닌 split만 처리
    if not splits:
        raise ValueError("xs_train/xs_val/xs_test 중 하나 이상 DataFrame이 필요")

    # --- 1) position ---
    if position_mode == 'ohe':
        # pos_1 ~ pos_4 더미 4개 생성 (이 unit 내 die 위치를 one-hot으로)
        added_pos = []
        for p in (1, 2, 3, 4):
            col = f'pos_{p}'
            for split_df in splits:
                split_df[col] = (split_df['position'] == p).astype(np.int8)
            added_pos.append(col)
        feat_cols += added_pos
    elif position_mode == 'raw':
        # 'position' 정수 컬럼을 그대로 feature로 등록 (트리/ET는 OHE 불필요)
        if 'position' not in feat_cols:
            feat_cols.append('position')
        added_pos = ['position']
    else:
        raise ValueError(
            f"position_mode must be 'ohe' or 'raw', got {position_mode!r}"
        )

    # --- 2) die_x, die_y (트리/ET 전용 연속형 좌표) ---
    if use_die_xy:
        for split_df in splits:
            # lot/wafer_no/die_x/die_y 컬럼을 추가하지만, feat_cols엔 die_x/die_y만 등록
            parse_run_wf_xy(split_df, prefix='', inplace=True, verbose=False)
        added_diexy = ['die_x', 'die_y']
        feat_cols += added_diexy
    else:
        added_diexy = []

    # --- 3) wafer_no (선택적, 1~25 정수) ---
    # run_wf_xy 두 번째 토큰을 int32로 파싱. use_die_xy=True 시 parse_run_wf_xy가 이미
    # 문자열 wafer_no를 추가하므로 여기서 정수로 덮어씀. use_die_xy=False 시에도 독립 동작.
    if use_wafer_no:
        from utils.config import DIE_KEY_COL
        for split_df in splits:
            split_df['wafer_no'] = (
                split_df[DIE_KEY_COL].str.split("_", expand=True)[1].astype(np.int32)
            )
        feat_cols.append('wafer_no')
        added_wafer_no = ['wafer_no']
    else:
        added_wafer_no = []

    # --- 4) 위치기반 X 컬럼의 OHE (ElasticNet 전용) ---
    # X1073 같은 컬럼은 값이 sector/strip ID라 "1<2<3"이라는 순서 가정이 부적절 → 더미로 펼침.
    # 더미 카테고리는 train 값 기준으로 정함. val/test에만 나오는 새 카테고리는 모든 더미가 0이 되어 자연스럽게 처리됨.
    added_loc, removed_loc = [], []
    if use_loc_x_ohe:
        if xs_train is None:
            raise ValueError(
                "use_loc_x_ohe=True 시 xs_train 필수 (train 카테고리 기준 가드)"
            )
        loc_cols = list(loc_x_required) + list(loc_x_optional)
        for col in loc_cols:
            if col not in feat_cols:
                continue  # cleaning이나 EXCLUDE_COLS에서 이미 빠진 컬럼이면 그냥 넘어감 (에러 X)
            cats = sorted(xs_train[col].dropna().unique().tolist())   # train에 등장한 카테고리
            cat_cols = [f"{col}_eq{int(cat)}" for cat in cats]
            for split_df in splits:
                for cat, cat_col in zip(cats, cat_cols):
                    split_df[cat_col] = (split_df[col] == cat).astype(np.int8)
            # 원본 ordinal 컬럼은 빼고(더미와 동시 유지 시 다중공선성) 더미들만 feat_cols에 추가
            feat_cols = [c for c in feat_cols if c != col] + cat_cols
            added_loc += cat_cols
            removed_loc.append(col)

    if verbose:
        msg = (f"[add_meta_features] position_mode={position_mode!r}, "
               f"use_die_xy={use_die_xy}, use_wafer_no={use_wafer_no}, use_loc_x_ohe={use_loc_x_ohe} → "
               f"position={added_pos}, die_xy={added_diexy}, wafer_no={added_wafer_no}")
        if use_loc_x_ohe:
            msg += f", loc_x: -{removed_loc} +{len(added_loc)} dummy"
        msg += f" (feat_cols: {len(feat_cols)})"
        print(msg)
    return feat_cols
