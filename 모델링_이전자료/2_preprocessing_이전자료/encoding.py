"""
범주형 feature 인코딩 모듈

현재 파이프라인에서 범주형으로 처리 가능한 컬럼:
  - meta_features에서 생성된 wf_pattern_* (이미 원핫 처리됨 → 재처리 불필요)
  - 이산형 X feature (고유값 수 적은 것, EDA Phase 4 기준 391개)
  - 필요 시 lot_id, wafer_no 등 ID성 컬럼

Data leakage 방지 원칙:
  - 모든 인코딩 기준(카테고리 목록, target mean 등)은 train에서만 계산
  - val/test는 train 기준으로 변환, 미등장 카테고리는 0 또는 전체 평균 처리

공개 함수:
    onehot_encode     — 원핫 인코딩 (train 기준 카테고리 고정)
    target_encode     — Target 인코딩 (train mean, smoothing 적용)
    frequency_encode  — 빈도 인코딩 (train 등장 빈도 비율)
    run_encoding      — 파이프라인 통합 실행
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


# ──────────────────────────────────────────────────────────────────────────────
# 원핫 인코딩
# ──────────────────────────────────────────────────────────────────────────────

def onehot_encode(xs_train, xs_val, xs_test, cat_cols,
                  max_cardinality=50, drop_first=False):
    """
    원핫 인코딩 (train 기준 카테고리 고정, data leakage 방지)

    train에서 카테고리 목록을 고정한 뒤 val/test에 동일하게 적용.
    train에 없는 카테고리는 전부 0으로 처리.
    고유값 수가 max_cardinality 초과하는 컬럼은 건너뜀.

    Parameters
    ----------
    xs_train, xs_val, xs_test : DataFrame
    cat_cols : list
        인코딩 대상 컬럼명 리스트
    max_cardinality : int
        이 값 초과하는 고유값 수의 컬럼은 스킵. 기본 50
    drop_first : bool
        True면 더미 변수 함정(dummy trap) 방지용 첫 번째 카테고리 제거

    Returns
    -------
    xs_train, xs_val, xs_test : DataFrame
        원핫 컬럼이 추가된 DataFrame (원본 cat_cols는 제거됨)
    ohe_cols : list
        새로 추가된 원핫 컬럼명 리스트
    skipped : list
        max_cardinality 초과로 스킵된 컬럼명 리스트
    """
    xs_train = xs_train.copy()
    xs_val   = xs_val.copy()
    xs_test  = xs_test.copy()

    ohe_cols = []
    skipped  = []
    to_drop  = []

    for col in cat_cols:
        if col not in xs_train.columns:
            continue

        n_unique = xs_train[col].nunique()
        if n_unique > max_cardinality:
            skipped.append(col)
            print(f"  [스킵] {col}: 고유값 {n_unique}개 > max_cardinality({max_cardinality})")
            continue

        # train 기준 카테고리 목록 고정
        categories = sorted(xs_train[col].dropna().unique().tolist())
        if drop_first:
            categories = categories[1:]

        for cat in categories:
            new_col = f"{col}__{cat}"
            xs_train[new_col] = (xs_train[col] == cat).astype(int)
            xs_val[new_col]   = (xs_val[col]   == cat).astype(int)
            xs_test[new_col]  = (xs_test[col]  == cat).astype(int)
            ohe_cols.append(new_col)

        to_drop.append(col)

    # 원본 범주형 컬럼 제거
    xs_train.drop(columns=[c for c in to_drop if c in xs_train.columns], inplace=True)
    xs_val.drop(  columns=[c for c in to_drop if c in xs_val.columns],   inplace=True)
    xs_test.drop( columns=[c for c in to_drop if c in xs_test.columns],  inplace=True)

    print(f"[원핫 인코딩] {len(to_drop)}개 컬럼 → {len(ohe_cols)}개 더미 컬럼 생성")
    if skipped:
        print(f"  스킵: {skipped}")

    return xs_train, xs_val, xs_test, ohe_cols, skipped


# ──────────────────────────────────────────────────────────────────────────────
# Target 인코딩
# ──────────────────────────────────────────────────────────────────────────────

def target_encode(xs_train, xs_val, xs_test, ys_train,
                  cat_cols, target_col, key_col,
                  smoothing=10, suffix="_te", n_splits=5):
    """
    Target 인코딩 (smoothing 적용, data leakage 방지)

    각 카테고리의 target 평균을 feature로 사용.
    smoothing으로 샘플 수 적은 카테고리의 과적합 방지.

    Train 누수 방지:
        - train은 GroupKFold(unit 단위) OOF 인코딩
          → 각 fold는 나머지 fold의 통계로만 인코딩되어
            자신의 y가 자기 인코딩에 섞이지 않음
        - die-level 행이 unit 단위로 그룹화되어 같은 unit의 4개 die가
          서로 다른 fold로 흩어지지 않음 (die→unit 누수 차단)
    Val/test:
        - train 전체 통계(smoothed)로 변환
        - 미등장 카테고리 → train global_mean

    공식: encoded = (n * cat_mean + smoothing * global_mean) / (n + smoothing)

    Parameters
    ----------
    xs_train, xs_val, xs_test : DataFrame
    ys_train : DataFrame
        target 데이터 (key_col, target_col 포함)
    cat_cols : list
        인코딩 대상 컬럼명 리스트
    target_col : str
        타깃 컬럼명
    key_col : str
        unit 식별자 컬럼 (xs와 ys 병합 키, GroupKFold group으로도 사용)
    smoothing : float
        smoothing 강도. 클수록 global_mean에 가까워짐. 기본 10
    suffix : str
        생성 컬럼명 접미사. 기본 '_te'
    n_splits : int
        train OOF 인코딩 fold 수. 기본 5

    Returns
    -------
    xs_train, xs_val, xs_test : DataFrame
        target 인코딩 컬럼이 추가된 DataFrame
    te_cols : list
        새로 추가된 컬럼명 리스트
    te_maps : dict
        {컬럼명: Series(카테고리 → 인코딩 값)} — val/test에 적용된
        full-train 통계. 재사용 가능 (OOF 통계는 fold마다 달라 저장 안 함)
    """
    xs_train = xs_train.copy()
    xs_val   = xs_val.copy()
    xs_test  = xs_test.copy()

    # xs_train에 target 병합 (unit 단위 평균)
    ys_unit = ys_train.groupby(key_col)[target_col].mean()
    train_with_y = xs_train.join(ys_unit, on=key_col, how="left")

    global_mean = train_with_y[target_col].mean()
    groups = train_with_y[key_col].values
    gkf = GroupKFold(n_splits=n_splits)
    fold_indices = list(gkf.split(train_with_y, groups=groups))

    te_cols = []
    te_maps = {}

    for col in cat_cols:
        if col not in xs_train.columns:
            continue

        new_col = f"{col}{suffix}"

        # ── (1) train: GroupKFold OOF 인코딩 ──
        oof = np.full(len(xs_train), np.nan, dtype=float)
        for tr_idx, va_idx in fold_indices:
            fold_train = train_with_y.iloc[tr_idx]
            fold_gmean = fold_train[target_col].mean()
            stats_f = fold_train.groupby(col)[target_col].agg(["mean", "count"])
            smoothed_f = (
                (stats_f["count"] * stats_f["mean"] + smoothing * fold_gmean)
                / (stats_f["count"] + smoothing)
            )
            oof[va_idx] = (
                train_with_y.iloc[va_idx][col]
                .map(smoothed_f)
                .fillna(fold_gmean)
                .values
            )
        xs_train[new_col] = oof

        # ── (2) val/test: full-train 통계 ──
        stats = train_with_y.groupby(col)[target_col].agg(["mean", "count"])
        smoothed = (
            (stats["count"] * stats["mean"] + smoothing * global_mean)
            / (stats["count"] + smoothing)
        )
        te_maps[col] = smoothed

        xs_val[new_col]  = xs_val[col].map(smoothed).fillna(global_mean)
        xs_test[new_col] = xs_test[col].map(smoothed).fillna(global_mean)
        te_cols.append(new_col)

    print(f"[Target 인코딩] {len(te_cols)}개 컬럼 생성 "
          f"(smoothing={smoothing}, OOF GroupKFold n_splits={n_splits})")
    return xs_train, xs_val, xs_test, te_cols, te_maps


# ──────────────────────────────────────────────────────────────────────────────
# 빈도 인코딩
# ──────────────────────────────────────────────────────────────────────────────

def frequency_encode(xs_train, xs_val, xs_test, cat_cols, suffix="_freq"):
    """
    빈도 인코딩 (train 등장 비율로 대체)

    카테고리 → train에서의 등장 비율(0~1)로 변환.
    val/test의 미등장 카테고리 → 0으로 대체.
    원핫보다 차원 증가 없이 빈도 정보 보존.

    Parameters
    ----------
    xs_train, xs_val, xs_test : DataFrame
    cat_cols : list
        인코딩 대상 컬럼명 리스트
    suffix : str
        생성 컬럼명 접미사. 기본 '_freq'

    Returns
    -------
    xs_train, xs_val, xs_test : DataFrame
    freq_cols : list
        새로 추가된 컬럼명 리스트
    freq_maps : dict
        {컬럼명: Series(카테고리 → 비율)}
    """
    xs_train = xs_train.copy()
    xs_val   = xs_val.copy()
    xs_test  = xs_test.copy()

    freq_cols = []
    freq_maps = {}

    for col in cat_cols:
        if col not in xs_train.columns:
            continue

        new_col = f"{col}{suffix}"
        freq = xs_train[col].value_counts(normalize=True)
        freq_maps[col] = freq

        xs_train[new_col] = xs_train[col].map(freq).fillna(0.0)
        xs_val[new_col]   = xs_val[col].map(freq).fillna(0.0)
        xs_test[new_col]  = xs_test[col].map(freq).fillna(0.0)
        freq_cols.append(new_col)

    print(f"[빈도 인코딩] {len(freq_cols)}개 컬럼 생성")
    return xs_train, xs_val, xs_test, freq_cols, freq_maps


# ──────────────────────────────────────────────────────────────────────────────
# 파이프라인 통합 실행
# ──────────────────────────────────────────────────────────────────────────────

def run_encoding(xs_train, xs_val, xs_test, feat_cols,
                 method="onehot",
                 cat_cols=None,
                 max_cardinality=50,
                 discrete_threshold=20,
                 drop_first=False,
                 ys_train=None,
                 target_col=None,
                 key_col=None,
                 smoothing=10):
    """
    인코딩 파이프라인 통합 실행

    cat_cols를 직접 지정하지 않으면 feat_cols 중
    고유값 수 <= discrete_threshold인 컬럼을 자동 탐지한다.
    (EDA Phase 4: 고유값 20 이하 = 이산형 391개 기준)

    wf_pattern_* 컬럼은 이미 meta_features에서 처리됐으므로 자동 제외.

    Parameters
    ----------
    xs_train, xs_val, xs_test : DataFrame
    feat_cols : list
        전체 feature 컬럼명 리스트
    method : str
        'onehot'    — 원핫 인코딩 (기본)
        'target'    — Target 인코딩 (ys_train, target_col, key_col 필수)
        'frequency' — 빈도 인코딩
    cat_cols : list or None
        인코딩 대상 컬럼 직접 지정. None이면 자동 탐지
    max_cardinality : int
        원핫 인코딩 시 고유값 수 상한. 기본 50
    discrete_threshold : int
        자동 탐지 기준 고유값 수. 기본 20
    drop_first : bool
        원핫 인코딩 시 첫 카테고리 제거 여부
    ys_train : DataFrame or None
        target 인코딩 시 필수
    target_col : str or None
        target 인코딩 시 필수
    key_col : str or None
        target 인코딩 시 필수
    smoothing : float
        target 인코딩 smoothing 강도

    Returns
    -------
    xs_train, xs_val, xs_test : DataFrame
    new_feat_cols : list
        인코딩 후 업데이트된 feature 컬럼 리스트
    report : dict
        인코딩 결과 요약
    """
    assert method in ("onehot", "target", "frequency"), \
        f"method='{method}' 지원하지 않음. 'onehot' / 'target' / 'frequency' 중 선택"

    print("=" * 60)
    print(f"인코딩 파이프라인 시작 (method={method})")
    print("=" * 60)

    # cat_cols 자동 탐지
    if cat_cols is None:
        # wf_pattern_* 제외 (이미 처리됨)
        candidates = [c for c in feat_cols
                      if not c.startswith("wf_pattern")
                      and c in xs_train.columns]
        cat_cols = [
            c for c in candidates
            if xs_train[c].nunique() <= discrete_threshold
        ]
        print(f"자동 탐지: 고유값 <= {discrete_threshold}인 컬럼 {len(cat_cols)}개")
    else:
        print(f"직접 지정: {len(cat_cols)}개 컬럼")

    if len(cat_cols) == 0:
        print("인코딩 대상 없음 — 종료")
        return xs_train, xs_val, xs_test, feat_cols, {"method": method, "encoded": []}

    report = {"method": method}

    if method == "onehot":
        xs_train, xs_val, xs_test, new_cols, skipped = onehot_encode(
            xs_train, xs_val, xs_test, cat_cols,
            max_cardinality=max_cardinality,
            drop_first=drop_first,
        )
        report["encoded_cols"]  = new_cols
        report["skipped_cols"]  = skipped
        removed_cols = [c for c in cat_cols if c not in skipped]

    elif method == "target":
        assert ys_train is not None and target_col and key_col, \
            "target 인코딩에는 ys_train, target_col, key_col 필수"
        xs_train, xs_val, xs_test, new_cols, te_maps = target_encode(
            xs_train, xs_val, xs_test, ys_train,
            cat_cols, target_col, key_col, smoothing=smoothing,
        )
        report["encoded_cols"] = new_cols
        report["te_maps"]      = te_maps
        removed_cols = []  # target 인코딩은 원본 컬럼 유지

    elif method == "frequency":
        xs_train, xs_val, xs_test, new_cols, freq_maps = frequency_encode(
            xs_train, xs_val, xs_test, cat_cols,
        )
        report["encoded_cols"]  = new_cols
        report["freq_maps"]     = freq_maps
        removed_cols = []  # 빈도 인코딩은 원본 컬럼 유지

    # feat_cols 업데이트
    new_feat_cols = [c for c in feat_cols if c not in removed_cols] + new_cols

    print(f"\n인코딩 완료")
    print(f"  feature: {len(feat_cols)} → {len(new_feat_cols)}개")
    print(f"  train: {xs_train.shape}")
    print(f"  val  : {xs_val.shape}")
    print(f"  test : {xs_test.shape}")
    print("=" * 60)

    return xs_train, xs_val, xs_test, new_feat_cols, report
