"""
데이터 클리닝 모듈.

die-level X에서 "예측에 방해되거나 의미 없는 컬럼"을 단계적으로 걷어내고 결측을 채운다.
순서대로:
  1) 상수 / 극저분산 feature 제거 (std ≤ threshold)
  2) 고결측 feature 제거 (결측률 ≥ threshold)
  3) 완전 중복 컬럼 제거 (값이 똑같은 쌍에서 한쪽)
  4) 고상관 feature 제거 (|r| > threshold 쌍에서 한쪽 — 다중공선성 감소)
  5) 결측치 imputation (median / KNN / 공간보간) + 결측 indicator 옵션
  6) (선택) imputation 후 2차 고상관 제거 — 보간으로 인위적으로 닮아진 컬럼 정리
별도로 binarize_degenerate(): "거의 한 값"인 feature를 0/1 indicator로 압축.

핵심 규칙: 모든 fit(median, 상관, KNN 등)은 train split에서만 하고 val/test엔 transform만 적용한다 (data leakage 방지).
"""
import pandas as pd
import numpy as np


def remove_constant_features(xs, feat_cols, threshold=1e-6):
    """
    분산이 threshold 이하인 feature 제거

    Parameters
    ----------
    xs : DataFrame
    feat_cols : list
    threshold : float
        std가 이 값 이하이면 제거. 0이면 완전 상수만 제거

    Returns
    -------
    keep_cols : list (남은 feature 컬럼)
    removed_cols : list (제거된 feature 컬럼)
    """
    # 컬럼별 표준편차 (ddof=1 = 표본 std). NaN은 무시하고 계산.
    stds = pd.Series(
        np.nanstd(xs[feat_cols].values, axis=0, ddof=1), index=feat_cols
    )
    removed = stds[stds <= threshold].index.tolist()        # std가 거의 0 → 정보량 없음
    keep = [c for c in feat_cols if c not in removed]        # 순서 보존하며 나머지만 남김

    print(f"[상수/극저분산 제거] threshold={threshold}")
    print(f"  제거: {len(removed)}개, 잔여: {len(keep)}개")
    return keep, removed


def remove_high_missing_features(xs, feat_cols, threshold=0.5):
    """
    결측률이 threshold 이상인 feature 제거

    Parameters
    ----------
    xs : DataFrame
    feat_cols : list
    threshold : float
        결측 비율 (0~1). 기본 0.5 = 50%

    Returns
    -------
    keep_cols : list
    removed_cols : list
    """
    # 컬럼별 결측 비율 = isnan 평균
    missing_pct = pd.Series(
        np.isnan(xs[feat_cols].values).mean(axis=0), index=feat_cols
    )
    removed = missing_pct[missing_pct >= threshold].index.tolist()   # 너무 비어 있어 채워도 신뢰 어려움
    keep = [c for c in feat_cols if c not in removed]

    print(f"[고결측 제거] threshold={threshold*100:.0f}%")
    print(f"  제거: {len(removed)}개, 잔여: {len(keep)}개")
    return keep, removed


def remove_duplicate_features(xs, feat_cols, sample_n=5000, seed=42):
    """
    완전 중복 컬럼 쌍에서 하나씩 제거 (샘플 기반, 해시 사전 필터링)

    Parameters
    ----------
    xs : DataFrame
    feat_cols : list
    sample_n : int
    seed : int

    Returns
    -------
    keep_cols : list
    removed_cols : list
    """
    from collections import defaultdict

    # 전체 17만 행을 다 비교하면 느리므로 sample_n행만 뽑아 비교 (중복이면 샘플에서도 같음)
    sample = xs[feat_cols].sample(n=min(sample_n, len(xs)), random_state=seed)
    removed = set()

    # 모든 쌍을 .equals()로 비교하면 O(F^2)라 비쌈 → 먼저 컬럼별 해시를 구해
    # 해시가 같은 컬럼끼리만 실제 비교(같은 버킷 안에서만 .equals())
    col_hashes = {}
    for col in feat_cols:
        h = pd.util.hash_pandas_object(sample[col], index=False)   # 행 순서 기준 해시 시퀀스
        col_hashes[col] = hash(h.values.tobytes())                  # 그 시퀀스를 다시 한 정수로 압축

    hash_groups = defaultdict(list)
    for col in feat_cols:  # feat_cols 순서를 유지해 "먼저 나온 컬럼을 남김"이 되도록
        hash_groups[col_hashes[col]].append(col)

    for h, cols in hash_groups.items():
        if len(cols) < 2:          # 해시가 유일하면 중복 아님
            continue
        # 같은 해시 버킷 안에서 i<j 쌍을 비교, 진짜 같으면 뒤 컬럼(j)을 제거 대상으로
        for i in range(len(cols)):
            if cols[i] in removed:
                continue
            for j in range(i + 1, len(cols)):
                if cols[j] in removed:
                    continue
                if sample[cols[i]].equals(sample[cols[j]]):
                    removed.add(cols[j])

    removed = list(removed)
    keep = [c for c in feat_cols if c not in removed]

    print(f"[중복 컬럼 제거] sample_n={sample_n}")
    print(f"  제거: {len(removed)}개, 잔여: {len(keep)}개")
    return keep, removed


def remove_high_corr_features(
    xs, feat_cols,
    threshold=0.95,
    keep_by="std",
    ys_train=None,
    winsorize_pct=0.0,
    sample_n=10000,
    seed=42,
):
    """
    피처 간 |상관계수| > threshold인 쌍에서 한쪽 제거 (다중공선성 감소)

    (이 데이터엔 |r|>0.95인 피처 쌍이 다수 존재 — X234↔X235↔X236↔X237 같은 거의 동일한 측정값들)

    쌍 중 어느 쪽을 남길지는 `keep_by`로 선택:
    - "std" (기본): std 큰 쪽을 남김.
        winsorize_pct > 0이면 각 컬럼을 [q, 1-q] 분위수로 클립한 뒤
        std를 계산하여 극단값의 영향을 줄임 (예: 0.01 → 1%/99% clip).
    - "target_corr": |corr(feature, target)| 큰 쪽을 남김. ys_train 필수.
        die → unit mean 집계 후 unit-level에서 target과 Pearson 상관 계산 (EDA의 상관 계산과 동일 방식).

    알고리즘: feat_cols 순서대로 i<j 쌍을 순회하고, |r|>threshold인 쌍에서
    score가 큰 쪽을 남김. 한 컬럼이 제거되면 그 컬럼의 나머지 쌍은 스킵.

    Parameters
    ----------
    xs : DataFrame
        die-level DataFrame. KEY_COL(ufs_serial) 컬럼이 있어야 target_corr 동작
    feat_cols : list
    threshold : float
        |r|이 이 값 초과인 쌍에서 한쪽 제거. 기본 0.95
    keep_by : {"std", "target_corr"}
        남길 컬럼을 고르는 기준. 기본 "std"
    ys_train : DataFrame or Series, optional
        keep_by="target_corr"일 때 필수. unit-level target.
        DataFrame이면 KEY_COL, TARGET_COL 컬럼이 있어야 함.
    winsorize_pct : float
        keep_by="std"일 때 std 계산 전 양쪽 분위수 클립 비율.
        0이면 생략. 기본 0.0
    sample_n : int
        std 경로에서 상관/score 계산 시 샘플 수 (속도).
        target_corr 경로에선 unit-level 집계라 전체를 사용.
    seed : int

    Returns
    -------
    keep_cols : list
    removed_cols : list
    """
    # 상관 행렬 계산은 비싸므로 sample_n행만 사용 (다중공선성 판정엔 샘플로 충분)
    sample = xs[feat_cols].sample(n=min(sample_n, len(xs)), random_state=seed)

    # 피처 간 상관 행렬의 절대값 — argwhere로 빠르게 고상관 쌍을 뽑기 위해 numpy 배열로
    corr_matrix = sample.corr().abs()
    corr_arr = corr_matrix.values

    # --- 쌍에서 "남길 쪽"을 정하는 점수 계산 ---
    # 점수를 tuple로 둠 → 파이썬의 사전식(lexicographic) 비교로 "1순위 기준, 동률이면 2순위 기준" 처리
    if keep_by == "target_corr":
        if ys_train is None:
            raise ValueError(
                "keep_by='target_corr'일 때는 ys_train을 반드시 전달해야 합니다"
            )
        from utils.config import KEY_COL, TARGET_COL
        if KEY_COL not in xs.columns:
            raise ValueError(
                f"keep_by='target_corr'는 xs에 '{KEY_COL}' 컬럼이 있어야 합니다"
            )
        # die feature를 unit mean으로 집계 → unit-level target과 |Pearson r| 계산 (EDA와 동일 방식)
        xs_unit_mean = xs.groupby(KEY_COL)[feat_cols].mean()
        merged = xs_unit_mean.merge(
            ys_train, left_index=True, right_on=KEY_COL, how='inner'
        )
        primary = merged[feat_cols].corrwith(merged[TARGET_COL]).abs()   # 1순위: |target 상관|
        tiebreak = sample.std()                                          # 2순위(동률): die-level std
        primary_d = primary.fillna(-np.inf).to_dict()                    # NaN 점수는 항상 지는 -inf로
        tiebreak_d = tiebreak.fillna(-np.inf).to_dict()
        scores_dict = {
            col: (primary_d.get(col, -np.inf), tiebreak_d.get(col, -np.inf))
            for col in feat_cols
        }
        criterion_desc = (
            f"|target 상관| 우선, 동률이면 std "
            f"(unit-level merge n={len(merged):,})"
        )
    elif keep_by == "std":
        if winsorize_pct > 0:
            # 극단값이 std를 부풀리는 걸 막으려고 양 끝 winsorize_pct만큼 분위수로 클립한 뒤 std
            q_low = sample.quantile(winsorize_pct)
            q_high = sample.quantile(1 - winsorize_pct)
            clipped = sample.clip(lower=q_low, upper=q_high, axis=1)
            primary = clipped.std()
            criterion_desc = f"winsorized({winsorize_pct*100:.0f}%) std"
        else:
            primary = sample.std()
            criterion_desc = "std"
        primary_d = primary.fillna(-np.inf).to_dict()
        scores_dict = {
            col: (primary_d.get(col, -np.inf),) for col in feat_cols   # 1-튜플 (기준 1개뿐)
        }
    else:
        raise ValueError(
            f"Unknown keep_by={keep_by!r}. Use 'std' or 'target_corr'"
        )

    # --- 고상관 쌍만 골라 greedy 제거 ---
    removed = set()
    # |r| > threshold인 (i, j) 좌표만 추출
    pairs = np.argwhere(corr_arr > threshold)
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]   # 대칭 행렬이므로 상삼각(i<j)만 사용

    for i, j in pairs:
        col_i = feat_cols[i]
        col_j = feat_cols[j]
        if col_i in removed or col_j in removed:   # 이미 한쪽이 빠진 쌍은 건너뜀
            continue
        # 점수가 크거나 같은 쪽을 남기고 작은 쪽을 제거 (동률이면 i를 남김 → 먼저 나온 컬럼 우대)
        if scores_dict[col_i] >= scores_dict[col_j]:
            removed.add(col_j)
        else:
            removed.add(col_i)

    removed = list(removed)
    keep = [c for c in feat_cols if c not in removed]

    print(f"[고상관 제거] threshold={threshold}, keep_by={keep_by} ({criterion_desc})")
    print(f"  제거: {len(removed)}개, 잔여: {len(keep)}개")
    return keep, removed


def impute_spatial(xs, feat_cols, max_dist=2.0, train_mask=None):
    """
    공간 보간 → lot 평균 → 전체 평균 순서로 결측치를 채우는 함수

    전략:
    1단계: 같은 lot/wafer 내 xy 좌표 기준 거리 가중 평균 (max_dist 이내 이웃)
    2단계: 같은 lot 내 평균
    3단계: train split의 컬럼 전체 평균

    Parameters
    ----------
    xs : DataFrame
        전체 데이터 (run_wf_xy 컬럼 필수)
    feat_cols : list
        imputation 대상 feature 컬럼명 리스트
    max_dist : float
        공간 보간 시 이웃으로 간주할 최대 거리 (기본 2.0, 인접 die까지)
    train_mask : Series[bool], optional
        xs.index와 정렬된 boolean 마스크. True인 행만 1단계 이웃 후보·2단계 lot 평균
        계산에 사용되어 data leakage를 차단한다. None이면 1·2단계는 전체 데이터를 사용.
        단 3단계(전체 평균 fallback)는 train_mask와 무관하게 항상 split=='train' 행만 쓰므로
        standalone 호출 시에도 SPLIT_COL 컬럼이 필요하다.

    Returns
    -------
    xs : DataFrame
        결측이 채워진 DataFrame (임시 컬럼은 정리됨)
    report : dict
        각 단계별 채운 결측 수
    """
    from utils.config import DIE_KEY_COL, SPLIT_COL
    from meta_features import parse_run_wf_xy

    # run_wf_xy("Lot_Wafer_X_Y")를 쪼개 임시 컬럼 생성 → _lot_wafer는 "같은 웨이퍼" 그룹 키
    parse_run_wf_xy(xs, prefix="_", inplace=True, verbose=False)
    xs['_lot_wafer'] = xs['_lot'] + '_' + xs['_wafer_no']
    tmp_cols = ['_lot', '_wafer_no', '_die_x', '_die_y', '_lot_wafer']   # 끝에 지울 임시 컬럼들

    total_na_before = xs[feat_cols].isnull().sum().sum()
    print(f"[공간 보간 imputation] 총 결측: {total_na_before:,}")
    if train_mask is not None:
        n_train = int(train_mask.sum())
        print(f"  train-only 모드: train {n_train:,} / 전체 {len(xs):,} 행")

    if total_na_before == 0:                       # 채울 게 없으면 임시 컬럼만 정리하고 끝
        xs.drop(columns=tmp_cols, inplace=True)
        return xs, {'spatial': 0, 'lot': 0, 'global': 0}

    # --- 1단계: 같은 웨이퍼 안에서 거리 가중 평균으로 보간 ---
    # 보간한 값이 다시 다른 행의 이웃으로 쓰이면 안 되므로, 이웃 참조는 항상 "보간 전 원본"에서만 한다.
    original_arr = xs[feat_cols].values.copy()   # (N, F) — 이웃 참조용 스냅샷 (불변)
    result_arr = original_arr.copy()             # 보간 결과를 누적 기록

    # DataFrame index → 0-based 정수 위치. 그룹 인덱스를 numpy 배열 위치로 바꿀 때 사용
    idx_to_pos = pd.Series(range(len(xs)), index=xs.index)
    die_x_all = xs['_die_x'].values.astype(float)
    die_y_all = xs['_die_y'].values.astype(float)

    # 이웃 후보를 train 행으로 제한 (leakage 차단). train_mask 없으면 전부 허용
    if train_mask is not None:
        is_train_all = train_mask.reindex(xs.index).fillna(False).values.astype(bool)
    else:
        is_train_all = np.ones(len(xs), dtype=bool)

    filled_spatial = 0
    for lw, group in xs.groupby('_lot_wafer'):     # 웨이퍼 단위로 묶어서 처리
        gpos = idx_to_pos.loc[group.index].values  # 이 그룹 행들의 정수 위치
        g_orig = original_arr[gpos]                # (G, F) 이 그룹의 원본 값
        g_na = np.isnan(g_orig)

        if not g_na.any():                         # 그룹에 결측이 없으면 스킵
            continue

        is_train_grp = is_train_all[gpos]          # (G,) 이 그룹에서 누가 train인지
        if not is_train_grp.any():
            # 그룹 안에 train 이웃이 하나도 없으면 공간 보간 불가 → 2단계(lot 평균)로 넘김
            continue

        # 그룹 내 모든 die 쌍 사이 유클리드 거리 행렬 (그룹당 한 번만 계산)
        gx = die_x_all[gpos]
        gy = die_y_all[gpos]
        dx = gx[:, None] - gx[None, :]
        dy = gy[:, None] - gy[None, :]
        dist_mat = np.sqrt(dx**2 + dy**2)          # (G, G)
        np.fill_diagonal(dist_mat, np.inf)         # 자기 자신은 이웃에서 제외 (거리 ∞로)

        na_rows = np.where(g_na.any(axis=1))[0]    # 그룹 내 결측이 하나라도 있는 행들

        for li in na_rows:
            row_na = g_na[li]                      # (F,) 이 행에서 어느 컬럼이 NaN인지
            nan_cols = np.where(row_na)[0]
            if len(nan_cols) == 0:
                continue

            dists = dist_mat[li]                   # (G,) 이 행에서 다른 die까지 거리
            # 이웃 조건: 거리 ≤ max_dist  그리고  train 행
            nbr_mask = (dists <= max_dist) & is_train_grp
            if not nbr_mask.any():
                continue

            nbr_idx = np.where(nbr_mask)[0]
            weights = 1.0 / np.maximum(dists[nbr_idx], 1e-6)   # 가까운 이웃일수록 큰 가중치 (0 나눗셈 방지)

            nbr_vals = g_orig[nbr_idx][:, nan_cols]            # (K, C) 이웃들의 원본 값 (NaN인 컬럼만)
            valid = ~np.isnan(nbr_vals)                        # 이웃도 그 컬럼이 NaN일 수 있음 → 그건 제외

            # 컬럼별로 "유효한 이웃들"의 가중평균: 분자 = Σ w·값, 분모 = Σ w (유효한 것만)
            w_valid = weights[:, None] * valid                 # (K, C)
            w_sum = w_valid.sum(axis=0)                        # (C,) 컬럼별 가중치 합
            has_valid = w_sum > 0                              # 유효 이웃이 하나라도 있는 컬럼

            if not has_valid.any():
                continue

            weighted_sum = (weights[:, None] * np.where(valid, nbr_vals, 0.0)).sum(axis=0)
            avg = weighted_sum / w_sum                         # (C,) 가중평균

            fill_cols = nan_cols[has_valid]                    # 실제로 채울 수 있는 컬럼들
            result_arr[gpos[li], fill_cols] = avg[has_valid]
            filled_spatial += int(has_valid.sum())

    # 1단계 결과를 DataFrame에 반영
    xs[feat_cols] = result_arr

    na_after_spatial = xs[feat_cols].isnull().sum().sum()
    print(f"  1단계 (공간 보간, dist<={max_dist}): {filled_spatial:,}개 채움"
          f" → 잔여: {na_after_spatial:,}")

    # --- 2단계: 1단계로 못 채운 칸은 "같은 lot의 평균"으로 (평균은 train 행에서만 계산) ---
    if na_after_spatial > 0:
        if train_mask is not None:
            train_lot_means = xs.loc[train_mask].groupby('_lot')[feat_cols].mean()
        else:
            train_lot_means = xs.groupby('_lot')[feat_cols].mean()

        # 각 행의 _lot 값에 해당하는 lot 평균을 가져옴 (train에 없는 lot은 NaN → 3단계로 넘어감)
        fill_df = train_lot_means.reindex(xs['_lot'].values)
        fill_df.index = xs.index

        filled_before = xs[feat_cols].isnull().sum().sum()
        xs[feat_cols] = xs[feat_cols].fillna(fill_df)
        filled_lot = filled_before - xs[feat_cols].isnull().sum().sum()
        na_after_lot = xs[feat_cols].isnull().sum().sum()
        lot_label = "train 기준" if train_mask is not None else "전체 기준"
        print(f"  2단계 (lot 평균, {lot_label}): {filled_lot:,}개 채움"
              f" → 잔여: {na_after_lot:,}")
    else:
        filled_lot = 0
        na_after_lot = 0

    # --- 3단계: 그래도 남은 칸은 "train 전체의 컬럼 평균"으로 ---
    if na_after_lot > 0:
        train_means = xs.loc[xs[SPLIT_COL] == 'train', feat_cols].mean()
        filled_before = xs[feat_cols].isnull().sum().sum()
        xs[feat_cols] = xs[feat_cols].fillna(train_means)
        filled_global = filled_before - xs[feat_cols].isnull().sum().sum()
        na_after_global = xs[feat_cols].isnull().sum().sum()
        print(f"  3단계 (train 전체 평균): {filled_global:,}개 채움"
              f" → 잔여: {na_after_global:,}")
    else:
        filled_global = 0

    # 보간용으로 만들었던 임시 컬럼 제거 → 원래 컬럼 구성으로 복원
    xs.drop(columns=tmp_cols, inplace=True)

    report = {
        'total_before': total_na_before,
        'spatial': filled_spatial,
        'lot': filled_lot,
        'global': filled_global,
        'remaining': xs[feat_cols].isnull().sum().sum(),
    }

    print(f"\n  [요약] {total_na_before:,} → "
          f"공간({filled_spatial:,}) → lot({filled_lot:,}) → "
          f"전체({filled_global:,}) → 잔여({report['remaining']:,})")

    return xs, report


def _add_missing_indicators(xs_train, xs_val, xs_test, feat_cols,
                            threshold=0.01):
    """
    train 결측률 기준으로 indicator 컬럼({feat}_missing)을 3개 split에 추가.

    모든 DataFrame을 inplace 수정하고 생성된 indicator 컬럼명 리스트를 반환.
    threshold 이상인 feature가 없으면 빈 리스트.
    """
    # "어느 feature에 indicator를 붙일지"는 train 결측률로만 결정 (val/test 결측률 보면 leakage)
    missing_pct = xs_train[feat_cols].isnull().mean()
    indicator_feats = missing_pct[missing_pct >= threshold].index.tolist()

    indicator_cols = []
    for col in indicator_feats:
        ind_col = f"{col}_missing"
        # "이 값이 원래 결측이었나"를 0/1로 — 결측 자체가 신호일 수 있어 imputation 전에 박제
        xs_train[ind_col] = xs_train[col].isnull().astype(int)
        xs_val[ind_col] = xs_val[col].isnull().astype(int)
        xs_test[ind_col] = xs_test[col].isnull().astype(int)
        indicator_cols.append(ind_col)

    if indicator_cols:
        print(f"[결측 indicator] {len(indicator_cols)}개 컬럼 추가 "
              f"(결측률 >= {threshold*100:.0f}%)")
    return indicator_cols


def impute_missing(xs_train, xs_val, xs_test, feat_cols, add_indicator=False,
                   indicator_threshold=0.01, method="median", knn_neighbors=5):
    """
    결측치 imputation (data leakage 방지: fit on train, transform all)

    Parameters
    ----------
    xs_train, xs_val, xs_test : DataFrame
    feat_cols : list
    add_indicator : bool
        True면 결측률이 indicator_threshold 이상인 feature에 대해
        결측 여부 indicator 컬럼({feat}_missing) 추가 (결측 자체가 가진 정보 보존)
    indicator_threshold : float
        이 비율 이상 결측인 feature에만 indicator 추가. 기본 0.01 (1%)
    method : str
        "median" — train median으로 채움 (빠름, 기본)
        "knn" — KNNImputer (지역 구조 보존)
    knn_neighbors : int
        method="knn"일 때 이웃 수. 기본 5

    Returns
    -------
    xs_train, xs_val, xs_test : DataFrame (결측 채워진 복사본)
    imputer_info : dict (method별 정보)
    indicator_cols : list (추가된 indicator 컬럼명, add_indicator=False면 빈 리스트)
    """
    # 원본을 건드리지 않게 복사본에서 작업
    xs_train = xs_train.copy()
    xs_val = xs_val.copy()
    xs_test = xs_test.copy()

    # indicator는 imputation으로 NaN이 사라지기 전에 만들어야 한다
    indicator_cols = []
    if add_indicator:
        indicator_cols = _add_missing_indicators(
            xs_train, xs_val, xs_test, feat_cols, threshold=indicator_threshold
        )

    imputer_info = {"method": method}

    if method == "median":
        medians = xs_train[feat_cols].median()       # fit: train의 컬럼별 중앙값
        xs_train[feat_cols] = xs_train[feat_cols].fillna(medians)   # transform: 전 split에 같은 값으로
        xs_val[feat_cols] = xs_val[feat_cols].fillna(medians)
        xs_test[feat_cols] = xs_test[feat_cols].fillna(medians)
        imputer_info["medians"] = medians
        print(f"[결측 imputation] method=median")

    elif method == "knn":
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=knn_neighbors, weights="uniform")
        print(f"[결측 imputation] method=knn (n_neighbors={knn_neighbors})")
        print(f"  fitting on train ({len(xs_train):,} rows)...")
        xs_train[feat_cols] = imputer.fit_transform(xs_train[feat_cols])   # fit + transform은 train에서만
        xs_val[feat_cols] = imputer.transform(xs_val[feat_cols])           # val/test는 transform만
        xs_test[feat_cols] = imputer.transform(xs_test[feat_cols])
        imputer_info["knn_neighbors"] = knn_neighbors

    else:
        raise ValueError(f"Unknown imputation method: {method}. Use 'median' or 'knn'")

    remaining = xs_train[feat_cols].isnull().sum().sum()
    print(f"  imputation 후 잔여 결측: {remaining}")
    return xs_train, xs_val, xs_test, imputer_info, indicator_cols


def run_cleaning(xs, feat_cols, xs_dict,
                 const_threshold=1e-6,
                 missing_threshold=0.5,
                 remove_duplicates=True,
                 corr_threshold=0.95,
                 corr_keep_by="std",
                 corr_winsorize_pct=0.0,
                 ys_train=None,
                 add_indicator=False,
                 indicator_threshold=0.01,
                 imputation_method="median",
                 knn_neighbors=5,
                 spatial_max_dist=2.0,
                 post_impute_corr_threshold=None,
                 post_impute_corr_keep_by="std",
                 protected_cols=None):
    """
    클리닝 파이프라인 전체 실행

    Parameters
    ----------
    xs : DataFrame (전체)
    feat_cols : list
    xs_dict : dict (split별 DataFrame)
    const_threshold : float
    missing_threshold : float
    remove_duplicates : bool
    corr_threshold : float or None
        |r| > 이 값인 피처 쌍에서 한쪽 제거. None이면 스킵
    corr_keep_by : {"std", "target_corr"}
        고상관 쌍에서 남길 기준. 기본 "std"
        - "std": std 큰 쪽 남김 (corr_winsorize_pct>0이면 분위수 클립 후 std)
        - "target_corr": |corr(feature, target)| 큰 쪽 남김 (ys_train 필수).
          내부에서 die→unit mean 집계 후 unit-level에서 상관 계산.
    corr_winsorize_pct : float
        corr_keep_by="std"일 때 std 계산 전 양쪽 분위수 클립 비율.
        예: 0.01 → 1%/99% clip. 0이면 생략. 기본 0.0
    ys_train : DataFrame, optional
        corr_keep_by="target_corr"일 때 필수.
        unit-level train Y 데이터 (KEY_COL, TARGET_COL 컬럼 필요).
    add_indicator : bool
        결측 indicator 컬럼 추가 여부
    indicator_threshold : float
        indicator 추가 기준 결측률
    imputation_method : {"median", "knn", "spatial"}
        - "median" (기본): train median으로 채움
        - "knn": KNNImputer
        - "spatial": 같은 lot/wafer 내 xy 거리 기반 공간 보간 → lot 평균
                     → train 전체 평균 3단계. xs_train/val/test를 합쳐서
                     같은 lot 내 이웃을 참조한 뒤 split별로 다시 분리한다.
    knn_neighbors : int
        imputation_method="knn"일 때 이웃 수
    spatial_max_dist : float
        imputation_method="spatial"일 때 공간 보간의 이웃 최대 거리. 기본 2.0
    post_impute_corr_threshold : float or None
        imputation 이후 한 번 더 고상관 제거를 수행할 때의 임계값.
        None이면 스킵. 권장: 0.99 (보수적). imputation으로 인해 인위적으로
        상관이 높아진 컬럼만 추가 제거하는 목적.
        대상은 일반 feature(current_cols)만이며 indicator 컬럼은 제외.
    post_impute_corr_keep_by : {"std", "target_corr"}
        2차 고상관 제거에서 남길 기준. 기본 "std"
    protected_cols : list of str, optional
        클리닝 전 과정(상수/결측/중복/고상관/imputation/post-impute corr)을
        우회시킬 컬럼 목록. 사전 산출된 encoding 컬럼처럼 cleaning이 건드리면
        안 되는 컬럼을 보호. 우회된 컬럼은 끝에 그대로 재부착되어 결과 feat_cols에
        포함된다. None이면 기존 동작 (모든 feat_cols가 cleaning 대상).
        용도: GroupTargetEncoder 산출물 보호 등. `get_default_protected_cols()` 헬퍼 활용.

    Returns
    -------
    xs_train, xs_val, xs_test : DataFrame (클리닝 완료)
    clean_feat_cols : list (남은 feature 컬럼, indicator 컬럼 포함)
    report : dict (각 단계별 제거 내역)
    """
    # protected_cols가 있으면: 그 컬럼들을 빼고 일반 cleaning을 재귀 호출 → 끝나면 다시 붙임
    # (target encoding 결과처럼 cleaning이 손대면 안 되는 컬럼을 통째로 우회시키는 장치)
    if protected_cols:
        protected_set = set(protected_cols)
        protected_in_feat = [c for c in feat_cols if c in protected_set]
        if protected_in_feat:
            other_cols = [c for c in feat_cols if c not in protected_set]
            xs_other = xs.drop(columns=protected_in_feat, errors="ignore")
            xs_dict_other = {
                k: v.drop(columns=protected_in_feat, errors="ignore")
                for k, v in xs_dict.items()
            }

            print(f"[Protected Cleaning] {len(protected_in_feat)} 컬럼 우회: "
                  f"{protected_in_feat}")
            xs_train, xs_val, xs_test, clean_cols, report = run_cleaning(
                xs_other, other_cols, xs_dict_other,
                const_threshold=const_threshold,
                missing_threshold=missing_threshold,
                remove_duplicates=remove_duplicates,
                corr_threshold=corr_threshold,
                corr_keep_by=corr_keep_by,
                corr_winsorize_pct=corr_winsorize_pct,
                ys_train=ys_train,
                add_indicator=add_indicator,
                indicator_threshold=indicator_threshold,
                imputation_method=imputation_method,
                knn_neighbors=knn_neighbors,
                spatial_max_dist=spatial_max_dist,
                post_impute_corr_threshold=post_impute_corr_threshold,
                post_impute_corr_keep_by=post_impute_corr_keep_by,
                protected_cols=None,  # 재귀 호출에선 protected를 비워 무한재귀 방지
            )

            # 우회시켰던 보호 컬럼을 cleaning 결과 DataFrame에 index 맞춰 다시 부착
            for split_name, target in [
                ("train", xs_train), ("validation", xs_val), ("test", xs_test),
            ]:
                src_df = xs_dict[split_name]
                for ec in protected_in_feat:
                    target[ec] = src_df[ec].reindex(target.index).values

            return xs_train, xs_val, xs_test, clean_cols + protected_in_feat, report

    print("=" * 60)
    print("클리닝 파이프라인 시작")
    print(f"원본 feature 수: {len(feat_cols)}")
    print("=" * 60)

    report = {}
    original_count = len(feat_cols)
    current_cols = feat_cols.copy()   # 단계마다 줄여 나갈 "현재 살아 있는 feature" 목록

    # split별 DataFrame 복사 (원본 xs_dict 보호). cleaning 결과는 이 복사본에 반영
    from utils.config import META_COLS
    xs_train = xs_dict["train"].copy()
    xs_val = xs_dict["validation"].copy()
    xs_test = xs_dict["test"].copy()

    def _drop_from_all(cols_to_drop):
        """train/val/test 3개 DataFrame에서 주어진 컬럼들을 한꺼번에 제거."""
        nonlocal xs_train, xs_val, xs_test
        existing = [c for c in cols_to_drop if c in xs_train.columns]
        if existing:
            xs_train = xs_train.drop(columns=existing)
            xs_val = xs_val.drop(columns=existing)
            xs_test = xs_test.drop(columns=existing)

    # 1) 상수/극저분산 제거
    before = len(current_cols)
    current_cols, removed = remove_constant_features(xs_train, current_cols, const_threshold)
    _drop_from_all(removed)
    report["constant"] = removed
    print(f"    컬럼: {before} → {len(current_cols)} ({before - len(current_cols)}개 제거)")
    print(f"    DataFrame: {xs_train.shape}\n")

    # 2) 고결측 제거
    before = len(current_cols)
    current_cols, removed = remove_high_missing_features(xs_train, current_cols, missing_threshold)
    _drop_from_all(removed)
    report["high_missing"] = removed
    print(f"    컬럼: {before} → {len(current_cols)} ({before - len(current_cols)}개 제거)")
    print(f"    DataFrame: {xs_train.shape}\n")

    # 3) 완전 중복 컬럼 제거 (옵션)
    if remove_duplicates:
        before = len(current_cols)
        current_cols, removed = remove_duplicate_features(xs_train, current_cols)
        _drop_from_all(removed)
        report["duplicate"] = removed
        print(f"    컬럼: {before} → {len(current_cols)} ({before - len(current_cols)}개 제거)")
        print(f"    DataFrame: {xs_train.shape}\n")

    # 4) 고상관 제거 (다중공선성 감소) — corr_threshold=None이면 생략
    if corr_threshold is not None:
        before = len(current_cols)
        current_cols, removed = remove_high_corr_features(
            xs_train, current_cols,
            threshold=corr_threshold,
            keep_by=corr_keep_by,
            ys_train=ys_train,
            winsorize_pct=corr_winsorize_pct,
        )
        _drop_from_all(removed)
        report["high_corr"] = removed
        print(f"    컬럼: {before} → {len(current_cols)} ({before - len(current_cols)}개 제거)")
        print(f"    DataFrame: {xs_train.shape}\n")

    # 5) 결측 imputation (+ indicator). 방식에 따라 분기
    if imputation_method in ("median", "knn"):
        xs_train, xs_val, xs_test, imputer_info, indicator_cols = impute_missing(
            xs_train, xs_val, xs_test, current_cols,
            add_indicator=add_indicator,
            indicator_threshold=indicator_threshold,
            method=imputation_method,
            knn_neighbors=knn_neighbors,
        )
        report["imputer_info"] = imputer_info
        report["indicator_cols"] = indicator_cols
    elif imputation_method == "spatial":
        from utils.config import SPLIT_COL

        # impute_spatial은 indicator 기능이 없으므로, 필요하면 보간 전에 여기서 직접 만든다
        indicator_cols = []
        if add_indicator:
            indicator_cols = _add_missing_indicators(
                xs_train, xs_val, xs_test, current_cols,
                threshold=indicator_threshold,
            )

        # 같은 lot/wafer 이웃을 찾으려면 3 split을 합쳐야 함. 단 이웃·평균 계산은 train_mask로 train에 한정 (leakage 차단)
        combined = pd.concat([xs_train, xs_val, xs_test], axis=0)
        train_mask = combined[SPLIT_COL] == 'train'
        combined, spatial_report = impute_spatial(
            combined, current_cols,
            max_dist=spatial_max_dist,
            train_mask=train_mask,
        )

        # 보간 끝났으면 split 컬럼 기준으로 다시 3개로 쪼갬
        xs_train = combined[combined[SPLIT_COL] == 'train'].copy()
        xs_val = combined[combined[SPLIT_COL] == 'validation'].copy()
        xs_test = combined[combined[SPLIT_COL] == 'test'].copy()

        report["imputer_info"] = {"method": "spatial", **spatial_report}
        report["indicator_cols"] = indicator_cols
    else:
        raise ValueError(
            f"Unknown imputation_method: {imputation_method!r}. "
            f"Use 'median', 'knn', or 'spatial'"
        )

    # 6) (선택) imputation 이후 2차 고상관 제거
    #    median/평균으로 결측을 채우다 보면 원래 NaN이던 자리가 다 같은 값이 되어 상관이 인위적으로 올라간다.
    #    그런 컬럼만 잡도록 보수적인 임계값(권장 0.99)을 쓰고, indicator 컬럼은 대상에서 제외(current_cols만).
    if post_impute_corr_threshold is not None:
        before = len(current_cols)
        print()
        current_cols, removed = remove_high_corr_features(
            xs_train, current_cols,
            threshold=post_impute_corr_threshold,
            keep_by=post_impute_corr_keep_by,
            ys_train=ys_train,
            winsorize_pct=corr_winsorize_pct,
        )
        _drop_from_all(removed)
        report["high_corr_post_impute"] = removed
        print(f"    [고상관 제거 2차 / imputation 후] threshold="
              f"{post_impute_corr_threshold}")
        print(f"    컬럼: {before} → {len(current_cols)} "
              f"({before - len(current_cols)}개 제거)")
        print(f"    DataFrame: {xs_train.shape}")

    # 최종 feature 목록 = 살아남은 일반 feature + (있으면) indicator 컬럼
    all_feat_cols = current_cols + indicator_cols

    print(f"\n{'=' * 60}")
    print(f"클리닝 완료: {original_count} → {len(current_cols)} features "
          f"({original_count - len(current_cols)}개 제거)")
    if indicator_cols:
        print(f"  + indicator 컬럼: {len(indicator_cols)}개 → 총 {len(all_feat_cols)}개")
    print(f"  train: {xs_train.shape}")
    print(f"  val:   {xs_val.shape}")
    print(f"  test:  {xs_test.shape}")
    print("=" * 60)

    return xs_train, xs_val, xs_test, all_feat_cols, report


# ------------------------------------------------------------
# Degenerate feature Binarize — "거의 한 값"인 feature를 0/1로 압축
# ------------------------------------------------------------

def binarize_degenerate(xs_train, xs_val, xs_test, feat_cols,
                        top_value_threshold=0.99,
                        max_unique=None,
                        verbose=True):
    """
    Degenerate feature를 binary indicator로 변환.

    "99%가 한 값 + 1%가 outlier" 또는 "2~5개 값만 가지는 이산형" feature는 어떤
    스케일링도 skew를 못 잡음. "특이값 여부 (0/1)"로 요약하여 파이프라인 안정화.

    변환 조건 (OR):
      1) train 기준 최빈값 비율 > top_value_threshold
      2) train 기준 nunique <= max_unique (max_unique=None이면 스킵)

    Parameters
    ----------
    xs_train, xs_val, xs_test : DataFrame (cleaning 이후, outlier 이전에 호출 권장)
    feat_cols : list
        대상 feature. 보통 clean_cols
    top_value_threshold : float, default 0.99
        train 기준 최빈값 비율이 이 값을 초과하면 변환.
        0.99 → "99% 이상이 한 값"인 feature 대상
    max_unique : int or None, default None
        train 기준 nunique가 이 값 이하이면 무조건 변환 (top%와 OR).
        None이면 nunique 조건 스킵 (top%만 사용).
        예: 5 → "train nunique ≤ 5인 이산형 feature도 binary로"
    verbose : bool, default True

    Returns
    -------
    xs_train, xs_val, xs_test : DataFrame (inplace 변경됨)
    report : dict
        {'n_converted', 'top_value_threshold', 'max_unique',
         'converted': [(feature, mode_val, top_pct, n_uniq, trigger), ...]}

    Notes
    -----
    - train 기준 최빈값/nunique를 val/test에도 동일하게 적용 (누수 방지)
    - 변환 후 dtype은 int8 (공간 절약, 모델 입력 호환)
    - feat_cols 리스트는 변경되지 않음 (컬럼 수 동일, 값만 이진화)
    - trigger 필드: 'top%', 'nuniq', 'top%+nuniq' 중 하나 (어떤 조건이 발동했는지)
    """
    converted = []

    for c in feat_cols:
        s = xs_train[c].dropna()              # 판정은 train의 결측 제외 값으로
        if len(s) == 0:
            continue
        mode_val = s.mode().iloc[0]           # 최빈값
        top_pct = float((s == mode_val).mean())   # 최빈값이 차지하는 비율
        n_uniq = int(s.nunique())             # 고유값 개수

        hit_top = top_pct > top_value_threshold
        hit_uniq = (max_unique is not None) and (n_uniq <= max_unique)

        if hit_top or hit_uniq:
            # "최빈값이면 0, 아니면 1" — train의 mode_val 기준을 val/test에도 그대로 적용 (leakage 방지)
            for df in [xs_train, xs_val, xs_test]:
                df[c] = (df[c] != mode_val).astype(np.int8)
            triggers = []
            if hit_top:  triggers.append("top%")
            if hit_uniq: triggers.append("nuniq")
            converted.append((c, float(mode_val), top_pct, n_uniq, "+".join(triggers)))

    if verbose:
        cond_str = f"top% > {top_value_threshold*100:.0f}%"
        if max_unique is not None:
            cond_str += f" OR nuniq ≤ {max_unique}"
        print(f"[Binarize] {cond_str} → binary 변환")
        print(f"  변환 대상: {len(converted)}개 / {len(feat_cols)}개")
        for c, v, p, n, why in converted[:25]:    # 너무 길지 않게 앞 25개만 출력
            print(f"  {c}: mode={v:.4g} (top%={p*100:.2f}%, nuniq={n}) → 0/1  [{why}]")
        if len(converted) > 25:
            print(f"  ... 외 {len(converted) - 25}개")

    report = {
        "n_converted": len(converted),
        "top_value_threshold": top_value_threshold,
        "max_unique": max_unique,
        "converted": converted,
    }
    return xs_train, xs_val, xs_test, report
