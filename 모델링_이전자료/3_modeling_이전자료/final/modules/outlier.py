"""
이상치 처리 모듈
- IQR 기반 탐지
- Winsorization (분위수 경계로 클리핑)
- Grubbs 검정 (통계적으로 유의한 이상치만 처리)
- 로트별 국소 기준 (AEC DPAT, run_id별 중앙값/백분위수 기반)

EDA 결과 기반:
- IQR 기준 이상치 5% 초과: 167개 feature
- X393(45.7%), X988(40.4%) 등 극단적 이상치 비율
- Feature 스케일 극도로 불균일 (mean 범위: -2,293 ~ 20,201,109)
"""
import pandas as pd
import numpy as np
from scipy import stats as sp_stats


def detect_outliers_iqr(xs, feat_cols, multiplier=1.5):
    """
    IQR 기반 이상치 탐지 (통계만 반환, 데이터 수정 안 함)

    Parameters
    ----------
    xs : DataFrame
    feat_cols : list
    multiplier : float
        IQR 배수. 기본 1.5

    Returns
    -------
    outlier_stats : DataFrame
        feature별 이상치 수, 비율
    """
    Q1 = xs[feat_cols].quantile(0.25)
    Q3 = xs[feat_cols].quantile(0.75)
    IQR = Q3 - Q1

    low = xs[feat_cols] < (Q1 - multiplier * IQR)
    high = xs[feat_cols] > (Q3 + multiplier * IQR)
    outlier_count = (low | high).sum()
    outlier_pct = (outlier_count / len(xs) * 100).round(2)

    stats = pd.DataFrame({
        "outlier_count": outlier_count,
        "outlier_pct": outlier_pct,
    }).sort_values("outlier_pct", ascending=False)

    print(f"[이상치 탐지] IQR × {multiplier}")
    print(f"  이상치 > 5%: {(stats['outlier_pct'] > 5).sum()}개")
    print(f"  이상치 > 10%: {(stats['outlier_pct'] > 10).sum()}개")
    return stats


def _match_bounds_dtype(xs_train, feat_cols, lower, upper):
    """
    bounds(lower/upper Series)를 입력 feat_cols의 dtype에 맞춤.

    pandas `quantile`은 입력 dtype과 무관하게 float64를 반환하므로,
    float32 파이프라인에서 clip 결과가 float64로 승격되는 것을 방지한다.
    """
    try:
        in_dtype = xs_train[feat_cols].dtypes.iloc[0]
    except (AttributeError, IndexError):
        return lower, upper
    if in_dtype == np.float32:
        if hasattr(lower, 'astype'):
            # X1086 파생 bounds는 float64 유지 (날짜값 8자리, float32 정밀도 부족)
            safe_idx = [i for i in lower.index if not str(i).startswith("X1086")]
            lower[safe_idx] = lower[safe_idx].astype('float32')
        if hasattr(upper, 'astype'):
            safe_idx = [i for i in upper.index if not str(i).startswith("X1086")]
            upper[safe_idx] = upper[safe_idx].astype('float32')
    return lower, upper


def _apply_clip_to_splits(xs_train, xs_val, xs_test, feat_cols, lower, upper):
    """
    train/val/test 3개 split을 동일한 lower/upper 경계로 clip.

    lower/upper는 Series(feat_cols 인덱스) 또는 scalar. train 기준으로
    이미 계산되어 있어야 하며, 이 헬퍼는 copy → clip만 수행한다.
    입력이 float32인 경우 bounds를 float32로 맞춰 dtype 일관성 유지.

    Returns
    -------
    xs_train, xs_val, xs_test : DataFrame (클리핑된 복사본)
    bounds : DataFrame (feature별 lower/upper 경계)
    """
    xs_train = xs_train.copy()
    xs_val = xs_val.copy()
    xs_test = xs_test.copy()

    lower, upper = _match_bounds_dtype(xs_train, feat_cols, lower, upper)

    xs_train[feat_cols] = xs_train[feat_cols].clip(lower, upper, axis=1)
    xs_val[feat_cols] = xs_val[feat_cols].clip(lower, upper, axis=1)
    xs_test[feat_cols] = xs_test[feat_cols].clip(lower, upper, axis=1)

    bounds = pd.DataFrame({"lower": lower, "upper": upper})
    return xs_train, xs_val, xs_test, bounds


def winsorize(xs_train, xs_val, xs_test, feat_cols,
              lower_pct=0.01, upper_pct=0.99):
    """
    Winsorization: train 기준 분위수로 클리핑 (data leakage 방지)

    Parameters
    ----------
    xs_train, xs_val, xs_test : DataFrame
    feat_cols : list
    lower_pct : float
        하위 분위수 (기본 1%)
    upper_pct : float
        상위 분위수 (기본 99%)

    Returns
    -------
    xs_train, xs_val, xs_test : DataFrame (클리핑된 복사본)
    bounds : DataFrame (feature별 lower/upper 경계)
    """
    lower = xs_train[feat_cols].quantile(lower_pct)
    upper = xs_train[feat_cols].quantile(upper_pct)

    xs_train, xs_val, xs_test, bounds = _apply_clip_to_splits(
        xs_train, xs_val, xs_test, feat_cols, lower, upper
    )

    print(f"[Winsorization] lower={lower_pct*100:.0f}%, upper={upper_pct*100:.0f}%")
    print(f"  적용 feature: {len(feat_cols)}개")
    return xs_train, xs_val, xs_test, bounds


def iqr_clip(xs_train, xs_val, xs_test, feat_cols, multiplier=1.5):
    """
    IQR 기반 클리핑: Q1-k*IQR ~ Q3+k*IQR 경계로 clip
    (Winsorization과 달리 사분위수 기반 경계 사용)

    Parameters
    ----------
    xs_train, xs_val, xs_test : DataFrame
    feat_cols : list
    multiplier : float
        IQR 배수 (1.5: 표준, 3.0: 극단값만)

    Returns
    -------
    xs_train, xs_val, xs_test : DataFrame (클리핑된 복사본)
    bounds : DataFrame (feature별 lower/upper 경계)
    """
    Q1 = xs_train[feat_cols].quantile(0.25)
    Q3 = xs_train[feat_cols].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR

    xs_train, xs_val, xs_test, bounds = _apply_clip_to_splits(
        xs_train, xs_val, xs_test, feat_cols, lower, upper
    )

    print(f"[IQR Clip] multiplier={multiplier}")
    print(f"  적용 feature: {len(feat_cols)}개")
    return xs_train, xs_val, xs_test, bounds


def grubbs_clip(xs_train, xs_val, xs_test, feat_cols,
                alpha=0.05, max_rounds=5):
    """
    Grubbs 검정 기반 이상치 처리 (논문 5-4 근거)
    통계적으로 유의한 이상치만 경계값으로 clip — 가장 보수적

    원리:
    1. feature별로 Grubbs 검정 수행 (양측)
    2. p < alpha이면 해당 극값을 이상치로 판정
    3. 이상치를 이상치 제외 데이터의 min/max로 clip
    4. max_rounds만큼 반복 (한 번에 하나씩 처리)

    Parameters
    ----------
    xs_train, xs_val, xs_test : DataFrame
    feat_cols : list
    alpha : float
        유의수준 (기본 0.05). 낮을수록 보수적 (확실한 이상치만 처리)
    max_rounds : int
        반복 검정 횟수 (기본 5). 높으면 더 많은 이상치 제거

    Returns
    -------
    xs_train, xs_val, xs_test : DataFrame (처리된 복사본)
    bounds : DataFrame (feature별 최종 lower/upper 경계)
    """
    xs_train = xs_train.copy()
    xs_val = xs_val.copy()
    xs_test = xs_test.copy()

    total_clipped = 0
    lower_bounds = {}
    upper_bounds = {}

    for col in feat_cols:
        data = xs_train[col].dropna().values
        if len(data) < 3:
            lower_bounds[col] = data.min() if len(data) > 0 else np.nan
            upper_bounds[col] = data.max() if len(data) > 0 else np.nan
            continue

        col_lower = data.min()
        col_upper = data.max()

        for _ in range(max_rounds):
            n = len(data)
            if n < 3:
                break

            mean = data.mean()
            std = data.std(ddof=1)
            if std == 0:
                break

            # 양쪽 극값 중 평균에서 더 먼 값 선택
            abs_dev = np.abs(data - mean)
            max_idx = abs_dev.argmax()
            G = abs_dev[max_idx] / std

            # Grubbs 임계값 계산 (t-분포 기반)
            t_crit = sp_stats.t.ppf(1 - alpha / (2 * n), n - 2)
            G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(
                t_crit**2 / (n - 2 + t_crit**2)
            )

            if G <= G_crit:
                break  # 유의한 이상치 없음 → 종료

            # 이상치 제거 후 경계 갱신
            data = np.delete(data, max_idx)
            total_clipped += 1

        col_lower = data.min()
        col_upper = data.max()
        lower_bounds[col] = col_lower
        upper_bounds[col] = col_upper

    lower_s = pd.Series(lower_bounds)
    upper_s = pd.Series(upper_bounds)

    lower_s, upper_s = _match_bounds_dtype(xs_train, feat_cols, lower_s, upper_s)

    xs_train[feat_cols] = xs_train[feat_cols].clip(lower_s, upper_s, axis=1)
    xs_val[feat_cols] = xs_val[feat_cols].clip(lower_s, upper_s, axis=1)
    xs_test[feat_cols] = xs_test[feat_cols].clip(lower_s, upper_s, axis=1)

    bounds = pd.DataFrame({"lower": lower_s, "upper": upper_s})

    print(f"[Grubbs 검정] alpha={alpha}, max_rounds={max_rounds}")
    print(f"  적용 feature: {len(feat_cols)}개")
    print(f"  총 이상치 clip 횟수: {total_clipped}")
    return xs_train, xs_val, xs_test, bounds


def lot_local_clip(xs_train, xs_val, xs_test, feat_cols,
                   run_id_col="run_wf_xy", lower_pct=0.01, upper_pct=0.99,
                   min_lot_size=10):
    """
    로트별 국소 기준 이상치 처리 (논문 5-3 근거, AEC DPAT 방식)
    전역 기준 대신 로트(lot)별로 분위수 경계를 세움

    원리:
    - 반도체 공정은 로트마다 수준이 다름
    - 전역 기준으로 하면 로트 간 정상적 차이를 이상치로 오판
    - 로트별로 기준을 세우면 "해당 로트 안에서 튀는 값"만 잡음

    Parameters
    ----------
    xs_train, xs_val, xs_test : DataFrame
    feat_cols : list
    run_id_col : str
        로트 구분 컬럼. "run_wf_xy"(기본)이면 parse_run_wf_xy로 파싱.
        이미 로트 ID 컬럼이 있으면 그 컬럼명을 지정
    lower_pct, upper_pct : float
        로트별 분위수 경계 (기본 1%, 99%)
    min_lot_size : int
        이 수 미만인 로트는 전역 기준 적용 (소량 로트는 통계 불안정)

    Returns
    -------
    xs_train, xs_val, xs_test : DataFrame (처리된 복사본)
    report : dict (로트별/전역 처리 통계)
    """
    from meta_features import parse_run_wf_xy

    xs_train = xs_train.copy()
    xs_val = xs_val.copy()
    xs_test = xs_test.copy()

    # 로트 컬럼 준비
    parsed_cols = None
    if run_id_col == "run_wf_xy" and "run_wf_xy" in xs_train.columns:
        for df in [xs_train, xs_val, xs_test]:
            parse_run_wf_xy(df, prefix="_", inplace=True, verbose=False)
        lot_col = "_lot"
        parsed_cols = ["_lot", "_wafer_no", "_die_x", "_die_y"]
    elif run_id_col in xs_train.columns:
        lot_col = run_id_col
    else:
        raise ValueError(f"컬럼 '{run_id_col}'을 찾을 수 없습니다.")

    # 전역 기준 (소량 로트 fallback용) — train 기준
    global_lower = xs_train[feat_cols].quantile(lower_pct)
    global_upper = xs_train[feat_cols].quantile(upper_pct)
    global_lower, global_upper = _match_bounds_dtype(
        xs_train, feat_cols, global_lower, global_upper
    )

    # 로트별 처리 (train 기준으로 경계 산출)
    lot_sizes = xs_train[lot_col].value_counts()
    large_lots = lot_sizes[lot_sizes >= min_lot_size].index
    large_lot_set = set(large_lots)

    # 대형 로트의 경계를 한 번만 계산해 캐싱 (val/test에서 재사용)
    lot_bounds = {}
    for lot in large_lots:
        mask_tr = xs_train[lot_col] == lot
        lot_lower = xs_train.loc[mask_tr, feat_cols].quantile(lower_pct)
        lot_upper = xs_train.loc[mask_tr, feat_cols].quantile(upper_pct)
        lot_lower, lot_upper = _match_bounds_dtype(
            xs_train, feat_cols, lot_lower, lot_upper
        )
        lot_bounds[lot] = (lot_lower, lot_upper)

        xs_train.loc[mask_tr, feat_cols] = (
            xs_train.loc[mask_tr, feat_cols].clip(lot_lower, lot_upper, axis=1)
        )

    # 소형 로트: 전역 기준 적용
    small_lots = lot_sizes[lot_sizes < min_lot_size].index
    for lot in small_lots:
        mask_tr = xs_train[lot_col] == lot
        xs_train.loc[mask_tr, feat_cols] = (
            xs_train.loc[mask_tr, feat_cols].clip(global_lower, global_upper, axis=1)
        )

    # val/test: 캐시된 경계 재사용, 없으면 전역 기준
    for df in [xs_val, xs_test]:
        for lot in df[lot_col].unique():
            mask = df[lot_col] == lot
            if lot in large_lot_set:
                lot_lower, lot_upper = lot_bounds[lot]
                df.loc[mask, feat_cols] = (
                    df.loc[mask, feat_cols].clip(lot_lower, lot_upper, axis=1)
                )
            else:
                df.loc[mask, feat_cols] = (
                    df.loc[mask, feat_cols].clip(global_lower, global_upper, axis=1)
                )

    # 임시 컬럼 제거
    if parsed_cols is not None:
        for df in [xs_train, xs_val, xs_test]:
            df.drop(columns=parsed_cols, inplace=True)

    report = {
        "n_large_lots": len(large_lots),
        "n_small_lots": len(small_lots),
        "min_lot_size": min_lot_size,
        "lower_pct": lower_pct,
        "upper_pct": upper_pct,
    }

    print(f"[로트별 국소 기준] lower={lower_pct*100:.0f}%, upper={upper_pct*100:.0f}%")
    print(f"  대형 로트 (>={min_lot_size}): {len(large_lots)}개 → 로트별 기준")
    print(f"  소형 로트 (<{min_lot_size}): {len(small_lots)}개 → 전역 기준")
    print(f"  적용 feature: {len(feat_cols)}개")
    return xs_train, xs_val, xs_test, report


def run_outlier_treatment(xs_train, xs_val, xs_test, feat_cols,
                          method="winsorize",
                          lower_pct=0.01, upper_pct=0.99,
                          iqr_multiplier=1.5,
                          grubbs_alpha=0.05, grubbs_max_rounds=5,
                          lot_run_id_col="run_wf_xy",
                          lot_min_size=10):
    """
    이상치 처리 파이프라인 전체 실행

    Parameters
    ----------
    xs_train, xs_val, xs_test : DataFrame
    feat_cols : list
    method : str
        "winsorize" — 분위수 경계 clip (기본)
        "iqr_clip" — IQR 기반 경계 clip
        "grubbs" — Grubbs 검정 (통계적으로 유의한 이상치만, 가장 보수적)
        "lot_local" — 로트별 국소 기준 (run_id별 분위수)
        "none" — 이상치 처리 안 함
    lower_pct, upper_pct : float
        winsorize / lot_local 시 하위/상위 분위수
    iqr_multiplier : float
        iqr_clip 시 IQR 배수 (1.5: 표준)
    grubbs_alpha : float
        grubbs 시 유의수준 (0.05: 표준, 0.01: 더 보수적)
    grubbs_max_rounds : int
        grubbs 시 반복 검정 횟수
    lot_run_id_col : str
        lot_local 시 로트 구분 컬럼
    lot_min_size : int
        lot_local 시 로트별 기준 적용 최소 로트 크기

    Returns
    -------
    xs_train, xs_val, xs_test : DataFrame (처리 완료)
    report : dict
    """
    print("=" * 60)
    print(f"이상치 처리 파이프라인 시작 (method={method})")
    print("=" * 60)

    report = {"method": method}

    # 1. 처리 전 이상치 현황
    stats_before = detect_outliers_iqr(xs_train, feat_cols)
    report["before"] = stats_before

    # 2. 메서드별 처리
    if method == "winsorize":
        xs_train, xs_val, xs_test, bounds = winsorize(
            xs_train, xs_val, xs_test, feat_cols, lower_pct, upper_pct
        )
        report["bounds"] = bounds
    elif method == "iqr_clip":
        xs_train, xs_val, xs_test, bounds = iqr_clip(
            xs_train, xs_val, xs_test, feat_cols, iqr_multiplier
        )
        report["bounds"] = bounds
    elif method == "grubbs":
        xs_train, xs_val, xs_test, bounds = grubbs_clip(
            xs_train, xs_val, xs_test, feat_cols,
            alpha=grubbs_alpha, max_rounds=grubbs_max_rounds
        )
        report["bounds"] = bounds
    elif method == "lot_local":
        xs_train, xs_val, xs_test, lot_report = lot_local_clip(
            xs_train, xs_val, xs_test, feat_cols,
            run_id_col=lot_run_id_col,
            lower_pct=lower_pct, upper_pct=upper_pct,
            min_lot_size=lot_min_size
        )
        report["lot_report"] = lot_report
    elif method == "none":
        print("[이상치 처리 스킵]")
    else:
        raise ValueError(f"Unknown outlier method: {method}. "
                         f"Use 'winsorize', 'iqr_clip', 'grubbs', "
                         f"'lot_local', or 'none'")

    print(f"\n{'=' * 60}")
    print(f"이상치 처리 완료 (method={method})")
    print(f"  train: {xs_train.shape}")
    print("=" * 60)

    return xs_train, xs_val, xs_test, report


# ============================================================
# 2차 funnel — IsolationForest 다변량 이상치 점수 (feature 추가)
# ============================================================

def multivariate_anomaly_score(xs_train, xs_val, xs_test, feat_cols,
                               contamination='auto',
                               n_estimators=200,
                               score_col='iso_anomaly_score',
                               max_samples='auto',
                               random_state=42):
    """
    IsolationForest 기반 다변량 이상치 점수를 컬럼으로 추가 (제거 아님).

    단변량 clip의 한계 보완 목적 (EDA: 단일 |r| max=0.037). 트리 모델이
    `iso_anomaly_score > τ` 분기로 자동 활용 가능.

    Parameters
    ----------
    xs_train, xs_val, xs_test : DataFrame
    feat_cols : list
        anomaly score 계산에 쓸 feature (일반적으로 cleaning 이후 남은 cols)
    contamination : 'auto' or float in (0, 0.5), default 'auto'
    n_estimators : int, default 200
    score_col : str, default 'iso_anomaly_score'
        추가될 컬럼 이름
    max_samples : 'auto' or int, default 'auto'
        학습 샘플 수 제한. 175K × 수백 feature 메모리 대응 시 10000 등으로 제한 가능
    random_state : int, default 42

    Returns
    -------
    xs_train, xs_val, xs_test : DataFrame (score_col 컬럼 추가됨)
    new_feat_cols : list (기존 feat_cols + [score_col])
    report : dict
    """
    from sklearn.ensemble import IsolationForest

    iso = IsolationForest(contamination=contamination,
                          n_estimators=n_estimators,
                          max_samples=max_samples,
                          random_state=random_state,
                          n_jobs=-1)
    iso.fit(xs_train[feat_cols].values)

    # decision_function: 높을수록 normal, 낮을수록 anomaly
    # 부호 뒤집어 "높을수록 이상"으로 통일. dtype은 feature와 동일하게 맞춰
    # 기존 파이프라인의 float32 일관성 유지
    in_dtype = xs_train[feat_cols].dtypes.iloc[0]
    for df in [xs_train, xs_val, xs_test]:
        scores = -iso.decision_function(df[feat_cols].values)
        if in_dtype == np.float32:
            scores = scores.astype(np.float32, copy=False)
        df[score_col] = scores

    new_feat_cols = list(feat_cols) + [score_col]
    report = {
        'n_estimators': n_estimators,
        'contamination': contamination,
        'max_samples': max_samples,
        'score_col': score_col,
        'train_score_range': (float(xs_train[score_col].min()),
                              float(xs_train[score_col].max())),
        'val_score_range':   (float(xs_val[score_col].min()),
                              float(xs_val[score_col].max())),
        'test_score_range':  (float(xs_test[score_col].min()),
                              float(xs_test[score_col].max())),
    }
    print(f"[IsoForest] anomaly score 컬럼 추가: '{score_col}'")
    print(f"  n_estimators={n_estimators}, contamination={contamination}")
    print(f"  train score 범위: [{report['train_score_range'][0]:.4f}, "
          f"{report['train_score_range'][1]:.4f}]")
    print(f"  val   score 범위: [{report['val_score_range'][0]:.4f}, "
          f"{report['val_score_range'][1]:.4f}]")
    print(f"  test  score 범위: [{report['test_score_range'][0]:.4f}, "
          f"{report['test_score_range'][1]:.4f}]")
    return xs_train, xs_val, xs_test, new_feat_cols, report
