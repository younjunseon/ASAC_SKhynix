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

    xs_train = xs_train.copy()
    xs_val = xs_val.copy()
    xs_test = xs_test.copy()

    xs_train[feat_cols] = xs_train[feat_cols].clip(lower, upper, axis=1)
    xs_val[feat_cols] = xs_val[feat_cols].clip(lower, upper, axis=1)
    xs_test[feat_cols] = xs_test[feat_cols].clip(lower, upper, axis=1)

    bounds = pd.DataFrame({"lower": lower, "upper": upper})

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

    xs_train = xs_train.copy()
    xs_val = xs_val.copy()
    xs_test = xs_test.copy()

    xs_train[feat_cols] = xs_train[feat_cols].clip(lower, upper, axis=1)
    xs_val[feat_cols] = xs_val[feat_cols].clip(lower, upper, axis=1)
    xs_test[feat_cols] = xs_test[feat_cols].clip(lower, upper, axis=1)

    bounds = pd.DataFrame({"lower": lower, "upper": upper})

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
    전역 기준 대신 로트(run_id)별로 분위수 경계를 세움

    원리:
    - 반도체 공정은 로트마다 수준이 다름
    - 전역 기준으로 하면 로트 간 정상적 차이를 이상치로 오판
    - 로트별로 기준을 세우면 "해당 로트 안에서 튀는 값"만 잡음

    Parameters
    ----------
    xs_train, xs_val, xs_test : DataFrame
    feat_cols : list
    run_id_col : str
        로트 구분 컬럼. run_wf_xy에서 파싱한 run_id 또는 run_wf_xy 자체
    lower_pct, upper_pct : float
        로트별 분위수 경계 (기본 1%, 99%)
    min_lot_size : int
        이 수 미만인 로트는 전역 기준 적용 (소량 로트는 통계 불안정)

    Returns
    -------
    xs_train, xs_val, xs_test : DataFrame (처리된 복사본)
    report : dict (로트별/전역 처리 통계)
    """
    xs_train = xs_train.copy()
    xs_val = xs_val.copy()
    xs_test = xs_test.copy()

    # run_id 파싱: run_wf_xy에서 첫 번째 '_' 앞 부분 = 작업번호(로트)
    if run_id_col == "run_wf_xy" and "run_wf_xy" in xs_train.columns:
        for df in [xs_train, xs_val, xs_test]:
            df["_lot_id"] = df["run_wf_xy"].str.split("_").str[0]
        lot_col = "_lot_id"
    elif run_id_col in xs_train.columns:
        lot_col = run_id_col
    else:
        raise ValueError(f"컬럼 '{run_id_col}'을 찾을 수 없습니다.")

    # 전역 기준 (소량 로트 fallback용) — train 기준
    global_lower = xs_train[feat_cols].quantile(lower_pct)
    global_upper = xs_train[feat_cols].quantile(upper_pct)

    # 로트별 처리 (train 기준으로 경계 산출)
    lot_sizes = xs_train[lot_col].value_counts()
    large_lots = lot_sizes[lot_sizes >= min_lot_size].index
    small_lots = lot_sizes[lot_sizes < min_lot_size].index

    # 대형 로트: 로트별 분위수 경계
    for lot in large_lots:
        mask_tr = xs_train[lot_col] == lot
        lot_lower = xs_train.loc[mask_tr, feat_cols].quantile(lower_pct)
        lot_upper = xs_train.loc[mask_tr, feat_cols].quantile(upper_pct)

        xs_train.loc[mask_tr, feat_cols] = (
            xs_train.loc[mask_tr, feat_cols].clip(lot_lower, lot_upper, axis=1)
        )

    # 소형 로트: 전역 기준 적용
    for lot in small_lots:
        mask_tr = xs_train[lot_col] == lot
        xs_train.loc[mask_tr, feat_cols] = (
            xs_train.loc[mask_tr, feat_cols].clip(global_lower, global_upper, axis=1)
        )

    # val/test: 해당 로트가 train에 있으면 train 로트 기준, 없으면 전역 기준
    for df in [xs_val, xs_test]:
        for lot in df[lot_col].unique():
            mask = df[lot_col] == lot
            if lot in large_lots:
                mask_tr = xs_train[lot_col] == lot
                lot_lower = xs_train.loc[mask_tr, feat_cols].quantile(lower_pct)
                lot_upper = xs_train.loc[mask_tr, feat_cols].quantile(upper_pct)
                df.loc[mask, feat_cols] = (
                    df.loc[mask, feat_cols].clip(lot_lower, lot_upper, axis=1)
                )
            else:
                df.loc[mask, feat_cols] = (
                    df.loc[mask, feat_cols].clip(global_lower, global_upper, axis=1)
                )

    # 임시 컬럼 제거
    if "_lot_id" in xs_train.columns:
        for df in [xs_train, xs_val, xs_test]:
            df.drop(columns=["_lot_id"], inplace=True)

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

    # 3. 처리 후 이상치 현황
    stats_after = detect_outliers_iqr(xs_train, feat_cols)
    report["after"] = stats_after

    # 전후 비교
    before_5pct = (stats_before["outlier_pct"] > 5).sum()
    after_5pct = (stats_after["outlier_pct"] > 5).sum()
    before_10pct = (stats_before["outlier_pct"] > 10).sum()
    after_10pct = (stats_after["outlier_pct"] > 10).sum()

    print(f"\n{'=' * 60}")
    print("이상치 처리 완료")
    print(f"  이상치 >5%  feature: {before_5pct} → {after_5pct} ({before_5pct - after_5pct}개 감소)")
    print(f"  이상치 >10% feature: {before_10pct} → {after_10pct} ({before_10pct - after_10pct}개 감소)")
    print(f"  train: {xs_train.shape}")
    print("=" * 60)

    return xs_train, xs_val, xs_test, report
