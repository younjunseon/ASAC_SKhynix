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







