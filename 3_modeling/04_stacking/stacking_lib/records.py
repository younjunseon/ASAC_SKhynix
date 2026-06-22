"""Record dataclass + 점수 계산.

v2의 Record와 score_rec를 모듈로 분리. die-level / unit-level RMSE를 별도 필드로 보관.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Record:
    """탐색/refit 결과 1건.

    Attributes
    ----------
    tag, stage, method : 디버깅용 식별자
    n_base             : subset에서 고른 base 모델 수
    val_rmse, test_rmse, oof_rmse :
        unit-level RMSE — postprocess.tune_and_apply 후 unit 집계된 최종 예측 기준.
        v3에서 stacking의 "원본 지표"이며, select_by="oof"는 이 oof_rmse를 본다.
    val_rmse_die, test_rmse_die, oof_rmse_die :
        die-level RMSE — broadcast된 y_die 기준. 진단용으로 같이 박제 (집계 전 메타 품질).
    pool_names         : 선택된 base 모델 이름들
    params             : 메타 학습기 하이퍼파라미터 (alpha/iso_weight/zero_tau 등)
    meta_cv_oof_rmse   : GroupKFold OOF로 평가한 unit-level RMSE (refit 단계만 채워짐, 외엔 NaN)
    aggregation        : 채택된 die→unit 집계 방식 ('mean', 'median', ..., 'weighted')
    pi_threshold       : ZITboost π gating 채택값 (안 쓰면 None)
    zero_clip          : zero_clip 채택값
    pos_weights        : weighted 집계가 채택됐을 때 position 가중치 [w1, w2, w3, w4]
    """
    tag: str
    stage: str
    method: str
    n_base: int

    val_rmse: float
    test_rmse: float
    oof_rmse: float

    val_rmse_die: float
    test_rmse_die: float
    oof_rmse_die: float

    pool_names: list[str]
    params: dict

    meta_cv_oof_rmse: float = float("nan")
    """**unit-level** GroupKFold OOF RMSE — record의 method 그대로 fold-fit/predict하여 계산.
    refit 단계에서만 채워짐 (fast/seed/local/optuna record는 NaN).
    `select_by="meta_cv_oof"` 시 가장 정직한 selection 기준. NaN인 record는 자동 제외.
    """

    aggregation: Optional[str] = None
    pi_threshold: Optional[float] = None
    zero_clip: Optional[float] = None
    pos_weights: Optional[list[float]] = None

    decisions: dict = field(default_factory=dict)
    """tune_and_apply 단계별 채택/거부 사유 (사람이 읽기 위한 메시지)."""

    # ─── v4 SHAP X-stacking (always_include / searchable 모두 박제) ───
    n_extra: int = 0
    """이 record가 사용한 SHAP 컬럼 수.
    - always_include 모드: subset에 안 들어가지만 메타 학습 시 항상 추가됨 → record마다 동일 값.
    - searchable 모드: pool_names 안에 SHAP 컬럼이 섞여 있을 수 있음 → 그 중 개수.
    """

    extra_tags: list[str] = field(default_factory=list)
    """이 record가 사용한 SHAP 캐시 폴더명 목록 (재현/추적용)."""


def score_rec(rec: Record, select_by: str, val_gap_penalty: float = 0.0) -> float:
    """탐색/선정 기준 점수. 작을수록 좋음.

    select_by:
      - "val"          : rec.val_rmse
      - "meta_cv_oof"  : rec.meta_cv_oof_rmse — **NaN이면 +inf** (select 후보에서 자동 제외).
                         fast/seed/local/optuna stage record는 mcv를 계산하지 않으므로 NaN →
                         meta_cv_oof 기준 selection 시 refit stage record만 비교 대상이 됨 (의도된 동작).
      - "oof"          : rec.oof_rmse + λ·max(0, val-oof)   (λ=val_gap_penalty)
    """
    if select_by == "val":
        return rec.val_rmse
    if select_by == "meta_cv_oof":
        if not math.isnan(rec.meta_cv_oof_rmse):
            return rec.meta_cv_oof_rmse
        return float("inf")  # mcv 없는 record는 비교에서 자동 제외
    # default: oof + optional val-gap penalty
    score = rec.oof_rmse
    if val_gap_penalty > 0 and not math.isnan(rec.val_rmse):
        score += val_gap_penalty * max(0.0, rec.val_rmse - rec.oof_rmse)
    return score
