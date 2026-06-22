"""die→unit 집계 — postprocess.tune_and_apply 호출 래퍼.

v3 stacking의 마지막 단계: die-level meta 예측을 unit 단위로 집계.
postprocess.tune_and_apply가 이미 다음을 다 처리한다:
  1. 8종 집계 (mean/median/max/min/trimmed_mean/weighted/Q25/Q75) 중 train OOF best 후보 선정
  2. val로 검증 → val 개선 시만 채택, 아니면 baseline_agg(mean) fallback
  3. zero_clip threshold 탐색 (val 개선 시만 채택)
  4. (옵션) π threshold gating — base 모델의 die-level π를 사용. stacking에서는 보통 OFF.

이 래퍼는 다음을 추가로 해준다:
  - die-level pred에 대한 die-level RMSE (집계 전 진단치) 계산
  - aggregate 결과의 unit-level RMSE를 oof/val/test 모두 보고
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .config import StackingConfig
from .meta import rmse

KEY_COL = "ufs_serial"
TARGET_COL = "health"


def build_unit_mean_aggregator(key_df: pd.DataFrame):
    """fast eval용 — die-level pred을 unit mean으로 1ms 이하 집계하는 closure 반환.

    내부적으로 np.bincount를 써서 매번 pandas.groupby를 안 부른다.
    탐색 단계(random/local)에서 수천 번 호출되므로 효율이 중요.

    Returns
    -------
    aggregate(die_pred) -> (unit_pred, unique_units)
        unit_pred: (N_unit,) ndarray
        unique_units: (N_unit,) ndarray — np.unique 순서 (정렬됨)
    y_to_unit(y_die)  : die-level y → unit y 평균 (검증용. broadcast된 y라면 그냥 unit 값과 동일)
    unique_units      : 위와 동일, 외부에서 캐싱용
    """
    keys = key_df[KEY_COL].values
    unique_units, unit_idx = np.unique(keys, return_inverse=True)
    counts = np.bincount(unit_idx).astype(float)

    def aggregate(die_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sums = np.bincount(unit_idx, weights=np.asarray(die_pred, dtype=float))
        return sums / counts, unique_units

    return aggregate, unique_units


def align_unit_y(unique_units: np.ndarray, y_unit: pd.DataFrame) -> np.ndarray:
    """unique_units 순서에 맞춰 y_unit DataFrame을 1D array로 정렬.

    fast eval에서 단일 unit RMSE를 빠르게 계산하기 위함.
    """
    return y_unit.set_index(KEY_COL)[TARGET_COL].loc[unique_units].values.astype(float)


def _ensure_postprocess_on_path(project_root: Path) -> None:
    """3_modeling/modules가 sys.path에 있어야 from modules.postprocess import ... 가능.

    노트북에서는 이미 setup.py가 처리하지만, 단독 호출 시도 대비.
    """
    cand = project_root / "3_modeling"
    if str(cand) not in sys.path:
        sys.path.insert(0, str(cand))


def aggregate_die_to_unit(
    cfg: StackingConfig,
    key_oof: pd.DataFrame,
    key_val: pd.DataFrame,
    key_test: pd.DataFrame,
    die_pred_oof: np.ndarray,
    die_pred_val: np.ndarray,
    die_pred_test: np.ndarray,
    y_unit_oof: pd.DataFrame,
    y_unit_val: pd.DataFrame | None,
    die_pi_oof: np.ndarray | None = None,
    die_pi_val: np.ndarray | None = None,
    die_pi_test: np.ndarray | None = None,
) -> dict:
    """tune_and_apply 호출 + die-level/unit-level RMSE 함께 반환.

    Parameters
    ----------
    key_oof, key_val, key_test : DataFrame (N_die, ≥1)
        최소 [ufs_serial] 컬럼 — postprocess.aggregate가 ufs_serial로 groupby.
        position 컬럼은 weighted 집계에서 사용 (있으면 pivot, 없으면 skip).
    die_pred_oof/val/test : (N_die,) — die-level meta 예측
    y_unit_oof : DataFrame [ufs_serial, health] — 집계 best 선정 기준
    y_unit_val : DataFrame [ufs_serial, health] | None
        주어지면 "val 개선 시만 채택" 모드. None이면 train OOF best 무조건 채택 (backward-compat).
    die_pi_*  : (N_die,) | None — ZIT π를 stacking에 쓰는 경우만 채움. 보통 None.

    Returns
    -------
    dict 합성:
      tune_and_apply 결과 (best_agg, val_rmse_final, ... ) +
      die-level RMSE (oof_rmse_die, val_rmse_die, test_rmse_die)
    """
    _ensure_postprocess_on_path(cfg.project_root)
    from modules import postprocess  # type: ignore

    # weighted 집계는 position 컬럼이 있어야 동작 — 없으면 자동 제외.
    agg_methods = list(cfg.agg_methods)
    if "position" not in key_oof.columns and "weighted" in agg_methods:
        agg_methods = [m for m in agg_methods if m != "weighted"]

    # postprocess는 xs로 (DataFrame, KEY_COL 포함, position 컬럼 있으면 weighted 가능)을 받음.
    # die_pred는 xs 행 순서와 동일 길이의 1D array.
    res = postprocess.tune_and_apply(
        xs_train=key_oof, xs_val=key_val, xs_test=key_test,
        die_pred_train=die_pred_oof,
        die_pred_val=die_pred_val,
        die_pred_test=die_pred_test,
        y_train_unit=y_unit_oof,
        y_val_unit=y_unit_val,
        die_pi_train=die_pi_oof,
        die_pi_val=die_pi_val,
        die_pi_test=die_pi_test,
        agg_methods=agg_methods,
        baseline_agg=cfg.baseline_agg,
        position_method=cfg.position_method,
        position_optuna_n_trials=cfg.position_optuna_n_trials,
        use_pi_threshold=cfg.use_pi_threshold,
        zero_clip_range=cfg.zero_clip_range,
        zero_clip_step=cfg.zero_clip_step,
        zero_clip_log_space=cfg.zero_clip_log_space,
    )

    # die-level RMSE — y_unit을 die-level로 broadcast해서 비교 (진단용)
    y_oof_map = y_unit_oof.set_index(KEY_COL)[TARGET_COL]
    y_die_oof = key_oof[KEY_COL].map(y_oof_map).values.astype(float)
    res["oof_rmse_die"] = rmse(die_pred_oof, y_die_oof)

    if y_unit_val is not None:
        y_val_map = y_unit_val.set_index(KEY_COL)[TARGET_COL]
        y_die_val = key_val[KEY_COL].map(y_val_map).values.astype(float)
        res["val_rmse_die"] = rmse(die_pred_val, y_die_val)
    else:
        res["val_rmse_die"] = float("nan")

    # test die-level RMSE: tune_and_apply가 final_test_unit을 반환하므로 거기서 unit RMSE 추출.
    # die-level test RMSE는 y_unit_test가 따로 와야 계산 가능 → 호출자가 unit-level test y를
    # 가지고 있을 때 추가로 계산. 여기선 NaN으로 둠.
    res["test_rmse_die"] = float("nan")

    return res
