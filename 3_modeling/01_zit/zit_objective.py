"""
01_zit/zit_objective.py — ZIT 4조합 공용 objective 빌더.

4조합(zit_only / bag × pearson / eql)은 unit-CV objective의 fold 루프가 동일하고, 차이는
딱 3가지뿐:
  - model_class : ZITboostRegressor / ZITboostEQLRegressor / BagZITboostRegressor / BagZITEQLRegressor
  - die→unit 집계 : zit_only=mean, bag=sum
  - fit 시그니처 : bag은 unit_id= 전달, zit_only는 미전달

ZIT 고유 로직(`predict_components` → `(1-π)·μ` clip → `tau_pi` 구조적-0 게이트 → die→unit)을
**여기 한 곳에 명시적으로** 둔다. 각 조합 hpo.py는 위 3가지(+search space)만 선언하는 thin
config가 된다 (refactor_strategy.md §1.1: 하네스는 트랙-무관 parallel_hpo, objective는 트랙
단위로 명시적 — 트랙 내부 변형은 이 빌더를 공유).

병렬 하네스(인자/study/mmap/optimize)는 modules.parallel_hpo가 담당한다.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "setup.py").exists() and (p / "utils").exists():
            return p
    raise RuntimeError(f"Project root not found from {start}")


_ROOT = _find_project_root(Path(__file__).resolve())
for _p in [_ROOT, _ROOT / "3_modeling"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import optuna  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

from modules import hpo as _hpo  # noqa: E402  (sample_from_space)
from utils.config import SEED  # noqa: E402


def _agg_die_to_unit(pred_die: np.ndarray, uid_die: np.ndarray, how: str) -> pd.Series:
    """die 예측을 unit 단위로 집계. how='mean'(zit_only) | 'sum'(bag)."""
    g = pd.DataFrame({"uid": uid_die, "p": pred_die}).groupby("uid", sort=False)["p"]
    return g.sum() if how == "sum" else g.mean()


def make_zit_objective(args, data, *, model_class, search, tau_range, use_unit_id, agg):
    """ZIT unit-CV objective(unit RMSE) 빌더.

    Parameters
    ----------
    args   : parallel_hpo.add_common_args 로 채운 Namespace (n_folds, n_jobs, worker_id 등)
    data   : parallel_hpo.load_pp_mmap 반환 (x, uid_die, y_unit_s, y_die, feat)
    model_class : ZIT 계열 클래스 (4조합 중 하나)
    search : sample_from_space 형식 search space dict
    tau_range : (low, high) — tau_pi(구조적-0 게이트) 탐색 범위
    use_unit_id : bag 이면 True (fit에 unit_id 전달)
    agg : 'mean'(zit_only) | 'sum'(bag)
    """
    x_train, uid_train_die, y_train_unit_s, y_train_die, _ = data
    unique_units = y_train_unit_s.index.values
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)
    folds = list(kf.split(unique_units))
    print(f"[fold split] n_folds={args.n_folds}, seed={SEED}, agg={agg}, use_unit_id={use_unit_id}")

    def objective(trial: optuna.Trial) -> float:
        t0 = time.time()
        params = _hpo.sample_from_space(trial, search)
        tau_pi = trial.suggest_float("tau_pi", tau_range[0], tau_range[1])

        # KFold split은 SEED 고정(모든 trial 동일 fold) → RMSE 비교 가능. 모델 seed만 trial별로.
        model_seed = int(SEED) + int(trial.number)
        params["random_state"] = model_seed
        params["n_jobs"] = args.n_jobs
        params["verbose"] = -1
        params["device"] = "cpu"
        params["em_tol"] = 1e-7

        trial.set_user_attr("worker_id", args.worker_id)
        trial.set_user_attr("tau_pi", tau_pi)
        trial.set_user_attr("model_seed", model_seed)

        fold_oof_rmse = []
        oof_pred_unit = pd.Series(np.nan, index=y_train_unit_s.index, dtype=np.float64)

        for fold_idx, (tr_uidx, vl_uidx) in enumerate(folds):
            tr_units = unique_units[tr_uidx]
            vl_units = unique_units[vl_uidx]
            tr_mask = np.isin(uid_train_die, tr_units)
            vl_mask = np.isin(uid_train_die, vl_units)

            model = model_class(**params)
            if use_unit_id:
                model.fit(x_train[tr_mask], y_train_die[tr_mask], unit_id=uid_train_die[tr_mask])
            else:
                model.fit(x_train[tr_mask], y_train_die[tr_mask])

            pi_vl, mu_vl, _ = model.predict_components(x_train[vl_mask])
            pred_die = np.clip((1.0 - pi_vl) * mu_vl, 0.0, None)
            pred_die = np.where(pi_vl > tau_pi, 0.0, pred_die)   # 구조적-0 게이트
            unit_pred = _agg_die_to_unit(pred_die, uid_train_die[vl_mask], agg)

            oof_pred_unit.loc[unit_pred.index] = unit_pred.values
            y_vl = y_train_unit_s.loc[unit_pred.index].values
            fold_oof_rmse.append(float(np.sqrt(np.mean((unit_pred.values - y_vl) ** 2))))

            avg = float(np.mean(fold_oof_rmse))
            trial.report(avg, step=fold_idx)
            if trial.should_prune():
                trial.set_user_attr("pruned_at_fold", fold_idx + 1)
                trial.set_user_attr("elapsed_sec", time.time() - t0)
                trial.set_user_attr("fold_oof_rmse", fold_oof_rmse)
                trial.set_user_attr("partial_val_rmse", avg)
                raise optuna.TrialPruned()

        if oof_pred_unit.isna().any():
            raise RuntimeError("OOF NaN: missing fold predictions")

        oof_rmse = float(np.sqrt(np.mean((oof_pred_unit.values - y_train_unit_s.values) ** 2)))
        elapsed = time.time() - t0
        trial.set_user_attr("elapsed_sec", elapsed)
        trial.set_user_attr("fold_oof_rmse", fold_oof_rmse)
        trial.set_user_attr("val_rmse", oof_rmse)
        trial.set_user_attr("oof_rmse", oof_rmse)
        print(
            f"trial #{trial.number}: worker={args.worker_id}, "
            f"tau_pi={tau_pi:.4f}, oof={oof_rmse:.9f}, elapsed={elapsed:.0f}s"
        )
        return oof_rmse

    return objective
