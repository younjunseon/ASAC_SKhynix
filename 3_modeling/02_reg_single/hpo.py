"""
02_reg_single/hpo.py — 단일 회귀모델 병렬 HPO 튜너 (--model 5종 분기, thin).

modules.parallel_hpo 하네스(워커 N개가 1 study + pp.npy mmap 공유) + modules.models 팩토리
(get_search_space / create_regressor). objective(fit/predict → die→unit mean → unit RMSE,
enet은 fold-local RobustScaler)는 이 파일에 명시적으로 둔다 (refactor_strategy.md §1.1).
탐색 공간은 models.SEARCH_SPACES의 넓은 범위를 그대로 쓴다.

실행 (PowerShell). 워커 3개 권장(3 × n_jobs ≤ 물리 스레드):
  python 3_modeling/01_zit/precompute_pp.py                              # 1회: pp.npy(전 트랙 공용)
  python 3_modeling/02_reg_single/hpo.py --model lgbm --worker-id w1 --n-jobs 4 --end-at 2026-06-20T05:00 > w1.log 2>&1
  python 3_modeling/02_reg_single/hpo.py --model lgbm --worker-id w2 --n-jobs 4 --end-at 2026-06-20T05:00 > w2.log 2>&1
  python 3_modeling/02_reg_single/hpo.py --model lgbm --worker-id w3 --n-jobs 4 --end-at 2026-06-20T05:00 > w3.log 2>&1
  # --model {lgbm,xgb,catboost,et,enet} — 모델마다 별도 study(reg_{model}).
"""
from __future__ import annotations

import argparse
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

from modules import models  # noqa: E402  (get_search_space / create_regressor)
from modules import parallel_hpo as ph  # noqa: E402
from utils.config import SEED  # noqa: E402


REG_MODELS = ["lgbm", "xgb", "catboost", "et", "enet"]
# 워커 스레드 캡을 위해 모델별 thread 파라미터 이름. enet은 스레드 인자 없음.
_THREAD_PARAM = {"lgbm": "n_jobs", "xgb": "n_jobs", "et": "n_jobs", "catboost": "thread_count"}


def build_objective(args, data):
    """reg unit-CV objective — fit/predict → die→unit mean → unit RMSE. enet은 fold-local RobustScaler."""
    X_train, uid_train_die, y_train_unit_s, y_train_die, _ = data
    unique_units = y_train_unit_s.index.values
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)
    folds = list(kf.split(unique_units))
    space_fn = models.get_search_space(args.model)
    needs_scaling = args.model == "enet"   # enet만 스케일링 (트리 4종은 pass-through)
    print(f"[fold split] n_folds={args.n_folds}, model={args.model}, scaling={needs_scaling}")

    def objective(trial: optuna.Trial) -> float:
        t0 = time.time()
        params = space_fn(trial)
        tp = _THREAD_PARAM.get(args.model)
        if tp:
            params[tp] = args.n_jobs   # space 함수의 n_jobs=-1 등을 워커 스레드로 캡
        trial.set_user_attr("worker_id", args.worker_id)

        fold_oof_rmse = []
        oof_pred_unit = pd.Series(np.nan, index=y_train_unit_s.index, dtype=np.float64)

        for fold_idx, (tr_uidx, vl_uidx) in enumerate(folds):
            tr_units = unique_units[tr_uidx]
            vl_units = unique_units[vl_uidx]
            tr_mask = np.isin(uid_train_die, tr_units)
            vl_mask = np.isin(uid_train_die, vl_units)

            X_tr, y_tr = X_train[tr_mask], y_train_die[tr_mask]
            X_vl = X_train[vl_mask]

            if needs_scaling:
                # enet: 이 fold train 기준 RobustScaler(median/IQR)를 fold holdout에 동일 적용.
                med = np.median(X_tr, axis=0)
                q75 = np.quantile(X_tr, 0.75, axis=0)
                q25 = np.quantile(X_tr, 0.25, axis=0)
                iqr = np.maximum(q75 - q25, 1e-8)
                X_tr = (X_tr - med) / iqr
                X_vl = (X_vl - med) / iqr

            model = models.create_regressor(args.model, params)
            model.fit(X_tr, y_tr)
            pred_die = np.clip(model.predict(X_vl), 0.0, None)

            unit_pred = (
                pd.DataFrame({"uid": uid_train_die[vl_mask], "p": pred_die})
                .groupby("uid", sort=False)["p"].mean()
            )
            oof_pred_unit.loc[unit_pred.index] = unit_pred.values
            y_vl = y_train_unit_s.loc[unit_pred.index].values
            fold_oof_rmse.append(float(np.sqrt(np.mean((unit_pred.values - y_vl) ** 2))))

            avg = float(np.mean(fold_oof_rmse))
            trial.report(avg, step=fold_idx)
            if trial.should_prune():
                trial.set_user_attr("pruned_at_fold", fold_idx + 1)
                trial.set_user_attr("elapsed_sec", time.time() - t0)
                trial.set_user_attr("fold_oof_rmse", fold_oof_rmse)
                raise optuna.TrialPruned()

        if oof_pred_unit.isna().any():
            raise RuntimeError("OOF NaN: missing fold predictions")

        oof_rmse = float(np.sqrt(np.mean((oof_pred_unit.values - y_train_unit_s.values) ** 2)))
        elapsed = time.time() - t0
        trial.set_user_attr("elapsed_sec", elapsed)
        trial.set_user_attr("fold_oof_rmse", fold_oof_rmse)
        trial.set_user_attr("oof_rmse", oof_rmse)
        trial.set_user_attr("val_rmse", oof_rmse)
        print(f"trial #{trial.number}: worker={args.worker_id}, model={args.model}, "
              f"oof={oof_rmse:.9f}, elapsed={elapsed:.0f}s")
        return oof_rmse

    return objective


def main() -> None:
    ap = argparse.ArgumentParser(description="reg_single parallel HPO worker (--model 5종).")
    ap.add_argument("--model", required=True, choices=REG_MODELS)
    ph.add_common_args(ap, default_exp_id="__auto__", default_n_jobs=4, default_n_startup=80)
    args = ap.parse_args()
    if args.exp_id == "__auto__":
        args.exp_id = f"reg_{args.model}"   # 모델별 study (예: reg_lgbm)

    data = ph.load_pp_mmap(args.precomputed_dir)
    out_dir = ph.resolve_out_dir(f"02_reg_single/{args.model}")
    db_path = ph.study_db_path(out_dir, args.user, args.exp_id)
    study = ph.make_study(args, db_path, study_meta={
        "model": args.model,
        "track": "02_reg_single",
        "out_subdir": f"02_reg_single/{args.model}",
        "scaling": "fold-local RobustScaler" if args.model == "enet" else "none",
    })
    print(f"MODEL={args.model}\nOUT_DIR={out_dir}\nDB={db_path}")
    ph.run_optimize(study, build_objective(args, data), args)


if __name__ == "__main__":
    main()
