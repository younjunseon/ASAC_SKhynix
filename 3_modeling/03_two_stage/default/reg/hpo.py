"""
03_two_stage/default/reg/hpo.py — Two-Stage Stage2 회귀 병렬 HPO 튜너 (--model 5종, thin).

02_reg_single/hpo.py와 ~90% 동일하되 **Y>0 only fit**가 차이 (Stage 2 = "Y>0만으로 학습 →
E[Y|Y>0,x] 예측"). 폴더 격리 우선 원칙(refactor_strategy.md §4)에 따라 objective는 이 파일에
명시적으로 둔다(02_reg와 중복 허용). fit_mask만 `tr_mask & (y_die>0)`로 좁히고, 예측·집계·RMSE는
전체 unit 기준(02_reg와 동일) — combine 단계에서 clf와 곱해 최종 Two-Stage 예측을 만든다.
탐색 공간은 models.SEARCH_SPACES의 넓은 범위를 그대로 쓴다.

실행 (워커 3개 권장):
  python 3_modeling/01_zit/precompute_pp.py                                       # 1회: pp.npy
  python 3_modeling/03_two_stage/default/reg/hpo.py --model lgbm --worker-id w1 --n-jobs 4 --end-at 2026-06-20T05:00 > w1.log 2>&1
  # --model {lgbm,xgb,catboost,et,enet} — 모델마다 별도 study(ts_reg_{model}).
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

from modules import models  # noqa: E402
from modules import parallel_hpo as ph  # noqa: E402
from utils.config import SEED  # noqa: E402


REG_MODELS = ["lgbm", "xgb", "catboost", "et", "enet"]
_THREAD_PARAM = {"lgbm": "n_jobs", "xgb": "n_jobs", "et": "n_jobs", "catboost": "thread_count"}


def build_objective(args, data):
    """ts/reg unit-CV objective — Y>0 only fit, 전체 vl die 예측 → unit mean → 전체 unit RMSE.

    enet은 fold-local RobustScaler (Y>0 fit 데이터 기준 통계량).
    """
    X_train, uid_train_die, y_train_unit_s, y_train_die, _ = data
    unique_units = y_train_unit_s.index.values
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)
    folds = list(kf.split(unique_units))
    space_fn = models.get_search_space(args.model)
    needs_scaling = args.model == "enet"
    print(f"[fold split] n_folds={args.n_folds}, model={args.model}, Y>0-only fit, scaling={needs_scaling}")

    def objective(trial: optuna.Trial) -> float:
        t0 = time.time()
        params = space_fn(trial)
        tp = _THREAD_PARAM.get(args.model)
        if tp:
            params[tp] = args.n_jobs
        trial.set_user_attr("worker_id", args.worker_id)

        fold_oof_rmse = []
        oof_pred_unit = pd.Series(np.nan, index=y_train_unit_s.index, dtype=np.float64)

        for fold_idx, (tr_uidx, vl_uidx) in enumerate(folds):
            tr_units = unique_units[tr_uidx]
            vl_units = unique_units[vl_uidx]
            tr_mask = np.isin(uid_train_die, tr_units)
            vl_mask = np.isin(uid_train_die, vl_units)

            # Stage 2: y>0 die만 학습 (E[Y|Y>0,x]). 예측은 전체 vl die.
            fit_mask = tr_mask & (y_train_die > 0)
            X_tr, y_tr = X_train[fit_mask], y_train_die[fit_mask]
            X_vl = X_train[vl_mask]

            if needs_scaling:
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
    ap = argparse.ArgumentParser(description="Two-Stage Stage2 reg parallel HPO worker (Y>0 only, --model 5종).")
    ap.add_argument("--model", required=True, choices=REG_MODELS)
    ph.add_common_args(ap, default_exp_id="__auto__", default_n_jobs=4, default_n_startup=80)
    args = ap.parse_args()
    if args.exp_id == "__auto__":
        args.exp_id = f"ts_reg_{args.model}"

    data = ph.load_pp_mmap(args.precomputed_dir)
    out_dir = ph.resolve_out_dir(f"03_two_stage/default/reg/{args.model}")
    db_path = ph.study_db_path(out_dir, args.user, args.exp_id)
    study = ph.make_study(args, db_path, study_meta={
        "model": args.model,
        "track": "03_two_stage/default/reg",
        "out_subdir": f"03_two_stage/default/reg/{args.model}",
        "y_positive_only": True,
        "scaling": "fold-local RobustScaler" if args.model == "enet" else "none",
    })
    print(f"MODEL={args.model} (Y>0 only)\nOUT_DIR={out_dir}\nDB={db_path}")
    ph.run_optimize(study, build_objective(args, data), args)


if __name__ == "__main__":
    main()
