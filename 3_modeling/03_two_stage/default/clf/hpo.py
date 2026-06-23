"""
03_two_stage/default/clf/hpo.py — Two-Stage Stage1 분류 병렬 HPO 튜너 (--model 4종, thin).

modules.parallel_hpo 하네스 + modules.models 분류 팩토리(get_clf_search_space / create_classifier
/ resolve_clf_imbalance). objective는 die-level binary(y>0) 확률을 unit으로 mean 집계 후
`proba × y_pos_const(E[Y|Y>0])`로 health 스케일 변환 → unit RMSE. (combine 단계에서 reg와 결합)
탐색 공간은 models.CLF_SEARCH_SPACES의 넓은 범위를 그대로 쓴다.

실행 (워커 3개 권장):
  python 3_modeling/01_zit/precompute_pp.py                                       # 1회: pp.npy
  python 3_modeling/03_two_stage/default/clf/hpo.py --model lgbm --worker-id w1 --n-jobs 4 --end-at 2026-06-20T05:00 > w1.log 2>&1
  # --model {lgbm,xgb,catboost,et} — 모델마다 별도 study(ts_clf_{model}).
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


CLF_MODELS = ["lgbm", "xgb", "catboost", "et"]
_THREAD_PARAM = {"lgbm": "n_jobs", "xgb": "n_jobs", "et": "n_jobs", "catboost": "thread_count"}


def build_objective(args, data):
    """clf unit-CV objective — die-level P(y>0) → unit mean → ×y_pos_const → unit RMSE."""
    X_train, uid_train_die, y_train_unit_s, y_train_die, _ = data
    y_die_bin = (y_train_die > 0).astype(np.int8)
    y_pos_const = float(y_train_unit_s[y_train_unit_s > 0].mean())   # E[Y|Y>0]
    unique_units = y_train_unit_s.index.values
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)
    folds = list(kf.split(unique_units))
    space_fn = models.get_clf_search_space(args.model)
    print(f"[fold split] n_folds={args.n_folds}, model={args.model}, y_pos_const={y_pos_const:.6f}")

    def objective(trial: optuna.Trial) -> float:
        t0 = time.time()
        params = space_fn(trial)
        tp = _THREAD_PARAM.get(args.model)
        if tp:
            params[tp] = args.n_jobs
        trial.set_user_attr("worker_id", args.worker_id)
        trial.set_user_attr("y_pos_const", y_pos_const)   # fit/stacking이 prob→health 변환에 사용

        fold_oof_rmse = []
        oof_pred_unit = pd.Series(np.nan, index=y_train_unit_s.index, dtype=np.float64)

        for fold_idx, (tr_uidx, vl_uidx) in enumerate(folds):
            tr_units = unique_units[tr_uidx]
            vl_units = unique_units[vl_uidx]
            tr_mask = np.isin(uid_train_die, tr_units)
            vl_mask = np.isin(uid_train_die, vl_units)

            # imbalance 옵션을 이 fold 클래스 비율로 보강 (search space 값이 있으면 그것이 우선).
            fold_params = models.resolve_clf_imbalance(args.model, params, y_die_bin[tr_mask])
            clf = models.create_classifier(args.model, fold_params)
            clf.fit(X_train[tr_mask], y_die_bin[tr_mask])
            proba = clf.predict_proba(X_train[vl_mask])[:, 1]
            if np.isnan(proba).any():
                trial.set_user_attr("nan_at_fold", fold_idx + 1)
                raise optuna.TrialPruned()

            unit_proba = (
                pd.DataFrame({"uid": uid_train_die[vl_mask], "p": proba})
                .groupby("uid", sort=False)["p"].mean()
            )
            unit_pred = unit_proba * y_pos_const
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
    ap = argparse.ArgumentParser(description="Two-Stage Stage1 clf parallel HPO worker (--model 4종).")
    ap.add_argument("--model", required=True, choices=CLF_MODELS)
    ph.add_common_args(ap, default_exp_id="__auto__", default_n_jobs=4, default_n_startup=80)
    args = ap.parse_args()
    if args.exp_id == "__auto__":
        args.exp_id = f"ts_clf_{args.model}"

    data = ph.load_pp_mmap(args.precomputed_dir)
    out_dir = ph.resolve_out_dir(f"03_two_stage/default/clf/{args.model}")
    db_path = ph.study_db_path(out_dir, args.user, args.exp_id)
    study = ph.make_study(args, db_path, study_meta={
        "model": args.model,
        "track": "03_two_stage/default/clf",
        "out_subdir": f"03_two_stage/default/clf/{args.model}",
        "task": "binary (y>0)",
    })
    print(f"MODEL={args.model}\nOUT_DIR={out_dir}\nDB={db_path}")
    ph.run_optimize(study, build_objective(args, data), args)


if __name__ == "__main__":
    main()
