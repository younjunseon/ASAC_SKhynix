"""
Parallel HPO worker for reg_single/XGBoost hp/002.

reg_xgb has only hp/001 prior (49 trials) -> hp/002 is the first "serious" HPO sweep,
with a deliberately wide search around hp/001's best to actually find the optimum.
1주일 budget + processed_trials so far being small = biggest improvement potential
across all 4 base models (see stacking analysis).

Same execution model as 01_zit/01_zit_only_parallel_hpo.py:
- Multiple worker processes share one Optuna SQLite study.
- Each worker only writes Optuna DB records (no refit / no artifact export).
- Final refit + postprocess must be run separately from a single process.

Assigned PC for 1-week run: PC4 (dedicated to XGBoost regressor only — highest ROI).

Run command (PowerShell on the assigned PC):

  # Step 1 — one-time init in any terminal. Enqueues hp/001 best anchor, then exits.
  python 3_modeling/02_reg_single/02_reg_xgb_parallel_hpo.py --enqueue-anchor --n-trials 0

  # Step 2 — open 3 separate terminals on the same PC, paste one of these per terminal.
  # Tune --n-jobs so 3 * n_jobs <= physical thread count:
  #   16-thread PC -> --n-jobs 5
  #   12-thread PC -> --n-jobs 4
  #    8-thread PC -> --n-jobs 2
  # "> wN.log 2>&1" redirects stdout+stderr (works in both cmd.exe and PowerShell).
  python 3_modeling/02_reg_single/02_reg_xgb_parallel_hpo.py --worker-id w1 --n-trials 3000 --n-jobs 4 --end-at 2026-06-01T05:00 > w1.log 2>&1
  python 3_modeling/02_reg_single/02_reg_xgb_parallel_hpo.py --worker-id w2 --n-trials 3000 --n-jobs 4 --end-at 2026-06-01T05:00 > w2.log 2>&1
  python 3_modeling/02_reg_single/02_reg_xgb_parallel_hpo.py --worker-id w3 --n-trials 3000 --n-jobs 4 --end-at 2026-06-01T05:00 > w3.log 2>&1

Objective:
  unit RMSE of mean(die_pred -> unit). Identical recipe to hpo.run_hpo (model='xgb').
  objective is a categorical -- tweedie variants are protected from NaN by max_delta_step=0.7.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "setup.py").exists() and (p / "utils").exists():
            return p
    raise RuntimeError(f"Project root not found from {start}")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PP_DIR = PROJECT_ROOT / "2_preprocessing"
MOD_DIR = PROJECT_ROOT / "3_modeling"
for p in [PP_DIR, MOD_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import optuna  # noqa: E402
import xgboost as xgb  # noqa: E402
from optuna.pruners import MedianPruner  # noqa: E402
from optuna.samplers import TPESampler  # noqa: E402
from optuna.storages import RDBStorage, RetryFailedTrialCallback  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

from meta_features import add_meta_features  # noqa: E402
from modules import preprocess  # noqa: E402
from utils.config import KEY_COL, OUTPUT_DIR, SEED, TARGET_COL  # noqa: E402
from utils.data import get_feat_cols, load_all, split_xs  # noqa: E402


DEFAULT_EXP_ID = "reg-xgb-002"
DEFAULT_USER = "jh"

PP_FIXED = {
    "missing_threshold": 0.30,
    "corr_threshold": 0.90,
    "corr_keep_by": "std",
    "add_indicator": True,
    "indicator_threshold": 0.05,
    "spatial_max_dist": 6.0,
    "post_impute_corr_threshold": 0.96,
    "post_impute_corr_keep_by": "std",
}

# hp/001 best trial #25 (OOF=0.005525). Anchor is enqueued only with --enqueue-anchor.
REG_XGB_ANCHOR = {
    "n_estimators": 1033,
    "learning_rate": 0.02647002735388861,
    "max_depth": 7,
    "min_child_weight": 0.8791483836653062,
    "subsample": 0.8651727478077347,
    "colsample_bytree": 0.7616838302884501,
    "reg_alpha": 0.01081652598322564,
    "reg_lambda": 2.7752856740325894e-06,
    "gamma": 2.4037262078877586e-06,
    "objective": "count:poisson",
}

# hp/002 search space: wide because hp/001 only ran 49 trials -> best is uncertain.
# Centered on hp/001 best but generous around every axis. objective remains a
# 3-way categorical (squarederror / count:poisson / reg:tweedie) so TPE can compare losses.
# When objective=='reg:tweedie' the tweedie_variance_power is also sampled.
REG_XGB_SEARCH = {
    "n_estimators":     {"type": "int",   "low": 400,    "high": 2500},
    "learning_rate":    {"type": "float", "low": 0.008,  "high": 0.08,  "log": True},
    "max_depth":        {"type": "int",   "low": 4,      "high": 13},
    "min_child_weight": {"type": "float", "low": 0.2,    "high": 4.0,   "log": True},
    "subsample":        {"type": "float", "low": 0.55,   "high": 1.0,   "log": False},
    "colsample_bytree": {"type": "float", "low": 0.40,   "high": 1.0,   "log": False},
    "reg_alpha":        {"type": "float", "low": 1e-5,   "high": 1.0,   "log": True},
    "reg_lambda":       {"type": "float", "low": 1e-8,   "high": 1e-2,  "log": True},
    "gamma":            {"type": "float", "low": 1e-8,   "high": 1e-2,  "log": True},
    "objective":        {"type": "cat",   "choices": ["reg:squarederror", "count:poisson", "reg:tweedie"]},
    # tweedie_variance_power is conditional (only used when objective == "reg:tweedie").
    "tweedie_variance_power": {"type": "float", "low": 1.05, "high": 1.95, "log": False},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="reg_single/XGBoost parallel HPO worker.")
    parser.add_argument("--exp-id", default=DEFAULT_EXP_ID)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--worker-id", default=f"pid{os.getpid()}")
    parser.add_argument("--n-trials", type=int, default=2500)
    parser.add_argument("--timeout-hours", type=float, default=120.0)
    parser.add_argument(
        "--end-at",
        default=None,
        help=(
            "Absolute wall-clock end datetime in ISO format, e.g. '2026-06-01T05:00'. "
            "When set, the worker stops at this moment regardless of --timeout-hours. "
            "Pass the same value to every worker so all stop simultaneously."
        ),
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=3)
    parser.add_argument("--n-startup-trials", type=int, default=100)
    parser.add_argument("--db-timeout", type=float, default=120.0)
    parser.add_argument("--heartbeat-interval", type=int, default=300)
    parser.add_argument("--grace-period", type=int, default=1800)
    parser.add_argument("--max-retry", type=int, default=1)
    parser.add_argument("--enqueue-anchor", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-clip-y-extreme", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def sample_reg_xgb(trial: optuna.Trial) -> dict:
    """Sample HP dict + resolve conditional tweedie_variance_power axis."""
    params = dict(
        n_estimators     = trial.suggest_int("n_estimators",     REG_XGB_SEARCH["n_estimators"]["low"],     REG_XGB_SEARCH["n_estimators"]["high"]),
        learning_rate    = trial.suggest_float("learning_rate", REG_XGB_SEARCH["learning_rate"]["low"], REG_XGB_SEARCH["learning_rate"]["high"], log=True),
        max_depth        = trial.suggest_int("max_depth",        REG_XGB_SEARCH["max_depth"]["low"],        REG_XGB_SEARCH["max_depth"]["high"]),
        min_child_weight = trial.suggest_float("min_child_weight", REG_XGB_SEARCH["min_child_weight"]["low"], REG_XGB_SEARCH["min_child_weight"]["high"], log=True),
        subsample        = trial.suggest_float("subsample", REG_XGB_SEARCH["subsample"]["low"], REG_XGB_SEARCH["subsample"]["high"]),
        colsample_bytree = trial.suggest_float("colsample_bytree", REG_XGB_SEARCH["colsample_bytree"]["low"], REG_XGB_SEARCH["colsample_bytree"]["high"]),
        reg_alpha        = trial.suggest_float("reg_alpha",  REG_XGB_SEARCH["reg_alpha"]["low"],  REG_XGB_SEARCH["reg_alpha"]["high"],  log=True),
        reg_lambda       = trial.suggest_float("reg_lambda", REG_XGB_SEARCH["reg_lambda"]["low"], REG_XGB_SEARCH["reg_lambda"]["high"], log=True),
        gamma            = trial.suggest_float("gamma",      REG_XGB_SEARCH["gamma"]["low"],      REG_XGB_SEARCH["gamma"]["high"],      log=True),
    )
    obj_choice = trial.suggest_categorical("objective", REG_XGB_SEARCH["objective"]["choices"])
    if obj_choice == "reg:tweedie":
        params["objective"] = "reg:tweedie"
        params["tweedie_variance_power"] = trial.suggest_float(
            "tweedie_variance_power",
            REG_XGB_SEARCH["tweedie_variance_power"]["low"],
            REG_XGB_SEARCH["tweedie_variance_power"]["high"],
        )
        # tweedie is log-space -> exp() overflow on zero-inflated/tiny targets without a delta cap
        params["max_delta_step"] = 0.7
    else:
        params["objective"] = obj_choice
    return params


def create_study(args: argparse.Namespace, db_path: Path) -> optuna.Study:
    storage = RDBStorage(
        url=f"sqlite:///{db_path.as_posix()}",
        engine_kwargs={"connect_args": {"timeout": args.db_timeout}},
        heartbeat_interval=args.heartbeat_interval,
        grace_period=args.grace_period,
        failed_trial_callback=RetryFailedTrialCallback(max_retry=args.max_retry),
    )
    sampler = TPESampler(
        seed=None,
        multivariate=True,
        group=True,
        n_startup_trials=args.n_startup_trials,
    )
    pruner = MedianPruner(
        n_startup_trials=args.n_startup_trials,
        n_warmup_steps=2,
    )
    study = optuna.create_study(
        study_name=args.exp_id,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        load_if_exists=not args.no_resume,
    )

    if args.enqueue_anchor:
        if len(study.trials) == 0:
            study.enqueue_trial(dict(REG_XGB_ANCHOR))
            print(f"[enqueue] anchor forced as trial 0 ({len(REG_XGB_ANCHOR)} HP)")
        else:
            print(f"[enqueue skip] existing trials={len(study.trials)}")

    study_meta = {
        "exp_id": args.exp_id,
        "user": args.user,
        "model": "XGBoost regressor parallel HPO worker",
        "n_trials_per_worker": args.n_trials,
        "n_folds": args.n_folds,
        "n_jobs": args.n_jobs,
        "pp_fixed": PP_FIXED,
        "anchor_source": "hp/001 best trial #25 (OOF 0.005525)",
        "anchor": REG_XGB_ANCHOR,
        "search_space": REG_XGB_SEARCH,
        "sampler": "TPE seed=None multivariate group",
        "pruner": f"MedianPruner n_startup={args.n_startup_trials} n_warmup=2",
        "clip_y_extreme": not args.no_clip_y_extreme,
        "seed": int(SEED),
        "worker_id_last_writer": args.worker_id,
        "objective_recipe": "unit_RMSE(mean(die_pred -> unit))",
    }
    for k, v in study_meta.items():
        study.set_user_attr(k, str(v))
    return study


def load_preprocessed_data(args: argparse.Namespace):
    xs, ys = load_all()
    feat_cols = get_feat_cols(xs)
    xs_dict = split_xs(xs)

    ys_input = {k: v.copy() for k, v in ys.items()}
    if not args.no_clip_y_extreme:
        y_raw = ys_input["train"][TARGET_COL]
        second_max = y_raw[y_raw < y_raw.max()].max()
        n_clipped = int((y_raw >= 1.0).sum())
        ys_input["train"][TARGET_COL] = y_raw.clip(upper=second_max)
        print(f"[CLIP_Y_EXTREME] 1.0 -> {second_max:.6f}, n={n_clipped}")

    pp = preprocess.run(xs, ys_input, feat_cols, xs_dict, params=PP_FIXED)
    xs_train = pp["xs_train"]
    xs_val = pp["xs_val"]
    xs_test = pp["xs_test"]
    feat_cols_clean = pp["feat_cols"]

    feat_cols_clean = add_meta_features(
        xs_train, xs_val, xs_test, feat_cols_clean,
        position_mode="raw", use_die_xy=True,
    )

    x_train = xs_train[feat_cols_clean].values.astype(np.float64)
    uid_train_die = xs_train[KEY_COL].values
    y_train_unit_s = ys_input["train"].set_index(KEY_COL)[TARGET_COL]
    y_die = xs_train[KEY_COL].map(y_train_unit_s).values.astype(np.float64)

    print(f"[preprocess done] n_features={len(feat_cols_clean)}")
    print(f"  X_train={x_train.shape}, train_units={len(y_train_unit_s):,}")
    return x_train, uid_train_die, y_train_unit_s, y_die, feat_cols_clean


def build_objective(args: argparse.Namespace):
    x_train, uid_train_die, y_train_unit_s, y_die, _ = load_preprocessed_data(args)

    unique_units = y_train_unit_s.index.values
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)
    folds = list(kf.split(unique_units))
    print(f"[fold split] n_folds={args.n_folds}, seed={SEED}")

    def objective(trial: optuna.Trial) -> float:
        t0 = time.time()
        params = sample_reg_xgb(trial)
        params["random_state"] = SEED
        params["n_jobs"] = args.n_jobs
        params["tree_method"] = "hist"
        params["verbosity"] = 0

        trial.set_user_attr("worker_id", args.worker_id)
        trial.set_user_attr("pid", os.getpid())

        fold_oof_rmse = []
        oof_pred_unit = pd.Series(np.nan, index=y_train_unit_s.index, dtype=np.float64)

        for fold_idx, (tr_uidx, vl_uidx) in enumerate(folds):
            tr_units = unique_units[tr_uidx]
            vl_units = unique_units[vl_uidx]
            tr_mask = np.isin(uid_train_die, tr_units)
            vl_mask = np.isin(uid_train_die, vl_units)

            model = xgb.XGBRegressor(**params)
            model.fit(x_train[tr_mask], y_die[tr_mask])
            pred_die = model.predict(x_train[vl_mask])
            if np.isnan(pred_die).any():
                # tweedie/poisson can diverge under extreme HP -> prune this trial only
                trial.set_user_attr("nan_at_fold", fold_idx + 1)
                trial.set_user_attr("elapsed_sec", time.time() - t0)
                trial.set_user_attr("fold_oof_rmse", fold_oof_rmse)
                trial.set_user_attr("nan_objective", params.get("objective"))
                raise optuna.TrialPruned()

            df = pd.DataFrame({KEY_COL: uid_train_die[vl_mask], "p": pred_die})
            unit_pred = df.groupby(KEY_COL, sort=False)["p"].mean()
            oof_pred_unit.loc[unit_pred.index] = unit_pred.values
            y_vl = y_train_unit_s.loc[unit_pred.index].values
            fold_rmse = float(np.sqrt(np.mean((unit_pred.values - y_vl) ** 2)))
            fold_oof_rmse.append(fold_rmse)

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
        print(
            f"trial #{trial.number}: worker={args.worker_id}, "
            f"obj={params['objective']}, oof={oof_rmse:.9f}, elapsed={elapsed:.0f}s"
        )
        return oof_rmse

    return objective


def main() -> None:
    args = parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    out_dir = (
        Path(OUTPUT_DIR) / "02_reg_single" / "xgb" / args.exp_id.split("-")[-1]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / f"optuna_{args.user}_{args.exp_id}.db"

    print(f"PROJECT_ROOT={PROJECT_ROOT}")
    print(f"optuna={optuna.__version__}")
    print(f"EXP_ID={args.exp_id}, worker_id={args.worker_id}")
    print(f"OUT_DIR={out_dir}")
    print(f"DB_PATH={db_path}")
    print(f"N_TRIALS={args.n_trials}, N_JOBS={args.n_jobs}, N_FOLDS={args.n_folds}")
    print(f"search HP={len(REG_XGB_SEARCH)}, n_startup={args.n_startup_trials}")

    study = create_study(args, db_path)
    print(f"study={study.study_name}, existing_trials={len(study.trials)}")

    if args.n_trials <= 0:
        print("[skip optimize] n_trials <= 0")
        return

    if len(study.trials) == 0 and not args.enqueue_anchor:
        print("[note] no existing trials and --enqueue-anchor was not used")

    objective = build_objective(args)

    t_start = time.time()
    if args.end_at:
        end_dt = datetime.fromisoformat(args.end_at)
        remaining = (end_dt - datetime.now()).total_seconds()
        if remaining <= 0:
            print(f"[end-at] {end_dt.isoformat()} already passed — nothing to do")
            return
        timeout = remaining
        print(
            f"[end-at] target={end_dt.isoformat()} "
            f"-> timeout={timeout:.0f}s ({timeout/3600:.2f}h)"
        )
    else:
        timeout = None if args.timeout_hours <= 0 else args.timeout_hours * 3600

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=timeout,
        n_jobs=1,
        show_progress_bar=args.progress,
    )
    elapsed = time.time() - t_start
    print(
        f"[HPO done] worker={args.worker_id}, elapsed={elapsed:.0f}s, "
        f"total_trials={len(study.trials)}"
    )
    try:
        print(f"best_value={study.best_value:.9f}, best_trial={study.best_trial.number}")
    except ValueError:
        print("best_value unavailable: no completed trials")


if __name__ == "__main__":
    main()
