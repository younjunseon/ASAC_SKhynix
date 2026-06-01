"""
Parallel HPO worker for BagZIT + EQL phi update -- PRECOMPUTED-MMAP variant.

This is the EQL sibling of 02_bag_zit_mmap_parallel_hpo.py (004, base/simplified
phi). It combines the two best pieces we already have:

  * mechanism  = 02_bag_zit_mmap_parallel_hpo.py
      - BLAS/numpy thread cap (1 per process) so many workers don't oversubscribe
      - mmap-shared precomputed feature matrix (00_precompute_pp.py) -> instant start,
        ONE copy of the matrix in RAM across all workers on one PC
  * model      = BagZITEQLRegressor (imported from modules.zit -- single source of truth)
      - phi M-step target = Tweedie unit deviance D_zeta(y, mu)   (paper-matched / EQL)
      - everything else (bag/unit constraint, EM, predict) inherited from BagZIT

The EQL change is purely in the phi M-step (model internals); preprocessing is
IDENTICAL to the base worker, so the SAME precomputed pp.npy (built from
02_bag_zit_parallel_hpo's preprocessing: median-imputation patch + same PP_FIXED +
clip_y_extreme) is valid here with ZERO data change. We do not modify
3_modeling/modules/zit.py.

THREE SEARCH-SPACE VARIANTS, ONE FILE  (--variant v3 / v4 / v5)
  All three warm-start from the bag-zit-eql-final-002 BEST trial (#58) and all three
  ranges CONTAIN that anchor, so the enqueue is always valid. They differ only in how
  they explore around it:
    v3 (bag-zit-eql-final-003)  EXPLOIT  -- narrow band tightly around #58 (fine local search)
    v4 (bag-zit-eql-final-004)  PUSH     -- extend exactly the walls #58 pressed against
    v5 (bag-zit-eql-final-005)  EXPLORE  -- broad re-scan to escape any local optimum
  One PC per variant. Each variant is its own Optuna study / DB (independent search).

PREREQUISITE -- build the precomputed matrix once per PC (or copy the folder):
  python 3_modeling/01_zit/hp/00_precompute_pp.py
  -> 0_data/precomputed/bag_zit_pp/{pp.npy, units.npy, feat_cols.json, manifest.json}

Run command (PowerShell). PC1->v3, PC2->v4, PC3->v5. Deadline 2026-06-08 05:00.

  # One-time init PER VARIANT (creates the study + enqueues the 002 best anchor, then exits).
  python 3_modeling/01_zit/hp/02_bag_zit_eql_mmap_parallel_hpo.py --variant v3 --enqueue-anchor --n-trials 0

  # Then on that PC open N independent workers. 7 workers x --n-jobs 2 = 14 threads.
  # Independent OS processes share ONE mmap pp.npy and each pulls the next trial from the
  # shared SQLite DB the instant it finishes -- no "wait for the whole batch" stall.
  python 3_modeling/01_zit/hp/02_bag_zit_eql_mmap_parallel_hpo.py --variant v3 --worker-id w1 --n-trials 100000 --n-jobs 2 --end-at 2026-06-08T05:00 > 4_output/logs/eql_v3_w1.log 2>&1
  ...  (w2 .. w7, identical except --worker-id)

This worker writes Optuna DB records only. Final refit, postprocess, and artifact
export should be run separately after selecting the best trial.
"""

from __future__ import annotations

import os

# Limit per-process BLAS/numpy threads BEFORE importing numpy. Otherwise each of the N
# parallel workers spawns an all-core BLAS pool for the numpy EM steps, so N workers x
# (all cores) = massive thread oversubscription (in the 004 first run this made every
# trial ~5x slower). LightGBM's own --n-jobs only caps LightGBM threads, NOT numpy/BLAS,
# so this is needed on top of it. setdefault -> a caller can still override via the env.
for _thr_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thr_var, "1")

import argparse
import json
import logging
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

MOD_DIR = PROJECT_ROOT / "3_modeling"
if str(MOD_DIR) not in sys.path:
    sys.path.insert(0, str(MOD_DIR))

import optuna  # noqa: E402
from optuna.pruners import MedianPruner  # noqa: E402
from optuna.samplers import TPESampler  # noqa: E402
from optuna.storages import RDBStorage, RetryFailedTrialCallback  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

from modules import hpo  # noqa: E402
from modules.zit import BagZITEQLRegressor  # noqa: E402  (EQL phi, single source of truth)
from utils.config import KEY_COL, OUTPUT_DIR, SEED  # noqa: E402


DEFAULT_USER = "jh"
DEFAULT_PRECOMPUTED_DIR = PROJECT_ROOT / "0_data" / "precomputed" / "bag_zit_pp"

# variant -> exp-id (study name). exp-id.split('-')[-1] also picks the output subfolder.
VARIANT_EXP_ID = {
    "v3": "bag-zit-eql-final-003",
    "v4": "bag-zit-eql-final-004",
    "v5": "bag-zit-eql-final-005",
}

# ------------------------------------------------------------------------------------
# Warm-start anchor = bag-zit-eql-final-002 BEST trial #58 (OOF RMSE 0.005494691).
# This is itself an EQL config, so it is the natural seed for all three new EQL studies.
# Every variant's search range below is designed to CONTAIN this anchor (so enqueue is
# always inside the distribution -- no clamp/warn).
# ------------------------------------------------------------------------------------
BAG_ZIT_EQL_ANCHOR_002 = {
    "zeta": 1.0268864552766277,
    "n_em_iters": 22,
    "mu_n_estimators": 70,
    "mu_learning_rate": 0.0060766387987241915,
    "mu_num_leaves": 165,
    "mu_max_depth": 3,
    "mu_min_child_samples": 135,
    "mu_subsample": 0.522143096780092,
    "mu_colsample_bytree": 0.5375972225613537,
    "mu_reg_alpha": 4.8410077812888856e-05,
    "mu_reg_lambda": 0.016081378287710623,
    "pi_n_estimators": 399,
    "pi_learning_rate": 0.017901499304118572,
    "pi_num_leaves": 103,
    "pi_max_depth": 17,
    "pi_min_child_samples": 53,
    "phi_n_estimators": 125,
    "phi_learning_rate": 0.00795967360245769,
    "phi_num_leaves": 52,
    "phi_max_depth": 4,
    "phi_min_child_samples": 200,
}
ANCHOR_TAU_PI_002 = 0.9397668379445991

# ------------------------------------------------------------------------------------
# 002 best-trial boundary read (#58 sits at): zeta@low(1.027), mu_n_estimators@low(70),
# mu_max_depth@low(3), mu_colsample~high(0.54), mu_reg_alpha~high(4.8e-5),
# phi_n_estimators~low(125), phi_learning_rate~high(0.0080), phi_max_depth@high(4),
# phi_min_child_samples~high(200), pi_learning_rate~low(0.018). Top-5 trials are nearly
# tied (0.0054947-0.0054952) -> very flat surface near the optimum.
#
# v3 EXPLOIT : tight band around #58 -- squeeze the known-good basin (fine local search).
# v4 PUSH    : extend exactly the walls #58 pressed against -- is the optimum just outside?
# v5 EXPLORE : broad re-scan -- hedge against a local optimum, let TPE re-find structure.
# All three CONTAIN #58 so the warm-start anchor is valid in each.
# ------------------------------------------------------------------------------------
EQL_SEARCH_SPACES = {
    # ---- v3: EXPLOIT (narrow, centered on #58) -------------------------------------
    "v3": {
        "zeta": {"type": "float", "low": 1.02, "high": 1.10, "log": False},
        "n_em_iters": {"type": "int", "low": 18, "high": 26},
        "mu_n_estimators": {"type": "int", "low": 70, "high": 120},
        "mu_learning_rate": {"type": "float", "low": 0.0050, "high": 0.0080, "log": True},
        "mu_num_leaves": {"type": "int", "low": 120, "high": 200},
        "mu_max_depth": {"type": "int", "low": 3, "high": 4},
        "mu_min_child_samples": {"type": "int", "low": 100, "high": 170},
        "mu_subsample": {"type": "float", "low": 0.45, "high": 0.70, "log": False},
        "mu_colsample_bytree": {"type": "float", "low": 0.45, "high": 0.65, "log": False},
        "mu_reg_alpha": {"type": "float", "low": 2.0e-5, "high": 8.0e-5, "log": True},
        "mu_reg_lambda": {"type": "float", "low": 0.010, "high": 0.022, "log": True},
        "pi_n_estimators": {"type": "int", "low": 340, "high": 460},
        "pi_learning_rate": {"type": "float", "low": 0.015, "high": 0.024, "log": True},
        "pi_num_leaves": {"type": "int", "low": 90, "high": 130},
        "pi_max_depth": {"type": "int", "low": 13, "high": 20},
        "pi_min_child_samples": {"type": "int", "low": 40, "high": 65},
        "phi_n_estimators": {"type": "int", "low": 110, "high": 170},
        "phi_learning_rate": {"type": "float", "low": 0.0060, "high": 0.0090, "log": True},
        "phi_num_leaves": {"type": "int", "low": 40, "high": 70},
        "phi_max_depth": {"type": "int", "low": 3, "high": 4},
        "phi_min_child_samples": {"type": "int", "low": 160, "high": 220},
    },
    # ---- v4: PUSH (extend the boundaries #58 pressed against) ----------------------
    "v4": {
        "zeta": {"type": "float", "low": 1.005, "high": 1.12, "log": False},
        "n_em_iters": {"type": "int", "low": 18, "high": 30},
        "mu_n_estimators": {"type": "int", "low": 50, "high": 110},
        "mu_learning_rate": {"type": "float", "low": 0.0045, "high": 0.0085, "log": True},
        "mu_num_leaves": {"type": "int", "low": 120, "high": 230},
        "mu_max_depth": {"type": "int", "low": 3, "high": 5},
        "mu_min_child_samples": {"type": "int", "low": 100, "high": 200},
        "mu_subsample": {"type": "float", "low": 0.40, "high": 0.70, "log": False},
        "mu_colsample_bytree": {"type": "float", "low": 0.45, "high": 0.85, "log": False},
        "mu_reg_alpha": {"type": "float", "low": 3.0e-5, "high": 3.0e-4, "log": True},
        "mu_reg_lambda": {"type": "float", "low": 0.010, "high": 0.040, "log": True},
        "pi_n_estimators": {"type": "int", "low": 350, "high": 520},
        "pi_learning_rate": {"type": "float", "low": 0.012, "high": 0.022, "log": True},
        "pi_num_leaves": {"type": "int", "low": 90, "high": 150},
        "pi_max_depth": {"type": "int", "low": 12, "high": 22},
        "pi_min_child_samples": {"type": "int", "low": 40, "high": 75},
        "phi_n_estimators": {"type": "int", "low": 100, "high": 180},
        "phi_learning_rate": {"type": "float", "low": 0.0060, "high": 0.0120, "log": True},
        "phi_num_leaves": {"type": "int", "low": 40, "high": 80},
        "phi_max_depth": {"type": "int", "low": 3, "high": 6},
        "phi_min_child_samples": {"type": "int", "low": 150, "high": 260},
    },
    # ---- v5: EXPLORE (broad re-scan around #58) ------------------------------------
    "v5": {
        "zeta": {"type": "float", "low": 1.01, "high": 1.30, "log": False},
        "n_em_iters": {"type": "int", "low": 12, "high": 30},
        "mu_n_estimators": {"type": "int", "low": 50, "high": 180},
        "mu_learning_rate": {"type": "float", "low": 0.0035, "high": 0.0120, "log": True},
        "mu_num_leaves": {"type": "int", "low": 80, "high": 256},
        "mu_max_depth": {"type": "int", "low": 3, "high": 7},
        "mu_min_child_samples": {"type": "int", "low": 60, "high": 220},
        "mu_subsample": {"type": "float", "low": 0.40, "high": 0.95, "log": False},
        "mu_colsample_bytree": {"type": "float", "low": 0.22, "high": 0.85, "log": False},
        "mu_reg_alpha": {"type": "float", "low": 1.0e-6, "high": 5.0e-4, "log": True},
        "mu_reg_lambda": {"type": "float", "low": 0.003, "high": 0.060, "log": True},
        "pi_n_estimators": {"type": "int", "low": 250, "high": 550},
        "pi_learning_rate": {"type": "float", "low": 0.010, "high": 0.045, "log": True},
        "pi_num_leaves": {"type": "int", "low": 70, "high": 180},
        "pi_max_depth": {"type": "int", "low": 8, "high": 24},
        "pi_min_child_samples": {"type": "int", "low": 30, "high": 90},
        "phi_n_estimators": {"type": "int", "low": 90, "high": 250},
        "phi_learning_rate": {"type": "float", "low": 0.0040, "high": 0.0140, "log": True},
        "phi_num_leaves": {"type": "int", "low": 24, "high": 110},
        "phi_max_depth": {"type": "int", "low": 2, "high": 6},
        "phi_min_child_samples": {"type": "int", "low": 80, "high": 260},
    },
}
TAU_PI_RANGES = {
    "v3": (0.90, 0.97),
    "v4": (0.88, 1.0),
    "v5": (0.80, 1.0),
}


def _read_manifest(precomputed_dir: Path) -> dict:
    mpath = precomputed_dir / "manifest.json"
    if not mpath.exists():
        return {}
    try:
        return json.loads(mpath.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_precomputed_data(precomputed_dir: Path):
    """mmap-load the single precomputed numeric matrix (00_precompute_pp.py).

    Returns (x_train, uid_train_die, y_train_unit_s, y_train_die, feat_cols) -- the
    SAME shape/semantics as the self-preprocessing path, so build_objective and the
    objective are unchanged. pp.npy stays memory-mapped (read-only) and shared across
    all worker processes via the OS page cache; only the per-fold x_train[mask] copies
    are private. uid is an int code here (str->int relabel done in precompute);
    die->unit groupby and KFold-by-position are invariant to the relabel.

    EQL note: the precomputed matrix is byte-identical to the base worker's preprocessing
    (median-impute patch + same PP_FIXED + clip). EQL changes only the phi M-step inside
    the model, never the data -- so reusing bag_zit_pp here is exact, not an approximation.
    """
    d = Path(precomputed_dir)
    pp_path = d / "pp.npy"
    if not pp_path.exists():
        raise FileNotFoundError(
            f"Precomputed matrix not found: {pp_path}\n"
            f"  Run the precompute step first (once per PC, or copy the folder):\n"
            f"    python 3_modeling/01_zit/hp/00_precompute_pp.py"
        )
    manifest = _read_manifest(d)
    pp = np.load(pp_path, mmap_mode="r")               # (n_dies, F+2) read-only mmap (shared)
    F = int(manifest.get("n_features", pp.shape[1] - 2))
    units = np.load(d / "units.npy")                    # (n_units, 2)
    feat = json.loads((d / "feat_cols.json").read_text(encoding="utf-8"))

    x_train = pp[:, :F]                                 # mmap view, no copy
    uid_train_die = pp[:, F].astype(np.int64)           # die-order unit codes
    y_train_die = np.asarray(pp[:, F + 1], dtype=np.float64)
    y_train_unit_s = pd.Series(
        units[:, 1].astype(np.float64),
        index=units[:, 0].astype(np.int64),
    )
    print(f"[precomputed] mmap pp={pp.shape} F={F} units={len(y_train_unit_s):,} from {d}")
    print(f"  X_train={x_train.shape} (mmap view), n_features={len(feat)}")
    return x_train, uid_train_die, y_train_unit_s, y_train_die, feat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BagZIT-EQL precomputed-mmap parallel HPO worker.")
    parser.add_argument(
        "--variant",
        required=True,
        choices=["v3", "v4", "v5"],
        help="Search-space variant -> study. v3=exploit(003), v4=push(004), v5=explore(005). "
             "One PC per variant.",
    )
    parser.add_argument(
        "--exp-id",
        default=None,
        help="Override the study/exp-id. Default derives from --variant (VARIANT_EXP_ID).",
    )
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--worker-id", default=f"pid{os.getpid()}")
    parser.add_argument("--n-trials", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=90.0)
    parser.add_argument(
        "--end-at",
        default=None,
        help=(
            "Absolute wall-clock end datetime in ISO format, e.g. '2026-06-08T05:00'. "
            "When set, the worker stops at this moment regardless of --timeout-hours. "
            "Pass the same value to every worker so all stop simultaneously."
        ),
    )
    parser.add_argument(
        "--precomputed-dir",
        default=str(DEFAULT_PRECOMPUTED_DIR),
        help="Directory with precomputed pp.npy/units.npy (00_precompute_pp.py output).",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--n-startup-trials", type=int, default=12)
    parser.add_argument("--db-timeout", type=float, default=120.0)
    parser.add_argument("--heartbeat-interval", type=int, default=300)
    parser.add_argument("--grace-period", type=int, default=1800)
    parser.add_argument("--max-retry", type=int, default=1)
    parser.add_argument("--enqueue-anchor", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def _sum_die_to_unit(pred_die: np.ndarray, uid_die: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({KEY_COL: uid_die, "pred": pred_die})
    return df.groupby(KEY_COL, sort=False)["pred"].sum().reset_index()


def _apply_tau_pi(pred_die: np.ndarray, pi_die: np.ndarray, tau_pi: float) -> np.ndarray:
    return np.where(pi_die > tau_pi, 0.0, pred_die)


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
            anchor = dict(BAG_ZIT_EQL_ANCHOR_002, tau_pi=ANCHOR_TAU_PI_002)
            study.enqueue_trial(anchor)
            print(f"[enqueue] bag-zit-eql-final-002 best #58 anchor ({len(anchor)} HP)")
        else:
            print(f"[enqueue skip] existing trials={len(study.trials)}")

    manifest = _read_manifest(Path(args.precomputed_dir))
    study_meta = {
        "exp_id": args.exp_id,
        "user": args.user,
        "model": "BagZIT-EQL PRECOMPUTED-MMAP parallel HPO worker",
        "variant": args.variant,
        "variant_intent": {"v3": "exploit", "v4": "push", "v5": "explore"}[args.variant],
        "n_trials_per_worker": args.n_trials,
        "n_folds": args.n_folds,
        "n_jobs": args.n_jobs,
        "data_source": "precomputed mmap (00_precompute_pp.py)",
        "precomputed_dir": str(args.precomputed_dir),
        "pp_fixed": manifest.get("pp_fixed"),
        "clip_y_extreme": manifest.get("clip_y_extreme"),
        "precompute_fingerprint": manifest.get("fingerprint_nansum_stride101"),
        "n_features": manifest.get("n_features"),
        "anchor_source": "bag-zit-eql-final-002 best trial #58 (OOF 0.005494691)",
        "anchor": BAG_ZIT_EQL_ANCHOR_002,
        "anchor_tau_pi": ANCHOR_TAU_PI_002,
        "search_space": args.search_space,
        "tau_pi_range": args.tau_pi_range,
        "sampler": "TPE seed=None multivariate group",
        "pruner": f"MedianPruner n_startup={args.n_startup_trials} n_warmup=2",
        "phi_update": "EQL Tweedie unit deviance target (modules.zit.BagZITEQLRegressor)",
        "seed": int(SEED),
        "model_random_state": "SEED + trial.number",
        "worker_id_last_writer": args.worker_id,
    }
    for k, v in study_meta.items():
        study.set_user_attr(k, str(v))
    return study


def build_objective(args: argparse.Namespace):
    x_train, uid_train_die, y_train_unit_s, y_train_die, _ = load_precomputed_data(args.precomputed_dir)

    search_space = args.search_space
    tau_lo, tau_hi = args.tau_pi_range

    unique_units = y_train_unit_s.index.values
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)
    folds = list(kf.split(unique_units))
    print(f"[fold split] n_folds={args.n_folds}, seed={SEED}")

    def objective(trial: optuna.Trial) -> float:
        t0 = time.time()
        params = hpo.sample_from_space(trial, search_space)
        tau_pi = trial.suggest_float("tau_pi", tau_lo, tau_hi)

        model_seed = int(SEED) + int(trial.number)
        params["random_state"] = model_seed
        params["n_jobs"] = args.n_jobs
        params["verbose"] = -1
        params["device"] = "cpu"
        params["em_tol"] = 1e-7

        trial.set_user_attr("worker_id", args.worker_id)
        trial.set_user_attr("pid", os.getpid())
        trial.set_user_attr("variant", args.variant)
        trial.set_user_attr("tau_pi", tau_pi)
        trial.set_user_attr("model_seed", model_seed)
        trial.set_user_attr("phi_update", "eql_deviance")

        fold_oof_rmse = []
        oof_pred_unit = pd.Series(np.nan, index=y_train_unit_s.index, dtype=np.float64)

        for fold_idx, (tr_uidx, vl_uidx) in enumerate(folds):
            tr_units = unique_units[tr_uidx]
            vl_units = unique_units[vl_uidx]
            tr_mask = np.isin(uid_train_die, tr_units)
            vl_mask = np.isin(uid_train_die, vl_units)

            model = BagZITEQLRegressor(**params)
            model.fit(
                x_train[tr_mask],
                y_train_die[tr_mask],
                unit_id=uid_train_die[tr_mask],
            )

            pi_vl, mu_vl, _ = model.predict_components(x_train[vl_mask])
            pred_die_raw = np.clip((1.0 - pi_vl) * mu_vl, 0.0, None)
            pred_die_taupi = _apply_tau_pi(pred_die_raw, pi_vl, tau_pi)
            unit_pred_df = _sum_die_to_unit(pred_die_taupi, uid_train_die[vl_mask])

            oof_pred_unit.loc[unit_pred_df[KEY_COL].values] = unit_pred_df["pred"].values
            y_vl = y_train_unit_s.loc[unit_pred_df[KEY_COL].values].values
            fold_rmse = float(np.sqrt(np.mean((unit_pred_df["pred"].values - y_vl) ** 2)))
            fold_oof_rmse.append(fold_rmse)

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
            f"trial #{trial.number}: variant={args.variant}, worker={args.worker_id}, "
            f"tau_pi={tau_pi:.4f}, oof={oof_rmse:.9f}, elapsed={elapsed:.0f}s"
        )
        return oof_rmse

    return objective


def main() -> None:
    args = parse_args()

    # Resolve variant -> exp-id + search space + tau_pi range (stored on args).
    if args.exp_id is None:
        args.exp_id = VARIANT_EXP_ID[args.variant]
    args.search_space = EQL_SEARCH_SPACES[args.variant]
    args.tau_pi_range = TAU_PI_RANGES[args.variant]

    logging.getLogger("lightgbm").setLevel(logging.ERROR)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    out_dir = Path(OUTPUT_DIR) / "01_zit" / "bag_zit" / "hp" / args.exp_id.split("-")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / f"optuna_{args.user}_{args.exp_id}.db"

    print(f"PROJECT_ROOT={PROJECT_ROOT}")
    print(f"optuna={optuna.__version__}")
    print(f"VARIANT={args.variant} -> EXP_ID={args.exp_id}, worker_id={args.worker_id}")
    print(f"PRECOMPUTED_DIR={args.precomputed_dir}")
    print(f"OUT_DIR={out_dir}")
    print(f"DB_PATH={db_path}")
    print(f"N_TRIALS={args.n_trials}, N_JOBS={args.n_jobs}, N_FOLDS={args.n_folds}")
    print(f"N_STARTUP_TRIALS={args.n_startup_trials}")
    print(f"search HP={len(args.search_space)} + tau_pi=1, tau_pi_range={args.tau_pi_range}")

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
            print(f"[end-at] {end_dt.isoformat()} already passed; nothing to do")
            return
        timeout = remaining
        print(
            f"[end-at] target={end_dt.isoformat()} "
            f"-> timeout={timeout:.0f}s ({timeout / 3600:.2f}h)"
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
    print(f"[HPO done] variant={args.variant}, worker={args.worker_id}, "
          f"elapsed={elapsed:.0f}s, total_trials={len(study.trials)}")
    try:
        print(f"best_value={study.best_value:.9f}, best_trial={study.best_trial.number}")
    except ValueError:
        print("best_value unavailable: no completed trials")


if __name__ == "__main__":
    main()
