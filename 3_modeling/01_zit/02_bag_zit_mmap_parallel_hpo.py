"""
Parallel HPO worker for BagZIT (base / simplified phi) -- PRECOMPUTED-MMAP variant.

This is the 004 sibling of 02_bag_zit_parallel_hpo.py (003). It is functionally
IDENTICAL to 003 -- same search space, same anchor, same fold logic, same tau_pi
gate, same objective -- with ONE change: it does NOT run preprocessing per worker.
Instead it mmap-loads a single precomputed numeric matrix produced once by
00_precompute_pp.py, so many workers on one PC share ONE copy of the feature matrix
via the OS page cache.

  003 (02_bag_zit_parallel_hpo.py):  each worker self-preprocesses -> ~5.3 GB startup
                                     peak + private ~0.48 GB matrix per worker.
  004 (this file):                   workers mmap-share 0.48 GB pp.npy (ONE copy in
                                     RAM), no preprocessing, instant startup.

Because the precomputed matrix is byte-identical to 003's preprocessing output
(uid is relabelled str->int code; die->unit groupby and KFold-by-position are
invariant to the relabel, and unit order is preserved), this is a clean A/B vs 003:
the only moving part is HOW the data is loaded, not the data. New exp-id
'bag-zit-final-004' = a fresh study (warm-started by the same 001 anchor).

PREREQUISITE -- build the precomputed matrix once:
  python 3_modeling/01_zit/00_precompute_pp.py
  -> 0_data/precomputed/bag_zit_pp/{pp.npy, units.npy, feat_cols.json, manifest.json}

Run command (PowerShell) -- 14 workers on one PC, n_jobs=1 (14 threads = 14 cores):

  # One-time init. Enqueues bag-zit-final-001 best as anchor, then exits.
  python 3_modeling/01_zit/02_bag_zit_mmap_parallel_hpo.py --enqueue-anchor --n-trials 0

  # Open 14 workers (or use a launcher). Only --worker-id changes; same --end-at to all.
  # All workers mmap-share ONE copy of pp.npy.
  python 3_modeling/01_zit/02_bag_zit_mmap_parallel_hpo.py --worker-id w1 --n-trials 100000 --n-jobs 1 --n-startup-trials 12 --end-at 2026-06-05T09:00 > 4_output/logs/bag004_w1.log 2>&1
  ...  (w2 .. w14, identical except --worker-id)

This worker writes Optuna DB records only. Final refit, postprocess, and artifact
export should be run separately after selecting the best trial.
"""

from __future__ import annotations

import os

# Limit per-process BLAS/numpy threads BEFORE importing numpy. Otherwise each of the N
# parallel workers spawns an all-core BLAS pool for the numpy EM steps, so N workers x
# (all cores) = massive thread oversubscription. In the 004 first run this made every
# trial ~5x slower (~10 h instead of ~2 h: 14 workers spawned ~530 threads on 14 cores).
# LightGBM's own --n-jobs only caps LightGBM threads, NOT numpy/BLAS -- so this is needed
# on top of it. setdefault -> a caller can still override via the environment.
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
from modules.zit import BagZITboostRegressor  # noqa: E402
from utils.config import KEY_COL, OUTPUT_DIR, SEED  # noqa: E402


DEFAULT_EXP_ID = "bag-zit-final-004"
DEFAULT_USER = "jh"
DEFAULT_PRECOMPUTED_DIR = PROJECT_ROOT / "0_data" / "precomputed" / "bag_zit_pp"

# bag-zit-final-001 best trial #122 (identical to 003 -- same anchor for clean A/B).
BAG_ZIT_ANCHOR_001 = {
    "zeta": 1.0993526288230726,
    "n_em_iters": 19,
    "mu_n_estimators": 90,
    "mu_learning_rate": 0.0072451235639959255,
    "mu_num_leaves": 127,
    "mu_max_depth": 5,
    "mu_min_child_samples": 123,
    "mu_subsample": 0.6800893374780662,
    "mu_colsample_bytree": 0.3733854981962628,
    "mu_reg_alpha": 2.635748480441289e-05,
    "mu_reg_lambda": 0.015384762992767538,
    "pi_n_estimators": 414,
    "pi_learning_rate": 0.02056316533003463,
    "pi_num_leaves": 113,
    "pi_max_depth": 16,
    "pi_min_child_samples": 52,
    "phi_n_estimators": 171,
    "phi_learning_rate": 0.005536671157137253,
    "phi_num_leaves": 54,
    "phi_max_depth": 3,
    "phi_min_child_samples": 140,
}
ANCHOR_TAU_PI_001 = 0.8998148839519274

# Identical search space to 003 (02_bag_zit_parallel_hpo.py). Keeping it identical
# makes 004 a clean A/B against 003 -- only the data-loading path differs. The 4-PC
# search-space variants branch from here later.
BAG_ZIT_SEARCH = {
    "zeta": {"type": "float", "low": 1.02, "high": 1.22, "log": False},
    "n_em_iters": {"type": "int", "low": 15, "high": 26},
    "mu_n_estimators": {"type": "int", "low": 70, "high": 145},
    "mu_learning_rate": {"type": "float", "low": 0.0048, "high": 0.0090, "log": True},
    "mu_num_leaves": {"type": "int", "low": 100, "high": 210},
    "mu_max_depth": {"type": "int", "low": 3, "high": 6},
    "mu_min_child_samples": {"type": "int", "low": 70, "high": 170},
    "mu_subsample": {"type": "float", "low": 0.48, "high": 0.90, "log": False},
    "mu_colsample_bytree": {"type": "float", "low": 0.24, "high": 0.55, "log": False},
    "mu_reg_alpha": {"type": "float", "low": 1.5e-5, "high": 5.0e-5, "log": True},
    "mu_reg_lambda": {"type": "float", "low": 0.008, "high": 0.022, "log": True},
    "pi_n_estimators": {"type": "int", "low": 280, "high": 500},
    "pi_learning_rate": {"type": "float", "low": 0.016, "high": 0.038, "log": True},
    "pi_num_leaves": {"type": "int", "low": 80, "high": 160},
    "pi_max_depth": {"type": "int", "low": 10, "high": 22},
    "pi_min_child_samples": {"type": "int", "low": 35, "high": 75},
    "phi_n_estimators": {"type": "int", "low": 120, "high": 230},
    "phi_learning_rate": {"type": "float", "low": 0.0045, "high": 0.0085, "log": True},
    "phi_num_leaves": {"type": "int", "low": 32, "high": 96},
    "phi_max_depth": {"type": "int", "low": 2, "high": 4},
    "phi_min_child_samples": {"type": "int", "low": 90, "high": 220},
}
TAU_PI_RANGE = (0.80, 1.0)


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
    SAME shape/semantics as 003's load_preprocessed_data, so build_objective and the
    objective are unchanged. pp.npy stays memory-mapped (read-only) and shared across
    all worker processes via the OS page cache; only the per-fold x_train[mask] copies
    are private. uid is an int code here (str->int relabel done in precompute);
    die->unit groupby and KFold-by-position are invariant to the relabel.
    """
    d = Path(precomputed_dir)
    pp_path = d / "pp.npy"
    if not pp_path.exists():
        raise FileNotFoundError(
            f"Precomputed matrix not found: {pp_path}\n"
            f"  Run the precompute step first:\n"
            f"    python 3_modeling/01_zit/00_precompute_pp.py"
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
    parser = argparse.ArgumentParser(description="BagZIT precomputed-mmap parallel HPO worker (004).")
    parser.add_argument("--exp-id", default=DEFAULT_EXP_ID)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--worker-id", default=f"pid{os.getpid()}")
    parser.add_argument("--n-trials", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=90.0)
    parser.add_argument(
        "--end-at",
        default=None,
        help=(
            "Absolute wall-clock end datetime in ISO format, e.g. '2026-06-05T09:00'. "
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
    parser.add_argument("--n-jobs", type=int, default=1)
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
            anchor = dict(BAG_ZIT_ANCHOR_001, tau_pi=ANCHOR_TAU_PI_001)
            study.enqueue_trial(anchor)
            print(f"[enqueue] bag-zit-final-001 best anchor ({len(anchor)} HP)")
        else:
            print(f"[enqueue skip] existing trials={len(study.trials)}")

    manifest = _read_manifest(Path(args.precomputed_dir))
    study_meta = {
        "exp_id": args.exp_id,
        "user": args.user,
        "model": "BagZIT (base, simplified phi) PRECOMPUTED-MMAP parallel HPO worker (004)",
        "n_trials_per_worker": args.n_trials,
        "n_folds": args.n_folds,
        "n_jobs": args.n_jobs,
        "data_source": "precomputed mmap (00_precompute_pp.py)",
        "precomputed_dir": str(args.precomputed_dir),
        "pp_fixed": manifest.get("pp_fixed"),
        "clip_y_extreme": manifest.get("clip_y_extreme"),
        "precompute_fingerprint": manifest.get("fingerprint_nansum_stride101"),
        "n_features": manifest.get("n_features"),
        "anchor_source": "bag-zit-final-001 best trial #122",
        "anchor": BAG_ZIT_ANCHOR_001,
        "anchor_tau_pi": ANCHOR_TAU_PI_001,
        "search_space": BAG_ZIT_SEARCH,
        "tau_pi_range": TAU_PI_RANGE,
        "sampler": "TPE seed=None multivariate group",
        "pruner": f"MedianPruner n_startup={args.n_startup_trials} n_warmup=2",
        "phi_update": "base ZITboost (y-mu)^2 / mu^zeta (non-EQL, like zit_only)",
        "ab_note": "identical search/anchor/fold/objective to 003; only data loading differs (mmap vs self-preprocess)",
        "seed": int(SEED),
        "model_random_state": "SEED + trial.number",
        "worker_id_last_writer": args.worker_id,
    }
    for k, v in study_meta.items():
        study.set_user_attr(k, str(v))
    return study


def build_objective(args: argparse.Namespace):
    x_train, uid_train_die, y_train_unit_s, y_train_die, _ = load_precomputed_data(args.precomputed_dir)

    unique_units = y_train_unit_s.index.values
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)
    folds = list(kf.split(unique_units))
    print(f"[fold split] n_folds={args.n_folds}, seed={SEED}")

    def objective(trial: optuna.Trial) -> float:
        t0 = time.time()
        params = hpo.sample_from_space(trial, BAG_ZIT_SEARCH)
        tau_pi = trial.suggest_float("tau_pi", TAU_PI_RANGE[0], TAU_PI_RANGE[1])

        model_seed = int(SEED) + int(trial.number)
        params["random_state"] = model_seed
        params["n_jobs"] = args.n_jobs
        params["verbose"] = -1
        params["device"] = "cpu"
        params["em_tol"] = 1e-7

        trial.set_user_attr("worker_id", args.worker_id)
        trial.set_user_attr("pid", os.getpid())
        trial.set_user_attr("tau_pi", tau_pi)
        trial.set_user_attr("model_seed", model_seed)
        trial.set_user_attr("phi_update", "pearson_resid_sq")

        fold_oof_rmse = []
        oof_pred_unit = pd.Series(np.nan, index=y_train_unit_s.index, dtype=np.float64)

        for fold_idx, (tr_uidx, vl_uidx) in enumerate(folds):
            tr_units = unique_units[tr_uidx]
            vl_units = unique_units[vl_uidx]
            tr_mask = np.isin(uid_train_die, tr_units)
            vl_mask = np.isin(uid_train_die, vl_units)

            model = BagZITboostRegressor(**params)
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
            f"trial #{trial.number}: worker={args.worker_id}, "
            f"tau_pi={tau_pi:.4f}, oof={oof_rmse:.9f}, elapsed={elapsed:.0f}s"
        )
        return oof_rmse

    return objective


def main() -> None:
    args = parse_args()
    logging.getLogger("lightgbm").setLevel(logging.ERROR)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    out_dir = Path(OUTPUT_DIR) / "01_zit" / "bag_zit" / args.exp_id.split("-")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / f"optuna_{args.user}_{args.exp_id}.db"

    print(f"PROJECT_ROOT={PROJECT_ROOT}")
    print(f"optuna={optuna.__version__}")
    print(f"EXP_ID={args.exp_id}, worker_id={args.worker_id}")
    print(f"PRECOMPUTED_DIR={args.precomputed_dir}")
    print(f"OUT_DIR={out_dir}")
    print(f"DB_PATH={db_path}")
    print(f"N_TRIALS={args.n_trials}, N_JOBS={args.n_jobs}, N_FOLDS={args.n_folds}")
    print(f"N_STARTUP_TRIALS={args.n_startup_trials}")
    print(f"search HP={len(BAG_ZIT_SEARCH)} + tau_pi=1")

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
    print(f"[HPO done] worker={args.worker_id}, elapsed={elapsed:.0f}s, total_trials={len(study.trials)}")
    try:
        print(f"best_value={study.best_value:.9f}, best_trial={study.best_trial.number}")
    except ValueError:
        print("best_value unavailable: no completed trials")


if __name__ == "__main__":
    main()
