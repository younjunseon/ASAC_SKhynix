"""
Parallel HPO worker for ZIT-only hp/004 (PC1) and hp/005 (PC2).

Same search space and tuning logic; only --exp-id differs per PC. Each process
claims trials through its own SQLite Optuna DB and writes only Optuna DB records.
Final refit, postprocess, and artifact export must be run separately.

Compared to hp/003:
  - 11 search ranges widened from 002+003 TOP boundary analysis
  - tau_pi lower bound 0.88 -> 0.84
  - n_em_iters upper 24 -> 28 (002 36% HI, 003 20% HI)
  - default n_startup_trials 60 -> 120 (larger search space needs more exploration)
  - per-trial model seed = SEED + trial.number (deterministic per-trial diversity)
  - val_rmse / partial_val_rmse recorded as trial.user_attrs (DB-side metric)
  - both 002 best and 003 best are enqueued as anchors

SQLite is local per-PC. Do not share the DB over a network mount.

Run command (PowerShell on each PC):

  ### PC1 (hp/004)
  # Step 1 — one-time init. Enqueues 002 + 003 best anchors, then exits.
  python 3_modeling/01_zit/hp/01_zit_only_parallel_hpo.py --exp-id zit-only-final-004 --enqueue-anchor --n-trials 0

  # Step 2 — open 3 terminals on PC1. Tune --n-jobs so 3 * n_jobs <= physical thread count.
  python 3_modeling/01_zit/hp/01_zit_only_parallel_hpo.py --exp-id zit-only-final-004 --worker-id w1 --n-trials 2000 --n-jobs 4 --end-at 2026-06-04T05:00 > w1.log 2>&1
  python 3_modeling/01_zit/hp/01_zit_only_parallel_hpo.py --exp-id zit-only-final-004 --worker-id w2 --n-trials 2000 --n-jobs 4 --end-at 2026-06-04T05:00 > w2.log 2>&1
  python 3_modeling/01_zit/hp/01_zit_only_parallel_hpo.py --exp-id zit-only-final-004 --worker-id w3 --n-trials 2000 --n-jobs 4 --end-at 2026-06-04T05:00 > w3.log 2>&1

  ### PC2 (hp/005) — identical commands, only --exp-id changes
  python 3_modeling/01_zit/hp/01_zit_only_parallel_hpo.py --exp-id zit-only-final-005 --enqueue-anchor --n-trials 0
  python 3_modeling/01_zit/hp/01_zit_only_parallel_hpo.py --exp-id zit-only-final-005 --worker-id w1 --n-trials 2000 --n-jobs 4 --end-at 2026-06-04T05:00 > w1.log 2>&1
  python 3_modeling/01_zit/hp/01_zit_only_parallel_hpo.py --exp-id zit-only-final-005 --worker-id w2 --n-trials 2000 --n-jobs 4 --end-at 2026-06-04T05:00 > w2.log 2>&1
  python 3_modeling/01_zit/hp/01_zit_only_parallel_hpo.py --exp-id zit-only-final-005 --worker-id w3 --n-trials 2000 --n-jobs 4 --end-at 2026-06-04T05:00 > w3.log 2>&1

  # --n-jobs guide:
  #   16-thread PC -> --n-jobs 5
  #   12-thread PC -> --n-jobs 4
  #    8-thread PC -> --n-jobs 2
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
from optuna.pruners import MedianPruner  # noqa: E402
from optuna.samplers import TPESampler  # noqa: E402
from optuna.storages import RDBStorage, RetryFailedTrialCallback  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

from meta_features import add_meta_features  # noqa: E402
from modules import hpo, preprocess  # noqa: E402
from modules.zit import ZITboostRegressor  # noqa: E402
from utils.config import DIE_KEY_COL, KEY_COL, OUTPUT_DIR, SEED, TARGET_COL  # noqa: E402
from utils.data import get_feat_cols, load_all, split_xs  # noqa: E402


DEFAULT_EXP_ID = "zit-only-final-004"
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

# hp/002 best trial #82. Enqueued together with hp/003 best when --enqueue-anchor is used.
ZIT_ONLY_ANCHOR_002 = {
    "zeta": 1.070592137915828,
    "n_em_iters": 17,
    "mu_n_estimators": 200,
    # hp/002 trial #82의 원본 mu_num_leaves=260은 max_depth=4와 함께 쓰여 LightGBM이
    # 실제로 쓴 leaf 수는 min(260, 2^4)=16. hp/004 range도 [16, 128]이라 effective
    # 값(16)으로 정합. anchor trial이 hp/002 best와 동일 동작.
    "mu_num_leaves": 16,
    "mu_max_depth": 4,
    "mu_min_child_samples": 202,
    "mu_subsample": 0.7490707207911603,
    "mu_colsample_bytree": 0.21034413805267055,
    "mu_learning_rate": 0.002724262170219916,
    "mu_reg_alpha": 0.002190807541322471,
    "mu_reg_lambda": 0.0008220461034125899,
    "pi_n_estimators": 117,
    "pi_learning_rate": 0.04641531862989267,
    "pi_num_leaves": 211,
    "pi_max_depth": 15,
    "pi_min_child_samples": 43,
    "phi_n_estimators": 50,
    "phi_learning_rate": 0.004006136103984497,
    "phi_num_leaves": 71,
    "phi_max_depth": 6,
    "phi_min_child_samples": 322,
}
ANCHOR_TAU_PI_002 = 0.9417341871098373

# hp/003 best trial #24 (val_rmse=0.005493945346).
ZIT_ONLY_ANCHOR_003 = {
    "zeta": 1.1200196096433799,
    "n_em_iters": 16,
    "mu_n_estimators": 206,
    "mu_learning_rate": 0.001544804994892871,
    "mu_num_leaves": 29,
    "mu_max_depth": 3,
    "mu_min_child_samples": 269,
    "mu_subsample": 0.8129218946739942,
    "mu_colsample_bytree": 0.18344245348572596,
    "mu_reg_alpha": 0.001211879891858438,
    "mu_reg_lambda": 0.003235224001159518,
    "pi_n_estimators": 125,
    "pi_learning_rate": 0.05221975843303289,
    "pi_num_leaves": 179,
    "pi_max_depth": 17,
    "pi_min_child_samples": 44,
    "phi_n_estimators": 44,
    "phi_learning_rate": 0.002809100266233647,
    "phi_num_leaves": 75,
    "phi_max_depth": 6,
    "phi_min_child_samples": 322,
}
ANCHOR_TAU_PI_003 = 0.8927453228423853

# hp/004 search space. Built from 002 TOP-25 + 003 TOP-10 boundary analysis.
# Widened low edges where best trials clustered low (mu_reg_alpha/lambda, mu_lr,
# phi_lr, mu_colsample), widened high edges where they clustered high
# (n_em_iters, mu_num_leaves, mu_max_depth, pi_max_depth, phi_max_depth,
# phi_num_leaves, phi_min_child_samples). mu_num_leaves cap stays loose because
# mu_max_depth still caps effective leaves to 2^max_depth.
ZIT_SEARCH = {
    "zeta": {"type": "float", "low": 1.02, "high": 1.20, "log": False},
    "n_em_iters": {"type": "int", "low": 15, "high": 28},
    "mu_n_estimators": {"type": "int", "low": 150, "high": 240},
    "mu_learning_rate": {"type": "float", "low": 0.0009, "high": 0.0032, "log": True},
    "mu_num_leaves": {"type": "int", "low": 16, "high": 128},
    "mu_max_depth": {"type": "int", "low": 2, "high": 7},
    "mu_min_child_samples": {"type": "int", "low": 160, "high": 320},
    "mu_subsample": {"type": "float", "low": 0.55, "high": 0.85, "log": False},
    "mu_colsample_bytree": {"type": "float", "low": 0.14, "high": 0.34, "log": False},
    "mu_reg_alpha": {"type": "float", "low": 1e-4, "high": 0.006, "log": True},
    "mu_reg_lambda": {"type": "float", "low": 5e-5, "high": 0.007, "log": True},
    "pi_n_estimators": {"type": "int", "low": 90, "high": 170},
    "pi_learning_rate": {"type": "float", "low": 0.032, "high": 0.065, "log": True},
    "pi_num_leaves": {"type": "int", "low": 160, "high": 260},
    "pi_max_depth": {"type": "int", "low": 12, "high": 22},
    "pi_min_child_samples": {"type": "int", "low": 25, "high": 50},
    "phi_n_estimators": {"type": "int", "low": 30, "high": 60},
    "phi_learning_rate": {"type": "float", "low": 0.0012, "high": 0.0065, "log": True},
    "phi_num_leaves": {"type": "int", "low": 40, "high": 110},
    "phi_max_depth": {"type": "int", "low": 5, "high": 10},
    "phi_min_child_samples": {"type": "int", "low": 180, "high": 520},
}
TAU_PI_RANGE = (0.84, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ZIT-only parallel HPO worker.")
    parser.add_argument("--exp-id", default=DEFAULT_EXP_ID)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--worker-id", default=f"pid{os.getpid()}")
    parser.add_argument("--n-trials", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=90.0)
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
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--n-startup-trials", type=int, default=120)
    parser.add_argument("--db-timeout", type=float, default=120.0)
    parser.add_argument("--heartbeat-interval", type=int, default=300)
    parser.add_argument("--grace-period", type=int, default=1800)
    parser.add_argument("--max-retry", type=int, default=1)
    parser.add_argument("--enqueue-anchor", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-clip-y-extreme", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def _mean_die_to_unit(pred_die: np.ndarray, uid_die: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({KEY_COL: uid_die, "pred": pred_die})
    return df.groupby(KEY_COL, sort=False)["pred"].mean().reset_index()


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
            anchor_002 = dict(ZIT_ONLY_ANCHOR_002, tau_pi=ANCHOR_TAU_PI_002)
            anchor_003 = dict(ZIT_ONLY_ANCHOR_003, tau_pi=ANCHOR_TAU_PI_003)
            study.enqueue_trial(anchor_002)
            study.enqueue_trial(anchor_003)
            print(
                f"[enqueue] 2 anchors -- 002 best ({len(anchor_002)} HP) + "
                f"003 best ({len(anchor_003)} HP)"
            )
        else:
            print(f"[enqueue skip] existing trials={len(study.trials)}")

    study_meta = {
        "exp_id": args.exp_id,
        "user": args.user,
        "model": "ZITboost (zit_only) parallel HPO worker",
        "n_trials_per_worker": args.n_trials,
        "n_folds": args.n_folds,
        "n_jobs": args.n_jobs,
        "pp_fixed": PP_FIXED,
        "anchor_002_source": "hp/002 best trial #82",
        "anchor_002": ZIT_ONLY_ANCHOR_002,
        "anchor_002_tau_pi": ANCHOR_TAU_PI_002,
        "anchor_003_source": "hp/003 best trial #24",
        "anchor_003": ZIT_ONLY_ANCHOR_003,
        "anchor_003_tau_pi": ANCHOR_TAU_PI_003,
        "search_space": ZIT_SEARCH,
        "tau_pi_range": TAU_PI_RANGE,
        "sampler": "TPE seed=None multivariate group",
        "pruner": f"MedianPruner n_startup={args.n_startup_trials} n_warmup=2",
        "clip_y_extreme": not args.no_clip_y_extreme,
        "seed": int(SEED),
        "model_random_state": "SEED + trial.number (per-trial)",
        "worker_id_last_writer": args.worker_id,
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
        xs_train,
        xs_val,
        xs_test,
        feat_cols_clean,
        position_mode="raw",
        use_die_xy=True,
    )

    x_train = xs_train[feat_cols_clean].values.astype(np.float64)
    uid_train_die = xs_train[KEY_COL].values
    y_train_unit_s = ys_input["train"].set_index(KEY_COL)[TARGET_COL]
    y_train_die = xs_train[KEY_COL].map(y_train_unit_s).values.astype(np.float64)

    print(f"[preprocess done] n_features={len(feat_cols_clean)}")
    print(f"  X_train={x_train.shape}, train_units={len(y_train_unit_s):,}")
    return x_train, uid_train_die, y_train_unit_s, y_train_die, feat_cols_clean


def build_objective(args: argparse.Namespace):
    x_train, uid_train_die, y_train_unit_s, y_train_die, _ = load_preprocessed_data(args)

    unique_units = y_train_unit_s.index.values
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)
    folds = list(kf.split(unique_units))
    print(f"[fold split] n_folds={args.n_folds}, seed={SEED}")

    def objective(trial: optuna.Trial) -> float:
        t0 = time.time()
        params = hpo.sample_from_space(trial, ZIT_SEARCH)
        tau_pi = trial.suggest_float("tau_pi", TAU_PI_RANGE[0], TAU_PI_RANGE[1])

        # 모델 seed를 trial별로 다르게 → 재현성 유지 + trial 간 다양성. KFold split은
        # SEED로 고정해 같은 study 내 모든 trial이 동일한 fold 분할을 공유 → RMSE 비교 가능.
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

        fold_oof_rmse = []
        oof_pred_unit = pd.Series(np.nan, index=y_train_unit_s.index, dtype=np.float64)

        for fold_idx, (tr_uidx, vl_uidx) in enumerate(folds):
            tr_units = unique_units[tr_uidx]
            vl_units = unique_units[vl_uidx]
            tr_mask = np.isin(uid_train_die, tr_units)
            vl_mask = np.isin(uid_train_die, vl_units)

            model = ZITboostRegressor(**params)
            model.fit(x_train[tr_mask], y_train_die[tr_mask])

            pi_vl, mu_vl, _ = model.predict_components(x_train[vl_mask])
            pred_die_raw = np.clip((1 - pi_vl) * mu_vl, 0, None)
            pred_die_taupi = _apply_tau_pi(pred_die_raw, pi_vl, tau_pi)
            unit_pred_df = _mean_die_to_unit(pred_die_taupi, uid_train_die[vl_mask])

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

    out_dir = Path(OUTPUT_DIR) / "01_zit" / "zit_only" / "hp" / args.exp_id.split("-")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / f"optuna_{args.user}_{args.exp_id}.db"

    print(f"PROJECT_ROOT={PROJECT_ROOT}")
    print(f"optuna={optuna.__version__}")
    print(f"EXP_ID={args.exp_id}, worker_id={args.worker_id}")
    print(f"OUT_DIR={out_dir}")
    print(f"DB_PATH={db_path}")
    print(f"N_TRIALS={args.n_trials}, N_JOBS={args.n_jobs}, N_FOLDS={args.n_folds}")
    print(f"search HP={len(ZIT_SEARCH)} + tau_pi=1")

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
    print(f"[HPO done] worker={args.worker_id}, elapsed={elapsed:.0f}s, total_trials={len(study.trials)}")
    try:
        print(f"best_value={study.best_value:.9f}, best_trial={study.best_trial.number}")
    except ValueError:
        print("best_value unavailable: no completed trials")


if __name__ == "__main__":
    main()
