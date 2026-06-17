"""
Parallel seed-sweep worker for BagZIT(-EQL) — the .py sibling of
06_bag_zit_eql_seed_sweep.ipynb.

The notebook fits the SOURCE study's best HP across many random seeds (each seed =
unit-level 5-fold refit -> OOF/val/test predict -> tau_pi gate -> aggregation/zero_clip
tune -> iso/tail calibration grid) and records one summary row per seed, keeping the
best-val artifacts. This file does the SAME per-seed work but lets many independent
processes share ONE Optuna study (GridSampler over the seed list + shared SQLite) so
that, exactly like the HPO workers, each process grabs the next un-run seed the instant
it finishes — no "wait for the whole batch" stall.

WHY Optuna GridSampler (not a plain loop)?
  - atomic, dynamic seed assignment across processes (no double-run, work-stealing)
  - free RESUME: already-completed seeds are skipped on restart
  - the per-seed summary lives in trial.user_attrs; the summary CSV + "best/" copy are
    regenerated from the DB by `--summarize` (so workers never race on one CSV/folder)

DATA: this worker SELF-PREPROCESSES per process (preprocess.run + add_meta_features),
like the notebook — it does NOT use the mmap pp.npy (that has train dies only; the sweep
needs val/test + die metadata too). Per-seed cost is ~1.75-2h so the ~1-2 min preprocess
startup is negligible; the cost is RAM (~5 GB peak/worker), so size worker count to RAM.

  ⚠ IMPUTATION: the bag-zit-eql studies (incl. 003/004/005) were tuned on MEDIAN-fallback
  imputation, but the mmap workers did NOT record impute_stage23 in study meta. So for eql
  sources we FORCE median here (default --impute auto resolves eql -> median). Override with
  --impute mean only if you really want the 01-notebook mean pipeline.

SOURCE: --src-num picks the study (bag-zit-{eql-}final-{NUM}). --src-num auto = the best
(lowest best_value) among 002/003/004/005 that have completed trials on THIS machine
(002 included so it is runnable now; 003/004/005 auto-win once they beat it). The chosen
study's DB must be present locally.

Run command (PowerShell/cmd) from the project root:

  # 1) (optional) sanity: resolve source + paths, then exit
  python 3_modeling/01_zit/06_bag_zit_eql_seed_sweep_parallel.py --src-num auto --n-trials 0

  # 2) launch N independent workers (size N to RAM; ~5 GB peak/worker). Same shared study.
  python 3_modeling/01_zit/06_bag_zit_eql_seed_sweep_parallel.py --src-num auto --worker-id w1 --n-jobs 2 > 4_output/logs/sweep_w1.log 2>&1
  ...  (w2 .. wN, identical except --worker-id)

  # 3) after seeds finish (or anytime): rebuild summary CSV + copy best seed -> best/
  python 3_modeling/01_zit/06_bag_zit_eql_seed_sweep_parallel.py --src-num auto --summarize
"""

from __future__ import annotations

import os

# Cap BLAS/numpy threads BEFORE importing numpy (same reason as the HPO workers): N parallel
# workers each spawning an all-core BLAS pool = massive oversubscription. LightGBM --n-jobs
# caps only LightGBM threads, not numpy/BLAS. setdefault -> overridable via environment.
for _thr_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thr_var, "1")

import argparse
import ast
import gc
import hashlib
import importlib.util
import itertools
import json
import logging
import pickle
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def _find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "setup.py").exists() and (p / "utils").exists():
            return p
    raise RuntimeError(f"Project root not found from {start}")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
for _sub in ["2_preprocessing", "3_modeling"]:
    _p = PROJECT_ROOT / _sub
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import optuna  # noqa: E402
from optuna.samplers import GridSampler  # noqa: E402
from optuna.storages import RDBStorage, RetryFailedTrialCallback  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402

from utils.config import (  # noqa: E402
    OUTPUT_DIR,
    TARGET_COL,
    KEY_COL,
    DIE_KEY_COL,
    PROJECT_ROOT as CFG_PROJECT_ROOT,
)
from utils.data import load_all, get_feat_cols, split_xs  # noqa: E402
from meta_features import add_meta_features  # noqa: E402
from modules import preprocess, postprocess  # noqa: E402
from modules.zit import BagZITboostRegressor, BagZITEQLRegressor  # noqa: E402

OUTPUT_DIR = Path(OUTPUT_DIR)
CFG_PROJECT_ROOT = Path(CFG_PROJECT_ROOT)

# ----------------------------------------------------------------------------------------
# Calibration / postprocess grids — identical to the notebook (cell 4). Module constants.
# ----------------------------------------------------------------------------------------
BASELINE_AGG = "sum"                       # BagZIT die->unit is SUM only (die learns unit_y/n_die share)
AGG_CANDIDATES = ("sum",)
POSITION_METHOD = "optuna"                 # no-op (no 'weighted' candidate); kept for signature
POSITION_OPTUNA_N_TRIALS = 50

ZERO_CLIP_RANGE = (0.0001, 0.003)
ZERO_CLIP_N = 30
ZERO_CLIP_LOG_SPACE = True

ISO_KINDS = ["step", "pchip"]
ISO_WEIGHTS = [0.25, 0.5, 0.75, 1.00, 1.25, 1.50]
TAIL_QS = [0.95, 0.975, 0.99]
TAIL_RESID_QS = [0.75, 0.90]
TAIL_GAINS = [0.0, 0.5, 1.0, 1.5, 2.5]
TAIL_POWERS = [1.0, 2.0]
IQR_TOP_KS = [0, 1, 2]
IQR_MARGIN = 1e-6

# ----------------------------------------------------------------------------------------
# Globals populated by set_paths() / load_source_and_data(). Functions below read these
# at call time (same structure as the notebook, which uses module-level names).
# ----------------------------------------------------------------------------------------
SRC_NUM = SRC_VARIANT = MODEL_NAME = None
SOURCE_STUDY_NAME = SOURCE_DB_PATH = SOURCE_BEST_DIR = None
OUT_DIR = BEST_DIR = SEEDS_DIR = SUMMARY_PATH = None
SWEEP_DB = SWEEP_STUDY_NAME = None
N_FOLDS = N_JOBS = None
SEEDS = None

X_train = X_val = X_test = None
uid_train_die = uid_val_die = uid_test_die = None
y_train_unit_s = y_val_unit_s = y_test_unit_s = None
y_train_die = None
xs_train = xs_val = xs_test = None
ys_input = None
ModelClass = None
base_model_params = None
best_tau_pi = None
pp_fixed = None
feat_cols_clean = None
unit_ids_hash = None
source_meta = None


# ====================================================================================
# Common functions (verbatim port from notebook cells 8 / 9 / 11 / 13).
# ====================================================================================
def rmse(pred, y):
    pred = np.asarray(pred, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def clip_nonneg(x):
    return np.clip(np.asarray(x, dtype=float), 0.0, None)


def apply_tau_pi(pred_die, pi_die, tau_pi):
    return np.where(pi_die > tau_pi, 0.0, pred_die)


def unit_rmse(unit_df, y_unit_s):
    p = unit_df.set_index(KEY_COL)["pred"].loc[y_unit_s.index].values
    return rmse(p, y_unit_s.values)


def aligned_unit_pred(unit_df, y_unit_s):
    return unit_df.set_index(KEY_COL)["pred"].loc[y_unit_s.index].values.astype(float)


def tune_unit_postprocess_train_val(
    xs_train, xs_val, xs_test,
    die_pred_train, die_pred_val, die_pred_test,
    y_train_unit_df, y_val_unit_df,
):
    """train/validation 전용 후처리 튜닝 (notebook과 동일). zero_clip은 log-spaced."""
    y_train_s = y_train_unit_df.set_index(KEY_COL)[TARGET_COL]
    y_val_s = y_val_unit_df.set_index(KEY_COL)[TARGET_COL]
    decisions = {}
    val_history = []

    train_unit = postprocess.aggregate(xs_train, die_pred_train, BASELINE_AGG)
    val_unit = postprocess.aggregate(xs_val, die_pred_val, BASELINE_AGG)
    test_unit = postprocess.aggregate(xs_test, die_pred_test, BASELINE_AGG)
    cur_val = unit_rmse(val_unit, y_val_s)
    val_history.append((f"baseline_{BASELINE_AGG}", cur_val))

    agg_res = postprocess.find_best_aggregation(
        xs_train, die_pred_train, y_train_unit_df,
        methods=AGG_CANDIDATES,
        position_method=POSITION_METHOD,
        optuna_n_trials=POSITION_OPTUNA_N_TRIALS,
    )
    best_agg_cand = agg_res["best_method"]
    pos_w_cand = agg_res["pos_weights"]

    if best_agg_cand == BASELINE_AGG:
        best_agg = BASELINE_AGG
        pos_w = None
        decisions["aggregation"] = f"{BASELINE_AGG} train OOF best -> 유지"
    else:
        cand_train = postprocess.aggregate(xs_train, die_pred_train, best_agg_cand, pos_w_cand)
        cand_val = postprocess.aggregate(xs_val, die_pred_val, best_agg_cand, pos_w_cand)
        cand_test = postprocess.aggregate(xs_test, die_pred_test, best_agg_cand, pos_w_cand)
        cand_val_rmse = unit_rmse(cand_val, y_val_s)
        if cand_val_rmse < cur_val:
            train_unit, val_unit, test_unit = cand_train, cand_val, cand_test
            best_agg, pos_w = best_agg_cand, pos_w_cand
            decisions["aggregation"] = f"{best_agg_cand} 채택 ({cur_val:.9f} -> {cand_val_rmse:.9f})"
            cur_val = cand_val_rmse
        else:
            best_agg, pos_w = BASELINE_AGG, None
            decisions["aggregation"] = f"{best_agg_cand} 거절 ({cur_val:.9f} <= {cand_val_rmse:.9f})"
    val_history.append((f"after_agg({best_agg})", cur_val))

    zc_arr = np.logspace(np.log10(ZERO_CLIP_RANGE[0]), np.log10(ZERO_CLIP_RANGE[1]), ZERO_CLIP_N)
    zc_res = postprocess.find_best_zero_clip(train_unit, y_train_unit_df, zc_arr, log_space=ZERO_CLIP_LOG_SPACE)
    cand_zc = zc_res["best_threshold"]
    cand_train = postprocess.apply_zero_clip(train_unit, cand_zc, log_space=ZERO_CLIP_LOG_SPACE)
    cand_val = postprocess.apply_zero_clip(val_unit, cand_zc, log_space=ZERO_CLIP_LOG_SPACE)
    cand_test = postprocess.apply_zero_clip(test_unit, cand_zc, log_space=ZERO_CLIP_LOG_SPACE)
    cand_val_rmse = unit_rmse(cand_val, y_val_s)

    best_zc = None
    if cand_val_rmse < cur_val:
        train_unit, val_unit, test_unit = cand_train, cand_val, cand_test
        best_zc = cand_zc
        decisions["zero_clip"] = f"{cand_zc:.6f} 채택 ({cur_val:.9f} -> {cand_val_rmse:.9f})"
        cur_val = cand_val_rmse
    else:
        decisions["zero_clip"] = f"{cand_zc:.6f} 거절 ({cur_val:.9f} <= {cand_val_rmse:.9f})"
    val_history.append(("after_zero_clip", cur_val))

    train_rmse = unit_rmse(train_unit, y_train_s)
    return {
        "best_agg": best_agg,
        "pos_weights": pos_w,
        "best_zero_clip": best_zc,
        "zero_clip_log_space": ZERO_CLIP_LOG_SPACE,
        "zero_clip_arr": zc_arr,
        "position_method": POSITION_METHOD,
        "agg_rmses": agg_res["rmse_per_method"],
        "decisions": decisions,
        "val_rmse_history": val_history,
        "train_rmse": train_rmse,
        "val_rmse_final": cur_val,
        "final_train_unit": train_unit,
        "final_val_unit": val_unit,
        "final_test_unit": test_unit,
    }


def iqr_stats(pred, y_true=None):
    pred = np.asarray(pred, dtype=float)
    q1, q3 = np.quantile(pred, [0.25, 0.75])
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    mask = pred > upper
    out = {
        "q1": float(q1), "q3": float(q3), "iqr": float(iqr),
        "upper_fence": float(upper),
        "n_upper_outliers": int(mask.sum()),
        "max_pred": float(np.max(pred)),
    }
    if y_true is not None and mask.any():
        yy = np.asarray(y_true, dtype=float)[mask]
        out.update({
            "outlier_true_mean": float(np.mean(yy)),
            "outlier_true_max": float(np.max(yy)),
            "outlier_true_ge_q95": int((yy >= np.quantile(y_true, 0.95)).sum()),
        })
    else:
        out.update({"outlier_true_mean": np.nan, "outlier_true_max": np.nan, "outlier_true_ge_q95": 0})
    return out


def push_top_k_to_iqr(pred, score, top_k=0, margin=1e-6):
    """예측 rank만 사용하는 batch 변환. y_true는 절대 보지 않는다."""
    pred = np.asarray(pred, dtype=float).copy()
    if top_k <= 0:
        return pred
    q1, q3 = np.quantile(pred, [0.25, 0.75])
    upper = q3 + 1.5 * (q3 - q1)
    idx = np.argsort(np.asarray(score, dtype=float))[-int(top_k):]
    pred[idx] = np.maximum(pred[idx], upper + margin)
    return pred


def build_iso_pchip_transform(iso):
    """IsotonicRegression step function을 PCHIP monotonic cubic으로 smoothing."""
    x_knots = np.asarray(iso.X_thresholds_, dtype=float)
    y_knots = np.asarray(iso.y_thresholds_, dtype=float)
    uniq_mask = np.concatenate([[True], np.diff(x_knots) > 0])
    x_knots = x_knots[uniq_mask]
    y_knots = y_knots[uniq_mask]
    if len(x_knots) < 2:
        def _fallback(x):
            return iso.transform(np.asarray(x, dtype=float))
        return _fallback, x_knots, y_knots

    pchip = PchipInterpolator(x_knots, y_knots, extrapolate=False)
    lo, hi = float(x_knots[0]), float(x_knots[-1])

    def _transform(x):
        x = np.asarray(x, dtype=float)
        x_clipped = np.clip(x, lo, hi)
        y_out = pchip(x_clipped)
        return np.clip(y_out, 0.0, None)

    return _transform, x_knots, y_knots


def fit_iso_tail_grid(train_unit, val_unit, test_unit, y_train_s, y_val_s, y_test_s):
    """unit-level 후처리 결과를 raw score로 보고 isotonic/tail 후보를 비교 (notebook과 동일)."""
    raw_train = aligned_unit_pred(train_unit, y_train_s)
    raw_val = aligned_unit_pred(val_unit, y_val_s)
    raw_test = aligned_unit_pred(test_unit, y_test_s)
    y_train = y_train_s.values.astype(float)
    y_val = y_val_s.values.astype(float)
    y_test = y_test_s.values.astype(float)

    rows = []
    best = None

    def add_candidate(name, pred_train, pred_val, pred_test, params, iso_model=None, iso_kind=None, pchip_knots=None):
        nonlocal best
        pred_train = clip_nonneg(pred_train)
        pred_val = clip_nonneg(pred_val)
        pred_test = clip_nonneg(pred_test)
        val_stats = iqr_stats(pred_val, y_val)
        top_idx = int(np.argmax(pred_val))
        rec = {
            "name": name,
            "train_rmse": rmse(pred_train, y_train),
            "val_rmse": rmse(pred_val, y_val),
            "test_rmse": rmse(pred_test, y_test),
            "val_iqr_outliers": val_stats["n_upper_outliers"],
            "val_iqr_upper_fence": val_stats["upper_fence"],
            "val_max_pred": val_stats["max_pred"],
            "val_outlier_true_mean": val_stats["outlier_true_mean"],
            "val_outlier_true_max": val_stats["outlier_true_max"],
            "val_outlier_true_ge_q95": val_stats["outlier_true_ge_q95"],
            "val_top_pred_y_true": float(y_val[top_idx]),
            **params,
        }
        rows.append(rec)
        if best is None or rec["val_rmse"] < best["record"]["val_rmse"]:
            best = {
                "record": rec,
                "train_pred": pred_train, "val_pred": pred_val, "test_pred": pred_test,
                "iso_model": iso_model, "iso_kind": iso_kind, "pchip_knots": pchip_knots,
                "raw_train": raw_train, "raw_val": raw_val, "raw_test": raw_test,
            }

    add_candidate(
        "base_postprocess", raw_train, raw_val, raw_test,
        {"uses_iso": False, "iso_kind": "none", "iso_weight": 0.0,
         "tail_q": np.nan, "tail_resid_q": np.nan,
         "tail_gain": 0.0, "tail_power": np.nan, "iqr_top_k": 0, "tail_resid_scale": 0.0},
    )

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0)
    iso.fit(raw_train, y_train)
    iso_train_step = iso.transform(raw_train)
    iso_val_step = iso.transform(raw_val)
    iso_test_step = iso.transform(raw_test)
    pchip_transform, pchip_x_knots, pchip_y_knots = build_iso_pchip_transform(iso)
    iso_train_pchip = pchip_transform(raw_train)
    iso_val_pchip = pchip_transform(raw_val)
    iso_test_pchip = pchip_transform(raw_test)

    iso_table = {
        "step": (iso_train_step, iso_val_step, iso_test_step),
        "pchip": (iso_train_pchip, iso_val_pchip, iso_test_pchip),
    }

    for iso_kind, iso_weight, tail_q, tail_resid_q, tail_gain, tail_power, iqr_top_k in itertools.product(
        ISO_KINDS, ISO_WEIGHTS, TAIL_QS, TAIL_RESID_QS, TAIL_GAINS, TAIL_POWERS, IQR_TOP_KS
    ):
        iso_train_arr, iso_val_arr, iso_test_arr = iso_table[iso_kind]

        base_train = raw_train + iso_weight * (iso_train_arr - raw_train)
        base_val = raw_val + iso_weight * (iso_val_arr - raw_val)
        base_test = raw_test + iso_weight * (iso_test_arr - raw_test)

        tail_start = float(np.quantile(raw_train, tail_q))
        tail_hi = float(np.quantile(raw_train, 0.999))
        tail_denom = max(tail_hi - tail_start, 1e-12)
        tail_mask = raw_train >= tail_start
        if int(tail_mask.sum()) >= 3:
            resid = y_train[tail_mask] - base_train[tail_mask]
            tail_resid_scale = max(0.0, float(np.quantile(resid, tail_resid_q)))
        else:
            tail_resid_scale = 0.0

        def transform(raw, base, ts=tail_start, td=tail_denom, tp=tail_power, tg=tail_gain, trs=tail_resid_scale):
            ramp = np.clip((raw - ts) / td, 0.0, None) ** tp
            return base + tg * trs * ramp

        pred_train = transform(raw_train, base_train)
        pred_val = transform(raw_val, base_val)
        pred_test = transform(raw_test, base_test)

        pred_train = push_top_k_to_iqr(pred_train, raw_train, iqr_top_k, IQR_MARGIN)
        pred_val = push_top_k_to_iqr(pred_val, raw_val, iqr_top_k, IQR_MARGIN)
        pred_test = push_top_k_to_iqr(pred_test, raw_test, iqr_top_k, IQR_MARGIN)

        name = f"iso{iso_kind}_w{iso_weight:g}_q{tail_q:g}_rq{tail_resid_q:g}_g{tail_gain:g}_p{tail_power:g}_iqr{iqr_top_k}"
        add_candidate(
            name, pred_train, pred_val, pred_test,
            {
                "uses_iso": True, "iso_kind": iso_kind, "iso_weight": float(iso_weight),
                "tail_q": float(tail_q), "tail_resid_q": float(tail_resid_q),
                "tail_gain": float(tail_gain), "tail_power": float(tail_power),
                "iqr_top_k": int(iqr_top_k),
                "tail_start": tail_start, "tail_denom": tail_denom,
                "tail_resid_scale": float(tail_resid_scale),
            },
            iso_model=iso, iso_kind=iso_kind,
            pchip_knots=(pchip_x_knots, pchip_y_knots) if iso_kind == "pchip" else None,
        )

    cand = pd.DataFrame(rows).sort_values("val_rmse").reset_index(drop=True)
    iqr12 = cand[cand["val_iqr_outliers"].between(1, 2)].copy()
    best_iqr12 = iqr12.iloc[0].to_dict() if len(iqr12) else None

    best["candidates"] = cand
    best["best_iqr12"] = best_iqr12
    best["raw_train"] = raw_train
    best["raw_val"] = raw_val
    best["raw_test"] = raw_test
    return best


def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def build_die_df(uid, die_id, position, pi, mu, pred_raw, pred_taupi, y_unit_s):
    out = pd.DataFrame({
        KEY_COL: uid,
        DIE_KEY_COL: die_id,
        "position": position,
        "pi": pi,
        "one_minus_pi": 1.0 - pi,
        "mu": mu,
        "pred_raw": pred_raw,
        "pred_taupi": pred_taupi,
    })
    out[TARGET_COL] = out[KEY_COL].map(y_unit_s)
    return out


def build_unit_output(y_unit_s, pred, pred_base):
    return pd.DataFrame({
        KEY_COL: y_unit_s.index.values,
        "pred": np.asarray(pred, dtype=float),
        "pred_base_postprocess": np.asarray(pred_base, dtype=float),
        TARGET_COL: y_unit_s.values,
    })


def serializable_calibrator(best_cal):
    rec = dict(best_cal["record"])
    iso = best_cal.get("iso_model")
    if iso is not None:
        rec["iso_x_thresholds"] = iso.X_thresholds_.tolist()
        rec["iso_y_thresholds"] = iso.y_thresholds_.tolist()
    pk = best_cal.get("pchip_knots")
    if pk is not None:
        rec["pchip_x_knots"] = pk[0].tolist()
        rec["pchip_y_knots"] = pk[1].tolist()
    return rec


def save_result_artifacts(res, target_dir):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    best_cal = res["calibration"]
    pp_res = res["postprocess"]

    fold_payload = {
        "fold_models": res["fold_models"],
        "feature_names": feat_cols_clean,
        "model_name": MODEL_NAME,
        "n_folds": N_FOLDS,
        "seed": int(res["seed"]),
        "fold_model_seeds": res["fold_model_seeds"],
        "em_history_per_fold": res["em_history_per_fold"],
        "source_best_dir": str(SOURCE_BEST_DIR),
        "calibrator": {
            "record": dict(best_cal["record"]),
            "iso_model": best_cal.get("iso_model"),
            "iso_kind": best_cal.get("iso_kind"),
            "pchip_knots": best_cal.get("pchip_knots"),
        },
    }
    with open(target_dir / "fold_models.pkl", "wb") as f:
        pickle.dump(fold_payload, f)

    raw_train = best_cal["raw_train"]
    raw_val = best_cal["raw_val"]
    raw_test = best_cal["raw_test"]
    build_unit_output(y_train_unit_s, best_cal["train_pred"], raw_train).to_csv(target_dir / "oof_unit.csv", index=False)
    build_unit_output(y_val_unit_s, best_cal["val_pred"], raw_val).to_csv(target_dir / "val_unit.csv", index=False)
    build_unit_output(y_test_unit_s, best_cal["test_pred"], raw_test).to_csv(target_dir / "test_unit.csv", index=False)

    build_die_df(
        uid_train_die, xs_train[DIE_KEY_COL].values, xs_train["position"].values,
        res["oof_die_pi"], res["oof_die_mu"], res["oof_die_pred_raw"], res["oof_die_pred_taupi"],
        y_train_unit_s,
    ).to_csv(target_dir / "oof_die.csv", index=False)
    build_die_df(
        uid_val_die, xs_val[DIE_KEY_COL].values, xs_val["position"].values,
        res["val_die_pi"], res["val_die_mu"], res["val_die_pred_raw"], res["val_die_pred_taupi"],
        y_val_unit_s,
    ).to_csv(target_dir / "val_die.csv", index=False)
    build_die_df(
        uid_test_die, xs_test[DIE_KEY_COL].values, xs_test["position"].values,
        res["test_die_pi"], res["test_die_mu"], res["test_die_pred_raw"], res["test_die_pred_taupi"],
        y_test_unit_s,
    ).to_csv(target_dir / "test_die.csv", index=False)

    best_cal["candidates"].to_csv(target_dir / "calibration_candidates.csv", index=False)

    meta = {
        "exp_id": f'bag-zit-{SRC_NUM}-seed-sweep-seed{res["seed"]}',
        "model_name": MODEL_NAME,
        "source_best_dir": str(SOURCE_BEST_DIR),
        "source_exp_id": source_meta.get("exp_id"),
        "seed": int(res["seed"]),
        "fold_model_seeds": res["fold_model_seeds"],
        "best_params_resolved": res["best_full_params"],
        "best_tau_pi": best_tau_pi,
        "feature_names": feat_cols_clean,
        "n_features": len(feat_cols_clean),
        "n_folds": N_FOLDS,
        "unit_ids_hash": unit_ids_hash,
        "n_units_train": int(len(y_train_unit_s)),
        "n_units_val": int(len(y_val_unit_s)),
        "n_units_test": int(len(y_test_unit_s)),
        "effective_pp_params": pp_fixed,
        "val_rmse": float(res["summary"]["val_rmse"]),
        "base_val_rmse": float(res["summary"]["base_val_rmse"]),
        "test_rmse": float(res["summary"]["test_rmse"]),
        "base_test_rmse": float(res["summary"]["base_test_rmse"]),
        "postprocess": {
            "best_agg": pp_res["best_agg"],
            "pos_weights": pp_res["pos_weights"].tolist() if pp_res["pos_weights"] is not None else None,
            "best_zero_clip": pp_res["best_zero_clip"],
            "zero_clip_log_space": pp_res["zero_clip_log_space"],
            "zero_clip_arr": pp_res["zero_clip_arr"].tolist(),
            "position_method": pp_res["position_method"],
            "agg_rmses": {k: float(v) for k, v in pp_res["agg_rmses"].items()},
            "train_rmse": float(pp_res["train_rmse"]),
            "val_rmse_final": float(pp_res["val_rmse_final"]),
            "decisions": pp_res["decisions"],
        },
        "calibration": serializable_calibrator(best_cal),
        "best_iqr12_candidate": best_cal.get("best_iqr12"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(target_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=json_default)

    with open(target_dir / "summary_record.json", "w", encoding="utf-8") as f:
        json.dump(res["summary"], f, indent=2, ensure_ascii=False, default=json_default)

    print(f"[저장 완료] {target_dir}")


def make_folds(seed):
    unique_units = y_train_unit_s.index.values
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=int(seed))
    return unique_units, list(kf.split(unique_units))


def params_for_seed(seed, fold_idx):
    p = dict(base_model_params)
    fold_seed = int(seed) * 1009 + int(fold_idx)
    p["random_state"] = fold_seed
    p["n_jobs"] = N_JOBS
    p["verbose"] = -1
    p["device"] = "cpu"
    p["em_tol"] = 1e-7
    return p, fold_seed


def fit_one_seed(seed):
    seed = int(seed)
    unique_units, folds = make_folds(seed)

    n_train_die = len(X_train)
    n_val_die = len(X_val)
    n_test_die = len(X_test)

    oof_die_pi = np.full(n_train_die, np.nan)
    oof_die_mu = np.full(n_train_die, np.nan)
    oof_die_pred_raw = np.full(n_train_die, np.nan)

    val_die_pi = np.zeros(n_val_die)
    val_die_mu = np.zeros(n_val_die)
    val_die_pred_raw = np.zeros(n_val_die)

    test_die_pi = np.zeros(n_test_die)
    test_die_mu = np.zeros(n_test_die)
    test_die_pred_raw = np.zeros(n_test_die)

    fold_models = []
    fold_model_seeds = []
    em_history_per_fold = []

    t0 = time.time()
    print(f"\n=== seed {seed} ===")
    for fold_idx, (tr_uidx, vl_uidx) in enumerate(folds):
        tr_units = unique_units[tr_uidx]
        vl_units = unique_units[vl_uidx]
        tr_mask = np.isin(uid_train_die, tr_units)
        vl_mask = np.isin(uid_train_die, vl_units)

        params, fold_seed = params_for_seed(seed, fold_idx)
        model = ModelClass(**params)
        model.fit(X_train[tr_mask], y_train_die[tr_mask], unit_id=uid_train_die[tr_mask])

        pi_vl, mu_vl, _ = model.predict_components(X_train[vl_mask])
        pred_vl = clip_nonneg((1.0 - pi_vl) * mu_vl)
        oof_die_pi[vl_mask] = pi_vl
        oof_die_mu[vl_mask] = mu_vl
        oof_die_pred_raw[vl_mask] = pred_vl

        pi_val, mu_val, _ = model.predict_components(X_val)
        val_die_pi += pi_val / N_FOLDS
        val_die_mu += mu_val / N_FOLDS
        val_die_pred_raw += clip_nonneg((1.0 - pi_val) * mu_val) / N_FOLDS

        pi_test, mu_test, _ = model.predict_components(X_test)
        test_die_pi += pi_test / N_FOLDS
        test_die_mu += mu_test / N_FOLDS
        test_die_pred_raw += clip_nonneg((1.0 - pi_test) * mu_test) / N_FOLDS

        fold_models.append(model)
        fold_model_seeds.append(fold_seed)
        em_history_per_fold.append(getattr(model, "em_history_", None))
        print(f"  fold {fold_idx + 1}/{N_FOLDS} done, model_seed={fold_seed}, elapsed={time.time() - t0:.0f}s")

    if np.isnan(oof_die_pred_raw).any():
        raise RuntimeError(f"seed {seed}: OOF die prediction has NaN")

    oof_die_pred_taupi = apply_tau_pi(oof_die_pred_raw, oof_die_pi, best_tau_pi)
    val_die_pred_taupi = apply_tau_pi(val_die_pred_raw, val_die_pi, best_tau_pi)
    test_die_pred_taupi = apply_tau_pi(test_die_pred_raw, test_die_pi, best_tau_pi)

    pp_res = tune_unit_postprocess_train_val(
        xs_train=xs_train, xs_val=xs_val, xs_test=xs_test,
        die_pred_train=oof_die_pred_taupi,
        die_pred_val=val_die_pred_taupi,
        die_pred_test=test_die_pred_taupi,
        y_train_unit_df=ys_input["train"],
        y_val_unit_df=ys_input["validation"],
    )

    cal = fit_iso_tail_grid(
        pp_res["final_train_unit"], pp_res["final_val_unit"], pp_res["final_test_unit"],
        y_train_unit_s, y_val_unit_s, y_test_unit_s,
    )
    best_rec = cal["record"]
    best_iqr12 = cal.get("best_iqr12")

    base_test_rmse = unit_rmse(pp_res["final_test_unit"], y_test_unit_s)

    elapsed = time.time() - t0
    summary = {
        "seed": seed,
        "elapsed_sec": float(elapsed),
        "base_train_rmse": float(pp_res["train_rmse"]),
        "base_val_rmse": float(pp_res["val_rmse_final"]),
        "base_test_rmse": float(base_test_rmse),
        "val_rmse": float(best_rec["val_rmse"]),
        "test_rmse": float(best_rec["test_rmse"]),
        "train_rmse": float(best_rec["train_rmse"]),
        "calibration_name": best_rec["name"],
        "uses_iso": bool(best_rec["uses_iso"]),
        "iso_kind": str(best_rec.get("iso_kind", "none")),
        "iso_weight": float(best_rec.get("iso_weight", 0.0)),
        "tail_q": float(best_rec.get("tail_q", np.nan)) if not pd.isna(best_rec.get("tail_q", np.nan)) else np.nan,
        "tail_resid_q": float(best_rec.get("tail_resid_q", np.nan)) if not pd.isna(best_rec.get("tail_resid_q", np.nan)) else np.nan,
        "tail_gain": float(best_rec.get("tail_gain", 0.0)),
        "tail_power": float(best_rec.get("tail_power", np.nan)) if not pd.isna(best_rec.get("tail_power", np.nan)) else np.nan,
        "tail_resid_scale": float(best_rec.get("tail_resid_scale", 0.0)),
        "iqr_top_k": int(best_rec.get("iqr_top_k", 0)),
        "val_iqr_outliers": int(best_rec["val_iqr_outliers"]),
        "val_iqr_upper_fence": float(best_rec["val_iqr_upper_fence"]),
        "val_max_pred": float(best_rec["val_max_pred"]),
        "val_outlier_true_mean": float(best_rec["val_outlier_true_mean"]) if not pd.isna(best_rec["val_outlier_true_mean"]) else np.nan,
        "val_outlier_true_max": float(best_rec["val_outlier_true_max"]) if not pd.isna(best_rec["val_outlier_true_max"]) else np.nan,
        "val_outlier_true_ge_q95": int(best_rec["val_outlier_true_ge_q95"]),
        "val_top_pred_y_true": float(best_rec["val_top_pred_y_true"]),
        "best_iqr12_val_rmse": float(best_iqr12["val_rmse"]) if best_iqr12 else np.nan,
        "best_iqr12_name": best_iqr12["name"] if best_iqr12 else None,
        "postprocess_best_agg": pp_res["best_agg"],
        "postprocess_best_zero_clip": pp_res["best_zero_clip"],
    }
    print(f'[seed {seed}] base_val={summary["base_val_rmse"]:.9f}, best_val={summary["val_rmse"]:.9f}, '
          f'test={summary["test_rmse"]:.9f}, cal={summary["calibration_name"]}, '
          f'iqr_outliers={summary["val_iqr_outliers"]}, elapsed={elapsed:.0f}s')

    best_full_params, _ = params_for_seed(seed, 0)
    return {
        "seed": seed,
        "summary": summary,
        "postprocess": pp_res,
        "calibration": cal,
        "fold_models": fold_models,
        "fold_model_seeds": fold_model_seeds,
        "em_history_per_fold": em_history_per_fold,
        "best_full_params": best_full_params,
        "oof_die_pi": oof_die_pi, "oof_die_mu": oof_die_mu,
        "oof_die_pred_raw": oof_die_pred_raw, "oof_die_pred_taupi": oof_die_pred_taupi,
        "val_die_pi": val_die_pi, "val_die_mu": val_die_mu,
        "val_die_pred_raw": val_die_pred_raw, "val_die_pred_taupi": val_die_pred_taupi,
        "test_die_pi": test_die_pi, "test_die_mu": test_die_mu,
        "test_die_pred_raw": test_die_pred_raw, "test_die_pred_taupi": test_die_pred_taupi,
    }


def concat_iqr_outlier_stats(cal):
    pred = np.concatenate([cal["train_pred"], cal["val_pred"], cal["test_pred"]])
    q1, q3 = np.quantile(pred, [0.25, 0.75])
    upper_fence = float(q3 + 1.5 * (q3 - q1))
    n_out = int((pred > upper_fence).sum())
    return {
        "concat_n_total": int(len(pred)),
        "concat_iqr_upper_fence": upper_fence,
        "concat_iqr_outliers": n_out,
        "concat_has_outlier": bool(n_out > 0),
        "concat_max_pred": float(pred.max()),
    }


# ====================================================================================
# Source / path resolution + data load (notebook cells 4 + 6) + Optuna harness.
# ====================================================================================
def _parse_user_attr(v):
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return v
    return v


def _src_db_path(num, variant):
    infix = f"{variant}-" if variant else ""
    name = f"bag-zit-{infix}final-{num}"
    return OUTPUT_DIR / "01_zit" / "bag_zit" / num / f"optuna_jh_{name}.db", name


def _best_value_of(db_path, study_name):
    """완료 trial 중 최저 value. 완료 trial 없으면 None."""
    try:
        st = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path.as_posix()}")
        completed = [t for t in st.get_trials(deepcopy=False)
                     if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
        if not completed:
            return None
        return min(t.value for t in completed)
    except Exception:
        return None


def resolve_src_num(args):
    if args.src_num != "auto":
        return args.src_num
    # auto: best (lowest best_value) among 002/003/004/005 that have completed trials locally.
    # 002를 포함하므로 003/004/005가 아직 best가 없어도 지금 바로 002로 돌릴 수 있고,
    # 나중에 003/004/005가 002를 이기면 자동으로 그쪽을 고른다.
    cands = []
    for num in ["002", "003", "004", "005"]:
        db, name = _src_db_path(num, args.variant)
        if db.exists():
            bv = _best_value_of(db, name)
            if bv is not None:
                cands.append((bv, num))
    if not cands:
        raise RuntimeError(
            "--src-num auto: 002/003/004/005 중 완료 trial이 있는 eql study가 로컬에 없습니다. "
            "해당 study DB를 이 머신에 두거나 --src-num 으로 명시하세요."
        )
    cands.sort()
    best_bv, best_num = cands[0]
    print(f"[src auto] 후보 {[(n, round(b, 9)) for b, n in cands]} -> best={best_num} (best_value={best_bv:.9f})")
    return best_num


def set_paths(args, src_num):
    """SRC_*/OUT_* 경로 전역을 채운다 (데이터 로드 없이 summarize에서도 사용)."""
    variant = args.variant
    infix = f"{variant}-" if variant else ""
    src_db, src_study = _src_db_path(src_num, variant)
    model_name = f"bag_zit_{variant}" if variant else "bag_zit"

    tag = f"_{args.out_tag}" if args.out_tag else ""
    out_dir = OUTPUT_DIR / "01_zit" / "bag_zit" / f"seed_sweep_{src_num}{tag}"
    sweep_db = out_dir / f"optuna_jh_seedsweep_{infix}{src_num}.db"
    sweep_study = f"seedsweep-{infix}final-{src_num}"

    globals().update({
        "SRC_NUM": src_num,
        "SRC_VARIANT": variant,
        "MODEL_NAME": model_name,
        "SOURCE_STUDY_NAME": src_study,
        "SOURCE_DB_PATH": src_db,
        "SOURCE_BEST_DIR": src_db,
        "OUT_DIR": out_dir,
        "BEST_DIR": out_dir / "best",
        "SEEDS_DIR": out_dir / "seeds",
        "SUMMARY_PATH": out_dir / "seed_sweep_summary.csv",
        "SWEEP_DB": sweep_db,
        "SWEEP_STUDY_NAME": sweep_study,
        "N_FOLDS": args.n_folds,
        "N_JOBS": args.n_jobs,
        "SEEDS": args.seeds,
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "seeds").mkdir(parents=True, exist_ok=True)


def load_source_and_data(args):
    """notebook cell 6 — source best params + data 로드 후 전역에 채운다."""
    storage = f"sqlite:///{SOURCE_DB_PATH.as_posix()}"
    if not SOURCE_DB_PATH.exists():
        raise FileNotFoundError(f"source study db 없음: {SOURCE_DB_PATH}")
    study = optuna.load_study(study_name=SOURCE_STUDY_NAME, storage=storage)
    best = study.best_trial
    ua = {k: _parse_user_attr(v) for k, v in study.user_attrs.items()}

    src_model = str(ua.get("model") or "")
    is_eql = ("eql" in src_model.lower()) or ("deviance" in src_model.lower()) or (SRC_VARIANT == "eql")
    model_class = BagZITEQLRegressor if is_eql else BagZITboostRegressor
    print(f'[model] source="{src_model}" -> {model_class.__name__} 재현')

    bp = dict(best.params)
    tau_pi = float(bp.pop("tau_pi"))

    clip_attr = ua.get("clip_y_extreme", ua.get("CLIP_Y_EXTREME", True))
    if isinstance(clip_attr, str):
        clip_attr = clip_attr.lower() in {"true", "1", "yes"}
    clip_y_extreme = bool(clip_attr)

    src_meta = {
        "exp_id": ua.get("exp_id"),
        "model_name": MODEL_NAME,
        "best_trial_number": best.number,
        "best_oof_rmse": float(best.value),
        "best_params_resolved": dict(bp),
        "best_tau_pi": tau_pi,
        "effective_pp_params": dict(ua.get("pp_fixed") or {}),
        "feature_names": None, "n_features": None, "unit_ids_hash": None,
        "study_meta": {"CLIP_Y_EXTREME": clip_y_extreme, **ua},
    }
    base_params = dict(src_meta["best_params_resolved"])
    for k in ["random_state", "n_jobs", "verbose", "device", "em_tol"]:
        base_params.pop(k, None)
    pp_params = dict(src_meta["effective_pp_params"])

    print("[기준 source study]")
    print(f'  exp_id      : {src_meta.get("exp_id")}')
    print(f'  best_trial# : {src_meta.get("best_trial_number")}')
    print(f'  best_oof    : {src_meta.get("best_oof_rmse"):.9f}')
    print(f"  tau_pi      : {tau_pi:.9f}")
    print(f"  clip extreme: {clip_y_extreme}")
    print(f"  pp_fixed    : {pp_params}")

    xs, ys = load_all()
    feat_cols = get_feat_cols(xs)
    xs_dict = split_xs(xs)

    ys_in = {k: v.copy() for k, v in ys.items()}
    if clip_y_extreme:
        y_raw = ys_in["train"][TARGET_COL]
        second_max = y_raw[y_raw < y_raw.max()].max()
        n_clipped = int((y_raw >= y_raw.max()).sum())
        ys_in["train"][TARGET_COL] = y_raw.clip(upper=second_max)
        print(f"[CLIP_Y_EXTREME] train max -> {second_max:.9f}, clipped={n_clipped}")

    # impute 결정. eql study는 median으로 튜닝됐으나 mmap 워커가 impute_stage23를 안 남겼다.
    #   --impute auto = (eql 이거나 meta가 median이면) median, 아니면 mean.
    src_uses_median = "median" in str(ua.get("impute_stage23") or "").lower()
    if args.impute == "median":
        apply_median = True
    elif args.impute == "mean":
        apply_median = False
    else:  # auto
        apply_median = is_eql or src_uses_median
    if apply_median:
        import importlib.util as _ilu, cleaning as _cleaning
        worker_fname = "02_bag_zit_eql_parallel_hpo.py" if is_eql else "02_bag_zit_parallel_hpo.py"
        wpath = CFG_PROJECT_ROOT / "3_modeling" / "01_zit" / worker_fname
        spec = _ilu.spec_from_file_location("bag_zit_worker_impute", wpath)
        wmod = _ilu.module_from_spec(spec); spec.loader.exec_module(wmod)
        _cleaning.impute_spatial = wmod._impute_spatial_median
        print(f"[impute patch] median fallback 적용 ({worker_fname}::_impute_spatial_median, "
              f"impute={args.impute}, src_meta_median={src_uses_median})")
    else:
        print(f"[impute] mean fallback (impute={args.impute}, src_meta_median={src_uses_median})")

    pp = preprocess.run(xs, ys_in, feat_cols, xs_dict, params=pp_params)
    xs_tr, xs_vl, xs_te = pp["xs_train"], pp["xs_val"], pp["xs_test"]
    fc = pp["feat_cols"]
    fc = add_meta_features(xs_tr, xs_vl, xs_te, fc, position_mode="raw", use_die_xy=True)

    Xtr = xs_tr[fc].values.astype(np.float64)
    Xvl = xs_vl[fc].values.astype(np.float64)
    Xte = xs_te[fc].values.astype(np.float64)

    ytr_unit = ys_in["train"].set_index(KEY_COL)[TARGET_COL]
    yvl_unit = ys_in["validation"].set_index(KEY_COL)[TARGET_COL]
    yte_unit = ys_in["test"].set_index(KEY_COL)[TARGET_COL]
    ytr_die = xs_tr[KEY_COL].map(ytr_unit).values.astype(np.float64)

    uid_hash = hashlib.sha1(",".join(map(str, ys_in["train"][KEY_COL].unique())).encode()).hexdigest()
    print(f"[data] X_train={Xtr.shape}, X_val={Xvl.shape}, X_test={Xte.shape}, "
          f"units train={len(ytr_unit):,}, val={len(yvl_unit):,}, test={len(yte_unit):,}")

    globals().update({
        "ModelClass": model_class,
        "base_model_params": base_params,
        "best_tau_pi": tau_pi,
        "pp_fixed": pp_params,
        "source_meta": src_meta,
        "xs_train": xs_tr, "xs_val": xs_vl, "xs_test": xs_te,
        "ys_input": ys_in,
        "feat_cols_clean": fc,
        "X_train": Xtr, "X_val": Xvl, "X_test": Xte,
        "uid_train_die": xs_tr[KEY_COL].values,
        "uid_val_die": xs_vl[KEY_COL].values,
        "uid_test_die": xs_te[KEY_COL].values,
        "y_train_unit_s": ytr_unit, "y_val_unit_s": yvl_unit, "y_test_unit_s": yte_unit,
        "y_train_die": ytr_die,
        "unit_ids_hash": uid_hash,
    })


def _to_py(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def make_study(args):
    storage = RDBStorage(
        url=f"sqlite:///{SWEEP_DB.as_posix()}",
        engine_kwargs={"connect_args": {"timeout": args.db_timeout}},
        heartbeat_interval=args.heartbeat_interval,
        grace_period=args.grace_period,
        failed_trial_callback=RetryFailedTrialCallback(max_retry=args.max_retry),
    )
    sampler = GridSampler({"seed": list(SEEDS)})
    study = optuna.create_study(
        study_name=SWEEP_STUDY_NAME,
        storage=storage,
        sampler=sampler,
        direction="minimize",
        load_if_exists=not args.no_resume,
    )
    study.set_user_attr("source_study", SOURCE_STUDY_NAME)
    study.set_user_attr("source_db", str(SOURCE_DB_PATH))
    study.set_user_attr("seeds", str(list(SEEDS)))
    study.set_user_attr("n_folds", N_FOLDS)
    study.set_user_attr("model_name", MODEL_NAME)
    study.set_user_attr("worker_id_last", args.worker_id)
    return study


def build_objective(args):
    def objective(trial: optuna.Trial) -> float:
        seed = int(trial.suggest_categorical("seed", list(SEEDS)))
        trial.set_user_attr("worker_id", args.worker_id)
        trial.set_user_attr("pid", os.getpid())

        # GridSampler가 grid 소진 시 이미 끝난(또는 다른 워커가 RUNNING 중인) seed를 재배정한다.
        # 그 경우 이 워커가 새로 할 seed가 없다는 뜻 -> fit하지 말고 이 워커만 멈춘다(다른 워커는 계속).
        # 중복 fit(~1.75h 낭비) 방지. 시작 시 단 한 번 빠르게 prune.
        _study = trial.study
        _taken = {
            t.params.get("seed")
            for t in _study.get_trials(deepcopy=False)
            if t.number != trial.number
            and t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.RUNNING)
        }
        if seed in _taken:
            trial.set_user_attr("skipped_duplicate_or_exhausted", True)
            _study.stop()
            raise optuna.TrialPruned()

        res = fit_one_seed(seed)
        concat = concat_iqr_outlier_stats(res["calibration"])
        res["summary"].update(concat)

        save_result_artifacts(res, SEEDS_DIR / f"seed_{seed}")

        for k, v in res["summary"].items():
            trial.set_user_attr(k, _to_py(v))
        val = float(res["summary"]["val_rmse"])
        print(f'  [concat IQR] n={concat["concat_n_total"]}, '
              f'상단 이상치={concat["concat_iqr_outliers"]}개, max={concat["concat_max_pred"]:.6f}')

        del res
        gc.collect()
        return val

    return objective


def run_summarize(args):
    """DB의 완료 trial로 summary CSV 재생성 + best seed -> best/ 복사."""
    if not SWEEP_DB.exists():
        raise FileNotFoundError(f"sweep db 없음: {SWEEP_DB}")
    study = optuna.load_study(study_name=SWEEP_STUDY_NAME, storage=f"sqlite:///{SWEEP_DB.as_posix()}")
    rows = []
    for t in study.get_trials(deepcopy=False):
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        rec = {k: v for k, v in t.user_attrs.items() if k not in ("pid",)}
        rec.setdefault("seed", t.params.get("seed"))
        rec["trial_number"] = t.number
        rows.append(rec)
    if not rows:
        print("[summarize] 완료된 seed trial이 아직 없습니다.")
        return
    df = pd.DataFrame(rows).sort_values("val_rmse").reset_index(drop=True)
    df.to_csv(SUMMARY_PATH, index=False)
    print(f"[summarize] {len(df)} seed -> {SUMMARY_PATH}")
    print(df[["seed", "val_rmse", "test_rmse", "base_val_rmse", "calibration_name", "val_iqr_outliers"]].head(15).to_string(index=False))

    best_seed = int(df.iloc[0]["seed"])
    src = SEEDS_DIR / f"seed_{best_seed}"
    if src.exists():
        if BEST_DIR.exists():
            shutil.rmtree(BEST_DIR)
        shutil.copytree(src, BEST_DIR)
        print(f"[summarize] best seed={best_seed} (val_rmse={df.iloc[0]['val_rmse']:.9f}) -> {BEST_DIR}")
    else:
        print(f"[summarize] ⚠ best seed={best_seed} 산출물 폴더 없음: {src}")


def parse_seeds(s: str):
    s = s.strip()
    if "-" in s and "," not in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="BagZIT(-EQL) parallel seed-sweep worker.")
    ap.add_argument("--src-num", default="auto",
                    help="소스 study 번호 002/003/004/005 또는 'auto'(002/003/004/005 중 best).")
    ap.add_argument("--variant", default="eql", choices=["eql", ""],
                    help="'eql'=bag-zit-eql-final-* / ''=bag-zit-final-*.")
    ap.add_argument("--seeds", type=parse_seeds, default="1000-1029",
                    help="'1000-1029'(범위) 또는 '1000,1001,...'(목록). 기본 30개.")
    ap.add_argument("--impute", default="auto", choices=["auto", "median", "mean"],
                    help="결측 fallback. auto=eql이면 median. (precompute/eql 워커는 median)")
    ap.add_argument("--out-tag", default=None, help="OUT_DIR 접미사(여러 run 분리용).")
    ap.add_argument("--worker-id", default=f"pid{os.getpid()}")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--n-trials", type=int, default=100000)
    ap.add_argument("--timeout-hours", type=float, default=0.0)
    ap.add_argument("--end-at", default=None,
                    help="ISO 종료 시각 e.g. '2026-06-08T05:00'. 워커 전체 동일 값 권장.")
    ap.add_argument("--db-timeout", type=float, default=120.0)
    ap.add_argument("--heartbeat-interval", type=int, default=300)
    ap.add_argument("--grace-period", type=int, default=1800)
    ap.add_argument("--max-retry", type=int, default=1)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--summarize", action="store_true",
                    help="fit 없이 DB로 summary CSV 재생성 + best 복사.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.getLogger("lightgbm").setLevel(logging.ERROR)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    src_num = resolve_src_num(args)
    set_paths(args, src_num)

    print(f"PROJECT_ROOT={PROJECT_ROOT}")
    print(f"SRC_NUM={SRC_NUM}, SRC_VARIANT={SRC_VARIANT!r}, MODEL_NAME={MODEL_NAME}")
    print(f"SOURCE_STUDY={SOURCE_STUDY_NAME}")
    print(f"SOURCE_DB={SOURCE_DB_PATH}")
    print(f"OUT_DIR={OUT_DIR}")
    print(f"SWEEP_DB={SWEEP_DB}, study={SWEEP_STUDY_NAME}")
    print(f"SEEDS(n={len(SEEDS)})={SEEDS[:3]}..{SEEDS[-1]}, N_FOLDS={N_FOLDS}, N_JOBS={N_JOBS}")

    if args.summarize:
        run_summarize(args)
        return

    load_source_and_data(args)
    study = make_study(args)
    print(f"study={study.study_name}, existing_trials={len(study.trials)}")

    if args.n_trials <= 0:
        print("[skip optimize] n_trials <= 0 (resolve/preprocess sanity only)")
        return

    # 모든 seed가 완료되면 더는 grid가 없으니 stop (GridSampler 재평가 방지).
    def _stop_when_grid_done(st, _trial):
        n_done = sum(1 for t in st.get_trials(deepcopy=False)
                     if t.state == optuna.trial.TrialState.COMPLETE)
        if n_done >= len(SEEDS):
            st.stop()

    if args.end_at:
        end_dt = datetime.fromisoformat(args.end_at)
        remaining = (end_dt - datetime.now()).total_seconds()
        if remaining <= 0:
            print(f"[end-at] {end_dt.isoformat()} already passed; nothing to do")
            return
        timeout = remaining
        print(f"[end-at] target={end_dt.isoformat()} -> timeout={timeout:.0f}s ({timeout/3600:.2f}h)")
    else:
        timeout = None if args.timeout_hours <= 0 else args.timeout_hours * 3600

    t_start = time.time()
    study.optimize(
        build_objective(args),
        n_trials=args.n_trials,
        timeout=timeout,
        n_jobs=1,
        callbacks=[_stop_when_grid_done],
    )
    elapsed = time.time() - t_start
    n_done = sum(1 for t in study.get_trials(deepcopy=False) if t.state == optuna.trial.TrialState.COMPLETE)
    print(f"[sweep done] worker={args.worker_id}, elapsed={elapsed:.0f}s, completed_seeds={n_done}/{len(SEEDS)}")
    try:
        print(f"best_value={study.best_value:.9f}, best_seed={study.best_trial.params.get('seed')}")
    except ValueError:
        print("best_value unavailable: no completed trials")


if __name__ == "__main__":
    main()
