"""
Parallel HPO worker for BagZIT + EQL phi update.

This file intentionally does not modify 3_modeling/modules/zit.py.  It defines a
local subclass of BagZITboostRegressor that changes only the phi M-step target:

  current module:  phi_target = (y - mu)^2 / mu^zeta
  this worker:     phi_target = Tweedie unit deviance D_zeta(y, mu)

The rest is the BagZIT idea from 02_bag_zit.ipynb:
  - unit-level y is allocated to dies inside BagZIT at each EM iteration
  - die predictions are summed to restore unit predictions
  - tau_pi is a die-level structural-zero gate

Run command (PowerShell):

  # One-time init. Enqueues bag-zit-final-001 best as anchor, then exits.
  python 3_modeling/01_zit/hp/02_bag_zit_eql_parallel_hpo.py --enqueue-anchor --n-trials 0

  # Open 3 terminals. Tune --n-jobs so 3 * n_jobs <= physical thread count.
  python 3_modeling/01_zit/hp/02_bag_zit_eql_parallel_hpo.py --worker-id w1 --n-trials 2000 --n-jobs 4 --n-startup-trials 12 --end-at 2026-06-01T09:00 > bag_eql_w1.log 2>&1
  python 3_modeling/01_zit/hp/02_bag_zit_eql_parallel_hpo.py --worker-id w2 --n-trials 2000 --n-jobs 4 --n-startup-trials 12 --end-at 2026-06-01T09:00 > bag_eql_w2.log 2>&1
  python 3_modeling/01_zit/hp/02_bag_zit_eql_parallel_hpo.py --worker-id w3 --n-trials 2000 --n-jobs 4 --n-startup-trials 12 --end-at 2026-06-01T09:00 > bag_eql_w3.log 2>&1

This worker writes Optuna DB records only.  Final refit, postprocess, and artifact
export should be run separately after selecting the best trial.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
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

import cleaning  # noqa: E402  (impute_spatial을 이 워커 프로세스에서만 median 버전으로 교체)
from meta_features import add_meta_features  # noqa: E402
from modules import hpo, preprocess  # noqa: E402
from modules.zit import BagZITboostRegressor  # noqa: E402
from utils.config import KEY_COL, OUTPUT_DIR, SEED, TARGET_COL  # noqa: E402
from utils.data import get_feat_cols, load_all, split_xs  # noqa: E402


DEFAULT_EXP_ID = "bag-zit-eql-final-002"
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

# bag-zit-final-001 best trial #122.
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

# bag-zit-final-001 경계분석 + 완료 76 trial 중요도/추세 분석 반영.
#   - 중요도(fANOVA/MDI): tau_pi(0.43) > zeta(0.18)가 지배. 나머지는 대체로 평평.
#   - 추세(Spearman): zeta(+0.64)/mu_max_depth(+0.41)/phi_max_depth(+0.88)는 "작을수록 좋음".
#   이전 버전이 깊은/높은 쪽으로 과확장했던 zeta-high, mu_max_depth, phi_max_depth를 적당히 다시 좁힘
#   (나쁜 영역만 잘라내고 anchor·상위 영역은 유지). tau_pi는 EQL이 π 분포를 바꾸므로 넓게 유지.
BAG_ZIT_EQL_SEARCH = {
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


def _impute_spatial_median(xs, feat_cols, max_dist=2.0, train_mask=None):
    """cleaning.impute_spatial의 로컬 사본 — 2·3단계 fallback fill만 mean→median으로 교체.

    cleaning.py는 수정하지 않는다. load_preprocessed_data에서 `cleaning.impute_spatial`을
    이 함수로 in-process monkeypatch해 **이 워커 프로세스에서만** 적용한다 (다른 실험/노트북 무영향).
      - 1단계(같은 wafer 내 거리 가중 공간 보간): 원본과 동일.
      - 2단계(lot fallback): lot 평균 → lot 중앙값(median).
      - 3단계(global fallback): train 전체 평균 → train 전체 중앙값(median).

    ⚠ 동기화 주의: cleaning.impute_spatial 원본 로직이 바뀌면 이 사본도 같이 갱신해야 한다
       (현재 2_preprocessing/cleaning.py와 1단계 로직 동일, 2·3단계만 median).
    """
    from utils.config import DIE_KEY_COL, SPLIT_COL  # noqa: F401  (원본 시그니처 유지)
    from meta_features import parse_run_wf_xy

    parse_run_wf_xy(xs, prefix="_", inplace=True, verbose=False)
    xs['_lot_wafer'] = xs['_lot'] + '_' + xs['_wafer_no']
    tmp_cols = ['_lot', '_wafer_no', '_die_x', '_die_y', '_lot_wafer']

    total_na_before = xs[feat_cols].isnull().sum().sum()
    print(f"[공간 보간 imputation / 2·3단계 median] 총 결측: {total_na_before:,}")
    if train_mask is not None:
        n_train = int(train_mask.sum())
        print(f"  train-only 모드: train {n_train:,} / 전체 {len(xs):,} 행")

    if total_na_before == 0:
        xs.drop(columns=tmp_cols, inplace=True)
        return xs, {'spatial': 0, 'lot': 0, 'global': 0}

    # --- 1단계: 같은 웨이퍼 안에서 거리 가중 평균으로 보간 (원본과 동일) ---
    original_arr = xs[feat_cols].values.copy()
    result_arr = original_arr.copy()

    idx_to_pos = pd.Series(range(len(xs)), index=xs.index)
    die_x_all = xs['_die_x'].values.astype(float)
    die_y_all = xs['_die_y'].values.astype(float)

    if train_mask is not None:
        is_train_all = train_mask.reindex(xs.index).fillna(False).values.astype(bool)
    else:
        is_train_all = np.ones(len(xs), dtype=bool)

    filled_spatial = 0
    for lw, group in xs.groupby('_lot_wafer'):
        gpos = idx_to_pos.loc[group.index].values
        g_orig = original_arr[gpos]
        g_na = np.isnan(g_orig)

        if not g_na.any():
            continue

        is_train_grp = is_train_all[gpos]
        if not is_train_grp.any():
            continue

        gx = die_x_all[gpos]
        gy = die_y_all[gpos]
        dx = gx[:, None] - gx[None, :]
        dy = gy[:, None] - gy[None, :]
        dist_mat = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist_mat, np.inf)

        na_rows = np.where(g_na.any(axis=1))[0]

        for li in na_rows:
            row_na = g_na[li]
            nan_cols = np.where(row_na)[0]
            if len(nan_cols) == 0:
                continue

            dists = dist_mat[li]
            nbr_mask = (dists <= max_dist) & is_train_grp
            if not nbr_mask.any():
                continue

            nbr_idx = np.where(nbr_mask)[0]
            weights = 1.0 / np.maximum(dists[nbr_idx], 1e-6)

            nbr_vals = g_orig[nbr_idx][:, nan_cols]
            valid = ~np.isnan(nbr_vals)

            w_valid = weights[:, None] * valid
            w_sum = w_valid.sum(axis=0)
            has_valid = w_sum > 0

            if not has_valid.any():
                continue

            weighted_sum = (weights[:, None] * np.where(valid, nbr_vals, 0.0)).sum(axis=0)
            avg = weighted_sum / w_sum

            fill_cols = nan_cols[has_valid]
            result_arr[gpos[li], fill_cols] = avg[has_valid]
            filled_spatial += int(has_valid.sum())

    xs[feat_cols] = result_arr

    na_after_spatial = xs[feat_cols].isnull().sum().sum()
    print(f"  1단계 (공간 보간, dist<={max_dist}): {filled_spatial:,}개 채움"
          f" → 잔여: {na_after_spatial:,}")

    # --- 2단계: 1단계로 못 채운 칸은 "같은 lot의 중앙값"으로 (median, train 행에서만 계산) ---
    if na_after_spatial > 0:
        if train_mask is not None:
            train_lot_fill = xs.loc[train_mask].groupby('_lot')[feat_cols].median()
        else:
            train_lot_fill = xs.groupby('_lot')[feat_cols].median()

        fill_df = train_lot_fill.reindex(xs['_lot'].values)
        fill_df.index = xs.index

        filled_before = xs[feat_cols].isnull().sum().sum()
        xs[feat_cols] = xs[feat_cols].fillna(fill_df)
        filled_lot = filled_before - xs[feat_cols].isnull().sum().sum()
        na_after_lot = xs[feat_cols].isnull().sum().sum()
        lot_label = "train 기준" if train_mask is not None else "전체 기준"
        print(f"  2단계 (lot 중앙값, {lot_label}): {filled_lot:,}개 채움"
              f" → 잔여: {na_after_lot:,}")
    else:
        filled_lot = 0
        na_after_lot = 0

    # --- 3단계: 그래도 남은 칸은 "train 전체의 컬럼 중앙값"으로 (median) ---
    if na_after_lot > 0:
        train_fill = xs.loc[xs[SPLIT_COL] == 'train', feat_cols].median()
        filled_before = xs[feat_cols].isnull().sum().sum()
        xs[feat_cols] = xs[feat_cols].fillna(train_fill)
        filled_global = filled_before - xs[feat_cols].isnull().sum().sum()
        na_after_global = xs[feat_cols].isnull().sum().sum()
        print(f"  3단계 (train 전체 중앙값): {filled_global:,}개 채움"
              f" → 잔여: {na_after_global:,}")
    else:
        filled_global = 0

    xs.drop(columns=tmp_cols, inplace=True)

    report = {
        'total_before': total_na_before,
        'spatial': filled_spatial,
        'lot': filled_lot,
        'global': filled_global,
        'remaining': xs[feat_cols].isnull().sum().sum(),
    }

    print(f"\n  [요약] {total_na_before:,} → "
          f"공간({filled_spatial:,}) → lot({filled_lot:,}) → "
          f"전체({filled_global:,}) → 잔여({report['remaining']:,})")

    return xs, report


def _tweedie_unit_deviance(y: np.ndarray, mu: np.ndarray, zeta: float) -> np.ndarray:
    """Tweedie unit deviance D_zeta(y, mu), valid for 1 < zeta < 2."""
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    y = np.maximum(y, 0.0)
    mu = np.maximum(mu, 1e-10)
    p = float(zeta)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        y_term = np.where(
            y > 0,
            np.power(y, 2.0 - p) / ((1.0 - p) * (2.0 - p)),
            0.0,
        )
        cross_term = -y * np.power(mu, 1.0 - p) / (1.0 - p)
        mu_term = np.power(mu, 2.0 - p) / (2.0 - p)
        dev = 2.0 * (y_term + cross_term + mu_term)

    dev = np.nan_to_num(dev, nan=1e6, posinf=1e6, neginf=1e-8)
    return np.clip(dev, 1e-8, 1e6)


class BagZITEQLRegressor(BagZITboostRegressor):
    """BagZIT with EQL/Tweedie-deviance target for the phi M-step."""

    def _m_step(self, X, y, posterior):
        w_tw = 1.0 - posterior

        lgb_pi = lgb.LGBMRegressor(**self._pi_params())
        lgb_pi.fit(X, posterior)
        pi_pred = lgb_pi.predict(X)
        pi_pred = np.clip(pi_pred, 1e-8, 1.0 - 1e-8)

        phi_for_weight = np.maximum(self._phi_current, 1e-10)
        mu_weight = w_tw / phi_for_weight

        lgb_mu = lgb.LGBMRegressor(**self._mu_params())
        lgb_mu.fit(X, y, sample_weight=mu_weight)
        mu_pred = lgb_mu.predict(X)
        mu_pred = np.maximum(mu_pred, 1e-10)

        phi_target = _tweedie_unit_deviance(y, mu_pred, self.zeta)
        lgb_phi = lgb.LGBMRegressor(**self._phi_params())
        lgb_phi.fit(X, phi_target, sample_weight=w_tw)
        phi_pred = lgb_phi.predict(X)
        phi_pred = np.clip(phi_pred, 1e-8, 1e6)

        return lgb_pi, lgb_mu, lgb_phi, pi_pred, mu_pred, phi_pred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BagZIT-EQL parallel HPO worker.")
    parser.add_argument("--exp-id", default=DEFAULT_EXP_ID)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--worker-id", default=f"pid{os.getpid()}")
    parser.add_argument("--n-trials", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=90.0)
    parser.add_argument(
        "--end-at",
        default=None,
        help=(
            "Absolute wall-clock end datetime in ISO format, e.g. '2026-06-01T09:00'. "
            "When set, the worker stops at this moment regardless of --timeout-hours."
        ),
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--n-startup-trials", type=int, default=12)
    parser.add_argument("--db-timeout", type=float, default=120.0)
    parser.add_argument("--heartbeat-interval", type=int, default=300)
    parser.add_argument("--grace-period", type=int, default=1800)
    parser.add_argument("--max-retry", type=int, default=1)
    parser.add_argument("--enqueue-anchor", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-clip-y-extreme", action="store_true")
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

    study_meta = {
        "exp_id": args.exp_id,
        "user": args.user,
        "model": "BagZIT-EQL parallel HPO worker",
        "n_trials_per_worker": args.n_trials,
        "n_folds": args.n_folds,
        "n_jobs": args.n_jobs,
        "pp_fixed": PP_FIXED,
        "anchor_source": "bag-zit-final-001 best trial #122",
        "anchor": BAG_ZIT_ANCHOR_001,
        "anchor_tau_pi": ANCHOR_TAU_PI_001,
        "search_space": BAG_ZIT_EQL_SEARCH,
        "tau_pi_range": TAU_PI_RANGE,
        "sampler": "TPE seed=None multivariate group",
        "pruner": f"MedianPruner n_startup={args.n_startup_trials} n_warmup=2",
        "phi_update": "EQL Tweedie unit deviance target",
        "impute_stage23": "lot/global fallback mean->median (spatial stage unchanged); cleaning.py 미수정, in-process patch",
        "clip_y_extreme": not args.no_clip_y_extreme,
        "seed": int(SEED),
        "model_random_state": "SEED + trial.number",
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

    # 결측 imputation 2·3단계(lot/global fallback)를 mean→median으로. cleaning.py는 수정하지 않고
    # 이 워커 프로세스 안에서만 impute_spatial을 로컬 median 사본으로 교체한다 (1단계 공간보간 동일).
    # run_cleaning(cleaning.py)이 impute_spatial을 모듈 전역으로 호출하므로 이 patch가 그대로 반영됨.
    cleaning.impute_spatial = _impute_spatial_median
    print("[patch] cleaning.impute_spatial -> 2·3단계 median (이 워커 프로세스 한정, cleaning.py 미수정)")

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
        params = hpo.sample_from_space(trial, BAG_ZIT_EQL_SEARCH)
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
            f"trial #{trial.number}: worker={args.worker_id}, "
            f"tau_pi={tau_pi:.4f}, oof={oof_rmse:.9f}, elapsed={elapsed:.0f}s"
        )
        return oof_rmse

    return objective


def main() -> None:
    args = parse_args()
    logging.getLogger("lightgbm").setLevel(logging.ERROR)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    out_dir = Path(OUTPUT_DIR) / "01_zit" / "bag_zit" / "hp" / args.exp_id.split("-")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / f"optuna_{args.user}_{args.exp_id}.db"

    print(f"PROJECT_ROOT={PROJECT_ROOT}")
    print(f"optuna={optuna.__version__}")
    print(f"EXP_ID={args.exp_id}, worker_id={args.worker_id}")
    print(f"OUT_DIR={out_dir}")
    print(f"DB_PATH={db_path}")
    print(f"N_TRIALS={args.n_trials}, N_JOBS={args.n_jobs}, N_FOLDS={args.n_folds}")
    print(f"N_STARTUP_TRIALS={args.n_startup_trials}")
    print(f"search HP={len(BAG_ZIT_EQL_SEARCH)} + tau_pi=1")

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
