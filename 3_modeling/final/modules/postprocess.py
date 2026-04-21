"""
Final 파이프라인 — 후처리 (die → unit, π threshold, zero_clip)

3단계 순차:
  1) aggregate: die-level 4개 → unit-level (6종 중 best)
     - mean / median / max / min / trimmed_mean / weighted(SLSQP position w)
  2) pi_threshold: ZITboost π > threshold → pred=0 (경로 A·B 한정)
  3) zero_clip: pred < threshold → 0

각 단계는 독립적으로 val/OOF RMSE 기준으로 best를 grid search.

사용법
------
    from final.modules import postprocess

    # 단일 경로 전체 후처리 (train OOF로 best 튜닝 → val/test에 동일 적용)
    pp = postprocess.tune_and_apply(
        xs_train, xs_val, xs_test,
        die_pred_train=final['oof_pred_die'],
        die_pred_val=final['val_pred_die'],
        die_pred_test=final['test_pred_die'],
        y_train_unit=ys_train_unit,
        die_pi_train=final['oof_pi'],      # ZIT이면 제공, 아니면 None
        die_pi_val=final['val_pi'],
        die_pi_test=final['test_pi'],
        agg_methods=['mean', 'median', 'weighted', 'trimmed_mean'],
        pi_threshold_range=(0.5, 0.95),
        zero_clip_range=(0.001, 0.015),
    )
    # pp['final_train_unit'], pp['final_val_unit'], pp['final_test_unit']
    # pp['best_agg'], pp['best_pi_threshold'], pp['best_zero_clip']
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import trim_mean

from utils.config import KEY_COL, TARGET_COL, POSITION_COL


AGG_METHODS = ("mean", "median", "max", "min", "trimmed_mean", "weighted")


# ═════════════════════════════════════════════════════════════
# 단일 집계 함수 (die-level pred → unit-level pred)
# ═════════════════════════════════════════════════════════════

def _agg_simple(xs, die_pred, func):
    df = pd.DataFrame({KEY_COL: xs[KEY_COL].values, "pred": die_pred})
    return df.groupby(KEY_COL, sort=False)["pred"].agg(func).reset_index()


def aggregate(xs, die_pred, method="mean", pos_weights=None):
    """die-level → unit-level 집계.

    Parameters
    ----------
    xs : DataFrame (die-level, KEY_COL + optionally POSITION_COL)
    die_pred : array (len=len(xs))
    method : str (AGG_METHODS 중 하나)
    pos_weights : array[4] or None
        method='weighted'일 때 필수. position 1~4 순서 가중치 (합=1).

    Returns
    -------
    DataFrame [KEY_COL, 'pred']  — unit 순서는 첫 등장 순
    """
    if method == "mean":
        return _agg_simple(xs, die_pred, "mean")
    if method == "median":
        return _agg_simple(xs, die_pred, "median")
    if method == "max":
        return _agg_simple(xs, die_pred, "max")
    if method == "min":
        return _agg_simple(xs, die_pred, "min")
    if method == "trimmed_mean":
        # 4 die → 상하 25% trimming = 중앙 2개 평균
        return _agg_simple(
            xs, die_pred, lambda x: trim_mean(x, proportiontocut=0.25)
        )
    if method == "weighted":
        if pos_weights is None:
            raise ValueError("weighted aggregation requires pos_weights")
        df = pd.DataFrame({
            KEY_COL: xs[KEY_COL].values,
            POSITION_COL: xs[POSITION_COL].values,
            "pred": die_pred,
        })
        pivot = df.pivot_table(
            index=KEY_COL, columns=POSITION_COL, values="pred",
            aggfunc="mean", sort=False,
        )
        # 4개 position이 모두 있는지 확인
        cols = sorted(pivot.columns.tolist())
        if cols != [1, 2, 3, 4]:
            raise ValueError(f"Expected positions [1,2,3,4], got {cols}")
        w = np.asarray(pos_weights, dtype=float)
        pred_unit = pivot[cols].values @ w
        out = pd.DataFrame({KEY_COL: pivot.index.values, "pred": pred_unit})
        return out
    raise ValueError(f"Unknown aggregation method: {method!r}")


# ═════════════════════════════════════════════════════════════
# position weight 최적화 (weighted 집계용)
# ═════════════════════════════════════════════════════════════

def fit_position_weights(xs, die_pred, y_true_unit, bounds=(0.15, 0.35)):
    """train OOF die 예측 → unit 정답으로 position 1~4 가중치 SLSQP 최적화.

    제약: w_i ∈ [bounds], Σw_i = 1.

    Returns
    -------
    np.array[4]  — position 1,2,3,4 순서
    """
    df = pd.DataFrame({
        KEY_COL: xs[KEY_COL].values,
        POSITION_COL: xs[POSITION_COL].values,
        "pred": die_pred,
    })
    pivot = df.pivot_table(
        index=KEY_COL, columns=POSITION_COL, values="pred",
        aggfunc="mean", sort=False,
    )
    cols = sorted(pivot.columns.tolist())
    pred_mat = pivot[cols].values  # (N_unit, 4)

    y = y_true_unit.set_index(KEY_COL)[TARGET_COL].loc[pivot.index].values

    def loss(w):
        return float(np.sqrt(np.mean((pred_mat @ w - y) ** 2)))

    w0 = np.full(4, 0.25)
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bnds = [bounds] * 4
    res = minimize(loss, w0, method="SLSQP", bounds=bnds, constraints=constraints)
    if not res.success:
        print(f"[SLSQP] warning: {res.message}. Fallback to equal weights.")
        return np.full(4, 0.25)
    return res.x


# ═════════════════════════════════════════════════════════════
# RMSE
# ═════════════════════════════════════════════════════════════

def _unit_rmse(unit_pred_df, y_true_unit):
    """unit_pred_df = [KEY_COL, 'pred'], y_true_unit = DataFrame[KEY_COL, TARGET_COL]."""
    y = y_true_unit.set_index(KEY_COL)[TARGET_COL]
    p = unit_pred_df.set_index(KEY_COL)["pred"].loc[y.index]
    return float(np.sqrt(np.mean((p.values - y.values) ** 2)))


# ═════════════════════════════════════════════════════════════
# 단계 1: 최적 집계 방식 탐색
# ═════════════════════════════════════════════════════════════

def find_best_aggregation(
    xs_train, die_pred_train, y_train_unit,
    methods=("mean", "median", "max", "min", "trimmed_mean", "weighted"),
):
    """train OOF 기준 각 집계 방식의 unit RMSE → best 선택.

    Returns
    -------
    dict  {'best_method', 'best_rmse', 'rmse_per_method', 'pos_weights'}
    """
    rmses = {}
    pos_w = None
    for m in methods:
        if m == "weighted":
            pos_w = fit_position_weights(xs_train, die_pred_train, y_train_unit)
            unit = aggregate(xs_train, die_pred_train, "weighted", pos_w)
        else:
            unit = aggregate(xs_train, die_pred_train, m)
        rmses[m] = _unit_rmse(unit, y_train_unit)
    best = min(rmses, key=rmses.get)
    print(f"[Aggregation] RMSEs: { {k: round(v, 6) for k, v in rmses.items()} }")
    print(f"[Aggregation] best={best} ({rmses[best]:.6f})")
    return {
        "best_method":     best,
        "best_rmse":       rmses[best],
        "rmse_per_method": rmses,
        "pos_weights":     pos_w,
    }


# ═════════════════════════════════════════════════════════════
# 단계 2: π threshold tuning (ZITboost π 기반 gate)
# ═════════════════════════════════════════════════════════════

def find_best_pi_threshold(
    unit_pred_df, unit_pi_df, y_true_unit,
    thresholds=np.arange(0.5, 0.96, 0.01),
):
    """π > threshold 인 unit은 pred=0으로 강제. grid search.

    Parameters
    ----------
    unit_pred_df : DataFrame [KEY_COL, 'pred']
    unit_pi_df   : DataFrame [KEY_COL, 'pi']
    thresholds   : iterable

    Returns
    -------
    dict  {'best_threshold', 'best_rmse', 'rmse_per_threshold'}
    """
    y = y_true_unit.set_index(KEY_COL)[TARGET_COL]
    p = unit_pred_df.set_index(KEY_COL)["pred"].loc[y.index].values
    pi = unit_pi_df.set_index(KEY_COL)["pi"].loc[y.index].values

    rmses = {}
    for th in thresholds:
        p_gated = np.where(pi > th, 0.0, p)
        rmses[float(th)] = float(np.sqrt(np.mean((p_gated - y.values) ** 2)))
    best_th = min(rmses, key=rmses.get)
    print(f"[π threshold] best={best_th:.2f} ({rmses[best_th]:.6f})")
    return {
        "best_threshold":   best_th,
        "best_rmse":        rmses[best_th],
        "rmse_per_threshold": rmses,
    }


def apply_pi_threshold(unit_pred_df, unit_pi_df, threshold):
    """π > threshold인 row의 pred → 0. 새 DataFrame 반환."""
    out = unit_pred_df.copy()
    pi_map = unit_pi_df.set_index(KEY_COL)["pi"]
    mask = out[KEY_COL].map(pi_map).values > threshold
    out.loc[mask, "pred"] = 0.0
    return out


# ═════════════════════════════════════════════════════════════
# 단계 3: zero_clip threshold tuning
# ═════════════════════════════════════════════════════════════

def find_best_zero_clip(
    unit_pred_df, y_true_unit,
    thresholds=np.arange(0.001, 0.016, 0.001),
):
    """pred < threshold → 0. grid search.

    Returns
    -------
    dict  {'best_threshold', 'best_rmse', 'rmse_per_threshold'}
    """
    y = y_true_unit.set_index(KEY_COL)[TARGET_COL]
    p = unit_pred_df.set_index(KEY_COL)["pred"].loc[y.index].values
    rmses = {}
    for th in thresholds:
        p_clipped = np.where(p < th, 0.0, p)
        rmses[float(th)] = float(np.sqrt(np.mean((p_clipped - y.values) ** 2)))
    best_th = min(rmses, key=rmses.get)
    print(f"[zero_clip] best={best_th:.4f} ({rmses[best_th]:.6f})")
    return {
        "best_threshold":     best_th,
        "best_rmse":          rmses[best_th],
        "rmse_per_threshold": rmses,
    }


def apply_zero_clip(unit_pred_df, threshold):
    out = unit_pred_df.copy()
    mask = out["pred"].values < threshold
    out.loc[mask, "pred"] = 0.0
    return out


# ═════════════════════════════════════════════════════════════
# 통합: train OOF로 3단계 튜닝 → val/test에 동일 적용
# ═════════════════════════════════════════════════════════════

def tune_and_apply(
    xs_train, xs_val, xs_test,
    die_pred_train, die_pred_val, die_pred_test,
    y_train_unit,
    die_pi_train=None, die_pi_val=None, die_pi_test=None,
    agg_methods=("mean", "median", "max", "min", "trimmed_mean", "weighted"),
    pi_threshold_range=(0.5, 0.95),
    pi_threshold_step=0.01,
    zero_clip_range=(0.001, 0.015),
    zero_clip_step=0.001,
    use_pi_threshold=True,
):
    """전체 후처리 파이프라인. train OOF로 best를 찾고 val/test에 동일 적용.

    Returns
    -------
    dict {
        'best_agg', 'pos_weights', 'best_pi_threshold', 'best_zero_clip',
        'final_train_unit', 'final_val_unit', 'final_test_unit',
        'train_rmse', 'rmse_per_stage' (진단용)
    }
    """
    # ── 1. 집계 ──
    agg_res = find_best_aggregation(
        xs_train, die_pred_train, y_train_unit, methods=agg_methods
    )
    best_agg = agg_res["best_method"]
    pos_w = agg_res["pos_weights"]

    train_unit = aggregate(xs_train, die_pred_train, best_agg, pos_w)
    val_unit   = aggregate(xs_val,   die_pred_val,   best_agg, pos_w)
    test_unit  = aggregate(xs_test,  die_pred_test,  best_agg, pos_w)

    # ── 2. π threshold (optional) ──
    # π 집계는 pred 용 best_agg (max/min/weighted 등) 와 무관하게 **항상 mean**.
    # π 는 확률이므로 pred 의 극단 집계(max/min)나 position 가중치가 의미상 맞지
    # 않고, threshold 튜닝도 mean-집계 단일 해석이 안정적이다.
    best_pi_th = None
    if use_pi_threshold and die_pi_train is not None:
        pi_train_unit = aggregate(xs_train, die_pi_train, "mean")
        pi_val_unit   = aggregate(xs_val,   die_pi_val,   "mean")
        pi_test_unit  = aggregate(xs_test,  die_pi_test,  "mean")
        pi_train_unit = pi_train_unit.rename(columns={"pred": "pi"})
        pi_val_unit   = pi_val_unit.rename(columns={"pred": "pi"})
        pi_test_unit  = pi_test_unit.rename(columns={"pred": "pi"})

        th_arr = np.arange(pi_threshold_range[0],
                           pi_threshold_range[1] + pi_threshold_step/2,
                           pi_threshold_step)
        pi_res = find_best_pi_threshold(train_unit, pi_train_unit, y_train_unit, th_arr)
        best_pi_th = pi_res["best_threshold"]
        train_unit = apply_pi_threshold(train_unit, pi_train_unit, best_pi_th)
        val_unit   = apply_pi_threshold(val_unit,   pi_val_unit,   best_pi_th)
        test_unit  = apply_pi_threshold(test_unit,  pi_test_unit,  best_pi_th)

    # ── 3. zero_clip ──
    zc_arr = np.arange(zero_clip_range[0],
                       zero_clip_range[1] + zero_clip_step/2,
                       zero_clip_step)
    zc_res = find_best_zero_clip(train_unit, y_train_unit, zc_arr)
    best_zc = zc_res["best_threshold"]
    train_unit = apply_zero_clip(train_unit, best_zc)
    val_unit   = apply_zero_clip(val_unit,   best_zc)
    test_unit  = apply_zero_clip(test_unit,  best_zc)

    # ── 최종 train RMSE ──
    final_train_rmse = _unit_rmse(train_unit, y_train_unit)

    print(f"[Postprocess] best_agg={best_agg}, "
          f"pi_th={best_pi_th}, zero_clip={best_zc:.4f}, "
          f"train_rmse={final_train_rmse:.6f}")

    return {
        "best_agg":            best_agg,
        "pos_weights":         pos_w,
        "best_pi_threshold":   best_pi_th,
        "best_zero_clip":      best_zc,
        "final_train_unit":    train_unit,
        "final_val_unit":      val_unit,
        "final_test_unit":     test_unit,
        "train_rmse":          final_train_rmse,
        "agg_rmses":           agg_res["rmse_per_method"],
    }
