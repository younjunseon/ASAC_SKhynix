"""
후처리 — die→unit 집계, π threshold, zero_clip.

모델은 die-level로 예측을 내지만 평가는 unit-level이다. tune_and_apply가 3단계를 순서대로 처리한다:
  1) 집계      : 한 unit의 die 4개 예측 → unit 예측 하나 (mean/median/max/min/trimmed_mean/Q25/Q75/weighted 중 best)
  2) π threshold : ZITboost류에서 unit의 π 추정치(die π를 unit 평균낸 값)가 th 초과면 그 unit 예측을 0으로 강제 (선택)
  3) zero_clip : pred < th 인 unit 예측을 0으로 (작은 양수 노이즈 제거)

각 단계의 후보는 train OOF로 골라 두되, val이 주어지면 "val RMSE가 실제로 좋아질 때만 채택"하고 아니면 baseline(mean / 미적용)을 유지한다. (val이 없으면 train OOF best를 그냥 채택 — backward-compat)

사용법
------
    from modules import postprocess

    pp = postprocess.tune_and_apply(
        xs_train, xs_val, xs_test,
        die_pred_train=final['oof_pred_die'],
        die_pred_val=final['val_pred_die'],
        die_pred_test=final['test_pred_die'],
        y_train_unit=ys_train_unit,
        y_val_unit=ys_val_unit,            # 주면 "val 개선 시만 채택" 모드
        die_pi_train=final['oof_pi'],      # ZIT이면 제공, 아니면 None
        ...
    )
    # pp['final_train_unit'], pp['final_val_unit'], pp['final_test_unit']
    # pp['best_agg'], pp['best_pi_threshold'], pp['best_zero_clip'], pp['decisions']
"""
import numpy as np
import pandas as pd
import optuna
from scipy.optimize import minimize
from scipy.stats import trim_mean

from utils.config import KEY_COL, TARGET_COL, POSITION_COL


# 시험해 볼 die→unit 집계 방식 8종
AGG_METHODS = (
    "mean", "median", "max", "min",
    "trimmed_mean", "weighted",
    "Q25", "Q75",
)


# ------------------------------------------------------------
# die-level pred → unit-level pred (집계)
# ------------------------------------------------------------

def _agg_simple(xs, die_pred, func):
    """ufs_serial로 groupby 해서 func 집계 → [KEY_COL, 'pred'] DataFrame."""
    df = pd.DataFrame({KEY_COL: xs[KEY_COL].values, "pred": die_pred})
    return df.groupby(KEY_COL, sort=False)["pred"].agg(func).reset_index()   # sort=False: 첫 등장 순서 유지


def aggregate(xs, die_pred, method="mean", pos_weights=None):
    """die-level → unit-level 집계.

    Parameters
    ----------
    xs : DataFrame (die-level, KEY_COL + optionally POSITION_COL)
    die_pred : array (len=len(xs))
    method : str (AGG_METHODS 중 하나, 'sum'도 허용)
    pos_weights : array[4] or None
        method='weighted'일 때 필수. position 1~4 순서 가중치 (합=1).

    Returns
    -------
    DataFrame [KEY_COL, 'pred']  — unit 순서는 첫 등장 순
    """
    if method == "mean":
        return _agg_simple(xs, die_pred, "mean")
    if method == "sum":
        return _agg_simple(xs, die_pred, "sum")
    if method == "median":
        return _agg_simple(xs, die_pred, "median")
    if method == "max":
        return _agg_simple(xs, die_pred, "max")
    if method == "min":
        return _agg_simple(xs, die_pred, "min")
    if method == "trimmed_mean":
        # die가 4개 → 위아래 25%씩 잘라내면 = 가운데 2개의 평균 (극단 die 1개씩 무시)
        return _agg_simple(
            xs, die_pred, lambda x: trim_mean(x, proportiontocut=0.25)
        )
    if method == "Q25":
        return _agg_simple(xs, die_pred, lambda x: np.quantile(x, 0.25))
    if method == "Q75":
        return _agg_simple(xs, die_pred, lambda x: np.quantile(x, 0.75))
    if method == "weighted":
        # position별 가중평균: die를 position 1~4 컬럼으로 펼친 뒤 (N,4)·(4,) 행렬곱
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
        cols = sorted(pivot.columns.tolist())
        if cols != [1, 2, 3, 4]:                          # 4개 position이 다 있어야 함
            raise ValueError(f"Expected positions [1,2,3,4], got {cols}")
        w = np.asarray(pos_weights, dtype=float)
        pred_unit = pivot[cols].values @ w                # 행렬곱 = 각 unit의 가중평균
        out = pd.DataFrame({KEY_COL: pivot.index.values, "pred": pred_unit})
        return out
    raise ValueError(f"Unknown aggregation method: {method!r}")


# ------------------------------------------------------------
# position 가중치 최적화 (weighted 집계용)
# ------------------------------------------------------------

def _build_position_pivot(xs, die_pred, y_true_unit):
    """die-level pred → (N_unit, 4) position 행렬 + 같은 순서의 정답 y 벡터."""
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
    pred_mat = pivot[cols].values                                       # (N, 4)
    y = y_true_unit.set_index(KEY_COL)[TARGET_COL].loc[pivot.index].values   # pivot 순서에 맞춰 정답 정렬
    return pred_mat, y


def fit_position_weights(xs, die_pred, y_true_unit, bounds=(0.15, 0.35)):
    """train OOF의 die 예측 → unit 정답을 가장 잘 맞추는 position 1~4 가중치를 SLSQP로 찾음.

    제약: w_i ∈ bounds, Σw_i = 1.

    Returns
    -------
    np.array[4]  — position 1,2,3,4 순서. 최적화 실패 시 균등(0.25씩) 반환.
    """
    pred_mat, y = _build_position_pivot(xs, die_pred, y_true_unit)

    def loss(w):
        return float(np.sqrt(np.mean((pred_mat @ w - y) ** 2)))   # 가중평균 예측의 RMSE

    w0 = np.full(4, 0.25)                                          # 균등에서 출발
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]   # 합=1 제약
    bnds = [bounds] * 4                                           # 각 weight 범위 제약
    res = minimize(loss, w0, method="SLSQP", bounds=bnds, constraints=constraints)
    if not res.success:
        print(f"[SLSQP] warning: {res.message}. Fallback to equal weights.")
        return np.full(4, 0.25)
    return res.x


def fit_position_weights_optuna(
    xs, die_pred, y_true_unit, n_trials=50, seed=42,
):
    """position 1~4 가중치를 Optuna로 탐색 (sum=1 제약은 Dirichlet 정규화 w_i = raw_i/Σraw로 자동 만족).

    Parameters
    ----------
    n_trials : int (default 50)
    seed : int (재현성)

    Returns
    -------
    np.array[4]
    """
    pred_mat, y = _build_position_pivot(xs, die_pred, y_true_unit)

    def objective(trial):
        raw = np.array([trial.suggest_float(f"w{i+1}", 0.05, 1.0) for i in range(4)])
        w = raw / raw.sum()                                       # 정규화 → 합=1
        return float(np.sqrt(np.mean((pred_mat @ w - y) ** 2)))   # 가중평균 RMSE 최소화

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    raw = np.array([study.best_params[f"w{i+1}"] for i in range(4)])
    w = raw / raw.sum()
    print(f"[Position weights / Optuna {n_trials}t] best={study.best_value:.6f}, w={w.round(3).tolist()}")
    return w


# ------------------------------------------------------------
# RMSE 헬퍼
# ------------------------------------------------------------

def _unit_rmse(unit_pred_df, y_true_unit):
    """unit_pred_df([KEY_COL,'pred'])와 y_true_unit([KEY_COL,TARGET_COL])를 정렬해 RMSE."""
    y = y_true_unit.set_index(KEY_COL)[TARGET_COL]
    p = unit_pred_df.set_index(KEY_COL)["pred"].loc[y.index]   # 정답 순서에 맞춰 예측 정렬
    return float(np.sqrt(np.mean((p.values - y.values) ** 2)))


# ------------------------------------------------------------
# 단계 1: 최적 집계 방식 탐색 (train OOF 기준)
# ------------------------------------------------------------

def find_best_aggregation(
    xs_train, die_pred_train, y_train_unit,
    methods=AGG_METHODS,
    position_method="optuna",
    optuna_n_trials=50,
):
    """train OOF로 각 집계 방식의 unit RMSE를 재서 best를 고름.

    Parameters
    ----------
    methods : iterable of str (default AGG_METHODS — 8종)
    position_method : 'optuna'(기본) or 'slsqp' — weighted 집계의 가중치 탐색 방법
    optuna_n_trials : int (position_method='optuna' 시)

    Returns
    -------
    dict  {'best_method', 'best_rmse', 'rmse_per_method', 'pos_weights'}
    """
    rmses = {}
    pos_w = None
    for m in methods:
        if m == "weighted":
            # weighted는 가중치를 먼저 학습해야 함 (train OOF로)
            if position_method == "optuna":
                pos_w = fit_position_weights_optuna(
                    xs_train, die_pred_train, y_train_unit, n_trials=optuna_n_trials,
                )
            else:
                pos_w = fit_position_weights(xs_train, die_pred_train, y_train_unit)
            unit = aggregate(xs_train, die_pred_train, "weighted", pos_w)
        else:
            unit = aggregate(xs_train, die_pred_train, m)
        rmses[m] = _unit_rmse(unit, y_train_unit)
    best = min(rmses, key=rmses.get)                  # train OOF RMSE가 가장 낮은 방식
    print(f"[Aggregation] RMSEs: { {k: round(v, 6) for k, v in rmses.items()} }")
    print(f"[Aggregation] best={best} ({rmses[best]:.6f})")
    return {
        "best_method":     best,
        "best_rmse":       rmses[best],
        "rmse_per_method": rmses,
        "pos_weights":     pos_w,           # weighted가 best가 아니어도 마지막 계산값이 들어 있을 수 있음
    }


# ------------------------------------------------------------
# 단계 2: π threshold 탐색 (ZITboost류 — π로 0 게이트)
# ------------------------------------------------------------

def find_best_pi_threshold(
    unit_pred_df, unit_pi_df, y_true_unit,
    thresholds=np.arange(0.5, 0.96, 0.01),
):
    """unit 평균 π > threshold인 unit의 pred를 0으로. threshold를 grid로 훑어 best를 찾음.

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
        p_gated = np.where(pi > th, 0.0, p)                       # π 큰 unit은 예측을 0으로
        rmses[float(th)] = float(np.sqrt(np.mean((p_gated - y.values) ** 2)))
    best_th = min(rmses, key=rmses.get)
    print(f"[π threshold] best={best_th:.2f} ({rmses[best_th]:.6f})")
    return {
        "best_threshold":   best_th,
        "best_rmse":        rmses[best_th],
        "rmse_per_threshold": rmses,
    }


def apply_pi_threshold(unit_pred_df, unit_pi_df, threshold):
    """π > threshold인 행의 pred를 0으로 만든 새 DataFrame 반환."""
    out = unit_pred_df.copy()
    pi_map = unit_pi_df.set_index(KEY_COL)["pi"]
    mask = out[KEY_COL].map(pi_map).values > threshold
    out.loc[mask, "pred"] = 0.0
    return out


# ------------------------------------------------------------
# 단계 3: zero_clip threshold 탐색 (작은 양수 → 0)
# ------------------------------------------------------------

def find_best_zero_clip(
    unit_pred_df, y_true_unit,
    thresholds=np.arange(0.001, 0.016, 0.001),
    log_space=False,
):
    """pred < threshold인 unit 예측을 0으로. threshold grid를 훑어 best를 찾음.

    Parameters
    ----------
    log_space : bool (default False)
        True면 log1p 공간에서 비교 — 모델이 log1p target으로 학습된 경우 학습 공간과 일관되게.

    Returns
    -------
    dict  {'best_threshold', 'best_rmse', 'rmse_per_threshold', 'log_space'}
    """
    y = y_true_unit.set_index(KEY_COL)[TARGET_COL]
    p = unit_pred_df.set_index(KEY_COL)["pred"].loc[y.index].values
    rmses = {}
    for th in thresholds:
        if log_space:
            mask = np.log1p(p) < np.log1p(th)
        else:
            mask = p < th
        p_clipped = np.where(mask, 0.0, p)
        rmses[float(th)] = float(np.sqrt(np.mean((p_clipped - y.values) ** 2)))
    best_th = min(rmses, key=rmses.get)
    tag = " (log)" if log_space else ""
    print(f"[zero_clip{tag}] best={best_th:.4f} ({rmses[best_th]:.6f})")
    return {
        "best_threshold":     best_th,
        "best_rmse":          rmses[best_th],
        "rmse_per_threshold": rmses,
        "log_space":          log_space,
    }


def apply_zero_clip(unit_pred_df, threshold, log_space=False):
    """pred < threshold인 행의 pred를 0으로 만든 새 DataFrame 반환 (log_space=True면 log1p 공간에서 비교)."""
    out = unit_pred_df.copy()
    p = out["pred"].values
    if log_space:
        mask = np.log1p(p) < np.log1p(threshold)
    else:
        mask = p < threshold
    out.loc[mask, "pred"] = 0.0
    return out


# ------------------------------------------------------------
# 통합: train OOF로 3단계 튜닝 → val에서 검증 → val 개선 시만 채택 → test에도 동일 적용
# ------------------------------------------------------------

def tune_and_apply(
    xs_train, xs_val, xs_test,
    die_pred_train, die_pred_val, die_pred_test,
    y_train_unit, y_val_unit=None,
    die_pi_train=None, die_pi_val=None, die_pi_test=None,
    agg_methods=AGG_METHODS,
    pi_threshold_range=(0.5, 0.95),
    pi_threshold_step=0.01,
    zero_clip_range=(0.001, 0.015),
    zero_clip_step=0.001,
    use_pi_threshold=True,
    zero_clip_log_space=False,
    position_method="optuna",
    position_optuna_n_trials=50,
    baseline_agg="mean",
):
    """전체 후처리 파이프라인.

    각 단계마다: (1) train OOF에서 best 후보 탐색 → (2) val에 적용해 val RMSE 측정 →
    (3) baseline보다 val이 좋아졌으면 채택, 아니면 baseline 유지.
    `y_val_unit=None`이면 val 검증 없이 train OOF best를 그냥 채택 (backward-compat).

    Baseline: 집계=mean, π threshold=미적용, zero_clip=미적용.

    Returns
    -------
    dict {
        'best_agg', 'pos_weights', 'best_pi_threshold', 'best_zero_clip',
        'final_train_unit', 'final_val_unit', 'final_test_unit',
        'train_rmse', 'val_rmse_final', 'val_rmse_history', 'agg_rmses',
        'zero_clip_log_space', 'position_method',
        'decisions': dict — 단계별 채택/미채택 + 사유
    }
    """
    has_val = y_val_unit is not None
    decisions = {}              # 각 단계에서 무엇을 채택/거부했는지 사람이 읽을 메시지
    val_rmse_history = []       # 각 단계 직후의 val RMSE (진단용)

    def _val_rmse_or_none(unit_df):
        return _unit_rmse(unit_df, y_val_unit) if has_val else None

    # --- 0) baseline 상태에서 출발: 집계만 baseline_agg, π/zero_clip 미적용 ---
    train_unit = aggregate(xs_train, die_pred_train, baseline_agg)
    val_unit   = aggregate(xs_val,   die_pred_val,   baseline_agg)
    test_unit  = aggregate(xs_test,  die_pred_test,  baseline_agg)
    cur_val_rmse = _val_rmse_or_none(val_unit)   # 현재까지의 val RMSE (채택 비교 기준)
    val_rmse_history.append((f"baseline_{baseline_agg}", cur_val_rmse))

    # --- 1) 집계 방식: train OOF best 후보 vs baseline_agg ---
    agg_res = find_best_aggregation(
        xs_train, die_pred_train, y_train_unit,
        methods=agg_methods,
        position_method=position_method,
        optuna_n_trials=position_optuna_n_trials,
    )
    best_agg_cand = agg_res["best_method"]
    pos_w_cand    = agg_res["pos_weights"]

    if best_agg_cand == baseline_agg:
        # train OOF에서도 baseline이 1등 → 그대로 둠
        best_agg = baseline_agg
        pos_w    = None
        decisions["aggregation"] = f"{baseline_agg} is best on train OOF — kept"
    else:
        # 후보 방식으로 3 split 다시 집계해서 val로 검증
        cand_train = aggregate(xs_train, die_pred_train, best_agg_cand, pos_w_cand)
        cand_val   = aggregate(xs_val,   die_pred_val,   best_agg_cand, pos_w_cand)
        cand_test  = aggregate(xs_test,  die_pred_test,  best_agg_cand, pos_w_cand)
        cand_val_rmse = _val_rmse_or_none(cand_val)

        if not has_val:
            # val이 없으면 train OOF best를 무조건 채택
            train_unit, val_unit, test_unit = cand_train, cand_val, cand_test
            best_agg, pos_w = best_agg_cand, pos_w_cand
            decisions["aggregation"] = f"{best_agg_cand} adopted (no val provided)"
            cur_val_rmse = None
        elif cand_val_rmse < cur_val_rmse:
            # val이 실제로 좋아짐 → 채택
            decisions["aggregation"] = (
                f"{best_agg_cand} adopted (val {cur_val_rmse:.6f} -> {cand_val_rmse:.6f})"
            )
            train_unit, val_unit, test_unit = cand_train, cand_val, cand_test
            best_agg, pos_w = best_agg_cand, pos_w_cand
            cur_val_rmse = cand_val_rmse
        else:
            # train OOF에선 좋았지만 val에선 손해 → 거부, baseline 유지
            decisions["aggregation"] = (
                f"{best_agg_cand} rejected (val {cur_val_rmse:.6f} <= {cand_val_rmse:.6f}) -> keep {baseline_agg}"
            )
            best_agg, pos_w = baseline_agg, None
    val_rmse_history.append((f"after_agg({best_agg})", cur_val_rmse))

    # --- 2) π threshold (ZITboost τ_π를 따로 쓰는 경로는 use_pi_threshold=False로 끔) ---
    best_pi_th = None
    if use_pi_threshold and die_pi_train is not None:
        # die π를 unit 평균 π로 집계 (컬럼명을 'pi'로)
        pi_train_unit = aggregate(xs_train, die_pi_train, "mean").rename(columns={"pred": "pi"})
        pi_val_unit   = aggregate(xs_val,   die_pi_val,   "mean").rename(columns={"pred": "pi"})
        pi_test_unit  = aggregate(xs_test,  die_pi_test,  "mean").rename(columns={"pred": "pi"})

        th_arr = np.arange(pi_threshold_range[0],
                           pi_threshold_range[1] + pi_threshold_step / 2,   # 끝값 포함되게 살짝 여유
                           pi_threshold_step)
        pi_res = find_best_pi_threshold(train_unit, pi_train_unit, y_train_unit, th_arr)
        cand_pi_th = pi_res["best_threshold"]

        cand_train = apply_pi_threshold(train_unit, pi_train_unit, cand_pi_th)
        cand_val   = apply_pi_threshold(val_unit,   pi_val_unit,   cand_pi_th)
        cand_test  = apply_pi_threshold(test_unit,  pi_test_unit,  cand_pi_th)
        cand_val_rmse = _val_rmse_or_none(cand_val)

        # 집계 단계와 동일한 "val 개선 시만 채택" 로직
        if not has_val:
            train_unit, val_unit, test_unit = cand_train, cand_val, cand_test
            best_pi_th = cand_pi_th
            decisions["pi_threshold"] = f"{cand_pi_th:.3f} adopted (no val)"
            cur_val_rmse = None
        elif cand_val_rmse < cur_val_rmse:
            decisions["pi_threshold"] = (
                f"{cand_pi_th:.3f} adopted (val {cur_val_rmse:.6f} -> {cand_val_rmse:.6f})"
            )
            train_unit, val_unit, test_unit = cand_train, cand_val, cand_test
            best_pi_th = cand_pi_th
            cur_val_rmse = cand_val_rmse
        else:
            decisions["pi_threshold"] = (
                f"{cand_pi_th:.3f} rejected (val {cur_val_rmse:.6f} <= {cand_val_rmse:.6f}) — skip"
            )
    val_rmse_history.append(("after_pi_th", cur_val_rmse))

    # --- 3) zero_clip (작은 양수 예측 → 0) ---
    zc_arr = np.arange(zero_clip_range[0],
                       zero_clip_range[1] + zero_clip_step / 2,
                       zero_clip_step)
    zc_res = find_best_zero_clip(
        train_unit, y_train_unit, zc_arr, log_space=zero_clip_log_space,
    )
    cand_zc = zc_res["best_threshold"]

    cand_train = apply_zero_clip(train_unit, cand_zc, log_space=zero_clip_log_space)
    cand_val   = apply_zero_clip(val_unit,   cand_zc, log_space=zero_clip_log_space)
    cand_test  = apply_zero_clip(test_unit,  cand_zc, log_space=zero_clip_log_space)
    cand_val_rmse = _val_rmse_or_none(cand_val)

    best_zc = None
    if not has_val:
        train_unit, val_unit, test_unit = cand_train, cand_val, cand_test
        best_zc = cand_zc
        decisions["zero_clip"] = f"{cand_zc:.4f} adopted (no val)"
        cur_val_rmse = None
    elif cand_val_rmse < cur_val_rmse:
        decisions["zero_clip"] = (
            f"{cand_zc:.4f} adopted (val {cur_val_rmse:.6f} -> {cand_val_rmse:.6f})"
        )
        train_unit, val_unit, test_unit = cand_train, cand_val, cand_test
        best_zc = cand_zc
        cur_val_rmse = cand_val_rmse
    else:
        decisions["zero_clip"] = (
            f"{cand_zc:.4f} rejected (val {cur_val_rmse:.6f} <= {cand_val_rmse:.6f}) — skip"
        )
    val_rmse_history.append(("after_zero_clip", cur_val_rmse))

    final_train_rmse = _unit_rmse(train_unit, y_train_unit)

    # 요약 출력
    tag_log = " (log)" if zero_clip_log_space else ""
    print(f"[Postprocess] best_agg={best_agg}, "
          f"pi_th={best_pi_th}, zero_clip{tag_log}={best_zc}, "
          f"train_rmse={final_train_rmse:.6f}"
          + (f", val_rmse={cur_val_rmse:.6f}" if has_val else ""))
    for stage, val_r in val_rmse_history:
        print(f"  {stage:30s} val_rmse={val_r}")
    for stage, msg in decisions.items():
        print(f"  [decision] {stage:14s} {msg}")

    return {
        "best_agg":            best_agg,
        "pos_weights":         pos_w,
        "best_pi_threshold":   best_pi_th,
        "best_zero_clip":      best_zc,
        "zero_clip_log_space": zero_clip_log_space,
        "position_method":     position_method,
        "final_train_unit":    train_unit,
        "final_val_unit":      val_unit,
        "final_test_unit":     test_unit,
        "train_rmse":          final_train_rmse,
        "val_rmse_final":      cur_val_rmse,
        "val_rmse_history":    val_rmse_history,
        "agg_rmses":           agg_res["rmse_per_method"],
        "decisions":           decisions,
    }
