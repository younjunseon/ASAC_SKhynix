"""zit_fit_lib — zit fit 4조합(zit_only/bag × pearson/eql) 공유 calibration 머신러리.

4개 fit.ipynb가 byte-identical로 복제하던 후처리/보정/직렬화 1차 함수들을 한 곳으로 모은다.
실험 드라이버(fit_one_seed / save_result_artifacts / make_folds / params_for_seed)는 데이터·메타와
강하게 결합돼 있어 각 노트북에 그대로 둔다 — 본 모듈은 그 드라이버가 호출하는 **재사용 1차 함수**만 제공.

구성
----
- die→unit 후처리: tune_unit_postprocess_train_val (집계 선택 → zero_clip, val 개선 시에만 채택)
- isotonic/tail 보정: build_iso_pchip_transform(step→PCHIP), fit_iso_tail_grid(iso×tail×iqr grid 탐색)
- 진단: iqr_stats, push_top_k_to_iqr (예측 rank만 사용 — val leakage 없음)
- 직렬화: json_default, build_die_df, build_unit_output, serializable_calibrator
- 공용: rmse, clip_nonneg, apply_tau_pi, unit_rmse, aligned_unit_pred

config(집계/zero_clip/iso/tail/iqr grid)는 notebook cell4에서 **명시 kwargs로 주입** — 모듈은
상태를 갖지 않는다(노트북별 BASELINE_AGG/AGG_CANDIDATES 차이를 인자로 흡수).
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from sklearn.isotonic import IsotonicRegression


def _find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "setup.py").exists() and (p / "utils").exists():
            return p
    raise RuntimeError(f"Project root not found from {start}")


_ROOT = _find_project_root(Path(__file__).resolve())
for _p in [_ROOT, _ROOT / "3_modeling"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from modules import postprocess  # noqa: E402
from utils.config import KEY_COL, TARGET_COL, DIE_KEY_COL  # noqa: E402


# 공통 RMSE 계산. 모든 선택 기준은 validation RMSE가 낮은 쪽이다.
def rmse(pred, y):
    pred = np.asarray(pred, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def clip_nonneg(x):
    return np.clip(np.asarray(x, dtype=float), 0.0, None)


# ZIT의 structural-zero 확률 pi가 tau_pi보다 큰 die는 0으로 강제한다.
def apply_tau_pi(pred_die, pi_die, tau_pi):
    return np.where(pi_die > tau_pi, 0.0, pred_die)


def unit_rmse(unit_df, y_unit_s):
    p = unit_df.set_index(KEY_COL)['pred'].loc[y_unit_s.index].values
    return rmse(p, y_unit_s.values)


def aligned_unit_pred(unit_df, y_unit_s):
    return unit_df.set_index(KEY_COL)['pred'].loc[y_unit_s.index].values.astype(float)


def tune_unit_postprocess_train_val(
    xs_train,
    xs_val,
    xs_test,
    die_pred_train,
    die_pred_val,
    die_pred_test,
    y_train_unit_df,
    y_val_unit_df,
    *, baseline_agg, agg_candidates, position_method, position_optuna_n_trials,
    zero_clip_range, zero_clip_n, zero_clip_log_space,
):
    """seed sweep과 같은 train/validation 전용 축약판.

    차이: zero_clip 후보 array를 np.arange linear가 아니라 np.logspace로 만든다.
    하한 0.0001부터 상한 0.015까지 log 균등 30개. 작은 양수도 촘촘히 본다.
    """
    y_train_s = y_train_unit_df.set_index(KEY_COL)[TARGET_COL]
    y_val_s = y_val_unit_df.set_index(KEY_COL)[TARGET_COL]
    decisions = {}
    val_history = []

    # 1단계: baseline 집계(mean)에서 출발. validation 개선 시에만 교체.
    train_unit = postprocess.aggregate(xs_train, die_pred_train, baseline_agg)
    val_unit = postprocess.aggregate(xs_val, die_pred_val, baseline_agg)
    test_unit = postprocess.aggregate(xs_test, die_pred_test, baseline_agg)
    cur_val = unit_rmse(val_unit, y_val_s)
    val_history.append((f'baseline_{baseline_agg}', cur_val))

    # 2단계: train OOF에서 best 집계 방식 후보, validation 개선 시에만 채택.
    agg_res = postprocess.find_best_aggregation(
        xs_train,
        die_pred_train,
        y_train_unit_df,
        methods=agg_candidates,
        position_method=position_method,
        optuna_n_trials=position_optuna_n_trials,
    )
    best_agg_cand = agg_res['best_method']
    pos_w_cand = agg_res['pos_weights']

    if best_agg_cand == baseline_agg:
        best_agg = baseline_agg
        pos_w = None
        decisions['aggregation'] = f'{baseline_agg} train OOF best -> 유지'
    else:
        cand_train = postprocess.aggregate(xs_train, die_pred_train, best_agg_cand, pos_w_cand)
        cand_val = postprocess.aggregate(xs_val, die_pred_val, best_agg_cand, pos_w_cand)
        cand_test = postprocess.aggregate(xs_test, die_pred_test, best_agg_cand, pos_w_cand)
        cand_val_rmse = unit_rmse(cand_val, y_val_s)
        if cand_val_rmse < cur_val:
            train_unit, val_unit, test_unit = cand_train, cand_val, cand_test
            best_agg, pos_w = best_agg_cand, pos_w_cand
            decisions['aggregation'] = f'{best_agg_cand} 채택 ({cur_val:.9f} -> {cand_val_rmse:.9f})'
            cur_val = cand_val_rmse
        else:
            best_agg, pos_w = baseline_agg, None
            decisions['aggregation'] = f'{best_agg_cand} 거절 ({cur_val:.9f} <= {cand_val_rmse:.9f})'
    val_history.append((f'after_agg({best_agg})', cur_val))

    # 3단계: zero_clip. log-spaced 후보.
    zc_arr = np.logspace(
        np.log10(zero_clip_range[0]),
        np.log10(zero_clip_range[1]),
        zero_clip_n,
    )
    zc_res = postprocess.find_best_zero_clip(train_unit, y_train_unit_df, zc_arr, log_space=zero_clip_log_space)
    cand_zc = zc_res['best_threshold']
    cand_train = postprocess.apply_zero_clip(train_unit, cand_zc, log_space=zero_clip_log_space)
    cand_val = postprocess.apply_zero_clip(val_unit, cand_zc, log_space=zero_clip_log_space)
    cand_test = postprocess.apply_zero_clip(test_unit, cand_zc, log_space=zero_clip_log_space)
    cand_val_rmse = unit_rmse(cand_val, y_val_s)

    best_zc = None
    if cand_val_rmse < cur_val:
        train_unit, val_unit, test_unit = cand_train, cand_val, cand_test
        best_zc = cand_zc
        decisions['zero_clip'] = f'{cand_zc:.6f} 채택 ({cur_val:.9f} -> {cand_val_rmse:.9f})'
        cur_val = cand_val_rmse
    else:
        decisions['zero_clip'] = f'{cand_zc:.6f} 거절 ({cur_val:.9f} <= {cand_val_rmse:.9f})'
    val_history.append(('after_zero_clip', cur_val))

    train_rmse = unit_rmse(train_unit, y_train_s)
    return {
        'best_agg': best_agg,
        'pos_weights': pos_w,
        'best_zero_clip': best_zc,
        'zero_clip_log_space': zero_clip_log_space,
        'zero_clip_arr': zc_arr,
        'position_method': position_method,
        'agg_rmses': agg_res['rmse_per_method'],
        'decisions': decisions,
        'val_rmse_history': val_history,
        'train_rmse': train_rmse,
        'val_rmse_final': cur_val,
        'final_train_unit': train_unit,
        'final_val_unit': val_unit,
        'final_test_unit': test_unit,
    }


def iqr_stats(pred, y_true=None):
    pred = np.asarray(pred, dtype=float)
    q1, q3 = np.quantile(pred, [0.25, 0.75])
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    mask = pred > upper
    out = {
        'q1': float(q1),
        'q3': float(q3),
        'iqr': float(iqr),
        'upper_fence': float(upper),
        'n_upper_outliers': int(mask.sum()),
        'max_pred': float(np.max(pred)),
    }
    if y_true is not None and mask.any():
        yy = np.asarray(y_true, dtype=float)[mask]
        out.update({
            'outlier_true_mean': float(np.mean(yy)),
            'outlier_true_max': float(np.max(yy)),
            'outlier_true_ge_q95': int((yy >= np.quantile(y_true, 0.95)).sum()),
        })
    else:
        out.update({'outlier_true_mean': np.nan, 'outlier_true_max': np.nan, 'outlier_true_ge_q95': 0})
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
    """sklearn IsotonicRegression의 step function을 PCHIP monotonic cubic으로 smoothing한다.

    - knot 값(`X_thresholds_`, `y_thresholds_`)은 그대로 둔다. → 상단 끌어올림 폭은 step과 동일.
    - knot 사이만 PCHIP cubic 보간. → 평탄 plateau가 부드러운 곡선이 된다.
    - PAV가 보장하는 단조 증가성이 PCHIP에서도 유지된다 (PCHIP는 입력 monotonicity를 보존).

    Returns
    -------
    transform : callable. iso.transform과 같은 시그니처(raw -> calibrated).
    """
    x_knots = np.asarray(iso.X_thresholds_, dtype=float)
    y_knots = np.asarray(iso.y_thresholds_, dtype=float)
    # PAV 산출에서 X_thresholds_는 strictly increasing이지만, 방어적 dedupe.
    uniq_mask = np.concatenate([[True], np.diff(x_knots) > 0])
    x_knots = x_knots[uniq_mask]
    y_knots = y_knots[uniq_mask]
    if len(x_knots) < 2:
        # knot이 1개 이하면 PCHIP 불가능 → step 그대로 반환.
        def _fallback(x):
            return iso.transform(np.asarray(x, dtype=float))
        return _fallback, x_knots, y_knots

    pchip = PchipInterpolator(x_knots, y_knots, extrapolate=False)
    lo, hi = float(x_knots[0]), float(x_knots[-1])

    def _transform(x):
        x = np.asarray(x, dtype=float)
        x_clipped = np.clip(x, lo, hi)  # iso의 out_of_bounds='clip'과 같은 동작
        y_out = pchip(x_clipped)
        return np.clip(y_out, 0.0, None)  # y_min=0 강제

    return _transform, x_knots, y_knots


def fit_iso_tail_grid(train_unit, val_unit, test_unit, y_train_s, y_val_s, y_test_s,
                      *, iso_kinds, iso_weights, tail_qs, tail_resid_qs,
                      tail_gains, tail_powers, iqr_top_ks, iqr_margin):
    """unit-level 후처리 결과를 raw score로 보고 isotonic/tail 후보를 비교한다.

    seed sweep 대비 차이:
    - `iso_kind ∈ {'step', 'pchip'}`이 grid 차원에 추가됨. PCHIP는 step의 knot 값을 그대로 두고 plateau만 smoothing.
    - `iso_weights`에 0.25, 0.5가 추가되어 raw 비중↑ 후보도 함께 탐색.

    train OOF raw -> train y로 fit하고, validation raw에는 transform만 적용한다.
    tail 강화는 raw score 상단부에만 추가 보정을 걸어 RMSE와 outlier 형성을 동시에 노린다.
    """
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
            'name': name,
            'train_rmse': rmse(pred_train, y_train),
            'val_rmse': rmse(pred_val, y_val),
            # test_rmse는 모니터링용. 후보 선택(best 판정)에는 절대 쓰지 않는다 (val_rmse만 기준).
            'test_rmse': rmse(pred_test, y_test),
            'val_iqr_outliers': val_stats['n_upper_outliers'],
            'val_iqr_upper_fence': val_stats['upper_fence'],
            'val_max_pred': val_stats['max_pred'],
            'val_outlier_true_mean': val_stats['outlier_true_mean'],
            'val_outlier_true_max': val_stats['outlier_true_max'],
            'val_outlier_true_ge_q95': val_stats['outlier_true_ge_q95'],
            'val_top_pred_y_true': float(y_val[top_idx]),
            **params,
        }
        rows.append(rec)
        if best is None or rec['val_rmse'] < best['record']['val_rmse']:
            best = {
                'record': rec,
                'train_pred': pred_train,
                'val_pred': pred_val,
                'test_pred': pred_test,
                'iso_model': iso_model,
                'iso_kind': iso_kind,
                'pchip_knots': pchip_knots,
                'raw_train': raw_train,
                'raw_val': raw_val,
                'raw_test': raw_test,
            }

    # base 후보: iso/tail 없이 postprocess 결과 그대로.
    add_candidate(
        'base_postprocess', raw_train, raw_val, raw_test,
        {'uses_iso': False, 'iso_kind': 'none', 'iso_weight': 0.0,
         'tail_q': np.nan, 'tail_resid_q': np.nan,
         'tail_gain': 0.0, 'tail_power': np.nan, 'iqr_top_k': 0, 'tail_resid_scale': 0.0},
    )

    # PAV fit. step / pchip transform을 둘 다 미리 만들어 두고 itertools.product에서 룩업.
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0)
    iso.fit(raw_train, y_train)
    iso_train_step = iso.transform(raw_train)
    iso_val_step = iso.transform(raw_val)
    iso_test_step = iso.transform(raw_test)
    pchip_transform, pchip_x_knots, pchip_y_knots = build_iso_pchip_transform(iso)
    iso_train_pchip = pchip_transform(raw_train)
    iso_val_pchip = pchip_transform(raw_val)
    iso_test_pchip = pchip_transform(raw_test)

    iso_table = {
        'step':  (iso_train_step,  iso_val_step,  iso_test_step),
        'pchip': (iso_train_pchip, iso_val_pchip, iso_test_pchip),
    }

    for iso_kind, iso_weight, tail_q, tail_resid_q, tail_gain, tail_power, iqr_top_k in itertools.product(
        iso_kinds, iso_weights, tail_qs, tail_resid_qs, tail_gains, tail_powers, iqr_top_ks
    ):
        iso_train_arr, iso_val_arr, iso_test_arr = iso_table[iso_kind]

        # iso_weight=1이면 순수 isotonic, 1보다 크면 isotonic 방향으로 더 강하게 당긴다.
        base_train = raw_train + iso_weight * (iso_train_arr - raw_train)
        base_val = raw_val + iso_weight * (iso_val_arr - raw_val)
        base_test = raw_test + iso_weight * (iso_test_arr - raw_test)

        # tail_start 이상 영역만 ramp. tail_resid_scale은 train tail의 양의 residual 분위수.
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

        # IQR outlier push는 y_true를 보지 않고 예측 rank/quantile만 사용 - validation leakage 방지.
        pred_train = push_top_k_to_iqr(pred_train, raw_train, iqr_top_k, iqr_margin)
        pred_val = push_top_k_to_iqr(pred_val, raw_val, iqr_top_k, iqr_margin)
        pred_test = push_top_k_to_iqr(pred_test, raw_test, iqr_top_k, iqr_margin)

        name = f'iso{iso_kind}_w{iso_weight:g}_q{tail_q:g}_rq{tail_resid_q:g}_g{tail_gain:g}_p{tail_power:g}_iqr{iqr_top_k}'
        add_candidate(
            name, pred_train, pred_val, pred_test,
            {
                'uses_iso': True,
                'iso_kind': iso_kind,
                'iso_weight': float(iso_weight),
                'tail_q': float(tail_q),
                'tail_resid_q': float(tail_resid_q),
                'tail_gain': float(tail_gain),
                'tail_power': float(tail_power),
                'iqr_top_k': int(iqr_top_k),
                'tail_start': tail_start,
                'tail_denom': tail_denom,
                'tail_resid_scale': float(tail_resid_scale),
            },
            iso_model=iso,
            iso_kind=iso_kind,
            pchip_knots=(pchip_x_knots, pchip_y_knots) if iso_kind == 'pchip' else None,
        )

    cand = pd.DataFrame(rows).sort_values('val_rmse').reset_index(drop=True)
    iqr12 = cand[cand['val_iqr_outliers'].between(1, 2)].copy()
    best_iqr12 = iqr12.iloc[0].to_dict() if len(iqr12) else None

    best['candidates'] = cand
    best['best_iqr12'] = best_iqr12
    best['raw_train'] = raw_train
    best['raw_val'] = raw_val
    best['raw_test'] = raw_test
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
        'position': position,
        'pi': pi,
        'one_minus_pi': 1.0 - pi,
        'mu': mu,
        'pred_raw': pred_raw,
        'pred_taupi': pred_taupi,
        # 'pred' = tau_pi 게이트 적용된 최종 die 예측. 04_stacking discovery._read_die_split이
        # 표준 'pred' 컬럼을 요구하므로(reg='pred'/clf='prob'와 동일 계약, §5.2) 명시 노출한다.
        # pred_raw/pred_taupi는 진단용으로 함께 유지.
        'pred': pred_taupi,
    })
    out[TARGET_COL] = out[KEY_COL].map(y_unit_s)
    return out


def build_unit_output(y_unit_s, pred, pred_base):
    return pd.DataFrame({
        KEY_COL: y_unit_s.index.values,
        'pred': np.asarray(pred, dtype=float),
        'pred_base_postprocess': np.asarray(pred_base, dtype=float),
        TARGET_COL: y_unit_s.values,
    })


def serializable_calibrator(best_cal):
    """best 후보의 파라미터와 isotonic/PCHIP knot을 JSON에 박제한다."""
    rec = dict(best_cal['record'])
    iso = best_cal.get('iso_model')
    if iso is not None:
        rec['iso_x_thresholds'] = iso.X_thresholds_.tolist()
        rec['iso_y_thresholds'] = iso.y_thresholds_.tolist()
    pk = best_cal.get('pchip_knots')
    if pk is not None:
        rec['pchip_x_knots'] = pk[0].tolist()
        rec['pchip_y_knots'] = pk[1].tolist()
    return rec
