"""
Final 파이프라인 — 3-path OOF 블랜딩

- 입력: 경로별 unit-level OOF 예측 dict ({'A': df, 'B': df, 'C': df})
  각 df는 columns=[KEY_COL, 'pred']
- 제약: w_i ≥ 0, Σw_i = 1
- 최적화: SLSQP (수학해) + Optuna (검증용)
- 최종: val/test 에 동일 가중치 적용

사용법
------
    from final.modules import blending

    w_res = blending.blend_slsqp(
        {'A': oof_A, 'B': oof_B, 'C': oof_C},
        ys_train_unit,
    )
    weights = w_res['weights']     # dict

    final_val  = blending.apply_weights(
        {'A': val_A,  'B': val_B,  'C': val_C },  weights)
    final_test = blending.apply_weights(
        {'A': test_A, 'B': test_B, 'C': test_C},  weights)
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from utils.config import KEY_COL, TARGET_COL


def _align_preds(oofs_dict, reference_keys):
    """각 DataFrame을 reference_keys 순서로 정렬한 예측 행렬로 변환.

    Parameters
    ----------
    oofs_dict : {name: DataFrame[KEY_COL, 'pred']}
    reference_keys : array-like (unit id 순서)

    Returns
    -------
    names : list (입력 순서)
    P : np.ndarray (N, K)  — 각 열이 모델 k의 예측
    """
    names = list(oofs_dict.keys())
    P_cols = []
    for name in names:
        df = oofs_dict[name]
        aligned = df.set_index(KEY_COL).loc[reference_keys, "pred"].values
        P_cols.append(aligned)
    P = np.column_stack(P_cols)
    return names, P


def _rmse(weights, P, y):
    w = np.asarray(weights, dtype=float)
    return float(np.sqrt(np.mean((P @ w - y) ** 2)))


# ═════════════════════════════════════════════════════════════
# SLSQP (수학해)
# ═════════════════════════════════════════════════════════════

def blend_slsqp(oofs_dict, y_true_unit, bounds=(0.0, 1.0)):
    """SLSQP로 가중치 최적화 (제약: 합=1, 0 ≤ w ≤ 1).

    Returns
    -------
    dict {
        'weights': {name: float},
        'train_rmse': float,
        'converged': bool,
        'method': 'slsqp',
    }
    """
    y = y_true_unit.set_index(KEY_COL)[TARGET_COL]
    names, P = _align_preds(oofs_dict, y.index)
    K = len(names)
    y_arr = y.values

    w0 = np.full(K, 1.0 / K)
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bnds = [bounds] * K

    res = minimize(
        lambda w: _rmse(w, P, y_arr),
        w0, method="SLSQP", bounds=bnds, constraints=constraints,
    )
    weights = {n: float(w) for n, w in zip(names, res.x)}
    print(f"[blend SLSQP] weights={ {k: round(v, 4) for k,v in weights.items()} }, "
          f"rmse={res.fun:.6f}, converged={res.success}")
    return {
        "weights":    weights,
        "train_rmse": float(res.fun),
        "converged":  bool(res.success),
        "method":     "slsqp",
    }


# ═════════════════════════════════════════════════════════════
# Optuna (검증용)
# ═════════════════════════════════════════════════════════════

def blend_optuna(oofs_dict, y_true_unit, n_trials=300, seed=42):
    """Optuna로 가중치 탐색 (Dirichlet-style: raw suggest → softmax normalize).

    Returns
    -------
    dict {
        'weights': {name: float},
        'train_rmse': float,
        'study': optuna.Study,
        'method': 'optuna',
    }
    """
    import optuna

    y = y_true_unit.set_index(KEY_COL)[TARGET_COL]
    names, P = _align_preds(oofs_dict, y.index)
    K = len(names)
    y_arr = y.values

    def objective(trial):
        raw = np.array([
            trial.suggest_float(f"w_{n}", 1e-4, 1.0) for n in names
        ])
        w = raw / raw.sum()
        return _rmse(w, P, y_arr)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    raw = np.array([study.best_params[f"w_{n}"] for n in names])
    w = raw / raw.sum()
    weights = {n: float(wi) for n, wi in zip(names, w)}
    print(f"[blend Optuna] weights={ {k: round(v, 4) for k,v in weights.items()} }, "
          f"rmse={study.best_value:.6f}")
    return {
        "weights":    weights,
        "train_rmse": float(study.best_value),
        "study":      study,
        "method":     "optuna",
    }


# ═════════════════════════════════════════════════════════════
# 가중치 적용 (val / test)
# ═════════════════════════════════════════════════════════════

def apply_weights(preds_dict, weights):
    """unit-level 예측들에 가중치 적용 → 단일 DataFrame 반환.

    Parameters
    ----------
    preds_dict : {name: DataFrame[KEY_COL, 'pred']}
    weights : {name: float}

    Returns
    -------
    DataFrame [KEY_COL, 'pred']  — 첫 경로의 unit 순서 유지
    """
    names = list(preds_dict)
    if set(names) != set(weights):
        raise KeyError(f"preds vs weights mismatch: {names} vs {list(weights)}")

    # 첫 경로의 순서를 기준으로 정렬
    ref = preds_dict[names[0]]
    ref_keys = ref[KEY_COL].values

    blended = np.zeros(len(ref_keys))
    for n in names:
        aligned = preds_dict[n].set_index(KEY_COL).loc[ref_keys, "pred"].values
        blended += weights[n] * aligned

    return pd.DataFrame({KEY_COL: ref_keys, "pred": blended})


# ═════════════════════════════════════════════════════════════
# 편의 함수: train OOF로 weights fit → val/test에 apply 한 번에
# ═════════════════════════════════════════════════════════════

def fit_and_apply(
    train_preds, val_preds, test_preds, y_train_unit,
    method="slsqp", n_trials=300,
):
    """3 단계를 한 번에: train OOF로 weights 찾고 val/test에 적용.

    Parameters
    ----------
    train_preds, val_preds, test_preds : dict {name: DataFrame[KEY_COL, 'pred']}
    y_train_unit : DataFrame
    method : 'slsqp' | 'optuna'
    """
    if method == "slsqp":
        fit = blend_slsqp(train_preds, y_train_unit)
    elif method == "optuna":
        fit = blend_optuna(train_preds, y_train_unit, n_trials=n_trials)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    weights = fit["weights"]
    return {
        "weights":      weights,
        "train_rmse":   fit["train_rmse"],
        "train_blend":  apply_weights(train_preds, weights),
        "val_blend":    apply_weights(val_preds,   weights),
        "test_blend":   apply_weights(test_preds,  weights),
        "method":       fit["method"],
    }
