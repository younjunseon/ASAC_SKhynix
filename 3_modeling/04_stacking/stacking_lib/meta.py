"""메타 학습기 5종 + 후처리.

base 모델들의 die-level 예측 (N_die, K) → die-level meta 예측 (N_die,).
- ridge       : closed-form L2 (standardize 후 fit)
- nnls        : 비음수 가중치 (scipy.nnls + L-BFGS-B refine)
- mean        : 단순 평균 (fit 없음)
- ENet        : ElasticNetCV (l1_ratio + alpha 그리드, KFold CV)
- ENetPositive: ENet w/ positive=True 제약
- Combo       : (ENet bag 5seeds + ENet single + NNLS) 평균

모든 학습기는 die-level 입력/출력만 다룬다. unit 집계는 aggregate.py에서 별도 수행.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, nnls
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------
def clip_nonneg(x: np.ndarray) -> np.ndarray:
    """음수 → 0. health는 비음수이므로 메타 예측도 음수면 잘라낸다."""
    return np.clip(np.asarray(x, dtype=float), 0.0, None)


def apply_iso(
    raw_oof: np.ndarray,
    raw_val: np.ndarray,
    raw_test: np.ndarray,
    y_oof: np.ndarray,
    iso_weight: float = 1.0,
    zero_tau: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Isotonic regression 후처리.

    iso_weight=1.0이면 순수 isotonic 출력. <1.0이면 raw와 blend.
    zero_tau>0이면 raw < zero_tau인 행을 0으로 (작은 양수 노이즈 cut).
    """
    raw_oof = clip_nonneg(raw_oof)
    raw_val = clip_nonneg(raw_val)
    raw_test = clip_nonneg(raw_test)

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0)
    iso.fit(raw_oof, y_oof)
    iso_oof = iso.transform(raw_oof)
    iso_val = iso.transform(raw_val)
    iso_test = iso.transform(raw_test)

    oof = iso_weight * iso_oof + (1.0 - iso_weight) * raw_oof
    val = iso_weight * iso_val + (1.0 - iso_weight) * raw_val
    test = iso_weight * iso_test + (1.0 - iso_weight) * raw_test

    if zero_tau > 0:
        oof = np.where(raw_oof < zero_tau, 0.0, oof)
        val = np.where(raw_val < zero_tau, 0.0, val)
        test = np.where(raw_test < zero_tau, 0.0, test)

    return clip_nonneg(oof), clip_nonneg(val), clip_nonneg(test)


def apply_no_iso(
    raw_oof: np.ndarray,
    raw_val: np.ndarray,
    raw_test: np.ndarray,
    zero_tau: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """iso 비적용 — raw 메타 예측을 non-negative clip만 해서 그대로 반환 (연속형 보존).

    cfg.use_iso=False 일 때 apply_iso 대신 호출. y_oof를 안 받는다 (iso fit 안 함).
    zero_tau는 그대로 지원.
    """
    raw_oof = clip_nonneg(raw_oof)
    raw_val = clip_nonneg(raw_val)
    raw_test = clip_nonneg(raw_test)

    if zero_tau > 0:
        raw_oof = np.where(raw_oof < zero_tau, 0.0, raw_oof)
        raw_val = np.where(raw_val < zero_tau, 0.0, raw_val)
        raw_test = np.where(raw_test < zero_tau, 0.0, raw_test)
    return raw_oof, raw_val, raw_test


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~(np.isnan(pred) | np.isnan(y))
    return float(np.sqrt(np.mean((pred[m] - y[m]) ** 2)))


# ---------------------------------------------------------------------------
# 학습기 — raw (iso 미적용) 예측 반환
# ---------------------------------------------------------------------------
def fit_ridge_raw(
    Xo: np.ndarray, yo: np.ndarray,
    Xv: np.ndarray, Xt: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """closed-form L2. standardize는 train(=oof) 통계로."""
    mu = Xo.mean(axis=0)
    sd = Xo.std(axis=0)
    sd[sd == 0] = 1.0

    def z(X):
        return (X - mu) / sd

    Zo = np.column_stack([np.ones(len(Xo)), z(Xo)])
    penalty = np.eye(Zo.shape[1]) * alpha
    penalty[0, 0] = 0.0
    lhs = Zo.T @ Zo + penalty
    rhs = Zo.T @ yo
    try:
        coef = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    def pred(X):
        return np.column_stack([np.ones(len(X)), z(X)]) @ coef

    return pred(Xo), pred(Xv), pred(Xt)


def fit_nnls_raw(
    Xo: np.ndarray, yo: np.ndarray,
    Xv: np.ndarray, Xt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """비음수 가중치. scipy.nnls를 초기값으로 L-BFGS-B 미세조정."""
    w0, _ = nnls(Xo, yo, maxiter=50000)
    res = minimize(
        lambda w: float(np.mean((Xo @ w - yo) ** 2)),
        w0,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * Xo.shape[1],
        options={"maxiter": 300, "ftol": 1e-12},
    )
    w = res.x if res.success else w0
    return Xo @ w, Xv @ w, Xt @ w


def fit_mean_raw(
    Xo: np.ndarray, yo: np.ndarray,
    Xv: np.ndarray, Xt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """단순 평균 — fit 데이터 yo는 사용 안 함."""
    return Xo.mean(axis=1), Xv.mean(axis=1), Xt.mean(axis=1)


def fit_enet_cv_raw(
    Xo: np.ndarray, yo: np.ndarray,
    Xv: np.ndarray, Xt: np.ndarray,
    seed: int = 42,
    positive: bool = False,
    l1_ratio_grid: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
    alpha_n: int = 30,
    max_iter: int = 20000,
    cv_folds: int = 5,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Pipeline]:
    """ElasticNetCV — l1_ratio × alpha 그리드 선택.

    groups가 주어지면 **GroupKFold(=unit 단위)** 로 fold를 쪼개서 alpha/l1_ratio 선택 시
    같은 unit의 4 die가 train/valid에 섞이지 않도록 한다 (leakage 방지).
    None이면 기존 동작(die-level KFold). 호출자(stacking)는 항상 groups를 넘긴다.
    """
    if groups is not None:
        # GroupKFold는 random_state를 받지 않는다 — group 순서 자체가 fold를 결정.
        # split 결과(인덱스 리스트)를 cv에 직접 넘기면 ElasticNetCV가 그대로 사용.
        gkf = GroupKFold(n_splits=cv_folds)
        cv_iter = list(gkf.split(Xo, yo, groups=groups))
    else:
        cv_iter = KFold(cv_folds, shuffle=True, random_state=seed)
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("en", ElasticNetCV(
            l1_ratio=list(l1_ratio_grid),
            alphas=np.logspace(-6, 0, alpha_n),
            cv=cv_iter,
            n_jobs=-1,
            max_iter=max_iter,
            random_state=seed,
            positive=positive,
        )),
    ])
    pipe.fit(Xo, yo)
    return pipe.predict(Xo), pipe.predict(Xv), pipe.predict(Xt), pipe


# ---------------------------------------------------------------------------
# 통합 dispatch — method 이름으로 fit_*_raw 호출
# ---------------------------------------------------------------------------
def fit_meta_raw(
    method: str,
    Xo: np.ndarray, yo: np.ndarray,
    Xv: np.ndarray, Xt: np.ndarray,
    *,
    alpha: float = 1e-5,
    seed: int = 42,
    positive: bool = False,
    enet_kwargs: dict | None = None,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """method ∈ {'ridge', 'nnls', 'mean', 'enet', 'enet_positive'} dispatch.

    groups가 주어지면 ENet 계열은 GroupKFold(unit) 내부 CV를 사용 (leakage 방지).
    """
    if method == "ridge":
        return fit_ridge_raw(Xo, yo, Xv, Xt, alpha)
    if method == "nnls":
        return fit_nnls_raw(Xo, yo, Xv, Xt)
    if method == "mean":
        return fit_mean_raw(Xo, yo, Xv, Xt)
    if method in ("enet", "enet_positive"):
        kwargs = enet_kwargs or {}
        ro, rv, rt, _ = fit_enet_cv_raw(
            Xo, yo, Xv, Xt,
            seed=seed,
            positive=(method == "enet_positive" or positive),
            groups=groups,
            **kwargs,
        )
        return ro, rv, rt
    raise ValueError(f"Unknown method: {method}")


def fit_combo_raw(
    Xo: np.ndarray, yo: np.ndarray,
    Xv: np.ndarray, Xt: np.ndarray,
    *,
    seeds: tuple[int, ...] = (42, 123, 456, 789, 2024),
    enet_kwargs: dict | None = None,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combo: bag(ENet over seeds) + single ENet + NNLS 의 평균.

    raw 단계까지만 (iso는 호출자에서 apply_iso).
    groups가 주어지면 ENet 계열은 GroupKFold(unit) 내부 CV 사용 (leakage 방지).
    """
    kwargs = enet_kwargs or {}

    oof_bag = np.zeros(len(yo))
    val_bag = np.zeros(len(Xv))
    test_bag = np.zeros(len(Xt))
    for s in seeds:
        ro, rv, rt, _ = fit_enet_cv_raw(Xo, yo, Xv, Xt, seed=s, positive=False, groups=groups, **kwargs)
        oof_bag += clip_nonneg(ro) / len(seeds)
        val_bag += clip_nonneg(rv) / len(seeds)
        test_bag += clip_nonneg(rt) / len(seeds)

    ro_en, rv_en, rt_en, _ = fit_enet_cv_raw(Xo, yo, Xv, Xt, seed=seeds[0], positive=False, groups=groups, **kwargs)
    oof_en, val_en, test_en = clip_nonneg(ro_en), clip_nonneg(rv_en), clip_nonneg(rt_en)

    oof_nn, val_nn, test_nn = fit_nnls_raw(Xo, yo, Xv, Xt)
    oof_nn, val_nn, test_nn = clip_nonneg(oof_nn), clip_nonneg(val_nn), clip_nonneg(test_nn)

    raw_oof = (oof_bag + oof_en + oof_nn) / 3.0
    raw_val = (val_bag + val_en + val_nn) / 3.0
    raw_test = (test_bag + test_en + test_nn) / 3.0
    return raw_oof, raw_val, raw_test


# ---------------------------------------------------------------------------
# meta_cv_oof — GroupKFold OOF 기반 ridge+iso die-level RMSE
# ---------------------------------------------------------------------------
def compute_meta_cv_oof_die(
    Xo: np.ndarray,
    yo: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    alpha: float = 1e-5,
    use_iso: bool = True,
) -> tuple[float, np.ndarray]:
    """die-level GroupKFold OOF: ridge(+iso) fit on train fold, predict holdout fold.

    use_iso=False면 iso 생략, raw ridge clip 출력 사용.

    Returns
    -------
    rmse_die : float — die-level RMSE (y가 unit broadcast이므로 unit RMSE보다 약간 큼)
    oof_pred : (N_die,) — fold별 predict을 stitch한 die-level 예측
    """
    meta_preds = np.zeros(len(yo))
    for tr_idx, va_idx in splits:
        X_tr, y_tr = Xo[tr_idx], yo[tr_idx]
        X_va = Xo[va_idx]
        mu = X_tr.mean(axis=0)
        sd = X_tr.std(axis=0)
        sd[sd == 0] = 1.0
        Z_tr = np.column_stack([np.ones(len(X_tr)), (X_tr - mu) / sd])
        Z_va = np.column_stack([np.ones(len(X_va)), (X_va - mu) / sd])
        penalty = np.eye(Z_tr.shape[1]) * alpha
        penalty[0, 0] = 0.0
        try:
            coef = np.linalg.solve(Z_tr.T @ Z_tr + penalty, Z_tr.T @ y_tr)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(Z_tr.T @ Z_tr + penalty, Z_tr.T @ y_tr, rcond=None)[0]
        raw_tr = clip_nonneg(Z_tr @ coef)
        raw_va = clip_nonneg(Z_va @ coef)
        if use_iso:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0)
            iso.fit(raw_tr, y_tr)
            meta_preds[va_idx] = iso.transform(raw_va)
        else:
            meta_preds[va_idx] = raw_va
    meta_preds = clip_nonneg(meta_preds)
    return rmse(meta_preds, yo), meta_preds
