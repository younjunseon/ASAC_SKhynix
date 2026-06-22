"""Record의 메타 학습 결과를 사람이 읽고 재현할 수 있는 weight dict로 추출.

v2의 extract_weights를 die-level + 모듈 분리 버전으로 재작성.
SHAP/extra 컬럼 로직 없음. base 모델 가중치만.

사용 시점:
- best record (or top-N) 가 정해진 후, 그 record의 fit을 한 번 더 돌려서 가중치 추출
- learner 별 (ridge/nnls/mean/ENet/Combo) 형식이 다르므로 분기 처리
- ridge의 standardize 통계, ENet의 alpha/l1_ratio/scaler 통계, Combo의 5seeds 컴포넌트까지 박제
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, nnls
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import meta
from .config import StackingConfig
from .records import Record
from .search import ArrayBundle

KEY_COL = "ufs_serial"


# ---------------------------------------------------------------------------
# learner별 컴포넌트 추출 — fit 로직과 정확히 동치
# ---------------------------------------------------------------------------
def _fit_ridge_with_coef(Xo: np.ndarray, yo: np.ndarray, alpha: float):
    """fit_ridge_raw와 같은 식. intercept/weights/mu/sd 반환."""
    mu = Xo.mean(axis=0)
    sd = Xo.std(axis=0)
    sd[sd == 0] = 1.0
    Zo = np.column_stack([np.ones(len(Xo)), (Xo - mu) / sd])
    penalty = np.eye(Zo.shape[1]) * alpha
    penalty[0, 0] = 0.0
    lhs = Zo.T @ Zo + penalty
    rhs = Zo.T @ yo
    try:
        coef = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return float(coef[0]), coef[1:].astype(float), mu.astype(float), sd.astype(float)


def _fit_nnls_with_w(Xo: np.ndarray, yo: np.ndarray) -> np.ndarray:
    """fit_nnls_raw와 동치 — w만 반환."""
    w0, _ = nnls(Xo, yo, maxiter=50000)
    res = minimize(
        lambda w: float(np.mean((Xo @ w - yo) ** 2)),
        w0, method="L-BFGS-B",
        bounds=[(0.0, None)] * Xo.shape[1],
        options={"maxiter": 300, "ftol": 1e-12},
    )
    w = res.x if res.success else w0
    return w.astype(float)


def _iso_step_for_weights(raw_oof: np.ndarray, y_oof: np.ndarray) -> dict:
    """apply_iso의 isotonic step을 박제 (x_thresholds / y_thresholds)."""
    raw_oof = meta.clip_nonneg(raw_oof)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0)
    iso.fit(raw_oof, y_oof)
    return {
        "x_thresholds": iso.X_thresholds_.astype(float).tolist(),
        "y_thresholds": iso.y_thresholds_.astype(float).tolist(),
    }


def _enet_components(
    Xo: np.ndarray, yo: np.ndarray,
    seed: int, positive: bool,
    cfg: StackingConfig,
    groups: np.ndarray | None = None,
) -> dict:
    """fit_enet_cv_raw와 동치 — coef/intercept/scaler 통계 추출.

    groups가 주어지면 GroupKFold(unit) 내부 CV — refit 단계와 동일하게 동작 (재현용).
    """
    if groups is not None:
        cv_iter = list(GroupKFold(n_splits=cfg.enet_cv_folds).split(Xo, yo, groups=groups))
    else:
        cv_iter = KFold(cfg.enet_cv_folds, shuffle=True, random_state=seed)
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("en", ElasticNetCV(
            l1_ratio=list(cfg.enet_l1_ratio_grid),
            alphas=np.logspace(-6, 0, cfg.enet_alpha_n),
            cv=cv_iter,
            n_jobs=-1,
            max_iter=cfg.enet_max_iter,
            random_state=seed,
            positive=positive,
        )),
    ])
    pipe.fit(Xo, yo)
    sc = pipe.named_steps["sc"]
    en = pipe.named_steps["en"]
    return {
        "alpha": float(en.alpha_),
        "l1_ratio": float(en.l1_ratio_),
        "scaler_mean": sc.mean_.astype(float).tolist(),
        "scaler_scale": sc.scale_.astype(float).tolist(),
        "intercept": float(en.intercept_),
        "coef": en.coef_.astype(float).tolist(),
        "raw_oof": pipe.predict(Xo).astype(float),
    }


# ---------------------------------------------------------------------------
# 통합 — record → weights dict (재현 가능)
# ---------------------------------------------------------------------------
def extract_weights(
    rec: Record,
    bundle: ArrayBundle,
    cfg: StackingConfig,
) -> dict:
    """record 1개에 대해 fit을 재실행하고 사람-읽을 수 있는 weights dict 반환.

    closed-form (ridge/nnls/mean) 또는 random_state 고정 (ENet/Combo)이므로
    원본 record의 RMSE와 정확히 일치하는 가중치를 얻을 수 있다.
    """
    base_cols = tuple(sorted(bundle.names.index(x) for x in rec.pool_names))
    base_col_names = [bundle.names[i] for i in base_cols]

    # v4: always_include 모드면 extra_idx를 합산해서 X 슬라이싱.
    # learner.weights에 base + SHAP 가중치가 모두 박제되어 재현 가능.
    from .search import _use_cols
    use_cols = _use_cols(base_cols, bundle)
    Xo = bundle.X_oof[:, use_cols]
    if bundle.extra_names and bundle.shap_mode == "always_include":
        # always_include: extra_idx 컬럼이 base 뒤에 그대로 붙음 (build_array_bundle 규약).
        extra_col_names = list(bundle.extra_names)
    else:
        # searchable 모드는 SHAP이 base_col_names 안에 이미 섞여있음. 또는 SHAP 없음.
        extra_col_names = []
    all_col_names = base_col_names + extra_col_names  # X 컬럼 순서와 정확히 매칭

    y_oof = bundle.y_die_oof
    groups_oof = bundle.key_oof[KEY_COL].values   # GroupKFold용 (refit과 동일하게)
    method_full = rec.method

    out = {
        "tag": rec.tag,
        "stage": rec.stage,
        "method": method_full,
        "n_base": rec.n_base,
        "n_extra": rec.n_extra,
        "shap_mode": bundle.shap_mode,
        "base_models": base_col_names,
        "extra_features": extra_col_names,     # SHAP 컬럼명 (always_include) — searchable은 base_models에 섞임
        "extra_tags": list(rec.extra_tags),
        "rec_val_rmse": rec.val_rmse,
        "rec_test_rmse": rec.test_rmse,
        "rec_oof_rmse": rec.oof_rmse,
        "rec_val_rmse_die": rec.val_rmse_die,
        "rec_test_rmse_die": rec.test_rmse_die,
        "rec_oof_rmse_die": rec.oof_rmse_die,
        "rec_params": rec.params,
        "iso_weight": float(rec.params.get("iso_weight", 1.0)),
        "zero_tau": float(rec.params.get("zero_tau", 0.0)),
        "aggregation": rec.aggregation,
        "pi_threshold": rec.pi_threshold,
        "zero_clip": rec.zero_clip,
        "pos_weights": rec.pos_weights,
    }

    # method dispatch
    if method_full.startswith("ridge"):
        alpha = float(rec.params.get("alpha", 1e-5))
        intercept, w, mu, sd = _fit_ridge_with_coef(Xo, y_oof, alpha)
        out["learner"] = {
            "kind": "ridge",
            "alpha": alpha,
            "intercept": intercept,
            "standardize_mu": mu.tolist(),
            "standardize_sd": sd.tolist(),
            "weights": [{"name": n, "w": float(wi)} for n, wi in zip(all_col_names, w)],
        }
        raw_oof = np.column_stack([np.ones(len(Xo)), (Xo - mu) / sd]) @ np.concatenate([[intercept], w])

    elif method_full.startswith("nnls"):
        w = _fit_nnls_with_w(Xo, y_oof)
        out["learner"] = {
            "kind": "nnls",
            "weights": [{"name": n, "w": float(wi)} for n, wi in zip(all_col_names, w)],
        }
        raw_oof = Xo @ w

    elif method_full.startswith("mean"):
        out["learner"] = {
            "kind": "mean",
            "weights": [{"name": n, "w": 1.0 / len(all_col_names)} for n in all_col_names],
        }
        raw_oof = Xo.mean(axis=1)

    elif method_full.startswith("ENetPositive"):
        comp = _enet_components(Xo, y_oof, seed=cfg.seed, positive=True, cfg=cfg, groups=groups_oof)
        out["learner"] = {
            "kind": "ENetPositive",
            "alpha": comp["alpha"],
            "l1_ratio": comp["l1_ratio"],
            "scaler_mean": comp["scaler_mean"],
            "scaler_scale": comp["scaler_scale"],
            "intercept": comp["intercept"],
            "weights": [{"name": n, "w": float(c)} for n, c in zip(all_col_names, comp["coef"])],
        }
        raw_oof = comp["raw_oof"]

    elif method_full.startswith("ENet"):
        comp = _enet_components(Xo, y_oof, seed=cfg.seed, positive=False, cfg=cfg, groups=groups_oof)
        out["learner"] = {
            "kind": "ENet",
            "alpha": comp["alpha"],
            "l1_ratio": comp["l1_ratio"],
            "scaler_mean": comp["scaler_mean"],
            "scaler_scale": comp["scaler_scale"],
            "intercept": comp["intercept"],
            "weights": [{"name": n, "w": float(c)} for n, c in zip(all_col_names, comp["coef"])],
        }
        raw_oof = comp["raw_oof"]

    elif method_full.startswith("Combo"):
        seeds = list(rec.params.get("seeds", cfg.combo_seeds))
        bag_components = []
        oof_bag = np.zeros(len(y_oof))
        for s in seeds:
            comp = _enet_components(Xo, y_oof, seed=s, positive=False, cfg=cfg, groups=groups_oof)
            bag_components.append({
                "seed": s,
                "alpha": comp["alpha"],
                "l1_ratio": comp["l1_ratio"],
                "scaler_mean": comp["scaler_mean"],
                "scaler_scale": comp["scaler_scale"],
                "intercept": comp["intercept"],
                "weights": [{"name": n, "w": float(c)} for n, c in zip(all_col_names, comp["coef"])],
            })
            oof_bag += meta.clip_nonneg(comp["raw_oof"]) / len(seeds)
        # ★ Combo의 single-ENet은 refit(meta.fit_combo_raw)에서 seeds[0]로 fit됨 → 동일한 seed 사용해야 재현
        comp_single = _enet_components(Xo, y_oof, seed=seeds[0], positive=False, cfg=cfg, groups=groups_oof)
        w_nnls = _fit_nnls_with_w(Xo, y_oof)
        out["learner"] = {
            "kind": "Combo",
            "seeds": seeds,
            "components": {
                "enet_bag": bag_components,
                "enet_single": {
                    "alpha": comp_single["alpha"],
                    "l1_ratio": comp_single["l1_ratio"],
                    "scaler_mean": comp_single["scaler_mean"],
                    "scaler_scale": comp_single["scaler_scale"],
                    "intercept": comp_single["intercept"],
                    "weights": [{"name": n, "w": float(c)} for n, c in zip(all_col_names, comp_single["coef"])],
                },
                "nnls": {
                    "weights": [{"name": n, "w": float(wi)} for n, wi in zip(all_col_names, w_nnls)],
                },
            },
            "aggregation_formula": "raw = (mean(enet_bag) + enet_single + nnls) / 3",
        }
        oof_en = meta.clip_nonneg(comp_single["raw_oof"])
        oof_nn = meta.clip_nonneg(Xo @ w_nnls)
        raw_oof = (oof_bag + oof_en + oof_nn) / 3.0

    else:
        out["learner"] = {"kind": "unknown", "note": f"method={method_full!r} not handled"}
        return out

    # iso는 cfg.use_iso=True일 때만 박제 (record에 +Iso 접미사가 붙어 있을 때만 적용된 것)
    if cfg.use_iso and "Iso" in method_full:
        out["iso"] = _iso_step_for_weights(raw_oof, y_oof)
    return out
