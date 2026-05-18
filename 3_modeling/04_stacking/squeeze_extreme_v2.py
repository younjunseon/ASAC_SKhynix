"""
squeeze_extreme_v2.py
=====================

squeeze_extreme.py의 업그레이드 버전.

핵심 차이
---------
- `--extra-cache <PATH>` (반복 지정 가능)로 추가 unit-level 메타 입력 컬럼을 로드한다.
  각 PATH 는 `oof_unit_shap.{parquet,csv}`, `val_unit_shap.*`, `test_unit_shap.*` 형식의 캐시 폴더.
  보통은 `build_shap_features.py`가 생성한 폴더를 가리킨다.
- 이 extra 컬럼들은 **subset search 후보에 들어가지 않고**, 메타 학습/평가 시에는 **항상 입력**으로 포함된다.
  즉 squeeze가 고르는 건 'base 예측값 subset'이고, SHAP 컬럼들은 항상 같이 들어가서 메타가 그 위에서 weighting 학습.
- val_rmse 는 v1 그대로 매 record에 기록 (모든 stage), OOF 최소 선택도 v1 그대로 (--select_by oof default).
- meta_cv_oof 계산도 extra cols 포함해서 — 추가 컬럼이 진짜 일반화에 기여하는지 측정.

사용 예시
--------
    # base만 (v1과 동일)
    python squeeze_extreme_v2.py --random_trials 4000 --top_refit 40

    # base + lgbm/hp/002 SHAP top-50 추가
    python squeeze_extreme_v2.py \
        --extra-cache shap_cache/02_reg_single__lgbm__hp__002 \
        --random_trials 4000 --top_refit 40

    # SHAP을 여러 base에서 (확장 단계)
    python squeeze_extreme_v2.py \
        --extra-cache shap_cache/02_reg_single__lgbm__hp__002 \
        --extra-cache shap_cache/02_reg_single__xgb__hp__002 \
        --random_trials 6000 --top_refit 50 --optuna_trials 300
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "4_output"
RESULT_DIR = Path(__file__).resolve().parent / "results_extreme_v2"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

KEY_COL = "ufs_serial"
DEFAULT_SEED = 42
KNOWN_STRONG_SUBSET = [
    "03_two_stage__default__clf__lgbm__hp__002",
    "01_zit__zit_only__hp__002",
    "02_reg_single__et__hp__001",
    "02_reg_single__catboost__raw__001",
]


@dataclass
class Record:
    tag: str
    stage: str
    method: str
    n_base: int            # subset에서 고른 base 모델 수
    n_extra: int           # extra(SHAP 등) 컬럼 수 — 항상 같은 값이지만 record에 박제
    val_rmse: float
    test_rmse: float
    oof_rmse: float
    pool_names: list[str]   # 선택된 base 이름들
    extra_tags: list[str]   # extra cache 식별자들 (재현용)
    params: dict
    meta_cv_oof_rmse: float = float("nan")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--oof_rmse_cutoff", type=float, default=0.00555)
    p.add_argument("--no_clf", action="store_true")
    p.add_argument("--include_combined", action="store_true",
                   help="Include two-stage/default/combined outputs. Default: excluded.")
    p.add_argument("--include_ts_reg", action="store_true",
                   help="Include two-stage/default/reg outputs. Default: excluded.")
    p.add_argument("--min_subset_size", type=int, default=2)
    p.add_argument("--max_subset_size", type=int, default=18)
    p.add_argument("--random_trials", type=int, default=6000)
    p.add_argument("--local_seeds", type=int, default=25)
    p.add_argument("--local_steps", type=int, default=30)
    p.add_argument("--local_candidate_limit", type=int, default=35)
    p.add_argument("--top_refit", type=int, default=50)
    p.add_argument("--combo_refit", type=int, default=20)
    p.add_argument("--optuna_trials", type=int, default=0)
    p.add_argument("--select_by", choices=["oof", "val", "meta_cv_oof"], default="oof",
                   help="Metric used for search/selection. Default: oof (사용자 요구).")
    p.add_argument("--allow_val_fit", action="store_true",
                   help="Use val labels directly for weights/postprocess. Strong peek-bias.")
    p.add_argument("--deadline", type=str, default=None,
                   help='Stop before this local datetime, e.g. "2026-05-16 08:00".')
    p.add_argument("--deadline_margin_minutes", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--extra-cache", action="append", default=[],
                   help="추가 메타 입력 컬럼 캐시 폴더(반복 지정 가능). build_shap_features.py의 출력 폴더.")
    p.add_argument("--extra-prefix-with-tag", action=argparse.BooleanOptionalAction, default=True,
                   help="extra cache 컬럼 이름 앞에 캐시 태그를 prefix 해서 충돌 방지 (default ON)")
    p.add_argument("--extra-top-k", type=int, default=0,
                   help="각 extra cache 내 feature 중 처음 K개만 사용 (0=전체). build_shap_features.py가 importance 내림차순으로 저장 → 자동 top-K importance.")
    p.add_argument("--extra-mode", choices=["always_include", "searchable"], default="always_include",
                   help="extra cache 컬럼을 다루는 방식. always_include=메타 학습 시 항상 포함(v2 기본), "
                        "searchable=base와 동등하게 subset search 후보로 두기 (ridge/L1이 알아서 선택).")
    return p.parse_args()


def rmse(pred, y) -> float:
    pred = np.asarray(pred, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~(np.isnan(pred) | np.isnan(y))
    return float(np.sqrt(np.mean((pred[m] - y[m]) ** 2)))


def score_rec(rec: Record, select_by: str) -> float:
    if select_by == "val":
        return rec.val_rmse
    if select_by == "meta_cv_oof":
        if not math.isnan(rec.meta_cv_oof_rmse):
            return rec.meta_cv_oof_rmse
        return rec.oof_rmse
    return rec.oof_rmse


def parse_deadline(text: str | None):
    if not text:
        return None
    text = text.strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%m-%d %H:%M"):
        try:
            dt = datetime.strptime(text, fmt)
            if fmt == "%m-%d %H:%M":
                dt = dt.replace(year=datetime.now().year)
            return dt
        except ValueError:
            pass
    raise ValueError(f"Unsupported --deadline format: {text!r}")


def seconds_left(deadline) -> float:
    if deadline is None:
        return float("inf")
    return (deadline - datetime.now()).total_seconds()


def should_stop(deadline, margin_minutes: float) -> bool:
    return seconds_left(deadline) <= margin_minutes * 60.0


def load_split(dirpath: Path, split: str) -> pd.DataFrame:
    return pd.read_csv(dirpath / f"{split}_unit.csv")


def get_oof_rmse(dirpath: Path) -> float:
    try:
        df = load_split(dirpath, "oof")
        return rmse(df["pred"].values, df["health"].values)
    except Exception:
        return 999.0


def discover_models(args: argparse.Namespace) -> list[dict]:
    skip = {"_old", "_subset_search", "base", "curated", "_cache", "_temp"}
    if not args.include_combined:
        skip.add("combined")
    if not args.include_ts_reg:
        skip.add("reg")

    models = []
    for dp in sorted(OUTPUT_DIR.rglob("oof_unit.csv")):
        dirpath = dp.parent
        parts_all = set(dirpath.parts)
        if any(s in parts_all for s in skip):
            continue
        if not (dirpath / "val_unit.csv").exists():
            continue
        if not (dirpath / "test_unit.csv").exists():
            continue

        rel = dirpath.relative_to(OUTPUT_DIR)
        parts = rel.parts
        if "clf" in parts:
            category = "clf"
        elif len(parts) and parts[0] == "01_zit":
            category = "zit"
        elif len(parts) and parts[0] == "02_reg_single":
            category = "reg_single"
        elif "reverse" in parts:
            category = "reverse"
        elif "combined" in parts:
            category = "combined"
        elif "reg" in parts:
            category = "ts_reg"
        else:
            category = "other"

        variant = next((x for x in parts if x in ("raw", "hp", "pphp")), "default")
        oof_r = get_oof_rmse(dirpath)
        if oof_r <= args.oof_rmse_cutoff and not (args.no_clf and category == "clf"):
            models.append({
                "name": "__".join(parts),
                "path": dirpath,
                "rel": str(rel),
                "category": category,
                "variant": variant,
                "oof_rmse": oof_r,
            })
    models.sort(key=lambda m: m["oof_rmse"])
    return models


def build_base_matrix(models: list[dict], split: str):
    """base 모델 예측값으로만 X 행렬 구성. y와 keys도 같이 반환."""
    dfs = {}
    for m in models:
        df = load_split(m["path"], split)
        dfs[m["name"]] = df.set_index(KEY_COL)["pred"]

    ref = load_split(models[0]["path"], split).set_index(KEY_COL)
    keys = ref.index
    y = ref["health"].values if "health" in ref.columns else np.zeros(len(keys))
    X = np.column_stack([dfs[m["name"]].reindex(keys).values for m in models])
    return X.astype(float), y.astype(float), list(keys)


# ---------------------------------------------------------------------------
# extra cache loader  — build_shap_features.py 출력 디렉토리를 받아서 컬럼 배열로
# ---------------------------------------------------------------------------
def _read_cache_split(dirpath: Path, split: str) -> pd.DataFrame:
    """{oof,val,test}_unit_shap.{parquet,csv}를 찾아서 DataFrame 반환 (KEY_COL 포함)."""
    base = f"{split}_unit_shap"
    p_pq = dirpath / f"{base}.parquet"
    p_csv = dirpath / f"{base}.csv"
    if p_pq.exists():
        return pd.read_parquet(p_pq)
    if p_csv.exists():
        return pd.read_csv(p_csv)
    raise FileNotFoundError(f"{p_pq} / {p_csv} 둘 다 없음")


def load_extra_caches(cache_paths: list[str], keys_oof, keys_val, keys_test,
                      prefix_with_tag: bool, top_k: int = 0
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[int]]:
    """여러 캐시 폴더를 합쳐서 (X_oof_extra, X_val_extra, X_test_extra, extra_col_names, cache_tags) 반환.

    각 캐시 split DataFrame을 KEY_COL 기준 재정렬한 뒤 컬럼 가로 결합.
    행 누락 시 RuntimeError. tag prefix는 컬럼 이름 충돌 방지용.
    """
    if not cache_paths:
        return None, None, None, [], [], []

    all_oof, all_val, all_test = [], [], []
    all_names: list[str] = []
    tags: list[str] = []
    cache_col_counts: list[int] = []   # cache별로 가져온 cols 수 (균형 candidate_pool용)
    for raw_path in cache_paths:
        dpath = Path(raw_path)
        if not dpath.is_absolute():
            # 04_stacking 디렉토리 기준 상대경로도 허용
            cand = Path(__file__).resolve().parent / raw_path
            if cand.exists():
                dpath = cand
        if not dpath.exists():
            raise FileNotFoundError(f"extra cache 폴더 없음: {raw_path}")

        tag = dpath.name
        tags.append(tag)
        print(f"  [extra cache] {tag}  ({dpath})")

        oof_df  = _read_cache_split(dpath, "oof")
        val_df  = _read_cache_split(dpath, "val")
        test_df = _read_cache_split(dpath, "test")

        # 캐시 내 컬럼명 = KEY_COL + feature cols (build_shap_features.py는 importance 내림차순으로 저장)
        feat_cols = [c for c in oof_df.columns if c != KEY_COL]
        if top_k > 0 and len(feat_cols) > top_k:
            feat_cols = feat_cols[:top_k]
            print(f"    [top-K] {top_k}개로 제한 (전체 cache cols 중 처음 {top_k})")
        if prefix_with_tag:
            renamed = {c: f"{tag}::{c}" for c in feat_cols}
            oof_df  = oof_df.rename(columns=renamed)
            val_df  = val_df.rename(columns=renamed)
            test_df = test_df.rename(columns=renamed)
            feat_cols = [renamed[c] for c in feat_cols]
        all_names.extend(feat_cols)
        cache_col_counts.append(len(feat_cols))

        for name, df, keys in [("oof", oof_df, keys_oof),
                               ("val", val_df, keys_val),
                               ("test", test_df, keys_test)]:
            if KEY_COL not in df.columns:
                raise RuntimeError(f"{tag}/{name}: KEY_COL 없음 ({list(df.columns)[:5]}...)")
            idx_df = df.set_index(KEY_COL)
            missing = set(keys) - set(idx_df.index)
            if missing:
                raise RuntimeError(
                    f"{tag}/{name}: base와 unit 매칭 누락 {len(missing)}개 (예: {list(missing)[:3]})"
                )
            arr = idx_df.reindex(keys)[feat_cols].values.astype(np.float32)
            if name == "oof":
                all_oof.append(arr)
            elif name == "val":
                all_val.append(arr)
            else:
                all_test.append(arr)

    X_oof  = np.hstack(all_oof).astype(float)
    X_val  = np.hstack(all_val).astype(float)
    X_test = np.hstack(all_test).astype(float)
    print(f"  [extra total] cols={len(all_names)} from {len(tags)} cache(s) "
          f"(per-cache cols: {cache_col_counts})")
    return X_oof, X_val, X_test, all_names, tags, cache_col_counts


def clip_nonneg(x):
    return np.clip(np.asarray(x, dtype=float), 0.0, None)


def apply_iso(raw_oof, raw_val, raw_test, y_oof, iso_weight=1.0, zero_tau=0.0):
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


def compute_meta_cv_oof(use_cols, arrays, n_splits=5, seed=42, alpha=1e-5) -> float:
    """5-fold CV on OOF matrix: ridge+iso fit on train fold, predict on holdout fold.

    use_cols: subset에서 고른 base index들 + (있다면) extra index들 — 모두 X_oof[:, use_cols]에 들어감.
    """
    X_oof, y_oof = arrays[0], arrays[1]
    Xo = X_oof[:, list(use_cols)]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    meta_preds = np.zeros(len(y_oof))
    for tr_idx, va_idx in kf.split(Xo):
        X_tr, y_tr = Xo[tr_idx], y_oof[tr_idx]
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
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0)
        iso.fit(raw_tr, y_tr)
        meta_preds[va_idx] = iso.transform(raw_va)
    return float(rmse(clip_nonneg(meta_preds), y_oof))


def fit_ridge_raw(Xo, yo, Xv, Xt, alpha: float):
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


def fit_nnls_raw(Xo, yo, Xv, Xt):
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


def fit_mean_raw(Xo, yo, Xv, Xt):
    return Xo.mean(axis=1), Xv.mean(axis=1), Xt.mean(axis=1)


def _combine_cols(base_cols: tuple, extra_idx: tuple) -> tuple:
    """base subset + extra(항상 포함) = 실제 X[:, use_cols] 슬라이싱 인덱스."""
    return tuple(base_cols) + tuple(extra_idx)


def eval_fast(
    base_cols: tuple[int, ...],
    extra_idx: tuple[int, ...],
    X_oof, y_oof, X_val, y_val, X_test, y_test,
    names, extra_tags,
    method: str,
    alpha: float = 1e-5,
    iso_weight: float = 1.0,
    zero_tau: float = 0.0,
    stage: str = "fast",
) -> tuple[Record, np.ndarray, np.ndarray]:
    use_cols = _combine_cols(base_cols, extra_idx)
    Xo = X_oof[:, use_cols]
    Xv = X_val[:, use_cols]
    Xt = X_test[:, use_cols]
    if method == "ridge":
        ro, rv, rt = fit_ridge_raw(Xo, y_oof, Xv, Xt, alpha)
    elif method == "nnls":
        ro, rv, rt = fit_nnls_raw(Xo, y_oof, Xv, Xt)
    elif method == "mean":
        ro, rv, rt = fit_mean_raw(Xo, y_oof, Xv, Xt)
    else:
        raise ValueError(method)

    po, pv, pt = apply_iso(ro, rv, rt, y_oof, iso_weight=iso_weight, zero_tau=zero_tau)
    rec = Record(
        tag=f"{stage}__{method}__k{len(base_cols)}",
        stage=stage,
        method=f"{method}+Iso",
        n_base=len(base_cols),
        n_extra=len(extra_idx),
        val_rmse=rmse(pv, y_val),
        test_rmse=rmse(pt, y_test),
        oof_rmse=rmse(po, y_oof),
        pool_names=[names[i] for i in base_cols],
        extra_tags=list(extra_tags),
        params={"alpha": alpha, "iso_weight": iso_weight, "zero_tau": zero_tau},
    )
    return rec, pv, pt


def eval_fast_arrays(base_cols, extra_idx, arrays, names, extra_tags, **kwargs):
    return eval_fast(
        base_cols, extra_idx,
        arrays[0], arrays[1], arrays[2], arrays[3], arrays[4], arrays[5],
        names, extra_tags,
        **kwargs,
    )


def best_fast_for_subset(base_cols, extra_idx, arrays, names, extra_tags, alpha_grid, select_by="oof"):
    best = None
    for method in ("mean", "nnls"):
        try:
            rec, _, _ = eval_fast_arrays(base_cols, extra_idx, arrays, names, extra_tags, method=method)
            if best is None or score_rec(rec, select_by) < score_rec(best, select_by):
                best = rec
        except Exception:
            pass
    for alpha in alpha_grid:
        try:
            rec, _, _ = eval_fast_arrays(base_cols, extra_idx, arrays, names, extra_tags,
                                          method="ridge", alpha=alpha)
            if best is None or score_rec(rec, select_by) < score_rec(best, select_by):
                best = rec
        except Exception:
            pass
    return best


def fit_enet_cv_raw(Xo, yo, Xv, Xt, seed=DEFAULT_SEED, positive=False):
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("en", ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            alphas=np.logspace(-6, 0, 30),
            cv=KFold(5, shuffle=True, random_state=seed),
            n_jobs=-1,
            max_iter=20000,
            random_state=seed,
            positive=positive,
        )),
    ])
    pipe.fit(Xo, yo)
    return pipe.predict(Xo), pipe.predict(Xv), pipe.predict(Xt), pipe


def eval_enet(base_cols, extra_idx, arrays, names, extra_tags, seed, positive=False, stage="refit"):
    X_oof, y_oof, X_val, y_val, X_test, y_test = arrays
    use_cols = _combine_cols(base_cols, extra_idx)
    ro, rv, rt, pipe = fit_enet_cv_raw(
        X_oof[:, use_cols], y_oof, X_val[:, use_cols], X_test[:, use_cols],
        seed=seed, positive=positive,
    )
    po, pv, pt = apply_iso(ro, rv, rt, y_oof)
    en = pipe.named_steps["en"]
    rec = Record(
        tag=f"{stage}__ENet{'Positive' if positive else ''}+Iso__k{len(base_cols)}",
        stage=stage,
        method=f"ENet{'Positive' if positive else ''}+Iso",
        n_base=len(base_cols),
        n_extra=len(extra_idx),
        val_rmse=rmse(pv, y_val),
        test_rmse=rmse(pt, y_test),
        oof_rmse=rmse(po, y_oof),
        pool_names=[names[i] for i in base_cols],
        extra_tags=list(extra_tags),
        params={
            "alpha": float(en.alpha_),
            "l1_ratio": float(en.l1_ratio_),
            "positive": positive,
        },
    )
    return rec, pv, pt


def eval_combo(base_cols, extra_idx, arrays, names, extra_tags, seed, stage="refit"):
    X_oof, y_oof, X_val, y_val, X_test, y_test = arrays
    use_cols = _combine_cols(base_cols, extra_idx)
    seeds = [seed, 123, 456, 789, 2024]

    oof_bag = np.zeros(len(y_oof))
    val_bag = np.zeros(len(y_val))
    test_bag = np.zeros(len(y_test))
    for s in seeds:
        ro, rv, rt, _ = fit_enet_cv_raw(
            X_oof[:, use_cols], y_oof, X_val[:, use_cols], X_test[:, use_cols],
            seed=s, positive=False,
        )
        oof_bag += clip_nonneg(ro) / len(seeds)
        val_bag += clip_nonneg(rv) / len(seeds)
        test_bag += clip_nonneg(rt) / len(seeds)

    ro, rv, rt, _ = fit_enet_cv_raw(
        X_oof[:, use_cols], y_oof, X_val[:, use_cols], X_test[:, use_cols],
        seed=seed, positive=False,
    )
    oof_en, val_en, test_en = clip_nonneg(ro), clip_nonneg(rv), clip_nonneg(rt)

    oof_nn, val_nn, test_nn = fit_nnls_raw(
        X_oof[:, use_cols], y_oof, X_val[:, use_cols], X_test[:, use_cols]
    )
    oof_nn, val_nn, test_nn = clip_nonneg(oof_nn), clip_nonneg(val_nn), clip_nonneg(test_nn)

    raw_oof = (oof_bag + oof_en + oof_nn) / 3.0
    raw_val = (val_bag + val_en + val_nn) / 3.0
    raw_test = (test_bag + test_en + test_nn) / 3.0
    po, pv, pt = apply_iso(raw_oof, raw_val, raw_test, y_oof)

    rec = Record(
        tag=f"{stage}__Combo+Iso__k{len(base_cols)}",
        stage=stage,
        method="Combo+Iso",
        n_base=len(base_cols),
        n_extra=len(extra_idx),
        val_rmse=rmse(pv, y_val),
        test_rmse=rmse(pt, y_test),
        oof_rmse=rmse(po, y_oof),
        pool_names=[names[i] for i in base_cols],
        extra_tags=list(extra_tags),
        params={"seeds": seeds},
    )
    return rec, pv, pt


def make_seed_subsets(models, names, args):
    n = len(names)
    subsets = set()
    by_cat = {}
    by_variant = {}
    for i, m in enumerate(models):
        by_cat.setdefault(m["category"], []).append(i)
        by_variant.setdefault(m["variant"], []).append(i)

    for k in [2, 3, 4, 5, 8, 10, 15, 20, n]:
        if args.min_subset_size <= k <= min(args.max_subset_size, n):
            subsets.add(tuple(range(k)))

    for xs in by_cat.values():
        if len(xs) >= args.min_subset_size:
            subsets.add(tuple(sorted(xs[:args.max_subset_size])))
    for xs in by_variant.values():
        if len(xs) >= args.min_subset_size:
            subsets.add(tuple(sorted(xs[:args.max_subset_size])))

    strong = [names.index(x) for x in KNOWN_STRONG_SUBSET if x in names]
    if len(strong) == len(KNOWN_STRONG_SUBSET):
        subsets.add(tuple(sorted(strong)))

    return sorted(subsets, key=lambda x: (len(x), x))


def sample_random_subsets(n_base, args, rng):
    max_k = min(args.max_subset_size, n_base)
    min_k = min(args.min_subset_size, max_k)
    possible = sum(math.comb(n_base, k) for k in range(min_k, max_k + 1))
    target = min(args.random_trials, possible)
    rank = np.arange(n_base, dtype=float)
    prob = 1.0 / np.power(rank + 1.0, 0.65)
    prob = prob / prob.sum()
    seen = set()
    while len(seen) < target:
        k = int(rng.integers(min_k, max_k + 1))
        cols = tuple(sorted(rng.choice(np.arange(n_base), size=k, replace=False, p=prob).tolist()))
        seen.add(cols)
    return list(seen)


def local_improve(seed_cols, extra_idx, arrays, names, extra_tags, args, alpha_grid, candidate_pool):
    current = tuple(sorted(seed_cols))
    best = best_fast_for_subset(current, extra_idx, arrays, names, extra_tags, alpha_grid, args.select_by)
    if best is None:
        return []

    records = [best]
    all_idx = list(candidate_pool)
    for _ in range(args.local_steps):
        proposals = set()
        if len(current) < args.max_subset_size:
            for j in all_idx:
                if j not in current:
                    proposals.add(tuple(sorted(current + (j,))))
        if len(current) > args.min_subset_size:
            for j in current:
                proposals.add(tuple(x for x in current if x != j))
        if len(current) > args.min_subset_size:
            for drop in current:
                base = tuple(x for x in current if x != drop)
                for add in all_idx:
                    if add not in current:
                        proposals.add(tuple(sorted(base + (add,))))

        step_best = best
        for cols in proposals:
            rec = best_fast_for_subset(cols, extra_idx, arrays, names, extra_tags, alpha_grid, args.select_by)
            if rec is not None and score_rec(rec, args.select_by) < score_rec(step_best, args.select_by) - 1e-10:
                step_best = rec
        if step_best is best:
            break
        current = tuple(names.index(x) for x in step_best.pool_names)
        best = step_best
        records.append(best)
    for r in records:
        r.stage = "local"
        r.tag = r.tag.replace("fast__", "local__")
    return records


def run_optuna_search(args, extra_idx, arrays, names, extra_tags, alpha_grid):
    if args.optuna_trials <= 0:
        return []
    try:
        import optuna
    except Exception as e:
        print(f"[WARN] optuna unavailable: {e}")
        return []

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n = len(names)
    max_k = min(args.max_subset_size, n)
    min_k = min(args.min_subset_size, max_k)
    records = []
    deadline = parse_deadline(args.deadline)

    def objective(trial):
        pool_n = trial.suggest_int("pool_n", min(n, max(6, min_k)), n)
        k = trial.suggest_int("k", min_k, min(max_k, pool_n))
        cols = set()
        for j in range(k):
            cols.add(trial.suggest_int(f"idx_{j}", 0, pool_n - 1))
        while len(cols) < min_k:
            cols.add(len(cols))
        cols = tuple(sorted(cols))
        method = trial.suggest_categorical("method", ["ridge", "nnls", "mean"])
        alpha = trial.suggest_float("alpha", 1e-9, 1e-2, log=True)
        iso_weight = trial.suggest_float("iso_weight", 0.65, 1.0)
        zero_tau = trial.suggest_float("zero_tau", 0.0, 0.0025)
        rec, _, _ = eval_fast_arrays(
            cols, extra_idx, arrays, names, extra_tags,
            method=method,
            alpha=alpha,
            iso_weight=iso_weight,
            zero_tau=zero_tau,
            stage="optuna",
        )
        rec.params["trial"] = trial.number
        records.append(rec)
        return score_rec(rec, args.select_by)

    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True, group=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    timeout = None
    if deadline is not None:
        timeout = max(0.0, seconds_left(deadline) - args.deadline_margin_minutes * 60.0)
        if timeout <= 0:
            print("[Optuna] skipped: deadline margin reached")
            return records
    study.optimize(objective, n_trials=args.optuna_trials, timeout=timeout, show_progress_bar=False)
    best_val_seen = min((r.val_rmse for r in records), default=float("nan"))
    print(f"[Optuna] best_{args.select_by}={study.best_value:.9f} "
          f"val_monitor_best={best_val_seen:.9f} params={study.best_params}")
    return records


def prediction_for_record(rec, extra_idx, arrays, names, extra_tags, args):
    base_cols = tuple(sorted(names.index(x) for x in rec.pool_names))
    if rec.method.startswith("ridge") or rec.method.startswith("nnls") or rec.method.startswith("mean"):
        base_method = rec.method.split("+", 1)[0]
        _, pv, pt = eval_fast_arrays(
            base_cols, extra_idx, arrays, names, extra_tags,
            method=base_method,
            alpha=float(rec.params.get("alpha", 1e-5)),
            iso_weight=float(rec.params.get("iso_weight", 1.0)),
            zero_tau=float(rec.params.get("zero_tau", 0.0)),
            stage=rec.stage,
        )
        return pv, pt
    if rec.method.startswith("ENetPositive"):
        _, pv, pt = eval_enet(base_cols, extra_idx, arrays, names, extra_tags,
                              seed=args.seed, positive=True, stage="cache")
        return pv, pt
    if rec.method.startswith("ENet"):
        _, pv, pt = eval_enet(base_cols, extra_idx, arrays, names, extra_tags,
                              seed=args.seed, positive=False, stage="cache")
        return pv, pt
    if rec.method.startswith("Combo"):
        _, pv, pt = eval_combo(base_cols, extra_idx, arrays, names, extra_tags,
                               seed=args.seed, stage="cache")
        return pv, pt
    return None


def save_outputs(records, pred_cache, keys_val, keys_test, args, models, extra_tags, extra_col_count):
    ts = datetime.now().strftime("%m%d_%H%M%S")
    run_dir = RESULT_DIR / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    records = sorted(records, key=lambda r: score_rec(r, args.select_by))
    slim = []
    for r in records:
        d = asdict(r)
        d["objective"] = args.select_by
        d["objective_score"] = score_rec(r, args.select_by)
        slim.append(d)
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "extra_tags": extra_tags,
            "extra_col_count": extra_col_count,
            "n_records": len(records),
            "models": [{k: str(v) for k, v in m.items() if k != "path"} for m in models],
            "records": slim,
        }, f, ensure_ascii=False, indent=2)
    pd.DataFrame(slim).drop(columns=["params", "pool_names", "extra_tags"], errors="ignore").to_csv(
        run_dir / "summary.csv", index=False
    )

    best = records[0]
    pv, pt = pred_cache[id(best)]
    pd.DataFrame({KEY_COL: keys_val, "pred": pv}).to_csv(run_dir / "best_val.csv", index=False)
    pd.DataFrame({KEY_COL: keys_test, "pred": pt}).to_csv(run_dir / "best_test.csv", index=False)

    print("\n" + "=" * 92)
    print(f"FINAL TOP 20 BY {args.select_by.upper()} OBJECTIVE  (extra cols={extra_col_count})")
    print("=" * 92)
    for i, r in enumerate(records[:20], 1):
        risk = "  [VAL-FIT]" if r.stage == "VAL_FIT" else ""
        mcv = f"mcv={r.meta_cv_oof_rmse:.9f} " if not math.isnan(r.meta_cv_oof_rmse) else ""
        print(f"{i:2d}. obj={score_rec(r, args.select_by):.9f} "
              f"oof={r.oof_rmse:.9f} {mcv}val={r.val_rmse:.9f} test={r.test_rmse:.9f} "
              f"k={r.n_base:2d}+x{r.n_extra} {r.tag}{risk}")
        print("    " + ", ".join(r.pool_names[:8]) + (" ..." if len(r.pool_names) > 8 else ""))

    mcv_records = [r for r in records if not math.isnan(r.meta_cv_oof_rmse)]
    if mcv_records:
        print("\nMETA-CV OOF TOP 10 (refit only)")
        for i, r in enumerate(sorted(mcv_records, key=lambda x: x.meta_cv_oof_rmse)[:10], 1):
            print(f"{i:2d}. mcv={r.meta_cv_oof_rmse:.9f} oof={r.oof_rmse:.9f} "
                  f"val={r.val_rmse:.9f} test={r.test_rmse:.9f} k={r.n_base:2d}+x{r.n_extra} {r.tag}")
    print("\nVAL MONITOR TOP 10 (reported only)")
    for i, r in enumerate(sorted(records, key=lambda x: x.val_rmse)[:10], 1):
        mcv = f"mcv={r.meta_cv_oof_rmse:.9f} " if not math.isnan(r.meta_cv_oof_rmse) else ""
        print(f"{i:2d}. val={r.val_rmse:.9f} oof={r.oof_rmse:.9f} "
              f"{mcv}test={r.test_rmse:.9f} k={r.n_base:2d}+x{r.n_extra} {r.tag}")
    print(f"\nSaved: {run_dir}")
    return run_dir


def main():
    args = parse_args()
    deadline = parse_deadline(args.deadline)
    rng = np.random.default_rng(args.seed)
    alpha_grid = np.logspace(-9, -2, 8)

    print("=" * 92)
    print("squeeze_extreme_v2.py  (base preds + optional extra cols, always included)")
    print(f"PROJECT_ROOT={PROJECT_ROOT}")
    print(f"two-stage/default/reg included? {args.include_ts_reg}")
    print(f"selection objective={args.select_by}  (val/test/oof/meta_cv_oof 모두 매 record 로그)")
    if deadline is not None:
        print(f"deadline={deadline.strftime('%Y-%m-%d %H:%M:%S')} local "
              f"(reserve {args.deadline_margin_minutes:.1f} min for saving)")
    print("=" * 92)

    models = discover_models(args)
    if not models:
        raise RuntimeError("No model outputs passed the filters.")
    names = [m["name"] for m in models]
    print(f"\nFiltered base models: {len(models)}")
    for i, m in enumerate(models):
        print(f"{i:2d}  {m['oof_rmse']:.6f}  {m['category']:10s} {m['variant']:7s} {m['rel']}")

    X_oof_base, y_oof, keys_oof = build_base_matrix(models, "oof")
    X_val_base, y_val, keys_val = build_base_matrix(models, "val")
    X_test_base, y_test, keys_test = build_base_matrix(models, "test")

    print("\n[extra cache] loading ...")
    X_oof_extra, X_val_extra, X_test_extra, extra_names, extra_tags, cache_col_counts = load_extra_caches(
        args.extra_cache, keys_oof, keys_val, keys_test, args.extra_prefix_with_tag,
        top_k=args.extra_top_k,
    )
    n_base = X_oof_base.shape[1]
    n_extra = X_oof_extra.shape[1] if X_oof_extra is not None else 0

    if n_extra > 0:
        X_oof  = np.hstack([X_oof_base,  X_oof_extra])
        X_val  = np.hstack([X_val_base,  X_val_extra])
        X_test = np.hstack([X_test_base, X_test_extra])
        if args.extra_mode == "searchable":
            # extra cols을 base와 동등하게 subset search 후보로 둠 → 메타가 알아서 선택
            names = names + extra_names                # 통합 풀
            extra_idx = ()                              # 항상 포함 안 함
            print(f"  [extra-mode] searchable: names 통합 ({len(names)} cols, base {n_base} + extra {n_extra})")
        else:
            extra_idx = tuple(range(n_base, n_base + n_extra))
            print(f"  [extra-mode] always_include: extra cols {n_extra}개를 메타 학습 시 항상 포함")
    else:
        X_oof, X_val, X_test = X_oof_base, X_val_base, X_test_base
        extra_idx = ()
    n_pool = len(names)   # subset search 후보 풀 크기 (mode에 따라 base만 / base+extra)

    arrays = (X_oof, y_oof, X_val, y_val, X_test, y_test)
    print(f"\nShapes: oof={X_oof.shape} (base={n_base}+extra={n_extra}), "
          f"val={X_val.shape}, test={X_test.shape}")
    print(f"y_val range: [{y_val.min():.6f}, {y_val.max():.6f}]")

    records: list[Record] = []
    pred_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    # 1) Seed pools.
    seed_subsets = make_seed_subsets(models, names, args)
    print(f"\n[seed subsets] {len(seed_subsets)}")
    for cols in seed_subsets:
        rec = best_fast_for_subset(cols, extra_idx, arrays, names, extra_tags, alpha_grid, args.select_by)
        if rec is not None:
            rec.stage = "seed"
            rec.tag = rec.tag.replace("fast__", "seed__")
            records.append(rec)

    # 2) Random broad search.
    # n_pool = n_base (always_include 모드) 또는 n_base+n_extra (searchable 모드)
    random_subsets = sample_random_subsets(n_pool, args, rng)
    print(f"\n[random search] {len(random_subsets)} subsets")
    t0 = time.time()
    seen_subsets = {tuple(names.index(x) for x in r.pool_names) for r in records}
    for i, cols in enumerate(random_subsets, 1):
        if should_stop(deadline, args.deadline_margin_minutes):
            print(f"  [deadline] random search stopped at {i-1}/{len(random_subsets)}")
            break
        if cols in seen_subsets:
            continue
        rec = best_fast_for_subset(cols, extra_idx, arrays, names, extra_tags, alpha_grid, args.select_by)
        if rec is not None:
            rec.stage = "random"
            rec.tag = rec.tag.replace("fast__", "random__")
            records.append(rec)
            seen_subsets.add(cols)
        if i % 500 == 0:
            best = min(score_rec(r, args.select_by) for r in records)
            best_val = min(r.val_rmse for r in records)
            print(f"  {i:5d}/{len(random_subsets)}  elapsed={time.time()-t0:7.1f}s  "
                  f"best_{args.select_by}={best:.9f}  val_monitor_best={best_val:.9f}")

    # 3) Local add/drop/swap around best random subsets.
    print(f"\n[local search] seeds={args.local_seeds}")
    top_for_local = sorted(records, key=lambda r: score_rec(r, args.select_by))[:args.local_seeds]
    # candidate_pool: always_include = base 일부만 (기존), searchable = base 일부 + cache별 균등
    if args.extra_mode == "searchable" and n_extra > 0:
        base_limit = min(args.local_candidate_limit, n_base)
        # cache별로 처음 K_per_cache개씩 (build_shap_features가 importance 내림차순 저장 → top-K importance)
        per_cache_k = max(1, args.local_candidate_limit // max(1, len(cache_col_counts)))
        extra_indices: list[int] = []
        offset = n_base
        for cnt in cache_col_counts:
            take = min(per_cache_k, cnt)
            extra_indices.extend(range(offset, offset + take))
            offset += cnt
        candidate_pool = list(range(base_limit)) + extra_indices
    else:
        candidate_pool = list(range(min(args.local_candidate_limit, n_base)))
    print(f"  [candidate_pool] size={len(candidate_pool)} (extra-mode={args.extra_mode})")
    for i, rec in enumerate(top_for_local, 1):
        if should_stop(deadline, args.deadline_margin_minutes):
            print(f"  [deadline] local search stopped at {i-1}/{len(top_for_local)}")
            break
        cols = tuple(sorted(names.index(x) for x in rec.pool_names))
        local_recs = local_improve(cols, extra_idx, arrays, names, extra_tags, args, alpha_grid, candidate_pool)
        records.extend(local_recs)
        if local_recs:
            print(f"  local {i:2d}: best_{args.select_by}={min(score_rec(r, args.select_by) for r in local_recs):.9f} "
                  f"val_monitor={min(r.val_rmse for r in local_recs):.9f} k={local_recs[-1].n_base}")

    # 4) Optional Optuna.
    optuna_records = run_optuna_search(args, extra_idx, arrays, names, extra_tags, alpha_grid)
    records.extend(optuna_records)

    # 5) Expensive refit on top unique subsets.
    print(f"\n[refit] top_refit={args.top_refit}, combo_refit={args.combo_refit}")
    unique_cols = []
    seen = set()
    for rec in sorted(records, key=lambda r: score_rec(r, args.select_by)):
        cols = tuple(sorted(names.index(x) for x in rec.pool_names))
        if cols in seen:
            continue
        seen.add(cols)
        unique_cols.append(cols)
        if len(unique_cols) >= args.top_refit:
            break

    for i, cols in enumerate(unique_cols, 1):
        if should_stop(deadline, args.deadline_margin_minutes):
            print(f"  [deadline] refit stopped at {i-1}/{len(unique_cols)}")
            break
        rec, pv, pt = eval_enet(cols, extra_idx, arrays, names, extra_tags,
                                seed=args.seed, positive=False, stage="refit")
        records.append(rec)
        pred_cache[id(rec)] = (pv, pt)

        rec_pos, pv_pos, pt_pos = eval_enet(cols, extra_idx, arrays, names, extra_tags,
                                            seed=args.seed, positive=True, stage="refit")
        records.append(rec_pos)
        pred_cache[id(rec_pos)] = (pv_pos, pt_pos)

        if i <= args.combo_refit:
            rec_combo, pv_combo, pt_combo = eval_combo(cols, extra_idx, arrays, names, extra_tags,
                                                       seed=args.seed, stage="refit")
            records.append(rec_combo)
            pred_cache[id(rec_combo)] = (pv_combo, pt_combo)

        # meta_cv_oof는 base + extra 모두 포함해서 계산 (= 실제 메타 입력과 동일)
        use_cols_for_mcv = _combine_cols(cols, extra_idx)
        mc_oof = compute_meta_cv_oof(use_cols_for_mcv, arrays, seed=args.seed)
        for r in records[-3 if i <= args.combo_refit else -2:]:
            r.meta_cv_oof_rmse = mc_oof

        print(f"  refit {i:3d}/{len(unique_cols)}  "
              f"best_{args.select_by}={min(score_rec(r, args.select_by) for r in records):.9f}  "
              f"meta_cv_oof={mc_oof:.9f}  val_monitor_best={min(r.val_rmse for r in records):.9f}")

    # Cache predictions for current best records too (val-fit postprocess 후보).
    for rec in sorted(records, key=lambda r: score_rec(r, args.select_by))[:50]:
        if id(rec) in pred_cache:
            continue
        try:
            pred = prediction_for_record(rec, extra_idx, arrays, names, extra_tags, args)
            if pred is not None:
                pred_cache[id(rec)] = pred
        except Exception:
            pass

    # 6) Optional val-label direct tuning — v2 에서는 일단 미지원 (필요시 v1 함수 포팅)
    if args.allow_val_fit:
        print("[WARN] --allow_val_fit는 v2에서 아직 미구현, 무시됨")

    # Ensure best has cached val/test predictions.
    for rec in sorted(records, key=lambda r: score_rec(r, args.select_by)):
        if id(rec) in pred_cache:
            break
        pred = prediction_for_record(rec, extra_idx, arrays, names, extra_tags, args)
        if pred is None:
            raise RuntimeError(f"Could not rebuild predictions for best record: {rec.tag}")
        pred_cache[id(rec)] = pred
        break

    save_outputs(records, pred_cache, keys_val, keys_test, args, models, extra_tags, n_extra)


if __name__ == "__main__":
    main()
