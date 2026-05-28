"""탐색 단계 — seed pool / random search / local improve / Optuna.

v2의 search 로직을 die-level + unit aggregate 기반으로 재구성.

핵심 분기:
- fast eval (탐색 중): mean+iso → 단순 unit mean 집계 → unit RMSE. 1 trial 수십 ms.
- refit eval (상위 K개): mean/ridge/nnls/ENet/ENetPositive/Combo + iso → **postprocess.tune_and_apply**로 unit 집계.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import aggregate, meta
from .config import SqueezeV4Config, parse_deadline, seconds_left, should_stop
from .records import Record, score_rec


# ---------------------------------------------------------------------------
# eval 1건 — fast (단순 mean unit 집계)
# ---------------------------------------------------------------------------
@dataclass
class ArrayBundle:
    """split별 die-level 행렬 + key + unit y. eval 함수에 통째로 넘긴다.

    v4의 SHAP 통합:
    - `extra_idx`가 비어 있지 않으면 (always_include 모드) — search/refit이 base_cols와
      extra_idx의 합집합을 메타 입력으로 사용. extra_idx는 subset search 후보에 들어가지 않음.
    - `extra_idx`가 비어 있으면 (searchable 모드 또는 SHAP 비활성) — names 전체가 후보 풀.
    - `extra_names` / `extra_tags`는 박제/diagnostics용.
    """
    X_oof:  np.ndarray   # (N_die_oof, K_total) — base + (always_include 모드면 SHAP)
    X_val:  np.ndarray
    X_test: np.ndarray
    y_die_oof:  np.ndarray
    y_die_val:  np.ndarray
    y_die_test: np.ndarray

    key_oof:  pd.DataFrame
    key_val:  pd.DataFrame
    key_test: pd.DataFrame

    y_unit_oof:  pd.DataFrame   # [ufs_serial, health]
    y_unit_val:  pd.DataFrame
    y_unit_test: pd.DataFrame

    # fast aggregator + unique_units (외부에서 1번 만들어 넣음)
    agg_oof:  Callable
    agg_val:  Callable
    agg_test: Callable
    units_oof:  np.ndarray
    units_val:  np.ndarray
    units_test: np.ndarray
    y_unit_oof_arr:  np.ndarray
    y_unit_val_arr:  np.ndarray
    y_unit_test_arr: np.ndarray

    names: list[str]

    # ─── v4 SHAP 통합 메타 (default 빈 값 → v3와 동일 동작 호환) ──────
    extra_idx: tuple[int, ...] = ()
    """always_include 모드일 때 메타 학습에 항상 포함될 컬럼 인덱스(SHAP).
    subset search 후보에는 안 들어감. 빈 튜플이면 SHAP 비활성 또는 searchable 모드."""

    extra_names: list[str] = None       # type: ignore[assignment]
    """SHAP 컬럼명 (always_include는 X에 있지만 names에 안 들어갈 수도 있어서 별도 박제).
    searchable 모드일 때는 names에 포함되어 있고 이 필드는 같은 컬럼명 또는 None.
    """

    extra_tags: list[str] = None        # type: ignore[assignment]
    """SHAP 캐시 폴더명 목록 (record.extra_tags로 박제)."""

    shap_mode: str = "none"
    """'none' | 'always_include' | 'searchable'."""


def _use_cols(base_cols: tuple[int, ...], bundle: ArrayBundle) -> list[int]:
    """base_cols ∪ extra_idx — 메타 학습에 실제로 들어갈 컬럼 인덱스 리스트.

    always_include 모드: base_cols 뒤에 extra_idx를 (중복 제거) append.
    searchable / none:   그냥 base_cols 그대로.
    추가 순서를 base_cols 뒤로 고정 → record.pool_names 내 base 컬럼 순서는 변형 안 됨.
    """
    if not bundle.extra_idx:
        return list(base_cols)
    seen = set(base_cols)
    cols = list(base_cols)
    for j in bundle.extra_idx:
        if j not in seen:
            cols.append(j)
            seen.add(j)
    return cols


def _fast_unit_rmse(
    die_pred: np.ndarray,
    aggregator: Callable,
    y_unit_arr: np.ndarray,
) -> float:
    """die pred → unit mean → RMSE. 수 ms."""
    unit_pred, _ = aggregator(die_pred)
    return meta.rmse(unit_pred, y_unit_arr)


def _count_extra_in_subset(base_cols: tuple[int, ...], bundle: ArrayBundle) -> int:
    """searchable 모드에서 base_cols 안의 SHAP 컬럼 개수를 센다 (record.n_extra 박제용).

    always_include 모드는 base_cols와 extra_idx가 분리되어 있으므로 별도로 len(extra_idx)를 박제.
    """
    if bundle.shap_mode == "searchable" and bundle.extra_names:
        extra_name_set = set(bundle.extra_names)
        return sum(1 for c in base_cols if bundle.names[c] in extra_name_set)
    return 0


def _extra_tags_for_record(bundle: ArrayBundle) -> list[str]:
    return list(bundle.extra_tags) if bundle.extra_tags else []


def eval_fast(
    base_cols: tuple[int, ...],
    bundle: ArrayBundle,
    method: str = "ridge",
    alpha: float = 1e-5,
    iso_weight: float = 1.0,
    zero_tau: float = 0.0,
    stage: str = "fast",
    use_iso: bool = True,
) -> Record:
    """단일 subset의 fast eval.

    die-level meta fit → (use_iso이면) iso → fast unit mean 집계 → Record. aggregation은 'mean' 고정.
    use_iso=False면 raw 메타 예측을 그대로 사용 (non-negative clip만 적용).
    ENet 계열은 내부 CV가 GroupKFold(unit)로 동작하도록 groups를 자동 주입한다.

    v4 SHAP: bundle.extra_idx가 비어 있지 않으면 base_cols와 합쳐서 메타 입력 컬럼 구성
    (always_include 모드). pool_names는 base_cols 기준만 박제 (subset search 추적용).
    """
    use_cols = _use_cols(base_cols, bundle)
    Xo = bundle.X_oof[:, use_cols]
    Xv = bundle.X_val[:, use_cols]
    Xt = bundle.X_test[:, use_cols]

    # OOF die의 unit ID — ENet 내부 CV의 GroupKFold용 (다른 method는 무시)
    groups_oof = bundle.key_oof[aggregate.KEY_COL].values
    ro, rv, rt = meta.fit_meta_raw(method, Xo, bundle.y_die_oof, Xv, Xt, alpha=alpha, groups=groups_oof)
    if use_iso:
        po, pv, pt = meta.apply_iso(ro, rv, rt, bundle.y_die_oof,
                                    iso_weight=iso_weight, zero_tau=zero_tau)
    else:
        po, pv, pt = meta.apply_no_iso(ro, rv, rt, zero_tau=zero_tau)
    # die-level RMSE
    rmse_die_o = meta.rmse(po, bundle.y_die_oof)
    rmse_die_v = meta.rmse(pv, bundle.y_die_val)
    rmse_die_t = meta.rmse(pt, bundle.y_die_test)
    # unit-level RMSE (단순 mean 집계)
    rmse_unit_o = _fast_unit_rmse(po, bundle.agg_oof,  bundle.y_unit_oof_arr)
    rmse_unit_v = _fast_unit_rmse(pv, bundle.agg_val,  bundle.y_unit_val_arr)
    rmse_unit_t = _fast_unit_rmse(pt, bundle.agg_test, bundle.y_unit_test_arr)

    # n_extra 박제: always_include면 len(extra_idx), searchable이면 base_cols 안의 SHAP 개수.
    if bundle.shap_mode == "always_include":
        n_extra = len(bundle.extra_idx)
    elif bundle.shap_mode == "searchable":
        n_extra = _count_extra_in_subset(base_cols, bundle)
    else:
        n_extra = 0

    method_label = f"{method}+Iso" if use_iso else method
    return Record(
        tag=f"{stage}__{method}__k{len(base_cols)}",
        stage=stage,
        method=method_label,
        n_base=len(base_cols),
        val_rmse=rmse_unit_v,
        test_rmse=rmse_unit_t,
        oof_rmse=rmse_unit_o,
        val_rmse_die=rmse_die_v,
        test_rmse_die=rmse_die_t,
        oof_rmse_die=rmse_die_o,
        pool_names=[bundle.names[i] for i in base_cols],
        params={"alpha": alpha, "iso_weight": iso_weight, "zero_tau": zero_tau, "use_iso": use_iso},
        aggregation="mean",   # fast 단계는 mean 고정
        n_extra=n_extra,
        extra_tags=_extra_tags_for_record(bundle),
    )


def _fast_score(rec: Record, cfg: SqueezeV4Config) -> float:
    """fast 탐색 단계 비교용 점수 — 항상 oof_rmse(+ val_gap_penalty) 기준.

    cfg.select_by="meta_cv_oof"는 refit 단계에서만 채워지는 값을 보므로,
    fast/seed/local/optuna 단계에서는 의미가 없다 (모두 NaN → inf 반환 → 정렬 무의미).
    따라서 fast 단계 내부 비교는 항상 oof_rmse 기준으로 한다.
    """
    return score_rec(rec, "oof", cfg.val_gap_penalty)


def best_fast_for_subset(
    base_cols: tuple[int, ...],
    bundle: ArrayBundle,
    alpha_grid: tuple[float, ...],
    cfg: SqueezeV4Config,
) -> Optional[Record]:
    """한 subset에 대해 mean/nnls/ridge(grid) 다 돌려서 score 최저 record 반환."""
    best: Optional[Record] = None
    for method in ("mean", "nnls"):
        try:
            rec = eval_fast(base_cols, bundle, method=method, stage="fast", use_iso=cfg.use_iso)
            if best is None or _fast_score(rec, cfg) < _fast_score(best, cfg):
                best = rec
        except Exception:
            pass
    for alpha in alpha_grid:
        try:
            rec = eval_fast(base_cols, bundle, method="ridge", alpha=alpha, stage="fast", use_iso=cfg.use_iso)
            if best is None or _fast_score(rec, cfg) < _fast_score(best, cfg):
                best = rec
        except Exception:
            pass
    return best


# ---------------------------------------------------------------------------
# subset 후보 생성기
# ---------------------------------------------------------------------------
def make_seed_subsets(
    models: list[dict],
    names: list[str],
    cfg: SqueezeV4Config,
) -> list[tuple[int, ...]]:
    """카테고리/variant별 + Top-K + KNOWN_STRONG_SUBSET을 모아 seed로 둠."""
    n = len(names)
    subsets: set[tuple[int, ...]] = set()
    by_cat: dict[str, list[int]] = {}
    by_variant: dict[str, list[int]] = {}
    for i, m in enumerate(models):
        by_cat.setdefault(m["category"], []).append(i)
        by_variant.setdefault(m["variant"], []).append(i)

    # Top-K (정렬 상위 k개)
    for k in (2, 3, 4, 5, 8, 10, 15, 20, n):
        if cfg.min_subset_size <= k <= min(cfg.max_subset_size, n):
            subsets.add(tuple(range(k)))

    # 카테고리별 / variant별
    for xs in by_cat.values():
        if len(xs) >= cfg.min_subset_size:
            subsets.add(tuple(sorted(xs[:cfg.max_subset_size])))
    for xs in by_variant.values():
        if len(xs) >= cfg.min_subset_size:
            subsets.add(tuple(sorted(xs[:cfg.max_subset_size])))

    # KNOWN_STRONG_SUBSET (4개 모두 있을 때만)
    if cfg.known_strong_subset:
        strong_in_names = [names.index(x) for x in cfg.known_strong_subset if x in names]
        if len(strong_in_names) == len(cfg.known_strong_subset):
            subsets.add(tuple(sorted(strong_in_names)))

    return sorted(subsets, key=lambda x: (len(x), x))


def sample_random_subsets(
    n_base: int,
    cfg: SqueezeV4Config,
    rng: np.random.Generator,
) -> list[tuple[int, ...]]:
    """rank-biased sampling — oof_rmse 정렬 상위에 더 높은 가중치를 줘서 k개 뽑음.

    뽑은 subset이 unique해질 때까지 반복.
    """
    max_k = min(cfg.max_subset_size, n_base)
    min_k = min(cfg.min_subset_size, max_k)
    possible = sum(math.comb(n_base, k) for k in range(min_k, max_k + 1))
    target = min(cfg.random_trials, possible)
    rank = np.arange(n_base, dtype=float)
    prob = 1.0 / np.power(rank + 1.0, 0.65)
    prob = prob / prob.sum()
    seen: set[tuple[int, ...]] = set()
    while len(seen) < target:
        k = int(rng.integers(min_k, max_k + 1))
        cols = tuple(sorted(rng.choice(np.arange(n_base), size=k, replace=False, p=prob).tolist()))
        seen.add(cols)
    return list(seen)


# ---------------------------------------------------------------------------
# local improve (add/drop/swap)
# ---------------------------------------------------------------------------
def local_improve(
    seed_cols: tuple[int, ...],
    bundle: ArrayBundle,
    cfg: SqueezeV4Config,
    alpha_grid: tuple[float, ...],
    candidate_pool: list[int],
) -> list[Record]:
    """seed subset 주변을 add/drop/swap으로 탐색. score가 더 안 좋아질 때까지 진행."""
    current = tuple(sorted(seed_cols))
    best = best_fast_for_subset(current, bundle, alpha_grid, cfg)
    if best is None:
        return []
    records = [best]
    all_idx = list(candidate_pool)
    for _ in range(cfg.local_steps):
        proposals: set[tuple[int, ...]] = set()
        # add (한 개 추가)
        if len(current) < cfg.max_subset_size:
            for j in all_idx:
                if j not in current:
                    proposals.add(tuple(sorted(current + (j,))))
        # drop (한 개 제거)
        if len(current) > cfg.min_subset_size:
            for j in current:
                proposals.add(tuple(x for x in current if x != j))
        # swap (1 drop + 1 add)
        if len(current) > cfg.min_subset_size:
            for drop in current:
                base = tuple(x for x in current if x != drop)
                for add in all_idx:
                    if add not in current:
                        proposals.add(tuple(sorted(base + (add,))))

        step_best = best
        for cols in proposals:
            rec = best_fast_for_subset(cols, bundle, alpha_grid, cfg)
            # fast 단계라 _fast_score(oof 기준) 사용 — select_by="meta_cv_oof"여도 정상 동작
            if rec is not None and _fast_score(rec, cfg) < _fast_score(step_best, cfg) - 1e-10:
                step_best = rec
        if step_best is best:
            break
        current = tuple(bundle.names.index(x) for x in step_best.pool_names)
        best = step_best
        records.append(best)

    # tag/stage 정리
    for r in records:
        r.stage = "local"
        r.tag = r.tag.replace("fast__", "local__")
    return records


# ---------------------------------------------------------------------------
# Optuna search (옵션)
# ---------------------------------------------------------------------------
def run_optuna_search(
    bundle: ArrayBundle,
    cfg: SqueezeV4Config,
    alpha_grid: tuple[float, ...],
    deadline: Optional[datetime] = None,
) -> list[Record]:
    if cfg.optuna_trials <= 0:
        return []
    try:
        import optuna
    except ImportError as e:
        if cfg.verbose:
            print(f"[WARN] optuna unavailable: {e}")
        return []

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n = len(bundle.names)
    max_k = min(cfg.max_subset_size, n)
    min_k = min(cfg.min_subset_size, max_k)
    records: list[Record] = []

    def objective(trial):
        pool_n = trial.suggest_int("pool_n", min(n, max(6, min_k)), n)
        k = trial.suggest_int("k", min_k, min(max_k, pool_n))
        cols: set[int] = set()
        for j in range(k):
            cols.add(trial.suggest_int(f"idx_{j}", 0, pool_n - 1))
        while len(cols) < min_k:
            cols.add(len(cols))
        cols_t = tuple(sorted(cols))
        method = trial.suggest_categorical("method", ["ridge", "nnls", "mean"])
        alpha = trial.suggest_float("alpha", 1e-9, 1e-2, log=True)
        iso_weight = trial.suggest_float("iso_weight", 0.65, 1.0)
        zero_tau = trial.suggest_float("zero_tau", 0.0, 0.0025)
        rec = eval_fast(
            cols_t, bundle,
            method=method, alpha=alpha,
            iso_weight=iso_weight, zero_tau=zero_tau,
            stage="optuna",
            use_iso=cfg.use_iso,
        )
        rec.params["trial"] = trial.number
        records.append(rec)
        # fast(optuna) 단계라 oof 기준 비교 — meta_cv_oof는 refit 후에만 채워짐
        return _fast_score(rec, cfg)

    sampler = optuna.samplers.TPESampler(seed=cfg.seed, multivariate=True, group=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    timeout: Optional[float] = None
    if deadline is not None:
        timeout = max(0.0, seconds_left(deadline) - cfg.deadline_margin_minutes * 60.0)
        if timeout <= 0:
            if cfg.verbose:
                print("[Optuna] skipped: deadline margin reached")
            return records
    study.optimize(objective, n_trials=cfg.optuna_trials, timeout=timeout, show_progress_bar=False)
    if cfg.verbose:
        best_val_seen = min((r.val_rmse for r in records), default=float("nan"))
        # optuna는 fast 단계라 oof 기준 (select_by와 무관하게 study.best_value는 _fast_score)
        print(f"[Optuna] best_oof={study.best_value:.9f} "
              f"val_monitor_best={best_val_seen:.9f} params={study.best_params}")
    return records


# ---------------------------------------------------------------------------
# 통합 wrapper — main 노트북에서 한 번에 호출 가능
# ---------------------------------------------------------------------------
def run_search_stages(
    models: list[dict],
    bundle: ArrayBundle,
    cfg: SqueezeV4Config,
) -> tuple[list[Record], np.random.Generator]:
    """seed → random → local → optuna 4단계 fast 탐색 통합.

    refit/save는 별도 함수.
    Returns (records, rng).
    """
    rng = np.random.default_rng(cfg.seed)
    deadline = parse_deadline(cfg.deadline)
    alpha_grid = tuple(cfg.ridge_alpha_grid)
    records: list[Record] = []
    # n_pool = subset search 후보 풀 크기 = len(names).
    # always_include 모드: names = base만 → 후보 풀 = base만.
    # searchable 모드:    names = base + SHAP → 후보 풀 = 통합.
    # bundle.X_oof.shape[1]은 (base + extra_idx)일 수 있어 항상 names 기준으로 한다.
    n_pool = len(bundle.names)
    names = bundle.names

    # 1) seed pools
    seed_subsets = make_seed_subsets(models, names, cfg)
    if cfg.verbose:
        print(f"\n[seed subsets] {len(seed_subsets)}")
    for cols in seed_subsets:
        rec = best_fast_for_subset(cols, bundle, alpha_grid, cfg)
        if rec is not None:
            rec.stage = "seed"
            rec.tag = rec.tag.replace("fast__", "seed__")
            records.append(rec)

    # 2) random broad
    if cfg.random_trials > 0:
        random_subsets = sample_random_subsets(n_pool, cfg, rng)
        if cfg.verbose:
            print(f"\n[random search] {len(random_subsets)} subsets")
        t0 = time.time()
        seen_subsets = {tuple(sorted(names.index(x) for x in r.pool_names)) for r in records}
        for i, cols in enumerate(random_subsets, 1):
            if should_stop(deadline, cfg.deadline_margin_minutes):
                if cfg.verbose:
                    print(f"  [deadline] random stopped at {i - 1}/{len(random_subsets)}")
                break
            if cols in seen_subsets:
                continue
            rec = best_fast_for_subset(cols, bundle, alpha_grid, cfg)
            if rec is not None:
                rec.stage = "random"
                rec.tag = rec.tag.replace("fast__", "random__")
                records.append(rec)
                seen_subsets.add(cols)
            if cfg.verbose and i % cfg.log_every == 0:
                # fast 단계 progress log — oof 기준 (select_by와 무관)
                best = min(_fast_score(r, cfg) for r in records)
                best_val = min(r.val_rmse for r in records)
                print(f"  {i:5d}/{len(random_subsets)}  t={time.time() - t0:7.1f}s  "
                      f"best_oof={best:.9f}  val_best={best_val:.9f}")

    # 3) local improve
    if cfg.local_seeds > 0 and cfg.local_steps > 0:
        if cfg.verbose:
            print(f"\n[local search] seeds={cfg.local_seeds}")
        # local 출발 seed도 fast 단계 점수(oof) 기준 정렬
        top_for_local = sorted(records, key=lambda r: _fast_score(r, cfg))[:cfg.local_seeds]
        candidate_pool = list(range(min(cfg.local_candidate_limit, n_pool)))
        if cfg.verbose:
            print(f"  [candidate_pool] size={len(candidate_pool)}")
        for i, rec in enumerate(top_for_local, 1):
            if should_stop(deadline, cfg.deadline_margin_minutes):
                if cfg.verbose:
                    print(f"  [deadline] local stopped at {i - 1}/{len(top_for_local)}")
                break
            cols = tuple(sorted(names.index(x) for x in rec.pool_names))
            local_recs = local_improve(cols, bundle, cfg, alpha_grid, candidate_pool)
            records.extend(local_recs)
            if cfg.verbose and local_recs:
                # local 단계도 oof 기준 진행 로그 (select_by와 무관)
                print(f"  local {i:2d}: best_oof="
                      f"{min(_fast_score(r, cfg) for r in local_recs):.9f} "
                      f"val={min(r.val_rmse for r in local_recs):.9f} k={local_recs[-1].n_base}")

    # 4) Optuna (옵션)
    records.extend(run_optuna_search(bundle, cfg, alpha_grid, deadline))

    return records, rng


# ---------------------------------------------------------------------------
# Refit — 상위 K개 subset에 ENet/ENetPositive/Combo + tune_and_apply
# ---------------------------------------------------------------------------
@dataclass
class RefitArtifact:
    """refit 결과 1건의 raw die-level pred + 채택된 집계 결과 보관.

    save_outputs에서 final unit pred CSV를 만들 때 이걸 본다.
    """
    record: Record
    raw_die_oof:  np.ndarray
    raw_die_val:  np.ndarray
    raw_die_test: np.ndarray
    unit_oof_df:  pd.DataFrame   # [ufs_serial, pred]
    unit_val_df:  pd.DataFrame
    unit_test_df: pd.DataFrame


def _make_record_from_refit(
    method_label: str,
    base_cols: tuple[int, ...],
    bundle: ArrayBundle,
    raw_oof: np.ndarray, raw_val: np.ndarray, raw_test: np.ndarray,
    cfg: SqueezeV4Config,
    extra_params: dict | None = None,
) -> RefitArtifact:
    """raw die-level pred 3종 → (use_iso이면) iso → tune_and_apply → Record + artifact 묶음 생성."""
    if cfg.use_iso:
        po, pv, pt = meta.apply_iso(raw_oof, raw_val, raw_test, bundle.y_die_oof)
    else:
        po, pv, pt = meta.apply_no_iso(raw_oof, raw_val, raw_test)

    # die-level RMSE
    rmse_die_o = meta.rmse(po, bundle.y_die_oof)
    rmse_die_v = meta.rmse(pv, bundle.y_die_val)
    rmse_die_t = meta.rmse(pt, bundle.y_die_test)

    # unit 집계 via tune_and_apply (val 개선 시만 채택 + zero_clip)
    res = aggregate.aggregate_die_to_unit(
        cfg=cfg,
        key_oof=bundle.key_oof, key_val=bundle.key_val, key_test=bundle.key_test,
        die_pred_oof=po, die_pred_val=pv, die_pred_test=pt,
        y_unit_oof=bundle.y_unit_oof,
        y_unit_val=bundle.y_unit_val,
    )
    train_unit_df = res["final_train_unit"]
    val_unit_df   = res["final_val_unit"]
    test_unit_df  = res["final_test_unit"]

    # unit-level RMSE
    y_oof_arr  = aggregate.align_unit_y(train_unit_df[aggregate.KEY_COL].values, bundle.y_unit_oof)
    y_val_arr  = aggregate.align_unit_y(val_unit_df[aggregate.KEY_COL].values,   bundle.y_unit_val)
    y_test_arr = aggregate.align_unit_y(test_unit_df[aggregate.KEY_COL].values,  bundle.y_unit_test)
    rmse_unit_o = meta.rmse(train_unit_df["pred"].values, y_oof_arr)
    rmse_unit_v = meta.rmse(val_unit_df["pred"].values,   y_val_arr)
    rmse_unit_t = meta.rmse(test_unit_df["pred"].values,  y_test_arr)

    params = {"refit_method": method_label, "use_iso": cfg.use_iso}
    if extra_params:
        params.update(extra_params)

    # n_extra 박제
    if bundle.shap_mode == "always_include":
        n_extra = len(bundle.extra_idx)
    elif bundle.shap_mode == "searchable":
        n_extra = _count_extra_in_subset(base_cols, bundle)
    else:
        n_extra = 0

    suffix = "+Iso" if cfg.use_iso else ""
    rec = Record(
        tag=f"refit__{method_label}{suffix}__k{len(base_cols)}",
        stage="refit",
        method=f"{method_label}{suffix}",
        n_base=len(base_cols),
        val_rmse=rmse_unit_v,
        test_rmse=rmse_unit_t,
        oof_rmse=rmse_unit_o,
        val_rmse_die=rmse_die_v,
        test_rmse_die=rmse_die_t,
        oof_rmse_die=rmse_die_o,
        pool_names=[bundle.names[i] for i in base_cols],
        params=params,
        aggregation=res.get("best_agg"),
        pi_threshold=res.get("best_pi_threshold"),
        zero_clip=res.get("best_zero_clip"),
        pos_weights=(list(res["pos_weights"]) if res.get("pos_weights") is not None else None),
        decisions=res.get("decisions", {}),
        n_extra=n_extra,
        extra_tags=_extra_tags_for_record(bundle),
    )
    return RefitArtifact(
        record=rec,
        raw_die_oof=po, raw_die_val=pv, raw_die_test=pt,
        unit_oof_df=train_unit_df,
        unit_val_df=val_unit_df,
        unit_test_df=test_unit_df,
    )


def refit_subset_enet(
    base_cols: tuple[int, ...],
    bundle: ArrayBundle,
    cfg: SqueezeV4Config,
    positive: bool = False,
) -> RefitArtifact:
    """ENet (또는 ENetPositive) refit. 내부 CV는 GroupKFold(unit).
    v4: always_include 모드면 base_cols + extra_idx로 X 슬라이싱."""
    enet_kwargs = dict(
        l1_ratio_grid=cfg.enet_l1_ratio_grid,
        alpha_n=cfg.enet_alpha_n,
        max_iter=cfg.enet_max_iter,
        cv_folds=cfg.enet_cv_folds,
    )
    use_cols = _use_cols(base_cols, bundle)
    Xo = bundle.X_oof[:, use_cols]
    Xv = bundle.X_val[:, use_cols]
    Xt = bundle.X_test[:, use_cols]
    groups_oof = bundle.key_oof[aggregate.KEY_COL].values
    ro, rv, rt, pipe = meta.fit_enet_cv_raw(
        Xo, bundle.y_die_oof, Xv, Xt,
        seed=cfg.seed, positive=positive, groups=groups_oof, **enet_kwargs,
    )
    label = "ENetPositive" if positive else "ENet"
    en = pipe.named_steps["en"]
    extra = {"alpha": float(en.alpha_), "l1_ratio": float(en.l1_ratio_), "positive": positive}
    return _make_record_from_refit(label, base_cols, bundle, ro, rv, rt, cfg, extra)


def refit_subset_combo(
    base_cols: tuple[int, ...],
    bundle: ArrayBundle,
    cfg: SqueezeV4Config,
) -> RefitArtifact:
    """Combo refit (Bag + ENet single + NNLS) / 3. ENet 내부 CV는 GroupKFold(unit)."""
    enet_kwargs = dict(
        l1_ratio_grid=cfg.enet_l1_ratio_grid,
        alpha_n=cfg.enet_alpha_n,
        max_iter=cfg.enet_max_iter,
        cv_folds=cfg.enet_cv_folds,
    )
    use_cols = _use_cols(base_cols, bundle)
    Xo = bundle.X_oof[:, use_cols]
    Xv = bundle.X_val[:, use_cols]
    Xt = bundle.X_test[:, use_cols]
    groups_oof = bundle.key_oof[aggregate.KEY_COL].values
    ro, rv, rt = meta.fit_combo_raw(
        Xo, bundle.y_die_oof, Xv, Xt,
        seeds=cfg.combo_seeds, enet_kwargs=enet_kwargs, groups=groups_oof,
    )
    extra = {"seeds": list(cfg.combo_seeds)}
    return _make_record_from_refit("Combo", base_cols, bundle, ro, rv, rt, cfg, extra)


def compute_meta_cv_oof_for_record(
    rec: Record,
    bundle: ArrayBundle,
    cfg: SqueezeV4Config,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    """Record의 base subset에 대해 GroupKFold OOF로 die-level RMSE 계산 (legacy, ridge 고정).

    refit 후 record.meta_cv_oof_rmse를 채워 넣는 데 사용. die-level 기준 (unit 집계는 비용 큼).
    """
    base_cols = tuple(sorted(bundle.names.index(x) for x in rec.pool_names))
    use_cols = _use_cols(base_cols, bundle)
    Xo = bundle.X_oof[:, use_cols]
    cv_rmse, _ = meta.compute_meta_cv_oof_die(Xo, bundle.y_die_oof, splits, alpha=cfg.cv_ridge_alpha, use_iso=cfg.use_iso)
    return cv_rmse


def compute_meta_cv_oof_unit_for_record(
    rec: Record,
    bundle: ArrayBundle,
    cfg: SqueezeV4Config,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    """**record의 method 그대로** GroupKFold(unit) 안에서 fit/predict → unit-level OOF RMSE.

    이게 "진짜 OOF" — record.oof_rmse(=fit한 data에 predict한 in-sample 값)와 달리
    한 번도 학습에 안 쓰인 unit에 대한 예측만 모아서 계산. 메타 모델 선정의 정직한 기준.

    동작:
      1. fold마다 train fold의 die로 메타 학습기(record method) fit
      2. val fold die에 predict → die-level meta_oof 채움
      3. iso는 cfg.use_iso이면 fold 내부에서 train fold로 fit, val fold로 transform
      4. die-level meta_oof를 simple unit mean으로 집계 → unit-level OOF predict
      5. y_unit_oof_arr와 RMSE
    tune_and_apply(8종 집계 + zero_clip)는 fold마다 적용하지 않는다 — 비용/단순성 trade-off.
    """
    base_method = rec.method.replace("+Iso", "")
    bml = base_method.lower()

    base_cols = tuple(sorted(bundle.names.index(x) for x in rec.pool_names))
    use_cols = _use_cols(base_cols, bundle)
    Xo_full = bundle.X_oof[:, use_cols]
    yo = bundle.y_die_oof
    units_full = bundle.key_oof[aggregate.KEY_COL].values

    alpha = float(rec.params.get("alpha", cfg.cv_ridge_alpha))
    enet_kwargs = dict(
        l1_ratio_grid=cfg.enet_l1_ratio_grid,
        alpha_n=cfg.enet_alpha_n,
        max_iter=cfg.enet_max_iter,
        cv_folds=cfg.enet_cv_folds,
    )

    meta_oof = np.full(len(yo), np.nan, dtype=float)
    for tr_idx, va_idx in splits:
        Xo_tr, y_tr = Xo_full[tr_idx], yo[tr_idx]
        Xo_va = Xo_full[va_idx]
        groups_tr = units_full[tr_idx]
        try:
            # method별 dispatch — fit_meta_raw/fit_combo_raw가 (X_v, X_t) 자리에 받은 행렬에 대해
            # predict한 결과를 (rv, rt)로 반환하므로 Xo_va를 두 자리 모두에 넘긴다.
            if bml == "combo":
                ro_tr_self, rv_va, _ = meta.fit_combo_raw(
                    Xo_tr, y_tr, Xo_va, Xo_va,
                    seeds=cfg.combo_seeds, enet_kwargs=enet_kwargs, groups=groups_tr,
                )
            elif bml in ("enet", "enetpositive"):
                positive = (bml == "enetpositive")
                ro_tr_self, rv_va, _ = meta.fit_meta_raw(
                    "enet_positive" if positive else "enet",
                    Xo_tr, y_tr, Xo_va, Xo_va,
                    alpha=alpha, seed=cfg.seed, groups=groups_tr,
                    enet_kwargs=enet_kwargs,
                )
            elif bml in ("ridge", "nnls", "mean"):
                ro_tr_self, rv_va, _ = meta.fit_meta_raw(
                    bml, Xo_tr, y_tr, Xo_va, Xo_va, alpha=alpha,
                )
            else:
                # 미지의 method (방어용) — ridge로 fallback
                ro_tr_self, rv_va, _ = meta.fit_meta_raw(
                    "ridge", Xo_tr, y_tr, Xo_va, Xo_va, alpha=alpha,
                )

            # iso 후처리: cfg.use_iso=True일 때만, train fold raw로 iso fit → val fold transform
            # (현재 노트북은 use_iso=False 기본이라 이 분기는 보통 건너뜀)
            if cfg.use_iso:
                _, rv_va_iso, _ = meta.apply_iso(ro_tr_self, rv_va, rv_va, y_tr)
                rv_va = rv_va_iso
            meta_oof[va_idx] = meta.clip_nonneg(rv_va)
        except Exception:
            # 한 fold 실패해도 다른 fold는 계속 — 최종 RMSE는 NaN 비율 보면서 해석
            continue

    # die-level meta_oof → unit mean 집계 → unit-level RMSE
    valid = ~np.isnan(meta_oof)
    if not valid.all():
        # NaN이 일부 있어도 unit mean 집계 시 평균에서 빠지도록 0으로 처리하면 잘못된 평균이 나오므로,
        # NaN은 그대로 두고 unit 평균 계산에서 제외하는 안전한 경로를 따로 구현.
        keys = bundle.key_oof[aggregate.KEY_COL].values
        df = pd.DataFrame({"u": keys, "p": meta_oof}).dropna()
        unit_pred = df.groupby("u", sort=True)["p"].mean()
        # bundle.units_oof와 정렬 일치 확보
        unit_pred = unit_pred.reindex(bundle.units_oof).values
    else:
        unit_pred, _ = bundle.agg_oof(meta_oof)
    # NaN unit은 평균에서 제외 후 RMSE 계산
    valid_u = ~np.isnan(unit_pred)
    if not valid_u.any():
        return float("nan")
    diff = unit_pred[valid_u] - bundle.y_unit_oof_arr[valid_u]
    return float(np.sqrt(np.mean(diff ** 2)))


def run_refit_stage(
    records: list[Record],
    bundle: ArrayBundle,
    cfg: SqueezeV4Config,
    cv_splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[list[Record], dict[int, RefitArtifact]]:
    """fast records의 상위 unique subset에 대해 정밀 refit + tune_and_apply.

    Returns
    -------
    refit_records : list[Record] — refit으로 추가된 record들
    artifacts     : {id(rec): RefitArtifact} — best 저장 시 unit pred 등에 접근
    """
    deadline = parse_deadline(cfg.deadline)
    # unique top subsets — refit 전이라 meta_cv_oof_rmse는 모두 NaN.
    # cfg.select_by="meta_cv_oof"일 때 score_rec이 inf를 반환해서 정렬이 무의미해지는 것을 방지하기 위해,
    # 후보 선정은 **항상 oof_rmse 기준**으로 한다 (refit 후 selection만 select_by로).
    seen: set[tuple[int, ...]] = set()
    unique_cols: list[tuple[int, ...]] = []
    for rec in sorted(records, key=lambda r: r.oof_rmse):
        cols = tuple(sorted(bundle.names.index(x) for x in rec.pool_names))
        if cols in seen:
            continue
        seen.add(cols)
        unique_cols.append(cols)
        if len(unique_cols) >= cfg.top_refit:
            break

    if cfg.verbose:
        print(f"\n[refit] top_refit={cfg.top_refit}, combo_refit={cfg.combo_refit}, "
              f"unique subsets={len(unique_cols)}")

    refit_records: list[Record] = []
    artifacts: dict[int, RefitArtifact] = {}

    for i, cols in enumerate(unique_cols, 1):
        if should_stop(deadline, cfg.deadline_margin_minutes):
            if cfg.verbose:
                print(f"  [deadline] refit stopped at {i - 1}/{len(unique_cols)}")
            break

        for positive in (False, True):
            art = refit_subset_enet(cols, bundle, cfg, positive=positive)
            refit_records.append(art.record)
            artifacts[id(art.record)] = art

        if i <= cfg.combo_refit:
            art_c = refit_subset_combo(cols, bundle, cfg)
            refit_records.append(art_c.record)
            artifacts[id(art_c.record)] = art_c

        # meta_cv_oof_rmse — **record의 method 그대로** GroupKFold(unit) OOF unit RMSE
        # (이전 버전은 모든 record에 ridge die-level OOF를 박제했지만, method 비교 불가했음.
        #  지금은 record method별로 fold-fit해서 unit RMSE를 박제 → select_by="meta_cv_oof"가 정직한 기준이 됨.)
        last_n = 3 if i <= cfg.combo_refit else 2
        new_records_this_subset = refit_records[-last_n:]
        if cv_splits is not None:
            for r in new_records_this_subset:
                r.meta_cv_oof_rmse = compute_meta_cv_oof_unit_for_record(r, bundle, cfg, cv_splits)
            mc_log = ", ".join(
                f"{r.method.split('+')[0]}={r.meta_cv_oof_rmse:.6f}"
                for r in new_records_this_subset
            )
        else:
            mc_log = "skipped"

        if cfg.verbose:
            best_score = min(score_rec(r, cfg.select_by, cfg.val_gap_penalty) for r in (records + refit_records))
            best_val = min(r.val_rmse for r in (records + refit_records))
            print(f"  refit {i:3d}/{len(unique_cols)}  "
                  f"best_{cfg.select_by}={best_score:.9f}  "
                  f"mcv_unit[{mc_log}]  val_best={best_val:.9f}")

    return refit_records, artifacts


# ---------------------------------------------------------------------------
# ArrayBundle factory — 노트북에서 한 줄로 호출
# ---------------------------------------------------------------------------
def build_array_bundle(
    cfg: SqueezeV4Config,
    models: list[dict],
) -> ArrayBundle:
    """cfg.shap_caches와 cfg.shap_mode를 반영한 ArrayBundle을 한 번에 생성.

    내부에서:
      1. discovery.build_die_matrix로 base die-level 행렬 oof/val/test 생성 (정렬 + key_df)
      2. shap.load_shap_caches로 SHAP die-level 행렬 로드 + key_df 순서로 정렬
      3. shap_mode에 따라 X 결합 + names/extra_idx 구성
      4. y_unit_oof/val/test 로드 (각 모델의 oof_unit.csv 등이 아니라 base 모델 die의 health에서 unit별 unique값)
      5. fast aggregator + unique_units + y_unit_array 사전 캐싱
      → ArrayBundle 반환.

    SHAP이 비활성(cfg.shap_caches가 비어 있으면)일 때는 v3와 동일 동작.
    """
    from . import discovery, shap as shap_mod

    # 1) base 행렬 + key_df
    X_oof_base, y_die_oof, key_oof = discovery.build_die_matrix(models, "oof")
    X_val_base, y_die_val, key_val = discovery.build_die_matrix(models, "val")
    X_test_base, y_die_test, key_test = discovery.build_die_matrix(models, "test")

    base_names = [m["name"] for m in models]
    n_base = len(base_names)

    # 2) SHAP 캐시 로드 (key_df 순서로 정렬되어 나옴)
    shap_res = shap_mod.load_shap_caches(cfg, key_oof, key_val, key_test)
    n_extra = len(shap_res["names"])

    # 3) 결합 + names/extra_idx 결정
    if n_extra == 0:
        X_oof, X_val, X_test = X_oof_base, X_val_base, X_test_base
        names = base_names
        extra_idx: tuple[int, ...] = ()
        extra_names: list[str] = []
        extra_tags: list[str] = []
        shap_mode_label = "none"
    else:
        # X에는 항상 base + extra 컬럼이 들어감 (한 번만 hstack)
        X_oof = np.hstack([X_oof_base, shap_res["X_oof"]])
        X_val = np.hstack([X_val_base, shap_res["X_val"]])
        X_test = np.hstack([X_test_base, shap_res["X_test"]])
        extra_names = list(shap_res["names"])
        extra_tags = list(shap_res["tags"])

        if cfg.shap_mode == "always_include":
            # names는 base만 → subset search 후보 = base만. extra_idx로 X 슬라이싱 시 합산.
            names = list(base_names)
            extra_idx = tuple(range(n_base, n_base + n_extra))
            shap_mode_label = "always_include"
            if cfg.verbose:
                print(f"  [shap_mode] always_include - extra_idx={n_extra}개 컬럼을 메타 학습 시 항상 포함")
        elif cfg.shap_mode == "searchable":
            # names = base + extra → subset search 후보에 등록. extra_idx는 비움.
            names = list(base_names) + extra_names
            extra_idx = ()
            shap_mode_label = "searchable"
            if cfg.verbose:
                print(f"  [shap_mode] searchable - base {n_base}개 + SHAP {n_extra}개가 통합 풀 (총 {n_base + n_extra}개)")
        else:
            raise ValueError(f"알 수 없는 cfg.shap_mode: {cfg.shap_mode!r}")

    # 4) y_unit_oof/val/test — base 모델 oof_unit.csv가 unit-level y 진실원천
    #    (key_df는 die-level이라 unit y는 따로 로드)
    def _load_y_unit(models, split):
        from pathlib import Path
        for m in models:
            p_unit = Path(m["path"]) / f"{split}_unit.csv"
            if p_unit.exists():
                df = pd.read_csv(p_unit)
                if "health" in df.columns and aggregate.KEY_COL in df.columns:
                    return df[[aggregate.KEY_COL, "health"]].copy()
        # fallback: die-level에서 unit별 first
        kdf = {"oof": key_oof, "val": key_val, "test": key_test}[split]
        ydie = {"oof": y_die_oof, "val": y_die_val, "test": y_die_test}[split]
        return (
            pd.DataFrame({aggregate.KEY_COL: kdf[aggregate.KEY_COL].values, "health": ydie})
            .groupby(aggregate.KEY_COL, as_index=False)
            .first()
        )

    y_unit_oof = _load_y_unit(models, "oof")
    y_unit_val = _load_y_unit(models, "val")
    y_unit_test = _load_y_unit(models, "test")

    # 5) fast aggregator + units + y_unit_array
    agg_oof, units_oof = aggregate.build_unit_mean_aggregator(key_oof)
    agg_val, units_val = aggregate.build_unit_mean_aggregator(key_val)
    agg_test, units_test = aggregate.build_unit_mean_aggregator(key_test)
    y_unit_oof_arr = aggregate.align_unit_y(units_oof, y_unit_oof)
    y_unit_val_arr = aggregate.align_unit_y(units_val, y_unit_val)
    y_unit_test_arr = aggregate.align_unit_y(units_test, y_unit_test)

    return ArrayBundle(
        X_oof=X_oof, X_val=X_val, X_test=X_test,
        y_die_oof=y_die_oof, y_die_val=y_die_val, y_die_test=y_die_test,
        key_oof=key_oof, key_val=key_val, key_test=key_test,
        y_unit_oof=y_unit_oof, y_unit_val=y_unit_val, y_unit_test=y_unit_test,
        agg_oof=agg_oof, agg_val=agg_val, agg_test=agg_test,
        units_oof=units_oof, units_val=units_val, units_test=units_test,
        y_unit_oof_arr=y_unit_oof_arr, y_unit_val_arr=y_unit_val_arr, y_unit_test_arr=y_unit_test_arr,
        names=names,
        extra_idx=extra_idx,
        extra_names=extra_names if extra_names else None,
        extra_tags=extra_tags if extra_tags else None,
        shap_mode=shap_mode_label,
    )
