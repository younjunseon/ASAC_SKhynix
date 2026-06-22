"""결과 저장 — die-level + unit-level 양쪽 + summary + best weights.

저장 구조:
    {result_dir}/run_{ts}/
      ├── summary.json           # cfg + 전체 records + models
      ├── summary.csv            # records를 평탄화한 CSV (정렬: score)
      ├── best_die_oof.csv       # 1등 record의 die-level OOF pred (iso 적용 후)
      ├── best_die_val.csv
      ├── best_die_test.csv
      ├── best_unit_oof.csv      # 1등 record의 unit-level OOF pred (집계 후 최종)
      ├── best_unit_val.csv
      ├── best_unit_test.csv
      ├── best_weights.json      # 1등 record의 메타 가중치 (재현용)
      └── (옵션) top{N}_weights.json

unit CSV는 v2와 schema 호환 ([ufs_serial, pred]) — 다른 노트북에서 그대로 부를 수 있음.
die CSV는 [ufs_serial, position, pred] 컬럼.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import StackingConfig
from .records import Record, score_rec
from .search import ArrayBundle, RefitArtifact
from .weights import extract_weights


def _record_to_dict(rec: Record, cfg: StackingConfig) -> dict:
    d = asdict(rec)
    d["objective"] = cfg.select_by
    d["objective_score"] = score_rec(rec, cfg.select_by, cfg.val_gap_penalty)
    return d


def save_outputs(
    records: list[Record],
    artifacts: dict[int, RefitArtifact],
    bundle: ArrayBundle,
    cfg: StackingConfig,
    models: list[dict],
    top_n_weights: int = 1,
) -> Path:
    """run 폴더 생성 + summary + best die/unit CSV + best weights JSON 저장.

    Parameters
    ----------
    records : list[Record] — 전체 (seed + random + local + optuna + refit)
    artifacts : refit 단계에서 생성된 die-level pred + unit DataFrame
        fast 단계 record는 여기 없음 → 최상위가 fast인 경우 weights만 박제하고 CSV는 refit 1등 기준.

    Returns
    -------
    run_dir : 저장된 폴더 경로
    """
    ts = datetime.now().strftime("%m%d_%H%M%S")
    run_dir = cfg.result_dir / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ─── summary ────────────────────────────────────────────────────
    sorted_records = sorted(records, key=lambda r: score_rec(r, cfg.select_by, cfg.val_gap_penalty))
    slim = [_record_to_dict(r, cfg) for r in sorted_records]

    # SHAP 메타 — bundle에서 추출 (v4: cfg.to_dict()에도 포함되지만 따로 정리해서 쉽게 검사 가능)
    shap_meta = {
        "shap_mode": bundle.shap_mode,
        "shap_caches": list(cfg.shap_caches),
        "shap_top_k": cfg.shap_top_k,
        "shap_prefix_with_tag": cfg.shap_prefix_with_tag,
        "n_extra_cols": len(bundle.extra_names) if bundle.extra_names else 0,
        "extra_tags": list(bundle.extra_tags) if bundle.extra_tags else [],
    }

    summary = {
        "config": cfg.to_dict(),
        "shap": shap_meta,
        "n_records": len(sorted_records),
        "models": [
            {k: (str(v) if k == "path" else v) for k, v in m.items()}
            for m in models
        ],
        "records": slim,
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)
    pd.DataFrame(slim).drop(
        columns=["params", "pool_names", "decisions", "pos_weights"],
        errors="ignore",
    ).to_csv(run_dir / "summary.csv", index=False)

    # ─── best 1등 (objective_score 기준) 의 CSV ─────────────────────
    best = sorted_records[0]
    best_art = artifacts.get(id(best))

    def _die_df(key_df, die_pred):
        # position 컬럼은 베이스 모델들에 따라 없을 수도 있음 (build_die_matrix가 attach 실패 시).
        # 그 경우 ufs_serial + run_wf_xy만으로 저장 — die unique key는 보존됨.
        cols = {"ufs_serial": key_df["ufs_serial"].values}
        if "position" in key_df.columns:
            cols["position"] = key_df["position"].values
        elif "run_wf_xy" in key_df.columns:
            cols["run_wf_xy"] = key_df["run_wf_xy"].values
        cols["pred"] = die_pred
        return pd.DataFrame(cols)

    def _write_artifact(art):
        _die_df(bundle.key_oof,  art.raw_die_oof ).to_csv(run_dir / "best_die_oof.csv",  index=False)
        _die_df(bundle.key_val,  art.raw_die_val ).to_csv(run_dir / "best_die_val.csv",  index=False)
        _die_df(bundle.key_test, art.raw_die_test).to_csv(run_dir / "best_die_test.csv", index=False)
        art.unit_oof_df.to_csv(run_dir / "best_unit_oof.csv", index=False)
        art.unit_val_df.to_csv(run_dir / "best_unit_val.csv", index=False)
        art.unit_test_df.to_csv(run_dir / "best_unit_test.csv", index=False)

    if best_art is not None:
        _write_artifact(best_art)
    else:
        # fast 단계 record가 1등 → refit 1등(sorted_records 기준)으로 fallback
        # (best_weights.json도 fast record 그대로 박제하지만 CSV는 refit에서 가져옴 = 의도된 분리)
        for r in sorted_records:
            art = artifacts.get(id(r))
            if art is not None:
                _write_artifact(art)
                break

    # ─── 상위 N record의 weights ────────────────────────────────────
    # **summary/CSV/weights를 모두 같은 record로 통일** — sorted_records (score_rec 기준) 사용.
    # 이전엔 weights만 oof_rmse 별도 기준이라 best 정의가 3개로 갈렸음 (불일치).
    weights_top = []
    selection_label = f"score_rec(select_by={cfg.select_by!r}, val_gap_penalty={cfg.val_gap_penalty})"
    for rank, rec in enumerate(sorted_records[:top_n_weights], 1):
        try:
            w = extract_weights(rec, bundle, cfg)
            w["rank_in_run"] = rank
            w["selection_criterion"] = selection_label
            weights_top.append(w)
        except Exception as e:
            weights_top.append({
                "rank_in_run": rank,
                "tag": rec.tag,
                "error": f"{type(e).__name__}: {e}",
            })

    out_name = "best_weights.json" if top_n_weights == 1 else f"top{top_n_weights}_weights.json"
    with open(run_dir / out_name, "w", encoding="utf-8") as f:
        json.dump({
            "source_run": str(run_dir),
            "selection_criterion": selection_label,
            "cfg_select_by": cfg.select_by,
            "cfg_val_gap_penalty": cfg.val_gap_penalty,
            "top_n": top_n_weights,
            "weights": weights_top,
        }, f, ensure_ascii=False, indent=2, default=_json_default)

    # ─── 콘솔 출력 ──────────────────────────────────────────────────
    if cfg.verbose:
        print("\n" + "=" * 92)
        print(f"FINAL TOP 20 BY {cfg.select_by.upper()} OBJECTIVE")
        print("=" * 92)
        for i, r in enumerate(sorted_records[:20], 1):
            # meta_cv_oof_rmse는 이제 unit-level (search.compute_meta_cv_oof_unit_for_record 기준)
            mcv = f"mcv_unit={r.meta_cv_oof_rmse:.6f} " if not math.isnan(r.meta_cv_oof_rmse) else ""
            agg = f"agg={r.aggregation:<13s}" if r.aggregation else "agg=---"
            print(f"{i:2d}. obj={score_rec(r, cfg.select_by, cfg.val_gap_penalty):.9f} "
                  f"oof={r.oof_rmse:.6f} val={r.val_rmse:.6f} test={r.test_rmse:.6f} "
                  f"{mcv}{agg} k={r.n_base:2d} {r.tag}")

        # die-level RMSE 별도 top
        print("\nDIE-LEVEL OOF RMSE TOP 10 (진단치)")
        for i, r in enumerate(sorted(sorted_records, key=lambda r: r.oof_rmse_die)[:10], 1):
            print(f"{i:2d}. die_oof={r.oof_rmse_die:.6f}  die_val={r.val_rmse_die:.6f}  "
                  f"die_test={r.test_rmse_die:.6f}  unit_oof={r.oof_rmse:.6f}  k={r.n_base:2d} {r.tag}")

        print(f"\nSaved: {run_dir}")
    return run_dir


def _json_default(o):
    """numpy/pathlib 객체를 JSON 직렬화 가능하게."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
