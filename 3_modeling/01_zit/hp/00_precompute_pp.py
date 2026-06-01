"""
Precompute the BagZIT preprocessing into a single mmap-shareable numeric .npy.

WHY
  Each parallel HPO worker currently re-runs the full preprocessing in its own
  process (load_preprocessed_data), which (a) peaks at ~5.3 GB during spatial
  imputation and (b) keeps a private ~0.48 GB feature matrix per worker. On one PC
  that caps how many workers fit in RAM. This script bakes the die-level result
  ONCE into a single numeric matrix so workers can `np.load(..., mmap_mode='r')`
  and share ONE copy of the heavy matrix via the OS page cache.

WHAT IS SAVED  (0_data/precomputed/<name>/)
  pp.npy        float64 (n_dies, n_features+2)   col[:F]=features, col[F]=uid_code, col[F+1]=y_die
  units.npy     float64 (n_units, 2)             col0=uid_code (UNIT order), col1=y_unit
  feat_cols.json                                 feature names (len F)
  uid_map.json  {code: ufs_serial}               int code -> original unit id (for later submission)
  manifest.json shapes / column layout / pp params / fingerprint

REPRODUCIBILITY
  Runs the EXACT preprocessing of 02_bag_zit_parallel_hpo.py by importing its
  load_preprocessed_data (single source of truth: same PP_FIXED, same in-process
  median-imputation patch, same CLIP_Y_EXTREME). uid is relabelled str->int code,
  but die->unit groupby and KFold-by-position are invariant to the relabel, and the
  UNIT order is preserved via units.npy, so CV folds and OOF RMSE match the
  self-preprocessing path byte-for-byte. Precomputed workers are a clean continuation
  / A-B of the same study.

RUN
  python 3_modeling/01_zit/hp/00_precompute_pp.py
  python 3_modeling/01_zit/hp/00_precompute_pp.py --name bag_zit_pp --no-clip-y-extreme
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

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
for _sub in ["2_preprocessing", "3_modeling"]:
    _p = PROJECT_ROOT / _sub
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from utils.config import DATA_DIR, KEY_COL, TARGET_COL  # noqa: E402

WORKER_PATH = PROJECT_ROOT / "3_modeling" / "01_zit" / "hp" / "02_bag_zit_parallel_hpo.py"


def _load_worker_module():
    """Import the parallel-HPO worker by file path (its name starts with a digit)."""
    spec = importlib.util.spec_from_file_location("bagworker", WORKER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute BagZIT preprocessing -> single mmap .npy.")
    ap.add_argument("--name", default="bag_zit_pp",
                    help="output subfolder under 0_data/precomputed/")
    ap.add_argument("--no-clip-y-extreme", action="store_true",
                    help="match the worker's --no-clip-y-extreme (default: clip on, same as worker default)")
    a = ap.parse_args()

    m = _load_worker_module()

    # Reuse the worker's exact preprocessing (PP_FIXED + median patch + clip).
    wargs = argparse.Namespace(no_clip_y_extreme=a.no_clip_y_extreme)
    x_train, uid_die_str, y_unit_s, y_die, feat = m.load_preprocessed_data(wargs)

    x_train = np.ascontiguousarray(x_train, dtype=np.float64)
    n_dies, F = x_train.shape

    # uid str -> int code (die order). Unit ids reuse the SAME code mapping.
    codes_die, uniques = pd.factorize(np.asarray(uid_die_str))      # codes_die:(n_dies,) int, uniques:(n_units,) str
    str_to_code = {s: i for i, s in enumerate(uniques)}
    unit_index = np.asarray(y_unit_s.index)
    missing = [s for s in unit_index if s not in str_to_code]
    if missing:
        raise RuntimeError(f"{len(missing)} unit ids in y have no die rows (e.g. {missing[:3]}) "
                           f"-- preprocessing/alignment mismatch, aborting.")
    unit_codes = np.array([str_to_code[s] for s in unit_index], dtype=np.float64)
    y_unit = np.asarray(y_unit_s.values, dtype=np.float64)

    # ONE big die-level numeric matrix: [features | uid_code | y_die]
    pp = np.empty((n_dies, F + 2), dtype=np.float64)
    pp[:, :F] = x_train
    pp[:, F] = codes_die.astype(np.float64)
    pp[:, F + 1] = np.asarray(y_die, dtype=np.float64)

    # small unit-level sidecar (preserves ORIGINAL unit order -> identical CV folds)
    units = np.column_stack([unit_codes, y_unit])                  # (n_units, 2)

    out = Path(DATA_DIR) / "precomputed" / a.name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pp.npy", pp)
    np.save(out / "units.npy", units)
    (out / "feat_cols.json").write_text(json.dumps(list(map(str, feat)), ensure_ascii=False), encoding="utf-8")
    (out / "uid_map.json").write_text(
        json.dumps({int(i): str(s) for i, s in enumerate(uniques)}, ensure_ascii=False), encoding="utf-8")

    fingerprint = float(np.nansum(pp[::101, :]))                   # cheap sanity fingerprint
    manifest = {
        "n_features": int(F),
        "n_dies": int(n_dies),
        "n_units": int(len(uniques)),
        "pp_shape": [int(n_dies), int(F + 2)],
        "layout": {"features": [0, int(F)], "uid_code_col": int(F), "y_die_col": int(F + 1)},
        "units_layout": {"uid_code_col": 0, "y_unit_col": 1},
        "dtype": "float64",
        "clip_y_extreme": not a.no_clip_y_extreme,
        "pp_fixed": m.PP_FIXED,
        "key_col": KEY_COL,
        "target_col": TARGET_COL,
        "fingerprint_nansum_stride101": fingerprint,
        "source_worker": WORKER_PATH.name,
        "reconstruct": ("x=pp[:,:F]; uid_die=pp[:,F].astype(int64); y_die=pp[:,F+1]; "
                        "y_unit_s=pd.Series(units[:,1], index=units[:,0].astype(int64))"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    def _mb(path: Path) -> float:
        return path.stat().st_size / 1024 / 1024

    print(f"[precompute done] out={out}")
    print(f"  pp.npy    : shape={pp.shape}  {_mb(out / 'pp.npy'):.1f} MB   <- mmap target (shared across workers)")
    print(f"  units.npy : shape={units.shape}  {_mb(out / 'units.npy'):.3f} MB")
    print(f"  feat_cols.json / uid_map.json / manifest.json written")
    print(f"  n_features={F}, n_dies={n_dies:,}, n_units={len(uniques):,}")
    print(f"  fingerprint(nansum stride101)={fingerprint:.6f}")


if __name__ == "__main__":
    main()
