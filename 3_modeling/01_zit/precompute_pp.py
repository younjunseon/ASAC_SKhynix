"""
ZIT 전처리 결과를 mmap으로 공유 가능한 단일 numeric .npy로 미리 구워 둔다.

배경
  병렬 HPO 워커는 각 프로세스에서 전체 전처리(load_preprocessed_data)를 매번 다시 돌린다.
  이때 (a) 공간 보간 단계에서 ~5.3 GB까지 피크가 튀고, (b) 워커마다 ~0.48 GB feature 행렬을
  따로 들고 있어, PC 한 대에 띄울 수 있는 워커 수가 RAM에 묶인다. 이 스크립트는 die-level
  결과를 **한 번만** 단일 numeric 행렬로 구워 두어, 워커가 `np.load(..., mmap_mode='r')`로
  무거운 행렬 **한 벌**을 OS 페이지 캐시를 통해 공유하게 한다.

저장 산출물  (0_data/precomputed/<name>/)
  pp.npy        float64 (n_dies, n_features+2)   col[:F]=features, col[F]=uid_code, col[F+1]=y_die
  units.npy     float64 (n_units, 2)             col0=uid_code (UNIT 순서), col1=y_unit
  feat_cols.json                                 feature 이름 (길이 F)
  uid_map.json  {code: ufs_serial}               int code -> 원래 unit id (이후 제출용)
  manifest.json shape / 컬럼 레이아웃 / pp 파라미터 / fingerprint

재현성
  zit_pp.load_preprocessed_data와 **완전히 동일한** 전처리를 돌린다(4개 ZIT 조합의 단일 기준:
  동일 PP_FIXED, cleaning.py 정본 공간 보간, 동일 CLIP_Y_EXTREME). uid는 str->int code로 다시
  라벨링하지만, die->unit groupby와 position 기반 KFold는 라벨 교체에 불변이고 UNIT 순서도
  units.npy로 보존되므로, CV fold와 OOF RMSE가 자체 전처리 경로와 byte 단위로 일치한다.

실행
  python 3_modeling/01_zit/precompute_pp.py
  python 3_modeling/01_zit/precompute_pp.py --name zit_pp --no-clip-y-extreme
"""

from __future__ import annotations

import argparse
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

# zit_pp.py(공용 PP 로더)를 normal import 하기 위해 이 스크립트 폴더(01_zit)도 path에 추가
_SELF_DIR = Path(__file__).resolve().parent
if str(_SELF_DIR) not in sys.path:
    sys.path.insert(0, str(_SELF_DIR))

from utils.config import DATA_DIR, KEY_COL, TARGET_COL  # noqa: E402
import zit_pp  # noqa: E402  (01_zit/zit_pp.py — 4조합 공통 전처리, cleaning.py 정본)


def main() -> None:
    ap = argparse.ArgumentParser(description="ZIT 전처리를 단일 mmap .npy로 미리 굽는다.")
    ap.add_argument("--name", default="zit_pp",
                    help="0_data/precomputed/ 아래 출력 하위 폴더")
    ap.add_argument("--no-clip-y-extreme", action="store_true",
                    help="워커의 --no-clip-y-extreme와 맞춤 (기본: clip on, 워커 기본값과 동일)")
    a = ap.parse_args()

    # 공용 ZIT 전처리 재사용 (PP_FIXED + clip, cleaning.py 정본 보간).
    x_train, uid_die_str, y_unit_s, y_die, feat = zit_pp.load_preprocessed_data(
        clip_y_extreme=not a.no_clip_y_extreme
    )

    x_train = np.ascontiguousarray(x_train, dtype=np.float64)
    n_dies, F = x_train.shape

    # uid str -> int code (die 순서). unit id도 같은 code 매핑을 그대로 쓴다.
    codes_die, uniques = pd.factorize(np.asarray(uid_die_str))      # codes_die:(n_dies,) int, uniques:(n_units,) str
    str_to_code = {s: i for i, s in enumerate(uniques)}
    unit_index = np.asarray(y_unit_s.index)
    missing = [s for s in unit_index if s not in str_to_code]
    if missing:
        raise RuntimeError(f"{len(missing)} unit ids in y have no die rows (e.g. {missing[:3]}) "
                           f"-- preprocessing/alignment mismatch, aborting.")
    unit_codes = np.array([str_to_code[s] for s in unit_index], dtype=np.float64)
    y_unit = np.asarray(y_unit_s.values, dtype=np.float64)

    # die-level 큰 numeric 행렬 한 벌: [features | uid_code | y_die]
    pp = np.empty((n_dies, F + 2), dtype=np.float64)
    pp[:, :F] = x_train
    pp[:, F] = codes_die.astype(np.float64)
    pp[:, F + 1] = np.asarray(y_die, dtype=np.float64)

    # 작은 unit-level 보조 배열 (원래 unit 순서 보존 -> CV fold 동일)
    units = np.column_stack([unit_codes, y_unit])                  # (n_units, 2)

    out = Path(DATA_DIR) / "precomputed" / a.name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pp.npy", pp)
    np.save(out / "units.npy", units)
    (out / "feat_cols.json").write_text(json.dumps(list(map(str, feat)), ensure_ascii=False), encoding="utf-8")
    (out / "uid_map.json").write_text(
        json.dumps({int(i): str(s) for i, s in enumerate(uniques)}, ensure_ascii=False), encoding="utf-8")

    fingerprint = float(np.nansum(pp[::101, :]))                   # 가벼운 sanity fingerprint
    manifest = {
        "n_features": int(F),
        "n_dies": int(n_dies),
        "n_units": int(len(uniques)),
        "pp_shape": [int(n_dies), int(F + 2)],
        "layout": {"features": [0, int(F)], "uid_code_col": int(F), "y_die_col": int(F + 1)},
        "units_layout": {"uid_code_col": 0, "y_unit_col": 1},
        "dtype": "float64",
        "clip_y_extreme": not a.no_clip_y_extreme,
        "pp_fixed": zit_pp.PP_FIXED,
        "key_col": KEY_COL,
        "target_col": TARGET_COL,
        "fingerprint_nansum_stride101": fingerprint,
        "source": "01_zit/zit_pp.py::load_preprocessed_data (cleaning.py canonical)",
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
