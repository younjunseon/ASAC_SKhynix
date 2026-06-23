"""
01_zit/zit_only_eql/hpo.py — ZITboostEQL(EQL φ, zit_only) 병렬 HPO 튜너 (thin).

zit_only_pearson/hpo.py 의 쌍둥이 — 모델 클래스만 ZITboostEQLRegressor(EQL φ M-step:
Tweedie unit deviance target)로 교체. 구조(병렬 하네스·objective·집계 mean)는 동일.
**eql 전용 튜너**: 독립 study로 eql 고유 best HP를 넓은 범위에서 탐색한다.

실행 (워커 3개 권장):
  python 3_modeling/01_zit/00_precompute_pp.py                                # 1회: pp.npy
  python 3_modeling/01_zit/zit_only_eql/hpo.py --worker-id w1 --n-jobs 4 --end-at 2026-06-20T05:00 > w1.log 2>&1
  ... w2, w3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "setup.py").exists() and (p / "utils").exists():
            return p
    raise RuntimeError(f"Project root not found from {start}")


_ROOT = _find_project_root(Path(__file__).resolve())
for _p in [_ROOT, _ROOT / "3_modeling", _ROOT / "3_modeling" / "01_zit"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import zit_objective  # noqa: E402
from modules import parallel_hpo as ph  # noqa: E402
from modules.zit import ZITboostEQLRegressor  # noqa: E402


OUT_SUBDIR = "01_zit/zit_only_eql"
DEFAULT_EXP_ID = "zit_only_eql"

# zit_only 모델군 HP를 넓은 범위로 탐색 (pearson과 동일 기준, eql 독립 탐색).
SEARCH = {
    "zeta": {"type": "float", "low": 1.02, "high": 1.20, "log": False},
    "n_em_iters": {"type": "int", "low": 15, "high": 28},
    "mu_n_estimators": {"type": "int", "low": 150, "high": 240},
    "mu_learning_rate": {"type": "float", "low": 0.0009, "high": 0.0032, "log": True},
    "mu_num_leaves": {"type": "int", "low": 16, "high": 128},
    "mu_max_depth": {"type": "int", "low": 2, "high": 7},
    "mu_min_child_samples": {"type": "int", "low": 160, "high": 320},
    "mu_subsample": {"type": "float", "low": 0.55, "high": 0.85, "log": False},
    "mu_colsample_bytree": {"type": "float", "low": 0.14, "high": 0.34, "log": False},
    "mu_reg_alpha": {"type": "float", "low": 1e-4, "high": 0.006, "log": True},
    "mu_reg_lambda": {"type": "float", "low": 5e-5, "high": 0.007, "log": True},
    "pi_n_estimators": {"type": "int", "low": 90, "high": 170},
    "pi_learning_rate": {"type": "float", "low": 0.032, "high": 0.065, "log": True},
    "pi_num_leaves": {"type": "int", "low": 160, "high": 260},
    "pi_max_depth": {"type": "int", "low": 12, "high": 22},
    "pi_min_child_samples": {"type": "int", "low": 25, "high": 50},
    "phi_n_estimators": {"type": "int", "low": 30, "high": 60},
    "phi_learning_rate": {"type": "float", "low": 0.0012, "high": 0.0065, "log": True},
    "phi_num_leaves": {"type": "int", "low": 40, "high": 110},
    "phi_max_depth": {"type": "int", "low": 5, "high": 10},
    "phi_min_child_samples": {"type": "int", "low": 180, "high": 520},
}
TAU_PI_RANGE = (0.84, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="ZIT-only (EQL phi) parallel HPO worker.")
    ph.add_common_args(ap, default_exp_id=DEFAULT_EXP_ID, default_n_jobs=4, default_n_startup=120)
    args = ap.parse_args()

    data = ph.load_pp_mmap(args.precomputed_dir)
    objective = zit_objective.make_zit_objective(
        args, data,
        model_class=ZITboostEQLRegressor,
        search=SEARCH,
        tau_range=TAU_PI_RANGE,
        use_unit_id=False,
        agg="mean",
    )
    out_dir = ph.resolve_out_dir(OUT_SUBDIR)
    db_path = ph.study_db_path(out_dir, args.user, args.exp_id)
    study = ph.make_study(args, db_path, study_meta={
        "model": "ZITboostEQLRegressor (zit_only, EQL phi)",
        "search_space": SEARCH,
        "tau_pi_range": TAU_PI_RANGE,
        "agg": "mean",
        "out_subdir": OUT_SUBDIR,
    })
    print(f"OUT_DIR={out_dir}\nDB={db_path}\nsearch HP={len(SEARCH)} + tau_pi=1")
    ph.run_optimize(study, objective, args)


if __name__ == "__main__":
    main()
