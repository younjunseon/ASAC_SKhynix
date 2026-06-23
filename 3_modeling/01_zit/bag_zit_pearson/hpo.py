"""
01_zit/bag_zit_pearson/hpo.py — BagZIT(pearson φ) 병렬 HPO 튜너 (thin).

zit_only 와 차이: die 예측을 unit 단위로 **sum** 집계, fit에 `unit_id=` 전달(같은 unit의
4 die에 unit health를 배분하는 bagging). 모델 = BagZITboostRegressor(pearson φ).
HP는 넓은 범위에서 탐색한다.

실행 (워커 3개 권장):
  python 3_modeling/01_zit/00_precompute_pp.py                                 # 1회: pp.npy
  python 3_modeling/01_zit/bag_zit_pearson/hpo.py --worker-id w1 --n-jobs 4 --end-at 2026-06-20T05:00 > w1.log 2>&1
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
from modules.zit import BagZITboostRegressor  # noqa: E402


OUT_SUBDIR = "01_zit/bag_zit_pearson"
DEFAULT_EXP_ID = "bag_zit_pearson"

# bag 모델군 HP를 넓은 범위로 탐색.
SEARCH = {
    "zeta": {"type": "float", "low": 1.01, "high": 1.30, "log": False},
    "n_em_iters": {"type": "int", "low": 12, "high": 30},
    "mu_n_estimators": {"type": "int", "low": 50, "high": 180},
    "mu_learning_rate": {"type": "float", "low": 0.0035, "high": 0.0120, "log": True},
    "mu_num_leaves": {"type": "int", "low": 80, "high": 256},
    "mu_max_depth": {"type": "int", "low": 3, "high": 7},
    "mu_min_child_samples": {"type": "int", "low": 60, "high": 220},
    "mu_subsample": {"type": "float", "low": 0.40, "high": 0.95, "log": False},
    "mu_colsample_bytree": {"type": "float", "low": 0.22, "high": 0.85, "log": False},
    "mu_reg_alpha": {"type": "float", "low": 1.0e-6, "high": 5.0e-4, "log": True},
    "mu_reg_lambda": {"type": "float", "low": 0.003, "high": 0.060, "log": True},
    "pi_n_estimators": {"type": "int", "low": 250, "high": 550},
    "pi_learning_rate": {"type": "float", "low": 0.010, "high": 0.045, "log": True},
    "pi_num_leaves": {"type": "int", "low": 70, "high": 180},
    "pi_max_depth": {"type": "int", "low": 8, "high": 24},
    "pi_min_child_samples": {"type": "int", "low": 30, "high": 90},
    "phi_n_estimators": {"type": "int", "low": 90, "high": 250},
    "phi_learning_rate": {"type": "float", "low": 0.0040, "high": 0.0140, "log": True},
    "phi_num_leaves": {"type": "int", "low": 24, "high": 110},
    "phi_max_depth": {"type": "int", "low": 2, "high": 6},
    "phi_min_child_samples": {"type": "int", "low": 80, "high": 260},
}
TAU_PI_RANGE = (0.80, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="BagZIT (pearson phi) parallel HPO worker.")
    ph.add_common_args(ap, default_exp_id=DEFAULT_EXP_ID, default_n_jobs=3, default_n_startup=12)
    args = ap.parse_args()

    data = ph.load_pp_mmap(args.precomputed_dir)
    objective = zit_objective.make_zit_objective(
        args, data,
        model_class=BagZITboostRegressor,
        search=SEARCH,
        tau_range=TAU_PI_RANGE,
        use_unit_id=True,
        agg="sum",
    )
    out_dir = ph.resolve_out_dir(OUT_SUBDIR)
    db_path = ph.study_db_path(out_dir, args.user, args.exp_id)
    study = ph.make_study(args, db_path, study_meta={
        "model": "BagZITboostRegressor (bag, pearson phi)",
        "search_space": SEARCH,
        "tau_pi_range": TAU_PI_RANGE,
        "agg": "sum",
        "out_subdir": OUT_SUBDIR,
    })
    print(f"OUT_DIR={out_dir}\nDB={db_path}\nsearch HP={len(SEARCH)} + tau_pi=1")
    ph.run_optimize(study, objective, args)


if __name__ == "__main__":
    main()
