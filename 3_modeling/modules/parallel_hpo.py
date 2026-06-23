"""
modules/parallel_hpo.py — 전 트랙(zit / reg / ts) 공용 병렬 HPO 하네스.

배경 (refactor_strategy.md §1.1): zit/reg/ts(clf)의 기존 병렬 워커를 대조하니 하네스
(인자 파싱·create_study·mmap 로드·end-at 최적화 루프·main)와 unit-CV objective 골격이
사실상 byte 단위로 동일했다. 트랙 간 차이는 fold 루프 안 '타깃 준비 + 모델 평가 + die→unit
집계'뿐. 그 공용 보일러플레이트를 이 모듈 한 곳에 모은다.

설계 원칙
---------
- 이 모듈은 **하네스만** 제공한다. objective 본문(명시적 fold 루프)은 각 트랙 hpo.py에
  그대로 둔다 — callback 과추상화 없이 트랙 고유 로직(tau_pi / proba·y_pos_const / Y>0
  마스크)이 그 파일에 보이게 한다.
- N개 독립 프로세스가 하나의 Optuna SQLite study(RDBStorage)를 공유하고, pp.npy를
  mmap(read-only)으로 RAM 공유한다. (precompute_pp.py가 1회 생성)
- 워커는 Optuna DB 기록만. refit·후처리·산출물 저장은 fit.ipynb에서 별도(modules.hpo).
- 모든 워커가 넓은 탐색 공간을 **처음부터** 탐색한다.

사용 패턴 (각 트랙 hpo.py)
-------------------------
    from modules import parallel_hpo as ph

    SEARCH = {...}
    def build_objective(args, data):
        x, uid_die, y_unit_s, y_die, _ = data
        ...                                    # 명시적 fold 루프 (트랙 고유)
        return objective

    def main():
        ap = argparse.ArgumentParser()
        ph.add_common_args(ap, default_exp_id="zit_only_pearson")
        # 트랙 고유 인자(--model 등)는 여기서 추가
        args = ap.parse_args()
        data = ph.load_pp_mmap(args.precomputed_dir)
        out_dir = ph.resolve_out_dir("01_zit/zit_only_pearson")
        db_path = out_dir / f"optuna_{args.user}_{args.exp_id}.db"
        study = ph.make_study(args, db_path, study_meta={"search_space": SEARCH})
        ph.run_optimize(study, build_objective(args, data), args)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "setup.py").exists() and (p / "utils").exists():
            return p
    raise RuntimeError(f"Project root not found from {start}")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
for _p in [PROJECT_ROOT, PROJECT_ROOT / "2_preprocessing", PROJECT_ROOT / "3_modeling"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import optuna  # noqa: E402
from optuna.pruners import MedianPruner  # noqa: E402
from optuna.samplers import TPESampler  # noqa: E402
from optuna.storages import RDBStorage, RetryFailedTrialCallback  # noqa: E402

from utils.config import OUTPUT_DIR  # noqa: E402

# 전 트랙이 공유하는 단일 precompute 디렉토리 (precompute_pp.py --name zit_pp).
DEFAULT_PRECOMPUTED_DIR = PROJECT_ROOT / "0_data" / "precomputed" / "zit_pp"


# ------------------------------------------------------------
# 1) 공용 워커 인자
# ------------------------------------------------------------
def add_common_args(
    parser,
    *,
    default_exp_id: str = "exp",
    default_user: str = "jh",
    default_n_trials: int = 2000,
    default_n_jobs: int = 4,
    default_n_startup: int = 80,
):
    """모든 트랙 hpo.py가 공유하는 16개 워커 인자를 parser에 추가.

    트랙 고유 인자(--model, --variant 등)는 호출 측에서 별도로 add_argument 한다.
    """
    parser.add_argument("--exp-id", default=default_exp_id)
    parser.add_argument("--user", default=default_user)
    parser.add_argument("--worker-id", default=f"pid{os.getpid()}")
    parser.add_argument("--n-trials", type=int, default=default_n_trials)
    parser.add_argument("--timeout-hours", type=float, default=120.0)
    parser.add_argument(
        "--end-at",
        default=None,
        help=(
            "Absolute wall-clock end datetime (ISO), e.g. '2026-06-01T05:00'. When set, "
            "overrides --timeout-hours. Pass the same value to every worker so all stop together."
        ),
    )
    parser.add_argument(
        "--precomputed-dir",
        default=str(DEFAULT_PRECOMPUTED_DIR),
        help="Directory with precomputed pp.npy/units.npy (precompute_pp.py output).",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=default_n_jobs)
    parser.add_argument("--n-startup-trials", type=int, default=default_n_startup)
    parser.add_argument("--db-timeout", type=float, default=120.0)
    parser.add_argument("--heartbeat-interval", type=int, default=300)
    parser.add_argument("--grace-period", type=int, default=1800)
    parser.add_argument("--max-retry", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-clip-y-extreme", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser


# ------------------------------------------------------------
# 2) 공용 Optuna study (RDBStorage 공유)
# ------------------------------------------------------------
def make_study(args, db_path, study_meta: dict | None = None) -> optuna.Study:
    """N개 워커가 공유하는 SQLite Optuna study 생성/로드.

    TPE(multivariate, group) + MedianPruner. `--no-resume` 아니면 load_if_exists.
    study_meta는 user_attrs로 저장(재현 메타).
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    storage = RDBStorage(
        url=f"sqlite:///{db_path.as_posix()}",
        engine_kwargs={"connect_args": {"timeout": args.db_timeout}},
        heartbeat_interval=args.heartbeat_interval,
        grace_period=args.grace_period,
        failed_trial_callback=RetryFailedTrialCallback(max_retry=args.max_retry),
    )
    sampler = TPESampler(
        seed=None,
        multivariate=True,
        group=True,
        n_startup_trials=args.n_startup_trials,
    )
    pruner = MedianPruner(n_startup_trials=args.n_startup_trials, n_warmup_steps=2)
    study = optuna.create_study(
        study_name=args.exp_id,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        load_if_exists=not args.no_resume,
    )

    meta = {
        "exp_id": args.exp_id,
        "user": args.user,
        "worker_id_last_writer": args.worker_id,
        "n_trials_per_worker": args.n_trials,
        "n_folds": args.n_folds,
        "n_jobs": args.n_jobs,
        "sampler": "TPE seed=None multivariate group",
        "pruner": f"MedianPruner n_startup={args.n_startup_trials} n_warmup=2",
        "clip_y_extreme": not args.no_clip_y_extreme,
    }
    if study_meta:
        meta.update(study_meta)
    for k, v in meta.items():
        study.set_user_attr(k, str(v))
    return study


# ------------------------------------------------------------
# 3) 공용 pp.npy mmap 로더 (precompute_pp.py 산출물)
# ------------------------------------------------------------
def load_pp_mmap(precomputed_dir):
    """precompute한 단일 numeric 행렬을 mmap-load.

    Returns (x_train, uid_train_die, y_train_unit_s, y_train_die, feat_cols) — 워커
    self-preprocess 경로(zit_pp.load_preprocessed_data)와 동일 shape/semantics.
    pp.npy는 read-only mmap이라 모든 워커 프로세스가 OS page cache로 행렬 1벌을 공유한다.

    clf의 y_die_bin/y_pos_const, ts의 Y>0 subset은 여기서 받은 y_train_die / y_train_unit_s로
    각 트랙 hpo.py에서 파생한다 (별도 전처리 불필요).
    """
    d = Path(precomputed_dir)
    pp_path = d / "pp.npy"
    if not pp_path.exists():
        raise FileNotFoundError(
            f"Precomputed matrix not found: {pp_path}\n"
            f"  Run the precompute step first:\n"
            f"    python 3_modeling/01_zit/precompute_pp.py"
        )
    manifest = {}
    mpath = d / "manifest.json"
    if mpath.exists():
        try:
            manifest = json.loads(mpath.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    pp = np.load(pp_path, mmap_mode="r")               # (n_dies, F+2) read-only mmap (shared)
    F = int(manifest.get("n_features", pp.shape[1] - 2))
    units = np.load(d / "units.npy")                    # (n_units, 2)
    feat = json.loads((d / "feat_cols.json").read_text(encoding="utf-8"))

    x_train = pp[:, :F]                                 # mmap view, no copy
    uid_train_die = pp[:, F].astype(np.int64)           # die-order unit codes
    y_train_die = np.asarray(pp[:, F + 1], dtype=np.float64)
    y_train_unit_s = pd.Series(
        units[:, 1].astype(np.float64),
        index=units[:, 0].astype(np.int64),
    )
    print(f"[precomputed] mmap pp={pp.shape} F={F} units={len(y_train_unit_s):,} from {d}")
    print(f"  X_train={x_train.shape} (mmap view), n_features={len(feat)}")
    return x_train, uid_train_die, y_train_unit_s, y_train_die, feat


# ------------------------------------------------------------
# 4) 출력 경로 (§5.1 의미 폴더명 — 실험번호 없음)
#    producer(hpo.py)·fit.ipynb·combine이 모두 같은 규약을 쓰도록 경로 조립을
#    이 한 곳에 둔다. discovery.py는 4_output을 rglob 스캔하므로 경로를 직접
#    조립하지 않지만(=쓰는 경로/스캔 경로가 어긋날 수 없음), 폴더명 규약은 여기서 정한다.
# ------------------------------------------------------------
def model_out_dir(*parts, make: bool = True) -> Path:
    """§5.1 의미 폴더 경로(4_output/<track>/<model>/). 실험번호 금지 — 폴더명이 곧 stacking 태그.

    가변 인자라 트랙/모델을 조각으로 주거나 한 문자열로 줘도 동일하게 동작한다::

        model_out_dir("01_zit", "zit_only_pearson")   # 4_output/01_zit/zit_only_pearson
        model_out_dir("01_zit/zit_only_pearson")       # 동일
        model_out_dir("03_two_stage", "default", "clf", "lgbm")
    """
    sub = "/".join(s for s in ("/".join(str(p) for p in parts)).split("/") if s)
    out_dir = Path(OUTPUT_DIR) / sub
    if make:
        out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def resolve_out_dir(out_subdir, *, make: bool = True) -> Path:
    """단일 문자열 형태의 §5.1 경로 헬퍼 (model_out_dir의 별칭, 기존 호출부 호환)."""
    return model_out_dir(out_subdir, make=make)


def study_db_path(out_dir, user, exp_id) -> Path:
    """Optuna DB 파일명 규약 단일 소스: ``<out_dir>/optuna_{user}_{exp_id}.db``.

    hpo.py(쓰기)와 fit.ipynb(읽기, source DB)가 같은 파일을 가리키도록 파일명 조립을
    한 곳에 둔다. out_dir은 model_out_dir/resolve_out_dir이 돌려준 Path를 그대로 넘긴다.
    """
    return Path(out_dir) / f"optuna_{user}_{exp_id}.db"


# ------------------------------------------------------------
# 5) 공용 최적화 루프 (end-at / timeout 환산 + study.optimize)
# ------------------------------------------------------------
def run_optimize(study, objective, args):
    """end-at(절대 시각) 또는 timeout-hours로 환산한 시간 예산만큼 study.optimize.

    각 워커는 optuna n_jobs=1(프로세스 간 병렬이 본체). 모델 자체 n_jobs는 args.n_jobs로
    objective 안에서 설정한다.
    """
    logging.getLogger("lightgbm").setLevel(logging.ERROR)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"study={study.study_name}, existing_trials={len(study.trials)}")
    if args.n_trials <= 0:
        print("[skip optimize] n_trials <= 0")
        return

    if args.end_at:
        end_dt = datetime.fromisoformat(args.end_at)
        remaining = (end_dt - datetime.now()).total_seconds()
        if remaining <= 0:
            print(f"[end-at] {end_dt.isoformat()} already passed — nothing to do")
            return
        timeout = remaining
        print(f"[end-at] target={end_dt.isoformat()} -> timeout={timeout:.0f}s ({timeout/3600:.2f}h)")
    else:
        timeout = None if args.timeout_hours <= 0 else args.timeout_hours * 3600

    t0 = time.time()
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=timeout,
        n_jobs=1,
        show_progress_bar=args.progress,
    )
    elapsed = time.time() - t0
    print(f"[HPO done] worker={args.worker_id}, elapsed={elapsed:.0f}s, total_trials={len(study.trials)}")
    try:
        print(f"best_value={study.best_value:.9f}, best_trial={study.best_trial.number}")
    except ValueError:
        print("best_value unavailable: no completed trials")
