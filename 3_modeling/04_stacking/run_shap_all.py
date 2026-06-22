"""
전체 모델 × PP그룹 SHAP 배치 추출 스크립트.

사용법:
    python 3_modeling/04_stacking/run_shap_all.py
    python 3_modeling/04_stacking/run_shap_all.py --dry-run      # 실행 목록만 출력
    python 3_modeling/04_stacking/run_shap_all.py --force        # npz 존재해도 재실행
    python 3_modeling/04_stacking/run_shap_all.py --workers 3    # 3개 병렬 실행 (기본 1)

제외 항목:
    - enet (선형 모델, pred_contrib 미지원)
    - ts_reg (03_two_stage/default/reg — Y>0 서브셋만 학습 → OOF SHAP sparse)
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STACKING_DIR = Path(__file__).resolve().parent
SCRIPT      = STACKING_DIR / "build_shap_features.py"
SHAP_CACHE  = STACKING_DIR / "shap_cache"
LOG_DIR     = STACKING_DIR / "_prev"
LOG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 작업 목록: (base_rel, zit_sub_model, ts_sub_model)
# zit_sub_model=None, ts_sub_model=None → 일반 reg/clf 모델
# ---------------------------------------------------------------------------
TASKS = []

# ── 02_reg_single (enet 제외: 선형, pred_contrib 미지원) ─────────────────────
REG_MODELS = ["lgbm", "xgb", "catboost", "et"]
for model in REG_MODELS:
    TASKS.append((f"02_reg_single/{model}", None, None))

# ── 01_zit 4조합 (ZIT 내부 lgb_pi_ / lgb_mu_ 둘 다 SHAP 추출) ────────────────
ZIT_COMBOS = ["zit_only_pearson", "zit_only_eql", "bag_zit_pearson", "bag_zit_eql"]
for combo in ZIT_COMBOS:
    for sub in ["pi", "mu"]:
        TASKS.append((f"01_zit/{combo}", sub, None))

# ── 03_two_stage/default/clf ───────────────────────────────────────────────
CLF_MODELS = ["lgbm", "xgb", "catboost", "et"]
for model in CLF_MODELS:
    TASKS.append((f"03_two_stage/default/clf/{model}", None, None))

# ── 03_two_stage/reverse (reg/clf 두 컴포넌트) ──────────────────────────────
for sub in ["reg", "clf"]:
    TASKS.append(("03_two_stage/reverse", None, sub))

# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------
def out_dir_of(base_rel: str, zit_sub: str | None, ts_sub: str | None) -> Path:
    tag = base_rel.replace("/", "__").replace("\\", "__")
    if zit_sub:
        tag = f"{tag}__{zit_sub}"
    if ts_sub:
        tag = f"{tag}__{ts_sub}"
    return SHAP_CACHE / tag


def is_done(base_rel: str, zit_sub: str | None, ts_sub: str | None) -> bool:
    return (out_dir_of(base_rel, zit_sub, ts_sub) / "die_shap.npz").exists()


def run_task(base_rel: str, zit_sub: str | None, ts_sub: str | None, log_path: Path,
             omp_threads: int = 0) -> int:
    cmd = [sys.executable, str(SCRIPT), "--base-rel", base_rel]
    if zit_sub:
        cmd += ["--zit-sub-model", zit_sub]
    if ts_sub:
        cmd += ["--ts-sub-model", ts_sub]
    sub_label = zit_sub or ts_sub or "-"
    label = f"{base_rel}  [{sub_label}]"
    print(f"\n{'='*60}", flush=True)
    print(f"  START  {label}", flush=True)
    print(f"  log → {log_path.name}", flush=True)
    print(f"{'='*60}", flush=True)
    t0 = time.time()
    env = os.environ.copy()
    if omp_threads > 0:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[var] = str(omp_threads)
    with open(log_path, "w", encoding="utf-8") as flog:
        proc = subprocess.run(
            cmd,
            stdout=flog,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
    elapsed = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
    print(f"  {status}  {elapsed:.0f}s  {label}", flush=True)
    return proc.returncode


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _worker(args_tuple) -> tuple[str, str | None, str | None, int]:
    """ProcessPoolExecutor용 래퍼. (base_rel, zit_sub, ts_sub, log_path, omp_threads) → (base_rel, zit_sub, ts_sub, rc)"""
    base_rel, zit_sub, ts_sub, log_path, omp_threads = args_tuple
    rc = run_task(base_rel, zit_sub, ts_sub, Path(log_path), omp_threads)
    return base_rel, zit_sub, ts_sub, rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run",   action="store_true", help="실행 목록만 출력")
    ap.add_argument("--force",     action="store_true", help="이미 완료된 항목도 재실행")
    ap.add_argument("--workers",   type=int, default=1,
                    help="병렬 프로세스 수 (기본 1=순차). RAM ~4GB×N 필요")
    args = ap.parse_args()
    skip_done = not args.force

    total   = len(TASKS)
    pending = [(r, z, t) for r, z, t in TASKS if not (skip_done and is_done(r, z, t))]
    done_n  = total - len(pending)

    # 병렬 실행 시 CPU를 worker 수로 나눠 스레드 경합 방지
    n_cpu = os.cpu_count() or 1
    omp_threads = max(1, n_cpu // args.workers) if args.workers > 1 else 0

    print(f"총 {total}개 작업  |  완료 {done_n}개 스킵  |  실행 예정 {len(pending)}개"
          f"  |  workers={args.workers}"
          + (f"  omp_threads/worker={omp_threads}" if args.workers > 1 else ""))

    if args.dry_run:
        for i, (base_rel, zit_sub, ts_sub) in enumerate(pending, 1):
            marker = "[done]" if is_done(base_rel, zit_sub, ts_sub) else "[ RUN ]"
            sub_str = zit_sub or ts_sub or "-"
            print(f"  {i:3d}. {marker}  {base_rel}  ({sub_str})")
        return

    def make_log_path(base_rel, zit_sub, ts_sub):
        sub_str = zit_sub or ts_sub or ""
        name = base_rel.replace("/", "__") + (f"__{sub_str}" if sub_str else "") + ".log"
        return LOG_DIR / name

    fails = []

    if args.workers <= 1:
        # 순차 실행
        for i, (base_rel, zit_sub, ts_sub) in enumerate(pending, 1):
            log_path = make_log_path(base_rel, zit_sub, ts_sub)
            print(f"\n[{i}/{len(pending)}]", flush=True)
            rc = run_task(base_rel, zit_sub, ts_sub, log_path)
            if rc != 0:
                fails.append((base_rel, zit_sub, ts_sub, rc))
    else:
        # 병렬 실행
        work_items = [
            (base_rel, zit_sub, ts_sub,
             str(make_log_path(base_rel, zit_sub, ts_sub)),
             omp_threads)
            for base_rel, zit_sub, ts_sub in pending
        ]
        completed = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_worker, item): item for item in work_items}
            for fut in as_completed(futs):
                completed += 1
                base_rel, zit_sub, ts_sub, rc = fut.result()
                sub_str = zit_sub or ts_sub or "-"
                status = "OK" if rc == 0 else f"FAIL(rc={rc})"
                print(f"\n[{completed}/{len(pending)} done]  {status}  {base_rel}  [{sub_str}]",
                      flush=True)
                if rc != 0:
                    fails.append((base_rel, zit_sub, ts_sub, rc))

    print("\n" + "="*60)
    print(f"완료: {len(pending) - len(fails)}개  실패: {len(fails)}개")
    if fails:
        print("실패 목록:")
        for base_rel, zit_sub, ts_sub, rc in fails:
            sub_str = zit_sub or ts_sub or "-"
            print(f"  rc={rc}  {base_rel}  [{sub_str}]")
    print("="*60)


if __name__ == "__main__":
    main()
