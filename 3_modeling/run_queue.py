#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_queue.py — 모델링 노트북 큐 실행기 (로컬 다대 PC 분산용).

여러 .ipynb 를 동시에 N개씩 실행하고, 하나 끝나면 큐에서 다음을 자동으로 투입한다.

핵심 규칙
- 원본 .ipynb 는 절대 수정하지 않는다. 같은 폴더에 임시 사본 `<stem>.__q__.ipynb` 를 만들어
  실행하고, 끝나면 지운다. (같은 폴더에 둬야 노트북 첫 셀의 `%run ../../setup.py` 같은
  상대경로가 그대로 동작한다.)
- 임시 사본에 한해 config 셀의 변수를 덮어쓸 수 있다 (N_TRIALS / N_JOBS / TIMEOUT_SEC / RESUME ...).
  `^<VAR> = <뭐든> [# 주석]` 형태의 줄만 RHS를 갈아끼우고 주석은 보존한다.
- Colab 전용 zip 셀(`<var> = shutil.make_archive('/content/...', ...)`)은 임시 사본에서 무력화
  (해당 줄을 `<var> = None` 으로 치환) → 로컬에서 `/content` 가 없어 터지는 것 방지.
- 산출물(4_output/...)은 노트북 코드가 EXP_ID로 경로를 만들어 저장하므로 임시 사본 이름과 무관하게
  정상 위치에 남는다.
- 로그: 노트북별 콘솔 로그(`_runlogs/<stem>.log`) + 실행본 노트북(`_runlogs/<stem>.executed.ipynb`)
  + 마스터 로그(`_runlogs/_master.log`). 마스터 로그/콘솔에 진행상황 tail 가능.

사용
    python run_queue.py config.json
    python run_queue.py --dry-run config.json        # 임시 사본 생성·덮어쓰기까지만, 실행 안 함 (변환 확인용)
    python run_queue.py nbA.ipynb nbB.ipynb ...       # 즉석 — 덮어쓰기 없이 원본 설정대로, concurrency 기본값

config.json
{
  "concurrency": 3,
  "log_dir": "_runlogs",
  "deadline": "2026-05-13 14:00",        // (선택) timeout_to_deadline 잡들이 이 시각에 멈추도록
  "deadline_buffer_min": 60,             // (선택) refit/저장 여유분. 기본 60
  "defaults": {"overrides": {"N_JOBS": 6}},  // (선택) 모든 잡 공통 override
  "jobs": [
    {"nb": "02_reg_single/lgbm_pphp.ipynb", "group": 1,
     "overrides": {"N_TRIALS": 1, "TIMEOUT_SEC": 3600}},
    {"nb": "01_zit/01_zit_only_pphp.ipynb", "group": 2,
     "timeout_to_deadline": true}
  ]
}
- "nb": config.json 파일 위치 기준 상대경로(또는 절대경로).
- "group": 정수(기본 1). 낮은 group 전부 끝난 뒤에 다음 group 시작 (단계 분리용).
- "overrides": 임시 사본에서 갈아끼울 변수들. defaults.overrides + job.overrides 머지(job 우선).
- "timeout_to_deadline": true 면 그 잡이 '시작되는 시점'에
  TIMEOUT_SEC = (deadline - now - buffer) 초 로 자동 계산해서 overrides 에 얹는다.

종료코드: 모든 잡 성공이면 0, 하나라도 실패면 1.
"""

import sys
import os
import re
import json
import time
import shutil
import argparse
import subprocess
import datetime as _dt
from concurrent.futures import ThreadPoolExecutor

import nbformat


# ---------------------------------------------------------------- 노트북 변환

_ASSIGN_RE_CACHE = {}


def _override_line_re(var):
    r = _ASSIGN_RE_CACHE.get(var)
    if r is None:
        # 줄 시작 + (들여쓰기) + VAR + = + RHS + (선택적 # 주석)
        r = re.compile(r"^(\s*" + re.escape(var) + r"\s*=\s*)([^#\n]*?)(\s*#.*)?$")
        _ASSIGN_RE_CACHE[var] = r
    return r


def _py_literal(v):
    # JSON 에서 온 값(bool/int/float/str) → 파이썬 소스 리터럴
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (int, float)):
        return repr(v)
    return repr(str(v))


_ZIP_LINE_RE = re.compile(
    r"^(\s*)(?:([A-Za-z_]\w*)\s*=\s*)?shutil\.make_archive\s*\(.*$"
)


def _transform_source(src, overrides):
    """한 코드 셀의 소스를 변환: override 적용 + colab zip 줄 무력화. (적용된 변수 set 반환)"""
    applied = set()
    out_lines = []
    for ln in src.splitlines():
        # 1) colab zip 줄 무력화
        mz = _ZIP_LINE_RE.match(ln)
        if mz and "/content" in ln:
            indent, var = mz.group(1), mz.group(2)
            if var:
                out_lines.append(f"{indent}{var} = None  # [run_queue] colab zip skipped")
            else:
                out_lines.append(f"{indent}pass  # [run_queue] colab zip skipped")
            continue
        # 2) override 적용
        replaced = False
        for var, val in overrides.items():
            m = _override_line_re(var).match(ln)
            if m:
                comment = m.group(3) or ""
                out_lines.append(f"{m.group(1)}{_py_literal(val)}{comment}")
                applied.add(var)
                replaced = True
                break
        if not replaced:
            out_lines.append(ln)
    return "\n".join(out_lines), applied


def make_temp_notebook(orig_path, overrides):
    """orig_path 와 같은 폴더에 <stem>.__q__.ipynb 임시 사본을 만들고 변환 적용.
    (임시 사본 경로, 적용된 override set, 못 찾은 override set) 반환."""
    nb = nbformat.read(orig_path, as_version=4)
    applied_all = set()
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        new_src, applied = _transform_source(cell["source"], overrides)
        cell["source"] = new_src
        applied_all |= applied
    d = os.path.dirname(os.path.abspath(orig_path))
    stem = os.path.splitext(os.path.basename(orig_path))[0]
    tmp = os.path.join(d, f"{stem}.__q__.ipynb")
    nbformat.write(nb, tmp)
    missing = set(overrides) - applied_all
    return tmp, applied_all, missing


# ---------------------------------------------------------------- 실행

def _now_str():
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _fmt_dur(sec):
    sec = int(sec)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


class MasterLog:
    def __init__(self, path):
        self.path = path
        self._fh = open(path, "a", encoding="utf-8")
        self.write(f"\n{'='*70}\n[{_now_str()}] run_queue 시작 (pid={os.getpid()})\n{'='*70}")

    def write(self, msg):
        line = msg if msg.startswith("\n") else f"[{_now_str()}] {msg}"
        print(line, flush=True)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()


def run_one(job, base_dir, log_dir, deadline_epoch, buffer_min, mlog, dry_run):
    """잡 1개: 임시 사본 생성 → nbconvert 실행 → 사본 삭제. 결과 dict 반환."""
    nb_rel = job["nb"]
    nb_path = nb_rel if os.path.isabs(nb_rel) else os.path.join(base_dir, nb_rel)
    nb_path = os.path.normpath(nb_path)
    stem = os.path.splitext(os.path.basename(nb_path))[0]
    nb_dir = os.path.dirname(nb_path)

    if not os.path.exists(nb_path):
        mlog.write(f"!! 노트북 없음: {nb_path} — 건너뜀")
        return {"nb": nb_rel, "ok": False, "reason": "missing notebook", "dur": 0}

    overrides = dict(job.get("overrides", {}))
    # timeout_to_deadline → 시작 시점에 TIMEOUT_SEC 계산
    if job.get("timeout_to_deadline"):
        if deadline_epoch is None:
            mlog.write(f"!! {stem}: timeout_to_deadline=true 인데 config.deadline 없음 — TIMEOUT 미설정")
        else:
            secs = int(deadline_epoch - time.time() - buffer_min * 60)
            if secs <= 60:
                mlog.write(f"!! {stem}: 데드라인까지 {secs}s 밖에 안 남음 — 건너뜀")
                return {"nb": nb_rel, "ok": False, "reason": "past deadline", "dur": 0}
            overrides["TIMEOUT_SEC"] = secs
            mlog.write(f">> {stem}: TIMEOUT_SEC={secs}s ({_fmt_dur(secs)}, 데드라인 {job.get('_deadline_str','?')} - {buffer_min}m)")

    tmp_path, applied, missing = make_temp_notebook(nb_path, overrides)
    if missing:
        mlog.write(f"   {stem}: override 못 찾은 변수 {sorted(missing)} (오타?)")
    if overrides:
        mlog.write(f">> {stem}: overrides 적용 {sorted(applied)} = " +
                   ", ".join(f"{k}={overrides[k]}" for k in sorted(applied)))

    log_path = os.path.join(log_dir, f"{stem}.log")

    if dry_run:
        mlog.write(f"[dry-run] {stem}: 임시 사본 생성됨 -> {tmp_path}  (실행 생략)")
        return {"nb": nb_rel, "ok": True, "reason": "dry-run", "dur": 0, "tmp": tmp_path}

    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook", "--execute",
        "--ExecutePreprocessor.timeout=-1",          # 셀 1개 실행시간 무제한 (HPO 셀이 길다)
        "--ExecutePreprocessor.allow_errors=False",  # 에러나면 즉시 실패로
        os.path.basename(tmp_path),                  # CWD=nb_dir 에서 실행
        "--output", f"{stem}.executed.ipynb",
        "--output-dir", os.path.abspath(log_dir),
    ]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    mlog.write(f"[START] {stem}  (log: {os.path.relpath(log_path, base_dir)})")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(f"[{_now_str()}] $ {' '.join(cmd)}  (cwd={nb_dir})\n")
        lf.flush()
        try:
            proc = subprocess.run(cmd, cwd=nb_dir, stdout=lf, stderr=subprocess.STDOUT, env=env)
            rc = proc.returncode
        except Exception as e:  # noqa: BLE001
            lf.write(f"\n[run_queue] subprocess 예외: {e!r}\n")
            rc = -999
    dur = time.time() - t0
    # 임시 사본 정리 (실행본은 _runlogs/ 에 따로 남으므로 사본은 지워도 됨)
    try:
        os.remove(tmp_path)
    except OSError:
        pass
    ok = (rc == 0)
    tag = "[DONE]" if ok else "[FAIL]"
    mlog.write(f"{tag} {stem}  rc={rc}  ({_fmt_dur(dur)})  log={os.path.relpath(log_path, base_dir)}")
    if not ok:
        # 로그 마지막 30줄을 마스터 로그에 덧붙여 즉시 원인 보이게
        try:
            with open(log_path, encoding="utf-8", errors="replace") as f:
                tail = f.readlines()[-30:]
            mlog.write(f"   -- {stem}.log 마지막 {len(tail)}줄 --\n" + "".join(tail).rstrip())
        except OSError:
            pass
    return {"nb": nb_rel, "ok": ok, "rc": rc, "dur": dur, "log": log_path}


def run_group(jobs, concurrency, base_dir, log_dir, deadline_epoch, buffer_min, mlog, dry_run):
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(run_one, j, base_dir, log_dir, deadline_epoch, buffer_min, mlog, dry_run)
                for j in jobs]
        for f in futs:
            results.append(f.result())
    return results


# ---------------------------------------------------------------- main

def main(argv):
    try:                                       # 콘솔 인코딩이 cp949여도 한글/기호 안 깨지게
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:                          # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="모델링 노트북 큐 실행기")
    ap.add_argument("config_or_nbs", nargs="+", help="config.json 1개  또는  .ipynb 여러 개")
    ap.add_argument("--dry-run", action="store_true", help="임시 사본 생성·변환까지만, 실행 안 함")
    ap.add_argument("--concurrency", type=int, default=None, help="동시 실행 개수 (config 값 덮어씀)")
    args = ap.parse_args(argv)

    first = args.config_or_nbs[0]
    if first.lower().endswith(".json") and len(args.config_or_nbs) == 1:
        cfg_path = os.path.abspath(first)
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        base_dir = os.path.dirname(cfg_path)
    else:
        # 즉석 모드: .ipynb 나열 → 기본 config
        base_dir = os.getcwd()
        cfg = {"jobs": [{"nb": os.path.abspath(p)} for p in args.config_or_nbs]}

    concurrency = args.concurrency or int(cfg.get("concurrency", 3))
    log_dir = os.path.join(base_dir, cfg.get("log_dir", "_runlogs"))
    os.makedirs(log_dir, exist_ok=True)
    buffer_min = int(cfg.get("deadline_buffer_min", 60))
    deadline_str = cfg.get("deadline")
    deadline_epoch = None
    if deadline_str:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"):
            try:
                deadline_epoch = _dt.datetime.strptime(deadline_str, fmt).timestamp()
                break
            except ValueError:
                continue
        if deadline_epoch is None:
            raise SystemExit(f"deadline 파싱 실패: {deadline_str!r} (예: '2026-05-13 14:00')")

    default_ov = dict(cfg.get("defaults", {}).get("overrides", {}))
    jobs = []
    for j in cfg["jobs"]:
        jj = dict(j)
        jj["overrides"] = {**default_ov, **dict(j.get("overrides", {}))}
        jj["group"] = int(j.get("group", 1))
        if deadline_str:
            jj["_deadline_str"] = deadline_str
        jobs.append(jj)
    jobs.sort(key=lambda x: x["group"])

    mlog = MasterLog(os.path.join(log_dir, "_master.log"))
    mlog.write(f"config={'(즉석)' if base_dir == os.getcwd() and not first.lower().endswith('.json') else first} | "
               f"jobs={len(jobs)} | concurrency={concurrency} | "
               f"deadline={deadline_str or '-'} (buffer {buffer_min}m) | log_dir={log_dir}")
    for j in jobs:
        ov = j["overrides"]
        td = " +timeout_to_deadline" if j.get("timeout_to_deadline") else ""
        mlog.write(f"  - g{j['group']} {j['nb']}  overrides={ov or '-'}{td}")

    t0 = time.time()
    all_results = []
    for grp in sorted({j["group"] for j in jobs}):
        gjobs = [j for j in jobs if j["group"] == grp]
        mlog.write(f"\n----- group {grp}: {len(gjobs)}개 시작 (동시 {concurrency}) -----")
        all_results += run_group(gjobs, concurrency, base_dir, log_dir,
                                 deadline_epoch, buffer_min, mlog, args.dry_run)

    n_ok = sum(1 for r in all_results if r["ok"])
    n_fail = len(all_results) - n_ok
    mlog.write(f"\n{'='*70}")
    mlog.write(f"전체 종료: {n_ok}/{len(all_results)} 성공, {n_fail} 실패  (총 {_fmt_dur(time.time()-t0)})")
    for r in all_results:
        mark = "OK  " if r["ok"] else "FAIL"
        extra = f" rc={r.get('rc')}" if not r["ok"] and "rc" in r else ""
        mlog.write(f"  [{mark}] {r['nb']}  ({_fmt_dur(r.get('dur', 0))}){extra}  {r.get('reason','')}")
    mlog.write(f"{'='*70}")
    mlog.close()
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
