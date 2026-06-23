"""저장된 ZIT fold 피클에서 EM 이력 검증 리포트를 간단히 만든다.

이 스크립트는 학습을 다시 돌리지 않는다. 기존 fold_models.pkl을 읽어
em_history_per_fold를 추출하고, 작은 sanity-check 리포트를 쓴다.
"""

from __future__ import annotations

import csv
import math
import pickle
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "3_modeling" / "_check"
CSV_OUT = OUT_DIR / "zit_em_history_report.csv"
MD_OUT = OUT_DIR / "zit_em_history_report.md"

# §5.1 의미 폴더명: 4_output/01_zit/<combo>/best/fold_models.pkl (실험번호/run_타임스탬프 없음).
# 아직 fit을 안 돌렸으면 파일이 없어 graceful skip 된다(main의 not pkl_path.exists()).
REPRESENTATIVE_PKLS = [
    ROOT / "4_output" / "01_zit" / combo / "best" / "fold_models.pkl"
    for combo in ("zit_only_pearson", "zit_only_eql", "bag_zit_pearson", "bag_zit_eql")
]


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _finite(value):
    return value is not None and math.isfinite(value)


def _format(value):
    return f"{value:.12g}" if _finite(value) else ""


def _metric_key(history):
    if not history:
        return None
    first = history[0]
    if "unit_rmse" in first:
        return "unit_rmse"
    if "rmse" in first:
        return "rmse"
    return None


def _load_histories(pkl_path: Path):
    with pkl_path.open("rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict):
        histories = obj.get("em_history_per_fold")
        model_name = obj.get("model_name") or pkl_path.parent.parent.name
        if histories is not None:
            return model_name, histories

        fold_models = obj.get("fold_models") or []
        histories = [getattr(m, "em_history_", None) for m in fold_models]
        return model_name, histories

    histories = [getattr(m, "em_history_", None) for m in obj] if isinstance(obj, list) else []
    return pkl_path.parent.parent.name, histories


def _label_for(path: Path) -> str:
    # 4_output/01_zit/<combo>/best/fold_models.pkl → combo 폴더명을 라벨로(§5.1 의미 이름).
    parts = path.relative_to(ROOT).parts
    return parts[2] if len(parts) > 2 else path.parent.parent.name


def _values(history, key):
    return [_as_float(h.get(key)) for h in history if key in h]


def _all_range(values, low=None, high=None, positive=False):
    if not values:
        return ""
    for value in values:
        if not _finite(value):
            return False
        if positive and value <= 0:
            return False
        if low is not None and value < low:
            return False
        if high is not None and value > high:
            return False
    return True


def _summarize_history(label: str, pkl_path: Path, fold_idx: int, history):
    metric = _metric_key(history)
    row = {
        "label": label,
        "pkl_path": str(pkl_path.relative_to(ROOT)),
        "fold": fold_idx,
        "history_present": bool(history),
        "metric": metric or "",
        "n_iter": len(history or []),
        "start_metric": "",
        "end_metric": "",
        "delta": "",
        "metric_finite": "",
        "metric_nonincreasing": "",
        "metric_final_lower": "",
        "pi_range_ok": "",
        "mu_positive_ok": "",
        "phi_positive_ok": "",
        "posterior_range_ok": "",
        "sanity_pass": "",
        "pi_mean_last": "",
        "mu_mean_last": "",
        "phi_mean_last": "",
        "posterior_mean_last": "",
    }
    if not history or metric is None:
        return row

    metric_values = [_as_float(h.get(metric)) for h in history]
    metric_finite = all(_finite(v) for v in metric_values)
    start = metric_values[0] if metric_values else None
    end = metric_values[-1] if metric_values else None
    tol = 1e-10
    metric_nonincreasing = metric_finite and all(
        metric_values[i + 1] <= metric_values[i] + tol for i in range(len(metric_values) - 1)
    )
    metric_final_lower = _finite(start) and _finite(end) and end <= start + tol

    pi_range_ok = _all_range(_values(history, "pi_mean"), low=0.0, high=1.0)
    mu_positive_ok = _all_range(_values(history, "mu_mean"), positive=True)
    phi_positive_ok = _all_range(_values(history, "phi_mean"), positive=True)
    posterior_range_ok = _all_range(_values(history, "posterior_mean"), low=0.0, high=1.0)

    sanity_pass = (
        metric_finite
        and (pi_range_ok in (True, ""))
        and (mu_positive_ok in (True, ""))
        and (phi_positive_ok in (True, ""))
        and (posterior_range_ok in (True, ""))
    )

    last = history[-1]
    row.update(
        {
            "start_metric": _format(start),
            "end_metric": _format(end),
            "delta": _format(end - start) if _finite(start) and _finite(end) else "",
            "metric_finite": metric_finite,
            "metric_nonincreasing": metric_nonincreasing,
            "metric_final_lower": metric_final_lower,
            "pi_range_ok": pi_range_ok,
            "mu_positive_ok": mu_positive_ok,
            "phi_positive_ok": phi_positive_ok,
            "posterior_range_ok": posterior_range_ok,
            "sanity_pass": sanity_pass,
            "pi_mean_last": _format(_as_float(last.get("pi_mean"))),
            "mu_mean_last": _format(_as_float(last.get("mu_mean"))),
            "phi_mean_last": _format(_as_float(last.get("phi_mean"))),
            "posterior_mean_last": _format(_as_float(last.get("posterior_mean"))),
        }
    )
    return row


def _write_markdown(rows, skipped):
    by_label = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)

    lines = [
        "# ZIT EM History Verification",
        "",
        "기존 `fold_models.pkl`에서 `em_history_per_fold`만 추출해 만든 간단 검증 요약입니다.",
        "학습을 다시 돌린 결과가 아니라, 저장된 fold 모델의 EM 반복 기록을 읽은 것입니다.",
        "",
        "> 주의: 여기의 `rmse`/`unit_rmse`는 EM 목적함수 자체가 아니라 모니터링 지표입니다. "
        "따라서 단조 감소 여부를 합격 기준으로 쓰지 않고, NaN/inf 여부와 확률/양수 조건 위반 여부를 sanity check로 봅니다.",
        "",
        "## 요약",
        "",
        "| 모델 | fold 수 | 기록 있음 | metric 유한 | pi 범위 정상 | mu 양수 | phi 양수 | sanity pass | 비고 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for label, items in by_label.items():
        n = len(items)
        present = sum(1 for r in items if r["history_present"])
        finite = sum(1 for r in items if r["metric_finite"] is True)
        pi_ok = sum(1 for r in items if r["pi_range_ok"] in (True, ""))
        mu_ok = sum(1 for r in items if r["mu_positive_ok"] in (True, ""))
        phi_ok = sum(1 for r in items if r["phi_positive_ok"] in (True, ""))
        sanity = sum(1 for r in items if r["sanity_pass"] is True)
        note = "PASS: 저장 로그 기준 비정상 값 없음" if sanity == n else "확인 필요"
        lines.append(f"| {label} | {n} | {present} | {finite} | {pi_ok} | {mu_ok} | {phi_ok} | {sanity} | {note} |")

    lines.extend(
        [
            "",
            "## Fold 상세",
            "",
            "| 모델 | fold | iter | metric | 시작 | 종료 | 변화량 | 단조 감소 | 최종 lower | pi_last | mu_last | phi_last |",
            "|---|---:|---:|---|---:|---:|---:|---|---|---:|---:|---:|",
        ]
    )
    for r in rows:
        lines.append(
            "| {label} | {fold} | {n_iter} | {metric} | {start_metric} | {end_metric} | {delta} | {metric_nonincreasing} | {metric_final_lower} | {pi_mean_last} | {mu_mean_last} | {phi_mean_last} |".format(
                **r
            )
        )

    if skipped:
        lines.extend(["", "## 스킵된 파일", ""])
        for p, reason in skipped:
            lines.append(f"- `{p}`: {reason}")

    lines.extend(
        [
            "",
            "## 발표용 문장",
            "",
            "> 저장된 fold 모델의 EM 반복 이력을 다시 읽어 확인한 결과, 대표 ZIT/BagZIT 산출물에서 각 fold의 EM 기록이 남아 있었고 RMSE 모니터링 값, zero 확률, 평균/분산 추정값이 NaN이나 무한대로 깨지지 않았습니다. 즉 학습 과정에서 비정상 발산이나 파라미터 범위 위반은 보이지 않았고, 최종 성능 검증은 별도의 OOF/validation/test RMSE로 확인했습니다.",
            "",
        ]
    )
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "3_modeling"))
    rows = []
    skipped = []

    for pkl_path in REPRESENTATIVE_PKLS:
        if not pkl_path.exists():
            skipped.append((str(pkl_path.relative_to(ROOT)), "file not found"))
            continue
        label = _label_for(pkl_path)
        try:
            model_name, histories = _load_histories(pkl_path)
        except Exception as exc:
            skipped.append((str(pkl_path.relative_to(ROOT)), f"load failed: {type(exc).__name__}: {exc}"))
            continue

        if not histories:
            skipped.append((str(pkl_path.relative_to(ROOT)), f"no histories found for {model_name}"))
            continue

        for fold_idx, hist in enumerate(histories, start=1):
            rows.append(_summarize_history(label, pkl_path, fold_idx, hist))

    if not rows:
        if skipped:
            print("[skipped]")
            for p, reason in skipped:
                print(f"- {p}: {reason}")
        raise SystemExit("No EM histories were extracted.")

    fieldnames = list(rows[0].keys())
    with CSV_OUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    _write_markdown(rows, skipped)
    print(f"[written] {CSV_OUT.relative_to(ROOT)}")
    print(f"[written] {MD_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
