"""
대시보드 Model 페이지용 정적 산출물 빌드 (1회 실행).

입력
----
- 4_output/final/zit_only/fold_models.pkl   (5-fold ZITboost)
- 4_output/final/zit_only/oof_unit.csv      (OOF 예측: ufs_serial, pred, health)
- 0_data/compet_xs_data.csv                 (원본 X — split, X0~X1086)

출력 → 5_dashboard/data/model/
- fold_metrics.json        : fold별 RMSE + 평균/표준편차
- feature_importance.csv   : feature, mu_gain, pi_gain, total_gain (5-fold 평균)
- psi.csv                  : feature, psi (train ↔ validation)
- var_compare.csv          : feature, cohens_d, p_value (위험 unit pred>p95 vs 정상)

산출물은 정적 (재실행 전까지 변경 X). FastAPI가 메모리 로드해 서빙.
재실행이 필요한 경우: 모델 재학습 / 원본 데이터 변경 / 위험 임계 변경.
"""
from __future__ import annotations
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
# utils.config (SEED, KEY_COL 등)와 final.modules import용 — 둘 다 PROJECT_ROOT 기준
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "3_modeling"))

# 학습 코드의 KFold 함수를 그대로 재사용 — fold split 재현성 보장
from final.modules.hpo import _make_unit_folds  # noqa: E402

SEED = 42
N_FOLDS = 5
RISK_PERCENTILE = 0.95  # pred 상위 5%를 위험군으로 정의

OUT_DIR = HERE / "data" / "model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

XS_PATH = PROJECT_ROOT / "0_data" / "compet_xs_data.csv"
OOF_PATH = PROJECT_ROOT / "4_output" / "final" / "zit_only" / "oof_unit.csv"
FM_PATH = PROJECT_ROOT / "4_output" / "final" / "zit_only" / "fold_models.pkl"


def log(msg: str) -> None:
    print(f"[build] {msg}", flush=True)


def calc_psi(a: np.ndarray, b: np.ndarray, n_bins: int = 10) -> float:
    """train 분포 a, val 분포 b 간 Population Stability Index.

    PSI < 0.1   안정
    0.1 ~ 0.25  보통 (모니터링)
    > 0.25      유의미한 분포 변화
    """
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0 or np.std(a) < 1e-12:
        return 0.0
    edges = np.unique(np.quantile(a, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return 0.0
    a_hist, _ = np.histogram(a, bins=edges)
    b_hist, _ = np.histogram(b, bins=edges)
    eps = 1e-6
    a_pct = (a_hist + eps) / (a_hist.sum() + eps * len(a_hist))
    b_pct = (b_hist + eps) / (b_hist.sum() + eps * len(b_hist))
    return float(np.sum((a_pct - b_pct) * np.log(a_pct / b_pct)))


def main() -> None:
    t0 = time.time()
    log(f"OUT_DIR: {OUT_DIR}")

    # ─── 1. fold별 RMSE ─────────────────────────────────────────
    log("[1/4] fold별 RMSE")
    oof = pd.read_csv(OOF_PATH)
    if not {"ufs_serial", "pred", "health"}.issubset(oof.columns):
        raise RuntimeError(f"oof_unit.csv 컬럼 부족: {oof.columns.tolist()}")
    train_units = np.array(sorted(oof["ufs_serial"].unique()))
    folds = _make_unit_folds(train_units, n_splits=N_FOLDS, seed=SEED)

    fold_rows = []
    for i, (_tr, vl) in enumerate(folds):
        sub = oof[oof["ufs_serial"].isin(vl)]
        if len(sub) == 0:
            continue
        rmse = float(np.sqrt(((sub["pred"] - sub["health"]) ** 2).mean()))
        fold_rows.append({"fold": i + 1, "rmse": rmse, "n_units": int(len(sub))})
        log(f"  fold{i+1}: RMSE={rmse:.5f} (n={len(sub):,})")

    fold_metrics = {
        "n_folds": N_FOLDS,
        "seed": SEED,
        "folds": fold_rows,
        "mean_rmse": float(np.mean([r["rmse"] for r in fold_rows])),
        "std_rmse": float(np.std([r["rmse"] for r in fold_rows])),
        "n_total_units": int(len(oof)),
    }
    with (OUT_DIR / "fold_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(fold_metrics, f, indent=2, ensure_ascii=False)
    log(
        f"  → fold_metrics.json (mean={fold_metrics['mean_rmse']:.5f}, "
        f"std={fold_metrics['std_rmse']:.5f})"
    )

    # ─── 2. feature importance (mu, pi 5-fold 평균) ─────────────
    log("[2/4] feature importance")
    with FM_PATH.open("rb") as f:
        fm = pickle.load(f)
    feature_names: list[str] = list(fm["feature_names"])
    n_feat = len(feature_names)
    mu_imp = np.zeros(n_feat)
    pi_imp = np.zeros(n_feat)
    for model in fm["fold_models"]:
        mu_imp += model.lgb_mu_.booster_.feature_importance(importance_type="gain")
        pi_imp += model.lgb_pi_.booster_.feature_importance(importance_type="gain")
    mu_imp /= len(fm["fold_models"])
    pi_imp /= len(fm["fold_models"])
    fi_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mu_gain": mu_imp,
                "pi_gain": pi_imp,
                "total_gain": mu_imp + pi_imp,
            }
        )
        .sort_values("total_gain", ascending=False)
        .reset_index(drop=True)
    )
    fi_df.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    log(
        f"  → feature_importance.csv (Top 5: "
        f"{', '.join(fi_df['feature'].head(5).tolist())})"
    )

    # ─── 3. PSI (train ↔ validation) ────────────────────────────
    log("[3/4] PSI (train ↔ validation)")
    # 원본 X csv의 실제 컬럼만 사용 (_missing 같은 derived feature는 제외)
    xs_header = pd.read_csv(XS_PATH, nrows=0)
    xs_x_cols = {c for c in xs_header.columns if c.startswith("X")}
    feat_in_xs = [c for c in feature_names if c in xs_x_cols]
    log(
        f"  feature 매칭 {len(feat_in_xs)}개 (모델 {n_feat}개 중 원본 X에 있는 것만, "
        f"derived {n_feat - len(feat_in_xs)}개 제외)"
    )
    use_cols = ["ufs_serial", "split"] + feat_in_xs
    log(f"  X 데이터 로드 (rows ~ 175k × cols {len(use_cols)})")
    xs = pd.read_csv(XS_PATH, usecols=use_cols)
    log(f"  unit 단위 mean 집계 ({len(xs):,} die)")
    xs_unit = (
        xs.groupby(["ufs_serial", "split"], as_index=False)[feat_in_xs].mean()
    )
    train_X = xs_unit[xs_unit["split"] == "train"][feat_in_xs]
    val_X = xs_unit[xs_unit["split"] == "validation"][feat_in_xs]
    log(f"  PSI 계산 (train {len(train_X):,} unit × val {len(val_X):,} unit)")

    psi_rows = []
    for c in feat_in_xs:
        psi_rows.append(
            {
                "feature": c,
                "psi": calc_psi(train_X[c].values, val_X[c].values),
            }
        )
    psi_df = (
        pd.DataFrame(psi_rows).sort_values("psi", ascending=False).reset_index(drop=True)
    )
    psi_df.to_csv(OUT_DIR / "psi.csv", index=False)
    log(
        f"  → psi.csv (Top 5: "
        f"{', '.join(psi_df['feature'].head(5).tolist())}, "
        f"max={psi_df['psi'].max():.3f})"
    )

    # ─── 4. 위험군 vs 정상군 t-test + Cohen's d ────────────────
    log("[4/4] 위험군 vs 정상군 변수 비교")
    threshold = float(oof["pred"].quantile(RISK_PERCENTILE))
    risk_units = set(oof[oof["pred"] > threshold]["ufs_serial"])
    log(
        f"  threshold pred>{threshold:.5f} (q={RISK_PERCENTILE:.2f}), "
        f"위험 unit {len(risk_units):,}"
    )

    # train split만 — val/test는 아직 health 미공개 가정
    train_unit_X = xs_unit[xs_unit["split"] == "train"].copy()
    is_risk = train_unit_X["ufs_serial"].isin(risk_units).values
    g_risk = train_unit_X.loc[is_risk, feat_in_xs]
    g_norm = train_unit_X.loc[~is_risk, feat_in_xs]
    log(f"  group 위험 {len(g_risk):,} / 정상 {len(g_norm):,}")

    var_rows = []
    for c in feat_in_xs:
        a = g_risk[c].values
        b = g_norm[c].values
        sa = np.nanstd(a, ddof=1) if len(a) > 1 else 0.0
        sb = np.nanstd(b, ddof=1) if len(b) > 1 else 0.0
        if sa + sb < 1e-12:
            continue
        try:
            tstat, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        except Exception:
            continue
        # Cohen's d (pooled std)
        pooled = np.sqrt(
            ((len(a) - 1) * sa ** 2 + (len(b) - 1) * sb ** 2)
            / max(len(a) + len(b) - 2, 1)
        )
        d = float((np.nanmean(a) - np.nanmean(b)) / pooled) if pooled > 0 else 0.0
        var_rows.append(
            {
                "feature": c,
                "cohens_d": d,
                "abs_d": abs(d),
                "p_value": float(p) if not np.isnan(p) else 1.0,
                "mean_risk": float(np.nanmean(a)),
                "mean_norm": float(np.nanmean(b)),
                "n_risk": int(len(a)),
                "n_norm": int(len(b)),
            }
        )
    var_df = (
        pd.DataFrame(var_rows)
        .sort_values("abs_d", ascending=False)
        .drop(columns="abs_d")
        .reset_index(drop=True)
    )
    var_df.to_csv(OUT_DIR / "var_compare.csv", index=False)
    log(
        f"  → var_compare.csv (Top 5: "
        f"{', '.join(var_df['feature'].head(5).tolist())})"
    )

    # ─── manifest ──────────────────────────────────────────────
    manifest = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": SEED,
        "n_folds": N_FOLDS,
        "risk_percentile": RISK_PERCENTILE,
        "n_features": n_feat,
        "n_features_in_xs": len(feat_in_xs),
        "outputs": {
            "fold_metrics.json": "fold별 RMSE",
            "feature_importance.csv": "5-fold 평균 LGBM gain (mu/pi)",
            "psi.csv": "train ↔ validation 분포 변화",
            "var_compare.csv": "위험 vs 정상 unit 변수 비교 (t-test + Cohen's d)",
        },
    }
    with (OUT_DIR / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    log(f"전체 완료 ({time.time() - t0:.1f}s) - 산출물 5종 (manifest 포함) -> {OUT_DIR}")


if __name__ == "__main__":
    main()
