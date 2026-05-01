"""
대시보드 백엔드 (FastAPI).

PI 운영 관점 status 분류:
  - today     : 오늘 검사 (가장 최근 lot, days_back=0)
  - pending   : 최근 검사 (days_back 1~3, Y 측정 대기)
  - completed : 과거 batch (days_back ≥ 4, Y 도착 완료)

날짜 할당 — X1086(YYYYMMDD) 기반:
  - 원본 X1086은 lot 단위로 거의 일정 (28 lots × ~5일 분포)
  - 각 lot에 X1086 mode를 lot date로 정의
  - X1086 큰 lot순으로 today, today-1, ..., today-27까지 매핑
  - 한 lot 안의 모든 unit은 동일 inspected_date 가짐

Health 마스킹 (사용자 시나리오: "Y가 안 도착한 척"):
  - val/test split  → 항상 health=NaN (Y 자체를 모르는 held-out 데이터)
  - today/pending status → split 무관 모두 health=NaN (검사 직후, Y 측정 진행 중)
  → past area(completed)의 train unit만 health 노출됨
"""
from __future__ import annotations

import json
import os
from datetime import date, timedelta
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# ─── 경로 ──────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
XS_CSV_PATH = os.path.join(PROJECT_ROOT, "0_data", "compet_xs_data.csv")


def _load_lot_dates(today: date) -> dict[str, str]:
    """X1086(YYYYMMDD)을 lot 단위 1개 date로 압축한 뒤 today-relative로 remap.

    가장 최근 X1086 lot → today, 다음 → today-1, ..., 28번째 → today-27.
    """
    xs = pd.read_csv(XS_CSV_PATH, usecols=["ufs_serial", "run_wf_xy", "X1086"])
    xs["run_id"] = xs["run_wf_xy"].str.split("_").str[0]
    # 일부 lot은 X1086이 2개 일자에 걸쳐있어 mode로 압축 (가장 흔한 일자)
    lot_x1086 = (
        xs.dropna(subset=["X1086"])
        .groupby("run_id")["X1086"]
        .agg(lambda s: s.mode().iloc[0])
        .sort_values(ascending=False)
    )
    return {
        lot: (today - timedelta(days=i)).isoformat()
        for i, lot in enumerate(lot_x1086.index)
    }


def _assign_status_and_date(df: pd.DataFrame, lot_dates: dict[str, str]) -> pd.DataFrame:
    """X1086 기반 lot date + date 기반 status 적용 + val/test health 마스킹."""
    today = date.today()
    df = df.copy()

    # 1) inspected_date — lot date 매핑 (lot dict에 없으면 fallback = 가장 오래된 날짜)
    fallback = (today - timedelta(days=len(lot_dates))).isoformat()
    df["inspected_date"] = df["run_id"].map(lambda l: lot_dates.get(l, fallback))

    # 2) status — date 기반
    days_back = (
        pd.to_datetime(today) - pd.to_datetime(df["inspected_date"])
    ).dt.days.astype(int)
    df["status"] = np.where(
        days_back == 0,
        "today",
        np.where(days_back <= 3, "pending", "completed"),
    )

    # 3) Y 시나리오 마스킹
    #    (a) val/test split → 항상 NaN (Y 자체를 모르는 데이터)
    #    (b) today/pending status → split 무관 NaN (검사 직후 Y 측정 진행 중)
    if "split" in df.columns:
        future_splits = {"val", "validation", "test"}
        df.loc[df["split"].isin(future_splits), "health"] = np.nan
    df.loc[df["status"].isin(["today", "pending"]), "health"] = np.nan

    return df


# ─── 데이터 로드 (앱 시작 시 1회) ────────────────────────
class DataStore:
    def __init__(self):
        print(f"[DataStore] 로드 시작: {DATA_DIR}")
        self.die = pd.read_parquet(os.path.join(DATA_DIR, "die_predictions.parquet"))
        unit = pd.read_parquet(os.path.join(DATA_DIR, "unit_predictions.parquet"))
        today = date.today()
        self.lot_dates = _load_lot_dates(today)
        print(f"[DataStore] lot dates: {len(self.lot_dates)}개 lot, range {today.isoformat()} ~ {(today - timedelta(days=len(self.lot_dates)-1)).isoformat()}")
        self.unit = _assign_status_and_date(unit, self.lot_dates)

        # wafer_summary는 prepare_data.py에서 (wafer_key, run_id, wafer_no, split)로 집계되어
        # 같은 wafer가 여러 split에 걸치면 행이 중복된다(React duplicate key의 근본 원인).
        # 여기서 unit으로부터 wafer_key 단위로 재집계해 항상 unique하게 만든다.
        wafer_recompute = (
            self.unit.groupby(["wafer_key", "run_id", "wafer_no"], as_index=False)
            .agg(
                n_units=("ufs_serial", "count"),
                mean_pred=("pred", "mean"),
                max_pred=("pred", "max"),
                n_risk=("is_risk", "sum"),
                mean_health=("health", "mean"),
            )
        )
        wafer_recompute["risk_ratio"] = (
            wafer_recompute["n_risk"] / wafer_recompute["n_units"].clip(lower=1)
        )
        # wafer_key 단위 status (해당 wafer의 unit 다수결)
        wafer_status = (
            self.unit.groupby("wafer_key")["status"]
            .agg(lambda s: s.mode().iat[0] if len(s) else "completed")
            .reset_index()
        )
        self.wafer = wafer_recompute.merge(wafer_status, on="wafer_key", how="left")
        self.wafer["status"] = self.wafer["status"].fillna("completed")
        with open(os.path.join(DATA_DIR, "overview_stats.json"), encoding="utf-8") as f:
            self.overview = json.load(f)

        # 조회 성능을 위해 인덱스 설정
        self.unit_indexed = self.unit.set_index("ufs_serial", drop=False)
        self.die_by_unit = {k: g for k, g in self.die.groupby("ufs_serial")}
        self.die_by_wafer = {k: g for k, g in self.die.groupby("wafer_key")}

        # 모델 산출물 (build_model_artifacts.py가 생성)
        model_dir = os.path.join(DATA_DIR, "model")
        self.model_artifacts = {}
        if os.path.exists(model_dir):
            try:
                with open(os.path.join(model_dir, "fold_metrics.json"), encoding="utf-8") as f:
                    self.model_artifacts["fold_metrics"] = json.load(f)
                self.model_artifacts["feature_importance"] = pd.read_csv(
                    os.path.join(model_dir, "feature_importance.csv")
                )
                self.model_artifacts["psi"] = pd.read_csv(
                    os.path.join(model_dir, "psi.csv")
                )
                self.model_artifacts["var_compare"] = pd.read_csv(
                    os.path.join(model_dir, "var_compare.csv")
                )
                print(f"  model artifacts: {list(self.model_artifacts.keys())}")
            except Exception as e:
                print(f"  ⚠ model artifacts 로드 실패: {e}")
        else:
            print(f"  ⚠ {model_dir} 없음 — build_model_artifacts.py 실행 필요")

        print(f"  die  : {len(self.die):,} rows")
        print(f"  unit : {len(self.unit):,} rows")
        print(f"  wafer: {len(self.wafer):,} rows")
        print(f"  status 분배: {self.unit['status'].value_counts().to_dict()}")


store: Optional[DataStore] = None


# ─── FastAPI 앱 ────────────────────────────────────────────
app = FastAPI(title="Wafer Health Dashboard API", version="0.2.0")

# React dev server (5173) 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    global store
    store = DataStore()


# ─── 유틸 ──────────────────────────────────────────────
def _filter_status(df: pd.DataFrame, status: Optional[str]) -> pd.DataFrame:
    """status 필터: 'all' 또는 None이면 전체, 그 외에는 컬럼 매칭."""
    if not status or status == "all":
        return df
    if status not in ("completed", "pending", "today"):
        raise HTTPException(400, f"invalid status: {status}")
    return df[df["status"] == status]


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


# ─── 엔드포인트 ────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "data_loaded": store is not None}


@app.get("/api/overview")
def overview():
    """status별 KPI."""
    out = {
        "risk_top_ratio": store.overview.get("risk_top_ratio", 0.05),
        "totals": {
            "n_units": int(len(store.unit)),
            "n_dies": int(len(store.die)),
            "n_wafers": int(store.unit["wafer_key"].nunique()),
        },
        "statuses": {},
    }
    for status_name in ("completed", "pending", "today"):
        sub = store.unit[store.unit["status"] == status_name]
        if len(sub) == 0:
            out["statuses"][status_name] = {
                "n_units": 0,
                "n_risk": 0,
                "risk_ratio": 0.0,
                "pred_mean": 0.0,
                "pred_max": 0.0,
                "health_zero_ratio": 0.0,
            }
            continue
        entry = {
            "n_units": int(len(sub)),
            "n_risk": int(sub["is_risk"].sum()),
            "risk_ratio": float(sub["is_risk"].mean()),
            "pred_mean": float(sub["pred"].mean()),
            "pred_max": float(sub["pred"].max()),
            "health_zero_ratio": float((sub["health"].fillna(0) == 0).mean()),
        }
        # completed만 health 실측 → RMSE 계산 가능
        if status_name == "completed":
            mask = sub["health"].notna()
            if mask.sum() > 0:
                err = sub.loc[mask, "health"].values - sub.loc[mask, "pred"].values
                entry["rmse"] = float(np.sqrt(np.mean(err ** 2)))
        out["statuses"][status_name] = entry
    return out


@app.get("/api/wafer-grid")
@lru_cache(maxsize=1)
def wafer_grid():
    """모든 wafer의 die 좌표 union → wafer 외형 마스크."""
    uniq = store.die[["die_x", "die_y"]].drop_duplicates().sort_values(["die_y", "die_x"])
    bx = (int(store.die["die_x"].min()), int(store.die["die_x"].max()))
    by = (int(store.die["die_y"].min()), int(store.die["die_y"].max()))
    return {
        "bounds": {"x_min": bx[0], "x_max": bx[1], "y_min": by[0], "y_max": by[1]},
        "width": bx[1] - bx[0] + 1,
        "height": by[1] - by[0] + 1,
        "mask": uniq.values.tolist(),
    }


@app.get("/api/wafers")
def list_wafers(
    status: Optional[str] = Query(None, description="completed/pending/today/all"),
    sort: str = Query("risk_ratio", description="risk_ratio | mean_pred | n_risk"),
    limit: int = Query(100, ge=1, le=5000),
):
    df = _filter_status(store.wafer, status)
    if sort not in df.columns:
        raise HTTPException(400, f"invalid sort key: {sort}")
    df = df.sort_values(sort, ascending=False).head(limit)
    return {"count": len(df), "items": _records(df)}


@app.get("/api/wafers/{wafer_key}")
def wafer_detail(wafer_key: str):
    if wafer_key not in store.die_by_wafer:
        raise HTTPException(404, f"wafer not found: {wafer_key}")
    dies = store.die_by_wafer[wafer_key]
    units = store.unit[store.unit["wafer_key"] == wafer_key]
    summary_row = store.wafer[store.wafer["wafer_key"] == wafer_key]
    summary = _records(summary_row)[0] if len(summary_row) else None
    return {
        "wafer_key": wafer_key,
        "summary": summary,
        "dies": _records(dies[["ufs_serial", "run_wf_xy", "die_x", "die_y", "pi", "mu", "pred", "health", "split"]]),
        "units": _records(units[["ufs_serial", "pred", "health", "is_risk", "status"]]),
    }


@app.get("/api/units")
def list_units(
    status: Optional[str] = Query(None),
    risk_only: bool = Query(False),
    sort: str = Query("pred"),
    order: str = Query("desc", pattern="^(asc|desc)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
):
    df = _filter_status(store.unit, status)
    if risk_only:
        df = df[df["is_risk"]]
    if sort not in df.columns:
        raise HTTPException(400, f"invalid sort key: {sort}")
    df = df.sort_values(sort, ascending=(order == "asc"))
    total = len(df)
    start = (page - 1) * page_size
    page_df = df.iloc[start : start + page_size]
    return {"total": total, "page": page, "page_size": page_size, "items": _records(page_df)}


@app.get("/api/units/{ufs_serial}")
def unit_detail(ufs_serial: str):
    if ufs_serial not in store.unit_indexed.index:
        raise HTTPException(404, f"unit not found: {ufs_serial}")
    unit_row = store.unit_indexed.loc[ufs_serial]
    if isinstance(unit_row, pd.DataFrame):
        unit_row = unit_row.iloc[0]
    dies = store.die_by_unit.get(ufs_serial, pd.DataFrame())
    return {
        "unit": _records(unit_row.to_frame().T)[0],
        "dies": _records(dies[["run_wf_xy", "die_x", "die_y", "pi", "one_minus_pi", "mu", "pred"]]),
    }


@app.get("/api/triage")
def triage(
    status: str = Query("today", description="completed/pending/today/all"),
    top_units: int = Query(20, ge=1, le=200),
    top_wafers: int = Query(10, ge=1, le=50),
    unit_cost: float = Query(1000.0),
):
    """status별 위험 트리아지 + 색상 스케일."""
    unit = _filter_status(store.unit, status)
    if len(unit) == 0:
        # 데이터가 없는 status도 빈 응답 반환 (404 대신 — 클라이언트 토글 안전)
        return {
            "status": status,
            "summary": {
                "n_units": 0, "n_risk": 0, "risk_ratio": 0.0,
                "estimated_loss": 0.0, "unit_cost": unit_cost, "rmse": 0.0,
            },
            "scale": {
                "pred_min": float(store.die["pred"].min()),
                "pred_max": float(store.die["pred"].max()),
                "risk_threshold": float(store.unit["pred"].quantile(0.95)),
            },
            "top_wafers": [],
            "top_units": [],
        }
    risk = unit[unit["is_risk"]].sort_values("pred", ascending=False)
    wafer_filtered = _filter_status(store.wafer, status)
    wafers_top = (
        wafer_filtered.sort_values(["risk_ratio", "n_risk"], ascending=False).head(top_wafers)
    )
    rmse = 0.0
    if status == "completed" or status == "all":
        sub = unit[unit["health"].notna()]
        if len(sub) > 0:
            err = sub["health"].values - sub["pred"].values
            rmse = float(np.sqrt(np.mean(err ** 2)))
    return {
        "status": status,
        "summary": {
            "n_units": int(len(unit)),
            "n_risk": int(len(risk)),
            "risk_ratio": float(len(risk) / len(unit)),
            "estimated_loss": float(len(risk) * unit_cost),
            "unit_cost": unit_cost,
            "rmse": rmse,
        },
        "scale": {
            "pred_min": float(store.die["pred"].min()),
            "pred_max": float(store.die["pred"].max()),
            "risk_threshold": float(store.unit["pred"].quantile(0.95)),
        },
        "top_wafers": _records(wafers_top),
        "top_units": _records(risk.head(top_units)),
    }


@app.get("/api/lots")
def lots(
    status: str = Query("today"),
    sort: str = Query("risk_ratio"),
    limit: int = Query(50, ge=1, le=500),
):
    unit = _filter_status(store.unit, status)
    baseline = float(store.unit["is_risk"].mean()) or 1e-9
    if len(unit) == 0:
        return {"status": status, "baseline_risk_ratio": baseline,
                "n_lots_total": int(store.unit["run_id"].nunique()), "items": []}
    grouped = (
        unit.groupby("run_id")
        .agg(
            n_units=("ufs_serial", "count"),
            n_wafers=("wafer_no", "nunique"),
            n_risk=("is_risk", "sum"),
            mean_pred=("pred", "mean"),
            max_pred=("pred", "max"),
            mean_health=("health", "mean"),
        )
        .reset_index()
    )
    grouped["risk_ratio"] = grouped["n_risk"] / grouped["n_units"]
    grouped["lift"] = grouped["risk_ratio"] / baseline
    if sort not in grouped.columns:
        raise HTTPException(400, f"invalid sort: {sort}")
    grouped = grouped.sort_values(sort, ascending=False).head(limit)
    return {
        "status": status,
        "baseline_risk_ratio": baseline,
        "n_lots_total": int(unit["run_id"].nunique()),
        "items": _records(grouped),
    }


@app.get("/api/lots/{run_id}/aggregate-map")
def lot_aggregate_map(
    run_id: str,
    status: Optional[str] = Query(None, description="completed/pending/today/all"),
    agg: str = Query("max", pattern="^(mean|max)$"),
):
    """lot 내 모든 wafer를 겹쳐 die 좌표별 pred를 집계 → wafer map용.

    같은 (die_x, die_y) 위치에 여러 wafer의 die가 있으므로, 좌표별로 mean 또는 max를 반환.
    Drilldown에서 lot 자체를 선택했을 때 누적 wafer map으로 표시.
    """
    die = store.die[store.die["run_wf_xy"].str.startswith(run_id + "_")].copy()
    if status and status != "all":
        # status 필터: 해당 unit이 status에 해당하는 die만
        if status not in ("completed", "pending", "today"):
            raise HTTPException(400, f"invalid status: {status}")
        unit_in_status = set(store.unit.loc[store.unit["status"] == status, "ufs_serial"])
        die = die[die["ufs_serial"].isin(unit_in_status)]
    if len(die) == 0:
        raise HTTPException(404, f"lot not found or empty: {run_id}")

    agg_func = "mean" if agg == "mean" else "max"
    aggregated = (
        die.groupby(["die_x", "die_y"])
        .agg(
            pred=("pred", agg_func),
            pi=("pi", "mean"),
            mu=("mu", "mean"),
            n_wafers=("run_wf_xy", "nunique"),
        )
        .reset_index()
    )
    # WaferMap이 기대하는 dies 스키마에 맞춤 (run_wf_xy, ufs_serial은 가짜 키로 채움)
    aggregated["run_wf_xy"] = (
        run_id + "_AGG_" + aggregated["die_x"].astype(str) + "_" + aggregated["die_y"].astype(str)
    )
    aggregated["ufs_serial"] = None  # 클릭해도 unit 진단 안 띄움 (집계 die)

    n_wafers_total = int(die["run_wf_xy"].apply(lambda s: s.split("_")[1]).nunique())
    return {
        "run_id": run_id,
        "agg": agg,
        "n_wafers": n_wafers_total,
        "n_dies_total": int(len(die)),
        "n_unique_positions": int(len(aggregated)),
        "dies": _records(aggregated[["run_wf_xy", "die_x", "die_y", "pi", "mu", "pred", "ufs_serial"]]),
    }


@app.get("/api/lots/{run_id}")
def lot_detail(run_id: str, status: str = Query("today")):
    unit = store.unit[store.unit["run_id"] == run_id]
    unit = _filter_status(unit, status)
    if len(unit) == 0:
        raise HTTPException(404, f"lot not found in status={status}: {run_id}")
    by_wafer = (
        unit.groupby(["wafer_key", "wafer_no"])
        .agg(
            n_units=("ufs_serial", "count"),
            n_risk=("is_risk", "sum"),
            mean_pred=("pred", "mean"),
            max_pred=("pred", "max"),
        )
        .reset_index()
    )
    by_wafer["risk_ratio"] = by_wafer["n_risk"] / by_wafer["n_units"]
    by_wafer = by_wafer.sort_values("risk_ratio", ascending=False)
    return {
        "run_id": run_id,
        "status": status,
        "summary": {
            "n_units": int(len(unit)),
            "n_wafers": int(unit["wafer_no"].nunique()),
            "n_risk": int(unit["is_risk"].sum()),
            "risk_ratio": float(unit["is_risk"].mean()),
            "mean_pred": float(unit["pred"].mean()),
            "max_pred": float(unit["pred"].max()),
        },
        "wafers": _records(by_wafer),
    }


@app.get("/api/units/{ufs_serial}/report")
def unit_report(ufs_serial: str):
    """unit 위험 진단 — 자연어 narrative 포함."""
    if ufs_serial not in store.unit_indexed.index:
        raise HTTPException(404, f"unit not found: {ufs_serial}")
    unit_row = store.unit_indexed.loc[ufs_serial]
    if isinstance(unit_row, pd.DataFrame):
        unit_row = unit_row.iloc[0]
    dies = store.die_by_unit.get(ufs_serial, pd.DataFrame())

    threshold = float(store.unit["pred"].quantile(0.95))
    same_status = store.unit[store.unit["status"] == unit_row["status"]]
    pred_rank = float((same_status["pred"] < unit_row["pred"]).mean())

    if len(dies) > 0:
        worst_idx = dies["pred"].idxmax()
        worst_die = dies.loc[worst_idx]
        worst = {
            "run_wf_xy": worst_die["run_wf_xy"],
            "die_x": int(worst_die["die_x"]),
            "die_y": int(worst_die["die_y"]),
            "pi": float(worst_die["pi"]),
            "mu": float(worst_die["mu"]),
            "pred": float(worst_die["pred"]),
        }
    else:
        worst = None

    pi_mean = float(unit_row["pi_mean"])
    mu_mean = float(unit_row["mu_mean"])
    is_risk = bool(unit_row["is_risk"])
    status_label = {"today": "오늘 검사", "pending": "결과 대기", "completed": "검사 완료"}[
        unit_row["status"]
    ]

    verdict = "WARNING" if is_risk else "NORMAL"
    sentences = [
        f"{ufs_serial}는 wafer {unit_row['wafer_key']}의 unit ({status_label}, {unit_row['inspected_date']}).",
    ]
    if is_risk:
        sentences.append(
            f"pred {unit_row['pred']:.5f} — 상위 5% 임계 {threshold:.5f} 초과로 위험 분류."
        )
    else:
        sentences.append(
            f"pred {unit_row['pred']:.5f} — 임계 {threshold:.5f} 이하 (백분위 {pred_rank*100:.1f}%)."
        )
    if worst:
        sentences.append(
            f"4 die 중 ({worst['die_x']}, {worst['die_y']}) 위치가 가장 위험 (pred {worst['pred']:.5f})."
        )
    if pi_mean < 0.3:
        sentences.append(f"π 평균 {pi_mean:.3f} — 비-zero 가능성이 큼.")
    elif pi_mean > 0.7:
        sentences.append(f"π 평균 {pi_mean:.3f} — zero 가능성 큼. μ {mu_mean:.5f}가 위험 기여.")

    return {
        "ufs_serial": ufs_serial,
        "verdict": verdict,
        "is_risk": is_risk,
        "pred": float(unit_row["pred"]),
        "health": float(unit_row["health"]) if pd.notna(unit_row["health"]) else None,
        "status": unit_row["status"],
        "inspected_date": unit_row["inspected_date"],
        "wafer_key": unit_row["wafer_key"],
        "pred_rank": pred_rank,
        "threshold": threshold,
        "pi_mean": pi_mean,
        "mu_mean": mu_mean,
        "worst_die": worst,
        "narrative": sentences,
    }


@app.get("/")
def root():
    return {
        "service": "Wafer Health Dashboard API",
        "docs": "/docs",
    }


# ─── Model 산출물 endpoint (build_model_artifacts.py 결과) ─────────────
def _require_model_artifact(name: str):
    if name not in store.model_artifacts:
        raise HTTPException(
            503,
            f"{name} 산출물 없음. 'python 5_dashboard/build_model_artifacts.py' 실행 필요",
        )
    return store.model_artifacts[name]


@app.get("/api/model/fold-metrics")
def model_fold_metrics():
    """5-fold OOF RMSE."""
    return _require_model_artifact("fold_metrics")


@app.get("/api/model/feature-importance")
def model_feature_importance(top: int = Query(20, ge=1, le=100)):
    """LightGBM gain 5-fold 평균 (mu, pi, total). top N."""
    df = _require_model_artifact("feature_importance").head(top)
    return {"items": df.to_dict(orient="records"), "n_total": int(len(df))}


@app.get("/api/model/psi")
def model_psi(top: int = Query(20, ge=1, le=100)):
    """train ↔ validation 변수별 PSI."""
    df = _require_model_artifact("psi").head(top)
    return {"items": df.to_dict(orient="records")}


@app.get("/api/model/var-compare")
def model_var_compare(top: int = Query(20, ge=1, le=100)):
    """위험(pred>p95) vs 정상 unit의 변수별 t-test + Cohen's d."""
    df = _require_model_artifact("var_compare").head(top)
    return {"items": df.to_dict(orient="records")}