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
# 런타임에 읽는 건 5_dashboard/data/ 뿐 — 부모 프로젝트 의존 없음 (이 폴더만 공유하면 실행 가능).
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")


# 시연용 시계열 분배 상수
COMPLETED_SPAN_DAYS = 70  # oof → today-70 ~ today-1 (10주)
PENDING_SPAN_DAYS = 56    # val/test → today+1 ~ today+56 (8주)
DAILY_VARIATION = 0.20    # 일별 ±20% 변동
WEEKLY_VARIATION = 0.15   # 주별 ±15% 변동 (주별 가중치 후 일별 재분배)


def _assign_status_and_date(df: pd.DataFrame) -> pd.DataFrame:
    """split 기반 status + 시연용 날짜 균등 분배 (±20% 변동).

    status / 날짜 매핑:
      - completed (split=='oof')          → today-30 ~ today-1 (과거 30일에 균등 분배)
      - pending   (split in val/test)     → today+1 ~ today+30 (미래 30일에 균등 분배)
      - today                              → 위 분배 후 oof 1일치를 오늘로 이동
    """
    today = date.today()
    df = df.copy()
    rng = np.random.default_rng(42)  # 재현성 고정

    pending_splits = {"val", "validation", "test"}
    is_pending = df["split"].isin(pending_splits) if "split" in df.columns else pd.Series(False, index=df.index)
    is_completed = ~is_pending

    # 1) 주별 → 일별 2단계 가중 분배 (모든 주가 0이 되지 않도록 최소량 보장)
    def weighted_dist(n_units: int, n_days: int) -> np.ndarray:
        """n_days 길이의 일별 unit 수 배열, 합 = n_units.
        주별 ±15% 변동으로 1차 분배 → 주 내부에서 일별 ±20% 변동으로 2차 분배.
        7일로 나누어 떨어지지 않으면 마지막 partial week도 별도 주로 취급.
        """
        n_full_weeks, rem_days = divmod(n_days, 7)
        n_weeks = n_full_weeks + (1 if rem_days > 0 else 0)
        # 1차: 주별 가중치
        week_weights = rng.uniform(1 - WEEKLY_VARIATION, 1 + WEEKLY_VARIATION, n_weeks)
        # partial week는 일수 비례로 가중치 축소
        if rem_days > 0:
            week_weights[-1] *= rem_days / 7
        week_weights = week_weights / week_weights.sum()
        week_counts = np.floor(week_weights * n_units).astype(int)
        leftover = n_units - week_counts.sum()
        if leftover > 0:
            extra_idx = rng.choice(n_weeks, leftover, replace=True)
            for idx in extra_idx:
                week_counts[idx] += 1

        # 2차: 각 주 내부에서 일별 ±20% 분배
        counts = np.zeros(n_days, dtype=int)
        for w in range(n_weeks):
            start = w * 7
            end = min(start + 7, n_days)
            days_in_week = end - start
            wn = week_counts[w]
            if wn == 0 or days_in_week == 0:
                continue
            day_weights = rng.uniform(1 - DAILY_VARIATION, 1 + DAILY_VARIATION, days_in_week)
            day_weights = day_weights / day_weights.sum()
            day_counts = np.floor(day_weights * wn).astype(int)
            day_leftover = wn - day_counts.sum()
            if day_leftover > 0:
                extra = rng.choice(days_in_week, day_leftover, replace=True)
                for idx in extra:
                    day_counts[idx] += 1
            counts[start:end] = day_counts
        return counts

    # 2) completed 분배: today-30 ~ today-1
    completed_idx = df.index[is_completed].to_numpy()
    rng.shuffle(completed_idx)
    completed_counts = weighted_dist(len(completed_idx), COMPLETED_SPAN_DAYS)
    completed_dates = []
    for day_offset in range(COMPLETED_SPAN_DAYS, 0, -1):  # 30일전 → 1일전
        completed_dates.extend([today - timedelta(days=day_offset)] * completed_counts[COMPLETED_SPAN_DAYS - day_offset])

    # 3) pending 분배: today+1 ~ today+30
    pending_idx = df.index[is_pending].to_numpy()
    rng.shuffle(pending_idx)
    pending_counts = weighted_dist(len(pending_idx), PENDING_SPAN_DAYS)
    pending_dates = []
    for day_offset in range(1, PENDING_SPAN_DAYS + 1):  # 1일후 → 30일후
        pending_dates.extend([today + timedelta(days=day_offset)] * pending_counts[day_offset - 1])

    # 4) inspected_date / status 컬럼 생성
    df["inspected_date"] = ""
    df["status"] = ""
    df.loc[completed_idx, "inspected_date"] = [d.isoformat() for d in completed_dates]
    df.loc[completed_idx, "status"] = "completed"
    df.loc[pending_idx, "inspected_date"] = [d.isoformat() for d in pending_dates]
    df.loc[pending_idx, "status"] = "pending"

    # 5) today: completed에서 1일치(today-1)를 오늘로 당겨오기
    today_iso = today.isoformat()
    yesterday_iso = (today - timedelta(days=1)).isoformat()
    today_mask = (df["inspected_date"] == yesterday_iso) & (df["status"] == "completed")
    # 어제분의 절반을 today로 이동 (시연용 적정량)
    today_idx_pool = df.index[today_mask].to_numpy()
    if len(today_idx_pool) > 0:
        n_today = len(today_idx_pool) // 2
        today_pick = rng.choice(today_idx_pool, n_today, replace=False)
        df.loc[today_pick, "inspected_date"] = today_iso
        df.loc[today_pick, "status"] = "today"

    # 6) Y 시나리오 마스킹
    if "split" in df.columns:
        df.loc[df["split"].isin(pending_splits), "health"] = np.nan
    df.loc[df["status"] == "today", "health"] = np.nan

    return df


# ─── 데이터 로드 (앱 시작 시 1회) ────────────────────────
class DataStore:
    def __init__(self):
        print(f"[DataStore] 로드 시작: {DATA_DIR}")
        self.die = pd.read_parquet(os.path.join(DATA_DIR, "die_predictions.parquet"))
        unit = pd.read_parquet(os.path.join(DATA_DIR, "unit_predictions.parquet"))
        today = date.today()
        self.unit = _assign_status_and_date(unit)
        print(
            f"[DataStore] 시연용 날짜 분배: completed={today - timedelta(days=COMPLETED_SPAN_DAYS)}~{today - timedelta(days=1)}, "
            f"today={today}, pending={today + timedelta(days=1)}~{today + timedelta(days=PENDING_SPAN_DAYS)}"
        )

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

        # position별 위험도 집계
        pos_path = os.path.join(DATA_DIR, "position_stats.parquet")
        self.position_stats = pd.read_parquet(pos_path) if os.path.exists(pos_path) else None

        # unit × feature 평균 매트릭스 + 정상 baseline (비정상 변수 분석용)
        unit_feat_path = os.path.join(DATA_DIR, "unit_features.parquet")
        baseline_path = os.path.join(DATA_DIR, "normal_baseline.parquet")
        if os.path.exists(unit_feat_path) and os.path.exists(baseline_path):
            self.unit_features = pd.read_parquet(unit_feat_path).set_index("ufs_serial")
            self.normal_baseline = pd.read_parquet(baseline_path).set_index("feature")
            print(f"  unit features: {self.unit_features.shape}, baseline: {len(self.normal_baseline)} feat")
        else:
            self.unit_features = None
            self.normal_baseline = None
            print("  [warn] unit_features.parquet / normal_baseline.parquet 없음 — 비정상 분석 비활성")

        # feature ↔ health Pearson 상관 (Model 탭용 — health 실측 있는 unit만)
        self.feature_corr = None
        if self.unit_features is not None:
            health_series = self.unit.dropna(subset=["health"]).set_index("ufs_serial")["health"]
            common = self.unit_features.index.intersection(health_series.index)
            if len(common) > 100:
                X = self.unit_features.loc[common]
                y = health_series.loc[common]
                # 벡터화 Pearson r: cov(X, y) / (std(X) * std(y))
                X_centered = X - X.mean()
                y_centered = y - y.mean()
                num = X_centered.mul(y_centered, axis=0).sum()
                denom = np.sqrt((X_centered ** 2).sum() * (y_centered ** 2).sum())
                r = (num / denom.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
                self.feature_corr = pd.DataFrame({
                    "feature": r.index,
                    "r": r.values,
                    "abs_r": r.abs().values,
                }).sort_values("abs_r", ascending=False).reset_index(drop=True)
                print(f"  feature-target corr: {len(self.feature_corr)} feats, max |r|={self.feature_corr['abs_r'].max():.4f} (n={len(common)})")
            else:
                print(f"  [warn] feature-target corr: 공통 unit 부족 ({len(common)})")

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

# React dev server (5173) + cloudflared 임시 터널 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_origin_regex=r"https://.*\.trycloudflare\.com",
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
        "dies": _records(dies[["ufs_serial", "run_wf_xy", "die_x", "die_y", "position", "pi", "mu", "pred", "health", "split"]]),
        "units": _records(units[["ufs_serial", "pred", "health", "is_risk", "status"]]),
    }




@app.get("/api/units")
def list_units(
    status: Optional[str] = Query(None),
    risk_only: bool = Query(False),
    sort: str = Query("pred"),
    order: str = Query("desc", pattern="^(asc|desc)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100_000),
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


@app.get("/api/anomaly_features_global")
def anomaly_features_global(
    status: Optional[str] = Query(None, description="completed/pending/today/all"),
    top_n: int = Query(10, ge=1, le=50),
):
    """현재 status 필터의 위험 unit 집합이 정상 baseline 대비 가장 비정상인 feature Top-N.

    위험 unit = status 필터된 unit 중 is_risk=True (split별 pred 상위 5%).
    z-score = |위험 unit 평균 - 정상 평균| / 정상 std
    """
    if store.unit_features is None or store.normal_baseline is None:
        raise HTTPException(503, "unit_features.parquet not loaded")

    # 1) status 필터된 위험 unit 집합 결정
    df = _filter_status(store.unit, status)
    risk_serials = df.loc[df["is_risk"], "ufs_serial"].tolist()
    if len(risk_serials) == 0:
        return {"status": status or "all", "n_risk_units": 0, "top_n": top_n, "items": []}

    # 2) 위험 unit들의 feature 평균
    risk_features = store.unit_features.loc[
        store.unit_features.index.intersection(risk_serials)
    ]
    if len(risk_features) == 0:
        return {"status": status or "all", "n_risk_units": 0, "top_n": top_n, "items": []}
    risk_mean = risk_features.mean()

    # 3) z-score 산출
    baseline = store.normal_baseline
    z = (risk_mean.reindex(baseline.index) - baseline["normal_mean"]).abs() / baseline["normal_std"]
    z = z.replace([np.inf, -np.inf], np.nan).dropna()
    top = z.nlargest(top_n)

    items = []
    for feat in top.index:
        items.append({
            "feature": feat,
            "risk_mean": float(risk_mean[feat]),
            "normal_mean": float(baseline.loc[feat, "normal_mean"]),
            "normal_std": float(baseline.loc[feat, "normal_std"]),
            "z_score": float(top[feat]),
        })
    return {
        "status": status or "all",
        "n_risk_units": int(len(risk_features)),
        "top_n": top_n,
        "items": items,
    }


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
        "dies": _records(dies[["run_wf_xy", "die_x", "die_y", "position", "pi", "one_minus_pi", "mu", "pred"]]),
    }


@app.get("/api/units/{ufs_serial}/anomaly_features")
def unit_anomaly_features(ufs_serial: str, top_n: int = Query(10, ge=1, le=50)):
    """이 unit이 정상 unit baseline 대비 가장 비정상인 feature Top-N.

    z-score = |unit_mean - normal_mean| / normal_std
    (unit_mean = die 4개 feature 평균, normal_mean = pred 하위 50% unit들의 평균)
    """
    if store.unit_features is None or store.normal_baseline is None:
        raise HTTPException(503, "unit_features.parquet not loaded")
    if ufs_serial not in store.unit_features.index:
        raise HTTPException(404, f"unit not found: {ufs_serial}")

    unit_row = store.unit_features.loc[ufs_serial]
    baseline = store.normal_baseline  # index=feature, cols=normal_mean/normal_std

    z = (unit_row.reindex(baseline.index) - baseline["normal_mean"]).abs() / baseline["normal_std"]
    z = z.replace([np.inf, -np.inf], np.nan).dropna()
    top = z.nlargest(top_n)

    items = []
    for feat in top.index:
        items.append({
            "feature": feat,
            "unit_value": float(unit_row[feat]),
            "normal_mean": float(baseline.loc[feat, "normal_mean"]),
            "normal_std": float(baseline.loc[feat, "normal_std"]),
            "z_score": float(top[feat]),
        })
    return {"ufs_serial": ufs_serial, "top_n": top_n, "items": items}


@app.get("/api/position_risk")
def position_risk():
    """position(1~4)별 die-level 평균/최대/p95 예측 ppm."""
    if store.position_stats is None:
        raise HTTPException(503, "position_stats.parquet not loaded — prepare_data.py를 재실행하세요")
    return {"items": store.position_stats.to_dict(orient="records")}


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


@app.get("/api/alerts/today")
def alerts_today(
    lot_unit_threshold: int = Query(3, ge=1, le=50, description="lot에서 위험 unit 이 수 이상이면 lot 알림"),
    point_die_threshold: int = Query(3, ge=1, le=50, description="같은 (die_x,die_y) 포인트에서 위험 die 이 수 이상이면 포인트 알림"),
    lot_lift_threshold: float = Query(2.0, ge=1.0, le=10.0, description="lot risk_ratio / baseline 이 값 이상이면 디펙성 알림"),
    lot_min_units: int = Query(5, ge=1, description="디펙성 알림 대상이 되는 lot의 최소 unit 수 (소표본 노이즈 방지)"),
    position_lift_threshold: float = Query(1.5, ge=1.0, le=5.0, description="position 위험률 / 평균 이 값 이상이면 position 알림"),
):
    """오늘 검사된 데이터에서 4가지 패턴의 알림을 추출.

    1. position_alerts        : 특정 position(1~4)의 위험 die 비율이 평균 대비 lift 이상
    2. lot_cluster_alerts     : 같은 run_id에 위험 unit이 임계값 이상
    3. point_cluster_alerts   : 같은 (die_x, die_y) 좌표에서 위험 die가 임계값 이상 (여러 wafer 걸쳐서)
    4. systematic_lot_alerts  : 특정 lot의 위험 비율이 baseline의 lift배 이상 (디펙성 의심)
    """
    today_unit = store.unit[store.unit["status"] == "today"]
    if len(today_unit) == 0:
        return {
            "position_alerts": [],
            "lot_cluster_alerts": [],
            "point_cluster_alerts": [],
            "systematic_lot_alerts": [],
        }

    # die-level risk threshold: 전체 die pred 상위 5%
    die_risk_threshold = float(store.die["pred"].quantile(0.95))

    # 오늘 unit에 속한 die만 추출 (status는 die에 없으므로 unit 통해 join)
    today_unit_set = set(today_unit["ufs_serial"])
    today_die = store.die[store.die["ufs_serial"].isin(today_unit_set)].copy()
    today_die["is_risk_die"] = today_die["pred"] >= die_risk_threshold

    # ─ 1. position 알림 ───────────────────────────
    position_alerts = []
    if "position" in today_die.columns and today_die["position"].notna().any():
        pos_grp = (
            today_die.groupby("position")
            .agg(
                n_dies=("pred", "size"),
                n_risk_dies=("is_risk_die", "sum"),
                mean_pred=("pred", "mean"),
            )
            .reset_index()
        )
        pos_grp["risk_ratio"] = pos_grp["n_risk_dies"] / pos_grp["n_dies"].clip(lower=1)
        baseline_pos_ratio = float(pos_grp["risk_ratio"].mean()) or 1e-9
        pos_grp["lift"] = pos_grp["risk_ratio"] / baseline_pos_ratio
        flagged = pos_grp[pos_grp["lift"] >= position_lift_threshold]
        for _, r in flagged.sort_values("lift", ascending=False).iterrows():
            position_alerts.append({
                "position": int(r["position"]),
                "n_dies": int(r["n_dies"]),
                "n_risk_dies": int(r["n_risk_dies"]),
                "risk_ratio": float(r["risk_ratio"]),
                "lift": float(r["lift"]),
                "mean_pred": float(r["mean_pred"]),
            })

    # ─ 2. lot 클러스터 (위험 unit 임계 이상) ───────
    lot_cluster_alerts = []
    risk_today_unit = today_unit[today_unit["is_risk"]]
    if len(risk_today_unit) > 0:
        lot_grp = (
            risk_today_unit.groupby("run_id")
            .agg(
                n_risk_units=("ufs_serial", "count"),
                mean_pred=("pred", "mean"),
                sample_wafer_key=("wafer_key", "first"),
            )
            .reset_index()
        )
        flagged = lot_grp[lot_grp["n_risk_units"] >= lot_unit_threshold]
        for _, r in flagged.sort_values("n_risk_units", ascending=False).iterrows():
            lot_cluster_alerts.append({
                "run_id": str(r["run_id"]),
                "n_risk_units": int(r["n_risk_units"]),
                "mean_pred": float(r["mean_pred"]),
                "sample_wafer_key": str(r["sample_wafer_key"]),
            })

    # ─ 3. 포인트 클러스터 (die 좌표 hotspot) ──────
    point_cluster_alerts = []
    risk_today_die = today_die[today_die["is_risk_die"]]
    if len(risk_today_die) > 0:
        pt_grp = (
            risk_today_die.groupby(["die_x", "die_y"])
            .agg(
                n_risk_dies=("pred", "size"),
                n_wafers=("run_wf_xy", "nunique"),
                mean_pred=("pred", "mean"),
                sample_wafer_key=("wafer_key", "first"),
            )
            .reset_index()
        )
        flagged = pt_grp[pt_grp["n_risk_dies"] >= point_die_threshold]
        for _, r in flagged.sort_values("n_risk_dies", ascending=False).head(20).iterrows():
            point_cluster_alerts.append({
                "die_x": int(r["die_x"]),
                "die_y": int(r["die_y"]),
                "n_risk_dies": int(r["n_risk_dies"]),
                "n_wafers": int(r["n_wafers"]),
                "mean_pred": float(r["mean_pred"]),
                "sample_wafer_key": str(r["sample_wafer_key"]),
            })

    # ─ 4. 디펙성(systematic) lot 과다 발생 ──────
    systematic_lot_alerts = []
    baseline_unit_risk_ratio = float(store.unit["is_risk"].mean()) or 1e-9
    lot_full = (
        today_unit.groupby("run_id")
        .agg(
            n_units=("ufs_serial", "count"),
            n_risk=("is_risk", "sum"),
            mean_pred=("pred", "mean"),
            sample_wafer_key=("wafer_key", "first"),
        )
        .reset_index()
    )
    lot_full["risk_ratio"] = lot_full["n_risk"] / lot_full["n_units"].clip(lower=1)
    lot_full["lift"] = lot_full["risk_ratio"] / baseline_unit_risk_ratio
    flagged = lot_full[
        (lot_full["lift"] >= lot_lift_threshold) & (lot_full["n_units"] >= lot_min_units)
    ]
    for _, r in flagged.sort_values("lift", ascending=False).iterrows():
        systematic_lot_alerts.append({
            "run_id": str(r["run_id"]),
            "n_units": int(r["n_units"]),
            "n_risk": int(r["n_risk"]),
            "risk_ratio": float(r["risk_ratio"]),
            "lift": float(r["lift"]),
            "mean_pred": float(r["mean_pred"]),
            "sample_wafer_key": str(r["sample_wafer_key"]),
        })

    return {
        "baseline": {
            "die_risk_threshold": die_risk_threshold,
            "unit_risk_ratio": baseline_unit_risk_ratio,
        },
        "position_alerts": position_alerts,
        "lot_cluster_alerts": lot_cluster_alerts,
        "point_cluster_alerts": point_cluster_alerts,
        "systematic_lot_alerts": systematic_lot_alerts,
    }


@app.get("/api")
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


@app.get("/api/model/feature-corr")
def model_feature_corr(top: int = Query(10, ge=1, le=100)):
    """feature ↔ health Pearson 상관 Top N (|r| 기준).

    health 실측 있는 unit만 사용 (split=train + status=completed).
    EDA 결과 max |r|≈0.037 — 단일 피처로는 예측 불가, 비선형 모델 필요성 시각화.
    """
    if store.feature_corr is None:
        raise HTTPException(503, "feature_corr 미준비 (unit_features.parquet 또는 health 부족)")
    df = store.feature_corr.head(top)
    return {
        "items": df[["feature", "r", "abs_r"]].to_dict(orient="records"),
        "n_total": int(len(store.feature_corr)),
        "max_abs_r": float(store.feature_corr["abs_r"].max()),
    }


@app.get("/api/model/shap")
def model_shap(top: int = Query(10, ge=1, le=100)):
    """SHAP 근사 (임시 시뮬레이션) — feature_importance × var_compare 조합.

    실제 SHAP은 추후 산출 예정. 현재는 기존 산출물로 의미 동등한 근사:
      shap_like = sign(mean_risk - mean_norm) × |cohens_d| × norm(total_gain)
        - 크기: 모델 importance × 위험/정상 분리 강도
        - 부호: 위험군에서 평균이 크면 +(빨강) / 작으면 -(파랑)
    """
    fi = _require_model_artifact("feature_importance")
    vc = _require_model_artifact("var_compare")
    merged = fi.merge(vc, on="feature", how="inner")
    if len(merged) == 0:
        return {"items": [], "n_total": 0, "max_abs_shap": 0.0}
    gain_max = merged["total_gain"].max() or 1.0
    gain_norm = merged["total_gain"] / gain_max
    direction = np.sign(merged["mean_risk"] - merged["mean_norm"])
    shap_like = direction * merged["cohens_d"].abs() * gain_norm
    merged["shap"] = shap_like
    merged["abs_shap"] = shap_like.abs()
    out = (
        merged.sort_values("abs_shap", ascending=False)
        .head(top)[["feature", "shap", "abs_shap", "total_gain", "cohens_d", "mean_risk", "mean_norm"]]
    )
    return {
        "items": out.to_dict(orient="records"),
        "n_total": int(len(merged)),
        "max_abs_shap": float(merged["abs_shap"].max()),
    }


@app.get("/api/model/var-compare")
def model_var_compare(top: int = Query(20, ge=1, le=100)):
    """위험(pred>p95) vs 정상 unit의 변수별 t-test + Cohen's d."""
    df = _require_model_artifact("var_compare").head(top)
    return {"items": df.to_dict(orient="records")}


# ─── React 정적 파일 서빙 (시연 공유용 — frontend/dist 빌드 결과물) ─────
# /api/* 는 위 라우트가 이미 잡았고, 나머지 경로는 React가 처리
_FRONTEND_DIST = os.path.join(os.path.dirname(HERE), "frontend", "dist")
if os.path.isdir(_FRONTEND_DIST):
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    # /assets, /icons.svg 등 빌드 산출물
    app.mount(
        "/assets",
        StaticFiles(directory=os.path.join(_FRONTEND_DIST, "assets")),
        name="assets",
    )

    @app.get("/{full_path:path}")
    def _spa_fallback(full_path: str):
        # API/문서 경로는 제외 (라우트 우선순위상 여기 도달하면 정적 파일)
        candidate = os.path.join(_FRONTEND_DIST, full_path)
        if full_path and os.path.isfile(candidate):
            return FileResponse(candidate)
        # SPA: 그 외 모두 index.html (React Router가 처리)
        return FileResponse(os.path.join(_FRONTEND_DIST, "index.html"))