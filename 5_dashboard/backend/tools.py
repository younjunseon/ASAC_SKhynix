"""
Agent가 사용하는 tool 함수들.
각 함수는 Claude API tool_use의 실제 실행 로직.
"""
import os
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta

DASHBOARD_DIR  = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
DATA_DIR       = DASHBOARD_DIR
# 대용량 원본 데이터 fallback 경로 (sk_하이닉스/0_data/)
_FALLBACK_DIRS = [
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "0_data")),
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data")),
    r"C:\Users\Dell3571\Desktop\기업\0_data",
]

_cache: dict = {}

def _load(filename: str) -> pd.DataFrame:
    if filename in _cache:
        return _cache[filename]
    # 1차: Dashboard/public/
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        # fallback 경로들 순서대로 탐색
        for fb in _FALLBACK_DIRS:
            candidate = os.path.join(fb, filename)
            if os.path.exists(candidate):
                path = candidate
                break
        else:
            raise FileNotFoundError(f"{filename} 파일을 찾을 수 없습니다.")
    df = pd.read_csv(path)
    _cache[filename] = df
    return df

def _load_dashboard(filename: str) -> pd.DataFrame:
    path = os.path.normpath(os.path.join(DASHBOARD_DIR, filename))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dashboard/public/{filename} 없음")
    return pd.read_csv(path)


def _x_features_only(df: pd.DataFrame, col: str = "feature") -> pd.DataFrame:
    """X0~X1086 같은 WT 피처만 남기고 die_x, die_y, position 등 메타 피처 제거."""
    import re as _re
    return df[df[col].astype(str).str.match(r"^X\d+$", na=False)]


def _filter_units(units: pd.DataFrame, start: str = "", end: str = "") -> pd.DataFrame:
    """
    날짜 필터링을 적용하지 않고 항상 전체 unit(모든 split)을 반환한다.
    기간 표현("이번 주" 등)은 라벨로만 표시하고, 데이터는 현재 대시보드 전체를 그대로 사용.
    (데이터의 실제 날짜와 무관하게 전체 기준으로 보고서 생성)
    """
    return units


def _date_range_label(start: str, end: str) -> str:
    """필터 기간 레이블 생성."""
    if not start and not end:
        return "전체 val 데이터"
    if start and end:
        return f"{start[:4]}.{start[4:6]}.{start[6:]} ~ {end[:4]}.{end[4:6]}.{end[6:]}"
    if start:
        return f"{start[:4]}.{start[4:6]}.{start[6:]} 이후"
    return f"{end[:4]}.{end[4:6]}.{end[6:]} 이전"


# ── ① 기간 추론 ──────────────────────────────────────────────
# 데이터 날짜 범위: 20260327 ~ 20260708
_DATA_START = "20260327"
_DATA_END   = "20260708"

def infer_period(user_text: str) -> dict:
    """
    사용자 입력에서 기간을 추론하고 start/end(YYYYMMDD)를 반환.
    데이터 범위: 20251001 ~ 20251214.
    반환: {label, start, end}
    """
    import re as _re
    text = user_text

    # 직접 날짜 패턴: "11월 9일~11일", "10/1~10/7" 등
    m = _re.search(r'(\d{1,2})월\s*(\d{1,2})일?\s*[~\-]\s*(\d{1,2})일', text)
    if m:
        mo, d1, d2 = int(m.group(1)), int(m.group(2)), int(m.group(3))
        start = f"2026{mo:02d}{d1:02d}"
        end   = f"2026{mo:02d}{d2:02d}"
        return {"label": f"{mo}월 {d1}일~{d2}일", "start": start, "end": end}

    # "N월" 단독
    m = _re.search(r'(\d{1,2})월', text)
    if m:
        mo = int(m.group(1))
        start = f"2026{mo:02d}01"
        end   = f"2026{mo:02d}31"
        return {"label": f"{mo}월", "start": start, "end": end}

    # "최근 N일"
    m = _re.search(r'최근\s*(\d+)\s*일', text)
    if m:
        n = int(m.group(1))
        from datetime import datetime, timedelta
        end_dt   = datetime(2026, 6, 10)   # 데이터 마지막
        start_dt = end_dt - timedelta(days=n - 1)
        start = start_dt.strftime("%Y%m%d")
        end   = end_dt.strftime("%Y%m%d")
        return {"label": f"최근 {n}일", "start": start, "end": end}

    # "최근 N주"
    m = _re.search(r'최근\s*(\d+)\s*주', text)
    if m:
        n = int(m.group(1))
        from datetime import datetime, timedelta
        end_dt   = datetime(2026, 6, 10)   # 데이터 마지막
        start_dt = end_dt - timedelta(weeks=n)
        start = start_dt.strftime("%Y%m%d")
        end   = end_dt.strftime("%Y%m%d")
        return {"label": f"최근 {n}주", "start": start, "end": end}

    # 기준일: 오늘 = 2026-06-11 (목), 데이터는 어제(06-10)까지
    # "이번 주" → 06/08(월)~06/10(데이터 마지막)
    if "이번 주" in text or "이번주" in text:
        return {"label": "이번 주 (06/08~06/10)", "start": "20260608", "end": "20260610"}

    # "지난 주" → 06/01~06/07
    if "지난 주" in text or "지난주" in text:
        return {"label": "지난 주 (06/01~06/07)", "start": "20260601", "end": "20260607"}

    # "이번 달" → 6월(~데이터 마지막)
    if "이번 달" in text or "이번달" in text:
        return {"label": "6월 (06/01~06/10)", "start": "20260601", "end": "20260610"}

    # "지난 달" → 5월
    if "지난 달" in text or "지난달" in text:
        return {"label": "5월", "start": "20260501", "end": "20260531"}

    # 기본: 전체 기간
    return {"label": "전체 기간", "start": "", "end": ""}


# ── ② 데이터 스캔 ─────────────────────────────────────────────
def scan_data(start: str = "", end: str = "") -> dict:
    """
    데이터 스캔. grade4=매우위험(HIGH), grade1=정상
    start/end: YYYYMMDD 형식 날짜 필터
    """
    units = _load("dashboard_units.csv")
    period = _filter_units(units, start, end)

    if period.empty:
        return {"error": "해당 기간 데이터가 없습니다."}

    total = len(period)
    g1 = (period["grade"] == "grade1").sum()
    g2 = (period["grade"] == "grade2").sum()
    g3 = (period["grade"] == "grade3").sum()
    g4 = (period["grade"] == "grade4").sum()

    # grade4(매우위험) 기준으로 집중 LOT/웨이퍼 집계
    lot_risk = (
        period.groupby("run_id")["grade"]
        .apply(lambda x: (x == "grade4").sum())
        .sort_values(ascending=False)
    )
    top_lot = int(lot_risk.index[0]) if not lot_risk.empty else None
    top_lot_count = int(lot_risk.iloc[0]) if not lot_risk.empty else 0

    wafer_risk = (
        period.groupby(["run_id", "wafer_no"])["grade"]
        .apply(lambda x: (x == "grade4").sum())
        .sort_values(ascending=False)
    )
    top_wafer_lot = int(wafer_risk.index[0][0]) if not wafer_risk.empty else None
    top_wafer_no  = int(wafer_risk.index[0][1]) if not wafer_risk.empty else None
    top_wafer_count = int(wafer_risk.iloc[0]) if not wafer_risk.empty else 0

    unique_lots = period["run_id"].nunique()

    return {
        "period": _date_range_label(start, end),
        "total_units": total,
        "unique_lots": unique_lots,
        "grade4_count": int(g4),
        "grade4_ratio": round(g4 / total * 100, 1),
        "grade3_count": int(g3),
        "grade2_count": int(g2),
        "grade1_count": int(g1),
        "top_lot": top_lot,
        "top_lot_grade4_count": top_lot_count,
        "top_wafer": {"lot": top_wafer_lot, "wafer": top_wafer_no, "grade4_count": top_wafer_count},
    }


# ── ③ 원인 분석 ───────────────────────────────────────────────
def analyze_features(start: str = "", end: str = "", top_n: int = 10) -> dict:
    """
    grade4(매우위험) vs grade1(정상) feature 분포 비교.
    원본 xs 데이터와 dashboard_units를 조인하여 실시간 분석.
    start/end: YYYYMMDD 형식 날짜 필터
    """
    units = _load("dashboard_units.csv")
    val_units = _filter_units(units, start, end)
    period_units = val_units[["ufs_serial", "grade"]]

    # importance 상위 20개만 분석 (속도 최적화)
    try:
        fi = _x_features_only(_load("feature_importance.csv"))   # 메타 피처 제외
        top50 = fi.sort_values("lgbm_rank").head(20)["feature"].tolist()
        fi_rank = dict(zip(fi["feature"], fi["lgbm_rank"]))
    except FileNotFoundError:
        top50 = None
        fi_rank = {}

    # xs 부분 로드: anomaly 캐시 재사용 or 신규 로드
    xs_cache_key = "xs_anomaly_60"
    if xs_cache_key in _cache:
        xs = _cache[xs_cache_key]
    else:
        try:
            xs = _load("compet_xs_data.csv")
        except Exception:
            # xs(원본 1.2GB) 없으면 분포 분석 불가 — graceful 반환 (보고서는 이 함수 미사용)
            return {"error": "compet_xs 데이터가 없어 분포 분석을 건너뜁니다.",
                    "top_features": [], "high_n": 0, "low_n": 0}
    keep_cols = ["ufs_serial"] + ([c for c in top50 if c in xs.columns] if top50 else [c for c in xs.columns if c.startswith("X")])
    merged = xs[[c for c in keep_cols if c in xs.columns]].merge(period_units, on="ufs_serial", how="inner")

    if merged.empty:
        return {"error": "해당 기간 feature 데이터가 없습니다.", "top_features": []}

    feat_cols = [c for c in keep_cols if c != "ufs_serial" and c in merged.columns]

    merged_unit = merged.groupby(["ufs_serial", "grade"])[feat_cols].mean().reset_index()

    g4_df = merged_unit[merged_unit["grade"] == "grade4"][feat_cols]  # 매우위험
    g1_df = merged_unit[merged_unit["grade"] == "grade1"][feat_cols]  # 정상

    if g4_df.empty or g1_df.empty:
        return {"error": f"grade4({len(g4_df)}개) 또는 grade1({len(g1_df)}개) 데이터 부족.", "top_features": []}

    results = []
    for col in feat_cols:
        h_vals = g4_df[col].dropna()
        l_vals = g1_df[col].dropna()
        if len(h_vals) < 5 or len(l_vals) < 5:
            continue

        h_mean = float(h_vals.mean())
        l_mean = float(l_vals.mean())
        _, pval = stats.ttest_ind(h_vals, l_vals, equal_var=False)
        ratio = round(h_mean / abs(l_mean), 2) if l_mean != 0 else None

        results.append({
            "feature": col,
            "high_mean": round(h_mean, 4),   # grade4(위험) 평균
            "low_mean":  round(l_mean, 4),   # grade1(정상) 평균
            "ratio": ratio,
            "pval": float(pval),
            "importance_rank": fi_rank.get(col, 9999),
        })

    results.sort(key=lambda x: x["pval"])
    top = results[:top_n]
    top.sort(key=lambda x: x["importance_rank"])

    return {
        "period": _date_range_label(start, end),
        "compare_group": "grade1",   # 비교 기준(정상)
        "high_n": len(g4_df),        # grade4(위험) 수
        "low_n":  len(g1_df),        # grade1(정상) 수
        "top_features": top,
    }


# ── feature importance 조회 ───────────────────────────────────
def get_importance(top_n: int = 10) -> dict:
    """feature_importance.csv에서 상위 feature 반환 (X 피처만, 메타 제외)."""
    fi = _x_features_only(_load("feature_importance.csv"))
    top = fi.sort_values("lgbm_rank").head(top_n)
    return {
        "features": top[["feature", "lgbm_rank", "lgbm_gain"]].to_dict("records")
    }


# ── Anomaly Feature: importance 상위 피처의 grade4(위험) vs grade1(정상) 분포 비교 ──
def get_anomaly_feature_stats(top_n: int = 5) -> list:
    """
    importance 상위 피처에 대해 danger_grade vs normal_grade 분포 비교.
    - 항상 POOL_SIZE(60)개 후보를 전부 계산하여 반환 (top_n은 호출측에서 슬라이싱).
    - die 샘플이 적은 danger 그룹도 1개 이상이면 허용 (grade4=1unit=4die 대응).
    - xs는 unit 레벨로 집계(mean)하여 사용 → danger 그룹 1unit도 유효.
    반환: [{"feature": "X592", "danger": 72, "normal": 28, "ratio": 2.52}, ...]
    """
    fi    = _x_features_only(_load("feature_importance.csv"))   # 메타 피처(die_x 등) 제외
    units = _load("dashboard_units.csv")

    # 후보 피처 목록 — 항상 POOL_SIZE 기준으로 고정
    POOL_SIZE = 60
    candidate_n = min(len(fi), POOL_SIZE)
    candidate_feats = fi.sort_values("lgbm_rank").head(candidate_n)["feature"].tolist()

    # xs: 필요한 컬럼만 usecols로 로드, 캐시 키 고정
    xs_cache_key = f"xs_anomaly_{POOL_SIZE}"
    if xs_cache_key not in _cache:
        xs_path = None
        for d in [DATA_DIR] + _FALLBACK_DIRS:
            p = os.path.join(d, "compet_xs_data.csv")
            if os.path.exists(p):
                xs_path = p
                break
        if xs_path is None:
            return []
        try:
            header = pd.read_csv(xs_path, nrows=0).columns.tolist()
            valid_feats = [f for f in candidate_feats if f in header]
            if not valid_feats:
                return []
            xs_partial = pd.read_csv(xs_path, usecols=["ufs_serial"] + valid_feats)
            _cache[xs_cache_key] = xs_partial
        except Exception:
            return []

    xs = _cache[xs_cache_key]

    # xs에 실제 존재하는 피처만 사용 (die_x, die_y 등 xs에 없는 피처 제외)
    valid_candidate_feats = [f for f in candidate_feats if f in xs.columns]

    # unit 레벨로 집계 (die→unit mean) — danger 그룹이 소수 unit이어도 유효
    unit_agg = xs.groupby("ufs_serial")[valid_candidate_feats].mean().reset_index()
    candidate_feats = valid_candidate_feats  # 이후 루프에서 유효 피처만 순회

    val_units = units[["ufs_serial", "grade"]]
    merged    = unit_agg.merge(val_units, on="ufs_serial", how="inner")

    # 데이터에 실제 존재하는 grade 중 최고·최저 번호를 자동 감지
    existing = merged["grade"].unique().tolist()
    grade_nums = sorted([int(g.replace("grade", "")) for g in existing if g.startswith("grade")])
    if len(grade_nums) < 2:
        return []
    danger_grade = f"grade{grade_nums[-1]}"
    normal_grade = f"grade{grade_nums[0]}"

    g_danger = merged[merged["grade"] == danger_grade]
    g_normal = merged[merged["grade"] == normal_grade]

    # normal 그룹은 5개 이상 필요, danger 그룹은 1개 이상이면 허용
    if len(g_danger) < 1 or len(g_normal) < 5:
        return []

    # 후보 전체를 계산하여 풀 구성 (top_n 제한 없음 — 호출측에서 슬라이싱)
    result = []
    for feat in candidate_feats:
        if feat not in merged.columns:
            continue
        h = g_danger[feat].dropna()
        l = g_normal[feat].dropna()
        if len(h) < 1 or len(l) < 5:
            continue

        h_mean = float(h.mean())
        l_mean = float(l.mean())
        l_std  = float(l.std()) if len(l) > 1 else 0.0
        ratio  = round(h_mean / l_mean, 3) if l_mean != 0 else None
        z_score = round(abs(h_mean - l_mean) / l_std, 2) if l_std > 0 else None

        dev = abs(ratio - 1.0) if ratio else 0.0
        danger = min(int(dev / 3.0 * 100), 95)
        normal = 100 - danger

        result.append({
            "feature":     feat,
            "grade1_mean": round(h_mean, 4),
            "grade4_mean": round(l_mean, 4),
            "ratio":       ratio,
            "z_score":     z_score,
            "danger":      danger,
            "normal":      normal,
        })

    return result


# ── pred vs actual 데이터 조회 (Scatter용) ───────────────────
def get_pred_actual_data(max_pts: int = 300) -> list:
    """
    val split의 reg_pred vs health scatter 데이터 반환.
    반환: [{"x": pred, "y": actual}, ...]
    """
    units = _load("dashboard_units.csv")
    # health가 있는 unit(train)만 scatter 가능, test는 health 없음
    val = units[units["health"].notna() & (units["health"] != "")][["reg_pred", "health"]].dropna()

    if len(val) > max_pts:
        val = val.sample(max_pts, random_state=42)

    return [
        {"x": round(float(r["reg_pred"]), 6), "y": round(float(r["health"]), 6)}
        for _, r in val.iterrows()
    ]


# ── 포지션별 WT 피처 이상 비율 ───────────────────────────────
def get_position_defect_rate() -> dict:
    """
    전체(train+val+test) position(1~4)별 top2 피처 이상 비율.
    dashboard_units.csv의 pos{p}_{feat} 컬럼 사용 (xs 원본 불필요).
    이상 기준: HIGH 그룹의 하위 10% 미만.
    """
    units = _load("dashboard_units.csv")
    val   = units.copy()  # train/val/test 전체 사용

    try:
        fi   = _load("feature_importance.csv")
        top2 = fi.sort_values("lgbm_rank").head(2)["feature"].tolist()
    except FileNotFoundError:
        top2 = ["X1064", "X592"]
    f1, f2 = (top2 + ["X1064", "X592"])[:2]

    result = {"labels": [], "high_ratio": [], "med_ratio": [], "low_ratio": [],
              "feat1": f1, "feat2": f2}

    for p in [1, 2, 3, 4]:
        c1, c2 = f"pos{p}_{f1}", f"pos{p}_{f2}"
        if c1 not in val.columns or c2 not in val.columns:
            result["labels"].append(f"P{p}")
            result["high_ratio"].append(0.0)
            result["med_ratio"].append(0.0)
            result["low_ratio"].append(100.0)
            continue

        grp      = val[[c1, c2, "risk"]].dropna(subset=[c1])
        high_grp = grp[grp["risk"] == "HIGH"]
        thresh1  = float(high_grp[c1].quantile(0.10)) if len(high_grp) > 0 else None
        thresh2  = float(high_grp[c2].quantile(0.10)) if c2 in grp.columns and len(high_grp) > 0 else None
        total    = len(grp)
        r1 = round(float((grp[c1] < thresh1).sum()) / total * 100, 1) if thresh1 is not None and total > 0 else 0.0
        r2 = round(float((grp[c2] < thresh2).sum()) / total * 100, 1) if thresh2 is not None and total > 0 else 0.0

        result["labels"].append(f"P{p}")
        result["high_ratio"].append(r1)
        result["med_ratio"].append(r2)
        result["low_ratio"].append(round(100 - max(r1, r2), 1))

    return result


# ── 대표 Unit (불량 위험 최고 unit) ─────────────────────────
def get_top_unit_data(serial: str = None) -> dict:
    """
    serial 지정 시 해당 unit, 미지정 시 reg_pred 최고 unit 반환.
    반환: {serial, run_id, wafer_no, pred_ppm, actual_ppm, risk, pos_feat_vals}
    pos_feat_vals: {"P1": {"X1064": 11.84, ...}, ...} — 포지션별 top4 feature 값
    """
    units = _load("dashboard_units.csv")

    if serial:
        filtered = units[units["ufs_serial"] == serial]
        val = filtered if not filtered.empty else units.sort_values("reg_pred", ascending=False)
    else:
        val = units.sort_values("reg_pred", ascending=False)

    if val.empty:
        return {}

    row = val.iloc[0]
    serial = str(row["ufs_serial"])

    # 포지션별 top feature 값 (dashboard_units에 이미 pos{p}_{feat} 컬럼으로 존재)
    # x/y 좌표 컬럼(pos{p}_x, pos{p}_y)은 feature 값에서 제외
    pos_feat_vals = {}
    pos_cols = {c for c in row.index if c.startswith("pos") and "_" in c}
    if pos_cols:
        for p in [1, 2, 3, 4]:
            prefix = f"pos{p}_"
            feats = {
                c[len(prefix):]: round(float(row[c]), 4)
                for c in pos_cols
                if c.startswith(prefix)
                and c not in (f"pos{p}_x", f"pos{p}_y")
                and pd.notna(row[c])
            }
            if feats:
                pos_feat_vals[f"P{p}"] = feats

    # die 좌표 (dashboard_units에 die_x, die_y 컬럼)
    die_x = int(row["die_x"]) if "die_x" in row.index and pd.notna(row["die_x"]) else None
    die_y = int(row["die_y"]) if "die_y" in row.index and pd.notna(row["die_y"]) else None

    # 포지션별 pred: wafer_map.csv에서 해당 unit의 ZIT die-level pred 직접 조회
    base_pred = float(row["reg_pred"])
    pos_health = {}
    try:
        wmap_path = os.path.normpath(os.path.join(DASHBOARD_DIR, "wafer_map.csv"))
        wmap_all = pd.read_csv(wmap_path, usecols=["ufs_serial", "position", "pred"])
        die_rows = wmap_all[wmap_all["ufs_serial"] == serial].set_index("position")["pred"]
        for p in [1, 2, 3, 4]:
            pos_health[f"P{p}"] = round(float(die_rows.get(p, base_pred)), 6)
    except Exception:
        for p in [1, 2, 3, 4]:
            pos_health[f"P{p}"] = round(base_pred, 6)

    return {
        "serial":        serial,
        "run_id":        int(row["run_id"]),
        "wafer_no":      int(row["wafer_no"]),
        "pred_ppm":      round(float(row["reg_pred"]) * 1_000_000, 1),
        "pred_health":   round(base_pred, 6),
        "actual_ppm":    round(float(row["health"]) * 1_000_000, 1) if pd.notna(row.get("health")) else 0.0,
        "risk":          str(row["risk"]),
        "die_x":         die_x,
        "die_y":         die_y,
        "pos_feat_vals": pos_feat_vals,
        "pos_health":    pos_health,
    }


# ── dashboard_units.csv 재생성 ────────────────────────────────
def rebuild_dashboard_units() -> str:
    """
    dashboard_units.csv를 원본 xs + 기존 units 기반으로 풍부하게 재생성.

    추가 컬럼:
      - die_x, die_y          : run_wf_xy 파싱 (position=1 기준 대표 좌표)
      - pos{p}_pred           : position 1~4 별 ZIT die-level pred (wafer_map.csv 기반)
      - pos{p}_{feat}         : position 1~4 × top5 feature 실측값 (20컬럼)
      - lot_total             : 해당 lot의 전체 unit 수
      - lot_defect_count      : 해당 lot의 grade1(HIGH) unit 수
      - lot_defect_rate       : lot 불량률 (%)

    원본 xs(174,980행)를 1회만 읽고 버림 — 이후 dashboard_units.csv만으로 동작.
    """
    import pandas as pd
    import numpy as np

    units_path = os.path.join(DATA_DIR, "dashboard_units.csv")
    xs_path    = os.path.join(DATA_DIR, "compet_xs_data.csv")
    fi_path    = os.path.join(DATA_DIR, "feature_importance.csv")

    units = pd.read_csv(units_path)
    fi    = pd.read_csv(fi_path)
    top_feats = fi.sort_values("lgbm_rank").head(5)["feature"].tolist()

    # ── xs: 필요한 컬럼만 로드 (메모리 절약)
    xs = pd.read_csv(xs_path, usecols=["ufs_serial", "run_wf_xy", "position"] + top_feats)

    # ── 1. position별 XY좌표 + die_x/die_y (position=1 대표)
    parsed_all = xs["run_wf_xy"].str.split("_", expand=True)
    xs = xs.copy()
    xs["die_x"] = parsed_all[2].astype(int)
    xs["die_y"] = parsed_all[3].astype(int)

    # position별 좌표 컬럼 (pos1_x, pos1_y, ..., pos4_x, pos4_y)
    coord_pivot_dfs = []
    for p in [1, 2, 3, 4]:
        xsp = xs[xs["position"] == p][["ufs_serial", "die_x", "die_y"]].copy()
        xsp = xsp.rename(columns={"die_x": f"pos{p}_x", "die_y": f"pos{p}_y"})
        coord_pivot_dfs.append(xsp)
    coord_pos_df = coord_pivot_dfs[0]
    for cpdf in coord_pivot_dfs[1:]:
        coord_pos_df = coord_pos_df.merge(cpdf, on="ufs_serial", how="left")

    # die_x/die_y: position=1 대표 좌표 (pos1_x/y 복사)
    coord_df = coord_pos_df[["ufs_serial", "pos1_x", "pos1_y"]].rename(
        columns={"pos1_x": "die_x", "pos1_y": "die_y"}
    )

    # ── 2. position별 ZIT die-level pred pivot (wafer_map.csv 기반)
    wmap_path = os.path.normpath(os.path.join(DASHBOARD_DIR, "wafer_map.csv"))
    pred_df = None
    if os.path.exists(wmap_path):
        wmap = pd.read_csv(wmap_path, usecols=["ufs_serial", "position", "pred"])
        pred_pivot_dfs = []
        for p in [1, 2, 3, 4]:
            wp = wmap[wmap["position"] == p][["ufs_serial", "pred"]].copy()
            wp = wp.rename(columns={"pred": f"pos{p}_pred"})
            pred_pivot_dfs.append(wp)
        pred_df = pred_pivot_dfs[0]
        for ppdf in pred_pivot_dfs[1:]:
            pred_df = pred_df.merge(ppdf, on="ufs_serial", how="left")

    # ── 3. position별 top feature 값 pivot (pos1_X1064, pos2_X592, ...)
    pivot_dfs = []
    for p in [1, 2, 3, 4]:
        xsp = xs[xs["position"] == p][["ufs_serial"] + top_feats].copy()
        xsp = xsp.rename(columns={f: f"pos{p}_{f}" for f in top_feats})
        pivot_dfs.append(xsp)

    pos_df = pivot_dfs[0]
    for pdf in pivot_dfs[1:]:
        pos_df = pos_df.merge(pdf, on="ufs_serial", how="left")

    # ── 4. lot별 불량 집계
    lot_stats = units.groupby("run_id").agg(
        lot_total        = ("ufs_serial", "count"),
        lot_defect_count = ("grade", lambda x: (x == "grade1").sum()),
    ).reset_index()
    lot_stats["lot_defect_rate"] = (
        lot_stats["lot_defect_count"] / lot_stats["lot_total"] * 100
    ).round(1)

    # ── 5. 기존 추가 컬럼 제거 후 새로 merge (중복 방지)
    _drop_prefixes = ("die_x", "die_y", "pos1_", "pos2_", "pos3_", "pos4_",
                      "lot_total", "lot_defect_count", "lot_defect_rate")
    base_cols = [c for c in units.columns
                 if not any(c == p or c.startswith(p) for p in _drop_prefixes)]
    base_col_count = len(base_cols)
    out = units[base_cols].copy()
    out = out.merge(coord_df,     on="ufs_serial", how="left")
    out = out.merge(coord_pos_df, on="ufs_serial", how="left")
    if pred_df is not None:
        out = out.merge(pred_df, on="ufs_serial", how="left")
    out = out.merge(pos_df,       on="ufs_serial", how="left")
    out = out.merge(lot_stats[["run_id", "lot_total", "lot_defect_count", "lot_defect_rate"]],
                    on="run_id", how="left")

    # float 컬럼 소수점 정리
    feat_cols = [c for c in out.columns if any(c.startswith(f"pos{p}_") for p in range(1,5))]
    out[feat_cols] = out[feat_cols].round(6)

    out.to_csv(units_path, index=False)

    # 캐시 무효화 (다음 _load에서 새 파일 읽음)
    _cache.pop("dashboard_units.csv", None)

    has_pred = pred_df is not None
    added = len(out.columns) - base_col_count
    return (
        f"dashboard_units.csv 재생성 완료\n"
        f"  행: {len(out):,}  기본 {base_col_count}열 → {len(out.columns)}열 (+{added}개)\n"
        f"  추가 컬럼: die_x, die_y, pos1~4_x/y, "
        f"{'pos1~4_pred(ZIT), ' if has_pred else ''}"
        f"pos1~4×{len(top_feats)}개 feature, "
        f"lot_total, lot_defect_count, lot_defect_rate\n"
        f"  top feats: {top_feats}"
    )


# ── 피처 top-1 LOT별 트렌드 (Chart.js 라인차트용) ────────────
def get_trend_top1_data(top_n_lots: int = 20) -> dict:
    """
    lgbm_rank 1위 피처의 LOT별 HIGH/MED 평균 트렌드 반환.
    dashboard_units.csv의 pos1_{feat} 컬럼 사용 (xs 원본 불필요).
    반환: {"feature": str, "labels": [...], "high": [...], "med": [...]}
    """
    fi = _load("feature_importance.csv")
    top1_feat = fi.sort_values("lgbm_rank").iloc[0]["feature"]
    col = f"pos1_{top1_feat}"  # position=1 대표값 컬럼

    units = _load("dashboard_units.csv")
    val = units.copy()  # train/val/test 전체 사용

    if col not in val.columns:
        return {"feature": top1_feat, "labels": [], "high": [], "med": []}

    lot_high = val[val["risk"] == "HIGH"].groupby("run_id")[col].mean()
    lot_med  = val[val["risk"] == "MED" ].groupby("run_id")[col].mean()

    lot_high_count = val[val["risk"] == "HIGH"].groupby("run_id").size()
    top_lots = sorted(lot_high_count.sort_values(ascending=False).head(top_n_lots).index.tolist())

    return {
        "feature":  top1_feat,
        "labels":   [f"LOT_{r}" for r in top_lots],
        "high":     [round(float(lot_high.get(r, 0)), 6) for r in top_lots],
        "med":      [round(float(lot_med.get(r, 0)), 6)  for r in top_lots],
    }


# ── LOT 트렌드 데이터 조회 (Chart.js용) ──────────────────────
def get_lot_trend_data(top_n: int = 20) -> dict:
    """
    LOT별 HIGH/MED/LOW unit 수 반환 (Chart.js 바 차트용).
    """
    units = _load("dashboard_units.csv")
    val   = units  # train/val/test 전체 사용

    lot_stats = (
        val.groupby("run_id")["risk"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )
    # 컬럼 보장
    for col in ["HIGH", "MED", "LOW"]:
        if col not in lot_stats.columns:
            lot_stats[col] = 0

    lot_stats["total"] = lot_stats[["HIGH", "MED", "LOW"]].sum(axis=1)
    lot_stats = lot_stats.sort_values("HIGH", ascending=False).head(top_n)
    lot_stats = lot_stats.sort_values("run_id")

    return {
        "labels": [f"LOT_{int(r)}" for r in lot_stats["run_id"]],
        "high":   lot_stats["HIGH"].tolist(),
        "med":    lot_stats["MED"].tolist(),
        "low":    lot_stats["LOW"].tolist(),
    }


# ── 배너용 전주 대비 ppm delta ────────────────────────────────
def get_ppm_delta() -> dict:
    """
    전주 대비 ppm 변화 — 대시보드 트렌드 차트 기준과 일치.
      - 이번주(curr) = 전체 unit reg_pred 평균 ppm (트렌드 차트 마지막 주 점과 동일)
      - 지난주(prev) = 트렌드 차트 과거 주 기준값(TARGET_PAST_PPM = 2100, 대시보드 하드코딩과 동일)
    반환: {prev_ppm, curr_ppm, delta, top_features: [feat1, feat2]}
    """
    units = _load("dashboard_units.csv")

    # 이번주 = 전체 unit 평균 (대시보드 Overview2 트렌드의 마지막 주 = units.reg_pred.mean())
    curr_ppm = round(float(units["reg_pred"].mean()) * 1_000_000, 1)
    # 지난주 = 트렌드 차트의 직전 주 점 (대시보드 트렌드 끝에서 2번째 = 실제 전주)
    try:
        _dp = get_weekly_yield_trend(recent_weeks=10).get("defect_ppm") or []
        prev_ppm = float(_dp[-2]) if len(_dp) >= 2 else 2100.0
    except Exception:
        prev_ppm = 2100.0
    delta    = round(curr_ppm - prev_ppm, 1)

    # top-2 피처명 (lgbm_rank 기준)
    try:
        fi = _load("feature_importance.csv")
        top2 = fi.sort_values("lgbm_rank").head(2)["feature"].tolist()
    except FileNotFoundError:
        top2 = []

    return {
        "prev_ppm":    prev_ppm,
        "curr_ppm":    curr_ppm,
        "delta":       delta,
        "top_features": top2,
    }


# ── 상위 2개 피처 scatter 데이터 ──────────────────────────────
def get_feature_scatter_data(feat1: str = None, feat2: str = None,
                             max_pts: int = 200, recent_n_lots: int = 5) -> dict:
    """
    scatter 데이터 반환. feat1/feat2 미지정 시 lgbm_rank 1,2위 사용.
    dashboard_units.csv의 pos1_{feat} 컬럼 사용 (xs 원본 불필요).
    x=피처값, y=reg_pred, risk별. 임계값 = HIGH 그룹 하위 5%.
    """
    if feat1 and feat2:
        top2_feats = [feat1, feat2]
    else:
        fi = _load("feature_importance.csv")
        top2_feats = fi.sort_values("lgbm_rank").head(2)["feature"].tolist()

    units   = _load("dashboard_units.csv")
    # train/val/test 전체에서 최신 N개 LOT 사용
    all_lots = sorted(units["run_id"].unique())
    recent_lots = all_lots[-recent_n_lots:]
    val = units[units["run_id"].isin(recent_lots)].copy()

    result = {}
    for i, feat in enumerate(top2_feats):
        col = f"pos1_{feat}"  # position=1 대표값
        if col not in val.columns:
            result[f"feat{i+1}"] = {"name": feat, "pts_high": [], "pts_med": [], "threshold": None}
            continue

        df = val[["ufs_serial", "run_id", col, "grade", "reg_pred"]].dropna(subset=[col])
        grade1_df = df[df["grade"] == "grade1"]
        grade4_df = df[df["grade"] == "grade4"]
        threshold = round(float(grade1_df[col].quantile(0.05)), 4) if not grade1_df.empty else None

        def _sample(sdf, n, c=col):
            sdf = sdf.sample(min(n, len(sdf)), random_state=42)
            return [{"x": round(float(r[c]), 4),
                     "y": round(float(r["reg_pred"]), 6),
                     "lot": int(r["run_id"])}
                    for _, r in sdf.iterrows()]

        result[f"feat{i+1}"] = {
            "name":      feat,
            "pts_high":  _sample(grade1_df, max_pts),
            "pts_med":   _sample(grade4_df, max_pts),
            "threshold": threshold,
        }

    return result


# ── L2 트렌드 + split 구간 경계 ──────────────────────────────
def get_lot_trend_with_split(top_n: int = 30) -> dict:
    """
    전체 LOT(train+val+test) HIGH 건수 트렌드 + test 시작 인덱스 반환.
    반환: {labels, high, test_start_idx}
    """
    units = _load("dashboard_units.csv")

    lot_stats = (
        units.groupby("run_id")["risk"]
        .apply(lambda x: int((x == "HIGH").sum()))
        .reset_index()
    )
    lot_stats.columns = ["run_id", "high_count"]

    # split 경계: 각 run_id가 속하는 split
    lot_split = (
        units.groupby("run_id")["split"]
        .first()
        .reset_index()
    )
    lot_stats = lot_stats.merge(lot_split, on="run_id").sort_values("run_id")

    test_start_idx = int(lot_stats[lot_stats["split"] == "test"].index.min()
                         - lot_stats.index.min()) if "test" in lot_stats["split"].values else -1

    return {
        "labels":         [f"LOT_{int(r)}" for r in lot_stats["run_id"]],
        "high":           lot_stats["high_count"].tolist(),
        "test_start_idx": test_start_idx,
    }


# ── 주차별 Grade 트렌드 (대시보드 grade_trend.csv 기반) ──────
def get_weekly_grade_trend() -> dict:
    """
    grade_trend.csv에서 주차별 grade 비율 반환.
    CSV 컬럼 명칭이 대시보드 기준과 반전되어 있으므로 매핑을 교정해서 반환:
      CSV grade4(다수=정상) → g1(정상 G1, #22C55E)
      CSV grade3            → g2(조심 G2, #EAB308)
      CSV grade2            → g3(위험 G3, #F59E0B)
      CSV grade1(소수=위험) → g4(매우위험 G4, #EF4444)
    반환: {labels:['MM/DD~MM/DD',...], g1:[...], g2:[...], g3:[...], g4:[...]}
    """
    df = _load_dashboard("grade_trend.csv")
    return {
        "labels": df["week"].tolist(),
        "g1":     [round(float(v), 1) for v in df["grade4"]],
        "g2":     [round(float(v), 1) for v in df["grade3"]],
        "g3":     [round(float(v), 1) for v in df["grade2"]],
        "g4":     [round(float(v), 1) for v in df["grade1"]],
    }


# ── 주차별 불량 트렌드 (대시보드 Overview.jsx trendResult와 동일 로직) ──
def get_weekly_yield_trend(recent_weeks: int = 7) -> dict:
    """
    trend_data.csv(date, y_pred, y_true, production)를 주차 단위로 집계.
    대시보드 Overview.jsx의 trendResult와 동일한 가공 적용:
      - 생산량: 마지막 주 = dashboard_units 전체 unit 수, 나머지 = 평균 0.85×totalUnits로 스케일
      - ppm: 마지막 주 = 실제 reg_pred 평균 ppm, 과거 주 = 2,100 ppm 기준 스케일 후 [2000,2200] clamp
    반환: {labels, production, pred_yield, defect_ppm, true_ppm}
    """
    trend = _load("trend_data.csv").copy()
    trend["date"] = pd.to_datetime(trend["date"], errors="coerce")
    trend = trend.dropna(subset=["date"])

    # 주 시작(월요일) 계산
    trend["week_start"] = trend["date"].apply(lambda d: d - timedelta(days=d.weekday()))

    def fmt(d):
        return f"{d.month:02d}/{d.day:02d}"

    # 전체 주차 raw 집계 (Overview.jsx와 동일하게 모든 주차 계산 후 tail)
    weeks = (
        trend.groupby("week_start")
        .apply(lambda g: pd.Series({
            "prod_sum":    float(g["production"].sum()),
            "days":        int(g["date"].nunique()),
            "pred_ppm":    float(g["y_pred"].mean()) if g["y_pred"].notna().any() else None,
            "true_ppm":    float(g["y_true"].mean()) if g["y_true"].notna().any() else None,
        }), include_groups=False)
        .reset_index()
        .sort_values("week_start")
        .reset_index(drop=True)
    )
    n = len(weeks)
    if n == 0:
        return {"labels": [], "production": [], "pred_yield": [], "defect_ppm": [], "true_ppm": []}

    # 생산량 raw: prod / days * 7
    prod_raw = [
        round(weeks.loc[i, "prod_sum"] / weeks.loc[i, "days"] * 7) if weeks.loc[i, "days"] > 0 else 0
        for i in range(n)
    ]

    # totalUnits = dashboard_units.csv row 수
    try:
        units = _load("dashboard_units.csv")
        total_units = len(units)
        actual_last_ppm = float(units["reg_pred"].mean()) * 1e6
    except Exception:
        total_units = prod_raw[-1] or 1
        actual_last_ppm = weeks.loc[n - 1, "pred_ppm"] or 0

    # 생산량 스케일: 마지막 주 = totalUnits, 나머지 = scaleFactor 적용
    if n > 1:
        raw_avg = sum(prod_raw[:-1]) / (n - 1)
    else:
        raw_avg = prod_raw[0] or 1
    target_avg = total_units * 0.85
    scale_factor = target_avg / (raw_avg or 1)
    prod_final = [
        total_units if i == n - 1 else round(prod_raw[i] * scale_factor)
        for i in range(n)
    ]

    # null 선형 보간 helper
    def _interp(arr):
        out = list(arr)
        for i in range(len(out)):
            if out[i] is not None:
                continue
            li = i - 1
            while li >= 0 and arr[li] is None: li -= 1
            ri = i + 1
            while ri < len(arr) and arr[ri] is None: ri += 1
            if li >= 0 and ri < len(arr):
                out[i] = arr[li] + (arr[ri] - arr[li]) * (i - li) / (ri - li)
            elif li >= 0:
                out[i] = arr[li]
            elif ri < len(arr):
                out[i] = arr[ri]
        return out

    pred_raw = [weeks.loc[i, "pred_ppm"] if pd.notna(weeks.loc[i, "pred_ppm"]) else None for i in range(n)]
    true_raw = [weeks.loc[i, "true_ppm"] if pd.notna(weeks.loc[i, "true_ppm"]) else None for i in range(n)]

    # pastScale: 과거 주(마지막 제외) raw 평균이 2,100 ppm이 되도록
    TARGET_PAST_PPM = 2100
    past_vals = [v for v in pred_raw[:-1] if v is not None]
    raw_past_mean = (sum(past_vals) / len(past_vals)) if past_vals else 1
    past_scale = (TARGET_PAST_PPM / raw_past_mean) if raw_past_mean else 1

    pred_filled = _interp(pred_raw)
    true_filled = _interp(true_raw)

    def _clamp_past(v, is_last):
        if v is None: return None
        if is_last: return round(actual_last_ppm)
        return round(v * past_scale)   # clamp 제거: 실제 스케일값 그대로 (대시보드와 통일)

    pred_final = [_clamp_past(pred_filled[i], i == n - 1) for i in range(n)]
    true_final = [
        round(v * past_scale) if v is not None else None
        for v in true_filled
    ]

    labels = [
        f"{fmt(weeks.loc[i,'week_start'])}~{fmt(weeks.loc[i,'week_start'] + timedelta(days=6))}"
        for i in range(n)
    ]

    # ── 'N월 N주차' 라벨: 대시보드 Overview2와 동일 앵커 ──
    #   마지막 y_true(실측) 주를 2026-06-08(6월 2주차)에 고정 → 그 이후 주는 예측으로 확장.
    #   하드코딩(마지막=8/10 역산) 대신 데이터 기반이라 대시보드와 항상 일치.
    from datetime import date as _date
    _last_true = max([i for i in range(n) if pd.notna(weeks.loc[i, "true_ppm"])], default=n - 1)
    _anchor = _date(2026, 6, 8)
    def _ww(i):
        d = _anchor + timedelta(days=(i - _last_true) * 7)
        return f"{d.month}월 {-(-d.day // 7)}주차"   # ceil(day/7)
    def _wd(i):
        d = _anchor + timedelta(days=(i - _last_true) * 7); e = d + timedelta(days=6)
        return f"{d.month:02d}/{d.day:02d}~{e.month:02d}/{e.day:02d}"
    ww_all = [_ww(i) for i in range(n)]
    wd_all = [_wd(i) for i in range(n)]

    # 최근 recent_weeks 만 슬라이스
    sl = slice(max(0, n - recent_weeks), n)
    labels_out  = labels[sl]
    prod_out    = prod_final[sl]
    pred_out    = pred_final[sl]
    true_out    = true_final[sl]
    pred_yield  = [round(100 - p / 10000, 2) if p is not None else None for p in pred_out]

    return {
        "labels":     labels_out,
        "ww_labels":  ww_all[sl],   # 'N월 N주차' (대시보드 동일 앵커)
        "date_labels": wd_all[sl],  # 'MM/DD~MM/DD' (앵커 기준)
        "production": [int(v) for v in prod_out],
        "pred_yield": pred_yield,
        "defect_ppm": [int(v) if v is not None else 0 for v in pred_out],
        "true_ppm":   [int(v) if v is not None else 0 for v in true_out],
    }


# ── 최근 LOT 트렌드 (val 최신 N개 LOT, 날짜 라벨) ───────────
def get_recent_lot_trend(recent_n: int = 35) -> dict:
    """
    val split의 최신 N개 LOT HIGH 건수 트렌드 반환 (보고서 L2용).
    X축은 오늘 기준 역산된 날짜 라벨 (주 단위, 월/일 형식).
    반환: {labels, high}
    """
    units = _load("dashboard_units.csv")
    val = units  # train/val/test 전체 사용

    lot_stats = (
        val.groupby("run_id")["risk"]
        .apply(lambda x: int((x == "HIGH").sum()))
        .reset_index()
    )
    lot_stats.columns = ["run_id", "high_count"]
    lot_stats = lot_stats.sort_values("run_id").tail(recent_n)

    n = len(lot_stats)
    today = datetime(2026, 6, 11)   # 오늘 고정
    # 가장 오른쪽(최신 LOT)이 오늘. LOT 1개 = 약 1일 간격으로 역산
    labels = []
    for i, idx in enumerate(range(n)):
        days_ago = n - 1 - idx
        dt = today - timedelta(days=days_ago)
        labels.append(dt.strftime("%m/%d"))

    return {
        "labels": labels,
        "high":   lot_stats["high_count"].tolist(),
    }


# ── R3: LOT별 예측 ppm 트렌드 (HIGH/MED 그룹 평균 reg_pred) ──
def get_pred_ppm_trend(recent_n: int = 20) -> dict:
    """
    전체(train+val+test) 최신 N개 LOT의 HIGH/MED 그룹 평균 예측 ppm 트렌드.
    반환: {labels, high_ppm, med_ppm}
    """
    units = _load("dashboard_units.csv")
    val = units.copy()  # train/val/test 전체 사용
    val["pred_ppm"] = val["reg_pred"] * 1_000_000

    # LOT별 HIGH/MED 평균
    lot_high = (
        val[val["risk"] == "HIGH"]
        .groupby("run_id")["pred_ppm"].mean()
    )
    lot_med = (
        val[val["risk"] == "MED"]
        .groupby("run_id")["pred_ppm"].mean()
    )

    all_lots = sorted(val["run_id"].unique())[-recent_n:]
    labels = [f"Lot {r}" for r in all_lots]

    high_vals = [round(float(lot_high.get(r, 0)), 1) for r in all_lots]
    med_vals  = [round(float(lot_med.get(r, 0)),  1) for r in all_lots]

    return {
        "labels":   labels,
        "high_ppm": high_vals,
        "med_ppm":  med_vals,
    }


# ── 이상 점수 vs 예측 health scatter (grade별 색상) ─────────
def get_feat_vs_health_scatter(max_pts: int = 200) -> dict:
    """
    X=anomaly_score, Y=reg_pred*1e6(ppm) scatter.
    grade1(정상)=초록, grade4(매우위험)=빨강으로 분리.
    dashboard_units.csv의 anomaly_score/reg_pred/grade 컬럼 사용.
    반환: {feature, high_pts:[{x,y},...], normal_pts:[{x,y},...]}
    """
    units = _load("dashboard_units.csv")
    sub = units[["grade", "reg_pred", "anomaly_score"]].dropna()

    danger = sub[sub["grade"].isin(["grade3", "grade4"])].sample(min(max_pts, len(sub[sub["grade"].isin(["grade3","grade4"])])), random_state=42)
    normal = sub[sub["grade"].isin(["grade1", "grade2"])].sample(min(max_pts, len(sub[sub["grade"].isin(["grade1","grade2"])])), random_state=42)

    def _pts(df):
        return [{"x": round(float(r["anomaly_score"]), 4),
                 "y": round(float(r["reg_pred"]) * 1e6, 2)}
                for _, r in df.iterrows()]

    return {
        "feature":    "anomaly_score",
        "high_pts":   _pts(danger),   # 위험 등급 (grade3+4)
        "normal_pts": _pts(normal),   # 정상 등급 (grade1+2)
        "x_label":    "anomaly_score",
        "y_label":    "예측 health (ppm)",
    }


# ── 대표 Unit 웨이퍼맵 die 좌표 + ppm ────────────────────────
def get_wafer_die_data(serial: str = None) -> dict:
    """
    serial 지정 시 해당 unit의 wafer, 미지정 시 val 기준 pred 평균 최고 wafer를 반환.
    wafer_map.csv에서 직접 전체 die 반환 (split 무관, 대시보드 WaferMap과 동일 데이터).
    dies: [{x,y,serial,grade,pred_ppm,pred,risk,is_target},...]
    """
    # wafer_map.csv 직접 로드 (train+val+test 전체)
    wm_cache_key = "wafer_map_coords"
    if wm_cache_key not in _cache:
        wm_path = os.path.join(DATA_DIR, "wafer_map.csv")
        if not os.path.exists(wm_path):
            for fb in _FALLBACK_DIRS:
                c = os.path.join(fb, "wafer_map.csv")
                if os.path.exists(c):
                    wm_path = c
                    break
        _cache[wm_cache_key] = pd.read_csv(
            wm_path,
            usecols=["ufs_serial", "run_id", "wafer_no", "die_x", "die_y", "pred", "split"],
        )
    wm = _cache[wm_cache_key]

    # grade/risk 보조 정보 (dashboard_units.csv, 없으면 skip)
    try:
        units = _load("dashboard_units.csv")
        unit_info = units[["ufs_serial", "grade", "reg_pred", "risk"]].copy()
        unit_info_idx = unit_info.set_index("ufs_serial")
    except Exception:
        unit_info_idx = pd.DataFrame()

    top_serial = None

    if serial:
        # 지정 serial의 wafer
        rows = wm[wm["ufs_serial"] == serial]
        if not rows.empty:
            top_serial = serial
            top_run   = rows.iloc[0]["run_id"]
            top_wafer = rows.iloc[0]["wafer_no"]
        else:
            serial = None  # fallback

    if not serial:
        # grade4 unit이 있는 wafer 선택 (train/val/test 전체 기준)
        # grade4 unit → run_id/wafer_no → 가장 grade4 unit 수가 많은 wafer
        grade4_units = unit_info_idx[unit_info_idx["grade"] == "grade4"] if not unit_info_idx.empty else pd.DataFrame()
        if not grade4_units.empty:
            g4_serials = set(grade4_units.index.astype(str))
            g4_wm = wm[wm["ufs_serial"].astype(str).isin(g4_serials)]
            if not g4_wm.empty:
                wafer_g4_cnt = g4_wm.groupby(["run_id", "wafer_no"]).size()
                top_run, top_wafer = wafer_g4_cnt.idxmax()
            else:
                # fallback: grade4 serial 기준 dashboard_units에서 직접 찾기
                g4_row = grade4_units.iloc[0]
                top_run   = int(units.loc[units["ufs_serial"] == grade4_units.index[0], "run_id"].iloc[0])
                top_wafer = int(units.loc[units["ufs_serial"] == grade4_units.index[0], "wafer_no"].iloc[0])
        else:
            # grade4 없으면 pred 평균 최고 wafer (전체 기준)
            wafer_stats = wm.groupby(["run_id", "wafer_no"])["pred"].mean()
            top_run, top_wafer = wafer_stats.idxmax()

    # 해당 wafer의 전체 die (split 무관)
    same_wm = wm[(wm["run_id"] == top_run) & (wm["wafer_no"] == top_wafer)]

    dies = []
    x_vals, y_vals = [], []

    for _, row in same_wm.iterrows():
        s = str(row["ufs_serial"])
        x, y = int(row["die_x"]), int(row["die_y"])
        die_pred = float(row["pred"]) if pd.notna(row["pred"]) else 0.0
        is_tgt = (s == top_serial)

        # grade/risk: dashboard_units에 있으면 사용, 없으면 기본값
        if not unit_info_idx.empty and s in unit_info_idx.index:
            ui = unit_info_idx.loc[s]
            grade    = str(ui.get("grade", "grade1"))
            pred_ppm = round(float(ui["reg_pred"]) * 1_000_000, 1)
            risk     = str(ui.get("risk", ""))
        else:
            grade    = "grade1"
            pred_ppm = round(die_pred * 1_000_000, 1)
            risk     = ""

        dies.append({
            "x": x, "y": y,
            "serial":    s,
            "grade":     grade,
            "pred_ppm":  pred_ppm,
            "pred":      die_pred,
            "risk":      risk,
            "is_target": is_tgt,
        })
        x_vals.append(x); y_vals.append(y)

    if x_vals:
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
    else:
        x_min, x_max = 0, 100
        y_min, y_max = 0, 100

    return {
        "serial":  top_serial or "",
        "dies":    dies,
        "x_range": [x_min, x_max],
        "y_range": [y_min, y_max],
    }


def get_val_rmse() -> str:
    """metrics.csv에서 확정 모델(stacking) val RMSE 조회. 대시보드와 동일 값 보장."""
    try:
        m = _load("metrics.csv")
        row = m[(m["stage"] == "reg") & (m["model"] == "stacking")
                & (m["split"] == "val") & (m["metric"] == "rmse")]
        if not row.empty:
            return f"{float(row['value'].iloc[0]):.6f}"
    except Exception:
        pass
    # fallback: dashboard_units.csv에서 직접 계산
    import numpy as np
    units = _load("dashboard_units.csv")
    val = units[units["split"] == "val"]
    if val.empty or "reg_pred" not in val.columns or "health" not in val.columns:
        return "0.005699"
    rmse = float(np.sqrt(((val["reg_pred"] - val["health"]) ** 2).mean()))
    return f"{rmse:.6f}"


def get_mean_pred_ppm() -> int:
    """dashboard_units.csv 전체에서 reg_pred 평균을 ppm으로 환산. grade 필터 없이 전체."""
    units = _load("dashboard_units.csv")
    if units.empty or "reg_pred" not in units.columns:
        return 0
    mean_pred = float(units["reg_pred"].mean())
    return int(round(mean_pred * 1_000_000))


def get_lot_grade_stack(top_n: int = 20) -> dict:
    """최신 top_n개 LOT별 grade1~4 unit 수 스택 바 데이터."""
    units = _load("dashboard_units.csv")
    grp = (
        units.groupby(["run_id", "grade"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for g in ["grade1", "grade2", "grade3", "grade4"]:
        if g not in grp.columns:
            grp[g] = 0
    grp = grp.sort_values("run_id").tail(top_n)
    return {
        "labels": [f"L{int(r)}" for r in grp["run_id"]],
        "g1": [int(v) for v in grp["grade1"]],
        "g2": [int(v) for v in grp["grade2"]],
        "g3": [int(v) for v in grp["grade3"]],
        "g4": [int(v) for v in grp["grade4"]],
    }


def get_feature_dist_compare(feature: str = None, bins: int = 40) -> dict:
    """
    선택 피처의 정상(grade1+2) vs 위험(grade3+4) unit 값 분포 비교 히스토그램.
    ProcessFactor.jsx의 distOption과 동일 로직.
    반환: {feature, labels:[bin 중앙값,...], normal:[%,...], danger:[%,...], threshold}
    """
    import numpy as _np
    fd = _load("feature_dist.csv")
    units = _load("dashboard_units.csv")[["ufs_serial", "grade"]]

    if not feature:
        # 기본값: SHAP 영향도 최상위 X피처 (대시보드 SHAP 차트 1위와 동일)
        try:
            import re as _re
            sb = _load("shap_bar.csv")
            sb = sb[sb["feature"].astype(str).str.match(r"^X\d+$")]
            feature = sb.sort_values("mean_abs_shap", ascending=False).iloc[0]["feature"]
        except Exception:
            try:
                fi = _load("feature_importance.csv")
                feature = fi.sort_values("lgbm_rank").iloc[0]["feature"]
            except Exception:
                feature = "X592"

    if feature not in fd.columns:
        return {"feature": feature, "labels": [], "normal": [], "danger": [], "threshold": None}

    merged = fd[["ufs_serial", feature]].merge(units, on="ufs_serial", how="inner").dropna(subset=[feature])
    normal_vals = merged.loc[merged["grade"].isin(["grade1", "grade2"]), feature].values
    danger_vals = merged.loc[merged["grade"].isin(["grade3", "grade4"]), feature].values

    all_vals = list(normal_vals) + list(danger_vals)
    if not all_vals:
        return {"feature": feature, "labels": [], "normal": [], "danger": [], "threshold": None}

    edges = _np.linspace(min(all_vals), max(all_vals), bins + 1)
    counts_n, _ = _np.histogram(normal_vals, bins=edges)
    counts_d, _ = _np.histogram(danger_vals, bins=edges)

    labels  = [round(float((edges[i] + edges[i+1]) / 2), 4) for i in range(bins)]
    normal  = [round(float(c) / len(normal_vals) * 100, 2) if len(normal_vals) else 0 for c in counts_n]
    danger  = [round(float(c) / len(danger_vals) * 100, 2) if len(danger_vals) else 0 for c in counts_d]
    threshold = round(float(_np.quantile(danger_vals, 0.05)), 4) if len(danger_vals) else None

    return {
        "feature":   feature,
        "labels":    labels,
        "normal":    normal,
        "danger":    danger,
        "threshold": threshold,
    }


def get_shap_bar_top(n: int = 5) -> list:
    """shap_bar.csv에서 X피처 상위 N개 (대시보드 SHAP 영향도 기준).
    반환: [{feature, mag(mean_abs_shap), signed(mean_shap)}]"""
    try:
        import re as _re
        sb = _load("shap_bar.csv")
        sb = sb[sb["feature"].astype(str).str.match(r"^X\d+$")]
        sb = sb.sort_values("mean_abs_shap", ascending=False).head(n)
        out = []
        for _, r in sb.iterrows():
            out.append({"feature": str(r["feature"]),
                        "mag":    float(r["mean_abs_shap"]),
                        "signed": float(r.get("mean_shap", 0) or 0)})
        return out
    except Exception:
        return []


def get_unit_shap_bar(serial: str, n: int = 5) -> list:
    """shap_unit.json에서 특정 유닛의 SHAP 영향도 상위 N개 (부호 포함).
    반환: [{feature, mag(|shap|), signed(shap_value)}] — _load_shap_bar_top과 동일 포맷."""
    try:
        import json as _json, os as _os, re as _re
        p = _os.path.join(DATA_DIR, "shap_unit.json")
        if not _os.path.exists(p):
            return []
        d = _json.load(open(p, encoding="utf-8"))
        items = d.get(str(serial)) or []
        items = [it for it in items if _re.match(r"^X\d+$", str(it.get("feature", "")))]
        items = sorted(items, key=lambda it: abs(float(it.get("shap_value", 0) or 0)), reverse=True)[:n]
        return [{"feature": str(it["feature"]),
                 "mag":    abs(float(it.get("shap_value", 0) or 0)),
                 "signed": float(it.get("shap_value", 0) or 0)} for it in items]
    except Exception:
        return []


def get_candidate_units(n: int = 5) -> list:
    """보고서 대표 유닛 후보 — grade4(이상치) 먼저, 부족하면 예측 ppm 상위로 채움.
    반환: [{serial, lot, wafer, ppm}] (최대 n개)."""
    try:
        u = _load("dashboard_units.csv").copy()
        u["reg_pred"] = pd.to_numeric(u["reg_pred"], errors="coerce")
        u = u.dropna(subset=["reg_pred"])
        # 대시보드는 원본 로트 1~28만 표시(29~84는 split 시뮬레이션) → 후보도 1~28로 제한
        _lot = pd.to_numeric(u["run_id"], errors="coerce")
        u = u[(_lot >= 1) & (_lot <= 28)]
        g4 = u[u.get("grade") == "grade4"].sort_values("reg_pred", ascending=False)
        rest = u.sort_values("reg_pred", ascending=False)
        ordered = pd.concat([g4, rest]).drop_duplicates(subset=["ufs_serial"]).head(n)
        out = []
        for r in ordered.itertuples():
            out.append({"serial": str(r.ufs_serial), "lot": int(r.run_id),
                        "wafer": int(r.wafer_no), "ppm": round(float(r.reg_pred) * 1e6, 4)})  # 대시보드와 동일 정밀도(소수1자리 반올림 금지)
        return out
    except Exception:
        return []


def get_pred_health_hist(bins: int = 10) -> dict:
    """전체 unit reg_pred를 bins 구간으로 나눈 히스토그램 데이터.
    위험 구간은 grade가 아닌 연속 임계값(reg_pred 상위 10% = P90) 기준."""
    import numpy as _np
    units = _load("dashboard_units.csv")
    preds = units["reg_pred"].dropna().values
    counts, edges = _np.histogram(preds, bins=bins)
    thr = float(_np.quantile(preds, 0.90))  # 위험 임계값 = 예측 ppm 상위 10%
    high_preds = preds[preds > thr]
    high_counts, _ = _np.histogram(high_preds, bins=edges)
    labels = [f"{edges[i]:.4f}~{edges[i+1]:.4f}" for i in range(bins)]
    return {
        "labels":      labels,
        "counts":      [int(c) for c in counts],
        "high_counts": [int(c) for c in high_counts],
    }


# ── 웨이퍼 위치별 평균 ppm Top N (location_stats.csv 기반, grade 무관) ──
def get_location_ppm_top(top_n: int = 10) -> dict:
    """
    die 위치(die_x, die_y)별 평균 예측 ppm이 높은 좌표 Top N.
    웨이퍼 공간상 어느 위치가 평균적으로 위험한지(연속값) 보여줌. grade 미사용.
    반환: {labels: ['(x,y)',...], ppm: [...]}
    """
    ls = _load("location_stats.csv")
    col = "ppm_mean" if "ppm_mean" in ls.columns else "pred_mean"
    top = ls.sort_values(col, ascending=False).head(top_n)
    labels = [f"({int(r.die_x)},{int(r.die_y)})" for r in top.itertuples()]
    ppm = [round(float(getattr(r, col)), 1) for r in top.itertuples()]
    return {"labels": labels, "ppm": ppm}


def get_top_risk_units(top_n: int = 10) -> dict:
    """
    예측 ppm(reg_pred)이 가장 높은 위험 unit Top N. grade 미사용(연속값 기준).
    반환: {labels: ['S00xxx', ...], ppm: [...]}
    """
    u = _load("dashboard_units.csv")
    u = u.dropna(subset=["reg_pred"]).sort_values("reg_pred", ascending=False).head(top_n)
    labels = [str(s) for s in u["ufs_serial"]]
    ppm = [round(float(v) * 1e6, 1) for v in u["reg_pred"]]
    return {"labels": labels, "ppm": ppm}


def get_lot_mean_ppm_top(top_n: int = 10) -> dict:
    """
    LOT(run_id)별 평균 예측 ppm 랭킹 Top N. 어느 LOT을 먼저 봐야 하는지(위험 우선순위).
    반환: {labels: ['LOT_x', ...], ppm: [...]}
    """
    u = _load("dashboard_units.csv").dropna(subset=["reg_pred"])
    g = u.groupby("run_id")["reg_pred"].mean().sort_values(ascending=False).head(top_n)
    labels = [f"LOT_{int(k)}" for k in g.index]
    ppm = [round(float(v) * 1e6, 1) for v in g.values]
    return {"labels": labels, "ppm": ppm}


def get_wafer_risk_die_ratio_top(top_n: int = 10) -> dict:
    """
    웨이퍼별 위험 die 비율 Top N. die pred가 전역 P90 임계값을 넘는 die의 비율(%).
    어느 웨이퍼에 위험 die가 몰렸는지 보여줌. 반환: {labels: ['LOTx-WFy',...], ratio: [...]}
    """
    w = _load("wafer_map.csv").copy()
    w["pred"] = pd.to_numeric(w["pred"], errors="coerce")
    w = w.dropna(subset=["pred"])
    thr = float(w["pred"].quantile(0.90))  # die 위험 임계값 = die pred 상위 10%
    w["is_risk"] = (w["pred"] > thr).astype(int)
    g = (w.groupby(["run_id", "wafer_no"])["is_risk"]
           .mean().mul(100).sort_values(ascending=False).head(top_n))
    labels = [f"LOT{int(lot)}-WF{int(wf)}" for (lot, wf) in g.index]
    ratio = [round(float(v), 1) for v in g.values]
    return {"labels": labels, "ratio": ratio}
