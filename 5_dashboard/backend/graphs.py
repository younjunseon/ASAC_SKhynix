"""
그래프 이미지 생성 및 캐시 관리.
data/graphs/ 폴더에 이미지가 있으면 재사용, 없으면 생성 후 저장.
"""
import os
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

# 한글 폰트
rcParams["font.family"] = "Malgun Gothic"
rcParams["axes.unicode_minus"] = False

DATA_DIR   = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
GRAPHS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "data", "graphs"))
os.makedirs(GRAPHS_DIR, exist_ok=True)

# 파일명 고정
PATH_SHAP_TREND   = os.path.join(GRAPHS_DIR, "shap_trend.png")
PATH_FEAT_DIST    = os.path.join(GRAPHS_DIR, "feature_dist.png")
PATH_FEAT_IMP     = os.path.join(GRAPHS_DIR, "feature_importance.png")
PATH_LOT_TREND    = os.path.join(GRAPHS_DIR, "lot_trend.png")

# 공통 스타일
BG      = "#FFFFFF"
C_HIGH  = "#EF4444"
C_MED   = "#3B82F6"
C_BAR   = "#3B82F6"
C_GRID  = "#E5E7EB"
C_TEXT  = "#1E293B"
C_SUB   = "#64748B"


def _fig_save(fig, path: str, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)


def _load(filename: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, filename))


# ── 1. SHAP Value Trend Graph ─────────────────────────────────
def get_shap_trend(importance_data: dict, force: bool = False) -> str:
    """
    feature importance gain 값을 SHAP Trend 스타일 수평 막대 그래프로 표현.
    반환: 이미지 파일 경로
    """
    if not force and os.path.exists(PATH_SHAP_TREND):
        return PATH_SHAP_TREND

    features = importance_data.get("features", [])
    if not features:
        return None

    names = [f["feature"] for f in features[:15]]
    gains = [f.get("lgbm_gain", 0) or 0 for f in features[:15]]

    # 내림차순 정렬
    paired = sorted(zip(gains, names), reverse=True)
    gains  = [p[0] for p in paired]
    names  = [p[1] for p in paired]

    fig, ax = plt.subplots(figsize=(5.5, 4.2), facecolor=BG)
    ax.set_facecolor(BG)

    max_g  = max(gains, default=1) or 1
    colors = [C_HIGH if g == max_g else C_BAR for g in gains]
    bars = ax.barh(names, gains, color=colors, height=0.65, zorder=3)

    # 값 레이블
    for bar, val in zip(bars, gains):
        ax.text(val + max_g * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}", va="center", ha="left",
                fontsize=8, color=C_TEXT)

    ax.set_xlabel("SHAP Gain", fontsize=9, color=C_SUB)
    ax.set_title("SHAP Value Trend  (Feature Importance)", fontsize=10,
                 fontweight="bold", color=C_TEXT, pad=8)
    ax.tick_params(axis="y", labelsize=8, colors=C_TEXT)
    ax.tick_params(axis="x", labelsize=8, colors=C_SUB)
    ax.xaxis.grid(True, color=C_GRID, linestyle="--", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()

    plt.tight_layout()
    _fig_save(fig, PATH_SHAP_TREND)
    return PATH_SHAP_TREND


# ── 2. 피처 분포 박스플롯 (HIGH vs MED) ───────────────────────
def get_feature_dist(analysis_data: dict, force: bool = False) -> str:
    """
    TOP 3 feature의 HIGH vs MED 분포를 박스플롯으로 표현.
    반환: 이미지 파일 경로
    """
    if not force and os.path.exists(PATH_FEAT_DIST):
        return PATH_FEAT_DIST

    top_features = analysis_data.get("top_features", [])
    compare_grp  = analysis_data.get("compare_group", "MED")
    if not top_features:
        return None

    # 실제 분포 데이터 로드
    try:
        units = _load("dashboard_units.csv")
        xs    = _load("compet_xs_data.csv")
        period_units = units[units["split"] == "val"][["ufs_serial", "risk"]]
        merged = xs.merge(period_units, on="ufs_serial", how="inner")
    except Exception:
        return None

    top3 = [f["feature"] for f in top_features[:3]
            if f["feature"] in merged.columns]
    if not top3:
        return None

    high_df = merged[merged["risk"] == "HIGH"]
    cmp_df  = merged[merged["risk"] == compare_grp]

    fig, axes = plt.subplots(1, len(top3), figsize=(5.5, 3.2), facecolor=BG)
    if len(top3) == 1:
        axes = [axes]

    for ax, feat in zip(axes, top3):
        h_vals = high_df[feat].dropna().values
        l_vals = cmp_df[feat].dropna().values

        bp = ax.boxplot(
            [h_vals, l_vals],
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color=C_SUB, linewidth=1),
            capprops=dict(color=C_SUB, linewidth=1),
            flierprops=dict(marker="o", markersize=2, alpha=0.3),
            zorder=3,
        )
        bp["boxes"][0].set_facecolor(C_HIGH)
        bp["boxes"][0].set_alpha(0.85)
        bp["fliers"][0].set_markerfacecolor(C_HIGH)
        bp["boxes"][1].set_facecolor(C_MED)
        bp["boxes"][1].set_alpha(0.85)
        bp["fliers"][1].set_markerfacecolor(C_MED)

        ax.set_facecolor(BG)
        ax.set_title(feat, fontsize=8, fontweight="bold", color=C_TEXT, pad=4)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["HIGH", compare_grp], fontsize=8, color=C_TEXT)
        ax.tick_params(axis="y", labelsize=7, colors=C_SUB)
        ax.yaxis.grid(True, color=C_GRID, linestyle="--", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(f"고위험 피처 분포  (HIGH vs {compare_grp})",
                 fontsize=10, fontweight="bold", color=C_TEXT, y=1.02)
    plt.tight_layout()
    _fig_save(fig, PATH_FEAT_DIST)
    return PATH_FEAT_DIST


# ── 3. 피처 임포턴스 바 차트 ──────────────────────────────────
def get_feature_importance(importance_data: dict, force: bool = False) -> str:
    """
    TOP 15 feature importance 수평 막대 그래프.
    반환: 이미지 파일 경로
    """
    if not force and os.path.exists(PATH_FEAT_IMP):
        return PATH_FEAT_IMP

    features = importance_data.get("features", [])
    if not features:
        return None

    names = [f["feature"] for f in features[:15]]
    gains = [f.get("lgbm_gain", 0) or 0 for f in features[:15]]
    ranks = list(range(1, len(names) + 1))

    fig, ax = plt.subplots(figsize=(5.5, 4.5), facecolor=BG)
    ax.set_facecolor(BG)

    bars = ax.barh(range(len(names)), gains, color=C_BAR,
                   height=0.65, zorder=3)

    # 순위 + feature명 레이블
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{r}  {n}" for r, n in zip(ranks, names)],
                       fontsize=8, color=C_TEXT)

    max_g2 = max(gains, default=1) or 1
    for bar, val in zip(bars, gains):
        ax.text(val + max_g2 * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}", va="center", ha="left",
                fontsize=7, color=C_TEXT)

    ax.set_xlabel("Gain", fontsize=9, color=C_SUB)
    ax.set_title("피처 임포턴스  Top 15", fontsize=10,
                 fontweight="bold", color=C_TEXT, pad=8)
    ax.tick_params(axis="x", labelsize=8, colors=C_SUB)
    ax.xaxis.grid(True, color=C_GRID, linestyle="--", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()

    plt.tight_layout()
    _fig_save(fig, PATH_FEAT_IMP)
    return PATH_FEAT_IMP


# ── 4. LOT 불량 트렌드 ────────────────────────────────────────
def get_lot_trend(scan_data: dict, force: bool = False) -> str:
    """
    LOT별 HIGH 위험 unit 수 막대 그래프.
    반환: 이미지 파일 경로
    """
    if not force and os.path.exists(PATH_LOT_TREND):
        return PATH_LOT_TREND

    try:
        units = _load("dashboard_units.csv")
        val_units = units[units["split"] == "val"]

        lot_high = (
            val_units.groupby("run_id")["risk"]
            .apply(lambda x: (x == "HIGH").sum())
            .reset_index()
        )
        lot_high.columns = ["run_id", "high_count"]
        lot_high = lot_high.sort_values("run_id").tail(20)
    except Exception:
        return None

    fig, ax = plt.subplots(figsize=(6.0, 3.0), facecolor=BG)
    ax.set_facecolor(BG)

    top_lot = scan_data.get("top_lot")
    colors  = [C_HIGH if str(r) == str(top_lot) else C_BAR
               for r in lot_high["run_id"]]

    ax.bar(range(len(lot_high)), lot_high["high_count"],
           color=colors, width=0.7, zorder=3)

    ax.set_xticks(range(len(lot_high)))
    ax.set_xticklabels(
        [f"LOT_{r}" for r in lot_high["run_id"]],
        rotation=45, ha="right", fontsize=7, color=C_TEXT
    )
    ax.tick_params(axis="y", labelsize=8, colors=C_SUB)
    ax.set_ylabel("HIGH 위험 unit 수", fontsize=9, color=C_SUB)
    ax.set_title("LOT별 불량 트렌드  (Val 데이터 기준)",
                 fontsize=10, fontweight="bold", color=C_TEXT, pad=8)
    ax.yaxis.grid(True, color=C_GRID, linestyle="--", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 집중 LOT 강조 패치
    if top_lot is not None:
        high_patch = mpatches.Patch(color=C_HIGH, label=f"집중 LOT_{top_lot}")
        norm_patch = mpatches.Patch(color=C_BAR,  label="일반 LOT")
        ax.legend(handles=[high_patch, norm_patch],
                  fontsize=8, frameon=False)

    plt.tight_layout()
    _fig_save(fig, PATH_LOT_TREND)
    return PATH_LOT_TREND


# ── 공개 API: 전체 그래프 일괄 생성 ──────────────────────────
def ensure_all_graphs(report_data: dict, force: bool = False) -> dict:
    """
    필요한 그래프를 모두 생성(또는 캐시에서 로드)하여 경로 dict 반환.
    force=True 이면 기존 캐시 무시하고 재생성.
    """
    importance = report_data.get("importance", {})
    analysis   = report_data.get("analysis",   {})
    scan       = report_data.get("scan",        {})

    return {
        "shap_trend":         get_shap_trend(importance, force=force),
        "feature_dist":       get_feature_dist(analysis,  force=force),
        "feature_importance": get_feature_importance(importance, force=force),
        "lot_trend":          get_lot_trend(scan, force=force),
    }
