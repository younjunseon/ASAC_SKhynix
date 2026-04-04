"""
EDA 모듈 5: Wafer Map 불량 시각화
노트북에서 import eda_wafer_map as wm 로 사용

run_wf_xy 포맷: "작업번호_웨이퍼번호_X_Y"
예시: "0000000_25_24_25" → lot=0000000, wafer=25, die_x=24, die_y=25
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False
from utils.config import KEY_COL, DIE_KEY_COL, TARGET_COL, SEED


def parse_wafer_coords(xs):
    """
    run_wf_xy를 파싱하여 lot, wafer_no, die_x, die_y, wafer_id 컬럼 추가

    Parameters
    ----------
    xs : DataFrame
        die-level 원본 데이터 (run_wf_xy 컬럼 필요, "작업번호_웨이퍼번호_X_Y" 형식)

    Returns
    -------
    DataFrame
        ufs_serial, run_wf_xy + lot, wafer_no, die_x, die_y, wafer_id 컬럼
    """
    parts = xs[DIE_KEY_COL].str.split('_', expand=True)
    result = xs[[KEY_COL, DIE_KEY_COL]].copy()
    result['lot'] = parts[0]
    result['wafer_no'] = parts[1].astype(int)
    result['die_x'] = parts[2].astype(int)
    result['die_y'] = parts[3].astype(int)
    result['wafer_id'] = result['lot'] + '_' + parts[1]  # lot + wafer 조합 = 고유 wafer 식별자

    print(f"파싱 완료: {len(result):,} dies")
    print(f"고유 lot 수: {result['lot'].nunique()}")
    print(f"고유 wafer 수: {result['wafer_id'].nunique()}")
    print(f"die_x 범위: {result['die_x'].min()} ~ {result['die_x'].max()}")
    print(f"die_y 범위: {result['die_y'].min()} ~ {result['die_y'].max()}")

    return result


def select_top_wafers(xs_parsed, ys_train, n=6):
    """
    불량률(health>0 비율)이 높은 wafer 상위 N장 선택

    Parameters
    ----------
    xs_parsed : DataFrame
        parse_wafer_coords()에서 반환된 파싱 결과
    ys_train : DataFrame
        train Y 데이터 (ufs_serial, health 컬럼)
    n : int
        선택할 wafer 수. 기본 6

    Returns
    -------
    list of str
        상위 N개 wafer_id
    """
    # die에 health merge (ufs_serial 기준)
    merged = xs_parsed.merge(ys_train[[KEY_COL, TARGET_COL]], on=KEY_COL, how='inner')

    # wafer별 불량률: health > 0인 die 비율
    wafer_stats = merged.groupby('wafer_id').agg(
        total_dies=('die_x', 'size'),
        defect_dies=(TARGET_COL, lambda x: (x > 0).sum()),
        mean_health=(TARGET_COL, 'mean'),
    ).reset_index()
    wafer_stats['defect_rate'] = (wafer_stats['defect_dies'] / wafer_stats['total_dies'] * 100).round(1)
    wafer_stats = wafer_stats.sort_values('defect_rate', ascending=False)

    print(f"\n전체 wafer 수 (train): {len(wafer_stats)}")
    print(f"\n불량률 상위 {n}장:")
    print(wafer_stats.head(n).to_string(index=False))

    top_wafers = wafer_stats.head(n)['wafer_id'].tolist()
    return top_wafers


def _draw_single_wafer(ax, wf, wafer_id):
    """
    웨이퍼 1장을 원형 형태로 시각화 — 각 die를 사각형 타일로 표현
    정상(Y=0): 하늘색, 불량(Y>0): 빨강 계열 (health 값에 따라 농도)
    x/y 범위가 달라도 Ellipse + aspect='auto'로 화면에서 원형으로 표시.

    Parameters
    ----------
    ax : matplotlib Axes
        그릴 subplot
    wf : DataFrame
        해당 웨이퍼의 die 데이터 (die_x, die_y, health 컬럼 필요)
    wafer_id : str
        "lot_waferNo" 형식의 웨이퍼 식별자
    """
    from matplotlib.patches import Ellipse, Rectangle
    from matplotlib.collections import PatchCollection

    # die 간격(step) 자동 계산
    dx_sorted = np.sort(wf['die_x'].unique())
    dy_sorted = np.sort(wf['die_y'].unique())
    step_x = np.min(np.diff(dx_sorted)) if len(dx_sorted) > 1 else 1
    step_y = np.min(np.diff(dy_sorted)) if len(dy_sorted) > 1 else 1
    gap = 0.06
    die_w = step_x * (1 - gap)
    die_h = step_y * (1 - gap)

    # 웨이퍼 중심 / x·y 반지름 (독립 계산)
    cx = (wf['die_x'].max() + wf['die_x'].min()) / 2
    cy = (wf['die_y'].max() + wf['die_y'].min()) / 2
    rx = (wf['die_x'].max() - wf['die_x'].min()) / 2 + step_x / 2
    ry = (wf['die_y'].max() - wf['die_y'].min()) / 2 + step_y / 2
    pad = 1.08  # 가장자리 die 사각형이 원 안에 들어오도록 여유

    # 웨이퍼 배경 (Ellipse — aspect='auto'에서 원으로 보임)
    ax.set_facecolor('#e8e8e8')
    wafer_bg = Ellipse((cx, cy), rx * 2 * pad, ry * 2 * pad,
                        fill=True, facecolor='#f8f8f8',
                        edgecolor='#444444', linewidth=2.5, zorder=0)
    ax.add_patch(wafer_bg)

    # 정상/불량 분리
    zero_mask = wf[TARGET_COL] == 0
    wf_zero = wf[zero_mask]
    wf_defect = wf[~zero_mask]

    # --- 정상 die: 사각형 타일 (하늘색) ---
    normal_patches = []
    for _, row in wf_zero.iterrows():
        rect = Rectangle((row['die_x'] - die_w / 2, row['die_y'] - die_h / 2),
                          die_w, die_h)
        normal_patches.append(rect)
    if normal_patches:
        pc_normal = PatchCollection(normal_patches, facecolor='#a8d8ea', edgecolor='#cccccc',
                                    linewidth=0.3, alpha=0.85, zorder=2)
        ax.add_collection(pc_normal)

    # --- 불량 die: 사각형 타일 (빨강 계열, health 값 기반 색상) ---
    if len(wf_defect) > 0:
        health_vals = wf_defect[TARGET_COL].values
        vmin = max(health_vals.min(), 1e-6)
        vmax = health_vals.max()
        if vmin >= vmax:
            norm = mcolors.Normalize(vmin=0, vmax=1)
        else:
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

        cmap = plt.cm.Reds
        defect_patches = []
        defect_colors = []
        for _, row in wf_defect.iterrows():
            rect = Rectangle((row['die_x'] - die_w / 2, row['die_y'] - die_h / 2),
                              die_w, die_h)
            defect_patches.append(rect)
            defect_colors.append(row[TARGET_COL])

        pc_defect = PatchCollection(defect_patches, cmap=cmap, norm=norm,
                                    edgecolor='#666666', linewidth=0.4,
                                    alpha=0.95, zorder=3)
        pc_defect.set_array(np.array(defect_colors))
        ax.add_collection(pc_defect)
        plt.colorbar(pc_defect, ax=ax, shrink=0.7, label='health', pad=0.02)

    # 웨이퍼 윤곽선 (상단 레이어)
    wafer_outline = Ellipse((cx, cy), rx * 2 * pad, ry * 2 * pad,
                            fill=False, edgecolor='#333333', linewidth=2.5, zorder=5)
    ax.add_patch(wafer_outline)

    # 축 설정 — x/y 독립 범위 + aspect='auto'로 화면에서 원형
    mx = rx * 0.08
    my = ry * 0.08
    ax.set_xlim(cx - rx * pad - mx, cx + rx * pad + mx)
    ax.set_ylim(cy - ry * pad - my, cy + ry * pad + my)
    ax.set_aspect('auto')
    ax.set_xlabel('die_x', fontsize=9)
    ax.set_ylabel('die_y', fontsize=9)

    # 제목 & 범례
    lot, wf_no = wafer_id.rsplit('_', 1)
    defect_rate = (1 - zero_mask.mean()) * 100
    n_defect = (~zero_mask).sum()
    ax.set_title(f'Lot {lot} / Wafer {wf_no}\n'
                 f'불량률 {defect_rate:.1f}% ({n_defect}/{len(wf)})',
                 fontsize=11, fontweight='bold', pad=10)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#a8d8ea', edgecolor='#cccccc', label='정상 (Y=0)'),
        Patch(facecolor='#e74c3c', edgecolor='#666666', label=f'불량 (Y>0, n={n_defect})'),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='upper left',
              framealpha=0.9, edgecolor='#cccccc')


def plot_wafer_map(xs_parsed, ys_train, wafers):
    """
    선택된 wafer들의 die 위치를 원형 웨이퍼 형태로 시각화.
    정상(Y=0): 하늘색, 불량(Y>0): 빨강 계열 (값에 따라 농도 변화)

    Parameters
    ----------
    xs_parsed : DataFrame
        parse_wafer_coords()에서 반환된 파싱 결과
    ys_train : DataFrame
        train Y 데이터 (ufs_serial, health 컬럼)
    wafers : list of str
        시각화할 wafer_id 리스트 (select_top_wafers() 반환값)
    """
    # die에 health merge
    merged = xs_parsed.merge(ys_train[[KEY_COL, TARGET_COL]], on=KEY_COL, how='inner')

    # subplot 레이아웃
    n = len(wafers)
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 7 * n_rows))
    if n == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    # 웨이퍼별 시각화
    for idx, wafer_id in enumerate(wafers):
        wf = merged[merged['wafer_id'] == wafer_id].copy()
        _draw_single_wafer(axes[idx], wf, wafer_id)

    # 남는 subplot 숨김
    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Wafer Map — 불량 공간 분포', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()
