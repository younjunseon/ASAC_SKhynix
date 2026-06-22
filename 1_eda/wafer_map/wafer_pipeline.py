"""
Wafer Map 전처리 + 스케일링 파이프라인 모듈

사용법:
    from wafer_pipeline import WaferPipeline

    ap = WaferPipeline(xs, feat_cols, impute='median', transform='auto')
    ap.plot('X22')
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# scaling.py는 같은 디렉토리(wafer_map/)에 있음
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scaling import scale as _scale


class WaferPipeline:
    """
    Wafer Map 전처리 + 스케일링 파이프라인

    Parameters
    ----------
    xs : DataFrame
        전체 die-level 데이터 (원본, 수정하지 않음)
    feat_cols : list
        feature 컬럼명 리스트
    impute : str
        결측치 처리 방법: 'mean', 'median'
    agg : str
        좌표별 집계 방법 (wafer map 시각화용): 'mean', 'median'
    transform : str or None
        스케일링 방법:
          'robust' → Robust 표준화 (median/IQR)
          'auto'   → 왜도 기반 자동 선택 (높으면 log, 낮으면 robust)
          None     → 변환 없음
    clip_pct : float or None
        이상치 clip 상한 percentile. None이면 clip 안함. 기본 99
    skew_threshold : float
        transform='auto'일 때 log/robust 분기 기준 왜도 절대값. 기본 2.0
    """

    IMPUTE_OPTIONS = ('mean', 'median')
    AGG_OPTIONS = ('mean', 'median')
    TRANSFORM_OPTIONS = ('robust', 'auto', None)

    def __init__(self, xs, feat_cols, impute='median', agg='mean',
                 transform='auto', clip_pct=99, skew_threshold=2.0):
        from utils.config import DIE_KEY_COL, SPLIT_COL

        assert impute in self.IMPUTE_OPTIONS, \
            f"impute must be one of {self.IMPUTE_OPTIONS}, got '{impute}'"
        assert agg in self.AGG_OPTIONS, \
            f"agg must be one of {self.AGG_OPTIONS}, got '{agg}'"
        assert transform in self.TRANSFORM_OPTIONS, \
            f"transform must be one of {self.TRANSFORM_OPTIONS}, got '{transform}'"

        self.impute = impute
        self.agg = agg
        self.transform = transform
        self.clip_pct = clip_pct
        self.skew_threshold = skew_threshold
        self.feat_cols = list(feat_cols)
        self._die_key_col = DIE_KEY_COL
        self._split_col = SPLIT_COL

        # train만 사용
        xs_train = xs[xs[SPLIT_COL] == 'train'].copy()

        # die 좌표 파싱
        split_parts = xs_train[DIE_KEY_COL].str.split('_', expand=True)
        xs_train['die_x'] = split_parts[2].astype(int)
        xs_train['die_y'] = split_parts[3].astype(int)

        # ── 1) 결측치 처리 ──
        xs_train = self._do_impute(xs_train)

        # ── 2) 이상치 clip ──
        if clip_pct is not None:
            upper_bounds = xs_train[self.feat_cols].quantile(clip_pct / 100)
            xs_train[self.feat_cols] = xs_train[self.feat_cols].clip(upper=upper_bounds, axis=1)

        # ── 3) 스케일링/변환 (scaling.py 사용) ──
        xs_train, _, self.transform_map = _scale(
            xs_train, self.feat_cols,
            transform=transform, skew_threshold=skew_threshold
        )

        # ── 4) 좌표별 집계 (wafer map용) ──
        self.coord_agg = self._do_agg(xs_train)

        # 레이블
        self._label = f"impute={impute}, agg={agg}, transform={transform}"
        if transform == 'auto':
            n_log = sum(1 for v in self.transform_map.values() if v == 'log')
            n_robust = sum(1 for v in self.transform_map.values() if v == 'robust')
            self._label += f" (log:{n_log}, robust:{n_robust}, threshold={skew_threshold})"

        print(f"[WaferPipeline] {self._label}")
        print(f"  좌표 수: {len(self.coord_agg)}, feature 수: {self.coord_agg.shape[1]}")

    # ── 내부 메서드 ──────────────────────────────────────────────

    def _do_impute(self, df):
        """결측치 처리 (train 통계 기반)"""
        fc = self.feat_cols

        if self.impute == 'mean':
            stats = df[fc].mean()
        else:  # median
            stats = df[fc].median()

        df[fc] = df[fc].fillna(stats)

        # 안전장치: 혹시 남은 결측
        remaining = df[fc].isnull().sum().sum()
        if remaining > 0:
            df[fc] = df[fc].fillna(df[fc].mean())

        return df

    def _do_agg(self, df):
        """좌표별 집계 (wafer map 시각화용)"""
        if self.agg == 'mean':
            return df.groupby(['die_x', 'die_y'])[self.feat_cols].mean()
        else:
            return df.groupby(['die_x', 'die_y'])[self.feat_cols].median()

    # ── 시각화 ──────────────────────────────────────────────────

    def plot(self, col_name, pct_lo=1, pct_hi=99, cmap='RdBu_r',
             figsize=(10, 8), ax=None):
        """
        단일 feature wafer map

        Returns
        -------
        im : QuadMesh
        """
        series = self.coord_agg[col_name]
        pivot = series.reset_index().pivot(
            index='die_y', columns='die_x', values=col_name)

        vmin = np.percentile(series.dropna(), pct_lo)
        vmax = np.percentile(series.dropna(), pct_hi)

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        im = ax.pcolormesh(
            pivot.columns, pivot.index, pivot.values,
            cmap=cmap, vmin=vmin, vmax=vmax, shading='auto',
        )

        # auto일 때 실제 적용된 변환 표시
        applied = self.transform_map.get(col_name, self.transform)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(f'{col_name}  [transform={applied}]\n({self._label})')
        ax.set_aspect('equal')
        ax.invert_yaxis()

        if standalone:
            plt.colorbar(im, ax=ax, label=col_name, shrink=0.8)
            plt.tight_layout()
            plt.show()

        return im

    def plot_grid(self, col_names, ncols=4, figsize_per=(4, 3.5), **kwargs):
        """여러 feature wafer map 격자"""
        n = len(col_names)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows))
        axes = np.array(axes).flatten()

        for i, col in enumerate(col_names):
            im = self.plot(col, ax=axes[i], **kwargs)
            plt.colorbar(im, ax=axes[i], shrink=0.7)

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(self._label, fontsize=13, y=1.01)
        plt.tight_layout()
        plt.show()

    def skew_summary(self):
        """
        transform='auto'일 때 feature별 왜도 및 적용된 변환 요약 출력
        """
        if self.transform != 'auto':
            print("transform='auto'일 때만 사용 가능합니다.")
            return

        rows = []
        for col, t in self.transform_map.items():
            rows.append({'feature': col, 'applied': t})
        df_summary = pd.DataFrame(rows)

        print(f"[skew_summary] threshold={self.skew_threshold}")
        print(df_summary['applied'].value_counts().to_string())
        return df_summary
