"""
EDA 모듈 1: 데이터 구조 & Target 분석
노트북에서 import eda_overview as ov 로 사용
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.config import (
    KEY_COL, DIE_KEY_COL, POSITION_COL, SPLIT_COL,
    TARGET_COL, META_COLS, SEED,
)


def describe_xs(xs, feat_cols):
    """
    Xs 기본 정보 출력 (shape, dtypes, 메타 컬럼 샘플, feature 기초통계)

    Parameters
    ----------
    xs : DataFrame
        die-level 원본 데이터
    feat_cols : list of str
        feature 컬럼명 리스트 (X0~X1086)
    """
    print(f"shape: {xs.shape}")
    print(f"\ndtypes:")
    print(xs.dtypes.value_counts())
    print(f"\n메타 컬럼 샘플:")
    print(xs[META_COLS].head(10))
    print(f"\nFeature 통계 (일부):")
    print(xs[feat_cols[:10]].describe().round(3))


def plot_dies_per_unit(xs):
    """
    Unit당 die 수 분포 + position 값 분포 시각화

    Parameters
    ----------
    xs : DataFrame
        die-level 원본 데이터 (ufs_serial, position 컬럼 필요)
    """
    dies_per_unit = xs.groupby(KEY_COL).size()
    print(dies_per_unit.describe())
    print(f"\n고유 unit 수: {xs[KEY_COL].nunique():,}")
    print(f"고유 die 수:  {xs[DIE_KEY_COL].nunique():,}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    dies_per_unit.hist(bins=30, ax=axes[0], edgecolor='black')
    axes[0].set_title('Unit당 Die 수 분포')
    axes[0].set_xlabel('Die 수')
    axes[0].set_ylabel('Unit 수')

    xs[POSITION_COL].value_counts().sort_index().plot(kind='bar', ax=axes[1], edgecolor='black')
    axes[1].set_title('Position 값 분포')
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('Die 수')

    plt.tight_layout()
    plt.show()


def print_split_ratio(xs, ys_all):
    """
    Xs(die level) / Ys(unit level) split별 비율 출력

    Parameters
    ----------
    xs : DataFrame
        die-level 데이터 (split 컬럼 필요)
    ys_all : DataFrame
        unit-level 전체 Y 데이터 (split 컬럼 필요)
    """
    print("Xs (die level)")
    xs_split = xs[SPLIT_COL].value_counts()
    print(xs_split)
    print((xs_split / xs_split.sum() * 100).round(1))

    print(f"\nYs (unit level)")
    ys_split = ys_all[SPLIT_COL].value_counts()
    print(ys_split)
    print((ys_split / ys_split.sum() * 100).round(1))


def plot_target_distribution(ys_all):
    """
    Target(health) 분포 시각화: 전체 히스토그램, Y>0 히스토그램, split별 zero 비율

    Parameters
    ----------
    ys_all : DataFrame
        전체 Y 데이터 (health, split 컬럼 필요)
    """
    print(ys_all[TARGET_COL].describe())

    zero_ratio = (ys_all[TARGET_COL] == 0).mean() * 100
    nonzero_ratio = (ys_all[TARGET_COL] > 0).mean() * 100
    print(f"\nY = 0 비율: {zero_ratio:.1f}%")
    print(f"Y > 0 비율: {nonzero_ratio:.1f}%")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 전체 분포 (0 포함)
    axes[0].hist(ys_all[TARGET_COL], bins=100, edgecolor='black', color='steelblue')
    axes[0].set_title('Health 전체 분포 (Zero-Inflated)')
    axes[0].set_xlabel('health')
    axes[0].set_ylabel('빈도')
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.7)

    # 0 제외 분포
    nonzero = ys_all[ys_all[TARGET_COL] > 0][TARGET_COL]
    axes[1].hist(nonzero, bins=100, edgecolor='black', color='coral')
    axes[1].set_title(f'Health > 0 분포 (n={len(nonzero):,})')
    axes[1].set_xlabel('health')
    axes[1].set_ylabel('빈도')

    # Zero vs Non-zero 비율 (split별)
    zero_by_split = ys_all.groupby(SPLIT_COL).apply(
        lambda g: pd.Series({
            'Y=0': (g[TARGET_COL] == 0).sum(),
            'Y>0': (g[TARGET_COL] > 0).sum()
        })
    )
    zero_by_split.plot(kind='bar', stacked=True, ax=axes[2], edgecolor='black')
    axes[2].set_title('Split별 Zero / Non-zero 비율')
    axes[2].set_xlabel('Split')
    axes[2].set_ylabel('Unit 수')
    axes[2].legend()
    axes[2].tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.show()
