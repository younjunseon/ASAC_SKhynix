"""
EDA 모듈 2: Feature 품질 점검
노트북에서 import eda_feature_quality as fq 로 사용
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.config import SEED


def analyze_missing(xs, feat_cols):
    """
    Feature별 결측률 통계 출력 + 상위 30개 시각화

    Parameters
    ----------
    xs : DataFrame
        die-level 원본 데이터
    feat_cols : list of str
        feature 컬럼명 리스트

    Returns
    -------
    DataFrame
        결측이 있는 feature만 포함. 컬럼: missing_count, missing_pct
    """
    missing = xs[feat_cols].isnull().sum()
    missing_pct = (missing / len(xs) * 100).round(2)
    missing_df = pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct})
    missing_df = missing_df[missing_df['missing_count'] > 0].sort_values('missing_pct', ascending=False)

    print(f"전체 feature 수: {len(feat_cols)}")
    print(f"결측 있는 feature 수: {len(missing_df)}")
    print(f"결측 없는 feature 수: {len(feat_cols) - len(missing_df)}")

    if len(missing_df) > 0:
        print(f"\n결측 비율 상위 20개:")
        print(missing_df.head(20))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(missing_df['missing_pct'], bins=50, edgecolor='black', color='salmon')
        axes[0].set_title(f'Feature별 결측 비율 분포 (결측 있는 {len(missing_df)}개)')
        axes[0].set_xlabel('결측 비율 (%)')
        axes[0].set_ylabel('Feature 수')

        top30 = missing_df.head(30)
        axes[1].barh(range(len(top30)), top30['missing_pct'], color='salmon', edgecolor='black')
        axes[1].set_yticks(range(len(top30)))
        axes[1].set_yticklabels(top30.index, fontsize=7)
        axes[1].set_xlabel('결측 비율 (%)')
        axes[1].set_title('결측 비율 상위 30개 Feature')
        axes[1].invert_yaxis()

        plt.tight_layout()
        plt.show()
    else:
        print("결측치 없음")

    return missing_df


def classify_features(xs, feat_cols, threshold=20):
    """
    Feature를 연속형/이산형으로 분류 + 고유값 수 분포 시각화

    Parameters
    ----------
    xs : DataFrame
    feat_cols : list of str
    threshold : int
        고유값 수가 이 값 이하면 이산형으로 분류. 기본 20

    Returns
    -------
    continuous_feats : list of str
    discrete_feats : list of str
    """
    nunique = xs[feat_cols].nunique()

    discrete_feats = nunique[nunique <= threshold].index.tolist()
    continuous_feats = nunique[nunique > threshold].index.tolist()

    print(f"연속형 feature: {len(continuous_feats)}개")
    print(f"이산형 feature: {len(discrete_feats)}개 (고유값 수 {threshold})")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.hist(nunique.values, bins=100, edgecolor='black', color='steelblue')
    ax.set_title('Feature별 고유값 수 분포')
    ax.set_xlabel('고유값 수')
    ax.set_ylabel('Feature 수')
    ax.axvline(x=threshold, color='red', linestyle='--', label=f'이산형 기준 ({threshold})')
    ax.legend()
    plt.tight_layout()
    plt.show()

    if discrete_feats:
        print(f"\n이산형 Feature 고유값 수:")
        print(nunique[discrete_feats].sort_values())

    return continuous_feats, discrete_feats


def plot_continuous_dist(xs, continuous_feats):
    """
    연속형 feature 중 랜덤 12개를 골라 히스토그램 시각화

    Parameters
    ----------
    xs : DataFrame
    continuous_feats : list of str
        연속형 feature 컬럼명 리스트
    """
    np.random.seed(SEED)
    sample_cont = np.random.choice(continuous_feats, min(12, len(continuous_feats)), replace=False)

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()
    for i, col in enumerate(sample_cont):
        xs[col].dropna().hist(bins=50, ax=axes[i], edgecolor='black', color='steelblue', alpha=0.7)
        axes[i].set_title(col, fontsize=10)
        axes[i].tick_params(labelsize=8)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('연속형 Feature 분포 (랜덤 샘플 12개)', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_discrete_dist(xs, discrete_feats):
    """
    이산형 feature 최대 12개의 값 분포를 bar chart로 시각화

    Parameters
    ----------
    xs : DataFrame
    discrete_feats : list of str
        이산형 feature 컬럼명 리스트
    """
    if not discrete_feats:
        print("이산형 feature가 없습니다.")
        return

    sample_disc = discrete_feats[:min(12, len(discrete_feats))]
    n_plots = len(sample_disc)
    n_rows = (n_plots + 3) // 4

    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 3.5 * n_rows))
    axes = np.array(axes).flatten()
    for i, col in enumerate(sample_disc):
        xs[col].dropna().value_counts().sort_index().plot(kind='bar', ax=axes[i],
                                                          edgecolor='black', color='coral')
        axes[i].set_title(col, fontsize=10)
        axes[i].tick_params(labelsize=8)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('이산형 Feature 분포', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


def _find_low_var_features(xs, feat_cols):
    """
    상수/극저분산 feature 탐지 (std 기반)

    Parameters
    ----------
    xs : DataFrame
    feat_cols : list of str

    Returns
    -------
    const_feats : list of str
    low_var_feats : list of str
    feat_std : Series (이후 시각화에서 재사용)
    """
    feat_std = xs[feat_cols].std()
    const_feats = feat_std[feat_std == 0].index.tolist()
    low_var_feats = feat_std[feat_std < 1e-6].index.tolist()

    print(f"상수 feature (std=0): {len(const_feats)}개")
    print(f"극저분산 feature (std<1e-6): {len(low_var_feats)}개")

    if const_feats:
        print(f"\n상수 features: {const_feats[:20]}{'...' if len(const_feats) > 20 else ''}")

    return const_feats, low_var_feats, feat_std


def _find_duplicate_columns(xs, feat_cols):
    """
    완전 중복 컬럼 쌍 탐색 (샘플 5,000행 기반)

    Parameters
    ----------
    xs : DataFrame
    feat_cols : list of str

    Returns
    -------
    dup_pairs : list of tuple
        (col_a, col_b) 중복 쌍
    """
    print(f"\n중복 컬럼 탐색 (샘플 기반):")
    sample_xs = xs[feat_cols].sample(n=min(5000, len(xs)), random_state=SEED)
    dup_pairs = []
    cols_checked = sample_xs.columns.tolist()
    for i in range(len(cols_checked)):
        for j in range(i + 1, len(cols_checked)):
            if sample_xs[cols_checked[i]].equals(sample_xs[cols_checked[j]]):
                dup_pairs.append((cols_checked[i], cols_checked[j]))

    print(f"완전 중복 컬럼 쌍: {len(dup_pairs)}개")
    if dup_pairs:
        for pair in dup_pairs[:10]:
            print(f"  {pair[0]} == {pair[1]}")
        if len(dup_pairs) > 10:
            print(f"  ... 외 {len(dup_pairs) - 10}개")

    return dup_pairs


def _plot_variance_dist(feat_std):
    """
    Feature 표준편차 분포 히스토그램 (log10 스케일)

    Parameters
    ----------
    feat_std : Series
        feature별 std 값
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.hist(np.log10(feat_std[feat_std > 0].values + 1e-10), bins=100, edgecolor='black', color='steelblue')
    ax.set_title('Feature 표준편차 분포 (log10 스케일)')
    ax.set_xlabel('log10(std)')
    ax.set_ylabel('Feature 수')
    plt.tight_layout()
    plt.show()


def detect_low_variance(xs, feat_cols):
    """
    상수/극저분산/중복 컬럼 탐지 + Feature 분산 분포 시각화

    Parameters
    ----------
    xs : DataFrame
    feat_cols : list of str

    Returns
    -------
    const_feats : list of str
        std=0인 상수 feature
    low_var_feats : list of str
        std<1e-6인 극저분산 feature (상수 포함)
    dup_pairs : list of tuple
        완전 중복 컬럼 쌍 (col_a, col_b)
    """
    const_feats, low_var_feats, feat_std = _find_low_var_features(xs, feat_cols)
    dup_pairs = _find_duplicate_columns(xs, feat_cols)
    _plot_variance_dist(feat_std)
    return const_feats, low_var_feats, dup_pairs
