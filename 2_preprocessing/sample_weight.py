"""
Sample Weight 모듈.

LDS (Label Distribution Smoothing, Yang et al. ICML 2021 "Delving into Deep Imbalanced Regression"):
Y>0 안에서도 분포가 long-tail이라(대부분 작은 값, 드물게 큰 값) 그냥 학습하면 흔한 값에 치우친다.
y 분포를 KDE로 부드럽게 추정해 "희소한 y 구간일수록 큰 가중치"를 주어 모델이 꼬리 구간도 학습하게 만든다.

쓰는 곳:
- 회귀 학습에서 `model.fit(X, y, sample_weight=w)` (주로 Two-Stage의 Stage 2 = y>0 회귀)
- reg_level='position'(die-level 학습)이면 unit weight를 그 unit의 die들에 복제해 die-level weight로 반환
"""
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def compute_lds_weights(y_train, sigma=0.01,
                        min_weight=0.1, max_weight=10.0,
                        only_positive=True,
                        expand_to_die=False,
                        ys_train_df=None,
                        pos_data=None,
                        key_col='ufs_serial'):
    """
    LDS (Label Distribution Smoothing) sample_weight 계산.

    y 분포를 Gaussian KDE로 smoothing → density 추정 →
    w_i = 1 / density(y_i), min/max 클립 후 mean=1로 정규화.

    Parameters
    ----------
    y_train : np.ndarray (1D) or pd.Series
        unit-level target (health). len = n_units (~26K)
    sigma : float, default 0.01
        Gaussian kernel bandwidth. 너무 작으면 노이즈 민감, 너무 크면 smoothing 효과 약화.
        (y>0 평균이 ~0.0087 수준이라 0.005~0.02 권장)
    min_weight, max_weight : float
        가중치 클립 범위 (극단 희소 구간이 학습 폭주시키는 것 방지)
    only_positive : bool, default True
        True: y>0 샘플만 가중치 적용, y=0은 weight=1 (Stage 2 전용)
        False: 모든 샘플에 적용
    expand_to_die : bool, default False
        True면 die-level weight 반환 (reg_level='position' 전용).
        각 unit의 weight를 그 unit에 속한 die들에 복제.
        ys_train_df + pos_data 인자 필수 (ufs_serial 매핑용).
        False면 unit-level weight만 반환 (~26K).
    ys_train_df : DataFrame, required if expand_to_die=True
        ys['train'] — key_col, TARGET_COL 컬럼 포함
    pos_data : dict, required if expand_to_die=True
        {position: {'train': df, 'val': df, 'test': df}}
        die 순서 결정용 (position concat 기준)
    key_col : str, default 'ufs_serial'
        unit 식별 컬럼

    Returns
    -------
    weights : np.ndarray (1D)
        - expand_to_die=False: shape (n_units,)  — unit-level
        - expand_to_die=True:  shape (n_dies,)   — die-level (position concat 순)
    info : dict
        {'effective_sigma', 'n_positive', 'weight_min', 'weight_max',
         'weight_std', 'expanded', 'n_die' (expand_to_die=True일 때)}
    """
    y_train = np.asarray(y_train, dtype=float)
    weight_unit = np.ones_like(y_train)   # 기본 가중치 1 (가중치를 안 줄 샘플은 1로 남음)

    # only_positive면 y>0 샘플만 가중치 재분배 대상 (y=0은 weight=1 유지)
    if only_positive:
        mask = y_train > 0
        y_sub = y_train[mask]
    else:
        mask = np.ones_like(y_train, dtype=bool)
        y_sub = y_train

    # 가중치 줄 샘플이 너무 적으면(0~1개, unit test 등) KDE가 의미 없으니 전부 1로 반환
    if len(y_sub) < 2:
        info = {
            'effective_sigma': sigma,
            'note': 'too few positive samples',
            'expanded': False,
            'n_positive': int(mask.sum()),
        }
        if expand_to_die:
            assert ys_train_df is not None and pos_data is not None, \
                "expand_to_die=True면 ys_train_df, pos_data 필요"
            n_die = sum(len(pos_data[p]['train']) for p in sorted(pos_data.keys()))
            return np.ones(n_die), info
        return weight_unit, info

    # y_sub의 분포를 Gaussian KDE로 추정.
    # scipy의 bw_method는 std 대비 상대 bandwidth라, 절대값 sigma를 std로 나눠 넘긴다.
    kde = gaussian_kde(y_sub, bw_method=sigma / (y_sub.std() + 1e-12))
    density = kde(y_sub)                                  # 각 y값 위치에서의 추정 밀도

    w_sub = 1.0 / (density + 1e-12)                      # 희소 구간(밀도 낮음)일수록 큰 가중치
    w_sub = w_sub / w_sub.mean()                         # 평균 1로 정규화 (전체 손실 스케일 유지)
    w_sub = np.clip(w_sub, min_weight, max_weight)       # 극단적으로 큰/작은 가중치는 잘라냄
    w_sub = w_sub / w_sub.mean()                         # 클립 후 다시 평균 1로

    weight_unit[mask] = w_sub                            # y>0 자리에만 채워 넣음 (y=0은 1 그대로)

    info = {
        'effective_sigma': sigma,
        'n_positive': int(mask.sum()),
        'weight_min': float(w_sub.min()),
        'weight_max': float(w_sub.max()),
        'weight_std': float(w_sub.std()),
        'clip_low_ratio':  float((w_sub <= min_weight + 1e-8).mean()),   # 하한에 닿은 비율
        'clip_high_ratio': float((w_sub >= max_weight - 1e-8).mean()),   # 상한에 닿은 비율
        'expanded': False,
    }

    # --- die-level 확장 (reg_level='position'에서 die 단위로 학습할 때) ---
    if not expand_to_die:
        return weight_unit, info

    assert ys_train_df is not None, "expand_to_die=True면 ys_train_df 필요"
    assert pos_data is not None,    "expand_to_die=True면 pos_data 필요"

    ufs_key = ys_train_df[key_col].values
    assert len(ufs_key) == len(weight_unit), \
        (f"y_train 길이({len(weight_unit)}) != "
         f"ys_train_df 길이({len(ufs_key)})")
    # ufs_serial → weight 매핑 (unit ID는 유일하다고 가정)
    weight_series = pd.Series(weight_unit, index=ufs_key)

    # die 순서는 position pivot 데이터(_prepare_unit_data, reg_level='position')와 똑같이 맞춰야 한다.
    #   pos_data[1]['train'] → pos_data[2]['train'] → ... 순서로 concat된 die 배열의 순서대로 weight를 복제
    die_weights = []
    for pos in sorted(pos_data.keys()):
        ufs_in_pos = pos_data[pos]['train'][key_col].values
        die_weights.append(weight_series.loc[ufs_in_pos].values)

    weight_die = np.concatenate(die_weights)
    info['expanded'] = True
    info['n_die'] = len(weight_die)
    info['n_unit'] = len(weight_unit)
    return weight_die, info
