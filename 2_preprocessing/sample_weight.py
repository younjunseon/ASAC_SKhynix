"""
Sample Weight 모듈 (2차 funnel 신규)

- LDS (Label Distribution Smoothing) — Yang et al., ICML 2021
  "Delving into Deep Imbalanced Regression"
- Y>0 내부 long-tail에 가중치 재분배 (Stage 2 회귀 전용)

사용 위치:
- 2차 앙상블 funnel의 회귀 학습에서 `model.fit(X, y, sample_weight=w)`
- `_run_preprocessing`에서 `expand_to_die=True`로 호출하면 die-level weight 반환

근거:
- 논문요약.md 8-1 (DIR, ICML 2021)
- IMDB-WIKI few-shot MAE 26.33 → 22.19 (15.6% 감소)
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
        EDA y>0 평균 0.0087 기준 0.005~0.02 권장.
    min_weight, max_weight : float
        가중치 클립 범위 (극단 희소 구간이 학습 폭주시키는 것 방지)
    only_positive : bool, default True
        True: y>0 샘플만 가중치 적용, y=0은 weight=1 (Stage 2 전용)
        False: 모든 샘플에 적용
    expand_to_die : bool, default False
        ★ True면 die-level weight 반환 (reg_level='position' 전용).
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
    weight_unit = np.ones_like(y_train)

    if only_positive:
        mask = y_train > 0
        y_sub = y_train[mask]
    else:
        mask = np.ones_like(y_train, dtype=bool)
        y_sub = y_train

    # 샘플 부족 방어 (unit test 등에서 0~1개인 경우)
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

    # Gaussian KDE: bw_method은 상대 bandwidth (kde.factor = bw_method)
    # sigma는 절대 값이지만 scipy.stats.gaussian_kde는 std 대비 비율로 받음
    kde = gaussian_kde(y_sub, bw_method=sigma / (y_sub.std() + 1e-12))
    density = kde(y_sub)

    w_sub = 1.0 / (density + 1e-12)
    w_sub = w_sub / w_sub.mean()                         # 1차 평균 1 정규화
    w_sub = np.clip(w_sub, min_weight, max_weight)       # 극단 클립
    w_sub = w_sub / w_sub.mean()                         # 클립 후 재정규화

    weight_unit[mask] = w_sub

    info = {
        'effective_sigma': sigma,
        'n_positive': int(mask.sum()),
        'weight_min': float(w_sub.min()),
        'weight_max': float(w_sub.max()),
        'weight_std': float(w_sub.std()),
        'clip_low_ratio':  float((w_sub <= min_weight + 1e-8).mean()),
        'clip_high_ratio': float((w_sub >= max_weight - 1e-8).mean()),
        'expanded': False,
    }

    # ── die-level 확장 (expand_to_die=True) ──
    if not expand_to_die:
        return weight_unit, info

    assert ys_train_df is not None, "expand_to_die=True면 ys_train_df 필요"
    assert pos_data is not None,    "expand_to_die=True면 pos_data 필요"

    ufs_key = ys_train_df[key_col].values
    assert len(ufs_key) == len(weight_unit), \
        (f"y_train 길이({len(weight_unit)}) != "
         f"ys_train_df 길이({len(ufs_key)})")
    # ufs_serial → weight 매핑 (중복 key 없다고 가정)
    weight_series = pd.Series(weight_unit, index=ufs_key)

    # die-level 순서: _prepare_unit_data(reg_level='position')와 동일
    #   pos_data[1]['train'] → pos_data[2]['train'] → ... 순으로 concat
    die_weights = []
    for pos in sorted(pos_data.keys()):
        ufs_in_pos = pos_data[pos]['train'][key_col].values
        die_weights.append(weight_series.loc[ufs_in_pos].values)

    weight_die = np.concatenate(die_weights)
    info['expanded'] = True
    info['n_die'] = len(weight_die)
    info['n_unit'] = len(weight_unit)
    return weight_die, info