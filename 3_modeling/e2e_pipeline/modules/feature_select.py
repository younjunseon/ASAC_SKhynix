"""
경량 Feature Selection — E2E HPO trial 내부용

Boruta/Null Importance 등은 trial당 수십 분이므로 사용 불가.
LightGBM importance (gain) 기반 top-K 선택만 수행한다.
"""
import numpy as np
import lightgbm as lgb

from utils.config import SEED
from .model_zoo import DEVICE


def select_top_k(X_train, y_train, feat_cols, top_k, lgbm_params=None):
    """
    LightGBM importance (gain) 기준 상위 K개 피처 선택

    Parameters
    ----------
    X_train : ndarray, shape (n_samples, n_features)
    y_train : ndarray, shape (n_samples,)
    feat_cols : list
        피처 컬럼명 (X_train 컬럼 순서와 일치해야 함)
    top_k : int
        선택할 피처 수
    lgbm_params : dict, optional
        LightGBM 파라미터. None이면 기본값 사용

    Returns
    -------
    selected_cols : list
        선택된 피처 컬럼명
    importances : ndarray
        전체 피처의 importance 값
    """
    if lgbm_params is None:
        lgbm_params = dict(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
            device=DEVICE,
        )

    lgbm_params["importance_type"] = "gain"
    model = lgb.LGBMRegressor(**lgbm_params)
    model.fit(X_train, y_train)

    importances = model.feature_importances_  # gain 기반
    top_k = min(top_k, len(feat_cols))
    top_indices = np.argsort(importances)[::-1][:top_k]

    selected_cols = [feat_cols[i] for i in top_indices]
    return selected_cols, importances


def remove_zero_variance(X_train, feat_cols):
    """
    분산 0인 피처 제거

    Parameters
    ----------
    X_train : ndarray, shape (n_samples, n_features)
    feat_cols : list

    Returns
    -------
    filtered_cols : list
        분산 > 0인 피처 컬럼명
    mask : ndarray
        True인 위치가 유지된 피처
    """
    var = np.var(X_train, axis=0)
    mask = var > 0
    filtered_cols = [feat_cols[i] for i, keep in enumerate(mask) if keep]
    return filtered_cols, mask
