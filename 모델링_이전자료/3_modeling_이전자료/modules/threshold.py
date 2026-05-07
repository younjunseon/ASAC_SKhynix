"""
CLF Threshold 튜닝 모듈

P(Y>0) < threshold인 샘플의 예측을 0으로 마스킹하여
RMSE를 최소화하는 최적 threshold를 탐색한다.

사용법:
    from modules.threshold import optimize_threshold

    best_thr, best_rmse, results = optimize_threshold(
        proba=unit_data['val']['clf_proba_mean'].values,
        reg_pred=best_result['val_pred'],
        y_true=y_val,
    )
"""
import numpy as np
from utils.evaluate import rmse, postprocess


def optimize_threshold(
    proba,
    reg_pred,
    y_true,
    search_range=(0.0, 0.95),
    n_points=200,
):
    """
    RMSE 기준으로 최적 CLF threshold를 그리드 서치

    원리:
    - P(Y>0) < threshold인 샘플의 예측을 0으로 마스킹
    - threshold가 높을수록 더 많은 샘플이 0으로 → zero-inflated 특성 활용
    - threshold=0이면 마스킹 없음 (기존과 동일)

    Parameters
    ----------
    proba : array-like
        각 샘플의 P(Y>0) 확률 (clf_proba_mean)
    reg_pred : array-like
        회귀 예측값 (이미 postprocess 된 상태)
    y_true : array-like
        실제 target 값
    search_range : tuple
        탐색 범위 (min, max)
    n_points : int
        그리드 포인트 수

    Returns
    -------
    best_threshold : float
        최적 threshold
    best_rmse : float
        최적 threshold에서의 RMSE
    results : dict
        {"thresholds": array, "rmses": array} — 시각화용
    """
    proba = np.asarray(proba)
    reg_pred = np.asarray(reg_pred)
    y_true = np.asarray(y_true)

    thresholds = np.linspace(search_range[0], search_range[1], n_points)
    rmses = np.zeros(n_points)

    for i, thr in enumerate(thresholds):
        masked_pred = reg_pred.copy()
        masked_pred[proba < thr] = 0.0
        rmses[i] = rmse(y_true, postprocess(masked_pred))

    best_idx = np.argmin(rmses)
    best_threshold = thresholds[best_idx]
    best_rmse_val = rmses[best_idx]

    # 상위 5개 출력
    top_indices = np.argsort(rmses)[:5]
    print("CLF Threshold 탐색 결과 (Top 5):")
    for idx in top_indices:
        print(f"  threshold={thresholds[idx]:.4f} -> RMSE={rmses[idx]:.6f}")

    return best_threshold, best_rmse_val, {
        "thresholds": thresholds,
        "rmses": rmses,
    }


def apply_threshold(proba, reg_pred, threshold):
    """
    주어진 threshold로 zero-masking 적용

    Parameters
    ----------
    proba : array-like
        P(Y>0) 확률
    reg_pred : array-like
        회귀 예측값
    threshold : float
        CLF threshold

    Returns
    -------
    masked_pred : ndarray
        마스킹된 예측값
    """
    proba = np.asarray(proba)
    reg_pred = np.asarray(reg_pred)
    masked = reg_pred.copy()
    masked[proba < threshold] = 0.0
    return postprocess(masked)