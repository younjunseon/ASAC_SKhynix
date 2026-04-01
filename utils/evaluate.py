"""
모델 평가 및 예측 후처리
"""
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import root_mean_squared_error as _sklearn_rmse
except ImportError:
    from sklearn.metrics import mean_squared_error as _mse
    def _sklearn_rmse(y_true, y_pred):
        return _mse(y_true, y_pred, squared=False)


def rmse(y_true, y_pred):
    """
    RMSE (Root Mean Squared Error) 계산

    Parameters
    ----------
    y_true : array-like
        실제값
    y_pred : array-like
        예측값

    Returns
    -------
    float
    """
    return _sklearn_rmse(y_true, y_pred)


def postprocess(pred):
    """
    예측값 후처리: 음수를 0으로 클리핑 (health는 0 이상)

    Parameters
    ----------
    pred : array-like
        모델 예측값

    Returns
    -------
    ndarray
        0 이상으로 클리핑된 예측값
    """
    return np.clip(pred, 0, None)


def evaluate(y_true, y_pred, label="", clip=True):
    """
    RMSE 계산 + 요약 출력

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    label : str
        모델/실험 이름
    clip : bool
        True면 음수 클리핑 후 평가

    Returns
    -------
    float : RMSE
    """
    if clip:
        y_pred = postprocess(y_pred)

    score = rmse(y_true, y_pred)
    n_zero_true = (np.array(y_true) == 0).sum()
    n_total = len(y_true)

    prefix = f"[{label}] " if label else ""
    print(f"{prefix}RMSE = {score:.6f}  "
          f"(n={n_total:,}, zero={n_zero_true:,}({n_zero_true/n_total*100:.1f}%))")
    return score


def compare_models(results_dict, y_true):
    """
    여러 모델 예측 결과를 한 번에 비교

    Parameters
    ----------
    results_dict : dict
        {모델명: 예측값 array}
    y_true : array-like

    Returns
    -------
    DataFrame : 모델별 RMSE 정렬 표
    """
    rows = []
    for name, y_pred in results_dict.items():
        score = rmse(y_true, postprocess(y_pred))
        rows.append({"model": name, "rmse": score})

    df = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    df.index += 1
    print(df.to_string())
    return df