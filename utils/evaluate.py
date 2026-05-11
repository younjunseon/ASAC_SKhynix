"""
모델 평가 및 예측 후처리.

대회 지표는 RMSE 하나뿐이라 여기 모인 함수도 RMSE 중심이다:
  - rmse: 버전에 상관없이 동작하는 RMSE 래퍼
  - postprocess: health는 0 이상이므로 음수 예측을 0으로 자르는 후처리
  - evaluate: RMSE 계산 + zero 비율까지 한 줄로 출력
  - compare_models: 여러 모델 예측을 RMSE 기준 정렬 표로
"""
import numpy as np
import pandas as pd

# sklearn 1.4+ 에는 root_mean_squared_error가 있고, 그 이전 버전엔 없다.
# 새 함수가 있으면 그대로 쓰고, 없으면 mean_squared_error(squared=False)로 같은 동작을 만든다.
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
    # np.clip(pred, 0, None): 하한 0, 상한 없음 → 음수만 0으로
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
        y_pred = postprocess(y_pred)   # 음수 예측을 0으로 — 보통 항상 켜고 평가

    score = rmse(y_true, y_pred)
    # zero-inflated 데이터라 정답 중 0의 비율을 같이 찍어 두면 결과 해석에 도움됨
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
        # 모든 모델에 동일하게 음수 clip 후처리를 적용해 공정 비교
        score = rmse(y_true, postprocess(y_pred))
        rows.append({"model": name, "rmse": score})

    df = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)   # RMSE 오름차순 (좋은 모델이 위)
    df.index += 1   # 1등부터 보이게 index를 1-base로
    print(df.to_string())
    return df
