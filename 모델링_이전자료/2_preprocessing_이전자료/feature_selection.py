"""
Feature Selection 모듈
- Boruta (RF 기반 통계적 선별)
- LightGBM Importance (gain 기반)
- Null Importance (셔플 비교로 noise 탐지)
- RFE (Recursive Feature Elimination)
- Permutation Importance (모델 무관 검증)
- Mutual Information (비선형 상관 포착)
- 투표 기반 최종 선정

CLAUDE.md 전략:
- Step 1 (사전 필터링) → cleaning.py에서 완료
- Step 2 (Boruta) + Step 3 (LightGBM + Null Importance) → 이 모듈
- Step 4 (Permutation Importance) → 검증용
- 투표: 방법들 중 min_votes 이상에서 선택된 feature만 채택
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from utils.config import SEED


def _detect_device():
    """GPU 사용 가능 여부 감지 → 'gpu' 또는 'cpu' 반환"""
    try:
        import torch
        if torch.cuda.is_available():
            return "gpu"
    except ImportError:
        pass
    try:
        # Colab 등에서 nvidia-smi 확인
        import subprocess
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=5,
        )
        if result.returncode == 0:
            return "gpu"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "cpu"


DEVICE = _detect_device()


def _coerce_xy(X_train, y_train, feat_cols):
    """
    feature_selection 함수들의 공통 입력 정규화.

    - X_train이 DataFrame이면 feat_cols만 추출, 아니면 feat_cols를 열 이름으로 붙여 래핑
    - y_train이 Series면 .values로, 아니면 np.array로 변환
    """
    if hasattr(X_train, "columns"):
        X = X_train[feat_cols]
    else:
        X = pd.DataFrame(X_train, columns=feat_cols)

    if hasattr(y_train, "values"):
        y = y_train.values
    else:
        y = np.array(y_train)
    return X, y


def _default_lgbm_params(seed=SEED):
    """LightGBM 기본 파라미터 (GPU 자동 감지 포함)"""
    return dict(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
        device=DEVICE,
    )


def select_by_boruta(X_train, y_train, feat_cols,
                     max_iter=100, max_depth=7, perc=80,
                     sample_n=None, seed=SEED):
    """
    Boruta: RF 기반 통계적 feature selection
    Shadow feature(셔플된 복사본)와 비교하여 유의미한 feature만 선별

    Parameters
    ----------
    X_train : DataFrame
    y_train : Series or ndarray
    feat_cols : list
    max_iter : int
        Boruta 최대 반복 횟수 (100~200 권장)
    max_depth : int
        RF max_depth
    perc : int
        Shadow feature 임계 백분위수 (80~100, 높을수록 엄격)
    sample_n : int or None
        서브샘플 수 (None이면 전체 사용)
    seed : int

    Returns
    -------
    selected : list of str
    report : dict
    """
    from boruta import BorutaPy

    X, y = _coerce_xy(X_train, y_train, feat_cols)

    # 서브샘플링 (속도)
    if sample_n and sample_n < len(X):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), sample_n, replace=False)
        X_sub = X.iloc[idx]
        y_sub = y[idx]
    else:
        X_sub = X
        y_sub = y

    # LGBMRegressor 사용 (GPU 자동 감지, sklearn RF 대비 대폭 빠름)
    estimator = lgb.LGBMRegressor(
        n_estimators=500,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
        device=DEVICE,
    )
    boruta = BorutaPy(
        estimator, n_estimators='auto', max_iter=max_iter,
        perc=perc, random_state=seed, verbose=0,
    )

    print(f"[Boruta] max_iter={max_iter}, max_depth={max_depth}, perc={perc}, "
          f"device={DEVICE}, n_samples={len(X_sub):,}")
    boruta.fit(X_sub.values, y_sub)

    confirmed = [feat_cols[i] for i in range(len(feat_cols)) if boruta.support_[i]]
    tentative = [feat_cols[i] for i in range(len(feat_cols)) if boruta.support_weak_[i]]

    # confirmed + tentative 모두 포함
    selected = confirmed + tentative

    print(f"  확정(confirmed): {len(confirmed)}개")
    print(f"  미결(tentative): {len(tentative)}개")
    print(f"  선택 합계: {len(selected)}개 / {len(feat_cols)}개")

    report = {
        "confirmed": confirmed,
        "tentative": tentative,
        "ranking": boruta.ranking_,
    }
    return selected, report


def select_by_lgbm_importance(X_train, y_train, feat_cols,
                               importance_type="gain",
                               top_k=None,
                               threshold=0,
                               lgbm_params=None,
                               seed=SEED):
    """
    LightGBM feature importance 기반 selection

    Parameters
    ----------
    X_train : DataFrame
    y_train : Series
    feat_cols : list
    importance_type : str
        "gain" or "split"
    top_k : int or None
        상위 K개 선택. None이면 threshold 기준
    threshold : float
        importance > threshold인 feature만 선택
    lgbm_params : dict or None
        LightGBM 파라미터. None이면 기본값 사용
    seed : int

    Returns
    -------
    selected : list of str
    report : dict
    """
    if lgbm_params is None:
        lgbm_params = _default_lgbm_params(seed)

    X, y = _coerce_xy(X_train, y_train, feat_cols)

    model = lgb.LGBMRegressor(**lgbm_params, importance_type=importance_type)
    model.fit(X, y)

    imp = model.feature_importances_.astype(float)
    imp_df = pd.DataFrame({
        "feature": feat_cols,
        "importance": imp,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    if top_k is not None:
        selected = imp_df.head(top_k)["feature"].tolist()
        print(f"[LightGBM Importance] type={importance_type}, top_k={top_k}")
    else:
        selected = imp_df[imp_df["importance"] > threshold]["feature"].tolist()
        print(f"[LightGBM Importance] type={importance_type}, threshold={threshold}")

    zero_imp = (imp_df["importance"] == 0).sum()
    print(f"  전체: {len(feat_cols)}개, importance=0: {zero_imp}개")
    print(f"  선택: {len(selected)}개")

    report = {"importance_df": imp_df}
    return selected, report


def select_by_null_importance(X_train, y_train, feat_cols,
                               n_runs=10, threshold=2.0,
                               lgbm_params=None,
                               seed=SEED):
    """
    Null Importance: target을 셔플한 상태에서의 importance와 비교
    z_score = (real_imp - mean(null_imp)) / std(null_imp) > threshold

    Parameters
    ----------
    X_train : DataFrame
    y_train : Series
    feat_cols : list
    n_runs : int
        셔플 반복 횟수 (10~20 권장)
    threshold : float
        z-score 기준 (2.0 = 상위 ~2.3%)
    lgbm_params : dict or None
    seed : int

    Returns
    -------
    selected : list of str
    report : dict
    """
    if lgbm_params is None:
        lgbm_params = _default_lgbm_params(seed)

    X, y = _coerce_xy(X_train, y_train, feat_cols)

    # 1. Real importance
    model = lgb.LGBMRegressor(**lgbm_params)
    model.fit(X, y)
    real_imp = model.feature_importances_.astype(float)

    # 2. Null importance (n_runs times)
    print(f"[Null Importance] n_runs={n_runs}, threshold={threshold}")
    null_imps = np.zeros((n_runs, len(feat_cols)))
    rng = np.random.RandomState(seed)

    for i in range(n_runs):
        y_shuffled = rng.permutation(y)
        model_null = lgb.LGBMRegressor(**lgbm_params)
        model_null.fit(X, y_shuffled)
        null_imps[i] = model_null.feature_importances_
        print(f"  run {i+1}/{n_runs} 완료", end="\r")
    print()

    # 3. Z-score 계산
    null_mean = null_imps.mean(axis=0)
    null_std = null_imps.std(axis=0)
    null_std[null_std == 0] = 1e-10  # div-by-zero 방지
    z_scores = (real_imp - null_mean) / null_std

    score_df = pd.DataFrame({
        "feature": feat_cols,
        "real_imp": real_imp,
        "null_mean": null_mean,
        "null_std": null_std,
        "z_score": z_scores,
    }).sort_values("z_score", ascending=False).reset_index(drop=True)

    selected = score_df[score_df["z_score"] > threshold]["feature"].tolist()

    print(f"  z_score > {threshold}: {len(selected)}개 / {len(feat_cols)}개")

    report = {"score_df": score_df}
    return selected, report


def select_by_rfe(X_train, y_train, feat_cols,
                  n_features_to_select=100,
                  step=50,
                  lgbm_params=None,
                  seed=SEED):
    """
    RFE (Recursive Feature Elimination): 반복적으로 중요도 낮은 feature 제거

    Parameters
    ----------
    X_train : DataFrame
    y_train : Series or ndarray
    feat_cols : list
    n_features_to_select : int
        최종 선택할 feature 수
    step : int
        매 반복에서 제거할 feature 수
    lgbm_params : dict or None
    seed : int

    Returns
    -------
    selected : list of str
    report : dict
    """
    from sklearn.feature_selection import RFE

    if lgbm_params is None:
        lgbm_params = _default_lgbm_params(seed)

    X, y = _coerce_xy(X_train, y_train, feat_cols)

    estimator = lgb.LGBMRegressor(**lgbm_params)
    n_select = min(n_features_to_select, len(feat_cols))
    step_val = min(step, len(feat_cols) - n_select) if len(feat_cols) > n_select else 1

    print(f"[RFE] n_features={n_select}, step={step_val}")
    rfe = RFE(estimator, n_features_to_select=n_select, step=step_val)
    rfe.fit(X, y)

    selected = [feat_cols[i] for i in range(len(feat_cols)) if rfe.support_[i]]
    print(f"  선택: {len(selected)}개 / {len(feat_cols)}개")

    report = {"ranking": rfe.ranking_}
    return selected, report


def select_by_permutation(X_train, y_train, feat_cols,
                          threshold=0,
                          n_repeats=5,
                          lgbm_params=None,
                          seed=SEED):
    """
    Permutation Importance: 모델 무관 feature 기여도 검증
    CLAUDE.md Step 4 — validation set 기준 측정으로 과적합 피처 탐지

    Parameters
    ----------
    X_train : DataFrame
    y_train : Series or ndarray
    feat_cols : list
    threshold : float
        importance > threshold인 feature 선택 (0이면 기여하는 모든 feature)
    n_repeats : int
        셔플 반복 횟수
    lgbm_params : dict or None
    seed : int

    Returns
    -------
    selected : list of str
    report : dict
    """
    from sklearn.inspection import permutation_importance

    if lgbm_params is None:
        lgbm_params = _default_lgbm_params(seed)

    X, y = _coerce_xy(X_train, y_train, feat_cols)

    model = lgb.LGBMRegressor(**lgbm_params)
    model.fit(X, y)

    print(f"[Permutation Importance] n_repeats={n_repeats}, threshold={threshold}")
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=seed, n_jobs=-1,
        scoring="neg_root_mean_squared_error",
    )

    imp_df = pd.DataFrame({
        "feature": feat_cols,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    selected = imp_df[imp_df["importance_mean"] > threshold]["feature"].tolist()
    print(f"  선택: {len(selected)}개 / {len(feat_cols)}개")

    report = {"importance_df": imp_df}
    return selected, report


def select_by_mutual_info(X_train, y_train, feat_cols,
                          top_k=None, threshold=0,
                          seed=SEED):
    """
    Mutual Information: 비선형 상관 포착 (Pearson이 못 잡는 관계 탐지)
    EDA에서 max|r|=0.037로 선형 상관이 극도로 낮으므로 비선형 지표가 중요

    Parameters
    ----------
    X_train : DataFrame
    y_train : Series or ndarray
    feat_cols : list
    top_k : int or None
        상위 K개 선택. None이면 threshold 기준
    threshold : float
        MI > threshold인 feature 선택
    seed : int

    Returns
    -------
    selected : list of str
    report : dict
    """
    from sklearn.feature_selection import mutual_info_regression

    X, y = _coerce_xy(X_train, y_train, feat_cols)

    # NaN 처리 (MI는 결측 허용 안 함)
    X_filled = X.fillna(0) if hasattr(X, 'fillna') else np.nan_to_num(X)

    print(f"[Mutual Information] computing...")
    mi_scores = mutual_info_regression(X_filled, y, random_state=seed, n_jobs=-1)

    mi_df = pd.DataFrame({
        "feature": feat_cols,
        "mi_score": mi_scores,
    }).sort_values("mi_score", ascending=False).reset_index(drop=True)

    if top_k is not None:
        selected = mi_df.head(top_k)["feature"].tolist()
        print(f"[Mutual Information] top_k={top_k}")
    else:
        selected = mi_df[mi_df["mi_score"] > threshold]["feature"].tolist()
        print(f"[Mutual Information] threshold={threshold}")

    print(f"  선택: {len(selected)}개 / {len(feat_cols)}개")

    report = {"mi_df": mi_df}
    return selected, report


def select_by_voting(selections, feat_cols, min_votes=2):
    """
    여러 방법의 결과를 투표로 결합

    Parameters
    ----------
    selections : dict
        {method_name: selected_features_list}
    feat_cols : list
    min_votes : int
        최소 투표 수 (이 이상 방법에서 선택된 feature만 채택)

    Returns
    -------
    final_selected : list of str
    vote_df : DataFrame
    """
    from collections import Counter

    vote_counter = Counter()
    for method, selected in selections.items():
        for feat in selected:
            vote_counter[feat] += 1

    vote_df = pd.DataFrame([
        {"feature": f, "votes": vote_counter.get(f, 0)}
        for f in feat_cols
    ]).sort_values("votes", ascending=False).reset_index(drop=True)

    for method, selected in selections.items():
        vote_df[method] = vote_df["feature"].isin(selected).astype(int)

    final_selected = vote_df[vote_df["votes"] >= min_votes]["feature"].tolist()

    print(f"\n{'='*50}")
    print(f"[투표 기반 선정] min_votes={min_votes}/{len(selections)}")
    for method, selected in selections.items():
        print(f"  {method}: {len(selected)}개")
    print(f"  최종 선택: {len(final_selected)}개 / {len(feat_cols)}개")
    print(f"{'='*50}")

    return final_selected, vote_df


def run_feature_selection(X_train, y_train, feat_cols,
                          methods=None,
                          min_votes=2,
                          boruta_params=None,
                          lgbm_params=None,
                          null_params=None,
                          rfe_params=None,
                          perm_params=None,
                          mi_params=None,
                          sample_n=None,
                          seed=SEED):
    """
    Feature Selection 파이프라인 전체 실행

    Parameters
    ----------
    X_train : DataFrame
    y_train : Series or ndarray
    feat_cols : list
    methods : list of str
        사용할 방법 목록. 기본: ["boruta", "lgbm_importance", "null_importance"]
        지원: "boruta", "lgbm_importance", "null_importance",
              "rfe", "permutation", "mutual_info"
    min_votes : int
        투표 최소 수 (methods가 3개면 2 = 과반)
    boruta_params : dict or None
        select_by_boruta에 전달할 추가 파라미터
    lgbm_params : dict or None
        select_by_lgbm_importance에 전달할 추가 파라미터
    null_params : dict or None
        select_by_null_importance에 전달할 추가 파라미터
    rfe_params : dict or None
        select_by_rfe에 전달할 파라미터 (n_features_to_select, step)
    perm_params : dict or None
        select_by_permutation에 전달할 파라미터 (threshold, n_repeats)
    mi_params : dict or None
        select_by_mutual_info에 전달할 파라미터 (top_k, threshold)
    sample_n : int or None
        Boruta용 서브샘플 수 (None이면 전체)
    seed : int

    Returns
    -------
    selected_cols : list of str
        최종 선택된 feature 컬럼 리스트
    report : dict
        각 방법별 상세 결과
    """
    if methods is None:
        methods = ["boruta", "lgbm_importance", "null_importance"]

    print("=" * 60)
    print("Feature Selection 파이프라인 시작")
    print(f"입력 feature 수: {len(feat_cols)}")
    print(f"방법: {methods}, 최소 투표: {min_votes}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    selections = {}
    report = {"methods": methods}

    if "boruta" in methods:
        params = boruta_params or {}
        sel, rep = select_by_boruta(
            X_train, y_train, feat_cols,
            sample_n=sample_n, seed=seed, **params,
        )
        selections["boruta"] = sel
        report["boruta"] = rep

    if "lgbm_importance" in methods:
        params = lgbm_params or {}
        sel, rep = select_by_lgbm_importance(
            X_train, y_train, feat_cols,
            seed=seed, **params,
        )
        selections["lgbm_importance"] = sel
        report["lgbm_importance"] = rep

    if "null_importance" in methods:
        params = null_params or {}
        sel, rep = select_by_null_importance(
            X_train, y_train, feat_cols,
            seed=seed, **params,
        )
        selections["null_importance"] = sel
        report["null_importance"] = rep

    if "rfe" in methods:
        params = rfe_params or {}
        sel, rep = select_by_rfe(
            X_train, y_train, feat_cols,
            seed=seed, **params,
        )
        selections["rfe"] = sel
        report["rfe"] = rep

    if "permutation" in methods:
        params = perm_params or {}
        sel, rep = select_by_permutation(
            X_train, y_train, feat_cols,
            seed=seed, **params,
        )
        selections["permutation"] = sel
        report["permutation"] = rep

    if "mutual_info" in methods:
        params = mi_params or {}
        sel, rep = select_by_mutual_info(
            X_train, y_train, feat_cols,
            seed=seed, **params,
        )
        selections["mutual_info"] = sel
        report["mutual_info"] = rep

    # 투표
    if len(selections) == 1:
        method_name = list(selections.keys())[0]
        final_selected = selections[method_name]
        vote_df = None
        print(f"\n단일 방법 → 투표 불필요, 선택: {len(final_selected)}개")
    else:
        final_selected, vote_df = select_by_voting(
            selections, feat_cols, min_votes=min_votes,
        )

    report["selections"] = selections
    report["vote_df"] = vote_df
    report["final_selected"] = final_selected

    print(f"\nFeature Selection 완료: {len(feat_cols)} → {len(final_selected)}개")

    return final_selected, report
