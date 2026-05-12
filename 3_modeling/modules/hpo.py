"""
HPO (Optuna) + Best Trial Refit + 산출물 저장.

이 모듈이 하는 일:
- die-level로 모델을 학습하되, objective는 unit-level RMSE다 (die 예측을 unit으로 집계한 뒤 RMSE).
- KFold split은 항상 unit ID 단위 — 같은 unit의 4 die가 train/val에 섞이면 leakage라서 절대 금지.
- run_hpo: 한 모델의 HP를 Optuna로 탐색 (objective = train OOF unit RMSE), 옵션으로 매 trial의 val/test RMSE도 기록.
- refit_best: best HP로 n_folds(기본 5)-fold 재학습 → die-level OOF / val / test 예측 (val·test는 fold 평균) 생성. ZITboost면 π·μ도 함께.
- save_artifacts: refit 결과를 디스크에 저장 (fold_models.pkl + best_params.json + die/unit CSV 6개), postprocess 튜닝도 옵션.
- run_clf_hpo / refit_clf_best / save_clf_artifacts: Two-Stage Stage 1(분류) 버전. objective = unit RMSE(unit평균확률 × E[Y|Y>0]).
- narrow_around / sample_from_space / enqueue_anchor: anchor(1차 best HP) 주변으로 좁힌 탐색 공간 만들기 + 첫 trial 강제 헬퍼.

부가 기능 키워드:
- extra_feature_* : reg 입력 X 뒤에 die-level 컬럼 1개를 더 붙임 (예: "1-π" 컬럼).
- multiplier_*    : 최종 예측을 reg_pred × multiplier 로 만듦 (정석 Two-Stage 경로 B에서 (1-π)를 곱함). objective RMSE도 곱셈 후 값으로 계산 → "HPO가 최적화하는 식 == 최종 제출 식".
- y_positive_only : fit 데이터에서 y==0 die를 제외 (Stage 2 회귀 = "Y>0만으로 학습 → E[Y|Y>0,x] 예측").
- target_transform_fn/inverse_fn : 학습은 변환 공간(예: log1p), 출력은 원본 공간. 단 objective/loss_function 문자열이 tweedie 계열이면 자동 OFF (tweedie가 이미 right-skew를 모델링 → log1p와 이중 변환 방지). zitboost는 내부가 ZI-Tweedie라도 그 키가 hp에 없어 자동 OFF 대상이 아니므로, 같이 쓸 땐 호출부에서 transform을 직접 끄거나 켜야 한다.

사용법
------
    res = hpo.run_hpo(
        xs_train, ys_train_unit, feat_cols,
        model_name='lgbm', n_trials=100, n_folds=5,
        study_name='exp', storage='sqlite:///...db',
    )
    study, best_params = res['study'], res['best_params']

    final = hpo.refit_best(
        xs_train, xs_val, xs_test, ys_train_unit, feat_cols,
        model_name='lgbm', best_params=best_params, n_folds=5,
    )
"""
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import KFold

from utils.config import SEED, KEY_COL, TARGET_COL

from . import models as _models
from . import scaler as _scaler


# ------------------------------------------------------------
# unit 단위 KFold split → die-level mask
# ------------------------------------------------------------

def _make_unit_folds(unit_ids, n_splits, seed=SEED):
    """unit ID 배열을 n_splits개 fold로 나눠 [(train_units, val_units), ...] 반환.

    여기서 나눈 건 unit이지 die가 아니다 — die mask는 _die_mask_from_units로 따로 만든다.
    """
    unique = np.asarray(unit_ids)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)   # seed 고정 → trial 간 동일 분할
    folds = []
    for tr_idx, vl_idx in kf.split(unique):
        folds.append((unique[tr_idx], unique[vl_idx]))
    return folds


def _die_mask_from_units(xs, unit_set):
    """xs(die-level)에서 ufs_serial이 unit_set에 들어 있는 행의 불리언 마스크."""
    return xs[KEY_COL].isin(unit_set).values


def _broadcast_y_to_die(xs, ys_unit):
    """unit-level y를 die-level로 펼침 — 각 die는 자기 unit의 health 값을 target으로 가짐 (xs 행 순서대로)."""
    y_map = ys_unit.set_index(KEY_COL)[TARGET_COL]
    return xs[KEY_COL].map(y_map).values.astype(float)


def _aggregate_die_to_unit(xs, die_pred):
    """die-level 예측 → unit-level (단순 평균). 반환: [KEY_COL, 'pred'] DataFrame (원본 unit 순서 보존)."""
    df = pd.DataFrame({KEY_COL: xs[KEY_COL].values, "pred": die_pred})
    grp = df.groupby(KEY_COL, sort=False)["pred"].mean().reset_index()
    return grp


# ------------------------------------------------------------
# fit + predict (모델별 분기) / N_JOBS 주입 / X 행렬 구성
# ------------------------------------------------------------

# 노트북 상단의 단일 N_JOBS 변수를 라이브러리마다 다른 키로 주입한다.
_NJOBS_KEYED   = {"lgbm", "xgb", "et", "zitboost"}   # 'n_jobs' 키를 쓰는 모델들
_THREADCOUNT   = {"catboost"}                        # CatBoost는 'thread_count'
# enet(sklearn ElasticNet)은 병렬 인자가 아예 없음 → 무시


def _inject_n_jobs(model_name, params, n_jobs):
    """모델별 적절한 키로 N_JOBS를 덮어쓴 새 dict 반환 (n_jobs=None이면 그대로).

    - lgbm/xgb/et/zitboost: params['n_jobs'] = n_jobs
    - catboost: params['thread_count'] = n_jobs (혹시 들어 있던 n_jobs 키는 제거)
    - enet: 아무것도 안 함
    """
    if n_jobs is None:
        return params   # 미지정 → search space의 -1(전체 코어) 등 기존 값 유지
    p = dict(params)
    if model_name in _THREADCOUNT:
        p["thread_count"] = int(n_jobs)
        p.pop("n_jobs", None)
    elif model_name in _NJOBS_KEYED:
        p["n_jobs"] = int(n_jobs)
    return p


def _fit_predict_fold(
    model_name, hp,
    X_tr, y_tr, X_vl,
    return_components=False,
):
    """fold 하나 학습 + val 예측. ZITboost면 return_components=True 시 (π, μ)도 같이.

    Returns
    -------
    dict  {'pred': array, 'pi': array|None, 'mu': array|None, 'model': fitted}
    """
    model = _models.create_regressor(model_name, hp)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_vl)

    pi = mu = None
    if return_components and model_name == "zitboost":
        pi, mu, _phi = model.predict_components(X_vl)
    return {"pred": pred, "pi": pi, "mu": mu, "model": model}


def _build_X(xs_split, feat_cols, extra_feature=None):
    """feat_cols로 X 행렬을 만들고, extra_feature((name, 1D array))가 있으면 한 컬럼 더 붙임.

    extra_feature 배열 길이는 xs_split 행 수와 같아야 함 (die 단위로 정렬돼 있어야 함).
    """
    X = xs_split[feat_cols].values
    if extra_feature is not None:
        _, arr = extra_feature
        arr = np.asarray(arr).reshape(-1, 1)
        if arr.shape[0] != len(xs_split):
            raise ValueError(
                f"extra_feature length {arr.shape[0]} != xs_split length {len(xs_split)}"
            )
        X = np.hstack([X, arr])   # X 오른쪽에 한 컬럼 추가
    return X


# ------------------------------------------------------------
# Optuna HPO (회귀)
# ------------------------------------------------------------

def run_hpo(
    xs_train, ys_train_unit, feat_cols,
    model_name,
    n_trials=100, n_folds=5,
    extra_feature_train=None,     # (name, die-level array) or None — reg 입력 X에 컬럼 추가
    multiplier_train=None,        # die-level array or None — 최종 예측 = reg_pred × multiplier (예: 1-π)
    y_positive_only=False,        # True면 fit 데이터에서 y==0 die 제외 (정석 Two-Stage Stage 2)
    target_transform_fn=None,     # y → y_transformed (fit 전 적용, 예: np.log1p)
    target_inverse_fn=None,       # y_transformed → y (predict 후 역변환, 예: np.expm1+clip)
    study_name=None, storage=None,
    resume_study=False,           # True여야만 기존 study에 trial을 이어 붙임 (아니면 중복 study는 에러)
    seed=SEED, direction="minimize",
    show_progress_bar=True,
    user_attrs=None,
    space_variant="default",      # 'default' | 'zitreg' — models.get_search_space의 variant
    sampler=None,                 # None이면 TPESampler(seed=seed, multivariate=True, group=True)
    pruner=None,                  # None이면 pruning 없음
    enqueue_trials=None,          # list[dict] — anchor(1차 best HP)를 첫 trial로 강제
    timeout=None,                 # 초 단위, None=무제한 (Colab 타임아웃 대비)
    n_jobs=None,                  # 모델 학습 병렬도. None이면 라이브러리 default(-1)
    # 매 trial마다 holdout 평가 (옵션) — xs_val/ys_val_unit 둘 다 주면 fold 평균 val 예측 → val_rmse 기록.
    xs_val=None, ys_val_unit=None,
    extra_feature_val=None, multiplier_val=None,
    xs_test=None, ys_test_unit=None,
    extra_feature_test=None, multiplier_test=None,
):
    """die-level KFold OOF → unit RMSE를 최소화하는 Optuna study 실행 (단일 모델).

    Parameters
    ----------
    xs_train : DataFrame (die-level, KEY_COL 컬럼 포함)
    ys_train_unit : DataFrame (unit-level, KEY_COL + TARGET_COL) — **원본 스케일**
    feat_cols : list[str]
    model_name : str — 'lgbm' / 'xgb' / 'catboost' / 'et' / 'enet' / 'zitboost'
    n_trials, n_folds : int
    extra_feature_train : (name, array) or None — reg 입력에 die-level 컬럼 1개 추가
    multiplier_train : array or None — 최종 예측을 reg_pred × multiplier로. (정석 Two-Stage 경로 B에서 (1-π)).
    y_positive_only : bool — True면 fold 학습 데이터에서 y==0 die 제외 → E[Y|Y>0,x] 학습
    target_transform_fn / target_inverse_fn : callable or None — 학습/출력 공간 변환쌍 (반드시 둘 다 주거나 둘 다 None)
    study_name, storage : Optuna study 옵션
    user_attrs : dict — study에 저장할 메타데이터
    xs_val, ys_val_unit : (옵션, 짝) — 주면 매 trial의 val unit RMSE 기록
    xs_test, ys_test_unit : (옵션, 짝) — 동일하게 test RMSE 기록
    extra_feature_val/test, multiplier_val/test : *_train과 동일 의미로 val/test에 적용

    Returns
    -------
    dict  {'study', 'best_params', 'model_name', 'best_value'}
    """
    # transform/inverse는 반드시 쌍으로, eval set들도 짝으로 — 한쪽만 주면 에러
    if (target_transform_fn is None) != (target_inverse_fn is None):
        raise ValueError("target_transform_fn / target_inverse_fn은 쌍으로 제공")
    if (xs_val is None) != (ys_val_unit is None):
        raise ValueError("xs_val / ys_val_unit 은 쌍으로 제공해야 함")
    if (xs_test is None) != (ys_test_unit is None):
        raise ValueError("xs_test / ys_test_unit 은 쌍으로 제공해야 함")
    space_fn = _models.get_search_space(model_name, variant=space_variant)

    # unit 단위 KFold를 한 번 만들어 모든 trial이 공유 (HP 비교의 공정성). die mask도 미리 계산해 둠 (속도).
    unit_ids = ys_train_unit[KEY_COL].unique()
    folds = _make_unit_folds(unit_ids, n_folds, seed)
    fold_masks = [
        (_die_mask_from_units(xs_train, set(tr)),
         _die_mask_from_units(xs_train, set(vl)))
        for tr, vl in folds
    ]

    y_die_orig = _broadcast_y_to_die(xs_train, ys_train_unit)   # die-level 정답 (원본 스케일)
    # 학습용 y(objective 안의 y_die_fit_local)는 trial별로 결정한다 — tweedie 계열 loss면 transform을 끄기 때문 (아래 objective 안에서)
    X_full      = _build_X(xs_train, feat_cols, extra_feature_train)
    y_true_unit = ys_train_unit.set_index(KEY_COL)[TARGET_COL]   # unit-level 정답 (원본 스케일)

    # val/test의 X와 정답은 trial 전체에서 한 번만 구성
    X_val_full = (_build_X(xs_val, feat_cols, extra_feature_val)
                  if xs_val is not None else None)
    X_test_full = (_build_X(xs_test, feat_cols, extra_feature_test)
                   if xs_test is not None else None)
    y_val_true_unit = (ys_val_unit.set_index(KEY_COL)[TARGET_COL]
                       if ys_val_unit is not None else None)
    y_test_true_unit = (ys_test_unit.set_index(KEY_COL)[TARGET_COL]
                        if ys_test_unit is not None else None)

    # multiplier 배열이 들어왔으면 길이가 해당 split die 수와 맞는지 검증하고 float 1D로 정규화
    def _check_mult(arr, n, name):
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=float).ravel()
        if len(arr) != n:
            raise ValueError(f"{name} 길이 {len(arr)} != 대상 die 수 {n}")
        return arr
    multiplier_train = _check_mult(multiplier_train, len(xs_train), "multiplier_train")
    multiplier_val   = _check_mult(multiplier_val,
                                    len(xs_val) if xs_val is not None else 0,
                                    "multiplier_val")
    multiplier_test  = _check_mult(multiplier_test,
                                    len(xs_test) if xs_test is not None else 0,
                                    "multiplier_test")

    def _is_tweedie_hp(hp):
        """이 trial이 고른 hp에 tweedie 계열 손실(objective/loss_function)이 들어 있으면 True.

        tweedie는 right-skew를 분포 자체로 모델링하므로 log1p target과 같이 쓰면 이중 변환이 되어버린다.
        """
        s = str(hp.get("objective") or hp.get("loss_function") or "")
        return s.startswith("tweedie") or s.startswith("reg:tweedie") or s.lower().startswith("tweedie")

    def objective(trial):
        hp = space_fn(trial)                                  # 이 trial의 HP 샘플
        hp = _inject_n_jobs(model_name, hp, n_jobs)           # search space의 -1을 노트북 N_JOBS로 덮어씀

        # tweedie 계열 loss면 target_transform을 끈다 (이중 변환 방지)
        if _is_tweedie_hp(hp):
            eff_transform_fn = None
            eff_inverse_fn   = None
        else:
            eff_transform_fn = target_transform_fn
            eff_inverse_fn   = target_inverse_fn
        trial.set_user_attr("target_transform_active", eff_transform_fn is not None)

        y_die_fit_local = eff_transform_fn(y_die_orig) if eff_transform_fn else y_die_orig

        def _eval_split_rmse_local(xs_split, die_pred_accum, multiplier, y_true_unit_split):
            """fold 평균된 die 예측 → (있으면) 역변환 → (있으면) ×multiplier → unit 집계 → unit RMSE."""
            pred = eff_inverse_fn(die_pred_accum) if eff_inverse_fn else die_pred_accum
            if multiplier is not None:
                pred = pred * multiplier
            unit = _aggregate_die_to_unit(xs_split, pred)
            aligned = unit.set_index(KEY_COL)["pred"].loc[y_true_unit_split.index]
            return float(np.sqrt(np.mean((aligned.values - y_true_unit_split.values) ** 2)))

        oof = np.full(len(xs_train), np.nan)                  # train 각 die의 OOF 예측 버퍼
        val_pred_accum  = (np.zeros(len(xs_val))  if xs_val  is not None else None)   # fold 합산용
        test_pred_accum = (np.zeros(len(xs_test)) if xs_test is not None else None)

        for tr_mask, vl_mask in fold_masks:
            # 정석 Two-Stage(Stage 2): fit 데이터에서 y==0 die 제외
            if y_positive_only:
                fit_mask = tr_mask & (y_die_orig > 0)
            else:
                fit_mask = tr_mask
            X_tr, y_tr = X_full[fit_mask], y_die_fit_local[fit_mask]
            X_vl       = X_full[vl_mask]

            # enet이면 이 fold의 train 기준으로 RobustScaler 통계량(median/IQR)을 잡아 → 이 fold holdout + 외부 val + 외부 test 전부에 동일 적용
            if _scaler.needs_scaling(model_name):
                med = np.median(X_tr, axis=0)
                q75 = np.quantile(X_tr, 0.75, axis=0)
                q25 = np.quantile(X_tr, 0.25, axis=0)
                iqr = np.maximum(q75 - q25, 1e-8)
                X_tr   = (X_tr - med) / iqr
                X_vl   = (X_vl - med) / iqr
                X_eval_v = (X_val_full  - med) / iqr if X_val_full  is not None else None
                X_eval_t = (X_test_full - med) / iqr if X_test_full is not None else None
            else:
                X_eval_v = X_val_full
                X_eval_t = X_test_full

            res = _fit_predict_fold(model_name, hp, X_tr, y_tr, X_vl)
            oof[vl_mask] = res["pred"]                         # 이 fold의 검증분 OOF 채움

            # val/test 예측은 n_folds개 fold 모델의 평균 (각 fold 예측에 1/n_folds씩 누적)
            if X_eval_v is not None:
                val_pred_accum  += res["model"].predict(X_eval_v)  / n_folds
            if X_eval_t is not None:
                test_pred_accum += res["model"].predict(X_eval_t) / n_folds

        # OOF에 NaN이 있으면 (fold 누락이 아니라) 모델 예측 자체가 NaN인 경우 — 보통 tweedie/poisson 류
        # log-space loss가 발산해서 exp() overflow → NaN. 셀 전체를 죽이지 말고 이 trial만 폐기(pruned)하고 계속.
        if np.isnan(oof).any():
            n_nan = int(np.isnan(oof).sum())
            print(f"[trial {trial.number}] 예측에 NaN {n_nan}개 (모델 발산 추정: "
                  f"objective={hp.get('objective')}, tweedie_power={hp.get('tweedie_variance_power')}) "
                  f"→ 이 trial 폐기")
            raise optuna.TrialPruned()

        # objective 반환값 = train OOF unit RMSE (역변환·곱셈까지 반영한 "최종 예측" 기준)
        train_rmse = _eval_split_rmse_local(xs_train, oof, multiplier_train, y_true_unit)
        trial.set_user_attr("train_rmse", train_rmse)         # dashboard에서 라벨 명확하게

        # val/test RMSE는 참고용으로 user_attr에 기록 (objective에는 안 들어감)
        if val_pred_accum is not None:
            val_rmse = _eval_split_rmse_local(xs_val, val_pred_accum,
                                              multiplier_val, y_val_true_unit)
            trial.set_user_attr("val_rmse", val_rmse)
        if test_pred_accum is not None:
            test_rmse = _eval_split_rmse_local(xs_test, test_pred_accum,
                                               multiplier_test, y_test_true_unit)
            trial.set_user_attr("test_rmse", test_rmse)

        return train_rmse

    # resume_study=False면 같은 study_name/storage가 이미 있을 때 에러를 내서 trial 누적으로 best가 오염되는 걸 막는다.
    # (load_if_exists는 storage가 있고 resume_study=True일 때만 켜짐)
    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(storage and resume_study),
        sampler=sampler if sampler is not None else optuna.samplers.TPESampler(
            seed=seed, multivariate=True, group=True
        ),
        pruner=pruner,    # None이면 Optuna 기본(pruning 없음)
    )
    if user_attrs:
        for k, v in user_attrs.items():
            study.set_user_attr(k, v)

    # anchor enqueue: 기존 trial이 하나도 없을 때만 (RESUME 시에는 중복 enqueue 안 함)
    if enqueue_trials and len(study.trials) == 0:
        for anchor in enqueue_trials:
            study.enqueue_trial(dict(anchor))
        print(f"[enqueue] {len(enqueue_trials)} anchor trial(s) 강제")
    elif enqueue_trials:
        print(f"[enqueue skip] 기존 trial {len(study.trials)}개 — resume 모드")

    if _scaler.needs_scaling(model_name):
        print(f"[scaler] {model_name} → fold-local RobustScaler 적용 "
              f"(매 fold train 기준 fit, holdout/val/test 동일 변환)")

    study.optimize(objective, n_trials=n_trials, timeout=timeout,
                   show_progress_bar=show_progress_bar)

    return {
        "study":            study,
        "best_params":      dict(study.best_trial.params),
        "model_name":       model_name,
        "best_value":       study.best_value,
    }


# ------------------------------------------------------------
# Best Trial Refit (K-fold)
# ------------------------------------------------------------

def _hp_from_best(best_params, model_name, n_jobs=None):
    """Optuna best_params dict → 모델 생성자에 넘길 kwargs로 정리.

    탐색 공간이 objective를 categorical로 다루므로 여기서 모델 형식에 맞게 풀어 준다:
    - LGBM/XGB objective: 그대로 (tweedie면 tweedie_variance_power도 그대로 전달)
    - CatBoost loss_function: 'Tweedie' + tweedie_variance_power → 'Tweedie:variance_power=...' 문자열로 합침
    또 search space가 이미 넣는 고정값(random_state, n_jobs=-1 등)을 refit 경로에서도 setdefault로 보장한다.
    """
    hp = dict(best_params)
    # CatBoost: 'Tweedie' + tweedie_variance_power → loss_function 문자열에 끼워 넣기
    if model_name == "catboost":
        loss = hp.get("loss_function")
        if loss == "Tweedie":
            power = hp.pop("tweedie_variance_power", 1.5)
            hp["loss_function"] = f"Tweedie:variance_power={power}"
    # LGBM / XGB는 키를 그대로 모델에 넘김

    # 모델별 고정값 보강 (search space에서 빠졌을 수 있는 REUSE 경로 대비)
    from utils.config import SEED as _S
    if model_name in {"lgbm", "zitboost"}:
        hp.setdefault("random_state", _S)
        hp.setdefault("n_jobs", -1)
        hp.setdefault("verbose", -1)
        hp.setdefault("device", _models.DEVICE)
        if model_name == "lgbm":
            hp.setdefault("subsample_freq", 1)   # 없으면 LGBM이 subsample을 무시함
    elif model_name == "xgb":
        hp.setdefault("random_state", _S)
        hp.setdefault("n_jobs", -1)
        hp.setdefault("tree_method", "hist")
        hp.setdefault("verbosity", 0)
        # xgb_space와 동일: reg:tweedie면 발산 가드(max_delta_step) — best_params엔 안 박혀 있을 수 있으므로 setdefault로 복원
        if str(hp.get("objective", "")).startswith("reg:tweedie"):
            hp.setdefault("max_delta_step", 0.7)
    elif model_name == "catboost":
        hp.setdefault("random_seed", _S)
        hp.setdefault("verbose", False)
        hp.setdefault("allow_writing_files", False)
    elif model_name == "et":
        # Optuna trial.params엔 max_features_kind / max_features_frac 만 기록됨 → et_space와 동일하게 max_features로 환원
        if "max_features_kind" in hp:
            mf_kind = hp.pop("max_features_kind")
            mf_frac = hp.pop("max_features_frac", None)
            hp["max_features"] = mf_frac if (mf_kind == "frac" and mf_frac is not None) else "sqrt"
        hp.setdefault("max_features", "sqrt")
        hp.setdefault("bootstrap", True)        # et_space는 항상 bootstrap=True (sklearn 기본은 False라 빠지면 안 됨)
        hp.setdefault("random_state", _S)
        hp.setdefault("n_jobs", -1)
    elif model_name == "enet":
        hp.setdefault("random_state", _S)
        hp.setdefault("tol", 1e-6)
        hp.setdefault("selection", "random")
        hp.setdefault("precompute", True)

    hp = _inject_n_jobs(model_name, hp, n_jobs)   # n_jobs 인자가 들어왔으면 모델별 키로 덮어씀
    return hp


def refit_best(
    xs_train, xs_val, xs_test,
    ys_train_unit, feat_cols,
    model_name, best_params,
    n_folds=5, seed=SEED,
    extra_feature_train=None,
    extra_feature_val=None,
    extra_feature_test=None,
    multiplier_train=None,
    multiplier_val=None,
    multiplier_test=None,
    y_positive_only=False,
    target_transform_fn=None,
    target_inverse_fn=None,
    already_resolved=False,
    n_jobs=None,
):
    """Best trial HP로 K-fold 재학습. die-level OOF + (fold 평균) val/test 예측 생성.

    ZITboost면 π·μ 컴포넌트도 같이 반환한다.
    target_transform_fn을 주면 학습은 변환 공간, 출력은 원본 공간. (HPO와 동일하게 tweedie loss면 자동 OFF.)
    multiplier_* + y_positive_only는 정석 Two-Stage 경로 B를 지원:
      - y_positive_only=True: fit 데이터에서 y==0 die 제외 → E[Y|Y>0,x] 학습
      - multiplier_*: 최종 예측을 reg_pred × multiplier 로 → (1-π)×E[Y|Y>0,x] = E[Y|x]

    Returns
    -------
    dict {
        'oof_pred_die' / 'val_pred_die' / 'test_pred_die' : array — **원본 공간, multiplier 적용 후**
        'oof_pi'/'val_pi'/'test_pi', 'oof_mu'/'val_mu'/'test_mu' : array or None (ZITboost만)
        'oof_pred_unit' / 'val_pred_unit' / 'test_pred_unit' : DataFrame [KEY_COL, pred]
        'fold_models' : list — fold별 fitted 모델
        'fold_scalers' : list — fold별 {'median','iqr'} dict 또는 None (스케일링 안 하는 모델)
        'best_params_resolved' : dict
        'model_name' : str
    }
    """
    if (target_transform_fn is None) != (target_inverse_fn is None):
        raise ValueError("target_transform_fn / target_inverse_fn은 쌍으로 제공")
    # already_resolved=True: best_params가 이미 모델에 넣을 형태(JSON에서 읽은 best_params_resolved 등)면 재변환 스킵
    if already_resolved:
        hp = dict(best_params)
        hp = _inject_n_jobs(model_name, hp, n_jobs)
    else:
        hp = _hp_from_best(best_params, model_name, n_jobs=n_jobs)

    # HPO와 동일: tweedie 계열 loss면 transform OFF
    _obj_or_loss = str(hp.get("objective") or hp.get("loss_function") or "")
    if (_obj_or_loss.startswith("tweedie") or _obj_or_loss.startswith("reg:tweedie")
            or _obj_or_loss.lower().startswith("tweedie")):
        if target_transform_fn is not None:
            print(f"[refit] tweedie loss 감지 → target_transform OFF (EXPERIMENT_LOG §5.1)")
        target_transform_fn = None
        target_inverse_fn   = None

    # unit 단위 fold (run_hpo와 동일 seed/방식이라야 OOF가 같은 분할 위에서 나옴)
    unit_ids = ys_train_unit[KEY_COL].unique()
    folds = _make_unit_folds(unit_ids, n_folds, seed)
    y_die_train_orig = _broadcast_y_to_die(xs_train, ys_train_unit)
    y_die_train_fit  = target_transform_fn(y_die_train_orig) \
        if target_transform_fn else y_die_train_orig

    X_train_full = _build_X(xs_train, feat_cols, extra_feature_train)
    X_val_full   = _build_X(xs_val,   feat_cols, extra_feature_val)
    X_test_full  = _build_X(xs_test,  feat_cols, extra_feature_test)

    n_tr, n_vl, n_te = len(xs_train), len(xs_val), len(xs_test)
    # multiplier 길이 검증
    def _check_mult(arr, n, name):
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=float).ravel()
        if len(arr) != n:
            raise ValueError(f"{name} 길이 {len(arr)} != 대상 die 수 {n}")
        return arr
    multiplier_train = _check_mult(multiplier_train, n_tr, "multiplier_train")
    multiplier_val   = _check_mult(multiplier_val,   n_vl, "multiplier_val")
    multiplier_test  = _check_mult(multiplier_test,  n_te, "multiplier_test")

    oof_pred  = np.full(n_tr, np.nan)   # train die OOF
    val_pred  = np.zeros(n_vl)          # val: fold 평균 누적
    test_pred = np.zeros(n_te)          # test: fold 평균 누적

    is_zit = model_name == "zitboost"
    # ZITboost면 π/μ도 같은 방식으로 OOF·fold평균으로 모음
    oof_pi = np.full(n_tr, np.nan) if is_zit else None
    oof_mu = np.full(n_tr, np.nan) if is_zit else None
    val_pi = np.zeros(n_vl) if is_zit else None
    val_mu = np.zeros(n_vl) if is_zit else None
    test_pi = np.zeros(n_te) if is_zit else None
    test_mu = np.zeros(n_te) if is_zit else None

    fold_models = []
    fold_scalers = []   # enet 등: fold별 RobustScaler 통계량 보관 (pkl로 저장해 추론 시 재현)

    if _scaler.needs_scaling(model_name):
        print(f"[scaler] {model_name} → fold-local RobustScaler 적용 "
              f"(매 fold train 기준 fit, holdout/val/test 동일 변환)")

    for i, (tr_units, vl_units) in enumerate(folds):
        tr_mask = _die_mask_from_units(xs_train, set(tr_units))
        vl_mask = _die_mask_from_units(xs_train, set(vl_units))

        # 정석 Two-Stage: y>0 die만 학습
        if y_positive_only:
            fit_mask = tr_mask & (y_die_train_orig > 0)
        else:
            fit_mask = tr_mask
        X_tr, y_tr = X_train_full[fit_mask], y_die_train_fit[fit_mask]
        X_vl       = X_train_full[vl_mask]

        # enet: 이 fold의 train 기준 RobustScaler → val-fold/val/test 모두 같은 통계로 변환
        if _scaler.needs_scaling(model_name):
            med = np.median(X_tr, axis=0)
            q75 = np.quantile(X_tr, 0.75, axis=0)
            q25 = np.quantile(X_tr, 0.25, axis=0)
            iqr = np.maximum(q75 - q25, 1e-8)
            X_tr = (X_tr - med) / iqr
            X_vl = (X_vl - med) / iqr
            X_val_tr  = (X_val_full  - med) / iqr
            X_test_tr = (X_test_full - med) / iqr
            fold_scalers.append({"median": med, "iqr": iqr})
        else:
            X_val_tr, X_test_tr = X_val_full, X_test_full
            fold_scalers.append(None)

        model = _models.create_regressor(model_name, hp)
        model.fit(X_tr, y_tr)

        # 예측은 학습 공간 → 역변환해서 누적
        pred_vl = model.predict(X_vl)
        pred_v  = model.predict(X_val_tr)
        pred_t  = model.predict(X_test_tr)
        if target_inverse_fn:
            pred_vl = target_inverse_fn(pred_vl)
            pred_v  = target_inverse_fn(pred_v)
            pred_t  = target_inverse_fn(pred_t)

        oof_pred[vl_mask] = pred_vl                # 이 fold 검증분 OOF
        val_pred  += pred_v / n_folds              # val/test는 fold 평균
        test_pred += pred_t / n_folds

        if is_zit:
            pi_vl, mu_vl, _ = model.predict_components(X_vl)
            oof_pi[vl_mask] = pi_vl
            oof_mu[vl_mask] = mu_vl
            pi_v, mu_v, _ = model.predict_components(X_val_tr)
            pi_t, mu_t, _ = model.predict_components(X_test_tr)
            val_pi += pi_v / n_folds
            val_mu += mu_v / n_folds
            test_pi += pi_t / n_folds
            test_mu += mu_t / n_folds

        fold_models.append(model)
        print(f"[refit fold {i+1}/{n_folds}] "
              f"tr_units={len(tr_units)}, vl_units={len(vl_units)}")

    if np.isnan(oof_pred).any():
        raise RuntimeError(
            "oof_pred has NaN — fold 커버리지 문제가 아니라 모델 예측 자체가 NaN "
            "(tweedie/poisson 류 log-space loss 발산 가능성). best_params 확인 필요."
        )

    # multiplier 곱셈을 마지막에 일괄 적용 → oof/val/test가 모두 "최종 예측" 의미로 통일됨
    if multiplier_train is not None:
        oof_pred = oof_pred * multiplier_train
    if multiplier_val is not None:
        val_pred = val_pred * multiplier_val
    if multiplier_test is not None:
        test_pred = test_pred * multiplier_test

    return {
        "oof_pred_die":  oof_pred,
        "val_pred_die":  val_pred,
        "test_pred_die": test_pred,
        "oof_pi":  oof_pi,  "val_pi":  val_pi,  "test_pi":  test_pi,
        "oof_mu":  oof_mu,  "val_mu":  val_mu,  "test_mu":  test_mu,
        "oof_pred_unit":  _aggregate_die_to_unit(xs_train, oof_pred),
        "val_pred_unit":  _aggregate_die_to_unit(xs_val,   val_pred),
        "test_pred_unit": _aggregate_die_to_unit(xs_test,  test_pred),
        "fold_models":    fold_models,
        "fold_scalers":   fold_scalers,
        "best_params_resolved": hp,
        "model_name": model_name,
    }


# ------------------------------------------------------------
# 산출물 저장 (pkl + CSV + JSON)
# ------------------------------------------------------------

from utils.config import DIE_KEY_COL as _DIE_KEY_COL


def _die_csv(xs_split, pred, pi=None, mu=None, y_unit=None):
    """die-level 예측을 [KEY_COL, DIE_KEY_COL, pred(, health)(, pi/one_minus_pi/mu)] DataFrame으로.

    y_unit이 주어지면 unit health를 die에 broadcast해서 'health' 컬럼 추가 (test처럼 y 없으면 그 컬럼 자체 빠짐).
    """
    out = pd.DataFrame({
        KEY_COL:      xs_split[KEY_COL].values,
        _DIE_KEY_COL: xs_split[_DIE_KEY_COL].values,
        "pred":       pred,
    })
    if y_unit is not None:
        h_map = (y_unit.set_index(KEY_COL)[TARGET_COL]
                 if isinstance(y_unit, pd.DataFrame) else y_unit)
        out[TARGET_COL] = out[KEY_COL].map(h_map)
    if pi is not None:
        out["pi"] = pi
        out["one_minus_pi"] = 1.0 - pi   # 경로 B에서 multiplier로 바로 쓰기 쉽게 미리 만들어 둠
    if mu is not None:
        out["mu"] = mu
    return out


def _add_health_to_unit(unit_df, y_unit):
    """unit-level [KEY_COL, 'pred'] DataFrame에 'health' 컬럼을 붙여 반환 (y_unit이 None이면 그대로)."""
    if y_unit is None:
        return unit_df
    h_map = (y_unit.set_index(KEY_COL)[TARGET_COL]
             if isinstance(y_unit, pd.DataFrame) else y_unit)
    out = unit_df.copy()
    out[TARGET_COL] = out[KEY_COL].map(h_map)
    return out


def save_artifacts(
    refit_result, xs_train, xs_val, xs_test,
    out_dir, exp_id=None,
    feature_names=None,
    extra_feature_name=None,
    y_train_unit=None,
    y_val_unit=None,
    y_test_unit=None,
    postprocess_config=None,
    study_meta=None,
):
    """refit_best 결과를 디스크에 저장 (fold_models.pkl + best_params.json + die/unit CSV 6개).

    Parameters
    ----------
    feature_names : list[str] or None — 학습에 쓴 피처 이름 (재현/importance용). json+pkl에 저장.
    extra_feature_name : str or None — X 뒤에 붙은 추가 피처 이름 (예: 'one_minus_pi'). 저장 전용 메타.
    y_train_unit : DataFrame or None — postprocess_config가 있으면 필수 (unit RMSE 튜닝 + CSV의 health 컬럼).
    y_val_unit, y_test_unit : DataFrame or None — 주면 val/test CSV에 health 컬럼 merge (없으면 컬럼 빠짐).
    postprocess_config : dict or None — None이면 refit의 mean 집계만 저장. dict면 postprocess.tune_and_apply에 kwargs로 넘겨 unit CSV를 튜닝값으로 대체.
    study_meta : dict or None — study.user_attrs 같은 재현성 메타. best_params.json에 그대로 저장.

    생성 파일: fold_models.pkl, best_params.json, oof/val/test_die.csv, oof/val/test_unit.csv.
    경로 B는 *_die.csv의 'one_minus_pi' 컬럼을 reg 입력 피처로 재사용한다.
    """
    import os, json, pickle
    os.makedirs(out_dir, exist_ok=True)

    # 1) fold 모델들 + feature 이름 등을 pickle로 저장 (추론 시 그대로 로드해서 예측)
    pkl_payload = {
        "fold_models":         refit_result["fold_models"],
        "fold_scalers":        refit_result.get("fold_scalers"),
        "feature_names":       list(feature_names) if feature_names is not None else None,
        "extra_feature_name":  extra_feature_name,
        "model_name":          refit_result["model_name"],
        "n_folds":             len(refit_result["fold_models"]),
    }
    with open(os.path.join(out_dir, "fold_models.pkl"), "wb") as f:
        pickle.dump(pkl_payload, f)

    # 2) best_params.json 메타 — resolved HP + 전처리 파라미터 + feature 이름 + study 메타
    meta = {
        "exp_id":                exp_id,
        "model_name":            refit_result["model_name"],
        "best_params_resolved":  refit_result["best_params_resolved"],
        # effective_pp_params는 study_meta 하위가 아니라 top-level에도 둠 (clf 쪽과 키 위치 정합)
        "effective_pp_params":   (study_meta or {}).get("effective_pp_params"),
        "feature_names":         list(feature_names) if feature_names is not None else None,
        "n_features":            len(feature_names) if feature_names is not None else None,
        "extra_feature_name":    extra_feature_name,
        "n_folds":               len(refit_result["fold_models"]),
        "study_meta":            study_meta or {},
    }
    # fold 분할 재현성 — train unit 목록의 해시를 박제 (zit↔reg↔stacking의 OOF가 같은 분할인지 검증용)
    if y_train_unit is not None:
        import hashlib
        uid_arr = y_train_unit[KEY_COL].unique()
        uid_bytes = ",".join(map(str, uid_arr)).encode("utf-8")
        meta["unit_ids_hash"] = hashlib.sha1(uid_bytes).hexdigest()
        meta["n_units_train"] = int(len(uid_arr))

    # 3) die-level CSV — postprocess 이전의 die-level 예측 (multiplier_* 경로면 (1-π) 곱셈까지 반영된 값) + health, ZIT이면 pi/mu
    _die_csv(xs_train, refit_result["oof_pred_die"],
             refit_result.get("oof_pi"), refit_result.get("oof_mu"),
             y_unit=y_train_unit,
             ).to_csv(os.path.join(out_dir, "oof_die.csv"), index=False)
    _die_csv(xs_val,   refit_result["val_pred_die"],
             refit_result.get("val_pi"), refit_result.get("val_mu"),
             y_unit=y_val_unit,
             ).to_csv(os.path.join(out_dir, "val_die.csv"), index=False)
    _die_csv(xs_test,  refit_result["test_pred_die"],
             refit_result.get("test_pi"), refit_result.get("test_mu"),
             y_unit=y_test_unit,
             ).to_csv(os.path.join(out_dir, "test_die.csv"), index=False)

    # 4) unit-level CSV — postprocess_config가 있으면 튜닝값, 없으면 refit의 mean 집계
    if postprocess_config is not None and y_train_unit is not None:
        from . import postprocess as _pp
        pp_res = _pp.tune_and_apply(
            xs_train, xs_val, xs_test,
            die_pred_train=refit_result["oof_pred_die"],
            die_pred_val=refit_result["val_pred_die"],
            die_pred_test=refit_result["test_pred_die"],
            y_train_unit=y_train_unit,
            y_val_unit=y_val_unit,                       # val 기준으로 채택 여부 결정
            die_pi_train=refit_result.get("oof_pi"),
            die_pi_val=refit_result.get("val_pi"),
            die_pi_test=refit_result.get("test_pi"),
            **postprocess_config,
        )
        _add_health_to_unit(pp_res["final_train_unit"], y_train_unit).to_csv(
            os.path.join(out_dir, "oof_unit.csv"), index=False)
        _add_health_to_unit(pp_res["final_val_unit"], y_val_unit).to_csv(
            os.path.join(out_dir, "val_unit.csv"), index=False)
        _add_health_to_unit(pp_res["final_test_unit"], y_test_unit).to_csv(
            os.path.join(out_dir, "test_unit.csv"), index=False)
        # 후처리 결정 내역도 메타에 기록
        meta["postprocess"] = {
            "best_agg":          pp_res["best_agg"],
            "pos_weights":       (pp_res["pos_weights"].tolist()
                                  if pp_res["pos_weights"] is not None else None),
            "best_pi_threshold": pp_res["best_pi_threshold"],
            "best_zero_clip":    pp_res["best_zero_clip"],
            "train_rmse":        pp_res["train_rmse"],
            "val_rmse_final":    pp_res.get("val_rmse_final"),
            "val_rmse_history":  pp_res.get("val_rmse_history"),
            "decisions":         pp_res.get("decisions"),
            "agg_rmses":         {k: float(v) for k, v in pp_res["agg_rmses"].items()},
            "config":            postprocess_config,
        }
    else:
        # 후처리 없이 refit 단계의 mean 집계 그대로 저장
        _add_health_to_unit(refit_result["oof_pred_unit"], y_train_unit).to_csv(
            os.path.join(out_dir, "oof_unit.csv"), index=False)
        _add_health_to_unit(refit_result["val_pred_unit"], y_val_unit).to_csv(
            os.path.join(out_dir, "val_unit.csv"), index=False)
        _add_health_to_unit(refit_result["test_pred_unit"], y_test_unit).to_csv(
            os.path.join(out_dir, "test_unit.csv"), index=False)
        meta["postprocess"] = None

    # best_params.json 저장 (후처리 결과까지 포함된 최종 meta)
    with open(os.path.join(out_dir, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    pp_tag = "tuned" if meta.get("postprocess") else "mean"
    print(f"[save_artifacts] {out_dir} 저장 완료 "
          f"(fold_models.pkl + best_params.json + 6 CSV, unit={pp_tag})")


def load_extra_feature_from_path(path_dir, xs_split, feature_col="one_minus_pi",
                                 split=None):
    """경로 B 용: 경로 A가 저장한 die-level CSV에서 (1-π) 같은 컬럼을 꺼내 xs_split die 순서로 정렬해 반환.

    Parameters
    ----------
    path_dir : str — 경로 A가 save_artifacts로 저장한 디렉토리
    xs_split : DataFrame — 현재 노트북의 xs_train / xs_val / xs_test 중 하나
    feature_col : str (default 'one_minus_pi')
    split : {'train','val','test'} or None — 명시하면 해당 CSV(oof/val/test_die)를 직접 로드. None이면 길이·키셋 일치로 자동 감지.

    Returns
    -------
    np.ndarray — xs_split과 같은 길이의 die-level 배열
    """
    import os

    CSV_BY_SPLIT = {
        "train": "oof_die.csv",
        "val":   "val_die.csv",
        "test":  "test_die.csv",
    }

    split_keys = set(xs_split[_DIE_KEY_COL].values)   # 이 split의 die 키 집합
    n_split = len(xs_split)

    def _try_load(csv_name):
        """csv_name을 읽어 feature_col을 xs_split die 순서로 정렬해 반환. 길이/키 불일치면 사유 튜플, 파일 없으면 None."""
        full = os.path.join(path_dir, csv_name)
        if not os.path.exists(full):
            return None
        df = pd.read_csv(full)
        if feature_col not in df.columns:
            raise ValueError(
                f"{csv_name}에 컬럼 {feature_col!r} 없음 — "
                f"경로 A 아티팩트가 맞는지 확인하세요."
            )
        # 부분 매칭 사고 방지: 길이도 같고 키 집합도 정확히 같아야 함
        if len(df) != n_split:
            return ("length_mismatch", len(df))
        if set(df[_DIE_KEY_COL].values) != split_keys:
            return ("key_mismatch", None)
        aligned = df.set_index(_DIE_KEY_COL).loc[
            xs_split[_DIE_KEY_COL].values, feature_col
        ].values
        return aligned.astype(float)

    # split을 명시했으면 그 파일만 시도하고, 안 되면 에러
    if split is not None:
        if split not in CSV_BY_SPLIT:
            raise ValueError(
                f"split={split!r} — 'train'/'val'/'test' 중 하나여야 함"
            )
        csv_name = CSV_BY_SPLIT[split]
        result = _try_load(csv_name)
        if result is None:
            raise FileNotFoundError(
                f"{os.path.join(path_dir, csv_name)} 없음"
            )
        if isinstance(result, tuple):
            kind, detail = result
            raise ValueError(
                f"{csv_name} 내용이 xs_split 과 불일치 ({kind}, "
                f"detail={detail}, xs_split len={n_split}) — "
                f"01(경로 A)과 03(경로 B)의 전처리/split 이 동일한지 확인."
            )
        print(f"[load_extra_feature] {csv_name}({split}) → {feature_col} "
              f"(n={len(result)})")
        return result

    # split 미지정: 세 CSV 중 길이+키가 모두 맞는 걸 자동 채택
    for split_name, csv_name in CSV_BY_SPLIT.items():
        result = _try_load(csv_name)
        if isinstance(result, np.ndarray):
            print(f"[load_extra_feature] {csv_name}(auto={split_name}) "
                  f"→ {feature_col} (n={len(result)})")
            return result
    raise FileNotFoundError(
        f"{path_dir} 안에서 xs_split(len={n_split})과 길이·키셋이 "
        f"모두 일치하는 die CSV를 찾지 못했습니다. split 인자를 명시하거나 "
        f"01 노트북을 다시 실행하세요."
    )


# ------------------------------------------------------------
# CLF (Two-Stage Stage 1) — binary 분류 (y>0 vs y=0)
# ------------------------------------------------------------
# die-level로 학습 + unit 평균으로 집계. objective는 "회귀를 평균 상수로 가정한 unit RMSE":
#   oof_proba_die → unit 평균 → × y_pos_const → unit pred → unit RMSE,  y_pos_const = mean(y_train_unit | y>0) = E[Y|Y>0]
# 이렇게 하면 분류기 calibration을 RMSE 척도로 직접 평가하게 됨. KFold는 회귀와 똑같이 unit 단위(leakage 방지).

def run_clf_hpo(
    xs_train, ys_train_unit, feat_cols,
    model_name,
    n_trials=100, n_folds=5,
    study_name=None, storage=None,
    resume_study=False,
    seed=SEED,
    show_progress_bar=True,
    user_attrs=None,
    sampler=None,                 # None이면 TPESampler(seed=seed, multivariate=True, group=True)
    pruner=None,                  # None이면 pruning 없음
    enqueue_trials=None,          # list[dict] — anchor 첫 trial 강제
    timeout=None,                 # 초 단위, None=무제한
    n_jobs=None,                  # 모델 학습 병렬도. None이면 라이브러리 default(-1)
    xs_val=None, ys_val_unit=None,
    xs_test=None, ys_test_unit=None,
):
    """die-level binary CLF HPO. objective = unit RMSE(unit평균확률 × y_pos_const).

    Returns
    -------
    dict {'study', 'best_params', 'model_name', 'best_value', 'y_pos_const'}
    """
    if (xs_val is None) != (ys_val_unit is None):
        raise ValueError("xs_val / ys_val_unit 은 쌍으로 제공")
    if (xs_test is None) != (ys_test_unit is None):
        raise ValueError("xs_test / ys_test_unit 은 쌍으로 제공")

    space_fn = _models.get_clf_search_space(model_name)

    # unit 단위 fold + die mask 미리 계산
    unit_ids = ys_train_unit[KEY_COL].unique()
    folds = _make_unit_folds(unit_ids, n_folds, seed)
    fold_masks = [
        (_die_mask_from_units(xs_train, set(tr)),
         _die_mask_from_units(xs_train, set(vl)))
        for tr, vl in folds
    ]

    y_die_orig = _broadcast_y_to_die(xs_train, ys_train_unit)
    y_die_bin  = (y_die_orig > 0).astype(int)             # 학습 target = "이 die의 unit이 y>0인가" 0/1
    X_train = xs_train[feat_cols].values
    X_val   = xs_val[feat_cols].values  if xs_val  is not None else None
    X_test  = xs_test[feat_cols].values if xs_test is not None else None

    y_true_unit      = ys_train_unit.set_index(KEY_COL)[TARGET_COL]
    y_val_true_unit  = (ys_val_unit.set_index(KEY_COL)[TARGET_COL]
                        if ys_val_unit is not None else None)
    y_test_true_unit = (ys_test_unit.set_index(KEY_COL)[TARGET_COL]
                        if ys_test_unit is not None else None)

    # E[Y | Y>0] — Stage 2 회귀를 "상수"로 대체한 값. clf 단독 평가에서 unit pred = prob × 이 상수.
    y_pos_const = float(y_true_unit[y_true_unit > 0].mean())

    def _eval_split_rmse(xs_split, die_proba, y_true_unit_split):
        """die 확률 → unit 평균 확률 → × y_pos_const → unit pred → 정답과 RMSE."""
        df = pd.DataFrame({KEY_COL: xs_split[KEY_COL].values, "p": die_proba})
        unit_proba = df.groupby(KEY_COL, sort=False)["p"].mean()
        unit_pred  = unit_proba * y_pos_const
        aligned    = unit_pred.loc[y_true_unit_split.index]
        return float(np.sqrt(np.mean(
            (aligned.values - y_true_unit_split.values) ** 2
        )))

    def objective(trial):
        params = space_fn(trial)
        params = _inject_n_jobs(model_name, params, n_jobs)   # search space의 -1을 노트북 N_JOBS로
        oof_proba = np.full(len(xs_train), np.nan)
        val_proba_accum  = (np.zeros(len(xs_val))  if xs_val  is not None else None)
        test_proba_accum = (np.zeros(len(xs_test)) if xs_test is not None else None)

        for tr_mask, vl_mask in fold_masks:
            X_tr, y_tr = X_train[tr_mask], y_die_bin[tr_mask]
            X_vl       = X_train[vl_mask]

            # 이 fold의 양/음 비율로 클래스 불균형 옵션(scale_pos_weight 등)을 자동 채워 넣음
            params_resolved = _models.resolve_clf_imbalance(model_name, params, y_tr)
            clf = _models.create_classifier(model_name, params_resolved)
            clf.fit(X_tr, y_tr)

            oof_proba[vl_mask] = clf.predict_proba(X_vl)[:, 1]   # 양성(=y>0) 클래스 확률
            if X_val is not None:
                val_proba_accum  += clf.predict_proba(X_val)[:, 1]  / n_folds
            if X_test is not None:
                test_proba_accum += clf.predict_proba(X_test)[:, 1] / n_folds

        # OOF 확률에 NaN이 있으면 모델 예측 자체가 NaN인 경우 — 셀을 죽이지 말고 이 trial만 폐기하고 계속
        if np.isnan(oof_proba).any():
            n_nan = int(np.isnan(oof_proba).sum())
            print(f"[clf trial {trial.number}] 예측 확률에 NaN {n_nan}개 → 이 trial 폐기")
            raise optuna.TrialPruned()

        train_rmse = _eval_split_rmse(xs_train, oof_proba, y_true_unit)   # objective 반환값
        trial.set_user_attr("train_rmse", train_rmse)

        if val_proba_accum is not None:
            val_rmse = _eval_split_rmse(xs_val, val_proba_accum, y_val_true_unit)
            trial.set_user_attr("val_rmse", val_rmse)
        if test_proba_accum is not None:
            test_rmse = _eval_split_rmse(xs_test, test_proba_accum, y_test_true_unit)
            trial.set_user_attr("test_rmse", test_rmse)

        return train_rmse

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(storage and resume_study),
        sampler=sampler if sampler is not None else optuna.samplers.TPESampler(
            seed=seed, multivariate=True, group=True
        ),
        pruner=pruner,
    )
    if user_attrs:
        for k, v in user_attrs.items():
            study.set_user_attr(k, v)
    study.set_user_attr("y_pos_const", y_pos_const)
    study.set_user_attr("clf_objective_recipe",
                        "unit_RMSE(unit_mean(die_proba) * y_pos_const)")

    # anchor enqueue (RESUME 시에는 안 함)
    if enqueue_trials and len(study.trials) == 0:
        for anchor in enqueue_trials:
            study.enqueue_trial(dict(anchor))
        print(f"[enqueue] {len(enqueue_trials)} anchor trial(s) 강제")
    elif enqueue_trials:
        print(f"[enqueue skip] 기존 trial {len(study.trials)}개 — resume 모드")

    study.optimize(objective, n_trials=n_trials, timeout=timeout,
                   show_progress_bar=show_progress_bar)

    return {
        "study":       study,
        "best_params": dict(study.best_trial.params),
        "model_name":  model_name,
        "best_value":  study.best_value,
        "y_pos_const": y_pos_const,
    }


def _clf_hp_with_defaults(model_name, best_params, n_jobs=None):
    """clf best_params dict에 모델별 고정값(objective='binary' 등)을 setdefault로 보강 (REUSE 모드 호환)."""
    hp = dict(best_params)
    if model_name == "lgbm":
        hp.setdefault("objective",      "binary")
        hp.setdefault("random_state",   SEED)
        hp.setdefault("n_jobs",         -1)
        hp.setdefault("verbose",        -1)
        hp.setdefault("device",         _models.DEVICE)
        hp.setdefault("subsample_freq", 1)
    elif model_name == "xgb":
        hp.setdefault("objective",   "binary:logistic")
        hp.setdefault("eval_metric", "logloss")
        hp.setdefault("random_state", SEED)
        hp.setdefault("n_jobs",      -1)
        hp.setdefault("tree_method", "hist")
        hp.setdefault("verbosity",   0)
    elif model_name == "catboost":
        hp.setdefault("loss_function",       "Logloss")
        hp.setdefault("random_seed",         SEED)
        hp.setdefault("verbose",             False)
        hp.setdefault("allow_writing_files", False)
    elif model_name == "et":
        # et_clf_space도 max_features_kind/max_features_frac, class_weight를 categorical로 둠 → 모델 kwargs로 환원
        if "max_features_kind" in hp:
            mf_kind = hp.pop("max_features_kind")
            mf_frac = hp.pop("max_features_frac", None)
            hp["max_features"] = mf_frac if (mf_kind == "frac" and mf_frac is not None) else "sqrt"
        hp.setdefault("max_features", "sqrt")
        if hp.get("class_weight") == "None":   # trial.params엔 "None" 문자열로 남음 (et_clf_space는 resolved None을 넣지만)
            hp["class_weight"] = None
        hp.setdefault("random_state", SEED)
        hp.setdefault("n_jobs",       -1)
        hp.setdefault("bootstrap",    True)
    hp = _inject_n_jobs(model_name, hp, n_jobs)   # n_jobs 인자 들어왔으면 모델별 키로 덮어씀
    return hp


def refit_clf_best(
    xs_train, xs_val, xs_test,
    ys_train_unit, feat_cols,
    model_name, best_params,
    n_folds=5, seed=SEED,
    already_resolved=False,
    n_jobs=None,
):
    """Best CLF HP로 K-fold 재학습. die-level 양성 확률(OOF/val/test, val·test는 fold 평균) 반환.

    Returns
    -------
    dict {'oof_proba_die', 'val_proba_die', 'test_proba_die': array,
          'fold_models': list, 'best_params_resolved': dict, 'model_name': str}
    """
    if already_resolved:
        hp = dict(best_params)
        hp = _inject_n_jobs(model_name, hp, n_jobs)
    else:
        hp = _clf_hp_with_defaults(model_name, best_params, n_jobs=n_jobs)

    unit_ids = ys_train_unit[KEY_COL].unique()
    folds = _make_unit_folds(unit_ids, n_folds, seed)   # 회귀와 동일 seed/방식
    y_die_orig = _broadcast_y_to_die(xs_train, ys_train_unit)
    y_die_bin  = (y_die_orig > 0).astype(int)

    X_train = xs_train[feat_cols].values
    X_val   = xs_val[feat_cols].values
    X_test  = xs_test[feat_cols].values

    n_tr, n_vl, n_te = len(xs_train), len(xs_val), len(xs_test)
    oof_proba  = np.full(n_tr, np.nan)
    val_proba  = np.zeros(n_vl)
    test_proba = np.zeros(n_te)
    fold_models = []

    for i, (tr_units, vl_units) in enumerate(folds):
        tr_mask = _die_mask_from_units(xs_train, set(tr_units))
        vl_mask = _die_mask_from_units(xs_train, set(vl_units))

        X_tr, y_tr = X_train[tr_mask], y_die_bin[tr_mask]
        X_vl       = X_train[vl_mask]

        hp_resolved = _models.resolve_clf_imbalance(model_name, hp, y_tr)   # fold 클래스 비율로 imbalance 옵션 채움
        clf = _models.create_classifier(model_name, hp_resolved)
        clf.fit(X_tr, y_tr)

        oof_proba[vl_mask] = clf.predict_proba(X_vl)[:, 1]
        val_proba  += clf.predict_proba(X_val)[:, 1]  / n_folds
        test_proba += clf.predict_proba(X_test)[:, 1] / n_folds

        fold_models.append(clf)
        print(f"[clf refit fold {i+1}/{n_folds}] "
              f"tr_units={len(tr_units)}, vl_units={len(vl_units)}, "
              f"pos_ratio={y_tr.mean():.3f}")

    if np.isnan(oof_proba).any():
        raise RuntimeError(
            "oof_proba has NaN — fold 커버리지 문제가 아니라 분류기 예측 자체가 NaN. best_params 확인 필요."
        )

    return {
        "oof_proba_die":  oof_proba,
        "val_proba_die":  val_proba,
        "test_proba_die": test_proba,
        "fold_models":    fold_models,
        "best_params_resolved": hp,
        "model_name":     model_name,
    }


def save_clf_artifacts(
    refit_result, xs_train, xs_val, xs_test,
    out_dir, exp_id=None,
    feature_names=None,
    y_train_unit=None,
    y_val_unit=None,
    y_test_unit=None,
    y_pos_const=None,
    study_meta=None,
):
    """clf refit 결과를 디스크에 저장.

    파일:
      - oof/val/test_die.csv   : [KEY, DIE_KEY, prob, (health)]
      - oof/val/test_unit.csv  : [KEY, prob(=unit 평균), pred(=prob×y_pos_const), (health)]
      - fold_models.pkl
      - best_params.json
    """
    import os, json, pickle

    os.makedirs(out_dir, exist_ok=True)

    oof_p  = refit_result["oof_proba_die"]
    val_p  = refit_result["val_proba_die"]
    test_p = refit_result["test_proba_die"]

    def _build_die(xs_split, prob, y_unit):
        out = pd.DataFrame({
            KEY_COL:      xs_split[KEY_COL].values,
            _DIE_KEY_COL: xs_split[_DIE_KEY_COL].values,
            "prob":       prob,
        })
        if y_unit is not None:
            h_map = (y_unit.set_index(KEY_COL)[TARGET_COL]
                     if isinstance(y_unit, pd.DataFrame) else y_unit)
            out[TARGET_COL] = out[KEY_COL].map(h_map)
        return out

    def _build_unit(xs_split, prob, y_unit):
        # die 확률 → unit 평균 확률, 그리고 pred = 평균확률 × E[Y|Y>0]
        df = pd.DataFrame({KEY_COL: xs_split[KEY_COL].values, "p": prob})
        unit_proba = df.groupby(KEY_COL, sort=False)["p"].mean().reset_index()
        unit_proba.columns = [KEY_COL, "prob"]
        if y_pos_const is not None:
            unit_proba["pred"] = unit_proba["prob"] * y_pos_const
        if y_unit is not None:
            h_map = (y_unit.set_index(KEY_COL)[TARGET_COL]
                     if isinstance(y_unit, pd.DataFrame) else y_unit)
            unit_proba[TARGET_COL] = unit_proba[KEY_COL].map(h_map)
        return unit_proba

    _build_die(xs_train, oof_p,  y_train_unit).to_csv(os.path.join(out_dir, "oof_die.csv"),  index=False)
    _build_die(xs_val,   val_p,  y_val_unit  ).to_csv(os.path.join(out_dir, "val_die.csv"),  index=False)
    _build_die(xs_test,  test_p, y_test_unit ).to_csv(os.path.join(out_dir, "test_die.csv"), index=False)

    _build_unit(xs_train, oof_p,  y_train_unit).to_csv(os.path.join(out_dir, "oof_unit.csv"),  index=False)
    _build_unit(xs_val,   val_p,  y_val_unit  ).to_csv(os.path.join(out_dir, "val_unit.csv"),  index=False)
    _build_unit(xs_test,  test_p, y_test_unit ).to_csv(os.path.join(out_dir, "test_unit.csv"), index=False)

    with open(os.path.join(out_dir, "fold_models.pkl"), "wb") as f:
        pickle.dump(refit_result["fold_models"], f)

    # best_params.json — 회귀 save_artifacts와 키 위치를 맞춰 둠 (effective_pp_params/n_folds/unit_ids_hash/postprocess 등)
    meta = {
        "exp_id":               exp_id,
        "model_name":           refit_result["model_name"],
        "best_params_resolved": refit_result["best_params_resolved"],
        "effective_pp_params":  (study_meta or {}).get("effective_pp_params"),
        "feature_names":        feature_names,
        "n_features":           len(feature_names) if feature_names else None,
        "n_folds":              len(refit_result["fold_models"]),
        "y_pos_const":          y_pos_const,
        "postprocess":          None,  # clf 단독은 후처리 없음 (combine 단계에서 reg와 곱한 뒤 적용)
        "study_meta":           study_meta,
    }
    # fold 분할 재현성 (save_artifacts와 동일 패턴)
    if y_train_unit is not None:
        import hashlib
        uid_arr = (y_train_unit[KEY_COL].unique()
                   if isinstance(y_train_unit, pd.DataFrame) else np.unique(y_train_unit))
        uid_bytes = ",".join(map(str, uid_arr)).encode("utf-8")
        meta["unit_ids_hash"] = hashlib.sha1(uid_bytes).hexdigest()
        meta["n_units_train"] = int(len(uid_arr))

    with open(os.path.join(out_dir, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    print(f"[save_clf_artifacts] {out_dir} 저장 완료 "
          f"(fold_models.pkl + best_params.json + 6 CSV)")


# ------------------------------------------------------------
# 탐색 범위 / anchor enqueue 헬퍼
# ------------------------------------------------------------

def narrow_around(anchor, log_keys=None, int_keys=None, cat_keys=None,
                  ratio=0.30, int_step_ratio=0.30):
    """anchor(1차 best HP) dict → 그 주변으로 좁힌(±ratio) 탐색 범위 dict 자동 생성.

    Parameters
    ----------
    anchor : dict[str, value] — 기준 HP. 값 타입(float/int/str)에 따라 처리 분기.
    log_keys : set[str] or None — log-uniform으로 다룰 float 키. log 공간에서 ±ratio.
    int_keys : set[str] or None — 정수형 키. 명시 안 하면 anchor 값이 int인 키를 자동 감지.
    cat_keys : set[str] or None — categorical 키. 범위가 아니라 anchor 값 하나만 choices로.
    ratio : float (default 0.30) — 연속형 ±폭 (0.30 → ±30%).
    int_step_ratio : float (default 0.30) — 정수형 ±폭 비율 (최소 ±1).

    Returns
    -------
    dict[str, dict] — 키별로
        {'type':'float','low':..,'high':..,'log':bool} / {'type':'int','low':..,'high':..} / {'type':'cat','choices':[..]}
    이 dict는 sample_from_space()에 넘겨 trial.suggest_*로 푼다.
    """
    log_keys = set(log_keys or [])
    int_keys_explicit = set(int_keys or [])
    cat_keys = set(cat_keys or [])
    space = {}
    for k, v in anchor.items():
        if k in cat_keys:
            space[k] = {"type": "cat", "choices": [v]}   # 고정값 1개
            continue
        # 정수 키 자동 감지 (명시된 int_keys 우선, bool은 제외, log_keys면 정수 취급 안 함)
        is_int = (k in int_keys_explicit) or (
            isinstance(v, (int, np.integer)) and not isinstance(v, bool)
            and k not in log_keys
        )
        if is_int:
            v_int = int(v)
            step = max(1, int(round(abs(v_int) * int_step_ratio)))
            space[k] = {
                "type": "int",
                "low":  max(1, v_int - step) if v_int > 0 else v_int - step,   # 양수면 1 미만으로 안 내려감
                "high": v_int + step,
            }
            continue
        # float 키
        v_f = float(v)
        if k in log_keys:
            if v_f <= 0:
                raise ValueError(f"log_keys '{k}'는 양수만 가능 (anchor={v_f})")
            # log 공간에서 ±log(1+ratio) → 곱셈적 ±ratio
            log_v = np.log(v_f)
            log_low = log_v - np.log(1.0 + ratio)
            log_high = log_v + np.log(1.0 + ratio)
            space[k] = {
                "type": "float",
                "low":  float(np.exp(log_low)),
                "high": float(np.exp(log_high)),
                "log":  True,
            }
        else:
            spread = abs(v_f) * ratio
            space[k] = {
                "type": "float",
                "low":  v_f - spread,
                "high": v_f + spread,
                "log":  False,
            }
    return space


def sample_from_space(trial, space):
    """narrow_around() 결과 dict → trial.suggest_*(...)를 일괄 호출해 sampled HP dict 반환."""
    out = {}
    for k, spec in space.items():
        t = spec["type"]
        if t == "float":
            out[k] = trial.suggest_float(k, spec["low"], spec["high"],
                                          log=spec.get("log", False))
        elif t == "int":
            out[k] = trial.suggest_int(k, spec["low"], spec["high"])
        elif t == "cat":
            out[k] = trial.suggest_categorical(k, spec["choices"])
        else:
            raise ValueError(f"Unknown spec type {t!r} for key {k}")
    return out


def enqueue_anchor(study, anchor):
    """anchor를 study의 첫 trial로 강제 (1차 best HP를 trial 0에 박제) — study.enqueue_trial의 얇은 래퍼.

    study.optimize() 호출 전에 부른다. RESUME 모드(이미 trial이 있는 study)면 다시 enqueue하지 않는다 (중복 방지).
    anchor의 categorical 값은 search space의 choices 안에 있어야 한다.
    """
    if len(study.trials) == 0:
        study.enqueue_trial(dict(anchor))
        print(f"[enqueue] anchor 첫 trial로 강제 ({len(anchor)} HP)")
    else:
        print(f"[enqueue skip] 기존 trial {len(study.trials)}개 — resume 모드")
