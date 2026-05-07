"""
Final 파이프라인 — HPO (Optuna) + Best Trial Refit

- die-level 학습, unit-level RMSE objective (OOF 예측을 unit 집계 후 RMSE)
- KFold split은 반드시 **unit 단위** (같은 unit의 4 die가 train/val에 섞이면 leakage)
- 모델 선택(categorical) + HP 동시 탐색 지원 (models_to_search 리스트)
- extra_feature 지원: 경로 B에서 `(1-π_zit)` 컬럼을 OOF 기반으로 삽입
- refit_best: best trial을 K-fold로 재학습, OOF/val/test 예측 저장
- ZITboost 한정: π/μ 컴포넌트도 함께 반환 (predict_components 호출)

사용법
------
    # HPO
    res = hpo.run_hpo(
        xs_train, ys_train_unit, feat_cols,
        models_to_search=['lgbm', 'xgb', 'catboost', 'et', 'enet'],
        n_trials=100, n_folds=5,
        study_name='final-C',  storage=None,
    )
    study       = res['study']
    best_params = res['best_params']
    best_model  = res['model_name']

    # Refit
    final = hpo.refit_best(
        xs_train, xs_val, xs_test, ys_train_unit, feat_cols,
        model_name=best_model, best_params=best_params, n_folds=5,
    )
"""
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import KFold

from utils.config import SEED, KEY_COL, TARGET_COL

from . import models as _models
from . import scaler as _scaler


# ═════════════════════════════════════════════════════════════
# Unit-level KFold split → die-level index
# ═════════════════════════════════════════════════════════════

def _make_unit_folds(unit_ids, n_splits, seed=SEED):
    """unit id 배열을 n_splits fold로 나눔.

    Returns
    -------
    list of (train_units, val_units) — 각 tuple은 np.array
    """
    unique = np.asarray(unit_ids)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tr_idx, vl_idx in kf.split(unique):
        folds.append((unique[tr_idx], unique[vl_idx]))
    return folds


def _die_mask_from_units(xs, unit_set):
    """xs[KEY_COL]이 unit_set에 포함된 die-level mask."""
    return xs[KEY_COL].isin(unit_set).values


def _broadcast_y_to_die(xs, ys_unit):
    """unit-level y → die-level y (xs의 ufs_serial 순서 기준)."""
    y_map = ys_unit.set_index(KEY_COL)[TARGET_COL]
    return xs[KEY_COL].map(y_map).values.astype(float)


def _aggregate_die_to_unit(xs, die_pred):
    """die-level 예측 → unit-level (mean 집계).

    Returns
    -------
    pd.DataFrame  columns=[KEY_COL, 'pred']  (원본 unit 순서 보존)
    """
    df = pd.DataFrame({KEY_COL: xs[KEY_COL].values, "pred": die_pred})
    grp = df.groupby(KEY_COL, sort=False)["pred"].mean().reset_index()
    return grp


# ═════════════════════════════════════════════════════════════
# Fit + Predict (모델별 분기 처리)
# ═════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════
# N_JOBS 주입 헬퍼 (strategy_common §8 정책 정상화)
# ═════════════════════════════════════════════════════════════
# 노트북 최상단의 단일 N_JOBS 변수를 모델별 적절한 키(`n_jobs` / `thread_count`)로
# 주입한다. n_jobs=None이면 라이브러리 default(보통 -1, 전체 코어)로 폴백 →
# 기존 (호출자가 미전달) 동작과 호환.

_NJOBS_KEYED   = {"lgbm", "xgb", "et", "zitboost"}     # n_jobs 키 사용
_THREADCOUNT   = {"catboost"}                          # thread_count 키 사용
# enet (sklearn ElasticNet) 은 n_jobs 파라미터 자체가 없음 → 무시


def _inject_n_jobs(model_name, params, n_jobs):
    """모델별 적절한 키로 N_JOBS 덮어쓰기 (override).

    - None이면 무처리 (search_space의 -1 또는 라이브러리 default 유지)
    - lgbm/xgb/et/zitboost: params['n_jobs'] = n_jobs
    - catboost: params['thread_count'] = n_jobs (n_jobs 키가 있으면 제거)
    - enet: 무시
    """
    if n_jobs is None:
        return params
    p = dict(params)
    if model_name in _THREADCOUNT:
        p["thread_count"] = int(n_jobs)
        p.pop("n_jobs", None)   # 혹시 베이크된 n_jobs는 catboost와 충돌하지 않지만 정리
    elif model_name in _NJOBS_KEYED:
        p["n_jobs"] = int(n_jobs)
    return p


def _fit_predict_fold(
    model_name, hp,
    X_tr, y_tr, X_vl,
    return_components=False,
):
    """단일 fold 학습 + val 예측. ZITboost는 필요 시 (π, μ, pred) 추가 반환.

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
    """feat_cols 기반 X matrix 구성. extra_feature는 die 인덱스에 맞는 1D array.

    Parameters
    ----------
    xs_split : DataFrame (die-level)
    feat_cols : list[str]
    extra_feature : (name, np.ndarray) or None
        die-level 배열. len(extra_feature[1]) == len(xs_split).
    """
    X = xs_split[feat_cols].values
    if extra_feature is not None:
        _, arr = extra_feature
        arr = np.asarray(arr).reshape(-1, 1)
        if arr.shape[0] != len(xs_split):
            raise ValueError(
                f"extra_feature length {arr.shape[0]} != xs_split length {len(xs_split)}"
            )
        X = np.hstack([X, arr])
    return X


# ═════════════════════════════════════════════════════════════
# Optuna HPO
# ═════════════════════════════════════════════════════════════

def run_hpo(
    xs_train, ys_train_unit, feat_cols,
    model_name,
    n_trials=100, n_folds=5,
    extra_feature_train=None,     # (name, die-level array) or None  — reg 입력 피처로 추가
    multiplier_train=None,        # die-level array or None  — 최종 예측 (1-π)×reg_pred 곱셈용
    y_positive_only=False,        # True면 fit 데이터에서 y==0 필터링 (정석 Two-Stage)
    target_transform_fn=None,     # y → y_transformed (fit 전 적용)
    target_inverse_fn=None,       # y_transformed → y (predict 후 적용)
    study_name=None, storage=None,
    resume_study=False,           # True로 명시해야만 기존 study에 trial append
    seed=SEED, direction="minimize",
    show_progress_bar=True,
    user_attrs=None,
    space_variant="default",      # 'default' | 'zitreg' — models.get_search_space 의 variant
    # ── strategy_common §4·§5·§25 ──
    sampler=None,                 # None이면 TPESampler(seed=seed). §4: TPESampler(seed=None, multivariate=True, group=True)
    pruner=None,                  # None이면 사용 안 함. §4: MedianPruner(n_warmup_steps=...)
    enqueue_trials=None,          # list[dict] — anchor 첫 trial 강제 (§5)
    timeout=None,                 # 초 단위, None=무제한 (§25)
    n_jobs=None,                  # 모델 학습 병렬도 (strategy_common §8). None이면 라이브러리 default(-1)
    # ── trial별 holdout 평가 (옵션) ──
    # xs_val/ys_val_unit 둘 다 주면 매 trial 마다 fold-평균 val 예측 → val_rmse 기록.
    # extra_feature_val/multiplier_val 은 *_train 과 동일한 의미로 val 측에 적용.
    xs_val=None, ys_val_unit=None,
    extra_feature_val=None, multiplier_val=None,
    xs_test=None, ys_test_unit=None,
    extra_feature_test=None, multiplier_test=None,
):
    """die-level KFold OOF → unit RMSE를 최소화하는 Optuna study 실행.

    **단일 모델 HPO** — 모델 선택은 노트북 레벨에서 하고,
    한 study는 한 모델의 HP만 탐색한다.

    Parameters
    ----------
    xs_train : DataFrame (die-level, KEY_COL 컬럼 포함)
    ys_train_unit : DataFrame (unit-level, KEY_COL + TARGET_COL)  **원본 스케일**
    feat_cols : list[str]
    model_name : str
        MODEL_REGISTRY 이름 1개. 'lgbm' / 'xgb' / 'catboost' / 'et' / 'enet' / 'zitboost'.
    n_trials, n_folds : int
    extra_feature_train : (name, array) or None
        reg 입력에 die-level 컬럼을 1개 추가.
    multiplier_train : array or None
        최종 예측을 `reg_pred × multiplier_train` 형태로 바꾼다. 경로 B 정석
        Two-Stage 에서 `(1-π_zit)` 를 전달. objective RMSE 도 곱셈 후 값으로 계산 →
        "HPO 가 최적화하는 수식 == 최종 제출 수식" 일관성 확보.
    y_positive_only : bool
        True 면 fold 학습 데이터에서 `y == 0` 인 die 를 제외. 정석 Two-Stage 의
        Stage 2 회귀 정의(“Y>0 서브셋으로 학습 → E[Y|Y>0,x] 예측”)를 따름.
        multiplier_train 과 함께 쓰면 최종 E[Y] = P(Y>0|x) × E[Y|Y>0,x].
    target_transform_fn : callable or None
        y → y_transformed (모델 fit 입력용). 예: np.log1p.
        None이면 원본 그대로.
    target_inverse_fn : callable or None
        y_transformed → y (모델 predict 출력 역변환용). 예: np.expm1 + clip.
        `target_transform_fn` 있으면 반드시 쌍으로 제공.
    study_name, storage : Optuna study 옵션 (SQLite 경로 등)
    user_attrs : dict 저장할 메타데이터
    xs_val, ys_val_unit : DataFrame, DataFrame (optional, 짝)
        주면 매 trial 마다 fold 평균 val 예측 → unit RMSE 계산하여
        `trial.set_user_attr("val_rmse", ...)` 로 기록.
    xs_test, ys_test_unit : (optional, 짝) 동일하게 test_rmse 기록.
    extra_feature_val/test, multiplier_val/test :
        *_train 과 동일한 의미로 val/test 측에 적용.
        `extra_feature_train` 있을 때 *_val/test 도 모양/길이 맞춰 제공해야 함.

    Returns
    -------
    dict  {'study', 'best_params', 'model_name', 'best_value'}
    """
    if (target_transform_fn is None) != (target_inverse_fn is None):
        raise ValueError("target_transform_fn / target_inverse_fn은 쌍으로 제공")
    if (xs_val is None) != (ys_val_unit is None):
        raise ValueError("xs_val / ys_val_unit 은 쌍으로 제공해야 함")
    if (xs_test is None) != (ys_test_unit is None):
        raise ValueError("xs_test / ys_test_unit 은 쌍으로 제공해야 함")
    space_fn = _models.get_search_space(model_name, variant=space_variant)

    # unit 수준 KFold split을 trial 전체에서 재사용 (공정성)
    unit_ids = ys_train_unit[KEY_COL].unique()
    folds = _make_unit_folds(unit_ids, n_folds, seed)
    # die-level mask 미리 계산 (성능)
    fold_masks = [
        (_die_mask_from_units(xs_train, set(tr)),
         _die_mask_from_units(xs_train, set(vl)))
        for tr, vl in folds
    ]

    y_die_orig = _broadcast_y_to_die(xs_train, ys_train_unit)
    # NOTE: y_die_fit은 trial 별로 결정 (tweedie 계열이면 transform OFF — strategy.md §4)
    X_full      = _build_X(xs_train, feat_cols, extra_feature_train)
    y_true_unit = ys_train_unit.set_index(KEY_COL)[TARGET_COL]   # 원본 스케일

    # ── eval set X 사전 구성 (trial 전체에서 1회만) ──
    X_val_full = (_build_X(xs_val, feat_cols, extra_feature_val)
                  if xs_val is not None else None)
    X_test_full = (_build_X(xs_test, feat_cols, extra_feature_test)
                   if xs_test is not None else None)
    y_val_true_unit = (ys_val_unit.set_index(KEY_COL)[TARGET_COL]
                       if ys_val_unit is not None else None)
    y_test_true_unit = (ys_test_unit.set_index(KEY_COL)[TARGET_COL]
                        if ys_test_unit is not None else None)

    # multiplier 배열 검증 (train/val/test 공통)
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
        """sample된 hp에 tweedie 계열 손실함수가 들어있으면 True.
        EXPERIMENT_LOG §5.1: tweedie 분포는 right-skew 자체 모델링 → log1p 와 충돌.
        """
        s = str(hp.get("objective") or hp.get("loss_function") or "")
        return s.startswith("tweedie") or s.startswith("reg:tweedie") or s.lower().startswith("tweedie")

    def objective(trial):
        hp = space_fn(trial)
        # N_JOBS 주입 (strategy_common §8): search_space의 -1 베이크값을 노트북 N_JOBS로 override
        hp = _inject_n_jobs(model_name, hp, n_jobs)

        # ── trial별 target_transform 분기 (strategy.md §4, EXPERIMENT_LOG §5.1) ──
        # tweedie 계열 loss가 sample되면 log1p OFF (이중 변환 방지)
        if _is_tweedie_hp(hp):
            eff_transform_fn = None
            eff_inverse_fn   = None
        else:
            eff_transform_fn = target_transform_fn
            eff_inverse_fn   = target_inverse_fn
        trial.set_user_attr("target_transform_active", eff_transform_fn is not None)

        y_die_fit_local = eff_transform_fn(y_die_orig) if eff_transform_fn else y_die_orig

        def _eval_split_rmse_local(xs_split, die_pred_accum, multiplier, y_true_unit_split):
            """fold 평균된 die 예측 → trial별 inverse → multiplier → unit 집계 → RMSE."""
            pred = eff_inverse_fn(die_pred_accum) if eff_inverse_fn else die_pred_accum
            if multiplier is not None:
                pred = pred * multiplier
            unit = _aggregate_die_to_unit(xs_split, pred)
            aligned = unit.set_index(KEY_COL)["pred"].loc[y_true_unit_split.index]
            return float(np.sqrt(np.mean((aligned.values - y_true_unit_split.values) ** 2)))

        oof = np.full(len(xs_train), np.nan)
        val_pred_accum  = (np.zeros(len(xs_val))  if xs_val  is not None else None)
        test_pred_accum = (np.zeros(len(xs_test)) if xs_test is not None else None)

        for tr_mask, vl_mask in fold_masks:
            # ── 학습 데이터 필터링 (정석 Two-Stage: Y>0 만) ──
            if y_positive_only:
                fit_mask = tr_mask & (y_die_orig > 0)
            else:
                fit_mask = tr_mask
            X_tr, y_tr = X_full[fit_mask], y_die_fit_local[fit_mask]
            X_vl       = X_full[vl_mask]

            # scaler: enet 일 때만. fit-on-train → val/test 도 동일 변환
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
            oof[vl_mask] = res["pred"]

            # ── eval set 예측 (fold 평균) ──
            if X_eval_v is not None:
                val_pred_accum  += res["model"].predict(X_eval_v)  / n_folds
            if X_eval_t is not None:
                test_pred_accum += res["model"].predict(X_eval_t) / n_folds

        if np.isnan(oof).any():
            raise RuntimeError("OOF has NaN — fold coverage bug")

        # ── train OOF RMSE (objective 반환값) ──
        train_rmse = _eval_split_rmse_local(xs_train, oof, multiplier_train, y_true_unit)
        # 명시적으로 trial.set_user_attr 에도 기록 — Optuna dashboard 에서 라벨링 명확화
        trial.set_user_attr("train_rmse", train_rmse)

        # ── val/test RMSE (옵션) ──
        if val_pred_accum is not None:
            val_rmse = _eval_split_rmse_local(xs_val, val_pred_accum,
                                              multiplier_val, y_val_true_unit)
            trial.set_user_attr("val_rmse", val_rmse)
        if test_pred_accum is not None:
            test_rmse = _eval_split_rmse_local(xs_test, test_pred_accum,
                                               multiplier_test, y_test_true_unit)
            trial.set_user_attr("test_rmse", test_rmse)

        return train_rmse

    # resume_study=False 기본: 같은 study_name/storage 조합이 이미 있으면 에러를
    # 명시적으로 내어 trial 누적으로 best 값이 오염되는 것을 막는다.
    # sampler default: strategy_common §4에 따라 multivariate=True, group=True 적용.
    # seed는 함수 인자 그대로 (None 권장 — 다양성 확보; 노트북이 명시 주입 권장).
    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(storage and resume_study),
        sampler=sampler if sampler is not None else optuna.samplers.TPESampler(
            seed=seed, multivariate=True, group=True
        ),
        pruner=pruner,    # None이면 NopPruner default (Optuna 자동)
    )
    if user_attrs:
        for k, v in user_attrs.items():
            study.set_user_attr(k, v)

    # ── anchor enqueue (§5): 첫 trial(들) 강제 ──
    if enqueue_trials:
        for anchor in enqueue_trials:
            study.enqueue_trial(dict(anchor))
        print(f"[enqueue] {len(enqueue_trials)} anchor trial(s) 강제")

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


# ═════════════════════════════════════════════════════════════
# Best Trial Refit (K-fold)
# ═════════════════════════════════════════════════════════════

def _hp_from_best(best_params, model_name, n_jobs=None):
    """Optuna best_params dict → MODEL_REGISTRY에 전달할 kwargs.

    새 search space (strategy.md §6):
    - LGBM `objective`: 'regression' | 'poisson' | 'tweedie'  (+ tweedie 시 별도 'tweedie_variance_power' float)
    - XGB  `objective`: 'reg:squarederror' | 'count:poisson' | 'reg:tweedie'  (+ tweedie 시 'tweedie_variance_power')
    - CatBoost `loss_function`: 'RMSE' | 'Poisson' | 'Tweedie'  (+ Tweedie 시 'tweedie_variance_power' → 'Tweedie:variance_power=…' 변환)
    """
    hp = dict(best_params)
    # CatBoost: 'Tweedie' + 'tweedie_variance_power' → 'Tweedie:variance_power=...'
    if model_name == "catboost":
        loss = hp.get("loss_function")
        if loss == "Tweedie":
            power = hp.pop("tweedie_variance_power", 1.5)
            hp["loss_function"] = f"Tweedie:variance_power={power}"
    # LGBM / XGB는 사이킷 키 그대로 ('tweedie_variance_power'는 그대로 model에 전달)

    # 공통 고정값 (search space가 이미 주입했지만 refit 경로에서도 보장)
    from utils.config import SEED as _S
    if model_name in {"lgbm", "zitboost"}:
        hp.setdefault("random_state", _S)
        hp.setdefault("n_jobs", -1)
        hp.setdefault("verbose", -1)
        hp.setdefault("device", _models.DEVICE)
        # subsample_freq 없으면 LGBM이 subsample을 무시함. search space에서도
        # 고정 1을 넣지만 REUSE 모드 하위호환용으로 refit에서도 보장.
        if model_name == "lgbm":
            hp.setdefault("subsample_freq", 1)
    elif model_name == "xgb":
        hp.setdefault("random_state", _S)
        hp.setdefault("n_jobs", -1)
        hp.setdefault("tree_method", "hist")
        hp.setdefault("verbosity", 0)
    elif model_name == "catboost":
        hp.setdefault("random_seed", _S)
        hp.setdefault("verbose", False)
        hp.setdefault("allow_writing_files", False)
    elif model_name == "et":
        hp.setdefault("random_state", _S)
        hp.setdefault("n_jobs", -1)
    elif model_name == "enet":
        hp.setdefault("random_state", _S)
        hp.setdefault("tol", 1e-6)
        hp.setdefault("selection", "random")
        hp.setdefault("precompute", True)

    # N_JOBS override (strategy_common §8): n_jobs 인자가 들어왔으면 모델별 키로 덮어쓰기
    hp = _inject_n_jobs(model_name, hp, n_jobs)
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
    """Best trial HP로 K-fold 재학습. die-level OOF + val/test 예측 (fold 평균) 생성.

    ZITboost일 때 π·μ 컴포넌트도 함께 반환.
    target_transform_fn을 주면 학습은 transformed space, 출력은 original space.

    multiplier_* + y_positive_only : 정석 Two-Stage (경로 B) 지원.
      - y_positive_only=True: fit 데이터에서 y==0 die 제외 → E[Y|Y>0,x] 학습
      - multiplier_*: 최종 예측을 `reg_pred × multiplier` 로 변환 → (1-π)×E[Y|Y>0,x] = E[Y|x]

    Returns
    -------
    dict {
        'oof_pred_die':  array (len train-die) — **original space, multiplier 적용 후**,
        'val_pred_die':  array (len val-die)   — 동일,
        'test_pred_die': array (len test-die)  — 동일,
        'oof_pi', 'val_pi', 'test_pi': array or None (ZITboost만),
        'oof_mu', 'val_mu', 'test_mu': array or None (ZITboost만),
        'oof_pred_unit': DataFrame [KEY_COL, pred] — original,
        'val_pred_unit': DataFrame — original,
        'test_pred_unit': DataFrame — original,
        'fold_models': list,
        'fold_scalers': list — fold별 {'median', 'iqr'} dict 또는 None (스케일링 안 하는 모델),
        'best_params_resolved': dict,
    }
    """
    if (target_transform_fn is None) != (target_inverse_fn is None):
        raise ValueError("target_transform_fn / target_inverse_fn은 쌍으로 제공")
    # REUSE 모드: best_params_resolved(JSON)를 그대로 받은 경우 재변환 스킵
    if already_resolved:
        hp = dict(best_params)
        hp = _inject_n_jobs(model_name, hp, n_jobs)   # REUSE도 N_JOBS override 보장
    else:
        hp = _hp_from_best(best_params, model_name, n_jobs=n_jobs)

    # ── refit도 HPO와 동일하게 tweedie 계열 시 transform OFF (EXPERIMENT_LOG §5.1) ──
    _obj_or_loss = str(hp.get("objective") or hp.get("loss_function") or "")
    if (_obj_or_loss.startswith("tweedie") or _obj_or_loss.startswith("reg:tweedie")
            or _obj_or_loss.lower().startswith("tweedie")):
        if target_transform_fn is not None:
            print(f"[refit] tweedie loss 감지 → target_transform OFF (EXPERIMENT_LOG §5.1)")
        target_transform_fn = None
        target_inverse_fn   = None

    # splits
    unit_ids = ys_train_unit[KEY_COL].unique()
    folds = _make_unit_folds(unit_ids, n_folds, seed)
    y_die_train_orig = _broadcast_y_to_die(xs_train, ys_train_unit)
    y_die_train_fit  = target_transform_fn(y_die_train_orig) \
        if target_transform_fn else y_die_train_orig

    X_train_full = _build_X(xs_train, feat_cols, extra_feature_train)
    X_val_full   = _build_X(xs_val,   feat_cols, extra_feature_val)
    X_test_full  = _build_X(xs_test,  feat_cols, extra_feature_test)

    n_tr, n_vl, n_te = len(xs_train), len(xs_val), len(xs_test)
    # multiplier 배열 검증
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

    oof_pred  = np.full(n_tr, np.nan)
    val_pred  = np.zeros(n_vl)
    test_pred = np.zeros(n_te)

    is_zit = model_name == "zitboost"
    oof_pi = np.full(n_tr, np.nan) if is_zit else None
    oof_mu = np.full(n_tr, np.nan) if is_zit else None
    val_pi = np.zeros(n_vl) if is_zit else None
    val_mu = np.zeros(n_vl) if is_zit else None
    test_pi = np.zeros(n_te) if is_zit else None
    test_mu = np.zeros(n_te) if is_zit else None

    fold_models = []
    fold_scalers = []   # enet 등 스케일링 모델: fold별 (median, iqr) 보관 → pkl 재현용

    if _scaler.needs_scaling(model_name):
        print(f"[scaler] {model_name} → fold-local RobustScaler 적용 "
              f"(매 fold train 기준 fit, holdout/val/test 동일 변환)")

    for i, (tr_units, vl_units) in enumerate(folds):
        tr_mask = _die_mask_from_units(xs_train, set(tr_units))
        vl_mask = _die_mask_from_units(xs_train, set(vl_units))

        # ── 정석 Two-Stage: Y>0 만 학습 ──
        if y_positive_only:
            fit_mask = tr_mask & (y_die_train_orig > 0)
        else:
            fit_mask = tr_mask
        X_tr, y_tr = X_train_full[fit_mask], y_die_train_fit[fit_mask]
        X_vl       = X_train_full[vl_mask]

        # 스케일링: enet이면 train-fold 기준으로 fit → val-fold/val/test 모두 transform
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

        # 예측은 transformed space → 역변환해서 accumulate
        pred_vl = model.predict(X_vl)
        pred_v  = model.predict(X_val_tr)
        pred_t  = model.predict(X_test_tr)
        if target_inverse_fn:
            pred_vl = target_inverse_fn(pred_vl)
            pred_v  = target_inverse_fn(pred_v)
            pred_t  = target_inverse_fn(pred_t)

        oof_pred[vl_mask] = pred_vl
        val_pred  += pred_v / n_folds
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
        raise RuntimeError("oof_pred has NaN — unit coverage bug")

    # ── multiplier 적용 (정석 Two-Stage 경로 B) ──
    # 곱셈을 여기서 해야 oof/val/test 가 모두 "최종 예측" 의미로 통일된다.
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


# ═════════════════════════════════════════════════════════════
# Artifact 저장 (pkl + CSV + JSON)
# ═════════════════════════════════════════════════════════════

from utils.config import DIE_KEY_COL as _DIE_KEY_COL


def _die_csv(xs_split, pred, pi=None, mu=None, y_unit=None):
    """die-level 예측을 KEY_COL + DIE_KEY_COL 과 함께 DataFrame으로.

    `y_unit` (DataFrame `[KEY_COL, TARGET_COL]` 또는 Series indexed by KEY_COL)이
    주어지면 unit-level health 를 die-level 로 broadcast 하여 `health` 컬럼 추가.
    test split 처럼 y 가 없으면 None 으로 두면 컬럼 자체가 빠진다.
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
        out["one_minus_pi"] = 1.0 - pi   # 경로 B에서 바로 쓰기 쉬우라고 파생
    if mu is not None:
        out["mu"] = mu
    return out


def _add_health_to_unit(unit_df, y_unit):
    """unit-level [KEY_COL, 'pred'] DataFrame 에 health 컬럼을 merge."""
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
    """refit_best 결과를 디스크에 저장.

    Parameters
    ----------
    feature_names : list[str] or None
        학습에 사용된 피처 이름 (재현/SHAP/importance 용). best_params.json + pkl 에 저장.
    extra_feature_name : str or None
        경로 B 처럼 X 뒤에 붙은 추가 피처 이름 (예: 'one_minus_pi'). 저장 전용 메타.
    y_train_unit : DataFrame or None
        postprocess_config 가 주어질 때 필수. unit RMSE 기반으로 집계/threshold 튜닝.
        die/unit CSV 의 `health` 컬럼 도 여기서 가져온다.
    y_val_unit, y_test_unit : DataFrame or None
        주어지면 val/test die·unit CSV 에 `health` 컬럼 merge.
        None 이면 컬럼 자체가 빠진다 (test 가 비공개일 때 등).
    postprocess_config : dict or None
        None 이면 기존 mean 집계만 저장 (backward-compat).
        dict 이면 postprocess.tune_and_apply 에 kwargs 로 전달하여
        unit CSV 를 튜닝된 값으로 대체 저장.
        예: {'agg_methods': (...), 'pi_threshold_range': (...), ...}
    study_meta : dict or None
        study.user_attrs 같은 재현성 메타. best_params.json 에 그대로 저장.

    생성물
    ------
    - {out_dir}/fold_models.pkl      : {'fold_models', 'fold_scalers', 'feature_names', ...}
    - {out_dir}/best_params.json     : model_name + resolved HP + feature_names + study_meta
    - {out_dir}/oof_die.csv          : train OOF die-level (+ health, +pi/mu if ZIT)
    - {out_dir}/val_die.csv          : val die-level (+ health if y_val_unit 제공)
    - {out_dir}/test_die.csv         : test die-level (+ health if y_test_unit 제공)
    - {out_dir}/oof_unit.csv         : train OOF unit-level (postprocess tuned if config 제공)
    - {out_dir}/val_unit.csv         : val unit-level (동일)
    - {out_dir}/test_unit.csv        : test unit-level (동일)

    경로 B는 {out_dir}/oof_die.csv · val_die.csv · test_die.csv 의
    `one_minus_pi` 컬럼을 reg 입력 피처로 재사용한다.
    """
    import os, json, pickle
    os.makedirs(out_dir, exist_ok=True)

    # 1) fold models + feature_names (pkl)
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

    # 2) best_params (JSON) + study 메타 + fold 재현성 정보
    # strategy_common.md §23.3 키 위치 정합: effective_pp_params를 study_meta 하위가 아닌 top-level로
    meta = {
        "exp_id":                exp_id,
        "model_name":            refit_result["model_name"],
        "best_params_resolved":  refit_result["best_params_resolved"],
        "effective_pp_params":   (study_meta or {}).get("effective_pp_params"),
        "feature_names":         list(feature_names) if feature_names is not None else None,
        "n_features":            len(feature_names) if feature_names is not None else None,
        "extra_feature_name":    extra_feature_name,
        "n_folds":               len(refit_result["fold_models"]),
        "study_meta":            study_meta or {},
    }
    # fold 분할 재현성 (01↔03 alignment 검증용)
    if y_train_unit is not None:
        import hashlib
        uid_arr = y_train_unit[KEY_COL].unique()
        uid_bytes = ",".join(map(str, uid_arr)).encode("utf-8")
        meta["unit_ids_hash"] = hashlib.sha1(uid_bytes).hexdigest()
        meta["n_units_train"] = int(len(uid_arr))

    # 3) die-level CSV (tune 이전 raw die 예측 그대로) + health merge
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

    # 4) unit-level CSV — postprocess_config 여부로 분기 + health merge
    if postprocess_config is not None and y_train_unit is not None:
        from . import postprocess as _pp
        pp_res = _pp.tune_and_apply(
            xs_train, xs_val, xs_test,
            die_pred_train=refit_result["oof_pred_die"],
            die_pred_val=refit_result["val_pred_die"],
            die_pred_test=refit_result["test_pred_die"],
            y_train_unit=y_train_unit,
            y_val_unit=y_val_unit,                       # ★ val 비교용 (§10·§11·§12)
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
        # best tuning 결과도 메타에 기록
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
        # 기존 동작: refit 단계의 mean 집계 그대로 + health merge
        _add_health_to_unit(refit_result["oof_pred_unit"], y_train_unit).to_csv(
            os.path.join(out_dir, "oof_unit.csv"), index=False)
        _add_health_to_unit(refit_result["val_pred_unit"], y_val_unit).to_csv(
            os.path.join(out_dir, "val_unit.csv"), index=False)
        _add_health_to_unit(refit_result["test_pred_unit"], y_test_unit).to_csv(
            os.path.join(out_dir, "test_unit.csv"), index=False)
        meta["postprocess"] = None

    # best_params.json 저장 (postprocess 결과까지 포함된 최종 meta)
    with open(os.path.join(out_dir, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    pp_tag = "tuned" if meta.get("postprocess") else "mean"
    print(f"[save_artifacts] {out_dir} 저장 완료 "
          f"(fold_models.pkl + best_params.json + 6 CSV, unit={pp_tag})")


def load_extra_feature_from_path(path_dir, xs_split, feature_col="one_minus_pi",
                                 split=None):
    """경로 B 용: 경로 A가 저장한 die-level CSV에서 (1-π)를 추출.

    die 순서는 xs_split의 DIE_KEY_COL 기준으로 정렬.

    Parameters
    ----------
    path_dir : str
        경로 A가 save_artifacts로 저장한 디렉토리.
    xs_split : DataFrame
        현재 노트북의 xs_train / xs_val / xs_test 중 하나.
    feature_col : str  (default 'one_minus_pi')
    split : {'train','val','test'} or None
        명시하면 oof/val/test CSV 를 **직접 지정**해 로드. 기본 None 이면
        길이/키셋 일치 기반으로 자동 감지.

    Returns
    -------
    np.ndarray  — xs_split과 동일 길이의 die-level 배열
    """
    import os

    CSV_BY_SPLIT = {
        "train": "oof_die.csv",
        "val":   "val_die.csv",
        "test":  "test_die.csv",
    }

    split_keys = set(xs_split[_DIE_KEY_COL].values)
    n_split = len(xs_split)

    def _try_load(csv_name):
        full = os.path.join(path_dir, csv_name)
        if not os.path.exists(full):
            return None
        df = pd.read_csv(full)
        if feature_col not in df.columns:
            raise ValueError(
                f"{csv_name}에 컬럼 {feature_col!r} 없음 — "
                f"경로 A 아티팩트가 맞는지 확인하세요."
            )
        # 엄격 검증: 길이 일치 + 키셋 일치 (부분 매칭 방지)
        if len(df) != n_split:
            return ("length_mismatch", len(df))
        if set(df[_DIE_KEY_COL].values) != split_keys:
            return ("key_mismatch", None)
        aligned = df.set_index(_DIE_KEY_COL).loc[
            xs_split[_DIE_KEY_COL].values, feature_col
        ].values
        return aligned.astype(float)

    # 명시 split: 해당 파일만 시도, 실패하면 에러
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

    # 자동 감지: 길이+키 모두 일치하는 CSV 탐색
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


# ═════════════════════════════════════════════════════════════
# CLF (Stage 1) — Two-Stage 03c 노트북 전용
# ═════════════════════════════════════════════════════════════
# binary classification (y>0 vs y=0). die-level 학습 + unit mean 집계.
#
# Objective (RMSE-aligned):
#   oof_proba_die → unit mean → × y_pos_const → unit pred → unit RMSE
#   "회귀를 평균값 상수로 가정" 한 RMSE 평가. clf calibration 단독 평가.
#   y_pos_const = mean(y_train_unit | y > 0)  ← E[Y | Y>0]
#
# KFold split 은 unit 단위 (run_hpo 와 동일 패턴, leakage 방지).

def run_clf_hpo(
    xs_train, ys_train_unit, feat_cols,
    model_name,
    n_trials=100, n_folds=5,
    study_name=None, storage=None,
    resume_study=False,
    seed=SEED,
    show_progress_bar=True,
    user_attrs=None,
    # ── strategy_common §4·§5·§25 ──
    sampler=None,                 # None이면 TPESampler(seed=seed). §4
    pruner=None,                  # None이면 사용 안 함. §4
    enqueue_trials=None,          # list[dict] — anchor 첫 trial 강제 (§5)
    timeout=None,                 # 초 단위, None=무제한 (§25)
    n_jobs=None,                  # 모델 학습 병렬도 (strategy_common §8). None이면 라이브러리 default(-1)
    xs_val=None, ys_val_unit=None,
    xs_test=None, ys_test_unit=None,
):
    """die-level binary CLF HPO. Objective = unit RMSE (clf_proba × y_pos_const).

    Returns
    -------
    dict {'study', 'best_params', 'model_name', 'best_value', 'y_pos_const'}
    """
    if (xs_val is None) != (ys_val_unit is None):
        raise ValueError("xs_val / ys_val_unit 은 쌍으로 제공")
    if (xs_test is None) != (ys_test_unit is None):
        raise ValueError("xs_test / ys_test_unit 은 쌍으로 제공")

    space_fn = _models.get_clf_search_space(model_name)

    unit_ids = ys_train_unit[KEY_COL].unique()
    folds = _make_unit_folds(unit_ids, n_folds, seed)
    fold_masks = [
        (_die_mask_from_units(xs_train, set(tr)),
         _die_mask_from_units(xs_train, set(vl)))
        for tr, vl in folds
    ]

    y_die_orig = _broadcast_y_to_die(xs_train, ys_train_unit)
    y_die_bin  = (y_die_orig > 0).astype(int)
    X_train = xs_train[feat_cols].values
    X_val   = xs_val[feat_cols].values  if xs_val  is not None else None
    X_test  = xs_test[feat_cols].values if xs_test is not None else None

    y_true_unit      = ys_train_unit.set_index(KEY_COL)[TARGET_COL]
    y_val_true_unit  = (ys_val_unit.set_index(KEY_COL)[TARGET_COL]
                        if ys_val_unit is not None else None)
    y_test_true_unit = (ys_test_unit.set_index(KEY_COL)[TARGET_COL]
                        if ys_test_unit is not None else None)

    # E[Y | Y>0] — 정석 Two-Stage 의 Stage 2 평균값 상수로 대체
    y_pos_const = float(y_true_unit[y_true_unit > 0].mean())

    def _eval_split_rmse(xs_split, die_proba, y_true_unit_split):
        df = pd.DataFrame({KEY_COL: xs_split[KEY_COL].values, "p": die_proba})
        unit_proba = df.groupby(KEY_COL, sort=False)["p"].mean()
        unit_pred  = unit_proba * y_pos_const
        aligned    = unit_pred.loc[y_true_unit_split.index]
        return float(np.sqrt(np.mean(
            (aligned.values - y_true_unit_split.values) ** 2
        )))

    def objective(trial):
        params = space_fn(trial)
        # N_JOBS 주입 (strategy_common §8): clf search_space의 -1 베이크값 override
        params = _inject_n_jobs(model_name, params, n_jobs)
        oof_proba = np.full(len(xs_train), np.nan)
        val_proba_accum  = (np.zeros(len(xs_val))  if xs_val  is not None else None)
        test_proba_accum = (np.zeros(len(xs_test)) if xs_test is not None else None)

        for tr_mask, vl_mask in fold_masks:
            X_tr, y_tr = X_train[tr_mask], y_die_bin[tr_mask]
            X_vl       = X_train[vl_mask]

            params_resolved = _models.resolve_clf_imbalance(model_name, params, y_tr)
            clf = _models.create_classifier(model_name, params_resolved)
            clf.fit(X_tr, y_tr)

            oof_proba[vl_mask] = clf.predict_proba(X_vl)[:, 1]
            if X_val is not None:
                val_proba_accum  += clf.predict_proba(X_val)[:, 1]  / n_folds
            if X_test is not None:
                test_proba_accum += clf.predict_proba(X_test)[:, 1] / n_folds

        if np.isnan(oof_proba).any():
            raise RuntimeError("oof_proba has NaN — fold coverage bug")

        train_rmse = _eval_split_rmse(xs_train, oof_proba, y_true_unit)
        trial.set_user_attr("train_rmse", train_rmse)

        if val_proba_accum is not None:
            val_rmse = _eval_split_rmse(xs_val, val_proba_accum, y_val_true_unit)
            trial.set_user_attr("val_rmse", val_rmse)
        if test_proba_accum is not None:
            test_rmse = _eval_split_rmse(xs_test, test_proba_accum, y_test_true_unit)
            trial.set_user_attr("test_rmse", test_rmse)

        return train_rmse

    # sampler default: strategy_common §4 (multivariate=True, group=True) 정합
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

    # ── anchor enqueue (§5) ──
    if enqueue_trials:
        for anchor in enqueue_trials:
            study.enqueue_trial(dict(anchor))
        print(f"[enqueue] {len(enqueue_trials)} anchor trial(s) 강제")

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
    """best_params dict 에 모델별 search space 고정값 보강 (REUSE 모드 호환)."""
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
        hp.setdefault("random_state", SEED)
        hp.setdefault("n_jobs",       -1)
        hp.setdefault("bootstrap",    True)
    # N_JOBS override (strategy_common §8)
    hp = _inject_n_jobs(model_name, hp, n_jobs)
    return hp


def refit_clf_best(
    xs_train, xs_val, xs_test,
    ys_train_unit, feat_cols,
    model_name, best_params,
    n_folds=5, seed=SEED,
    already_resolved=False,
    n_jobs=None,
):
    """Best CLF HP 로 K-fold 재학습. die-level prob (OOF/val/test) 반환.

    Returns
    -------
    dict {
        'oof_proba_die', 'val_proba_die', 'test_proba_die': np.array,
        'fold_models': list,
        'best_params_resolved': dict,
        'model_name': str,
    }
    """
    if already_resolved:
        hp = dict(best_params)
        hp = _inject_n_jobs(model_name, hp, n_jobs)   # REUSE도 N_JOBS override
    else:
        hp = _clf_hp_with_defaults(model_name, best_params, n_jobs=n_jobs)

    unit_ids = ys_train_unit[KEY_COL].unique()
    folds = _make_unit_folds(unit_ids, n_folds, seed)
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

        hp_resolved = _models.resolve_clf_imbalance(model_name, hp, y_tr)
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
        raise RuntimeError("oof_proba has NaN — unit coverage bug")

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

    저장 파일:
      - oof_die.csv / val_die.csv / test_die.csv   (KEY, DIE_KEY, prob, [health])
      - oof_unit.csv / val_unit.csv / test_unit.csv (KEY, prob_unit_mean, pred=prob*y_pos_const, [health])
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

    # strategy_common.md §23.3 검수 체크리스트 6필드 top-level 보강
    # - effective_pp_params: study_meta에서 끌어올려 top-level로 (회귀 save_artifacts와 키 위치 정합)
    # - n_folds / unit_ids_hash / n_units_train: fold split 재현성 (zit↔reg↔stacking alignment)
    # - postprocess: clf 단독은 후처리 없음 — None + 사유 명시
    meta = {
        "exp_id":               exp_id,
        "model_name":           refit_result["model_name"],
        "best_params_resolved": refit_result["best_params_resolved"],
        "effective_pp_params":  (study_meta or {}).get("effective_pp_params"),
        "feature_names":        feature_names,
        "n_features":           len(feature_names) if feature_names else None,
        "n_folds":              len(refit_result["fold_models"]),
        "y_pos_const":          y_pos_const,
        "postprocess":          None,  # clf 단독은 후처리 없음 (combine 단계에서 적용)
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


# ═════════════════════════════════════════════════════════════
# Search range / anchor enqueue helpers (strategy.md §6, §5)
# ═════════════════════════════════════════════════════════════

def narrow_around(anchor, log_keys=None, int_keys=None, cat_keys=None,
                  ratio=0.30, int_step_ratio=0.30):
    """anchor dict → 자동 narrow search range dict.

    [strategy_common.md §5] 'A. Narrow' 방식 helper.

    Parameters
    ----------
    anchor : dict[str, value]
        기준 HP. value 타입(float/int/str)으로 처리 분기.
    log_keys : set[str] or None
        log-uniform 처리할 float 키. log-space에서 ±ratio.
    int_keys : set[str] or None
        정수형 키. anchor ± int_step_ratio*|anchor| (최소 1).
        명시 안 하면 anchor 값이 int 인 키 자동 감지.
    cat_keys : set[str] or None
        categorical 키. range가 아닌 anchor 값만 single-choice로.
    ratio : float (default 0.30)
        연속형 ±폭. 0.30이면 ±30%.
    int_step_ratio : float (default 0.30)
        정수형 ±폭 비율.

    Returns
    -------
    dict[str, dict] — 각 HP에 대해
        {'type': 'float'|'int'|'cat', 'low': ..., 'high': ..., 'log': bool, 'choices': [...]}

    Notes
    -----
    Optuna trial.suggest_*(...) 호출 시 이 dict의 키들을 풀어서 사용:

        for k, spec in space.items():
            if spec['type'] == 'float':
                trial.suggest_float(k, spec['low'], spec['high'], log=spec.get('log', False))
            elif spec['type'] == 'int':
                trial.suggest_int(k, spec['low'], spec['high'])
            elif spec['type'] == 'cat':
                trial.suggest_categorical(k, spec['choices'])
    """
    log_keys = set(log_keys or [])
    int_keys_explicit = set(int_keys or [])
    cat_keys = set(cat_keys or [])
    space = {}
    for k, v in anchor.items():
        if k in cat_keys:
            space[k] = {"type": "cat", "choices": [v]}
            continue
        # int 자동 감지 (명시 우선)
        is_int = (k in int_keys_explicit) or (
            isinstance(v, (int, np.integer)) and not isinstance(v, bool)
            and k not in log_keys
        )
        if is_int:
            v_int = int(v)
            step = max(1, int(round(abs(v_int) * int_step_ratio)))
            space[k] = {
                "type": "int",
                "low":  max(1, v_int - step) if v_int > 0 else v_int - step,
                "high": v_int + step,
            }
            continue
        # float
        v_f = float(v)
        if k in log_keys:
            if v_f <= 0:
                raise ValueError(f"log_keys '{k}'는 양수만 가능 (anchor={v_f})")
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
    """narrow_around() 결과 dict → trial.suggest_*(...) 일괄 호출.

    Returns
    -------
    dict[str, value] — sampled HP
    """
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
    """[strategy_common.md §5] anchor를 첫 trial로 강제 — 1차 best 보존.

    study.enqueue_trial(anchor)의 thin wrapper. 호출은 study.optimize 전.

    Parameters
    ----------
    study : optuna.Study
    anchor : dict[str, value]
        narrow_around()의 anchor와 동일 키/값. categorical은 choices 안의 값이어야.
    """
    study.enqueue_trial(dict(anchor))
    print(f"[enqueue] anchor 첫 trial로 강제 ({len(anchor)} HP)")
