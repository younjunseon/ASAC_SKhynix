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


def _scale_fold_if_needed(model_name, X_tr, X_vl):
    """enet이면 fold-local RobustScaler, 아니면 pass-through."""
    if not _scaler.needs_scaling(model_name):
        return X_tr, X_vl
    med = np.median(X_tr, axis=0)
    q75 = np.quantile(X_tr, 0.75, axis=0)
    q25 = np.quantile(X_tr, 0.25, axis=0)
    iqr = np.maximum(q75 - q25, 1e-8)
    return (X_tr - med) / iqr, (X_vl - med) / iqr


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

    Returns
    -------
    dict  {'study', 'best_params', 'model_name', 'best_value'}
    """
    if (target_transform_fn is None) != (target_inverse_fn is None):
        raise ValueError("target_transform_fn / target_inverse_fn은 쌍으로 제공")
    space_fn = _models.get_search_space(model_name)

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
    y_die_fit  = target_transform_fn(y_die_orig) if target_transform_fn else y_die_orig
    X_full = _build_X(xs_train, feat_cols, extra_feature_train)
    y_true_unit = ys_train_unit.set_index(KEY_COL)[TARGET_COL]   # 원본 스케일

    # multiplier 배열 검증
    if multiplier_train is not None:
        multiplier_train = np.asarray(multiplier_train, dtype=float).ravel()
        if len(multiplier_train) != len(xs_train):
            raise ValueError(
                f"multiplier_train 길이 {len(multiplier_train)} "
                f"!= xs_train 길이 {len(xs_train)}"
            )

    def objective(trial):
        hp = space_fn(trial)
        oof = np.full(len(xs_train), np.nan)
        for tr_mask, vl_mask in fold_masks:
            # ── 학습 데이터 필터링 (정석 Two-Stage: Y>0 만) ──
            if y_positive_only:
                fit_mask = tr_mask & (y_die_orig > 0)
            else:
                fit_mask = tr_mask
            X_tr, y_tr = X_full[fit_mask], y_die_fit[fit_mask]
            X_vl       = X_full[vl_mask]
            X_tr, X_vl = _scale_fold_if_needed(model_name, X_tr, X_vl)
            res = _fit_predict_fold(model_name, hp, X_tr, y_tr, X_vl)
            oof[vl_mask] = res["pred"]

        if np.isnan(oof).any():
            raise RuntimeError("OOF has NaN — fold coverage bug")

        # ── 역변환 → 원본 스케일에서 (1-π) 곱셈 적용 → 집계 + RMSE ──
        oof_orig = target_inverse_fn(oof) if target_inverse_fn else oof
        if multiplier_train is not None:
            oof_orig = oof_orig * multiplier_train   # die-level element-wise
        unit_pred = _aggregate_die_to_unit(xs_train, oof_orig)
        aligned = unit_pred.set_index(KEY_COL)["pred"].loc[y_true_unit.index]
        rmse = float(np.sqrt(np.mean((aligned.values - y_true_unit.values) ** 2)))
        return rmse

    # resume_study=False 기본: 같은 study_name/storage 조합이 이미 있으면 에러를
    # 명시적으로 내어 trial 누적으로 best 값이 오염되는 것을 막는다.
    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(storage and resume_study),
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    if user_attrs:
        for k, v in user_attrs.items():
            study.set_user_attr(k, v)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)

    return {
        "study":            study,
        "best_params":      dict(study.best_trial.params),
        "model_name":       model_name,
        "best_value":       study.best_value,
    }


# ═════════════════════════════════════════════════════════════
# Best Trial Refit (K-fold)
# ═════════════════════════════════════════════════════════════

def _hp_from_best(best_params, model_name):
    """Optuna best_params dict → MODEL_REGISTRY에 전달할 kwargs.

    (search space 내부의 objective/loss_function 처리 재현)
    """
    hp = dict(best_params)
    # LGBM
    if model_name == "lgbm":
        obj = hp.pop("objective", None)
        if obj == "poisson":
            hp["objective"] = "poisson"
        elif obj and obj.startswith("tweedie"):
            hp["objective"] = "tweedie"
            hp["tweedie_variance_power"] = float(obj.split("_")[1])
    # XGB
    elif model_name == "xgb":
        obj = hp.pop("objective", None)
        if obj == "reg:squarederror":
            hp["objective"] = "reg:squarederror"
        elif obj and obj.startswith("reg:tweedie"):
            hp["objective"] = "reg:tweedie"
            hp["tweedie_variance_power"] = float(obj.split("_")[1])
    # CatBoost
    elif model_name == "catboost":
        loss = hp.pop("loss_function", None)
        if loss == "RMSE":
            hp["loss_function"] = "RMSE"
        elif loss and loss.startswith("Tweedie"):
            power = float(loss.split("_")[1])
            hp["loss_function"] = f"Tweedie:variance_power={power}"

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
        hp.setdefault("tol", 1e-4)

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
        'best_params_resolved': dict,
    }
    """
    if (target_transform_fn is None) != (target_inverse_fn is None):
        raise ValueError("target_transform_fn / target_inverse_fn은 쌍으로 제공")
    # REUSE 모드: best_params_resolved(JSON)를 그대로 받은 경우 재변환 스킵
    hp = dict(best_params) if already_resolved else _hp_from_best(best_params, model_name)

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
        else:
            X_val_tr, X_test_tr = X_val_full, X_test_full

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
        "best_params_resolved": hp,
        "model_name": model_name,
    }


# ═════════════════════════════════════════════════════════════
# Artifact 저장 (pkl + CSV + JSON)
# ═════════════════════════════════════════════════════════════

from utils.config import DIE_KEY_COL as _DIE_KEY_COL


def _die_csv(xs_split, pred, pi=None, mu=None):
    """die-level 예측을 KEY_COL + DIE_KEY_COL 과 함께 DataFrame으로."""
    out = pd.DataFrame({
        KEY_COL:      xs_split[KEY_COL].values,
        _DIE_KEY_COL: xs_split[_DIE_KEY_COL].values,
        "pred":       pred,
    })
    if pi is not None:
        out["pi"] = pi
        out["one_minus_pi"] = 1.0 - pi   # 경로 B에서 바로 쓰기 쉬우라고 파생
    if mu is not None:
        out["mu"] = mu
    return out


def save_artifacts(
    refit_result, xs_train, xs_val, xs_test,
    out_dir, exp_id=None,
    feature_names=None,
    extra_feature_name=None,
    y_train_unit=None,
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
    postprocess_config : dict or None
        None 이면 기존 mean 집계만 저장 (backward-compat).
        dict 이면 postprocess.tune_and_apply 에 kwargs 로 전달하여
        unit CSV 를 튜닝된 값으로 대체 저장.
        예: {'agg_methods': (...), 'pi_threshold_range': (...), ...}
    study_meta : dict or None
        study.user_attrs 같은 재현성 메타. best_params.json 에 그대로 저장.

    생성물
    ------
    - {out_dir}/fold_models.pkl      : {'fold_models', 'feature_names', ...}
    - {out_dir}/best_params.json     : model_name + resolved HP + feature_names + study_meta
    - {out_dir}/oof_die.csv          : train OOF die-level (+ pi, mu if ZIT)
    - {out_dir}/val_die.csv          : val die-level
    - {out_dir}/test_die.csv         : test die-level
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
        "feature_names":       list(feature_names) if feature_names is not None else None,
        "extra_feature_name":  extra_feature_name,
        "model_name":          refit_result["model_name"],
        "n_folds":             len(refit_result["fold_models"]),
    }
    with open(os.path.join(out_dir, "fold_models.pkl"), "wb") as f:
        pickle.dump(pkl_payload, f)

    # 2) best_params (JSON) + study 메타 + fold 재현성 정보
    meta = {
        "exp_id":                exp_id,
        "model_name":            refit_result["model_name"],
        "best_params_resolved":  refit_result["best_params_resolved"],
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

    # 3) die-level CSV (tune 이전 raw die 예측 그대로)
    _die_csv(xs_train, refit_result["oof_pred_die"],
             refit_result.get("oof_pi"), refit_result.get("oof_mu")
             ).to_csv(os.path.join(out_dir, "oof_die.csv"), index=False)
    _die_csv(xs_val,   refit_result["val_pred_die"],
             refit_result.get("val_pi"), refit_result.get("val_mu")
             ).to_csv(os.path.join(out_dir, "val_die.csv"), index=False)
    _die_csv(xs_test,  refit_result["test_pred_die"],
             refit_result.get("test_pi"), refit_result.get("test_mu")
             ).to_csv(os.path.join(out_dir, "test_die.csv"), index=False)

    # 4) unit-level CSV — postprocess_config 여부로 분기
    if postprocess_config is not None and y_train_unit is not None:
        from . import postprocess as _pp
        pp_res = _pp.tune_and_apply(
            xs_train, xs_val, xs_test,
            die_pred_train=refit_result["oof_pred_die"],
            die_pred_val=refit_result["val_pred_die"],
            die_pred_test=refit_result["test_pred_die"],
            y_train_unit=y_train_unit,
            die_pi_train=refit_result.get("oof_pi"),
            die_pi_val=refit_result.get("val_pi"),
            die_pi_test=refit_result.get("test_pi"),
            **postprocess_config,
        )
        pp_res["final_train_unit"].to_csv(
            os.path.join(out_dir, "oof_unit.csv"), index=False)
        pp_res["final_val_unit"].to_csv(
            os.path.join(out_dir, "val_unit.csv"), index=False)
        pp_res["final_test_unit"].to_csv(
            os.path.join(out_dir, "test_unit.csv"), index=False)
        # best tuning 결과도 메타에 기록
        meta["postprocess"] = {
            "best_agg":          pp_res["best_agg"],
            "pos_weights":       (pp_res["pos_weights"].tolist()
                                  if pp_res["pos_weights"] is not None else None),
            "best_pi_threshold": pp_res["best_pi_threshold"],
            "best_zero_clip":    pp_res["best_zero_clip"],
            "train_rmse":        pp_res["train_rmse"],
            "agg_rmses":         {k: float(v) for k, v in pp_res["agg_rmses"].items()},
            "config":            postprocess_config,
        }
    else:
        # 기존 동작: refit 단계의 mean 집계 그대로
        refit_result["oof_pred_unit"].to_csv(
            os.path.join(out_dir, "oof_unit.csv"), index=False)
        refit_result["val_pred_unit"].to_csv(
            os.path.join(out_dir, "val_unit.csv"), index=False)
        refit_result["test_pred_unit"].to_csv(
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
