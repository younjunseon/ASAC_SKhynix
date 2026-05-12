"""
PP+HP HPO — 전처리(PP)를 Optuna trial 축에 넣는 래퍼 + 전처리 결과 캐시.

`*_pphp.ipynb` 노트북 전용. 기존 `hpo.py` / `preprocess.py` / `models.py` / `postprocess.py`는
**수정하지 않고** 이들의 심볼만 import해서 쓴다. 즉 pp+hp는 순수 추가 모듈이다.

기존 hp-only 노트북과의 유일한 차이: PP 6축이 trial 축에 들어간다 → 같은 모델이라도 다른 전처리에서
학습한 OOF가 나옴 → 스태킹 base 다양성 ↑. HP 탐색 범위·anchor·CV·후처리는 hp-only와 동일하게 쓴다.

PP 탐색 6축 (`strategy_common.md` 보완 문서 `pp_hp_strategy.md §3` 확정):
    missing_threshold           [0.1, 0.2, ..., 0.9]
    corr_threshold              [0.80, 0.84, 0.88, 0.92, 0.96, 0.98]
    add_indicator               [True, False]   (False면 indicator_threshold는 0.05 고정 = no-op)
    indicator_threshold         [0.01, 0.05, 0.10, 0.15, 0.20, 0.25]   (add_indicator=True일 때만)
    spatial_max_dist            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    post_impute_corr_threshold  [0.96, 0.97, 0.98, 0.99]
나머지(corr_keep_by='std', post_impute_corr_keep_by='std', const_threshold, remove_duplicates,
imputation_method='spatial', outlier winsorize 0.0/0.99, Stage 0 EXCLUDE_COLS)는 PP_FIXED와 동일하게 고정.

공개 API
--------
- PP_SEARCH_CANDIDATES        : 6축 후보 dict (노트북에서 mutate해서 좁히기 가능)
- pp_search_space(trial)      : 6축 샘플 → preprocess.run(params=...)에 넘길 dict 반환
- pp_params_from_best(best)   : study.best_trial.params에서 pp_* 키를 꺼내 위 dict로 복원
- make_cached_preprocess(...) : (xs, ys, feat_cols, xs_dict) 고정 → prep(pp_dict) 함수 반환 (LRU 캐시)
- run_pp_hpo(...)             : 회귀 — run_hpo + PP 축. objective = train OOF unit RMSE
- run_pp_clf_hpo(...)         : Two-Stage Stage 1 분류 — run_clf_hpo + PP 축
- refit_pp_best(...)          : best PP로 preprocess 재실행 → hpo.refit_best 위임 (refit 결과 + 재구성 데이터 반환)
- refit_pp_clf_best(...)      : 분류 버전

사용 흐름 (회귀)
----------------
    from modules import pp_hpo, hpo
    prep = pp_hpo.make_cached_preprocess(xs, ys_input, feat_cols, xs_dict,
                                         position_mode='raw', use_die_xy=True, maxsize=PP_CACHE_SIZE)
    res = pp_hpo.run_pp_hpo(prep, xs_dict['train'], ys_input['train'], 'lgbm',
                            n_trials=N_TRIALS, n_folds=N_FOLDS, n_jobs=N_JOBS,
                            study_name=EXP_ID, storage=f'sqlite:///{DB_PATH}', resume_study=RESUME,
                            enqueue_trials=[ANCHOR_PPHP], timeout=TIMEOUT_SEC, user_attrs=study_meta,
                            xs_val_raw=xs_dict['validation'], ys_val_unit=ys_input['validation'],
                            xs_test_raw=xs_dict['test'],       ys_test_unit=ys_input['test'])
    out = pp_hpo.refit_pp_best(prep, xs_dict, ys_input, 'lgbm', res['best_params'],
                               n_folds=N_FOLDS, n_jobs=N_JOBS)
    hpo.save_artifacts(refit_result=out['refit_result'],
                       xs_train=out['xs_train'], xs_val=out['xs_val'], xs_test=out['xs_test'],
                       out_dir=OUT_DIR, exp_id=EXP_ID, feature_names=out['feat_cols'],
                       y_train_unit=ys_input['train'], y_val_unit=ys_input['validation'], y_test_unit=ys_input['test'],
                       postprocess_config=POSTPROCESS_CONFIG,
                       study_meta={**study_meta, 'effective_pp_params': out['effective_pp_params']})
"""
import io
import contextlib
from collections import OrderedDict

import numpy as np
import pandas as pd
import optuna

from utils.config import SEED, KEY_COL, TARGET_COL

from . import models as _models
from . import scaler as _scaler
from . import hpo as _hpo            # 헬퍼 재사용 (수정 안 함)
from . import preprocess as _preprocess

# hpo.py 내부 헬퍼 — 같은 패키지라 import 허용 (재사용)
from .hpo import (
    _make_unit_folds, _die_mask_from_units, _broadcast_y_to_die,
    _aggregate_die_to_unit, _fit_predict_fold, _inject_n_jobs,
)


# ============================================================
# PP 탐색 6축
# ============================================================

PP_SEARCH_CANDIDATES = {
    "missing_threshold":          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "corr_threshold":             [0.80, 0.84, 0.88, 0.92, 0.96, 0.98],
    "add_indicator":              [True, False],
    "indicator_threshold":        [0.01, 0.05, 0.10, 0.15, 0.20, 0.25],
    "spatial_max_dist":           [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "post_impute_corr_threshold": [0.96, 0.97, 0.98, 0.99],
}

# add_indicator=False일 때 indicator_threshold에 박는 값 (preprocess.run의 DEFAULT_PARAMS와 동일 — no-op이라 캐시 키 안정용)
_INDICATOR_THRESHOLD_OFF = 0.05


def pp_search_space(trial):
    """PP 6축을 Optuna trial로 샘플 → preprocess.run(params=...)에 넘길 dict 반환.

    trial param 이름은 모두 'pp_' prefix (모델 HP 키와 충돌 방지). add_indicator=False면
    indicator_threshold는 샘플하지 않고 _INDICATOR_THRESHOLD_OFF로 채움 (conditional sampling).
    """
    c = PP_SEARCH_CANDIDATES
    p = {
        "missing_threshold":          trial.suggest_categorical("pp_missing_threshold",          c["missing_threshold"]),
        "corr_threshold":             trial.suggest_categorical("pp_corr_threshold",             c["corr_threshold"]),
        "spatial_max_dist":           trial.suggest_categorical("pp_spatial_max_dist",           c["spatial_max_dist"]),
        "post_impute_corr_threshold": trial.suggest_categorical("pp_post_impute_corr_threshold", c["post_impute_corr_threshold"]),
    }
    add_ind = trial.suggest_categorical("pp_add_indicator", c["add_indicator"])
    p["add_indicator"] = add_ind
    if add_ind:
        p["indicator_threshold"] = trial.suggest_categorical("pp_indicator_threshold", c["indicator_threshold"])
    else:
        p["indicator_threshold"] = _INDICATOR_THRESHOLD_OFF
    return p


def pp_params_from_best(best_params):
    """study.best_trial.params(또는 best_params.json) → preprocess.run에 넘길 PP dict 복원.

    add_indicator=False였던 trial은 pp_indicator_threshold 키가 없으므로 _INDICATOR_THRESHOLD_OFF로 채움.
    """
    add_ind = bool(best_params["pp_add_indicator"])
    return {
        "missing_threshold":          best_params["pp_missing_threshold"],
        "corr_threshold":             best_params["pp_corr_threshold"],
        "spatial_max_dist":           best_params["pp_spatial_max_dist"],
        "post_impute_corr_threshold": best_params["pp_post_impute_corr_threshold"],
        "add_indicator":              add_ind,
        "indicator_threshold":        (best_params.get("pp_indicator_threshold", _INDICATOR_THRESHOLD_OFF)
                                       if add_ind else _INDICATOR_THRESHOLD_OFF),
    }


def _pp_key(pp_dict):
    """PP dict → 캐시용 hashable key (정렬된 (key, value) 튜플)."""
    return tuple(sorted(pp_dict.items()))


# ============================================================
# 전처리 결과 캐시
# ============================================================

def make_cached_preprocess(xs, ys_input, feat_cols, xs_dict,
                           position_mode="raw", use_die_xy=True,
                           maxsize=2, suppress_stdout=True):
    """(xs, ys, feat_cols, xs_dict) 고정 → prep(pp_dict)를 반환. prep은 LRU 캐시된다.

    prep(pp_dict) 반환값: (xs_train, xs_val, xs_test, feat_cols_clean, effective_pp_params)
      - preprocess.run(xs, ys, feat_cols, xs_dict, params=pp_dict) 실행 (Stage0 → cleaning → outlier winsorize)
      - 이어서 meta_features.add_meta_features(position_mode, use_die_xy)로 메타피처 추가
      - 행 순서는 xs_dict['train'/'validation'/'test']와 동일 (cleaning은 컬럼만 떨굼)

    Parameters
    ----------
    maxsize : int — 캐시 보관 개수. cleaned 3-split 한 벌 ≈ 1~1.5GB → 메모리 보고 조정. 기본 2.
    suppress_stdout : bool — preprocess.run / add_meta_features의 진행 출력 억제 (trial 수천 회면 스팸). effective_pp_params는 항상 반환되므로 재현성에는 영향 없음.

    노트북: `prep.counters` (= {'hit': n, 'miss': n}), `prep.cache` (OrderedDict) 로 캐시 상태 확인 가능.
    """
    # meta_features는 2_preprocessing/ — 노트북 setup.py가 sys.path에 등록해 둠
    from meta_features import add_meta_features

    cache = OrderedDict()                       # _pp_key → (xs_tr, xs_vl, xs_te, fcols, eff)
    counters = {"hit": 0, "miss": 0}

    def prep(pp_dict):
        key = _pp_key(pp_dict)
        if key in cache:
            cache.move_to_end(key)
            counters["hit"] += 1
            return cache[key]

        counters["miss"] += 1
        _ctx = contextlib.redirect_stdout(io.StringIO()) if suppress_stdout else contextlib.nullcontext()
        with _ctx:
            pp = _preprocess.run(xs, ys_input, feat_cols, xs_dict, params=dict(pp_dict))
            xs_tr, xs_vl, xs_te = pp["xs_train"], pp["xs_val"], pp["xs_test"]
            fcols = add_meta_features(
                xs_tr, xs_vl, xs_te, list(pp["feat_cols"]),
                position_mode=position_mode, use_die_xy=use_die_xy, verbose=False,
            )
        eff = pp["effective_params"]
        entry = (xs_tr, xs_vl, xs_te, fcols, eff)
        cache[key] = entry
        if len(cache) > maxsize:
            cache.popitem(last=False)           # LRU evict
        print(f"[pp cache miss #{counters['miss']}] n_features={len(fcols)} "
              f"(miss={counters['miss']}, hit={counters['hit']}, cached={len(cache)}) "
              f"key={dict(pp_dict)}")
        return entry

    prep.cache = cache
    prep.counters = counters
    return prep


# ============================================================
# 회귀 PP+HP HPO
# ============================================================

def run_pp_hpo(
    cached_prep, xs_train_raw, ys_train_unit, model_name,
    pp_space_fn=pp_search_space,
    n_trials=100, n_folds=5,
    y_positive_only=False,
    target_transform_fn=None, target_inverse_fn=None,
    study_name=None, storage=None, resume_study=False,
    seed=SEED, direction="minimize",
    show_progress_bar=True,
    user_attrs=None,
    space_variant="default",
    sampler=None, pruner=None,
    enqueue_trials=None,
    timeout=None,
    n_jobs=None,
    xs_val_raw=None, ys_val_unit=None,
    xs_test_raw=None, ys_test_unit=None,
):
    """PP 6축 + 모델 HP를 함께 Optuna로 탐색. objective = train OOF unit RMSE (run_hpo와 동일 정의).

    hp-only `hpo.run_hpo`와의 차이: (1) 매 trial PP를 샘플해 cached_prep으로 전처리, (2) xs_*는 전처리 후 X.
    fold 분할(unit-level KFold, seed 고정)·tweedie loss시 transform OFF·enet 스케일링·NaN trial pruning·
    val/test RMSE user_attr 기록은 run_hpo와 동일하게 수행.

    Parameters
    ----------
    cached_prep : callable — make_cached_preprocess(...)가 반환한 prep 함수
    xs_train_raw : DataFrame — 전처리 전 xs_train (= xs_dict['train']). KEY_COL/DIE_KEY_COL만 쓰임 (fold mask·집계). feature는 안 씀
    ys_train_unit : DataFrame — unit-level [KEY_COL, TARGET_COL] (원본 스케일)
    model_name : 'lgbm' / 'xgb' / 'catboost' / 'et' / 'enet' / 'zitboost'
    y_positive_only : True면 fold 학습 데이터에서 y==0 die 제외 (Two-Stage Stage 2)
    target_transform_fn / target_inverse_fn : 쌍으로. tweedie 계열 objective면 자동 OFF
    xs_val_raw / ys_val_unit, xs_test_raw / ys_test_unit : (옵션, 짝) — 주면 매 trial val/test RMSE 기록
    그 외 인자: hpo.run_hpo와 동일 의미

    Returns
    -------
    dict {'study', 'best_params', 'model_name', 'best_value'}
      - best_params는 study.best_trial.params 그대로 (pp_* 키 + 모델 HP 키 혼재). refit_pp_best에 그대로 넘김.
    """
    if (target_transform_fn is None) != (target_inverse_fn is None):
        raise ValueError("target_transform_fn / target_inverse_fn은 쌍으로 제공")
    if (xs_val_raw is None) != (ys_val_unit is None):
        raise ValueError("xs_val_raw / ys_val_unit 은 쌍으로 제공")
    if (xs_test_raw is None) != (ys_test_unit is None):
        raise ValueError("xs_test_raw / ys_test_unit 은 쌍으로 제공")

    space_fn = _models.get_search_space(model_name, variant=space_variant)

    # PP와 무관한 것들 — 한 번만 계산 (KEY_COL/DIE_KEY_COL은 전처리해도 안 바뀜)
    unit_ids = ys_train_unit[KEY_COL].unique()
    folds = _make_unit_folds(unit_ids, n_folds, seed)
    fold_masks = [
        (_die_mask_from_units(xs_train_raw, set(tr)),
         _die_mask_from_units(xs_train_raw, set(vl)))
        for tr, vl in folds
    ]
    y_die_orig  = _broadcast_y_to_die(xs_train_raw, ys_train_unit)   # die-level 정답 (원본 스케일)
    y_true_unit = ys_train_unit.set_index(KEY_COL)[TARGET_COL]
    n_tr = len(xs_train_raw)
    n_vl = len(xs_val_raw)  if xs_val_raw  is not None else 0
    n_te = len(xs_test_raw) if xs_test_raw is not None else 0
    y_val_true_unit  = ys_val_unit.set_index(KEY_COL)[TARGET_COL]  if ys_val_unit  is not None else None
    y_test_true_unit = ys_test_unit.set_index(KEY_COL)[TARGET_COL] if ys_test_unit is not None else None

    def _is_tweedie_hp(hp):
        s = str(hp.get("objective") or hp.get("loss_function") or "")
        return s.startswith("tweedie") or s.startswith("reg:tweedie") or s.lower().startswith("tweedie")

    def objective(trial):
        # 1) PP 샘플 → 전처리(캐시)
        pp_params = pp_space_fn(trial)
        xs_tr, xs_vl, xs_te, fcols, eff_pp = cached_prep(pp_params)
        trial.set_user_attr("pp_params", {k: v for k, v in pp_params.items()})
        trial.set_user_attr("n_features", len(fcols))

        X_full      = xs_tr[fcols].values
        X_val_full  = xs_vl[fcols].values if (xs_val_raw  is not None and xs_vl is not None) else None
        X_test_full = xs_te[fcols].values if (xs_test_raw is not None and xs_te is not None) else None

        # 2) 모델 HP 샘플
        hp = space_fn(trial)
        hp = _inject_n_jobs(model_name, hp, n_jobs)

        # tweedie 계열이면 target_transform OFF (이중 변환 방지 — run_hpo와 동일)
        if _is_tweedie_hp(hp):
            eff_transform_fn = eff_inverse_fn = None
        else:
            eff_transform_fn, eff_inverse_fn = target_transform_fn, target_inverse_fn
        trial.set_user_attr("target_transform_active", eff_transform_fn is not None)

        y_die_fit_local = eff_transform_fn(y_die_orig) if eff_transform_fn else y_die_orig

        def _eval_split_rmse_local(xs_split, die_pred_accum, y_true_unit_split):
            pred = eff_inverse_fn(die_pred_accum) if eff_inverse_fn else die_pred_accum
            unit = _aggregate_die_to_unit(xs_split, pred)
            aligned = unit.set_index(KEY_COL)["pred"].loc[y_true_unit_split.index]
            return float(np.sqrt(np.mean((aligned.values - y_true_unit_split.values) ** 2)))

        oof = np.full(n_tr, np.nan)
        val_pred_accum  = np.zeros(n_vl) if X_val_full  is not None else None
        test_pred_accum = np.zeros(n_te) if X_test_full is not None else None

        for tr_mask, vl_mask in fold_masks:
            fit_mask = (tr_mask & (y_die_orig > 0)) if y_positive_only else tr_mask
            X_tr, y_tr = X_full[fit_mask], y_die_fit_local[fit_mask]
            X_vl       = X_full[vl_mask]

            if _scaler.needs_scaling(model_name):
                med = np.median(X_tr, axis=0)
                iqr = np.maximum(np.quantile(X_tr, 0.75, axis=0) - np.quantile(X_tr, 0.25, axis=0), 1e-8)
                X_tr   = (X_tr - med) / iqr
                X_vl   = (X_vl - med) / iqr
                X_eval_v = (X_val_full  - med) / iqr if X_val_full  is not None else None
                X_eval_t = (X_test_full - med) / iqr if X_test_full is not None else None
            else:
                X_eval_v, X_eval_t = X_val_full, X_test_full

            res = _fit_predict_fold(model_name, hp, X_tr, y_tr, X_vl)
            oof[vl_mask] = res["pred"]
            if X_eval_v is not None:
                val_pred_accum  += res["model"].predict(X_eval_v)  / n_folds
            if X_eval_t is not None:
                test_pred_accum += res["model"].predict(X_eval_t) / n_folds

        if np.isnan(oof).any():
            n_nan = int(np.isnan(oof).sum())
            print(f"[pp trial {trial.number}] 예측 NaN {n_nan}개 (모델 발산 추정: "
                  f"objective={hp.get('objective')}, tweedie_power={hp.get('tweedie_variance_power')}) → 이 trial 폐기")
            raise optuna.TrialPruned()

        train_rmse = _eval_split_rmse_local(xs_train_raw, oof, y_true_unit)
        trial.set_user_attr("train_rmse", train_rmse)
        if val_pred_accum is not None:
            trial.set_user_attr("val_rmse", _eval_split_rmse_local(xs_val_raw, val_pred_accum, y_val_true_unit))
        if test_pred_accum is not None:
            trial.set_user_attr("test_rmse", _eval_split_rmse_local(xs_test_raw, test_pred_accum, y_test_true_unit))
        return train_rmse

    study = optuna.create_study(
        direction=direction,
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
    study.set_user_attr("pp_search_candidates", {k: list(v) for k, v in PP_SEARCH_CANDIDATES.items()})

    if enqueue_trials and len(study.trials) == 0:
        for anchor in enqueue_trials:
            study.enqueue_trial(dict(anchor))
        print(f"[enqueue] {len(enqueue_trials)} anchor trial(s) 강제 (pp_* + HP 키 포함되어 있어야 함)")
    elif enqueue_trials:
        print(f"[enqueue skip] 기존 trial {len(study.trials)}개 — resume 모드")

    if _scaler.needs_scaling(model_name):
        print(f"[scaler] {model_name} → fold-local RobustScaler 적용 (매 fold train 기준 fit)")

    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=show_progress_bar)

    return {
        "study":       study,
        "best_params": dict(study.best_trial.params),
        "model_name":  model_name,
        "best_value":  study.best_value,
    }


# ============================================================
# Two-Stage Stage 1 (분류) PP+HP HPO
# ============================================================

def run_pp_clf_hpo(
    cached_prep, xs_train_raw, ys_train_unit, model_name,
    pp_space_fn=pp_search_space,
    n_trials=100, n_folds=5,
    study_name=None, storage=None, resume_study=False,
    seed=SEED,
    show_progress_bar=True,
    user_attrs=None,
    sampler=None, pruner=None,
    enqueue_trials=None,
    timeout=None,
    n_jobs=None,
    xs_val_raw=None, ys_val_unit=None,
    xs_test_raw=None, ys_test_unit=None,
):
    """die-level binary CLF (y>0 여부) + PP 6축 HPO. objective = unit RMSE(unit평균확률 × E[Y|Y>0]).

    run_clf_hpo와 동일 정의 + 매 trial PP 샘플·전처리. Returns: {'study','best_params','model_name','best_value','y_pos_const'}.
    """
    if (xs_val_raw is None) != (ys_val_unit is None):
        raise ValueError("xs_val_raw / ys_val_unit 은 쌍으로 제공")
    if (xs_test_raw is None) != (ys_test_unit is None):
        raise ValueError("xs_test_raw / ys_test_unit 은 쌍으로 제공")

    space_fn = _models.get_clf_search_space(model_name)

    unit_ids = ys_train_unit[KEY_COL].unique()
    folds = _make_unit_folds(unit_ids, n_folds, seed)
    fold_masks = [
        (_die_mask_from_units(xs_train_raw, set(tr)),
         _die_mask_from_units(xs_train_raw, set(vl)))
        for tr, vl in folds
    ]
    y_die_orig  = _broadcast_y_to_die(xs_train_raw, ys_train_unit)
    y_die_bin   = (y_die_orig > 0).astype(int)
    y_true_unit = ys_train_unit.set_index(KEY_COL)[TARGET_COL]
    y_pos_const = float(y_true_unit[y_true_unit > 0].mean())
    n_tr = len(xs_train_raw)
    n_vl = len(xs_val_raw)  if xs_val_raw  is not None else 0
    n_te = len(xs_test_raw) if xs_test_raw is not None else 0
    y_val_true_unit  = ys_val_unit.set_index(KEY_COL)[TARGET_COL]  if ys_val_unit  is not None else None
    y_test_true_unit = ys_test_unit.set_index(KEY_COL)[TARGET_COL] if ys_test_unit is not None else None

    def _eval_split_rmse(xs_split, die_proba, y_true_unit_split):
        df = pd.DataFrame({KEY_COL: xs_split[KEY_COL].values, "p": die_proba})
        unit_proba = df.groupby(KEY_COL, sort=False)["p"].mean()
        unit_pred  = unit_proba * y_pos_const
        aligned    = unit_pred.loc[y_true_unit_split.index]
        return float(np.sqrt(np.mean((aligned.values - y_true_unit_split.values) ** 2)))

    def objective(trial):
        pp_params = pp_space_fn(trial)
        xs_tr, xs_vl, xs_te, fcols, eff_pp = cached_prep(pp_params)
        trial.set_user_attr("pp_params", {k: v for k, v in pp_params.items()})
        trial.set_user_attr("n_features", len(fcols))

        X_train = xs_tr[fcols].values
        X_val   = xs_vl[fcols].values if (xs_val_raw  is not None and xs_vl is not None) else None
        X_test  = xs_te[fcols].values if (xs_test_raw is not None and xs_te is not None) else None

        params = space_fn(trial)
        params = _inject_n_jobs(model_name, params, n_jobs)

        oof_proba = np.full(n_tr, np.nan)
        val_proba_accum  = np.zeros(n_vl) if X_val  is not None else None
        test_proba_accum = np.zeros(n_te) if X_test is not None else None

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
            n_nan = int(np.isnan(oof_proba).sum())
            print(f"[pp clf trial {trial.number}] 예측 확률 NaN {n_nan}개 → 이 trial 폐기")
            raise optuna.TrialPruned()

        train_rmse = _eval_split_rmse(xs_train_raw, oof_proba, y_true_unit)
        trial.set_user_attr("train_rmse", train_rmse)
        if val_proba_accum is not None:
            trial.set_user_attr("val_rmse", _eval_split_rmse(xs_val_raw, val_proba_accum, y_val_true_unit))
        if test_proba_accum is not None:
            trial.set_user_attr("test_rmse", _eval_split_rmse(xs_test_raw, test_proba_accum, y_test_true_unit))
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
    study.set_user_attr("clf_objective_recipe", "unit_RMSE(unit_mean(die_proba) * y_pos_const)")
    study.set_user_attr("pp_search_candidates", {k: list(v) for k, v in PP_SEARCH_CANDIDATES.items()})

    if enqueue_trials and len(study.trials) == 0:
        for anchor in enqueue_trials:
            study.enqueue_trial(dict(anchor))
        print(f"[enqueue] {len(enqueue_trials)} anchor trial(s) 강제")
    elif enqueue_trials:
        print(f"[enqueue skip] 기존 trial {len(study.trials)}개 — resume 모드")

    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=show_progress_bar)

    return {
        "study":       study,
        "best_params": dict(study.best_trial.params),
        "model_name":  model_name,
        "best_value":  study.best_value,
        "y_pos_const": y_pos_const,
    }


# ============================================================
# Refit (best PP → 전처리 재실행 → hpo.refit_best/refit_clf_best 위임)
# ============================================================

def _split_best_params(best_params):
    """best_params(study.best_trial.params)를 (pp_dict, hp_dict)로 분리.

    pp_dict : preprocess.run에 넘길 형태 (pp_params_from_best)
    hp_dict : 모델 HP — 'pp_' 안 붙은 키 전부 (hpo._hp_from_best / _clf_hp_with_defaults가 처리할 형식)
    """
    pp_dict = pp_params_from_best(best_params)
    hp_dict = {k: v for k, v in best_params.items() if not str(k).startswith("pp_")}
    return pp_dict, hp_dict


def refit_pp_best(
    cached_prep, xs_dict, ys_input, model_name, best_params,
    n_folds=5, seed=SEED, n_jobs=None,
    y_positive_only=False,
    target_transform_fn=None, target_inverse_fn=None,
):
    """best PP로 preprocess 재실행 → 그 데이터로 hpo.refit_best 호출.

    Parameters
    ----------
    cached_prep : make_cached_preprocess(...) 반환 함수 (best PP 조합은 보통 캐시에 이미 있음)
    xs_dict : {'train': df, 'validation': df, 'test': df} — 전처리 전 원본 split
    ys_input : {'train': df, 'validation': df, 'test': df} — unit-level (학습 입력용, 원본 또는 CLIP된 것)
    best_params : run_pp_hpo가 돌려준 study.best_trial.params (pp_* + HP 혼재)
    y_positive_only / target_transform_fn / target_inverse_fn : hpo.refit_best에 그대로 전달

    Returns
    -------
    dict {
        'refit_result'        : hpo.refit_best 반환 (oof/val/test die·unit, fold_models, ...)
        'xs_train'/'xs_val'/'xs_test' : 전처리+메타 적용된 DataFrame (save_artifacts에 그대로 넘김)
        'feat_cols'           : 전처리 후 feature 이름 리스트
        'effective_pp_params' : 실제 적용된 PP 파라미터 (재현성 — study_meta에 넣어 저장)
        'pp_params'           : best PP dict (preprocess.run에 넘긴 형태)
    }
    """
    pp_dict, hp_dict = _split_best_params(best_params)
    xs_tr, xs_vl, xs_te, fcols, eff_pp = cached_prep(pp_dict)

    refit_result = _hpo.refit_best(
        xs_train=xs_tr, xs_val=xs_vl, xs_test=xs_te,
        ys_train_unit=ys_input["train"], feat_cols=fcols,
        model_name=model_name, best_params=hp_dict,
        n_folds=n_folds, seed=seed,
        y_positive_only=y_positive_only,
        target_transform_fn=target_transform_fn, target_inverse_fn=target_inverse_fn,
        n_jobs=n_jobs,
    )
    return {
        "refit_result":        refit_result,
        "xs_train":            xs_tr,
        "xs_val":              xs_vl,
        "xs_test":             xs_te,
        "feat_cols":           fcols,
        "effective_pp_params": eff_pp,
        "pp_params":           pp_dict,
    }


def refit_pp_clf_best(
    cached_prep, xs_dict, ys_input, model_name, best_params,
    n_folds=5, seed=SEED, n_jobs=None,
):
    """clf 버전 — best PP로 preprocess 재실행 → hpo.refit_clf_best 호출.

    Returns
    -------
    dict {'refit_result' (hpo.refit_clf_best 반환), 'xs_train'/'xs_val'/'xs_test', 'feat_cols',
          'effective_pp_params', 'pp_params'}
    """
    pp_dict, hp_dict = _split_best_params(best_params)
    xs_tr, xs_vl, xs_te, fcols, eff_pp = cached_prep(pp_dict)

    refit_result = _hpo.refit_clf_best(
        xs_train=xs_tr, xs_val=xs_vl, xs_test=xs_te,
        ys_train_unit=ys_input["train"], feat_cols=fcols,
        model_name=model_name, best_params=hp_dict,
        n_folds=n_folds, seed=seed, n_jobs=n_jobs,
    )
    return {
        "refit_result":        refit_result,
        "xs_train":            xs_tr,
        "xs_val":              xs_vl,
        "xs_test":             xs_te,
        "feat_cols":           fcols,
        "effective_pp_params": eff_pp,
        "pp_params":           pp_dict,
    }
