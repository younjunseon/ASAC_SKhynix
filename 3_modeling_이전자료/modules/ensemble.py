"""
Ensemble — Blending / Voting / Stacking 모듈

여러 base 모델의 예측을 결합하여 RMSE를 낮춘다.
모든 동작은 ENSEMBLE_CONFIG 스위치로 제어된다.

설계 원칙:
1. 가중치/meta-learner는 **train OOF 예측**으로만 학습 (val double-dip 방지)
   → 이를 위해 rerun_best_trial을 mode='kfold'로 강제
2. Equal weight는 SLSQP 옆에 항상 같이 출력 (가중치 튜닝 효과 검증)
3. Ridge 기본 meta-learner (안정성), LightGBM은 옵션
4. 스위치로 blend/stacking/both/none 자유 토글
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold

from utils.config import SEED, TARGET_COL, KEY_COL
from utils.evaluate import rmse, postprocess
from .e2e_hpo import run_e2e_optimization, rerun_best_trial


# ═════════════════════════════════════════════════════════════
# 기본 ENSEMBLE_CONFIG (전부 토글 가능)
# ═════════════════════════════════════════════════════════════
DEFAULT_ENSEMBLE_CONFIG = dict(
    # ── 마스터 스위치 ──
    enabled=True,                     # False → 앙상블 스킵, 단일 best만 반환

    # ── 방법 선택 ──
    method="both",                    # 'blend' | 'stacking' | 'both' | 'none'

    # ── 블렌딩 옵션 ──
    blend_optimizer="slsqp",          # 'slsqp' | 'optuna' | 'equal'
    blend_include_equal=True,         # SLSQP/Optuna 써도 equal은 항상 같이 계산
    blend_optuna_trials=200,          # blend_optimizer='optuna' 일 때만 사용

    # ── 스태킹 옵션 ──
    stacking_meta="ridge",            # 'ridge' | 'lgbm' | 'auto' (val RMSE 낮은 쪽)
    stacking_include_clf_proba=True,  # meta feature에 clf proba 추가
    stacking_ridge_alpha=None,        # None → RidgeCV 자동, 숫자 → 고정
    stacking_nested_cv=True,          # True → meta learner를 k-fold로 학습 (진짜 meta-OOF)
    stacking_meta_folds=5,            # stacking_nested_cv=True 일 때 fold 수
    stacking_lgbm_n_estimators=300,   # LGBM meta 고정 n_estimators (early stop 제거)

    # ── base 모델 rerun 모드 ──
    force_kfold_mode=True,            # True → rerun을 kfold로 강제 (OOF 확보)
    n_folds_rerun=5,                  # force_kfold_mode=True 일 때 fold 수

    # ── 오버피팅 방어 ──
    overfit_gap_warn=0.0005,          # |val-train| RMSE gap 경고 임계값
    conservative_selection=False,     # True → gap이 overfit_gap_warn 초과하는 방법은 best 선정에서 제외

    # ── 후처리 ──
    clip_negative=True,               # 최종 예측 np.clip(pred, 0, None)

    # ── 로깅 ──
    verbose=True,
)


def _merge_ens_config(user_config):
    cfg = DEFAULT_ENSEMBLE_CONFIG.copy()
    cfg.update(user_config or {})

    if cfg["method"] not in ("blend", "stacking", "both", "none"):
        raise ValueError(f"method must be one of blend/stacking/both/none, got '{cfg['method']}'")
    if cfg["blend_optimizer"] not in ("slsqp", "optuna", "equal"):
        raise ValueError(f"blend_optimizer must be slsqp/optuna/equal, got '{cfg['blend_optimizer']}'")
    if cfg["stacking_meta"] not in ("ridge", "lgbm", "auto"):
        raise ValueError(f"stacking_meta must be ridge/lgbm/auto, got '{cfg['stacking_meta']}'")

    if not cfg["force_kfold_mode"]:
        print("[ensemble] INFO: force_kfold_mode=False. "
              "base 후보가 single 모드로 돌면 oof_pred가 없어 "
              "실제 앙상블 실행 시점(collect_base_predictions)에서 차단됩니다.")
    return cfg


# ═════════════════════════════════════════════════════════════
# 1) Base 모델 예측 수집
# ═════════════════════════════════════════════════════════════
def collect_base_predictions(
    candidates,
    pos_data,
    feat_cols,
    ensemble_config=None,
    pipeline_config=None,
    n_trials=100,
    n_folds=3,
    clf_early_stop=50,
    reg_early_stop=50,
    label_col="label_bin",
    imbalance_method="scale_pos_weight",
    agg_funcs=None,
    top_k_range=(50, 500),
    top_k_fixed=200,
    clf_fixed=None,
    reg_fixed=None,
    unit_data_input=None,
    unit_feat_cols_input=None,
    precomputed=None,
    rerun_params=None,
):
    """
    candidates 리스트를 순회하며 각 (clf_model, reg_model) 조합의
    HPO → rerun(kfold) → OOF/val/test 예측을 수집.

    Parameters
    ----------
    candidates : list of tuple
        [(clf_model, reg_model), ...] 예: [("lgbm","lgbm"), ("xgb","catboost")]
    pos_data, feat_cols : e2e_hpo와 동일
    ensemble_config : dict
        ENSEMBLE_CONFIG. force_kfold_mode=True면 rerun을 kfold로 강제.
    precomputed : dict, optional
        이미 학습한 (clf_model, reg_model)의 결과를 재사용.
        형식: {(clf_model, reg_model): {"hpo_result": ..., "rerun_result": ...}}
        - hpo_result : run_e2e_optimization 반환값 ("best_params" 키 필요)
        - rerun_result : rerun_best_trial 반환값 ("oof_pred" 필요, kfold 모드)
        해당 key에 매칭되는 후보는 HPO/rerun을 스킵하고 주입된 결과 재사용.
        oof_pred가 없으면(single 모드) 경고 후 새로 학습.
    rerun_params : dict, optional
        base 후보의 rerun_best_trial 호출에 사용할 파라미터.
        키: mode, n_folds, clf_early_stop, reg_early_stop, es_holdout.
        None이면 기존 동작(ensemble_config의 force_kfold_mode/n_folds_rerun +
        clf/reg_early_stop*2 하드코딩)으로 폴백 — backward compat.
        단, ensemble_config['force_kfold_mode']=True면 rerun_params['mode']가
        'single'이어도 'kfold'로 덮어쓴다 (앙상블은 OOF 필요).
    나머지 : run_e2e_optimization / rerun_best_trial에 그대로 전달

    Returns
    -------
    base_results : dict
        {name: {
            "oof_pred":  train OOF 예측 (n_train_units,),
            "val_pred":  val 예측       (n_val_units,),
            "test_pred": test 예측      (n_test_units,),
            "val_rmse":  val RMSE,
            "clf_model": str,
            "reg_model": str,
            "best_params": dict,
            "clf_proba_train": None or (n_train_units,),   # clf proba (meta feature용)
            "clf_proba_val":   None or (n_val_units,),
            "clf_proba_test":  None or (n_test_units,),
            "y_train":   (n_train_units,),   # 첫 후보에서만 저장, 이후 재사용
            "y_val":     (n_val_units,),
        }}
    shared : dict
        {"y_train": ..., "y_val": ..., "key_train": ..., "key_val": ..., "key_test": ...}
        모든 후보가 공유하는 unit 키/라벨 (정렬 검증용)
    """
    ens_cfg = _merge_ens_config(ensemble_config)
    base_results = {}
    shared = None

    # ── Rerun 설정 소스 결정 ──
    # rerun_params가 주어지면 거기서 값을 가져오고, 없으면 기존(구버전) 방식.
    # 기존 방식: ens_cfg['force_kfold_mode']/'n_folds_rerun' + early_stop*2 하드코딩.
    if rerun_params is not None:
        rp_mode = rerun_params.get("mode", "kfold")
        rp_n_folds = rerun_params.get("n_folds", 5)
        rp_clf_es = rerun_params.get("clf_early_stop", clf_early_stop * 2)
        rp_reg_es = rerun_params.get("reg_early_stop", reg_early_stop * 2)
        rp_es_holdout = rerun_params.get("es_holdout", 0.1)
    else:
        rp_mode = "kfold" if ens_cfg["force_kfold_mode"] else "single"
        rp_n_folds = ens_cfg["n_folds_rerun"]
        rp_clf_es = clf_early_stop * 2
        rp_reg_es = reg_early_stop * 2
        rp_es_holdout = 0.1

    # 안전장치: force_kfold_mode=True면 rerun_params의 mode와 무관하게 kfold 강제.
    # 앙상블은 blending/stacking에 OOF가 필요하므로 single 모드는 진입 불가.
    if ens_cfg["force_kfold_mode"] and rp_mode != "kfold":
        if ens_cfg["verbose"]:
            print(f"[ensemble] force_kfold_mode=True → rerun mode='{rp_mode}' 를 'kfold'로 덮어씀")
        rp_mode = "kfold"

    rerun_mode = rp_mode
    n_folds_rerun = rp_n_folds

    for i, (clf_model, reg_model) in enumerate(candidates):
        name = f"{clf_model}-{reg_model}"
        # 중복 이름 방지 (동일 조합이 여러 번 들어올 수 있음)
        if name in base_results:
            name = f"{name}_{i}"

        if ens_cfg["verbose"]:
            print(f"\n{'=' * 70}")
            print(f"[{i+1}/{len(candidates)}] Base 모델: {name}")
            print(f"{'=' * 70}")

        # ── precomputed 재사용 체크 ──
        # 동일 (clf_model, reg_model)이 precomputed에 있고 oof_pred가 있으면
        # HPO/rerun을 건너뛰고 주입된 결과를 그대로 사용한다.
        use_precomputed = False
        if precomputed is not None and (clf_model, reg_model) in precomputed:
            pre = precomputed[(clf_model, reg_model)]
            pre_hpo = pre.get("hpo_result")
            pre_rerun = pre.get("rerun_result")
            if pre_hpo is None or pre_rerun is None:
                if ens_cfg["verbose"]:
                    print(f"[{name}] precomputed 항목이 불완전 "
                          f"(hpo_result={'O' if pre_hpo else 'X'}, "
                          f"rerun_result={'O' if pre_rerun else 'X'}) → 새로 학습")
            elif "oof_pred" not in pre_rerun:
                if ens_cfg["verbose"]:
                    print(f"[{name}] precomputed rerun_result에 oof_pred 없음 "
                          f"(single 모드) → 새로 학습")
            else:
                hpo_result = pre_hpo
                rerun_result = pre_rerun
                use_precomputed = True
                if ens_cfg["verbose"]:
                    print(f"[{name}] precomputed 재사용 — HPO/rerun 스킵")

        if not use_precomputed:
            # ── ① HPO ──
            hpo_result = run_e2e_optimization(
                pos_data=pos_data,
                feat_cols=feat_cols,
                pipeline_config=pipeline_config,
                clf_model=clf_model,
                reg_model=reg_model,
                n_trials=n_trials,
                n_folds=n_folds,
                clf_early_stop=clf_early_stop,
                reg_early_stop=reg_early_stop,
                label_col=label_col,
                imbalance_method=imbalance_method,
                agg_funcs=agg_funcs,
                top_k_range=top_k_range,
                top_k_fixed=top_k_fixed,
                clf_fixed=clf_fixed,
                reg_fixed=reg_fixed,
                unit_data_input=unit_data_input,
                unit_feat_cols_input=unit_feat_cols_input,
            )

            # ── ② Rerun (kfold 강제 → OOF 확보) ──
            if ens_cfg["verbose"]:
                print(f"\n[{name}] Rerun mode={rerun_mode}, folds={n_folds_rerun}")

            rerun_result = rerun_best_trial(
                pos_data=pos_data,
                feat_cols=feat_cols,
                best_params=hpo_result["best_params"],
                pipeline_config=pipeline_config,
                clf_model=clf_model,
                reg_model=reg_model,
                mode=rerun_mode,
                n_folds=n_folds_rerun,
                es_holdout=rp_es_holdout,
                clf_early_stop=rp_clf_es,
                reg_early_stop=rp_reg_es,
                label_col=label_col,
                imbalance_method=imbalance_method,
                agg_funcs=agg_funcs,
                top_k_fixed=top_k_fixed,
                clf_fixed=clf_fixed,
                reg_fixed=reg_fixed,
                unit_data_input=unit_data_input,
                unit_feat_cols_input=unit_feat_cols_input,
            )

        # ── ③ Unit-level 키/라벨 정합성 확인 ──
        unit_data = rerun_result["unit_data"]
        y_train = unit_data["train"][TARGET_COL].values
        y_val = unit_data["val"][TARGET_COL].values
        key_train = unit_data["train"][KEY_COL].values
        key_val = unit_data["val"][KEY_COL].values
        key_test = unit_data["test"][KEY_COL].values

        if shared is None:
            shared = dict(
                y_train=y_train, y_val=y_val,
                key_train=key_train, key_val=key_val, key_test=key_test,
            )
        else:
            # 후보 간 unit 순서 동일한지 체크
            if len(key_train) != len(shared["key_train"]) or \
               len(key_val) != len(shared["key_val"]) or \
               len(key_test) != len(shared["key_test"]):
                raise ValueError(
                    f"[ensemble] 후보 '{name}'의 unit 수가 이전 후보와 다릅니다. "
                    f"train={len(key_train)} vs {len(shared['key_train'])}, "
                    f"val={len(key_val)} vs {len(shared['key_val'])}, "
                    f"test={len(key_test)} vs {len(shared['key_test'])}"
                )
            if not np.array_equal(key_train, shared["key_train"]):
                # 순서가 다르면 reindex 필요 → 예외로 알림 (사일런트 미스매치 방지)
                raise ValueError(
                    f"[ensemble] 후보 '{name}'의 train unit 순서가 이전 후보와 다릅니다. "
                    f"pipeline_config가 candidate 간 일관되어야 합니다."
                )

        # ── ④ oof_pred 키 검증 (single 모드 차단) ──
        # rerun이 single 모드면 train 예측이 in-sample이라 oof_pred 키가 없음.
        # 이 경우 blending/stacking 메타학습이 in-sample 타깃을 학습하게 되어
        # 심각한 과적합이 발생하므로 앙상블 진입 자체를 거부한다.
        if "oof_pred" not in rerun_result:
            raise ValueError(
                f"[ensemble] 후보 '{name}'의 rerun 결과에 'oof_pred' 키가 없습니다. "
                f"single 모드의 train 예측은 in-sample이라 블렌딩/스태킹에 쓸 수 없습니다. "
                f"ensemble_config에 force_kfold_mode=True로 설정하세요."
            )

        # ── ⑤ clf proba (meta feature용, 있을 때만) ──
        clf_proba_train = None
        clf_proba_val = None
        clf_proba_test = None
        if "clf_proba_mean" in unit_data["train"].columns:
            clf_proba_train = unit_data["train"]["clf_proba_mean"].values
            clf_proba_val = unit_data["val"]["clf_proba_mean"].values
            clf_proba_test = unit_data["test"]["clf_proba_mean"].values

        base_results[name] = dict(
            oof_pred=np.asarray(rerun_result["oof_pred"], dtype=float),
            val_pred=np.asarray(rerun_result["val_pred"], dtype=float),
            test_pred=np.asarray(rerun_result["test_pred"], dtype=float),
            val_rmse=float(rerun_result["val_rmse"]),
            clf_model=clf_model,
            reg_model=reg_model,
            best_params=hpo_result["best_params"],
            clf_proba_train=clf_proba_train,
            clf_proba_val=clf_proba_val,
            clf_proba_test=clf_proba_test,
        )

        if ens_cfg["verbose"]:
            print(f"[{name}] 수집 완료 — val RMSE: {rerun_result['val_rmse']:.6f}")

    return base_results, shared


# ═════════════════════════════════════════════════════════════
# 2) 블렌딩 가중치 옵티마이저
# ═════════════════════════════════════════════════════════════
def _rmse_from_weights(weights, P, y):
    """가중 평균 예측의 RMSE (음수 클리핑 후)"""
    blend = P @ weights
    blend = np.clip(blend, 0, None)
    return float(np.sqrt(np.mean((y - blend) ** 2)))


def blend_weights_slsqp(P_oof, y_train):
    """
    제약(w≥0, Σw=1) 볼록 최적화. 초기값은 equal weight.

    Parameters
    ----------
    P_oof : ndarray, shape (n_samples, n_models)
    y_train : ndarray, shape (n_samples,)

    Returns
    -------
    weights : ndarray, shape (n_models,)
    """
    n_models = P_oof.shape[1]
    w0 = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0)] * n_models
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    res = minimize(
        _rmse_from_weights,
        w0,
        args=(P_oof, y_train),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not res.success:
        print(f"[blend-slsqp] WARN: SLSQP 수렴 실패 — {res.message}. equal weight로 폴백.")
        return w0
    return res.x


def blend_weights_optuna(P_oof, y_train, n_trials=200):
    """Optuna 기반 가중치 탐색 (비볼록 문제/비교용)"""
    try:
        import optuna
    except ImportError:
        print("[blend-optuna] optuna 미설치 — equal weight로 폴백")
        return np.ones(P_oof.shape[1]) / P_oof.shape[1]

    n_models = P_oof.shape[1]

    def objective(trial):
        raw = np.array([
            trial.suggest_float(f"w_{i}", 0.0, 1.0) for i in range(n_models)
        ])
        s = raw.sum()
        if s < 1e-12:
            return float("inf")
        w = raw / s
        return _rmse_from_weights(w, P_oof, y_train)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler())  # 시드 제거: 매 실행마다 다른 trial 시퀀스
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    raw = np.array([study.best_params[f"w_{i}"] for i in range(n_models)])
    return raw / raw.sum()


def blend_weights_equal(n_models):
    return np.ones(n_models) / n_models


# ═════════════════════════════════════════════════════════════
# 3) 블렌딩 실행
# ═════════════════════════════════════════════════════════════
def run_blending(base_results, shared, ens_cfg):
    """
    base_results의 OOF/val/test 예측을 가중 평균으로 결합.

    Returns
    -------
    blend_out : dict
        {
            method_key: {
                "weights": dict(model_name -> weight),
                "val_pred": ndarray,
                "test_pred": ndarray,
                "val_rmse": float,
                "train_rmse": float,   # OOF 기준
            }
        }
        method_key 예: 'blend_slsqp', 'blend_equal', 'blend_optuna'
    """
    names = list(base_results.keys())
    P_oof = np.column_stack([base_results[n]["oof_pred"] for n in names])
    P_val = np.column_stack([base_results[n]["val_pred"] for n in names])
    P_test = np.column_stack([base_results[n]["test_pred"] for n in names])
    y_train = shared["y_train"]
    y_val = shared["y_val"]

    blend_out = {}
    methods = []
    if ens_cfg["blend_optimizer"] == "slsqp":
        methods.append("slsqp")
    elif ens_cfg["blend_optimizer"] == "optuna":
        methods.append("optuna")
    elif ens_cfg["blend_optimizer"] == "equal":
        methods.append("equal")

    if ens_cfg["blend_include_equal"] and "equal" not in methods:
        methods.append("equal")

    for method in methods:
        if method == "slsqp":
            w = blend_weights_slsqp(P_oof, y_train)
        elif method == "optuna":
            w = blend_weights_optuna(P_oof, y_train, ens_cfg["blend_optuna_trials"])
        else:  # equal
            w = blend_weights_equal(len(names))

        val_pred = P_val @ w
        test_pred = P_test @ w
        oof_pred = P_oof @ w
        if ens_cfg["clip_negative"]:
            val_pred = np.clip(val_pred, 0, None)
            test_pred = np.clip(test_pred, 0, None)
            oof_pred = np.clip(oof_pred, 0, None)

        blend_out[f"blend_{method}"] = dict(
            weights={n: float(w[i]) for i, n in enumerate(names)},
            val_pred=val_pred,
            test_pred=test_pred,
            oof_pred=oof_pred,
            val_rmse=float(rmse(y_val, val_pred)),
            train_rmse=float(rmse(y_train, oof_pred)),
        )

    return blend_out


# ═════════════════════════════════════════════════════════════
# 4) 스태킹 (Ridge / LightGBM meta-learner)
# ═════════════════════════════════════════════════════════════
def _build_meta_features(base_results, shared, split, include_clf_proba):
    """
    split별 메타 피처 행렬 구성

    Parameters
    ----------
    split : str
        'oof' | 'val' | 'test' (oof는 train 대체)
    """
    names = list(base_results.keys())
    key_map = {"oof": "oof_pred", "val": "val_pred", "test": "test_pred"}
    proba_map = {"oof": "clf_proba_train", "val": "clf_proba_val", "test": "clf_proba_test"}

    cols = [base_results[n][key_map[split]] for n in names]
    feat_names = list(names)

    if include_clf_proba:
        # 모든 후보의 clf proba가 동일할 리 없으므로 각자 추가
        # 단, clf 모델이 같은 후보끼리는 proba가 같을 수 있어 중복 제거
        seen_proba = {}
        for n in names:
            proba = base_results[n][proba_map[split]]
            if proba is None:
                continue
            # 동일 array는 한 번만 (hash 기반 dedupe는 비용↑, 후보 간 clf_model 이름으로 dedupe)
            clf_name = base_results[n]["clf_model"]
            if clf_name in seen_proba:
                continue
            seen_proba[clf_name] = proba
            cols.append(proba)
            feat_names.append(f"clf_proba_{clf_name}")

    X_meta = np.column_stack(cols)
    return X_meta, feat_names


def _make_meta_model(meta_type, ens_cfg):
    """스태킹 meta-learner 인스턴스 생성 (early stopping 없음)"""
    if meta_type == "ridge":
        if ens_cfg["stacking_ridge_alpha"] is None:
            # 범위 확장: 더 강한 정규화 허용
            return RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
        return Ridge(alpha=float(ens_cfg["stacking_ridge_alpha"]))

    elif meta_type == "lgbm":
        import lightgbm as lgb
        # 고정 n_estimators → early stopping 없음 → val double-dip 차단
        # 강한 regularization으로 overfit 억제
        return lgb.LGBMRegressor(
            n_estimators=ens_cfg["stacking_lgbm_n_estimators"],
            max_depth=3,
            learning_rate=0.02,
            num_leaves=7,
            min_child_samples=100,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        raise ValueError(f"unknown meta_type: {meta_type}")


def run_stacking(base_results, shared, ens_cfg):
    """
    base_results의 OOF를 meta-learner에 학습, val/test에 예측.

    Overfit 방어:
    - LGBM meta는 val set을 건드리지 않음 (early stopping 제거, 고정 n_estimators)
    - stacking_nested_cv=True → meta learner를 k-fold로 학습해 진짜 meta-OOF 생성
    - RidgeCV alpha 범위 [0.01 ~ 1000]로 강한 정규화 허용

    Returns
    -------
    stack_out : dict
        {
            method_key: {
                "meta_features": list,
                "val_pred": ndarray,
                "test_pred": ndarray,
                "oof_pred": ndarray,      # nested_cv=True면 진짜 meta-OOF, 아니면 in-sample
                "val_rmse": float,
                "train_rmse": float,      # oof_pred 기준
                "coef": dict or None,     # Ridge (nested=False)일 때만
                "meta_type": str,
                "nested_cv": bool,
            }
        }
    """
    y_train = shared["y_train"]
    y_val = shared["y_val"]
    include_proba = ens_cfg["stacking_include_clf_proba"]
    nested = ens_cfg["stacking_nested_cv"]
    meta_folds = ens_cfg["stacking_meta_folds"]

    X_train_meta, feat_names = _build_meta_features(base_results, shared, "oof", include_proba)
    X_val_meta, _ = _build_meta_features(base_results, shared, "val", include_proba)
    X_test_meta, _ = _build_meta_features(base_results, shared, "test", include_proba)

    candidates_meta = []
    if ens_cfg["stacking_meta"] in ("ridge", "auto"):
        candidates_meta.append("ridge")
    if ens_cfg["stacking_meta"] in ("lgbm", "auto"):
        candidates_meta.append("lgbm")

    stack_out = {}

    for meta_type in candidates_meta:
        try:
            if nested:
                # ── Nested CV: 진짜 meta-OOF ──
                kf = KFold(n_splits=meta_folds, shuffle=True, random_state=SEED)
                oof_pred = np.zeros(len(y_train))
                val_preds_folds = []
                test_preds_folds = []
                fold_models = []

                for tr_idx, va_idx in kf.split(X_train_meta):
                    model = _make_meta_model(meta_type, ens_cfg)
                    model.fit(X_train_meta[tr_idx], y_train[tr_idx])
                    oof_pred[va_idx] = model.predict(X_train_meta[va_idx])
                    val_preds_folds.append(model.predict(X_val_meta))
                    test_preds_folds.append(model.predict(X_test_meta))
                    fold_models.append(model)

                val_pred = np.mean(val_preds_folds, axis=0)
                test_pred = np.mean(test_preds_folds, axis=0)

                # 계수: Ridge일 때만 첫 fold 기준 저장 (진단용)
                if meta_type == "ridge":
                    first = fold_models[0]
                    coef = {feat_names[i]: float(first.coef_[i]) for i in range(len(feat_names))}
                    coef["_intercept"] = float(first.intercept_)
                    best_alpha = getattr(first, "alpha_", None)
                    if best_alpha is not None:
                        coef["_best_alpha_fold0"] = float(best_alpha)
                    coef["_note"] = f"fold 0/{meta_folds} 기준 (nested CV)"
                else:
                    coef = None

            else:
                # ── 단순 학습: OOF → in-sample 예측 (진단용) ──
                model = _make_meta_model(meta_type, ens_cfg)
                model.fit(X_train_meta, y_train)
                oof_pred = model.predict(X_train_meta)  # in-sample: 진짜 OOF 아님
                val_pred = model.predict(X_val_meta)
                test_pred = model.predict(X_test_meta)

                if meta_type == "ridge":
                    coef = {feat_names[i]: float(model.coef_[i]) for i in range(len(feat_names))}
                    coef["_intercept"] = float(model.intercept_)
                    best_alpha = getattr(model, "alpha_", None)
                    if best_alpha is not None:
                        coef["_best_alpha"] = float(best_alpha)
                else:
                    coef = None

        except ImportError as e:
            print(f"[stacking-{meta_type}] 의존성 미설치 — 스킵: {e}")
            continue

        if ens_cfg["clip_negative"]:
            val_pred = np.clip(val_pred, 0, None)
            test_pred = np.clip(test_pred, 0, None)
            oof_pred = np.clip(oof_pred, 0, None)

        stack_out[f"stacking_{meta_type}"] = dict(
            meta_features=feat_names,
            val_pred=val_pred,
            test_pred=test_pred,
            oof_pred=oof_pred,
            val_rmse=float(rmse(y_val, val_pred)),
            train_rmse=float(rmse(y_train, oof_pred)),
            coef=coef,
            meta_type=meta_type,
            nested_cv=nested,
        )

    return stack_out


# ═════════════════════════════════════════════════════════════
# 5) 마스터 함수
# ═════════════════════════════════════════════════════════════
def run_ensemble(
    candidates,
    pos_data,
    feat_cols,
    ensemble_config=None,
    pipeline_config=None,
    precomputed=None,
    **run_kwargs,
):
    """
    앙상블 파이프라인 통합 실행.

    Parameters
    ----------
    candidates : list of tuple
        [(clf_model, reg_model), ...]
    pos_data, feat_cols : e2e_hpo와 동일
    ensemble_config : dict
        ENSEMBLE_CONFIG. enabled=False면 base만 학습 후 단일 best 반환.
    pipeline_config : dict
        e2e_hpo의 DEFAULT_CONFIG. 모든 candidate에 동일하게 적용.
    precomputed : dict, optional
        {(clf_model, reg_model): {"hpo_result": ..., "rerun_result": ...}}
        이미 학습된 후보 결과를 재사용해 HPO/rerun 중복 학습을 방지.
        collect_base_predictions에 그대로 전달된다.
    **run_kwargs :
        collect_base_predictions에 그대로 전달
        (n_trials, n_folds, clf_early_stop, reg_early_stop, label_col,
         imbalance_method, agg_funcs, top_k_range, top_k_fixed,
         unit_data_input, unit_feat_cols_input)

    Returns
    -------
    result : dict
        {
            "base_results":   dict (collect_base_predictions 결과),
            "shared":         dict (y_train/y_val/keys),
            "blend":          dict or None,
            "stacking":       dict or None,
            "comparison":     pd.DataFrame (모든 결과의 val RMSE 비교),
            "best_method":    str (val RMSE 기준 최저),
            "best_val_pred":  ndarray,
            "best_test_pred": ndarray,
            "best_val_rmse":  float,
            "ensemble_config": dict,
        }
    """
    ens_cfg = _merge_ens_config(ensemble_config)

    if ens_cfg["verbose"]:
        print(f"\n{'█' * 70}")
        print(f"█  ENSEMBLE 파이프라인 시작")
        print(f"█  후보: {len(candidates)}개 — {[f'{c}-{r}' for c,r in candidates]}")
        print(f"█  method={ens_cfg['method']}, blend_opt={ens_cfg['blend_optimizer']}, "
              f"stacking_meta={ens_cfg['stacking_meta']}")
        print(f"█  force_kfold={ens_cfg['force_kfold_mode']} (rerun folds={ens_cfg['n_folds_rerun']})")
        print(f"{'█' * 70}")

    # ── Base 예측 수집 ──
    base_results, shared = collect_base_predictions(
        candidates=candidates,
        pos_data=pos_data,
        feat_cols=feat_cols,
        ensemble_config=ens_cfg,
        pipeline_config=pipeline_config,
        precomputed=precomputed,
        **run_kwargs,
    )

    # ── 스위치별 앙상블 실행 ──
    blend_out = None
    stack_out = None

    if ens_cfg["enabled"] and ens_cfg["method"] in ("blend", "both"):
        if ens_cfg["verbose"]:
            print(f"\n{'─' * 70}")
            print(f"[블렌딩 실행] optimizer={ens_cfg['blend_optimizer']}")
            print(f"{'─' * 70}")
        blend_out = run_blending(base_results, shared, ens_cfg)

    if ens_cfg["enabled"] and ens_cfg["method"] in ("stacking", "both"):
        if ens_cfg["verbose"]:
            print(f"\n{'─' * 70}")
            print(f"[스태킹 실행] meta={ens_cfg['stacking_meta']}, "
                  f"include_clf_proba={ens_cfg['stacking_include_clf_proba']}")
            print(f"{'─' * 70}")
        stack_out = run_stacking(base_results, shared, ens_cfg)

    # ── 비교표 ──
    rows = []
    for name, r in base_results.items():
        rows.append({"method": f"base_{name}", "val_rmse": r["val_rmse"],
                     "train_rmse": float(rmse(shared["y_train"], r["oof_pred"]))})
    if blend_out:
        for k, r in blend_out.items():
            rows.append({"method": k, "val_rmse": r["val_rmse"], "train_rmse": r["train_rmse"]})
    if stack_out:
        for k, r in stack_out.items():
            rows.append({"method": k, "val_rmse": r["val_rmse"], "train_rmse": r["train_rmse"]})

    comparison = pd.DataFrame(rows)
    # overfit_gap = val - train (양수면 val이 더 나쁨 = 일반화 양호, 음수면 val이 더 좋음 = 의심)
    # 단, RMSE 특성상 train이 더 낮은 게 정상. |gap|이 너무 작으면 의심 X, 너무 크면 base가 train overfit
    comparison["overfit_gap"] = comparison["val_rmse"] - comparison["train_rmse"]
    comparison["abs_gap"] = comparison["overfit_gap"].abs()

    gap_thresh = float(ens_cfg["overfit_gap_warn"])
    comparison["warn"] = comparison["abs_gap"] > gap_thresh
    comparison = comparison.sort_values("val_rmse").reset_index(drop=True)

    # ── 최종 best 선정 ──
    if ens_cfg["conservative_selection"]:
        # gap이 임계값 이하인 것 중에서 val RMSE 최저
        safe = comparison[~comparison["warn"]]
        if len(safe) == 0:
            if ens_cfg["verbose"]:
                print(f"[ensemble] WARN: conservative_selection=True인데 overfit_gap 임계값({gap_thresh})을 "
                      f"통과한 방법이 없습니다. 전체에서 val RMSE 최저 선택.")
            best_row = comparison.iloc[0]
        else:
            best_row = safe.iloc[0]
            if ens_cfg["verbose"]:
                n_excluded = len(comparison) - len(safe)
                print(f"[ensemble] conservative_selection: {n_excluded}개 방법이 overfit 의심으로 제외됨")
    else:
        best_row = comparison.iloc[0]

    best_method = best_row["method"]
    if best_method.startswith("base_"):
        key = best_method.replace("base_", "", 1)
        best_val_pred = base_results[key]["val_pred"]
        best_test_pred = base_results[key]["test_pred"]
    elif best_method.startswith("blend_"):
        best_val_pred = blend_out[best_method]["val_pred"]
        best_test_pred = blend_out[best_method]["test_pred"]
    else:  # stacking_
        best_val_pred = stack_out[best_method]["val_pred"]
        best_test_pred = stack_out[best_method]["test_pred"]

    if ens_cfg["verbose"]:
        print(f"\n{'█' * 70}")
        print(f"█  앙상블 결과 비교 (val RMSE 오름차순)")
        print(f"{'█' * 70}")
        print(comparison.to_string(index=False))
        print(f"\n★ Best method: {best_method} (val RMSE: {best_row['val_rmse']:.6f})")

        if blend_out and "blend_slsqp" in blend_out:
            print(f"\n[블렌딩 SLSQP 가중치]")
            for n, w in blend_out["blend_slsqp"]["weights"].items():
                print(f"  {n:30s} : {w:.4f}")

        if stack_out and "stacking_ridge" in stack_out:
            coef = stack_out["stacking_ridge"]["coef"]
            if coef is not None:
                print(f"\n[스태킹 Ridge 계수]")
                for k, v in coef.items():
                    print(f"  {k:30s} : {v:+.4f}")

    return dict(
        base_results=base_results,
        shared=shared,
        blend=blend_out,
        stacking=stack_out,
        comparison=comparison,
        best_method=best_method,
        best_val_pred=best_val_pred,
        best_test_pred=best_test_pred,
        best_val_rmse=float(best_row["val_rmse"]),
        ensemble_config=ens_cfg,
    )