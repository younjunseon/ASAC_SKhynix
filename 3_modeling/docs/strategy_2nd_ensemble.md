# 2차 Funnel — 앙상블 모델링 코드 설계

> **작성일**: 2026-04-16
> **대상**: 2차 ensemble 새 노트북 + 기존 모듈 패치
> **관련 문서**: [strategy_2nd_preprocessing.md](strategy_2nd_preprocessing.md) (전처리 모듈)
> **베이스 노트북**: [baseline.ipynb](baseline.ipynb) (1차 funnel)

---

## 1. Overview

### 1.1 2차 목표

- **앙상블 베이스코드** 구축 (3차에서 가중치 정밀 튜닝 시 재사용)
- **모델 다양성 확보**: 회귀 = LGBM + ExtraTrees + ElasticNet / 분류 = LGBM + ExtraTrees + LogReg(elasticnet)
- **모델별 die-level OOF CSV 저장** → 3차에서 재학습 없이 가중치 튜닝 가능

### 1.2 사용자 확정 결정 (2026-04-16 대화)

| 항목 | 결정 | 근거 |
|---|---|---|
| Optuna 구조 | **옵션 A** (1 study, trial당 6 모델 동시 학습) | Two-Stage + A4 구조 정합성, trial-level 앙상블 평가 |
| 분류 3-모델 | LGBM + ExtraTrees + LogReg(elasticnet) | 회귀와 미러링 |
| 회귀 3-모델 | LGBM + ExtraTrees + ElasticNet | 약신호 + 다중공선성 47쌍 대응 |
| Calibration | Isotonic on LGBM/ET만 (LogReg는 자체) | `final=P×R`의 P bias 보정 |
| A4 (proba→feature) | True | 논문 2-1 R² 0.117→0.716 |
| 모델 가중치 튜닝 | 후처리 그리드 (step=0.025, ~861조합) | 학습 X, 빠름 |
| 포지션 가중치 튜닝 | sub-study Optuna (per-model, Dirichlet 정규화) | 학습 X, 가중치만 |
| DB 정공 | `study.add_trial`로 rerun도 DB에 append | 1차에서 누락됐던 재현성 |
| 시각화 | feature 제거 후보 + 가중치 + 예측 vs 실측 + RMSE 비교 | baseline 패턴 + 진단 |
| LDS 확장 | `compute_lds_weights(expand_to_die=True)` — 함수 내부에서 unit→die 확장 | 호출자 단순화 (결정 1 = A) |
| final_pred CSV 스키마 | **최종 베스트만 스키마 C** (ufs_serial, y_true, y_pred, clf_proba_unit, R_combined_unit, R_{m}_unit × 3, clf_proba_{m} × 3) | 3차 재사용 시 재학습 없이 즉시 (결정 2) |
| hybrid_scale 구현 | **sklearn 표준 fit/transform 패턴 (HybridScaler 클래스)** | 누수 안전 + 대시보드 재사용 (결정 3 = B) |
| 웨이퍼맵 기반 사전 제외 | **`EXCLUDE_COLS` 37개 하드코딩** — cleaning/outlier/scaling 이전 적용 | 35개: [1_eda/wafer_map_image/★분류완료/1_꼭빼야함/](../1_eda/wafer_map_image/★분류완료/1_꼭빼야함/) 수동 분류 + 2개 (X1056, X1072): 공간 패턴 feature 수동 선별 결과. Colab 폴더 미동기화 대비 하드코딩 |
| Stage 2 LGBM objective 탐색 | **`['regression','poisson','tweedie_1.2','tweedie_1.5']` Optuna 축** | 논문 2-2/2-5/2-6/2-7: zero-inflated + 비음수에서 Tweedie/Poisson이 MSE보다 우위 |
| Spearman 진단 metric | **Cell 11 시각화에 Pearson·Spearman 병기** | 논문 8-3: RMSE만으론 극단값 under-predict 놓침 (진단 보조) |
| **Position 인코딩** | **Ordinal + OHE 병행** (정수 `POSITION_COL` + `p_1`~`p_4` binary 4개) | OHE 단독은 순서 정보 파괴 (p1~p4 독립 취급), Ordinal 단독은 선형 모델이 "3=1의 3배"로 오해. 병행으로 트리·선형 모두 대응 |
| **run_wf_xy 메타 (die_x/die_y/run_id)** | **제외 유지** (파싱 안 함) | EDA Phase 23 (radial_dist r=0.006 무상관), Phase 24 (NNR 잔차 개선 0%), raw X에 공간 패턴 feature 6개 존재 — 수동 선별 결과: **유지 6개 `X708, X1059, X1073, X1075, X1076, X1077`** / **제거 2개 `X1056` (Ring, 기여 모호) · `X1072` (X708과 radial 중복)** |

### 1.3 pipeline_config — baseline 그대로

```python
pipeline_config = dict(
    input_level='die',
    run_clf=True,          # Two-Stage 유지
    clf_output='proba',
    clf_filter=False,      # True는 3차
    clf_optuna=True,
    run_fs=False,          # FS 활성화는 3차
    fs_optuna=False,
    reg_level='position',
    reg_optuna=True,
    zero_clip=True,        # threshold 정밀 튜닝은 3차
)
```

---

## 2. 모듈 패치 명세

### 2.1 `model_zoo.py` — 신규 모델 등록

#### 2.1.1 import 추가

```python
from sklearn.linear_model import ElasticNet, LogisticRegression
```

#### 2.1.2 `MODEL_REGISTRY` 확장

```python
MODEL_REGISTRY = {
    "lgbm": {"clf": lgb.LGBMClassifier, "reg": lgb.LGBMRegressor,
             "supports_early_stopping": True},
    "rf":   {"clf": RandomForestClassifier, "reg": RandomForestRegressor,
             "supports_early_stopping": False},
    "et":   {"clf": ExtraTreesClassifier, "reg": ExtraTreesRegressor,
             "supports_early_stopping": False},

    # ★ 신규 (2차)
    "logreg_enet": {
        "clf": LogisticRegression,
        "reg": None,                               # 분류 전용
        "supports_early_stopping": False,
    },
    "enet": {
        "clf": None,                               # 회귀 전용
        "reg": ElasticNet,
        "supports_early_stopping": False,
    },
}
```

#### 2.1.3 `get_default_params()` 확장

```python
def get_default_params(name, task, device=None):
    # ... 기존 ...

    elif name == "logreg_enet":
        params = dict(
            penalty='elasticnet',
            solver='saga',
            C=1.0,
            l1_ratio=0.5,
            max_iter=2000,
            tol=1e-3,
            random_state=SEED,
            n_jobs=-1,
        )
    elif name == "enet":
        params = dict(
            alpha=0.001,
            l1_ratio=0.5,
            max_iter=5000,
            tol=1e-4,
            random_state=SEED,
        )
    return params
```

#### 2.1.4 `fit_model()` 확장 — sample_weight 지원

```python
def fit_model(model, X_train, y_train,
              X_val=None, y_val=None, early_stop=50,
              sample_weight=None):              # ★ 신규
    model_cls = type(model).__name__.lower()

    if "lgbm" in model_cls:
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if X_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(early_stop, verbose=False),
                lgb.log_evaluation(0),
            ]
        model.fit(X_train, y_train, **fit_kwargs)

    else:
        # RF/ET/ElasticNet/LogReg 모두 sklearn 표준 시그니처
        if sample_weight is not None:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)

    return model
```

**주의**: sklearn ElasticNet은 `sample_weight` 지원. LogReg는 `sample_weight` 지원 (분류이므로 Stage 1 불균형 대응에도 유용).

---

### 2.2 `search_space.py` — 모델별 HP 함수

#### 2.2.0 `lgbm_space()` — reg prefix에 `objective` 축 추가 (★ 2차 신규)

```python
def lgbm_space(trial, prefix=""):
    p = prefix
    if p == "reg_":
        params = dict(
            # ... 기존 HP (n_estimators, learning_rate 등) 유지 ...
            n_estimators=trial.suggest_int(f"{p}n_estimators", 100, 1500),
            learning_rate=trial.suggest_float(f"{p}learning_rate", 0.005, 0.1, log=True),
            # ... (생략, 기존 reg_ space 그대로) ...
            random_state=SEED, n_jobs=-1, verbose=-1, device=DEVICE,
        )

        # ★ 2차 신규: objective 탐색 (논문 2-5/2-6/2-7 근거)
        obj_choice = trial.suggest_categorical(
            f"{p}objective",
            ['regression', 'poisson', 'tweedie_1.2', 'tweedie_1.5'],
        )
        if obj_choice == 'poisson':
            params['objective'] = 'poisson'
        elif obj_choice.startswith('tweedie'):
            params['objective'] = 'tweedie'
            params['tweedie_variance_power'] = float(obj_choice.split('_')[1])
        # 'regression' (MSE, LGBM default)은 파라미터 주입 없음
        return params

    # clf_ / 기본 prefix는 기존 그대로
    ...
```

**주의**:
- ET / ElasticNet은 objective 고정 (ElasticNet=MSE, ET=variance 기반) → 탐색 X
- **LGBM만 objective 축 탐색**
- Tweedie variance_power는 1.0 (Poisson 한계)~2.0 (Gamma 한계). zero-inflated + 비음수엔 1.1~1.5 권장

#### 2.2.1 `et_space(trial, prefix="")` — 신규

```python
def et_space(trial, prefix=""):
    """ExtraTrees 탐색 공간 (분류·회귀 공통)"""
    p = prefix
    return dict(
        n_estimators=trial.suggest_int(f"{p}n_estimators", 300, 1500),
        max_depth=trial.suggest_int(f"{p}max_depth", 6, 30),
        min_samples_leaf=trial.suggest_int(f"{p}min_samples_leaf", 1, 30),
        min_samples_split=trial.suggest_int(f"{p}min_samples_split", 2, 40),
        max_features=trial.suggest_categorical(
            f"{p}max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]),
        bootstrap=trial.suggest_categorical(f"{p}bootstrap", [True, False]),
        random_state=SEED,
        n_jobs=-1,
    )
```

#### 2.2.2 `logreg_enet_space(trial, prefix="")` — 신규

```python
def logreg_enet_space(trial, prefix=""):
    """LogisticRegression(penalty='elasticnet') 탐색 공간"""
    p = prefix
    return dict(
        penalty='elasticnet',
        solver='saga',
        C=trial.suggest_float(f"{p}C", 1e-3, 100.0, log=True),
        l1_ratio=trial.suggest_float(f"{p}l1_ratio", 0.1, 0.9),
        max_iter=trial.suggest_int(f"{p}max_iter", 1000, 4000, step=500),
        tol=1e-3,
        random_state=SEED,
        n_jobs=-1,
    )
```

#### 2.2.3 `enet_space(trial, prefix="")` — 신규

```python
def enet_space(trial, prefix=""):
    """ElasticNet 회귀 탐색 공간"""
    p = prefix
    return dict(
        alpha=trial.suggest_float(f"{p}alpha", 1e-5, 1.0, log=True),
        l1_ratio=trial.suggest_float(f"{p}l1_ratio", 0.1, 0.9),
        max_iter=trial.suggest_int(f"{p}max_iter", 2000, 8000, step=1000),
        tol=1e-4,
        random_state=SEED,
    )
```

#### 2.2.4 `SEARCH_SPACES` 확장

```python
SEARCH_SPACES = {
    "lgbm":        lgbm_space,
    "rf":          rf_space,
    "et":          et_space,                    # 기존 rf_space였음 → 전용으로 교체
    "logreg_enet": logreg_enet_space,           # 신규
    "enet":        enet_space,                  # 신규
}
```

---

### 2.3 `e2e_hpo.py` — 시그니처 확장

#### 2.3.1 `run_e2e_optimization_with_pp(...)` 확장 인자

```python
def run_e2e_optimization_with_pp(
    # ── 기존 인자 (baseline과 동일) ──
    xs, xs_dict, ys, feat_cols,
    pipeline_config=None,
    n_trials=100, n_folds=3,
    clf_early_stop=50, reg_early_stop=50,
    label_col="label_bin",
    imbalance_method="scale_pos_weight",
    top_k_range=(50, 500), top_k_fixed=200,
    clf_filter_threshold_range=(0.05, 0.5),
    clf_filter_threshold_fixed=0.5,
    zero_clip_threshold_range=(0.0, 0.02),
    zero_clip_threshold_fixed=0.0,
    clf_fixed=None, reg_fixed=None,
    use_sampling=False, sample_frac=1.0,
    exclude_cols=None,
    target_transform_fn=None, target_inverse_fn=None,
    exp_id=None, db_path=None,
    study_user_attrs=None,
    warm_start_top_k=30, warm_start_enabled=True,
    pp_cache_size=10,

    # ── ★ 2차 신규 인자 ──
    clf_models=("lgbm",),              # tuple/list — 분류 모델 목록
    reg_models=("lgbm",),              # tuple/list — 회귀 모델 목록
    calibration=None,                  # dict: {'method': 'isotonic', 'cv': 3, 'models': ['lgbm', 'et']}
    add_clf_proba_to_reg=False,        # A4: clf_proba_mean을 reg feature로 주입
):
    ...
```

#### 2.3.2 `rerun_best_trial_with_pp(...)` 확장 인자

```python
def rerun_best_trial_with_pp(
    # ── 기존 (baseline과 동일) ──
    xs, xs_dict, ys, feat_cols, best_params,
    pipeline_config=None,
    mode='single', n_folds=5, es_holdout=0.1,
    clf_early_stop=100, reg_early_stop=100,
    label_col="label_bin",
    imbalance_method="scale_pos_weight",
    top_k_fixed=200,
    clf_filter_threshold_fixed=0.5,
    zero_clip_threshold_fixed=0.0,
    clf_fixed=None, reg_fixed=None,
    use_sampling=False, sample_frac=1.0,
    exclude_cols=None,
    target_transform_fn=None, target_inverse_fn=None,

    # ── ★ 2차 신규 ──
    clf_models=("lgbm",),
    reg_models=("lgbm",),
    calibration=None,
    add_clf_proba_to_reg=False,
    save_per_model_oof=False,          # OOF CSV 저장 여부
    oof_dir=None,                      # 저장 위치
):
    """
    Returns
    -------
    final : dict
        ── 앙상블 RMSE ──
        - 'val_rmse', 'test_rmse': 단순 평균 앙상블 RMSE
        - 'per_model_val_rmse': dict {model: rmse}        ★ 신규
        - 'per_model_test_rmse': dict {model: rmse}       ★ 신규

        ── OOF / 저장 ──
        - 'oof_files': list[str] (저장된 CSV 경로)        ★ 신규

        ── 사후 분석용 (Cell 7 진단 시각화 + 3차 재사용) ──
        - 'per_model_fold_models': dict {model: list[fitted_model]}   ★ 신규
            예: {'lgbm': [m1, m2, ...], 'et': [...], 'enet': [...]}
            각 fold에서 학습된 모델 객체 리스트 — feature_importances_ / coef_ 추출용
        - 'per_model_clf_fold_models': dict {model: list[fitted_model]}  ★ 신규 (분류기)
        - 'selected_cols': list[str]                                  ★ 신규
            rerun에서 실제 회귀 학습에 사용된 컬럼 (feat_cols_clean 순서)
        - 'scaler': HybridScaler (fitted)                             ★ 신규
            pickle 저장 후 대시보드 / 예측 파이프라인 재사용 가능

        ── 기존 구조 유지 ──
        - 'clf_result', 'reg_result': 원래 dict 형태 그대로
    """
```

---

### 2.4 `e2e_hpo.py` — 내부 흐름 변경

#### 2.4.1 Trial 1회 구조 (옵션 A)

```
[Trial 진입]
  1) PP params 샘플링 (cleaning + outlier + iso + lds + agg)
  2) 모델 HP 샘플링 (prefix로 모델 분리):
     - clf_lgbm_*, clf_et_*, clf_logreg_enet_*
     - reg_lgbm_*, reg_et_*, reg_enet_*

  3) ▶ PP 실행 (LRU 캐시)
     cleaning → outlier → IsoForest → hybrid_scale → pos_data → LDS die-level
     → pos_data, feat_cols_clean, sample_weight (die-level), scaler

  4) ▶ 분류 3-모델 학습 (K-fold OOF)
     for clf_m in clf_models:
       OOF proba per position per model
     → calibration(LGBM/ET, Isotonic, cv=3) — LogReg는 스킵
     → soft voting: clf_proba_mean = mean(3 probas)

  5) ▶ 회귀 3-모델 학습 (K-fold OOF)
     ※ Stage 2 학습 데이터 = Y>0 서브셋만 (use_clf=True + clf_filter=False 기본 동작)
       — e2e_hpo._run_reg_oof의 `pos_tr = y_train > 0` 마스크로 자동 처리
       — 논문 1-4/2-1/2-2/2-8의 Hybrid Two-Stage 표준 구조
     for reg_m in reg_models:
       # Position 인코딩: Ordinal + OHE 병행 (§2.6 참조)
       unit_feat_cols = feat_cols_clean + [POSITION_COL, 'p_1', 'p_2', 'p_3', 'p_4']
       if add_clf_proba_to_reg:
           unit_feat_cols += ['clf_proba_mean']
       ※ run_wf_xy 메타(die_x/die_y/run_id)는 feat_cols_clean에 포함 X
       ※ LGBM만 objective ∈ {regression, poisson, tweedie(1.2/1.5)} Optuna 탐색
         (ET/ElasticNet은 MSE 고정)
       fit(sample_weight=sample_weight[fold_tr_idx])    # LDS die-level
       → die-level OOF pred per model

  6) ▶ 앙상블 (단순 평균 1/3씩)
     ensemble_pred = mean(reg_preds) × clf_proba_mean
     → unit-level mean → RMSE = objective

  7) trial.user_attrs 기록:
     - 'per_model_oof_rmse': dict
     - 'clf_val_auc', 'clf_val_ap' (기존)
     - 'n_feat_clean', 'n_feat_selected'
     - 'iso_enabled', 'lds_enabled' 등
```

#### 2.4.1-B Objective 안 HP 샘플링 구체 코드 스케치

```python
def objective(trial):
    # ── PP params 샘플링 ──
    pp_params = preprocessing_space(trial)
    cleaning_args, outlier_args, iso_args, lds_args, agg_funcs = split_pp_params(pp_params)

    # ── 모델별 HP 샘플링 (prefix 분리) ──
    clf_params_by_model = {
        m: SEARCH_SPACES[m](trial, prefix=f'clf_{m}_')
        for m in clf_models
    }
    reg_params_by_model = {
        m: SEARCH_SPACES[m](trial, prefix=f'reg_{m}_')
        for m in reg_models
    }

    # ── PP 실행 (LRU 캐시) ──
    cache_key = _pp_hash(pp_params)
    pp_result = _get_or_run_pp(cache, cache_key, xs, xs_dict, ys, feat_cols,
                               cleaning_args, outlier_args, iso_args, lds_args,
                               label_col, exclude_cols,
                               use_sampling, sample_frac, pp_cache_size)
    pos_data       = pp_result['pos_data']
    feat_cols_clean = pp_result['feat_cols']
    sample_weight  = pp_result['sample_weight']       # die-level or None

    # ── 분류 3-모델 + Calibration + Soft voting ──
    clf_result, per_model_clf = _run_clf_oof_multi(
        pos_data, feat_cols_clean, clf_params_by_model, clf_models,
        n_folds, clf_early_stop, label_col, imbalance_method,
        calibration=calibration, clf_output=cfg['clf_output'],
    )

    # ── 회귀 3-모델 (LDS + A4) ──
    unit_data, unit_feat_cols = _prepare_unit_data(
        pos_data, feat_cols_clean, clf_result, cfg, agg_funcs)
    # A4: clf_proba_mean은 _prepare_unit_data에서 unit_data에 이미 합쳐짐
    ensemble_reg, per_model_reg = _run_reg_oof_multi(
        unit_data, unit_feat_cols, reg_params_by_model, reg_models,
        n_folds, reg_early_stop,
        sample_weight=sample_weight,
        add_clf_proba_to_reg=add_clf_proba_to_reg,
        target_transform_fn=target_transform_fn,
        target_inverse_fn=target_inverse_fn,
    )

    # ── 앙상블 최종 RMSE (objective) ──
    if cfg['reg_level'] == 'position':
        ensemble_reg, agg_unit = _die_pred_to_unit(unit_data, ensemble_reg)
        y_train_unit = agg_unit['train'][TARGET_COL].values
    else:
        y_train_unit = unit_data['train'][TARGET_COL].values
    # zero_clip 후처리
    if cfg['zero_clip']:
        zclip = zero_clip_threshold_fixed  # Optuna 탐색 OFF
        ensemble_reg = _apply_zero_clip(ensemble_reg, y_train_unit, zclip)

    # trial.user_attrs 기록
    trial.set_user_attr('per_model_oof_rmse',
                        {m: per_model_reg[m]['train_rmse'] for m in reg_models})
    trial.set_user_attr('n_feat_clean', len(feat_cols_clean))
    trial.set_user_attr('iso_enabled', iso_args.get('iso_enabled', False))
    trial.set_user_attr('lds_enabled', lds_args.get('lds_enabled', False))
    # ... 기존 clf metrics 등도 기록

    return ensemble_reg['train_rmse']   # OOF RMSE = objective
```

#### 2.4.2 `_run_clf_oof_multi()` 신규

```python
def _run_clf_oof_multi(pos_data, feat_cols, clf_params_by_model, clf_models,
                      n_folds, early_stop, label_col, imbalance_method,
                      calibration=None, clf_output="proba"):
    """
    여러 분류 모델의 OOF proba 생성 + Calibration + Soft voting

    Returns
    -------
    clf_result : dict {position: {'train_proba': arr, 'val_proba': arr, 'test_proba': arr}}
        값은 soft-voted clf_proba_mean
    per_model_clf_result : dict {model: {position: {...}}}  ★ 신규
        각 모델의 독립 OOF proba (시각화/진단용)
    """
    per_model_results = {}
    for clf_m in clf_models:
        # 기존 _run_clf_oof 호출
        r = _run_clf_oof(pos_data, feat_cols, clf_params_by_model[clf_m], clf_m,
                         n_folds, early_stop, label_col, imbalance_method,
                         clf_output=clf_output)
        # Calibration (LGBM/ET만)
        if calibration and clf_m in calibration.get('models', []):
            r = _apply_isotonic_calibration(r, pos_data, label_col,
                                            cv=calibration.get('cv', 3))
        per_model_results[clf_m] = r

    # Soft voting (positon별)
    clf_result = {}
    positions = sorted(per_model_results[clf_models[0]].keys())
    for pos in positions:
        clf_result[pos] = {
            'train_proba': np.mean([per_model_results[m][pos]['train_proba']
                                    for m in clf_models], axis=0),
            'val_proba':   np.mean([per_model_results[m][pos]['val_proba']
                                    for m in clf_models], axis=0),
            'test_proba':  np.mean([per_model_results[m][pos]['test_proba']
                                    for m in clf_models], axis=0),
        }
    return clf_result, per_model_results
```

#### 2.4.3 `_apply_isotonic_calibration()` 신규

```python
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

def _apply_isotonic_calibration(clf_result, pos_data, label_col, cv=3):
    """
    Isotonic calibration을 position별 OOF proba에 적용

    train OOF proba + label로 isotonic regression fit → 모든 split에 transform
    """
    calibrated = {}
    for pos, d in clf_result.items():
        y_train = pos_data[pos]['train'][label_col].values
        p_train = d['train_proba']

        # K-fold로 calibrator 학습 (train OOF → 추가 OOF calibration)
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(p_train, y_train)

        calibrated[pos] = {
            'train_proba': iso.transform(p_train),
            'val_proba':   iso.transform(d['val_proba']),
            'test_proba':  iso.transform(d['test_proba']),
        }
    return calibrated
```

#### 2.4.4 `_run_reg_oof_multi()` 신규

```python
def _run_reg_oof_multi(unit_data, feat_cols, reg_params_by_model, reg_models,
                      n_folds, early_stop, sample_weight=None,
                      add_clf_proba_to_reg=False, **kwargs):
    """
    여러 회귀 모델의 die-level OOF pred 생성 (LDS + A4)

    Returns
    -------
    per_model_reg : dict {model: {'oof_pred': arr, 'val_pred': arr, 'test_pred': arr}}
    ensemble_reg  : dict (단순 평균 1/3)
    """
    per_model_results = {}
    effective_feat_cols = list(feat_cols)
    if add_clf_proba_to_reg and 'clf_proba_mean' in unit_data['train'].columns:
        effective_feat_cols.append('clf_proba_mean')

    for reg_m in reg_models:
        r = _run_reg_oof(
            unit_data, effective_feat_cols, reg_params_by_model[reg_m], reg_m,
            n_folds, early_stop,
            sample_weight=sample_weight,                 # ★ LDS
            **kwargs
        )
        per_model_results[reg_m] = r

    # 단순 평균 앙상블
    ensemble_reg = {
        'oof_pred':  np.mean([per_model_results[m]['oof_pred']  for m in reg_models], axis=0),
        'val_pred':  np.mean([per_model_results[m]['val_pred']  for m in reg_models], axis=0),
        'test_pred': np.mean([per_model_results[m]['test_pred'] for m in reg_models], axis=0),
    }
    ensemble_reg['train_rmse'] = rmse(
        unit_data['train'][TARGET_COL].values, ensemble_reg['oof_pred']
    )
    return ensemble_reg, per_model_results
```

#### 2.4.5 OOF CSV 저장 (rerun 내부)

```python
def _save_per_model_oof(per_model_clf, per_model_reg, clf_proba_mean,
                        sample_weight, pos_data, oof_dir, clf_models, reg_models):
    """
    모델별 die-level OOF CSV 7개 저장 (meta 1 + clf 3 + reg 3)
    """
    import os
    os.makedirs(oof_dir, exist_ok=True)

    # ── oof_meta.csv ──
    # 행: 1 die, 컬럼: ufs_serial, position, split, y_true, clf_proba_mean, lds_weight
    meta_rows = []
    for split in ['train', 'val', 'test']:
        for pos in sorted(pos_data.keys()):
            df = pos_data[pos][split]
            meta_rows.append(pd.DataFrame({
                KEY_COL:         df[KEY_COL].values,
                POSITION_COL:    pos,
                'split':         split,
                TARGET_COL:      df[TARGET_COL].values,
                'clf_proba_mean': clf_proba_mean[pos][f'{split}_proba'],
                'lds_weight':    (sample_weight if split == 'train' and
                                  sample_weight is not None else np.nan),
            }))
    meta = pd.concat(meta_rows, ignore_index=True)
    meta.to_csv(os.path.join(oof_dir, 'oof_meta.csv'), index=False)

    # ── oof_clf_{model}.csv × 3 ──
    for m in clf_models:
        rows = []
        for split in ['train', 'val', 'test']:
            for pos in sorted(per_model_clf[m].keys()):
                df = pos_data[pos][split]
                rows.append(pd.DataFrame({
                    KEY_COL:      df[KEY_COL].values,
                    POSITION_COL: pos,
                    'split':      split,
                    TARGET_COL:   df[TARGET_COL].values,
                    'clf_proba':  per_model_clf[m][pos][f'{split}_proba'],
                }))
        pd.concat(rows, ignore_index=True).to_csv(
            os.path.join(oof_dir, f'oof_clf_{m}.csv'), index=False)

    # ── oof_reg_{model}.csv × 3 ──
    # reg_level='position'이므로 per_model_reg는 die-level 예측을 모두 concat한 형태
    for m in reg_models:
        r = per_model_reg[m]
        # train=OOF, val/test=fold ensemble
        # row 순서는 unit_data['train'/'val'/'test']와 동일 (position concat)
        oof_df = pd.DataFrame({
            KEY_COL:        np.concatenate([
                unit_data_split[KEY_COL].values for unit_data_split in
                [unit_data['train'], unit_data['val'], unit_data['test']]]),
            POSITION_COL:   np.concatenate([
                unit_data_split[POSITION_COL].values for unit_data_split in
                [unit_data['train'], unit_data['val'], unit_data['test']]]),
            'split':        np.concatenate([
                ['train']*len(r['oof_pred']),
                ['val']*len(r['val_pred']),
                ['test']*len(r['test_pred'])]),
            TARGET_COL:     np.concatenate([
                unit_data_split[TARGET_COL].values for unit_data_split in
                [unit_data['train'], unit_data['val'], unit_data['test']]]),
            'reg_pred_die': np.concatenate([
                r['oof_pred'], r['val_pred'], r['test_pred']]),
        })
        oof_df.to_csv(os.path.join(oof_dir, f'oof_reg_{m}.csv'), index=False)
```

---

### 2.5 `add_rerun_to_study()` 신규 함수

[e2e_hpo.py](modules/e2e_hpo.py) 끝부분에 추가.

```python
def add_rerun_to_study(study, best_params, rerun_value, user_attrs=None):
    """
    Rerun 결과를 study에 FrozenTrial로 추가 (같은 DB에 append).

    Parameters
    ----------
    study : optuna.Study
        main study 객체
    best_params : dict
        main study의 best_params
    rerun_value : float
        rerun 결과 objective 값 (val RMSE)
    user_attrs : dict, optional
        {'is_rerun': True, 'rerun_mode': 'kfold', ...}

    Notes
    -----
    - best_params는 study의 기존 distribution과 동일 key set이어야 함
    - is_rerun=True user_attr로 나중에 필터 가능
    """
    import optuna
    if len(study.trials) == 0:
        raise ValueError("study has no trials yet; cannot extract distributions")

    # distributions는 기존 trial에서 추출
    distributions = study.trials[-1].distributions

    # best_params 중 study에 등록된 파라미터만 사용 (안전)
    valid_params = {k: v for k, v in best_params.items() if k in distributions}

    trial = optuna.trial.create_trial(
        params=valid_params,
        distributions={k: distributions[k] for k in valid_params},
        value=rerun_value,
        user_attrs=user_attrs or {'is_rerun': True},
    )
    study.add_trial(trial)
    print(f"[add_rerun_to_study] trial 추가 완료 (총 {len(study.trials)} trials, "
          f"value={rerun_value:.6f})")
```

**DB 영향**: SQLite에 INSERT 1행. Optuna viewer에서 `is_rerun=True`로 필터 가능.

---

### 2.6 Position 인코딩 — Ordinal + OHE 병행 (★ 신규)

#### 2.6.1 배경 — OHE vs Ordinal trade-off

`POSITION_COL`(1~4)은 unit 내 die 위치를 나타냄. 인코딩 방식별 특성:

| 방식 | 트리 모델 | 선형 모델 | 순서 정보 |
|---|---|---|---|
| Ordinal 정수 그대로 (1,2,3,4) | ✅ 분할로 효율 활용 | ❌ "3=1의 3배"로 오해 → 왜곡 | ✅ 보존 |
| OHE (p_1, p_2, p_3, p_4) | ⚠️ 분할 redundant (4컬럼) | ✅ 각 위치 독립 coef | ❌ 파괴 (p1과 p2 "인접" 무시) |
| Ordinal + OHE 병행 | ✅ 둘 다 활용 가능 | ✅ OHE로 처리 | ✅ Ordinal이 보존 |

→ **Ordinal + OHE 병행**이 트리·선형 혼합 앙상블(우리 케이스)에 가장 안전.

#### 2.6.2 구현 위치

`_prepare_unit_data` 또는 `_run_preprocessing`의 `pos_data` 빌드 단계에서 OHE 컬럼 4개를 추가:

```python
# _prepare_unit_data 내부 (reg_level='position' 경로)
for split_name in ["train", "val", "test"]:
    frames = []
    for pos in sorted(pos_data.keys()):
        df = pos_data[pos][split_name].copy()
        # ★ Position OHE 추가 (p_1, p_2, p_3, p_4)
        for p in [1, 2, 3, 4]:
            df[f'p_{p}'] = int(pos == p)
        if clf_result is not None:
            df["clf_proba_mean"] = clf_result[pos][f"{split_name}_proba"]
        frames.append(df)
    unit_data[split_name] = pd.concat(frames, ignore_index=True)

# unit_feat_cols에 Ordinal + OHE 모두 포함
unit_feat_cols = feat_cols + [POSITION_COL, 'p_1', 'p_2', 'p_3', 'p_4']
```

#### 2.6.3 모델별 활용 양상 (예상)

- **LGBM/ET**: `POSITION_COL` 분할이 주력, OHE는 보조 (둘 다 importance에 나타날 수 있음)
- **ElasticNet/LogReg-enet**: OHE 4컬럼이 각각 coef 가짐, `POSITION_COL` ordinal은 misleading일 수 있어 coef=0 수렴 예상

#### 2.6.4 run_wf_xy 메타 — 제외 확정

`run_wf_xy` 파싱(`run_id`, `wafer_no`, `die_x`, `die_y`)은 **2차에서 완전 제외**:

| 근거 | 결과 |
|---|---|
| EDA Phase 23: `radial_dist = sqrt(die_x² + die_y²)` 상관 | r = 0.006 (무상관) |
| EDA Phase 23: `is_edge` 이진 플래그 | ring별 불량률 차이 2%p 이내 |
| EDA Phase 24: NNR(인접 die 가중평균) 잔차 | 원본 대비 개선 feature 0개 (0.0%) |
| raw X 전체 스캔 | 공간 패턴 feature 8개 발견 → **수동 선별 6개 유지** (2026-04-16 웨이퍼맵 시각 검토) |
| run_id Target Encoding | 누수 위험 + fold마다 재계산 |

→ [meta_features.py](../2_preprocessing/meta_features.py) `run_meta_features` 호출 **안 함**. `_run_preprocessing`에서 파싱 단계 생략.

#### 공간 패턴 feature 수동 선별 결과 (2026-04-16)

사용자가 웨이퍼맵 시각화 결과 기반으로 수동 검토:

| Feature | 패턴 | 결정 | 근거 |
|---|---|---|---|
| X708 | Radial gradient (center-edge) | ✅ 유지 | 대표 radial 정보 |
| X1059 | Y축 수평 밴드 | ✅ 유지 | Y 좌표 정보 (대표) |
| X1073 | 4분면 sector | ✅ 유지 | Angle 정보 (독립) |
| X1075 | 복합 블록 | ✅ 유지 | 복합 공간 패턴 |
| X1076 | Y축 선명 스트립 | ✅ 유지 | Y 좌표 보조 |
| X1077 | X축 수직 밴드 | ✅ 유지 | X 좌표 정보 (독립) |
| **X1056** | Ring 패턴 (타원 고리) | ❌ **제거** | 기여 모호 + cleaning에서 상관 제거될 여지 |
| **X1072** | Radial gradient | ❌ **제거** | X708과 중복 (거의 동일 패턴) |

→ **`EXCLUDE_COLS`에 X1056, X1072 추가** (총 37개 제외).

---

## 3. 노트북 셀 구성 (11개 셀)

### 3.1 셀 목록

| # | 제목 | 역할 | baseline 대비 |
|---|---|---|---|
| 1 | 환경 + 데이터 로드 | Colab/Local 자동, gdown, utils/modules import | 같음 + 신규 함수 import |
| 2 | 실험 설정 | EXP_ID, USER, e2e_params, rerun_params, weight_params | 3-모델 list, calibration, A4, 가중치 설정 추가 |
| 3 | PP candidate 축소 | 1차 결과 기반 사용자 영역 | 신규 (1차 결과 viewer 사용) |
| 4 | 타겟 변환 + 시각화 | log1p + clip y extreme + 분포 plot | 같음 |
| 5 | Main Optuna | 단일 study, trial=6 모델 | list 모델, calibration, A4 |
| 6 | Rerun + OOF + DB | best params로 kfold=5 재학습 + OOF CSV + add_rerun_to_study | `save_per_model_oof=True`, `add_rerun_to_study` 호출 |
| 7 | ★ 진단 시각화 | feature 제거 후보 (규제/importance 기반) | 신규 |
| 8 | 포지션 가중치 sub-study | per-model Optuna, Dirichlet 정규화 | 신규 |
| 9 | 모델 가중치 그리드 | simplex step=0.025, ~861 조합 | 신규 |
| 10 | 최종 ensemble + 로그 | final_pred_val/test + log_experiment + Drive upload | 앙상블 경로 확장 |
| 11 | ★ 시각화 | 가중치 + 예측 vs 실측 + RMSE 비교 | 신규 |

### 3.2 셀 코드 — 요약

각 셀의 핵심 코드는 본 설계의 부록(섹션 7)에 분리 기재. 본 섹션은 상위 흐름만 기록.

---

## 4. 진단·시각화 셀 상세

### 4.1 셀 7 — Feature 제거 후보 진단

#### 목적

3-모델의 importance / coefficient를 교차 참조해서 "모두 무관 판정한 feature" 후보를 뽑고 시각화.

#### 출력

1. **ElasticNet `coef_=0` feature 리스트** (규제로 빠진 feature)
2. **LGBM `feature_importance(gain)` bottom K**
3. **ET `feature_importances_` bottom K**
4. **3-모델 교집합**: "모두 무관이라고 본" feature 후보
5. **시각화**:
   - Top-20 importance (3 모델 각각, horizontal bar)
   - coef/importance 분포 histogram (3 모델)
   - Venn diagram — 무관 판정 feature 교집합

#### 코드 스켈레톤

```python
# ── 1) rerun 결과에서 모델별 importance 추출 ──
# rerun은 fold별 모델 list를 반환함. 각 fold 평균 importance 사용.
# 주의: ElasticNet coef_는 1D (n_features,), LogReg coef_는 2D (1, n_features)
#       → .ravel()로 1D 강제해서 shape 맞춤
def _model_imp(models, kind):
    if kind == 'tree':
        return np.mean([m.feature_importances_ for m in models], axis=0)
    elif kind == 'linear':
        # ravel로 1D 강제 (LogReg 2D 케이스 대응)
        return np.mean([np.abs(m.coef_).ravel() for m in models], axis=0)
    else:
        raise ValueError(kind)

lgbm_imp  = _model_imp(final['per_model_fold_models']['lgbm'], 'tree')
et_imp    = _model_imp(final['per_model_fold_models']['et'],   'tree')
enet_coef = _model_imp(final['per_model_fold_models']['enet'], 'linear')

feat_names = final['selected_cols']   # rerun에서 실제 사용된 컬럼
df_imp = pd.DataFrame({
    'feature': feat_names,
    'lgbm_gain':  lgbm_imp,
    'et_impurity': et_imp,
    'enet_abs_coef': enet_coef,
})

# ── 2) 3-모델 "무관" 판정 기준 ──
threshold_lgbm = df_imp['lgbm_gain'].quantile(0.10)     # 하위 10%
threshold_et   = df_imp['et_impurity'].quantile(0.10)
mask_enet_zero = (df_imp['enet_abs_coef'] < 1e-8)       # coef=0

df_imp['lgbm_weak'] = df_imp['lgbm_gain']   < threshold_lgbm
df_imp['et_weak']   = df_imp['et_impurity'] < threshold_et
df_imp['enet_zero'] = mask_enet_zero

# ── 3) 3-모델 교집합 ──
intersection_cols = df_imp[
    df_imp['lgbm_weak'] & df_imp['et_weak'] & df_imp['enet_zero']
]['feature'].tolist()
print(f"3-모델 모두 무관 판정: {len(intersection_cols)}개 feature")
print(intersection_cols[:20])

# ── 4) 시각화 ──
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 4-1) Top-20 importance (bar)
for ax, (col, title) in zip(
    axes[0], [('lgbm_gain', 'LGBM gain'),
              ('et_impurity', 'ET impurity'),
              ('enet_abs_coef', 'ElasticNet |coef|')]):
    top = df_imp.nlargest(20, col)
    ax.barh(range(20), top[col].values)
    ax.set_yticks(range(20))
    ax.set_yticklabels(top['feature'].values, fontsize=8)
    ax.set_title(f'Top 20: {title}')
    ax.invert_yaxis()

# 4-2) Importance 분포 histogram
for ax, (col, title) in zip(
    axes[1], [('lgbm_gain', 'LGBM gain'),
              ('et_impurity', 'ET impurity'),
              ('enet_abs_coef', 'ENet |coef|')]):
    ax.hist(df_imp[col], bins=60)
    ax.set_title(f'분포: {title}')
    ax.set_yscale('log')
    ax.axvline(df_imp[col].quantile(0.10), color='red', linestyle='--',
               label='하위 10%')
    ax.legend()

plt.tight_layout()
plt.show()

# ── 5) 저장 (intersection 후보 CSV) ──
if SAVE_OUTPUTS:
    df_imp.to_csv(os.path.join(OOF_DIR, 'feature_importance.csv'), index=False)
    pd.Series(intersection_cols).to_csv(
        os.path.join(OOF_DIR, 'feature_remove_candidates.csv'),
        index=False, header=['feature'])
```

#### 주의

- 이 셀은 **진단만** 수행, 실제 feature 제거는 3차에서 판단
- `intersection_cols`가 너무 많으면 (예: 500+) 기준 완화, 너무 적으면 강화
- ElasticNet `coef_=0`은 `l1_ratio` 높고 alpha 큰 경우에만 효과적 → best l1_ratio 확인 후 해석

### 4.2 셀 11 — 가중치 + 예측 시각화

#### 출력

1. **포지션 가중치 bar (모델별)** — 4개 x축, 모델 3개 subplot
2. **모델 가중치 simplex scatter** — ternary plot or 2D projection
3. **예측 vs 실측 scatter** (val split)
4. **Residual plot** (val split)
5. **RMSE 비교 bar** — 단독 모델 3개 vs 앙상블 단순평균 vs 앙상블 최종 가중치

#### 코드 스켈레톤

```python
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# ── 1) 포지션 가중치 (모델별 bar) ──
for ax, m in zip(axes[0], e2e_params['reg_models']):
    w = position_weight_results[m]['best_w_pos']
    ax.bar(['P1', 'P2', 'P3', 'P4'], w,
           color=['C0', 'C1', 'C2', 'C3'])
    ax.axhline(0.25, color='gray', linestyle='--', label='균등 0.25')
    ax.set_title(f'{m}: best w_pos')
    ax.set_ylim(0, 0.5)
    ax.set_ylabel('weight')
    ax.legend()
    for i, wi in enumerate(w):
        ax.text(i, wi + 0.01, f'{wi:.3f}', ha='center', fontsize=9)

# ── 2) 모델 가중치 그리드 히트맵 (w_lgbm, w_et 기준, w_enet=implicit) ──
df_grid = pd.read_csv(os.path.join(OOF_DIR, 'model_weights_grid.csv'))
# 2D pivot: w_lgbm × w_et → val_rmse
pivot = df_grid.pivot_table(
    index='w_lgbm', columns='w_et', values='val_rmse', aggfunc='min')
im = axes[1, 0].imshow(pivot.values, cmap='viridis_r', origin='lower',
                       extent=[0, 1, 0, 1], aspect='auto')
axes[1, 0].set_xlabel('w_et')
axes[1, 0].set_ylabel('w_lgbm')
axes[1, 0].set_title('Model weight grid (val RMSE, lower=better)')
plt.colorbar(im, ax=axes[1, 0])
# best 점 마킹
best = df_grid.iloc[df_grid['val_rmse'].idxmin()]
axes[1, 0].scatter([best['w_et']], [best['w_lgbm']], color='red', s=100,
                   marker='x', label=f'best ({best["val_rmse"]:.6f})')
axes[1, 0].legend()

# ── 3) 예측 vs 실측 scatter (val) + Pearson·Spearman 병기 (★ 2차 신규) ──
from scipy.stats import spearmanr, pearsonr
final_val = pd.read_csv(os.path.join(OOF_DIR, 'final_pred_val.csv'))
pearson_r, _  = pearsonr(final_val['y_true'], final_val['y_pred'])
spearman_r, _ = spearmanr(final_val['y_true'], final_val['y_pred'])

# Y>0 서브셋에서도 따로 계산 (극단값 under-predict 진단, 논문 8-3)
pos_mask = final_val['y_true'] > 0
spearman_pos, _ = spearmanr(final_val.loc[pos_mask, 'y_true'],
                             final_val.loc[pos_mask, 'y_pred'])

ax = axes[1, 1]
ax.scatter(final_val['y_true'], final_val['y_pred'], alpha=0.3, s=8)
ax.plot([0, final_val['y_true'].max()], [0, final_val['y_true'].max()],
        color='red', linestyle='--', label='y=x')
ax.set_xlabel('y_true')
ax.set_ylabel('y_pred')
ax.set_title(f'Val Prediction vs True')
ax.text(0.05, 0.95,
        f'RMSE:          {best_val_rmse:.6f}\n'
        f'Pearson r:     {pearson_r:.4f}\n'
        f'Spearman ρ:    {spearman_r:.4f}\n'
        f'Spearman (Y>0):{spearman_pos:.4f}',
        transform=ax.transAxes, va='top', fontsize=9, family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
ax.legend()

# ── 4) Residual plot ──
residuals = final_val['y_pred'] - final_val['y_true']
ax = axes[0, 1]
ax.scatter(final_val['y_true'], residuals, alpha=0.3, s=8)
ax.axhline(0, color='red', linestyle='--')
ax.set_xlabel('y_true')
ax.set_ylabel('residual (pred - true)')
ax.set_title('Val Residual')

# ── 5) RMSE 비교 bar ──
ax = axes[0, 2]
labels = ['LGBM단독', 'ET단독', 'ENet단독', '앙상블 단순평균', '앙상블 가중']
values = [
    final['per_model_val_rmse']['lgbm'],
    final['per_model_val_rmse']['et'],
    final['per_model_val_rmse']['enet'],
    final['val_rmse'],                      # 단순 평균
    best_val_rmse,                           # cell 8 최종
]
ax.bar(labels, values, color=['C0','C1','C2','gray','green'])
for i, v in enumerate(values):
    ax.text(i, v + 1e-5, f'{v:.6f}', ha='center', fontsize=9)
ax.set_ylabel('val RMSE')
ax.set_title('RMSE 비교')
ax.tick_params(axis='x', rotation=20)

# ── 6) Feature importance mini (bonus) ──
ax = axes[1, 2]
top10 = df_imp.nlargest(10, 'lgbm_gain')
ax.barh(range(10), top10['lgbm_gain'].values)
ax.set_yticks(range(10))
ax.set_yticklabels(top10['feature'].values, fontsize=8)
ax.set_title('LGBM Top-10 feature')
ax.invert_yaxis()

plt.tight_layout()
plt.show()
```

---

## 5. 데이터 플로우 다이어그램

```
┌─────────────────────────────────────────────────────────────┐
│ [원본 데이터]                                               │
│   xs (die-level, 175K rows × 1,091 cols)                   │
│   ys (unit-level, 43K units)                                │
└─────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────┐
│ [Main Optuna Trial 1회 — 옵션 A]                            │
│                                                             │
│  ┌─ PP (cached, LRU) ──────────────────────────────────┐   │
│  │ cleaning → outlier → IsoForest → hybrid_scale → LDS │   │
│  │   → pos_data (die-level by position)                │   │
│  │   → sample_weight (train, unit-level)               │   │
│  └─────────────────────────────────────────────────────┘   │
│              ↓                                              │
│  ┌─ 분류 3-모델 ───────────────────────────────────────┐   │
│  │ LGBM_clf  → OOF proba ─┐                            │   │
│  │ ET_clf    → OOF proba ─┤→ Isotonic Cal (LGBM/ET)    │   │
│  │ LogReg    → OOF proba ─┘     ↓                      │   │
│  │                              soft voting             │   │
│  │                                 ↓                   │   │
│  │                          clf_proba_mean             │   │
│  └─────────────────────────────────────────────────────┘   │
│              ↓                                              │
│  ┌─ 회귀 3-모델 (LDS + A4, Y>0 only) ────────────────────┐   │
│  │ 학습 데이터: Y>0 서브셋만 (Hybrid Two-Stage 표준)     │   │
│  │ input: [X | clf_proba_mean], sample_weight=LDS       │   │
│  │ LGBM_reg  (objective: regression/poisson/tweedie)    │   │
│  │ ET_reg    (MSE 고정) → die-level OOF                 │   │
│  │ ENet_reg  (MSE 고정) → die-level OOF                 │   │
│  │              ↓                                      │   │
│  │    mean(3 die-level OOFs)                           │   │
│  │              ↓                                      │   │
│  │    unit-level (position mean)                       │   │
│  └─────────────────────────────────────────────────────┘   │
│              ↓                                              │
│  final = clf_proba_unit × R_unit_mean                       │
│  → postprocess(clip 0) → zero_clip                          │
│  → OOF RMSE = objective                                     │
└─────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────┐
│ [Rerun (kfold=5) + die-level OOF CSV 저장]                 │
│   best (PP + HP) 기반 재학습                                │
│   → oof_clf_{lgbm,et,logreg_enet}.csv  (die-level)         │
│   → oof_reg_{lgbm,et,enet}.csv         (die-level)         │
│   → oof_meta.csv (ufs_serial, position, split, y_true,     │
│                   clf_proba_mean, lds_weight)              │
│   → add_rerun_to_study(study, best_params, val_rmse)       │
└─────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────┐
│ [포지션 가중치 sub-study (per-model Optuna)]               │
│   입력: die-level OOF (모델별)                              │
│   trial: w1~w4 Dirichlet 정규화 → unit pred → RMSE          │
│   150 trials × 3 모델 = 450 trials (~1초)                   │
│   → position_weight_results[model]['best_w_pos']            │
│   → position_weights_substudy.csv                           │
└─────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────┐
│ [모델 가중치 그리드서치]                                    │
│   입력: unit-level R_m (per-model best w_pos 적용)          │
│   simplex step=0.025 → ~861 조합                            │
│   final = P × R_combined → RMSE                             │
│   → best (w_lgbm, w_et, w_enet)                             │
│   → model_weights_grid.csv                                  │
└─────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────┐
│ [최종 Ensemble]                                             │
│   unit_pred = clf_proba × Σ_m w_model[m] ·                  │
│               Σ_p w_pos[m,p] · die_pred[m,p]                │
│   → final_pred_val.csv, final_pred_test.csv                 │
│   → combined_best.json (w_pos, w_model 묶음)                │
│   → log_experiment + Drive upload                           │
└─────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────┐
│ [시각화 셀 11]                                              │
│   - 포지션 가중치 bar (모델별)                              │
│   - 모델 가중치 grid heatmap                                │
│   - 예측 vs 실측 scatter + residual                         │
│   - RMSE 비교 bar (단독 vs 앙상블)                          │
│   - feature importance Top-10                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. zip 재배포 가이드

### 6.1 재생성 필요한 zip

| zip | 포함 파일 | 변경점 |
|---|---|---|
| `preprocessing.zip` | `cleaning.py`, `outlier.py`, `scaling.py`, **`sample_weight.py` (신규)** | scaling 확장 + outlier 확장 + sample_weight 신규 |
| `modules.zip` | `model_zoo.py`, `search_space.py`, `e2e_hpo.py`, `aggregate.py`, `ensemble.py`, `feature_select.py`, `threshold.py`, `__init__.py` | model_zoo + search_space + e2e_hpo 갱신 |
| `code.zip` | `setup.py`, `requirements.txt`, `utils/` | 변경 없음 (scipy는 이미 포함 가정) |
| `dataset.zip` | CSV 4개 | 변경 없음 |

### 6.2 재생성 절차 (로컬)

```bash
# preprocessing.zip
cd "c:/Users/Dell5371/Desktop/기업연계프로젝트"
zip -r "2_preprocessing/preprocessing.zip" \
    2_preprocessing/cleaning.py \
    2_preprocessing/outlier.py \
    2_preprocessing/scaling.py \
    2_preprocessing/sample_weight.py

# modules.zip
zip -r "3_modeling/modules/modules.zip" \
    3_modeling/modules/*.py
```

### 6.3 Drive 업로드

- 기존 파일 ID 유지 (덮어쓰기):
  - `preprocessing.zip` ID: `1Rh0ByOS4Gama8XHuvY7KkOHo278H9YLr`
  - `modules.zip` ID: `1Vrn5LBl611rWbag7d09LZH68_lfpu6wP`
- Drive UI에서 "버전 관리" → 새 파일 업로드 (같은 ID 유지)
- Colab에서 다시 `gdown`하면 신규 버전이 받아짐

### 6.4 requirements.txt 확인

추가 필요 없음 (이미 포함):
- `scipy`: LDS gaussian_kde
- `sklearn`: QuantileTransformer, IsolationForest, IsotonicRegression, ElasticNet, LogisticRegression

---

## 7. 노트북 셀 코드 상세 (부록)

> 본 섹션은 대화에서 확정한 셀 1~11의 완성 코드를 모아 기록. 실제 구현 시 `ensemble_2nd.ipynb`로 복사.

### Cell 1 — 환경 + 데이터 로드

베이스라인과 동일. 추가 import:
- `from modules.e2e_hpo import add_rerun_to_study`
- `from modules.search_space import PP_ISO_ANOMALY_CANDIDATES, PP_LDS_CANDIDATES`

### Cell 2 — 실험 설정

```python
EXP_ID = '3-200-001'
EXP_TYPE = 'ensemble + HPO'
EXP_MEMO = '2차 funnel: 분류·회귀 3-모델 앙상블, hybrid_scale 외부 1회, LDS, IsoForest score, FS 교집합'

EVAL_VAL, EVAL_TEST = True, True
SAVE_OUTPUTS = True
USER = 'jh'
CSV_GDRIVE_ID = ''

DB_PATH = os.path.join(OUTPUT_DIR, 'experiments', f'optuna_{USER}_{EXP_ID}.db')
OOF_DIR = os.path.join(OUTPUT_DIR, 'oof', EXP_ID)
os.makedirs(OOF_DIR, exist_ok=True)

LABEL_COL = 'label_bin'
TARGET_TRANSFORM = 'log1p'
CLIP_Y_EXTREME = True
sampling_params = dict(use_sampling=False, sample_frac=1.0)

# 웨이퍼맵 기반 사전 제외 (총 37개)
# - 35개: 1_eda/wafer_map_image/★분류완료/1_꼭빼야함/ 수동 분류 결과
# - 2개 (★ 2026-04-16 추가): 공간 패턴 feature 수동 선별 결과
#   · X1056: Ring 패턴, 기여 모호 → 제거
#   · X1072: X708과 radial 중복 (거의 동일) → 제거
# cleaning/outlier/scaling 이전에 적용 — `feat_cols`에서 제거한 뒤 파이프라인 진입.
# Colab 환경에서 wafer_map_image 폴더 미동기화 가능성 → 하드코딩으로 재현성 확보.
EXCLUDE_COLS = [
    'X124', 'X300', 'X301',
    'X443', 'X444', 'X447', 'X448', 'X449',
    'X451', 'X452', 'X455', 'X456', 'X457', 'X458', 'X459', 'X460',
    'X463', 'X464',
    'X503', 'X504', 'X505', 'X506',
    'X517',
    'X658', 'X659', 'X671', 'X672',
    'X674', 'X675', 'X676', 'X677',
    'X680', 'X687',
    'X1056',   # ★ 2차 추가: Ring 패턴, 기여 모호
    'X1072',   # ★ 2차 추가: X708과 radial 중복
    'X1074', 'X1078',
]

pipeline_config = dict(
    input_level='die',
    run_clf=True, clf_output='proba', clf_filter=False, clf_optuna=True,
    run_fs=False, fs_optuna=False,
    reg_level='position', reg_optuna=True,
    zero_clip=True,
)

e2e_params = dict(
    clf_models=['lgbm', 'et', 'logreg_enet'],
    reg_models=['lgbm', 'et', 'enet'],
    n_trials=150,
    n_folds=3,
    clf_early_stop=50, reg_early_stop=50,
    imbalance_method='scale_pos_weight',
    top_k_range=(50, 500), top_k_fixed=200,
    clf_filter_threshold_range=(0.05, 0.5),
    clf_filter_threshold_fixed=0.5,
    zero_clip_threshold_range=(0.002, 0.002),
    zero_clip_threshold_fixed=0.0,
    calibration=dict(method='isotonic', cv=3, models=['lgbm', 'et']),
    add_clf_proba_to_reg=True,
    clf_fixed={}, reg_fixed={},
    pp_cache_size=10,
)

rerun_params = dict(
    mode='kfold', n_folds=5, es_holdout=0.1,
    add_to_study=True,
)

model_weight_params = dict(
    step=0.025,
    save_path='model_weights_grid.csv',
    save_top_k=20,
)

position_weight_substudy = dict(
    n_trials_per_model=150,
    weight_range=(0.15, 0.35),
    weight_step=0.01,
    normalize='dirichlet',
    per_model=True,
    save_path='position_weights_substudy.csv',
    save_top_k=20,
)

ensemble_save = dict(
    save_combined='combined_best.json',
    save_final_val='final_pred_val.csv',
    save_final_test='final_pred_test.csv',
)

check_exp_id(EXP_ID)
```

### Cell 3 — PP candidate 축소 (사용자 영역)

1차 `optuna_merged.db` 참고하여 `PP_CLEAN_CANDIDATES` / `PP_OUTLIER_CANDIDATES` 축소. 사용자가 직접 편집.

### Cell 4 — 타겟 변환 + 시각화

baseline cell 4 그대로 (log1p + clip_y_extreme + 2×3 분포 히스토그램).

### Cell 5 — Main Optuna 실행 (warm_start 명시적 전달)

```python
result = run_e2e_optimization_with_pp(
    xs=xs, xs_dict=xs_dict, ys=ys_input, feat_cols=feat_cols,
    pipeline_config=pipeline_config,
    label_col=LABEL_COL,
    use_sampling=sampling_params['use_sampling'],
    sample_frac=sampling_params['sample_frac'],
    exclude_cols=EXCLUDE_COLS,
    **e2e_params,                       # clf_models/reg_models/calibration/add_clf_proba_to_reg 포함
    target_transform_fn=target_transform_fn,
    target_inverse_fn=target_inverse_fn,
    exp_id=EXP_ID,
    db_path=DB_PATH,
    study_user_attrs=study_user_attrs,
    # ★ warm_start 명시적 전달 (끊긴 study 이어서)
    warm_start_top_k=30,
    warm_start_enabled=True,
)
study = result['study']
best_params = result['best_params']
```

### Cell 6 — Rerun + die-level OOF + add_rerun_to_study

섹션 2.3.2 시그니처 + 반환 dict 구조 참조. 저장되는 파일: 섹션 3.1 표 참조.

### Cell 7 — 진단 시각화

섹션 4.1 코드 그대로. 주의: `_model_imp` 헬퍼로 LogReg `coef_` 2D shape 대응.

### Cell 8 — 포지션 가중치 sub-study

```python
# OOF 로드 + pre-pivot (벡터화 가속)
# per-model Optuna sub-study (Dirichlet 정규화)
# 섹션 2.4 흐름 + 대화 확정 코드
# 저장: position_weights_substudy.csv (모델별 top_k=20)
```

### Cell 9 — 모델 가중치 그리드서치

```python
# simplex_grid(n_dim=3, step=0.025) ≈ 861 조합
# Two-Stage 곱 (P × R_combined) + postprocess + zero_clip
# 저장: model_weights_grid.csv (top_k=20)
```

### Cell 10 — 최종 ensemble + log_experiment (★ 스키마 C)

```python
# ── best (w_pos, w_model) 적용 → 최종 예측 ──
final_val_df = pd.DataFrame({
    # ── 스키마 C (풍부) ──
    KEY_COL:              unit_ids_val,
    'y_true':             y_val_unit,
    'y_pred':             final_pred_val,                       # 최종 앙상블 예측
    'clf_proba_unit':     clf_proba_unit_val,                   # P (Two-Stage)
    'R_combined_unit':    R_combined_val,                       # R 가중 결합
    'R_lgbm_unit':        unit_R_preds[('lgbm','val')],         # 모델별 R (best w_pos 적용)
    'R_et_unit':          unit_R_preds[('et','val')],
    'R_enet_unit':        unit_R_preds[('enet','val')],
    'clf_proba_lgbm':     proba_unit_val_by_model['lgbm'],      # 모델별 P (calibrated)
    'clf_proba_et':       proba_unit_val_by_model['et'],
    'clf_proba_logreg_enet': proba_unit_val_by_model['logreg_enet'],
})
final_val_df.to_csv(os.path.join(OOF_DIR, 'final_pred_val.csv'), index=False)

# test도 동일 schema
final_test_df = pd.DataFrame({...})   # 동일 컬럼
final_test_df.to_csv(os.path.join(OOF_DIR, 'final_pred_test.csv'), index=False)

# ── combined_best.json ──
combined_best = {
    'best_w_pos':    {m: position_weight_results[m]['best_w_pos'] for m in reg_models},
    'best_w_model':  dict(zip(reg_models, best_w_model)),
    'val_rmse':      best_val_rmse,
    'test_rmse':     best_test_rmse,
    'scaler_path':   'hybrid_scaler.pkl',     # pickle 저장 필요
    'exp_id':        EXP_ID,
    'saved_at':      datetime.now().isoformat(),
}
with open(os.path.join(OOF_DIR, ensemble_save['save_combined']), 'w') as f:
    json.dump(combined_best, f, indent=2)

# HybridScaler 객체 pickle 저장 (대시보드 재사용용)
import joblib
joblib.dump(final['scaler'], os.path.join(OOF_DIR, 'hybrid_scaler.pkl'))

# log_experiment + Drive upload
log_experiment(
    exp_id=EXP_ID, exp_type=EXP_TYPE,
    best_model='ensemble(lgbm+et+enet)',
    val_rmse=best_val_rmse, test_rmse=best_test_rmse,
    n_features=len(final['selected_cols']),
    memo=EXP_MEMO, user=USER,
    n_trials=e2e_params['n_trials'],
    csv_gdrive_id=CSV_GDRIVE_ID,
)
```

#### final_pred_{val,test}.csv 스키마 (★ 사용자 결정: 최종 베스트만 C)

| 컬럼 | 설명 |
|---|---|
| `ufs_serial` | unit 식별자 |
| `y_true` | 실측 health |
| `y_pred` | 최종 앙상블 예측 (P × R_combined) |
| `clf_proba_unit` | Two-Stage의 P (3 분류기 soft-voted, unit-level) |
| `R_combined_unit` | Two-Stage의 R (모델 가중치로 결합, unit-level) |
| `R_lgbm_unit` | LGBM 회귀 unit-level 예측 (best w_pos 적용됨) |
| `R_et_unit` | ET 회귀 동 |
| `R_enet_unit` | ElasticNet 회귀 동 |
| `clf_proba_lgbm` | LGBM 분류 unit-level proba (Isotonic calibrated) |
| `clf_proba_et` | ET 분류 동 |
| `clf_proba_logreg_enet` | LogReg-enet 분류 unit-level proba |

→ 3차 재사용 시 재학습 없이 가중치 재튜닝 가능.

### Cell 11 — 시각화

섹션 4.2 참조.

---

## 8. 실행 시간 추정

### 8.1 Main Optuna (cell 5)

| 항목 | 시간 |
|---|---|
| 1 trial당 | 3 clf + 3 reg × 3-fold = ~3분 |
| n_trials=150 | ~7.5시간 |
| warm_start 누적 가능 | 끊기면 이어서 |

### 8.2 Rerun (cell 6)

| 항목 | 시간 |
|---|---|
| kfold=5 × 3 clf × 3 reg | ~20분 |

### 8.3 후처리 (cell 7~9)

| 항목 | 시간 |
|---|---|
| 포지션 sub-study (450 trials) | ~1초 |
| 모델 가중치 그리드 (861 조합) | ~1초 |
| 시각화 | ~5초 |

### 8.4 전체

**약 8시간** (n_trials=150 기준). Colab에서 중간 끊김 가능 → warm_start로 이어짐.

---

## 9. 트러블슈팅

### 9.1 "saga solver 수렴 안 됨"

LogReg(elasticnet)의 saga는 scaled data + max_iter=4000 이상 권장.
- hybrid_scale 외부 1회 적용이 보장되면 수렴 문제 감소
- 안 되면 `tol=1e-2`로 완화

### 9.2 "ElasticNet 학습 중 warning"

`ConvergenceWarning` 많으면 `max_iter` 증가 (예: 10000). `filterwarnings`로 숨김 가능하지만 실제 수렴 확인.

### 9.3 "IsoForest 메모리 폭증"

175K × 960 feature → 메모리 부담. `max_samples=10000` 설정으로 제한 가능. contamination='auto' 유지.

### 9.4 "LDS weight 극단값"

`max_weight=10` 기본이지만 y>0 분포가 너무 편중되면 일부 sample weight가 상한에 몰림. `max_weight=5`로 조절. 또는 `sigma`를 크게 (smoothing 강화).

### 9.5 "add_rerun_to_study 시 distribution 불일치"

main study의 trial이 하나 이상 완료된 후에만 호출. best_params에 없는 key가 distribution에 있어도 무시됨 (안전 처리).

### 9.6 "OOF CSV 용량 큼"

175K × 3 모델 × 클래스 1개 ≈ 5MB/파일. 7 파일(meta 1 + clf 3 + reg 3) = ~35MB. Drive 부담 없음. 단 로컬 storage 주의.

---

## 10. 참고

- [CLAUDE.md](../CLAUDE.md) L515~564 (2차 funnel 계획)
- [strategy_2nd_preprocessing.md](strategy_2nd_preprocessing.md) (전처리 모듈 설계 — 본 문서의 전제)
- [미반영_항목_검토.md](../99_학습자료/스터디자료/미반영_항목_검토.md) (Phase 1/2 작업 근거)
- [baseline.ipynb](baseline.ipynb) (1차 funnel 참조)
- [논문요약.md](../99_학습자료/스터디자료/paper/논문요약.md) 2-1 (A4), 2-2 (Calibration), 8-1 (LDS), 9-2/9-3 (Stacking/GEM)

---

## 11. 구현 체크리스트

### 11.1 모듈 패치

- [ ] `model_zoo.py`: `enet`, `logreg_enet` 등록 + `fit_model` sample_weight 지원
- [ ] `search_space.py`: `et_space`, `logreg_enet_space`, `enet_space` + `SEARCH_SPACES`
- [ ] `search_space.py`: `lgbm_space(prefix='reg_')`에 **objective 축** 추가 (`regression`/`poisson`/`tweedie_1.2`/`tweedie_1.5`)
- [ ] `e2e_hpo.py`: `clf_models`/`reg_models` list 지원, `calibration`, `add_clf_proba_to_reg`
- [ ] `e2e_hpo.py`: `_run_clf_oof_multi`, `_apply_isotonic_calibration`, `_run_reg_oof_multi`
- [ ] `e2e_hpo.py`: `_save_per_model_oof` 신규
- [ ] `e2e_hpo.py`: `rerun_best_trial_with_pp`에 `save_per_model_oof`/`oof_dir` 인자
- [ ] `e2e_hpo.py`: rerun 반환 dict에 `per_model_fold_models`, `per_model_clf_fold_models`, `selected_cols`, `scaler` 키 추가
- [ ] `e2e_hpo.py`: `add_rerun_to_study` 신규
- [ ] `e2e_hpo.py`: `_run_preprocessing` 반환 4-tuple로 확장 (+ sample_weight, scaler)
- [ ] `e2e_hpo.py`: `_prepare_unit_data`에 **Position OHE 4컬럼 추가** (`p_1, p_2, p_3, p_4`) + `unit_feat_cols`에 Ordinal + OHE 모두 포함
- [ ] `e2e_hpo.py`: run_wf_xy 파싱 / `run_meta_features` 호출 **안 함** (제외 확정)
- [ ] 전처리 모듈 (T1.*): `strategy_2nd_preprocessing.md` 참조 — `HybridScaler` 클래스 + `compute_lds_weights(expand_to_die=True)`

### 11.2 노트북

- [ ] `ensemble_2nd.ipynb` 생성
- [ ] Cell 1~11 구성 (시각화 셀 2개 포함)
- [ ] smoke test (n_trials=2, n_folds=2)
- [ ] 정식 실행

### 11.3 zip 재배포

- [ ] `preprocessing.zip` 재생성 + Drive 업로드
- [ ] `modules.zip` 재생성 + Drive 업로드
- [ ] Colab에서 재다운로드 동작 확인

### 11.4 실행 후 로깅

- [ ] `log_experiment` 호출 (`exp_type='ensemble + HPO'`)
- [ ] `experiments.csv` Drive 업로드
- [ ] OOF 파일 백업 (Drive 또는 로컬)