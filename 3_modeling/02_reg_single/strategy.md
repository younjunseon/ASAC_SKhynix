# 02_reg_single 전략 — 단일 회귀 모델 5종

> 공통 규칙은 [strategy_common.md](../strategy_common.md). 본 문서는 02_reg_single 전용 사항만.

---

## 1. 목표

5개 회귀 모델 각각 evidence-driven HPO + 후처리. 모델별 노트북 1개씩.

| 노트북 | 모델 | 카테고리 | search 방식 |
|---|---|---|---|
| `lgbm.ipynb` | LightGBM | 부스팅 트리 | Evidence-driven Wide |
| `xgb.ipynb` | XGBoost | 부스팅 트리 | Anchor + Narrow ±50% |
| `catboost.ipynb` | CatBoost | 부스팅 트리 | Evidence-driven Wide |
| `et.ipynb` | ExtraTrees | 베깅 트리 | Evidence-driven Wide |
| `enet.ipynb` | ElasticNet | 선형 | Evidence-driven Wide + PP/scaling joint |

원본 [02_reg_only.ipynb](../../3_modeling_이전자료/final/02_reg_only.ipynb)의 `MODEL_NAME` 스위치 구조를 모델별로 분리.

---

## 2. 원본 파일 매핑

| 신규 (`3_modeling/02_reg_single/`) | 원본 (`3_modeling_이전자료/final/`) | 1차 산출물 (`4_output_이전자료/final/reg_only/`) |
|---|---|---|
| `lgbm.ipynb` | `02_reg_only.ipynb` | `lgbm/best_params.json` |
| `xgb.ipynb` | 동일 | (없음 — 1차 미실행) |
| `catboost.ipynb` | 동일 | `catboost/best_params.json` |
| `et.ipynb` | 동일 | `et/best_params.json` |
| `enet.ipynb` | 동일 | `enet/best_params.json` |

**모듈 의존** ([strategy_common.md §22](../strategy_common.md)):
- 전처리: `2_preprocessing/{cleaning, outlier, scaling, ...}.py` (직접 import)
- 모델링: `3_modeling/modules/{models, hpo, postprocess, preprocess}.py`

---

## 3. PP 정책 — 모델별 분기

[strategy_common.md §1·§3](../strategy_common.md):

| 모델 | PP 처리 | scaling |
|---|---|---|
| lgbm / xgb / catboost / et | **PP_FIXED 고정** (§1) | none |
| enet | **PP+scaling+HP joint Optuna** (§3) | Standard / Robust / Yeo-Johnson / Quantile / Hybrid |

> 사용자 결정: 트리 4종 PP 영향 작아 PP_FIXED 그대로. enet만 joint.

---

## 4. Target Transform — log1p 기본 + tweedie 시 자동 OFF

| 손실함수 | log1p 적용 |
|---|---|
| RMSE / squarederror / regression / poisson | **ON** (학습 시 log1p, 평가/저장 시 expm1) |
| **tweedie / Tweedie / reg:tweedie** | **OFF** (Tweedie 분포가 right-skew 자체 모델링 — 이중 변환 방지) |

**근거**: [EXPERIMENT_LOG §5.1](../../3_modeling_이전자료/final/EXPERIMENT_LOG.md) — Tweedie objective와 log1p 충돌. 조건부 분기 처리.

```python
TARGET_TRANSFORM = 'log1p'  # 디폴트
CLIP_Y_EXTREME   = True      # train의 max=1.0 한 샘플 → 두 번째로 큰 값으로 clip

# trial별 분기 (loss가 tweedie 계열이면 log1p 비활성화)
if loss in TWEEDIE_NAMES:
    target_transform_fn = None
    target_inverse_fn   = None
else:
    target_transform_fn = np.log1p
    target_inverse_fn   = lambda y: np.clip(np.expm1(y), 0, None)
```

> 노트북 작성 시 [hpo.py:run_hpo](../modules/hpo.py)의 `target_transform_fn` 인자를 trial 안에서 동적으로 결정하도록 변경 필요. 현재 코드는 study 단위 고정 — 분기 추가.

---

## 5. Anchor + Enqueue (1차 best 보존)

각 모델 첫 trial은 anchor 그대로 강제: `study.enqueue_trial(anchor)`.

이후 trial들은 §7의 wide range에서 자유 탐색.

### 5.1 lgbm anchor (1차 OOF=0.005521)
```python
LGBM_ANCHOR = {
    'objective':         'poisson',
    'n_estimators':      957,
    'learning_rate':     0.00602,
    'num_leaves':        379,
    'max_depth':         10,
    'min_child_samples': 343,
    'subsample':         0.711,
    'subsample_freq':    1,
    'colsample_bytree':  0.597,
    'reg_alpha':         0.00840,
    'reg_lambda':        0.000151,
    'min_split_gain':    1.016e-04,
    'path_smooth':       24.81,
}
```

### 5.2 catboost anchor (1차 OOF=0.005523)
```python
CATBOOST_ANCHOR = {
    'loss_function':       'RMSE',
    'iterations':          1269,
    'learning_rate':       0.02712,
    'depth':               9,
    'l2_leaf_reg':         22.96,
    'random_strength':     0.1593,
    'bagging_temperature': 0.7136,
    'border_count':        183,
    'rsm':                 0.7349,
}
```

### 5.3 et anchor (1차 OOF=0.005547)
```python
ET_ANCHOR = {
    'n_estimators':      583,
    'max_depth':         21,
    'min_samples_leaf':  8,
    'min_samples_split': 34,
    'max_features':      'sqrt',
}
```

### 5.4 enet anchor (1차 OOF=0.005563)
```python
ENET_ANCHOR = {
    'alpha':       5.377e-06,
    'l1_ratio':    0.885,
    'max_iter':    15000,
    'tol':         1e-06,
    'selection':   'random',
    'precompute':  True,
}
```

### 5.5 xgb (우회 anchor — Y>0 컨텍스트)
```python
XGB_ANCHOR = {  # ⚠ Y>0 subset에서 학습 — narrow ±50%로 탐색
    'objective':              'reg:tweedie',
    'tweedie_variance_power': 1.5,
    'n_estimators':           1423,
    'learning_rate':          0.0363,
    'max_depth':              10,
    'min_child_weight':       0.621,
    'subsample':              0.728,
    'colsample_bytree':       0.618,
    'reg_alpha':              0.01680,
    'reg_lambda':             3.890e-06,
    'gamma':                  3.837e-06,
}
```

---

## 6. 손실함수 후보 (부스팅 3종 통일)

| 모델 | 후보 | tweedie_variance_power |
|---|---|---|
| lgbm | `regression`, `poisson`, `tweedie` | `float(1.05, 1.95)` |
| xgb | `reg:squarederror`, `count:poisson`, `reg:tweedie` | `float(1.05, 1.95)` |
| catboost | `RMSE`, `Poisson`, `Tweedie` | `float(1.05, 1.95)` |
| et | (옵션 없음 — sklearn ExtraTrees는 MSE 고정) | — |
| enet | (옵션 없음 — L1+L2 고정) | — |

**근거**: 1차 reg_only DB top-2가 박빙(차이 <0.000005), top-3부터 명백히 worse. xgb는 1차 데이터 없어 통일성 + power 연속(1.05~1.95) 통합 탐색.

> **모듈 코드 변경 필요** ([models.py](../modules/models.py)):
> - xgb space에 `count:poisson` 추가 (현재 미포함)
> - catboost space에 `Poisson` 추가 (현재 미포함)
> - tweedie_1.2 / tweedie_1.5 categorical 분기 → `suggest_float(1.05, 1.95)` 통합

---

## 7. HPO 탐색 범위 — Evidence-driven Wide (lgbm/catboost/et/enet) + Anchor Narrow ±50% (xgb)

각 모델 1차 reg_only top 10 trial 분포의 **p5~p95 영역**을 search range로 설정.

### 7.1 LightGBM
| HP | 새 range | 1차 top10 영역 |
|---|---|---|
| `objective` | `[regression, poisson, tweedie]` | poisson 8, tweedie_1.2 2 |
| `learning_rate` | `log(0.005, 0.05)` | 0.005~0.017 |
| `num_leaves` | `int(64, 384)` | 19~379 |
| `max_depth` | `int(7, 14)` | 8~14 |
| `min_child_samples` | `int(50, 380)` | 120~348 |
| `subsample` | `float(0.60, 0.95)` | 0.71~0.88 |
| `subsample_freq` | `int(0, 5)` | anchor=1 |
| `colsample_bytree` | `float(0.20, 0.80)` | 0.23~0.64 |
| `reg_alpha` | `log(1e-8, 1)` | 1e-8 ~ 8e-3 |
| `reg_lambda` | `log(1e-7, 1e-1)` | 9e-7 ~ 5e-3 |
| `min_split_gain` | `log(1e-8, 1e-3)` | 3e-8 ~ 1e-4 |
| `path_smooth` | `float(0, 50)` | 10~39 |
| `n_estimators` | `int(400, 2500)` | 632~2370 |

### 7.2 CatBoost
| HP | 새 range | 1차 top10 영역 |
|---|---|---|
| `loss_function` | `[RMSE, Poisson, Tweedie]` | RMSE 10/10 (§6 통일) |
| `iterations` | `int(800, 2000)` | 934~1680 |
| `learning_rate` | `log(0.005, 0.1)` | 0.009~0.058 |
| `depth` | `int(6, 10)` | 7~9 |
| `l2_leaf_reg` | `log(5, 30)` | 8.9~30 |
| `random_strength` | `log(0.1, 5)` | 0.1~0.69 |
| `bagging_temperature` | `float(0, 1)` | 0.03~0.80 |
| `border_count` | `int(64, 254)` | 119~190 |
| `rsm` | `float(0.4, 1.0)` | 0.45~0.96 |

### 7.3 ExtraTrees
| HP | 새 range | 1차 top10 영역 |
|---|---|---|
| `n_estimators` | `int(300, 800)` | 409~583 |
| `max_depth` | `int(15, 25)` | 19~22 |
| `min_samples_leaf` | `int(2, 20)` | 5~13 |
| `min_samples_split` | `int(15, 50)` | 26~37 |
| `max_features` | `'sqrt'` 고정 | top10 모두 sqrt |

### 7.4 ElasticNet (모델 HP)
| HP | 새 range | 1차 top10 영역 |
|---|---|---|
| `alpha` | `log(1e-7, 1e-4)` | 3e-6 ~ 8.6e-6 |
| `l1_ratio` | `float(0.50, 0.95)` | 0.57~0.96 |
| `max_iter` | `int(8000, 20000)` | 8000~15000 |
| `tol` | `1e-6` 고정 | — |
| `selection` | `'random'` 고정 | — |

### 7.5 XGBoost (anchor ±50% — 1차 reg_only 데이터 없음)
[XGB_ANCHOR §5.5](#55-xgb-우회-anchor--y0-컨텍스트) 기준 ±50% (정수형은 ±적은 step):

| HP | 새 range |
|---|---|
| `objective` | `[reg:squarederror, count:poisson, reg:tweedie]` |
| `n_estimators` | `int(700, 2200)` |
| `learning_rate` | `log(0.018, 0.073)` |
| `max_depth` | `int(7, 14)` |
| `min_child_weight` | `log(0.31, 1.24)` |
| `subsample` | `float(0.55, 0.95)` |
| `colsample_bytree` | `float(0.45, 0.95)` |
| `reg_alpha` | `log(0.0084, 0.0336)` |
| `reg_lambda` | `log(1.95e-6, 7.78e-6)` |
| `gamma` | `log(1.92e-6, 7.67e-6)` |

---

## 8. Optuna 설정

| 모델 | N_TRIALS (참고값) | 비고 |
|---|---|---|
| lgbm | 150 | wide range, anchor enqueue |
| catboost | 150 | wide range |
| et | 100 | wide range, trial당 60~90초 |
| xgb | 150 | anchor ±50% |
| enet | 200 | PP+scaling+HP joint |

> N_TRIALS는 사용자 임의 조정. 위는 참고값.

공통:
- `N_FOLDS = 5` (§6)
- `sampler = TPESampler(seed=None, multivariate=True, group=True)` (§4)
- `pruner = MedianPruner(n_warmup_steps=10)` (선택)
- `direction = 'minimize'` (objective = OOF unit RMSE)
- `study_name = 'reg-{model}-002'` (1차와 분리)
- **`study.enqueue_trial(ANCHOR)`로 첫 trial 강제** (5종 모두)

---

## 9. enet 전용 — PP + scaling + HP joint search space

[strategy_common.md §3](../strategy_common.md). enet 노트북 trial이 sample하는 축:

### 9.1 PP 8축 (1차 joint 분포 기반 재조정)

| 키 | 새 range | 근거 (1차 top 390 trial 분포) |
|---|---|---|
| `missing_threshold` | `float(0.30, 0.90)` | p25=0.4, p50=0.7 |
| `corr_threshold` | `float(0.88, 0.98)` | p50=0.9 |
| `corr_keep_by` | **`'std'` 고정** | leakage 회피 ([plan.md §9.5](../../3_modeling_이전자료/final/plan.md)) |
| `add_indicator` | `[True, False]` | True 우세 |
| `indicator_threshold` | `float(0.05, 0.20)` | p25=0.05, p50=0.10, p75=0.15 |
| `spatial_max_dist` | `float(1.0, 6.0)` | top max=5 + 트리 6 일치 |
| `post_impute_corr_threshold` | `float(0.96, 0.99)` | min=0.97 |
| `post_impute_corr_keep_by` | **`'std'` 고정** | leakage 회피 |

> `corr_keep_by`/`post_impute_corr_keep_by`는 `'target_corr'` 후보를 **제거**. 이유: KFold 안에서 train target 전체를 feature selection에 사용 = supervised leak.

### 9.2 scaling 5종 (categorical)

- `StandardScaler`
- `RobustScaler`
- `Yeo-Johnson` (sklearn `PowerTransformer(method='yeo-johnson')`)
- `QuantileTransformer`
- `HybridScaler` ([2_preprocessing/scaling.py](../../2_preprocessing/scaling.py)) — `skew_threshold`도 HP 탐색

### 9.3 enet HP

[7.4](#74-elasticnet-모델-hp) 참조 — wide range로 통일.

trial당 cleaning + scaling 새로 돌아 비용 큼 → N_TRIALS 200 부여.

---

## 10. 후처리 매트릭스

[strategy_common.md §9~12](../strategy_common.md):

| 룰 | 적용? | 비고 |
|---|---|---|
| 분류 threshold (§9) | N/A | 회귀 단독 |
| die→unit 집계 다양성 (§10) | APPLY | 8후보: mean/median/max/min/trimmed_mean/weighted/Q25/Q75 |
| Position 가중치 (§11) | APPLY | Optuna sub-study 50 trial로 w1~w4 (Dirichlet) |
| zero_clip (§12) | APPLY | 0.001~0.015 step 0.001, **log space에서 비교** |

**zero_clip log space 적용** (전략 common §12 보강):
- 모델 출력 = log1p(y) space
- 임계값 비교: `pred_log < log1p(th)` 인 unit → 0
- 이후 `np.expm1(pred_log_clipped)` → original space로 복원
- target_transform이 'none' 또는 'tweedie' 분기에서 OFF인 경우 기존(original space) 흐름

원칙: train OOF best 탐색 → val 적용 → val 개선 시만 채택.

> **모듈 코드 변경 필요**:
> - [postprocess.py](../modules/postprocess.py) `apply_zero_clip` — log space 입력 받도록 분기 추가 (5~10줄)
> - [postprocess.py](../modules/postprocess.py) 집계 후보에 `Q25`, `Q75` 추가 (1차 6종 → 신규 8종)

**1차 후처리 best (참고)**:

| 모델 | best_agg | best_zero_clip |
|---|---|---|
| lgbm | median | 0.001 |
| catboost | mean | 0.001 |
| et | mean | 0.002 |
| enet | mean | 0.001 |

---

## 11. 출력 경로

| 노트북 | OUT_DIR |
|---|---|
| lgbm | `4_output/final/reg_only/lgbm/` |
| xgb | `4_output/final/reg_only/xgb/` |
| catboost | `4_output/final/reg_only/catboost/` |
| et | `4_output/final/reg_only/et/` |
| enet | `4_output/final/reg_only/enet/` |

산출물 9개 ([strategy_common.md §15](../strategy_common.md)) 표준화.

---

## 12. 실행 순서 + 병렬

[strategy_common.md §8](../strategy_common.md) 학원 14코어:

| 시나리오 | N_JOBS | 비고 |
|---|---|---|
| 트리 2개 병렬 | 7 | 권장 (lgbm + catboost, xgb + et) |
| enet 단독 | 14 | PP joint trial 무거움 |

**권장 순서**:
1. lgbm + catboost 병렬
2. xgb + et 병렬
3. enet 단독

---

## 13. 모듈 코드 변경 요약 (노트북 작성 전 사전 작업)

| # | 파일 | 변경 |
|---|---|---|
| 1 | [models.py](../modules/models.py) | xgb space에 `count:poisson` 추가 |
| 2 | [models.py](../modules/models.py) | catboost space에 `Poisson` 추가 |
| 3 | [models.py](../modules/models.py) | tweedie_1.2/1.5 categorical → `suggest_float(1.05, 1.95)` |
| 4 | [hpo.py](../modules/hpo.py) | trial 안에서 `target_transform_fn` 분기 (tweedie 시 None) |
| 5 | [hpo.py](../modules/hpo.py) | `study.enqueue_trial(anchor)` 헬퍼 추가 |
| 6 | [postprocess.py](../modules/postprocess.py) | `apply_zero_clip` log space 분기 |
| 7 | [postprocess.py](../modules/postprocess.py) | 집계 후보 8종으로 확장 (Q25/Q75 추가) |

→ 모듈 작업 끝나면 노트북 5개 작성 들어감.

---

## 14. 결정 필요 사항 (노트북 작성 시)

| # | 항목 | default |
|---|---|---|
| 1 | xgb `count:poisson` + `log1p=ON` 학습 가능 여부 | 첫 fold 학습 후 확인 |
| 2 | catboost `Poisson` loss 학습 안정성 | 첫 fold 학습 후 확인 |
| 3 | enet `Yeo-Johnson` + 낮은 `corr_threshold` 조합 학습 발산 가능 | trial 실패 처리 후 사용자 보고 |

---

## 15. 검증 — 노트북 작성 시 확인

[strategy_common.md §13](../strategy_common.md) 구현 문제 보고 원칙 준수.

특히:
- tweedie 선택 시 log1p가 진짜 OFF되는지 trial별 user_attr 로깅
- log space zero_clip 적용 후 RMSE가 original space보다 개선되는지 비교 측정
- 모든 anchor가 첫 trial로 enqueue 되었는지 확인 (Optuna study trial[0]의 params == ANCHOR)
