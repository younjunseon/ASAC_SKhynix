# 03 — Two-Stage Default (clf → reg)

> Path A: Stage 1 분류 → Stage 2 회귀(y>0 only) → final = `prob × pred`. Path B (reverse)는 [`../reverse/`](../reverse/) 별도.

## 1. 구조

| 단계 | 위치 | 노트북 |
|---|---|---|
| **Stage 1 (CLF)** | `clf/` | `lgbm.ipynb`, `xgb.ipynb`, `catboost.ipynb`, `et.ipynb` (4개) |
| **Stage 2 (REG)** | `reg/` | `lgbm.ipynb`, `xgb.ipynb`, `catboost.ipynb`, `et.ipynb`, `enet.ipynb` (5개) |
| **Combine** | (root) | `combine.ipynb` (1개) |

총 **10개 노트북** + strategy.md.

## 2. 공통 정책 (strategy_common.md 참조)

| 항목 | 정책 | 근거 |
|---|---|---|
| PP | `PP_FIXED` (트리 4종) / PP joint Optuna (ENet) | §1, §3 |
| KFold | unit-level, N=5, SEED=42 | §6, §7 |
| Sampler | `TPESampler(seed=None, multivariate=True, group=True)` | §4 |
| Pruner | `MedianPruner(n_warmup_steps=10)` | §4 |
| `N_JOBS` | 7 (병렬 2 노트북) / 14 (단독) | §8 |
| `N_TIMEOUT_SEC` | None (override 가능) | §25 |
| anchor | 1차 best HP enqueue (REG 트리 + ENet) / 없음 (CLF, 1차 신뢰도 낮음) | §5 |
| **트리 `target_transform`** | **`'none'` 통일** | **§24 (log1p_check 검증)** |
| ENet `target_transform` | Optuna 카테고리 4종 | §3 |
| `CLIP_Y_EXTREME` | True | 1차 동일 |

## 3. CLF (Stage 1)

### 3.1 공통 설계

- die-level 학습 (`y_unit > 0` broadcast)
- objective = `RMSE(unit_mean(die_prob) × y_pos_const, y_train_unit)` (clf 단독 평가)
- imbalance 자동 처리 (각 모델 default + `scale_pos_weight` / `auto_class_weights` / `class_weight` Optuna 탐색)
- `y_pos_const = E[Y | Y>0]` ≈ 0.008496
- 출력: die_prob csv → combine에서 reg_pred와 곱

### 3.2 anchor 정책 — **없음 (Wide search)**

⚠️ 1차 CLF는 모두 1 trial만 — anchor 신뢰도 낮음. 추가로 1차 best `lr=0.245`가 신규 clf space 상한 0.15 초과.

→ **anchor enqueue 안 함**. `models.get_clf_search_space()` Wide range에서 처음부터 탐색.

### 3.3 N_TRIALS

`N_TRIALS = 100` default (1차는 1 trial만이라 본격 탐색 필요). 사용자 환경 따라 수정.

## 4. REG (Stage 2 — y>0 only conditional)

### 4.1 공통 설계 (1차 03d 그대로)

- **`y_positive_only = True`** — y_unit==0인 unit의 die 제외 후 학습
- **die-level broadcast** (y_unit → 4 die)
- objective = OOF die-level pred → unit mean → unit RMSE (reg 단독)
- combine 시 clf prob과 곱셈 → final RMSE

### 4.2 트리 4종 (lgbm/xgb/catboost/et)

| 항목 | 값 |
|---|---|
| `TARGET_TRANSFORM` | **`'none'`** (strategy_common §24) |
| target_transform_fn | `None` (identity) |
| target_inverse_fn | `None` (clip(0)만 적용) |
| 손실함수 | `regression / poisson / tweedie_1.2 / tweedie_1.5` (LGBM 기준), 모델별 native — `models.get_search_space()`에서 자동 탐색 |

⚠️ **1차 reg DB는 log1p ON 환경**:
- lgbm best loss = `tweedie_1.2`
- xgb best loss = `reg:tweedie_1.5`
- catboost best loss = `Tweedie_1.5`
- et: 분포 무관

→ 신규는 `'none'` 환경 (log1p_check 결과 reg와 log1p 동등)이라 best loss가 1차와 다를 수 있음. anchor는 1차 best HP 그대로 enqueue 후 Optuna가 재탐색.

### 4.3 모델별 anchor (1차 best)

| 모델 | 1차 best | trials | anchor (주요 HP) | best loss (1차) |
|---|---|---|---|---|
| **lgbm** | val=0.006806 | 100 | n_est=1839, lr=0.0527, num_leaves=334, max_depth=14, min_child=5, subsample=0.925 | tweedie_1.2 |
| **xgb** | val=0.006821 | 200 | n_est=1423, lr=0.0363, max_depth=10, min_child_weight=0.62, subsample=0.728 | reg:tweedie_1.5 |
| **catboost** | val=0.007050 | 90 | iter=2408, lr=0.212, depth=10, l2_leaf_reg=1.51, bagging_temp=0.648 | Tweedie_1.5 |
| **et** | val=0.008218 | 300 | n_est=753, max_depth=10, min_samples_leaf=37, min_samples_split=31 | (분포 무관) |

→ anchor에 손실함수 키는 **빼고** enqueue (Optuna가 OFF 환경에 맞게 재선택).

### 4.4 search range

`models.get_search_space(model_name)` 그대로 사용 (1차와 동일 search space). anchor enqueue로 1차 best 영역에서 시작 + Optuna 자유 탐색.

### 4.5 ENet (별개, PP joint + target_transform Optuna)

[strategy_common §3](../../strategy_common.md) 정책:
- PP 6축 (범위 탐색)
- X scaling 5종 (`StandardScaler / RobustScaler / YeoJohnson / Quantile / Hybrid`)
- y target_transform 4종 (`none / log1p / yeo-johnson / quantile`)
- ENet HP (alpha / l1_ratio / max_iter)
- **Y_POSITIVE_ONLY=True** (Stage 2 정석)
- anchor: 1차 reg-enet best (val=0.008230)

### 4.6 N_TRIALS (REG)

| 모델 | N_TRIALS default | 비고 |
|---|---|---|
| lgbm | 100 | 1차 동일 |
| xgb | 100 | 1차 200 → 줄임 |
| catboost | 100 | 1차 90 |
| et | 100 | 1차 300 → 줄임 (효율) |
| enet | 200 | PP joint라 비용 큼 |

## 5. Combine (M×N grid + position weighted)

[1차 03e](../../../모델링_이전자료/3_modeling_이전자료/final/03e_ts_combine.ipynb) **그대로**:
- Step 1: die-level `final_die = clf_prob × reg_pred`
- Step 2 (a): die→unit mean (baseline) → `oof|val|test_unit.csv`
- Step 2 (b): position weighted avg (Optuna 50 trial × 그리드) → `oof|val|test_unit_weighted.csv`
- 출력: `combined/{clf}_x_{reg}/...` + `grid_summary.csv` + `weighted_summary.csv` + `combine_meta.json`

CLF 4 × REG 5 = **20개 그리드 조합** (1차 동일).

`N_TRIALS_POS = 50` (1차 동일), `TIMEOUT_SEC = None`.

## 6. 출력 경로

| 노트북 | OUT_DIR |
|---|---|
| `clf/{model}.ipynb` | `4_output/03_two_stage/default/clf/{model}/` |
| `reg/{model}.ipynb` | `4_output/03_two_stage/default/reg/{model}/` |
| `combine.ipynb` | `4_output/03_two_stage/default/combined/` |

산출물 9개 표준 ([strategy_common §15](../../strategy_common.md)). 추가로 combine은 `grid_summary.csv` / `weighted_summary.csv` / `combine_meta.json`.

## 7. 실행 순서

| 순서 | 노트북 | 의존 | N_JOBS |
|---|---|---|---|
| 1 | `clf/lgbm` + `clf/xgb` 병렬 | (없음) | 7 |
| 2 | `clf/catboost` + `clf/et` 병렬 | (없음) | 7 |
| 3 | `reg/lgbm` + `reg/xgb` 병렬 | (없음) | 7 |
| 4 | `reg/catboost` + `reg/et` 병렬 | (없음) | 7 |
| 5 | `reg/enet` 단독 | (없음) | 14 |
| 6 | `combine.ipynb` | 1~5 완료 | (Optuna 단독) |

## 8. 모듈 의존

기존 `modules.preprocess / hpo / models` 그대로 사용. `hpo.run_hpo` / `run_clf_hpo`에 §4·§5·§25 적용을 위해 4개 인자 추가됨:
- `sampler=` (None이면 `TPESampler(seed=seed)` default)
- `pruner=` (None이면 NopPruner)
- `enqueue_trials=` (list[dict], None이면 안 함)
- `timeout=` (초, None이면 무제한)

backward-compat 보장 (default=None).

## 9. 작성 시 검수 (§23 6항목)

각 노트북 작성 후 [strategy_common §23](../../strategy_common.md) 6항목 검수.

특히:
- §23.5: REG 트리 anchor가 1차 best HP와 일치 (단, 손실함수 키 제외)
- §23.5: 트리 reg는 transform=none (1차와 다름, 정책 §24 따름)
- §23.6: KFold unit-level + corr_keep_by='std'
