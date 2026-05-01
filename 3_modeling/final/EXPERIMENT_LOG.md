# BagZITboost (안 C / B3) — 실험 로그

작성일: 2026-04-30
위치: `3_modeling/two_stage/_temp_label_em/`

---

## 0. 한 줄 요약

> **ZIT 단독은 die 4개에 unit y를 그대로 broadcast 했지만, BagZIT은 unit y를 4 die에 비례 분배**하여 학습. die-level 학습 구조는 동일, 라벨과 합산 방식만 다름.

| 비교 | ZIT 단독 | BagZIT (B3) |
|---|---|---|
| 모델 구조 | π + μ + φ via EM | **동일** |
| 학습 라벨 | unit y broadcast (4 die 모두 같은 값) | unit y를 (1-π)μ 비례 분배 (Σ = unit y) |
| 예측 합산 | unit-level mean | **unit-level sum** = Σ (1-π_i)μ_i |
| 추가 코드 | — | EM E-step 직전 분배 1단계, ~30줄 |

---

## 1. 배경 — 출발점이 된 가설 (label_em_experiment)

### 1.1 원래 가설

ZIT 단독은 unit Y>0이면 4 die 모두에 broadcast → 멀쩡한 die도 "불량"으로 라벨링됨 → 분류 신호 흐림.
**"진짜 범인 die만 골라서 다시 학습하면 RMSE 개선될 것"**

### 1.2 실험 설정 (`label_em_experiment.ipynb` — **현재는 삭제됨**)

| Round | 라벨 |
|---|---|
| 0 | broadcast (4 die 모두 1) |
| 1 | IsolationForest anomaly + Round 0 P 결합 → max-score die 1개만 1 |
| 2~5 | 직전 P(die)의 max → unit당 1 culprit (iterative EM) |

### 1.3 결과 → **가설 기각**

| Round | OOF unit RMSE | Δ from R0 |
|---|---|---|
| **0 (broadcast)** | **0.005560** ← best | 0 |
| 1 | 0.005820 | +0.000261 |
| 2 | 0.005795 | +0.000235 |
| 3~5 | 0.005754~5795 | +0.0002 |

**진단**:
- IsoForest anomaly 범위 0.39~0.62 (좁음) — die간 차이 거의 없음
- culprit position 분포 균등 (1857/1950/1937/1902) — 어느 die가 더 의심스럽다는 신호 없음
- die-level X feature 자체가 culprit 식별에 부족 (EDA max\|r\|=0.037)

→ "unit당 1개 culprit" 가설 부적합. broadcast가 정답.

### 1.4 후속 논의

"unit당 1~4개 자유면?" → 단순한 hard selection으로는 부족. **bag-level constraint** (같은 unit의 die들이 묶여 있다는 정보)을 EM에 명시적으로 넣는 방식이 필요 → **안 C 도출**.

---

## 2. 안 C / B3 설계 결정

### 2.1 안 비교 (제안 → 폐기 → 최종)

| 안 | 내용 | 평가 |
|---|---|---|
| 안 A | hard threshold k 가변 (score > τ) | 폐기 — label_em 변형, 동일 한계 |
| 안 B | soft label EM (P(die culprit)) | 폐기 — ZIT의 부분 재발명 |
| **안 C** | **joint EM + bag-level constraint** | **채택** |

### 2.2 모델 수 결정 (b → a로 정정)

처음에 "(a) 2개 (CLF + REG)" 추천했으나, 분석 후 **(b) 3개 (π + μ + φ)** 가 합리적이라 정정:
- ZITboost 모델 구조는 통계적으로 검증됨 (Tweedie likelihood)
- 차별성은 bag constraint 자체에 있음 — 모델 구조까지 바꾸는 건 불필요
- 구현 복잡도도 ZITboost 코드 재활용으로 낮아짐

### 2.3 bag constraint 적용 방식 (B1/B2/B3)

| 방식 | 강제 조건 |
|---|---|
| B1 | Y>0 unit 안에서 max(1-π_i) ≥ 0.5 (페널티) |
| B2 | Y>0 unit 안에서 (1-π_i) 합 ≥ 1 |
| **B3** | **Σ_{i in unit} (1-π_i)μ_i = unit_y (예측 합 = 실제값)** |

→ **B3 채택**. 이유: 임의 숫자 없음, "예측 = 실제"라는 학습 목적과 정확히 일치.

### 2.4 최종 알고리즘 (안 C / B3)

```
초기화: y_die_alloc[i] = unit_y / 4   (uniform 분배)

EM iter 0~N:
  if iter > 0:
      # B3 재할당 — Σ shares = 1 per unit
      contribution[i] = (1 - π_i) × μ_i
      share[i] = contribution[i] / Σ_{j in same unit} contribution[j]
      y_die_alloc[i] = unit_y × share[i]
      # Y=0 unit dies → y_die_alloc[i] = 0

  E-step: posterior Π_i = P(structural zero | y_die_alloc[i], x_i)
  M-step: π / μ / φ 모델 업데이트 (target = y_die_alloc)
  
  unit RMSE 추적, em_tol 미만이면 early stop

최종 unit 예측: Σ_{i in unit} (1 - π_i) × μ_i  (sum aggregation)
```

---

## 3. 구현 — `bag_zit_experiment.ipynb` (HP 고정 baseline)

### 3.1 파일 흐름

| 단계 | 작업 |
|---|---|
| `label_em_experiment.ipynb` | **삭제** |
| origin/main의 `01_zit_only.ipynb` (die-level 버전) | git에서 추출 |
| `bag_zit_experiment.ipynb` | 위 베이스 + BagZITboostRegressor 정의 + 5-fold OOF + holdout |

### 3.2 BagZITboostRegressor 구조

`modules/zi_tweedie.py`의 `ZITboostRegressor` 상속:
- **fit(X, y, unit_id)** override — unit_id 인자 추가 + B3 분배 단계
- **`_allocate_b3()`** 추가 — 비례 분배 staticmethod
- **`predict_unit(X, unit_id)`** 추가 — die 예측 합산 (sum)
- 나머지 (`_initialize`, `_e_step`, `_m_step`, `_mu_params`, `_pi_params`, `_phi_params`, `predict`, `predict_components`)는 부모 그대로 사용

기존 `modules/zi_tweedie.py`는 **수정 안 함** (read-only).

### 3.3 실험 설정

```python
EXP_ID = 'bag-zit-temp-001'
N_FOLDS = 5
CLIP_Y_EXTREME = True   # y_train 1.0 → second_max(0.097417) clip
TARGET_TRANSFORM = 'none'  # ZIT는 Tweedie가 분포 직접 모델링 → log1p OFF

BAG_PARAMS = dict(
    zeta=1.5, n_em_iters=10, em_tol=1e-7,
    mu_n_estimators=300, mu_learning_rate=0.05, mu_num_leaves=63,
    mu_max_depth=6, mu_min_child_samples=20,
    mu_subsample=0.8, mu_colsample_bytree=0.8,
    mu_reg_alpha=1e-3, mu_reg_lambda=1e-1,
    pi_n_estimators=200, pi_learning_rate=0.05, pi_num_leaves=31,
    pi_max_depth=6, pi_min_child_samples=20,
    phi_n_estimators=100, phi_learning_rate=0.05, phi_num_leaves=31,
    phi_max_depth=6, phi_min_child_samples=20,
    random_state=SEED, n_jobs=-1, verbose=-1, device='cpu',
)

PARAMS = {
    'missing_threshold':          0.4,
    'corr_threshold':             0.90,
    'corr_keep_by':               'std',
    'add_indicator':              True,
    'indicator_threshold':        0.05,
    'spatial_max_dist':           5.0,
    'post_impute_corr_threshold': 0.98,
    'post_impute_corr_keep_by':   'std',
}
```

### 3.4 실행 결과 — **bag constraint 효과 미세하지만 일관**

| 지표 | BagZIT (B3) | ZIT 단독 (zit-final-999, 1 trial) | Δ |
|---|---|---|---|
| OOF unit | **0.005524** | 0.005592 | **−0.000068** ✅ |
| val unit | **0.005729** | 0.005772 | **−0.000043** ✅ |
| test unit | **0.008428** | 0.008450 | **−0.000022** ✅ |

전체 학습 시간: 1228.8s (~20분, 5-fold × EM 10 iter × 3 LGBM fit)

### 3.5 EM 학습 곡선 (fold 0)

| EM iter | unit_RMSE |
|---|---|
| 1 | **0.004998** ← best |
| 2 | 0.005012 |
| 3~10 | 0.0050~0.0052 변동 |

→ **iter 1이 거의 best**. 이후 미세 변동만. → HPO에서 `n_em_iters` 하한을 낮추는 것이 합리적이지만, 안전 마진 위해 ZIT 범위 따름.

### 3.6 진단

- π_mean = **0.7050** ≈ 데이터 Y=0 비율 (70.8%) ✓
- die pred mean = 0.000554 → unit pred mean = **0.002214 ≈ 4 × die_mean** ✓ (B3 sum 합산 정상)
- y_unit true mean = 0.002481 → 거의 일치
- **Y>0 unit pred mean = 0.002826, true = 0.008496** → **3× under-predict** ⚠
  - 대부분 unit이 Y=0이라 RMSE에는 크게 안 나타나지만, 고불량 unit 식별은 여전히 약함

### 3.7 데이터 누수 점검 ✅

| 항목 | 결과 |
|---|---|
| `preprocess.run()` train-only fit | ✅ 출력 로그 "train-only 모드" 확인 |
| KFold 분할 단위 | ✅ `unit_ids_train_unique` 기준 (4 die가 분리되지 않음) |
| BagZIT.fit | ✅ train fold만 입력 |
| OOF 예측 | ✅ `predict_unit(X_vl, uid_vl)` — fit 호출 없음 |
| val/test 예측 | ✅ 5 fold 모델 holdout 예측 후 평균 (snapshot ensemble) |
| `CLIP_Y_EXTREME` second_max | ✅ train y에서만 계산 |

---

## 4. HPO — `bag_zit_hpo.ipynb`

### 4.1 시간 예산 / trial 수

- 가용 시간: ~11시간 (overnight)
- 1 trial 평균: ~30분 (n_em_iters 평균 15, 5-fold × 3 LGBM)
- **N_TRIALS = 10** (안전 마진 포함, ~6시간)
- DB resumable (SQLite) — 중단 시 재개 가능

### 4.2 탐색공간 — ZIT top 20 anchor + ±10% 마진

`optuna_jh_zit-final-100.db` (180 trials) 의 top 20 best trial 분포 분석:

| HP | ZIT top 20 range | BagZIT 적용 범위 |
|---|---|---|
| zeta | 1.13~1.25 | [1.10, 1.30] |
| n_em_iters | 14~19 | [10, 20] (BagZIT 하한 ↓) |
| mu_n_estimators | 100~284 | [100, 300] |
| mu_learning_rate | 0.006~0.009 | [0.005, 0.012] log |
| mu_num_leaves | 123~167 | [110, 180] |
| mu_max_depth | **5 (locked)** | [4, 6] |
| mu_min_child_samples | 71~90 | [60, 100] |
| mu_subsample | 0.57~0.69 | [0.55, 0.75] |
| mu_colsample_bytree | 0.35~0.42 | [0.30, 0.50] |
| mu_reg_alpha | 1.1e-4~2.7e-3 | [5e-5, 5e-3] log |
| mu_reg_lambda | 0.02~0.13 | [0.01, 0.20] log |
| pi_n_estimators | 126~215 | [120, 230] |
| pi_learning_rate | 0.070~0.086 | [0.06, 0.10] log |
| pi_num_leaves | 66~76 | [60, 80] |
| pi_max_depth | **8 (locked)** | [7, 9] |
| pi_min_child_samples | 24~49 | [20, 55] |
| phi_n_estimators | 50~90 | [40, 100] |
| phi_learning_rate | 0.012~0.016 | [0.010, 0.020] log |
| phi_num_leaves | 61~89 | [55, 95] |
| phi_max_depth | **5 (locked)** | [4, 6] |
| phi_min_child_samples | 22~92 | [20, 100] |

→ 총 21 HP, 모든 max_depth는 ZIT top 20에서 단일 값으로 수렴했으므로 ±1만 허용.

### 4.3 샘플러 / Pruner

```python
sampler = TPESampler(
    multivariate=True,    # joint 분포 (HP 상호작용 학습)
    group=True,           # μ_*, π_*, φ_* 자동 그룹화
    n_startup_trials=4,   # 4 random + 6 multivariate
    seed=SEED,
)
pruner = MedianPruner(
    n_startup_trials=4,
    n_warmup_steps=2,     # fold 2 이후 prune 가능
)
```

**왜 multivariate?** 기본 TPE는 marginal univariate (HP 독립). `multivariate=True`로 joint 분포 학습 → mu_learning_rate ↑ × mu_n_estimators ↑ = 과적합 같은 상호작용 잡힘. `group=True`는 conditional/관련 HP를 자동 그룹화하여 그룹 내 상호작용을 강하게 학습.

### 4.4 출력

```
4_output/_temp/bag_zit_hpo/
├── optuna_jh_bag-zit-hpo-001.db   # SQLite DB (resumable)
├── oof_unit.csv                    # train OOF (best params refit)
├── val_unit.csv                    # val (5 fold avg)
├── test_unit.csv                   # test (5 fold avg)
├── best_params.json
└── meta.json                       # search space, sampler, RMSE 결과 등
```

trial별 user_attrs: `val_rmse`, `test_rmse`, `fold_oof_rmse`, `elapsed_sec`

---

## 5. 부속 결정

### 5.1 log1p (target transform) — **OFF**

이유:
- ZIT의 μ 모델은 LightGBM `objective='tweedie'` 사용
- Tweedie 분포는 zero-inflation + 양수 right-skew 데이터 분포 자체를 모델링
- log1p 변환은 MSE 손실용 trick. Tweedie objective는 이미 right-skew 처리 내장
- BagZIT의 sum aggregation도 log1p 비호환 (`Σ log1p(y_i) ≠ log1p(unit_y)`)

→ origin/main `01_zit_only` 도 `TARGET_TRANSFORM='none'` 고정. 동일 결정 유지.

### 5.2 후처리 — **OFF (CSV로 사후 적용 가능)**

기존 ZIT 단독 (`01_zit_only`)의 `POSTPROCESS_CONFIG`:
```python
{
    'agg_methods':         ('mean', 'median', 'max', 'min', 'trimmed_mean', 'weighted'),
    'pi_threshold_range':  (0.5, 0.95),
    'pi_threshold_step':   0.01,
    'zero_clip_range':     (0.001, 0.015),
    'zero_clip_step':      0.001,
    'use_pi_threshold':    True,
}
```
→ `hpo.save_artifacts` 안에서 grid search로 best 후처리 파라미터 튜닝.

BagZIT은 후처리 미적용 (raw 예측). 단:
- **zero_clip** (`pred < thr → 0`): unit pred만 필요 → CSV로 사후 적용 가능
- **negative clip**: 이미 적용됨 (`np.clip(pred, 0, None)` in predict)
- **agg_methods** (mean/median/max): BagZIT은 sum 직접 사용 → 의미 없음
- **pi_threshold**: die-level π 필요 → 별도 저장해야 함 (현재 미저장)

→ HPO 끝나고 **zero_clip만 OOF 기반 grid search로 적용**하면 추가 ~0.0001~0.0003 개선 기대.

---

## 6. 다음 단계 — Lot/Wafer Target Encoding 적용 계획

### 6.1 출처

`_scratch/lot_wafer_baseline_compare.ipynb` (v3) + `_scratch/lot_wafer_alpha_grid.ipynb` (v4)

### 6.2 의도

각 die가 속한 lot / wafer / lot×position / wafer×position 단위에서 target 통계량을 미리 계산해서 **die의 입력 X에 12개 컬럼으로 추가**.

> "이 die가 속한 그룹의 평균 health는 얼마인가" → 모델이 구조적 정보 학습 쉬워짐

### 6.3 컬럼 (12개)

3종 인코딩 × 4그룹:

| 그룹 | te (mean) | zero_rate | pos_mean |
|---|---|---|---|
| **lot** | lot_te | lot_zero_rate | lot_pos_mean |
| **wafer** | wafer_te | wafer_zero_rate | wafer_pos_mean |
| **lot × position** (lp) | lp_te | lp_zero_rate | lp_pos_mean |
| **wafer × position** (wp) | wp_te | wp_zero_rate | wp_pos_mean |

각 인코딩 의미:
- **te (target encoding)**: 그룹 내 die들의 평균 health (`y` 평균)
- **zero_rate**: 그룹에서 health=0 인 die 비율
- **pos_mean**: 그룹 내 health>0 die만의 평균

### 6.4 Leak 방지 — GroupKFold 5-fold

train 안에서 단순 평균 내면 자기 자신의 health를 보고 학습 = leak.

```python
from sklearn.model_selection import GroupKFold

ALPHA = 20.0   # smoothing
N_FOLD_ENC = 5

gkf = GroupKFold(n_splits=N_FOLD_ENC)
unique_train_units = np.unique(ufs_tr)
fold_splits = [...]   # unit 단위 분할

# train: 각 fold die에는 다른 fold의 평균만 매핑
# val/test: train 전체 매핑 사용
```

`_smoothed_mean`: `(n × group_mean + α × global_mean) / (n + α)` (Bayesian smoothing)

### 6.5 Two-Stage 베이스라인에서 본 효과

| | A_baseline | B_+12enc | Δ |
|---|---|---|---|
| HPO 3-fold OOF | 0.005524 | **0.005489** | **-0.64%** |
| Rerun val | 0.005735 | **0.005706** | -0.51% |
| Rerun test | 0.008428 | **0.008403** | -0.30% |

**판정: BORDER** (개선 미세하지만 일관됨, 12개 모두 importance top 30 안에 들어옴)

Top importance: wafer_zero_rate (874), wafer_te (674), lot_te (587), wp_te (553).

### 6.6 BagZIT 적용 방법

**핵심**: BagZIT의 fit() 인자 X는 die-level numpy array. 12개 인코딩 컬럼을 X에 그냥 추가하면 끝. 모델 코드 수정 불필요.

#### 통합 단계

1. **인코딩 헬퍼 함수 복사**
   - `_scratch/lot_wafer_baseline_compare.ipynb` cell[6]의 `build_te_3way`, `_smoothed_mean` 함수
   - bag_zit_hpo 노트북에 셀 추가 (또는 utils 모듈로 분리)

2. **인코딩 컬럼 생성** (전처리 완료 후, BagZIT 학습 직전)
   - `lot`, `wafer`, `position` 컬럼이 xs에 있어야 함 (`run_wf_xy` 파싱)
   - GroupKFold 기반 leak-free 인코딩 수행
   - 12개 컬럼을 `xs_train_die`, `xs_val_die`, `xs_test_die`에 join

3. **`feat_cols_clean` 확장**
   - 인코딩 12개 컬럼 추가
   - `X_train_die = xs_train_die[feat_cols_clean].values` 시점에서 자연 반영

4. **BagZIT 학습** — 기존 코드 수정 없이 X가 12개 더 많은 컬럼으로 들어감

#### 누수 점검 추가

- Encoding GroupKFold 분할 ≠ BagZIT 5-fold 분할 → **인코딩 fold가 BagZIT 학습 시 leak 안 만드는지 확인 필요**
  - 가장 안전한 방식: BagZIT의 5-fold와 인코딩 5-fold를 **다른 seed**로 분리해도 OK (같은 train pool 안에서만 작용)
  - 또는 BagZIT 5-fold 분할에 맞춰 인코딩을 fold별로 다시 계산 (가장 strict, but 5× 시간)
  - v3 결과는 단순 GroupKFold(5) 사용 + train 전체 평균으로 val/test 매핑 → 표준 방식 채택

#### 예상 효과

- Two-Stage 결과 (-0.30~0.64%) 와 유사 폭 가능
- BagZIT 현재 OOF 0.005524 → **0.005490 부근** 기대
- BagZIT의 약점인 "Y>0 unit 3× under-predict" 일부 완화 가능 (lot/wafer mean이 Y>0 unit pred 신호 보강)

### 6.7 적용 시점 결정

| 옵션 | 설명 | 추천 |
|---|---|---|
| **A** | 현재 HPO 끝나고 → 별도 노트북 `bag_zit_lot_wafer.ipynb` 생성 | ✅ 추천 (HPO 결과 보존) |
| B | bag_zit_hpo 노트북 수정 후 재실행 | × HPO 결과 손실 |
| C | bag_zit_experiment (HP 고정) 에 적용 → 직접 비교 | 가능 (빠른 검증용) |

**추천: A**. 내일 HPO 완료 후 best_params 사용해서 인코딩 적용 비교.

---

## 7. 파일 목록 / 위치

```
3_modeling/two_stage/_temp_label_em/
├── EXPERIMENT_LOG.md            ← 본 문서
├── bag_zit_experiment.ipynb     ← HP 고정 baseline (실행 완료, OOF 0.005524)
└── bag_zit_hpo.ipynb            ← Optuna 10 trial HPO (실행 대기)

4_output/_temp/bag_zit/          ← experiment 결과
4_output/_temp/bag_zit_hpo/      ← HPO 결과 (생성 예정)

참조:
modules/zi_tweedie.py            ← read-only, ZITboostRegressor (BagZIT 부모)
final/modules/preprocess.py      ← read-only, 전처리
_scratch/lot_wafer_baseline_compare.ipynb  ← lot/wafer encoding 출처
_scratch/lot_wafer_alpha_grid.ipynb        ← α grid sweep
4_output/final/zit_only/optuna_jh_zit-final-100.db  ← ZIT 180 trials (탐색공간 anchor)
```

---

## 8. 비교 기준선 (참고)

| 모델 | EXP | OOF | val | test | 비고 |
|---|---|---|---|---|---|
| ZIT 단독 (180 trial best) | zit-final-100 | 0.005501 | — | — | origin/main 01_zit_only |
| ZIT 단독 (1 trial) | zit-final-999 | 0.005592 | 0.005772 | 0.008450 | 1 trial 빠른 비교용 |
| **BagZIT (HP fixed)** | bag-zit-temp-001 | **0.005524** | **0.005729** | **0.008428** | 본 실험 |
| Two-Stage + 12 enc (v3) | scratch | 0.005489 | 0.005706 | 0.008403 | lot/wafer encoding 효과 (Two-Stage) |
| **BagZIT HPO** | bag-zit-hpo-001 | TBD | TBD | TBD | 실행 대기 |
| **BagZIT + 12 enc** | (예정) | TBD | TBD | TBD | 다음 단계 |

---

## 9. 주요 근거 / 참고

- **Gu 2024 (논문 2-16)**: ZI-Tweedie + LightGBM EM (ZITboost 원논문)
- **CLAUDE.md 7-B2 섹션**: ZITboost 적용 가이드
- **EDA 결과**: max\|r\|=0.037 (단일 die feature와 health), Y=0 비율 70.8%
- **B3 직관**: "예측 합 = 실제 unit y" 학습 목표가 우리 RMSE 평가 지표와 정확히 일치

---

## 10. 알려진 제약 / 향후 과제

1. **Y>0 unit 3× under-predict**: BagZIT은 RMSE 우위에도 불구하고 고불량 unit 식별은 약함 → 후처리 또는 추가 feature 필요
2. **n_em_iters 효율성**: 베이스라인에서 EM iter 1이 거의 best, iter 2+ 미세 변동만 → HPO에서 짧은 n_em_iters가 best로 나올 가능성
3. **die-level signal 한계**: max\|r\|=0.037 자체가 천장 — feature engineering (lot/wafer encoding 등) 없이는 모델 구조만 바꿔도 큰 개선 어려움
4. **후처리 누락**: zero_clip 등 사후 적용 가능하나 현재 단계에서는 미적용
5. **단일 seed**: SEED=42 하나로만 학습. 안정성 검증을 위해 multi-seed 평균 필요할 수도

---

(끝)
