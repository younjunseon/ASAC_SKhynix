# ZITboost 구현 계획

## 배경

1차 실험 5,750 trial 분석 결과, Two-Stage의 Stage 1 분류 성능이 **Recall 평균 0.011, AUC 0.576**으로 거의 랜덤 수준.
`P(Y>0) × E[Y|Y>0]` 곱셈이 예측을 0으로 수렴시키는 구조적 문제 → **분류와 회귀를 joint loss로 동시 학습하는 ZITboost**로 우회.

## 설계 결정 (2026-04-17 확정)

| 항목 | 결정 | 이유 |
|------|------|------|
| 통합 방식 | **별도 노트북** (e2e_hpo.py 수정 없음) | ZITboost는 내부 π로 분류 처리 → 외부 clf 불필요. 기존 `run_e2e_optimization_with_pp(run_clf=False, reg_models=['zitboost'])` 호출 |
| HP 분리 | **μ full(9) + π medium(5) + φ medium(5) + ZIT(2) = 21개** | μ가 핵심. π/φ는 보조 모델이므로 축소 |
| position weight | **그대로 가져감** | die-level 학습 → 4 die 예측 → position 가중치로 unit 집계 |
| target transform | **`'none'` 고정** | Tweedie가 분포 자체를 모델링하므로 외부 변환 불필요 |
| clf | **비활성 (`run_clf=False`)** | ZITboost 내부 π가 분류 담당 |

## 핵심 구조

```
Y_i = 0        확률 π(x_i)           ← structural zero (정상 제품)
Y_i ~ Tweedie(μ_i, φ_i, ζ)  확률 1-π(x_i)  ← Tweedie 상태

최종 예측: E[Y] = (1-π) × μ
```

- LightGBM 3개: `lgb_pi`(zero 확률), `lgb_mu`(Tweedie 평균), `lgb_phi`(분산)
- EM 알고리즘으로 동시 학습 (5~20회 반복)
- Two-Stage와 달리 **전체 데이터 100%** 사용, 분류 약해도 μ가 보상

### Tweedie 수학 공식

```
P(Y=0 | Tweedie) = exp(-μ^(2-ζ) / ((2-ζ)φ))     [compound Poisson-Gamma]
Var(Y | Tweedie) = φ × μ^ζ

E-step (y_i=0일 때):
  Π_i = π_i / [π_i + (1-π_i) × f_Tw(0; μ_i, φ_i, ζ)]

M-step:
  lgb_pi:  objective='binary',  target=Π_i
  lgb_mu:  objective='tweedie', weight=(1-Π_i)/φ_i
  lgb_phi: objective='gamma',   weight=(1-Π_i), target=deviance_residual²
```

## 작업 목록

### A. 모듈 작성 — `modules/zi_tweedie.py` (~250줄)

| # | 작업 | 상세 |
|---|------|------|
| A1 | ZITboostRegressor 클래스 골격 | `BaseEstimator, RegressorMixin` 상속. `__init__`에 21개 HP를 3그룹 수용: `mu_*`(9), `pi_*`(5), `phi_*`(5), `zeta`, `n_em_iters` |
| A2 | Tweedie 수학 유틸 | `_tweedie_p0(mu, phi, zeta)`: P(Y=0). `_tweedie_variance(mu, zeta)`: φ×μ^ζ |
| A3 | `_initialize()` | Y>0 mean으로 μ, Tweedie deviance 기반 φ, 전체 zero 비율로 π 초기화 |
| A4 | `_e_step()` | y=0: `Π_i = π_i / [π_i + (1-π_i)×f_Tw(0)]`, y>0: `Π_i = 0` |
| A5 | `_m_step()` | lgb_pi: binary + target=Π_i. lgb_mu: tweedie + weight=(1-Π_i)/φ_i. lgb_phi: gamma + weight=(1-Π_i) |
| A6 | `fit()` 통합 | 초기화 → (E→M)×n_em_iters. 매 iter `em_history_` 저장 (RMSE, Π 통계) |
| A7 | `predict()` + `predict_components()` | predict: `clip((1-π)×μ, 0)`. components: `(π, μ, φ)` 튜플 (π threshold 후처리용) |
| A8 | 코드 검증 | 문법, import, sklearn 호환, `get_params()`/`set_params()` 동작 |

### B. 기존 모듈 수정 (최소)

| # | 작업 | 상세 |
|---|------|------|
| B1 | `search_space.py` — `zitboost_space()` 추가 | μ: 9 HP (n_est, lr, leaves, depth, min_child, subsample, colsample, alpha, lambda). π: 5 HP (n_est, lr, leaves, depth, min_child). φ: 5 HP (동일). ZIT: zeta, n_em_iters. 총 21개 |
| B2 | `model_zoo.py` — MODEL_REGISTRY 등록 | `"zitboost": {"clf": None, "reg": ZITboostRegressor, "supports_early_stopping": False}` (3줄) |

### C. 실험 노트북 — `zitboost/zitboost_experiment.ipynb` (ensemble_2nd 미러링)

| # | 셀 | 작업 | ensemble_2nd 대응 |
|---|-----|------|-------------------|
| C1 | 1-2 | Colab/Local setup + import | 동일 + `from modules.zi_tweedie import ZITboostRegressor` |
| C2 | 3-4 | 실험 설정 (스위치) | `run_clf=False`, `reg_models=['zitboost']`, `TARGET_TRANSFORM='none'`, **`postprocess_config`** 추가 |
| C3 | 5-6 | 전처리 candidates + 범위 축소 | 동일 (1차 best 전처리로 축소) |
| C4 | 7-8 | 타겟 변환(none 고정) + 데이터 로드 + 샘플링 | 동일 구조 |
| C5 | 9-10 | Optuna HPO | `run_e2e_optimization_with_pp(run_clf=False, reg_models=['zitboost'])` |
| C6 | 11-12 | Rerun best trial (K-fold OOF) + CSV 저장 | 동일 |
| C7 | 13-14 | 성능 지표 + EM convergence 진단 | metrics.csv + EM iter별 RMSE/Π 통계 |
| C8 | 15-16 | **집계 함수 탐색** + position 가중치 | **신규**: mean/weighted/max/min/median/trimmed_mean 비교 테이블 |
| C9 | 17-18 | **π threshold tuning** | **신규**: predict_components()로 π 추출 → grid search (0.5~0.95) |
| C10 | 19-20 | **zero_clip** + 최종 예측 조합 + CSV 저장 | 기존 패턴 + Schema C 호환 |
| C11 | 21-22 | Two-Stage OOF 비교 + 시각화 (3×3) | EM convergence, π 분포, residual, y_true vs y_pred |
| C12 | — | 노트북 JSON 검증 | 이스케이프, 경로, 변수명 전수 점검 |

## 후처리 파이프라인 (3단계)

```
die-level 예측 (4개/unit)
    │
    ▼ C8: 집계 함수 탐색
    ├─ mean / weighted(SLSQP) / max / min / median / trimmed_mean
    ├─ 각각 val RMSE 계산 → 비교 테이블 출력
    └─ best 선택 (스위치: agg_method = 'auto' or 고정)
    │
    ▼ C9: π threshold tuning (ZITboost 전용)
    ├─ predict_components()로 π 추출
    ├─ grid search: π > threshold → pred=0 (range: 0.5~0.95)
    └─ best threshold 선택 (스위치: pi_threshold = 'auto' or 고정 or None)
    │
    ▼ C10: zero_clip (기존 패턴)
    ├─ pred < threshold → 0 (range: 0.001~0.015)
    └─ 최종 CSV 저장
```

## Optuna HPO 탐색 축 (21개)

### μ 모델 (핵심, 9개)
| 파라미터 | 범위 | 비고 |
|----------|------|------|
| `mu_n_estimators` | 100 ~ 2000 | |
| `mu_learning_rate` | 0.005 ~ 0.1 (log) | |
| `mu_num_leaves` | 32 ~ 256 | |
| `mu_max_depth` | 5 ~ 12 | |
| `mu_min_child_samples` | 5 ~ 100 | |
| `mu_subsample` | 0.5 ~ 1.0 | |
| `mu_colsample_bytree` | 0.3 ~ 1.0 | |
| `mu_reg_alpha` | 1e-8 ~ 10.0 (log) | |
| `mu_reg_lambda` | 1e-8 ~ 10.0 (log) | |

### π 모델 (분류, 5개)
| 파라미터 | 범위 | 비고 |
|----------|------|------|
| `pi_n_estimators` | 50 ~ 500 | μ보다 축소 |
| `pi_learning_rate` | 0.01 ~ 0.1 (log) | |
| `pi_num_leaves` | 16 ~ 128 | |
| `pi_max_depth` | 3 ~ 8 | |
| `pi_min_child_samples` | 10 ~ 100 | |

### φ 모델 (분산, 5개)
| 파라미터 | 범위 | 비고 |
|----------|------|------|
| `phi_n_estimators` | 50 ~ 500 | μ보다 축소 |
| `phi_learning_rate` | 0.01 ~ 0.1 (log) | |
| `phi_num_leaves` | 16 ~ 128 | |
| `phi_max_depth` | 3 ~ 8 | |
| `phi_min_child_samples` | 10 ~ 100 | |

### ZIT 전용 (2개)
| 파라미터 | 범위 | 비고 |
|----------|------|------|
| `zeta` (Tweedie power) | 1.1 ~ 1.9 | 1.0=Poisson, 2.0=Gamma 사이 |
| `n_em_iters` | 3 ~ 15 | EM 반복 횟수 |

## 노트북 스위치 전체 목록

```python
# ── 실험 식별 ──
EXP_ID = 'zit-001'
USER = 'jh'
EVAL_VAL = True
EVAL_TEST = True
SAVE_OUTPUTS = True

# ── Pipeline ──
pipeline_config = dict(
    input_level='die',
    run_clf=False,              # ZITboost 내부 π 사용
    reg_level='position',       # die-level 학습 → position weight 집계
    reg_optuna=True,
    zero_clip=True,
)

# ── E2E ──
e2e_params = dict(
    reg_models=['zitboost'],
    n_trials=100,
    n_folds=3,
    reg_early_stop=50,
)

# ── 후처리 ──
postprocess_config = dict(
    agg_method='auto',          # 'auto'=전체 탐색, 'mean'/'weighted'/... 고정
    agg_candidates=['mean', 'weighted', 'max', 'min', 'median', 'trimmed_mean'],
    position_weight_range=(0.15, 0.35),

    pi_threshold='auto',        # 'auto'=grid search, float=고정, None=비활성
    pi_threshold_range=(0.5, 0.95),
    pi_threshold_step=0.01,

    zero_clip_threshold='auto',
    zero_clip_range=(0.001, 0.015),
)

# ── 타겟 ──
TARGET_TRANSFORM = 'none'      # 고정
CLIP_Y_EXTREME = True
```

## 코드 수정 영향도

```
modules/zi_tweedie.py     ← 신규 (~250줄)
modules/search_space.py   ← 추가 (~30줄, 기존 무수정)
modules/model_zoo.py      ← 추가 (3줄, 기존 무수정)
modules/e2e_hpo.py        ← 수정 없음 ✓
utils/*                   ← 수정 없음 ✓
zitboost/experiment.ipynb ← 신규
```

## Two-Stage vs ZITboost 핵심 차이

| | Two-Stage | ZITboost |
|---|---|---|
| 구조 | 분류 → 회귀 (순차) | π + μ + φ 동시 학습 (EM) |
| 오차 전파 | 분류 오차가 곱셈으로 증폭 | joint loss로 상호 보완 |
| 학습 데이터 | Stage 2는 Y>0 29.2%만 | 전체 100% 사용 |
| 분류 성능 의존 | 치명적 (Recall 0.01이면 무력화) | 분류 약해도 μ가 보상 |
| 후처리 | clf_proba × R_combined | (1-π)×μ → agg → π threshold → zero_clip |
| position weight | SLSQP only | **6종 집계 함수 탐색** + SLSQP |

## 논문 근거

| 논문 | 핵심 기여 |
|------|----------|
| 2-16 (Gu 2024, ZITboost) | ZI-Tweedie + LightGBM EM, varying π(x)/φ(x), semicontinuous SOTA |
| 2-12 (Feng 2021, ZI vs Hurdle) | ZI vs Hurdle 체계적 비교, zero-deflation 시 Hurdle 우세 |
| 2-13 (Sidumo 2023, ML vs ZIR) | ML(SVM/RF)이 ZIP/ZINB outperform → 트리 기반 접근 정당화 |