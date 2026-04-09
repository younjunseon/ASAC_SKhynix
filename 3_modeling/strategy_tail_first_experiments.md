# Tail-First 실험 전략 — 꼬리(high/extreme y) 공략 우선

> **작성일**: 2026-04-09
> **베이스 노트북**: [3_modeling/e2e_twostage.ipynb](e2e_twostage.ipynb) (현 best Val RMSE = **0.005810**, LGBM Two-Stage, 742 features)
> **근거 문서**:
> - [99_학습자료/스터디자료/미반영_항목_검토.md](../99_학습자료/스터디자료/미반영_항목_검토.md)
> - [1_eda/y_dist_check.ipynb](../1_eda/y_dist_check.ipynb)
> - [CLAUDE.md](../CLAUDE.md) Stage 4·7-B·8

---

## 1. 문제 진단 — 왜 "꼬리"부터인가

`y_dist_check.ipynb`에서 드러난 핵심 사실:

| 관찰 | 위치 | 시사점 |
|---|---|---|
| train max=**1.0** (다음 값 0.0974, 10배 차이) | cell-4, cell-27 | 단일 극단치가 L2 gradient를 지배할 가능성 매우 높음 |
| train `high` 69개 / `extreme` 8개 unit | cell-23 | Stage 2 회귀의 꼬리 학습 신호 극도 희소 |
| y>0 서브셋 값 범위 0.001~1.0 (1000배 skew) | cell-6 | L2 loss는 꼬리 한두 점이 좌우함 |
| train/val/test 분포 KS p>0.7 | cell-10 | 공변량 shift 없음 → pseudo-labeling 전제 성립 |
| val+test X = 17,498 unit (train의 67%) 미사용 | — | 학습 데이터 67% 유휴 |
| extreme-y 유닛이 결측/이상치 **더 적음** | cell-23 | "측정 잘 된 unit일수록 고불량" — missing indicator 방향이 직관과 반대 |

**결론**: 현재 모델의 주된 병목은 **꼬리(high/extreme y) 처리**이다. RMSE 제곱 특성상 꼬리 몇 점이 전체 점수를 좌우하므로, 꼬리를 공략하는 기법을 가장 먼저 실험한다.

---

## 2. 현재 적용 상태 (기준선)

`e2e_twostage.ipynb` cell-4 config 기준:

- **Two-Stage**: LGBM CLF (`proba`) × LGBM REG, `scale_pos_weight`
- **집계 7종**: `['mean','std','cv','range','min','max','median']` — Q25/Q75/skew/kurt 미포함
- **전처리**: const/고결측(≥25%)/중복/고상관(>0.95) 제거, `add_indicator=True`, spatial imputation, winsorize upper 0.5%
- **회귀 레벨**: `reg_level='position'`
- **HPO**: Optuna 15 trials × 3-fold, 5-fold rerun
- **앙상블**: LGBM/XGB/CatBoost/RF blend+stacking

**건드리지 않을 것**: 위 구조. 본 전략은 **Stage 2 loss/target 변환 + 집계 보강 + 데이터 증강**으로 꼬리를 공략한다.

---

## 3. 우선순위 정리

### 3-1. 🔥 P1 — 바로 실험 (파라미터 한두 개 변경)

| # | 실험 | 근거 | 구현 |
|---|------|------|------|
| **E1** | Stage 2 `log1p(y)` target 변환 | 2-8, y_dist_check tail | reg 학습 직전 `y_tr = np.log1p(y>0)`, 예측 후 `np.expm1()` |
| **E2** | Stage 2 Tweedie objective | 2-5, 2-7 | `reg_fixed={'objective':'tweedie','tweedie_variance_power':1.2}` |
| **E3** | 집계에 **Q25, Q75** 추가 | CLAUDE.md Stage 4 9종 기본 | `agg_funcs += ['q25','q75']` |
| **E4** | 단일 극단치 S39398 ablation | y_dist_check cell-27 | ① 그대로 ② `y_clip=0.2` ③ 해당 unit 제외 3-way 비교 |

**E1/E2는 독립 실험으로 각 1회씩** 돌려 Val RMSE 변화를 확인한 뒤, 더 나은 쪽을 default로 확정.

### 3-2. 🟡 P2 — P1 결과 확정 후

| # | 실험 | 근거 | 구현 난이도 |
|---|------|------|:---:|
| **E5** | LDS re-weighting (Stage 2 sample_weight) | 8-1 | 낮음 (`sample_weight`만) |
| **E6** | Pseudo-labeling (CLF 먼저) | 9-1 | 중 (학습 루프 훅 1곳) |
| **E7** | Pseudo-labeling (REG, confident only) | 9-1 | 중 |
| **E8** | `clf_filter=True` + Youden J threshold | 2-8 | 낮음 |

### 3-3. 🟢 P3 — 장기 고려

| # | 실험 | 근거 |
|---|------|------|
| E9  | Attention MIL die→unit 집계 | 10-1 (구현비용 큼) |
| E10 | AutoGluon baseline (상한선 확인용) | 8-4 |
| E11 | PIMP p-value 피처 선택 | 5-5 |
| E12 | SHAP Stage 1/2 분리 해석 | 2-2 (프로젝트 목표 2번, 성능엔 직접 기여 없음) |

### 3-4. ❌ 기각 / 낮은 우선순위

| 항목 | 사유 |
|---|---|
| Separated scaling per lot (10-2) | 트리 모델엔 스케일 무의미. 선형/딥러닝 쓸 때만. |
| Fused LASSO (1-8) | 피처 비식별화로 "인접"의 물리 의미 없음. |
| 밀도 영역 피처 (3-2) | EDA Phase 23에서 radial/edge 무상관 확인. |
| NNR 공간 잔차 | EDA Phase 24에서 개선 feature 0개. |
| 전체 일괄 z-score | EDA Phase 21에서 76.9% feature 악화. |

---

## 4. 실험 실행 순서

```
[Baseline: Val RMSE 0.005810]
    ↓
(Step 1) E4 진단 — S39398 영향 격리
    ↓ train max=1.0이 차지하는 RMSE 비중 확인 후
(Step 2) E1 (log1p) 단독 실험
    ↓
(Step 3) E2 (Tweedie) 단독 실험 + variance_power 2~3점 sweep
    ↓ E1 vs E2 중 승자 확정
(Step 4) + E3 (Q25/Q75) 누적
    ↓ 여기까지가 "손쉬운 저수준 개선"
(Step 5) + E5 (LDS) 누적
    ↓
(Step 6) + E6 (CLF pseudo-label) — 1 round만 먼저
    ↓ val RMSE 개선이면 (Step 7), 아니면 중단
(Step 7) + E7 (REG pseudo-label, confident only)
    ↓
(Step 8) + E8 clf_filter + Youden 탐색 (optional)
```

**Checkpoint 규칙**: 각 Step은 단독 실험으로 5-fold rerun Val RMSE 기록. **악화 시 해당 변경 롤백**. test는 모든 Step에서 `EVAL_TEST=False` 유지, 최종 제출 직전에만 켠다.

---

## 5. 구현 위치 메모

### 5-1. E1 / E2 / E3 — 셀 4 config 수정만

[3_modeling/e2e_twostage.ipynb](e2e_twostage.ipynb) cell-4:

```python
# E2 (Tweedie)
e2e_params = dict(
    ...
    reg_fixed={'objective': 'tweedie', 'tweedie_variance_power': 1.2},
)

# E3 (Q25/Q75)
e2e_params['agg_funcs'] = ['mean','std','cv','range','min','max','median','q25','q75']
```

**E1 (log1p)은 셀 수정으로 안 됨** — [3_modeling/modules/e2e_hpo.py](modules/e2e_hpo.py)의 `_run_reg_oof` (또는 rerun 회귀 호출부) 에 `reg_target_transform` 옵션을 추가해야 한다. LGBM `fit` 직전에 `y_tr = np.log1p(y_tr)`, `predict` 직후 `np.expm1(pred)`. 음수 pred는 clip 후 expm1. `reg_fixed`와 직교하는 스위치로 둔다.

### 5-2. E4 — 진단 전용 분기

셀 9(health merge) 직후에 분기 3개:

```python
EXP_E4_MODE = 'raw'  # 'raw' | 'clip_02' | 'drop_s39398'

if EXP_E4_MODE == 'clip_02':
    die_train[TARGET_COL] = die_train[TARGET_COL].clip(upper=0.2)
elif EXP_E4_MODE == 'drop_s39398':
    die_train = die_train[die_train[KEY_COL] != 'S39398'].reset_index(drop=True)
```

> **주의**: val/test는 절대 건드리지 않는다. 평가 지표는 원본 y 기준.

### 5-3. E5 — LDS sample_weight

`modules/e2e_hpo.py`의 reg 학습부에 한 줄 훅. y>0 히스토그램을 Gaussian kernel로 smoothing한 뒤 `w = 1 / density(y_i)`로 정규화. scipy gaussian_kde 1개로 충분. CV 안쪽에서 train fold만 계산 (leakage 방지).

### 5-4. E6 / E7 — Pseudo-labeling 루프

새 함수 `run_pseudo_label_round(result, tau_low=0.05, tau_high=0.8)`:

1. 기존 E2E 결과로 val+test X에 대해 `P(Y>0)` 예측
2. `P(Y>0) < tau_low` → pseudo-label 0 (CLF 학습 데이터 추가)
3. `P(Y>0) > tau_high` → pseudo-label = reg 예측값 (Stage 2 학습 데이터 추가)
4. 통합 데이터로 E2E 재학습 → Val RMSE 비교
5. 개선되면 tau를 점진적 완화하며 round 2 (curriculum)

**반드시 val RMSE만 보고 판단**. test는 보지 않는다.

---

## 6. 성공 기준 & Ablation 표

실험 완료 시 `strategy_tail_first_experiments.md`에 아래 표를 누적 기록한다:

| Step | 실험 | Val RMSE | Δ vs Baseline | Δ vs 직전 | 결정 |
|:---:|---|---:|---:|---:|:---:|
| 0 | Baseline | 0.005810 | — | — | ✅ |
| 1 | E4 raw/clip/drop 비교 | | | | ⬜ |
| 2 | E1 log1p | | | | ⬜ |
| 3 | E2 tweedie (power sweep) | | | | ⬜ |
| 4 | + E3 (Q25/Q75) | | | | ⬜ |
| 5 | + E5 (LDS) | | | | ⬜ |
| 6 | + E6 (CLF pseudo) | | | | ⬜ |
| 7 | + E7 (REG pseudo) | | | | ⬜ |
| 8 | + E8 (clf_filter+Youden) | | | | ⬜ |

**판정 룰**:
- 개선 ≥ 0.00005 → 채택
- 변화 |Δ| < 0.00005 → 복잡도 낮은 쪽 유지
- 악화 → 롤백, 기각 사유 기록

---

## 7. 열어둔 질문

- **Q1**: E4에서 S39398 1개가 RMSE 기여도의 몇 %를 차지하는지 먼저 수치 확인(fold별 제곱오차 합 기여도)하면 E1/E2 기대효과 해석이 더 명확.
- **Q2**: Tweedie variance_power는 {1.1, 1.3, 1.5, 1.7} 4점 sweep이면 충분할지 vs Optuna 연속 탐색으로 돌릴지.
- **Q3**: Pseudo-labeling에서 CLF만 증강하고 REG는 그대로 두는 안(E6 only) vs 둘 다 증강(E6+E7) 비교. 보수적으로는 CLF only가 안전.
- **Q4**: E3 (Q25/Q75 추가) 시 피처 수 742 → 742 × (9/7) ≈ 954. `run_fs=False` 현 설정 그대로 둘지, `run_fs=True` + `top_k_fixed=200` 켤지.

---

## 8. 메모

- 본 문서는 **RMSE 최소화 전용**. 프로젝트 목표 2번(공정 개선 인사이트)는 E12(SHAP)로 분리.
- 앙상블(blend/stacking)은 현 구조를 유지. 각 P1/P2 실험은 **single E2E rerun의 Val RMSE**로 판정한 뒤, 최종 확정본에만 앙상블을 다시 얹는다.
- 실험 로그는 `log_experiment` 기존 스키마를 그대로 쓰고, `EXP_MEMO`에 `E1-log1p`, `E2-tweedie-1.2` 같은 태그만 추가한다.
