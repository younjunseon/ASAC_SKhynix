# Final 파이프라인 — 성능 개선 계획 v2

> 작성: 2026-04-27 (v1 → v2 갱신)
> 통합: 코드 검증 + GPT-5.5 피드백 + 데이터 재진단
> 베이스: `origin/main` (3_modeling/final/, 4_output/final/)
>
> **현재 best**: `zit_only` val RMSE ≈ 0.005703~0.005706 (anchor)
> 단일 reg ≈ 0.005733, two-stage·blending 모두 zit_only 미달

---

## 0. 의사결정 원칙

| 원칙 | 이유 |
|---|---|
| **비교 기준 = val RMSE** | test는 outlier 영향 큼(아래 0-2). HPO·실험 비교는 val 기준. test는 참고 |
| **항상 segment RMSE 같이 출력** | 전체 / Y=0(70.8%) / Y>0(29.2%) / Y bin 4종. 어디서 이기고 어디서 지는지 봐야 채택 가능 |
| **ZIT는 anchor — 희석 금지** | ZIT의 zero RMSE 강점(0.0025)이 어떤 변형에서도 0.0001 이상 악화되면 폐기 |
| **분포 shift 단정 금지** | val→test gap은 outlier 분포 차이일 가능성. AV로 진단 후 결정 |

---

## 1. 현재 진단 (3가지 별개 문제)

### 1-1. Segment 문제 — 가장 결정적

val 기준 segment 분해:

| 모델 | val 전체 | true zero RMSE (n=6,194, 70.8%) | true positive RMSE (n=2,555, 29.2%) |
|---|---|---|---|
| **ZIT** (anchor) | **0.005709** | **0.002471** ✅ | 0.009838 |
| reg_lgbm | 0.005731 | 0.002463 | 0.009887 |
| **ZIT+reg** | 0.005766 ❌ | **0.003194** ❌ | 0.009439 ✅ |
| blend | 0.005710 | 0.002478 | 0.009836 |

**핵심**: ZIT+reg는 positive를 잡아오는 대가로 zero를 망쳐 net 손해.
- positive 1단위 SE = 0.0098² = 0.000096
- zero 1단위 SE = 0.0025² = 0.0000063
- positive 1개당 영향력이 zero의 15배지만, zero 비중이 70.8%라 **zero가 전체 SE의 14%, positive가 86% 차지**

→ 모델 비교 시 전체 RMSE 차이 0.0001 안에서는 segment를 봐야 진짜 win/loss 보임.

### 1-2. val→test gap 진단 정정

```
              zero%   mean      median    q90      q99      max
train (26,247): 70.80%  0.002515  0.0      0.00923  0.02424  1.000
val   (8,749):  70.80%  0.002505  0.0      0.00921  0.02381  0.172
test  (8,749):  70.80%  0.002610  0.0      0.00946  0.02503  0.602
```

- zero%·mean·median·q90·q99: 세 split 거의 동일
- **max만 다름**: train 1.0, val 0.17, test 0.60

val RMSE 0.005710 → SE 합 ≈ 0.285
test RMSE 0.008414 → SE 합 ≈ 0.620
test의 max=0.602 unit 1개를 모델이 0.01로 예측만 해도 SE += 0.35

→ **test SE 차이의 거의 전부가 outlier 1개에서 옴**. 분포 shift라고 단정 금지. val/test gap은 평가지표 outlier 민감성 문제일 가능성이 큼.

### 1-3. Two-stage 곱셈 구조 위험

[hpo.py:515](modules/hpo.py#L515) `y_positive_only=True` (Y>0만 학습) + [hpo.py:573](modules/hpo.py#L573) `(1-π_zit)` 곱셈 = **이중 zero suppression**:
- Y>0만 학습 → reg는 항상 양수 출력 + zero 영역 미학습
- (1-π) 곱셈 → π 높은 곳을 추가로 누름
- net 효과 = zero RMSE 0.0025 → 0.0032 (28% 악화) → 1-1 segment 표 그대로

---

## 2. 우선순위 (Tier 0 → 4)

| Tier | 분류 | 시간 | 위험 | 효과 기대 |
|---|---|---|---|---|
| **0** | 진단 (3종) | 반나절 | 없음 | 의사결정 분기 |
| **1** | ZIT anchor 보강 | 1~2일 | 낮음 | 1e-6 ~ 5e-5 |
| **2** | Two-stage 재정의 (4 변형) | 2~3일 | 중 | zero 보존하며 positive 개선 가능 |
| **3** | 분포 shift 대응 | Tier 0-2 결과에 따라 | 중 | AUC>0.6일 때만 |
| **4** | 보류 | — | — | 낮음 또는 음의 효과 |

---

## 3. Tier 0 — 진단 (먼저 무조건)

### 3-1. Segment RMSE 고정 지표

**대상**: `4_output/final/` 의 oof/val/test CSV 6벌(zit, reg_lgbm, reg_catboost, reg_et, reg_enet, zitreg_lgbm).

**산출**:
```
표 1: 모델 × split × (전체 / Y=0 RMSE / Y>0 RMSE) — 4 모델 × 3 split × 3 metric = 36칸
표 2: 모델 × Y bin (Y=0, (0,0.005), [0.005,0.015), [0.015,0.05), [0.05,1.0]) × RMSE + 기여도(SE 합/전체 SE)
```

**위치**: 새 노트북 `3_modeling/final/_diagnostic.ipynb` (가벼운 분석 노트북)
**모듈 변경**: 없음
**판단 기준**:
- ZIT 대비 zero RMSE가 0.0025 → 0.0026 이상 악화되면 그 변형 폐기
- positive RMSE가 0.0001 이상 개선 + zero 0.0001 이내 악화면 채택 검토

### 3-2. Adversarial Validation

**대상**: train+val concat → binary classifier로 split 구분 가능한지 (AUC).

**산출**:
- AUC train↔val (1 숫자)
- AUC > 0.55면 importance top-20 feature 출력

**판단 기준**:
- **AUC < 0.55**: 분포 거의 같음 → Tier 3 보류
- **0.55 ≤ AUC < 0.65**: 약한 drift, 진단만
- **AUC ≥ 0.65**: drift 확정 → Tier 3 진행

**위치**: `_diagnostic.ipynb` 의 같은 노트북에 셀 추가
**모듈 변경**: 없음

### 3-3. GroupKFold(by=run_id) 진단 (대체 아님)

**대상**: 현재 best HP로 1회만 실험. 같은 ZIT 학습을 random KFold(현재) vs GroupKFold(by=run_id)로 비교.

**산출**:
- random KFold OOF RMSE (현재 0.00550)
- GroupKFold OOF RMSE
- 차이 = lot leak으로 인한 OOF 낙관도 정량화

**판단 기준**:
- 차이 < 0.0003: 거의 leak 없음. random KFold 그대로 사용
- 0.0003 ≤ 차이 < 0.0010: 약한 leak, 인지하되 random KFold 유지
- 차이 ≥ 0.0010: lot leak 큼. **OOF 절대 수치 신뢰도 재평가 필요** (모든 비교 기준 재고)

**중요**: 이 결과로 즉시 GroupKFold로 갈아타지 말 것. val 기준이라 KFold 자체를 바꾸면 평가지표 괴리. 어디까지나 진단.

**위치**: `_diagnostic.ipynb` 또는 별도 한번 실행 스크립트
**모듈 변경**: 없음 (KFold만 임시 교체)

---

## 4. Tier 1 — ZIT anchor 보강

### 4-1. ZIT seed bagging

**대상**: 현재 `4_output/final/zit_only/best_params.json` 기준 ZITboost를 seed 5~10개로 재학습 후 die-level 예측 평균.

**구현**:
- 새 노트북 `3_modeling/final/05_zit_bagging.ipynb`
- best HP 고정, fold split 동일 (`unit_ids_hash` 검증)
- model `random_state` 만 변경: 42, 43, 44, 45, 46
- die-level OOF/val/test 예측 평균 → 후처리는 기존 ZIT 단일 결과와 동일하게

**기대 효과**: 안전. 1e-6 ~ 5e-5 수준. zero 강점 보존.
**위험**: 학습 시간 5~10배. EM 수렴은 동일 HP라 안정.

### 4-2. Learned unit aggregator

**대상**: die-level 4개 예측 + π + μ + 통계량 → unit-level 집계 학습.

**입력 (unit당)**:
```
pred_p1, pred_p2, pred_p3, pred_p4
pi_p1, pi_p2, pi_p3, pi_p4
mu_p1, mu_p2, mu_p3, mu_p4
mean(pred), median(pred), std(pred), min(pred), max(pred), max-min(pred)
```

**meta 모델**: Ridge(α=0.01~10 grid) 또는 shallow LGBM(num_leaves=15, max_depth=3, n_estimators=100, 강한 정규화)

**필수**:
- meta 학습 = **train OOF die prediction** (final fitted prediction X — leak)
- val/test = fold 평균 die prediction

**구현 위치**: 새 노트북 `3_modeling/final/06_unit_aggregator.ipynb`
**모듈 변경**: 없음 (Ridge/LGBM은 sklearn/lightgbm 직접 호출)

**기대 효과**: mean/median 고정 집계보다 정보량 큼. positive 영역에서 die 분산 활용 가능.
**위험**: meta 과적합 (특히 LGBM). Ridge 우선.

### 4-3. zeta search range 확장

**대상**: 현재 zeta search range (1.1, 1.4), best=1.149 (boundary hit).

**변경**: [models.py:194](modules/models.py#L194) `zeta=trial.suggest_float("zeta", 1.01, 1.95)` 1줄.

**구현**: `01_zit_only.ipynb` REUSE_BEST_PARAMS=False로 재실행. n_trials는 기존과 동일.

**기대 효과**: 70.8% zero 데이터에서 zeta 1.0(Poisson) 근처가 적합할 가능성. EM 수렴 안정성 모니터링 필수.
**위험**: zeta 너무 낮으면(<1.05) `_tweedie_p0` 계산 불안정. n_em_iters 모니터링.

---

## 5. Tier 2 — Two-stage 재정의 (4 변형 비교)

현재 03 (`y_positive_only=True` + 무조건 (1-π) 곱셈)은 zero 망침 → 변형 4종 비교.

### 5-A. y_positive_only=False
```
reg 학습: 전체 die (Y=0 포함)
predict: reg_pred × (1-π_zit)
```
- 학습 데이터 100% 활용
- 곱셈은 유지 (zero 영역에서 reg가 작은 양수 학습 → 곱셈으로 더 작게)

### 5-B. 곱셈 약화
```
reg 학습: y_positive_only=True (현재)
predict: reg_pred × (1-π_zit)^λ,  λ ∈ {0.3, 0.5, 0.7, 1.0}
```
- λ < 1이면 곱셈 효과 완화. λ=1=현재, λ=0=reg 단독

### 5-C. Gated residual (GPT 핵심 제안)
```
zit_pred 그대로 가져오기
residual = y - zit_pred (die-level broadcast) 로 LGBM 학습
gate(π) = 0 if π > 0.95 else (1-π)
final = zit_pred + λ × gate(π) × residual_pred,  λ ∈ {0.05, 0.1, 0.2}
```
- zero 영역(π 높음) 완전 보호
- positive 영역에서만 보정

### 5-D. Hard-gated positive expert
```
final = zit_pred 기본
if (1-π_zit) > 0.7 and zit_pred > 0.005:
    final = w × zit_pred + (1-w) × reg_pred,  w ∈ {0.3, 0.5, 0.7}
```
- ZIT가 양수라고 강하게 예측한 unit에서만 reg 의견 반영

**비교 방법**: 같은 ZIT OOF/val/test 기반으로 4종 다 돌려서 segment 표 출력.

| 변형 | val 전체 | val zero | val pos | 채택? |
|---|---|---|---|---|
| zit only | 0.00571 | 0.00247 | 0.00984 | (anchor) |
| 5-A | ? | ? | ? | zero ≤ 0.00257 + pos < 0.00984면 ○ |
| 5-B (best λ) | ? | ? | ? | 동일 기준 |
| 5-C (best λ) | ? | ? | ? | 동일 기준 |
| 5-D (best w) | ? | ? | ? | 동일 기준 |

**구현 위치**:
- 5-A: `03_zit_plus_reg.ipynb` cell 12에서 `y_positive_only=True` → `False` 토글
- 5-B: `03_zit_plus_reg.ipynb` cell 12 multiplier 함수 수정 (`omp_train` → `omp_train**λ`)
- 5-C: 새 노트북 `3_modeling/final/07_gated_residual.ipynb`
- 5-D: 새 노트북 `3_modeling/final/08_positive_expert.ipynb`

---

## 6. Tier 3 — 분포 shift 대응 (조건부)

**조건**: Tier 0-2 AUC ≥ 0.65일 때만 진행.

### 6-1. drift feature 제외
0-2의 importance top-20 feature를 EXCLUDE_COLS에 추가. `preprocess.py` `exclude_cols` 인자로 전달. 모듈 무수정.

### 6-2. Pseudo-labeling on test X
```python
# ZIT로 test 예측 → 신뢰도 높은 행만 train에 추가
high_conf = (test_pi > 0.95) | ((test_pi < 0.05) & (test_pred > 0.005))
X_aug = np.vstack([X_train, X_test[high_conf]])
y_aug = np.concat([y_train, np.where(test_pi[high_conf] > 0.95, 0, test_pred[high_conf])])
ZITboostRegressor(**best_params).fit(X_aug, y_aug)
```

### 6-3. Adversarial weighting (선택)
0-2 분류기로 train의 각 행에 P(test 같은 분포) 계산 → sample_weight. 단 ZIT의 sample_weight는 EM에서 미지원이라 zit 적용 불가, reg 경로만.

---

## 7. Tier 4 — 보류 (낮은 ROI 또는 음의 효과)

| 항목 | 보류 이유 |
|---|---|
| Sample weight (구 plan 2순위) | zero/positive zero-sum + enet `precompute=True` 호환성 silent fail + ZIT EM 미지원 |
| `(1-π) → P(Y>0|x)` multiplier 교체 (구 plan 1순위) | 더 작아져 zero 더 누름. 5-B 곱셈 약화가 안전 |
| zero_clip 0.02까지 확장 (구 plan 3순위) | target 평균 0.0025인데 0.02면 양수 다 0으로 → Y>0 underestimate 폭증 |
| pi_threshold step 0.005 정밀화 (구 plan 3순위) | 0.93~0.99 영역 평평. 무의미 |
| HPO mean→median 정렬 (구 plan 4순위) | agg_rmses 차이 6번째 자리수(0.0000003). ROI 0 |
| 단순 LGBM 잔차 학습 (구 plan 5순위) | gate 없으면 zero 망침 + 같은 모델 클래스 잔차 신호 작음. 5-C로 대체 |
| 단순 Ridge/LGBM stacking (구 plan 6순위) | 모델 다양성 부족. 4-2 learned aggregator가 더 정보량 큼 |
| 새 base 모델 (TabNet 등) | val 0.00571 천장. 모델 종류 늘려도 같은 패턴 |
| 전처리 global search 추가 | EDA max\|r\|=0.037 천장. group encoding 같은 zero/positive 분리 feature는 Tier 1·2 끝난 뒤 재평가 |

---

## 8. 실행 순서 (체크리스트)

```
[Day 0.5]  Tier 0-1 (segment RMSE 표)
           Tier 0-2 (AV)
           Tier 0-3 (GroupKFold 진단)
           → 분기 결정
[Day 1~2]  Tier 1-1 (seed bagging)
           Tier 1-2 (learned aggregator)
           Tier 1-3 (zeta range 재학습)
[Day 3~5]  Tier 2 4 변형 (5-A~5-D)
           segment 표로 채택 결정
[Day 6~]   Tier 3 (Tier 0-2 조건 충족 시)
[제출 직전] train+val 합쳐 final refit
```

---

## 9. 산출물 / 파일 변경

### 신규 노트북
| 파일 | 역할 |
|---|---|
| `3_modeling/final/_diagnostic.ipynb` | Tier 0-1·2·3 진단 (분석 노트북) |
| `3_modeling/final/05_zit_bagging.ipynb` | Tier 1-1 seed bagging |
| `3_modeling/final/06_unit_aggregator.ipynb` | Tier 1-2 learned aggregator |
| `3_modeling/final/07_gated_residual.ipynb` | Tier 2-C |
| `3_modeling/final/08_positive_expert.ipynb` | Tier 2-D |

### 기존 파일 변경
| 파일 | 변경 |
|---|---|
| `modules/models.py` (line 194) | zeta 범위 (1.1, 1.4) → (1.01, 1.95) |
| `01_zit_only.ipynb` | REUSE_BEST_PARAMS=False로 zeta 재탐색 |
| `03_zit_plus_reg.ipynb` cell 12 | 5-A·5-B 토글 |

### 산출 디렉토리
| 경로 | 내용 |
|---|---|
| `4_output/final/diagnostic/` | segment RMSE 표 PNG/CSV, AV AUC 결과, GroupKFold 비교 |
| `4_output/final/zit_bagging/` | seed별 OOF·val·test + 평균 |
| `4_output/final/unit_aggregator/` | meta 모델 OOF·val·test |
| `4_output/final/gated_residual/` | 5-C 결과 |
| `4_output/final/positive_expert/` | 5-D 결과 |

---

## 10. 변경 이력

| 일자 | 변경 |
|---|---|
| v1 (이전) | plan 1~6 (π 정정, sample_weight, boundary, agg 정렬, 잔차, stacking) |
| v2 (2026-04-27) | 코드 검증 + GPT-5.5 segment 분석 + 데이터 진단 통합. **plan 1·5·6은 변형으로 재구성** (1→5-B, 5→5-C, 6→4-2), **plan 2·4·일부 3은 보류로 강등**, **Tier 0 진단(segment·AV·GroupKFold) 신설**, **die=position·val 기준·outlier-driven test variance 명시** |