# SK Hynix - Wafer Test 기반 Field Health Data(RCC) 예측 프로젝트

## 프로젝트 개요

**주제**: Wafer Test 결과를 통한 고객 Field Health Data(RCC) 예측

WT(Wafer Test) 단계에서 Field 불량 위험이 높은 unit을 사전 식별하는 **회귀 예측 모델**을 구축한다. SK Hynix 사내 경진대회 최우수 성능 대비 **10% 이상 향상**이 목표이다.

## 핵심 목표

1. Field 불량 위험이 높은 unit 사전 식별 (예측 모델)
2. 주요 WT 불량 인자 파악 → 공정 개선 인사이트 도출
3. 예측 결과 모니터링 인터랙티브 대시보드 개발
4. 자연어 질의 기반 AI Agent 구현

## 데이터 구조

### 데이터 레벨 관계

- Feature는 **die level** 데이터, Label은 **unit level** 데이터
- 하나의 unit은 여러 개의 die(chip)로 구성됨
- Feature와 Label의 최소 단위가 다름 (die vs unit)
- `ufs_serial`이 unit 명이자 X-Y 데이터의 mapping key

```
Unit #1
  ├── Chip #1-1 (Features from WT) ──┐
  ├── Chip #1-2 (Features from WT) ──┼── Label #1 (Health data / RCC)
  └── Chip #1-N (Features from WT) ──┘
  ... × K개 Unit
```

### 설명변수 (X) - `compet_xs_data.csv`

| 변수명 | 설명 | 비고 |
|--------|------|------|
| `ufs_serial` | unit 명, Y data와의 mapping key | 문자열 (S00000~) |
| `run_wf_xy` | die 구분 key (Lot_Wafer_Die좌표XY) | 문자열 |
| `position` | 해당 die의 unit 내 위치 | 정수 |
| `split` | train/validation/test 자료 구분 | 문자열 |
| `X0` ~ `X1086` | Wafer test data (비식별화) | 연속형, 이산형 혼합. 총 1,087개 |

- **총 행 수**: 174,980 (die 수)
- **총 열 수**: 1,091

### 종속변수 (Y) - `compet_ys_*.csv`

| 변수명 | 설명 |
|--------|------|
| `ufs_serial` | mapping key |
| `health` | 고객 Field health data (RCC), 비식별화. **Zero-inflated** 분포 |

- Y=0: 70.8%, Y>0: 29.2%
- 각 unit 별 고객 health data로 구성된 **zero-inflated data**

### 데이터 분할

| 구분 | X (die 수) | Y (unit 수) | 비율 | Y 공개 여부 |
|------|-----------|------------|------|-----------|
| Train | 104,988 | 26,247 | 60% | 공개 |
| Validation | 34,996 | 8,749 | 20% | 비공개 |
| Test | 34,996 | 8,749 | 20% | 비공개 |

- X 데이터의 `split` 컬럼으로 구분
- Y의 validation/test는 비공개 (제출용)
- die:unit 비율은 약 **4:1** (174,980 dies → 43,745 units)

## 평가 기준

- **RMSE** (Root Mean Squared Error)를 사용하여 health data 예측 성능 평가
- test zero 분포 값 = 0.015구간 (참고)

## 파이프라인 (수행 순서)

### Phase 1: EDA (1~2주차)
- 반도체 도메인 지식 학습
- Target 변수 분포 시각화 및 정규성 검정
- WT Data와 Target 변수 상관관계 파악
- 범주형 변수에 따른 Target 변수의 변화
- WT Data 분석
- RCC Data 분석

### Phase 2: 데이터 전처리 (3~4주차)
- 중복 Data 및 Column 제거
- Outlier 탐색 및 치환
- 분산이 작은 Feature 제거
- Data Scaling (Standardization)
- One-Hot Encoding
- Feature Selection

### Phase 3: 모델링 (5~6주차)
- train/validation dataset 분할
- Ridge / Lasso Regression
- SVR (Support Vector Regressor)
- RandomForest / XGBOOST / LightGBM Regressor
- 최종 모델 구축
- 결과 해석

### Phase 4: 대시보드 & AI Agent (9~11주차)
- 예측 결과 모니터링 인터랙티브 대시보드 개발
- 자연어 질의 기반 AI Agent 설계 및 구현

## Feature Selection 전략 (추천 순서)

1,087개 피처에서 영양가 있는 피처를 빠르게 찾기 위한 단계별 전략. 앞 단계에서 피처를 줄인 뒤 뒷 단계로 넘긴다.

### Step 1: 빠른 사전 필터링 (수 분)
불필요한 피처를 빠르게 제거하여 이후 단계의 연산량을 줄인다.
- **분산 기반 제거**: `VarianceThreshold`로 분산이 거의 0인 피처 제거
- **높은 상관 피처 제거**: 피처 간 상관계수 > 0.95인 쌍에서 한쪽 제거 (다중공선성 감소)
- **Target 무상관 제거**: 각 피처와 Y의 상관계수 절대값이 임계값 미만인 피처 제거 (빠른 1차 스크리닝)

### Step 2: Boruta (권장, 수십 분)
RandomForest 기반으로 **모든 유의미한 피처**를 통계적으로 선별한다.
- Shadow feature(셔플된 복사본)와 비교하여 진짜 중요한 피처만 남김
- `BorutaPy` 라이브러리 사용, estimator는 `RandomForestRegressor(n_jobs=-1)`
- 장점: 피처 간 상호작용을 고려하며, 다른 방법들이 놓치는 약한 피처도 잡아냄
- `max_iter=100` 이상 권장, `perc=80~100` 조정으로 엄격도 조절

```python
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_jobs=-1, max_depth=7, random_state=42)
boruta = BorutaPy(rf, n_estimators='auto', max_iter=200, random_state=42)
boruta.fit(X_train.values, y_train.values)
selected = X_train.columns[boruta.support_].tolist()
```

### Step 3: Optuna + LightGBM Feature Importance (권장, 수십 분)
Optuna로 HPO를 하면서 동시에 피처 중요도를 뽑는다. Boruta 결과와 교차 검증하면 더 견고해짐.
- LightGBM의 `feature_importance(importance_type='gain')` 활용
- Optuna `IntegrationCallback`으로 pruning 적용하면 탐색 시간 단축
- **Null Importance** 기법 병행: target을 셔플한 상태에서의 importance와 비교하여 노이즈 피처 제거

```python
import optuna
import lightgbm as lgb

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 8, 256),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    model = lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    return mean_squared_error(y_val, pred, squared=False)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

### Step 4: Permutation Importance (검증용)
최종 선택된 피처의 실제 기여도를 모델에 구애받지 않는 방식으로 검증한다.
- `sklearn.inspection.permutation_importance` 사용
- validation set 기준으로 측정하여 과적합된 피처 탐지

### 피처 선택 조합 가이드
| 시나리오 | 추천 조합 |
|---------|----------|
| 빠른 베이스라인 | Step 1 → LightGBM importance Top-K |
| 균형잡힌 접근 | Step 1 → Boruta → Optuna HPO |
| 최고 성능 추구 | Step 1 → Boruta + Null Importance → Optuna HPO → Permutation 검증 |

## EDA 결과 요약 (2026-03-29)

### 데이터 구조 확인
- 모든 unit은 정확히 **4개 die**로 구성 (position 1~4, 각 43,745개 균등)
- Split 비율 60:20:20 정확히 일치, split 간 target 분포도 동일 → stratified split 확인

### Target (health) 특성
- **Zero-inflated**: Y=0 70.8%, Y>0 29.2% (train/val/test 모두 동일 비율)
- 전체 mean=0.0025, Y>0일 때 mean=0.0087, median=0.0065
- max=1.0 존재 → 극단적 이상치 가능성
- 값의 범위가 극도로 작음 (대부분 0~0.015 구간)

### 결측치 현황
- **전체 1,087개 feature에 결측 존재**
- 1,054개는 결측률 0.23%로 매우 낮음
- **고결측 feature**: X729(99.5%), X1051(94.5%), X517(92.1%), X597(91.9%), X710(90.6%), X657(89.6%), X1055(89.6%), X1052(88.7%)
- 50% 이상 결측: 17개, 25% 이상: 22개

### Feature 유형
- **연속형**: 696개, **이산형**(고유값 ≤20): 391개
- **상수 feature**(std=0): **98개** → 즉시 제거 대상
- **극저분산**(std<1e-6): **105개** → 제거 대상
- 실질 유효 feature: ~982개

### Feature-Target 상관관계 (핵심 발견)
- **최대 |r| = 0.037** (X1083), 모든 feature가 target과 극도로 약한 선형 상관
- |r| > 0.1인 feature: **0개**, |r| > 0.05: **0개**
- 평균 |r| = 0.011
- **시사점**: 단일 feature로는 예측 불가 → feature 간 상호작용과 비선형 관계가 핵심

### 이상치 / 스케일
- IQR 기준 이상치 5% 초과: 167개, 10% 초과: 62개
- X393(45.7%), X988(40.4%), X989(36.5%) 등 극단적 이상치 비율
- **Feature 스케일 극도로 불균일**: mean 범위 [-2,293 ~ 20,201,109], range 최대 45,839
- **왜도**: |skew|>2: 314개, |skew|>5: 175개, |skew|>10: 101개

---

## 전처리 + 모델링 통합 전략

EDA 결과를 바탕으로 도출한 end-to-end 전략. 핵심은 **(1) die→unit 집계의 풍부함**, **(2) zero-inflated 대응**, **(3) 비선형 모델**이다.

### Stage 1: Feature 정리 (제거)

불필요한 feature를 사전에 제거하여 연산량과 노이즈를 줄인다.

| 제거 대상 | 기준 | 예상 수량 |
|-----------|------|----------|
| 상수 feature | std = 0 | 98개 |
| 극저분산 | std < 1e-6 | +7개 (총 105개) |
| 고결측 | 결측률 ≥ 50% | ~17개 |
| 중복 컬럼 | 값이 완전히 동일한 쌍 | 확인 후 결정 |

→ 약 **1,087 → ~960개** feature로 축소

### Stage 2: 결측치 처리

- 결측률 0.23% 수준(1,054개): **train set 기준 median으로 imputation** (fit on train, transform all)
- 결측률 1~50% (약 11개): median imputation + 결측 여부 indicator 컬럼 추가 고려
- 데이터 누수 방지: split별로 분리한 후 처리
- **k-NN Imputation 비교 실험** (논문 1-3 근거): `sklearn.impute.KNNImputer`로 median 대비 성능 비교. k-NN이 클래스 분리에 더 효과적일 수 있음
- **XGBoost/LightGBM 네이티브 결측 처리 실험** (XGBoost 원논문 근거): Sparsity-Aware Algorithm으로 imputation 없이 결측 처리 가능. manual imputation과 성능 비교하여 더 나은 쪽 채택

### Stage 3: 이상치 처리

- **Winsorization** (상하위 1% or 5% clip): 극단값을 분위수 경계로 대체
- **DBSCAN 기반 다변량 이상치 탐지** (논문 1-2 근거): R² 0.742→0.950 극적 개선 사례. health의 극단값이나 다변량 공간에서의 이상 패턴 탐지에 활용
- **강건 통계 기반 이상치 처리** (논문 5-3, 5-4 근거): 단순 mean±kσ보다 **중앙값/백분위수 기반(AEC DPAT)** 또는 **Grubbs 검정 + Johnson 변환**이 우수. 로트/위치별 국소 기준이 전역 기준보다 효과적
- 트리 기반 모델은 이상치에 비교적 강건하므로, 선형 모델용으로만 강하게 처리
- health의 max=1.0 이상치: 제거보다는 유지 (실제 불량 제품일 가능성)

> **Pre-optimization 원칙** (논문 3-3 근거, 7.9%p 향상): 이상치/노이즈 처리는 **반드시 die 레벨에서 먼저 수행**한 후 unit 레벨로 집계한다. 집계 후 처리하면 die 간 노이즈가 집계 통계에 오염됨.

### Stage 4: Die → Unit 집계 (Feature Engineering 핵심)

단일 feature 상관이 max 0.037로 극도로 낮으므로, **집계 통계량의 다양성**이 성능을 좌우한다.

```
die-level feature (4 dies per unit)
    ↓ groupby('ufs_serial')
unit-level aggregated features
```

| 집계 함수 | 설명 | 기대 효과 |
|-----------|------|----------|
| mean | 평균 | 기본 중심 경향 |
| std | 표준편차 | die 간 편차 → 불균일성 지표 |
| min, max | 최소/최대 | 극단 die 포착 |
| max - min (range) | 범위 | die 간 산포 |
| median | 중앙값 | robust 중심 경향 |
| skew | 왜도 | die 분포 비대칭 |

- 960개 feature × 7개 통계 = **~6,720개** unit-level feature 생성
- position별 feature도 고려: position 1/2/3/4 각각의 값을 별도 컬럼으로 (960 × 4 = 3,840개 추가 가능)
- 메모리 이슈 시 mean + std + range 3개로 시작 후 점진 확장

### Stage 4.5: Wafer 메타 피처 엔지니어링 (run_wf_xy 활용)

`run_wf_xy`는 `작업번호_웨이퍼번호_X_Y` 형식이며, 기존 X0~X1086에는 없는 **공정 맥락 정보**를 담고 있다. 이를 파싱하여 모델이 "어디서, 어떤 조건으로 만들어진 제품인가"를 학습할 수 있게 한다.

#### 4.5.1 파싱

```python
split = xs['run_wf_xy'].str.split('_', expand=True)
xs['run_id']   = split[0]   # 작업번호
xs['wafer_no'] = split[1]   # 웨이퍼 번호
xs['die_x']    = split[2].astype(int)  # die X 좌표
xs['die_y']    = split[3].astype(int)  # die Y 좌표
```

#### 4.5.2 공간 피처 (die_x, die_y → 연속형, 직접 투입)

웨이퍼 내 위치는 불량률과 직접적 관계가 있다 (edge 효과, 장비 균일성 등).

| 피처 | 산출 방법 | 기대 효과 |
|------|----------|----------|
| `die_x`, `die_y` | 파싱 그대로 | 위치 자체의 영향 |
| `radial_dist` | `sqrt(die_x² + die_y²)` (웨이퍼 중심 기준 보정 필요) | edge vs center 효과 |
| `is_edge` | `radial_dist > threshold` (분포 확인 후 결정) | 이진 edge 플래그 |

- die→unit 집계 시: unit 내 4개 die의 `radial_dist`에 대해 mean, std, max, min 생성
- die_x, die_y 자체도 mean, std, range로 집계 → unit의 공간 분포 특성 반영

**공간 잔차 피처** (논문 5-3 NNR, 논문 5-4 GPR 근거):
- die의 WT 측정값에서 **인접 die 가중 평균(가우시안 가중)**을 뺀 잔차(residual)를 별도 피처로 생성
- 공간 트렌드에서 크게 벗어난 die = 불량 위험 높음 → 잠재 이상치 15.6% 추가 검출 사례
- 구현: 같은 웨이퍼 내 die 좌표 기반 거리 가중 평균 → 잔차 계산 → die→unit 집계

#### 4.5.3 로트/작업 피처 (run_id, wafer_no → 집계 통계 & 인코딩)

문자열 ID는 직접 모델에 넣을 수 없으므로 수치화가 필요하다.

**방법 A: 집계 통계 피처 (누수 없음, 우선 적용)**

```python
# 로트(run_id)별 주요 WT feature의 평균/std → 해당 로트의 공정 상태 지표
lot_stats = xs.groupby('run_id')[feat_cols].agg(['mean', 'std'])
# unit별로 merge → "이 unit이 속한 로트의 전반적 품질"
```

- run_id별: WT feature 평균, std → 로트 품질 지표
- wafer_no별: 같은 로트 내 웨이퍼 간 차이 포착
- **로트별 정규화 (공식 피처)** (논문 5-3 근거, Multi-Site 분리로 양품 폐기율 40%→8.9%): `(die_value - lot_mean) / lot_std` — 로트 간 공정 편차를 제거하여 die의 상대적 이탈 정도를 수치화. 단순 편차보다 정규화된 z-score가 로트 간 비교 가능

**방법 B: Target Encoding (성능 향상 기대, 누수 주의)**

```python
# CV fold 기반 Target Encoding으로 누수 방지
# train 내에서 K-fold로 나눠, fold-out 방식으로 인코딩
from category_encoders import TargetEncoder
# 또는 수동 구현: fold별로 해당 fold 제외한 나머지의 target 평균으로 인코딩
```

- run_id → 해당 로트의 평균 불량률 (target mean)
- wafer_no → 해당 웨이퍼 번호의 평균 불량률
- **반드시 CV fold 기반**으로 인코딩하여 데이터 누수 방지

**방법 C: Frequency Encoding (단순, 보조)**

- run_id, wafer_no 각각의 출현 빈도 → 빈도가 낮은 로트 = 소량 생산 = 다른 패턴 가능

#### 4.5.4 die→unit 집계 시 메타 피처 처리

| 원본 피처 | 집계 방법 | 결과 피처 |
|-----------|----------|----------|
| `die_x`, `die_y` | mean, std, range | unit 내 die 공간 분포 |
| `radial_dist` | mean, std, max, min | unit의 웨이퍼 내 위치 특성 |
| `run_id` 집계 통계 | unit 내 동일하므로 first | 로트 품질 지표 |
| `wafer_no` | first (unit 내 동일) | 웨이퍼 번호 |

#### 4.5.5 적용 우선순위

| 순서 | 피처 | 이유 |
|------|------|------|
| 1 | die_x, die_y, radial_dist | 파싱만으로 즉시 생성, 누수 위험 없음 |
| 2 | run_id별 WT 집계 통계 | 누수 없는 로트 품질 지표 |
| 3 | Target Encoding (CV fold) | RMSE 개선 기대, 구현 주의 필요 |
| 4 | Frequency Encoding | 보조 피처, 간단 |

### Stage 5: 스케일링 & 인코딩

- **RobustScaler** 권장 (이상치에 강건, IQR 기반)
- 이산형 feature: 고유값 수에 따라 **One-Hot Encoding** (고유값 ≤ 5) 또는 그대로 사용 (트리 모델은 인코딩 불필요)
- 왜도 큰 feature: log1p 변환 시도 (선형 모델 한정)

### Stage 6: Feature Selection

1,087개에서 집계로 ~6,720개로 확장되므로 선별이 필수.

1. **분산 기반 제거** → 2. **높은 상관 쌍 제거** (|r|>0.95) → 3. **Boruta** → 4. **LightGBM importance + Null Importance** → 5. **Permutation Importance 검증** → 6. **투표 기반 최종 선정**

**투표 기반 피처 선택** (논문 1-3 근거, 12가지 알고리즘 투표로 robust 선정):
- 3~5번 단계(Boruta, LightGBM importance, Null Importance, Permutation Importance) 결과를 교차 검증
- **최종 피처 = 3가지 이상 방법에서 동시에 선택된 피처**만 채택
- 단일 방법의 편향을 줄이고, 다수 알고리즘이 합의한 피처만 남겨 노이즈 피처 혼입을 방지

### Stage 7: 모델링

#### 7-A: Baseline 비교 (Cross Validation + cuML GPU 가속)

전처리 완료된 데이터에 **기본 파라미터**로 10개 모델을 돌려 RMSE를 한눈에 비교한다. 스케일링은 선형/딥러닝 모델에 필요하지만, 트리 모델은 스케일링해도 결과가 동일하므로 **RobustScaler 적용한 데이터 1벌**로 전체 모델을 통일한다.

**CV 방식**: sklearn `cross_val_score` (5-Fold, postprocess 포함 RMSE scorer) — sklearn 호환 8개 모델에 적용. TabNet/FT-Transformer는 수동 CV.

**GPU 가속 (cuML)**: Colab GPU 환경에서 RF, Ridge, Lasso, ElasticNet을 cuML로 가속. 로컬은 sklearn CPU.
- 설치: `rapidsai-csp-utils` 스크립트 (Colab 권장) 또는 `pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com`
- cuML은 sklearn API 호환 → `cross_val_score` 사용 가능
- cuML RF: `n_jobs` 없음(GPU 자동 병렬), `max_depth` 기본값 16
- cuML 선형 모델: `random_state` 불필요 (closed-form solution)
- ExtraTrees는 cuML 미지원 → 항상 sklearn

| 구분 | 모델 | 스케일링 필요 | GPU 가속 | 비고 |
|------|------|:---:|:---:|------|
| 트리 | **LightGBM** | X | 자체 GPU | 빠르고 대규모 feature에 강함 |
| 트리 | **XGBoost** | X | 자체 GPU | 정규화 옵션 풍부 |
| 트리 | **RandomForest** | X | cuML | 안정적, 분산 낮음 |
| 트리 | **CatBoost** | X | 자체 GPU | 범주형 네이티브, 벤치마크 111개 중 트리 최다 1위 (논문 Shmuel 2024) |
| 트리 | **ExtraTrees** | X | CPU only | RF보다 빠르고 분산 더 낮음 (cuML 미지원) |
| 선형 | **Ridge** | O | cuML | L2 정규화 |
| 선형 | **Lasso** | O | cuML | L1 정규화, 자동 feature selection 효과 |
| 선형 | **ElasticNet** | O | cuML | L1+L2 혼합 |
| 딥러닝 | **TabNet** | O | PyTorch GPU | Attention 기반, feature selection 내장, **val+test X로 self-supervised pretraining 가능** (TabNet 원논문 근거) |
| 딥러닝 | **FT-Transformer** | O | PyTorch GPU | Transformer를 tabular에 적용, 벤치마크 상위권 |

- `compare_models()`로 10개 모델의 val RMSE를 정렬하여 출력
- 이 결과를 바탕으로 이후 단계(HPO, Two-Stage, 앙상블)에 투입할 모델을 선별

#### 7-B: Two-Stage Model (zero-inflated 대응)

Y의 70.8%가 0인 zero-inflated 특성에 맞는 구조. 논문 1-4에서 R² 0.192→0.538, 논문 2-2에서 RMSE 18.3% 감소 실증.

```
Stage 1: 분류 (Y=0 vs Y>0)
    → Classifier → P(Y>0) 확률 출력

Stage 2: 회귀 (Y>0인 샘플만)
    → Regressor (Y>0 서브셋으로 학습) → health 값 예측

최종 예측 = P(Y>0) × E[Y | Y>0]
```

- 분류기의 확률과 회귀 예측값을 곱하면 자연스럽게 zero-inflation 반영
- 각 stage를 별도 최적화할 수 있어 튜닝 유연성 높음
- 7-A에서 성능 좋았던 모델을 Stage 1/2에 각각 배치

**Stage 1 클래스 불균형 대응** (논문 1-3, 2-2 근거):
- **SMOTE + undersampling 결합** (논문 1-3: SMOTE 40% oversampling + 다수 80% undersampling이 최적)
- 또는 **class_weight / scale_pos_weight** 조정 (더 간단)
- **Custom Weighted Log-Loss** (논문 2-2 근거): `w_Cj = 1/n_Cj`(클래스 빈도 역수) 가중치로 Recall 0.26→0.77 달성. 분류 정확도가 회귀 성능에 직결되므로 Stage 1 최적화 필수

**Stage 2 Loss Function** (논문 2-2 근거):
- **Poisson/Tweedie Loss > MSE**: health 데이터는 비음수 + zero-inflated이므로 Poisson 계열 loss가 데이터 특성에 적합
- LightGBM: `objective='poisson'` 또는 `objective='tweedie'` (tweedie_variance_power 1.0~1.5 실험)
- XGBoost: `objective='count:poisson'`

#### 7-C: Ensemble (최고 성능 추구)

7-A의 baseline 결과를 기반으로 상위 모델들을 조합한다.

**Stacking**
- Level 0: 7-A에서 선별된 상위 모델들 (트리 + 선형 + 딥러닝 혼합)
- Level 1: meta-learner (Ridge 또는 LightGBM)로 level 0 예측을 결합
- Two-Stage(7-B)의 예측 결과를 level 0 feature로 포함 가능

**Voting / Blending**
- 상위 N개 모델의 예측값을 가중 평균 (weighted average)
- 가중치는 val RMSE 역수 비례 또는 Optuna로 최적화
- Stacking보다 단순하지만 과적합 위험이 낮음

#### 7-D: HPO (하이퍼파라미터 최적화)

7-A ~ 7-C 각 단계에서 유망한 모델에 Optuna HPO를 적용한다.
- baseline에서는 기본 파라미터로 빠르게 비교
- 유망 모델 선별 후 Optuna로 정밀 튜닝
- early stopping + pruning으로 탐색 시간 단축

### Stage 8: 후처리 & 평가

- `np.clip(pred, 0, None)`: 음수 예측값 → 0으로 보정
- zero threshold tuning: P(Y>0)의 임계값을 0.5가 아닌 최적값으로 조정 (RMSE 기준)
- validation set RMSE로 모델 간 비교, 최종 test 예측

### 우선순위 실행 계획

| 순서 | 작업 | 기대 효과 | 논문 근거 |
|------|------|----------|----------|
| 1 | Stage 1~3 (정리/결측/이상치, **Pre-optimization 원칙 적용**) | 노이즈 제거, die 레벨 선처리 | 3-3, 1-2, 5-3, 5-4 |
| 2 | Stage 4 (die→unit 집계) + Stage 7-A (**10개** 모델 baseline 비교) | **모델별 RMSE 확보, 유망 모델 선별** | 4-1, 4-2, Shmuel 2024 |
| 3 | Stage 4 확장 (mean+std+range+position) | feature 풍부화 → RMSE 개선 | 3-3 |
| 3.5 | Stage 4.5 (공간 피처 + **공간 잔차** + **로트별 정규화**) | 공정 맥락 정보 추가 → RMSE 개선 | 2-3, 5-3, 5-4, Kang 2015 |
| 4 | Stage 6 (Boruta + Null Importance + **투표 기반 선정**) | 노이즈 feature 제거 → RMSE 개선 | 1-3, 1-4, 5-1 |
| 5 | Stage 7-D (유망 모델 HPO) | 하이퍼파라미터 최적화 → RMSE 개선 | 4-1 |
| 6 | Stage 7-B (Two-Stage + **Custom Loss + SMOTE**) | zero-inflated 대응 → RMSE 개선 | 1-4, 2-1, 2-2 |
| 7 | Stage 7-C (Stacking/Voting) + Stage 8 (후처리) | 최종 성능 극대화 | 4-1 |

---

## 핵심 주의사항

1. **Die → Unit 집계**: X는 die level, Y는 unit level이므로 모델링 시 die-level feature를 unit-level로 집계해야 함 (mean, max, min, std 등)
2. **Zero-inflated 분포**: Y의 70.8%가 0이므로 이를 고려한 모델링 전략 필요 (two-stage model, zero-inflated regression 등 고려)
3. **비식별화 데이터**: 변수명이 X0~X1086으로 비식별화되어 있으므로 도메인 지식 기반 해석보다 데이터 기반 접근 필요
4. **평가 지표는 RMSE**: 회귀 문제이며 분류가 아님

## 논문 참조

모델링/전처리 의사결정 시 아래 논문 요약을 근거 자료로 참조한다.

- **`99_학습자료/스터디자료/paper/논문요약.md`** — 17편 (PDF 기반 상세 요약)
- **`99_학습자료/스터디자료/paper/논문요약_미다운로드_13편.md`** — 13편 (웹 기반 요약)

주요 논문 번호와 CLAUDE.md 전략의 매핑:

| 논문 | 반영 위치 | 핵심 기여 |
|------|----------|----------|
| 1-3 (Rare Class, 2024) | Stage 6, 7-B | 투표 기반 피처 선택, SMOTE+XGBoost |
| 1-4 (Two-Step, 2025) | Stage 7-B | Two-Stage R² 180% 향상, Boruta 효과 |
| 2-2 (CatBoost Hurdle, 2024) | Stage 7-B | Custom Loss, Poisson Loss, RMSE 18.3% 감소 |
| 3-3 (NAND Flash, 2021) | Stage 3→4 | Pre-optimization 원칙 (7.9%p 향상) |
| 5-3 (Outlier Screening, 2013) | Stage 3, 4.5 | DBSCAN, NNR 잔차, 로트별 정규화 |
| 5-4 (GPR Outlier, 2024) | Stage 4.5 | 공간 잔차 피처 |
| Shmuel 2024 (벤치마크 111개) | Stage 7-A | CatBoost 추가 근거 |
| Kang 2015 (SK Hynix 공저) | Stage 4.5 | 공간 피처의 직접적 근거 |

## 디렉토리 구조

```
기업연계프로젝트/
├── CLAUDE.md              # 이 파일
├── setup.py               # 노트북 부트스트랩 (%run ../setup.py)
├── utils/                 # 공통 모듈
│   ├── __init__.py
│   ├── config.py          # 환경 감지, 경로, 상수 (Colab/Local 자동)
│   ├── data.py            # 데이터 로드, split 분리, 캐싱
│   ├── aggregate.py       # die→unit 집계, position 피벗
│   └── evaluate.py        # RMSE 평가, 모델 비교, 후처리
├── 0_data/                # 원본 데이터
│   ├── compet_xs_data.csv
│   ├── compet_ys_train_data.csv
│   ├── compet_ys_validation_data.csv
│   ├── compet_ys_test_data.csv
│   └── dataset.zip        # Colab용 (Google Drive ID: 1yOUo0_wPLcuZBSJIK592b00YkSIlk4zO)
├── 1_eda/
│   └── eda.ipynb
├── 2_preprocessing/       # 전처리 노트북
├── 3_modeling/            # 모델링 노트북
├── 4_output/              # 예측 결과, submission 파일
└── 90_기획/               # 기획 문서
```

## 코드 작성 규칙

### 최우선 원칙: 모델 RMSE 최소화
- **모든 의사결정은 RMSE를 낮추는 방향으로** 한다. 메모리 효율, 코드 간결성, 학습 시간은 후순위
- SK Hynix 사내 경진대회 최우수 성능을 넘기는 것이 목표. 시간이 오래 걸려도, 코드가 복잡해져도 성능이 높은 방법을 선택
- 모델링 시 단순한 방법부터 시작하되, 성능 개선 여지가 있으면 적극적으로 고도화 (앙상블, 스태킹, 피처 조합 등)
- 전처리/피처 엔지니어링 선택지가 여럿이면 여러 방법을 실험하고 RMSE가 가장 낮은 것을 채택

### 환경 호환 규칙 (Colab / Local 공통)

노트북은 로컬과 Colab에서 **코드 수정 없이** 동일하게 실행되어야 한다.

#### 노트북 첫 셀 템플릿 (모든 노트북에 동일하게 적용)

```python
import os, sys

try:
    import google.colab
    if not os.path.exists("/content/project/setup.py"):
        os.system("pip install -q gdown")
        os.system("gdown --id 1AD4PDBnDVjp-LSna6puB7qLnpBqB7j_I -O /content/code.zip")
        os.system("unzip -qo /content/code.zip -d /content/project")
        os.makedirs("/content/project/0_data", exist_ok=True)
        os.system("gdown --id 1yOUo0_wPLcuZBSJIK592b00YkSIlk4zO -O /content/project/0_data/dataset.zip")
        os.system("unzip -qo /content/project/0_data/dataset.zip -d /content/project/0_data")
        os.remove("/content/project/0_data/dataset.zip")
    sys.path.insert(0, "/content/project")
    %run /content/project/setup.py
except ImportError:
    %run ../setup.py
```

#### Google Drive 파일 ID

| 파일 | ID | 내용 |
|------|-----|------|
| `code.zip` | `1AD4PDBnDVjp-LSna6puB7qLnpBqB7j_I` | setup.py, requirements.txt, utils/ |
| `preprocessing.zip` | `1Rh0ByOS4Gama8XHuvY7KkOHo278H9YLr` | cleaning.py, outlier.py, aggregation.py |
| `dataset.zip` | `1yOUo0_wPLcuZBSJIK592b00YkSIlk4zO` | CSV 4개 (1.2GB) |

#### 환경 판별 원리

- `import google.colab` 성공 → Colab, 실패 → Local
- 이 판별은 `config.py`의 `ENV` 변수에도 반영됨 (`ENV == "colab"` 또는 `ENV == "local"`)

#### 코드 작성 시 지켜야 할 것

- **경로 하드코딩 금지**: `"c:\\Users\\..."` 또는 `"/content/..."` 직접 쓰지 않고, `config.py`의 `DATA_DIR`, `PROJECT_ROOT` 등 상수 사용
- **컬럼명 하드코딩 금지**: `"health"`, `"ufs_serial"` 대신 `TARGET_COL`, `KEY_COL` 등 상수 사용
- **패키지 추가 시**: `requirements.txt`에 추가하면 `setup.py`가 자동 설치 처리
- **utils 코드 수정 시**: `code.zip`을 재생성하여 Google Drive에 재업로드 필요 (같은 파일 ID 유지)
- **한글 폰트**: `setup.py`에서 자동 처리 (Local→Malgun Gothic, Colab→NanumGothic). 별도 설정 불필요

### 기본 규칙
- Python 사용, Jupyter Notebook 기반
- 데이터 경로는 `config.py` 상수 사용: `DATA_DIR`, `XS_PATH`, `YS_TRAIN_PATH` 등
- 재현성을 위해 random_state는 `SEED` 상수 사용 (값: 42)
- 모델 성능은 RMSE로 비교 (`evaluate()` 또는 `compare_models()` 함수 사용)

### 코드 수정 시 규칙 (필수)
1. **수정 전 원본 먼저 읽기**: 수정 대상 코드를 반드시 먼저 읽고, 기존 로직/변수명/출력 형태를 파악한 뒤 수정
2. **수정 계획을 사용자에게 먼저 보고**: 무엇을 왜 어떻게 바꿀 건지 설명하고, 승인 후 수정. 임의로 수정하지 않는다
3. **기존 동작 보존**: 리팩토링이나 utils 적용 시 출력값, 컬럼명, 변수명 등 기존 결과가 달라지지 않는지 확인
4. **최소 범위 수정**: 요청된 부분만 수정하고, 관련 없는 코드는 건드리지 않기
5. **수정 전후 비교**: 수정 후 변경된 부분과 기존과 동일하게 유지되는 부분을 사용자에게 보고

### 코드 작성 후 검증 (필수)
- 코드 작성/수정이 끝나면, **반드시 작성한 코드를 다시 읽어서** 아래 항목을 검증한 뒤 사용자에게 보고할 것
  1. 문법 오류 없는지 (괄호 짝, 들여쓰기, 콜론 등)
  2. 변수명이 이전 셀에서 정의한 것과 일치하는지 (오타, 대소문자)
  3. DataFrame 컬럼명이 실제 데이터와 일치하는지 (`health`, `ufs_serial`, `split`, `position`, `X0`~`X1086`)
  4. merge/join 시 key 컬럼과 how 파라미터가 올바른지
  5. 노트북(.ipynb) JSON 편집 시 이스케이프 문자(`\n`, `\"` 등)가 의도대로 적용되었는지

### 데이터 처리 시 주의사항
- **die↔unit 레벨 혼동 금지**: X는 die(174,980행), Y는 unit(43,745행). 병합 전 반드시 집계(groupby) 필요
- **데이터 누수(leakage) 방지**: split 컬럼 기준으로 train/val/test를 분리한 뒤 전처리할 것. fit은 train에만, transform은 전체에
- **컬럼 범위 주의**: X0~X1086은 1,087개. 하드코딩보다 `xs.filter(like='X')` 또는 `xs.loc[:, 'X0':'X1086']` 사용
- **제거/필터링 시 전후 비교 필수 출력**: 행 또는 컬럼을 제거할 때 반드시 `before → after` 수치를 출력한다 (예: `컬럼: 1,087 → 951 (136개 제거)`, `행: 174,980 → 170,000 (4,980개 제거)`)
- 결측값/무한값 처리 시 처리 전후 shape를 출력하여 의도치 않은 행 삭제 확인
- groupby 집계 후에는 반드시 `.reset_index()`와 결과 shape 확인

### 모델링 시 주의사항
- 학습 데이터에 Y를 merge한 뒤, Y가 NaN인 행이 없는지 확인
- 예측값에 음수가 나올 수 있으므로 `np.clip(pred, 0, None)` 후처리 고려 (health는 0 이상)
- 모델 학습 전 X, Y의 shape와 index 정렬 상태를 출력할 것
- RMSE 계산 시 `sklearn.metrics.mean_squared_error(squared=False)` 사용
