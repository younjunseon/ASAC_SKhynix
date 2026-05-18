# SHAP/XStacking 정식 구현 계획

> 목적: 일반 스태킹의 입력인 `base 예측값`만으로는 잔차 상관 0.99+ 천장에 걸리므로, base 모델의 **예측 이유(SHAP 기여도)** 를 메타 입력으로 추가해 더 낮은 OOF RMSE를 노린다.

---

## 1. 한 줄 요약

일반 스태킹은 각 base 모델의 최종 예측값만 조합한다.

```text
unit -> [zit_only_pred, bag_zit_pred, lgbm_pred, xgb_pred, ...] -> meta model
```

SHAP/XStacking은 여기에 각 base 모델이 어떤 피처 때문에 그 예측을 했는지까지 넣는다.

```text
unit -> [
  base predictions,
  lgbm_feature_A_shap, lgbm_feature_B_shap, ...
  xgb_feature_A_shap,  xgb_feature_B_shap,  ...
  cat_feature_A_shap,  cat_feature_B_shap,  ...
] -> meta model
```

즉, **예측값의 크기**뿐 아니라 **예측값이 나온 이유**를 메타 모델이 보게 만드는 방식이다.

---

## 2. 왜 필요한가

현재 스태킹의 가장 큰 병목은 base 모델끼리 너무 비슷하게 틀린다는 점이다.

기존 실험에서 반복적으로 확인된 사실:

- base 예측/잔차 상관이 대부분 `0.99+`
- RF/LGBM/Tweedie 같은 복잡한 메타러너는 과적합 또는 over-smooth
- subset selection만으로는 개선 폭이 작음
- `isotonic calibration`은 효과가 있지만 후처리성 개선
- 진짜 개선은 `old_bagzit`처럼 서로 다른 전처리/학습 흐름에서 나온 base 다양성에서 발생

일반 스태킹 입력은 각 모델당 숫자 1개다.

```text
모델 A 예측 = 0.010
모델 B 예측 = 0.010
```

이 둘은 메타 모델 입장에서 거의 같은 정보다.

하지만 SHAP을 보면 다를 수 있다.

```text
모델 A: spatial feature 때문에 +0.006, missing indicator 때문에 -0.001
모델 B: corr-filter 이후 남은 sensor feature 때문에 +0.005, xy feature 때문에 +0.002
```

예측값은 같아도 근거가 다르면, 특정 샘플군에서 어떤 모델을 더 믿을지 배울 여지가 생긴다.

---

## 3. 기대효과

### 3.1 기대하는 개선 방향

SHAP/XStacking이 기대되는 이유는 다음 세 가지다.

1. **base 간 redundancy 완화**
   - 예측값은 비슷해도 SHAP 패턴은 다를 수 있다.
   - 잔차 상관 0.99+ 상태에서 메타 입력의 분리도를 늘릴 수 있다.

2. **instance-level 가중치 효과**
   - 현재 ElasticNet 스태킹은 전체 데이터에 대해 거의 고정 가중치를 쓴다.
   - SHAP feature를 넣으면 “이런 feature 패턴에서는 lgbm을 더 믿고, 저런 패턴에서는 zit 계열을 더 믿는” 방향으로 학습 가능하다.

3. **Y=0 / Y>0 trade-off 조정 가능성**
   - 기존 Stage 1/2는 Y>0에서는 좋아졌지만 Y=0 false positive가 커져 실패했다.
   - SHAP은 분류기를 새로 세게 붙이는 대신, base 모델의 설명 패턴을 활용하므로 더 부드러운 분리가 가능하다.

### 3.2 현실적 기대치

과거 실험상 현재 스태킹/iso만으로는 `0.005700` 근처가 천장이다.

정식 SHAP/XStacking의 기대치는 보수적으로:

```text
OOF 기준: -0.000001 ~ -0.000005 가능성
val monitor: 0.005700 근처에서 0.00569x 진입 가능성
```

다만 이는 보장되지 않는다. 차원이 커져 과적합하면 오히려 악화될 수 있다.

---

## 4. 필요한 입력물

학원에 있는 `fold_models.pkl`이 핵심이다.

필요 파일:

```text
각 base 모델 디렉토리/
  fold_models.pkl
  oof_unit.csv
  val_unit.csv
  test_unit.csv
  best_params.json
```

가능하면 die-level 예측과 전처리 후 feature matrix도 필요하다.

```text
oof_die.csv / val_die.csv / test_die.csv
전처리된 X_train / X_val / X_test 재생성 가능 코드
fold별 validation index 또는 unit fold split
```

SHAP 계산은 모델 객체만으로 끝나지 않는다. 모델이 본 **전처리 후 feature matrix**와 feature column 순서가 일치해야 한다.

---

## 5. 구현 원칙

### 5.1 val은 선택 기준으로 쓰지 않는다

현재 프로젝트 기준:

```text
선택 기준: OOF 또는 meta-CV OOF
val: monitor/log only
```

SHAP/XStacking도 동일하다.

```text
금지:
  val SHAP 결과 보고 best subset/feature/meta를 선택

허용:
  OOF 기준으로 선택
  val은 리포트에 같이 출력
```

### 5.2 train OOF SHAP은 fold별 holdout으로 만든다

가장 중요한 leakage 방지 규칙이다.

잘못된 방식:

```text
전체 train으로 학습된 모델 -> 전체 train SHAP 계산
```

올바른 방식:

```text
fold 1 모델 -> fold 1 holdout unit/die SHAP 계산
fold 2 모델 -> fold 2 holdout unit/die SHAP 계산
...
fold 5 모델 -> fold 5 holdout unit/die SHAP 계산
```

이렇게 해야 `OOF SHAP`이 된다.

### 5.3 val/test SHAP은 fold 모델 평균

val/test는 정답을 선택에 쓰면 안 된다. SHAP feature 생성은 다음 둘 중 하나로 통일한다.

권장:

```text
각 fold model로 val/test SHAP 계산
-> fold 평균
```

대안:

```text
refit full model이 있으면 full model SHAP 사용
```

다만 OOF와 val/test의 생성 방식이 달라질 수 있으므로, 처음에는 fold 평균이 더 일관적이다.

---

## 6. 1차 구현 범위

처음부터 모든 base에 SHAP을 붙이면 복잡도가 크다. 1차는 tree 계열부터 시작한다.

우선순위:

1. `02_reg_single/lgbm`
2. `02_reg_single/xgb`
3. `02_reg_single/catboost`
4. `02_reg_single/et`
5. `03_two_stage/default/clf/*` 중 성능 상위
6. ZIT 계열은 모델 구조 확인 후 2차

이유:

- LGBM/XGB/CatBoost/ET는 SHAP 또는 tree importance 계산이 비교적 표준적이다.
- ZIT 계열은 custom 구조라 SHAP 계산 방식 확인이 필요하다.
- two-stage reg는 현재 의도적으로 제외 중이므로 1차에서는 제외한다.

---

## 7. 구현 단계

### Step 1. pkl 구조 확인

학원 pkl을 받아서 먼저 구조만 확인한다.

확인 항목:

```text
type(fold_models)
len(fold_models)
각 fold object type
feature_names 보존 여부
fold별 validation index 보존 여부
pipeline/scaler/preprocess 포함 여부
```

가능한 형태:

```python
fold_models = [
    model_fold_0,
    model_fold_1,
    ...
]
```

또는:

```python
fold_models = {
    "models": [...],
    "folds": [...],
    "feature_cols": [...],
}
```

여기서 fold index와 feature column 순서가 없으면, 노트북의 KFold seed와 unit 순서로 재구성해야 한다.

### Step 2. 전처리 후 X 재생성

각 base가 학습할 때 사용한 전처리 설정을 `best_params.json` 또는 notebook config에서 복원한다.

필요 split:

```text
X_train_die
X_val_die
X_test_die
```

그리고 train OOF fold를 재현하기 위한 unit fold split:

```text
KFold(n_splits=5, shuffle=True, random_state=SEED)
```

주의:

- unit 기준 fold여야 한다.
- die row 기준으로 split하면 leakage다.
- feature column 순서가 학습 당시와 정확히 같아야 한다.

### Step 3. die-level SHAP 계산

모델별로 계산 방식을 나눈다.

LGBM/XGB:

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
```

CatBoost:

```python
model.get_feature_importance(type="ShapValues", data=pool)
```

ExtraTrees:

```python
shap.TreeExplainer(model).shap_values(X)
```

계산량이 크면 샘플링 없이 전체를 하는 것이 원칙이다. 다만 시간이 과하면 feature selection용 mean abs SHAP은 샘플로 먼저 만들고, 최종 top-K만 전체 계산한다.

### Step 4. train OOF SHAP 생성

fold별로:

```text
fold model 로드
holdout unit 목록 확인
holdout die rows 추출
SHAP 계산
해당 holdout 위치에 저장
```

결과:

```text
SHAP_oof_die: train 전체 die row 수 x feature 수
```

### Step 5. die SHAP -> unit SHAP 집계

평가는 unit-level이므로 SHAP도 unit-level로 집계한다.

기본 집계:

```text
mean SHAP: feature별 평균 기여도
abs mean SHAP: feature별 평균 영향 크기
```

초기 추천:

```text
signed mean만 사용
```

확장 후보:

```text
mean
abs_mean
max
min
std
```

하지만 처음부터 집계를 많이 넣으면 차원이 폭발한다. 1차는 `mean`만 둔다.

### Step 6. feature selection

전체 feature를 다 넣지 않는다.

추천 방식:

```text
각 base별 mean(|SHAP|) 상위 K개 선택
K = 20, 30, 50
```

또는:

```text
모든 base-feature 조합 중 global top K
K = 100, 200, 300
```

초기 실험 매트릭스:

| 설정 | 설명 |
|---|---|
| `base_pred_only` | 기존 스태킹 baseline |
| `base_pred + shap_top20_each` | base별 SHAP 20개 |
| `base_pred + shap_top50_each` | base별 SHAP 50개 |
| `base_pred + shap_global100` | 전체 top 100 |
| `base_pred + shap_global300` | 전체 top 300 |

### Step 7. meta input 구성

최종 meta matrix:

```text
X_meta_oof = [
  base prediction columns,
  selected SHAP columns
]
```

컬럼 예:

```text
pred__zit_only
pred__bag_zit
pred__reg_lgbm
shap__reg_lgbm__feature_A
shap__reg_lgbm__feature_B
shap__reg_xgb__feature_A
...
```

### Step 8. meta model 학습

1차 추천:

```text
RidgeCV 또는 ElasticNetCV
```

이유:

- SHAP feature가 많으므로 강한 정규화 필요
- LGBM/RF meta는 기존 실험에서 과적합/over-smooth 위험이 컸음
- linear meta가 해석과 안정성 측면에서 먼저 적합

후보:

| meta | 목적 |
|---|---|
| RidgeCV | 고차원 안정화 |
| ElasticNetCV | feature selection |
| LassoCV | 강한 sparse 선택 |
| NNLS on selected features | 보수적 조합 |
| shallow LGBM | 2차 실험, 과적합 주의 |

### Step 9. calibration

기존과 동일하게 OOF 기준 isotonic을 붙인다.

```text
raw meta OOF pred
-> OOF 정답으로 IsotonicRegression fit
-> val/test raw pred에 transform
```

단, 선택 기준은 항상 OOF 또는 meta-CV OOF다.

---

## 8. 더 엄격한 평가 방식: meta-CV OOF

SHAP/XStacking은 feature 수가 많아서 과적합 위험이 크다. 단순히 `X_meta_oof` 전체에 meta를 fit하고 같은 OOF에서 점수를 보면 낙관적이다.

권장 평가:

```text
X_meta_oof를 다시 5-fold split
각 fold:
  train fold로 scaler/meta/iso 학습
  holdout fold 예측
전체 holdout 예측으로 meta-CV OOF RMSE 계산
```

선택 기준:

```text
meta_cv_oof_rmse 최소
```

리포트:

```text
meta_cv_oof_rmse
full_fit_oof_rmse
val_monitor_rmse
test_monitor_rmse
segment Y=0 / Y>0
```

---

## 9. 실험 매트릭스

1차 실험은 작게 시작한다.

| 실험 | base | SHAP source | SHAP K | meta | iso |
|---|---|---|---:|---|---|
| A | current top subset | 없음 | 0 | ElasticNetCV | O |
| B | current top subset | reg lgbm/xgb/cat/et | 20 each | RidgeCV | O |
| C | current top subset | reg lgbm/xgb/cat/et | 50 each | RidgeCV | O |
| D | current top subset | reg lgbm/xgb/cat/et | global 100 | ElasticNetCV | O |
| E | top OOF 20 base | reg + clf tree | global 300 | RidgeCV | O |

2차 실험:

| 실험 | 추가 |
|---|---|
| F | old_bagzit 또는 pphp ZIT 계열 pkl 포함 |
| G | SHAP signed mean + abs_mean 같이 사용 |
| H | PCA/SVD로 SHAP 50~200차원 압축 |
| I | shallow LGBM meta, depth 2~3, leaf 크게 |

---

## 10. 산출물 설계

추천 출력 경로:

```text
4_output/04_stacking/shap_xstacking/
  shap_cache/
    reg_lgbm_oof_unit_shap.parquet
    reg_lgbm_val_unit_shap.parquet
    reg_lgbm_test_unit_shap.parquet
    ...
  feature_selection.csv
  meta_results.csv
  best_config.json
  best_oof_unit.csv
  best_val_unit.csv
  best_test_unit.csv
  segment_report.csv
```

`best_config.json`에는 반드시 기록:

```json
{
  "selection_metric": "meta_cv_oof_rmse",
  "base_models": [],
  "shap_sources": [],
  "shap_aggregation": "unit_mean",
  "feature_selection": {},
  "meta_model": {},
  "isotonic": true,
  "rmse": {
    "meta_cv_oof": 0.0,
    "full_oof": 0.0,
    "val_monitor": 0.0,
    "test_monitor": 0.0
  }
}
```

---

## 11. 예상 리스크

### 11.1 차원 폭발

예:

```text
base 10개 x feature 500개 = 5,000 SHAP columns
```

대응:

- top-K 제한
- Ridge/ElasticNet 사용
- meta-CV OOF 기준 선택
- segment별 과적합 확인

### 11.2 SHAP 계산 비용

Tree SHAP이라도 fold x split x 모델 수가 많으면 오래 걸린다.

대응:

- 처음에는 3~4개 tree base만
- 캐시 저장 필수
- float32 저장
- parquet 권장

### 11.3 feature matrix 재현 실패

pkl이 있어도 모델이 학습 당시 본 feature column 순서를 재현하지 못하면 SHAP 계산이 틀어진다.

대응:

- `best_params.json`의 effective PP params 확인
- `fold_models.pkl` 내부 feature names 확인
- 모델 객체의 `feature_name_`, `feature_names_in_` 확인
- 재현 불가 모델은 1차 제외

### 11.4 val 과적합

SHAP/XStacking은 실험 자유도가 크다. val monitor를 보며 사람이 고르면 val leakage가 된다.

대응:

- `meta_cv_oof_rmse` 기준으로만 선택
- val top 결과는 `DIAGNOSTIC_ONLY`로 분리
- 최종 리포트에 OOF 기준 rank와 val rank를 따로 표기

---

## 12. 구현 난이도별 권장 순서

### 1단계: pkl 구조 확인

가장 먼저 학원 pkl 하나를 열어 구조를 확인한다.

대상 추천:

```text
02_reg_single/lgbm/fold_models.pkl
02_reg_single/xgb/fold_models.pkl
02_reg_single/catboost/fold_models.pkl
```

### 2단계: lgbm 1개만 SHAP 캐시 생성

목표:

```text
reg_lgbm_oof_unit_shap
reg_lgbm_val_unit_shap
reg_lgbm_test_unit_shap
```

이 단계에서 fold/feature/aggregation 문제가 대부분 드러난다.

### 3단계: lgbm SHAP만 메타에 추가

기존 base prediction + lgbm SHAP top-K로 meta-CV OOF를 본다.

이 단계에서 OOF가 전혀 안 좋아지면, 전체 확장 ROI가 낮을 수 있다.

### 4단계: xgb/catboost/et 확장

개선 신호가 있으면 tree source를 늘린다.

### 5단계: ZIT 계열 pkl 검토

ZIT 계열이 현재 성능 핵심이므로, 가능하면 ZIT의 설명 feature까지 넣고 싶다.

다만 custom 모델이면 SHAP이 바로 안 될 수 있다. 이 경우 대안:

- permutation importance style local perturbation
- leaf index embedding
- base prediction interaction feature 확장

---

## 13. 의사코드

```python
# 1. base prediction matrix
P_oof, y_oof = load_base_preds("oof")
P_val, y_val = load_base_preds("val")
P_test, y_test = load_base_preds("test")

# 2. SHAP cache 생성
for base in shap_sources:
    models = load_fold_models(base)
    X_train, X_val, X_test, feature_cols = rebuild_preprocessed_X(base)
    folds = rebuild_unit_folds(seed=42)

    shap_oof_die = empty_like_train()
    for fold_id, model in enumerate(models):
        holdout_units = folds[fold_id].valid_units
        X_holdout = X_train[train_units in holdout_units]
        shap_oof_die[holdout_rows] = compute_shap(model, X_holdout)

    shap_val_die = mean([compute_shap(m, X_val) for m in models])
    shap_test_die = mean([compute_shap(m, X_test) for m in models])

    shap_oof_unit = aggregate_die_to_unit(shap_oof_die, method="mean")
    shap_val_unit = aggregate_die_to_unit(shap_val_die, method="mean")
    shap_test_unit = aggregate_die_to_unit(shap_test_die, method="mean")

    save_cache(...)

# 3. feature selection
selected_shap_cols = select_top_k_by_mean_abs_shap(shap_oof_unit, k=K)

# 4. meta matrix
X_meta_oof = concat([P_oof, shap_oof_unit[selected_shap_cols]])
X_meta_val = concat([P_val, shap_val_unit[selected_shap_cols]])
X_meta_test = concat([P_test, shap_test_unit[selected_shap_cols]])

# 5. meta-CV OOF 평가
meta_oof_pred = crossfit_meta_with_iso(X_meta_oof, y_oof)
score = rmse(meta_oof_pred, y_oof)

# 6. full fit 후 val/test monitor
meta.fit(X_meta_oof, y_oof)
raw_oof = meta.predict(X_meta_oof)
raw_val = meta.predict(X_meta_val)
raw_test = meta.predict(X_meta_test)
iso.fit(raw_oof, y_oof)
final_val = iso.transform(raw_val)
final_test = iso.transform(raw_test)
```

---

## 14. Go / No-Go 기준

1차 lgbm-only SHAP 실험에서:

```text
meta_cv_oof가 기존보다 -0.000001 이상 개선
```

이면 확장 가치 있음.

반대로:

```text
meta_cv_oof 악화
val monitor만 개선
```

이면 val 과적합 가능성이 높으므로 중단 또는 feature 수 축소.

최종 채택 기준:

```text
1. meta_cv_oof 개선
2. val monitor가 크게 악화되지 않음
3. Y=0 / Y>0 segment 중 한쪽이 크게 망가지지 않음
4. 기존 combo+iso 대비 복잡도 증가를 설명 가능
```

---

## 15. 결론

SHAP/XStacking은 현재 프로젝트에서 아직 남아 있는 몇 안 되는 큰 실험이다.

기대하는 핵심 효과는:

```text
예측값만 비슷한 base들을
예측 이유 기준으로 다시 분리해
메타 모델이 샘플별로 더 나은 신뢰 패턴을 배우게 하는 것
```

다만 구현 비용과 과적합 위험이 크므로, 바로 전체 확장하지 말고:

```text
lgbm pkl 1개 구조 확인
-> lgbm SHAP top-K만 추가
-> meta-CV OOF 개선 확인
-> xgb/catboost/et 확장
-> 가능하면 ZIT 계열까지 확장
```

순서로 진행하는 것이 가장 안전하다.
