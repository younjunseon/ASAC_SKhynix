# Final 파이프라인 계획

> 1차(LGBM Two-Stage HPO) → 2차(3-모델 앙상블) → 3차(ZITboost) 실험 종료.
> 유망 조합 선별 완료 → **제출용 파이프라인**을 `3_modeling/final/` 에 새로 구축.
> **기존 실험 코드(`two_stage/`, `zitboost/`, `modules/`)는 동결** — 참조만, 수정 없음.

---

## 1. 설계 원칙

| 원칙 | 이유 |
|---|---|
| **실험 코드와 완전 격리** | 실험 브랜치 건드려서 재현성 깨지는 사고 방지. final은 self-contained. |
| **전처리·스케일링 Optuna 탐색 제거** | 값은 사용자가 직접 지정. trial 예산을 모델 HP에 집중. PP 캐시·warm-start 등 `e2e_hpo` 복잡도 버림. |
| **단순한 모듈** | 모델 학습 + 후처리 + 블랜딩만. "1 trial = 전처리~예측 전부" 구조 폐기. |
| **die-level 유지, 집계는 후처리** | 경로별로 독립 학습 → 후처리에서 집계 방식 탐색 → 블랜딩. |

---

## 2. 전체 흐름

```
1. 데이터 로드
2. Cleaning (고정 파라미터, 사용자 지정)
3. Outlier winsorize(0.0, 0.99)  ← 확정
4. Scaling (고정)
5. 모델 HPO (3 경로 병렬)
   ├─ A: ZITboost 단일                    → E[Y] = (1-π)×μ
   ├─ B: ZIT.π + REG (기존 Two-Stage 구조) → P(Y>0)_zit × reg_pred
   └─ C: REG 단일                          → 순수 회귀
6. 후처리 (die→unit)
   - 집계 6종: mean / weighted(SLSQP) / max / min / median / trimmed_mean
   - π threshold (경로 A·B만): grid 0.5~0.95
   - zero_clip threshold: 0.001~0.015
7. 블랜딩 Optuna: 3 경로 OOF → 가중치
8. 최종 test 예측 + CSV 저장
```

---

## 3. 전처리 (Stage 2~3) — 고정 파라미터

### 3.1 Cleaning (값은 사용자가 지정)

| 파라미터 | 의미 | 기본값 자리 |
|---|---|---|
| `const_threshold` | 극저분산 제거 기준 (std < ?) | `TBD` |
| `missing_threshold` | 고결측 제거 기준 (결측률 > ?) | `TBD` |
| `remove_duplicates` | 중복 컬럼 제거 | `True` 고정 |
| `corr_threshold` | 1차 상관 제거 임계값 (>?) | `TBD` |
| `corr_keep_by` | 동률 시 남길 기준 | `'target_corr'` or `'std'` |
| `add_indicator` | 결측 indicator 컬럼 추가 여부 | `True` / `False` |
| `indicator_threshold` | indicator 생성 기준 (결측률 > ?) | `TBD` |
| `imputation_method` | `'spatial'` 고정 | `'spatial'` |
| `spatial_max_dist` | spatial 보간 거리 | `TBD` |
| `post_impute_corr_threshold` | 2차(보간 후) 상관 제거 임계값 | `TBD` |
| `post_impute_corr_keep_by` | 2차 동률 시 남길 기준 | `'target_corr'` or `'std'` |

### 3.2 Outlier (확정)

```python
method='winsorize', lower_pct=0.0, upper_pct=0.99
```

### 3.3 집계 — **없음**

die-level 그대로 유지. 집계는 6번 후처리 단계에서 탐색.

### 3.4 사전 제외

`EXCLUDE_COLS` (웨이퍼맵 수동 분류 53개) 그대로 사용.

---

## 4. 모델링 (Stage 5)

### 4.1 모델 풀

`xgb`, `catboost`, `et`, `enet`, `lgbm` + `zitboost`

**현재 상태**:
- `model_zoo.py` 에 등록된 것: `lgbm / rf / et / logreg_enet / enet / zitboost`
- **새로 추가 필요**: `xgb`, `catboost` (MODEL_REGISTRY + search space)
- `3_modeling/catboost_info/` 폴더가 존재 → CatBoost 학습 흔적 있음 (이전 실험 어딘가)

### 4.2 3 경로 정의

#### 경로 A — ZITboost 단일
- 모델: `zitboost` 1종
- 학습: die-level (ZITboost 자체가 EM으로 joint 학습)
- 예측: `(1-π) × μ` (die-level 4개)
- 후처리: 집계 6종 탐색 + π threshold + zero_clip

#### 경로 B — ZIT.π + REG (기존 Two-Stage + A4 피처 투입)
- **Stage 1 (clf 대체)**: 별도 ZIT 모델 학습 → die별 `(1-π_zit)` = "Y>0 확률" 추출
  - 근거: 1차 실험에서 외부 clf의 Stage 1 recall 평균 0.011, ZIT의 π 기반 확률이 **recall 유의미하게 높음**
- **Stage 2 (reg)**: reg 입력 피처에 `(1-π_zit)` 를 **추가 피처로 투입** (A4 방식)
  - reg 모델: xgb/catboost/et/enet/lgbm 중 Optuna가 선택
- **최종 예측**: `pred_B = (1-π_zit) × reg_pred` (기존 two-stage 곱셈 공식)
- 학습: die-level
- 후처리: 집계 6종 + π threshold + zero_clip
- **π 우려 해소**: reg가 학습 시 `(1-π_zit)` 를 피처로 보므로, 곱셈에서의 π에 맞춰 reg 출력이 자동 조정됨

#### 경로 C — REG 단일
- 모델: xgb/catboost/et/enet/lgbm 중 Optuna가 선택
- 학습: die-level
- 후처리: 집계 6종 + zero_clip (π threshold 없음)

### 4.3 HPO 구조

- KFold(5) die-level OOF + Optuna
- objective: unit-level RMSE (OOF 4 die → 집계(당시 best) → unit RMSE)
- trial 1회 = 모델 학습 1회 (전처리 없음)

---

## 5. 후처리 (Stage 6)

ZITboost 노트북에서 설계된 3단계 후처리 그대로:

```
die-level 예측 (4개/unit)
  │
  ▼ 5-1: 집계 함수 탐색
  ├─ mean / weighted(SLSQP) / max / min / median / trimmed_mean
  └─ val RMSE 최소 자동 선택
  │
  ▼ 5-2: π threshold tuning  (경로 A·B 한정)
  ├─ predict_components() 로 π 추출
  ├─ grid: π > threshold → pred=0 (range 0.5~0.95)
  └─ val RMSE 최소 선택
  │
  ▼ 5-3: zero_clip
  ├─ pred < threshold → 0 (range 0.001~0.015)
  └─ 최종 예측
```

---

## 6. 블랜딩 (Stage 7)

- 입력: 3 경로의 OOF 예측 (unit level)
- 가중치 제약: `w_A + w_B + w_C = 1`, `w_i ≥ 0`
- 최적화: SLSQP (수학해) + Optuna (검증용) 양쪽 제공
- 최종: test 예측에도 동일 가중치 적용

---

## 7. 폴더/모듈 구조

```
3_modeling/
  modules/                  ← 기존 (동결)
  two_stage/                ← 기존 (동결)
  zitboost/                 ← 기존 (동결)
  final/                    ← ★ 신규
    plan.md                 ← 이 파일
    modules/
      __init__.py
      preprocess.py         # 고정 cleaning + outlier 래핑 (캐시 없음, 매번 실행)
      scaler.py             # enet 한정 스케일링 (트리 모델은 pass-through)
      models.py             # xgb/catboost/et/enet/lgbm/zitboost + search space
      hpo.py                # KFold OOF + Optuna (모델 HP만)
      zit.py                # ZITboostRegressor (zi_tweedie.py 복사)
      postprocess.py        # die→unit 6종 + π threshold + zero_clip
      blending.py           # 3-path OOF 가중치
    01_zit_only.ipynb       # 경로 A: ZITboost HPO + OOF(π, μ, (1-π)×μ) 저장
    02_reg_only.ipynb       # 경로 C: 단일 회귀 HPO (MODEL_NAME 스위치로 5회 실행)
    03_zit_plus_reg.ipynb   # 경로 B: 01의 (1-π) 로드 → reg 피처 투입 + HPO
    04_blend.ipynb          # 3-path OOF 블랜딩 + test 예측 + CSV 제출

출력 디렉토리:
  4_output/final/
    zit_only/                                           # 경로 A
    reg_only/{lgbm,xgb,catboost,et,enet}/               # 경로 C (모델별)
    zit_plus_reg/{lgbm,xgb,catboost,et,enet}/           # 경로 B (모델별)
    blend/                                              # 블랜딩 결과
```

**노트북 실행 순서**: 01 → 02 (독립, 순서 무관) → 03 (01의 π OOF 필요) → 04 (3개 OOF 전부 필요)

**전처리 공유**: 각 노트북 상단에서 `preprocess.run(params)` 호출 — **캐시 없음, 매번 실행**. 파라미터는 노트북 상단 셀에 변수로 노출. 네 노트북 모두 동일 파라미터 사용.

**ZIT 재사용**: 경로 B(03)는 경로 A(01)가 저장한 **die-level OOF `(1-π_zit)`** + **val/test `(1-π_zit)` 예측**을 로드 → reg 입력 피처에 **1개 컬럼만** 추가 (기존 two-stage baseline A4 방식과 동일). 같은 KFold seed/split 사용하여 leakage 방지. ZIT 두 번 학습 안 함.

**스케일링 분기**: 경로 C(02)에서 Optuna가 enet을 선택하면 RobustScaler 적용, 트리 모델(xgb/catboost/lgbm/et)이면 pass-through. 경로 A(01)의 ZIT와 경로 B(03)의 reg도 동일 규칙.

---

## 8. 피드백 / 확인 필요 포인트 (해소 기록)

### ✅ F1. `reg_level='position'` 실제 동작 — 해소

**내 초기 오해**: CLAUDE.md 표현("die 4개를 unit 행에 옆으로 펼치는")을 읽고 column pivot으로 착각.

**실제 동작** (e2e_hpo.py:779-814 확인):
- die-level 데이터 유지 (~170K rows)
- 모델도 die-level로 학습
- 예측 후 `groupby(KEY_COL).agg({"_pred": "mean"})` 로 unit 집계
- "position"은 die의 위치 컬럼(1~4)을 보존한다는 뜻일 뿐, column pivot 아님

**결론**: 세 경로 모두 **die-level 학습 + unit groupby 집계** 로 통일. 경로 분기 없음. 후처리의 "집계 6종 탐색"이 바로 이 groupby 집계 단계를 확장하는 것.

### ✅ F2. 경로 B의 π 사용 방식 — 해소

**사용자 지시**: `(1-π_zit)` 를 **reg의 입력 피처로 추가** + 최종 예측은 `(1-π_zit) × reg_pred`.
**증거**: ZITboost 학습으로 나온 분류 확률이 기존 Two-Stage 외부 clf 확률보다 **recall 높음**.
**π 우려 해소 근거**: reg가 학습 시 `(1-π_zit)` 를 피처로 보므로, 곱셈 단계의 π에 맞춰 reg 출력이 자동 조정됨.

→ 섹션 4.2 경로 B 정의 업데이트 완료.

### ✅ F3. 전처리 파라미터 입력 위치 — 해소

노트북(01_preprocess_fix.ipynb) 첫 셀에 **파라미터 변수로 노출** — 사용자가 직접 값을 적음. 모듈(`preprocess.py`)은 dict를 받아 실행만 담당.

### ✅ F4. catboost_info — 해소

`learn_error.tsv / test_error.tsv / catboost_training.json` 등. 과거 CatBoost 학습 로그. final에 영향 없음.

### ✅ F5. target transform 노출 정책 — 해소

**정책**: 경로별로 변경 불가한 고정값은 **모듈 내부 하드코딩**(사용자 수정 불가), 변경 가능한 것만 노트북에 스위치로 노출. 단 **결과 로그/주석/study user_attrs에는 "이 경로는 log1p/none 썼음"을 명시** (재현성).

| 경로 | target transform | 노출 위치 |
|---|---|---|
| A (ZIT) | `'none'` 고정 (Tweedie 분포 직접 모델링) | 모듈 내부 하드코딩, 노트북 출력에 "transform=none" 표시 |
| B (ZIT.π + REG) | `'none'` 고정 (곱셈 구조 유지 위함) | 모듈 내부 하드코딩, 출력에 표시 |
| C (REG 단일) | 노트북 스위치 (`'log1p'` / `'none'` / `'yeo-johnson'`) | 노트북 첫 셀 |

---

## 9. 작업 순서 (TodoList와 대응)

1. plan.md 작성 (v1 → v2 갱신)
2. 사용자 피드백 수신 (F1/F2/F5 해소, 노트북 4개로 축소)
3. `final/` 폴더 + `final/modules/` 생성
4. 기존 `utils/`, `2_preprocessing/`, `3_modeling/modules/zi_tweedie.py` 필요한 파일만 `final/modules/` 로 복사
5. `preprocess.py` — 고정 파라미터 기반 단일 함수 (`run(params) -> dict`, 캐시 없음)
6. `scaler.py` — enet 선택 시만 fit/transform, 트리 모델은 pass-through
7. `models.py` — 5+1 모델 등록, xgb·catboost 신규, search space 단순화
8. `hpo.py` — KFold OOF + Optuna (모델 HP만 탐색)
9. `postprocess.py` — 집계 6종 + π threshold + zero_clip
10. `blending.py` — SLSQP + Optuna
11. `01_zit.ipynb` — 경로 A HPO + rerun(kfold) + OOF π·μ·(1-π)μ 저장
12. `02_reg_single.ipynb` — 경로 C HPO + rerun + OOF 저장
13. `03_zit_plus_reg.ipynb` — 01 OOF π 로드 → reg 피처 투입 + HPO + rerun + OOF 저장
14. `04_blend_final.ipynb` — 3 OOF 로드 → 가중치 탐색 → test 예측 → CSV 제출

---

## 9.5 기존 구조의 알려진 한계 (코드 수정 안 함, 해석 시 주의)

제출 직전 리뷰에서 다음 네 가지가 확인되었다. **코드 복잡도 증가를 피하기 위해 plan 측을 맞추는 방향**으로 정책을 고정한다.

| # | 항목 | 현재 동작 | 정책 |
|---|---|---|---|
| L1 | **경로 B val/test 곱셈 순서** | val/test = `avg_fold(reg) × avg_fold(1-π_zit)`, OOF = per-fold `reg_k × (1-π_k)` | **인정**. fold 간 분산이 작아 실질 차이 미미. HPO objective가 OOF라서 "HPO 기준 ≠ 제출 기준" 편향은 있으나, 구조 변경(per-fold π 저장) 대비 이득 작음. |
| L2 | **경로 B π threshold 후처리** | `use_pi_threshold=False` 고정 | **off 유지**. 경로 B는 이미 `(1-π_zit)` 곱셈으로 π 정보가 반영되므로, π threshold를 추가 적용하면 **중복 감쇠**. 경로 A만 threshold 튜닝, 경로 B는 `agg + zero_clip`만. |
| L3 | **전처리(cleaning/imputation/winsorize)가 KFold 바깥에서 fit** | 전체 train 기준 통계로 전처리 → 그 결과로 KFold OOF 생성 | **인정**. val-fold 분포가 train-fold 통계 산출에 약간 섞이는 fold-내 편향(within-train) 존재. 외부 validation/test target 누수는 아님. **OOF RMSE는 모델/경로 간 상대 비교용으로만 사용**하고, 절대 수치는 외부 val로 재확인. |
| L4 | **`corr_keep_by='target_corr'` 옵션** | preprocess `PARAMS`로 override 가능 | **금기**. 전체 train target으로 feature 선택 → KFold에서 명확한 supervised leak. 기본값 `'std'`를 **절대 건드리지 말 것**. 필요 시 `corr_winsorize_pct`만 조정. |

## 10. 제외(하지 않는 것)

- Optuna 전처리 탐색 ❌
- PP LRU 캐시, warm-start, trial CSV callback ❌ (e2e_hpo 복잡도 전부 버림)
- Feature Selection ❌ (집계 없이 die-level이라 피처 수 = cleaning 이후 그대로)
- LDS 가중치, Isotonic calibration ❌ (일단 제외, 필요 시 v2)
- ※ A4(proba→feat)는 경로 B에서 **사용** (ZIT π를 reg 피처로 투입)

---

## 11. 노트북 입력값 노출 방침 (일관 규칙)

| 유형 | 위치 | 예시 |
|---|---|---|
| **경로별 불변 설정** (수정 시 설계 깨짐) | 모듈 내부 하드코딩 | 경로 A·B의 `target_transform='none'`, ZIT의 `run_clf=False`, outlier `(winsorize, 0.0, 0.99)` |
| **기본값 + 오버라이드 가능 값** | 모듈에 `DEFAULT_PARAMS` + 노트북에서 `params={'key': value}` 로 오버라이드 | cleaning 파라미터 11개, Optuna `n_trials`/`n_folds` |
| **실험 식별** | 노트북 첫 셀 스위치 | `EXP_ID`, `USER`, 경로 C의 `target_transform`, `EVAL_TEST`, `SAVE_OUTPUTS` |
| **재현성 로깅** | study user_attrs + 노트북 출력 | 경로별 `target_transform` 실제 값, **effective 전처리 파라미터 dict**(기본값 + 오버라이드 병합 결과), best params 전체 |

### 오버라이드 패턴 (전처리 파라미터 예시)

```python
# 모듈: preprocess.py
DEFAULT_PARAMS = {
    'const_threshold': 1e-6,
    'missing_threshold': 0.5,
    'corr_threshold': 0.94,
    'corr_keep_by': 'target_corr',
    'add_indicator': False,
    'indicator_threshold': 0.05,
    'spatial_max_dist': 2.0,
    'post_impute_corr_threshold': 0.98,
    'post_impute_corr_keep_by': 'std',
    # ...기본값은 1차/2차 실험 best 분석 기반으로 박아둠
}

def run(xs, ys, params=None):
    effective = {**DEFAULT_PARAMS, **(params or {})}
    # 실제 전처리 실행
    ...
    return cleaned_data, effective  # effective를 로그에 남김
```

```python
# 노트북 상단 셀
PARAMS = {
    # 원하는 것만 덮어쓰기. 안 적으면 DEFAULT 자동 적용.
    # 'const_threshold': 1e-5,
    # 'missing_threshold': 0.3,
}
cleaned, effective = preprocess.run(xs, ys, PARAMS)
print(effective)  # 재현성 확인용
```
