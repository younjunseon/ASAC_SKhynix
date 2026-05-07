# 모델링 공통 전략

> 모든 노트북(`01_zit`, `02_reg_single`, `03_two_stage`, `04_stacking`, `05_diagnostics`)에 공통 적용. 노트북별 개별 전략은 각 폴더의 `strategy.md`.

---

## 1. 트리계열 + ZIT 공통 PP 고정

**대상**: LGBM, XGBoost, CatBoost, ExtraTrees, ZITboost, BagZIT

**근거 (데이터 기반)**:
- 1차 실험 LGBM 12 study (3,500+ trial) RF feature importance 분석:
  - PP HP importance 평균 **3.1%**
  - 모델 HP importance 평균 **96.9%**
- BagZIT 직접 비교 (각 30 trial):
  - HP만 흔든 RMSE range: **0.000179**
  - PP만 흔든 RMSE range: **0.000059**
  - **HP가 PP의 약 3배 영향**
- 일반론: 트리 모델은 split-based 알고리즘 → 스케일/분포에 비민감

```python
PP_FIXED = {
    'missing_threshold':          0.30,
    'corr_threshold':             0.90,
    'corr_keep_by':               'std',
    'add_indicator':              True,
    'indicator_threshold':        0.05,
    'spatial_max_dist':           6.0,
    'post_impute_corr_threshold': 0.96,
    'post_impute_corr_keep_by':   'std',
}
```

**각 키 의미**:
- `missing_threshold=0.30` — 결측률 ≥ 30%인 컬럼 제거
- `corr_threshold=0.90` — 피처 간 |r| > 0.9 쌍에서 한쪽 제거 (다중공선성)
- `corr_keep_by='std'` — 제거 시 std 큰 쪽 유지 (분산 보존)
- `add_indicator=True` — 결측률 ≥ `indicator_threshold` 컬럼은 결측 여부 binary 컬럼 추가
- `indicator_threshold=0.05` — 5% 이상만 indicator
- `spatial_max_dist=6.0` — spatial imputation 시 같은 lot/wafer 내 직선거리(Euclidean) ≤ 6 die 이웃 사용 (wafer 평균 51×20 grid 기준 반경 6 ≈ 100 die 후보)
- `post_impute_corr_threshold=0.96` — imputation 후 다시 |r| > 0.96 쌍 정리
- `post_impute_corr_keep_by='std'` — std 큰 쪽 유지

---

## 2. 결측치 imputation cascade

[2_preprocessing/cleaning.py:258](../2_preprocessing/cleaning.py)의 `impute_spatial` 3단계:

| 단계 | 방법 | leakage 차단 |
|---|---|---|
| 1 | 같은 `lot_wafer` 그룹 내 거리 ≤ `spatial_max_dist` 이웃의 `1/distance` 가중 평균 | train_mask 적용 |
| 2 | 같은 `lot` 내 평균 | train만 사용 (train_mask) |
| 3 | train split의 컬럼 전체 평균 | train split만 사용 |

cross-wafer 이웃은 보지 않음. lot에 train 데이터 없으면 자동으로 다음 단계로 fallback.

---

## 3. ElasticNet — PP + X scaling + y transform + HP joint Optuna

**대상**: `02_reg_single/enet.ipynb`, `03_two_stage/default/reg/enet.ipynb`

**근거**: 선형 모델은 스케일/분포에 민감 → PP·X scaling·y 변환 모두 학습 결과에 직접 영향. 트리(split 기반)와 달리 ENet은 **OLS 가정(잔차 정규성·등분산성)**의 영향을 받으므로 y 변환이 의미 있음.

**탐색 축**:
- PP 8축 (PP_FIXED 키와 동일, 단 범위로 탐색)
- **X scaling** (categorical): `StandardScaler` / `RobustScaler` / `Yeo-Johnson` / `Quantile` / **`HybridScaler`**
- **y target_transform** (categorical, 신규): `none` / `log1p` / `yeo-johnson` / `quantile`
- ElasticNet HP: `alpha`, `l1_ratio` (0.1~0.9, 0/1 양 끝 제외), `max_iter`

`l1_ratio` 양 끝(순수 Lasso/Ridge)은 제외 — 약신호(EDA max\|r\|=0.037) + 다중공선성 환경에서 위험.

**HybridScaler**: 자체 구현 ([2_preprocessing/scaling.py:155](../2_preprocessing/scaling.py)). `skew_threshold` 기준으로 컬럼별 다른 스케일러 자동 선택 (high-skew → Yeo-Johnson, low-skew → Robust 등). 2차 funnel에서 만든 우리만의 스케일러. ElasticNet 탐색 후보로 반드시 포함.

**target_transform 후보**:
| 후보 | 처리 |
|---|---|
| `none` | identity (변환 없음) |
| `log1p` | `np.log1p` / `np.expm1` — 1차 enet best 검증된 변환 |
| `yeo-johnson` | `PowerTransformer(method='yeo-johnson')` — `lambda` 자동 추정 |
| `quantile` | `QuantileTransformer(output_distribution='normal')` — rank 기반 정규 매핑 |

→ 우리 데이터(y 0~0.097, zero-inflated 70.8%)에서 어떤 변환이 best인지 **Optuna가 자동 탐색**.

**구현 주의**:
- **fold별로 train fold y에만 transformer fit** (val/test의 y가 transformer에 새어들면 leakage)
- predict → `inverse_transform` → `np.clip(0, None)` (선형 모델은 음수 예측 가능)
- 후처리 `zero_clip_log_space`는 best `target_transform`에 맞춰 분기: `'log1p'`이면 `True`, 그 외(`none`/`yeo`/`quantile`)면 `False`
- `target_transform`이 trial 축에 들어가므로 `best_params.json`에 기록 필수 — refit 시 동일 변환 적용

---

## 4. Optuna sampler 설정

```python
from optuna.samplers import TPESampler

sampler = TPESampler(
    seed=None,            # ★ 시드 풀기 — 탐색 다양성 확보
    multivariate=True,    # ★ 파라미터 간 상관 고려
    group=True,           # ★ 그룹 단위 conditional sampling (자연 블록 활용)
)
```

**근거**:
- `seed=None`: 재현성 포기 대신 탐색 영역 다양화. 같은 노트북 여러 번 돌릴 때 다른 영역 탐색 → stacking pool 다양성 ↑
- `multivariate=True`: 22 HP의 상호작용 (예: `mu_lr × mu_n_estimators`) 고려
- `group=True`: ZIT의 `mu_*/pi_*/phi_*` 같은 자연 블록 conditional sampling. 기존 검증된 설정

---

## 5. 탐색 범위 — Anchor + Narrow / Evidence-driven Wide

두 가지 방식 중 노트북별 결정:

**A. Narrow (anchor ±n%)**: anchor의 신뢰도가 높고 좁은 영역에 mass가 모인 경우.
- 연속형: anchor의 ±30% (log-uniform이면 log-space 기준)
- 정수형: anchor ± 적은 step (max_depth ±1, n_estimators ±50)
- categorical: 그대로 유지

**B. Evidence-driven Wide**: 1차 top-N trial 분포가 anchor 변두리에 위치하거나 광범위하게 펴진 경우. **분포의 p5~p95 영역**을 search range로 잡음.

**공통**: anchor 있으면 항상 `study.enqueue_trial(anchor)`로 **첫 trial 강제** — 1차 best가 보존되도록.

- best 없으면 default search space 사용
- 구체 range는 노트북별 strategy.md에 명시

---

## 6. Cross Validation

- `N_FOLDS = 5` (모든 노트북 공통)
- **메커니즘**: sklearn `KFold(n_splits=5, shuffle=True, random_state=42)`를 **unique `ufs_serial` 배열에 적용** ([hpo.py:_make_unit_folds](modules/hpo.py))
- 즉 `GroupKFold`가 아닌 `KFold(unit_ids)` — unit ID를 먼저 fold로 나눈 뒤 die index에 매핑하는 방식
- 결과적으로 같은 unit의 4 die는 같은 fold에 들어감 (leakage 방지)
- die-level KFold 직접 적용은 leakage → **금지**

```python
unique_units = np.asarray(unit_ids)              # ufs_serial 유니크
kf = KFold(n_splits=5, shuffle=True, random_state=42)
folds = []
for tr_idx, vl_idx in kf.split(unique_units):
    folds.append((unique_units[tr_idx], unique_units[vl_idx]))
# → die index 매핑은 isin(unit_ids, fold_units)로 처리
```

---

## 7. random_state 정책

| 대상 | 값 | 이유 |
|---|---|---|
| 모델 학습 (`random_state`) | **42** 고정 | 재현성 |
| Optuna sampler (`seed`) | **None** | 다양성 (위 4번) |
| KFold split (`random_state`) | **42** 고정 | fold 일관성 (zit↔reg↔stacking 호환) |

---

## 8. 병렬 실행 / n_jobs 정책

여러 노트북을 동시에 돌릴 수 있으므로, **모델 학습의 `n_jobs`는 노트북 최상단 셀의 단일 변수로 제어** 가능하게 작성한다. 학원 환경 **14 코어** 기준.

**선언 위치 — 노트북 최상단 (환경 설정 셀)**:
```python
N_JOBS = 7   # ★ 이 값 하나만 바꾸면 모든 모델 fit/predict 병렬도 변경
```

**사용 위치 — 모든 모델 fit에 명시 전달**:
- LightGBM: `LGBMRegressor(n_jobs=N_JOBS, ...)`
- XGBoost: `XGBRegressor(n_jobs=N_JOBS, ...)`
- CatBoost: `CatBoostRegressor(thread_count=N_JOBS, ...)` ← 키 이름 다름 주의
- sklearn (RF/ET/Ridge/Lasso/ElasticNet): `RandomForestRegressor(n_jobs=N_JOBS, ...)`
- ZIT 내부 LightGBM: `params['n_jobs'] = N_JOBS` — `mu`/`pi`/`phi` 모두에 전달
- **Optuna `study.optimize(..., n_jobs=1)`** — trial 병렬은 끔 (모델 내부 병렬과 곱셈 효과 방지)

**14 코어 기준 N_JOBS 권장**:

| 동시 실행 노트북 수 | N_JOBS | 코어 사용 | 비고 |
|---|---|---|---|
| 1개 (전용) | 14 | 14 | 단일 모델 최대 속도 |
| **2개 (병렬)** | **7** | **14** | 두 모델 동등, throughput 최대 |
| 3개 (병렬) | 4 | 12 (여유 2) | 트리 plateau 영역, 무리 없음 |
| 4개 (병렬) | 3 | 12 | 약간 느려짐, 메모리 부담 ↑ |

**주의**:
- LightGBM/XGBoost는 8 코어 근처 scaling plateau — 14 몰빵해도 +20% 정도. 노트북 2개 병렬(N_JOBS=7)이 throughput 우세
- CatBoost `thread_count` 미지정 시 14 코어 다 먹음 → 명시 필수
- Optuna `n_jobs>1`은 multi-process라 모델 내부 병렬과 곱해짐 → study `n_jobs=1` 고정
- 메모리: LGBM 1 trial ≈ 4-8GB. 노트북 4개 병렬은 RAM 부족 위험 → RAM 모니터링

**금지**:
- `n_jobs=-1` 사용 (코어 다 먹어 다른 노트북 영향)
- 노트북 안에서 `n_jobs`를 hard-coded 숫자로 (`n_jobs=4` 직접 작성) — 항상 `N_JOBS` 변수 참조

---

## 9. 분류 threshold (Two-Stage 전용)

- Stage 1 clf의 `prob`에 임계값 적용 → `prob < τ`이면 0으로 강제
- 후보: τ_low (예: 0.0~0.5), 필요시 τ_high (0.5~1.0)도
- **Train OOF**에서 RMSE 최소 τ 탐색 → val 적용 후 비교
- val 개선되면 채택, 아니면 미적용
- **ZIT/BagZIT**: 이미 `τ_π`가 die-level threshold 역할 → SKIP (중복 불요)

---

## 10. die→unit 집계 다양성

- 후보: `mean`, `median`, `max`, `min`, `trimmed_mean`, `weighted`, `Q25`, `Q75`
- Train OOF에서 best agg 탐색 → val 적용 후 비교
- val 개선되면 채택, 아니면 `mean` 유지
- 모든 노트북 적용

---

## 11. Position 가중 평균 (4 die per unit)

- 후보 가중치: `w_p1, w_p2, w_p3, w_p4` (Dirichlet 정규화 또는 `sum=1`)
- Optuna sub-study로 best weight 탐색 (50 trial 정도)
- Train OOF에서 best, val 적용 후 비교
- val 개선되면 채택, 아니면 균등 (0.25 × 4) 유지
- 모든 노트북 적용

---

## 12. zero_clip

- 예측값 ≤ 임계값이면 0으로 강제 (zero-inflation 보정)
- 후보: 0.001 ~ 0.015 step 0.001
- Train OOF best, val 적용 후 비교
- val 개선되면 채택

**임계값 적용 공간**: target_transform 사용 시(log1p, yeo-johnson 등) 임계값 비교는 **transformed space에서 수행**한 뒤 inverse 적용.
- 이유: 모델이 학습한 공간과 일관성 유지. 작은 값 영역에서 log space가 더 민감
- 코드 분기: `apply_zero_clip(pred_log, th_log)` → `np.expm1(pred_clipped)` 순
- target_transform이 'none'인 경우 기존과 동일

> 후처리 공통 원칙 (8~11 모두): **train OOF best 탐색 → val 적용 → val_loss 개선되면 채택, 아니면 미적용**. 이미 동등 기능 구현된 모델(예: ZIT τ_π)은 중복 적용 안 함.

---

## 13. 구현 중 문제 발생 시 처리 원칙

**금지 (자동 우회/억지 구현)**:
- 에러나 비호환 상황을 임의 우회 (예: `try/except: pass`로 숨기기, default 값으로 대체, 임시 mock)
- import 실패를 감추는 fallback 코드
- "일단 동작하게" 만들기 위한 hack
- 사용자가 결정할 사항(API 변경, 모듈 위치, 시그니처 차이)을 추측으로 채우기

**원칙 (보고 후 중단)**:
- 모듈 import 실패, 함수 시그니처 불일치, 데이터 경로 누락, 의존성 충돌 등 발생 시
  1. **해당 위치에서 작업 중단**
  2. 사용자에게 다음을 보고:
     - 어떤 작업 중에 발생했는지
     - 정확한 에러/문제 내용 (메시지 그대로)
     - 가능한 원인 후보
     - 사용자 결정이 필요한 선택지 (있다면)
  3. 사용자 지시 받기 전까지 우회 시도하지 않음

**예외 (진행 OK)**:
- 명백한 typo, 문법 오류 등 자명한 자기 수정
- 사용자가 미리 위임한 영역의 합리적 default 사용 (예: HP 범위가 strategy.md에 명시된 경우)

---

## 14. 노트북 작성 규칙

이건 **최종 노트북**. 진척 보고/홍보 자료가 아니라 실행 코드 + 산출물 생성이 목적.

**금지**:
- "기존 대비 n% 개선" 같은 결과 요약 셀
- baseline 비교 셀 / 자축 메시지
- 어차피 산출물 csv·json에 다 있는 RMSE/메트릭 재출력 셀

**권장**:
- 노트북 첫 마크다운: 목적, 입력 데이터, 출력 산출물 명시
- 단계 헤더 (`## 1. 환경 설정`, `## 2. 데이터 로드` 등)
- 비자명한 결정에 대한 짧은 주석 (왜 이 HP인지, 왜 이 PP인지) — 발표 시 설명 도움

**예시 — 노트북 첫 셀 마크다운**:
```markdown
# 01 — ZIT only

ZITboost 단독 (joint EM, π·μ·φ 동시 학습) HPO + refit + 후처리.

- **입력**: `0_data/compet_xs_data.csv`, `compet_ys_train_data.csv`
- **출력**: `4_output/final/zit_only/{best_params.json, fold_models.pkl, optuna_*.db, oof|val|test_die.csv, oof|val|test_unit.csv}`
- **PP**: 트리 공통 `PP_FIXED` (strategy_common.md §1)
- **HPO**: 100 trial, anchor=trial 99 best, ±30% narrow
```

---

## 15. 출력 산출물 — 표준 9개

| # | 파일 | 내용 |
|---|---|---|
| 1 | `best_params.json` | Optuna best HP + study_meta + effective_pp_params + postprocess 메타 |
| 2 | `fold_models.pkl` | 5-fold 학습된 모델 (재현/SHAP/refit reuse) |
| 3 | `optuna_*.db` | Optuna sqlite study (trial 분포 분석용) |
| 4 | `oof_die.csv` | train OOF die-level 예측 |
| 5 | `val_die.csv` | val fold-mean die-level 예측 |
| 6 | `test_die.csv` | test fold-mean die-level 예측 |
| 7 | `oof_unit.csv` | die→unit 집계 + 후처리 적용 train OOF |
| 8 | `val_unit.csv` | 동일 처리 val |
| 9 | `test_unit.csv` | 동일 처리 test |

저장 helper: [`hpo.save_artifacts()`](modules/hpo.py) 활용. 9개 자동 저장.

---

## 16. 단계별 확장 산출물

- **03_two_stage/combine**: `combined/` 폴더에 `grid_summary.csv`, 페어별 `{clf}_x_{reg}/(oof|val|test)_unit.csv`
- **04_stacking**: `stack/blend` 결과 csv, `residual_corr_*.csv` (다양성 진단)
- **05_diagnostics**: `error_analysis_summary.png`, `segment_*.csv`, `fp_distribution.csv`, `se_decomposition.csv`

---

## 17. 출력 경로 정책

- 기존 `4_output/` 디렉토리는 통째 백업 (`4_output_이전자료/`로 rename)
- 새 모델링 출력은 깨끗한 `4_output/`로 들어감
- 노트북 내부 `OUT_DIR` 패턴은 이전과 동일 유지:
  - `4_output/final/zit_only/`
  - `4_output/final/reg_only/{MODEL_NAME}/`
  - `4_output/final/two_stage/clf/{MODEL_NAME}/`
  - 등

코드 변경 0줄 — `config.py`의 `OUTPUT_DIR` 상수만 `4_output/`을 가리키면 OK.

---

## 18. 공통 룰 적용 매트릭스

| 룰 | 01_zit | 02_reg_single (트리 4종) | 02_reg_single (enet) | 03_two_stage clf/reg | 04_stacking |
|---|---|---|---|---|---|
| 1. PP 고정 (PP_FIXED) | ✅ | ✅ | ❌ joint Optuna | ✅ (트리) / ❌ (enet) | N/A |
| 3. ElasticNet PP+HP joint | N/A | N/A | ✅ | ✅ (enet 한정) | N/A |
| 4. Optuna seed=None + multivariate + group | ✅ | ✅ | ✅ | ✅ | N/A |
| 6. N_FOLDS=5, unit-level KFold | ✅ | ✅ | ✅ | ✅ | ✅ |
| 8. N_JOBS override 가능 코드 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 9. 분류 threshold | SKIP (τ_π) | N/A (회귀만) | N/A | APPLY | N/A |
| 10. die→unit 집계 다양성 | APPLY | APPLY | APPLY | APPLY | APPLY |
| 11. Position 가중치 | APPLY | APPLY | APPLY | APPLY | APPLY |
| 12. zero_clip | APPLY | APPLY | APPLY | APPLY | APPLY |
| 14. 노트북 작성 규칙 | ✅ | ✅ | ✅ | ✅ | ✅ |
| **24. y target_transform** | **`'none'` (이미)** | **`'none'` 통일** | **Optuna 카테고리 (§3)** | **트리=`'none'`, ENet=Optuna** | N/A |
| 25. study timeout (N_TIMEOUT_SEC) | ✅ | ✅ | ✅ | ✅ | N/A (Optuna 없음) |

---

## 19. 결정 원칙

1. **모든 후처리는 train OOF best → val 적용 → 개선 확인** 순서. val 개선 안 되면 적용 안 함
2. **ZIT 등 이미 구현된 기능과 중복되는 후처리는 강제 적용 안 함** (예: τ_π는 분류 threshold 자리)
3. **트리계열 PP는 고정** — 모델별 차별화 효과 3% 미만, 발표 명료성 + 코드 단순성 우선
4. **ElasticNet만 PP+scaling joint** — 선형 모델 스케일/분포 민감성 보전
5. **재현성 포기, 다양성 확보** — Optuna seed 풀기로 같은 노트북 여러 회 실행 시 stacking pool 다양화
6. **트리/ZIT 호환 유지** — fold split SEED 일관 (zit OOF ↔ reg OOF ↔ stacking 매핑 가능)
7. **노트북에 자축 메시지 금지** — 산출물 csv가 곧 결과. 노트북은 실행기

---

## 20. 실행 순서

| 순서 | 노트북 | 의존 |
|---|---|---|
| 1 | `01_zit/01_zit_only.ipynb` | (없음) |
| 2 | `01_zit/02_bag_zit.ipynb` | (없음, 1과 병렬 가능) |
| 3 | `02_reg_single/reg_single.ipynb` × 5 모델 | (없음, 모델별 병렬) |
| 4 | `03_two_stage/default/clf/` clf HPO × 4 모델 | (없음) |
| 5 | `03_two_stage/default/reg/` reg HPO × 5 모델 | (없음) |
| 6 | `03_two_stage/reverse/` reverse HPO | (없음) |
| 7 | `03_two_stage` combine (M×N grid + position weighted) | 4·5·6 완료 |
| 8 | `04_stacking/stacking_meta.ipynb` | 1·2·3·7 OOF 완료 |
| 9 | `05_diagnostics/error_analysis.ipynb` | 8 완료 |

학원 14코어 환경: 1·2·3 (총 7 모델)을 3 노트북 병렬 × 각 4코어 식으로 2턴이면 끝남. 4·5·6은 그 후 진행.

---

## 21. 검증 근거

- **PP importance 3% 검증**: [4_output/experiments/1차 실험/optuna_merged.db](../4_output/experiments/1차%20실험/optuna_merged.db) 12 study × 평균 330 trial RF feature importance
- **HP가 PP의 3배**: [4_output/_temp/bag_zit_hpo](../4_output/_temp/bag_zit_hpo) (HP only 30trial) vs [4_output/_temp/bag_zit_pp_hpo](../4_output/_temp/bag_zit_pp_hpo) (PP only 30trial) RMSE range 비교
- **PP_FIXED 값**: zit_only joint best PP 기준 + 사용자 결정 (corr 0.9, missing 0.3, **spatial 6** [10→6 조정, wafer y축 20칸 대비 반경 10이 과대], indicator_thr 0.05, post_impute 0.96)

---

## 22. 모듈 조직

| 종류 | 위치 | 내용 |
|---|---|---|
| **전처리 모듈** | `2_preprocessing/*.py` | cleaning / outlier / scaling (HybridScaler 포함) / encoding / group_encoder / meta_features / feature_selection / sample_weight / aggregation |
| **모델링 모듈** | `3_modeling/modules/*.py` | zit / models / hpo / postprocess / blending / preprocess(파이프라인 wrapper) |

**원칙**:
- **전처리 모듈을 모델링 폴더로 복사하지 않는다** — 한쪽만 수정되어 동기화 사고 발생.
- 노트북 import:
  - 전처리: `setup.py`가 `2_preprocessing/`을 `sys.path` 등록 → `from cleaning import run_cleaning` 등 직접 import
  - 모델링: `3_modeling/modules/`를 `sys.path` 등록 → `from modules.hpo import run_hpo` 등

**기존 → 신규 이관 매핑** (이전 `3_modeling_이전자료/final/modules/`에서):

| 기존 파일 | 신규 위치 | 비고 |
|---|---|---|
| `cleaning.py`, `outlier.py`, `scaling.py` | **삭제** — `2_preprocessing/` 동명 파일 직접 사용 | 복사본이었음 |
| `zit.py`, `models.py`, `hpo.py`, `postprocess.py`, `blending.py`, `preprocess.py`, `scaler.py` | `3_modeling/modules/`로 이동 | 모델링 전용 |

`scaler.py`(enet 한정 RobustScaler 래퍼)는 `2_preprocessing/scaling.py`의 `maybe_scale` / `HybridScaler`로 흡수 가능하면 통합. 시그니처 차이 있으면 사용자 보고.

---

## 23. 노트북 검수 체크리스트

매 노트북 작성·수정 후 5개 항목을 점검한다. 위반 발견 시 strategy.md 수정 또는 사용자 보고.

### 23.1 설계 문서 일관성

노트북 폴더의 `strategy.md` (또는 본 `strategy_common.md`)에 명시된 사항이 노트북에 **그대로** 반영됐는가?

- [ ] `PP_FIXED` 키/값 (또는 PP joint search range)
- [ ] HP search range — narrow ±n% (anchor 기반) 또는 evidence-driven wide (1차 분포 기반)
- [ ] `N_TRIALS`, `N_FOLDS`, `N_JOBS`, `N_STARTUP_TRIALS`
- [ ] sampler 설정 (`TPESampler(seed=None, multivariate=True, group=True)`)
- [ ] pruner 설정 (예: `MedianPruner`)
- [ ] anchor enqueue (`study.enqueue_trial(anchor)` 첫 trial 강제)
- [ ] 후처리 매트릭스 (집계 8후보, position Optuna sub-study, zero_clip log_space, π threshold APPLY/SKIP)
- [ ] 출력 경로 패턴

### 23.2 기존 코드 컨셉/기능 보존

- [ ] 모듈 함수의 시그니처·리턴 구조가 기존 호출부와 호환 (default 인자로 backward-compat 유지)
- [ ] 기존 핵심 로직 누락 없음 — 예:
  - τ_π 적용 위치 (die-level → unit 집계)
  - fold split 메커니즘 (unit-level KFold, seed 고정)
  - EM 수렴 검증 (`em_history_` 단조감소)
  - target_inverse_fn 적용 시점 (refit 후 die-level로 복원)
- [ ] 함수 default 변경 시 영향 범위 점검 (기존 호출부에서 명시 전달 안 하면 동작 변경 가능)

### 23.3 재현성 — 산출물의 완전성

`best_params.json` 검증:
- [ ] `best_params_resolved` — 모델 HP 전체 + `random_state` 포함
- [ ] `effective_pp_params` — 실제 적용된 PP 8축 (joint trial인 경우 best 값)
- [ ] `feature_names` — 전처리 후 살아남은 feature 리스트
- [ ] `n_folds`, `unit_ids_hash`, `n_units_train` — fold 분할 재현 가능
- [ ] `study_meta` — sampler/pruner/anchor/CLIP_Y_EXTREME/SEED/N_JOBS
- [ ] `postprocess` 메타 — best_agg, pos_weights, best_zero_clip, zero_clip_log_space, position_method, agg_rmses

기타:
- [ ] `fold_models.pkl` — 모든 fold 모델 + feature_names + (ZIT는 `em_history_per_fold` 포함)
- [ ] `optuna_*.db` — OUT_DIR에 저장 (storage URL은 절대경로 권장)
- [ ] CSV health 컬럼 — val/test 미공개라도 train의 `health`는 반드시 merge

### 23.4 입출력 경로

- [ ] 입력: `utils.config`의 상수만 사용 (`DATA_DIR`, `XS_PATH`, `YS_*_PATH`, `OUTPUT_DIR`) — 하드코딩 금지
- [ ] 출력: §17의 패턴(`4_output/final/{exp_name}/`) 준수
- [ ] Colab 분기: `GDRIVE_CODE_ID` / `GDRIVE_DATASET_ID` / `GDRIVE_PREPROCESSING_ID` / `GDRIVE_MODELING_ID` 4종 정의 (modeling은 신규)
- [ ] `sys.path` 등록 순서: `2_preprocessing` → `3_modeling` (전처리/모델링 모듈 import)

### 23.5 1차 실험 결과 반영 점검

신규 노트북이 1차 실험 DB·산출물에서 도출된 사실과 모순되지 않는지 확인.

- [ ] anchor가 1차 best HP와 정확히 일치 (다른 컨텍스트에서 fork한 경우 출처·차이 명시)
- [ ] search range가 1차 top-N 분포의 **p5~p95**를 커버 (Evidence-driven Wide 채택 시)
- [ ] `PP_FIXED`가 1차 PP joint 분포의 mode와 일치 — 보수적 fix면 근거 명시 (예: "ZIT context"라 reg와 다름)
- [ ] 손실함수 후보가 1차 DB top 분포와 일치 — 명백히 worse(`Δ > 0.0001`) 옵션은 search space에서 제거
- [ ] 과거 발견된 충돌 조합 회피 — 예: log1p × tweedie (EXPERIMENT_LOG §5.1), `corr_keep_by='target_corr'` (plan.md L9.5 leakage)

### 23.6 데이터 누수(leakage) 점검 — Critical

| 점검 | 위반 시 영향 |
|---|---|
| KFold 분할이 **unit-level**인가 (die-level 직접 분할 금지, §6) | 같은 unit 4 die가 train/val에 섞여 OOF 부풀림 |
| Imputation·scaler `fit()`이 **train_mask만** 사용하는가 | val/test 정보 새어 들어가 OOF가 비현실적으로 좋아짐 |
| Target encoding이 **fold-out**(K-fold에서 자기 fold 제외) 방식인가 | self-target leak |
| `corr_keep_by='target_corr'` / `post_impute_corr_keep_by='target_corr'` 회피 (§22, plan.md L9.5) | KFold 안에서 train target 전체 사용 → supervised leak |
| 후처리 best 탐색 시 train OOF만 사용 (val/test 미참조) | val/test 직접 fit 시 cherry-picking |

---

### 검수 실행 프로토콜

**노트북/모듈 작업 완료 시점에 사용자에게 다음과 같이 묻는다**:

> "01_zit 작업 완료. **§23 검수 6항목 실행할까요?**"

사용자가 **"진행"** 또는 동의 신호를 주면:
1. §23.1~23.6 6항목을 순서대로 점검
2. 각 항목별 통과/미통과 + 근거(파일 라인 번호) 보고
3. 미통과 항목 발견 시 작업 중단·사용자 결정 받기 (§13 원칙)

사용자가 검수 스킵 지시(예: "그냥 커밋해", "다음으로 가자")를 주면 검수 생략 가능 — 단 strategy.md에 "23 검수 미수행" 메모.

자동 검수 진행 금지 — 항상 사용자 확인 후 실행.

---

## 24. 트리계열 y target_transform — `'none'` 통일

**대상**: LGBM, XGBoost, CatBoost, ExtraTrees (모든 트리계열, 모든 학습 컨텍스트 — die broadcast / y>0 only / Two-Stage clf · reg / reverse)

**근거 (실측 검증)**: [`3_modeling/log1p_check.ipynb`](log1p_check.ipynb) — LGBM 단일 회귀, 1차 best HP 고정, 4 loss × 4 transform = **16조합 직접 비교** (die broadcast, 5-fold OOF unit RMSE, 80 모델 학습)

| loss \ transform | none | log1p | yeo-johnson | quantile |
|---|---|---|---|---|
| regression | **0.005523** | **0.005523** | 0.005814 | 0.006123 |
| poisson | **0.005521** | **0.005521** | 0.005785 | INCOMPAT |
| tweedie_1.2 | **0.005522** | **0.005522** | 0.005788 | INCOMPAT |
| tweedie_1.5 | **0.005527** | **0.005527** | 0.005801 | INCOMPAT |

**핵심 발견**:
1. **`none` = `log1p` (모든 손실에서 소수점 6자리까지 정확히 동일)** — y가 0~0.097 영역이라 `log1p ≈ identity` (Taylor: `log1p(0.1)=0.0953` ≈ y)
2. **`yeo-johnson` 평균 +0.00027 악화** — 변환이 너무 강해 정보 손실
3. **`quantile` 분포 손실에서 INCOMPAT** — LightGBM `poisson`/`tweedie`는 음수 target 거부, `QuantileTransformer(output_distribution='normal')`은 음수 매핑 발생
4. **`quantile` regression에서 +0.000600 악화** — 분포 변형이 y의 의미 왜곡

**정책: 트리계열 `TARGET_TRANSFORM = 'none'` 고정**

이유:
1. **`none` = `log1p` 동등** → 단순한 쪽 채택 (분기 코드 불필요)
2. **분포 손실 충돌 회피** — `tweedie`/`poisson` objective는 분포 가정 자체 모델링하므로 변환 시 충돌 (실측: y broadcast 컨텍스트에서 yeo/quantile 모두 악화)
3. **모든 손실(regression/poisson/tweedie/binary)에 일관 적용** — 분기 없는 단일 룰
4. **후처리 `zero_clip_log_space=False` 통일** — 학습 공간(original)과 일치

**주의 — y>0 only 컨텍스트의 1차 결과**:
- 1차 `ts-reg-{lgbm,xgb,catboost}` best는 모두 `log1p ON + tweedie_X` 조합 (`y_positive_only=True` 컨텍스트)
- log1p_check은 die broadcast 컨텍스트라 직접 비교는 아님
- 그러나 **`none` = `log1p`라 수치적으로 동일하므로 y>0 only 컨텍스트에서도 영향 거의 없을 것** (이론적으로도 y 작은 영역이라 log1p ≈ identity 동일 적용)
- 정책 통일성 + 코드 단순성 우선 → 트리는 모든 컨텍스트에서 `'none'`

**적용 대상 노트북**:
- `01_zit/01_zit_only.ipynb`, `02_bag_zit.ipynb` — 이미 `TARGET_TRANSFORM='none'` (변경 불필요)
- `02_reg_single/{lgbm, xgb, catboost, et}.ipynb` — 신규 작성 시 `'none'` 적용
- `03_two_stage/default/clf/{lgbm, xgb, catboost, et}.ipynb` — binary 분류, transform 무관 (`'none'` 명시)
- `03_two_stage/default/reg/{lgbm, xgb, catboost, et}.ipynb` — 신규 작성 시 `'none'` 적용
- `03_two_stage/reverse/ts_reverse.ipynb` — 1차 코드는 `log1p ON` → 정책 따라 `'none'`으로 수정 필요

**ENet은 별개** (§3 참조): PP joint Optuna에 `target_transform` 카테고리 4종 (`none/log1p/yeo-johnson/quantile`) 자동 탐색. 선형 모델은 OLS 가정 (잔차 정규성·등분산성)의 영향을 받아 변환 효과가 데이터 의존적.

---

## 25. Optuna study timeout — Colab 세션 끊김 대응

Colab은 24시간 강제 세션 종료. 학원 환경도 장시간 실행 시 세션 타임아웃 위험. **`study.optimize` timeout으로 명시적 제한** + sqlite storage로 다음 세션 resume.

**선언 위치 — 노트북 최상단 (환경 설정 셀)**:

```python
N_TIMEOUT_SEC = 82800   # ★ 23시간 default (Colab 24시간 마진 1시간)
                        #    None → 무제한 (로컬 무한 실행 등)
```

**사용 위치 — Optuna study.optimize**:

```python
study.optimize(
    objective,
    n_trials=N_TRIALS,
    n_jobs=1,
    timeout=N_TIMEOUT_SEC,   # ★ 시간 도달 시 진행 중 trial 완료 후 종료
)
```

**동작**:
- `N_TRIALS` 또는 `N_TIMEOUT_SEC` 중 **먼저 도달하는 시점에 종료**
- 종료 시 진행 중인 trial은 완료까지 기다림 (Optuna가 강제 kill 안 함)
- sqlite storage + `load_if_exists=True`이면 다음 세션에서 자동 resume

**default 결정 원칙**:
- Colab: **82800초 (23시간)** — 24시간 마진 1시간
- 로컬 무한 실행: `None`
- 노트북 작성 시 환경에 맞춰 override 가능

**주의**:
- `timeout=None` 사용 가능 — **무조건 걸지 않음**. 노트북별 선택사항
- 노트북 안에서 hard-coded 숫자 (`timeout=82800` 직접 작성) 금지 — 항상 `N_TIMEOUT_SEC` 변수 참조 (§8 동일 원칙)
- **trial별 timeout은 Optuna 미지원** — 무거운 trial 보호는 별도 메커니즘 필요 (signal/multiprocessing). 예: 0_baseline의 `impute='knn'` cell이 20분+ 걸리는 사례
