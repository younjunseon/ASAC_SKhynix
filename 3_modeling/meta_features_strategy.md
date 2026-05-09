# 메타피처 추가 전략 (전체 노트북 공통 가이드)

`3_modeling/` 하위 모든 노트북에 공통 적용되는 wafer 메타피처 추가 전략. 모델 family별 적용 방식 + 비선형 신호 처리 + leak 분석을 단일 문서로 통합.

> **본 문서는 가이드**입니다. 모델별 구체 실험 리스트는 별도 md 참조:
> - ElasticNet 실험 (reg_only + ts_reg 양쪽 적용): [enet_experiments.md](enet_experiments.md)
> - 트리 모델 실험: 추후 [02_reg_single/tree_experiments.md](02_reg_single/tree_experiments.md) 추가 예정

---

## 0. 한 줄 요약 (2026-05-09 결정)

Wafer 메타피처 적용 범위를 아래로 축소:

| 메타피처 | ElasticNet | 트리 부스팅 (LGBM/XGB/CatBoost/ZITboost) | ExtraTrees |
|---|:---:|:---:|:---:|
| **position** | ✅ OHE 4 | ✅ raw int 1 | ✅ raw int 1 |
| **die_x, die_y** | ❌ **제외** | ✅ continuous | ✅ continuous |
| **lot / wafer_no / lot_wafer** | ❌ **제외** | ❌ **제외** | ❌ **제외** |

target encoding 계열도 모든 모델에서 **금지** (이전자료 효과 미세 + leak 강함).

> **결정 근거**:
> - **lot/wafer/lot_wafer 전 모델 제외**: leak 위험 (트리의 lot raw int split, ElasticNet alpha=5.4e-6 약정규화 환경 모두) + production(새 wafer) 일반화 약화. competition 점수 향상 trade-off 거부.
> - **ElasticNet die_x/die_y 제외**: 선형 단조 가정 한계로 직접 효과 미세, polynomial/spline 보강 시 컬럼 폭증 대비 효과 미미.
> - **position 공통 사용**: unit 안에서 die마다 다른 위치 → leak 0, 가장 안전한 메타피처.

---

## 1. 배경: split 구조와 모델별 표현력 차이

### 1.1 Split 구조 검증

#### Raw csv (`compet_xs_data.csv`) 기준

| | die 수 | lot | wafer_no | lot_wafer |
|---|---:|---:|---:|---:|
| train | 104,988 | 28 | 25 | 432 |
| val | 34,996 | 28 | 25 | 432 |
| test | 34,996 | 28 | 25 | 431 |
| 합계 | 174,980 | 28 | 25 | 432 |

#### `utils.data.load_all()` 호출 후 (실제 모델 입력)

`load_all()` 이 all-NaN 행 407개 + 4 position 미만 unit 1개 (4 die)를 자동 제거 → 모델은 아래 카운트로 학습:

| | die 수 | lot | wafer_no | lot_wafer |
|---|---:|---:|---:|---:|
| train | 104,748 | 28 | 25 | 431 |
| val | 34,908 | 28 | 25 | 431 |
| test | 34,916 | 28 | 25 | 430 |

- **train ∩ val ∩ test (lot_wafer): 430** — 모든 wafer를 셋이 공유 (load_all 후 기준)
- 샘플 wafer `0000000_1` (377 die): train 241 / val 72 / test 64 die — 같은 wafer 안에서도 서로 다른 unit이 train/val/test에 배치된 결과. 같은 unit의 4 die는 항상 같은 split (unit 단위 분할 보장 → 그렇지 않으면 leak)
- val/test 신규 lot/wafer **0개**

→ **lot/wafer ID는 train/val/test 모두 정확히 28/25 카디널리티로 일치, lot_wafer는 거의 일치** (raw csv: 432/432/431, load_all 후: 431/431/430 — test가 1개 적음). ID-기반 dummy/split은 그 그룹의 train 잔차 평균을 흡수 → val/test 예측에 broadcast.

### 1.2 모델별 lot/wafer 정보 처리 차이

| | 트리 부스팅 | ElasticNet (선형) |
|---|---|---|
| 표현력 | `if X42<0.5 ∧ X87>1.2 ∧ …` 형태로 lot/wafer를 X 조합에서 implicit 식별 가능 | `y = β₀ + Σβᵢ·Xᵢ` 글로벌 계수만. lot-conditional / 비선형 / 상호작용 표현 불가 |
| 비선형 표현 | split으로 자동 흡수 (`if X<0.5 then a else b`) | 피처 엔지니어링 필요 (X² / spline / OHE binning) |
| lot/wafer 메타 추가 효과 | 학습 효율 ↑ (depth/leaf 제한 완화) | **유의미한 신호 추가** (mean shift를 dummy로 흡수) |

### 1.3 EDA 근거 — 신호의 본질이 비선형

CLAUDE.md EDA 결과:
- 최대 |r| = 0.037 (X1083), |r|>0.1인 feature **0개**, 평균 |r| = 0.011
- → **단일 feature 선형 신호로는 예측 불가, feature 간 상호작용과 비선형 관계가 핵심**

이 사실 때문에 트리는 강하고 ElasticNet은 약함. ElasticNet에 OHE/spline 기반 비선형 처리는 본질적으로 표현력 보강 의미.

---

## 2. 메타피처별 위험도·효과 매트릭스 (2026-05-09 결정 반영)

| 메타피처 | 카디널리티 | leak 위험 | 모델별 권장 |
|---|---:|---|---|
| `position` (1~4) | 4 | ✅ 없음 (unit 안에서 die마다 다름) | ✅ **공통 적용** — enet OHE 4, 트리/ET raw int 1 |
| `die_x`, `die_y` (continuous) | 연속 | ✅ 없음 | ✅ 트리/ExtraTrees, ❌ **ElasticNet 제외** (선형 단조 한계) |
| 위치기반 X 5개 (`X1059/X1073/X1075/X1076/X1077`) | nunique 6~11 | ✅ 없음 (X에서 파생) | enet OHE (X1073 필수), 트리는 raw int 그대로 — **메타피처 아닌 X 피처 처리** |
| `lot` raw int | 28 | ⚠️ 약 | 🚫 **모든 모델 제외** (2026-05-09) |
| `wafer_no` raw int | 25 | ⚠️ 약 | 🚫 **모든 모델 제외** (2026-05-09) |
| `lot_wafer` 묶음 | 431 | 🔴 강 (group memorization) | 🚫 **모든 모델 제외** (2026-05-09) |
| Target Encoding (lot/wafer mean health 등) | — | 🔥 명백한 leak | 🚫 **금지** (모든 모델) |
| Per-lot centered X (lot mean 뺀 deviation) | — | ✅ 없음 (target 안 건드림) | 🚫 **자동 폐기** (lot 메타 자체를 안 씀) |

### 2.1 Target Encoding 명시적 제외 사유

- 이전자료 [final/EXPERIMENT_LOG.md §6](../3_modeling_이전자료/final/EXPERIMENT_LOG.md) (BagZIT) 결과: A_baseline 0.005524 → B_+12enc 0.005489 (-0.64% OOF). 효과 미세 + 같은 wafer의 train health → val/test 예측에 직접 broadcast = leak
- 사용자 결정: **명시적 GroupKFold target encoding 모든 모델 금지** (lot_te / wafer_te / lp_te / wp_te 같은 12 enc cols)

> ~~**예외 — CatBoost `cat_features` (Ordered TS)**~~ — lot/wafer 자체를 제외하므로 검토 자동 폐기 (2026-05-09).

### 2.2 lot/wafer 제외 결정 근거 (2026-05-09)

> **이전 분석 (참고용 보존)**: ElasticNet `lot_A` dummy 계수 = train lot A의 잔차 평균 broadcast (약 leak). 트리 `if lot==A` split = lot A의 train health 평균 직접 외움 (강 leak). 같은 메타피처라도 트리에서 더 위험. wafer-within split이라 competition test 점수엔 fair 반영, **production(새 wafer) 일반화는 enet < 트리 모두 약화**.

**결정**: competition 점수 향상보다 production 일반화 + leak-free 우선 → **lot / wafer_no / lot_wafer 전 모델 제외**.

---

## 3. 모델 family별 적용 트랙

각 family별로 적용할 메타피처 + 처리 방식. 구체 실험 리스트는 family별 실험 md 참조.

### 3.1 ElasticNet 트랙 — `02_reg_single/enet.ipynb` 한정 (2026-05-09 축소)

| 단계 | 메타피처 / X 처리 | 처리 | leak |
|---:|---|---|---|
| 1 | position | OHE 4 | 0 |
| 2c | EXCLUDE_COLS에 X1056, X1072 추가 | 명시 제외 (자동 cleaning 변동 대비) | — |
| 2a | X1073 (4분면 sector) | **OHE 6 필수** (monotonic 가정 명백히 틀림) | 0 |
| 2b | X1059/X1075/X1076/X1077 | OHE 32 (A/B로 ordinal vs OHE 결정) | 0 |
| ~~3~~ | ~~die_x, die_y~~ | ❌ **제외** (선형 단조 한계, polynomial/spline 보강 비용 대비 효과 미미) | — |
| ~~4~~ | ~~lot OHE 28~~ | ❌ **제외** (leak 위험 + production 일반화 약화) | — |
| ~~5~~ | ~~per-lot centered X~~ | ❌ **자동 폐기** (lot 메타 자체를 안 씀) | — |
| 6 (옵션) | 트리 importance top 20 X에 SplineTransformer | 비선형 흡수 (X 피처 처리, 메타 아님) | 0 |
| 7 (옵션) | ColumnTransformer (dummy passthrough) | scaler 우회 | — |

상세 실험 리스트 (reg_only + ts_reg 양쪽 적용): [enet_experiments.md](enet_experiments.md)

### 3.2 트리 부스팅 트랙 — LightGBM / XGBoost / CatBoost / ZITboost (2026-05-09 축소)

| 단계 | 메타피처 | 처리 | leak |
|---:|---|---|---|
| 1 | position | raw int 1 | 0 |
| 2 | die_x, die_y | continuous (트리는 split 자동 비선형) | 0 |
| ~~3~~ | ~~lot_wafer 묶음~~ | ❌ **제외** (leak 위험 + production 일반화 약화) | — |

> **2026-05-09 결정**: 트리 부스팅도 lot/wafer/lot_wafer 메타 전부 제외.
> - LightGBM `categorical_feature` / XGBoost `enable_categorical` / CatBoost `cat_features` 도입 검토 **모두 폐기**.
> - CatBoost Ordered TS leak 평가, ZITboost `cat_feature_indices` 인자 추가 작업 (zit.py ~30줄 + 노트북 ~5줄) **모두 불필요**.
> - 트리 부스팅 메타 적용 작업 = `position` (raw int) + `die_x`/`die_y` (continuous) 2개 단계로 끝.

### 3.3 ExtraTrees 트랙 — `02_reg_single/et.ipynb` + 03_two_stage 의 et (2026-05-09 축소)

| 단계 | 메타피처 | 처리 | 비고 |
|---:|---|---|---|
| 1 | position | raw int 1 | 0 |
| 2 | die_x, die_y | continuous | 0 |
| ~~3~~ | ~~lot raw int + wafer_no raw int~~ | ❌ **제외** (leak 위험 + production 일반화 약화) | — |

> ExtraTrees도 트리 부스팅과 동일하게 `position` + `die_x`/`die_y` 2개 단계로 끝. lot_wafer 묶음 / lot+wafer_no 분리 둘 다 폐기.

---

## 4. 적용 노트북 매핑

전체 `3_modeling/` 하위 노트북별 적용 범위. 자동 전파 안 되니 노트북마다 헬퍼 호출 추가 필요.

| 노트북 | family | 메타 처리 (2026-05-09) | 우선순위 | 상세 가이드 |
|---|---|---|---|---|
| [02_reg_single/enet.ipynb](02_reg_single/enet.ipynb) | enet | pos OHE 4 | 🥇 P0 | [enet_experiments.md](enet_experiments.md) |
| [02_reg_single/lgbm.ipynb](02_reg_single/lgbm.ipynb) | 트리 부스팅 | pos + die_xy | 🥇 P0 | tree_experiments.md (추후) |
| [02_reg_single/xgb.ipynb](02_reg_single/xgb.ipynb) | 트리 부스팅 | pos + die_xy | 🥇 P0 | 〃 |
| [02_reg_single/catboost.ipynb](02_reg_single/catboost.ipynb) | 트리 부스팅 | pos + die_xy | 🥇 P0 | 〃 |
| [02_reg_single/et.ipynb](02_reg_single/et.ipynb) | ExtraTrees | pos + die_xy | 🥇 P0 | 〃 |
| [01_zit/01_zit_only.ipynb](01_zit/01_zit_only.ipynb) | ZITboost | pos + die_xy | 🥈 P1 | (별도 검토 필요 — Stage 1 분류 + Stage 2 회귀 구조) |
| [01_zit/02_bag_zit.ipynb](01_zit/02_bag_zit.ipynb) | ZITboost | pos + die_xy | 🥈 P1 | 〃 |
| [03_two_stage/default/reg/{enet,lgbm,xgb,catboost,et}.ipynb](03_two_stage/default/reg/) | family 동일 | family 동일 | 🥉 P2 | enet은 [enet_experiments.md](enet_experiments.md) 양쪽 동시 적용. 다른 family는 02_reg_single 결과 검증 후 동일 패턴 복제 |
| [03_two_stage/default/clf/{lgbm,xgb,catboost,et}.ipynb](03_two_stage/default/clf/) | 분류 | family 동일 (분류 효과는 별도 검증) | 🥉 P2 | 분류는 메타피처 효과가 회귀와 다를 수 있음 |
| [03_two_stage/reverse/ts_reverse.ipynb](03_two_stage/reverse/ts_reverse.ipynb) | 분류 → 회귀 reverse | family 동일 | 🥉 P2 | |
| [04_stacking/stacking.ipynb](04_stacking/stacking.ipynb) | meta-learner | base OOF만 받음 | 📌 자동 | base 노트북이 메타피처 OOF 저장 시 자동 반영. meta-learner에 추가 메타피처는 별도 검토 |
| [0_baseline/](0_baseline/) | OAT/group study | — | ❌ 적용 외 | 탐색 노트북 |

**진행 순서 권장**:
1. **P0 (02_reg_single)** — 5개 모델 baseline 측정 → 메타피처 추가 효과 검증
2. **P1 (01_zit)** — ZITboost가 P0 결과와 일관 행동하는지 확인
3. **P2 (03_two_stage)** — Stage 1/2 분리 효과 측정
4. **04_stacking** — base가 메타피처 적용 OOF 생성하면 자동 적용. meta-learner 자체엔 추가 안 해도 됨

---

## 5. 헬퍼 모듈 시그니처

[2_preprocessing/meta_features.py](../2_preprocessing/meta_features.py) — 기존 파일에 `add_meta_features()` 함수 추가 (이미 `parse_run_wf_xy()` 존재). 모든 노트북에서 한 줄 호출.

```python
def add_meta_features(xs_train, xs_val, xs_test, feat_cols,
                       use_position=True,        # bool — 모든 모델 True
                       position_mode='ohe',      # 'ohe' (enet) | 'raw' (트리/ET)
                       use_die_xy=True,          # bool — 트리/ET True, ElasticNet False
                       use_loc_x_ohe=False,      # enet 전용: 위치기반 X 5개 OHE
                       loc_x_required=("X1073",),
                       loc_x_optional=("X1059", "X1075", "X1076", "X1077"),
                       ):
    """
    모든 모델 family 공통. 모델별 인자 조합 (2026-05-09 결정 반영):

      ElasticNet:
        position_mode='ohe', use_die_xy=False, use_loc_x_ohe=True

      LightGBM/XGBoost/CatBoost/ZITboost/ExtraTrees:
        position_mode='raw', use_die_xy=True, use_loc_x_ohe=False

    lot/wafer/lot_wafer 메타는 전 모델 제외 결정 (2026-05-09) → 인자 없음.
    die_xy_nonlinear (poly/spline) 옵션도 ElasticNet die_x/die_y 제외로 폐기.

    Returns
    -------
    new_feat_cols : list[str]
    cat_cols : list[str]   — 빈 리스트 (lot_wafer 제외로 categorical 네이티브 처리 무관)
    """
```

**왜 단일 함수**: 메타피처 정의를 한 곳에서 관리. 노트북에는 인자만 다르게 1줄 호출. 변경 시 한 함수만 수정.

### 5.1 enet 전용: ColumnTransformer 분기

위 헬퍼 호출 후 enet은 추가로 ColumnTransformer 도입 (옵션). **반드시 fold loop *안*에서 매 fold `ct.fit_transform` 호출** — train fold 기준으로 X 피처 통계를 fit해야 leak 없음. fold loop 밖에서 한 번만 fit하면 다른 fold의 train 정보가 새는 leak 발생.

```python
from sklearn.compose import ColumnTransformer

dummy_cols  = [c for c in feat_cols_clean
               if "_eq" in c
               or c.startswith(("pos_", "lot_"))
               or c.endswith("_missing")]   # cleaning.py {col}_missing 형식 indicator 포함
x_only_cols = [c for c in feat_cols_clean if c not in dummy_cols]

# ↓ fold loop 안에서 매 fold 새로 fit (★ 절대 fold loop 밖에서 fit 금지)
for tr_units, vl_units in FOLDS:
    tr_mask = ...
    vl_mask = ...
    ct = ColumnTransformer([
        ("scale",    make_scaler(scaling_name), x_only_cols),
        ("passthru", "passthrough",             dummy_cols),
    ])
    X_tr_s = ct.fit_transform(xs_train_c.loc[tr_mask, feat_cols_clean])
    X_vl_s = ct.transform(xs_train_c.loc[vl_mask, feat_cols_clean])
    # val/test도 동일 ct로 transform (refit 단계에서)
```

**적용 여부는 enet 실험 단계 7에서 A/B로 결정** (RMSE 차이 미세 예상).

---

## 6. 비선형 신호 처리 (ElasticNet 전용) — 2026-05-09 부분 폐기

> **2026-05-09 결정 반영**: ElasticNet에서 die_x/die_y 자체를 제외하므로 die_x/die_y polynomial / spline 논의 (§6.1, §6.3-1) 자동 폐기. 위치기반 X OHE (§6.3-2) 와 트리 importance top 20 X spline (§6.3-3) 은 X 피처 처리이므로 **유효**.

### ~~6.1 문제: 단순 좌표는 ElasticNet에 단조 효과만 학습~~ (폐기)

> ~~예시: die_x/die_y 가 health에 U-shape 영향. ElasticNet은 직선 효과만 학습 가능 → U-shape 흡수 불가.~~ → die_x/die_y 자체를 ElasticNet 입력에서 제외하므로 무관 (2026-05-09).

### 6.2 해결법 (X 피처 한정으로 유효)

| 방법 | 어떻게 | U-shape 흡수 | 비용 | 권장도 |
|---|---|---|---|---|
| ~~**다항 항**~~ | ~~`die_x` + `die_x²` 두 컬럼~~ | ~~✅ β₁<0, β₂>0 면 U~~ | ~~컬럼 ×2~~ | ❌ **폐기** (die_x/die_y 자체 제외) |
| **비선형 변환** | log/sqrt/inv (양수 입력 한정) | 단조 비선형만 | 컬럼 +N | ⚠️ 음수 가능 X엔 부적용. 일반 X는 PowerTransformer/Yeo-Johnson 권장 |
| **Binning + OHE** | X를 5~10 구간 dummy | 임의 비선형 흡수 | 컬럼 ×bin | ✅ 효과 강함 (X1073 OHE가 이 카테고리) |
| **Spline** | `SplineTransformer(n_knots=5)` | 부드러운 비선형 | 컬럼 ×knot수 | ✅ 트리 importance top 20에 한정 적용 (fold-안 fit 필수) |

### 6.3 적용 우선순위 (수정)

1. ~~**die_x, die_y polynomial** — 단계 3 추가~~ ❌ **폐기** (die_x/die_y 자체 제외)
2. **위치기반 X 5개 OHE** — 단계 2 (이미 md에 박힘). binning 의 일종
3. **트리 importance top 20 X에 spline** — 단계 6 (옵션). knot 4~5개

**원칙**: 1087 X 전체에 일괄 polynomial/spline 적용 X (약신호 노이즈 폭증). 트리 importance 또는 EDA 결과로 선별한 핵심 피처에만 적용.

### 6.4 트리는 적용 외

트리는 split으로 비선형 자동 흡수 → polynomial/spline 추가 무의미 또는 redundant. 트리에는 raw 그대로 입력 (die_x/die_y 포함).

---

## 7. 검증 프로토콜 (모든 모델 공통)

각 단계마다:

1. **anchor params 고정** — 각 모델 노트북의 anchor (예: `ENET_ANCHOR`)
2. **5-fold KFold seed 동일** (`SEED=42`)
3. 5 trial 빠른 HPO + 1 best refit → `oof / val / test RMSE` 기록
4. **Δval_rmse ≥ 0** 이면 그 단계 reject, 다음 단계 진행 안 함
5. 결과는 family별 실험 md 결과 기록표에 추가

### 7.1 Marginal 효과 측정 시 주의

- **fold별 RMSE 5개의 mean과 std** 까지 보기 (KFold OOF 단일 array의 표준편차가 아니라, 각 fold의 RMSE 5개 → mean/std). 단계 추가의 RMSE 변화 폭이 fold별 std보다 작으면 noise → reject 권장
  ```python
  fold_rmses = []
  for fold_i, (tr, vl) in enumerate(FOLDS):
      ...   # fit + predict
      fold_rmses.append(rmse_fold_i)
  print(f"fold RMSEs: {fold_rmses}")
  print(f"mean={np.mean(fold_rmses):.6f}, std={np.std(fold_rmses):.6f}")
  ```
- PP 결과가 trial마다 달라지면 marginal effect 분리 어려움 → **PP를 best 1set으로 고정** 한 채 메타피처만 변동
- 모델별로 anchor params 차이 있어 동일 메타피처도 RMSE 영향 다를 수 있음 → 모델 family 내 일관성 우선

### 7.2 Stage 1 (분류) vs Stage 2 (회귀) 차등

03_two_stage 분류 노트북은 메타피처 효과가 회귀와 다를 수 있음:
- lot_wafer 같은 group identifier가 분류 정확도에 더 큰 영향 가능
- Recall/F1 metric 기준으로 별도 측정 필요

---

## 8. 참고

- 위치기반 8개 → 6개 명시 결정: [3_modeling_이전자료/two_stage/docs/strategy_2nd_ensemble.md "공간 패턴 feature 수동 선별 결과 (2026-04-16)"](../3_modeling_이전자료/two_stage/docs/strategy_2nd_ensemble.md)
  - X708/X1059/X1073/X1075/X1076/X1077 유지, X1056/X1072 제거
- Anchor cleaning 후 실측 생존: X1059/X1073/X1075/X1076/X1077 (5개). X708은 corr_threshold=0.90 단계에서 X1059와의 r=0.964로 자동 제거
- Target Encoding 미세 효과 근거: [3_modeling_이전자료/final/EXPERIMENT_LOG.md §6](../3_modeling_이전자료/final/EXPERIMENT_LOG.md) (BagZIT, lot/wafer/lp/wp × 3종 = 12 enc cols, OOF -0.64%)
- e2e_hpo OHE p_1~p_4 코드 참조: [3_modeling_이전자료/modules/e2e_hpo.py:931-959](../3_modeling_이전자료/modules/e2e_hpo.py#L931-L959) (`reg_level='position'` 분기)
- Direct comparison data 부재 — 4_output 모든 best_params.json n_features=568~573, position 미적용. 본 전략은 신규 측정 필요
