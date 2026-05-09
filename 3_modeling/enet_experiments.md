# ElasticNet 메타피처 실험 리스트

ElasticNet 노트북 **2개에 동일하게 적용**되는 실험 리스트:

- [02_reg_single/enet.ipynb](02_reg_single/enet.ipynb) — reg_only (전체 health 회귀)
- [03_two_stage/default/reg/enet.ipynb](03_two_stage/default/reg/enet.ipynb) — ts_reg (Two-Stage 의 Stage 2, Y>0 서브셋 회귀)

우선순위 순으로 진행, 각 단계마다 val/test RMSE 측정 → Δval ≥ 0 이면 reject.

> 전체 모델 공통 가이드는 [meta_features_strategy.md](meta_features_strategy.md) 참조. 본 문서는 **enet 한정 구체 실험 리스트**.

---

## 0. 진행 원칙

- **anchor params 고정**: 각 노트북 cell 4의 `ENET_ANCHOR` 그대로 (reg_only / ts_reg 각각 별도 정의 — 두 노트북 anchor 값이 다를 수 있음)
- **5-fold KFold seed=42** 고정
- 단계별 5 trial 빠른 HPO + 1 best refit
- **Δval_rmse ≥ 0 이면 그 단계 reject**, 다음 단계 진행 안 함
- 베이스라인 (단계 0) 측정 먼저 → 모든 후속 단계는 baseline 대비 Δ로 평가
- 메타피처는 PP(`preprocess.run`) **이후 / scaler 이전**에 추가
- 결과는 [§3 결과 기록표](#3-결과-기록표) 채워가기 — **reg_only / ts_reg 별도 기록**

### 0.1 두 노트북 구조 일치 — 동일 코드 패치 적용 가능

| 영역 | reg_only cell | ts_reg cell | 비고 |
|---|---:|---:|---|
| ENET_ANCHOR 정의 | 4 | 4 | 별도 anchor 값 |
| objective 정의 | 10 | 10 | 동일 패턴 |
| refit (best params) | 12 | 12 | 동일 패턴 |
| final summary | 14 | 14 | 동일 패턴 |

→ 본 md의 모든 단계 코드 패치는 **양쪽 노트북에 동일 위치 (cell 10/12) 에 동일 코드** 로 적용 가능.

### 0.2 ts_reg 의 차이점 — Y_POSITIVE_ONLY 마스크와 메타피처는 직교

ts_reg cell 10 fold loop 안에는 `Y_POSITIVE_ONLY=True` 분기로 `pos_mask = y_tr_orig > 0` 마스크가 X_tr / y_tr 에 적용됨 (Stage 2: Y>0 서브셋만 fit). 이는 본 md의 메타피처 작업과 **직교**:

- 메타피처(OHE/poly/centered) 추가는 **PP 이후 / fold loop 진입 *이전***. 이 시점에 xs_train_c 의 모든 die (Y=0 포함) 에 컬럼이 박힘.
- Y_POSITIVE_ONLY 마스크는 fold loop 안의 X_tr ndarray 에서 행 선택만. 컬럼 구조 안 바뀜.
- → 동일 코드로 양쪽 노트북에 적용 안전.

~~**의도된 부수 효과**: Y_POSITIVE_ONLY=True 인 ts_reg 에서 일부 lot 이 Y>0 서브셋에 한 die 도 없으면 그 lot dummy 는 X_tr_fit 에서 all-zero 상수 → ElasticNet coef 자동 0.~~ → lot OHE 폐기 (2026-05-09)로 무관.

---

## 1. 우선순위 매트릭스 (2026-05-09 축소)

| # | 실험 | 대분류 | 우선순위 | 추가 컬럼 | leak | 예상 효과 |
|---:|---|---|---|---:|---|---|
| **0** | baseline (X+ind 572) | 측정 기준 | 🥇 P0 (선행) | 0 | — | — |
| **1** | + position OHE 4 | 메타 추가 | 🥇 P0 | +4 | 0 | 작아도 일관 + |
| **2c** | EXCLUDE_COLS에 X1056, X1072 명시 | 안정성 | 🥇 P0 | -2 (cleaning과 동일) | — | corr_threshold 변동 대비 |
| **2a** | + X1073 OHE 6 (4분면 sector, 원본 1개 제거) | 비선형 흡수 | 🥇 P0 (필수) | +5 | 0 | **강함** (monotonic 가정 명백히 틀림) |
| **2b** | + X1059/1075/1076/1077 OHE 32 (원본 4개 제거) | 비선형 흡수 | 🥈 P1 | +28 | 0 | + 가능 (A/B로 ordinal vs OHE 비교) |
| ~~**3**~~ | ~~+ die_x, die_y (continuous)~~ | — | ❌ **폐기** | — | — | ElasticNet die_x/die_y 제외 결정 (2026-05-09) |
| ~~**3'**~~ | ~~+ die_x², die_y² (polynomial)~~ | — | ❌ **폐기** | — | — | 단계 3 폐기 → 자동 폐기 |
| ~~**4**~~ | ~~+ lot OHE 28~~ | — | ❌ **폐기** | — | — | lot/wafer 전 모델 제외 결정 (2026-05-09) |
| ~~**5**~~ | ~~+ per-lot centered X~~ | — | ❌ **폐기** | — | — | lot 메타 자체를 안 씀 → 자동 폐기 |
| **6** | + 트리 top 20 X에 SplineTransformer | 비선형 흡수 (옵션, X 피처 한정) | 📌 P3 (옵션) | +80~100 | 0 | 트리와 RMSE gap 좁힘 |
| **7** | ColumnTransformer (dummy passthrough) | 처리 정밀화 | 📌 P3 (옵션) | 0 | — | 미세 (RMSE 차이 사실상 noise) |
| ❌ | wafer_no OHE 25 | 비추 | — | — | 약 | 2026-05-09 결정으로 자동 제외 |
| ❌ | lot_wafer OHE 431 | 비추 | — | — | 강 | 2026-05-09 결정으로 자동 제외 |
| 🚫 | Target Encoding | **금지** | — | — | 명백 | 이전자료 -0.64% (효과 미세 + leak 위험 대비 수익 작음) |

> **2026-05-09 결정**: ElasticNet은 `position OHE` + `위치기반 X OHE` (X1073 등) + 옵션 (spline / ColumnTransformer) 만 진행. die_x/die_y / lot / per-lot centered 단계 모두 폐기.

---

## 2. 단계별 상세

### 단계 0 — Baseline 측정

**목적**: 모든 후속 단계 비교 기준. 각 노트북 (`02_reg_single/enet.ipynb`, `03_two_stage/default/reg/enet.ipynb`) 그대로 5 trial 돌려 값 확보.

**조건**: PP anchor params 고정, scaling=`'RobustScaler'`, target_transform=`'log1p'`, n_features=572 (568 X + 4 indicator).

**산출**: oof_rmse, val_rmse, test_rmse 1세트 → [§3 결과 기록표](#3-결과-기록표) row 0 채움 (**reg_only / ts_reg 별도**).

**선행 조건**: HybridScaler 버그 수정 (별건). HybridScaler 사용 안 하면 나머지 4 scaler (Standard/Robust/YeoJohnson/Quantile) 로 baseline 가능.

---

### 단계 1 — Position OHE (필수)

**근거**: position 1~4는 unit 안에서 die마다 다른 위치. ordinal 가정(1<2<3<4 monotonic) 부적절. OHE로 풀면 각 position이 독립 계수로 health 영향 흡수.

**구현 위치**: 양쪽 노트북 모두 cell 10 `objective` 함수 안 — `preprocess.run` 호출 직후 `feat_cols_clean` 확정 시점. **반드시 fold loop *진입 이전*** (fold loop 안에 넣으면 매 fold dataframe 컬럼 재추가 → 비효율 + 의도 깨짐). cell 12 refit 코드에도 동일 패턴 복제.

```python
# objective 함수 내부, PP 직후, fold loop 진입 *전*
for p in [1, 2, 3, 4]:
    xs_train_c[f"pos_{p}"] = (xs_train_c["position"] == p).astype(np.int8)
    xs_val_c[f"pos_{p}"]   = (xs_val_c["position"] == p).astype(np.int8)
    xs_test_c[f"pos_{p}"]  = (xs_test_c["position"] == p).astype(np.int8)
feat_cols_clean = list(feat_cols_clean) + [f"pos_{p}" for p in [1,2,3,4]]

# ↓ 이 시점부터 fold loop 진입
```

**원본 `position` 컬럼 처리**: `position` 은 [utils/data.py](../utils/data.py) `get_feat_cols()` 가 X 시작만 반환하므로 **`feat_cols_clean` 에 처음부터 없음**. 별도 제거 불필요. dummy 4개만 추가하면 끝.

**Scaler 호환**:
- HybridScaler → binary passthrough 자동 분류 (0/1 유지)
- 다른 scaler → 0/1이 임의 두 실수로 변환되지만 β 자동 비례 조정 → RMSE 사실상 동일

**ts_reg 추가 메모**: Y_POSITIVE_ONLY 마스크 후에도 4 position 모두 Y>0 서브셋에 존재할 가능성 매우 높음 (position 별 health 분포 큰 차이 없음). 사실상 reg_only 와 동일 효과.

**예상 효과**: 작아도 일관 +. competition 점수 미세 향상.

---

### 단계 2c — EXCLUDE_COLS에 X1056, X1072 명시 추가

**근거**: 위치기반 X 8개 중 **X1056, X1072는 이전자료에서 명시 제거 결정**:
- X1056: Ring (타원 고리), 기여 모호
- X1072: X708과 r=0.997 → 사실상 중복

자동 cleaning(corr_threshold=0.90)에서도 자연스레 제거되지만, HPO에서 corr_threshold가 0.95~0.98 sample되면 X1072가 살아남을 가능성. 명시 제외로 안정성 ↑.

> X1072 쌍 처리 주의: r=0.997 X708/X1072 쌍에서 누가 살아남는지는 `corr_keep_by='std'` 결정 — std 비교 결과에 따라 X708 이 살 수도, X1072 가 살 수도. 명시 제외로 결정 의존성 끊음.

**구현 위치**: [3_modeling/modules/preprocess.py:56-73](modules/preprocess.py#L56-L73)

[3_modeling/modules/preprocess.py:56-73](modules/preprocess.py#L56-L73) 의 기존 EXCLUDE_COLS는 **54개** (X124/X300/X301, X441~X464, X499~X506, X658~X687 일부, X1041/X1074/X1078/X1086). 그 끝에 X1056, X1072 두 줄 추가:

```python
EXCLUDE_COLS = [
    # ... 기존 54개 ...
    # 위치기반 추가 제외 (이전자료 strategy_2nd_ensemble.md 결정)
    "X1056",   # Ring (타원 고리) — 기여 모호
    "X1072",   # Radial gradient — X708과 r=0.997 (사실상 중복)
]
# 총 56개
```

**영향 범위**: preprocess.run 호출하는 **모든 모델**. 즉 enet 양쪽 (reg_only + ts_reg) 뿐 아니라 트리/ZITboost 노트북에도 동일 적용. 단계 2c는 **enet 단독 실험 아님 — 전체 영향**.

**예상 효과**: 변화 없음 (anchor 0.90에서 자동 제거되던 컬럼). 그러나 전체 노트북의 안정성 ↑.

---

### 단계 2a — X1073 (4분면 sector) OHE (필수)

**근거**: X1073 nunique=6, 1~6 임의 sector ID. monotonic 가정(1<6 의미 있음) 명백히 틀림. ElasticNet에 ordinal로 들어가면 잘못된 신호 학습.

**구현**: PP 후 / 단계 1 직후 / fold loop 진입 *이전* 에서 실행. **train 카테고리 기준으로만 dummy 생성** + val/test에 train에 없는 새 카테고리 있으면 **모든 dummy=0 으로 가드** (정보 손실 발생하지만 임의 신규 카테고리 처리 안전).

```python
# objective 함수 내부, preprocess.run 결과 받은 직후, fold loop *전*
for col in ["X1073"]:
    if col not in feat_cols_clean:
        continue
    cats = sorted(xs_train_c[col].dropna().unique().tolist())   # ★ train 카테고리 기준
    cat_cols = [f"{col}_eq{int(cat)}" for cat in cats]

    # train: 카테고리별 dummy
    for cat, cat_col in zip(cats, cat_cols):
        xs_train_c[cat_col] = (xs_train_c[col] == cat).astype(np.int8)

    # ★ val/test: 동일 카테고리 기준만 매칭 (신규 카테고리는 자동으로 모든 dummy=0)
    for split_df in (xs_val_c, xs_test_c):
        for cat, cat_col in zip(cats, cat_cols):
            split_df[cat_col] = (split_df[col] == cat).astype(np.int8)
        # 진단용: train에 없는 카테고리 행 수 확인 (디버그 시)
        # n_unseen = (~split_df[col].isin(cats)).sum()
        # print(f"  [{col}] train에 없는 카테고리 행: {n_unseen}")

    feat_cols_clean = [c for c in feat_cols_clean if c != col] + cat_cols
```

**가드 동작**: val/test에 train에 없는 X1073 값이 있으면 그 행은 모든 `X1073_eq*` dummy = 0. 원본 X1073 컬럼은 제거되므로 그 행은 X1073 정보 자체를 잃음. **wafer-within split이라 X1073 cardinality 6이 셋 모두에 있을 가능성 높지만, 단계 4의 `pd.get_dummies(...).reindex(fill_value=0)` 패턴과 일관성 확보 + 안전 가드** 차원에서 명시.

> 단계 2b의 X1059/X1075/X1076/X1077 OHE도 **동일한 train-카테고리 기준 가드** 패턴 적용.

**왜 원본 X1073 제거**: ordinal int + OHE 동시 유지하면 multicollinearity + 정보 중복.

**왜 drop_first=False**: ElasticNet의 L1/L2가 redundant dummy 자동 0으로 떨굼.
- ⚠️ **ENET_ANCHOR alpha=5.4e-6 (약한 정규화) 환경에서 다중공선성 우려**: position OHE 4 + fit_intercept=True → sum-to-1 perfect collinearity. 약한 정규화는 symmetry breaking 불완전 → coordinate descent 수렴 시 어느 dummy 가 살아남느냐가 임의로 결정 (RMSE 자체는 거의 영향 없음, 계수 해석만 비결정).
- → 단계 채택 후 안정성 확보 위해 `drop_first=True` 또는 `fit_intercept=False` A/B 측정 권장. RMSE 차이 noise 수준이면 drop_first=False 유지.

**컬럼 변화**: 1 (원본 제거) + 6 (dummy) = 순증 +5.

**예상 효과**: **강함** — 비-monotonic 패턴 흡수. 단계 1보다 더 큰 RMSE 개선 기대.

---

### 단계 2b — X1059/X1075/X1076/X1077 OHE (A/B 결정)

**근거**: X1073과 달리 이 4개는 좌표/strip index — monotonic 신호도 *가능*. ordinal 유지 vs OHE 변환 A/B로 비교.

| 피처 | nunique | 패턴 | ordinal 적합성 |
|---|---:|---|---|
| X1059 | 11 | Y축 수평 밴드 | 둘 다 가능 |
| X1075 | 9 | 복합 블록 | OHE 우세 |
| X1076 | 6 | Y축 strip | 둘 다 가능 |
| X1077 | 6 | X축 strip | 둘 다 가능 |

**A 시나리오 (ordinal 유지)**: 단계 2a까지만 적용, 4개는 raw int 그대로
**B 시나리오 (모두 OHE)**: 4개도 OHE → +32 dummies (원본 4개 제거 시 순증 +28)

**진행**: B 시나리오로 한 번 측정 → A 대비 val RMSE 비교. B가 우세하면 채택.

**예상 효과**: 미세~중간 수준. X1075(복합 블록)에서 가장 큰 효과 기대.

---

### ~~단계 3 — die_x, die_y (continuous)~~ ❌ 폐기 (2026-05-09)

> **폐기 사유**: ElasticNet die_x/die_y 제외 결정. 선형 단조 가정 한계 + polynomial/spline 보강 비용 대비 효과 미미. 트리/ExtraTrees는 [meta_features_strategy.md](meta_features_strategy.md) §3.2/§3.3 에서 계속 사용.

> **이전 검토 (참고용 보존)**: wafer-within split이라 leak 0. continuous로 RobustScaler/Quantile에 자연 흡수. `radial_dist`, `is_edge` 는 EDA Phase 24 무효 확인.

---

### ~~단계 3' — die_x², die_y² (polynomial 비선형 흡수)~~ ❌ 폐기 (2026-05-09)

> **폐기 사유**: 단계 3 폐기 → 자동 폐기. die_x/die_y 자체가 ElasticNet 입력에 없으므로 polynomial 항도 무관.

---

### ~~단계 4 — lot OHE 28~~ ❌ 폐기 (2026-05-09)

> **폐기 사유**: lot/wafer/lot_wafer 전 모델 제외 결정 (2026-05-09). competition 점수 향상보다 production 일반화 + leak-free 우선.

> **이전 검토 (참고용 보존)**: lot raw ID 28종, wafer-within split이라 train/val/test 모두 같은 28 lot 공유. dummy 계수 = 그 lot의 train 잔차 평균 broadcast (약한 leak). ENET_ANCHOR `alpha=5.4e-6` 는 약 정규화라 lot dummy 효과 살아남기 쉬워 leak 영향 작지 않았을 가능성. 의식적 trade-off 였으나 결정으로 폐기.

---

### ~~단계 5 (옵션) — Per-lot centered X (target-leak-free 대안)~~ ❌ 폐기 (2026-05-09)

> **폐기 사유**: lot 메타 자체를 안 씀 → per-lot centered도 자동 폐기. lot_means_tr 계산이 무의미.

> **이전 검토 (참고용 보존)**: lot 정보를 target leak 없이 주는 방법. 같은 lot 안의 X 평균을 빼서 "die가 자기 lot에서 얼마나 벗어났는가"만 학습. fold-local lot_means + dummy 분리 + 컬럼 ~563 추가 검토했으나 단계 4 폐기로 자동 폐기.

---

### 단계 6 (옵션) — 트리 importance top 20 X에 SplineTransformer

**근거**: ElasticNet은 단일 X 컬럼에 monotonic 효과만 학습. 트리가 학습한 비선형 신호는 못 잡음. SplineTransformer로 부드러운 비선형 흡수.

**선행 조건**: lgbm 노트북에서 best 모델의 `feature_importance(gain)` top 20 X 피처 추출 → 이 20개에만 spline 적용.

**중요 — fold-안 fit 필수**: `SplineTransformer.fit` 의 knot 위치는 train 분위수에 의해 결정. fold loop 밖에서 한 번 fit하면 다른 fold val의 분위수가 knot에 반영 = leak 발생. **반드시 fold loop 안에서 매 fold 새로 spline.fit** (단계 1, 2a, 2b 같은 단순 OHE는 카테고리 식별만이라 fold-밖 OK, spline은 통계 추정이라 fold-안 필수).

**구현**:
```python
from sklearn.preprocessing import SplineTransformer

TOP_X = ["X384", "X385", ...]  # lgbm best의 feature_importance top 20

# ★ fold loop 안에서 매 fold 새로 fit (objective / refit 둘 다)
for tr_units, vl_units in FOLDS:
    tr_mask = xs_train_c[KEY_COL].isin(set(tr_units)).values
    vl_mask = xs_train_c[KEY_COL].isin(set(vl_units)).values

    spline = SplineTransformer(n_knots=5, degree=3, include_bias=False)
    spline.fit(xs_train_c.loc[tr_mask, TOP_X])   # ★ train fold 만 fit

    spline_cols = [f"spl_{c}" for c in spline.get_feature_names_out(TOP_X)]

    # train fold / val fold / val / test 모두 동일 spline으로 transform
    arr_tr = spline.transform(xs_train_c.loc[tr_mask, TOP_X])
    arr_vl = spline.transform(xs_train_c.loc[vl_mask, TOP_X])
    arr_v  = spline.transform(xs_val_c[TOP_X])
    arr_te = spline.transform(xs_test_c[TOP_X])

    # X_tr_s = np.hstack([X_tr_s, arr_tr]) 같은 형태로 fold-local로만 사용
    # feat_cols_clean에 영구 추가 안 함 (fold-local 변수)
```

**왜 fold-local 변수**: fold마다 spline 다시 fit하면 spline 출력값이 fold마다 다름. `xs_train_c` 자체에 spline 컬럼을 저장하면 어느 fold의 fit 결과를 영구 저장할지 모호 → 매 fold loop에서 ndarray로 직접 X_tr_s에 hstack 해서 사용.

**컬럼 추가**: SplineTransformer 기본 (n_knots=5, degree=3, include_bias=False) 시 입력 컬럼당 (n_knots + degree - 1) = 7 features → 20 × 7 = 140. include_bias / extrapolation 등 옵션에 따라 변동 가능하므로 `get_feature_names_out()` 사용 권장.

**예상 효과**: 트리와의 RMSE gap 일부 좁힘. 다만 약신호(|r|=0.037) 데이터에 컬럼 +140은 부담 → 효과 작을 수도 있음. **단계 1~4 모두 채택 후 마지막 옵션**.

---

### 단계 7 (옵션) — ColumnTransformer (dummy passthrough)

**근거**: 일반 scaler가 OHE dummy도 함께 변환 (`(0-m)/iqr`, `(1-m)/iqr` 형태). β 자동 비례 조정으로 RMSE 사실상 동일하지만 dummy 0/1 sparse 신호 의미가 약간 흐려짐. 엄밀히 0/1 강제하려면 ColumnTransformer로 분리.

**구현**: 양쪽 노트북 cell 10/12 fold loop에 ~10줄 추가.

```python
from sklearn.compose import ColumnTransformer

dummy_cols  = [c for c in feat_cols_clean
               if "_eq" in c
               or c.startswith(("pos_", "lot_"))
               or c.endswith("_missing")]   # cleaning.py 의 {col}_missing indicator 도 dummy
x_only_cols = [c for c in feat_cols_clean if c not in dummy_cols]

ct = ColumnTransformer([
    ("scale",    make_scaler(scaling_name), x_only_cols),
    ("passthru", "passthrough",             dummy_cols),
])
X_tr_s = ct.fit_transform(xs_train_c.loc[tr_mask, feat_cols_clean])
X_vl_s = ct.transform(xs_train_c.loc[vl_mask, feat_cols_clean])
```

**예상 효과**: 미세 (RMSE 차이 사실상 noise). **단계 1~4 채택 후 A/B 검증용**으로만.

---

## 3. 결과 기록표

각 단계 측정 후 채워넣기. **Δval ≥ 0 이면 그 단계 reject + 다음 단계 진행 안 함**.

> n_features 는 **추정값** (anchor PP 결과 + 단계 누적 가정). 실측은 측정 후 채움.
> reg_only / ts_reg 별도 기록 — **두 노트북 anchor 값이 다르면 RMSE 절대값 비교 불가**, Δval 만 비교.

### 3.1 reg_only (`02_reg_single/enet.ipynb`)

| 단계 | 추가 피처 | n_features (추정) | OOF RMSE | val RMSE | test RMSE | Δval | 채택? | 메모 |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 0  | baseline | ~572 | — | — | — | — | — | 측정 예정 |
| 1  | + pos OHE 4 | ~576 | — | — | — | — | — | |
| 2c | EXCLUDE에 X1056, X1072 | ~572 (변화 없음) | — | — | — | — | — | preprocess.py 수정 → 전 모델 영향 |
| 2a | + X1073 OHE 6 (원본 -1) | ~581 | — | — | — | — | — | sector 비선형 흡수 (필수) |
| 2b | + X1059/1075/1076/1077 OHE 32 (원본 -4) | ~609 | — | — | — | — | — | A/B로 ordinal vs OHE 비교 |
| ~~3~~  | ~~+ die_x, die_y~~ | — | — | — | — | — | ❌ 폐기 | 2026-05-09: ElasticNet die_x/die_y 제외 |
| ~~3'~~ | ~~+ die_x², die_y²~~ | — | — | — | — | — | ❌ 폐기 | 단계 3 폐기로 자동 폐기 |
| ~~4~~  | ~~+ lot OHE 28~~ | — | — | — | — | — | ❌ 폐기 | 2026-05-09: lot/wafer 전 모델 제외 |
| ~~5~~  | ~~+ per-lot centered X~~ | — | — | — | — | — | ❌ 폐기 | 단계 4 폐기로 자동 폐기 |
| 6  | + spline top 20 (~×7 dummy) | ~~~1344~~ ~715 | — | — | — | — | — | 옵션 (X 피처 한정) |
| 7  | ColumnTransformer | (변화 없음) | — | — | — | — | — | A/B 검증 |

### 3.2 ts_reg (`03_two_stage/default/reg/enet.ipynb`)

| 단계 | 추가 피처 | n_features (추정) | OOF RMSE | val RMSE | test RMSE | Δval | 채택? | 메모 |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 0  | baseline | ~572 | — | — | — | — | — | Y_POSITIVE_ONLY=True (Stage 2) |
| 1  | + pos OHE 4 | ~576 | — | — | — | — | — | |
| 2c | EXCLUDE에 X1056, X1072 | ~572 (변화 없음) | — | — | — | — | — | preprocess.py 공유 |
| 2a | + X1073 OHE 6 (원본 -1) | ~581 | — | — | — | — | — | |
| 2b | + X1059/1075/1076/1077 OHE 32 (원본 -4) | ~609 | — | — | — | — | — | |
| ~~3~~  | ~~+ die_x, die_y~~ | — | — | — | — | — | ❌ 폐기 | 2026-05-09: ElasticNet die_x/die_y 제외 |
| ~~3'~~ | ~~+ die_x², die_y²~~ | — | — | — | — | — | ❌ 폐기 | 단계 3 폐기로 자동 폐기 |
| ~~4~~  | ~~+ lot OHE 28~~ | — | — | — | — | — | ❌ 폐기 | 2026-05-09: lot/wafer 전 모델 제외 |
| ~~5~~  | ~~+ per-lot centered X~~ | — | — | — | — | — | ❌ 폐기 | 단계 4 폐기로 자동 폐기 |
| 6  | + spline top 20 (~×7 dummy) | ~~~1344~~ ~715 | — | — | — | — | — | 옵션 (X 피처 한정) |
| 7  | ColumnTransformer | (변화 없음) | — | — | — | — | — | A/B 검증 |

> 측정 후 본 표 채워서 PR/commit. 채택 단계가 결정되면 양쪽 노트북 cell 4의 `feat_cols_clean` 확장 코드를 영속 적용.

---

## 4. 진행 의사결정 흐름 (2026-05-09 축소)

```
[단계 0 baseline] 측정 (reg_only + ts_reg 각각)
       ↓
[단계 1 pos OHE] → val 개선? → No → 단계 1 reject (드물 것)
       ↓ Yes
[단계 2c EXCLUDE 추가] → preprocess.py 수정 (전체 영향)
       ↓
[단계 2a X1073 OHE] → val 개선? (필수 단계, 거의 확실)
       ↓ Yes
[단계 2b X1059/1075/1076/1077 OHE] → 개선? → No → ordinal 유지
       ↓ Yes/No
[단계 3 die_x/y / 3' poly / 4 lot OHE / 5 per-lot centered] ❌ 모두 폐기 (2026-05-09)
       ↓
[단계 6 spline top 20] (옵션, X 피처 한정) → 개선?
       ↓ Yes/No
[단계 7 ColumnTransformer] (옵션, 정밀화) → A/B
       ↓
[최종 채택 단계 영속 적용 — reg_only / ts_reg 동시]
```

> **양쪽 노트북 동기화 원칙**: 한 단계가 reg_only 에서 채택되면 ts_reg 에도 같은 단계 측정. 둘 중 하나만 reject 면 그 노트북에서만 reject (각 노트북의 anchor / Y_POSITIVE_ONLY 차이로 효과 다를 수 있음).

---

## 5. 헬퍼 모듈 사용 예시

[3_modeling/modules/meta_features.py](modules/meta_features.py) (신설 예정) 한 줄 호출 — 양쪽 노트북에 동일:

```python
from modules.meta_features import add_meta_features

feat_cols_clean, _ = add_meta_features(
    xs_train_c, xs_val_c, xs_test_c, feat_cols_clean,
    use_position=True,
    position_mode='ohe',           # enet
    use_die_xy=False,              # ★ 2026-05-09: ElasticNet die_x/die_y 제외
    use_loc_x_ohe=True,            # 단계 2a/2b 채택 시
    loc_x_required=("X1073",),
    loc_x_optional=("X1059", "X1075", "X1076", "X1077"),  # 단계 2b 채택 시
)
# lot_mode 인자 없음 — 2026-05-09 결정으로 lot/wafer 전 모델 제외
# die_xy_nonlinear 인자 없음 — die_x/die_y 자체가 제외라 무관
```

ColumnTransformer (단계 7) 분기는 별도. fold loop 안에서.

---

## 6. 참고

- 전체 가이드: [meta_features_strategy.md](meta_features_strategy.md)
- 이전자료 enet 결과: [4_output/final/reg_only/enet/best_params.json](../4_output/final/reg_only/enet/best_params.json) — n_features=568, position 미적용, OOF 0.005563
- 위치기반 X 분류 근거: [3_modeling_이전자료/two_stage/docs/strategy_2nd_ensemble.md](../3_modeling_이전자료/two_stage/docs/strategy_2nd_ensemble.md) "공간 패턴 feature 수동 선별 결과 (2026-04-16)"
