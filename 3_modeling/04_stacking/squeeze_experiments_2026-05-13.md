# 스태킹 Squeeze 실험 일지 (2026-05-13)

> 목적: 학원 풀 천장 val=0.005704 / test=0.008410 을 메타러너 단에서 얼마나 짜낼 수 있는지 검증.
> 결과: **val 0.005700 / test 0.008406 달성** — 학원 best ever 동등 (학원 best는 옛 old_bagzit_* 5종 포함 풀, 우리는 그게 없는 현재 풀 10-base만 사용).
> 내일: pp+hp 변형이 추가로 들어오면 본 일지의 최종 레시피를 그대로 적용.

---

## 0. 사용한 풀 (현재 10-base)

학원 `4_output/04_stacking/curated/config_comparison.csv` 의 `zit3 + reg4 + grid3` 구성:

```python
POOL = [
    "zit_only",                      # 4_output/01_zit/zit_only/001
    "bag_zit",                       # 4_output/01_zit/bag_zit/001
    "ts_reverse",                    # 4_output/03_two_stage/reverse/001
    "reg__lgbm",                     # 4_output/02_reg_single/lgbm/002
    "reg__xgb",                      # 4_output/02_reg_single/xgb/001
    "reg__catboost",                 # 4_output/02_reg_single/catboost/001
    "reg__et",                       # 4_output/02_reg_single/et/002
    "grid__lgbm_x_et",               # 4_output/03_two_stage/default/combined/lgbm_x_et
    "grid__xgb_x_et",                # 4_output/03_two_stage/default/combined/xgb_x_et
    "grid__catboost_x_et",           # 4_output/03_two_stage/default/combined/catboost_x_et
]
```

각 모델 디렉토리에 `oof_unit.csv`, `val_unit.csv`, `test_unit.csv` (column: `ufs_serial, pred, health`).

### baseline (단일 ElasticNetCV)
- val **0.005704**
- test **0.008410**

---

## 1. 각 실험 상세 — 무엇을 어떻게 했나

### 1.0 공통 환경
- Train: 26,187 unit / Val: 8,727 / Test: 8,729
- `KFold(n_splits=5, shuffle=True, random_state=42)`
- All experiments: oof_unit.csv / val_unit.csv / test_unit.csv 만 사용 (학원이 제공한 base predictions)
- 평가: `RMSE = sqrt(mean((pred - y)**2))`, `np.clip(pred, 0, None)` 후처리 일관 적용
- 스크립트 위치: `_runs/quantile_stacking/`

---

### 1.1 Baseline 단일 ElasticNetCV
**파일**: 모든 스크립트 공통 baseline
**가설**: 학원의 baseline 재현 (val 0.005704, test 0.008410)
**구현**:
```python
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("enet", ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        alphas=np.logspace(-6, 0, 30),
        cv=KFold(5, shuffle=True, random_state=42),
        positive=False, n_jobs=8, max_iter=20000,
    ))
])
pipe.fit(X_oof, y_oof)  # X_oof: (26187, 10), y_oof: (26187,)
pred = np.clip(pipe.predict(X), 0, None)
```
**결과**: val=0.005704, test=0.008410. 학원 결과와 정확히 일치 (재현 성공).
**메모**: 28-base 전체 풀로 돌려도 동일 결과 (0.005704) — base 추가의 marginal 효과 없음.

---

### 1.2 Isotonic calibration ⭐
**파일**: `squeeze_all.py` step [5]
**가설**: ZI 데이터의 작은 양수 false-positive를 monotonic 변환으로 0 쪽으로 끌어내림 (zero-clip의 부드러운 버전).
**구현**:
```python
from sklearn.isotonic import IsotonicRegression
# 1. baseline ENet 으로 stack_oof, stack_val, stack_test 생성
# 2. iso fit on OOF
iso = IsotonicRegression(out_of_bounds="clip", y_min=0)
iso.fit(stack_oof, y_oof)  # 1D → 1D
final_val  = iso.transform(stack_val)
final_test = iso.transform(stack_test)
```
**결과**: val=0.005704→**0.005701**, test=0.008410→**0.008406**.
**해석**: Δval=-0.000003, Δtest=-0.000004. **5분 작업으로 가장 큰 ROI**. 학원이 안 시도한 듯.
**왜 작동**: ZI 분포에서 ENet 이 평균을 추정하다 보니 Y=0 unit 에 0.0005~0.002 같은 작은 양수 출력. iso 가 OOF 분포 보고 "출력 0.001 이하는 다 0으로" 같은 step function 학습.

---

### 1.3 Bagged ElasticNet (5 CV seeds)
**파일**: `squeeze_all.py` step [2], `squeeze_v2.py` META 1
**가설**: 5 seed 평균으로 ElasticNet 의 KFold 분산 줄임.
**구현**: 같은 ENet 파라미터 (l1_ratio grid, alphas log space) but `KFold(seed=s)` 와 `ElasticNetCV(random_state=s)` 둘 다 변주. seed=[42, 123, 456, 789, 2024]. 5 모델 예측 평균.
**결과**: val=0.005704 (baseline 과 동일), test=0.008410. **단독으론 효과 없음**.
**해석**: ENet 자체가 variance 가 작은 모델 (closed-form solution 비슷). seed 흔들기로 얻는 분산 감소 ≈ 0. 단, combo 의 한 구성원으로는 유용 (다른 메타와 평균 시 안정화).

---

### 1.4 Greedy NNLS + L-BFGS-B
**파일**: `squeeze_all.py` step [6]
**가설**: ENet 의 L1+L2 정규화 없이 **순수 RMSE 최소화** 가중치.
**구현**:
```python
from scipy.optimize import nnls, minimize
# Step 1: NNLS 초기화 (closed-form non-negative least squares)
w0, _ = nnls(X_oof, y_oof, maxiter=50000)
# Step 2: L-BFGS-B 로 refine
res = minimize(
    lambda w: float(np.mean((X_oof @ w - y_oof) ** 2)),
    w0, method="L-BFGS-B",
    bounds=[(0, None)] * 10,
    options={"maxiter": 500, "ftol": 1e-12},
)
w = res.x  # 10-dim non-negative weights
pred = np.clip(X @ w, 0, None)
```
**결과**: val=0.005703 (Δ=-0.000001 미미), test=0.008409, sum_w=1.0161.
**해석**: NNLS 가 9개 base 에 양수 가중치 분산. ENet 의 음수 corrector 효과 없어서 val 살짝 떨어졌지만 test 는 더 안정.
**메모**: 가중치 합이 1.02 → "convex combination" 거의 만족. SLSQP sum=1 강제하면 val=0.005704 로 살짝 악화.

---

### 1.5 Combo (3 meta 평균) + Isotonic ⭐ — **현재 best**
**파일**: `squeeze_all.py` step [8]
**가설**: 3가지 메타 (bagged ENet + plain ENet + greedy NNLS) 평균으로 분산 줄이고 iso 로 calibrate.
**구현**:
```python
# 위 1.2, 1.3, 1.4 의 OOF/val/test 예측 사용
combo_oof  = (oof_bag  + oof_en  + oof_gr)  / 3.0
combo_val  = (val_bag  + val_en  + val_gr)  / 3.0
combo_test = (test_bag + test_en + test_gr) / 3.0
# Isotonic
iso = IsotonicRegression(out_of_bounds="clip", y_min=0)
iso.fit(combo_oof, y_oof)
final_val  = iso.transform(combo_val)
final_test = iso.transform(combo_test)
```
**결과**: val=**0.005700**, test=**0.008406**. 학원 best ever 동등.
**Δ vs baseline**: val -0.000004, test -0.000004.
**해석**: 단순 iso 단독 (0.005701) 보다 한 자리 더. 3 메타 평균이 분산 줄이고, iso 가 ZI false positive 잡음. 가중치 SLSQP 튜닝해봐도 균등 1/3 이 optimal (1.5 참조).

---

### 1.6 Combo 가중치 SLSQP / L-BFGS-B 튜닝
**파일**: `squeeze_v2.py` Task 1
**가설**: 균등 1/3 대신 OOF MSE 최소화 가중치가 더 좋을 수 있음.
**구현**:
```python
M_oof = np.column_stack([oof_bag, oof_en, oof_gr])  # (26187, 3)
def combo_loss(w):
    return float(np.mean((M_oof @ w - y_oof) ** 2))

# SLSQP sum=1
res1 = minimize(combo_loss, [1/3]*3, method="SLSQP",
                bounds=[(0,1)]*3,
                constraints=[{"type":"eq", "fun":lambda w: w.sum()-1}])
# L-BFGS-B nonneg (sum 자유)
res2 = minimize(combo_loss, [1/3]*3, method="L-BFGS-B",
                bounds=[(0,None)]*3)
```
**결과**: 둘 다 `w=[0.333, 0.333, 0.333]` 으로 수렴. val 변화 없음.
**해석**: 3 메타가 균등하게 유익. 즉 메타 간 잔차 패턴이 비슷해서 가중치 차별화로 짜낼 게 없음.

---

### 1.7 RandomForest meta (논문 9-2)
**파일**: `squeeze_v2.py` Task 2
**가설**: RF 의 bootstrap + random feature 가 base 간 correlated error 에 강건 + 비선형 상호작용 포착.
**구현**: 5 config grid:
```python
rf_configs = [
    {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 5},
    {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 20},
    {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 50},
    {"n_estimators": 500, "max_depth": 6,    "min_samples_leaf": 50},
    {"n_estimators": 1000,"max_depth": 8,    "min_samples_leaf": 30},
]
# 5-fold OOF 학습 후 full refit → val/test 예측
```
**결과**: 모든 config 가 baseline 보다 나쁨. best (depth=6, leaf=50): val=0.005705, test=0.008409. iso 추가도 동일.
**해석**: RF 평균화로 over-smooth → ZI 극단값 손실. min_samples_leaf 큼 (50) 이 best 인 게 그 증거 — leaf 가 작으면 overfit, 크면 under-fit. 우리 데이터에 RF meta 안 맞음.

---

### 1.8 4-meta combo (ENet 3개 + RF)
**파일**: `squeeze_v2.py` Task 3
**가설**: 비선형 메타(RF) 1개를 ENet 3개에 추가하면 다양성.
**구현**: SLSQP sum=1 가중치 4개.
**결과**: 균등 0.25 4개로 수렴. val=0.005701, test=0.008406 (현재 best 0.005700 보다 살짝 위).
**해석**: RF 가 평균화 효과로 combo 약화. ENet 3개만 쓰는 게 best.

---

### 1.9 Multi-layer skip connection (논문 9-1 AutoGluon)
**파일**: `multilayer_skip.py`
**가설**: raw X 피처와 base predictions 를 둘 다 메타 입력으로 → base 가 못 잡은 패턴 추가 학습.
**구현**:
```python
# PP 돌려서 raw X 얻기 (PP_FIXED, 573 features)
pp = preprocess.run(xs, ys_input, feat_cols, xs_dict, params=PP_FIXED)
# die → unit 집계 (mean)
X_raw_unit = xs_train[[KEY_COL] + feat_clean].groupby(KEY_COL).mean()
# concat
BR_oof  = np.hstack([B_oof,  X_oof_raw])   # (26187, 10 + 573 = 583)
BR_val  = np.hstack([B_val,  X_val_raw])
BR_test = np.hstack([B_test, X_test_raw])
# Sparse ENet 으로 selection
ElasticNetCV(l1_ratio=[0.7, 0.9, 0.95, 1.0], alphas=np.logspace(-6, 0, 30))
```
4가지 시나리오:
- A: base only (10) — baseline
- B: base + derived stats (mean/std/min/max/range/median of 10 base) (16)
- C: base + raw X (583)
- D: base + derived + raw X (589)

**결과**:
- A: val=0.005704, A+iso: 0.005701
- B: 0.005704, B+iso: 0.005701
- C: 0.005705, C+iso: 0.005701
- D: 0.005705, D+iso: 0.005701

**해석**: ElasticNet 이 sparse selection 으로 base 10개만 active. raw X 피처는 전부 0 가중치. **이유**: base 모델들이 이미 raw X 를 학습에 사용했기 때문에, raw X 가 base predictions 와 redundant. ElasticNet 이 redundancy 인식하고 raw X 무시.

---

### 1.10 Forward selection + 잔차 다양성 패널티
**파일**: `squeeze_all.py` step [1]
**가설**: best-by-CV 가 아니라 잔차 상관 낮은 base 선호하는 selection.
**구현**:
```python
# greedy 추가: 각 step 에서
score(candidate) = -cv_rmse(current + candidate) - λ × max_corr(candidate, current)
# λ = [0.0, 1e-5, 5e-5]
```
28-base 풀에서 selection.
**결과**: 세 λ 값 모두 동일하게 val=0.005704, test=0.008411. 선택된 base 도 거의 유사 (9~11개).
**해석**: 학원 forward_selection 결과와 동일. 풀이 이미 정제된 상태라 selection 만으로 짜낼 게 없음. λ 효과 미미 (28-base 잔차 상관 너무 균일).

---

### 1.11 Stage 1 / Stage 2 분리 메타
**파일**: `stage12_meta.py`
**가설**: 단일 메타가 전체 RMSE 최적화 → Y>0 30% 의 다양성 못 짜냄. Stage 1 분류 + Stage 2 회귀 (Y>0만 학습) 분리하면 Y>0 에서 메타가 진짜 일함.
**구현**:
```python
# Stage 1: LGBM 분류 (Y>0 vs Y=0)
clf = LGBMClassifier(objective="binary", n_estimators=300, num_leaves=15, max_depth=4, ...)
# 5-fold OOF → stage1_oof, stage1_val, stage1_test (P(Y>0))

# Stage 2: ENet 회귀 (Y>0 unit 만 학습)
pos_mask = z_oof == 1   # 7,646개
meta = ElasticNetCV(positive=True, ...)
meta.fit(X_oof[pos_mask], y_oof[pos_mask])
# predict on full val/test (Y=0 unit 에도 양수 출력)

# 결합
final = clip(stage1 * stage2, 0, None)
```
**결과**: val=0.005713 (Δ=+0.000009 악화), test=0.008409 (-0.000001 살짝 개선).
**Segment 분석**:
- Y=0 segment val: 0.002562 → 0.002616 (악화 +0.000054)
- Y>0 segment val: 0.009771 → 0.009754 (개선 -0.000017)

**해석**: Stage 2 가 Y>0 에서 진짜 일하긴 함 (-0.000017). 하지만 Stage 1 분류 AUC 0.67 약함 → Y=0 unit 에 P(Y>0)=0.2~0.3 출력 → Stage 2 의 양수 × 0.2 = 0.001~0.003 양수 잔차. Y=0 unit 18,541개 × 잔차^2 합 > Y>0 7,646개 개선분.

---

### 1.12 Stage 1 / Stage 2 + Threshold cutoff
**파일**: `stage12_threshold.py`
**가설**: P(Y>0) < τ 이면 hard cutoff → Y=0 leak 막음.
**구현**:
```python
def combine(s1, s2, tau):
    pred = s1 * s2
    pred = np.where(s1 < tau, 0.0, pred)
    return np.clip(pred, 0, None)
# OOF 에서 best τ grid search
taus = np.arange(0.0, 1.01, 0.025)
```
**결과**: **best τ = 0.0** (cutoff 없는 게 best). τ ≥ 0.3 부터 급격히 악화 (Y>0 unit 까지 0 처리됨).
**해석**: Stage 1 AUC 0.67 너무 약해서 τ 어떻게 잡아도 Y>0 손실 > Y=0 개선. Stage 1 자체를 강화해야 답.

---

### 1.13 LGBM Tweedie meta (shallow)
**파일**: `squeeze_all.py` step [4]
**가설**: ZI 데이터에 적합한 Tweedie loss + 작은 LGBM 으로 비선형 메타.
**구현**:
```python
params = dict(
    objective="tweedie", tweedie_variance_power=1.3,
    n_estimators=200, learning_rate=0.03, num_leaves=7, max_depth=3,
    min_child_samples=300, ...
)
# 5-fold OOF
```
**결과**: val=0.005707 (Δ=+0.000003 악화), test=0.008409.
**해석**: 메타 LGBM 이 base predictions(상관 0.99+) 에서 과적합. depth=3 으로 매우 작게 했음에도 train 패턴에 fit 하다 val 일반화 실패.

---

### 1.14 Y-bucket 메타 (3 bucket soft mix)
**파일**: `squeeze_all.py` step [3]
**가설**: Y 구간별 별도 메타 학습 (Stage 1/2 의 일반화).
**구현**:
```python
cuts = [0, 0.0025, 0.010, ∞]   # 3 bucket
# train: true Y 로 bucket assignment → bucket 별 ENet 학습
# val/test: baseline pred 로 bucket assignment (true Y 모름)
```
**결과**: val=0.006072 (Δ=+0.000368 큰 악화), test=0.008637.
**해석**: val/test 에서 bucket assignment 가 baseline pred 기준이라 부정확. Y=0 unit 의 baseline pred 가 0.001~0.005 사이라 bucket 0 (0~0.0025) 와 bucket 1 (0.0025~0.010) 사이에 잘못 분류. **fundamentally flawed approach** for unknown Y.

---

### 1.15 XStacking lite (poly2 + rank + qt)
**파일**: `poly_xstacking_lite.py`
**가설**: 정식 XStacking 은 SHAP 필요 (모델 pkl 없어 비현실적). 대안으로 base predictions 의 비선형 표현:
- **poly2**: degree-2 interaction (10 base → 45 pairwise products = 55 features)
- **rank**: per-unit rank across 10 base (10 features)
- **qt**: QuantileTransformer(output="normal") 변환 (10 features)

**구현**:
```python
from sklearn.preprocessing import PolynomialFeatures, QuantileTransformer
from scipy.stats import rankdata

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
PO_oof = poly.fit_transform(B_oof)  # (26187, 45)

R_oof = np.apply_along_axis(rankdata, 1, B_oof) / 10  # per-unit rank
Q_oof = QuantileTransformer(n_quantiles=1000, output_distribution="normal").fit_transform(B_oof)
```
5가지 시나리오 (A: base, B: base+poly2, C: base+rank, D: base+qt, E: all 75).
**결과**:
- B (base + poly2): val=0.005702, B+iso: **0.005700** (val 동등이지만 test 0.008408)
- D (base + qt): val=0.005701, D+iso: 0.005702
- E (all): val=0.005703, E+iso: 0.005705 (과적합)

**해석**: poly2 가 val 에서는 base 의 interaction 패턴 잡음. iso 추가하면 우리 best combo (val 0.005700, test 0.008406) 와 val 동등. 단 test 가 0.008408 로 살짝 위 → **val 과적합 위험**. combo (1.5) 가 여전히 test 안정성에서 우위.

---

## 2. 시도한 모든 실험 (요약 표)

총 10가지 기법 × 30+ variant. val 오름차순 정렬 (test 함께 표시).

| 순위 | 기법 | val | test | 비고 |
|---:|---|---|---|---|
| **★1** | **combo: bagged+greedy+enet → iso** | **0.005700** | **0.008406** | **현재 best** |
| ★1 | XStacking lite: base+poly2 → iso | 0.005700 | 0.008408 | val 동등, test 살짝 위 |
| 3 | combo: bagged + iso | 0.005701 | 0.008406 | |
| 3 | isotonic on baseline (10-base ENet) | 0.005701 | 0.008406 | **iso 단독으로도 -0.000003** |
| 3 | combo ABD (base + poly2 + qt) + iso | 0.005701 | 0.008407 | |
| 3 | Stage 1/2 + threshold 모든 시나리오 | 0.005701~0.005713 | 0.008408~0.008409 | Stage 1 AUC 0.67 약함 |
| 7 | Multi-layer skip (raw X 583dim) + iso | 0.005701 | 0.008409 | ENet 이 raw X 무시 |
| 7 | base + quantile transform + iso | 0.005702 | 0.008408 | |
| 9 | base + poly2 (iso 없음) | 0.005702 | 0.008408 | |
| 10 | multilevel 3-meta avg | 0.005703 | 0.008408 | |
| 10 | greedy weight (NNLS+LBFGS) | 0.005703 | 0.008409 | |
| 12 | baseline ENet (10-base / 28-base 동일) | 0.005704 | 0.008410 | 학원 single ENet |
| 12 | bagged 5-seed ENet alone | 0.005704 | 0.008410 | variance 이미 작음 |
| 12 | Forward selection (λ=0~5e-5) | 0.005704 | 0.008411 | 학원 FS와 중복 |
| 15 | RF meta best (depth=6, leaf=50) | 0.005705 | 0.008409 | 모든 RF config 가 baseline 이상으로 안 좋음 |
| 16 | LGBM Tweedie meta (shallow) | 0.005707 | 0.008409 | 메타 과적합 |
| 17 | Stage 1/Stage 2 분리 메타 (no threshold) | 0.005713 | 0.008409 | Y=0 segment +0.000054 손실 |
| 18 | Y-bucket meta (3 bucket soft mix) | 0.006072 | 0.008637 | bucket 분류 부정확 |

---

## 3. 핵심 발견 — 작동한 것

### 3.1 Isotonic calibration ⭐ (가장 큰 ROI)

**단순 적용만으로 val 0.005704 → 0.005701 (-0.000003), test 0.008410 → 0.008406 (-0.000004)**.

```python
from sklearn.isotonic import IsotonicRegression
iso = IsotonicRegression(out_of_bounds="clip", y_min=0)
iso.fit(stack_oof, y_oof)
final_val  = iso.transform(stack_val)
final_test = iso.transform(stack_test)
```

작동 이유: ZI 데이터의 작은 양수 예측(false positive)을 monotonic하게 0으로 끌어내림. `zero_clip` 의 부드러운 버전. 학원이 안 시도.

### 3.2 Combo (3 meta 평균) → Isotonic

**Isotonic 단독 + variance reduction 으로 한 자리 더**.

3 메타: bagged ENet (5 seed) + plain ENet + greedy NNLS+LBFGS. 균등 1/3 평균 (SLSQP 튜닝해도 균등이 optimal).

---

## 4. 최종 best 레시피 (재현 가능)

```python
# ─── pool & data load ──────────────────────────────
import os, numpy as np, pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import nnls, minimize

PROJECT_ROOT = r"c:\Users\COM\Desktop\기업연계프로젝트"
KEY_COL = "ufs_serial"
SEED = 42

POOL_DIRS = {
    # 기존 10-base
    "zit_only":   "4_output/01_zit/zit_only/001",
    "bag_zit":    "4_output/01_zit/bag_zit/001",
    "ts_reverse": "4_output/03_two_stage/reverse/001",
    "reg__lgbm":     "4_output/02_reg_single/lgbm/002",
    "reg__xgb":      "4_output/02_reg_single/xgb/001",
    "reg__catboost": "4_output/02_reg_single/catboost/001",
    "reg__et":       "4_output/02_reg_single/et/002",
    "grid__lgbm_x_et":     "4_output/03_two_stage/default/combined/lgbm_x_et",
    "grid__xgb_x_et":      "4_output/03_two_stage/default/combined/xgb_x_et",
    "grid__catboost_x_et": "4_output/03_two_stage/default/combined/catboost_x_et",
    # ★ 내일: pp+hp 변형 추가
    # "pp_hp__bag_zit_v1": "4_output/.../pp_hp_bag_zit_v1",
    # ...
}
POOL = list(POOL_DIRS.keys())

def load_split(d, s):
    return pd.read_csv(os.path.join(PROJECT_ROOT, d, f"{s}_unit.csv"))

def build_wide(pool, split):
    first = pool[0]
    df = load_split(POOL_DIRS[first], split)[[KEY_COL, "health", "pred"]].rename(columns={"pred": first})
    for n in pool[1:]:
        sub = load_split(POOL_DIRS[n], split)[[KEY_COL, "pred"]].rename(columns={"pred": n})
        df = df.merge(sub, on=KEY_COL, how="inner")
    return df

def rmse(p, y): return float(np.sqrt(np.mean((p-y)**2)))

P_oof  = build_wide(POOL, "oof")
P_val  = build_wide(POOL, "val")
P_test = build_wide(POOL, "test")
X_oof, X_val, X_test = P_oof[POOL].values, P_val[POOL].values, P_test[POOL].values
y_oof, y_val, y_test = P_oof["health"].values, P_val["health"].values, P_test["health"].values

# ─── 3 메타 학습 ───────────────────────────────────
# META 1: Bagged ENet (5 seed 평균)
oof_bag = np.zeros_like(y_oof); val_bag = np.zeros_like(y_val); test_bag = np.zeros_like(y_test)
for s in [42, 123, 456, 789, 2024]:
    pipe = Pipeline([("sc", StandardScaler()),
                     ("en", ElasticNetCV(l1_ratio=[0.1,0.3,0.5,0.7,0.9,1.0],
                                         alphas=np.logspace(-6, 0, 30),
                                         cv=KFold(5, shuffle=True, random_state=s),
                                         n_jobs=8, max_iter=20000, random_state=s))])
    pipe.fit(X_oof, y_oof)
    oof_bag  += np.clip(pipe.predict(X_oof),  0, None) / 5
    val_bag  += np.clip(pipe.predict(X_val),  0, None) / 5
    test_bag += np.clip(pipe.predict(X_test), 0, None) / 5

# META 2: Plain ENet
pipe2 = Pipeline([("sc", StandardScaler()),
                  ("en", ElasticNetCV(l1_ratio=[0.1,0.3,0.5,0.7,0.9,1.0],
                                      alphas=np.logspace(-6, 0, 30),
                                      cv=KFold(5, shuffle=True, random_state=SEED),
                                      n_jobs=8, max_iter=20000, random_state=SEED))])
pipe2.fit(X_oof, y_oof)
oof_en  = np.clip(pipe2.predict(X_oof),  0, None)
val_en  = np.clip(pipe2.predict(X_val),  0, None)
test_en = np.clip(pipe2.predict(X_test), 0, None)

# META 3: Greedy NNLS+LBFGS
w0, _ = nnls(X_oof, y_oof, maxiter=50000)
res = minimize(lambda w: float(np.mean((X_oof @ w - y_oof)**2)),
               w0, method="L-BFGS-B",
               bounds=[(0, None)] * X_oof.shape[1],
               options={"maxiter": 500, "ftol": 1e-12})
w = res.x
oof_gr  = np.clip(X_oof  @ w, 0, None)
val_gr  = np.clip(X_val  @ w, 0, None)
test_gr = np.clip(X_test @ w, 0, None)

# ─── Combo 평균 + Isotonic ────────────────────────
combo_oof  = (oof_bag  + oof_en  + oof_gr)  / 3.0
combo_val  = (val_bag  + val_en  + val_gr)  / 3.0
combo_test = (test_bag + test_en + test_gr) / 3.0

iso = IsotonicRegression(out_of_bounds="clip", y_min=0)
iso.fit(combo_oof, y_oof)

final_oof  = iso.transform(combo_oof)
final_val  = iso.transform(combo_val)
final_test = iso.transform(combo_test)

print(f"oof  RMSE = {rmse(final_oof,  y_oof):.6f}")
print(f"val  RMSE = {rmse(final_val,  y_val):.6f}")
print(f"test RMSE = {rmse(final_test, y_test):.6f}")
# 기대: oof=0.005486, val=0.005700, test=0.008406
```

---

## 5. 작동 안 한 것 — 함정 모음

| 기법 | 왜 안 됐나 |
|---|---|
| RF meta | bootstrap 평균화로 극단값 손실. ZI 데이터에서 over-smooth |
| LGBM Tweedie meta | 메타가 base predictions(상관 0.99+) 에서 과적합 |
| Multi-layer skip (raw X 583dim) | ElasticNet 이 base 10개만 active 선택. raw X 정보 base 학습에 이미 사용됨 |
| Y-bucket meta | val/test에 true Y 모름 → baseline pred 로 bucket assignment 부정확. Y=0 bucket 분리 실패 |
| Stage 1/2 분리 메타 | Stage 1 분류 AUC 0.67 약함. Y=0 unit 에 P(Y>0)=0.2~0.3 → Stage 2 양수 출력 × P → false positive |
| Stage 1/2 + threshold | OOF 기준 best τ=0.0 (cutoff 없는 게 best). τ↑ 시 Y>0 unit 도 0 처리 → 큰 손실 |
| Forward selection diversity penalty | 학원 FS와 결과 동일. 풀이 이미 정제된 상태 |
| Bagged 5-seed alone | ENet variance 가 이미 작아서 평균 효과 미미 |
| Combo 가중치 SLSQP 튜닝 | 균등 1/3 이 optimal — 3 메타가 균등하게 유익 |
| Forward selection (λ=0~5e-5) | 풀이 작아 selection 효과 marginal |

---

## 6. 내일 — pp+hp 변형 들어왔을 때 실행 절차

### Step 1. 새 변형 OOF 디렉토리 확인 (§4 best 레시피 참조)
```bash
ls 4_output/[pp+hp 변형 경로]/
# 기대: oof_unit.csv, val_unit.csv, test_unit.csv 3개 (column: ufs_serial, pred, health)
```

### Step 2. `POOL_DIRS` 에 새 변형 추가
위 §4 코드의 `POOL_DIRS` dict 에 항목 추가. 이름 prefix 통일 (예: `pphp__bag_zit_v1`).

### Step 3. 잔차 상관 매트릭스 확인 (새 변형 진짜 다양성 추가?)
```python
res_oof = pd.DataFrame({k: P_oof[k].values - y_oof for k in POOL})
C = res_oof.corr()
print("새 변형과 기존 base 잔차 상관 (낮을수록 다양함):")
for new in NEW_VARIANTS:
    pairs = sorted([(b, float(C.loc[new, b])) for b in OLD_POOL], key=lambda x: x[1])
    print(f"  [{new}]")
    for b, c in pairs[:5]:
        print(f"    {b}: r = {c:.4f}")

# 학원 old_bagzit_* 5종은 서로 r > 0.997 인데도 ensemble 이득 있었음.
# 새 변형이 0.997 미만 페어 만들면 ensemble 효과 기대.
```

### Step 4. §4 best 레시피 그대로 실행

POOL 확장된 상태에서 동일 코드 → 새 final_val, final_test 산출.

### Step 5. 결과 비교
```python
# 기존 best (only 10-base) vs 확장 풀
print(f"기존 풀 best: val=0.005700, test=0.008406")
print(f"확장 풀 result: val={rmse(final_val,y_val):.6f}, test={rmse(final_test,y_test):.6f}")
```

기대치 (학원 데이터 근거):
- 5개 변형 추가 → val **0.005696~0.005698** 영역
- 10개 변형 추가 → val **0.005695** 영역 가능성

### Step 6. (선택) Forward selection 다시
풀이 커지면 일부 변형이 redundant 일 수 있음. forward selection 으로 best subset 찾기:

```python
# best subset 선택 — squeeze_all.py 의 fs_with_diversity() 참조
```

---

## 7. 추가로 고려할 점

### 7.1 새 변형 학습 가이드

기존 `3_modeling_이전자료/final/_temp_bagzit/` 5종 패턴 참고:

| 변형 | 핵심 |
|---|---|
| HP only HPO | `PP_FIXED` 고정 + Optuna HP 30 trial |
| PP only HPO | `HP_FIXED` 고정 + Optuna PP 8축 30 trial |
| HP+PP joint | search space 통합 + 30~50 trial |
| + FE 변주 | xy, GroupTargetEncoder 추가 |

각 변형마다 5-fold OOF → `oof_unit.csv`, `val_unit.csv`, `test_unit.csv` 저장.

### 7.2 base 모델별 적합성

| 모델 | pp+hp 변주 ROI |
|---|---|
| bag_zit (ZIT 계열) | ★★★ 학원 5종 검증, 가장 추천 |
| zit_only | ★★ |
| ts_reverse | ★★ |
| reg__lgbm | ★ (단일 RMSE 0.005731 약하지만 잔차 패턴 다름) |
| reg__xgb, catboost, et | ★ |
| grid__*_x_et | ☆ (이미 25개 grid 존재) |

→ **bag_zit 5~10개 변형이 최고 ROI**. 그 다음 zit_only / ts_reverse 각 2~3개.

### 7.3 학습 비용 추정

각 변형 4~6시간 (HPO 30 trial 기준). 14코어 환경에서 2개 병렬 → 하루 4~6개 가능.

5개 변형 = 1일 / 10개 = 2일.

---

## 8. 핵심 깨달음 (한 줄 요약)

1. **Isotonic calibration 이 ZI 데이터의 비밀 무기** (학원 안 시도)
2. **메타 다양화(RF, LGBM, Tweedie)는 잔차 상관 0.99+ 풀에서 마이너스**
3. **변형 추가가 메타 변경보다 ROI 큼** — 학원 5종 → -0.000003
4. **val/test gap 안정성은 positive=True 메타가 살짝 유리**
5. **풀이 정제되어 있을 때 forward selection 은 marginal**
6. **0.000001~0.000002 차이는 노이즈 (SE ≈ 0.000043)** — combo 가 best 인 건 통계적으로 robust 아님

---

## 부록. 관련 파일

```
3_modeling/
├── stacking_strategy.md           # 전체 전략 + 논문 인사이트
├── squeeze_experiments_2026-05-13.md  # 본 문서 (실험 일지)
└── ...

_runs/quantile_stacking/
├── squeeze_all.py                 # 7기법 (FS, bagged, iso, greedy 등)
├── squeeze_v2.py                  # combo weight + RF meta
├── multilayer_skip.py             # raw X + base preds
├── poly_xstacking_lite.py         # poly2 / rank / qt
├── stage12_meta.py                # Stage 1/2 분리
├── stage12_threshold.py           # Stage 1/2 + threshold
├── train_q70.py                   # LGBM Quantile τ=0.7 학습
└── *_results.json                 # 결과 JSON
```

## 부록 2. 논문 참조 (스태킹 관련, 논문요약.md)

| 논문 | 본 실험 반영 | 결과 |
|---|---|---|
| 9-1 AutoGluon (multi-layer + bagging) | Multi-layer skip 시도 | 효과 없음 (ElasticNet 이 raw X 무시) |
| 9-2 RF meta | RF meta 5종 grid | 효과 없음 (over-smooth) |
| 9-3 GEM-ITH (HP+가중치 joint) | Combo SLSQP weight 시도 | 균등 1/3 이 optimal |
| 9-4 XStacking (SHAP 기반) | poly2 / qt / rank 변형으로 lite 시도 | val 동등, test 살짝 위 |

정식 XStacking 미시도 — base 모델 pkl 없음 (재학습 시 SHAP 계산 추가 가능).
