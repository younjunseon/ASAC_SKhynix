# 2차 Funnel — 전처리 모듈 설계

> **작성일**: 2026-04-16
> **대상**: 2차 ensemble funnel에서 신규/확장할 전처리 모듈
> **관련 문서**: [strategy_2nd_ensemble.md](strategy_2nd_ensemble.md) (앙상블 모델링 코드)
> **베이스**: [CLAUDE.md](../CLAUDE.md) Stage 1~4 + 2차 funnel 계획 (L515~564)

---

## 1. Overview

1차 funnel은 LGBM 단일 모델 + `cleaning` / `outlier` / `agg_preset` 3축만 Optuna 탐색. 2차는 **앙상블 베이스코드 + 모델 다양성**(LGBM·ExtraTrees·ElasticNet)을 위해 전처리 축을 확장한다.

### 사용자 결정 (2026-04-16 대화 확정)

| 항목 | 결정 | 근거 |
|---|---|---|
| Scaling 방식 | **하이브리드** (skew>5 → quantile, 나머지 → power, binary는 passthrough) | test.ipynb 4종 비교: `skew_threshold=5` 정규근사율 68%로 최고 |
| Scaling 탐색 축 | **없음** (옵튜나 밖 확정, `skew_threshold=5.0` 고정) | 사용자 명시 "가장 좋은 걸로 1회만" |
| Binarize 스텝 (신규) | **cleaning 직후 고정 적용** (top%>0.95 OR nunique≤5 → 0/1) | Degenerate feature winsorize·scaling 무의미 회피, 선형 모델 수렴 개선 |
| IsoForest score | 컬럼 추가 (제거 X), 옵튜나 on/off | 단변량 clip의 한계 보완 |
| LDS re-weighting | 신규 모듈, 옵튜나 sigma 탐색 | Stage 2 회귀 Y>0 long-tail 대응 |
| 메타 피처 (die_x/die_y, lot stats) | **제외** | EDA Phase 23/24에서 무상관 확인, raw X에 공간 패턴 이미 존재 |
| Target encoding | **제외** | fold마다 재계산 + 누수 위험 |
| Separated scaling per lot | **제외** | "스케일링 1회" 정책 충돌 |

---

## 2. 신규/확장 대상 일람

| 파일 | 변경 | 함수 | 상태 |
|---|---|---|---|
| [2_preprocessing/scaling.py](../2_preprocessing/scaling.py) | 클래스·함수 추가 | `HybridScaler` (binary passthrough 포함), `hybrid_scale()` | ✅ 구현 완료 |
| [2_preprocessing/cleaning.py](../2_preprocessing/cleaning.py) | 함수 추가 | `binarize_degenerate()` (cleaning 직후 고정 호출) | ✅ 구현 완료 |
| [2_preprocessing/outlier.py](../2_preprocessing/outlier.py) | 함수 추가 | `multivariate_anomaly_score()` | ✅ 구현 완료 |
| `2_preprocessing/sample_weight.py` | **신규 파일** | `compute_lds_weights()` | ✅ 구현 완료 |
| [3_modeling/modules/search_space.py](modules/search_space.py) | 상수·함수 확장 | `PP_ISO_ANOMALY_CANDIDATES`, `PP_LDS_CANDIDATES`, `preprocessing_space()`, `split_pp_params()` | ⏳ 미반영 |
| [3_modeling/modules/e2e_hpo.py](modules/e2e_hpo.py) | `_run_preprocessing` 확장 | binarize + IsoForest + hybrid_scale + LDS 통합, 4-tuple 반환 | ⏳ 미반영 |

---

## 3. scaling.py 확장 (방식 B — sklearn fit/transform 패턴)

### 3.0 설계 원칙 (사용자 결정 2026-04-16)

- **sklearn 표준 fit/transform 패턴** 채택 — 모델과 동일한 인터페이스
- **train fit → val/test transform** 명시적 분리 (누수 방지 + 재사용성)
- 향후 대시보드 / AI Agent (CLAUDE.md Phase 4)에서 같은 scaler 재사용 가능

### 3.1 `HybridScaler` 클래스 (신규)

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, QuantileTransformer


class HybridScaler:
    """
    Skew 기반 하이브리드 스케일러

    |skew| > threshold → QuantileTransformer (rank 기반, heavy-tail 평탄화)
    |skew| ≤ threshold → PowerTransformer (Yeo-Johnson + standardize)

    sklearn 표준 fit/transform 패턴 — 객체 보관 후 predict 파이프라인 재사용 가능.
    """

    def __init__(self, skew_threshold=10.0, n_quantiles=1000,
                 quantile_output='normal', random_state=42):
        self.skew_threshold = skew_threshold
        self.n_quantiles = n_quantiles
        self.quantile_output = quantile_output
        self.random_state = random_state

    def fit(self, X, feat_cols=None):
        """
        Parameters
        ----------
        X : DataFrame (train only)
        feat_cols : list, optional
            스케일링 대상. None이면 X의 모든 컬럼

        Returns
        -------
        self
        """
        if feat_cols is None:
            feat_cols = list(X.columns)
        self.feat_cols_ = list(feat_cols)

        # 1) Skew 측정 → 두 그룹 분기
        skew_vals = X[self.feat_cols_].skew().abs()
        self.skew_vals_ = skew_vals
        self.quantile_cols_ = skew_vals[skew_vals > self.skew_threshold].index.tolist()
        self.power_cols_ = [c for c in self.feat_cols_ if c not in self.quantile_cols_]

        # 2) 각 그룹에 대해 train fit
        self.qt_ = None
        if self.quantile_cols_:
            n_q = min(self.n_quantiles, len(X))
            self.qt_ = QuantileTransformer(
                n_quantiles=n_q,
                output_distribution=self.quantile_output,
                subsample=int(1e6),
                random_state=self.random_state,
            )
            self.qt_.fit(X[self.quantile_cols_])

        self.pt_ = None
        if self.power_cols_:
            self.pt_ = PowerTransformer(method='yeo-johnson', standardize=True)
            self.pt_.fit(X[self.power_cols_])

        print(f"[HybridScaler.fit] skew_threshold={self.skew_threshold}")
        print(f"  Quantile 적용 : {len(self.quantile_cols_)}개 (|skew| > {self.skew_threshold})")
        print(f"  Power 적용    : {len(self.power_cols_)}개 (Yeo-Johnson)")
        return self

    def transform(self, X, inplace=True):
        """
        Parameters
        ----------
        X : DataFrame (train/val/test 중 하나)
        inplace : bool, default True
            True면 X 수정, False면 복사본 반환

        Returns
        -------
        X_transformed : DataFrame
        """
        if not inplace:
            X = X.copy()
        if self.quantile_cols_ and self.qt_ is not None:
            X[self.quantile_cols_] = self.qt_.transform(X[self.quantile_cols_])
        if self.power_cols_ and self.pt_ is not None:
            X[self.power_cols_] = self.pt_.transform(X[self.power_cols_])
        return X

    @property
    def transform_map_(self):
        """컬럼별 적용된 변환 종류 {feature: 'quantile'|'power'}"""
        return {
            **{c: 'quantile' for c in self.quantile_cols_},
            **{c: 'power'    for c in self.power_cols_},
        }
```

### 3.2 편의 함수 `hybrid_scale(xs_train, xs_val, xs_test, feat_cols, skew_threshold)`

```python
def hybrid_scale(xs_train, xs_val, xs_test, feat_cols, skew_threshold=10.0):
    """
    fit-on-train + transform-all 편의 함수.

    _run_preprocessing에서 이 함수 1번 호출하면 모든 split이 스케일링됨.
    scaler 객체도 반환 → 대시보드 / 예측 파이프라인 재사용 가능.

    Returns
    -------
    xs_train, xs_val, xs_test : DataFrame (in-place 수정됨)
    scaler : HybridScaler (fitted, pickle 저장 가능)
    """
    scaler = HybridScaler(skew_threshold=skew_threshold).fit(xs_train, feat_cols)
    scaler.transform(xs_train, inplace=True)
    scaler.transform(xs_val,   inplace=True)
    scaler.transform(xs_test,  inplace=True)
    return xs_train, xs_val, xs_test, scaler
```

### 3.3 기존 `scale()` 통합 함수에 `'hybrid'` 분기 추가 (deprecated 권장)

기존 `scale(xs, feat_cols, train_mask, transform)` 형태는 index 처리가 까다로워 **신규 노트북에서는 `hybrid_scale(xs_train, xs_val, xs_test, ...)` 편의 함수 사용 권장**. 기존 `scale()`은 호환성 유지용으로만 존속.

### 3.4 사용 예시

```python
from scaling import hybrid_scale, HybridScaler

# 방법 1: 편의 함수 (일반적 사용)
xs_train, xs_val, xs_test, scaler = hybrid_scale(
    xs_train, xs_val, xs_test, feat_cols,
    skew_threshold=10.0,
)
print(f"Quantile 적용 feature: {len(scaler.quantile_cols_)}")
print(f"Power 적용 feature:    {len(scaler.power_cols_)}")

# 방법 2: 직접 클래스 사용 (세밀 제어)
scaler = HybridScaler(skew_threshold=10.0).fit(xs_train, feat_cols)
scaler.transform(xs_train)
scaler.transform(xs_val)
scaler.transform(xs_test)

# 방법 3: 저장/재사용 (예측 파이프라인)
import joblib
joblib.dump(scaler, 'hybrid_scaler.pkl')
# ... later ...
scaler = joblib.load('hybrid_scaler.pkl')
scaler.transform(xs_new)
```

---

## 4. outlier.py 확장

### 4.1 `multivariate_anomaly_score(xs_train, xs_val, xs_test, feat_cols, contamination='auto', n_estimators=200)`

IsolationForest로 **die-level anomaly score를 feature 컬럼으로 추가**. 제거 아님. 트리 모델이 `score > τ` 분기로 자동 활용 가능.

```python
from sklearn.ensemble import IsolationForest

def multivariate_anomaly_score(xs_train, xs_val, xs_test, feat_cols,
                               contamination='auto',
                               n_estimators=200,
                               score_col='iso_anomaly_score',
                               random_state=42):
    """
    IsolationForest 기반 다변량 이상치 점수를 컬럼으로 추가

    Parameters
    ----------
    xs_train, xs_val, xs_test : DataFrame
    feat_cols : list
        anomaly score 계산에 쓸 feature (일반적으로 cleaning 이후 남은 cols)
    contamination : 'auto' or float in (0, 0.5)
    n_estimators : int, default 200
    score_col : str, default 'iso_anomaly_score'
        추가될 컬럼 이름
    random_state : int

    Returns
    -------
    xs_train, xs_val, xs_test : DataFrame (score_col 컬럼 추가됨)
    new_feat_cols : list (기존 feat_cols + [score_col])
    report : dict
    """
    iso = IsolationForest(contamination=contamination,
                          n_estimators=n_estimators,
                          random_state=random_state,
                          n_jobs=-1)
    iso.fit(xs_train[feat_cols].values)

    # decision_function: 높을수록 normal, 낮을수록 anomaly
    # anomaly score로 쓸 땐 부호 뒤집어 "높을수록 이상" 으로 통일
    for df in [xs_train, xs_val, xs_test]:
        df[score_col] = -iso.decision_function(df[feat_cols].values)

    new_feat_cols = feat_cols + [score_col]
    report = {
        'n_estimators': n_estimators,
        'contamination': contamination,
        'score_col': score_col,
        'train_score_range': (float(xs_train[score_col].min()),
                              float(xs_train[score_col].max())),
    }
    print(f"[IsoForest] anomaly score 컬럼 추가: '{score_col}'")
    print(f"  train score 범위: [{report['train_score_range'][0]:.4f}, "
          f"{report['train_score_range'][1]:.4f}]")
    return xs_train, xs_val, xs_test, new_feat_cols, report
```

### 4.2 근거 (CLAUDE.md L530)

> "다변량 이상치 점수 피처: IsolationForest로 die-level anomaly score를 컬럼으로 추가 (제거 아님). 단일 \|r\| max=0.037 → 단변량 clip의 한계 보완. 트리 모델이 `score > τ` 분기로 자동 활용"

### 4.3 옵튜나 축

`PP_ISO_ANOMALY_CANDIDATES`:
- `iso_enabled`: `[True, False]` — 컬럼 추가 여부 (marginal 효과 측정용)
- `iso_contamination`: `[0.05, 0.1, 'auto']` — anomaly 비율 가정

---

## 5. sample_weight.py 신규

### 5.1 파일 생성 위치

`2_preprocessing/sample_weight.py` (신규)

### 5.2 `compute_lds_weights(...)` (사용자 결정 2026-04-16: 옵션 A — `expand_to_die=True` 플래그)

Label Distribution Smoothing — Y>0 내부 long-tail에 가중치 재분배. **unit-level 계산 후 die-level로 자동 확장** 기능 내장.

```python
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

def compute_lds_weights(y_train, sigma=0.01,
                        min_weight=0.1, max_weight=10.0,
                        only_positive=True,
                        expand_to_die=False,
                        ys_train_df=None,
                        pos_data=None,
                        key_col='ufs_serial'):
    """
    LDS (Label Distribution Smoothing) sample_weight 계산

    논문: Yang et al., ICML 2021 "Delving into Deep Imbalanced Regression"
    구현: y 분포를 Gaussian KDE smoothing → density 추정 →
          w_i = 1 / density(y_i), min/max 클립 후 정규화

    Parameters
    ----------
    y_train : np.ndarray (1D) or pd.Series
        unit-level target (health). len = n_units (~26K)
    sigma : float, default 0.01
        Gaussian kernel bandwidth. EDA y>0 평균 0.0087 기준 0.005~0.02 권장
    min_weight, max_weight : float
        가중치 클립 범위 (학습 폭주 방지)
    only_positive : bool, default True
        True: y>0 샘플만 가중치 적용, y=0은 weight=1 (Stage 2 전용)
    expand_to_die : bool, default False
        ★ True면 die-level weight 반환 (reg_level='position' 전용).
        각 unit의 weight를 그 unit에 속한 die들에 복제.
        ys_train_df + pos_data 인자 필수 (ufs_serial 매핑용).
        False면 unit-level weight만 반환 (~26K).
    ys_train_df : DataFrame, required if expand_to_die=True
        ys_input['train'] — ufs_serial, TARGET_COL 컬럼 포함
    pos_data : dict, required if expand_to_die=True
        {position: {'train': df, 'val': df, 'test': df}}
        die 순서 결정용 (position concat 기준)
    key_col : str, default 'ufs_serial'
        unit 식별 컬럼

    Returns
    -------
    weights : np.ndarray (1D)
        - expand_to_die=False: shape (n_units,)  — unit-level
        - expand_to_die=True:  shape (n_dies,)   — die-level (position concat 순)
    info : dict
        {'effective_sigma', 'n_positive', 'weight_min', 'weight_max',
         'weight_std', 'expanded', 'n_die' (expand_to_die=True일 때)}
    """
    y_train = np.asarray(y_train, dtype=float)
    weight_unit = np.ones_like(y_train)

    if only_positive:
        mask = y_train > 0
        y_sub = y_train[mask]
    else:
        mask = np.ones_like(y_train, dtype=bool)
        y_sub = y_train

    if len(y_sub) < 2:
        info = {'effective_sigma': sigma, 'note': 'too few samples',
                'expanded': False, 'n_positive': int(mask.sum())}
        if expand_to_die:
            # 확장도 건너뛰고 die-level np.ones 반환
            assert ys_train_df is not None and pos_data is not None
            n_die = sum(len(pos_data[p]['train']) for p in sorted(pos_data.keys()))
            return np.ones(n_die), info
        return weight_unit, info

    # Gaussian KDE
    kde = gaussian_kde(y_sub, bw_method=sigma / y_sub.std())
    density = kde(y_sub)

    w_sub = 1.0 / (density + 1e-12)
    w_sub = w_sub / w_sub.mean()                              # 평균 1 정규화
    w_sub = np.clip(w_sub, min_weight, max_weight)            # 극단 클립
    w_sub = w_sub / w_sub.mean()                              # 재정규화

    weight_unit[mask] = w_sub

    info = {
        'effective_sigma': sigma,
        'n_positive': int(mask.sum()),
        'weight_min': float(w_sub.min()),
        'weight_max': float(w_sub.max()),
        'weight_std': float(w_sub.std()),
        'expanded': False,
    }

    # ── die-level 확장 (expand_to_die=True일 때만) ──
    if not expand_to_die:
        return weight_unit, info

    assert ys_train_df is not None, "expand_to_die=True면 ys_train_df 필요"
    assert pos_data is not None, "expand_to_die=True면 pos_data 필요"

    # unit-level weight를 ufs_serial 기준 Series로
    ufs_key = ys_train_df[key_col].values
    assert len(ufs_key) == len(weight_unit), \
        f"y_train 길이({len(weight_unit)}) != ys_train_df({len(ufs_key)})"
    weight_series = pd.Series(weight_unit, index=ufs_key)

    # die-level 순서: _prepare_unit_data(reg_level='position')와 동일
    #   pos_data[1]['train'] → pos_data[2]['train'] → ... concat
    die_weights = []
    for pos in sorted(pos_data.keys()):
        ufs_in_pos = pos_data[pos]['train'][key_col].values
        # ufs_serial로 lookup하여 weight 복제
        die_weights.append(weight_series.loc[ufs_in_pos].values)

    weight_die = np.concatenate(die_weights)
    info['expanded'] = True
    info['n_die'] = len(weight_die)
    info['n_unit'] = len(weight_unit)
    return weight_die, info
```

### 5.3 적용 위치

- **fit on train**: train y로 unit-level weight 계산
- **`expand_to_die=True`** (2차 `reg_level='position'` 전용): die-level weight 반환 → 학습 X와 길이 일치
- **apply on train only**: val/test는 평가용이라 weight 불필요 (None 전달)
- **Stage 2 회귀 `model.fit(X, y, sample_weight=w)`** 인자로 주입
- **E2E 통합**: `_run_preprocessing`에서 `expand_to_die=True`로 호출 → die-level weight 받아 `_run_reg_oof_multi`까지 전달
  - `model_zoo.fit_model()` 시그니처에 `sample_weight=None` 인자 추가 필요 (MD#2 §2.1.4 참조)

### 5.4 옵튜나 축

`PP_LDS_CANDIDATES`:
- `lds_enabled`: `[True, False]` — LDS 적용 여부
- `lds_sigma`: `[0.005, 0.01, 0.02]` — kernel bandwidth 후보
- `lds_max_weight`: `[5.0, 10.0]` — 극단 가중치 상한

### 5.5 근거

- [논문요약.md:1591](../99_학습자료/스터디자료/paper/논문요약.md#L1591) 8-1 (DIR, ICML 2021)
- [strategy_tail_first_experiments.md:147](strategy_tail_first_experiments.md#L147) E5 실험 계획
- SqInv + LDS + FDS: IMDB-WIKI few-shot MAE 26.33 → 22.19 (15.6% 감소)

---

## 6. search_space.py 확장

### 6.1 신규 상수

```python
# ── IsoForest 이상치 점수 후보 ──
PP_ISO_ANOMALY_CANDIDATES = {
    "iso_enabled":       [True, False],          # marginal 효과 측정용 'none' 포함
    "iso_contamination": [0.05, 0.1, 'auto'],
    "iso_n_estimators":  [100, 200],
}

# ── LDS 가중치 후보 ──
PP_LDS_CANDIDATES = {
    "lds_enabled":    [True, False],
    "lds_sigma":      [0.005, 0.01, 0.02],       # y>0 평균 0.0087 기준
    "lds_max_weight": [5.0, 10.0],
}

# ── Hybrid scale 파라미터 (옵튜나 탐색 X — 고정) ──
# skew_threshold: 5.0 (test.ipynb 4종 비교 결과 확정 — 정규근사율 68% 최고)
# binary passthrough: nunique≤2 feature는 변환 없음 (binarize 스텝 결과 보존)
PP_SCALE_CONFIG = {
    "transform":         "hybrid",
    "skew_threshold":    5.0,
    "binary_passthrough": True,
}

# ── Binarize 스텝 파라미터 (옵튜나 탐색 X — cleaning 직후 고정) ──
PP_BINARIZE_CONFIG = {
    "apply":               True,
    "top_value_threshold": 0.95,   # 최빈값 비율 > 0.95 OR
    "max_unique":          5,      # nunique ≤ 5 → 0/1 변환
}
```

### 6.2 `preprocessing_space()` 확장

```python
def preprocessing_space(trial):
    params = {}

    # ── 기존 ──
    for key, candidates in PP_CLEAN_CANDIDATES.items():
        params[key] = trial.suggest_categorical(f"pp_{key}", candidates)
    for key, candidates in PP_OUTLIER_CANDIDATES.items():
        params[f"outlier__{key}"] = trial.suggest_categorical(
            f"pp_outlier_{key}", candidates)
    params["agg_preset_idx"] = trial.suggest_categorical(
        "pp_agg_preset_idx", PP_AGG_PRESET_IDX_CANDIDATES)

    # ── 신규: IsoForest ──
    for key, candidates in PP_ISO_ANOMALY_CANDIDATES.items():
        params[f"iso__{key}"] = trial.suggest_categorical(
            f"pp_iso_{key}", candidates)

    # ── 신규: LDS ──
    for key, candidates in PP_LDS_CANDIDATES.items():
        params[f"lds__{key}"] = trial.suggest_categorical(
            f"pp_lds_{key}", candidates)

    return params
```

### 6.3 `split_pp_params()` 반환 확장

```python
def split_pp_params(pp_params):
    cleaning_args = {k: v for k, v in pp_params.items() if k in CLEANING_KEYS}

    outlier_args = {k[len("outlier__"):]: v
                    for k, v in pp_params.items() if k.startswith("outlier__")}

    # 신규: IsoForest 인자
    iso_args = {k[len("iso__"):]: v
                for k, v in pp_params.items() if k.startswith("iso__")}

    # 신규: LDS 인자
    lds_args = {k[len("lds__"):]: v
                for k, v in pp_params.items() if k.startswith("lds__")}

    agg_idx = pp_params.get("agg_preset_idx", 0)
    agg_funcs = AGG_PRESETS[agg_idx]

    return cleaning_args, outlier_args, iso_args, lds_args, agg_funcs
```

### 6.4 `extract_pp_params_from_best()` 확장

기존 `pp_outlier_*` 파싱에 이어 `pp_iso_*`, `pp_lds_*`도 내부 prefix로 복원.

---

## 7. `_run_preprocessing` 통합 순서

### 7.1 실행 순서 (엄격)

```
[Step 1] cleaning (기존)
    ├─ 상수/고결측/중복/고상관 feature 제거
    ├─ imputation (spatial/median/knn)
    └─ post-impute corr 2차 제거
         ↓
[Step 1.5] ★ binarize_degenerate (신규, 고정)
    ├─ top% > 0.95 OR nunique ≤ 5 → 0/1 (int8)
    ├─ clean_cols 수 유지, 값만 변환
    └─ train 기준 최빈값 판정 → val/test 동일 매핑
         ↓
[Step 2] outlier (기존)
    └─ winsorize/iqr_clip/grubbs/lot_local/none
       (binarized 컬럼은 winsorize 영향 없음)
         ↓
[Step 3] ★ IsoForest anomaly score (신규)
    ├─ iso_enabled=False면 스킵
    └─ train fit → train/val/test에 컬럼 추가
         ↓
[Step 4] ★ hybrid_scale (신규, 옵튜나 탐색 X, 방식 B)
    ├─ HybridScaler(skew_threshold=5.0).fit(xs_train)  ← train만 fit
    ├─ 그룹 분기: binary(nunique≤2 passthrough) / quantile(|skew|>5) / power
    ├─ scaler.transform(xs_train) / xs_val / xs_test   ← in-place
    └─ binarize 결과(Step 1.5)는 binary passthrough로 자동 보존
         ↓
[Step 5] exclude_cols 필터 (기존)
         ↓
[Step 6] pos_data 빌드 (기존)
    └─ die-level X + position별 분리
         ↓
[Step 7] ★ LDS sample_weight 계산 (신규, die-level 확장)
    ├─ lds_enabled=False면 sample_weight=None
    └─ expand_to_die=True → die-level weight 반환
       (pos_data의 die 순서에 맞춰 ufs_serial로 매핑)
```

> **주의 — LDS `max_weight`는 soft cap**: `np.clip(..., max_weight)` 이후 mean=1 재정규화를 거치므로, 최종 weight max는 `max_weight`를 **초과 가능** (test.ipynb 실측: `max_weight=10` 설정에 최종 max≈25). 학습 폭주 방지는 clip 시점에 이뤄지고, 재정규화는 gradient 스케일 유지 목적. 파라미터 해석 주의.

### 7.2 `_run_preprocessing` 시그니처 변경

```python
def _run_preprocessing(xs, xs_dict, ys, feat_cols,
                       cleaning_args, outlier_args,
                       iso_args, lds_args,                 # ★ 신규
                       label_col, exclude_cols,
                       use_sampling, sample_frac):
    """
    cleaning + outlier + IsoForest + hybrid_scale + LDS weight + pos_data 빌드

    Returns
    -------
    pos_data : dict
    feat_cols_clean : list (IsoForest score 컬럼 포함)
    sample_weight : np.ndarray (unit-level, train 전용)
    """
    from cleaning import run_cleaning
    from outlier import run_outlier_treatment
    from outlier import multivariate_anomaly_score      # ★ 신규
    from scaling import hybrid_scale                    # ★ 신규
    from sample_weight import compute_lds_weights       # ★ 신규
    from utils.config import PP_SCALE_CONFIG            # 고정값

    ys_train = ys["train"]

    # Step 1: cleaning
    xs_train, xs_val, xs_test, clean_cols, _ = run_cleaning(
        xs, feat_cols, xs_dict, ys_train=ys_train, **cleaning_args)

    # Step 2: outlier
    xs_train, xs_val, xs_test, _ = run_outlier_treatment(
        xs_train, xs_val, xs_test, clean_cols, **outlier_args)

    # Step 3: IsoForest (옵션)
    if iso_args.get('iso_enabled', False):
        xs_train, xs_val, xs_test, clean_cols, _ = multivariate_anomaly_score(
            xs_train, xs_val, xs_test, clean_cols,
            contamination=iso_args.get('iso_contamination', 'auto'),
            n_estimators=iso_args.get('iso_n_estimators', 200),
        )

    # Step 4: hybrid_scale (방식 B — fit on train, transform on all)
    # ★ concat/split 방식 X → 각 split에 transform 직접 적용 (index 안전)
    xs_train, xs_val, xs_test, scaler = hybrid_scale(
        xs_train, xs_val, xs_test, clean_cols,
        skew_threshold=PP_SCALE_CONFIG['skew_threshold'],
    )
    # scaler는 향후 대시보드 / 예측 파이프라인 재사용 가능 (pickle 권장)

    # Step 5: exclude_cols 필터 (pos_data 빌드 전에 수행)
    if exclude_cols:
        clean_cols = [c for c in clean_cols if c not in set(exclude_cols)]

    # Step 6: pos_data 빌드 (기존)
    pos_data = _build_pos_data(xs_train, xs_val, xs_test, ys, label_col,
                               use_sampling=use_sampling,
                               sample_frac=sample_frac, silent=True)

    # Step 7: LDS sample_weight (★ die-level 확장, pos_data 필요)
    if lds_args.get('lds_enabled', False):
        y_unit_train = ys_train[TARGET_COL].values
        sample_weight, _ = compute_lds_weights(
            y_unit_train,
            sigma=lds_args.get('lds_sigma', 0.01),
            max_weight=lds_args.get('lds_max_weight', 10.0),
            expand_to_die=True,            # ★ die-level weight 반환
            ys_train_df=ys_train,          # ★ ufs_serial 매핑용
            pos_data=pos_data,             # ★ die 순서 결정용
            key_col=KEY_COL,
        )
    else:
        sample_weight = None        # = weight 없음

    return pos_data, clean_cols, sample_weight, scaler
```

**주의**: Step 순서가 기존 제안과 다름 — **pos_data 빌드를 LDS보다 먼저** 해야 `expand_to_die=True`가 die 순서를 알 수 있음. 또한 `scaler` 객체도 반환에 포함 (향후 재사용).

### 7.3 호출부 변경

```python
# 기존 (baseline):
pos_data, feat_cols_clean = _run_preprocessing(
    xs, xs_dict, ys, feat_cols,
    cleaning_args, outlier_args,
    label_col, exclude_cols, ...)

# 신규 (2차):
pos_data, feat_cols_clean, sample_weight, scaler = _run_preprocessing(
    xs, xs_dict, ys, feat_cols,
    cleaning_args, outlier_args, iso_args, lds_args,    # ★ 인자 추가
    label_col, exclude_cols, ...)
# scaler: HybridScaler 객체 — 대시보드/예측 파이프라인 재사용용
```

---

## 8. LRU 캐시 key 구성

### 8.1 기존 `_pp_hash()`

```python
def _pp_hash(pp_params):
    key = json.dumps(pp_params, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()
```

### 8.2 변경 불필요

`pp_params`에 이미 모든 축 (cleaning / outlier / iso / lds / agg_preset) 포함되므로 **자동 반영됨**. 해시 충돌 없음.

### 8.3 캐시된 값 구조 변경

```python
# 기존:
value = {"pos_data": ..., "feat_cols": ..., "n_feat": ...}

# 신규 (sample_weight + scaler 추가):
value = {
    "pos_data": ...,
    "feat_cols": ...,
    "n_feat": ...,
    "sample_weight": sample_weight,    # ★ die-level train 가중치 (~105K)
    "scaler": scaler,                  # ★ HybridScaler (fitted)
}
```

### 8.4 캐시 크기 주의

- `sample_weight`는 train **die 수** (~105K) float 벡터 → ~840KB per cache entry
- `scaler` 객체 (PowerTransformer + QuantileTransformer): ~수 MB (feature 수에 비례)
- `pp_cache_size=10` 기준 총 ~30~50MB 추가 — 허용 범위

---

## 9. 검증 체크리스트

### 9.1 함수 단위

- [ ] `HybridScaler.fit`: skew 측정이 train 기반인지, quantile/power 분기가 정확히 `|skew|>10`으로 가는지
- [ ] `HybridScaler.transform`: val/test 적용 시 내부 qt_/pt_ 객체로 transform만 (fit 재호출 X)
- [ ] `HybridScaler`: pickle 저장/로드 후 동일 출력 나오는지 (대시보드 재사용 검증)
- [ ] `hybrid_scale` 편의함수: train fit → 3 split transform이 순서대로 실행되는지
- [ ] `multivariate_anomaly_score`: train fit, val/test는 decision_function transform만
- [ ] `compute_lds_weights(expand_to_die=False)`: unit-level weight (~26K) + weights.mean() ≈ 1
- [ ] `compute_lds_weights(expand_to_die=True)`: die-level weight (~105K) + unit당 4 die에 동일 weight
- [ ] `compute_lds_weights`: ufs_serial로 매핑 시 pos_data[pos]['train']의 순서와 일치

### 9.2 통합 단위 (`_run_preprocessing`)

- [ ] 모든 스텝이 train 기반 fit (누수 없음)
- [ ] IsoForest score 컬럼이 `feat_cols`에 추가되어 하위 모델 학습에 전달됨
- [ ] hybrid_scale이 cleaning/outlier/IsoForest 이후에 적용됨 (순서 중요)
- [ ] **die-level sample_weight가 `_run_reg_oof_multi`까지 전달되고 길이 일치** (~105K)
- [ ] **pos_data 빌드 후에 LDS 호출** (die 순서 결정 필요)
- [ ] `iso_enabled=False` / `lds_enabled=False` 조합도 정상 동작 (passthrough)
- [ ] 반환값 4-tuple: `(pos_data, feat_cols, sample_weight, scaler)`

### 9.3 Optuna 통합

- [ ] `pp_iso_*`, `pp_lds_*` 파라미터가 trial.params에 기록
- [ ] `study.trials_dataframe()`에서 각 축 분포 확인 가능
- [ ] `extract_pp_params_from_best()`가 신규 축 복원 정확

### 9.4 성능 확인

- [ ] 1차 best 전처리 + 하이브리드 스케일 적용 시 단일 LGBM RMSE 재측정 → 개선 또는 유지
- [ ] IsoForest 컬럼 추가 시 LGBM feature importance에 상위권 진입 여부 확인
- [ ] LDS sigma sweep 결과 Y>0 내부 RMSE 개선 여부

---

## 10. 구현 순서 제안

1. **T1.1 scaling.py** — `quantile_scale()`, `hybrid_scale()` 구현 + 단위 테스트
2. **T1.2 outlier.py** — `multivariate_anomaly_score()` 구현
3. **T1.3 sample_weight.py** — `compute_lds_weights()` 신규 파일 생성
4. **T1.4 search_space.py** — 상수/함수 확장
5. **T1.5 e2e_hpo._run_preprocessing** — 통합
6. **smoke test** — n_trials=2, n_folds=2로 1회 실행 → 모든 축 정상 동작 확인
7. `preprocessing.zip` 재생성 → Drive 업로드

---

## 11. 참고

- [CLAUDE.md](../CLAUDE.md) Stage 3 / Stage 4.5 / L515~564 (2차 funnel 계획)
- [strategy_tail_first_experiments.md](strategy_tail_first_experiments.md) E5 (LDS 원안)
- [논문요약.md:1591](../99_학습자료/스터디자료/paper/논문요약.md#L1591) 8-1 (DIR/LDS 원논문)
- [미반영_항목_검토.md](../99_학습자료/스터디자료/미반영_항목_검토.md) Phase 2 #10/#11 (IsoForest/QT)