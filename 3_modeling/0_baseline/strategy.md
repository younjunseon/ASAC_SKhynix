# 0_baseline 전략 — Ablation 실험 (발표용)

> 공통 규칙은 [strategy_common.md](../strategy_common.md) 참조. 본 문서는 0_baseline 전용 사항만 명시.

---

## 1. 목적

발표/기록용 베이스라인 ablation. **계획했던 베이스라인 실험들을 깨끗하게 한 번 더 정리**해 발표 자료 + 영구 기록 남기기.

**문제 의식**:
- [optuna_merged.db](../../4_output/experiments/1차%20실험/optuna_merged.db)는 1차 실험 trial 모음이지만 (a) 계획했던 실험을 다 한 게 아님 (b) 변환된 환경에서 찍힌 값이라 절대값 의미 없고 참고용
- 한 study에 모든 축을 때려넣으면 TPE가 좋은 영역에 trial 몰아주므로 **조합 간 공정 비교 불가** (편향)

**해결 방향**: **OAT(One-At-A-Time) + Group Study 2단계**, 7+7 코어 병렬

**원칙**: baseline.ipynb의 실제 동작과 같은 코드를 쓰고, 거기에 **축 흔들기 + 축 추가**만 얹는다.

---

## 2. 디렉토리 구조

```
3_modeling_신규/0_baseline/
├── strategy.md                # 이 문서
├── axes.py                    # 축 정의 + reference + 그리드 생성 helper + run_one()
├── 01_oat.ipynb               # OAT 노트북 (한 축씩 흔들기)
├── 02_group_study.ipynb       # Group study 노트북 (Optuna multivariate)
└── 03_summary.ipynb           # 두 결과 종합 + 시각화 (발표 직전 1회)
```

산출물 위치 (베이스라인 산출물임을 경로에서 식별):
```
4_output/baseline/
├── oat/
│   ├── master.csv             # cell × seed × fold RMSE 영구 박제
│   ├── checkpoint.json        # 진행 상황 (재실행 시 이어서)
│   └── plots/                 # tornado.png 등
├── group/
│   ├── optuna.db              # study sqlite
│   ├── trials.csv             # study.trials_dataframe() export
│   └── plots/                 # contour, importance.png 등
└── summary/
    ├── comparison.csv         # OAT marginal vs Group best 비교표
    └── summary_report.png     # 발표용 종합 1장
```

---

## 3. 설계 — 2단계 병렬

### 3.1 전략 개요

```
[슬롯 A — 코어 7개]                [슬롯 B — 코어 7개]
01_oat.ipynb                       02_group_study.ipynb
  ├ 12축 × 옵션 × seed                ├ 12축 동시 Optuna 탐색
  ├ 한 번에 한 축만 흔듦               ├ TPE multivariate
  ├ marginal 효과 측정                ├ 상호작용 학습
  └ → master.csv                    └ → optuna.db
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
              03_summary.ipynb
              tornado / heatmap / 종합 비교
```

### 3.2 두 노트북의 역할

| | OAT (01) | Group study (02) |
|---|---|---|
| **목적** | 축별 marginal 영향력 측정 | 축들 상호작용 + best 조합 탐색 |
| **방식** | reference 1점에서 한 축씩 갈아끼움 | 12축 동시 샘플 (Optuna) |
| **편향** | 없음 (모든 비교가 1:1) | 있음 (TPE가 좋은 영역 집중) — 의도된 편향 |
| **발표 용도** | tornado plot ("이 축이 X 영향") | "최종 best 조합" |
| **결과 저장** | csv | sqlite db |

### 3.3 두 노트북의 일관성 (필수)

발표에서 두 결과를 나란히 보이려면 환경이 같아야 함:

- **데이터 split**: 같은 5-fold (KFold seed=42, unit-level — strategy_common §6)
- **PP base**: baseline.ipynb의 `pin` 값과 동일 (`const_threshold=1e-6`, `missing_threshold=0.4`, `corr_threshold=0.9`, `corr_keep_by='std'`, `add_indicator=True`, `indicator_threshold=0.05`, `spatial_max_dist=5.0`, `post_impute_corr_threshold=0.98`, `lower_pct=0.0`, `upper_pct=0.99`)
- **모델 HP**: LGBM default 고정 (HPO 없음 — 축 효과만 분리)
- **reference 1점**: OAT의 reference = Group study의 anchor (동일 영점)
- **axes.py에서 단일 정의**: 두 노트북이 import → 축 변경 시 1군데 수정

---

## 4. OAT 설계 (01_oat.ipynb)

### 4.1 Reference 1점

[experiment_status_summary.xlsx](../../4_output/experiments/1차%20실험/experiment_status_summary.xlsx) val_rmse 최소 셀(0.001141) + baseline.ipynb의 sensible default를 종합:

```python
REFERENCE = {
    'CLF':              'off',         # run_clf=False (xlsx best는 clf off)
    'clf_level':        'die',         # CLF=off면 무관, conditional
    'reg_level':        'position',
    'TARGET_TRANSFORM': 'yeo-johnson',
    'CLIP_Y_EXTREME':   True,
    'loss':             'regression',  # LGBM default = mse
    'impute':           'spatial',     # baseline.ipynb pin
    'outlier':          'winsorize',   # baseline.ipynb pin
    'agg_preset':       'P00_full6',   # baseline.ipynb의 AGG_PRESETS[0]
    'binarize':         False,         # default
    'iso_anomaly':      False,         # default
    'lds':              False,         # default
}
```

### 4.2 흔들 12축

**baseline.ipynb의 실제 변수와 1:1 매핑**. xlsx 6축 + 추가 6축.

| # | 축 | 옵션 | baseline.ipynb 위치 | 비고 |
|---|---|---|---|---|
| 1 | `CLF` | `off` / `proba` / `proba+filter` / `binary` | `pipeline_config.run_clf` × `clf_output` × `clf_filter` | xlsx 그대로 |
| 2 | `clf_level` ★ | `die` / `unit` | (현재 미구현) | **코드 구현 필요** — `unit` 옵션은 e2e_hpo에 unit-level 분류 분기 추가. `CLF=off`면 무관 (conditional) |
| 3 | `reg_level` | `unit` / `position` | `pipeline_config.reg_level` | xlsx 그대로 |
| 4 | `TARGET_TRANSFORM` | `none` / `log1p` / `yeo-johnson` | `TARGET_TRANSFORM` | xlsx 그대로. sqrt는 코드 미지원이라 제외 |
| 5 | `CLIP_Y_EXTREME` | `True` / `False` | `CLIP_Y_EXTREME` | xlsx 그대로 |
| 6 | `loss` | `regression` / `poisson` / `tweedie` / `huber` | `e2e_params.reg_fixed = {'objective': ...}` | LGBM objective. `regression` = mse |
| 7 | `impute` | `spatial` / `median` / `knn` | `PP_CLEAN_CANDIDATES.imputation_method` | baseline.ipynb 옵션 그대로 |
| 8 | `outlier` ★ | `none` / `winsorize` / `iqr_clip` / `grubbs` / `lot_local` | `PP_OUTLIER_CANDIDATES.method` | **`none` 분기 코드 추가 필요** ([outlier.py:425](../../2_preprocessing/outlier.py#L425) `run_outlier_treatment`에 passthrough 추가) |
| 9 | `agg_preset` | **12종 (§4.3)** | `AGG_PRESETS` | 확장 — `reg_level='unit'`일 때만 의미 |
| 10 | `binarize` | `True` / `False` | `PP_BINARIZE_CANDIDATES.apply` | baseline.ipynb 그대로 |
| 11 | `iso_anomaly` | `True` / `False` | `PP_ISO_ANOMALY_CANDIDATES.iso_enabled` | baseline.ipynb 그대로 |
| 12 | `lds` | `True` / `False` | `PP_LDS_CANDIDATES.lds_enabled` | baseline.ipynb 그대로 (Label Distribution Smoothing — zero-inflation 대응) |

**제외한 축** (코드 미지원 또는 효과 미미):
- `scaling` — `HybridScaler`가 컬럼별 자동 분기, 외부에서 강제 못 흔듦
- `zero_clip` — 사용자 지시로 제외
- `path` (회귀) — `reg_level=position` vs `unit`이 사실상 동일 역할
- `run_fs` / `clf_optuna` / `reg_optuna` — OAT는 default 고정이 원칙이라 무의미

### 4.3 agg_preset 확장 — 12종 ★

baseline.ipynb 4종 + 추가 8종. die→unit 집계 함수 조합 다양성 확보.

**코드 수정 필요**: [aggregate.py:65](../../3_modeling/modules/aggregate.py#L65) `aggregate_die_to_unit`에 `Q25, Q75` 분기 추가 (현재 native 지원: `mean / std / cv / range / min / max / median` 7종만). `CV`는 이미 지원.

```python
AGG_PRESET_LIB = {
    # ── baseline.ipynb 원본 4종 ──
    'P00_full6':     ['mean', 'std', 'range', 'min', 'max', 'median'],
    'P01_meanstd':   ['mean', 'std'],
    'P02_basic4':    ['mean', 'std', 'median', 'range'],
    'P03_disp3':     ['std', 'range', 'median'],
    # ── 단일 함수 4종 ──
    'P04_mean':      ['mean'],
    'P05_median':    ['median'],
    'P06_max':       ['max'],
    'P07_min':       ['min'],
    # ── 극단/분위수 ──
    'P08_extremes':  ['max', 'min', 'range'],
    'P09_quartiles': ['Q25', 'median', 'Q75'],
    # ── 풀세트 ──
    'P10_full8':     ['mean', 'std', 'range', 'min', 'max', 'median', 'Q25', 'Q75'],
    'P11_full9_cv':  ['mean', 'std', 'range', 'min', 'max', 'median', 'Q25', 'Q75', 'CV'],
}
```

총 **12종**. reference = `P00_full6`.

### 4.4 Cell 수 계산

각 축에서 reference와 다른 옵션만 변형 cell:

| 축 | 옵션 수 | 변형 cell |
|---|---|---|
| CLF | 4 | 3 |
| clf_level | 2 | 1 |
| reg_level | 2 | 1 |
| TARGET_TRANSFORM | 3 | 2 |
| CLIP_Y_EXTREME | 2 | 1 |
| loss | 4 | 3 |
| impute | 3 | 2 |
| outlier | 5 | 4 |
| agg_preset | 12 | 11 |
| binarize | 2 | 1 |
| iso_anomaly | 2 | 1 |
| lds | 2 | 1 |
| **합계** | | **31 변형** + 1 reference = **32 cell** |

× 5 seed = **160 cell-seed**
× 5 fold = **800 model fit**

LGBM default + position pivot 기준 fit당 4~6분 가정 → **53~80시간 ≈ 2.5~3.5일** (단일 슬롯).

### 4.5 출력 — master.csv

row 단위 영구 박제. 발표 시 csv 로딩 → tornado plot 즉석 생성.

| 컬럼 | 예시 |
|---|---|
| `axis` | `TARGET_TRANSFORM` |
| `option` | `log1p` |
| `seed` | 42 |
| `fold` | 0 |
| `val_rmse` | 0.005442 |
| `is_reference` | False |
| `elapsed_sec` | 187.3 |
| `timestamp` | 2026-05-08T14:23:01 |

`is_reference=True`인 row가 5 seed × 5 fold = 25개 (모든 축에서 reference 옵션과 같은 셀 1개).

### 4.6 Checkpoint — 끊겨도 이어서

`checkpoint.json`에 완료한 (axis, option, seed) 튜플 기록. 재실행 시 master.csv에 이미 있으면 skip.

```python
done = set(zip(master.axis, master.option, master.seed))
for cell in oat_grid:
    if cell.tuple in done: continue
    ...
```

---

## 5. Group Study 설계 (02_group_study.ipynb)

### 5.1 Search space — OAT의 12축 그대로

axes.py의 `AXES` dict를 그대로 Optuna `suggest_categorical`로 변환:

```python
def objective(trial):
    cfg = {
        axis: trial.suggest_categorical(axis, options)
        for axis, options in AXES.items()
    }
    return run_one(cfg, seed=42)   # 5-fold mean RMSE
```

LGBM HP는 default 고정. **축 12개만 흔듦** (HP 같이 흔들면 축 효과와 섞여 발표 일관성 깨짐).

### 5.2 Sampler

[strategy_common.md §4](../strategy_common.md):

```python
TPESampler(
    seed=None,            # 다양성
    multivariate=True,    # ★ 축들의 결합 분포 학습 = 상호작용
    group=True,           # ★ 자연 블록 (예: CLF=off면 clf_level 무관)
)
```

`group=True`로 `CLF=off`일 때 `clf_level` 미샘플 등 conditional 처리.

### 5.3 Pruner

LGBM default fit + 5-fold라 fold별 중간값 잘 안 나옴. **MedianPruner는 비활성화** (5 fold 다 끝나야 점수 나옴).

### 5.4 Trial 예산

- 12축 카르테시안 = 4×2×2×3×2×4×3×5×12×2×2×2 ≈ **552,960 조합**
- 모두 못 보니까 TPE로 좋은 영역 학습

| trial | fit | 예상 시간 (5-fold, fit당 ~5분) | 발표 적합성 |
|---|---|---|---|
| 200 | 1,000 | ~83h ≈ 3.5일 | 최소 |
| 300 | 1,500 | ~125h ≈ 5일 | **권장** |
| 500 | 2,500 | ~208h ≈ 8.5일 | 시간 빡빡 |

**default = 300 trial** (사용자 §11에서 변경 가능).

### 5.5 Optuna 산출물

- `optuna.db` — sqlite study (그대로 보존)
- `trials.csv` — `study.trials_dataframe()` export (master.csv와 같은 형식으로 변환 가능)
- `param_importance.csv` — `optuna.importance.get_param_importances()` (OAT tornado와 비교 가능)

---

## 6. 7+7 병렬 운용 — N_JOBS 설정

[strategy_common.md §8](../strategy_common.md) 14 코어 기준.

```python
# 두 노트북 상단 동일하게
N_JOBS = 7
```

- `LGBMRegressor(n_jobs=N_JOBS)` / `LGBMClassifier(n_jobs=N_JOBS)` — 모델 내부 병렬
- `study.optimize(..., n_jobs=1)` — Optuna trial 직렬 (모델 내부와 곱셈 효과 방지)
- OAT 노트북도 `for cell in grid:` 직렬 (모델 내부 N_JOBS만 활용)

**시작 순서**:
1. 슬롯 B (Group study) 먼저 시작 — TPE 학습 시간 필요
2. 슬롯 A (OAT) 동시 시작 — 시간 짧음

---

## 7. axes.py — 두 노트북 공통 모듈

```python
# 3_modeling_신규/0_baseline/axes.py

REFERENCE = { ... }     # §4.1
AGG_PRESET_LIB = { ... } # §4.3

AXES = {
    'CLF':              ['off', 'proba', 'proba+filter', 'binary'],
    'clf_level':        ['die', 'unit'],   # CLF=off면 conditional skip
    'reg_level':        ['unit', 'position'],
    'TARGET_TRANSFORM': ['none', 'log1p', 'yeo-johnson'],
    'CLIP_Y_EXTREME':   [True, False],
    'loss':             ['regression', 'poisson', 'tweedie', 'huber'],
    'impute':           ['spatial', 'median', 'knn'],
    'outlier':          ['none', 'winsorize', 'iqr_clip', 'grubbs', 'lot_local'],
    'agg_preset':       list(AGG_PRESET_LIB.keys()),   # 12종
    'binarize':         [True, False],
    'iso_anomaly':      [True, False],
    'lds':              [True, False],
}

SEEDS = [42, 7, 123, 2024, 31415]


def generate_oat_grid():
    """OAT cell list 생성. reference 1개 + 각 축의 non-reference 옵션."""
    grid = [(REFERENCE.copy(), seed) for seed in SEEDS]   # reference × 5 seed
    for axis, options in AXES.items():
        ref_val = REFERENCE[axis]
        for opt in options:
            if opt == ref_val: continue
            cfg = REFERENCE.copy()
            cfg[axis] = opt
            for seed in SEEDS:
                grid.append((cfg, seed))
    return grid


def run_one(cfg: dict, seed: int) -> dict:
    """단일 cell 학습 + 5-fold val RMSE 반환.

    1) cfg를 baseline.ipynb의 PP_*_CANDIDATES / pipeline_config / e2e_params 형태로 매핑
    2) baseline.ipynb의 run_e2e_optimization_with_pp() 와 동일 함수 호출 (단, n_trials=1, default HP)
    3) 5-fold OOF RMSE 반환
    """
    ...
```

핵심: **`run_one`이 두 노트북 공통 entrypoint**. OAT는 grid 순회, Group study는 Optuna가 호출.

---

## 8. 03_summary.ipynb — 시각화

발표 직전 1회 실행. master.csv + optuna.db 읽어 종합:

### 8.1 OAT — Tornado plot

축별 RMSE range (max - min) 정렬, 가로 bar.

### 8.2 Group study — Param importance

`optuna.importance.get_param_importances()` 막대 그래프. OAT tornado와 나란히 비교 → "marginal과 multivariate가 일치하면 신뢰성 ↑".

### 8.3 비교표 (comparison.csv)

| axis | OAT marginal best | Group study best | 일치? |
|---|---|---|---|
| TARGET_TRANSFORM | yeo-johnson | yeo-johnson | O |
| CLF | off | off | O |
| ... | ... | ... | ... |

### 8.4 출력

- `summary/tornado.png` — OAT
- `summary/group_importance.png` — Group study
- `summary/comparison.csv` — 표
- `summary/summary_report.png` — 위 3개 합본 1장 (발표용)

---

## 9. 발표 스토리

```
"우리는 베이스라인 단계에서 12개 축의 효과를 측정했다.

1) OAT (편향 없는 marginal):
   - 32 cell × 5 seed = 160 fit
   - tornado plot: TARGET_TRANSFORM이 가장 큰 축 (Δ 0.0044)
   - scaling은 거의 무영향 (트리 모델이라 예상대로)

2) Group study (상호작용 포함):
   - 300 Optuna trial, multivariate TPE
   - best 조합: target=yeo-johnson + clip=True + clf=off + ...
   - param importance ranking이 OAT tornado와 일치
   → 상호작용 효과 작음 = OAT marginal이 충분 신뢰 가능

3) 결론: 12개 축 중 X개가 RMSE 좌우.
   나머지는 default 그대로 가도 안전 → 모델링 단계로 진입할 때
   탐색 공간을 X축으로 좁혀 효율화."
```

---

## 10. 노트북 작성 규칙

[strategy_common.md §14](../strategy_common.md) 준수.

특히 0_baseline은 **공정 비교가 목적**이라:
- LGBM HP는 default 고정 (HPO 안 함)
- PP base는 baseline.ipynb pin 값 (§3.3) 고정. 단, axes.py의 impute/outlier/binarize/iso/lds 축으로 흔드는 셀은 그 축만 변형
- 후처리 (zero_clip, position weight 등) **모두 비적용** — 축 효과만 분리

---

## 11. 결정 필요 사항

| # | 사항 | 제안 default | 결정 시점 |
|---|---|---|---|
| 1 | Group study trial 수 | 300 | 노트북 작성 전 |
| 2 | seed 5개 (`42, 7, 123, 2024, 31415`) OK? | OK | 노트북 작성 전 |
| 3 | `clf_level='unit'` 코드 구현 | e2e_hpo.py에 분기 추가 | **`clf_level` 축 켜기 전 필수** |
| 4 | LGBM default HP — `n_estimators` 명시 | sklearn default(100) | 노트북 작성 전 |
| 5 | `outlier` 'none' 분기 추가 | outlier.py `run_outlier_treatment`에 passthrough | 노트북 작성 전 |
| 6 | `agg_preset` Q25/Q75 분기 추가 | aggregate.py `aggregate_die_to_unit`에 quantile 분기 | 노트북 작성 전 |

---

## 12. 검증 근거

- **편향 우려**: TPE는 좋은 영역에 trial 몰아주므로 한 study 다축 학습 시 조합 간 trial 수 불균형 → 비교 불가. OAT는 reference 1점에서 1:1 비교라 편향 없음
- **OAT 한계 (상호작용)**: xlsx 데이터에서 CLIP 효과가 TARGET에 따라 큰 차이 (yeo-johnson + CLIP=F: Δ +0.00001 vs none + CLIP=F: Δ +0.00273) → OAT만으로는 못 잡음 → Group study로 보완
- **공정 비교 핵심**: 같은 5-fold split + 같은 PP pin + 같은 LGBM default HP로만 축 효과 분리 가능. axes.py의 `run_one()`이 단일 entrypoint
- **재현성**: master.csv에 모든 fit row 박제 → 발표 자료 png는 매번 csv에서 즉석 생성. 다음에 축 추가해도 csv 합치면 됨
- **baseline.ipynb 정합**: 모든 축이 baseline.ipynb의 실제 변수(`pipeline_config`, `PP_*_CANDIDATES`, `AGG_PRESETS`, `TARGET_TRANSFORM`, `CLIP_Y_EXTREME`)와 1:1 매핑. `clf_level='unit'`만 코드 추가 필요

---

## 13. 실행 순서

1. 사용자 §11 결정 사항 확인
2. `clf_level='unit'` 코드 구현 (e2e_hpo.py 분기)
3. axes.py 작성 (`run_one()` 포함)
4. 02_group_study.ipynb 시작 (코어 7) — 먼저 (시간 김)
5. 01_oat.ipynb 시작 (코어 7) — 동시
6. 둘 다 끝나면 03_summary.ipynb 1회 실행 → 발표 자료
