# 0_baseline 전략 — Ablation 실험 (발표용)

> 공통 규칙은 [strategy_common.md](../strategy_common.md) 참조. 본 문서는 0_baseline 전용 사항만 명시.

---

## 1. 목적

발표/기록용 베이스라인 ablation. **계획했던 베이스라인 실험들을 깨끗하게 한 번 더 정리**해 발표 자료 + 영구 기록 남기기.

**문제 의식**:
- [optuna_merged.db](../../모델링_이전자료/4_output_이전자료/experiments/1차%20실험/optuna_merged.db)는 1차 실험 trial 모음이지만 (a) 계획했던 실험을 다 한 게 아님 (b) 변환된 환경에서 찍힌 값이라 절대값 의미 없고 참고용
- 한 study에 모든 축을 때려넣으면 TPE가 좋은 영역에 trial 몰아주므로 **조합 간 공정 비교 불가** (편향)

**해결 방향**: **OAT(One-At-A-Time) + Group Study 2단계**, 7+7 코어 병렬

**원칙**: baseline.ipynb의 실제 동작과 같은 코드를 쓰고, 거기에 **축 흔들기 + 축 추가**만 얹는다.

---

## 2. 디렉토리 구조

```
3_modeling/0_baseline/
├── strategy.md                # 이 문서
├── _modules/                  # 0_baseline 전용 격리 모듈 (모델링_이전자료/3_modeling_이전자료/modules/ 복사 + 수정)
│   ├── e2e_hpo.py             # baseline.ipynb 그대로 사용 (분기 추가 없음)
│   ├── search_space.py
│   ├── model_zoo.py           # ZITboost 의존 제거
│   ├── aggregate.py           # Q25/Q75 분기 추가
│   ├── feature_select.py
│   ├── training.py
│   └── __init__.py
├── axes.py                    # 축 정의 + reference + 그리드 생성 helper + run_one()
├── 01_oat.ipynb               # OAT 노트북 (한 축씩 흔들기)
├── 02_group_study.ipynb       # Group study 노트북 (Optuna multivariate)
└── 03_summary.ipynb           # 두 결과 종합 + 시각화 (발표 직전 1회)
```

산출물 위치 (베이스라인 산출물임을 경로에서 식별):
```
4_output/baseline/
├── oat/
│   ├── master.csv             # row 단위 영구 박제 (155 row, cfg 11축 컬럼 포함)
│   ├── checkpoint.json        # 진행 상황 (재실행 시 이어서)
│   └── meta.json              # 실행 설정 + axis/PP/seed/exclude_cols (재현성)
├── group/
│   ├── optuna.db              # study sqlite (300 trial 누적)
│   ├── trials.csv             # study.trials_dataframe() export
│   ├── param_importance.csv   # fANOVA importance
│   └── meta.json              # 실행 설정 + sampler/pruner/study_name 등
└── summary/
    ├── tornado.png            # OAT 축별 RMSE range
    ├── group_importance.png   # Group fANOVA importance
    ├── comparison.csv         # OAT marginal best vs Group best (RMSE 값 포함)
    └── summary_report.png     # 발표용 종합 1장 (tornado + importance)
```

총 **11개 파일 고정** (oat 3 + group 4 + summary 4). 파일 개수는 trial/cell 수와 무관하게 일정 — master.csv는 row append, optuna.db는 sqlite 누적.

---

## 3. 설계 — 2단계 병렬

### 3.1 전략 개요

```
[슬롯 A — 코어 7개]                [슬롯 B — 코어 7개]
01_oat.ipynb                       02_group_study.ipynb
  ├ 11축 × 옵션 × seed                ├ 11축 동시 Optuna 탐색
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
| **방식** | reference 1점에서 한 축씩 갈아끼움 | 11축 동시 샘플 (Optuna) |
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

[experiment_status_summary.xlsx](../../모델링_이전자료/4_output_이전자료/experiments/1차%20실험/experiment_status_summary.xlsx) val_rmse 최소 셀(0.001141) + baseline.ipynb의 sensible default를 종합:

```python
REFERENCE = {
    'CLF':              'off',         # run_clf=False (xlsx best는 clf off)
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

### 4.2 흔들 11축

**baseline.ipynb의 실제 변수와 1:1 매핑**. xlsx 6축(clf_output·clf_filter는 CLF에 통합) + 추가 5축.

> **제외 결정**: `clf_level` (분류를 die-level vs unit-level) — 1차 실험에서 **집계 후 분류 성능이 안 좋았던 이력**이 있어 축 자체 제거. baseline은 die-level 분류만 사용 (baseline.ipynb 동작 그대로).

| # | 축 | 옵션 | baseline.ipynb 위치 | 비고 |
|---|---|---|---|---|
| 1 | `CLF` | `off` / `proba` / `proba+filter` / `binary` | `pipeline_config.run_clf` × `clf_output` × `clf_filter` | xlsx 그대로 |
| 2 | `reg_level` | `unit` / `position` | `pipeline_config.reg_level` | xlsx 그대로 |
| 3 | `TARGET_TRANSFORM` | `none` / `log1p` / `yeo-johnson` | `TARGET_TRANSFORM` | xlsx 그대로. sqrt는 코드 미지원이라 제외 |
| 4 | `CLIP_Y_EXTREME` | `True` / `False` | `CLIP_Y_EXTREME` | xlsx 그대로 |
| 5 | `loss` | `regression` / `poisson` / `tweedie` / `huber` | `e2e_params.reg_fixed = {'objective': ...}` | LGBM objective. `regression` = mse. **`tweedie` 시 `TARGET_TRANSFORM` 자동 OFF** ([02_reg_single/strategy.md §4](../02_reg_single/strategy.md), EXPERIMENT_LOG §5.1 — Tweedie + log1p 충돌 방지). poisson은 ON 유지 |
| 6 | `impute` | `spatial` / `median` / `knn` | `PP_CLEAN_CANDIDATES.imputation_method` | baseline.ipynb 옵션 그대로 |
| 7 | `outlier` | `none` / `winsorize` / `iqr_clip` / `grubbs` / `lot_local` | `PP_OUTLIER_CANDIDATES.method` | `none` 분기 [outlier.py:449](../../2_preprocessing/outlier.py#L449)에 이미 구현됨 |
| 8 | `agg_preset` | **12종 (§4.3)** | `AGG_PRESETS` | 확장 — `reg_level='unit'`일 때만 의미 |
| 9 | `binarize` | `True` / `False` | `PP_BINARIZE_CANDIDATES.apply` | baseline.ipynb 그대로 |
| 10 | `iso_anomaly` | `True` / `False` | `PP_ISO_ANOMALY_CANDIDATES.iso_enabled` | baseline.ipynb 그대로 |
| 11 | `lds` | `True` / `False` | `PP_LDS_CANDIDATES.lds_enabled` | baseline.ipynb 그대로 (Label Distribution Smoothing — zero-inflation 대응) |

**제외한 축** (코드 미지원, 효과 미미, 또는 명시 제외):
- `clf_level` — 1차 실험에서 집계 후 분류 성능 나쁨 → 제거
- `scaling` — `HybridScaler`가 컬럼별 자동 분기, 외부에서 강제 못 흔듦
- `zero_clip` — 사용자 지시로 제외
- `path` (회귀) — `reg_level=position` vs `unit`이 사실상 동일 역할
- `run_fs` / `clf_optuna` / `reg_optuna` — OAT는 default 고정이 원칙이라 무의미

### 4.3 agg_preset 확장 — 12종 ★

baseline.ipynb 4종 + 추가 8종. die→unit 집계 함수 조합 다양성 확보.

**코드 수정 완료**: [_modules/aggregate.py](_modules/aggregate.py) `aggregate_die_to_unit`에 `Q25, Q75` 분기 추가됨. native 지원 함수 = `mean / std / cv / range / min / max / median / Q25 / Q75` 9종.

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
    'P11_full9_cv':  ['mean', 'std', 'range', 'min', 'max', 'median', 'Q25', 'Q75', 'cv'],
}
```

총 **12종**. reference = `P00_full6`.

### 4.4 Cell 수 계산

각 축에서 reference와 다른 옵션만 변형 cell:

| 축 | 옵션 수 | 변형 cell |
|---|---|---|
| CLF | 4 | 3 |
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
| **합계** | | **30 변형** + 1 reference = **31 cell** |

× 5 seed = **155 cell-seed**
× 5 fold = **775 model fit**

LGBM default + position pivot 기준 fit당 4~6분 가정 → **52~78시간 ≈ 2~3.5일** (단일 슬롯).

### 4.5 출력 — master.csv

row 단위 영구 박제. 발표 시 csv 로딩 → tornado plot 즉석 생성.

**한 row = 한 (axis, option, seed) 셀의 5-fold 평균 결과**. fold별 분리는 안 함 (변동성은 seed 5개로 측정).

| 컬럼 | 예시 | 비고 |
|---|---|---|
| `axis` | `TARGET_TRANSFORM` | reference 셀은 `'reference'` |
| `option` | `log1p` | 그 축의 옵션 값 |
| `seed` | 42 | 5개 중 1개 |
| `is_reference` | False | reference 셀 식별 |
| `oof_rmse` | 0.008275 | 5-fold OOF RMSE |
| `val_rmse` | 0.005442 | 5-fold val 평균 RMSE |
| `test_rmse` | 0.008446 | 5-fold test 평균 RMSE |
| `elapsed_sec` | 187.3 | |
| `effective_target_transform` | `none` | ★ 학습에 실제 적용된 변환. cfg와 다를 수 있음 (loss=tweedie 시 자동 'none' override — [02_reg_single/strategy.md §4](../02_reg_single/strategy.md) 룰) |
| `timestamp` | 2026-05-08T14:23:01 | |
| `cfg_CLF` | `off` | ★ 셀 cfg 11축 풀어 저장 (재현/복원용) |
| `cfg_reg_level` | `position` | |
| `cfg_TARGET_TRANSFORM` | `log1p` | (이 row의 axis와 일치) |
| ... (총 11개 cfg_* 컬럼) | | |

`is_reference=True`인 row가 5 seed = **5개** (cfg는 모두 REFERENCE). 변형 cell × 5 seed = 150 row + reference 5 = **155 row**.

### 4.6 메타 산출물 — meta.json (재현성)

`oat/meta.json`, `group/meta.json` 각 디렉토리에 1개. run마다 덮어쓰기. 내용:
- 실행 설정: `n_jobs`, `n_estimators`, `n_trials` (group)
- axis 정의: `reference`, `axes`, `seeds`, `agg_preset_lib`
- PP pin 값: `pp_pin.{cleaning, outlier, binarize, iso, lds, ge}`
- `exclude_cols` (54개)
- `master_cols` (oat) / `study_name`, `sampler`, `pruner` (group)

### 4.7 Checkpoint — 끊겨도 이어서

`checkpoint.json`에 완료한 (axis, option, seed) 튜플 기록. 재실행 시 master.csv에 이미 있으면 skip.

```python
done = set(zip(master.axis, master.option, master.seed))
for cell in oat_grid:
    if cell.tuple in done: continue
    ...
```

---

## 5. Group Study 설계 (02_group_study.ipynb)

### 5.1 Search space — OAT의 11축 그대로

axes.py의 `AXES` dict를 그대로 Optuna `suggest_categorical`로 변환:

```python
def objective(trial):
    cfg = {
        axis: trial.suggest_categorical(axis, options)
        for axis, options in AXES.items()
    }
    return run_one(cfg, seed=42)   # 5-fold mean RMSE
```

LGBM HP는 default 고정. **축 11개만 흔듦** (HP 같이 흔들면 축 효과와 섞여 발표 일관성 깨짐).

### 5.2 Sampler

[strategy_common.md §4](../strategy_common.md):

```python
TPESampler(
    seed=None,            # 다양성
    multivariate=True,    # ★ 축들의 결합 분포 학습 = 상호작용
    group=True,           # ★ 자연 블록 (예: CLF=off면 clf_output/clf_filter 무관)
)
```

`group=True`로 `CLF=off`일 때 clf 관련 sub-옵션 미샘플 등 conditional 처리.

### 5.3 Pruner

LGBM default fit + 5-fold라 fold별 중간값 잘 안 나옴. **MedianPruner는 비활성화** (5 fold 다 끝나야 점수 나옴).

### 5.4 Trial 예산

- 11축 카르테시안 = 4×2×3×2×4×3×5×12×2×2×2 ≈ **276,480 조합**
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
# 3_modeling/0_baseline/axes.py

REFERENCE = { ... }     # §4.1
AGG_PRESET_LIB = { ... } # §4.3

AXES = {
    'CLF':              ['off', 'proba', 'proba+filter', 'binary'],
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
    2) baseline.ipynb의 rerun_best_trial_with_pp() 호출 (best_params=defaults, n_folds=5)
    3) OOF + val + test RMSE 반환
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
"우리는 베이스라인 단계에서 11개 축의 효과를 측정했다.

1) OAT (편향 없는 marginal):
   - 31 cell × 5 seed = 155 fit
   - tornado plot: TARGET_TRANSFORM이 가장 큰 축 (Δ 0.0044)
   - scaling은 거의 무영향 (트리 모델이라 예상대로)

2) Group study (상호작용 포함):
   - 300 Optuna trial, multivariate TPE
   - best 조합: target=yeo-johnson + clip=True + clf=off + ...
   - param importance ranking이 OAT tornado와 일치
   → 상호작용 효과 작음 = OAT marginal이 충분 신뢰 가능

3) 결론: 11개 축 중 X개가 RMSE 좌우.
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

## 11. 검증 근거

- **편향 우려**: TPE는 좋은 영역에 trial 몰아주므로 한 study 다축 학습 시 조합 간 trial 수 불균형 → 비교 불가. OAT는 reference 1점에서 1:1 비교라 편향 없음
- **OAT 한계 (상호작용)**: xlsx 데이터에서 CLIP 효과가 TARGET에 따라 큰 차이 (yeo-johnson + CLIP=F: Δ +0.00001 vs none + CLIP=F: Δ +0.00273) → OAT만으로는 못 잡음 → Group study로 보완
- **공정 비교 핵심**: 같은 5-fold split + 같은 PP pin + 같은 LGBM default HP로만 축 효과 분리 가능. axes.py의 `run_one()`이 단일 entrypoint
- **재현성**: master.csv에 모든 fit row 박제 → 발표 자료 png는 매번 csv에서 즉석 생성. 다음에 축 추가해도 csv 합치면 됨
- **baseline.ipynb 정합**: 모든 축이 baseline.ipynb의 실제 변수(`pipeline_config`, `PP_*_CANDIDATES`, `AGG_PRESETS`, `TARGET_TRANSFORM`, `CLIP_Y_EXTREME`)와 1:1 매핑. 코드 수정은 `_modules/aggregate.py`의 Q25/Q75 분기 추가뿐 (e2e_hpo.py는 손 안 댐)

---

## 12. 실행 순서

1. axes.py 작성 (`run_one()` 포함)
2. 02_group_study.ipynb 시작 (코어 7) — 먼저 (시간 김)
3. 01_oat.ipynb 시작 (코어 7) — 동시
4. 둘 다 끝나면 03_summary.ipynb 1회 실행 → 발표 자료
