# E2E LGBM-only + 전처리 통합 HPO — 수정 전략

> **목표**: 전처리·집계·CLF·REG의 모든 파라미터를 단일 Optuna study 안에서 동시 탐색.
> 1 trial = (전처리 파라미터 → cleaning/outlier → die→unit 집계 → CLF OOF → FS → REG OOF → val RMSE)

---

## 1. 변경 / 유지 매트릭스

| 항목 | 변경 여부 | 비고 |
|------|:---:|------|
| **모델 종류** | 변경 | LGBM only (CLF·REG 모두) — XGB/CatBoost/RF 코드 경로 제거 |
| **앙상블 섹션** | 삭제 | `ensemble_*` 셀, `ensemble_candidates`, `ensemble_config` 전부 제거 |
| **전처리 (cleaning + outlier)** | 변경 | 셀 4에서 1회 실행하던 것 → Optuna trial 내부로 이동 (캐싱 적용) |
| **die → unit 집계 로직** | 유지 | `aggregate_die_to_unit`, `agg_funcs` 그대로 |
| **CLF / FS / REG 구조** | 유지 | `_run_clf_oof`, `select_top_k`, `_run_reg_oof` 그대로 |
| **pipeline_config 스위치** | 유지 | `input_level`, `run_clf`, `clf_output`, `clf_filter`, `run_fs`, `reg_level` 등 그대로 동작 |
| **rerun_best_trial** | 유지 + 일부 수정 | 베스트 전처리 파라미터를 받아서 rerun 시점에 cleaning/outlier 재실행 필요 |
| **실험 로깅 (`log_experiment`)** | 유지 + 확장 | `model_params`에 `best_preprocessing_params` 추가 |
| **EVAL_TEST 스위치** | 유지 | test peeking 방지 동일 |
| **EXP_ID 입력 / xlsx·json gdrive sync** | 유지 | 동일 |

---

## 2. 파일 작업

### 2-1. 백업
```
3_modeling/baseline.ipynb       → 3_modeling/_backup/baseline_old.ipynb
3_modeling/e2e_twostage.ipynb   → 3_modeling/_backup/e2e_twostage_old.ipynb (선택)
```
> `_backup/` 폴더 신규 생성. `e2e_twostage.ipynb`도 같이 백업할지는 사용자가 정해줘.

### 2-2. 새 baseline
- 새 파일 이름 후보: `3_modeling/baseline.ipynb` (덮어쓰기)
  - 또는 `3_modeling/baseline_e2e_lgbm.ipynb` (별도 파일)
- **이게 새로운 baseline.** 향후 모든 실험은 여기서 fork.

### 2-3. 모듈 수정
- `3_modeling/modules/e2e_hpo.py` — objective 함수에 전처리 단계 추가
- `3_modeling/modules/search_space.py` — 전처리 search space 함수 추가 (`preprocessing_space`)
- `2_preprocessing/cleaning.py`, `outlier.py` — **수정 없음** (호출만 trial 안으로 이동)

---

## 3. Optuna Search Space 설계

> **range는 일단 합리적인 기본값으로 잡아둠. 사용자가 보고 빼거나 좁힐 예정.**
> 모든 search space는 `search_space.py`에 dict-of-list 형태로 모아두어 **노트북 셀에서 한눈에 보고 수정**할 수 있게 한다.

### 3-1. 전처리 search space (신규)

```python
# search_space.py (신규 추가)
def preprocessing_space(trial):
    """전처리 + 이상치 + 집계 search space"""
    return dict(
        # ── cleaning ──
        const_threshold     = trial.suggest_categorical("pp_const_threshold",
                                  [1e-8, 1e-6, 1e-4]),
        missing_threshold   = trial.suggest_categorical("pp_missing_threshold",
                                  [0.25, 0.50, 0.75, 0.90]),
        remove_duplicates   = trial.suggest_categorical("pp_remove_duplicates",
                                  [True, False]),
        corr_threshold      = trial.suggest_categorical("pp_corr_threshold",
                                  [0.90, 0.95, 0.99, None]),
        corr_keep_by        = trial.suggest_categorical("pp_corr_keep_by",
                                  ["std", "target_corr"]),
        add_indicator       = trial.suggest_categorical("pp_add_indicator",
                                  [True, False]),
        indicator_threshold = trial.suggest_categorical("pp_indicator_threshold",
                                  [0.01, 0.05, 0.10]),
        imputation_method   = trial.suggest_categorical("pp_imputation_method",
                                  ["median", "knn", "spatial"]),
        knn_neighbors       = trial.suggest_categorical("pp_knn_neighbors",
                                  [3, 5, 10]),
        spatial_max_dist    = trial.suggest_categorical("pp_spatial_max_dist",
                                  [1.0, 2.0, 3.0]),
        # ── outlier ──
        outlier_method      = trial.suggest_categorical("pp_outlier_method",
                                  ["winsorize", "iqr_clip", "grubbs", "lot_local", "none"]),
        outlier_lower_pct   = trial.suggest_categorical("pp_outlier_lower_pct",
                                  [0.0, 0.005, 0.01]),
        outlier_upper_pct   = trial.suggest_categorical("pp_outlier_upper_pct",
                                  [0.99, 0.995, 0.999, 1.0]),
        iqr_multiplier      = trial.suggest_categorical("pp_iqr_multiplier",
                                  [1.5, 3.0, 5.0]),
        # ── 집계 ──
        # agg_funcs는 list of list로 후보 정의 → trial에서 인덱스 선택
        agg_preset_idx      = trial.suggest_categorical("pp_agg_preset_idx",
                                  [0, 1, 2]),
    )

AGG_PRESETS = [
    ["mean", "std", "cv", "range", "min", "max", "median"],     # 0: 7종 (현 기본)
    ["mean", "std", "cv"],                                      # 1: 핵심 3종
    ["mean", "std", "cv", "range", "min", "max", "median",
     "q25", "q75", "skew", "kurt"],                             # 2: 풀 11종
]
```

> **핵심**: 모든 categorical을 list로 정의 → 사용자가 빼고 싶은 후보는 list에서 원소만 지우면 됨.
> log/uniform이 필요한 항목(예: spatial_max_dist를 연속으로 탐색)은 사용자 요청 시 변경.

### 3-2. CLF / REG search space (기존 유지)

`lgbm_space(trial, prefix="clf_" or "reg_")` 그대로. range는 기존 [search_space.py:10](modules/search_space.py#L10) 사용.

### 3-3. FS top_k

기존 동일. `fs_optuna=True`이면 `top_k_range=(50, 500)`, `False`이면 `top_k_fixed=200`.

---

## 4. 전처리 캐싱 전략 (성능 핵심)

> 전처리는 1회당 약 3분 (특히 spatial imputation). n_trials=15면 +45분, 50이면 +2.5시간.
> **캐싱 없이는 불가능.**

### 4-1. 캐시 키
```python
import hashlib, json
def _pp_hash(pp_params):
    """전처리 파라미터 dict → 고정 hash 키"""
    s = json.dumps(pp_params, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:12]
```

### 4-2. 캐시 자료구조
```python
PP_CACHE = {}  # 전역 dict (run_e2e_optimization 함수 내부)
# key: pp_hash
# value: dict(
#     pos_data_clean=...,        # cleaning + outlier 적용된 pos_data
#     feat_cols_clean=...,       # cleaning 후 feature list
# )
```

### 4-3. objective 흐름
```python
def objective(trial):
    # ── ⓪ 전처리 (캐싱) ──
    pp_params = preprocessing_space(trial)
    key = _pp_hash(pp_params)
    if key in PP_CACHE:
        cached_pp = PP_CACHE[key]
    else:
        # cleaning + outlier 실행
        xs_train_c, xs_val_c, xs_test_c, clean_cols, _ = run_cleaning(
            xs, feat_cols, xs_dict, ys_train=ys_train,
            **{k: v for k, v in pp_params.items() if k in CLEANING_KEYS},
        )
        xs_train_c, xs_val_c, xs_test_c, _ = run_outlier_treatment(
            xs_train_c, xs_val_c, xs_test_c, clean_cols,
            **{k: v for k, v in pp_params.items() if k in OUTLIER_KEYS},
        )
        # health merge + label + position split
        pos_data_c = _build_pos_data(xs_train_c, xs_val_c, xs_test_c, ys, ...)
        PP_CACHE[key] = dict(pos_data=pos_data_c, feat_cols=clean_cols, agg_funcs=AGG_PRESETS[pp_params['agg_preset_idx']])
        cached_pp = PP_CACHE[key]

    # ── 이하 기존 ① CLF → ② 집계 → ③ FS → ④ REG → val RMSE 동일 ──
    ...
```

- TPESampler가 후반엔 비슷한 조합을 반복 → 캐시 hit률 높아짐
- 메모리: pos_data 하나당 die-level DataFrame 4개 × 약 100MB → 캐시 N개면 ~400N MB. **상한 두는 LRU 권장** (예: 최근 10개)
- LRU는 `collections.OrderedDict` 또는 `functools.lru_cache` 패턴

### 4-4. rerun_best_trial 변경
- 베스트 trial의 pp_params를 추출해서 cleaning/outlier 1회 재실행
- 캐시는 study가 끝나면 폐기 (메모리 회수)
- 원래 cleaning/outlier 함수는 건드리지 않음 — 호출 위치만 이동

---

## 5. 코드 변경 위치 (요약)

### 5-1. `modules/search_space.py`
- `preprocessing_space()` 함수 추가
- `AGG_PRESETS` 모듈 상수 추가
- `CLEANING_KEYS`, `OUTLIER_KEYS` 상수 추가 (pp_params를 cleaning/outlier 함수 인자로 분배할 때 필요)

### 5-2. `modules/e2e_hpo.py`
- `run_e2e_optimization` 시그니처에 `xs`, `xs_dict`, `ys` (혹은 raw 데이터 진입점) 추가
  - 현재는 `pos_data`를 외부에서 받는 구조 — 전처리가 trial 안으로 들어오면 raw 데이터 필요
- objective 안에 ⓪ 전처리 단계 + 캐싱 로직 추가
- `_build_pos_data` 헬퍼 신규 (현 노트북 셀 5의 health merge + label + position split을 함수화)
- `rerun_best_trial`에도 동일 전처리 단계 추가
- `clf_model`, `reg_model` 인자는 남기되 기본값 `'lgbm'` 고정 (방어적)
- LGBM-only 가정에 따라 catboost/xgb/rf branch는 제거 가능 (선택)

### 5-3. `baseline.ipynb` (새 노트북)
- Cell 1: env + data load (기존 동일)
- **Cell 2** (신규 통합): 모든 설정 dict
  - `EXP_ID`, `EXP_TYPE`, `EVAL_TEST`
  - `pipeline_config` (기존)
  - `e2e_params` (n_trials, n_folds 등)
  - `rerun_params`
  - `preprocessing_search_space_overrides` ← 사용자가 range 좁힐 때 쓰는 dict (선택)
  - **앙상블 dict 전부 제거**
- Cell 3: ~~전처리~~ → 삭제 (Optuna 안으로 이동)
- Cell 4: ~~health merge~~ → 삭제 (Optuna 안으로 이동)
- Cell 5: `run_e2e_optimization(xs, xs_dict, ys, feat_cols, ...)` 호출
- Cell 6: `rerun_best_trial(...)` (best pp_params + best lgbm params)
- Cell 7: 결과 시각화 (trial RMSE 추이 + feature importance)
- ~~Cell 8~10~~: 앙상블 → 삭제
- Cell 8 (구 12): 최종 CSV 출력 (single E2E만)
- Cell 9 (구 13): `log_experiment` — `model_params`에 `best_preprocessing_params`, `preprocessing_search_space` 추가

### 5-4. `modules/__init__.py`
- 변경 없음 (있다면 새 함수 export 추가)

---

## 6. 로깅 변경 사항

`log_experiment` 호출 시 `model_params`에 다음 키 추가:
```python
model_params={
    'pipeline': 'E2E Joint HPO (preprocessing + lgbm)',
    'pipeline_config': pipeline_config,
    'e2e_params': e2e_params,
    'rerun_params': rerun_params,
    # ── 신규 ──
    'preprocessing_search_space': PREPROCESSING_SEARCH_SPACE_DUMP,  # 후보 list 그대로
    'best_preprocessing_params': best_pp_params,                    # 베스트 trial의 pp_*
    # ── 기존 ──
    'best_params': best_params,
    'best_val_rmse_hpo': result['best_value'],
    'best_val_rmse_rerun': final['val_rmse'],
    'final_method': 'e2e_single',
    ...
}
```

`feature_sel_params`, `cleaning_params`, `outlier_params`도 `log_experiment`의 인자로 들어가므로 → **베스트 trial의 값**으로 채워서 전달 (재현 가능성 보장).

---

## 7. 미해결 — Q5 (사용자 결정 필요)

### Q5. trial별 결과 저장 vs 베스트만 저장

| 옵션 | 장점 | 단점 |
|------|------|------|
| **(A) 베스트만 저장** | 디스크 절약, 명확 | trial별 분석 불가 (단, study.trials에 파라미터·val RMSE는 자동 저장됨) |
| **(B) trial별 전부 저장** | 모든 trial 재현 가능, 사후 분석 자유 | n_trials × (val.csv + test.csv) → 디스크 폭발. 50 trials면 100개 파일 |
| **(C) 절충: study pickle + best CSV** | study 객체 1개로 모든 trial 정보 보존 + best CSV만 출력 | trial별 예측 CSV 없음 (필요하면 study에서 best params 꺼내 재실행) |

**내 추천**: **(C) 절충**
- `4_output/{EXP_ID}_study.pkl` 1개 + `{EXP_ID}_final_val.csv`/`{EXP_ID}_final_test.csv` 2개만 저장
- study에는 trial별 params + val RMSE + user_attrs(train_rmse, n_features 등) 다 들어있음
- 사후 분석 필요 시 `optuna.load_study()` 또는 `pickle.load()`로 불러와 `study.trials_dataframe()` 호출

→ 결정해주면 그대로 반영. 다른 의견 있으면 말해줘.

---

## 8. 작업 순서 (확정 후 실행)

1. **백업**: `baseline.ipynb` → `_backup/`
2. **search_space.py 수정**: `preprocessing_space()` + 상수 추가
3. **e2e_hpo.py 수정**: 시그니처 + objective ⓪단계 + 캐싱 + rerun
4. **baseline.ipynb 신규 작성**: 위 5-3 구조
5. **smoke test**: n_trials=2, n_folds=2로 1회 실행 → 동작 검증
6. **로그 검증**: JSON에 best_preprocessing_params 정상 저장 확인

---

## 9. 추가 확인 필요한 미세 항목

작업 들어가기 직전에 확인할 디테일:

- **Q5-1**: `e2e_twostage.ipynb` 백업할지? (현재 노트북도 `_backup/`로 옮길지)
- **Q5-2**: 새 baseline의 파일명 `baseline.ipynb`로 덮어쓸지, `baseline_e2e_lgbm.ipynb` 별도 이름으로 둘지
- **Q5-3**: PP_CACHE LRU 상한 (default 10이면 충분?)
- **Q5-4**: `pipeline_config`의 `clf_filter`, `reg_level` 등 기본값은 현 e2e_twostage.ipynb와 동일하게(`reg_level='position'`, `run_fs=False`, `clf_filter=False`) 가져갈지
- **Q5-5**: `EXP_ID` prefix는 그대로 `3-2-xxx`?
