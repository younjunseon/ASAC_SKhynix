# 04 — Stacking

> 1차 `05_stacking_11base.ipynb` + `05b_stacking_full_with_nn.ipynb`를 **노트북 1개로 통합** (`stacking.ipynb`). POOL_PATHS만 재매핑하면 11base/full 둘 다 자동 탐지로 커버.

## 1. 노트북

| 노트북 | 1차 원본 | 비고 |
|---|---|---|
| `stacking.ipynb` | `05_stacking_11base.ipynb` + `05b_stacking_full_with_nn.ipynb` 통합 | POOL_PATHS 자동 탐지 + 옵션 토글 (A·D) + Optuna 통합 모드 (A_OPT) |

## 2. 메타러너 (1차 그대로)

`Pipeline(StandardScaler → ElasticNetCV)`
- `alpha`: `np.logspace(-6, 0, 30)` 그리드
- `l1_ratio`: `[0.1, 0.3, 0.5, 0.7, 0.9, 1.0]` 그리드
- `KFold(n_splits=5, shuffle=True, random_state=SEED)` CV
- `positive=False` (음수 weight 허용 — 일부 base가 corrector 역할)
- `max_iter=20000`
- 예측 후 `np.clip(0, None)` (health ≥ 0)

비교 baseline: SLSQP blending (`Σw=1, w≥0`)

## 3. 옵션

### 3.1 옵션 A — 핵심 피처 추가 (단순 토글, `USE_EXTRA`)

| 변수 | 기본 | 설명 |
|---|---|---|
| `USE_EXTRA` | `False` | True 켜면 importance 상위 K 피처를 메타 입력에 합침 |
| `EXTRA_K` | `20` | 상위 K 피처 |
| `EXTRA_AGG` | `['mean', 'std']` | die→unit 집계 함수 list |
| `IMPORTANCE_SOURCE` | `4_output/02_reg_single/lgbm` | `fold_models.pkl` 가진 안정적인 트리 base (1개) |

**메커니즘**: 1개 source → fold importance 평균 → top K → die→unit 집계 → P_oof/val/test에 join.

**한계**: 단일 모델 의존, gain importance 함정. 본격 실험은 §3.2 옵션 A_OPT 참조.

### 3.2 옵션 A_OPT — Optuna 통합 importance (본격 실험)

A안의 한계(단일 base 편향, K/agg 임의 고정)를 Optuna로 동시 탐색해 통합 importance 운영.

#### 탐색 축

| 축 | 후보 | 설명 |
|---|---|---|
| `w_shap` | `float [0, 1]` | gain vs SHAP 가중치 (`w_gain = 1 - w_shap`) |
| `shap_lgbm`, `shap_xgb`, `shap_cb` | `float [0, 1]` 각각 | 3 base SHAP 혼합 비율 (정규화) |
| `K` | `int [10, 100] step 10` | top-K 피처 수 |
| `agg_{name}` | `bool` 7개 (`mean / std / max / min / range / Q25 / Q75`) | die→unit 집계 binary 다중선택 |

#### 사전 신호 (gain × 3 + SHAP × 3)

- **gain importance** (3 base): `02_reg_single/{lgbm, xgb, catboost}`의 `fold_models.pkl`에서 fold별 평균 → 컬럼 정규화 (`/sum`) → 3개 평균 = `gain_score`
- **SHAP importance** (3 base): TreeExplainer로 fold별 `|shap|` 평균 → 정규화 → csv 캐시. **사전 1회 계산 필수** (§3.4 참조)

#### Trial별 통합

```
shap_score = (s_lgbm·shap_lgbm + s_xgb·shap_xgb + s_cb·shap_cb) / Σs
final_imp  = w_gain·gain_score + w_shap·shap_score
top_feats  = final_imp.nlargest(K).index
extra_unit = die→unit aggregate(top_feats, aggs)
```

→ `P_oof_full = P_oof_base.join(extra_unit)` → ElasticNetCV → val RMSE 반환.

#### Sampler/Pruner

- `TPESampler(seed=None, multivariate=True, group=True)` — strategy_common §4
- `MedianPruner(n_warmup_steps=10)` — strategy_common §4
- `n_trials=100~200`, `timeout=None`

### 3.3 옵션 D — Zero clip 후처리 (`USE_ZERO_CLIP`)

| 변수 | 기본 | 설명 |
|---|---|---|
| `USE_ZERO_CLIP` | `False` | True 켜면 zero clip 탐색 + val 검증 |
| `ZERO_CLIP_GRID` | `np.arange(0.001, 0.016, 0.001)` | 후보 th 1e-3 ~ 1.5e-2 step 1e-3 |

**메커니즘** (strategy_common §12 원칙 따름):
1. train OOF에서 best th 탐색 (RMSE 최소)
2. val 적용 후 비교
3. **val 개선되면 채택** (`stack_oof / stack_val / stack_test` 덮어씀)
4. **개선 없거나 악화 시 미적용** (원본 stack 유지)

채택 결과는 meta.json `options.zero_clip_applied`, `zero_clip_best_th`, `rmse_pre_zero_clip`에 기록.

### 3.4 SHAP 캐시 사전 준비 (옵션 A_OPT 전용)

옵션 A_OPT 사용 전 **SHAP 캐시를 stacking 노트북 안에서 1회 생성**한다. 04_stacking 책임 범위로 묶어 base 노트북(02_reg_single)은 SHAP을 모르도록 분리.

#### 위치

```
4_output/04_stacking/_cache/
  shap_lgbm.csv      # columns: feature, shap_score
  shap_xgb.csv
  shap_catboost.csv
```

#### 생성 로직 (cell-shap-cache, cell-extra 직전 신규 셀)

```python
SHAP_SOURCES = {
    'lgbm':     '02_reg_single/lgbm',
    'xgb':      '02_reg_single/xgb',
    'catboost': '02_reg_single/catboost',
}
SHAP_CACHE_DIR = '4_output/04_stacking/_cache/'
os.makedirs(SHAP_CACHE_DIR, exist_ok=True)

for name, src in SHAP_SOURCES.items():
    cache_path = os.path.join(SHAP_CACHE_DIR, f'shap_{name}.csv')
    if os.path.exists(cache_path):
        continue   # 캐시 있으면 skip
    # fold_models.pkl 로드 → TreeExplainer → fold별 |shap| 평균 → 정규화 → csv
    ...
```

#### 비용

- 첫 실행: **30~60분** (lgbm + xgb + catboost 각 fold 5개 × 26K row)
- 두 번째 이후: **0초** (csv 로드)
- 1차 백업으로 검증 시: `IMPORTANCE_SOURCE`만 `모델링_이전자료/4_output_이전자료/final/reg_only/{model}` 임시 변경

#### base 추가/변경 시

- 02_reg_single 모델 재학습 → 해당 csv 수동 삭제 → stacking 다음 실행 시 자동 재계산
- 또는 base 모델 변경 없이 stacking만 재실험 → 캐시 그대로 재사용 (비용 0)

### 3.5 Segment 분해 진단 (cell-segment, cell-summary 직전 신규 셀)

전체 val RMSE는 zero-inflated 환경에서 noise를 가림. **Y=0 / Y>0 분해 출력 필수** (사용자 메모리 정책 — `평가 기준 정책`).

```python
seg_y0   = (y_val == 0)
seg_ypos = (y_val > 0)
print(f'Y=0 RMSE: {_rmse(stack_val[seg_y0],   y_val[seg_y0]):.6f}  (n={seg_y0.sum()})')
print(f'Y>0 RMSE: {_rmse(stack_val[seg_ypos], y_val[seg_ypos]):.6f}  (n={seg_ypos.sum()})')
```

전체 val=0.005701 plateau여도 **Y>0에서 0.001 개선**이면 진짜 효과. 반대면 trade-off → 가중치/source 재조정 신호.

meta.json `stacking.segment_rmse`에 `{y0, ypos}` 별 RMSE 기록.

### 3.6 OUT_DIR 옵션 태그

옵션 조합으로 디렉토리 자동 분리 → 같은 노트북 여러 번 돌려 비교 가능:

| `USE_EXTRA` | `USE_OPTUNA_EXTRA` | `USE_ZERO_CLIP` | `EXP_TAG` |
|---|---|---|---|
| F | F | F | `base` |
| T | F | F | `extra_K20_mean_std` |
| F | T | F | `optuna_extra` |
| F | F | T | `zclip` |
| T | F | T | `extra_K20_mean_std_zclip` |
| F | T | T | `optuna_extra_zclip` |

`USE_OPTUNA_EXTRA=True` 시 `USE_EXTRA`는 자동 OFF (Optuna가 K/agg/SHAP 모두 탐색).

## 4. 신규 환경 호환

### 4.1 환경 셋업 셀
- `try/except google.colab` 분기 + `GDRIVE_CODE_ID` / `GDRIVE_DATASET_ID` (이미 신규 패턴)
- 로컬: `%run ../setup.py`

### 4.2 POOL_PATHS

미러링 정책(strategy_common §17) 따라 신규 경로로 매핑. 자동 탐지 로직은 그대로 — POOL_PATHS에 등록된 base 중 산출물 있는 것만 사용, 누락 모델 자동 제외.

| 풀 | 경로 |
|---|---|
| `zit_only` | `01_zit/zit_only/` |
| `bag_zit` | `01_zit/bag_zit/` |
| `reg__{model}` | `02_reg_single/{model}/` (lgbm/xgb/catboost/et/enet) |
| `grid__{clf}_x_{reg}` | `03_two_stage/default/combined/{clf}_x_{reg}/` |

### 4.3 출력 경로 — strategy_common §17 따름

신규 정책: 모델링 폴더와 동일한 깊이로 미러링, `final` 한 단계 제거.

→ 본 노트북 OUT_DIR: `4_output/04_stacking/{exp_tag}/`

## 5. 실행 순서

### 5.1 첫 실행 (베이스라인)
1. 01_zit, 02_reg_single, 03_two_stage 모든 base 산출물 완료
2. **POOL_PATHS** 신규 경로로 재매핑 (cell-pool)
3. **base 1회 (옵션 모두 OFF, USE_EXTRA=False, USE_OPTUNA_EXTRA=False, USE_ZERO_CLIP=False)** → 기준선 확보

### 5.2 옵션 A_OPT 본격 실험 (Optuna 통합)
1. `USE_OPTUNA_EXTRA = True`, `OPTUNA_N_TRIALS = 100` 또는 `200`
2. 첫 실행 시 SHAP 캐시 자동 생성 (30~60분, §3.4)
3. Optuna trial (50분 ~ 2h, §3.2)
4. Best trial 재학습 → segment 분해 진단 (§3.5)
5. `4_output/04_stacking/optuna_extra/` 결과 + base 비교

### 5.3 D안 + 조합
- A_OPT의 best 결과에 `USE_ZERO_CLIP=True` 얹어 zero clip 효과 검증
- 결과 폴더 자동 분리 (`optuna_extra_zclip/`)

## 6. 산출물 (옵션별 디렉토리)

각 `4_output/04_stacking/{exp_tag}/` 안에:
- `oof_unit_stack.csv` / `val_unit_stack.csv` / `test_unit_stack.csv` (ElasticNetCV)
- `oof_unit_blend.csv` / `val_unit_blend.csv` / `test_unit_blend.csv` (SLSQP baseline)
- `single_base_rmse.csv` / `residual_corr_oof.csv` / `residual_corr_test.csv` / `comparison.csv`
- `meta.json` (`options`, `meta_learner`, `blending_slsqp`, `stacking`, `delta_vs_11base`, `segment_rmse`)
- A_OPT 모드 추가: `optuna_*.db`, `weight_K_agg_grid.csv`, `extra_importance.csv`, `best_extra_config.json`

영구 캐시 (옵션별 공유):
- `4_output/04_stacking/_cache/shap_{lgbm,xgb,catboost}.csv` (한 번 만들면 재사용)
