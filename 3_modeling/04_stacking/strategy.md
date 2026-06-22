# 04 — Stacking (die-level)

> base 모델들의 **die-level 예측(`*_die.csv`)** 을 메타 학습해 unit RMSE를 최소화한다.
> 노트북 `stacking.ipynb` + 패키지 `stacking_lib/`. (구 unit-level `stacking.ipynb` + squeeze v2/v3 계열은
> die-level로 단일화하며 폐기 — 버전 표기 제거.)

## 1. 구성

| 파일 | 역할 |
|------|------|
| `stacking.ipynb` | 파이프라인 실행 노트북 (config → discover → bundle → search → refit → save) |
| `stacking_lib/` | 패키지: `config / discovery / shap / meta / weights / search / aggregate / io / records` |
| `build_shap_features.py` | base 1개의 die-level SHAP(`die_shap.npz`)를 생성 (`--base-rel`, `--zit-sub-model`, `--ts-sub-model`) |
| `run_shap_all.py` | 전 base × 컴포넌트 배치로 `build_shap_features.py` 호출 (TASKS 목록) |

## 2. 입력 계약 (producer ↔ stacking)

`discovery.discover_models(cfg)`가 `4_output`을 **`rglob("oof_die.csv")`** 로 재귀 스캔해, die CSV 3종
(`oof_die.csv` / `val_die.csv` / `test_die.csv`)을 **모두 가진 폴더만** base 후보로 채택한다.
경로를 조립하지 않고 *스캔*하므로, fit 노트북이 §5.1 경로에 번들만 떨구면 자동으로 잡힌다.

- **모델 태그** = 폴더 상대경로의 `"__".join` (예: `02_reg_single__lgbm`, `03_two_stage__default__clf__lgbm`).
- **매칭 키** = `(ufs_serial, run_wf_xy)` — die 단위 unique. 한 행이라도 누락이면 `build_die_matrix`가 RuntimeError.
- **clf die CSV는 `prob`(0~1)** → `discovery`가 `prob × y_pos_const`(=E[Y|Y>0], `best_params.json`에 박제)로
  health 스케일 변환. `y_pos_const` 없는 clf 폴더는 자동 제외.
- **category 자동 분류**: `clf` / `zit`(01_zit) / `reg_single`(02_reg_single) / `reverse` / `combined` / `ts_reg`(reg).
  `cfg.include_combined` / `cfg.include_ts_reg` / `cfg.no_clf`로 포함 여부 토글.

## 3. 메타 학습 (die-level)

- **GroupKFold by `ufs_serial`** — 같은 unit의 4 die가 train/val에 섞이면 leakage라 절대 금지.
- 메타 학습기 5종: `ridge` / `nnls` / `mean` / `ENet`(ElasticNetCV) / `Combo`(Bag+ENet+NNLS) + `iso` 후처리(옵션).
- 메타 raw 예측 → `np.clip(0, None)` → **die→unit 집계**(`aggregate` = `postprocess.tune_and_apply`):
  `mean/median/max/min/trimmed_mean/weighted/Q25/Q75` 중 train OOF로 1등 선택, weighted는 position 가중 최적화.
- 선택 기준 `cfg.select_by` ∈ {`oof`, `val`, `meta_cv_oof`}. `meta_cv_oof`는 GroupKFold OOF로 과적합을 덜 본다.

## 4. SHAP X-stacking (옵션)

base 예측 행렬에 **die-level SHAP 컬럼**을 덧붙여 메타 입력을 확장한다.

1. `run_shap_all.py` 실행 → 각 base의 `shap_cache/<tag>/die_shap.npz` 생성
   (npz에 `oof/val/test_run_wf_xy` 동봉 → `stacking_lib.shap`이 `(ufs_serial, run_wf_xy)` 키로 정렬 일치).
2. `cfg.shap_caches`에 캐시 폴더 리스트 지정. `shap_top_k`(캐시당 상위 K feature), `shap_mode`
   (`always_include` 항상 입력 / `searchable` subset 후보 등록), `shap_prefix_with_tag`(컬럼명 충돌 방지).
3. 캐시가 없는 깨끗한 repo에선 `shap_caches=()`(비활성) — 노트북 셀 11에 새 §5.1 태그 예시를 주석으로 둠.

> SHAP 추출 제외: `enet`(선형, pred_contrib 미지원), `03_two_stage/default/reg`(Y>0 서브셋만 → OOF sparse).
> ZIT는 내부 `lgb_pi_`/`lgb_mu_` 둘 다 `--zit-sub-model {pi,mu}`로 별도 추출.

## 5. config 주요 파라미터 (`StackingConfig`)

| 파라미터 | 기본 | 의미 |
|----------|------|------|
| `oof_rmse_cutoff` | `0.006` | 후보 base의 oof_rmse 상한(안전 필터). plateau(~0.0057) 위로 완화 — run 후 실측 보고 조정 |
| `min/max_subset_size` | `2 / 18` | 메타에 들어가는 base 개수 범위 |
| `random_trials / local_seeds·steps / top_refit / combo_refit / optuna_trials` | — | 탐색 단계별 예산 |
| `select_by` | `oof` | 탐색·선정 기준 metric |
| `agg_methods / baseline_agg / position_method` | — | die→unit 집계 후보·fallback·position 최적화 |
| `use_iso` | `True` | 메타 출력 IsotonicRegression 후처리 |
| `shap_caches / shap_top_k / shap_mode / shap_prefix_with_tag` | — | SHAP X-stacking |
| `known_strong_subset` | `()` | 강제 후보 base 태그(있을 때만). 앵커 해제로 비움 |
| `output_subdir` | `04_stacking` | 결과 위치 `4_output/04_stacking/run_{ts}/` (버전 표기 없음) |

## 6. 실행 순서

1. `01_zit` / `02_reg_single` / `03_two_stage`(clf·reg·combine·reverse) fit 노트북으로 **base die CSV 번들** 생성.
2. (옵션) `python 3_modeling/04_stacking/run_shap_all.py` → `shap_cache/<tag>/die_shap.npz`.
3. `stacking.ipynb` 실행: config(셀 5) → `discover_models`(셀 7) → (옵션 SHAP 셀 11) →
   `build_array_bundle`(셀 13) → `run_search_stages`(셀 18) → `run_refit_stage`(셀 20) → `io.save_outputs`(셀 22).
4. 결과: `4_output/04_stacking/run_{ts}/` (`summary.csv`, `best_weights.json`, 메타 산출물).

## 7. 산출물 (`run_{ts}/`)

- `summary.csv` — 전 record(fast/refit)의 die/unit/val/test RMSE + 집계 방식 + tag.
- `best_weights.json` — 최종 선정 메타의 base/extra 구성 + 학습기 가중치 박제.
- `summary.json` — config + SHAP 메타 + 선정 결과.

## 8. 대용량 바이너리 (git 제외)

- `shap_cache/*/die_shap.npz` (SHAP 입력) · `_prev/*.log` (run 로그) → `.gitignore` 등록(추적 제외).
  코드는 그대로 동작하되 repo엔 올리지 않는다.
