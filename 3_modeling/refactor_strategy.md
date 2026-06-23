# 모델링 실험 리팩토링 전략 (zit / reg / ts)

> GitHub 공개용 정리. 실험 잔재를 걷어내고 **모델별 1개 튜너(py) + fit(ipynb)** 구조로 통일한다.
> 공통 규칙은 [strategy_common.md](strategy_common.md) 참조. 본 문서는 리팩토링 작업 범위만 명시.
>
> **범위**: 리팩토링 주대상은 `01_zit` / `02_reg_single` / `03_two_stage`. 단 `0_baseline`·`modules/`도 **리팩토링 후 정상 작동**해야 한다 — 모델링만 고치는 게 아니라 전부 정상이어야 함. scaler 정리·모듈 삭제가 baseline·e2e를 깨지 않도록 함께 손본다.

---

## 0. 핵심 원칙

| # | 원칙 | 의미 |
|---|------|------|
| 1 | **py 튜닝 → ipynb fit** | Optuna 파라미터 탐색은 `hpo.py`(병렬 워커), 최종 fit/run은 `fit.ipynb`. 모델 폴더마다 한 쌍, 역할 분리 |
| 2 | **앵커 완전 해제** | `enqueue_trial(ANCHOR)` 제거 + `narrow_around`/`±50%` → wide search space. 이전 실험 편향 0 |
| 3 | **최종 = 진짜 최종** | `최종/` 폴더 개념 제거. 정식 fit 노트북이 곧 최종. 중복 zit는 `최종/` 버전으로 대체 |
| 4 | **죽는 코드 즉시 제거** | 대체되는 순간 옛 파일 삭제. 좀비/copy/orphan 모듈 잔존 금지 |
| 5 | **실험번호 제거** | `hp00X` / `*-final-00X` / `005t17` 등 → 깨끗한 의미 이름 (§5) |
| 6 | **폴더 3개 유지** | `01_zit` / `02_reg_single` / `03_two_stage` 구조는 유지. 병합은 **폴더 내부**에서만 |

> ⚠ §2 앵커 해제 = 기존 Optuna DB·`4_output` 산출물이 옛 exp-id로 묶여 있던 링크를 버리고 **새 study로 재실험**한다는 뜻. (어차피 앵커를 풀면 재탐색이므로 옛 DB는 폐기 대상)

---

## 1. 목표 구조 한눈에

```
3_modeling/
├── modules/                     # zit.py(ZIT 4종 + EMTboost 통합), zit_EMT.py(shim), hpo.py(refit/저장), models.py, postprocess.py, preprocess.py
│                                #  └ parallel_hpo.py 신설(전 트랙 공용 HPO 하네스 §1.1). scaler.py·pp_hpo.py·modules.zip 삭제(scaler→scaling.py 흡수)
├── 0_baseline/                  # (작동 보장 대상) 10모델 비교 + §4e EMT 흡수(§4d zit_gu 제거)
├── 01_zit/
│   ├── 00_precompute_pp.py      # (유지) CSV→pp.npy 사전계산 — 전 트랙 공용 (zit_pp 사용)
│   ├── zit_pp.py                # 신설: 공용 PP 로더 (cleaning.py 정본, median 패치 폐기) §2.1
│   ├── zit_fit_lib.py           # 신설: fit 4조합 공유 calibration/serialization 1차함수 (isotonic/tail grid 등) §2.6
│   ├── trial_to_json.py         # (유지) study trial → param json — 공용
│   ├── zit_only_pearson/        #   hpo.py(thin, parallel_hpo) + fit.ipynb
│   ├── zit_only_eql/            #   hpo.py + fit.ipynb   ← 튜너 신설
│   ├── bag_zit_pearson/         #   hpo.py + fit.ipynb
│   └── bag_zit_eql/             #   hpo.py + fit.ipynb   (단일 best — 4-param 앙상블 폐기)
├── 02_reg_single/
│   ├── hpo.py                   # --model {lgbm,xgb,catboost,et,enet}
│   └── fit.ipynb                # MODEL_NAME 스위치
├── 03_two_stage/
│   ├── default/
│   │   ├── clf/                 #   hpo.py  +  fit.ipynb   (--model {lgbm,xgb,catboost,et})
│   │   ├── reg/                 #   hpo.py  +  fit.ipynb   (--model {lgbm,xgb,catboost,et,enet})
│   │   └── combine.ipynb        # (유지) CLF 4 × REG 5 grid
│   └── reverse/ts_reverse.ipynb # (유지) Path B
└── 04_stacking/                 # (유지·rename) die-level 메타. stacking.ipynb + stacking_lib/ (§7)
```

*파일명은 제안값. 변경 가능.*

### 1.1 병렬 실행 모델 — `modules/parallel_hpo.py` (전 트랙 공용 하네스)

**왜 공용 하네스인가 (코드 실측 근거, 2026-06-17)**: zit/reg/ts(clf)의 기존 병렬 워커를 대조하니 **하네스(인자 파싱·`create_study`·mmap 로드·end-at 최적화 루프·`main`)와 unit-CV objective 골격(KFold-by-unit → fold 루프 → die→unit 집계 → unit RMSE → prune)이 사실상 byte 단위로 동일**했다. `PP_FIXED`·`load_preprocessed_data`(clip + `preprocess.run` + `add_meta_features(raw, die_xy)`)까지 동일. 트랙 간 차이는 fold 루프 안 **2가지뿐**:

| 트랙 | 타깃 준비 | 모델 평가(pred_die, health 스케일) | die→unit |
|------|----------|----------------------------------|---------|
| reg | `y_die` | `model.predict(x_vl)` | mean |
| zit_only | `y_die` | `(1-π)·μ` → `tau_pi` 게이트 | mean |
| bag | `y_die` | 위 + `fit(unit_id=)` | **sum** |
| clf | `y_die_bin=(y>0)` | `proba · y_pos_const(=E[Y\|Y>0])` | mean |
| ts/reg | `y_die` (Y>0 only fit) | `model.predict` | mean |

→ 250줄짜리 워커를 트랙·모델마다 복제하는 대신, **`modules/parallel_hpo.py`가 공용 보일러플레이트를 제공**하고 각 `hpo.py`는 위 2가지(+search space)만 둔다. objective 본문은 각 `hpo.py`에 **명시적으로** 남겨 callback 과추상화 없이 가독성 유지.

**`modules/parallel_hpo.py` 공개 API**:
- `add_common_args(parser)` — 16개 워커 인자 (exp-id/user/worker-id/n-trials/end-at/n-jobs/n-folds/n-startup/db-timeout/heartbeat/grace/max-retry/no-resume/no-clip-y-extreme/progress/precomputed-dir)
- `make_study(args, db_path, study_meta)` — RDBStorage + TPE(multivariate,group) + MedianPruner, `load_if_exists=not no_resume`. **앵커 enqueue 없음**(§0-2)
- `load_pp_mmap(precomputed_dir)` → `(x, uid_die, y_unit_s, y_die, feat)` — pp.npy mmap view
- `run_optimize(study, objective, args)` — end-at/timeout 환산 + `study.optimize`
- `resolve_out_dir(out_subdir, ...)` — §5.1 의미 폴더 경로

| 단계 | 동작 |
|------|------|
| **원천 입력** | **CSV** (`compet_xs_data.csv` 등) — 진실의 원천 |
| **사전계산** | `00_precompute_pp.py`(→`zit_pp.load_preprocessed_data`)가 CSV를 전처리해 `pp.npy`(float64 `(n_dies, F+2)`)로 1회 저장. 4트랙이 동일 PP → **pp.npy 1벌을 전 트랙이 mmap 공유**. npy = 빌드 캐시, git 추적 제외 |
| **워커** | N개 독립 프로세스가 `np.load(mmap_mode='r')`로 **행렬 1벌만 RAM 공유** + **1개 Optuna SQLite study 공유**(`--worker-id w1..wN`) |
| **스레드** | 프로세스당 BLAS/numpy 스레드 1로 캡 (오버서브스크립션 방지) |
| **역할 분리** | 워커는 Optuna DB 기록만. refit·후처리·산출물 저장은 `fit.ipynb`에서 별도 |

- **입력 정책**: "**입력의 진실은 CSV, `pp.npy`는 빌드 캐시**". 재현은 `python 00_precompute_pp.py` 한 줄. 레포에는 CSV/코드만, 이진 캐시는 비추적.
- **PP 통일**: clf의 `y_die_bin`·`y_pos_const`, ts의 Y>0 subset은 모두 같은 pp.npy(+units.npy)에서 파생 → 트랙별 별도 전처리 불필요.
- **권장 워커 수 = 3** (`3 × n_jobs ≤ 물리 스레드`). 구 mmap 헤더의 7워커는 과다 구독 → **3으로 표준화**.
- **enqueue-anchor 제거**: `--enqueue-anchor`(옛 best 주입) + `*_ANCHOR` dict 모두 앵커 해제 원칙(§0-2)에 따라 삭제.

> **변경 이력(2026-06-17)**: 본래 안은 트랙별 `hpo.py`를 각각 독립 mmap 워커로 두는 것이었으나, 4트랙 워커가 하네스+objective 골격까지 동일함을 코드 실측으로 확인 → **`modules/parallel_hpo.py` 단일 하네스 + 트랙별 thin `hpo.py`**로 통합. 파일당 중복 ~150줄 제거, 통일감·가독성 동시 확보. (reg/ts도 이 하네스를 쓰므로 §3·§4의 `hpo.py`도 동일 패턴.)

---

## 2. 01_zit — 4조합 완전 분리

`modules/zit.py`의 4개 클래스가 4조합에 1:1 대응한다.

각 조합 = 폴더 1개. 폴더 안 `hpo.py`(튜너) + `fit.ipynb`(fit)는 아래 source에서 유도.

| 조합 폴더 | 모델 클래스 | `hpo.py` ← source | `fit.ipynb` ← source |
|------|------------|------------------|--------------------|
| `zit_only_pearson/` | `ZITboostRegressor` | `01_zit_only_parallel_hpo.py` (앵커 해제) | `05_zit_only_hp003_seed_sweep.ipynb` |
| `zit_only_eql/` | `ZITboostEQLRegressor` | **신설** = pearson `hpo.py` 복제 + 클래스 교체 + 앵커 해제 | `08_zit_only_eql_hp003_seed_sweep.ipynb` |
| `bag_zit_pearson/` | `BagZITboostRegressor` | `02_bag_zit_mmap_parallel_hpo.py` (004, 앵커 해제) | `06_bag_zit_seed_sweep.ipynb` |
| `bag_zit_eql/` | `BagZITEQLRegressor` | `02_bag_zit_eql_mmap_parallel_hpo.py` (v3/4/5, 앵커 해제) | **`최종/06_bag_zit_eql_seed_sweep_final.ipynb`** → **단일 best로 단순화** (param1~4.json 4-세트 앙상블 폐기 — 앵커 해제와 모순) |

### 2.1 선행 완료 — `zit_pp.py` 공용 PP (median 패치 폐기, cleaning.py 정본) ✅

**실측 결과 (2026-06-17)**: 4조합 워커의 `load_preprocessed_data`는 모두 `modules.preprocess.run()`(=cleaning.py 래핑)을 호출하며 구조가 동일했다. 정본과의 **유일한 차이는 bag 계열 2개 워커**(`02_bag_zit_parallel_hpo.py`, `02_bag_zit_eql_parallel_hpo.py`)가 `cleaning.impute_spatial`을 **2·3단계 fallback만 median으로 바꾼 사본(`_impute_spatial_median`)으로 in-process monkeypatch**한 것뿐. zit_only는 패치 없이 이미 cleaning.py 정본을 썼고, mmap 워커는 PP 없이 `pp.npy`만 읽었다.

**결정** (사용자 지침 "모든 전처리는 현재 cleaning.py로 통일"): **median monkeypatch 폐기**. 1단계 공간보간은 원래 동일, fallback도 cleaning.py 정본(mean)으로 → zit_only ≡ bag 동일 PP. 앵커 해제로 재실험이므로 PP 수치 변화는 의도된 것. `clip_y_extreme` + `add_meta_features(raw, die_xy)`는 100% 보존.

→ **작업 완료**:
- `01_zit/zit_pp.py` 신설 — `PP_FIXED`(4조합 byte 동일) + `load_preprocessed_data(clip_y_extreme=True)`, cleaning.py 정본(패치 없음).
- `00_precompute_pp.py`가 구 워커 동적 import(`spec_from_file`)를 폐기하고 `zit_pp`를 import. `pp.npy` 레이아웃(`[features|uid_code|y_die]`) 불변 → mmap 워커 `load_precomputed_data` 호환.
- reg/ts 워커도 동일 PP라 `zit_pp`(또는 `parallel_hpo`)를 공용으로 쓴다.
- 비-mmap 워커 2개는 fit 노트북 이관(§2 fit) 후 삭제. fit 노트북의 `_impute_spatial_median` 동적 로드도 `zit_pp` import로 교체.

> **부수 참조 갱신**: `최종` fit 노트북이 `hp/trial_to_json.py` 옛 경로를 호출한다. `trial_to_json.py`를 `01_zit/` 루트로 옮긴 뒤 이 참조도 갱신.

### 2.2 eql 튜너 신설 (zit_only-eql)

- 현재 zit_only-eql은 **전용 HPO가 없음**. `08` 노트북이 pearson study(`zit-only-final-003`) best_params를 그대로 받아 모델만 `ZITboostEQLRegressor`로 교체해 seed-sweep만 했다.
- **작업**: `zit_only_pearson/hpo.py`를 복제 → `from modules.zit import ZITboostEQLRegressor`로 교체(거의 1줄, `__init__` 동일) → 앵커 해제 → 독립 study로 eql 고유 best HP 확보.
- 그 study best를 `zit_only_eql/fit.ipynb`이 로드.

### 2.3 앵커 해제 (zit)

- `01_zit_only_parallel_hpo.py`: `ZIT_ONLY_ANCHOR_002`/`_003` enqueue 제거, `narrow_around` → wide range.
- bag mmap 워커: `BAG_ZIT_*_ANCHOR_*` enqueue 제거, v3/v4/v5(EXPLOIT/PUSH/EXPLORE) 3변형 구조 → **wide 단일 search space**로 통일(EXPLORE v5의 넓은 범위를 기준 삼으면 손쉬움).
- fit 노트북의 study DB 참조 → 새 study 이름으로 갱신(§5).

### 2.4 zit 삭제 대상

| 파일 | 사유 |
|------|------|
| `01_zit_only.ipynb`, `02_bag_zit.ipynb` | 옛 HPO 노트북 → py 튜너로 대체 |
| `04_zit_only_hp002_seed_sweep.ipynb` | `05`(hp003)로 대체 |
| `06_bag_zit_eql_seed_sweep.ipynb` | `최종/...final.ipynb`로 대체(중복) |
| `06_bag_zit_eql_seed_sweep_parallel.py` (1184줄) | fit이 ipynb로 가므로 parallel .py sweep 중복 |
| `02_bag_zit_parallel_hpo.py`, `02_bag_zit_eql_parallel_hpo.py` | mmap이 대체 (§2.1 모듈화 후) |
| `03_zit_et_hpo.ipynb` | ZIT+ExtraTrees, 4조합 외 — **삭제 확정** |
| `*copy*.ipynb` 5개 | 좀비 |
| `최종/param{1,2,3,4}_*.json` | 4-세트 앙상블 폐기(앵커 해제) → 불필요 |
| `최종/` 폴더 | final 노트북을 `01_zit/bag_zit_eql/fit.ipynb`로 승격 후 폴더 제거 |

### 2.5 EMT — 베이스라인으로 이관 (완료)

EMT(EMTboost)는 4조합에 넣지 않고 **`0_baseline`에만 남긴다** (베이스라인 비교 모델). `01_zit`에 `emt/` 트랙 만들지 않음. `07_emtboost_hpo.ipynb`는 **앵커(`EMT_ANCHOR`)를 enqueue 하는 구(舊) 인라인 HPO 노트북**이라 앵커 해제 방침과 맞지 않는다 → HPO 노트북 자체는 폐기하고, **default 비교 1행으로만 흡수**한다.

| 파일 | 처리 |
|------|------|
| `01_zit/07_emtboost_hpo.ipynb` | `리펙토링대상_모델링_아웃풋/3_modeling/01_zit/`로 백업 이동 (구 01~08 실험 노트북과 동일 처리) |
| `modules/zit_EMT.py` | **zit.py로 병합**(2026-06-19, 사용자 요청) — EMTboost + `_estimate_phi_scalar`를 zit.py로 이관, 중복 3헬퍼(`_tweedie_p0`/`_tweedie_unit_deviance`/`_zitweedie_loglik`, 함수 동치 확인) 제거. `zit_EMT.py`는 하위호환 **shim**(`from modules.zit import EMTboost ...`)으로 유지 |
| `0_baseline/04_default_compare.ipynb` | EMT를 **§4e 섹션**으로 흡수 + **§4d `zit_gu` 섹션 제거**. import는 `from modules.zit import EMTboost` |

**흡수 방식 (04_default_compare.ipynb)** — "각 모델 자기 라이브러리 기본값" 철학 그대로:

- **§4e EMTboost default**: μ-LightGBM은 §4b ZIT의 `LGBM_DEFAULTS` 중 `mu_*`만 사용(라이브러리 기본값), `n_em_iters=N_EM_ITERS`. ζ는 §4b와 동일하게 **profile likelihood**(`ZETA_GRID` 중 train loglik 최대 `score_loglik`)로 결정 — 앵커/탐색 없음. die↔unit은 `zit_only` 방식(unit health→4 die broadcast 학습 → die 예측 `q·μ`를 **mean** 집계, scalar π라 `tau_pi` 게이트 없음). `zit_cache['emt_boost']` 저장 + `ZIT_ALL_SPECS`에 등록 → §5가 ZIT-family 한 줄로 함께 집계.
- **§4d `zit_gu` 제거**: `ZITGuInline` 인라인 클래스(`zit_Gu.py` 의존)는 사용자가 모듈을 수동 제거 → 셀(markdown+class+fit) 삭제. EMT 섹션이 그 자리(셀 id `emt4e_md`/`emt4e_fit`)를 대체.
- **결과**: `results.csv` = 회귀 5 + 투스테이지 25 + ZIT 17(4 base + 백엔드 믹스 12 + **EMTboost 1**) = **47행**. §4a(트리 15 refit)·§4b(ZIT 4종)·§4c(믹스 12종) 기능 **그대로 유지**.

### 2.6 zit fit 공유 처리 로직 모듈화 — `zit_fit_lib.py` (완료, 사용자 요청 2026-06-18)

4개 fit.ipynb가 **byte-identical로 복제**하던 calibration/serialization 1차 함수(isotonic+tail grid 등, cell8 344줄 + cell9 helper)를 `01_zit/zit_fit_lib.py` 한 곳으로 추출.

- **추출 대상(14함수)**: `tune_unit_postprocess_train_val`(집계선택→zero_clip), `build_iso_pchip_transform`(step→PCHIP), `fit_iso_tail_grid`(iso×tail×iqr grid), `iqr_stats`/`push_top_k_to_iqr`(rank만 사용, val leakage 없음), `rmse`/`clip_nonneg`/`apply_tau_pi`/`unit_rmse`/`aligned_unit_pred`, `json_default`/`build_die_df`/`build_unit_output`/`serializable_calibrator`.
- **config 주입 방식**: 노트북별 차이(`BASELINE_AGG`/`AGG_CANDIDATES`)와 공통 grid(zero_clip/iso/tail/iqr)를 **명시 kwargs**로 주입 → 모듈은 무상태(상수 박제 없음).
- **노트북에 유지**: 실험 드라이버 `fit_one_seed`/`save_result_artifacts`/`make_folds`/`params_for_seed`는 데이터·메타 결합이라 각 fit에 둠(라이브러리 1차함수를 호출). 37-필드 컨텍스트 객체를 강요해 correctness-critical 경로를 흔드는 것을 회피.
- **효과**: fit.ipynb ~970 → ~595 코드줄(×4 ≈ **1,500줄 중복 제거**), 모듈 443줄 1벌. **통일성 유지**(여전히 cell0/cell4만 차이).
- **검증**: zit_fit_lib **미해결 free var 0**(노트북 글로벌 누출 없음=동작 보존), 4 노트북 구문/uniformity OK, `fit_iso_tail_grid` 합성데이터 functional smoke 통과(17 candidate, pchip best).

---

## 3. 02_reg_single — 5모델 → 1튜너

> reg strategy.md에 *"원본 `02_reg_only.ipynb`의 `MODEL_NAME` 스위치 구조를 모델별로 분리"*라고 명시 — **병합 = 원래 단일 스위치 구조로 복귀**. 모델·search space가 이미 `modules.models.get_search_space(model_name)`로 파라미터화돼 있어 난이도 낮음.

위치: `02_reg_single/hpo.py` + `02_reg_single/fit.ipynb` (폴더 루트, 모델은 `--model`로 분기).

| 항목 | 설계 |
|------|------|
| `hpo.py` | `--model {lgbm,xgb,catboost,et,enet}` → 모델/search space 룩업. `01_reg_et_parallel_hpo.py` + `02_reg_xgb_parallel_hpo.py` + (catboost/enet/lgbm 노트북 inline HPO) 통합 |
| enet 분기 | enet만 **PP+scaling+target_transform joint** 탐색(`strategy_common §3`) → `--model enet`일 때 별도 trial 축 활성화 |
| `fit.ipynb` | `MODEL_NAME` 스위치로 해당 study best 로드 → fit/후처리/산출물 |
| 앵커 해제 | 5종 모두 `enqueue_trial(ANCHOR)` 제거. xgb `±50%` → wide. 트리 4종은 이미 wide range라 enqueue만 제거 |

**삭제**: `01_reg_et_parallel_hpo.py`, `02_reg_xgb_parallel_hpo.py`, `catboost.ipynb`, `enet.ipynb`, `et.ipynb`, `lgbm.ipynb`, `xgb.ipynb` (전부 `02_reg_single/hpo.py` + `fit.ipynb`로 흡수).

---

## 4. 03_two_stage — clf 1 + reg 1

ts/reg는 02_reg_single과 거의 동일하되 **Y>0 only + die-level broadcast** 차이뿐. 둘 다 `modules.models` + `modules.hpo`를 이미 사용.

| 묶음 | 설계 | source |
|------|------|--------|
| `default/clf/hpo.py` | `--model {lgbm,xgb,catboost,et}`. die-level 분류, `scale_pos_weight` 등 imbalance 축. `run_clf_hpo` | `clf/01_clf_lgbm_parallel_hpo.py` + `clf/{cat,et,xgb}.ipynb` |
| `default/clf/fit.ipynb` | MODEL_NAME 스위치 fit | (clf 노트북 fit부) |
| `default/reg/hpo.py` | `--model {...5}`. Y>0 only, die-level. enet joint 분기 | `reg/*.ipynb` inline HPO |
| `default/reg/fit.ipynb` | MODEL_NAME 스위치 fit | (reg 노트북 fit부) |
| `default/combine.ipynb` | **유지** (CLF 4 × REG 5 = 20 grid + position weighted) | 그대로 |
| `reverse/ts_reverse.ipynb` | **유지** (Path B, 노트북 형태) — 단 **앵커 완전 해제(wide)·실험번호 제거** | 그대로 두지 않음 |

**앵커 해제**: ts/reg 트리+enet enqueue 제거. ts/clf는 이미 앵커 없음(wide) — 변경 없음. **reverse도 완전 해제**(사용자 결정 2026-06-18): `narrow_around(ANCHOR,±30%)`→wide(`models.lgbm_space` 폭) + `enqueue_anchor`/`TS_REVERSE_ANCHOR` 제거, `EXP_ID 'ts-reverse-final-002'`→`'ts_reverse'`, 경로 `reverse/002/`→`reverse/`. 마지막 앵커 소비처가 사라져 `hpo.narrow_around`/`hpo.enqueue_anchor`도 모듈에서 삭제(단 `sample_from_space`는 reverse가 dict 공간 샘플링에 계속 사용 → 유지).
**삭제**: `clf/*.ipynb`, `clf/01_clf_lgbm_parallel_hpo.py`, `reg/*.ipynb` (튜너/ fit으로 흡수).

**중복 정책**: `default/reg/`는 `02_reg_single`과 ~90% 동일(차이 = Y>0 only + die broadcast). **폴더 격리 우선 → 중복 허용**(공통부를 억지로 모듈화하지 않음). 폴더 3개 독립 유지가 원칙.
**combine 입력 전제**: `combine.ipynb`는 `clf/{model}/`·`reg/{model}/` per-model 산출물 20조합을 읽는다. clf/reg를 1튜너로 병합해도 **fit이 모델별 OOF/val/test CSV를 §5.1 경로 구조 그대로** 떨궈야 combine이 동작.

---

## 5. 실험번호 제거 — 범위 평가 & 네이밍 규칙

**실험번호 제거는 스코프를 키우지 않는다.** 출현은 `hp00X` 22파일 + `*-final-00X` 9 py지만, 어차피 (a) 튜너 재작성/병합, (b) fit 노트북 재유도, (c) **앵커 완전 해제 → 새 study**를 하므로 깨끗한 이름으로 새로 시작하면 자연 흡수된다.

| 옛 패턴 | 새 패턴 |
|---------|---------|
| study/exp-id `zit-only-final-004`, `bag-zit-eql-final-005`, `reg-lgbm-002` | `zit_only_pearson`, `bag_zit_eql`, `reg_lgbm` 등 의미 이름 |
| DB `optuna_jh_zit-only-final-003.db` | `optuna_zit_only_pearson.db` |
| 출력 dir `4_output/.../004/`, `005/` | `4_output/.../zit_only_pearson/` |
| param `param1_005t17.json`, `param2_004t46.json` | `param1.json`~ 또는 의미 이름(`balanced.json` 등) |
| 코드 주석 `# hp/002 best trial #43 (OOF=...)` | 앵커 해제로 주석째 제거 |

**유일한 비용**: 옛 exp-id로 묶인 기존 Optuna DB·`4_output` 산출물 링크가 끊긴다. 앵커 해제 = 재실험이므로 폐기해도 무방. (보존이 필요하면 `모델링_이전자료/`로 백업 후 진행)

### 5.1 출력 경로 (4_output) 컨벤션

fit 노트북 산출물은 새 폴더명에 미러링한다.

| fit | OUT_DIR |
|-----|---------|
| `01_zit/zit_only_pearson/fit.ipynb` | `4_output/01_zit/zit_only_pearson/` |
| `01_zit/zit_only_eql/fit.ipynb` | `4_output/01_zit/zit_only_eql/` |
| `01_zit/bag_zit_pearson/fit.ipynb` | `4_output/01_zit/bag_zit_pearson/` |
| `01_zit/bag_zit_eql/fit.ipynb` | `4_output/01_zit/bag_zit_eql/` |
| `02_reg_single/fit.ipynb` (`--model M`) | `4_output/02_reg_single/{M}/` |
| `03_two_stage/default/clf/fit.ipynb` (`--model M`) | `4_output/03_two_stage/default/clf/{M}/` |
| `03_two_stage/default/reg/fit.ipynb` (`--model M`) | `4_output/03_two_stage/default/reg/{M}/` |
| `03_two_stage/default/combine.ipynb` | `4_output/03_two_stage/default/combined/` |

`combine`은 위 `clf/{M}/`·`reg/{M}/`를 그대로 읽으므로 **경로 구조 보존 필수**(§4).

### 5.2 출력 산출물 번들 규약 (producer ↔ stacking 계약)

> **핵심**: `4_output`의 폴더 구조 + 각 leaf의 산출물 파일셋이 곧 **fit 노트북(producer) ↔ 04_stacking(consumer) 간의 API 계약**이다. `04_stacking/_v4/discovery.py`가 `4_output`을 재귀 스캔해 **die CSV 3종을 모두 가진 폴더만** base 모델로 채택하고, **폴더 경로(`__`.join)를 모델 ID로** 쓴다. 따라서 출력 폴더명·파일셋을 먼저 확정하고 코드를 거기에 맞춘다.

**모든 fit leaf가 떨궈야 하는 표준 번들** (코드 실측 기준 — reg/clf/ts-reg는 `modules/hpo.py::save_artifacts`가 일괄 생성, zit·reverse는 동일 스키마를 인라인 작성):

| 파일 | 레벨 | 컬럼 | 소비처 |
|------|------|------|--------|
| `oof_die.csv` `val_die.csv` `test_die.csv` | die | `ufs_serial, run_wf_xy, pred, health`<br>(zit: `+ pi, one_minus_pi, mu` / clf: `pred` 대신 **`prob`**) | **stacking v4 필수 3종** + combine |
| `oof_unit.csv` `val_unit.csv` `test_unit.csv` | unit | `ufs_serial, pred, health` | cutoff RMSE 추정, 최종 제출 |
| `best_params.json` | — | `exp_id, model_name, best_params_resolved, effective_pp_params, feature_names, n_folds, unit_ids_hash, study_meta` (**clf은 `y_pos_const` 필수**) | stacking clf 스케일 변환, 재현 |
| `fold_models.pkl` | — | `fold_models, fold_scalers, feature_names, model_name, n_folds` | SHAP/refit 재사용 |
| `optuna_{user}_{study}.db` | — | Optuna study | RESUME |

규약 포인트:
- **die CSV 매칭 키 = `(ufs_serial, run_wf_xy)`** — stacking이 이 키로 base 행을 정렬·merge. 한 행이라도 누락되면 `discovery.build_die_matrix`가 RuntimeError.
- **clf die CSV는 `prob`(0~1) 컬럼** — stacking이 `prob × y_pos_const`(=E[Y\|Y>0], `best_params.json`에 박제)로 health 스케일 변환. `y_pos_const` 없는 clf 폴더는 자동 제외.
- **health 컬럼**: val/test의 y가 비공개면 그 컬럼이 NaN/누락될 수 있음(정상). cutoff RMSE는 `oof_unit.csv` 우선, 없으면 die-level fallback.
- **번들 *쓰기*는 이미 통일**(`save_artifacts`). 리팩토링이 손볼 **진짜 갭은 *경로***: 현재 모든 fit이 `OUT_DIR = os.path.join(OUTPUT_DIR, '...', EXP_ID.split('-')[-1])` 식으로 **실험번호를 경로에 박는다**(`reg-lgbm-raw-001` → `.../001/`). 이걸 §5.1 의미 폴더명으로 바꾸는 게 출력 정리의 본체.

**권장 작업 — 경로 단일 소스화** (구현 완료 2026-06-18): `modules/parallel_hpo.py`에 경로 헬퍼를 두었다 — `model_out_dir(*parts)`(§5.1 폴더 경로, 가변 인자라 `('01_zit','zit_only_pearson')`/`('01_zit/zit_only_pearson')` 동일) + `study_db_path(out_dir, user, exp_id)`(`optuna_{user}_{exp_id}.db` 파일명 규약 단일 소스). `resolve_out_dir`은 기존 호출 호환용 별칭으로 유지. **producer(hpo.py 7개)는 `study_db_path`를 호출**해 DB 파일명을 더 이상 각자 조립하지 않는다. `discovery.py`는 4_output을 `rglob` 스캔하므로 경로를 *조립*하지 않는다 → 쓰는 경로/스캔 경로가 구조적으로 어긋날 수 없음. (fit.ipynb의 OUT_DIR은 §5.1 그대로의 명시 경로를 유지 — zit fit 4종 byte-동일 제약 보호 + 최소수정. 모두 §5.1 일치 확인.) 번들 *내용* 작성은 `save_artifacts`가 담당하므로 신규 writer 불필요 — 경로 규약만 통일.

### 5.3 완성 `4_output` 트리 (빈 상태에서 채워질 목표 형상)

각 leaf 아래에 §5.2 표준 번들이 통째로 들어간다. **폴더명에 실험번호(`001`/`hp`/`raw`) 금지** — 폴더명이 곧 stacking 모델 태그(`__`.join)다.

```
4_output/
├── 01_zit/
│   ├── zit_only_pearson/     ┐
│   ├── zit_only_eql/         │ leaf = §5.2 번들 1세트
│   ├── bag_zit_pearson/      │ (die3 + unit3 + json + pkl + db)
│   └── bag_zit_eql/          ┘
├── 02_reg_single/
│   └── {lgbm,xgb,catboost,et,enet}/      # fit --model M → {M}/
├── 03_two_stage/
│   └── default/
│       ├── clf/{lgbm,xgb,catboost,et}/        # die CSV는 prob + best_params.json에 y_pos_const
│       ├── reg/{lgbm,xgb,catboost,et,enet}/   # Y>0 only
│       ├── combined/{clf}_x_{reg}/            # combine.ipynb 20조합 (clf/·reg/ 읽어 생성)
│       └── reverse/                           # ts_reverse (Path B)
└── 04_stacking/
    ├── die_shap/{base_tag}/   # build_shap_features → die_shap.npz (구 shap_cache/)
    ├── run_{ts}/              # stacking 탐색 결과 (구 results_extreme_v4 — 버전표기 제거 §7.1)
    └── final/                 # 최종 선정 메타 + submission
```

> **§5.1 ↔ §5.3 정합**: §5.1 경로표는 fit별 `OUT_DIR`을, §5.3은 그 결과 전체 트리를 그린다. 두 표의 폴더명이 어긋나면 stacking이 base를 못 찾으므로 **경로 헬퍼(§5.2)로 한 소스에서 생성**한다.

---

## 6. 죽은 코드 / 불필요 모듈 / 비효율 — 짚어주기

### 6.1 즉시 제거 (죽은 코드 / 좀비)

| 대상 | 근거 |
|------|------|
| **`modules/pp_hpo.py`** | `*_pphp.ipynb` 전용 모듈인데 pphp 노트북 전부 삭제됨. 활성 import **0건** (참조는 `모델링_이전자료/`·`modules.zip`뿐). → 고아 모듈 |
| **`pp_hp_strategy.md`** | 위 pp_hpo 경로 전략 문서 → 동반 고아 |
| **`modules/modules.zip`** | Drive 업로드용 빌드 산출물 → **git 추적 제외 + `.gitignore`** (Colab 부팅 깨는 것 아님 — git에서만 빼는 것). 빌드/재업로드는 모듈 수정 후 **맨 마지막 단계**이고 실질 재실행 계획은 없음. 단 코드 자체는 작동 가능하게 유지 |
| `01_zit/*copy*.ipynb` 5개 | copy 좀비 |
| `04_stacking/squeeze_extreme_v2.py`, `v3.py`, `v3_bundle.zip` | 구 monolithic 계열 — die-level `_v4/` 패키지가 대체. **v2는 치팅으로 무효** (§7) |
| `04_stacking/_prev/*.log` | 옛 실행 로그 |

### 6.2 비효율 / 실험 잔재 (구조 개선)

| 대상 | 문제 | 개선 |
|------|------|------|
| non-mmap 워커가 PP를 각자 재구현 | `_impute_spatial_median` 등 전처리가 워커 파일마다 중복 정의 | 공유 모듈 1곳으로(§2.1). 이게 4조합 분리의 선행 블로커 |
| `narrow_around` + `*_ANCHOR` dict 산재 | 1차 실험 결과를 코드에 박제 → 공개 레포에 실험 흔적 | 앵커 완전 해제 시 함께 제거(§2.3, §3, §4) |
| mmap 워커 v3/v4/v5 3변형 | EXPLOIT/PUSH/EXPLORE는 탐색 전략 실험용 | wide 단일 search space로 통일 |

### 6.3 검토 / 정리 방향

| 대상 | 관찰 | 처리 |
|------|------|------|
| `modules/scaler.py` | **왜 모델링에 있나**: ① 모델명→스케일 여부 분기(`needs_scaling`, `_SCALING_REQUIRED={'enet'}` = 모델링 관심사) + ② Robust(median/IQR) 변환 재구현(전처리 관심사)이 한 파일에 섞임. ②는 `2_preprocessing/scaling.py`의 `robust_scale`와 **중복 재구현**. enet 전략(scaling.py 5종 joint)과도 불일치(여긴 Robust 하드코딩 = 구버전 흔적) | **변환은 `scaling.py` 단일 소스로**, 모델명 게이트만 호출부 inline. 단 `hpo.py`·`0_baseline/_modules/e2e_hpo.py`·enet 노트북 2개가 `maybe_scale` 의존 → 호출부 교체 필요(중간 규모). 즉시 삭제 불가 |
| `04_stacking/` 전반 | die-level(v4) vs unit-level 2계열 + shap_cache 대량 | **§7에서 die-level v4 단일화로 확정** |

---

## 7. 04_stacking — die-level 단일화

**배경**: 04_stacking엔 unit-level(`stacking.ipynb`)과 die-level squeeze(v2→v3→v4) 두 계열이 공존했다. 실측 val RMSE는 전부 **0.0057 plateau**이고, unit-level v2의 근소 최저(0.005693)는 **치팅으로 무효**. die-level은 성능 우위는 아니지만(plateau) **가장 최신·패키지화·확장성(position 가중 + die-level SHAP)**이 좋다. → **die-level v4를 유일 정식으로 단일화하고 버전 명칭(v4) 제거** ("원래 그거였던 것처럼").

### 7.1 정식 = die-level v4 (clean rename)

| 현재 | → 새 이름 |
|------|----------|
| `squeeze_extreme_v4.ipynb` | `stacking.ipynb` (구 unit-level `stacking.ipynb` 삭제 후 이 자리로) |
| `_v4/` 패키지 | `stacking_lib/` (버전 표기 제거 — 이름 조정 가능) |
| `build_shap_features.py`, `run_shap_all.py` | 유지 (die-level `die_shap.npz` 생성 = 파이프라인 입력) |
| `4_output/04_stacking/results_extreme_v4/` | `4_output/04_stacking/` 정식 결과로 (버전 dir 제거) |

내부 `from _v4 import config, ...` → `from stacking_lib import ...` 전부 교체.

### 7.2 삭제 (죽은 계열 + 좀비)

| 대상 | 사유 |
|------|------|
| `stacking.ipynb` (구 unit-level) | die-level로 단일화 → 삭제하고 v4를 이 이름으로 승격 |
| `squeeze_extreme_v2.py` | unit-level + **치팅** SHAP → 무효 |
| `squeeze_extreme_v3.py` | die-level이나 `_v4/` 패키지가 대체 |
| `squeeze_extreme_v3_bundle.zip` | 빌드 좀비 |
| `_prev/*.log` | 옛 실행 로그 |
| `4_output/.../results_extreme`, `_v2`, `_v3` | 구 계열 결과 (archive 또는 삭제) |
| `docs/squeeze_experiments_*.md`, `shap_xstacking_plan.md` | 실험 로그 → strategy.md로 통합 또는 삭제 |

### 7.3 POOL_PATHS — 새 base 경로 의존 (필수 갱신)

v4의 base 입력 경로가 **옛 네이밍**(`01_zit/zit_only/`, `bag_zit/`)을 참조. 새 4조합(`zit_only_pearson`/`zit_only_eql`/`bag_zit_pearson`/`bag_zit_eql`) + reg/ts 경로(§5.1)로 갱신.
- **die-level 전제**: 메타가 die행 학습이므로 각 base가 **`*_die.csv`(die-level 예측)**를 떨궈야 함 → zit/reg/ts fit 노트북이 die-level 산출물도 저장하는지 확인.

### 7.4 strategy.md 재작성

`04_stacking/strategy.md`는 현재 **삭제될 unit-level `stacking.ipynb`를 설명** → die-level v4 파이프라인 기준으로 전면 재작성.

### 7.5 shap_cache / 대용량 바이너리

`shap_cache/*.npz`(die_shap = v4 입력)는 유지하되 **git 추적 제외**(.gitignore). 구 unit-level `*_unit_shap.parquet`는 v4 미사용 → 삭제 가능.

---

## 8. 작업 순서 (체크리스트)

1. [x] **백업/.gitignore** — 구 `4_output`은 `리펙토링대상_모델링_아웃풋/`로 이동(사용자), 백업 폴더 + `preprocessing.zip` 추적 제외 (.gitignore 이미 `*.zip`/`*.db`/shap_cache 커버)
2. [x] **zit_pp 공용 PP** — `01_zit/zit_pp.py` 신설(cleaning.py 정본, median 패치 폐기), `00_precompute_pp.py` 재배선 (§2.1)
3. [x] **scaler 정리** — `modules/scaler.py`·`pp_hpo.py` 제거(백업 이동), `hpo.py`의 게이트를 `_SCALING_REQUIRED` 인라인 상수로, `__init__.py` 도크스트링 갱신 (§6.3). e2e_hpo·enet은 `maybe_scale` 미사용(이미 `2_preprocessing/scaling.py` 사용) 확인
4. [x] **공용 하네스** — `modules/parallel_hpo.py` 신설 (전 트랙 공용 HPO 보일러플레이트, §1.1)
5. [x] **zit 4조합** — thin `hpo.py` 4 + fit 4(셀0/4만 차이, 나머지 byte-동일) 정리, `zit_objective.py` 공용 objective, eql 튜너 신설, 앵커 해제(wide), bag_zit_eql 단일 best화, `최종/` 승격. 비-mmap 워커 2개·`trial_to_json`·옛 노트북·copy 좀비·zit_et 정리 (§2)
6. [x] **reg 병합** — `02_reg_single/{hpo.py(--model 5종), fit.ipynb}`, parallel_hpo 사용, 앵커 해제 (§3)
7. [x] **ts 병합** — `default/clf/{hpo.py(4종), fit.ipynb}` + `default/reg/{hpo.py(5종,Y>0 only), fit.ipynb}`, parallel_hpo 사용, combine `_list_models` §5.1 직접경로 패치·reverse 유지, 앵커 해제 (§4)
8. [x] **EMT 이관** — `07_emtboost_hpo.ipynb` → 백업 이동, `04_default_compare.ipynb`에 **§4e EMTboost default**(profile-ζ, μ=LGBM 기본값) 흡수 + **§4d `zit_gu` 제거**, `zit_EMT.py` 유지. results 47행 (§2.5)
9. [x] **경로 헬퍼 단일화 + reverse/combine 점검** — `parallel_hpo.model_out_dir`/`study_db_path` 신설, hpo.py 7개 wire. combine §5.1 OK·discovery rglob(경로 무조립) 확인. **reverse 완전 해제**(앵커+번호) + `hpo.narrow_around`/`enqueue_anchor` 제거 (§4, §5.2)
10. [x] **스태킹 단일화** — `squeeze_extreme_v4.ipynb`→`stacking.ipynb`, `_v4/`→`stacking_lib/`, `SqueezeV4Config`→`StackingConfig`, `from _v4`→`from stacking_lib`. 구 unit-level `stacking.ipynb`·v2·v3·zip·docs → 백업 이동. config: `KNOWN_STRONG_SUBSET=()`·cutoff 0.006·`output_subdir="04_stacking"`. 노트북 cell8 variant필터 제거·cell11/16 SHAP 태그 §5.1화. `run_shap_all.py` TASKS 18개 §5.1 재작성. `04_stacking/strategy.md` die-level 전면 재작성 (§7)
11. [x] **실험번호 제거** — 활성 코드 genuine 토큰(hp00X/-final-00X/pphp/__hp__) **0건**. EXP_ID/study명 전부 의미이름. `build_shap` 경로·4 zit hpo.py 프로비넌스·4 zit fit 주석(통일 유지) 정리. 구 폴더별 strategy.md 3개(01_zit/02_reg/03_ts default)·`pp_hp_strategy.md` → 백업 이동 (refactor_strategy.md가 신구조 대체) (§5, §5.1)
12. [x] **baseline 작동 확인** — `0_baseline`(e2e_hpo·EMT) import-chain 전수 통과. `e2e_hpo`는 `scaling.hybrid_scale`(2_preprocessing) 사용·삭제된 `modules/scaler` 참조 0. EMTboost·create_regressor functional smoke OK
13. [x] **검증(§9)** — 동치성(registry search space·anchor-free)·combine §5.1·fit DB 신이름·stacking `from stacking_lib`(_v4 0)·die-level GroupKFold(ufs_serial)·공개 grep(추적 db/zip/npz/modules.zip 0, 워킹트리 금지토큰 0) 모두 통과. 활성 .py 29개 컴파일 OK. 백업 이동 55건은 미스테이징 삭제(커밋 시 정리). ⚠ `5_dashboard/data/*.parquet`(38M 등)은 리팩토링 범위 밖 — 별도 판단
14. [ ] (필요 시, 맨 마지막) `modules.zip` 재빌드 → Drive 재업로드 (재실행 계획 없으면 생략)
15. [x] (추가, 사용자 요청) **zit_fit_lib 모듈화** — isotonic/calibration 14함수 추출, ~1,500줄 중복 제거 (§2.6)
16. [x] (추가, 2026-06-19) **부트스트랩 통일** — 17 노트북 첫 셀 단일 stub(code.zip 1개 + cwd 자동탐색 + runpy), dataset fetch를 setup.py로 이관 (§10.1)
17. [x] (추가, 2026-06-19) **코드 번들 3→1** — `code.zip` 단일화(지원 .py 54개), Drive 동일 ID 새 버전 업로드. preprocessing.zip/modeling.zip 폐기 (§10.2)
18. [x] (추가, 2026-06-19) **ROOT 버그 수정** — 4 zit fit `str(ROOT/…)`→`str(PROJECT_ROOT/…)` (runpy 전환 부작용) (§10.3)
19. [x] (추가, 2026-06-19) **precompute 경로 통일** — `00_precompute_pp.py` 기본 `--name` `bag_zit_pp`→`zit_pp` + 디스크 폴더 rename, parallel_hpo와 일치 (§10.4)

---

## 9. 리스크 / 검증

- **impute 모듈화 회귀**: 이동 전후 PP 결과(컬럼 수·결측 처리)가 byte 동일한지 1개 fold로 확인. 다르면 bag fit OOF가 어긋남.
- **병합 튜너 동치성**: `02_reg_single/hpo.py --model lgbm`, `default/clf/hpo.py --model lgbm`, `default/reg/hpo.py --model lgbm` 각각이 옛 단일 노트북 한 trial과 동일 search space/objective인지(앵커 enqueue만 빠진 채) 확인.
- **앵커 해제 후 성능**: wide 탐색이 옛 narrow best를 회복하는지 study 진행 중 모니터링. 회복 못 하면 trial 예산 부족 신호.
- **baseline 작동**: `scaler` 정리·모듈 삭제 후 `0_baseline` 노트북(e2e_hpo·EMT)이 import 에러 없이 실행되는지 확인. **모델링만이 아니라 baseline까지 정상이어야 함.**
- **combine 입력 경로**: clf/reg fit이 §5.1 경로로 모델별 산출물을 떨궈 `combine.ipynb`의 20조합 glob이 그대로 동작하는지 확인.
- **fit DB 참조**: 새 study 이름으로 갱신됐는지(옛 `*-final-00X.db` 잔존 참조 0건). bag_zit_eql fit이 `param1~4.json`을 더 이상 참조하지 않는지.
- **스태킹 rename 무결성**: `from _v4 import ...` → `from stacking_lib import ...` 전부 교체됐는지(잔존 import 0건). `stacking.ipynb`가 옛 unit-level이 아니라 die-level v4 본문인지.
- **스태킹 die-level 누수/입력**: 메타 CV가 `ufs_serial` GroupKFold인지(같은 unit의 4 die 분리 금지). base의 `*_die.csv`가 새 경로에 존재해 POOL_PATHS가 모두 잡히는지.
- **공개 점검**: 레포에 `modules.zip`·`*.db`·대용량 parquet/npz·실험번호(`hp00X`/`*-final-00X`/`squeeze_extreme`/`_v4`) 잔존 0건 grep 확인.

---

## 10. 배포·부트스트랩 정리 (2026-06-19, 추가 요청)

§1~§9(코드 구조 리팩토링)와 별개로, GitHub/Colab 공개 직전 **실행 환경 호환**을 위해 처리한 운영성 변경. 학원 PC→D드라이브 복사 후에도, 노트북 폴더 깊이가 달라도, Colab에서도 **코드 수정 없이** 돌도록 한다.

### 10.1 노트북 부트스트랩 통일 (17개)

- **이전**: 노트북마다 `%run ../setup.py`(폴더 깊이 `../` 의존) + 3개 zip(code/preprocessing/modeling) 분기 if-스파게티 + dataset 다운로드까지 첫 셀에 흩어짐.
- **변경**: 모든 노트북 첫 셀을 **동일 thin stub**으로 교체.
  - Colab: `GDRIVE_CODE_ID`(code.zip **1개**)만 gdown→unzip→`/content/project`로 `chdir`.
  - 공통: **cwd에서 위로 `setup.py`(+`utils/`)를 자동탐색**한 뒤 `runpy.run_path`로 실행 → 노트북이 몇 단계 깊이에 있든, 드라이브 위치가 바뀌어도(D드라이브 복사) 동작. `../` 하드코딩 제거.
- **데이터 fetch 이관**: dataset.zip(1.2GB) 자동 다운로드를 노트북 if문에서 **`setup.py`로 이관**(`ENV=='colab'` & `compet_xs_data.csv` 없을 때만 gdown→unzip). 노트북은 코드 번들만 책임.

### 10.2 코드 번들 3→1 (`code.zip` 단일화)

- `code.zip`/`preprocessing.zip`/`modeling.zip` 3개 → **`code.zip` 1개**로 통합. 내용 = `setup.py`+`requirements.txt`+`utils/`+`2_preprocessing/`+`3_modeling/`의 지원 `.py` (54개) + `1_eda/eda_style.mplstyle`. 노트북(.ipynb)·데이터·db·npz·캐시·백업 폴더 제외.
- Drive ID `1AD4PDBnDVjp-LSna6puB7qLnpBqB7j_I`는 **유지**(우클릭 → 버전 관리 → 새 버전 업로드) → 노트북 `GDRIVE_CODE_ID` 불변. `preprocessing.zip`/`modeling.zip`은 폐기(코드가 code.zip에 흡수), `dataset.zip`은 그대로.
- → §6.1의 `modules/modules.zip` 항목 및 체크리스트 14의 "modules.zip 재빌드"는 이 단일화로 **대체**됨.

### 10.3 ROOT 버그 수정 (4 zit fit)

- `%run`은 setup.py 전역을 노트북 네임스페이스에 주입했지만 `runpy.run_path`는 주입하지 않는다. 4 zit fit의 cell2 `_ZIT_DIR = str(ROOT / '3_modeling' / '01_zit')`가 **미정의 `ROOT`** 참조 → `NameError`(refactor 후 미재실행이라 잠복).
- **수정**: `ROOT` → `PROJECT_ROOT`(config 셀에서 `Path(CFG_PROJECT_ROOT)`로 정의됨). 4개 byte-동일 유지(동일 수정). 다른 13 노트북은 누출 전역 의존 없음 확인(`ENV` 미사용, `plt`/`np`/`pd` 자체 import).

### 10.4 precompute 경로 통일 (`bag_zit_pp`→`zit_pp`)

- 만드는 쪽 `00_precompute_pp.py`의 `--name` 기본값이 `bag_zit_pp`(bag 전용이던 시절 잔재)인데, 읽는 쪽 `parallel_hpo.DEFAULT_PRECOMPUTED_DIR`는 `zit_pp` → **경로 불일치**로 HPO가 mmap을 못 찾는 블로커. (코드 전체 표준은 로더 모듈명 `zit_pp.py`·parallel_hpo 모두 `zit_pp`)
- **수정(A안)**: `00_precompute_pp.py` 기본 `--name` `bag_zit_pp`→`zit_pp`(+docstring 예시·헤더 문구), 디스크 폴더 `0_data/precomputed/bag_zit_pp/`→`zit_pp/` rename(pp.npy 461.9MB 재계산 0). 읽는 쪽(5 hpo.py가 의존하는 parallel_hpo)은 무수정.
- 결과: 만드는/읽는 쪽 모두 `zit_pp`. **로컬**은 기존 npy 그대로 재사용(00pp 재실행 불필요), **Colab**은 `python 00_precompute_pp.py` 1회(기본값 zit_pp)로 생성 후 병렬 HPO.
