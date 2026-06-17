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
├── modules/                     # zit.py, zit_EMT.py, hpo.py, models.py, postprocess.py, preprocess.py, scaler.py(정리 대상)
│                                #  └ pp_hpo.py·modules.zip 삭제. zit_EMT.py 유지(0_baseline EMT용). scaler.py→scaling.py 흡수
├── 0_baseline/                  # (작동 보장 대상) 10모델 비교 + EMT(07) 이관. e2e_hpo의 scaler 의존 갱신
├── 01_zit/
│   ├── 00_precompute_pp.py      # (유지) mmap 전처리 사전계산 — 공용
│   ├── trial_to_json.py         # (유지) study trial → param json — 공용
│   ├── zit_only_pearson/        #   hpo.py  +  fit.ipynb
│   ├── zit_only_eql/            #   hpo.py  +  fit.ipynb   ← 튜너 신설
│   ├── bag_zit_pearson/         #   hpo.py  +  fit.ipynb
│   └── bag_zit_eql/             #   hpo.py  +  fit.ipynb   (단일 best — 4-param 앙상블 폐기)
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

### 1.1 병렬 실행 모델 (mmap 계승 — 모든 `hpo.py` 공통)

모든 트랙(`zit`/`reg`/`ts`)의 `hpo.py`는 **최신 `02_bag_zit_eql_mmap_parallel_hpo.py`의 mmap 병렬 방식을 계승**한다. (구 non-mmap 워커가 아니라 mmap이 표준.)

| 단계 | 동작 |
|------|------|
| **원천 입력** | **CSV** (`compet_xs_data.csv` 등) — 진실의 원천 |
| **사전계산** | `00_precompute_pp.py`가 CSV를 전처리해 `pp.npy`(float64 `(n_dies, F+2)`)로 1회 저장. **npy = 빌드 산출물(캐시), git 추적 제외(`.gitignore`)** |
| **워커** | N개 독립 프로세스가 `np.load(mmap_mode='r')`로 **행렬 1벌만 RAM 공유** + **1개 Optuna SQLite study 공유**(`--worker-id w1..wN`) |
| **스레드** | 프로세스당 BLAS/numpy 스레드 1로 캡 (오버서브스크립션 방지) |
| **역할 분리** | 워커는 Optuna DB 기록만. refit·후처리·산출물 저장은 `fit.ipynb`에서 별도 |

- **입력 정책**: "**입력의 진실은 CSV, `pp.npy`는 빌드 캐시**". 재현은 `python 00_precompute_pp.py` 한 줄로 CSV에서 npy 재생성 → 레포에는 CSV/코드만, 이진 캐시는 비추적. (CSV를 워커가 직접 읽으면 mmap RAM 공유가 깨지므로 npy 캐시는 유지)
- **권장 워커 수 = 3** (`3 × n_jobs ≤ 물리 스레드`). 구 mmap 헤더의 7워커는 과다 구독(7×n_jobs2=14 threads) → **3으로 표준화**.
- **enqueue-anchor 제거**: 병렬 실행 메커니즘은 계승하되, `--enqueue-anchor`(옛 best 주입)는 앵커 해제 원칙(§0-2)에 따라 삭제.

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

### 2.1 선행 필수 — `_impute_spatial_median` 모듈화 (블로커)

- 현재 `_impute_spatial_median`은 **non-mmap 워커 2개에만** 정의(`02_bag_zit_parallel_hpo.py`, `02_bag_zit_eql_parallel_hpo.py`).
- bag fit 노트북 3개(`06_bag`, `06_bag_eql`, `최종`)가 `spec_from_file`로 그 함수를 **동적 로드**.
- mmap 워커(최신)엔 이 함수가 없음 → PP를 `00_precompute_pp.py`로 분리했기 때문.
- **mmap 채택 + non-mmap 삭제를 동시에 하면 fit 노트북이 깨진다.**

→ **작업**: `_impute_spatial_median`을 공유 위치로 이동(권장: `2_preprocessing/`의 imputation 계열 또는 `modules/preprocess.py`). 그 후 mmap 워커·precompute·fit 노트북 모두 그 한 곳에서 import. 이러면 non-mmap 워커 2개 삭제 가능.

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

### 2.5 EMT — 베이스라인으로 이관

EMT(EMTboost)는 4조합에 넣지 않고 **`0_baseline`에만 남긴다** (베이스라인 비교 모델). `01_zit`에 `emt/` 트랙 만들지 않음.

| 파일 | 처리 |
|------|------|
| `01_zit/07_emtboost_hpo.ipynb` | `0_baseline/`로 이관 (baseline 비교에 통합) |
| `modules/zit_EMT.py` | **유지** — baseline EMT가 사용 |

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
| `reverse/ts_reverse.ipynb` | **유지** (Path B) | 그대로 |

**앵커 해제**: ts/reg 트리+enet enqueue 제거. ts/clf는 이미 앵커 없음(wide) — 변경 없음.
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

**권장 작업 — 경로 단일 소스화**: `OUT_DIR` 조립을 노트북마다 애드혹으로 두지 말고 `modules`에 경로 헬퍼 한 곳(예: `model_out_dir(track, model, variant=None)`)을 두고 producer와 `discovery.py`가 **둘 다 그걸 참조**한다. 이러면 "쓰는 경로"와 "스캔하는 경로"가 구조적으로 어긋날 수 없다. (번들 *내용* 작성은 `save_artifacts`가 이미 담당하므로 신규 writer는 불필요 — 경로 규약만 통일.)

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

1. [ ] **백업** — 기존 `4_output`·optuna DB가 필요하면 `모델링_이전자료/`로 이동 (앵커 해제 = 폐기 전제)
2. [ ] **선행 모듈화** — `_impute_spatial_median` 공유 위치 이동 + 워커/노트북/precompute import 교체, `trial_to_json.py` 경로 참조 갱신 (§2.1)
3. [ ] **죽은 코드 제거** — §6.1 일괄 삭제, `.gitignore`에 `*.zip`·`*.db`·대용량 parquet/npz 등록
4. [ ] **zit 4조합** — 튜너 4 + fit 4 정리, eql 튜너 신설, 앵커 해제, bag_zit_eql 단일 best화(앙상블 폐기), `최종/` 승격 (§2)
5. [ ] **reg 병합** — `02_reg_single/{hpo.py(--model), fit.ipynb}`, 앵커 해제 (§3)
6. [ ] **ts 병합** — `default/clf/{hpo.py, fit.ipynb}` + `default/reg/{hpo.py, fit.ipynb}`, combine/reverse 유지, 앵커 해제 (§4)
7. [ ] **scaler 정리** — 변환은 `scaling.py` 단일화, `hpo.py`·`0_baseline/_modules/e2e_hpo.py`·enet 노트북 2개의 `maybe_scale` 호출부 교체 (§6.3)
8. [ ] **EMT 이관** — `07_emtboost_hpo.ipynb` → `0_baseline/`, `zit_EMT.py` 유지 (§2.5)
9. [ ] **스태킹 단일화** — v4 die-level → `stacking.ipynb`/`stacking_lib` rename, 구 계열 삭제, `from _v4` import 교체, POOL_PATHS·strategy.md 갱신 (§7)
10. [ ] **실험번호 제거** — study/DB/output/param/exp-id 네이밍 일괄 (§5, §5.1)
11. [ ] **baseline 작동 확인** — `0_baseline`(e2e_hpo·EMT 포함)이 scaler 정리·모듈 삭제 후에도 정상 실행
12. [ ] **검증** — §9
13. [ ] (필요 시, 맨 마지막) `modules.zip` 재빌드 → Drive 재업로드 (재실행 계획 없으면 생략)

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
