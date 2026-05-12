# PP+HP 노트북 추가 전략 — 스태킹 다양성 확보

> 작성: 2026-05-12. 본 문서는 기존 `strategy_common.md` / 각 폴더 `strategy.md`를 **보완**한다.
> 충돌 시 우선순위: 본 문서(신규 pp+hp 노트북 한정) > strategy_common.md.

---

## 1. 목적 & 배경

현재 모델마다 **HP만 흔드는** 노트북이 있고 PP는 `PP_FIXED`(8키 고정값)로 박혀 있다 (`strategy_common.md §1`). 트리계열은 PP HP importance가 평균 3.1%로 작지만, **스태킹 base의 다양성**을 위해서는 같은 모델이라도 서로 다른 영역을 학습한 예측 2벌이 1벌보다 낫다.

→ **enet을 제외한 15개 모델 노트북에 "PP도 흔드는" 쌍둥이 노트북(`*_pphp.ipynb`)을 추가**한다. 기존 17 + 신규 15 = **32개 모델 노트북**.

- enet 제외 이유: `02_reg_single/enet.ipynb` / `03_two_stage/default/reg/enet.ipynb`는 **이미** PP + X스케일링 + y변환 + HP를 통째로 joint Optuna로 돌리는 구조(`strategy_common.md §3`). pp+hp 컨셉이 이미 들어가 있으므로 그대로 둔다.
- 신규 노트북은 enet의 joint 방식(스케일러·y변환까지)을 **복사하지 않는다**. 트리용으로 가볍게 — cleaning 6축만 흔든다 (아래 §3).

---

## 2. 신규 노트북 15개

기존 노트북 옆에 `_pphp` 접미사로 생성. 코드는 기존 hp-only 노트북의 **클론 + 외과수술**(PP_FIXED → PP 탐색, `run_hpo` → `run_pp_hpo`, `EXP_ID`/`OUT_DIR` 변경)이다 — 구조·anchor·RESUME·Colab 부트스트랩 그대로 유지.

| # | 신규 노트북 | 원본(hp-only) | 학습 경로 / HPO 호출 | EXP_ID | OUT_DIR |
|---|---|---|---|---|---|
| 1 | `01_zit/01_zit_only_pphp.ipynb` | `01_zit_only.ipynb` | zitboost, inline `optuna.create_study` + 커스텀 objective | `zit-only-pphp` | `4_output/01_zit/zit_only/pphp/` |
| 2 | `01_zit/02_bag_zit_pphp.ipynb` | `02_bag_zit.ipynb` | bagged zitboost, inline objective | `bag-zit-pphp` | `4_output/01_zit/bag_zit/pphp/` |
| 3 | `02_reg_single/lgbm_pphp.ipynb` | `lgbm.ipynb` | reg, `run_pp_hpo` | `reg-lgbm-pphp` | `4_output/02_reg_single/lgbm/pphp/` |
| 4 | `02_reg_single/xgb_pphp.ipynb` | `xgb.ipynb` | reg, `run_pp_hpo` | `reg-xgb-pphp` | `4_output/02_reg_single/xgb/pphp/` |
| 5 | `02_reg_single/catboost_pphp.ipynb` | `catboost.ipynb` | reg, `run_pp_hpo` | `reg-catboost-pphp` | `4_output/02_reg_single/catboost/pphp/` |
| 6 | `02_reg_single/et_pphp.ipynb` | `et.ipynb` | reg, `run_pp_hpo` | `reg-et-pphp` | `4_output/02_reg_single/et/pphp/` |
| 7 | `03_two_stage/default/clf/lgbm_pphp.ipynb` | `clf/lgbm.ipynb` | clf, `run_pp_clf_hpo` | `ts-clf-lgbm-pphp` | `4_output/03_two_stage/default/clf/lgbm/pphp/` |
| 8 | `03_two_stage/default/clf/xgb_pphp.ipynb` | `clf/xgb.ipynb` | clf, `run_pp_clf_hpo` | `ts-clf-xgb-pphp` | `.../clf/xgb/pphp/` |
| 9 | `03_two_stage/default/clf/catboost_pphp.ipynb` | `clf/catboost.ipynb` | clf, `run_pp_clf_hpo` | `ts-clf-catboost-pphp` | `.../clf/catboost/pphp/` |
| 10 | `03_two_stage/default/clf/et_pphp.ipynb` | `clf/et.ipynb` | clf, `run_pp_clf_hpo` | `ts-clf-et-pphp` | `.../clf/et/pphp/` |
| 11 | `03_two_stage/default/reg/lgbm_pphp.ipynb` | `reg/lgbm.ipynb` | reg, `run_pp_hpo(y_positive_only=True)` | `ts-reg-lgbm-pphp` | `.../reg/lgbm/pphp/` |
| 12 | `03_two_stage/default/reg/xgb_pphp.ipynb` | `reg/xgb.ipynb` | reg, `run_pp_hpo(y_positive_only=True)` | `ts-reg-xgb-pphp` | `.../reg/xgb/pphp/` |
| 13 | `03_two_stage/default/reg/catboost_pphp.ipynb` | `reg/catboost.ipynb` | reg, `run_pp_hpo(y_positive_only=True)` | `ts-reg-catboost-pphp` | `.../reg/catboost/pphp/` |
| 14 | `03_two_stage/default/reg/et_pphp.ipynb` | `reg/et.ipynb` | reg, `run_pp_hpo(y_positive_only=True)` | `ts-reg-et-pphp` | `.../reg/et/pphp/` |
| 15 | `03_two_stage/reverse/ts_reverse_pphp.ipynb` | `ts_reverse.ipynb` | reverse(die→agg), inline objective | `ts-reverse-pphp` | `4_output/03_two_stage/reverse/pphp/` |

- `combine.ipynb`(two_stage), `stacking.ipynb`은 집계 노트북이라 추가 없음. **stacking.ipynb는 신규 OOF 15벌을 base pool에 더 받도록 나중에 별도 수정** (본 작업 범위 밖, todolist 후반).
- `03_two_stage/default/combine.ipynb`도 clf×reg 페어 grid에 `_pphp` 산출물을 포함하도록 나중에 입력 경로 추가 (별도 항목).

---

## 3. PP 탐색공간 (트리·zit 공통 — 확정)

기존 `PP_FIXED` 8키 중 6개를 흔들고 나머지는 고정. **단위(간격)는 이전자료 후보처럼 coarse하게**, 범위만 확장.

### 흔드는 6축 (categorical)

| 키 | 후보 리스트 | 비고 |
|---|---|---|
| `missing_threshold` | `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]` | 이전자료 그대로. 현재값 0.30 포함 |
| `corr_threshold` | `[0.80, 0.84, 0.88, 0.92, 0.96, 0.98]` | 범위 0.80~0.98로 확장, 간격 0.04(이전자료 [0.90,0.94,0.98]와 동일 간격). 끝 구간만 0.02 |
| `add_indicator` | `[True, False]` | 둘 다. False면 `indicator_threshold` 무관 → conditional sampling |
| `indicator_threshold` | `[0.01, 0.05, 0.10, 0.15, 0.20, 0.25]` | 이전자료 그대로. `add_indicator=True`일 때만 샘플 |
| `spatial_max_dist` | `[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]` | 범위 1~6으로 확장(현재값 6.0 포함), 간격 1.0 |
| `post_impute_corr_threshold` | `[0.96, 0.97, 0.98, 0.99]` | 범위 0.96~0.99로 확장(현재값 0.96 포함), 간격 0.01 |

조합 수: `add_indicator=True` → 9×6×6×6×4 = 7,776, `False` → 9×6×6×4 = 1,296 → 합 **9,072**. (지금처럼 캐시 작게 잡으면 거의 다 새로 전처리.)

### 고정 (PP_FIXED와 동일, 흔들지 않음)

- `corr_keep_by = 'std'`
- `post_impute_corr_keep_by = 'std'`
- `corr_winsorize_pct = 0.0` (DEFAULT_PARAMS 기본값 — std 계산 전 분위수 clip 안 함)
- `const_threshold = 1e-6`
- `remove_duplicates = True` (`_FIXED`)
- `imputation_method = 'spatial'` (`_FIXED`)
- 이상치: `outlier_method='winsorize'`, `lower_pct=0.0`, `upper_pct=0.99` (`_FIXED`)
- Stage 0 `EXCLUDE_COLS` 하드코딩 그대로
- 메타피처: `add_meta_features(position_mode='raw', use_die_xy=True)` 그대로

### die→unit 집계

**PP 탐색축으로 넣지 않는다.** 현재 구조 그대로 — die-broadcast(또는 reg는 y>0 only)로 학습 → **postprocess에서 `agg_methods` 8종(mean/median/max/min/trimmed_mean/weighted/Q25/Q75) 중 train OOF best 자동 선택 + `zero_clip` 0.001~0.015 탐색**. 이전자료의 `AGG_PRESETS` 방식(unit-level 학습)은 우리 구조와 안 맞아 채택 안 함.

---

## 4. PP 인프라 — 방식 A (per-trial 재호출 + 캐시)

`3_modeling/modules/`에 `preprocess.run()`을 trial마다 다시 도는 인프라가 없으므로 **얇은 wrapper 모듈 신규 추가**: `3_modeling/modules/pp_hpo.py`.

### 핵심 설계

1. **`pp_search_space(trial) -> dict`** — 위 §3의 6축을 conditional sampling으로 샘플. trial param 이름은 `pp_` prefix (`pp_missing_threshold`, `pp_corr_threshold`, ...). `preprocess.run(params=...)`에 넘길 수 있는 형태로 반환 (`add_indicator=False`면 `indicator_threshold`는 DEFAULT 값으로 채움).

2. **closure 기반 캐시** — `make_cached_preprocess(xs, ys, feat_cols, xs_dict)` → 내부에 LRU dict(`maxsize` 작게, **기본 2**)를 들고 `prep(pp_tuple) -> (xs_train, xs_val, xs_test, feat_cols_clean, effective_params)`를 반환. key는 6축 값의 정렬된 tuple. cache miss면 `preprocess.run()` + `add_meta_features()` 실행 후 저장.
   - ⚠ **메모리**: cleaned 3-split DataFrame 한 벌 ≈ 1~1.5GB (174,980행 × ~900열 float). `maxsize=2`면 ~3GB. 노트북 동시 실행 수에 따라 조정 — 노트북 최상단 변수 `PP_CACHE_SIZE`로 노출 (기본 2).
   - ⚠ **재현성**: `preprocess.run`은 `fit on train, transform all` 구조라 캐시해도 leakage 없음 (PP 자체가 PP params만의 함수).

3. **`run_pp_hpo(...)`** — `hpo.run_hpo`와 동일 시그니처 + `cached_prep` 인자 추가. objective:
   ```
   def objective(trial):
       pp = pp_search_space(trial)                       # PP 6축
       xs_tr, xs_vl, xs_te, fcols, eff_pp = cached_prep(_pp_key(pp))
       hp = models.get_search_space(model_name, variant)(trial)   # 기존 HP 그대로
       # ... 이하 hpo.run_hpo의 objective와 동일: fold loop, _fit_predict_fold, OOF unit RMSE ...
   ```
   - **fold 분할은 PP와 무관**: `ys_train_unit[KEY_COL].unique()`는 PP가 어떤 컬럼을 떨궈도 불변 → `_make_unit_folds`를 study 시작 시 1회만 (기존과 동일, leakage·일관성 OK).
   - 단, `_die_mask_from_units`, `_build_X`는 PP마다 xs가 바뀌므로 **objective 안에서** 재계산.
   - **재사용**: `hpo._make_unit_folds`, `_die_mask_from_units`, `_broadcast_y_to_die`, `_fit_predict_fold`, `_build_X`, `_aggregate_die_to_unit`, `_inject_n_jobs`, `_is_tweedie_hp`, `enqueue_anchor`, `sample_from_space` 전부 그대로 import해서 씀. **hpo.py는 수정하지 않음** (필요하면 private 함수 import만).
   - val/test RMSE 기록(`user_attr`)도 기존처럼. anchor enqueue도 기존처럼 — 단 anchor에 `pp_*` 키도 추가해야 enqueue가 완전한 trial이 됨 (없으면 Optuna가 그 trial에서 빠진 pp 키를 따로 샘플 → 동작은 함). **anchor는 1차 best HP + `PP_FIXED` 값을 pp_ 키로** 채워 첫 trial로.

4. **`run_pp_clf_hpo(...)`** — 동일 패턴으로 `hpo.run_clf_hpo`를 래핑 (clf objective는 die-level proba → unit RMSE 평가).

5. **`refit_pp_best(...)`** — best trial의 `pp_*`로 `preprocess.run()` + `add_meta_features()` 한 번 더 돌려 xs_train/val/test 재구성 → 그 다음은 **기존 `hpo.refit_best` / `hpo.refit_clf_best`를 그대로 호출**. 산출물 저장도 기존 `hpo.save_artifacts` / `save_clf_artifacts` 그대로. `best_params.json`에 `effective_pp_params`가 best PP 값으로 들어가도록 `study_meta`에 반영.

6. **zit_only / bag_zit / ts_reverse** (inline objective 노트북): wrapper 함수가 아니라 **노트북 안 objective에 PP 샘플 + cached_prep 두 줄 추가**. cached_prep은 `pp_hpo.make_cached_preprocess`를 그대로 import해 씀. 나머지 EM/bagging/inner-CV 로직은 손대지 않음.

### hpo.py 수정 여부

**원칙: hpo.py·preprocess.py·models.py·postprocess.py는 수정하지 않는다.** pp_hpo.py는 이들의 public/private 심볼을 import만 한다. (private `_` 함수 import는 같은 패키지 내부라 허용 — 단 `__init__.py`에서 re-export는 안 함.) 만약 어쩔 수 없이 수정이 필요해지면 → `strategy_common.md §13` 원칙대로 **작업 중단·사용자 보고**.

### 모듈 zip 재업로드

`3_modeling/modules/pp_hpo.py` 추가 → `modeling.zip`(ID `1Vrn5LBl611rWbag7d09LZH68_lfpu6wP`) 재생성·재업로드 필요 (`CLAUDE.md` 환경 호환 규칙).

---

## 5. HP 탐색공간 — 기존 그대로

신규 노트북은 **HP 범위를 새로 안 만든다.** `models.get_search_space(model_name, variant)` (트리 4종 + zitboost)와 `models.get_clf_search_space(model_name)` (clf 4종)를 hp-only 노트북과 동일하게 쓴다. anchor도 동일(1차 best HP) — 단 enqueue dict에 `pp_*` 키를 `PP_FIXED` 값으로 추가.

→ 신규 노트북과 hp-only 노트북의 차이는 **PP 6축이 trial 축에 들어가느냐**뿐. 이게 OOF 다양성의 원천.

---

## 6. 출력 / 산출물

- 출력 경로는 §2 표대로 `.../{model}/pphp/` — hp-only(`.../002/` 등)와 **물리적으로 분리**되어 덮어쓰지 않음.
- 표준 9개 산출물(`strategy_common.md §15`) 그대로: `best_params.json`, `fold_models.pkl`, `optuna_*.db`, `oof|val|test_die.csv`, `oof|val|test_unit.csv`.
- `best_params.json` 추가 검증: `effective_pp_params`가 **best trial의 PP 값**이어야 함 (PP_FIXED 값이 아니라). `study_meta`에 `pp_search_space_candidates`(§3 후보 리스트들)와 `pp_cache_size` 기록.
- RESUME / `GDRIVE_*_ID` / Colab 부트스트랩 셀: 원본 노트북 그대로 복사. EXP_ID·OUT_DIR만 변경.
- Colab에서 `4_output.zip`(RESUME용)에 `pphp/` 폴더가 포함되도록 — 최초 1회 실행 후 zip 재생성·재업로드 (운영 항목).

---

## 7. 작업 순서 (todolist 골격)

1. **`3_modeling/modules/pp_hpo.py` 작성** — `pp_search_space`, `make_cached_preprocess`, `run_pp_hpo`, `run_pp_clf_hpo`, `refit_pp_best`(+clf), `_pp_key`. hpo.py 등 기존 모듈 무수정 원칙 준수.
   - 작성 전 `hpo.py` 전체(`run_hpo`/`objective`/`refit_best`/`run_clf_hpo`/`refit_clf_best`/`save_artifacts`/`save_clf_artifacts`), `preprocess.py`, `meta_features` 시그니처, `postprocess.py` 정독.
   - `__init__.py`에 `pp_hpo` 노출.
2. **`pp_hpo.py` 자체 sanity** — `ast.parse` + 작은 입력으로 import·1 trial 실제 실행(런타임 검증, `memory/feedback_runtime_verify.md` 준수). 사용자 검수 요청.
3. **reg_single pphp 4개** (`02_reg_single/{lgbm,xgb,catboost,et}_pphp.ipynb`) — 원본 클론 + 외과수술. lgbm 먼저 만들고 검수 → 나머지 3개 동형 복제.
4. **ts_clf pphp 4개**, **ts_reg pphp 4개** — 동형.
5. **zit_only / bag_zit / ts_reverse pphp 3개** — inline objective에 PP 2줄 추가.
6. **`modeling.zip` 재생성·재업로드** (pp_hpo.py 포함).
7. (후반, 별도) `stacking.ipynb` / `combine.ipynb` 입력 풀에 `pphp/` 산출물 추가.

### 검수

각 노트북·모듈 작업 후 `strategy_common.md §23` 6항목 체크리스트 준용 — 특히 §23.6 leakage(unit-level KFold·fit on train only·후처리 train OOF only)와 §23.2 기존 코드 컨셉 보존(hpo private 함수 시그니처 호환). 자동 검수 금지 — 사용자 확인 후.

---

## 8. 결정 로그 (2026-05-12, 사용자 확정)

- 신규 노트북: enet 2개 제외 15개. 17 + 15 = 32.
- PP 흔드는 6축 + 범위/단위: §3 표대로 확정.
- PP 인프라: **방식 A** (per-trial `preprocess.run` 재호출 + closure LRU 캐시). e2e_hpo.py 포팅(B)·PP 사전고정(C) 기각.
- 집계: postprocess에서 OOF best 자동선택 — PP 축 아님.
- HP 범위: hp-only 노트북 것 그대로 재사용.
