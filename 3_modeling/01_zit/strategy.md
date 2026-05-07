# 01_zit 전략 — ZITboost 2종 (zit_only + bag_zit)

> 공통 규칙은 [strategy_common.md](../strategy_common.md) 참조. 본 문서는 01_zit 전용 사항만 명시.

---

## 1. 목표

ZITboost 변형 2종을 narrow HPO + 후처리로 마무리:
- `01_zit_only.ipynb`: ZITboost 단독 (joint EM, π·μ·φ 동시 학습)
- `02_bag_zit.ipynb`: BagZIT (unit-aware 변형, 매 EM iter마다 same-unit dies에 unit_y 비례 분배)

두 노트북 모두 기존 PP+HPO joint 실험의 best HP를 anchor로 narrow 재탐색.

---

## 2. 원본 파일 매핑

| 신규 (3_modeling/01_zit/) | 원본 (모델링_이전자료/3_modeling_이전자료/) | 결과 (모델링_이전자료/4_output_이전자료/final/zit_only/) |
|---|---|---|
| `01_zit_only.ipynb` | `_temp_bagzit/zit_only_hpo.ipynb` | `zit_only_hpo.txt` |
| `02_bag_zit.ipynb` | `_temp_bagzit/zit_bag_hpo.ipynb` | `zit_bag_hpo.txt` |

**모듈 의존**:
- 전처리: `2_preprocessing/{cleaning, scaling, outlier, ...}.py`
- 모델: `3_modeling/modules/zit.py` (BagZITboost 클래스 포함)
- HPO: `3_modeling/modules/{hpo, search_space, postprocess}.py`

---

## 3. PP 고정값

[strategy_common.md §1](../strategy_common.md)의 `PP_FIXED` 그대로 사용:
```python
PP_FIXED = {
    'missing_threshold':          0.30,
    'corr_threshold':             0.90,
    'corr_keep_by':               'std',
    'add_indicator':              True,
    'indicator_threshold':        0.05,
    'spatial_max_dist':           6.0,
    'post_impute_corr_threshold': 0.96,
    'post_impute_corr_keep_by':   'std',
}
```

zit_only / bag_zit 두 노트북 동일하게 적용.

---

## 4. Anchor — 01_zit_only HP

[zit_only_hpo.txt](../../모델링_이전자료/4_output_이전자료/final/zit_only/zit_only_hpo.txt) trial 99 best:
- OOF=0.005496, val=0.005707, test=0.008410, τ_π=0.944
- n_feats=714 (PP 후 살아남은 feature)
- elapsed=1281s/trial

```python
ZIT_ONLY_ANCHOR = {
    # ── 모델 공통 ──
    'zeta':                  1.149,    # Tweedie power
    'n_em_iters':            13,
    # ── μ (Tweedie mean, 9개) ──
    'mu_n_estimators':       240,
    'mu_learning_rate':      0.00309,
    'mu_num_leaves':         212,
    'mu_max_depth':          3,
    'mu_min_child_samples':  132,
    'mu_subsample':          0.649,
    'mu_colsample_bytree':   0.255,
    'mu_reg_alpha':          0.00576,
    'mu_reg_lambda':         0.00155,
    # ── π (zero prob, 5개) ──
    'pi_n_estimators':       125,
    'pi_learning_rate':      0.0408,
    'pi_num_leaves':         165,
    'pi_max_depth':          11,
    'pi_min_child_samples':  38,
    # ── φ (dispersion, 5개) ──
    'phi_n_estimators':      57,
    'phi_learning_rate':     0.00628,
    'phi_num_leaves':        65,
    'phi_max_depth':         4,
    'phi_min_child_samples': 190,
}
```

**참고 — 상위 5 trial** (anchor 신뢰도 검증):

| trial | oof | val | test | τ_π |
|---|---|---|---|---|
| 76 | 0.005498 | **0.005703** | **0.008407** | 0.886 |
| 94 | 0.005496 | 0.005704 | 0.008408 | 0.949 |
| 75 | 0.005496 | 0.005706 | 0.008411 | 0.974 |
| 78 | 0.005500 | 0.005705 | 0.008408 | 0.917 |
| 99 | 0.005496 | 0.005707 | 0.008410 | 0.944 |

trial 99(공식 best)는 OOF 기준 선정. trial 76이 val 더 낮지만 anchor는 trial 99로 가되 narrow range 안에 76이 들어오도록 폭 확보.

> **DB 저장 에러로 trial 76의 HP 값은 알 수 없음**. anchor=trial 99 단일 사용.

---

## 5. Anchor — 02_bag_zit HP

[zit_bag_hpo.txt](../../모델링_이전자료/4_output_이전자료/final/zit_only/zit_bag_hpo.txt) best:
- OOF=0.005497, val=0.005705, test=0.008410, τ_π=0.8246
- trial history는 txt에 없음 (best 1개만 저장)

```python
BAG_ZIT_ANCHOR = {
    # ── 모델 공통 ──
    'zeta':                  1.193,
    'n_em_iters':            18,
    # ── μ (9개) ──
    'mu_n_estimators':       103,
    'mu_learning_rate':      0.00658,
    'mu_num_leaves':         147,
    'mu_max_depth':          3,
    'mu_min_child_samples':  99,
    'mu_subsample':          0.686,
    'mu_colsample_bytree':   0.357,
    'mu_reg_alpha':          2.74e-05,
    'mu_reg_lambda':         0.0170,
    # ── π (5개) ──
    'pi_n_estimators':       348,
    'pi_learning_rate':      0.0256,
    'pi_num_leaves':         94,
    'pi_max_depth':          12,
    'pi_min_child_samples':  53,
    # ── φ (5개) ──
    'phi_n_estimators':      185,
    'phi_learning_rate':     0.00631,
    'phi_num_leaves':        68,
    'phi_max_depth':         3,
    'phi_min_child_samples': 128,
}
```

zit_only와 비교 시 차이:
- `n_em_iters` 13→18 (BagZIT는 EM 더 오래)
- `mu_n_estimators` 240→103 (μ 트리 절반 미만)
- `pi_n_estimators` 125→348 (π 트리 거의 3배)
- `mu_learning_rate` 약 2배

→ 두 모델은 다른 HP 영역에서 best. 각자 anchor로 따로 narrow 탐색.

---

## 6. HPO 탐색 범위 — Anchor + Narrow

[strategy_common.md §5](../strategy_common.md): 연속형 anchor의 ±30% (log uniform이면 log-space).

**zit_only 탐색 범위**:

| HP | type | anchor | range |
|---|---|---|---|
| `zeta` | float | 1.149 | [1.05, 1.50] (Tweedie 유효 범위 내) |
| `n_em_iters` | int | 13 | [10, 20] |
| `mu_n_estimators` | int | 240 | [170, 320] |
| `mu_learning_rate` | float-log | 0.00309 | [0.0021, 0.0046] |
| `mu_num_leaves` | int | 212 | [148, 280] |
| `mu_max_depth` | int | 3 | [3, 5] (anchor가 floor라 편향 확장) |
| `mu_min_child_samples` | int | 132 | [90, 180] |
| `mu_subsample` | float | 0.649 | [0.50, 0.85] |
| `mu_colsample_bytree` | float | 0.255 | [0.18, 0.35] |
| `mu_reg_alpha` | float-log | 0.00576 | [0.002, 0.015] |
| `mu_reg_lambda` | float-log | 0.00155 | [5e-4, 5e-3] |
| `pi_n_estimators` | int | 125 | [90, 175] |
| `pi_learning_rate` | float-log | 0.0408 | [0.028, 0.060] |
| `pi_num_leaves` | int | 165 | [115, 220] |
| `pi_max_depth` | int | 11 | [8, 13] |
| `pi_min_child_samples` | int | 38 | [25, 55] |
| `phi_n_estimators` | int | 57 | [40, 80] |
| `phi_learning_rate` | float-log | 0.00628 | [0.004, 0.010] |
| `phi_num_leaves` | int | 65 | [45, 90] |
| `phi_max_depth` | int | 4 | [3, 6] |
| `phi_min_child_samples` | int | 190 | [130, 260] |

**bag_zit 탐색 범위**: 동일 ±30% 규칙으로 [BAG_ZIT_ANCHOR](#5-anchor--02_bag_zit-hp) 기반 적용. 노트북 작성 시 anchor dict에서 자동 산출하는 helper 사용 권장 (수작업 누락 방지).

```python
def narrow_around(anchor: dict, log_keys: set) -> dict:
    """anchor ±30% 자동 산출. log_keys는 log-uniform으로 처리."""
    ...
```

---

## 7. Optuna 설정

| 항목 | 값 | 비고 |
|---|---|---|
| `N_TRIALS` | **80** | narrow space + 22 HP, anchor 안정 영역. 100보다 적게 |
| `N_FOLDS` | 5 | strategy_common §6 (unit-level KFold) |
| `sampler` | TPESampler(seed=None, multivariate=True, group=True) | strategy_common §4 |
| `pruner` | MedianPruner(n_warmup_steps=10) | EM 수렴 모니터링용 |
| `direction` | 'minimize' | OOF unit RMSE |
| `study_name` | `zit_only` / `bag_zit` | 기존 `4_output/final/`은 `모델링_이전자료/4_output_이전자료/`로 백업되므로 별도 suffix 불필요 |

`zeta`는 Tweedie compound Poisson-Gamma 유효 범위가 (1, 2)이므로 [1.05, 1.95] hard limit 내에서만.

---

## 8. 후처리 적용 매트릭스

[strategy_common.md §9~12](../strategy_common.md) 준수. ZIT 한정 SKIP/APPLY:

| 룰 | 적용? | 비고 |
|---|---|---|
| 분류 threshold (§9) | **SKIP** | `τ_π`가 die-level threshold 역할 — ZIT 모델 내부에서 처리 |
| die→unit 집계 다양성 (§10) | **APPLY** | mean/median/max/min/trimmed_mean/weighted/Q25/Q75 8후보 |
| Position 가중치 (§11) | **APPLY** | Optuna sub-study 50 trial로 w1~w4 |
| zero_clip (§12) | **APPLY** | 0.001~0.015 step 0.001 |

`τ_π`는 별도 HP로 ZIT search space에 포함 (위 §6 anchor에 포함시키거나 별도 sub-search). 기존 best τ_π:
- zit_only: 0.944
- bag_zit: 0.825

→ τ_π range도 anchor 기준 narrow ([anchor-0.10, min(1.0, anchor+0.05)]) 권장.

---

## 9. 출력 경로

| 노트북 | OUT_DIR |
|---|---|
| 01_zit_only | `4_output/01_zit/zit_only/` |
| 02_bag_zit | `4_output/01_zit/bag_zit/` |

산출물 9개 + 메타: [strategy_common.md §15](../strategy_common.md) 참조.

기존 1차 결과(`모델링_이전자료/4_output_이전자료/final/zit_only/`)는 `모델링_이전자료/4_output_이전자료/`로 백업됐고, 신규 결과는 위 미러링 경로에 새로 생성된다.

---

## 10. 실행 순서

1. `01_zit_only.ipynb` 실행 — 80 trial × 5 fold
2. `02_bag_zit.ipynb` 실행 — 80 trial × 5 fold

**병렬화**: 두 노트북은 의존 없음. 학원 14 코어 기준 [strategy_common.md §8](../strategy_common.md) 따라:

| 시나리오 | N_JOBS (각 노트북) | 코어 사용 |
|---|---|---|
| 두 노트북 병렬 (권장) | **7** | 14 |
| 한 노트북씩 순차 | 14 | 14 (단일) |

BagZIT가 EM iter 더 많음(18 vs 13) — 시간 더 걸림 → 먼저 시작 권장.

**elapsed 추정** (anchor 기준 trial당, N_JOBS=7 가정):
- zit_only: ~17분/trial → 80 trial × 5fold ≈ 23시간
- bag_zit: ~25분/trial → 80 trial × 5fold ≈ 33시간

→ trial 예산 80은 23~33시간 단위. 시간 부족하면 50 trial로 축소 가능.

---

## 11. 검증 — 노트북 실행 시 확인 사항

[strategy_common.md §13](../strategy_common.md) 구현 문제 보고 원칙 준수.

특히 ZIT 관련:
- `BagZITboostRegressor.fit(X, y, unit_id)` 시그니처 — `unit_id` 인자 필수. `hpo.refit_best`가 이 인자 전달하는지 확인.
- `τ_π` 적용 위치 — die-level pred에 적용 후 unit 집계 (기존 코드 기준)
- EM 수렴 여부 — `em_history_`에서 unit_rmse 단조감소 확인. 발산하면 학습 중단
