# 4_output/ — 모델링 산출물

모델링 트랙(`3_modeling/`)이 생성하는 예측·모델·실험 기록 저장소.

> ⚠️ **대용량 실물은 git에 올리지 않는다.** 예측 CSV·`fold_models.pkl`·`optuna_*.db`·
> `best_params.json` 등은 `.gitignore`로 제외되며, **이 README만 추적**한다.
> 실물 결과는 아카이브에서 참고한다 → `4_output.zip` (Google Drive ID `1ts73qEMmjX8cKIb-QeDQ-TMeyudFGWzs`, ~2.5GB, RESUME/재현용).

## 폴더 구조

```
4_output/
├── 01_zit/                     # ZITboost 4조합 (ZI-Tweedie + LightGBM, EM)
│   ├── zit_only_pearson/best/  # φ=Pearson 잔차 추정
│   ├── zit_only_eql/best/      # φ=extended quasi-likelihood (논문충실)
│   ├── bag_zit_pearson/best/   # + unit bag 제약
│   └── bag_zit_eql/best/
│       └── best_params.json · summary_record.json · (die/unit CSV·pkl)
├── 02_reg_single/              # 단일 회귀 5종
│   ├── lgbm/ · xgb/ · catboost/ · et/ · enet/
│   │   └── best_params.json · fold_models.pkl
├── 03_two_stage/               # Two-Stage (분류 → 회귀)
│   ├── default/{clf, reg, combined}/
│   └── reverse/                # 회귀→집계 경로
│       └── best_params.json · fold_models.pkl
├── 04_stacking/                # 메타 스태킹 (die-level, GroupKFold)
│   └── run_0620_185039/
│       └── summary.json · best_weights.json
├── 0_baseline/                 # 초기 스크리닝 (LEGACY)
│   ├── oat/        # one-at-a-time 전처리 비교 (checkpoint.json · meta.json)
│   ├── group/      # 그룹 스터디 (meta.json)
│   ├── default_compare/
│   └── summary/    # tornado.png · group_importance.png · summary_report.png
└── experiments/experiments.csv # 실험 요약 테이블 (현재 비어 있음)
```

## 산출물 종류 (모두 `.gitignore` 제외 — 아카이브 참조)

| 파일 | 내용 |
|------|------|
| `*_die.csv` / `*_unit.csv` | die-level / unit-level 예측 (OOF·val·test) |
| `fold_models.pkl` | fold별 학습 모델 (refit 결과) |
| `best_params.json` | best 하이퍼파라미터 + val/test RMSE |
| `summary_record.json` / `summary.json` | 트랙 요약 (지표·설정) |
| `optuna_*.db` | Optuna trial 상세 (SQLite, RESUME 가능) |
| `*.png` | 0_baseline 요약 차트 |

## 결과 요약 (unit-level RMSE — 대회 지표)

각 트랙 `best/`의 `summary_record.json` / `summary.json` 스냅샷 기준. 전체 trial·fold 상세는 아카이브 참조.

| 트랙 | 모델 | val RMSE | test RMSE |
|------|------|---------:|----------:|
| 01_zit | zit_only_pearson | **0.005986** | 0.005346 |
| 01_zit | bag_zit_pearson | 0.006093 | **0.005293** |
| 01_zit | zit_only_eql | 0.006138 | 0.005334 |
| 01_zit | bag_zit_eql | 0.006168 | 0.005475 |
| 04_stacking | best blend | oof 0.006906 | — |
| 02_reg_single | lgbm/xgb/cat/et/enet | (아카이브 참조) | (아카이브 참조) |
| 03_two_stage | default · reverse | (아카이브 참조) | (아카이브 참조) |

- 최저 **val** = `zit_only_pearson` (0.005986), 최저 **test** = `bag_zit_pearson` (0.005293)
- ZITboost(01_zit) 계열이 현재 단일 트랙 최고 성능. 스태킹은 die-level OOF 기준이라 직접 비교 시 집계 단계 차이 유의.

## 재생성 / 재현

- 트랙별 `hpo.py` / `fit.ipynb` 실행 (`3_modeling/<track>/`). 첫 셀 `RESUME` 플래그로 기존 `optuna_*.db`에 이어서 가능.
- 아카이브에서 복원: `4_output.zip` 압축 해제 → `4_output/`. (Colab은 `RESUME=True` 시 자동 다운로드)
