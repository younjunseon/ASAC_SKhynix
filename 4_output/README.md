# 4_output

모델링 트랙(`3_modeling/`)을 돌리면 결과가 여기 저장된다.

예측값 CSV, 학습된 모델(`fold_models.pkl`), 하이퍼파라미터(`best_params.json`),
Optuna DB 같은 실제 파일은 용량이 커서 저장소에는 올리지 않는다(`.gitignore` 처리).
이 문서에는 폴더 구성만 적어 둔다. 파일 자체가 필요하면
`4_output.zip`(Google Drive ID `1ts73qEMmjX8cKIb-QeDQ-TMeyudFGWzs`)을 받아서 풀면 된다.

## 폴더 구성

```
4_output/
├── 01_zit/          ZITboost 4조합 (zit_only / bag × pearson / eql)
├── 02_reg_single/   단일 회귀 5종 (lgbm, xgb, catboost, et, enet)
├── 03_two_stage/    Two-Stage (default: clf + reg + combined / reverse)
├── 04_stacking/     메타 스태킹 (die-level, GroupKFold)
├── 0_baseline/      초기 스크리닝 (oat, group, default_compare, summary)
└── experiments/     실험 요약 CSV (지금은 비어 있음)
```

각 트랙 폴더에는 보통 `best_params.json`, `fold_models.pkl`, 예측 CSV(die/unit),
요약 JSON이 들어간다. `0_baseline/summary/`에는 요약 차트 PNG(tornado 등)가 있다.

## 다시 돌리기

트랙별 `hpo.py`나 `fit.ipynb`를 실행하면 된다. 첫 셀의 `RESUME`를 켜면
기존 `optuna_*.db`에 이어서 돌고, zip을 미리 풀어 두면 그 상태에서 이어진다.
