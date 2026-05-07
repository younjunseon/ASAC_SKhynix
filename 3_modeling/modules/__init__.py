"""final/modules — 제출용 파이프라인 모듈.

구성:
- preprocess : 전처리 (cleaning + outlier, 파라미터 override 가능)
- scaler     : enet 한정 RobustScaler (트리 모델은 pass-through)
- models     : 6종 regressor (xgb/catboost/lgbm/et/enet/zitboost) + search space
- hpo        : KFold OOF + Optuna + best trial refit
- postprocess: die→unit 집계 6종 + π threshold + zero_clip
- blending   : 3-path OOF 가중치 (SLSQP / Optuna)

복사본 (원본 2_preprocessing/ 및 3_modeling/modules/ 에서):
- cleaning, outlier, scaling (2_preprocessing)
- zit ← zi_tweedie (3_modeling/modules)
"""
