"""3_modeling/modules — 모델링 파이프라인 모듈 모음.

구성:
- preprocess : 전처리 래퍼 (cleaning + outlier를 한 번에, 파라미터 override 가능)
- scaler     : ElasticNet 한정 RobustScaler (트리 모델은 pass-through)
- models     : regressor/classifier 팩토리 + Optuna 탐색 공간 (xgb/catboost/lgbm/et/enet ...)
- hpo        : unit 단위 KFold OOF + Optuna HPO + best trial refit + 산출물 저장
- postprocess: die→unit 집계 여러 종 + π threshold + zero_clip + position 가중평균
- zit        : ZITboost (ZI-Tweedie + LightGBM, EM) regressor

cleaning / outlier 등 전처리 코드는 2_preprocessing/ 모듈을 그대로 import 해서 쓴다 (preprocess.py). scaling만은 modules/scaler.py로 따로 두었다 — ElasticNet 한정 RobustScaler.
"""
