"""3_modeling/modules — 모델링 파이프라인 모듈 모음.

구성:
- preprocess : 전처리 래퍼 (cleaning + outlier를 한 번에, 파라미터 override 가능)
- models     : regressor/classifier 팩토리 + Optuna 탐색 공간 (xgb/catboost/lgbm/et/enet ...)
- hpo        : unit 단위 KFold OOF + Optuna HPO + best trial refit + 산출물 저장
- postprocess: die→unit 집계 여러 종 + π threshold + zero_clip + position 가중평균
- zit        : ZITboost (ZI-Tweedie + LightGBM, EM) regressor

cleaning / outlier 등 전처리 코드는 2_preprocessing/ 모듈을 그대로 import 해서 쓴다 (preprocess.py).
스케일링 변환의 정본도 2_preprocessing/scaling.py이며, "어떤 모델이 스케일링을 켜는가"라는
모델링 게이트(enet 한정)만 hpo.py 안에 _SCALING_REQUIRED 상수로 인라인되어 있다.
"""
