"""stacking_lib — die-level stacking 파이프라인 + SHAP X-stacking.

base 모델들의 die-level 예측(`*_die.csv`)을 GroupKFold(ufs_serial)로 메타 학습해
unit RMSE를 최소화한다. die-level SHAP 캐시(`die_shap.npz`)를 입력 행렬에 컬럼으로 덧붙이는
SHAP X-stacking을 지원한다.

핵심
----
- die-level stacking, GroupKFold by ufs_serial (같은 unit의 4 die가 train/val에 안 섞이게)
- 메타 학습기 5종 (ridge / nnls / mean / ENet / Combo) + iso 후처리
- postprocess.tune_and_apply 기반 die→unit 집계 (position 가중 포함)
- die_shap.npz를 (ufs_serial, run_wf_xy) 키로 정렬 일치시켜 base 행렬에 stack:
  build_shap_features.py가 npz에 `oof/val/test_run_wf_xy`를 함께 저장 → shap.py가 그 키로
  build_die_matrix 결과와 행 순서를 맞춘다.
  * `always_include` : 메타 학습 시 항상 입력. subset search는 base 모델만 후보.
  * `searchable`     : base와 동등하게 subset search 후보 풀에 등록 (ridge/L1이 자동 선택).

모듈 구성
--------
- config     : StackingConfig — 모든 파라미터 dataclass (SHAP 필드 포함)
- records    : Record + score_rec (die/unit RMSE 별도 보관)
- discovery  : die-level 모델 탐색(4_output rglob) + 행렬 구성 (SHAP 컬럼은 shap.py가 attach)
- shap       : die_shap.npz 로더 + (ufs_serial, run_wf_xy) 정렬 + top-K + tag prefix
- meta       : 메타 학습기
- weights    : 최종 가중치 박제 (SHAP 컬럼 포함)
- search     : 탐색 단계 (seed → random → local → optuna) + refit
- aggregate  : postprocess.tune_and_apply 호출 래퍼 (die→unit)
- io         : 결과 저장 (summary.json 등)

모든 모듈은 `from stacking_lib import config, ...` 형태로 import (노트북 기준).
"""
from . import aggregate, config, discovery, io, meta, records, search, shap, weights

__all__ = [
    "aggregate",
    "config",
    "discovery",
    "io",
    "meta",
    "records",
    "search",
    "shap",
    "weights",
]
