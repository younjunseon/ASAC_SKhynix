"""squeeze_extreme_v4 — die-level stacking 파이프라인 + SHAP X-stacking.

v3 (_v3/) 대비 변경점
--------------------
- **SHAP 통합 부활**: build_shap_features.py가 생성한 die-level SHAP 캐시(`die_shap.npz`)를
  base 예측 행렬에 die-level 컬럼으로 stack. v2의 `--extra-cache`와 동일한 컨셉을 die-level로
  재구현. unit-level parquet은 사용하지 않음.
- **2가지 SHAP 모드**:
  * `always_include` : 메타 학습 시 항상 입력. subset search는 base 모델만 후보로 (v2 기본).
  * `searchable`     : base와 동등하게 subset search 후보 풀에 등록. ridge/L1이 자동 선택.
- **die_shap.npz의 (ufs_serial, run_wf_xy) 키로 bundle 정렬 일치 확보** —
  build_shap_features.py가 npz에 `oof_run_wf_xy / val_run_wf_xy / test_run_wf_xy`를 함께 저장하므로
  shap.py가 그걸 읽어 (ufs_serial, run_wf_xy) 키로 bundle의 build_die_matrix 결과와 정렬 일치.

v3에서 그대로 가져온 것
----------------------
- die-level stacking, GroupKFold by ufs_serial
- 메타 학습기 5종 (ridge/nnls/mean/ENet/Combo) + iso 후처리
- postprocess.tune_and_apply 기반 die→unit 집계
- SqueezeV4Config dataclass (구 v3의 SqueezeV3Config 확장)

모듈 구성
--------
- config     : SqueezeV4Config — SHAP 관련 필드 4개 추가 (shap_caches, shap_top_k, shap_mode, shap_prefix_with_tag)
- records    : Record + n_extra/extra_tags 필드 추가
- discovery  : 기존 die-level 모델 탐색 + 행렬 구성 (SHAP 컬럼은 shap.py가 attach)
- shap       : (신규) die_shap.npz 로더 + (ufs_serial, run_wf_xy) 정렬 + top-K + tag prefix
- meta       : v3와 동일
- weights    : SHAP 컬럼 가중치 박제 포함
- search     : ArrayBundle에 extra_idx 추가, always_include 모드는 모든 subset에 자동 합집합
- aggregate  : v3와 동일 (postprocess.tune_and_apply 호출 래퍼)
- io         : summary.json에 shap 메타 추가

모든 모듈은 `from _v4 import config, ...` 형태로 import (노트북 기준).
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
