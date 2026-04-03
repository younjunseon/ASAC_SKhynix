# 1_eda/modules — EDA 분석 모듈 패키지
#
# Phase 1: 데이터 기본 파악
#   eda_overview (ov)         - 데이터 구조, target 분포, split 검증
#
# Phase 2: Feature 품질 검사
#   eda_feature_quality (fq)  - 결측치, 분포, 분산, 유형 분류
#
# Phase 3: 이상치 및 스케일
#   eda_outlier_scale (out)   - IQR 이상치, 스케일 불균형
#   eda_outlier_methods (om)  - 4가지 이상치 탐지법 비교 (IQR/Winsor/AEC DPAT/MAD)
#
# Phase 4: Feature-Target 관계
#   eda_relationships (rel)   - Pearson 상관, 다중공선성
#   eda_nonlinear (nl)        - Spearman + Mutual Information
#   eda_interaction (ia)      - 2-way 상호작용 + multi-way split 분석
#
# Phase 5: Target 심층 분석
#   eda_group_compare (gc)    - Y=0 vs Y>0 통계 검정
#   eda_zero_structure (zs)   - Y=0 내부 이질성 (클러스터링, 경계 분석)
#   eda_target_segment (ts)   - health 구간별 세분화 분석
#
# Phase 6: 공간/구조 분석
#   eda_wafer_map (wm)        - 웨이퍼 맵 시각화
#   eda_lot_wafer (lw)        - 로트/웨이퍼 품질 분석
#   eda_lot_normalize (ln)    - 로트 정규화 효과 분석
#   eda_position (pos)        - die position 분석
#   eda_spatial (sp)          - 공간 패턴 (radial, ring, NNR)
#   eda_spatial_residual (sr) - NNR 공간 잔차 피처 분석
#   eda_neighbor_die (nd)     - 인접 die 불량 연관성, Moran's I
#
# Phase 7: 집계 전략
#   eda_agg_compare (agg)     - 7가지 집계 방법 비교
