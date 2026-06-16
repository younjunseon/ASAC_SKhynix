# 보고서 양식 설계

`report.py`의 `build_html()`로 생성되는 HTML 보고서 양식.

## 레이아웃 구조

```
┌───────────────────────────────────────────────────────────────┐
│  발행일자: YYYY.MM.DD   품 질 불 량 예 측 보 고 서    [대외비]  │
├───────────────────────────────────────────────────────────────┤
│  전주 대비 품질불량 ▲ ppm 이상  |  X1050, X350 특성 → Inline  │  ← 알림 배너 (전체 너비)
├──────────────────────────┬────────────────────────────────────┤
│  [ 모 델 링  현 황  보 고 ]│  [ 불 량  유 닛  분 석 현 황 ]    │
├──────────────────────────┼────────────────────────────────────┤
│ L1. 요약 3열              │ R1. 대표 Unit (2열 레이아웃)        │
│  · 모델 성능 (Val RMSE)   │  · 왼쪽: 웨이퍼맵 이미지 영역      │
│  · 분석 유닛              │  · 오른쪽: 시리얼/예측ppm/불량좌표  │
│  · 예측 ppm               │           검사일자/생산일자 테이블  │
│                           │                                    │
│ L2. 불량률 트렌드          │ R2. 3열 그리드                     │
│  · LOT별 HIGH 건수 라인   │  · 어노멀리 피처 (정상/위험 바 차트) │
│                           │  · 포지션별 불량률 (Position 1~4)   │
│ L3. SHAP 분석 (2열)       │  · 피처 임포턴스 비율               │
│  · 왼쪽: bee-swarm scatter│                                    │
│    (HIGH빨강/MED파랑)      │ R3. 피처 top-1 트렌드              │
│  · 오른쪽: Pred vs Actual │  · 빨강=HIGH, 초록=MED 라인차트     │
│    scatter                │                                    │
└──────────────────────────┴────────────────────────────────────┘
│              푸터 (날짜 · 모델 · RMSE)                         │
└───────────────────────────────────────────────────────────────┘
```

## 섹션별 데이터 소스

| 섹션 | 데이터 소스 | 상태 |
|------|-----------|------|
| 알림 배너 | `analysis.top_features` top-2 피처명 | ✅ 실데이터 |
| L1. 요약 3열 | `meta.val_rmse`, `scan.total_units`, `scan.high_ratio` | ✅ 실데이터 |
| L2. 불량률 트렌드 | `get_lot_trend_data()` | ✅ 실데이터 |
| L3. SHAP bee-swarm | `importance.features` (high_mean, low_mean) | ✅ 실데이터 |
| L3. Pred vs Actual | `analysis.pred_actual` | ✅ 실데이터 (없으면 더미) |
| R1. 대표 Unit 정보 | - | 🔶 더미 (추후 구현) |
| R2. 어노멀리 피처 | `analysis.top_features` (ratio 기반 위험도) | ✅ 실데이터 |
| R2. 포지션별 불량률 | - | 🔶 더미 (추후 구현) |
| R2. 피처 임포턴스 비율 | `importance.features` (lgbm_gain) | ✅ 실데이터 |
| R3. 피처 top-1 트렌드 | `analysis.trend_top1` — `get_trend_top1_data()` | ✅ 실데이터 |
| L3. Pred vs Actual | `analysis.pred_actual` — `get_pred_actual_data()` | ✅ 실데이터 |

## 더미 섹션 표시
더미 섹션 제목 옆에 노란 뱃지 `더미` 자동 표시.
```python
DUMMY_SECTIONS = {"대표 Unit 정보", "포지션별 불량률"}
```

## AI 수정 어시스턴트 (report_editor)
보고서 열람 중 우측 챗봇으로 수정 가능한 항목:

| 요청 예시 | 동작 |
|---------|------|
| "SHAP 상위 5개만" | `d["chart_params"] = {"chart":"shap","top_n":"5"}` |
| "LOT 트렌드 10개만" | `d["chart_params"] = {"chart":"lot","top_n":"10"}` |
| "왼쪽에 표 추가해줘" | `d["custom_sections"].append({..., "position":"left_col"})` |

## 파일 위치
- HTML 보고서: `report.py` → `build_html()`
- PPTX 보고서: `report.py` → `build_pptx()`
- AI 수정 에이전트: `agent.py` → `run_report_editor()`
