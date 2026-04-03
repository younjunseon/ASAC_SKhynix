# Wafer Test 결과를 통한 고객 Field Health Data(RCC) 예측

## 프로젝트 개요
본 프로젝트는 **Wafer Test(WT) 단계의 데이터를 활용하여 고객 Field Health Data(RCC)를 예측**하는 것을 목표로 합니다.  
WT 단계에서 이미 나타나는 신호를 바탕으로, 향후 Field 불량 위험이 높은 unit을 사전에 식별하고, 불량에 영향을 주는 주요 인자를 찾아 공정 개선에 활용하고자 합니다.

## 프로젝트 목표
- WT 단계에서 **Field 불량 위험이 높은 unit을 사전에 식별**할 수 있는 예측 모델 구축
- 불량에 영향을 미치는 **주요 WT 인자 파악** 및 공정 개선 인사이트 도출
- **SK hynix 사내 최우수 성능 대비 10% 이상 향상** 달성

## 수행 내용
- 반도체 WT Data **EDA(탐색적 데이터 분석)**
- **Outlier Detection 및 치환**
- **Feature Engineering & Selection**
- **Data Standardization & One-Hot Encoding**
- **Machine Learning 회귀 모델링**
- **Cross Validation & Hyper Parameter Optimization**
- 예측 모델 결과 해석 및 최종 모델 구축
- 예측 결과 모니터링 **대시보드 개발**
- 자연어 질의 기반 대시보드 연동 **AI Agent 구현**

## 데이터 개요

### 설명변수
- Row 수: **174,980**
- 변수:
  - `ufs_serial`
  - `run_wf_xy`
  - `position`
  - `split`
  - `X0 ~ X1086`

### 종속변수
- Row 수: **43,745**
- 분포:
  - `Y = 0` : **70.80%**
  - `Y > 0` : **29.20%**

### 변수 설명
- `ufs_serial` : unit명, Y data의 mapping key
- `run_wf_xy` : die 구분 key (Lot / Wafer / Die x, y position)
- `position` : 해당 die의 unit 내 위치
- `split` : Train / Validation / Test 자료 구분
- `X0 ~ X1086` : Wafer Test Data (비식별화)

> 설명변수는 배경지식으로 인한 편향을 최소화하기 위해 비식별화되어 제공됩니다.

## 수행 계획

| 주차 | 내용 |
|---|---|
| 1~2주차 | 반도체 도메인 지식 학습 및 탐색적 데이터 분석 |
| 3~4주차 | Outlier Detection 및 치환 / Feature Engineering |
| 5~6주차 | Data Standardization & Encoding / Feature Selection |
| 7~8주차 | Cross Validation & Hyper Parameter Optimization |
| 9~10주차 | 예측 결과 모니터링 인터랙티브 대시보드 개발 |
| 11주차 | 자연어 질의 기반 AI Agent 설계 및 구현 / 최종 발표 준비 |

## 기대 효과
- 출하 전 리스크 사전 선별
- 품질 비용 절감
- 고객 신뢰도 향상
- 주요 불량 인자 파악

## 기술 스택 예시
- Python
- Pandas / NumPy
- Scikit-learn
- XGBoost / LightGBM
- Matplotlib / Seaborn
- Dashboard Framework (예: Streamlit, Dash)
- LLM 기반 AI Agent 연동

## 향후 확장 방향
- 모델 성능 고도화를 위한 고급 앙상블 기법 적용
- 공정/품질 담당자가 쉽게 활용할 수 있는 실시간 대시보드 개선
- 자연어 질의를 통한 예측 결과 조회 및 원인 분석 자동화

TEST