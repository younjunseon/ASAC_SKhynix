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
- Row 수: **174,980** (die level)
- 변수: `ufs_serial`, `run_wf_xy`, `position`, `split`, `X0 ~ X1086`

### 종속변수
- Row 수: **43,745** (unit level)
- 분포: `Y = 0` **70.80%** / `Y > 0` **29.20%**

### 변수 설명
- `ufs_serial` : unit명, Y data의 mapping key
- `run_wf_xy` : die 구분 key (Lot / Wafer / Die x, y position)
- `position` : 해당 die의 unit 내 위치
- `split` : Train / Validation / Test 자료 구분
- `X0 ~ X1086` : Wafer Test Data (비식별화)

> 설명변수는 배경지식으로 인한 편향을 최소화하기 위해 비식별화되어 제공됩니다.

## 빠른 시작

### 1. 클론

```bash
git clone https://github.com/younjunseon/ASAC_SKhynix.git
cd ASAC_SKhynix
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

또는 노트북 첫 셀에서 `%run ../setup.py` 실행하면 **자동 설치**됩니다.

### 3. 데이터 배치

CSV 파일(비공개)을 `0_data/` 폴더에 넣어주세요:

```
0_data/
├── compet_xs_data.csv           # 설명변수 (1.2GB)
├── compet_ys_train_data.csv     # 종속변수 train
├── compet_ys_validation_data.csv
└── compet_ys_test_data.csv
```

### 4. 노트북 실행

모든 노트북의 **첫 셀**에서:

```python
%run ../setup.py
```

이 한 줄로 패키지 설치, 경로 설정, 한글 폰트, 유틸리티 함수가 모두 준비됩니다.

## Colab에서 실행

```python
# 셀 1: 프로젝트 가져오기
!git clone https://github.com/younjunseon/ASAC_SKhynix.git /content/project

# 셀 2: 셋업 (패키지 설치 + 데이터 자동 다운로드)
import sys; sys.path.insert(0, "/content/project")
%run /content/project/setup.py
```

Colab에서는 Google Drive의 `dataset.zip`을 자동 다운로드합니다.

## 프로젝트 구조

```
├── setup.py               # 노트북 부트스트랩 (환경 자동 감지)
├── requirements.txt       # 의존 패키지 목록
├── utils/                 # 공통 유틸리티 모듈
│   ├── config.py          #   환경 감지, 경로, 상수
│   ├── data.py            #   데이터 로드, split 분리, 캐싱
│   ├── aggregate.py       #   die→unit 집계, position 피벗
│   └── evaluate.py        #   RMSE 평가, 모델 비교, 후처리
├── 0_data/                # 원본 데이터 (git 미포함)
├── 1_eda/                 # 탐색적 데이터 분석
├── 2_preprocessing/       # 전처리
├── 3_modeling/            # 모델링
├── 4_output/              # 예측 결과
└── 90_기획/               # 기획 문서
```

## 수행 계획

| 주차 | 내용 |
|---|---|
| 1~2주차 | 반도체 도메인 지식 학습 및 탐색적 데이터 분석 |
| 3~4주차 | Outlier Detection 및 치환 / Feature Engineering |
| 5~6주차 | Data Standardization & Encoding / Feature Selection |
| 7~8주차 | Cross Validation & Hyper Parameter Optimization |
| 9~10주차 | 예측 결과 모니터링 인터랙티브 대시보드 개발 |
| 11주차 | 자연어 질의 기반 AI Agent 설계 및 구현 / 최종 발표 준비 |

## 기술 스택
- Python, Pandas, NumPy
- Scikit-learn, XGBoost, LightGBM, CatBoost
- Matplotlib, Seaborn
- Optuna, Boruta
- Dashboard Framework (Streamlit / Dash)
- LLM 기반 AI Agent 연동

## 기대 효과
- 출하 전 리스크 사전 선별
- 품질 비용 절감
- 고객 신뢰도 향상
- 주요 불량 인자 파악

## 팀원
- SK하이닉스 기업연계 ASAC 팀
