# Wafer Test 결과를 통한 고객 Field Health Data 예측

SK하이닉스 기업연계 프로젝트 — WT(Wafer Test) 데이터 기반 Field Health(RCC) 예측 모델

## 빠른 시작

### 1. 클론

```bash
git clone https://github.com/<REPO>.git
cd 기업연계프로젝트
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

`dataset.zip`이 있다면 `0_data/`에 넣고 압축 해제하면 됩니다.

### 4. 노트북 실행

모든 노트북의 **첫 셀**에서:

```python
%run ../setup.py
```

이 한 줄로 패키지 설치, 경로 설정, 한글 폰트, 유틸리티 함수가 모두 준비됩니다.

## Colab에서 실행

```python
# 셀 1: 프로젝트 가져오기
!git clone https://github.com/<REPO>.git /content/project

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

## utils 모듈 사용법

`%run ../setup.py` 실행 후 바로 사용 가능합니다.

### 데이터 로드

```python
xs, ys = load_all()                    # Xs + Ys 한 번에 로드
feat_cols = get_feat_cols(xs)          # ["X0", "X1", ..., "X1086"]
xs_dict = split_xs(xs)                 # {"train": ..., "validation": ..., "test": ...}
ys_train = ys["train"]                 # train Y만
```

### die → unit 집계

```python
# 기본 (mean, std, min, max)
unit_df = aggregate_to_unit(xs_dict["train"])

# 집계 함수 지정
unit_df = aggregate_to_unit(xs_dict["train"], agg_funcs=["mean", "std", "range"])

# position별 피벗 포함
unit_df = build_unit_features(xs_dict["train"], agg_funcs=["mean", "std"], use_position_pivot=True)

# target merge → 바로 모델에 넣을 수 있는 X, y
X_train, y_train = merge_with_target(unit_df, split="train")
```

### 모델 평가

```python
evaluate(y_val, model.predict(X_val), label="LightGBM v1")
# [LightGBM v1] RMSE = 0.007832  (n=8,749, zero=6,193(70.8%))

compare_models({
    "LightGBM": lgbm_pred,
    "XGBoost": xgb_pred,
    "Ridge": ridge_pred,
}, y_val)
```

### 주요 상수

| 상수 | 값 | 용도 |
|------|-----|------|
| `SEED` | `42` | 랜덤 시드 |
| `TARGET_COL` | `"health"` | Y 컬럼명 |
| `KEY_COL` | `"ufs_serial"` | unit 매핑 키 |
| `META_COLS` | `["ufs_serial", "run_wf_xy", "position", "split"]` | 메타 컬럼 |
| `DATA_DIR` | `{프로젝트루트}/0_data` | 데이터 경로 |

## 의존 패키지

`requirements.txt` 참조. 주요 패키지:

- pandas, numpy, matplotlib, seaborn
- scikit-learn, lightgbm, xgboost
- optuna, boruta
- gdown (Colab 데이터 다운로드용)

## 팀원

- SK하이닉스 기업연계 ASAC 팀