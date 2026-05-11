"""
프로젝트 공통 설정 및 경로 관리.

이 모듈은 import 되는 순간(아래 모듈 레벨 코드)에:
  1) 실행 환경이 Colab인지 로컬인지 감지하고
  2) 그에 맞는 PROJECT_ROOT / 데이터 경로 / 출력 경로 상수를 만든 뒤
  3) Colab이면 Google Drive에서 dataset.zip을 받아 풀고
  4) PROJECT_ROOT를 sys.path에 등록하고 필요한 디렉토리를 만들어 둔다.
그래서 노트북은 이 모듈만 import 하면 경로/시드/컬럼명을 별도 설정 없이 그대로 쓸 수 있다.
"""
import os
import sys

# dataset.zip(원본 CSV 4개)의 Google Drive 파일 ID — Colab에서 자동 다운로드할 때만 사용
GDRIVE_FILE_ID = "1yOUo0_wPLcuZBSJIK592b00YkSIlk4zO"


def _detect_env():
    """현재 실행 환경(Colab/Local)과 그에 맞는 프로젝트 루트 경로를 반환한다.

    google.colab 모듈이 import 되면 Colab, 안 되면 로컬로 판단한다.
    - Colab: 루트는 부트스트랩 셀이 zip을 풀어 둔 고정 경로 /content/project
    - Local: 루트는 이 파일(utils/config.py)의 두 단계 상위 디렉토리(= 프로젝트 폴더)
    """
    try:
        import google.colab  # noqa: F401  (존재 여부만 확인 — import 자체가 환경 감지)
        _env = "colab"
        _root = "/content/project"
    except ImportError:
        _env = "local"
        # __file__ = .../프로젝트/utils/config.py → dirname 두 번 = .../프로젝트
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return _env, _root


def _ensure_colab_data(data_dir):
    """Colab 세션에 원본 데이터가 없으면 Google Drive에서 dataset.zip을 받아 풀어 둔다.

    compet_xs_data.csv가 이미 있으면 받지 않고 즉시 반환(세션 재실행 시 중복 다운로드 방지).
    없으면: data_dir 생성 → gdown으로 zip 다운로드 → 압축 해제 → zip 삭제.
    """
    xs_path = os.path.join(data_dir, "compet_xs_data.csv")
    if os.path.exists(xs_path):
        return  # 이미 풀려 있음 — 아무것도 안 함

    print("Colab: 데이터 다운로드 중...")
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "dataset.zip")

    import gdown
    gdown.download(id=GDRIVE_FILE_ID, output=zip_path, quiet=False)

    print("압축 해제 중...")
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    os.remove(zip_path)  # CSV만 남기고 zip은 정리
    print(f"데이터 준비 완료 → {data_dir}")


# --- 여기부터는 import 시 1회 실행되는 모듈 레벨 초기화 ---

ENV, PROJECT_ROOT = _detect_env()   # 이후 모든 경로는 PROJECT_ROOT 기준 상대 조립

# 프로젝트 표준 디렉토리 (단계별 폴더 + 출력 폴더)
DATA_DIR = os.path.join(PROJECT_ROOT, "0_data")
EDA_DIR = os.path.join(PROJECT_ROOT, "1_eda")
PREP_DIR = os.path.join(PROJECT_ROOT, "2_preprocessing")
MODEL_DIR = os.path.join(PROJECT_ROOT, "3_modeling")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "4_output")
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")

# Colab이면 데이터가 준비됐는지 확인하고, 없으면 지금 받아 둔다 (로컬은 이미 0_data/에 있다고 가정)
if ENV == "colab":
    _ensure_colab_data(DATA_DIR)

# 원본 데이터 파일 경로 — 노트북은 이 상수를 쓰고 경로 문자열을 직접 쓰지 않는다
XS_PATH = os.path.join(DATA_DIR, "compet_xs_data.csv")            # X (die-level, X0~X1086)
YS_TRAIN_PATH = os.path.join(DATA_DIR, "compet_ys_train_data.csv")        # y train (unit-level, 공개)
YS_VAL_PATH = os.path.join(DATA_DIR, "compet_ys_validation_data.csv")   # y validation (제출용, 라벨 공개 X)
YS_TEST_PATH = os.path.join(DATA_DIR, "compet_ys_test_data.csv")         # y test (제출용, 라벨 공개 X)

# 공통 상수 — 시드와 컬럼명을 한 곳에서 관리해 노트북 전체가 동일한 이름/시드를 쓰게 함
SEED = 42                  # 모든 random_state / KFold 시드
TARGET_COL = "health"      # 예측 대상 (Field RCC, zero-inflated)
KEY_COL = "ufs_serial"     # unit 식별자 — X(die)와 y(unit)를 잇는 매핑 키
DIE_KEY_COL = "run_wf_xy"  # die 식별자 (Lot_Wafer_X_Y 형식)
POSITION_COL = "position"  # unit 내 die 위치 (1~4)
SPLIT_COL = "split"        # train/validation/test 구분 컬럼 (X에 들어 있음)
META_COLS = [KEY_COL, DIE_KEY_COL, POSITION_COL, SPLIT_COL]  # feature가 아닌 메타 컬럼 모음 (feature 목록에서 제외할 때 사용)

# PROJECT_ROOT를 import 경로에 등록 — 노트북이 sys.path 손대지 않고 `from utils...`, `from modules...` 가능하게
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 출력/전처리/모델 디렉토리가 없으면 만들어 둔다 (DATA_DIR/EDA_DIR는 이미 있다고 가정)
for _d in [PREP_DIR, MODEL_DIR, OUTPUT_DIR]:
    os.makedirs(_d, exist_ok=True)
