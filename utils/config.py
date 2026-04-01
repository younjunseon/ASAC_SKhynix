"""
프로젝트 공통 설정 및 경로 관리
- 로컬 / Colab 자동 감지
- Colab: Google Drive에서 dataset.zip 다운로드 & 압축 해제
- 공통 상수, 시드, 컬럼명 관리
"""
import os
import sys

# ─── Google Drive 파일 ID (dataset.zip) ───────────────────
GDRIVE_FILE_ID = "1yOUo0_wPLcuZBSJIK592b00YkSIlk4zO"

# ─── 환경 감지 ──────────────────────────────────────────────
def _detect_env():
    """Colab / Local 자동 감지 후 프로젝트 루트 반환"""
    try:
        import google.colab  # noqa: F401
        _env = "colab"
        _root = "/content/project"
    except ImportError:
        _env = "local"
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return _env, _root


def _ensure_colab_data(data_dir):
    """Colab 환경에서 데이터가 없으면 Google Drive에서 다운로드 & 압축 해제"""
    xs_path = os.path.join(data_dir, "compet_xs_data.csv")
    if os.path.exists(xs_path):
        return  # 이미 존재하면 스킵

    print("Colab: 데이터 다운로드 중...")
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "dataset.zip")

    import gdown
    gdown.download(id=GDRIVE_FILE_ID, output=zip_path, quiet=False)

    print("압축 해제 중...")
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    os.remove(zip_path)
    print(f"데이터 준비 완료 → {data_dir}")


ENV, PROJECT_ROOT = _detect_env()

# ─── 경로 ──────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "0_data")
EDA_DIR = os.path.join(PROJECT_ROOT, "1_eda")
PREP_DIR = os.path.join(PROJECT_ROOT, "2_preprocessing")
MODEL_DIR = os.path.join(PROJECT_ROOT, "3_modeling")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "4_output")
UTILS_DIR = os.path.join(PROJECT_ROOT, "utils")

# ─── Colab이면 데이터 자동 다운로드 ───────────────────────
if ENV == "colab":
    _ensure_colab_data(DATA_DIR)

# ─── 파일 경로 ─────────────────────────────────────────────
XS_PATH = os.path.join(DATA_DIR, "compet_xs_data.csv")
YS_TRAIN_PATH = os.path.join(DATA_DIR, "compet_ys_train_data.csv")
YS_VAL_PATH = os.path.join(DATA_DIR, "compet_ys_validation_data.csv")
YS_TEST_PATH = os.path.join(DATA_DIR, "compet_ys_test_data.csv")

# ─── 상수 ──────────────────────────────────────────────────
SEED = 42
TARGET_COL = "health"
KEY_COL = "ufs_serial"
DIE_KEY_COL = "run_wf_xy"
POSITION_COL = "position"
SPLIT_COL = "split"
META_COLS = [KEY_COL, DIE_KEY_COL, POSITION_COL, SPLIT_COL]

# ─── utils를 import path에 등록 ───────────────────────────
# 노트북에서 `sys.path` 걱정 없이 바로 사용 가능하게
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ─── 디렉토리 자동 생성 ───────────────────────────────────
for _d in [PREP_DIR, MODEL_DIR, OUTPUT_DIR]:
    os.makedirs(_d, exist_ok=True)
