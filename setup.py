"""
Colab / Local 공통 부트스트랩

노트북 첫 셀에서 아래 코드로 모든 설정 완료:

    ──── 로컬 ────
    %run ../setup.py

    ──── Colab ────
    !git clone https://github.com/<REPO>.git /content/project  # 또는 아래 방식
    # !pip install gdown && gdown --id <REPO_ZIP_ID> -O /content/project.zip && unzip ...
    import sys; sys.path.insert(0, "/content/project")
    %run /content/project/setup.py
"""
import sys
import os
import subprocess

# ─── 필수 패키지 자동 설치 (requirements.txt 기반) ─────────
def _ensure_packages():
    # requirements.txt 경로 찾기
    _setup_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    req_path = os.path.join(_setup_dir, "requirements.txt")

    if not os.path.exists(req_path):
        return

    # 패키지명 → import명 매핑 (pip명과 import명이 다른 경우)
    _import_map = {
        "scikit-learn": "sklearn",
        "lightgbm": "lightgbm",
        "Pillow": "PIL",
    }

    missing = []
    with open(req_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            pkg = line.split(">=")[0].split("==")[0].split("<")[0].strip()
            import_name = _import_map.get(pkg, pkg.replace("-", "_"))
            try:
                __import__(import_name)
            except ImportError:
                missing.append(line)  # 버전 조건 포함해서 설치

    if missing:
        print(f"패키지 설치 중: {[m.split('>=')[0] for m in missing]}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q"] + missing
        )

_ensure_packages()

# ─── 프로젝트 루트 결정 ──────────────────────────────────
try:
    import google.colab  # noqa: F401
    _this = "/content/project"
    # Colab: utils가 없으면 git clone 안내
    if not os.path.exists(os.path.join(_this, "utils")):
        raise FileNotFoundError(
            "utils 폴더가 없습니다. 먼저 프로젝트를 /content/project 에 배치하세요.\n"
            "예: !git clone <REPO_URL> /content/project"
        )
except ImportError:
    # Local: setup.py의 위치 = 프로젝트 루트
    _this = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

if _this not in sys.path:
    sys.path.insert(0, _this)

# 2_preprocessing 모듈 경로 추가 (final/modules/cleaning.py 가
# `from meta_features import parse_run_wf_xy` 를 수행하기 때문)
_pp_dir = os.path.join(_this, "2_preprocessing")
if os.path.isdir(_pp_dir) and _pp_dir not in sys.path:
    sys.path.insert(0, _pp_dir)

# ─── matplotlib 한글 설정 ────────────────────────────────
import matplotlib
import matplotlib.pyplot as plt

from utils.config import ENV

if ENV == "colab":
    # Colab: 나눔고딕 설치 후 캐시 초기화
    os.system("apt-get -qq -y install fonts-nanum > /dev/null 2>&1")
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    if os.path.exists(font_path):
        import matplotlib.font_manager as fm
        fm.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = "NanumGothic"
else:
    plt.rcParams["font.family"] = "Malgun Gothic"

plt.rcParams["axes.unicode_minus"] = False

# ─── EDA 시각화 스타일 적용 ────────────────────────────────
_style_path = os.path.join(_this, "1_eda", "eda_style.mplstyle")
if os.path.exists(_style_path):
    plt.style.use(_style_path)
    # 스타일 적용 후 한글 폰트 재설정 (style.use가 font.family를 덮어쓰므로)
    if ENV == "colab":
        plt.rcParams["font.family"] = "NanumGothic"
    else:
        plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

# U+2212 (−) 글리프 경고 방지: matplotlib 내부 로거 필터링
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("matplotlib.mathtext").setLevel(logging.ERROR)

print("setup 완료")