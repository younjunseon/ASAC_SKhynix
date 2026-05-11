"""
Colab / Local 공통 부트스트랩.

노트북 첫 셀에서 `%run ../setup.py` (Colab은 절대경로) 한 줄이면 끝나도록, 이 파일을 import(=실행)하면:
  1) requirements.txt를 보고 빠진 패키지를 pip로 설치
  2) 프로젝트 루트를 정해 sys.path에 등록 (+ 2_preprocessing도 경로에 추가)
  3) matplotlib 한글 폰트 설정 (Colab=NanumGothic, Local=Malgun Gothic)
  4) 1_eda/eda_style.mplstyle 가 있으면 시각화 스타일 적용
  5) 폰트 관련 잡 경고 로거를 죽임
까지 한 번에 처리한다.

사용 예:
    [로컬]
    %run ../setup.py

    [Colab]
    !git clone https://github.com/<REPO>.git /content/project   # 또는 zip 다운로드 방식
    import sys; sys.path.insert(0, "/content/project")
    %run /content/project/setup.py
"""
import sys
import os
import subprocess


def _ensure_packages():
    """requirements.txt에 적힌 패키지 중 import가 안 되는 것만 골라 pip로 설치한다."""
    # requirements.txt는 setup.py와 같은 폴더(=프로젝트 루트)에 있다 (%run 환경에선 __file__이 없을 수 있어 cwd로 폴백)
    _setup_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    req_path = os.path.join(_setup_dir, "requirements.txt")

    if not os.path.exists(req_path):
        return   # requirements.txt가 없으면 설치 단계 자체를 건너뜀

    # pip 패키지명과 실제 import 이름이 다른 경우의 매핑 (예: pip install scikit-learn → import sklearn)
    _import_map = {
        "scikit-learn": "sklearn",
        "lightgbm": "lightgbm",
        "Pillow": "PIL",
    }

    missing = []
    with open(req_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):   # 빈 줄·주석 줄은 건너뜀
                continue
            # "lightgbm>=4.0" 같은 줄에서 패키지명만 떼어냄 (>=, ==, < 를 순서대로 잘라냄 — ~=,!= 등은 미처리지만 결과는 사실상 동일)
            pkg = line.split(">=")[0].split("==")[0].split("<")[0].strip()
            import_name = _import_map.get(pkg, pkg.replace("-", "_"))   # 매핑에 없으면 '-'→'_' 추정
            try:
                __import__(import_name)
            except ImportError:
                missing.append(line)   # 버전 조건이 붙은 원본 줄 그대로 설치 목록에 추가

    if missing:
        print(f"패키지 설치 중: {[m.split('>=')[0] for m in missing]}")
        # 현재 파이썬의 pip로 조용히(-q) 설치 — check_call이라 실패하면 예외로 멈춤
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q"] + missing
        )

_ensure_packages()

# --- 프로젝트 루트 결정 + sys.path 등록 ---
try:
    import google.colab  # noqa: F401
    _this = "/content/project"   # Colab은 부트스트랩 셀이 여기에 코드를 풀어 둔다는 약속
    # 그런데 utils 폴더가 없다 = 코드가 아직 안 풀렸다 → 무엇을 해야 하는지 알려주고 멈춤
    if not os.path.exists(os.path.join(_this, "utils")):
        raise FileNotFoundError(
            "utils 폴더가 없습니다. 먼저 프로젝트를 /content/project 에 배치하세요.\n"
            "예: !git clone <REPO_URL> /content/project"
        )
except ImportError:
    # 로컬은 이 setup.py가 놓인 폴더가 곧 프로젝트 루트
    _this = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

if _this not in sys.path:
    sys.path.insert(0, _this)

# 2_preprocessing도 경로에 추가 — cleaning.py가 `from meta_features import ...`처럼
# 패키지 접두사 없이 import 하기 때문에 이 폴더가 sys.path에 있어야 동작한다
_pp_dir = os.path.join(_this, "2_preprocessing")
if os.path.isdir(_pp_dir) and _pp_dir not in sys.path:
    sys.path.insert(0, _pp_dir)

# --- matplotlib 한글 폰트 ---
import matplotlib
import matplotlib.pyplot as plt

from utils.config import ENV   # config가 import 시점에 환경 감지를 끝내 ENV에 담아 둠

if ENV == "colab":
    # Colab 기본 이미지에는 한글 폰트가 없으므로 나눔고딕을 설치해서 등록
    os.system("apt-get -qq -y install fonts-nanum > /dev/null 2>&1")
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    if os.path.exists(font_path):
        import matplotlib.font_manager as fm
        fm.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = "NanumGothic"
else:
    plt.rcParams["font.family"] = "Malgun Gothic"   # Windows 기본 한글 폰트

plt.rcParams["axes.unicode_minus"] = False   # 축 눈금의 음수 기호가 깨지지 않게 (유니코드 − 대신 ASCII -)

# --- EDA 공용 시각화 스타일 ---
_style_path = os.path.join(_this, "1_eda", "eda_style.mplstyle")
if os.path.exists(_style_path):
    plt.style.use(_style_path)
    # style.use()가 rcParams를 통째로 덮어써 font.family까지 날아가므로, 폰트 설정을 다시 적용
    if ENV == "colab":
        plt.rcParams["font.family"] = "NanumGothic"
    else:
        plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

# 한글/수식 폰트에 일부 글리프(U+2212 등)가 없을 때 뜨는 경고가 시끄러우므로 해당 로거만 ERROR로 올려 숨김
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("matplotlib.mathtext").setLevel(logging.ERROR)

print("setup 완료")
