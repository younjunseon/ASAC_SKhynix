"""
실험 결과 기록 모듈

- xlsx: 실험별 핵심 지표 테이블 (1행 = 1실험)
- json: 실험별 상세 파라미터 (key = 실험번호)

사용법:
    from utils.experiment import log_experiment

    log_experiment(
        exp_id="1-1-001",
        exp_type="baseline",
        best_model="LightGBM",
        val_rmse=0.00456,
        test_rmse=0.00478,
        n_features=960,
        memo="mean only 집계",
        cleaning_params=cleaning_params,
        outlier_params=outlier_params,
        ...
    )
"""
import os
import sys
import json
import hashlib
from datetime import datetime

import pandas as pd

from utils.config import ENV, OUTPUT_DIR


# ─── 저장 경로 ────────────────────────────────────────────
EXP_DIR = os.path.join(OUTPUT_DIR, "experiments")
XLSX_PATH = os.path.join(EXP_DIR, "experiments.xlsx")
JSON_PATH = os.path.join(EXP_DIR, "experiments.json")

# ─── xlsx 컬럼 정의 ───────────────────────────────────────
XLSX_COLUMNS = [
    "실험번호", "날짜", "타입", "베스트모델",
    "val_rmse", "test_rmse", "val_증감", "test_증감",
    "피처수", "메모",
]


def _parse_exp_id(exp_id: str):
    """실험번호 파싱 → (팀, 모델, 실험번호)"""
    parts = exp_id.split("-")
    if len(parts) != 3:
        raise ValueError(
            f"실험번호 형식 오류: '{exp_id}' → '팀-모델-실험번호' (예: 1-1-001)"
        )
    return parts[0], parts[1], parts[2]


def _get_baseline_id(exp_id: str) -> str:
    """같은 팀-모델의 기준 실험번호(001) 반환"""
    team, model, _ = _parse_exp_id(exp_id)
    return f"{team}-{model}-001"


def _load_xlsx() -> pd.DataFrame:
    """기존 xlsx 로드. 없거나 비어있으면 빈 DataFrame 반환"""
    if os.path.exists(XLSX_PATH) and os.path.getsize(XLSX_PATH) > 0:
        try:
            return pd.read_excel(XLSX_PATH, dtype={"실험번호": str})
        except Exception:
            return pd.DataFrame(columns=XLSX_COLUMNS)
    return pd.DataFrame(columns=XLSX_COLUMNS)


def _save_xlsx(df: pd.DataFrame):
    """xlsx 저장"""
    os.makedirs(EXP_DIR, exist_ok=True)
    df.to_excel(XLSX_PATH, index=False)


def _load_json() -> dict:
    """기존 json 로드. 없거나 비어있으면 빈 dict 반환"""
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    return {}


def _save_json(data: dict):
    """json 저장"""
    os.makedirs(EXP_DIR, exist_ok=True)
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def download_from_drive(xlsx_gdrive_id: str = "", json_gdrive_id: str = ""):
    """
    Colab에서 Google Drive의 실험 파일을 gdown으로 다운로드.
    로컬이거나 ID가 비어있으면 스킵.
    """
    if ENV != "colab":
        return
    if not xlsx_gdrive_id and not json_gdrive_id:
        return

    import gdown
    os.makedirs(EXP_DIR, exist_ok=True)

    if xlsx_gdrive_id:
        print(f"  Drive → xlsx 다운로드 중...")
        gdown.download(id=xlsx_gdrive_id, output=XLSX_PATH, quiet=True)
        print(f"  완료: {XLSX_PATH}")

    if json_gdrive_id:
        print(f"  Drive → json 다운로드 중...")
        gdown.download(id=json_gdrive_id, output=JSON_PATH, quiet=True)
        print(f"  완료: {JSON_PATH}")


def upload_to_drive(xlsx_gdrive_id: str = "", json_gdrive_id: str = ""):
    """
    Colab에서 실험 파일을 Google Drive에 업로드 (기존 파일 덮어쓰기).
    로컬이거나 ID가 비어있으면 스킵.
    """
    if ENV != "colab":
        return
    if not xlsx_gdrive_id and not json_gdrive_id:
        return

    from google.colab import auth
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    auth.authenticate_user()
    service = build("drive", "v3")

    if xlsx_gdrive_id and os.path.exists(XLSX_PATH):
        media = MediaFileUpload(
            XLSX_PATH,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        service.files().update(fileId=xlsx_gdrive_id, media_body=media).execute()
        print(f"  xlsx → Drive 업로드 완료 (ID: {xlsx_gdrive_id})")

    if json_gdrive_id and os.path.exists(JSON_PATH):
        media = MediaFileUpload(JSON_PATH, mimetype="application/json")
        service.files().update(fileId=json_gdrive_id, media_body=media).execute()
        print(f"  json → Drive 업로드 완료 (ID: {json_gdrive_id})")


def check_exp_id(exp_id: str):
    """
    실험번호 중복 검사. 이미 존재하면 에러 발생.
    노트북 상단(실험 설정 셀)에서 호출하여 코드 실행 전에 미리 차단.
    """
    _parse_exp_id(exp_id)
    df = _load_xlsx()
    if exp_id in df["실험번호"].values:
        raise ValueError(f"이미 존재하는 실험번호입니다: '{exp_id}'")


def _params_hash(params_dict: dict) -> str:
    """파라미터 딕셔너리 → 결정적 해시 (비교용)"""
    serialized = json.dumps(params_dict, sort_keys=True, default=str)
    return hashlib.md5(serialized.encode()).hexdigest()


def check_duplicate_params(
    exp_id: str,
    *,
    cleaning_params: dict = None,
    outlier_params: dict = None,
    meta_params: dict = None,
    agg_params: dict = None,
    feature_sel_params: dict = None,
    model_params: dict = None,
):
    """
    실험 내용(파라미터) 중복 검사.
    동일한 파라미터 조합의 실험이 이미 존재하면 에러 발생.

    Parameters
    ----------
    exp_id : str
        현재 실험번호 (자기 자신은 비교에서 제외)
    cleaning_params ~ model_params : dict
        현재 실험의 파라미터들

    Raises
    ------
    ValueError
        동일한 파라미터 조합이 이미 존재할 때
    """
    current = {}
    if cleaning_params is not None:
        current["cleaning_params"] = cleaning_params
    if outlier_params is not None:
        current["outlier_params"] = outlier_params
    if meta_params is not None:
        current["meta_params"] = meta_params
    if agg_params is not None:
        current["agg_params"] = agg_params
    if feature_sel_params is not None:
        current["feature_sel_params"] = feature_sel_params
    if model_params is not None:
        current["model_params"] = model_params

    if not current:
        return  # 비교할 파라미터가 없으면 스킵

    current_hash = _params_hash(current)

    existing = _load_json()
    for existing_id, existing_params in existing.items():
        if existing_id == exp_id:
            continue  # 자기 자신 제외

        # 기존 실험에서 같은 키만 추출하여 비교
        existing_subset = {
            k: v for k, v in existing_params.items() if k in current
        }
        if not existing_subset:
            continue

        if _params_hash(existing_subset) == current_hash:
            raise ValueError(
                f"실험 '{existing_id}'과 동일한 파라미터 조합입니다. "
                f"파라미터를 변경하거나 기존 실험을 확인하세요."
            )


def _calc_delta(df: pd.DataFrame, exp_id: str, val_rmse: float, test_rmse: float):
    """
    증감 계산: 같은 팀-모델의 001 실험 대비 차이

    Returns
    -------
    (val_delta, test_delta) : (float|None, float|None)
        001이면 None, 기준이 없으면 None
    """
    _, _, exp_num = _parse_exp_id(exp_id)

    # 본인이 001이면 기준점 → 증감 없음
    if exp_num == "001":
        return None, None

    baseline_id = _get_baseline_id(exp_id)
    match = df[df["실험번호"] == baseline_id]

    if match.empty:
        print(f"  주의: 기준 실험 '{baseline_id}'이 없어 증감을 계산할 수 없습니다.")
        return None, None

    ref = match.iloc[0]
    val_delta = (
        val_rmse - ref["val_rmse"]
        if pd.notna(val_rmse) and pd.notna(ref["val_rmse"])
        else None
    )
    test_delta = (
        test_rmse - ref["test_rmse"]
        if pd.notna(test_rmse) and pd.notna(ref["test_rmse"])
        else None
    )
    return val_delta, test_delta


def log_experiment(
    exp_id: str,
    exp_type: str,
    best_model: str,
    val_rmse: float,
    test_rmse: float,
    n_features: int,
    memo: str = "",
    *,
    cleaning_params: dict = None,
    outlier_params: dict = None,
    meta_params: dict = None,
    agg_params: dict = None,
    feature_sel_params: dict = None,
    model_params: dict = None,
    feature_cols: list = None,
    xlsx_gdrive_id: str = "",
    json_gdrive_id: str = "",
):
    """
    실험 결과를 xlsx + json에 기록

    Parameters
    ----------
    exp_id : str
        실험번호 (예: "1-1-001"). 형식: 팀-모델-실험번호
    exp_type : str
        실험 타입 (예: "baseline", "two-stage", "ensemble")
    best_model : str
        최고 성능 모델명
    val_rmse : float
        Validation RMSE
    test_rmse : float
        Test RMSE
    n_features : int
        사용된 피처 수
    memo : str
        실험 메모 (한줄 요약)
    cleaning_params ~ model_params : dict
        상세 파라미터 (json에 저장)
    """
    # 형식 검증
    _parse_exp_id(exp_id)

    # ── xlsx 처리 ──────────────────────────────────────────
    df = _load_xlsx()

    # 중복 체크
    if exp_id in df["실험번호"].values:
        print(f"경고: 실험번호 '{exp_id}'가 이미 존재합니다.")
        answer = input("덮어쓰시겠습니까? (y/n): ").strip().lower()
        if answer != "y":
            print("저장 취소.")
            return
        df = df[df["실험번호"] != exp_id].reset_index(drop=True)

    # 증감 계산
    val_delta, test_delta = _calc_delta(df, exp_id, val_rmse, test_rmse)

    # 행 추가
    new_row = pd.DataFrame([{
        "실험번호": exp_id,
        "날짜": datetime.now().strftime("%Y-%m-%d"),
        "타입": exp_type,
        "베스트모델": best_model,
        "val_rmse": round(val_rmse, 6),
        "test_rmse": round(test_rmse, 6) if test_rmse is not None else None,
        "val_증감": round(val_delta, 6) if val_delta is not None else None,
        "test_증감": round(test_delta, 6) if test_delta is not None else None,
        "피처수": n_features,
        "메모": memo,
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    _save_xlsx(df)

    # ── json 처리 ─────────────────────────────────────────
    data = _load_json()
    detail = {}
    if cleaning_params is not None:
        detail["cleaning_params"] = cleaning_params
    if outlier_params is not None:
        detail["outlier_params"] = outlier_params
    if meta_params is not None:
        detail["meta_params"] = meta_params
    if agg_params is not None:
        detail["agg_params"] = agg_params
    if feature_sel_params is not None:
        detail["feature_sel_params"] = feature_sel_params
    if model_params is not None:
        detail["model_params"] = model_params
    if feature_cols is not None:
        detail["feature_cols"] = list(feature_cols)

    data[exp_id] = detail
    _save_json(data)

    # ── 결과 출력 ─────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"실험 기록 완료: {exp_id}")
    print(f"  타입: {exp_type} | 베스트: {best_model}")
    print(f"  Val RMSE:  {val_rmse:.6f}", end="")
    if pd.notna(val_delta):
        sign = "+" if val_delta >= 0 else ""
        print(f"  ({sign}{val_delta:.6f})", end="")
    print()
    print(f"  Test RMSE: {test_rmse:.6f}" if pd.notna(test_rmse) else "  Test RMSE: N/A", end="")
    if pd.notna(test_delta):
        sign = "+" if test_delta >= 0 else ""
        print(f"  ({sign}{test_delta:.6f})", end="")
    print()
    print(f"  피처수: {n_features} | 메모: {memo}")
    print(f"  xlsx: {XLSX_PATH}")
    print(f"  json: {JSON_PATH}")
    print(f"{'='*50}")

    # ── Colab이면 Google Drive에 업로드 ─────────────────────
    if xlsx_gdrive_id or json_gdrive_id:
        upload_to_drive(xlsx_gdrive_id, json_gdrive_id)
