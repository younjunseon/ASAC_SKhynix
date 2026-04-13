"""
실험 결과 기록 모듈

- csv: 실험별 핵심 지표 테이블 (1행 = 1실험, 공용)
- sqlite: 모든 trial 상세 + 재현 메타데이터 (팀원별 optuna_{USER}.db)

사용법:
    from utils.experiment import log_experiment

    log_experiment(
        exp_id="1-1-001",
        exp_type="baseline",
        best_model="LightGBM",
        val_rmse=0.00456,
        test_rmse=0.00478,
        n_features=960,
        n_trials=30,
        user="jw",
        memo="mean only 집계",
    )
"""
import os
import sys
from datetime import datetime

import pandas as pd

from utils.config import ENV, OUTPUT_DIR


# ─── 저장 경로 ────────────────────────────────────────────
EXP_DIR = os.path.join(OUTPUT_DIR, "experiments")
CSV_PATH = os.path.join(EXP_DIR, "experiments.csv")

# ─── csv 컬럼 정의 ────────────────────────────────────────
CSV_COLUMNS = [
    "실험번호", "날짜", "타입", "베스트모델",
    "val_rmse", "test_rmse", "val_증감", "test_증감",
    "피처수", "메모", "user", "n_trials",
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


def _load_csv() -> pd.DataFrame:
    """기존 csv 로드. 없거나 비어있으면 빈 DataFrame 반환"""
    if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0:
        try:
            return pd.read_csv(CSV_PATH, dtype={"실험번호": str})
        except Exception:
            return pd.DataFrame(columns=CSV_COLUMNS)
    return pd.DataFrame(columns=CSV_COLUMNS)


def _save_csv(df: pd.DataFrame):
    """csv 저장 (UTF-8 BOM: 한글 Excel 호환)"""
    os.makedirs(EXP_DIR, exist_ok=True)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")


def download_from_drive(
    csv_gdrive_id: str = "",
    db_gdrive_id: str = "",
    db_path: str = None,
):
    """
    Colab에서 Google Drive의 실험 파일을 gdown으로 다운로드.
    로컬이거나 ID가 비어있으면 스킵.

    Parameters
    ----------
    csv_gdrive_id : str
        공용 experiments.csv 파일 ID
    db_gdrive_id : str
        본인 optuna_{USER}.db 파일 ID
    db_path : str
        .db 파일의 로컬 저장 경로 (없으면 새 파일로 생성)
    """
    if ENV != "colab":
        return
    if not csv_gdrive_id and not db_gdrive_id:
        return

    import gdown
    os.makedirs(EXP_DIR, exist_ok=True)

    if csv_gdrive_id:
        print(f"  Drive → csv 다운로드 중...")
        gdown.download(id=csv_gdrive_id, output=CSV_PATH, quiet=True)
        print(f"  완료: {CSV_PATH}")

    if db_gdrive_id and db_path:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        print(f"  Drive → db 다운로드 중...")
        gdown.download(id=db_gdrive_id, output=db_path, quiet=True)
        print(f"  완료: {db_path}")


def upload_to_drive(
    csv_gdrive_id: str = "",
    db_gdrive_id: str = "",
    db_path: str = None,
):
    """
    Colab에서 실험 파일을 Google Drive에 업로드 (기존 파일 덮어쓰기).
    로컬이거나 ID가 비어있으면 스킵.

    Parameters
    ----------
    csv_gdrive_id : str
        공용 experiments.csv 파일 ID
    db_gdrive_id : str
        본인 optuna_{USER}.db 파일 ID
    db_path : str
        .db 파일의 로컬 경로
    """
    if ENV != "colab":
        return
    if not csv_gdrive_id and not db_gdrive_id:
        return

    from google.colab import auth
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    auth.authenticate_user()
    service = build("drive", "v3")

    if csv_gdrive_id and os.path.exists(CSV_PATH):
        media = MediaFileUpload(CSV_PATH, mimetype="text/csv")
        service.files().update(fileId=csv_gdrive_id, media_body=media).execute()
        print(f"  csv → Drive 업로드 완료 (ID: {csv_gdrive_id})")

    if db_gdrive_id and db_path and os.path.exists(db_path):
        media = MediaFileUpload(db_path, mimetype="application/x-sqlite3")
        service.files().update(fileId=db_gdrive_id, media_body=media).execute()
        print(f"  db → Drive 업로드 완료 (ID: {db_gdrive_id})")


def check_exp_id(exp_id: str):
    """
    실험번호 중복 검사. 이미 존재하면 에러 발생.
    노트북 상단(실험 설정 셀)에서 호출하여 코드 실행 전에 미리 차단.
    """
    _parse_exp_id(exp_id)
    df = _load_csv()
    if exp_id in df["실험번호"].values:
        raise ValueError(f"이미 존재하는 실험번호입니다: '{exp_id}'")


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
    user: str = "",
    n_trials: int = None,
    csv_gdrive_id: str = "",
):
    """
    실험 결과를 csv에 기록 (1행 = 1실험).
    상세 trial 정보는 SQLite(optuna_{USER}.db)에 저장되므로 여기선 저장 안 함.

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
    user : str
        실험 수행자 (예: 'jw', 'kim')
    n_trials : int
        총 HPO trial 수
    csv_gdrive_id : str
        공용 experiments.csv Drive 파일 ID (Colab에서 업로드용)
    """
    # 형식 검증
    _parse_exp_id(exp_id)

    # ── csv 처리 ───────────────────────────────────────────
    df = _load_csv()

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
        "user": user,
        "n_trials": n_trials,
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    _save_csv(df)

    # ── 결과 출력 ─────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"실험 기록 완료: {exp_id}")
    print(f"  타입: {exp_type} | 베스트: {best_model} | user: {user}")
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
    print(f"  피처수: {n_features} | n_trials: {n_trials} | 메모: {memo}")
    print(f"  csv: {CSV_PATH}")
    print(f"{'='*50}")

    # ── Colab이면 Google Drive에 업로드 ─────────────────────
    if csv_gdrive_id:
        upload_to_drive(csv_gdrive_id=csv_gdrive_id)


def make_trial_upload_callback(db_path: str, db_gdrive_id: str):
    """
    매 trial 완료 직후 .db 파일을 Drive에 업로드하는 Optuna 콜백 팩토리.

    사용법:
        upload_cb = make_trial_upload_callback(db_path=DB_PATH, db_gdrive_id=DB_GDRIVE_ID)
        study.optimize(objective, n_trials=N, callbacks=[upload_cb])

    Parameters
    ----------
    db_path : str
        로컬 SQLite 파일 경로 (optuna_{USER}.db)
    db_gdrive_id : str
        Drive에 저장된 해당 .db 파일 ID

    Returns
    -------
    callable
        Optuna callback 함수 (study, trial) → None
    """
    def _callback(study, trial):
        upload_to_drive(db_gdrive_id=db_gdrive_id, db_path=db_path)

    return _callback
