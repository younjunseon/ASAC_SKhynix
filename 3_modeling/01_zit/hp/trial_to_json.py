"""
trial_to_json.py — Optuna db의 특정 trial 1개를 seed-sweep 노트북용 best_params JSON으로 변환.

ZIT 계열(zit_only / bag_zit / bag_zit_eql) 병렬 HPO 워커가 만든 SQLite Optuna study에서,
"내가 원하는 trial 번호" 하나를 골라 seed-sweep 노트북이 그대로 읽는 source_meta JSON으로 저장한다.
best trial이 아니어도 된다 — seed sweep은 HP만 받아 처음부터 재학습하므로 어떤 완료 trial이든 입력으로 쓸 수 있다.

생성 JSON 스키마 = 06_bag_zit_eql_seed_sweep.ipynb cell 6이 db에서 합성하던 source_meta와 동일:
  - best_params_resolved : trial.params 에서 tau_pi를 뺀 모델 HP
  - best_tau_pi          : trial.params['tau_pi']
  - effective_pp_params  : study.user_attrs['pp_fixed'] (전처리 파라미터)
  - study_meta           : study.user_attrs 전체 + CLIP_Y_EXTREME (model/impute_stage23 등 재현 메타 포함)
  - exp_id / model_name / best_trial_number / best_trial_state / best_oof_rmse
  - feature_names / n_features / unit_ids_hash = null (노트북이 pp_fixed로 재현하므로 불필요, feature check 자동 skip)

사용 (CLI):
  python 3_modeling/01_zit/hp/trial_to_json.py \
      --db 4_output/01_zit/bag_zit/hp/002/optuna_jh_bag-zit-eql-final-002.db \
      --trial 82 \
      --out 4_output/01_zit/bag_zit/hp/_custom/eql_002_t82/best_params.json

  # db에 study가 1개면 study명 자동 감지. 여러 개면 --study-name 으로 지정.
  # 산출물 model_name을 직접 지정하려면 --model-name bag_zit_eql.

사용 (노트북/파이썬 import):
  from trial_to_json import trial_to_json
  trial_to_json(db_path, trial_number, out_path)      # JSON 저장 + meta dict 반환
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import optuna


def _parse_user_attr(v):
    """optuna user_attr가 문자열 repr로 박제돼 있으면 원래 타입으로 복원한다.
    병렬 워커는 study.set_user_attr(k, str(v))로 dict/bool/int를 모두 str로 저장한다.
    'bag-zit-eql-final-002' 같은 순수 문자열은 literal_eval이 실패하므로 원본을 그대로 돌려준다.
    (06_bag_zit_eql_seed_sweep.ipynb cell 6의 동일 함수와 같은 동작)"""
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return v
    return v


def _resolve_study_name(storage_url: str, study_name: str | None) -> str:
    try:
        names = optuna.study.get_all_study_names(storage=storage_url)
    except AttributeError:  # 구버전 optuna 호환
        names = [s.study_name for s in optuna.get_all_study_summaries(storage=storage_url)]
    if not names:
        raise RuntimeError(f"db에 study가 없습니다: {storage_url}")
    if study_name is None:
        if len(names) != 1:
            raise RuntimeError(
                f"db에 study가 {len(names)}개 있습니다. --study-name 으로 지정하세요: {names}"
            )
        return names[0]
    if study_name not in names:
        raise RuntimeError(f"study '{study_name}' 없음. 존재하는 study: {names}")
    return study_name


def trial_to_json(
    db_path,
    trial_number: int,
    out_path=None,
    study_name: str | None = None,
    model_name: str | None = None,
):
    """db_path의 trial_number 1개를 seed-sweep용 source_meta dict로 변환한다.
    out_path가 주어지면 JSON으로도 저장한다. 반환값은 meta dict.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"db 파일 없음: {db_path}")
    storage = f"sqlite:///{db_path.as_posix()}"
    study_name = _resolve_study_name(storage, study_name)
    study = optuna.load_study(study_name=study_name, storage=storage)

    # trial.number로 정확히 찾는다 (리스트 index와 다를 수 있음).
    trial = next(
        (t for t in study.get_trials(deepcopy=False) if t.number == trial_number),
        None,
    )
    if trial is None:
        nums = [t.number for t in study.get_trials(deepcopy=False)]
        rng = f"0~{max(nums)}" if nums else "없음"
        raise ValueError(
            f"trial #{trial_number} 없음 (study='{study_name}', 가용 trial {len(nums)}개, 범위 {rng})"
        )

    # 모델 HP + tau_pi가 trial.params 한 dict에 섞여 있다. tau_pi를 분리한다.
    params = dict(trial.params)
    if "tau_pi" not in params:
        raise ValueError(
            f"trial #{trial_number}.params에 tau_pi가 없습니다 (state={trial.state}). "
            f"샘플링 전에 실패한 불완전 trial일 수 있습니다."
        )
    tau_pi = float(params.pop("tau_pi"))

    study_ua = {k: _parse_user_attr(v) for k, v in study.user_attrs.items()}
    trial_ua = {k: _parse_user_attr(v) for k, v in trial.user_attrs.items()}

    # best_oof_rmse: COMPLETE면 trial.value, 아니면(PRUNED 등) user_attr로 fallback (정보용).
    oof = trial.value
    if oof is None:
        oof = trial_ua.get("oof_rmse", trial_ua.get("val_rmse", trial_ua.get("partial_val_rmse")))
    oof = float(oof) if oof is not None else None

    # clip_y_extreme: 대문자/소문자 키 모두 대응.
    clip_attr = study_ua.get("clip_y_extreme", study_ua.get("CLIP_Y_EXTREME", True))
    if isinstance(clip_attr, str):
        clip_attr = clip_attr.lower() in {"true", "1", "yes"}
    clip_y_extreme = bool(clip_attr)

    if model_name is None:
        # study_meta의 'model' 문자열로 variant를 추정해 라벨을 만든다 (노트북도 같은 규칙으로 재판별).
        src_model = str(study_ua.get("model") or "").lower()
        is_eql = ("eql" in src_model) or ("deviance" in src_model)
        model_name = "bag_zit_eql" if is_eql else "bag_zit"

    meta = {
        "exp_id": study_ua.get("exp_id", study_name),
        "model_name": model_name,
        "source_db": str(db_path),
        "source_study_name": study_name,
        "best_trial_number": int(trial.number),
        "best_trial_state": str(trial.state),
        "best_oof_rmse": oof,
        "best_params_resolved": params,        # ① 모델 HP (tau_pi 제외)
        "best_tau_pi": tau_pi,                  # ② tau_pi
        "effective_pp_params": dict(study_ua.get("pp_fixed") or {}),  # ③ 전처리
        "feature_names": None,
        "n_features": None,
        "unit_ids_hash": None,
        "study_meta": {"CLIP_Y_EXTREME": clip_y_extreme, **study_ua},  # ④ + 재현 메타
    }

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Optuna db의 trial 1개 -> seed-sweep용 best_params JSON 변환 (best 아니어도 됨)"
    )
    ap.add_argument("--db", required=True, help="optuna SQLite db 경로 (예: 4_output/.../optuna_jh_*.db)")
    ap.add_argument("--trial", type=int, required=True, help="변환할 trial 번호 (trial.number)")
    ap.add_argument("--out", required=True, help="출력 JSON 경로 (예: .../_custom/eql_002_t82/best_params.json)")
    ap.add_argument("--study-name", default=None, help="db에 study가 여러 개일 때만 지정")
    ap.add_argument("--model-name", default=None, help="산출물 model_name 강제 (기본: study메타로 자동)")
    args = ap.parse_args()

    meta = trial_to_json(args.db, args.trial, args.out, args.study_name, args.model_name)

    oof = meta["best_oof_rmse"]
    oof_str = f"{oof:.9f}" if isinstance(oof, (int, float)) else str(oof)
    print(f"[OK] study='{meta['source_study_name']}'  trial #{meta['best_trial_number']} ({meta['best_trial_state']})")
    print(f"     best_oof_rmse = {oof_str}")
    print(f"     best_tau_pi   = {meta['best_tau_pi']:.9f}")
    print(f"     model_name    = {meta['model_name']}")
    print(f"     n HP          = {len(meta['best_params_resolved'])}")
    print(f"     pp_fixed keys = {list(meta['effective_pp_params'].keys())}")
    print(f"     study 'model' = {meta['study_meta'].get('model')}")
    print(f"     impute_stage23= {meta['study_meta'].get('impute_stage23')}")
    print(f"  -> {Path(args.out)}")


if __name__ == "__main__":
    main()
