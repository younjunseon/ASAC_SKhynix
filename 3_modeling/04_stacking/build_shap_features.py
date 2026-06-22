"""
build_shap_features.py
======================

LGBM (또는 호환 트리 모델)의 fold_models.pkl을 그대로 활용해
fold-aware OOF + fold-평균 val/test SHAP(pred_contrib)을 추출하고,
die→unit 단위로 signed mean 집계해서 parquet으로 캐시한다.

stacking_lib.shap이 base preds 위에 이 SHAP cols를 메타 입력으로 얹는다.

사용 예시
--------
    python build_shap_features.py
    python build_shap_features.py --base-rel 02_reg_single/lgbm --top-k 50
    python build_shap_features.py --base-rel 02_reg_single/lgbm --top-k 0   # top-K 선택 안 함 (전체 저장)

검증
----
- best_params.json의 unit_ids_hash 와 재현된 train unit_ids hash 일치
- best_params.json의 feature_names (576개) 와 재현된 feat_cols 일치
- 재현된 OOF prediction(=exp(sum(SHAP)+bias) for poisson/tweedie) 의 RMSE 가
  best_params.json의 postprocess.train_rmse 와 일치 (objective 별 변환 고려)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


# Windows cp949 콘솔에서 UTF-8 출력 (한글/체크마크 깨짐 방지)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# 경로 부트스트랩 -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "2_preprocessing"))
sys.path.insert(0, str(PROJECT_ROOT / "3_modeling"))

from utils.config import KEY_COL, OUTPUT_DIR, SEED, TARGET_COL                # noqa: E402
from utils.data import get_feat_cols, load_all, split_xs                       # noqa: E402
from modules import preprocess                                                 # noqa: E402
from meta_features import add_meta_features                                    # noqa: E402

# ZITETRegressor는 노트북 인라인 정의 → pickle 역직렬화용 stub
try:
    from modules.zit import ZITboostRegressor as _ZITBase
    class ZITETRegressor(_ZITBase):  # noqa: N801
        pass
except Exception:
    pass


# 기본 PP_FIXED — lgbm/xgb/et/catboost 노트북 공통값 (strategy_common.md §1)
PP_FIXED_TREE = {
    "missing_threshold":          0.30,
    "corr_threshold":             0.90,
    "corr_keep_by":               "std",
    "add_indicator":              True,
    "indicator_threshold":        0.05,
    "spatial_max_dist":           6.0,
    "post_impute_corr_threshold": 0.96,
    "post_impute_corr_keep_by":   "std",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-rel", default="02_reg_single/lgbm",
                   help="OUTPUT_DIR 기준 base 모델 폴더. 기본: 02_reg_single/lgbm")
    p.add_argument("--out-dir", default=None,
                   help="캐시 저장 폴더. 기본: 3_modeling/04_stacking/shap_cache/<base-tag>")
    p.add_argument("--top-k", type=int, default=50,
                   help="저장할 SHAP feature 수 (mean|SHAP| 상위 K). 0이면 전체.")
    p.add_argument("--zit-sub-model", choices=("pi", "mu", "phi"), default=None,
                   help="ZIT(ZITboostRegressor) fold_models일 때 어느 내부 LGBM에서 SHAP을 뽑을지. "
                        "pi=zero 분류, mu=Tweedie mean, phi=dispersion. lgbm/xgb/cat 단일 모델이면 무시.")
    p.add_argument("--ts-sub-model", choices=("reg", "clf"), default=None,
                   help="ts_reverse fold_models는 (LGBMRegressor, LGBMClassifier) tuple. "
                        "reg=index 0 (회귀), clf=index 1 (분류). ts_reverse 이외에는 무시.")
    p.add_argument("--clip-y-extreme", action=argparse.BooleanOptionalAction, default=True,
                   help="train의 y>=1.0 행을 두 번째 큰 값으로 clip (노트북 기본 동작)")
    p.add_argument("--position-mode", default="raw", choices=("raw", "ohe"),
                   help="add_meta_features.position_mode. 트리는 raw.")
    p.add_argument("--use-die-xy", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed-kfold", type=int, default=SEED,
                   help="best_params.json의 seed_kfold 와 일치해야 OOF 분할 재현됨")
    p.add_argument("--n-folds", type=int, default=5)
    return p.parse_args()


# ---------------------------------------------------------------------------
# fold split 재현 — hpo._make_unit_folds 와 동일
# ---------------------------------------------------------------------------
def make_unit_folds(unit_ids: np.ndarray, n_splits: int, seed: int):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(unit_ids[tr], unit_ids[vl]) for tr, vl in kf.split(unit_ids)]


def compute_unit_ids_hash(unit_ids: np.ndarray) -> str:
    uid_bytes = ",".join(map(str, unit_ids)).encode("utf-8")
    return hashlib.sha1(uid_bytes).hexdigest()


# ---------------------------------------------------------------------------
# SHAP 추출 (per fold)
# ---------------------------------------------------------------------------
def _predict_contrib(model, X: np.ndarray, zit_sub: str | None = None) -> np.ndarray:
    """LGBM/XGB/CatBoost/ZIT 어느 쪽이든 pred_contrib 인터페이스로 통일.

    반환 shape: (n_samples, n_features + 1)  — 마지막 컬럼은 expected_value (bias).
    ZITboostRegressor면 zit_sub ∈ {'pi','mu','phi'} 로 내부 LGBM 선택.
    """
    cls = type(model).__name__
    if cls in {"ZITboostRegressor", "BagZITboostRegressor", "ZITETRegressor"}:
        if zit_sub == "pi":
            inner = model.lgb_pi_
        elif zit_sub == "mu":
            inner = model.lgb_mu_
        elif zit_sub == "phi":
            inner = model.lgb_phi_
        else:
            raise ValueError(
                f"ZIT 모델인데 --zit-sub-model 미지정 (현재: {zit_sub!r}). "
                "{{pi, mu, phi}} 중 하나 지정 필요."
            )
        return _predict_contrib(inner, X)  # inner가 LGBM이든 ET든 자동 분기
    if cls in {"LGBMRegressor", "LGBMClassifier"}:
        return model.predict(X, pred_contrib=True)
    if cls in {"XGBRegressor", "XGBClassifier"}:
        # XGB sklearn API는 pred_contribs 인자가 없어 booster 직접 호출
        import xgboost as xgb
        return model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    if cls in {"CatBoostRegressor", "CatBoostClassifier"}:
        import catboost as cb
        pool = cb.Pool(X)
        return model.get_feature_importance(type="ShapValues", data=pool)
    if cls in {"ExtraTreesRegressor", "ExtraTreesClassifier",
               "RandomForestRegressor", "RandomForestClassifier"}:
        import shap as shap_lib
        explainer = shap_lib.TreeExplainer(model)
        sv = explainer.shap_values(X)
        ev = explainer.expected_value
        # 이진 분류기: shap_values()가 [class0_arr, class1_arr] 리스트를 반환할 수 있음
        # → class 1(양성) SHAP 값만 사용 (LGBM/XGB pred_contrib과 동일 기준)
        if isinstance(sv, list):
            sv = sv[-1]
        # expected_value도 클래스별 배열일 수 있음 → class 1 값 사용
        if hasattr(ev, "__len__") and len(ev) > 1:
            ev_val = float(ev[-1])
        else:
            ev_val = float(ev) if not hasattr(ev, "__len__") else float(ev[0])
        return np.column_stack([sv.astype(np.float32),
                                np.full(len(X), ev_val, dtype=np.float32)])
    raise NotImplementedError(f"pred_contrib 미지원 모델: {cls}")


def extract_oof_shap_die(fold_models, X_train_full: np.ndarray, xs_train: pd.DataFrame,
                         folds, n_features: int, zit_sub: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """fold-aware OOF SHAP. 각 fold model은 그 fold의 holdout die에 대해서만 SHAP 계산.

    Returns
    -------
    shap_oof : (n_die_train, n_features)   — feature contribution
    bias_oof : (n_die_train,)              — expected_value (fold별 다를 수 있음)
    """
    n = len(xs_train)
    shap_oof = np.full((n, n_features), np.nan, dtype=np.float32)
    bias_oof = np.full(n,               np.nan, dtype=np.float32)
    covered  = np.zeros(n, dtype=bool)

    for i, (_, vl_units) in enumerate(folds):
        vl_mask = xs_train[KEY_COL].isin(set(vl_units)).values
        X_vl = X_train_full[vl_mask]
        t0 = time.time()
        contrib = _predict_contrib(fold_models[i], X_vl, zit_sub=zit_sub)
        shap_oof[vl_mask, :] = contrib[:, :n_features].astype(np.float32)
        bias_oof[vl_mask]    = contrib[:, n_features].astype(np.float32)
        covered |= vl_mask
        print(f"  [oof fold {i+1}/{len(folds)}] {vl_mask.sum():>7d} die, {time.time()-t0:.1f}s")

    if not covered.all():
        raise RuntimeError(f"OOF fold coverage 불완전: {(~covered).sum()} die 누락")
    if np.isnan(shap_oof).any():
        raise RuntimeError("shap_oof에 NaN 잔존 — fold coverage 또는 모델 예측 문제")
    return shap_oof, bias_oof


def extract_avg_shap_die(fold_models, X_full: np.ndarray, n_features: int,
                         split_name: str, zit_sub: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """val/test: 모든 fold model로 SHAP 계산 후 평균."""
    n = len(X_full)
    shap_acc = np.zeros((n, n_features), dtype=np.float64)
    bias_acc = np.zeros(n,               dtype=np.float64)
    for i, m in enumerate(fold_models):
        t0 = time.time()
        contrib = _predict_contrib(m, X_full, zit_sub=zit_sub)
        shap_acc += contrib[:, :n_features]
        bias_acc += contrib[:, n_features]
        print(f"  [{split_name} fold {i+1}/{len(fold_models)}] {time.time()-t0:.1f}s")
    shap_acc /= len(fold_models)
    bias_acc /= len(fold_models)
    return shap_acc.astype(np.float32), bias_acc.astype(np.float32)


# ---------------------------------------------------------------------------
# die→unit signed mean (4 die per unit)
# ---------------------------------------------------------------------------
def die_to_unit_signed_mean(xs_split: pd.DataFrame, shap_die: np.ndarray, feat_names: list[str]) -> pd.DataFrame:
    """ufs_serial 기준 signed mean. 원본 unit 순서 보존."""
    df = pd.DataFrame(shap_die, columns=feat_names, copy=False)
    df.insert(0, KEY_COL, xs_split[KEY_COL].values)
    unit = df.groupby(KEY_COL, sort=False).mean()
    unit.reset_index(inplace=True)
    return unit


# ---------------------------------------------------------------------------
# Sanity check — reconstructed prediction RMSE
# ---------------------------------------------------------------------------
def reconstruct_pred_die(shap_die: np.ndarray, bias_die: np.ndarray, objective: str) -> np.ndarray:
    """sum(SHAP, axis=1) + bias = raw_score → objective 별 변환 적용."""
    raw = shap_die.sum(axis=1).astype(np.float64) + bias_die.astype(np.float64)
    if objective in ("poisson", "tweedie", "regression_log") or "tweedie" in objective:
        return np.exp(raw)
    return raw  # regression / regression_l2 / mse 등


def main():
    args = parse_args()
    base_dir = Path(OUTPUT_DIR) / args.base_rel
    base_tag = args.base_rel.replace("/", "__").replace("\\", "__")
    # sub-model suffix 자동 추가 (ZIT: ...__pi, ts_reverse: ...__reg/__clf)
    out_tag = base_tag
    if args.zit_sub_model:
        out_tag = f"{out_tag}__{args.zit_sub_model}"
    if args.ts_sub_model:
        out_tag = f"{out_tag}__{args.ts_sub_model}"
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parent / "shap_cache" / out_tag
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[base]      {base_dir}")
    print(f"[out_dir]   {out_dir}")
    print(f"[top_k]     {args.top_k if args.top_k > 0 else 'ALL'}")
    if args.zit_sub_model:
        print(f"[zit-sub]   {args.zit_sub_model} (ZITboostRegressor 내부 lgb_{args.zit_sub_model}_)")
    if args.ts_sub_model:
        print(f"[ts-sub]    {args.ts_sub_model} (ts_reverse tuple index {'0=reg' if args.ts_sub_model=='reg' else '1=clf'})")

    # ---- 1. best_params.json 로드 + 검증값 추출
    with open(base_dir / "best_params.json", encoding="utf-8") as f:
        bp = json.load(f)
    expected_feat_names = bp["feature_names"]
    expected_n_units    = bp["n_units_train"]
    expected_uid_hash   = bp["unit_ids_hash"]
    seed_kfold          = int(bp.get("study_meta", {}).get("seed_kfold", args.seed_kfold))
    n_folds             = int(bp.get("n_folds", args.n_folds))
    eff_pp              = bp["effective_pp_params"]
    if eff_pp is None:
        # 리팩토링 후 PP가 zit_pp.PP_FIXED로 통일되며 fit이 effective_pp_params=None을 기록한다.
        # 학습 때 실제로 쓰인 고정 PP(= PP_FIXED_TREE, zit_pp.PP_FIXED와 동일)로 재현한다.
        eff_pp = dict(PP_FIXED_TREE)
    objective           = bp["best_params_resolved"].get("objective", "regression")
    print(f"[base meta] objective={objective}  n_folds={n_folds}  seed_kfold={seed_kfold}  "
          f"n_features={len(expected_feat_names)}  n_units_train={expected_n_units}")

    # ---- 2. fold_models.pkl 로드
    with open(base_dir / "fold_models.pkl", "rb") as f:
        fm = pickle.load(f)
    # dict 형식(ts_reverse 등)과 list 형식 모두 처리
    fold_models = fm["fold_models"] if isinstance(fm, dict) else fm
    if len(fold_models) != n_folds:
        raise RuntimeError(f"fold_models 길이 {len(fold_models)} != n_folds {n_folds}")

    # ts_reverse: 각 fold가 (LGBMRegressor, LGBMClassifier) tuple → 선택적 unwrap
    # fold_models_raw 보존 — ts_reverse clf 577번째 피처(reg OOF 예측) 생성에 필요
    fold_models_raw = list(fold_models)
    if args.ts_sub_model is not None:
        idx = 0 if args.ts_sub_model == "reg" else 1
        fold_models = [m[idx] for m in fold_models]

    model_cls = type(fold_models[0]).__name__
    ZIT_CLASSES = {"ZITboostRegressor", "BagZITboostRegressor", "ZITETRegressor"}
    if model_cls in ZIT_CLASSES and args.zit_sub_model is None:
        raise SystemExit(
            f"[ABORT] base가 {model_cls}인데 --zit-sub-model 미지정. "
            "{pi, mu, phi} 중 하나 지정 (각각 별도 cache로 저장 권장)."
        )
    print(f"[fold model class] {model_cls}")

    # ---- 3. 데이터 로드 + 노트북 동일 흐름으로 전처리
    print("\n[1] load_all + split_xs ...")
    xs, ys = load_all()
    feat_cols = get_feat_cols(xs)
    xs_dict = split_xs(xs)

    ys_input = {k: v.copy() for k, v in ys.items()}
    if args.clip_y_extreme:
        y_raw = ys_input["train"][TARGET_COL]
        second_max = y_raw[y_raw < y_raw.max()].max()
        n_clipped = int((y_raw >= 1.0).sum())
        ys_input["train"][TARGET_COL] = y_raw.clip(upper=second_max)
        print(f"  [CLIP_Y_EXTREME] 1.0 → {second_max:.6f} clip, {n_clipped}개")

    # PP_FIXED — best_params.json의 effective_pp_params에서 ‘_fixed_’/‘_exclude_’ 빼고 그대로 사용
    pp_params = {k: v for k, v in eff_pp.items() if not k.startswith("_")}
    # preprocess.run이 받지 않는 키만 골라내 drop (혹시 모를 추가 키 대비)
    pp_params = {k: v for k, v in pp_params.items() if k in preprocess.DEFAULT_PARAMS}

    print(f"\n[2] preprocess.run(params={pp_params}) ...")
    pp = preprocess.run(xs, ys_input, feat_cols, xs_dict, params=pp_params)
    xs_train, xs_val, xs_test = pp["xs_train"], pp["xs_val"], pp["xs_test"]
    feat_cols_clean = pp["feat_cols"]

    print(f"\n[3] add_meta_features(position_mode={args.position_mode!r}, "
          f"use_die_xy={args.use_die_xy}) ...")
    feat_cols_clean = add_meta_features(
        xs_train, xs_val, xs_test, feat_cols_clean,
        position_mode=args.position_mode, use_die_xy=args.use_die_xy,
    )

    # ---- 4. 재현성 검증 — feat_cols / unit_ids_hash
    if feat_cols_clean != expected_feat_names:
        missing = [f for f in expected_feat_names if f not in set(feat_cols_clean)]
        if missing:
            # preprocessing이 걸러낸 피처를 원본 xs에서 복원
            # preprocess.run()은 xs를 수정하지 않으므로 xs.loc[xs_train.index] 정렬 가능
            recoverable   = [f for f in missing if f in xs.columns]
            unrecoverable = [f for f in missing if f not in xs.columns]
            if unrecoverable:
                raise RuntimeError(
                    f"feat_cols 재현 실패: 원본 xs에도 없는 피처: {unrecoverable[:10]}"
                )
            print(f"  [WARN] {len(missing)}개 피처를 preprocessing이 제거 → 원본에서 복원 "
                  f"(train 기준 median impute). 예: {missing[:5]}")
            for col in recoverable:
                med = xs.loc[xs_train.index, col].median()
                xs_train[col] = xs.loc[xs_train.index, col].fillna(med).values
                xs_val[col]   = xs.loc[xs_val.index,   col].fillna(med).values
                xs_test[col]  = xs.loc[xs_test.index,  col].fillna(med).values
            print(f"  [OK] {len(recoverable)}개 피처 복원 완료")
        # extra 피처(pipeline이 추가한 것) 드롭 + expected 기준 정렬
        extra = [f for f in feat_cols_clean if f not in set(expected_feat_names)]
        if extra:
            print(f"  [WARN] extra {len(extra)}개 드롭, expected 기준 정렬")
        feat_cols_clean = expected_feat_names
    print(f"  [OK] feat_cols 일치 ({len(feat_cols_clean)}개)")

    train_uid = ys_input["train"][KEY_COL].unique()
    if len(train_uid) != expected_n_units:
        raise RuntimeError(f"n_units_train 불일치: {len(train_uid)} != {expected_n_units}")
    got_hash = compute_unit_ids_hash(train_uid)
    if got_hash != expected_uid_hash:
        raise RuntimeError(f"unit_ids_hash 불일치: {got_hash} != {expected_uid_hash}")
    print(f"  [OK] unit_ids_hash 일치 ({got_hash[:8]}..., n_units={len(train_uid)})")

    # ---- 5. X 행렬 build (fold_models이 본 그대로)
    X_train_full = xs_train[feat_cols_clean].values
    X_val_full   = xs_val[feat_cols_clean].values
    X_test_full  = xs_test[feat_cols_clean].values
    n_features = len(feat_cols_clean)
    print(f"\n[4] X shapes: train={X_train_full.shape}, val={X_val_full.shape}, test={X_test_full.shape}")

    # ---- 6. fold split 재현
    folds = make_unit_folds(train_uid, n_folds, seed_kfold)
    fold_sizes = [(len(tr), len(vl)) for tr, vl in folds]
    print(f"  fold (tr_units, vl_units): {fold_sizes}")

    # ---- 6.5. ts_reverse clf 전용: reg OOF 예측을 577번째 feature로 추가
    # clf 학습 시 X_tr_aug = np.hstack([X_tr, clip(reg_oof, 0)]) 로 훈련됐으므로 동일하게 재현
    if args.ts_sub_model == "clf":
        print("\n[4.5] ts_reverse clf: reg OOF 예측 → 577번째 feature 추가 ...")
        reg_models_ts = [m[0] for m in fold_models_raw]
        # train: outer fold-aware OOF reg 예측 (fold_models[i][0]이 outer full reg)
        reg_oof_arr = np.full(len(X_train_full), np.nan, dtype=np.float64)
        for i, (_, vl_units) in enumerate(folds):
            vl_mask = xs_train[KEY_COL].isin(set(vl_units)).values
            reg_oof_arr[vl_mask] = np.clip(
                reg_models_ts[i].predict(X_train_full[vl_mask]), 0.0, None
            )
        if np.isnan(reg_oof_arr).any():
            raise RuntimeError("ts_reverse clf: reg OOF에 NaN 잔존 — fold coverage 문제")
        # val / test: 전체 fold reg 모델 평균
        reg_val_arr  = np.mean([np.clip(m.predict(X_val_full),  0.0, None) for m in reg_models_ts], axis=0)
        reg_test_arr = np.mean([np.clip(m.predict(X_test_full), 0.0, None) for m in reg_models_ts], axis=0)
        X_train_full = np.column_stack([X_train_full, reg_oof_arr.astype(np.float32)])
        X_val_full   = np.column_stack([X_val_full,   reg_val_arr.astype(np.float32)])
        X_test_full  = np.column_stack([X_test_full,  reg_test_arr.astype(np.float32)])
        feat_cols_clean = list(feat_cols_clean) + ["ts_reg_pred"]
        n_features = len(feat_cols_clean)
        print(f"  ts_reg_pred 추가 완료 → X_train: {X_train_full.shape}, n_features: {n_features}")

    # ---- 7. SHAP 추출
    print("\n[5] SHAP 추출 (OOF, fold-aware) ...")
    t0 = time.time()
    shap_oof_die, bias_oof_die = extract_oof_shap_die(
        fold_models, X_train_full, xs_train, folds, n_features, zit_sub=args.zit_sub_model
    )
    print(f"  done {time.time()-t0:.1f}s, shap_oof_die shape={shap_oof_die.shape}")

    print("\n[6] SHAP 추출 (val, fold mean) ...")
    t0 = time.time()
    shap_val_die, bias_val_die = extract_avg_shap_die(fold_models, X_val_full, n_features, "val",
                                                       zit_sub=args.zit_sub_model)
    print(f"  done {time.time()-t0:.1f}s, shap_val_die shape={shap_val_die.shape}")

    print("\n[7] SHAP 추출 (test, fold mean) ...")
    t0 = time.time()
    shap_test_die, bias_test_die = extract_avg_shap_die(fold_models, X_test_full, n_features, "test",
                                                         zit_sub=args.zit_sub_model)
    print(f"  done {time.time()-t0:.1f}s, shap_test_die shape={shap_test_die.shape}")

    # ---- 8. Sanity check — reconstructed prediction RMSE (unit mean)
    # ZIT sub-model 또는 ts_reverse clf(분류기)는 RMSE 재현 skip
    if args.zit_sub_model or args.ts_sub_model == "clf":
        sub_label = args.zit_sub_model or f"ts_clf({args.ts_sub_model})"
        print(f"\n[8] sanity: {sub_label} → final RMSE 재현 skip.")
        oof_rmse_recon = float("nan")
        oof_rmse_saved = float("nan")
    else:
        print("\n[8] sanity: reconstructed OOF die→unit RMSE vs best_params.postprocess ...")
        oof_pred_die = reconstruct_pred_die(shap_oof_die, bias_oof_die, objective)
        oof_pred_unit = pd.DataFrame({KEY_COL: xs_train[KEY_COL].values, "pred": oof_pred_die}) \
                           .groupby(KEY_COL, sort=False)["pred"].mean()
        y_true = ys_input["train"].set_index(KEY_COL)[TARGET_COL].loc[oof_pred_unit.index]
        oof_rmse_recon = float(np.sqrt(np.mean((oof_pred_unit.values - y_true.values) ** 2)))
        pp_block = bp.get("postprocess") or {}   # clf 모델 등은 postprocess=null일 수 있음
        oof_rmse_saved = float(pp_block.get("train_rmse", float("nan")))
        print(f"  reconstructed OOF unit RMSE = {oof_rmse_recon:.9f}")
        print(f"  saved postprocess.train_rmse= {oof_rmse_saved:.9f}")
        if not np.isnan(oof_rmse_saved):
            diff = abs(oof_rmse_recon - oof_rmse_saved)
            if diff > 5e-7:
                print(f"  [WARN] OOF RMSE 차이 {diff:.2e} > 5e-7 — 재현성 확인 권장")
            else:
                print(f"  [OK] OOF RMSE 일치 (diff={diff:.2e})")

    # ---- 9. feature importance (meta.json 기록용)
    feat_names = list(feat_cols_clean)
    abs_means = np.abs(shap_oof_die).mean(axis=0)
    importance = pd.Series(abs_means, index=feat_names).sort_values(ascending=False)
    print(f"\n[9] feature importance top-10 (mean|SHAP|):")
    for nm, v in importance.head(10).items():
        print(f"    {nm:>20s}  {v:.6e}")

    # ---- 10. 저장 — die-level npz (oof+val+test 통합)
    # die-level stacking(v4)에서 (ufs_serial, run_wf_xy)로 매칭하기 위해 run_wf_xy도 함께 저장.
    npz_path = out_dir / "die_shap.npz"
    print(f"\n[10] die-level npz 저장: {npz_path} ...")
    np.savez_compressed(
        npz_path,
        oof_shap=shap_oof_die,
        val_shap=shap_val_die,
        test_shap=shap_test_die,
        feature_names=np.array(feat_names, dtype=object),
        oof_serials=xs_train[KEY_COL].values.astype(str),
        val_serials=xs_val[KEY_COL].values.astype(str),
        test_serials=xs_test[KEY_COL].values.astype(str),
        oof_run_wf_xy=xs_train["run_wf_xy"].values.astype(str),
        val_run_wf_xy=xs_val["run_wf_xy"].values.astype(str),
        test_run_wf_xy=xs_test["run_wf_xy"].values.astype(str),
    )
    npz_mb = os.path.getsize(npz_path) / 1024 / 1024
    print(f"  die_shap.npz  oof={shap_oof_die.shape}  val={shap_val_die.shape}  "
          f"test={shap_test_die.shape}  {npz_mb:.1f} MB")

    meta = {
        "base_rel":         args.base_rel,
        "base_tag":         base_tag,
        "zit_sub_model":    args.zit_sub_model,
        "model_class":      type(fold_models[0]).__name__,
        "objective":        objective,
        "n_folds":          n_folds,
        "seed_kfold":       seed_kfold,
        "n_features":       n_features,
        "feature_names":    feat_names,
        "importance_top50": {nm: float(v) for nm, v in importance.head(50).items()},
        "unit_ids_hash":    got_hash,
        "n_units_train":    int(len(train_uid)),
        "oof_rmse_recon":   oof_rmse_recon,
        "oof_rmse_saved":   oof_rmse_saved,
        "clip_y_extreme":   bool(args.clip_y_extreme),
        "position_mode":    args.position_mode,
        "use_die_xy":       bool(args.use_die_xy),
        "npz_path":         str(npz_path),
        "npz_mb":           round(npz_mb, 2),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  meta.json 저장")

    # ---- 11. unit-level parquet 저장 (die→unit signed mean, importance 내림차순 열 정렬)
    ordered_feat = importance.index.tolist()   # 이미 mean|SHAP| 내림차순
    print(f"\n[11] unit-level parquet 저장 ...")
    for split_name, shap_die, serials in [
        ("oof",  shap_oof_die,  xs_train[KEY_COL].values.astype(str)),
        ("val",  shap_val_die,  xs_val[KEY_COL].values.astype(str)),
        ("test", shap_test_die, xs_test[KEY_COL].values.astype(str)),
    ]:
        df = pd.DataFrame(shap_die.astype(np.float32), columns=feat_names)
        df[KEY_COL] = serials
        unit_df = df.groupby(KEY_COL, sort=False)[feat_names].mean().reset_index()
        unit_df = unit_df[[KEY_COL] + ordered_feat]   # importance 순 열 정렬
        pq_path = out_dir / f"{split_name}_unit_shap.parquet"
        unit_df.to_parquet(pq_path, index=False)
        pq_mb = os.path.getsize(pq_path) / 1024 / 1024
        print(f"  {split_name}_unit_shap.parquet  {unit_df.shape}  {pq_mb:.1f} MB")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
