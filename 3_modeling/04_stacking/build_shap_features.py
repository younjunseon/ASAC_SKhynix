"""
build_shap_features.py
======================

LGBM (또는 호환 트리 모델)의 fold_models.pkl을 그대로 활용해
fold-aware OOF + fold-평균 val/test SHAP(pred_contrib)을 추출하고,
die→unit 단위로 signed mean 집계해서 parquet으로 캐시한다.

squeeze_extreme_v2.py가 base preds 위에 이 SHAP cols를 메타 입력으로 얹는다.

사용 예시
--------
    python build_shap_features.py
    python build_shap_features.py --base-rel 02_reg_single/lgbm/hp/002 --top-k 50
    python build_shap_features.py --base-rel 02_reg_single/lgbm/hp/002 --top-k 0   # top-K 선택 안 함 (전체 저장)

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
    p.add_argument("--base-rel", default="02_reg_single/lgbm/hp/002",
                   help="OUTPUT_DIR 기준 base 모델 폴더. 기본: lgbm/hp/002")
    p.add_argument("--out-dir", default=None,
                   help="캐시 저장 폴더. 기본: 3_modeling/04_stacking/shap_cache/<base-tag>")
    p.add_argument("--top-k", type=int, default=50,
                   help="저장할 SHAP feature 수 (mean|SHAP| 상위 K). 0이면 전체.")
    p.add_argument("--zit-sub-model", choices=("pi", "mu", "phi"), default=None,
                   help="ZIT(ZITboostRegressor) fold_models일 때 어느 내부 LGBM에서 SHAP을 뽑을지. "
                        "pi=zero 분류, mu=Tweedie mean, phi=dispersion. lgbm/xgb/cat 단일 모델이면 무시.")
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
    if cls == "ZITboostRegressor":
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
        return inner.predict(X, pred_contrib=True)
    if cls in {"LGBMRegressor", "LGBMClassifier"}:
        return model.predict(X, pred_contrib=True)
    if cls in {"XGBRegressor", "XGBClassifier"}:
        # XGB sklearn API는 pred_contribs 인자가 없어 booster 직접 호출
        import xgboost as xgb
        return model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    if cls in {"CatBoostRegressor", "CatBoostClassifier"}:
        # ShapValues type — (n, n_features+1) 같은 형태
        return model.get_feature_importance(type="ShapValues", data=X)
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
    # ZIT면 out_dir 이름에 sub-model suffix 자동 추가 (예: ...__pi, ...__mu)
    out_tag = f"{base_tag}__{args.zit_sub_model}" if args.zit_sub_model else base_tag
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parent / "shap_cache" / out_tag
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[base]      {base_dir}")
    print(f"[out_dir]   {out_dir}")
    print(f"[top_k]     {args.top_k if args.top_k > 0 else 'ALL'}")
    if args.zit_sub_model:
        print(f"[zit-sub]   {args.zit_sub_model} (ZITboostRegressor 내부 lgb_{args.zit_sub_model}_)")

    # ---- 1. best_params.json 로드 + 검증값 추출
    with open(base_dir / "best_params.json", encoding="utf-8") as f:
        bp = json.load(f)
    expected_feat_names = bp["feature_names"]
    expected_n_units    = bp["n_units_train"]
    expected_uid_hash   = bp["unit_ids_hash"]
    seed_kfold          = int(bp["study_meta"].get("seed_kfold", args.seed_kfold))
    n_folds             = int(bp.get("n_folds", args.n_folds))
    eff_pp              = bp["effective_pp_params"]
    objective           = bp["best_params_resolved"].get("objective", "regression")
    print(f"[base meta] objective={objective}  n_folds={n_folds}  seed_kfold={seed_kfold}  "
          f"n_features={len(expected_feat_names)}  n_units_train={expected_n_units}")

    # ---- 2. fold_models.pkl 로드
    with open(base_dir / "fold_models.pkl", "rb") as f:
        fm = pickle.load(f)
    fold_models = fm["fold_models"]
    if len(fold_models) != n_folds:
        raise RuntimeError(f"fold_models 길이 {len(fold_models)} != n_folds {n_folds}")
    model_cls = type(fold_models[0]).__name__
    if model_cls == "ZITboostRegressor" and args.zit_sub_model is None:
        raise SystemExit(
            f"[ABORT] base가 ZITboostRegressor인데 --zit-sub-model 미지정. "
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
        diff_a = set(feat_cols_clean) - set(expected_feat_names)
        diff_b = set(expected_feat_names) - set(feat_cols_clean)
        raise RuntimeError(
            f"feat_cols 재현 실패. extra={sorted(diff_a)[:10]}... missing={sorted(diff_b)[:10]}..."
            f" (len cur={len(feat_cols_clean)}, expected={len(expected_feat_names)})"
        )
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
    # ZIT는 (1-π)×μ 의 mixture라 sub-model 하나만으로 final RMSE 재현 불가 → skip
    if args.zit_sub_model:
        print(f"\n[8] sanity: ZIT sub-model({args.zit_sub_model}) → final RMSE 재현 skip "
              "((1-π)×μ mixture라 단일 sub-model로 비교 부적합).")
        oof_rmse_recon = float("nan")
        oof_rmse_saved = float("nan")
    else:
        print("\n[8] sanity: reconstructed OOF die→unit RMSE vs best_params.postprocess ...")
        oof_pred_die = reconstruct_pred_die(shap_oof_die, bias_oof_die, objective)
        oof_pred_unit = pd.DataFrame({KEY_COL: xs_train[KEY_COL].values, "pred": oof_pred_die}) \
                           .groupby(KEY_COL, sort=False)["pred"].mean()
        y_true = ys_input["train"].set_index(KEY_COL)[TARGET_COL].loc[oof_pred_unit.index]
        oof_rmse_recon = float(np.sqrt(np.mean((oof_pred_unit.values - y_true.values) ** 2)))
        pp_block = bp.get("postprocess", {})
        oof_rmse_saved = float(pp_block.get("train_rmse", float("nan")))
        print(f"  reconstructed OOF unit RMSE = {oof_rmse_recon:.9f}")
        print(f"  saved postprocess.train_rmse= {oof_rmse_saved:.9f}")
        if not np.isnan(oof_rmse_saved):
            diff = abs(oof_rmse_recon - oof_rmse_saved)
            if diff > 5e-7:
                print(f"  [WARN] OOF RMSE 차이 {diff:.2e} > 5e-7 — 재현성 확인 권장")
            else:
                print(f"  [OK] OOF RMSE 일치 (diff={diff:.2e})")

    # ---- 9. die→unit signed mean 집계
    print("\n[9] die→unit signed mean 집계 ...")
    feat_names = list(feat_cols_clean)
    oof_unit = die_to_unit_signed_mean(xs_train, shap_oof_die, feat_names)
    val_unit = die_to_unit_signed_mean(xs_val,   shap_val_die, feat_names)
    test_unit = die_to_unit_signed_mean(xs_test,  shap_test_die, feat_names)
    print(f"  oof_unit  {oof_unit.shape}")
    print(f"  val_unit  {val_unit.shape}")
    print(f"  test_unit {test_unit.shape}")

    # ---- 10. top-K feature 선택 (mean|SHAP| on OOF)
    abs_means = np.abs(shap_oof_die).mean(axis=0)
    importance = pd.Series(abs_means, index=feat_names).sort_values(ascending=False)
    if args.top_k > 0:
        keep = importance.head(args.top_k).index.tolist()
    else:
        keep = feat_names
    keep_cols = [KEY_COL] + keep
    print(f"\n[10] top-K={len(keep)} (전체 {len(feat_names)})")
    print("  top 10:")
    for nm, v in importance.head(10).items():
        print(f"    {nm:>20s}  mean|SHAP|={v:.6e}")

    # ---- 11. 저장 — parquet (없으면 csv fallback)
    print(f"\n[11] 저장 ({out_dir}) ...")
    saved = {}
    for name, df in [("oof_unit_shap",  oof_unit[keep_cols]),
                     ("val_unit_shap",  val_unit[keep_cols]),
                     ("test_unit_shap", test_unit[keep_cols])]:
        try:
            path = out_dir / f"{name}.parquet"
            df.to_parquet(path, index=False)
        except Exception as e:
            path = out_dir / f"{name}.csv"
            df.to_csv(path, index=False)
            print(f"  parquet 실패({e}) → csv fallback {path.name}")
        saved[name] = str(path)
        print(f"  {path.name:30s}  shape={df.shape}  size={os.path.getsize(path)/1024:,.1f} KB")

    meta = {
        "base_rel":            args.base_rel,
        "base_tag":            base_tag,
        "model_class":         type(fold_models[0]).__name__,
        "objective":           objective,
        "n_folds":             n_folds,
        "seed_kfold":          seed_kfold,
        "n_features":          n_features,
        "top_k":               len(keep),
        "kept_features":       keep,
        "importance_full":     {nm: float(v) for nm, v in importance.items()},
        "unit_ids_hash":       got_hash,
        "n_units_train":       int(len(train_uid)),
        "oof_rmse_recon":      oof_rmse_recon,
        "oof_rmse_saved":      oof_rmse_saved,
        "clip_y_extreme":      bool(args.clip_y_extreme),
        "position_mode":       args.position_mode,
        "use_die_xy":          bool(args.use_die_xy),
        "saved":               saved,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  meta.json 저장")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
