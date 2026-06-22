"""die-level SHAP X-stacking — die_shap.npz 캐시 로더.

build_shap_features.py가 만든 캐시 폴더(`shap_cache/<base_tag>/`)에서 `die_shap.npz`를 읽어
bundle의 `build_die_matrix` 결과와 (ufs_serial, run_wf_xy) 키로 정렬 일치시킨 die-level 컬럼 행렬을 반환.

캐시 폴더 구조 (build_shap_features.py 출력):
    shap_cache/<base_tag>/
      ├── die_shap.npz          ← (oof_shap, val_shap, test_shap, feature_names,
      │                            oof_serials, val_serials, test_serials,
      │                            oof_run_wf_xy, val_run_wf_xy, test_run_wf_xy)
      ├── meta.json             ← importance_top50 (mean|SHAP| 내림차순)
      ├── oof_unit_shap.parquet ← v2가 쓰던 unit-level (v4는 사용 안 함)
      ├── val_unit_shap.parquet
      └── test_unit_shap.parquet

run_wf_xy 키 부재 (구버전 캐시) 시 명시적 RuntimeError로 실패 — 재생성 유도.

v2의 `--extra-cache` 컨셉을 die-level로 재구현. 컬럼 prefix(`<tag>::feature_name`)는 여러 캐시
합칠 때 충돌 방지용.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import StackingConfig
from .discovery import KEY_COL, DIE_KEY_COL


def _resolve_cache_dir(raw_path: str, cfg: StackingConfig) -> Path:
    """상대경로면 (shap_cache_root or 04_stacking/) 기준으로 확장."""
    p = Path(raw_path)
    if p.is_absolute() and p.exists():
        return p
    base = (
        Path(cfg.shap_cache_root)
        if cfg.shap_cache_root
        else cfg.project_root / "3_modeling" / "04_stacking"
    )
    cand = base / raw_path
    if cand.exists():
        return cand
    # 마지막 fallback — raw_path 그대로 (상대경로지만 cwd 기준)
    if p.exists():
        return p
    raise FileNotFoundError(
        f"SHAP 캐시 폴더 없음: {raw_path} (시도: {base / raw_path}, {p})"
    )


def _load_npz_split(npz_path: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """npz에서 한 split의 (shap_arr, serials, run_wf_xy, feature_names) 반환.

    run_wf_xy 키가 없으면 RuntimeError — 구버전 캐시는 재생성 필요.
    split ∈ {'oof', 'val', 'test'}.
    """
    z = np.load(npz_path, allow_pickle=True)
    files = set(z.files)

    arr_key = f"{split}_shap"
    sid_key = f"{split}_serials"
    xy_key = f"{split}_run_wf_xy"

    if arr_key not in files or sid_key not in files:
        raise RuntimeError(
            f"{npz_path}: {arr_key} 또는 {sid_key} 키 없음. "
            f"있는 키: {sorted(files)}"
        )
    if xy_key not in files:
        raise RuntimeError(
            f"{npz_path}: {xy_key} 키 없음 — build_shap_features.py(v4 이후)로 캐시 재생성 필요. "
            "run_shap_all.py --force 권장."
        )

    feat_names = list(z["feature_names"]) if "feature_names" in files else None
    if feat_names is None:
        raise RuntimeError(f"{npz_path}: feature_names 키 없음")

    return (
        z[arr_key],
        z[sid_key].astype(str),
        z[xy_key].astype(str),
        [str(x) for x in feat_names],
    )


def _select_top_k(
    feat_names: list[str],
    shap_arr: np.ndarray,
    meta_path: Path,
    top_k: int,
) -> tuple[list[str], np.ndarray]:
    """meta.json의 importance_top50 순서대로 상위 K 컬럼만 추출.

    meta.json이 없거나 키가 비면 mean|SHAP|을 OOF 자체에서 계산하여 정렬 (fallback).
    top_k=0이면 전체 feat_names 그대로 반환.
    """
    if top_k <= 0:
        return feat_names, shap_arr

    ordered: list[str] | None = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            imp = meta.get("importance_top50")
            if isinstance(imp, dict) and imp:
                ordered = [n for n in imp.keys() if n in set(feat_names)]
        except Exception:
            ordered = None
    if not ordered:
        abs_means = np.abs(shap_arr).mean(axis=0)
        order = np.argsort(-abs_means)
        ordered = [feat_names[i] for i in order]

    keep = ordered[:top_k]
    name_to_idx = {n: i for i, n in enumerate(feat_names)}
    idx = [name_to_idx[n] for n in keep]
    return keep, shap_arr[:, idx]


def _align_to_keydf(
    shap_arr: np.ndarray,
    serials: np.ndarray,
    run_wf_xy: np.ndarray,
    key_df: pd.DataFrame,
    tag: str,
    split: str,
) -> np.ndarray:
    """die_shap.npz의 행 순서 → bundle key_df의 (ufs_serial, run_wf_xy) 순서로 reorder.

    매칭 누락 행이 있으면 RuntimeError (key set 불일치).
    """
    if len(serials) != len(run_wf_xy) or len(serials) != len(shap_arr):
        raise RuntimeError(
            f"[shap/{tag}/{split}] 캐시 npz 내부 길이 불일치: "
            f"shap={len(shap_arr)} serials={len(serials)} run_wf_xy={len(run_wf_xy)}"
        )

    cache_df = pd.DataFrame({
        KEY_COL: serials,
        DIE_KEY_COL: run_wf_xy,
        "_pos": np.arange(len(serials), dtype=np.int64),
    })
    merged = key_df[[KEY_COL, DIE_KEY_COL]].merge(
        cache_df, on=[KEY_COL, DIE_KEY_COL], how="left",
    )
    if merged["_pos"].isna().any():
        n_missing = int(merged["_pos"].isna().sum())
        raise RuntimeError(
            f"[shap/{tag}/{split}] (ufs_serial, run_wf_xy) 매칭 누락 {n_missing}개. "
            f"캐시와 base die set 불일치 — preprocess 차이 가능성."
        )
    idx = merged["_pos"].astype(np.int64).values
    return shap_arr[idx]


def load_shap_caches(
    cfg: StackingConfig,
    key_df_oof: pd.DataFrame,
    key_df_val: pd.DataFrame,
    key_df_test: pd.DataFrame,
) -> dict:
    """모든 cfg.shap_caches를 로드 + key_df 순서로 정렬 + 가로 결합.

    Returns
    -------
    dict {
        'X_oof':  (N_die_oof, sum_n_shap) | None,
        'X_val':  (N_die_val, sum_n_shap) | None,
        'X_test': (N_die_test, sum_n_shap) | None,
        'names':  list[str] (사용자에게 보이는 컬럼명, tag prefix 적용 후),
        'tags':   list[str] (캐시 폴더명 = tag),
        'cache_col_counts': list[int] (캐시별로 가져온 컬럼 수),
    }

    cfg.shap_caches가 비어 있으면 모든 X_*를 None으로 반환 (호출자는 base만 사용).
    """
    if not cfg.shap_caches:
        return {
            "X_oof": None, "X_val": None, "X_test": None,
            "names": [], "tags": [], "cache_col_counts": [],
        }

    all_oof: list[np.ndarray] = []
    all_val: list[np.ndarray] = []
    all_test: list[np.ndarray] = []
    all_names: list[str] = []
    tags: list[str] = []
    counts: list[int] = []

    for raw_path in cfg.shap_caches:
        dpath = _resolve_cache_dir(raw_path, cfg)
        tag = dpath.name
        tags.append(tag)
        if cfg.verbose:
            print(f"  [shap cache] {tag}  ({dpath})")

        npz_path = dpath / "die_shap.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"die_shap.npz 없음: {npz_path}")

        meta_path = dpath / "meta.json"

        # OOF 먼저 — top-K 컬럼 결정 (각 캐시는 importance 정렬이 self-consistent)
        oof_shap, oof_sid, oof_xy, feat_names_full = _load_npz_split(npz_path, "oof")
        feat_names_kept, oof_shap_top = _select_top_k(
            feat_names_full, oof_shap, meta_path, cfg.shap_top_k
        )
        if cfg.shap_top_k > 0 and cfg.verbose:
            print(f"    [top-K] {len(feat_names_kept)}/{len(feat_names_full)} 컬럼")

        # val/test는 같은 컬럼 인덱스 적용
        name_to_idx_full = {n: i for i, n in enumerate(feat_names_full)}
        kept_idx = [name_to_idx_full[n] for n in feat_names_kept]

        val_shap, val_sid, val_xy, val_feat_names = _load_npz_split(npz_path, "val")
        if val_feat_names != feat_names_full:
            raise RuntimeError(f"{tag}: val/oof feature_names 불일치")
        val_shap_top = val_shap[:, kept_idx]

        test_shap, test_sid, test_xy, test_feat_names = _load_npz_split(npz_path, "test")
        if test_feat_names != feat_names_full:
            raise RuntimeError(f"{tag}: test/oof feature_names 불일치")
        test_shap_top = test_shap[:, kept_idx]

        # bundle key_df 순서로 reorder
        X_oof_aligned = _align_to_keydf(
            oof_shap_top, oof_sid, oof_xy, key_df_oof, tag, "oof"
        )
        X_val_aligned = _align_to_keydf(
            val_shap_top, val_sid, val_xy, key_df_val, tag, "val"
        )
        X_test_aligned = _align_to_keydf(
            test_shap_top, test_sid, test_xy, key_df_test, tag, "test"
        )

        # 컬럼명 prefix
        if cfg.shap_prefix_with_tag:
            col_names = [f"{tag}::{n}" for n in feat_names_kept]
        else:
            col_names = list(feat_names_kept)

        all_oof.append(X_oof_aligned.astype(float))
        all_val.append(X_val_aligned.astype(float))
        all_test.append(X_test_aligned.astype(float))
        all_names.extend(col_names)
        counts.append(len(feat_names_kept))

    X_oof = np.hstack(all_oof)
    X_val = np.hstack(all_val)
    X_test = np.hstack(all_test)

    if cfg.verbose:
        print(
            f"  [shap total] cols={len(all_names)} from {len(tags)} cache(s) "
            f"(per-cache cols: {counts})"
        )

    return {
        "X_oof": X_oof,
        "X_val": X_val,
        "X_test": X_test,
        "names": all_names,
        "tags": tags,
        "cache_col_counts": counts,
    }
