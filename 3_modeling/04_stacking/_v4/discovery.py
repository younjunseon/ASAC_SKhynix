"""die-level 모델 풀 탐색 + 행렬 구성 + GroupKFold helper.

v2의 discover_models / build_base_matrix를 die-level CSV 기준으로 재작성.

핵심:
- oof_die.csv / val_die.csv / test_die.csv를 보유한 모델만 후보로 (없으면 자동 제외).
- 행렬은 die-level: shape (N_die, K_base). 동일 unit의 die들이 인접하지 않을 수 있으므로 ufs_serial로 정렬 보장.
- GroupKFold는 ufs_serial 단위로 split → 같은 unit의 4 die가 train/val에 섞이지 않음 (leakage 방지).
- clf 모델의 die CSV는 raw `prob`(0~1)이 들어 있어 reg pred(health 스케일 ~0.001)와 스케일이 다르다.
  build_die_matrix에서 clf 컬럼만 자동으로 `prob × y_pos_const` 변환하여 reg와 동일 스케일로 맞춤.
  y_pos_const는 각 모델의 best_params.json에 저장되어 있다 (=E[Y|Y>0], 학습 시점 train OOF 기준).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import SqueezeV4Config

KEY_COL = "ufs_serial"
POS_COL = "position"
DIE_KEY_COL = "run_wf_xy"   # 모든 die CSV가 보유. (ufs_serial, run_wf_xy)가 die 단위 unique key.


def _read_die_split(dirpath: Path, split: str) -> pd.DataFrame:
    """{split}_die.csv를 읽어 정규화된 컬럼만 남긴 DataFrame 반환.

    - 매칭 키: (ufs_serial, run_wf_xy) — 이게 die 단위 unique. `position`은 모델에 따라 없을 수 있음.
    - 예측 컬럼: `pred` 우선, 없으면 `prob`을 pred로 rename (clf 모델 호환).
    - 정답 컬럼: `health` (있으면).
    """
    df = pd.read_csv(dirpath / f"{split}_die.csv")
    if "pred" not in df.columns:
        if "prob" in df.columns:
            df = df.rename(columns={"prob": "pred"})
        else:
            raise RuntimeError(f"{dirpath}/{split}_die.csv: pred도 prob도 없음 ({list(df.columns)})")
    keep = [c for c in (KEY_COL, DIE_KEY_COL, POS_COL, "pred", "health") if c in df.columns]
    return df[keep]


def _get_oof_rmse_unit_estimate(dirpath: Path) -> float:
    """모델 후보 필터링용 — oof_unit.csv가 있으면 unit RMSE, 없으면 die-level fallback.

    discover의 oof_rmse_cutoff 기준은 v2와 호환 유지를 위해 unit-level RMSE를 우선.
    """
    p_unit = dirpath / "oof_unit.csv"
    if p_unit.exists():
        df = pd.read_csv(p_unit)
        return float(np.sqrt(np.mean((df["pred"].values - df["health"].values) ** 2)))
    # fallback: die-level RMSE (broadcast된 y이므로 정확히 unit RMSE는 아니지만 컷오프 근사용)
    p_die = dirpath / "oof_die.csv"
    df = pd.read_csv(p_die)
    return float(np.sqrt(np.mean((df["pred"].values - df["health"].values) ** 2)))


def _get_y_pos_const(dirpath: Path) -> float | None:
    """clf 모델의 best_params.json에서 y_pos_const(=E[Y|Y>0])를 로드.

    clf가 아니거나 키가 없으면 None. discover_models에서 category='clf'인 모델에
    한해 호출되어, 없을 경우 RuntimeError로 명시적 실패시킨다 (silent skip 금지).
    """
    p = dirpath / "best_params.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    val = d.get("y_pos_const")
    return float(val) if val is not None else None


def discover_models(cfg: SqueezeV4Config) -> list[dict]:
    """4_output을 재귀 스캔해 die-level CSV 3종(oof/val/test_die)을 모두 가진 모델을 후보로 반환.

    Returns
    -------
    list[dict] — 각 dict: {name, path, rel, category, variant, oof_rmse}.
    oof_rmse 오름차순 정렬.
    """
    skip = {"_old", "_subset_search", "base", "curated", "_cache", "_temp"}
    if not cfg.include_combined:
        skip.add("combined")
    if not cfg.include_ts_reg:
        skip.add("reg")

    out_root = cfg.output_dir
    models: list[dict] = []
    for dp in sorted(out_root.rglob("oof_die.csv")):
        dirpath = dp.parent
        parts = dirpath.parts
        if any(s in parts for s in skip):
            continue
        # die CSV 3종이 다 있어야 채택
        if not (dirpath / "val_die.csv").exists():
            continue
        if not (dirpath / "test_die.csv").exists():
            continue

        rel = dirpath.relative_to(out_root)
        rel_parts = rel.parts
        if "clf" in rel_parts:
            category = "clf"
        elif rel_parts and rel_parts[0] == "01_zit":
            category = "zit"
        elif rel_parts and rel_parts[0] == "02_reg_single":
            category = "reg_single"
        elif "reverse" in rel_parts:
            category = "reverse"
        elif "combined" in rel_parts:
            category = "combined"
        elif "reg" in rel_parts:
            category = "ts_reg"
        else:
            category = "other"

        variant = next((x for x in rel_parts if x in ("raw", "hp", "pphp")), "default")
        try:
            oof_r = _get_oof_rmse_unit_estimate(dirpath)
        except Exception:
            oof_r = 999.0

        if oof_r > cfg.oof_rmse_cutoff:
            continue
        if cfg.no_clf and category == "clf":
            continue

        # clf 모델은 die-level pred가 raw prob(0~1)이므로 build_die_matrix에서
        # `prob × y_pos_const`로 health 스케일 변환이 필요하다. y_pos_const를 미리 로드.
        # clf인데 best_params.json에 y_pos_const가 없으면 변환할 수 없으므로 제외.
        y_pos_const = None
        if category == "clf":
            y_pos_const = _get_y_pos_const(dirpath)
            if y_pos_const is None:
                # 변환 불가 → 스태킹 입력에서 제외 (스케일 mismatch 방지)
                continue

        models.append({
            "name": "__".join(rel_parts),
            "path": dirpath,
            "rel": str(rel),
            "category": category,
            "variant": variant,
            "oof_rmse": oof_r,
            "y_pos_const": y_pos_const,   # clf만 값, reg는 None
        })

    models.sort(key=lambda m: m["oof_rmse"])
    return models


def build_die_matrix(models: list[dict], split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """die-level base 예측 행렬 + y_die_broadcast + key_df 반환.

    Returns
    -------
    X : (N_die, K) — float64
        K = len(models). 컬럼 순서는 models 입력 순서와 일치.
    y_die : (N_die,) — float64
        die-level broadcast된 health (한 unit의 4 die가 같은 값).
    key_df : DataFrame (N_die, ≥2)
        반드시 포함: [KEY_COL, DIE_KEY_COL].
        가능하면 포함: POS_COL (ref 모델이 가지고 있을 때만).
        unit 집계 시 xs로 그대로 넘기면 postprocess가 KEY_COL 기준 groupby.
        weighted 집계는 POS_COL이 있어야 동작 (없으면 postprocess가 무시).

    Notes
    -----
    - 행 정렬: ref(=models[0])의 (ufs_serial, run_wf_xy) 사전순.
    - 이후 모델들은 (ufs_serial, run_wf_xy) 키로 left-merge → 행 순서 보존.
    - 한 모델이라도 매칭 누락(같은 키 조합 없음) 시 RuntimeError.
    """
    if not models:
        raise RuntimeError("discover_models가 빈 리스트를 반환했습니다.")

    ref = _read_die_split(models[0]["path"], split).copy()
    ref = ref.sort_values([KEY_COL, DIE_KEY_COL]).reset_index(drop=True)
    key_cols = [KEY_COL, DIE_KEY_COL]
    if POS_COL in ref.columns:
        key_cols.append(POS_COL)
    key_df = ref[key_cols].copy()

    # POS_COL이 ref에 없으면, position 가진 다른 모델에서 attach.
    # weighted 집계는 POS_COL이 있어야 동작 — 가능한 한 attach 시도.
    if POS_COL not in key_df.columns:
        for m in models:
            cand = _read_die_split(m["path"], split)
            if POS_COL in cand.columns:
                key_df = key_df.merge(
                    cand[[KEY_COL, DIE_KEY_COL, POS_COL]],
                    on=[KEY_COL, DIE_KEY_COL], how="left",
                )
                if key_df[POS_COL].notna().all():
                    break
        if POS_COL in key_df.columns and key_df[POS_COL].isna().any():
            # 매칭 못한 die가 일부 남음 → POS_COL 자체를 떨군다 (weighted 집계 비활성).
            key_df = key_df.drop(columns=[POS_COL])
    n_die = len(ref)
    y_die = ref["health"].values.astype(float) if "health" in ref.columns else np.zeros(n_die)

    cols = []
    for m in models:
        df = _read_die_split(m["path"], split)
        merged = key_df[[KEY_COL, DIE_KEY_COL]].merge(
            df[[KEY_COL, DIE_KEY_COL, "pred"]],
            on=[KEY_COL, DIE_KEY_COL], how="left",
        )
        if merged["pred"].isna().any():
            n_missing = int(merged["pred"].isna().sum())
            raise RuntimeError(
                f"[build_die_matrix/{split}] {m['name']}에서 (ufs_serial, run_wf_xy) 매칭 누락 {n_missing}개. "
                f"die CSV의 행 set이 base와 일치하지 않습니다."
            )
        col = merged["pred"].values.astype(float)
        # clf 모델은 raw prob(0~1) → reg pred(health 스케일)와 맞추기 위해 y_pos_const 곱
        # y_pos_const = E[Y|Y>0] (학습 시점 train OOF 기준, best_params.json에 박제)
        ypc = m.get("y_pos_const")
        if ypc is not None:
            col = col * float(ypc)
        cols.append(col)

    X = np.column_stack(cols) if cols else np.zeros((n_die, 0))
    return X.astype(float), y_die, key_df


def make_group_kfold_splits(
    key_df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold by ufs_serial — 같은 unit의 4 die가 train/val에 섞이지 않게 split.

    Returns
    -------
    list of (tr_idx, vl_idx) tuples (n_splits개) — die-level 인덱스.

    Notes
    -----
    sklearn.GroupKFold는 deterministic이고 random_state 인자가 없다.
    seed로 결과를 흔들고 싶을 때는 unit unique 순서를 seed 기반으로 permute하여
    group 라벨을 부여 → 같은 unit은 같은 group → split 결과가 seed에 의존.
    """
    rng = np.random.default_rng(seed)
    units = key_df[KEY_COL].values
    unique_units = pd.unique(units)
    permuted = rng.permutation(unique_units)
    unit_to_gid = {u: i for i, u in enumerate(permuted)}
    groups = np.array([unit_to_gid[u] for u in units])
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(np.zeros(len(units)), groups=groups))
