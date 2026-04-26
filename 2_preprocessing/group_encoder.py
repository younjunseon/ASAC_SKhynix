"""
그룹 키(lot, wafer, (wafer, position) 등) 기반 Out-of-Fold Target Encoding 모듈.

`run_wf_xy` (lot_wafer_x_y)에서 파싱한 그룹 ID에 대해 die-level OOF target encoding을
empirical Bayes smoothing과 함께 생성한다.

3종 encoding per group (zero-inflated 분리):
    f"{name}_te"        : smoothed E[Y | group]            (raw mean)
    f"{name}_zero_rate" : smoothed P(Y=0 | group)          (제로 비율)
    f"{name}_pos_mean"  : smoothed E[Y | group, Y>0]       (양수만의 평균)

Train 산출:
    - GroupKFold(n_folds=5)을 unit-level(`ufs_serial`)에서 수행 → fold-out OOF
    - 같은 unit의 4 die가 서로 다른 fold에 들어가지 않도록 leakage 차단

Val/Test 산출:
    - train 전체 fit lookup 사용. 미등장 그룹은 global mean(zero_rate/pos_mean도 동일)으로 대체

Smoothing:
    enc = (n * group_mean + alpha * global_mean) / (n + alpha)
    - alpha 작음: 그룹 고유 신호 살림 (소그룹에 noise 위험)
    - alpha 큼  : global로 끌어당김 (안전, 신호 약화)

핵심 발견 (2026-04-25):
    - v3 12 encoding feature importance: (lot, position) 그룹이 lot 단독과 redundant
      (|r|이 소수점 4자리까지 동일, importance rank 25/26/29). DEFAULT_GROUP_SPECS는
      이를 제외한 9 encoding (3 그룹 × 3종)으로 구성.
    - v4 alpha grid sweep [2, 195] 결과 alpha 둔감 — test RMSE 변동 0.028%
      lot 28개 × 평균 3,700 die/group으로 너무 안정적이라 smoothing 효과 미미.
      기본값 alpha=20이면 충분. 별도 튜닝 권장 안 함.

기본 사용 예시:
    >>> from group_encoder import GroupTargetEncoder, DEFAULT_GROUP_SPECS, get_default_protected_cols
    >>> xs = GroupTargetEncoder.parse_group_columns(xs)  # adds lot_id, wafer_id
    >>> enc = GroupTargetEncoder(alpha=20.0)
    >>> enc_dict = enc.fit_transform(xs, ys["train"])
    >>> for col, arr in enc_dict.items():
    ...     xs[col] = arr
    >>> # cleaning에서 보호:
    >>> protected = get_default_protected_cols()
    >>> run_cleaning(xs, feat_cols + protected, xs_dict, protected_cols=protected, ...)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from utils.config import KEY_COL, SPLIT_COL, TARGET_COL, POSITION_COL


# 9 encoding 기본 그룹 (v3 importance 분석 결과 (lot, position) 제외)
DEFAULT_GROUP_SPECS = [
    ("lot",   "lot_id"),
    ("wafer", "wafer_id"),
    ("wp",    ["wafer_id", POSITION_COL]),
]

# 12 encoding (CL의 ALL_GROUP_SPECS — 검증 / 비교용)
ALL_GROUP_SPECS = [
    ("lot",   "lot_id"),
    ("wafer", "wafer_id"),
    ("lp",    ["lot_id", POSITION_COL]),
    ("wp",    ["wafer_id", POSITION_COL]),
]


class GroupTargetEncoder:
    """
    그룹 키 기반 OOF target encoder (3종 encoding/group, empirical Bayes smoothing).

    Parameters
    ----------
    alpha : float, default 20.0
        Smoothing 강도. enc = (n*group + alpha*global) / (n + alpha).
        클수록 global 평균으로 수축(소그룹에 안전), 작을수록 그룹 신호 살림.
    n_folds : int, default 5
        GroupKFold splits (train OOF 산출 시 unit-level CV).
    random_state : int, default 42
        sklearn은 GroupKFold에 random_state를 받지 않지만, 호환성/문서화 용도로 유지.
    """

    def __init__(self, alpha=20.0, n_folds=5, random_state=42):
        self.alpha = float(alpha)
        self.n_folds = int(n_folds)
        self.random_state = random_state

    # ------------------------------------------------------------
    # Helpers (static)
    # ------------------------------------------------------------

    @staticmethod
    def parse_group_columns(xs, src_col="run_wf_xy",
                            lot_col="lot_id", wafer_col="wafer_id"):
        """
        run_wf_xy(예: "0000000_25_24_25") → lot_id, wafer_id 컬럼 추가.

        - lot_id   = 첫 번째 토큰 (작업번호)
        - wafer_id = "{lot_id}_{wafer_no}" (lot 내 웨이퍼 단위)

        Returns
        -------
        DataFrame (xs.copy()에 lot_id, wafer_id 추가)
        """
        if src_col not in xs.columns:
            raise ValueError(f"'{src_col}' 컬럼이 xs에 없음")
        parts = xs[src_col].str.split("_", expand=True)
        xs = xs.copy()
        xs[lot_col]   = parts[0]
        xs[wafer_col] = parts[0] + "_" + parts[1]
        return xs

    @staticmethod
    def _pids_for_spec(xs, group_col):
        """단일 컬럼은 그대로, list/tuple은 join — 그룹 ID 시리즈 생성."""
        if isinstance(group_col, (list, tuple)):
            return xs[list(group_col)].astype(str).agg("_".join, axis=1).values
        return xs[group_col].values

    @staticmethod
    def _smoothed_mean(pids_arr, y_arr, gm, alpha):
        """그룹별 (n, mean) → smoothed mean dict."""
        df = pd.DataFrame({"pid": pids_arr, "y": y_arr})
        agg = df.groupby("pid")["y"].agg(["size", "mean"])
        agg["enc"] = (agg["size"] * agg["mean"] + alpha * gm) / (agg["size"] + alpha)
        return agg["enc"].to_dict()

    # ------------------------------------------------------------
    # Core: per-group encoding
    # ------------------------------------------------------------

    def _build_one_group(self, name, group_col, xs, ufs_tr, die_y_tr,
                         is_tr, is_va, is_te, fold_splits):
        """단일 그룹 spec → 3종 encoding (te / zero_rate / pos_mean) dict."""
        pids = self._pids_for_spec(xs, group_col)
        train_pids = pids[is_tr]

        gm_mean = float(die_y_tr.mean())
        is_zero = (die_y_tr == 0).astype(np.float64)
        gm_zero = float(is_zero.mean())
        pos_y = die_y_tr[die_y_tr > 0]
        gm_pos = float(pos_y.mean()) if len(pos_y) else 0.0

        n_tr = is_tr.sum()
        oof_mean = np.zeros(n_tr, dtype=np.float64)
        oof_zero = np.zeros(n_tr, dtype=np.float64)
        oof_pos  = np.zeros(n_tr, dtype=np.float64)

        ufs_s = pd.Series(ufs_tr)
        for tr_units, vl_units in fold_splits:
            tr_mask = ufs_s.isin(tr_units).values
            vl_mask = ufs_s.isin(vl_units).values

            tr_pids = train_pids[tr_mask]
            tr_y = die_y_tr[tr_mask]
            tr_zero = (tr_y == 0).astype(np.float64)

            m_mean = self._smoothed_mean(tr_pids, tr_y, gm_mean, self.alpha)
            oof_mean[vl_mask] = (
                pd.Series(train_pids[vl_mask]).map(m_mean).fillna(gm_mean).values
            )

            m_zero = self._smoothed_mean(tr_pids, tr_zero, gm_zero, self.alpha)
            oof_zero[vl_mask] = (
                pd.Series(train_pids[vl_mask]).map(m_zero).fillna(gm_zero).values
            )

            pos_msk = tr_y > 0
            if pos_msk.sum() > 0:
                m_pos = self._smoothed_mean(
                    tr_pids[pos_msk], tr_y[pos_msk], gm_pos, self.alpha
                )
                oof_pos[vl_mask] = (
                    pd.Series(train_pids[vl_mask]).map(m_pos).fillna(gm_pos).values
                )
            else:
                oof_pos[vl_mask] = gm_pos

        # train 전체 fit lookup (val/test용)
        m_mean_full = self._smoothed_mean(train_pids, die_y_tr, gm_mean, self.alpha)
        m_zero_full = self._smoothed_mean(train_pids, is_zero, gm_zero, self.alpha)
        pos_msk_full = die_y_tr > 0
        m_pos_full = (
            self._smoothed_mean(
                train_pids[pos_msk_full], die_y_tr[pos_msk_full],
                gm_pos, self.alpha,
            )
            if pos_msk_full.sum() > 0
            else {}
        )

        def _broadcast(oof_arr, m_full, gm_default):
            out = np.full(len(xs), gm_default, dtype=np.float64)
            out[is_tr] = oof_arr
            out[is_va] = (
                pd.Series(pids[is_va]).map(m_full).fillna(gm_default).values
            )
            out[is_te] = (
                pd.Series(pids[is_te]).map(m_full).fillna(gm_default).values
            )
            return out

        return {
            f"{name}_te":        _broadcast(oof_mean, m_mean_full, gm_mean),
            f"{name}_zero_rate": _broadcast(oof_zero, m_zero_full, gm_zero),
            f"{name}_pos_mean":  _broadcast(oof_pos,  m_pos_full,  gm_pos),
        }

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def fit_transform(self, xs, ys_train, group_specs=None):
        """
        Build encoded columns for every group_spec.

        Parameters
        ----------
        xs : DataFrame
            train + validation + test concat된 전체 die-level DataFrame.
            반드시 KEY_COL, SPLIT_COL, group_specs에서 참조하는 컬럼을 포함해야 함.
        ys_train : DataFrame
            train target (KEY_COL, TARGET_COL — unit-level).
        group_specs : list of (name, col_or_cols), optional
            (encoding prefix, source col 이름 단일 또는 list).
            None이면 DEFAULT_GROUP_SPECS (lot, wafer, wp — 9 encoding 3 그룹).

        Returns
        -------
        enc_dict : dict {col_name: np.ndarray}
            len(arr) == len(xs). 호출자가 `xs[col] = arr`로 주입.

        Notes
        -----
        - GroupKFold split은 unit-level → 같은 unit의 4 die가 fold 간 분리 안 됨 (leakage 차단).
        - val/test는 train 전체 fit dictionary로 lookup. 미등장 그룹은 global mean으로 대체.
        - dtype은 모두 float64. 트리 모델은 무영향, 선형/스케일러는 호출자가 적절히 처리.
        """
        if group_specs is None:
            group_specs = DEFAULT_GROUP_SPECS

        # 컬럼 존재 검증
        for name, col_or_cols in group_specs:
            cols = [col_or_cols] if isinstance(col_or_cols, str) else list(col_or_cols)
            missing = [c for c in cols if c not in xs.columns]
            if missing:
                raise ValueError(
                    f"Group '{name}' references missing columns: {missing}. "
                    f"Did you call GroupTargetEncoder.parse_group_columns(xs) first?"
                )

        # split mask + train Y broadcast
        is_tr = (xs[SPLIT_COL] == "train").values
        is_va = (xs[SPLIT_COL] == "validation").values
        is_te = (xs[SPLIT_COL] == "test").values
        if not (is_tr.any() and is_va.any() and is_te.any()):
            raise ValueError(
                f"xs[{SPLIT_COL}]에 train/validation/test 모두 존재해야 함. "
                f"counts: train={is_tr.sum()}, val={is_va.sum()}, test={is_te.sum()}"
            )

        ufs_all = xs[KEY_COL].values
        ufs_tr  = ufs_all[is_tr]

        y_train_unit = ys_train.set_index(KEY_COL)[TARGET_COL]
        die_y_tr = pd.Series(ufs_tr).map(y_train_unit).values

        if pd.isna(die_y_tr).any():
            n_nan = int(pd.isna(die_y_tr).sum())
            raise ValueError(
                f"{n_nan} train die have unmapped Y "
                f"(ufs_serial in xs but not in ys_train). "
                f"ys_train이 train 전체 unit을 커버하는지 확인."
            )

        # GroupKFold splits at unit level
        unique_train_units = np.unique(ufs_tr)
        gkf = GroupKFold(n_splits=self.n_folds)
        fold_splits = []
        for tr_idx, vl_idx in gkf.split(
            np.zeros(len(unique_train_units)),
            groups=unique_train_units,
        ):
            fold_splits.append((
                set(unique_train_units[tr_idx]),
                set(unique_train_units[vl_idx]),
            ))

        # 각 그룹별 encoding 빌드
        enc_dict = {}
        for name, col_or_cols in group_specs:
            group_enc = self._build_one_group(
                name, col_or_cols, xs, ufs_tr, die_y_tr,
                is_tr, is_va, is_te, fold_splits,
            )
            enc_dict.update(group_enc)

        return enc_dict


# ============================================================
# Helpers
# ============================================================

def get_default_protected_cols(group_specs=None):
    """
    GroupTargetEncoder 산출물의 컬럼명 list 반환.

    `cleaning.run_cleaning(..., protected_cols=...)` 인자로 그대로 넘길 수 있음.

    Parameters
    ----------
    group_specs : list, optional
        None이면 DEFAULT_GROUP_SPECS (9 encoding).

    Returns
    -------
    list of str
        예: ["lot_te", "lot_zero_rate", "lot_pos_mean",
             "wafer_te", "wafer_zero_rate", "wafer_pos_mean",
             "wp_te", "wp_zero_rate", "wp_pos_mean"]
    """
    if group_specs is None:
        group_specs = DEFAULT_GROUP_SPECS
    cols = []
    for name, _ in group_specs:
        cols.extend([f"{name}_te", f"{name}_zero_rate", f"{name}_pos_mean"])
    return cols
