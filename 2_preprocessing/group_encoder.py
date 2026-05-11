"""
그룹 키(lot, wafer, (wafer, position)) 기반 Out-of-Fold Target Encoding 모듈.

`run_wf_xy`("lot_wafer_x_y")에서 뽑은 그룹 ID마다 die-level target encoding을
empirical Bayes smoothing과 함께 만든다. 그룹당 3종 encoding을 생성한다 (zero-inflated 특성 분리):
    f"{name}_te"        : smoothed E[Y | group]            (그룹 평균)
    f"{name}_zero_rate" : smoothed P(Y=0 | group)          (그룹의 zero 비율)
    f"{name}_pos_mean"  : smoothed E[Y | group, Y>0]       (그룹 안 양수 값들의 평균)

train OOF 산출:
    - unit ID(`ufs_serial`) 단위로 GroupKFold(기본 5-fold, n_folds 인자) → 각 fold의 검증분은 나머지 fold로 학습한 encoding으로 채움
    - 같은 unit의 4 die가 서로 다른 fold로 흩어지지 않게 막아 leakage 차단
val/test 산출:
    - train 전체로 fit한 lookup 사용. train에 없던 그룹은 global mean(zero_rate/pos_mean도 동일)으로 대체

Smoothing 공식:
    enc = (n * group_mean + alpha * global_mean) / (n + alpha)
    - alpha 작음 → 그룹 고유 신호를 살림 (소그룹에선 noise 위험)
    - alpha 큼   → global 평균 쪽으로 끌어당김 (안전하지만 신호 약화)
    이 데이터는 lot이 28개 × 평균 ~3,700 die/group으로 그룹이 매우 커서 alpha에 거의 둔감하다. 기본 alpha=20이면 충분.

사용 예시:
    >>> from group_encoder import GroupTargetEncoder, DEFAULT_GROUP_SPECS, get_default_protected_cols
    >>> xs = GroupTargetEncoder.parse_group_columns(xs)  # lot_id, wafer_id 컬럼 추가
    >>> enc = GroupTargetEncoder(alpha=20.0)
    >>> enc_dict = enc.fit_transform(xs, ys["train"])
    >>> for col, arr in enc_dict.items():
    ...     xs[col] = arr
    >>> # cleaning이 이 컬럼들을 건드리지 않게 보호:
    >>> protected = get_default_protected_cols()
    >>> run_cleaning(xs, feat_cols + protected, xs_dict, protected_cols=protected, ...)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from utils.config import KEY_COL, SPLIT_COL, TARGET_COL, POSITION_COL


# 기본 그룹 = lot / wafer / (wafer, position) 3개 → 3종씩 = 9 encoding.
# (lot, position) 그룹은 lot 단독과 거의 동일(중복)이라 기본에서 뺌.
DEFAULT_GROUP_SPECS = [
    ("lot",   "lot_id"),
    ("wafer", "wafer_id"),
    ("wp",    ["wafer_id", POSITION_COL]),
]

# 검증/비교용 풀 세트 — (lot, position)까지 포함한 4그룹 → 12 encoding
ALL_GROUP_SPECS = [
    ("lot",   "lot_id"),
    ("wafer", "wafer_id"),
    ("lp",    ["lot_id", POSITION_COL]),
    ("wp",    ["wafer_id", POSITION_COL]),
]


class GroupTargetEncoder:
    """
    그룹 키 기반 OOF target encoder (그룹당 3종 encoding, empirical Bayes smoothing).

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

    # --- 정적 헬퍼들 ---

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
        xs = xs.copy()                              # 원본 보호
        xs[lot_col]   = parts[0]                    # 작업번호
        xs[wafer_col] = parts[0] + "_" + parts[1]   # 작업번호_웨이퍼번호 (lot 안에서 유일한 웨이퍼 ID)
        return xs

    @staticmethod
    def _pids_for_spec(xs, group_col):
        """그룹 ID 시리즈 생성. 컬럼 하나면 그대로, 여러 컬럼이면 '_'로 이어붙여 복합 키로."""
        if isinstance(group_col, (list, tuple)):
            return xs[list(group_col)].astype(str).agg("_".join, axis=1).values
        return xs[group_col].values

    @staticmethod
    def _smoothed_mean(pids_arr, y_arr, gm, alpha):
        """그룹별 (개수 n, 평균 mean)을 구해 smoothed 평균 {group_id: enc} dict로 반환.

        enc = (n*그룹평균 + alpha*전체평균) / (n + alpha) — 작은 그룹일수록 전체평균(gm) 쪽으로 당겨짐.
        """
        df = pd.DataFrame({"pid": pids_arr, "y": y_arr})
        agg = df.groupby("pid")["y"].agg(["size", "mean"])
        agg["enc"] = (agg["size"] * agg["mean"] + alpha * gm) / (agg["size"] + alpha)
        return agg["enc"].to_dict()

    # --- 그룹 하나에 대한 encoding 빌드 ---

    def _build_one_group(self, name, group_col, xs, ufs_tr, die_y_tr,
                         is_tr, is_va, is_te, fold_splits):
        """그룹 spec 하나 → 3종 encoding(te / zero_rate / pos_mean) dict."""
        pids = self._pids_for_spec(xs, group_col)   # 전 행의 그룹 ID
        train_pids = pids[is_tr]                    # train 행만

        # 3종 각각의 전역 평균 (smoothing의 gm, 그리고 미등장 그룹의 fallback)
        gm_mean = float(die_y_tr.mean())                       # E[Y]
        is_zero = (die_y_tr == 0).astype(np.float64)
        gm_zero = float(is_zero.mean())                        # P(Y=0)
        pos_y = die_y_tr[die_y_tr > 0]
        gm_pos = float(pos_y.mean()) if len(pos_y) else 0.0    # E[Y|Y>0]

        n_tr = is_tr.sum()
        oof_mean = np.zeros(n_tr, dtype=np.float64)   # train 각 행의 OOF encoding이 채워질 버퍼
        oof_zero = np.zeros(n_tr, dtype=np.float64)
        oof_pos  = np.zeros(n_tr, dtype=np.float64)

        ufs_s = pd.Series(ufs_tr)
        for tr_units, vl_units in fold_splits:        # GroupKFold fold 순회 (unit ID 단위 분할)
            tr_mask = ufs_s.isin(tr_units).values     # 이 fold의 학습 die
            vl_mask = ufs_s.isin(vl_units).values     # 이 fold의 검증 die (여기를 채울 차례)

            tr_pids = train_pids[tr_mask]
            tr_y = die_y_tr[tr_mask]
            tr_zero = (tr_y == 0).astype(np.float64)

            # te: 학습 fold로 그룹평균 lookup 만들고, 검증 fold 행에 매핑 (없는 그룹은 gm)
            m_mean = self._smoothed_mean(tr_pids, tr_y, gm_mean, self.alpha)
            oof_mean[vl_mask] = (
                pd.Series(train_pids[vl_mask]).map(m_mean).fillna(gm_mean).values
            )

            # zero_rate: 같은 방식으로 P(Y=0) lookup
            m_zero = self._smoothed_mean(tr_pids, tr_zero, gm_zero, self.alpha)
            oof_zero[vl_mask] = (
                pd.Series(train_pids[vl_mask]).map(m_zero).fillna(gm_zero).values
            )

            # pos_mean: 양수 die만 모아 E[Y|Y>0] lookup (학습 fold에 양수가 하나도 없으면 그냥 gm_pos)
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

        # train 전체로 다시 fit한 lookup — val/test는 OOF가 의미 없으므로 이걸로 채운다
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
            """전 행 길이의 배열 만들기: train=OOF값, val/test=전체 lookup값(없으면 gm_default)."""
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

    # --- 공개 API ---

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

        # 그룹 spec이 참조하는 컬럼이 xs에 다 있는지 먼저 확인 (없으면 parse_group_columns를 빼먹은 것)
        for name, col_or_cols in group_specs:
            cols = [col_or_cols] if isinstance(col_or_cols, str) else list(col_or_cols)
            missing = [c for c in cols if c not in xs.columns]
            if missing:
                raise ValueError(
                    f"Group '{name}' references missing columns: {missing}. "
                    f"Did you call GroupTargetEncoder.parse_group_columns(xs) first?"
                )

        # split별 마스크 — xs에 train/val/test가 모두 있어야 함
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

        # unit-level y를 die에 broadcast (각 die는 자기 unit의 health 값을 target으로 가짐)
        y_train_unit = ys_train.set_index(KEY_COL)[TARGET_COL]
        die_y_tr = pd.Series(ufs_tr).map(y_train_unit).values

        if pd.isna(die_y_tr).any():
            # train die의 ufs_serial이 ys_train에 없으면 매핑이 NaN → ys_train이 train 전체를 덮는지 확인 필요
            n_nan = int(pd.isna(die_y_tr).sum())
            raise ValueError(
                f"{n_nan} train die have unmapped Y "
                f"(ufs_serial in xs but not in ys_train). "
                f"ys_train이 train 전체 unit을 커버하는지 확인."
            )

        # unit ID를 그룹으로 한 GroupKFold → fold마다 (학습 unit 집합, 검증 unit 집합) 튜플
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

        # 그룹 spec마다 3종 encoding을 만들어 한 dict에 모음
        enc_dict = {}
        for name, col_or_cols in group_specs:
            group_enc = self._build_one_group(
                name, col_or_cols, xs, ufs_tr, die_y_tr,
                is_tr, is_va, is_te, fold_splits,
            )
            enc_dict.update(group_enc)

        return enc_dict


# ------------------------------------------------------------
# 헬퍼
# ------------------------------------------------------------

def get_default_protected_cols(group_specs=None):
    """
    GroupTargetEncoder 산출물의 컬럼명 list 반환.

    `cleaning.run_cleaning(..., protected_cols=...)` 인자로 그대로 넘길 수 있음
    (target encoding 결과는 cleaning의 상관/상수 제거가 손대면 안 되므로 보호 목록에 넣는다).

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
