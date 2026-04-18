# 시간 단축 수정안 (A: CLF 통합 + C: 포지션 가중치 수학해)

## A. CLF position 루프 통합 — fit 횟수 1/4

### 현재 구조

```
_run_clf_oof: for pos in [1,2,3,4]:
    pos별 독립 학습 (26K die × n_folds)
→ 3모델 × 4pos × 3fold = 36 fits (main)
→ 3모델 × 4pos × 5fold = 60 fits (rerun)
```

### 변경 후 구조

```
_run_clf_oof: 4 position concat → 1번 학습 (105K die × n_folds)
    → predict 후 position별로 분리해서 반환
→ 3모델 × 1concat × 3fold = 9 fits (main)
→ 3모델 × 1concat × 5fold = 15 fits (rerun)
```

### 성능 영향: 동일하거나 향상

- 4개 position의 die는 동일한 WT feature(X0~X1086), 동일한 label(unit health 기반)
- label 분포도 position간 동일 (stratified split으로 확인됨)
- 학습 데이터 4배 증가 → 트리 분할 기준이 더 안정적

### 수정 대상 함수 (e2e_hpo.py)

| 함수 | 위치 | 변경 내용 |
|------|------|----------|
| `_run_clf_oof` | line 108 | position 루프 제거, concat 학습 + 분리 반환 |
| `_run_clf_single` | line 310 | 동일 패턴 적용 |

### 반환 포맷: 변경 없음

```python
# 변경 전후 모두 동일한 포맷
clf_result = {
    1: {"train_proba": arr, "val_proba": arr, "test_proba": arr},
    2: {"train_proba": arr, "val_proba": arr, "test_proba": arr},
    3: {"train_proba": arr, "val_proba": arr, "test_proba": arr},
    4: {"train_proba": arr, "val_proba": arr, "test_proba": arr},
}
```

하위 소비자 15개 전수 확인 완료 — 포맷만 유지하면 전부 OK:
- `_run_clf_oof_multi`, `_apply_isotonic_calibration`
- `_prepare_unit_data` (position/unit 모두)
- `aggregate_die_to_unit`, `_compute_clf_metrics`
- `_save_per_model_oof`, main objective, rerun, 노트북 셀 등

파이프라인 스위치 전수 확인 완료 — 전부 OK:
- `clf_filter=True/False`, `clf_output='proba'/'binary'`
- `run_clf=True/False`, `reg_level='position'/'unit'`
- `calibration` isotonic, `mode='single'/'kfold'`
- `add_clf_proba_to_reg=True`, `imbalance_method`

### 필수 처리: GroupKFold로 leakage 방지

**문제**: 4 position concat 시 같은 unit의 die 4개가 동일 label 공유.
`StratifiedKFold`가 이걸 모르고 분할하면 같은 unit의 die가 train/val에 섞임 → label leakage.

**해결**: `GroupKFold(groups=ufs_serial)` 사용.

```python
# 변경 전 (position별 독립 — leakage 없음)
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
for tr_idx, val_idx in skf.split(X_tr, y_tr):
    ...

# 변경 후 (concat — GroupKFold로 unit 단위 분할)
groups = concat_df[KEY_COL].values  # ufs_serial
gkf = GroupKFold(n_splits=n_folds)
for tr_idx, val_idx in gkf.split(X_all, y_all, groups=groups):
    ...
```

`_run_clf_single`의 ES holdout도 동일하게 group-aware split 필요:
```python
# GroupShuffleSplit 또는 수동 group split
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=es_holdout, random_state=SEED)
tr_idx, es_idx = next(gss.split(X_all, y_all, groups=groups))
```

### position feature 추가

현재 CLF는 `feat_cols`(X0~X1086 서브셋)만 사용.
concat하면 "이 die가 몇번 position인지" 정보가 사라지므로 position을 feature로 추가:

- 트리 모델(LGBM/ET): 정수 position 컬럼 1개면 충분
- 선형 모델(logreg_enet): one-hot encoding (p_1~p_4)
- REG의 `_prepare_unit_data`에 이미 동일 패턴 존재 (line 823~844) → 그대로 참고

### 구현 스케치 (_run_clf_oof 변경)

```python
def _run_clf_oof(pos_data, feat_cols, clf_params, model_name,
                 n_folds, early_stop, label_col, imbalance_method,
                 clf_output="proba"):

    positions = sorted(pos_data.keys())

    # ── 1. concat ──
    train_frames, val_frames, test_frames = [], [], []
    for pos in positions:
        d = pos_data[pos]
        for split_name, container in [("train", train_frames),
                                       ("val", val_frames),
                                       ("test", test_frames)]:
            df = d[split_name][feat_cols].copy()
            df[POSITION_COL] = pos
            # OHE (선형 모델 대응)
            for p in positions:
                df[f"p_{p}"] = np.int8(1 if pos == p else 0)
            container.append((pos, df, d[split_name][KEY_COL].values,
                              d[split_name][label_col].values))

    X_tr_all = np.vstack([f[1].values for f in train_frames])
    y_tr_all = np.concatenate([f[3] for f in train_frames])
    groups_all = np.concatenate([f[2] for f in train_frames])  # ufs_serial
    pos_ids_tr = np.concatenate([np.full(len(f[1]), f[0]) for f in train_frames])

    # position별 길이 기록 (split-back용)
    pos_lengths_tr = {pos: len(pos_data[pos]["train"]) for pos in positions}

    # ── 2. 불균형 처리 ──
    actual_params = clf_params.copy()
    # ... (기존과 동일, concat 전체 기준으로 1회만 계산)

    # ── 3. GroupKFold ──
    gkf = GroupKFold(n_splits=n_folds)
    clf_feat_cols = feat_cols + [POSITION_COL] + [f"p_{p}" for p in positions]
    oof_proba = np.zeros(len(X_tr_all))
    fold_models = []

    for tr_idx, val_idx in gkf.split(X_tr_all, y_tr_all, groups=groups_all):
        clf = create_model(model_name, "clf", actual_params)
        # ... (학습 — 기존 ES 로직 그대로, 단 inner holdout도 group-aware)
        oof_proba[val_idx] = clf.predict_proba(X_tr_all[val_idx])[:, 1]
        fold_models.append(clf)

    # ── 4. val/test predict ──
    X_val_all = np.vstack([...])  # 동일 패턴
    X_test_all = np.vstack([...])
    val_proba_all = np.mean([m.predict_proba(X_val_all)[:, 1] for m in fold_models], axis=0)
    test_proba_all = np.mean([m.predict_proba(X_test_all)[:, 1] for m in fold_models], axis=0)

    # ── 5. position별 split-back ──
    clf_result = {}
    tr_offset, val_offset, test_offset = 0, 0, 0
    for pos in positions:
        n_tr = pos_lengths_tr[pos]
        n_val = len(pos_data[pos]["val"])
        n_test = len(pos_data[pos]["test"])
        clf_result[pos] = {
            "train_proba": oof_proba[tr_offset:tr_offset + n_tr],
            "val_proba":   val_proba_all[val_offset:val_offset + n_val],
            "test_proba":  test_proba_all[test_offset:test_offset + n_test],
        }
        tr_offset += n_tr; val_offset += n_val; test_offset += n_test

    return clf_result
```

---

## C. 포지션 가중치 sub-study — Optuna → scipy 수학해

### 현재 (Cell 18)

```python
# 모델당 150 Optuna trial → w1~w4 이산 그리드 탐색
# 3모델 × 150 trial = 450 trial
sub.optimize(_objective, n_trials=150)
```

### 변경 후

```python
from scipy.optimize import minimize

def find_best_pos_weights(pred_mat, y_val, bounds=(0.15, 0.35)):
    """
    pred_mat: (n_units, 4) — 각 position의 die-level 예측을 unit pivot한 것
    y_val:    (n_units,)   — unit-level 정답
    bounds:   w_i 범위 제약
    
    Returns: (w_best, best_rmse)
    """
    def obj(w):
        w_norm = w / w.sum()  # Dirichlet 정규화
        pred = pred_mat @ w_norm
        return np.sqrt(np.mean((y_val - pred) ** 2))

    res = minimize(
        obj,
        x0=np.array([0.25, 0.25, 0.25, 0.25]),
        method='SLSQP',
        bounds=[(bounds[0], bounds[1])] * 4,
        constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1.0},
    )
    w_best = res.x / res.x.sum()
    return w_best, res.fun
```

### 변경 근거

- 목적함수: `RMSE(y, Pw)` — P는 (n_units, 4) 고정 행렬, w는 4차원 벡터
- 이것은 **제약 조건 선형 최소자승** 문제 → 닫힌 해 존재
- `scipy.optimize.minimize(SLSQP)` + bounds `[0.15, 0.35]` + 합=1 제약으로 **정확한 최적해**를 0.01초에 반환
- Optuna 150 trial은 step=0.01 이산 그리드의 **근사치**이므로, 수학해가 **더 정확**

### 수정 위치

- Cell 18 전체를 위 함수로 교체
- `position_weight_results` dict 포맷은 그대로 유지 (Cell 20, Cell 22가 사용)
- Optuna study 객체는 더 이상 생성하지 않으므로, `position_weight_results[m]['study']` 키 제거
  - 이 키를 사용하는 후속 코드가 있으면 None으로 채움

### Cell 18 교체 코드

```python
from scipy.optimize import minimize

position_weight_results = {}
POS_ORDER = [1, 2, 3, 4]

def find_best_pos_weights(pred_mat, y_val, bounds=(0.15, 0.35)):
    def obj(w):
        w_norm = w / w.sum()
        return np.sqrt(np.mean((y_val - pred_mat @ w_norm) ** 2))
    res = minimize(
        obj, x0=[0.25]*4, method='SLSQP',
        bounds=[bounds]*4,
        constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1.0},
    )
    return res.x / res.x.sum(), res.fun

for m in e2e_params['reg_models']:
    val = _reg_oof_df(m, 'val')
    pivot = (val.pivot_table(index=KEY_COL, columns='position',
                              values='reg_pred_die', aggfunc='mean')
                .reindex(columns=POS_ORDER))
    y_val = (val.groupby(KEY_COL, sort=False)[TARGET_COL].first()
                .reindex(pivot.index).values)

    cfg = position_weight_substudy
    w_best, best_rmse = find_best_pos_weights(
        pivot.values, y_val,
        bounds=(cfg['weight_range'][0], cfg['weight_range'][1]),
    )
    position_weight_results[m] = {
        'best_w_pos': w_best,
        'best_rmse':  best_rmse,
        'study':      None,        # Optuna 제거 — 후속 호환용
        'pred_pivot': pivot,
        'y_val':      y_val,
    }
    print(f'{m:5s} | best w_pos = [{w_best[0]:.3f}, {w_best[1]:.3f}, '
          f'{w_best[2]:.3f}, {w_best[3]:.3f}]  | val RMSE = {best_rmse:.6f}')

# position_weight_rows (저장용) — scipy에는 top_k 개념이 없으므로 best 1개만
position_weight_rows = []
for m, r in position_weight_results.items():
    w = r['best_w_pos']
    position_weight_rows.append({
        'model': m, 'val_rmse': r['best_rmse'],
        'w_p1': w[0], 'w_p2': w[1], 'w_p3': w[2], 'w_p4': w[3],
    })

if SAVE_OUTPUTS:
    out_path = os.path.join(OOF_DIR, position_weight_substudy['save_path'])
    pd.DataFrame(position_weight_rows).to_csv(out_path, index=False)
    print(f'\n저장: {out_path}')
```

---

## 시간 절감 요약

| 항목 | 현재 | 변경 후 | 절감 |
|------|------|---------|------|
| **A. CLF fits (main, 10 trial)** | 360 fits | 90 fits | **75%** |
| **A. CLF fits (rerun)** | 60 fits | 15 fits | **75%** |
| **C. 포지션 가중치** | 450 Optuna trial (~90초) | scipy 3회 (~0.03초) | **~100%** |

A가 압도적으로 큰 효과. C는 구현이 쉽고 코드도 깔끔해짐.