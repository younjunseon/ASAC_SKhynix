# 추가 전처리 적용 계획 (final 파이프라인 기준)

> 1·2차 funnel 종료 후, final/ 제출 파이프라인에 추가로 시도해볼 수 있는
> 전처리 8종에 대한 우선순위 + 복잡도 + 안전 적용 plan.
>
> 코드 검증 기반: `final/modules/{preprocess, hpo, scaler, postprocess, models}.py`,
> `final/modules/{cleaning, outlier, scaling}.py`,
> `2_preprocessing/{meta_features, encoding, feature_selection}.py`,
> `utils/{data, config, aggregate}.py`, `final/{01,02,03,04}.ipynb`.

---

## 0. 한눈에 보기

| 순위 | 항목 | 부류 | hook 필요 | 신규 모듈 LoC | 파일 변경 | 위험 |
|---|---|:---:|:---:|:---:|---|:---:|
| 1 | OOF-safe Feature Selection | B | O | ~80 | hpo.py, preprocess.py | 중 |
| 2 | WoE Binning (supervised) | B | O | ~80 | hpo.py | 중 |
| 3 | Degenerate Binarization | A | X | 0 (기존) | preprocess.py | 저 |
| 4 | AutoFE pairwise (OOF-safe 재배선) | B | O | ~50 (wrap) | hpo.py | 중 |
| 5 | Die-level deviation features | A | X | ~60 | preprocess.py | 저 |
| 6 | Adversarial Validation 진단 + 필터 | A | X | ~50 + 노트북 | preprocess.py(EXCLUDE_COLS) | 저 |
| 7 | GBDT Leaf Encoding (enet 전용) | B | O | ~50 | hpo.py | 중 |
| 8 | Imputation Ablation (spatial vs median+ind vs LGBM-native) | A | X | 0 (기존) | preprocess.py(_FIXED 해제) | 중 |

**부류 A**: `preprocess.run()` 안에 단계 추가만. fold loop 무수정.
**부류 B**: `hpo.py` fold loop에 callback hook 1개 추가 + 항목별 callback 함수.

> "hook" = `objective`/`refit_best` fold loop 안의 `_fit_predict_fold` 호출 직전에
> `if fold_transform_fn is not None: X_tr, X_vl = fold_transform_fn(X_tr, y_tr, X_vl)`
> 한 줄을 끼우는 것. 모델 코드, search space, postprocess, blending 무수정.

---

## 1. 코드베이스 검증 결과 (건드리는 곳/안 건드리는 곳)

### 1.1 변경 대상 파일 (8종 누적, 모든 항목 적용 시)

| 파일 | 변경 내용 | 영향 범위 |
|---|---|---|
| `final/modules/preprocess.py` | `DEFAULT_PARAMS`에 토글 추가, `run()`에 단계 호출, `_FIXED` 일부 해제(8번) | A군 전체 |
| `final/modules/hpo.py` | `run_hpo`/`refit_best`에 `fold_transform_fn` 파라미터 추가, fold loop 내부 1줄 | B군 전체 |
| `final/modules/feature_engineering/` (신규 디렉토리) | `voting_fs.py`, `woe.py`, `autofe_oof.py`, `leaf_encoding.py` | B군 callback 4개 |
| `final/modules/cleaning.py` | (기존 `binarize_degenerate` 그대로 사용) | 0줄 |
| `final/01~03_*.ipynb` | `PARAMS` dict에 토글 키 추가, hook callback 주입 | 노트북 3개 |
| `final/04_blend.ipynb` | 변경 없음 | — |

### 1.2 절대 건드리지 않는 파일

- `final/modules/models.py` — 모델 클래스/search space 무수정
- `final/modules/zit.py` — ZITboost 정의 무수정
- `final/modules/scaler.py` — RobustScaler 분기 무수정
- `final/modules/postprocess.py` — 집계/threshold 튜닝 무수정 (별도 결정 사항)
- `final/modules/blending.py` — 무수정
- `utils/data.py`, `utils/config.py`, `utils/aggregate.py`, `utils/evaluate.py` — 무수정
- `2_preprocessing/`의 기존 모듈들 — wrap만, 직접 수정 없음

### 1.3 큰 흐름 (변하지 않음)

```
load_xs/load_ys → split_xs                                 [무수정]
   ↓
preprocess.run() → xs_train/val/test, feat_cols_clean      [단계 추가만]
   ↓
hpo.run_hpo() → fold loop:                                 [hook 1줄 추가]
   for fold:
     X_tr, X_vl 분할
     [★ hook 자리 — fold-internal 변환]
     [★ scaler (기존)]
     model.fit → predict (die-level)                       [무수정]
   ↓
hpo.refit_best() → K-fold 재학습                           [hook 1줄 추가]
   ↓
postprocess.tune_and_apply() → die→unit 집계 + threshold   [별도 결정]
   ↓
blending                                                   [무수정]
```

**불변 보장**: die-level 학습, position 보존, unit aggregate, RMSE objective 모두 그대로.

---

## 2. hook 인프라 (B군 공통, 1회 작업)

### 2.1 hpo.py 변경

`run_hpo(...)`와 `refit_best(...)` 시그니처에 인자 1개 추가:

```python
def run_hpo(..., fold_transform_fn=None, ...):
    ...
    def objective(trial):
        for tr_mask, vl_mask in fold_masks:
            X_tr, y_tr = X_full[fit_mask], y_die_fit[fit_mask]
            X_vl       = X_full[vl_mask]

            # ★ NEW: fold-internal 변환 hook (Y leak 차단)
            if fold_transform_fn is not None:
                X_tr, X_vl = fold_transform_fn(X_tr, y_tr, X_vl)

            # 기존 스케일러
            if _scaler.needs_scaling(model_name):
                ...
            res = _fit_predict_fold(model_name, hp, X_tr, y_tr, X_vl)
            ...
```

`refit_best`도 동일 위치에 동일 한 줄. val/test에 대한 transform은 콜백 내부에서
`X_tr` 기준 fit한 변환을 fold마다 X_val_full/X_test_full에 적용해 fold 평균에 반영.

### 2.2 callback 시그니처 표준

```python
def callback(X_tr: np.ndarray, y_tr: np.ndarray,
             X_vl: np.ndarray,
             X_val: np.ndarray = None, X_test: np.ndarray = None,
             ) -> tuple:
    """
    Returns
    -------
    (X_tr_new, X_vl_new) 또는 (X_tr_new, X_vl_new, X_val_new, X_test_new)
    """
```

콜백이 X_val/X_test도 받으면 fold마다 변환된 val/test도 반환 → `refit_best`가
fold 평균에 사용. 받지 않으면 X_val/X_test는 변환 없음 (학습 시점 한정 변환).

### 2.3 위험 요소

- **컬럼 수 증가/감소 호환**: callback이 컬럼 수를 바꾸면 (FS는 감소, leaf encoding은
  증가) `_fit_predict_fold` 이후 단계는 X 차원 무관하므로 문제 없음.
  단 **enet 스케일러는 컬럼 수에 의존** → callback이 enet에 적용될 때는 callback이
  스케일러 호출 이전에 위치해야 함 (현 plan: callback이 scaler 앞에 있음, 안전).
- **fold 간 컬럼 수 일치 보장 안 됨**: FS가 fold마다 다른 컬럼을 고를 수 있음 →
  val/test 예측 시 fold 평균이 의미 없음. 해결: callback이 X_tr, X_vl, X_val, X_test를
  같은 컬럼 인덱스로 변환해서 반환하도록 설계.
- **검증 방법**: hook=`None`으로 1회 실행 → 기존 결과와 byte-level 일치 확인 후 진행.

---

## 3. A군 (preprocess.run에 단계 추가, hook 불필요)

### 3.1 [순위 3] Degenerate Binarization

**상태**: 이미 [cleaning.py:747](modules/cleaning.py#L747) `binarize_degenerate()` 구현됨.
final `preprocess.run()`에서 호출만 안 함.

**수정**:
```python
# preprocess.py DEFAULT_PARAMS에 추가
"binarize_degenerate":   False,    # True면 cleaning 후·outlier 전 적용
"binarize_top_pct":      0.99,     # 최빈값 비율 > 이 값
"binarize_max_unique":   None,     # nunique <= 이 값 (OR 조건). None=top_pct만
```

```python
# preprocess.py run() 안, run_cleaning 직후·run_outlier_treatment 직전
if effective["binarize_degenerate"]:
    from .cleaning import binarize_degenerate
    xs_train, xs_val, xs_test, bin_report = binarize_degenerate(
        xs_train, xs_val, xs_test, clean_feat_cols,
        top_value_threshold=effective["binarize_top_pct"],
        max_unique=effective["binarize_max_unique"],
    )
    report["binarize"] = bin_report
```

**위험 검토**:
- ✅ binarize는 컬럼 수 불변(값만 0/1로) → feat_cols 무수정
- ✅ train 기준 mode를 val/test에 동일 적용 (이미 모듈 내 처리됨)
- ✅ winsorize 직전이라 0/1로 변한 컬럼이 winsorize에 들어가도 무해 (0/1은 이미 분위수 경계 안)
- ⚠️ dtype: int8로 변환됨 → 스케일러 직전에 float32로 cast 필요? scaler.py 확인:
  `(df[feat_cols] - medians) / iqr_safe` — pandas가 자동 upcast → float64 가능성.
  **대응**: scaler.py 변경 없이도 동작하나 dtype 일관성 위해 binarize 결과를 float32로
  cast하는 한 줄을 binarize 호출 직후 추가 권장.

**검증 스모크 테스트**:
1. `binarize_degenerate=False`로 실행 → 기존 결과와 동일한지
2. `top_pct=0.99`로 켜고 변환 컬럼 수 출력
3. ZITboost 1 trial 돌려서 RMSE 얻고 baseline과 비교

---

### 3.2 [순위 5] Die-level Deviation Features

**상태**: [meta_features.py:65](../../2_preprocessing/meta_features.py#L65) `create_lot_stats_features()`는
unit-level lot mean/std merge만 함. **die가 자기 lot 평균에서 얼마나 벗어나는가**의
deviation은 아직 만든 적 없음.

**수정**:
```python
# preprocess.py DEFAULT_PARAMS에 추가
"add_deviation":         False,
"deviation_top_k":       50,         # LGBM importance top-K feature에만 deviation 생성
"deviation_levels":      ("lot",),   # ("lot",) 또는 ("lot", "wafer")
```

```python
# preprocess.py run(), outlier 직후 (winsorize된 값에 대해 계산)
if effective["add_deviation"]:
    from .feature_engineering.deviation import add_deviation_features
    xs_train, xs_val, xs_test, dev_cols = add_deviation_features(
        xs_train, xs_val, xs_test,
        feat_cols=clean_feat_cols,
        ys_train=ys.get("train"),
        top_k=effective["deviation_top_k"],
        levels=effective["deviation_levels"],
    )
    clean_feat_cols = clean_feat_cols + dev_cols
```

**신규 모듈 `final/modules/feature_engineering/deviation.py`** (~60줄):
- top-K 선정: train 데이터에 LGBM 1회 fit → importance top-K
- lot 파싱: `meta_features.parse_run_wf_xy(xs, prefix="_", inplace=True)`
- 각 (col, level) 쌍에 대해:
  - **train 행만으로** `lot_mean = xs_train.groupby(_lot)[col].mean()` 계산
  - 새 컬럼 `dev_{col}_lot = xs[col] - lot_mean.loc[xs[_lot]]`
  - val/test도 train의 lot_mean 매핑 (unknown lot은 global train mean fallback)
- 임시 컬럼 정리

**위험 검토**:
- ✅ Y 사용 안 함 (top-K 선정에만 LGBM이 Y 사용하지만, 선정 자체는 train만 — 이건 covariate-only 방식이라 fold-internal 안 해도 안전. 더 엄격히 가려면 fold-internal로 옮길 수 있으나 cost 큼)
- ⚠️ **leak 가능성**: top-K 선정에 train 전체 Y 사용 → "선정된 feature 자체"는 fold val 정보 약하게 새지만, deviation 값은 X에서만 계산됨. 보수적으로 가려면 top-K 선정도 fold-internal로 옮기거나, 통째로 hook으로 옮기는 것을 고려. **plan**: 1차는 단순 train 전체 top-K, 효과 확인 후 fold-internal 검토
- ⚠️ lot_mean 계산 시 **train 행만** 사용 (val/test 분포 누수 차단)
- ⚠️ dtype: float32 유지하려면 deviation 결과를 `astype('float32')` 명시
- ⚠️ X1086은 int32라 deviation 계산 시 자동 float64 승격 → top-K에서 X1086 제외 권장 또는 deviation 컬럼만 float32 cast
- ✅ 컬럼 수 증가 (top_k × len(levels) × 1, 기본 50개) → `_build_X(feat_cols)`가 그대로 처리

**검증 스모크 테스트**:
1. `add_deviation=False` → 기존과 동일
2. `top_k=10, levels=("lot",)` → +10 컬럼 확인, deviation 값이 mean 0 근처인지
3. unit RMSE 비교

---

### 3.3 [순위 6] Adversarial Validation 진단 + 필터

**상태**: 미구현. 진단 결과를 `EXCLUDE_COLS`에 추가하는 형태로 적용.

**구조** (다른 항목과 다름 — 노트북 + 모듈):

신규 노트북 `final/00_adversarial_validation.ipynb`:
1. xs_train + xs_val concat → label `is_val = (split=="validation")`
2. cleaning + winsorize 적용 (final preprocess.run의 `train+val` 합본 변형 호출)
3. LGBM 5-fold CV `objective='binary'` → AUC 측정
4. AUC > 0.55면 importance top-N (예: 20개) 추출
5. AUC > 0.6이면 top-N을 `DRIFT_COLS_VAL.json` 으로 저장
6. xs_train + xs_test 도 동일 절차 → `DRIFT_COLS_TEST.json`

신규 모듈 `final/modules/feature_engineering/adversarial.py` (~50줄):
- `compute_drift_features(xs_a, xs_b, feat_cols, threshold_auc=0.6, top_n=20)` 한 함수

**적용**: 노트북에서 `EXCLUDE_COLS` 변수에 union 추가:
```python
import json
with open("DRIFT_COLS_VAL.json") as f:
    drift_cols = json.load(f)
EXCLUDE_COLS_EXTENDED = list(set(EXCLUDE_COLS + drift_cols))
pp = preprocess.run(xs, ys_input, feat_cols, xs_dict,
                    params=PARAMS, exclude_cols=EXCLUDE_COLS_EXTENDED)
```

**위험 검토**:
- ✅ Y 사용 안 함 (split label은 Y가 아님, 데이터 분할의 메타 정보)
- ⚠️ split 라벨 자체가 일종의 정보 — train+val 합본으로 분류기를 학습하면 val 의 X 특성이
  분류기에 사용됨. drift 진단 목적으론 정상, 단 이걸 "feature 제거" 액션으로 옮기면
  사실상 **val의 X 분포를 보고 EXCLUDE_COLS를 결정**한 것 → val 평가가 살짝 낙관 편향.
  **대응**: test 비공개 구조에서 train+test AV는 실제 운영처럼 가능하지만,
  train+val AV는 신중히. 1차로는 train+val AV를 진단으로만 보고, 제거는 보수적으로
  (예: AUC > 0.7 이상에서만)
- ✅ preprocess.run 무수정. exclude_cols 인자가 이미 있음 (preprocess.py:117)
- ✅ 다른 모든 단계에 무영향

**검증 스모크 테스트**:
1. 노트북 실행 → AUC, drift 컬럼 리스트 출력
2. 빈 리스트면 AV 결과 "drift 없음" 로깅하고 끝
3. drift 있으면 ablation: drift 컬럼 포함/제외로 ZITboost 1 trial × 2 비교

---

### 3.4 [순위 8] Imputation Ablation

**상태**: `_FIXED["imputation_method"]="spatial"` 하드코딩. 다른 옵션은 코드상 지원되나
플래그가 막혀 있음.

**수정**:
```python
# preprocess.py
# _FIXED 에서 imputation_method 제거, DEFAULT_PARAMS 로 이동
DEFAULT_PARAMS = {
    ...,
    "imputation_method":  "spatial",   # 'spatial' | 'median' | 'knn' | 'native'
    "knn_neighbors":      5,
}

_FIXED = {
    "remove_duplicates":  True,
    "outlier_method":     "winsorize",
    "outlier_lower_pct":  0.0,
    "outlier_upper_pct":  0.99,
}
```

```python
# run() 안
xs_train, xs_val, xs_test, clean_feat_cols, report = run_cleaning(
    ..., imputation_method=effective["imputation_method"],
    knn_neighbors=effective["knn_neighbors"], ...
)
```

**`'native'` 옵션 추가**: cleaning.py의 `run_cleaning`에 한 분기 추가 — imputation을
스킵하고 NaN을 그대로 LGBM/XGB에 넘김. ElasticNet은 NaN 불가 → 사용 시 모델
호환성 책임은 사용자가.

**위험 검토**:
- ⚠️ **재현성 변경**: `_FIXED`에서 `imputation_method`를 빼면 이전 실험 결과의
  `effective_params`에 기록된 `_fixed_imputation_method` 키가 사라짐. JSON 호환성
  깨짐. **대응**: `_FIXED` 키 이름을 유지하되 `DEFAULT_PARAMS`에서 override 가능하게.
  더 나은 방법: `_FIXED` 키 자체를 삭제하지 말고, `DEFAULT_PARAMS["imputation_method"]`
  값이 `_FIXED["imputation_method"]`를 덮어쓰도록 `_merge_params` 수정. 또는
  `_FIXED`를 그대로 두고 `DEFAULT_PARAMS["imputation_method_override"]` 선택적 키
  추가 (None이면 _FIXED 값 사용)
- ⚠️ ElasticNet은 NaN 불가 → `'native'` 선택 시 enet 모델 자동 비활성 또는 에러
- ⚠️ `'spatial'` vs `'median'` 결과 비교 시 학습 데이터가 달라짐 → 다른 모든 trial
  결과와 직접 비교 어려움. 각 imputation_method마다 별도 EXP_ID로 진행
- ✅ cleaning.py의 `impute_missing(method='knn')` 이미 구현 (line 463)
- ✅ cleaning.py의 `impute_spatial` 이미 구현

**검증 스모크 테스트**:
1. `imputation_method='spatial'` → 기존과 동일
2. `imputation_method='median'` → 결측 다 채워졌는지, shape 동일한지
3. `imputation_method='knn', knn_neighbors=5` → 메모리/시간 측정
4. `imputation_method='native'` → NaN 잔존 확인, LGBM에 들어가는지

---

## 4. B군 (hpo.py에 hook 추가, fold-internal 적용)

> **공통 전제**: 2장의 hook 인프라가 먼저 적용되어야 함.

### 4.1 [순위 1] OOF-safe Feature Selection

**상태**: [feature_selection.py:1](../../2_preprocessing/feature_selection.py#L1)에 5종 +
voting 구현. 모두 **whole-train fit** 가정 → 그대로 fold loop 안에서 호출하면
OK (fold마다 새로 fit). 단 시간 비용이 큼.

**신규 모듈 `final/modules/feature_engineering/voting_fs.py`** (~80줄):
```python
def make_voting_fs_callback(
    feat_cols, methods=("boruta", "lgbm_importance", "null_importance"),
    min_votes=2, sample_n=10000, n_runs_null=5, ...,
):
    """run_feature_selection을 fold-internal callback으로 wrap."""
    def callback(X_tr, y_tr, X_vl, X_val=None, X_test=None):
        from feature_selection import run_feature_selection
        selected, _ = run_feature_selection(
            pd.DataFrame(X_tr, columns=feat_cols), y_tr, feat_cols,
            methods=methods, min_votes=min_votes,
            sample_n=sample_n, ...,
        )
        sel_idx = [feat_cols.index(c) for c in selected]
        out = (X_tr[:, sel_idx], X_vl[:, sel_idx])
        if X_val  is not None: out += (X_val[:, sel_idx],)
        if X_test is not None: out += (X_test[:, sel_idx],)
        return out
    return callback
```

**호출** (노트북):
```python
from final.modules.feature_engineering.voting_fs import make_voting_fs_callback
fs_cb = make_voting_fs_callback(feat_cols_clean, min_votes=2, sample_n=8000)

res = hpo.run_hpo(..., fold_transform_fn=fs_cb)
final = hpo.refit_best(..., fold_transform_fn=fs_cb)
```

**위험 검토**:
- ⚠️ **시간 비용**: Boruta가 fold당 수십 분 → 5 fold × 100 trial = 비현실적.
  **대응**: 1차로는 `lgbm_importance`만 (fold당 ~30초), 효과 보이면 Null Importance
  추가. Boruta는 3차 funnel 최종 1조합에서만.
- ⚠️ **컬럼 수가 fold마다 다름**: voting은 fold마다 다른 컬럼을 고를 수 있음 →
  refit_best의 fold 평균 val/test 예측이 무의미.
  **대응**: `union_across_folds=True` 옵션 추가 — 5 fold 결과의 합집합을 최종 set으로
  사용. 또는 `min_votes_across_folds`로 fold 간 합의도 추가 필터. **권장**: refit
  시점에는 전체 train 1회 fit으로 union 만든 뒤 모든 fold가 동일 컬럼 사용 (HPO
  trial은 fold마다 달라도 됨).
- ⚠️ extra_feature와의 호환: 경로 B의 `(1-π_zit)`는 `extra_feature_train`으로
  `_build_X` 시점에 추가됨. callback이 X_full 받을 때 이미 hstack된 상태 →
  feat_cols에 'one_minus_pi'를 추가해 길이 일치시켜야 함. **대응**:
  callback factory에 `extra_feature_names` 인자 추가
- ⚠️ enet 호환: callback이 컬럼 수를 줄이고 → scaler가 줄어든 컬럼에 대해 fit
  → 정상 동작. 단 callback이 fold마다 다른 컬럼을 주면 scaler stats도 달라져
  fold 평균 val/test에 영향. union 방식 권장 (위 참조)
- ⚠️ dtype: callback 입출력 모두 numpy → float32 유지. `X_tr[:, sel_idx]`는 dtype 보존

**검증 스모크 테스트**:
1. `methods=("lgbm_importance",)` 단일 + `top_k=200`로 빠르게
2. fold별 selected 컬럼 수 출력 → fold 간 분산 확인
3. union vs per-fold 결과 비교 (val RMSE)
4. baseline(전체 컬럼) 대비 RMSE 개선 확인

---

### 4.2 [순위 2] WoE Binning

**상태**: 미구현.

**신규 모듈 `final/modules/feature_engineering/woe.py`** (~80줄):
- `fit_woe(X, y, top_k=50, n_bins=10)`:
  - top-K feature 선정 (LGBM importance, fit 데이터 한정)
  - 각 feature를 quantile bin (`pd.qcut`)
  - bin별 `P(Y>0|bin)` 계산 → `logit = log(p / (1-p))`
  - smoothing: bin 샘플 < 50이면 global mean으로 끌어당김
- `transform_woe(X, woe_maps)`: 각 컬럼을 woe 값으로 매핑, 신규 bin은 global mean
- `make_woe_callback(feat_cols, top_k=50, n_bins=10, append=True)`:
  - `append=True` (기본): 원본 컬럼 + `woe_X{i}` 컬럼 추가
  - `append=False`: 원본을 woe로 대체

**호출 예**:
```python
from final.modules.feature_engineering.woe import make_woe_callback
woe_cb = make_woe_callback(feat_cols_clean, top_k=50, n_bins=10, append=True)
res = hpo.run_hpo(..., fold_transform_fn=woe_cb)
```

**위험 검토**:
- ⚠️ **target leak 위험**: WoE는 정의상 Y를 봄 → fold-internal 필수. callback이
  X_tr와 함께 받은 y_tr만 사용해 woe_maps fit → X_vl/X_val/X_test에는 transform만
  적용. ✅ hook 자체가 이를 강제
- ⚠️ Y는 die-level broadcast된 unit y → 같은 unit의 4 die가 같은 y. WoE 계산 시
  bin 안에 같은 unit이 여러 die로 들어감 → bin 통계가 unit 다양성 부풀려질 가능성.
  **대응**: 1차는 die-level y 그대로, 2차에 unit-level 집계로 fit하는 옵션 추가
- ⚠️ `top_k` 선정에 LGBM 1회 fit → fold 학습 모델보다 단순 LGBM (n_estimators=200
  정도)로 빠르게. cost 허용 범위
- ⚠️ Y=0 비율 70.8% → 각 bin의 `P(Y>0)`가 0~0.5 좁은 범위 → logit 변환 후 분산
  작을 수 있음. **대응**: 출력값 정규화 옵션 (woe.subtract_mean)
- ⚠️ 신규 bin (val/test에 train에 없는 값): qcut의 경계는 train 기준이라 자동 처리.
  단 outlier가 train의 q[0.99] 밖이면 마지막 bin으로 들어감 — winsorize가 이미
  처리해서 안전
- ⚠️ append=True일 경우 차원 +50 → enet 스케일러가 추가 컬럼도 스케일. WoE 자체가
  logit 스케일이라 RobustScaler 통과 OK

**검증 스모크 테스트**:
1. `top_k=10, n_bins=5` 작게
2. fit 데이터에서 woe 값 분포 (mean ≈ 0인지)
3. ZITboost 1 trial로 RMSE
4. Stage 1 분류 (ZITboost의 π) Recall 변화 확인 — 기존 0.011 대비

---

### 4.3 [순위 4] AutoFE Pairwise (OOF-safe 재배선)

**상태**: [auto_features.py:215](modules/auto_features.py#L215) `run_auto_feature_engineering()`
이미 있음. 단 `ys_train_unit` 전체로 페어 corr 계산 → fold leak.

**신규 모듈 `final/modules/feature_engineering/autofe_oof.py`** (~50줄):
- 기존 `auto_features.py`를 fold-internal로 wrap
- callback이 X_tr 받으면 그 fold 내에서:
  1. top-K LGBM importance 선정
  2. 페어 × ops 생성 → unit-level corr (fit 데이터의 unit만으로)
  3. baseline × gain_ratio 임계값으로 채택
  4. X_tr / X_vl / X_val / X_test 모두에 채택된 페어 컬럼 추가

```python
def make_autofe_callback(feat_cols, ys_train_unit, key_arr_train,
                         k=20, ops=("mul","sub","ratio"),
                         gain_ratio=1.5, max_keep=30):
    def callback(X_tr, y_tr, X_vl, X_val=None, X_test=None):
        # fit 데이터의 unit id 복원 → ys_train_unit에서 해당 unit y만 사용
        # auto_features 내부 함수 호출
        ...
    return callback
```

**위험 검토**:
- ⚠️ unit id 복원이 까다로움: callback은 numpy X만 받고 KEY_COL을 모름.
  **대응**: callback factory에 `key_array`(전체 train의 unit id 배열)를 클로저로 전달
  + fold mask 정보를 callback이 외부에서 받아야 함. 이 부분이 hook 시그니처를 더
  복잡하게 만들 수 있음 (현재는 X_tr, y_tr, X_vl만). 두 가지 선택:
  - (a) hook 시그니처에 `tr_mask, vl_mask` 추가 — 모든 callback이 받지만 안 쓰면 그만
  - (b) AutoFE callback이 die-level y_tr만으로 작동하도록 (unit 집계 없이) — 약간
        품질 떨어지지만 단순
  - **권장**: (a). 시그니처에 mask 추가는 1회 변경, 모든 callback이 미사용 시 무영향
- ⚠️ 페어 채택 fold 간 다양성: WoE/FS와 동일 문제 (fold마다 다른 페어). 동일 대응
  (refit 시점은 train 전체 union 사용)
- ⚠️ 컬럼 수 증가 (max_keep × 4 ops 수=120 정도) → enet 스케일러 OK, 트리 OK
- ⚠️ 페어 값의 스케일이 매우 다를 수 있음 (mul, ratio는 극단적) → winsorize 후
  적용이라 어느 정도 안정. enet 사용 시 추가 클리핑 고려

**검증 스모크 테스트**:
1. `k=10, max_keep=10` 작게
2. 채택된 페어의 unit-level |r|이 baseline보다 큰지
3. unit RMSE 비교

---

### 4.4 [순위 7] GBDT Leaf Encoding (enet 전용)

**상태**: 미구현. enet 경로에서만 적용.

**신규 모듈 `final/modules/feature_engineering/leaf_encoding.py`** (~50줄):
- callback 안에서:
  1. fit 데이터에 LGBM 1회 fit (n_estimators=100, max_depth=6 정도, 작게)
  2. `pred_leaf=True`로 leaf id 매트릭스 추출 (n × n_estimators)
  3. one-hot or ordinal: one-hot은 차원 폭증 (수만 컬럼) → ordinal 권장 또는
     `n_estimators=50` + one-hot
  4. 기존 X와 hstack
  5. val/test도 동일 LGBM으로 leaf id 추출 → 인코딩 후 hstack

**호출 (enet 경로 전용)**:
```python
if MODEL_NAME == "enet":
    leaf_cb = make_leaf_encoding_callback(n_estimators=50, encode="ordinal")
    res = hpo.run_hpo(..., model_name="enet", fold_transform_fn=leaf_cb)
```

**위험 검토**:
- ⚠️ enet의 RobustScaler가 leaf id (정수)도 스케일 → 이상하지만 동작은 함.
  **대응**: leaf encoding은 별도로 0-mean/std=1로 정규화하거나 그대로 두기
- ⚠️ ordinal vs one-hot: one-hot은 컬럼 폭증 (50 × 평균 leaf 31 = 1550). enet은
  L1 정규화로 sparse selection 가능하지만 시간 증가. ordinal은 단순 정수 → 트리
  분기처럼 의미 있게 작동하진 않지만 enet엔 노이즈 추가
- ⚠️ fold마다 LGBM이 다름 → leaf id 의미가 fold 간 다름. refit_best의 fold 평균
  val/test 예측 OK (각 fold가 자기 leaf encoder로 transform한 결과 평균)
- ⚠️ 학습 시간 증가: fold당 LGBM 1회 추가 fit (~30초) → 100 trial × 5 fold = +2.5
  hour. 무시 못 할 수준 → trial 예산 줄이거나 leaf encoder는 trial 시작 시 1회 fit
  (LGBM HP를 trial이 안 바꾸므로 가능) — 단 fold-internal은 유지
- ⚠️ enet 외 모델에 적용하면 무의미하거나 해로움 → 노트북에서 `if MODEL_NAME == "enet"`
  분기 명확히

**검증 스모크 테스트**:
1. enet baseline RMSE 측정
2. leaf encoding 추가 후 RMSE 비교
3. enet 가중치(앙상블)가 의미 있게 변하는지

---

## 5. 적용 순서 (rollout 단계)

### Phase 0: hook 인프라 (1회, 30분)
1. `hpo.py`에 `fold_transform_fn=None` 인자 추가 (`run_hpo`, `refit_best`)
2. fold loop 안에 `if fn is not None` 한 줄
3. **회귀 테스트**: hook=None으로 ZITboost 1 trial 실행 → 기존 결과와 byte 일치

### Phase 1: A군 cheap wins (1일)
1. **3순위 Degenerate Binarization**: PARAMS 토글 추가, 5분 작업
2. **8순위 Imputation Ablation**: `_FIXED` 일부 해제, 30분 작업
3. **6순위 Adversarial Validation**: 신규 노트북 1개 + 모듈, 1시간

각각 ablation 1회씩 → A·B·C 경로 1 trial로 RMSE 영향 측정.

### Phase 2: A군 deviation (반나절)
4. **5순위 Die-level deviation**: 신규 모듈 + PARAMS 토글, 2시간 + ablation 1시간

### Phase 3: B군 가벼운 것 (1일)
5. **1순위 OOF-safe FS** (`lgbm_importance` 단일로 시작): 신규 모듈 callback, 3시간
   + ablation. 효과 확인 후 Null Imp 추가

### Phase 4: B군 무거운 것 (3차 funnel용, 1주)
6. **2순위 WoE**: 신규 모듈, Stage 1 회생용 — Path B (zit_plus_reg) 위주 측정
7. **4순위 AutoFE OOF-safe**: 기존 모듈 wrap
8. **7순위 Leaf Encoding**: enet 경로 전용

---

## 6. 회귀 검증 (각 Phase 끝마다)

각 Phase 완료 시 다음을 확인:

| 검증 항목 | 기준 |
|---|---|
| hook=None 또는 신규 옵션=False로 실행 시 | 기존 결과와 byte/수치 일치 |
| 신규 옵션=True로 실행 시 | shape, dtype, NaN 잔존 여부 정상 |
| `effective_params` 로깅 | 신규 키가 정확히 기록 |
| Optuna study DB | trial.user_attrs에 신규 옵션 값 저장 |
| 4_output/ 산출물 | best_params.json에 신규 옵션 명시 |

특히 **REUSE 모드** 호환:
- 기존 `best_params.json` 로드 시 신규 키가 없어도 에러 안 나야 함
- `_merge_params`가 누락 키를 DEFAULT로 채우는 동작 확인

---

## 7. 위험 요약 (전체 통합)

| ID | 위험 | 영향 항목 | 대응 |
|---|---|---|---|
| R1 | float32 dtype 깨짐 | 3, 5, 7 | 신규 컬럼 생성 후 `.astype('float32')` 명시 |
| R2 | X1086 (int32) 처리 누락 | 3, 5 | top-K 선정 시 X1086 제외 또는 별도 dtype 처리 |
| R3 | lot_mean 계산 시 val/test 행 포함 (covariate leak) | 5 | `xs_train`만 groupby, val/test는 매핑만 |
| R4 | top-K 선정에 train 전체 Y 사용 (약한 leak) | 5, 2, 4 | 1차 허용, 효과 보고 fold-internal 검토 |
| R5 | fold마다 다른 컬럼 선정 → fold 평균 val/test 무의미 | 1, 2, 4 | refit 시점은 train 전체 union 컬럼 사용 |
| R6 | callback이 X_val/X_test를 변환 안 하면 fold 평균 부정확 | 1, 2, 4, 7 | callback 시그니처에 X_val, X_test 옵션 |
| R7 | enet 스케일러가 신규 컬럼(0/1, leaf id) 스케일 | 3, 7 | 별도 처리 또는 RobustScaler에 맡김 (대부분 OK) |
| R8 | REUSE best_params.json 호환성 깨짐 | 8 | `_FIXED` 키 이름 유지 + override 메커니즘 |
| R9 | Boruta fold당 시간 비용 폭발 | 1 | lgbm_importance 단일로 시작, Boruta는 3차 한정 |
| R10 | AV가 val 정보로 EXCLUDE_COLS 결정 (낙관 편향) | 6 | AUC>0.7 보수적 임계, train+test AV 우선 |
| R11 | imputation_method 변경 시 기존 trial과 직접 비교 불가 | 8 | 별도 EXP_ID로 분리 운영 |
| R12 | WoE의 die-level y broadcast (같은 unit 4번 세기) | 2 | unit-level fit 옵션 추가 |
| R13 | leaf encoding 차원 폭증 | 7 | ordinal 권장, one-hot은 n_estimators 작게 |
| R14 | hook 시그니처 변경이 모든 callback에 파급 | 4 | `tr_mask, vl_mask` 추가 시 모든 callback 키워드 인자 통일 |
| R15 | enet은 NaN 불가 | 8 (`'native'` 옵션) | enet 자동 비활성 또는 명시 에러 |

---

## 8. 권장 우선 적용 (시간 빠듯할 때)

ROI 큰 순서로 단 2개:

1. **3순위 Degenerate Binarization** — 5분 작업, downside 거의 0
2. **1순위 OOF-safe FS (lgbm_importance만)** — feature 1,087 → 200~400으로 줄여
   후속 모든 실험 5~10× 가속. 다른 항목들의 비용도 같이 떨어짐

이 둘만으로 짧은 시간에 효과 확인 가능. 효과 검증되면 Phase 2 이후 진행.

---

## 9. 결정 필요한 사항 (사용자)

1. **deviation top-K 선정**을 fold-internal로 옮길 것인가? (안전성↑, 비용↑)
2. **WoE는 die-level y vs unit-level y** 중 어느 것으로 fit?
3. **AV 임계 AUC**: 0.6 (공격적) vs 0.7 (보수적)?
4. **leaf encoding**: ordinal vs one-hot?
5. **Imputation ablation**: 1 EXP_ID 안에서 모든 imputation 비교 vs EXP_ID 분리?
6. **hook 시그니처**: `(X_tr, y_tr, X_vl)` 단순형 vs `(X_tr, y_tr, X_vl, tr_mask,
   vl_mask, X_val, X_test)` 풀스펙?

각 질문에 답이 정해지면 Phase별 모듈 코드 초안을 바로 작성 가능.