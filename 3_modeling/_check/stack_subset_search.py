"""
스태킹 — base 부분집합 전수/우선 탐색 (pred CSV만 사용, 학습 재실행 없음)

목적: "어떤 base 조합이 가장 좋은가"를, 잔차상관 corr(pred_i - y, pred_j - y) 관점으로
      다양성을 본 뒤, 가능한 모든 (또는 우선순위 상위) 조합에 대해 메타러너를 돌려 비교한다.

풀(pool, 기본 ~10개):
  - zit 계열 3: zit_only, bag_zit, reverse (path B / joint)
  - reg_single 5: lgbm, xgb, catboost, et, enet  (각 모델은 OOF best 실험 자동 선택)
  - two-stage 곱 ts_best 1: combined/* 중 val RMSE 최저  (`INCLUDE_TS_BEST`)
  - two-stage 곱 ts_decorr 1: combined/* 중 "나머지 풀과 잔차상관 평균이 가장 낮은" 1개  (`INCLUDE_TS_DECORR`)
  combined/* 20개는 자기들끼리 잔차상관 ≈ 0.999 라 전부 넣을 가치가 없어 대표 1~2개만 쓴다.

탐색:
  - 풀 크기 N 이면 size>=2 부분집합 = 2^N - N - 1 개. N<=MAX_FULL_N(기본 13)이면 전수, 아니면
    "잔차상관 가장 낮은 페어/트리플릿을 seed로 + 다양성 우선 확장"으로 N_SAMPLE 개만.
  - 각 부분집합: ElasticNetCV(StandardScaler) 메타 (stacking.ipynb 와 동일 설정, alpha 그리드만 약간 축소)
    + SLSQP blend(Σw=1,w>=0) baseline. 선택 기준은 ElasticNetCV 내부 5-fold CV RMSE (val 안 봄 = 정직).
  - val/test RMSE 는 진단(모니터링)용으로만 기록. "2000개 중 val 최저 1개 뽑기"는 cherry-pick임을 인지.

산출: 4_output/04_stacking/_subset_search/{results.csv, best.json, residual_corr_pool.csv,
      best_oof_unit.csv, best_val_unit.csv, best_test_unit.csv} + 진행 로그(stdout).
"""
import os, sys, json, time, itertools
import numpy as np
import pandas as pd

# Windows 콘솔/로그가 cp949 라 '—' 같은 문자에서 죽는 것 방지
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# --- 프로젝트 루트 import 경로 ---
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from utils.config import KEY_COL, SEED, OUTPUT_DIR  # noqa: E402

from sklearn.linear_model import ElasticNetCV       # noqa: E402
from sklearn.preprocessing import StandardScaler     # noqa: E402
from sklearn.pipeline import Pipeline                # noqa: E402
from sklearn.model_selection import KFold            # noqa: E402
from scipy.optimize import minimize                  # noqa: E402

# ============================== 설정 =========================================
INCLUDE_TS_BEST          = True     # combined/* 중 val 최저 1개를 풀에 포함
INCLUDE_TS_DECORR        = True     # combined/* 중 나머지 풀과 잔차상관 평균 최저인 1개도 포함
INCLUDE_STACK_OUTPUTS    = False    # 04_stacking/{stack,blend} 를 base 로 재투입(보통 비권장 — 순환적)
INCLUDE_CAT_ZIT          = True     # CatBoost-BagZIT (gpu zip 풀어둔 것) 을 풀에 포함
CAT_ZIT_DIR              = os.path.join(OUTPUT_DIR, '01_zit', 'temp_cat_bag_zit', '001')
INCLUDE_OLD_BAGZIT       = True     # 이전자료의 LGBM-BagZIT 변종들 (옛 curated 0.005701 만든 것들)
OLD_BAGZIT_SOURCES       = {        # 라벨: 디렉토리 (모델링_이전자료/4_output_이전자료/_temp/*)
    'old_bagzit_fixed_ge':     os.path.join(PROJECT_ROOT, '모델링_이전자료', '4_output_이전자료', '_temp', 'bag_zit_fixed_ge'),
    'old_bagzit_hpo':          os.path.join(PROJECT_ROOT, '모델링_이전자료', '4_output_이전자료', '_temp', 'bag_zit_hpo'),
    'old_bagzit_combined_xy':  os.path.join(PROJECT_ROOT, '모델링_이전자료', '4_output_이전자료', '_temp', 'bag_zit_combined_best_xy'),
}
MAX_FULL_N               = 16       # 풀 크기 <= 이 값이면 size>=2 부분집합 전수 탐색 (14개면 16369 조합)
N_SAMPLE                 = 4000     # 전수가 불가능할 때 샘플링할 부분집합 수
MIN_SUBSET_SIZE          = 2
ENET_L1_GRID             = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
ENET_ALPHAS              = np.logspace(-6, 0, 12)   # 16k+ 부분집합 돌려야 해서 alpha 그리드 축소 (20→12)
N_FOLDS                  = 5
N_JOBS                   = 1        # ElasticNetCV 내부 — 1로 둠 (부분집합 1000+개를 순차 호출하므로 joblib 풀 스폰 오버헤드 회피; 한 fit 자체가 작음)
TOP_K_REPORT             = 25
RNG                      = np.random.default_rng(SEED)

OUT_DIR = os.path.join(OUTPUT_DIR, '04_stacking', '_subset_search')
os.makedirs(OUT_DIR, exist_ok=True)

REQUIRED = ['oof_unit.csv', 'val_unit.csv', 'test_unit.csv']


# ============================== 헬퍼 =========================================
def _oof_rmse(exp_dir):
    p = os.path.join(exp_dir, 'oof_unit.csv')
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    if 'pred' not in df.columns or 'health' not in df.columns:
        return None
    d = df.dropna(subset=['pred', 'health'])
    return float(np.sqrt(np.mean((d['pred'].values - d['health'].values) ** 2))) if len(d) else None


def _resolve_best_exp(base):
    """base 바로 아래 REQUIRED 있으면 그대로, 아니면 숫자 하위폴더 중 OOF RMSE 최저."""
    if not os.path.isdir(base):
        return None
    if all(os.path.exists(os.path.join(base, f)) for f in REQUIRED):
        return base
    cands = []
    for d in sorted(os.listdir(base)):
        sub = os.path.join(base, d)
        if d.isdigit() and os.path.isdir(sub) and all(os.path.exists(os.path.join(sub, f)) for f in REQUIRED):
            cands.append((d, sub, _oof_rmse(sub)))
    if not cands:
        return None
    scored = [c for c in cands if c[2] is not None]
    return min(scored, key=lambda c: c[2])[1] if scored else max(cands, key=lambda c: c[0])[1]


def _read_split(path, split, fname=None, col='pred'):
    df = pd.read_csv(os.path.join(path, fname or f'{split}_unit.csv'))
    return df.set_index(KEY_COL)[[col, 'health']].rename(columns={col: 'v'})


def _rmse(p, y):
    p = np.asarray(p, float); y = np.asarray(y, float)
    m = ~(np.isnan(p) | np.isnan(y))
    return float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))


def log(*a):
    print(*a, flush=True)


# ============================== 1) 풀 구성 ===================================
log('=' * 78)
log(f' stack subset search  |  SEED={SEED}  |  OUT_DIR={OUT_DIR}')
log('=' * 78)

# (label, dir, split_fname_template_or_None, col)
pool_spec = []  # 일반: (label, dir)   특수파일: (label, dir, fname_template, col)

# zit 계열
for fam in ['zit_only', 'bag_zit']:
    p = _resolve_best_exp(os.path.join(OUTPUT_DIR, '01_zit', fam))
    if p:
        pool_spec.append((fam, p))
p_rev = _resolve_best_exp(os.path.join(OUTPUT_DIR, '03_two_stage', 'reverse'))
if p_rev:
    pool_spec.append(('reverse', p_rev))

# reg_single 5
for n in ['lgbm', 'xgb', 'catboost', 'et', 'enet']:
    p = _resolve_best_exp(os.path.join(OUTPUT_DIR, '02_reg_single', n))
    if p:
        pool_spec.append((f'reg_{n}', p))

# cat_zit (CatBoost-BagZIT)
if INCLUDE_CAT_ZIT and all(os.path.exists(os.path.join(CAT_ZIT_DIR, f)) for f in REQUIRED):
    pool_spec.append(('cat_zit', CAT_ZIT_DIR))
    log(f"cat_zit = {os.path.relpath(CAT_ZIT_DIR, OUTPUT_DIR)}")
elif INCLUDE_CAT_ZIT:
    log(f"[WARN] cat_zit 산출물 없음: {CAT_ZIT_DIR} — 제외")

# 옛 LGBM-BagZIT 변종 (이전자료)
if INCLUDE_OLD_BAGZIT:
    for lbl, d in OLD_BAGZIT_SOURCES.items():
        if all(os.path.exists(os.path.join(d, f)) for f in REQUIRED):
            pool_spec.append((lbl, d))
        else:
            log(f"[WARN] {lbl} 산출물 없음: {d} — 제외")

# combined/* (two-stage 곱)
combined_root = os.path.join(OUTPUT_DIR, '03_two_stage', 'default', 'combined')
combined_dirs = {}
if os.path.isdir(combined_root):
    for combo in sorted(os.listdir(combined_root)):
        cdir = os.path.join(combined_root, combo)
        if os.path.isdir(cdir) and os.path.exists(os.path.join(cdir, 'oof_unit.csv')):
            combined_dirs[combo] = cdir

# stacking outputs (옵션)
stack_specs_extra = []
if INCLUDE_STACK_OUTPUTS:
    sroot = os.path.join(OUTPUT_DIR, '04_stacking')
    if os.path.isdir(sroot):
        for tag in sorted(os.listdir(sroot)):
            tdir = os.path.join(sroot, tag)
            if not os.path.isdir(tdir):
                continue
            if os.path.exists(os.path.join(tdir, 'oof_unit_stack.csv')):
                stack_specs_extra.append((f'stack_{tag}', tdir, '{split}_unit_stack.csv', 'pred'))
            if os.path.exists(os.path.join(tdir, 'oof_unit_blend.csv')):
                stack_specs_extra.append((f'blend_{tag}', tdir, '{split}_unit_blend.csv', 'pred'))

# ts_best: combined/* 중 val RMSE 최저
ts_best_combo = None
if INCLUDE_TS_BEST and combined_dirs:
    rows = []
    for combo, cdir in combined_dirs.items():
        d = _read_split(cdir, 'val').dropna()
        rows.append((combo, cdir, _rmse(d['v'], d['health'])))
    ts_best_combo = min(rows, key=lambda r: r[2])
    pool_spec.append((f'ts_best({ts_best_combo[0]})', ts_best_combo[1]))
    log(f"ts_best = combined/{ts_best_combo[0]}  (val RMSE {ts_best_combo[2]:.6f})")

# ---- 로드 헬퍼 ----
def _load(spec, split):
    if len(spec) == 2:
        lbl, path = spec
        return lbl, _read_split(path, split)
    lbl, path, ftmpl, col = spec
    return lbl, _read_split(path, split, fname=ftmpl.format(split=split), col=col)


def _build_matrix(specs, split):
    """specs → (P[label]=pred, y=health).  y 는 'health.max() 가 가장 작은 base'(=CLIP_Y_EXTREME 적용본)에서 가져온다.
    — combined/* 산출물은 OOF health 를 클립 안 한 raw 로 박아놔서(max 1.0) 다른 base(클립, max≈0.1)와 불일치 → 일관 기준 강제."""
    loaded = {lbl: d for lbl, d in (_load(s, split) for s in specs)}
    common = None
    for d in loaded.values():
        common = d.index if common is None else common.intersection(d.index)
    common = sorted(common)
    # y: health.max 최소인 base 기준
    ref_lbl = min(loaded, key=lambda l: loaded[l]['health'].reindex(common).max())
    y = loaded[ref_lbl]['health'].reindex(common)
    # 불일치 경고
    for lbl, d in loaded.items():
        h = d['health'].reindex(common)
        if not np.allclose(h.values, y.values):
            log(f"  [WARN] {split}/{lbl}: health 가 기준({ref_lbl})과 불일치 (max {h.max():.4f} vs {y.max():.4f}) → 기준 health 로 평가")
    P = pd.DataFrame({lbl: d['v'].reindex(common) for lbl, d in loaded.items()})
    return P, y, ref_lbl


all_specs = list(pool_spec) + list(stack_specs_extra)
P_oof_full, y_oof, _ref_oof = _build_matrix(all_specs, 'oof')
log(f"OOF y 기준 base = {_ref_oof}  (health max {y_oof.max():.4f})")

# ts_decorr: combined/* 중 "현재 풀(=non-combo)과 OOF 잔차상관 평균"이 가장 낮은 1개
if INCLUDE_TS_DECORR and combined_dirs:
    R_pool = P_oof_full.subtract(y_oof, axis=0)
    best_combo, best_mc = None, np.inf
    already = set()
    if ts_best_combo:
        already.add(ts_best_combo[0])
    for combo, cdir in combined_dirs.items():
        if combo in already:
            continue
        d = _read_split(cdir, 'oof').reindex(P_oof_full.index)
        r = d['v'] - y_oof
        mc = np.mean([np.corrcoef(r.values, R_pool[c].values)[0, 1] for c in R_pool.columns])
        if mc < best_mc:
            best_mc, best_combo = mc, combo
    if best_combo is not None:
        pool_spec.append((f'ts_decorr({best_combo})', combined_dirs[best_combo]))
        all_specs = list(pool_spec) + list(stack_specs_extra)
        P_oof_full, y_oof, _ref_oof = _build_matrix(all_specs, 'oof')   # 한 컬럼 추가됐으니 재구성
        log(f"ts_decorr = combined/{best_combo}  (나머지 풀과 OOF 잔차상관 평균 {best_mc:.4f})")

LABELS = list(P_oof_full.columns)
N = len(LABELS)
log(f"\n풀 {N}개: {LABELS}")

# val / test 도 동일 기준으로 로드 (컬럼 순서 LABELS 로 통일)
P_val_full, y_val, _ref_val = _build_matrix(all_specs, 'val')
P_test_full, y_test, _ref_test = _build_matrix(all_specs, 'test')
P_oof_full = P_oof_full[LABELS]; P_val_full = P_val_full[LABELS]; P_test_full = P_test_full[LABELS]
log(f"행 수: oof={len(P_oof_full)}  val={len(P_val_full)}  test={len(P_test_full)}  | val y기준={_ref_val} test y기준={_ref_test}")

# ============================== 2) 풀 잔차상관 ===============================
R_oof = P_oof_full.subtract(y_oof, axis=0)
C_resid = R_oof.corr()           # corr(pred_i - y, pred_j - y)
C_pred = P_oof_full.corr()
C_resid.to_csv(os.path.join(OUT_DIR, 'residual_corr_pool.csv'))
_off = lambda C: C.values[np.triu_indices_from(C.values, k=1)]
rv, pv = _off(C_resid), _off(C_pred)
log(f"\n[풀 OOF 잔차상관]  mean={rv.mean():.4f}  median={np.median(rv):.4f}  min={rv.min():.4f}  max={rv.max():.4f}")
log(f"[풀 OOF 예측상관]  mean={pv.mean():.4f}  median={np.median(pv):.4f}  min={pv.min():.4f}  max={pv.max():.4f}")
# 가장 다른(잔차상관 낮은) 페어 top
iu = np.triu_indices_from(C_resid.values, k=1)
order = np.argsort(C_resid.values[iu])
log("  잔차상관 가장 낮은 페어:")
for t in order[:min(8, len(order))]:
    log(f"    {LABELS[iu[0][t]]:24s} ~ {LABELS[iu[1][t]]:24s} = {C_resid.values[iu][t]:.4f}")

# 단일 base RMSE
log("\n[단일 base RMSE]")
for lbl in LABELS:
    log(f"  {lbl:28s}  oof={_rmse(P_oof_full[lbl], y_oof):.6f}  val={_rmse(P_val_full[lbl], y_val):.6f}  test={_rmse(P_test_full[lbl], y_test):.6f}")
best_single_val = min(_rmse(P_val_full[l], y_val) for l in LABELS)

# ============================== 3) 부분집합 목록 =============================
idx = list(range(N))
n_all = (1 << N) - 1 - N  # size>=2
if N <= MAX_FULL_N:
    subsets = []
    for r in range(MIN_SUBSET_SIZE, N + 1):
        subsets += [list(c) for c in itertools.combinations(idx, r)]
    log(f"\n전수 탐색: {len(subsets)} 개 부분집합 (size {MIN_SUBSET_SIZE}~{N})")
else:
    # 잔차상관 낮은 페어를 seed → 다양성 우선 그리디 확장으로 N_SAMPLE 개
    log(f"\n샘플 탐색: {N_SAMPLE} 개 (전수 {n_all} 너무 큼)")
    pair_order = [(iu[0][t], iu[1][t]) for t in np.argsort(C_resid.values[iu])]
    seen, subsets = set(), []
    def _div_extend(s):
        # 현재 집합과 잔차상관 평균이 가장 낮은 base 들 순서로 확장 후보 반환
        rest = [j for j in idx if j not in s]
        scored = sorted(rest, key=lambda j: np.mean([C_resid.values[j, k] for k in s]))
        return scored
    for (a, b) in pair_order:
        s = [a, b]
        while True:
            key = tuple(sorted(s))
            if key not in seen:
                seen.add(key); subsets.append(list(s))
                if len(subsets) >= N_SAMPLE:
                    break
            if len(s) == N:
                break
            ext = _div_extend(s)
            # 약간의 무작위성
            j = ext[0] if RNG.random() < 0.7 else RNG.choice(ext)
            s = sorted(set(s) | {int(j)})
        if len(subsets) >= N_SAMPLE:
            break
    # 모자라면 랜덤으로 채움
    while len(subsets) < min(N_SAMPLE, n_all):
        r = RNG.integers(MIN_SUBSET_SIZE, N + 1)
        s = sorted(RNG.choice(idx, size=r, replace=False).tolist())
        if tuple(s) not in seen:
            seen.add(tuple(s)); subsets.append(s)

# 다양성 점수(=1 - 부분집합 내부 잔차상관 평균)가 높은 순으로 정렬 → "예측상관 많이 다른 조합 우선"
def _subset_diversity(s):
    if len(s) < 2:
        return 0.0
    vals = [C_resid.values[i, j] for i, j in itertools.combinations(s, 2)]
    return float(1.0 - np.mean(vals))
subsets.sort(key=_subset_diversity, reverse=True)

# ============================== 4) 메타러너 루프 =============================
Po = P_oof_full.values; Pv = P_val_full.values; Pt = P_test_full.values
yo = y_oof.values; yv = y_val.values; yt = y_test.values
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

def _enet_eval(cols):
    Xo, Xv, Xt = Po[:, cols], Pv[:, cols], Pt[:, cols]
    pipe = Pipeline([('sc', StandardScaler()),
                     ('en', ElasticNetCV(l1_ratio=ENET_L1_GRID, alphas=ENET_ALPHAS,
                                         cv=kf, random_state=SEED, n_jobs=N_JOBS,
                                         max_iter=20000, positive=False))])
    pipe.fit(Xo, yo)
    en = pipe.named_steps['en']
    cv_rmse = float(np.sqrt(en.mse_path_.mean(axis=-1).min()))
    oof_in = _rmse(np.clip(pipe.predict(Xo), 0, None), yo)   # in-sample (낙관적, 진단만)
    val_r = _rmse(np.clip(pipe.predict(Xv), 0, None), yv)
    test_r = _rmse(np.clip(pipe.predict(Xt), 0, None), yt)
    n_active = int(np.sum(np.abs(en.coef_) > 1e-9))
    n_neg = int(np.sum(en.coef_ < -1e-9))
    return dict(cv_rmse=cv_rmse, oof_insample=oof_in, val=val_r, test=test_r,
                l1_ratio=float(en.l1_ratio_), alpha=float(en.alpha_),
                n_active=n_active, n_neg=n_neg,
                pipe=pipe)

def _blend_eval(cols):
    Xo, Xv, Xt = Po[:, cols], Pv[:, cols], Pt[:, cols]
    k = len(cols)
    res = minimize(lambda w: _rmse(Xo @ w, yo), np.full(k, 1.0 / k), method='SLSQP',
                   bounds=[(0.0, 1.0)] * k, constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}],
                   options={'ftol': 1e-10, 'maxiter': 500})
    w = res.x
    return dict(blend_oof=_rmse(Xo @ w, yo), blend_val=_rmse(Xv @ w, yv), blend_test=_rmse(Xt @ w, yt))

log(f"\n메타 평가 시작 — {len(subsets)} 부분집합 (ElasticNetCV + SLSQP blend)\n")
t0 = time.time()
rows = []
for i, s in enumerate(subsets):
    e = _enet_eval(s)
    b = _blend_eval(s)
    rows.append(dict(
        n_bases=len(s), bases='+'.join(LABELS[j] for j in s),
        cv_rmse=e['cv_rmse'], oof_insample=e['oof_insample'], val=e['val'], test=e['test'],
        blend_val=b['blend_val'], blend_test=b['blend_test'],
        diversity=_subset_diversity(s),
        mean_resid_corr=float(np.mean([C_resid.values[a, b2] for a, b2 in itertools.combinations(s, 2)])) if len(s) > 1 else np.nan,
        mean_pred_corr=float(np.mean([C_pred.values[a, b2] for a, b2 in itertools.combinations(s, 2)])) if len(s) > 1 else np.nan,
        l1_ratio=e['l1_ratio'], alpha=e['alpha'], n_active=e['n_active'], n_neg=e['n_neg'],
    ))
    if (i + 1) % 100 == 0 or (i + 1) == len(subsets):
        dt = time.time() - t0
        eta = dt / (i + 1) * (len(subsets) - i - 1)
        log(f"  {i+1:5d}/{len(subsets)}  ({dt:6.1f}s, ETA {eta:6.1f}s)  best cv_rmse so far={min(r['cv_rmse'] for r in rows):.6f}")

df = pd.DataFrame(rows)
df_by_cv = df.sort_values('cv_rmse').reset_index(drop=True)
df.sort_values('cv_rmse').to_csv(os.path.join(OUT_DIR, 'results.csv'), index=False)

# ============================== 5) 리포트 ===================================
log("\n" + "=" * 78)
log(f" 결과 — {len(df)} 부분집합  ·  best_single val={best_single_val:.6f}  ·  기존 stacking val≈0.005701")
log("=" * 78)
log(f"\n[CV RMSE 기준 top {TOP_K_REPORT}] (정직한 선택 기준 — val 안 봄)")
log(df_by_cv.head(TOP_K_REPORT)[['n_bases', 'cv_rmse', 'oof_insample', 'val', 'test', 'blend_val',
                                  'mean_resid_corr', 'l1_ratio', 'alpha', 'n_active', 'n_neg', 'bases']]
    .to_string(index=False, float_format='%.6f'))

df_by_val = df.sort_values('val').reset_index(drop=True)
log(f"\n[val RMSE 기준 top {TOP_K_REPORT}]  ⚠ {len(df)}개 중 val 최저 뽑기 = cherry-pick (peek-bias). 참고용.")
log(df_by_val.head(TOP_K_REPORT)[['n_bases', 'val', 'test', 'cv_rmse', 'blend_val',
                                   'mean_resid_corr', 'l1_ratio', 'n_active', 'n_neg', 'bases']]
    .to_string(index=False, float_format='%.6f'))

# 다양성(잔차상관 낮음) 상위 조합들의 성능
df_by_div = df.sort_values('diversity', ascending=False).reset_index(drop=True)
log(f"\n[다양성(낮은 잔차상관) 기준 top 12 — '예측상관 많이 다른 조합'들이 실제론 어떤지]")
log(df_by_div.head(12)[['n_bases', 'mean_resid_corr', 'mean_pred_corr', 'cv_rmse', 'val', 'test', 'bases']]
    .to_string(index=False, float_format='%.6f'))

# best (CV 기준) 재학습 → 예측 저장 + segment
best_s_labels = df_by_cv.iloc[0]['bases'].split('+')
best_cols = [LABELS.index(l) for l in best_s_labels]
be = _enet_eval(best_cols)
pipe = be['pipe']
for split_name, P_, y_, ids in [('oof', Po, yo, P_oof_full.index), ('val', Pv, yv, P_val_full.index), ('test', Pt, yt, P_test_full.index)]:
    pred = np.clip(pipe.predict(P_[:, best_cols]), 0, None)
    pd.DataFrame({KEY_COL: ids, 'pred': pred, 'health': y_}).to_csv(os.path.join(OUT_DIR, f'best_{split_name}_unit.csv'), index=False)

def _seg(p, y):
    p = np.asarray(p, float); y = np.asarray(y, float)
    return {'y0': _rmse(p[y == 0], y[y == 0]), 'ypos': _rmse(p[y > 0], y[y > 0]),
            'n_y0': int((y == 0).sum()), 'n_ypos': int((y > 0).sum())}
seg = {sp: _seg(np.clip(pipe.predict(P_[:, best_cols]), 0, None), y_)
       for sp, P_, y_ in [('oof', Po, yo), ('val', Pv, yv), ('test', Pt, yt)]}

best_json = {
    'pool_labels': LABELS,
    'pool_paths': {lbl: (spec[1]) for lbl, spec in zip(LABELS, all_specs)},
    'n_subsets_evaluated': int(len(df)),
    'enumeration': 'full' if N <= MAX_FULL_N else f'sampled({N_SAMPLE})',
    'pool_resid_corr': {'mean': float(rv.mean()), 'median': float(np.median(rv)), 'min': float(rv.min()), 'max': float(rv.max())},
    'best_single_val': float(best_single_val),
    'baseline_stacking_11base_val': 0.005701,
    'best_by_cv': {
        'bases': best_s_labels, 'cv_rmse': float(be['cv_rmse']),
        'val': float(be['val']), 'test': float(be['test']),
        'l1_ratio': float(be['l1_ratio']), 'alpha': float(be['alpha']),
        'n_active': int(be['n_active']), 'n_neg': int(be['n_neg']),
        'segment_rmse': seg,
    },
    'best_by_val_DIAGNOSTIC_ONLY': {
        'bases': df_by_val.iloc[0]['bases'].split('+'),
        'val': float(df_by_val.iloc[0]['val']), 'test': float(df_by_val.iloc[0]['test']),
        'cv_rmse': float(df_by_val.iloc[0]['cv_rmse']),
    },
    'SEED': int(SEED),
}
with open(os.path.join(OUT_DIR, 'best.json'), 'w', encoding='utf-8') as f:
    json.dump(best_json, f, indent=2, ensure_ascii=False, default=str)

log("\n" + "=" * 78)
log(" BEST (CV 기준)")
log("=" * 78)
log(f"  bases   : {' + '.join(best_s_labels)}  ({len(best_s_labels)}개)")
log(f"  cv_rmse : {be['cv_rmse']:.6f}   (alpha={be['alpha']:.2e}, l1={be['l1_ratio']:.2f}, active={be['n_active']}/{len(best_s_labels)}, neg={be['n_neg']})")
log(f"  val     : {be['val']:.6f}   test: {be['test']:.6f}")
log(f"  segment : val Y=0 {seg['val']['y0']:.6f} (n={seg['val']['n_y0']}) | val Y>0 {seg['val']['ypos']:.6f} (n={seg['val']['n_ypos']})")
log(f"  vs best_single_val={best_single_val:.6f}  Δ={be['val']-best_single_val:+.6f}   |   vs stacking_11base 0.005701  Δ={be['val']-0.005701:+.6f}")
log(f"\n저장: {OUT_DIR}/  (results.csv, best.json, residual_corr_pool.csv, best_{{oof,val,test}}_unit.csv)")
log(f"총 소요: {time.time()-t0:.1f}s")
