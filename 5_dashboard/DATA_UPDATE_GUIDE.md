# 대시보드·보고서 새 모델 데이터 업데이트 가이드

> 새 모델 run 산출물로 **dashboard_fin** 의 대시보드 + 보고서 데이터를 교체하는 실행 체크리스트.
> 다른 Claude 세션이 이 문서만 보고 그대로 실행할 수 있도록 경로·명령·수정 위치(file:line)를 명시한다.

---

## 0. 전제 / 입력

- **대상 레포**: `C:/Users/Dell3571/Desktop/dashboard_fin` (오직 fin만 수정. v3 등은 건드리지 않음)
- **입력 = 새 모델 run 디렉터리** (예시: `C:/Users/Dell3571/Downloads/run_0605_194155_3-20260609T010651Z-3-001/run_0605_194155_3`)
  ```
  run_xxx/
  ├── best/
  │   ├── oof_die.csv  oof_unit.csv      # train (die: pred_taupi/pred_raw/mu/pi, unit: pred)
  │   ├── val_die.csv  val_unit.csv      # val
  │   ├── test_die.csv test_unit.csv     # test
  │   ├── fold_models.pkl                # SHAP/FI 재계산용 (lgb_mu_ 포함)
  │   ├── best_params.json
  │   ├── summary_record.json            # ★ val_rmse 등 평가지표
  │   └── calibration_candidates.csv
  └── seed_sweep_summary.csv
  ```
  - **die pred = `pred_taupi`** (unit = die 4개의 **합** 구조), **unit pred = `pred`**
  - **val RMSE = `best/summary_record.json` → `"val_rmse"`** (6자리 반올림해 사용. 예: 0.005699107… → `0.005699`)

- **외부 의존 (반드시 존재해야 함)**:
  - 원본 피처: `C:/Users/Dell3571/Desktop/ASAC_SKhynix/0_data/compet_xs_data.csv`  (SHAP·feature_dist 재계산에 필요)
  - 모델 모듈: `C:/Users/Dell3571/Desktop/ASAC_SKhynix/3_modeling` (ZITboost/BagZITEQL — `fold_models.pkl` unpickle에 필요)

- **출력 흐름**: `data/processed/*` 재생성 → `Dashboard/public/` 복사 → 하드코딩(val RMSE) 갱신 → 빌드/검증 → 푸시

> ⚠️ **경로 하드코딩 주의**: 아래 스크립트 다수가 `dashboard_v3` 또는 레거시 경로로 하드코딩되어 있다. fin 적용 시 각 스크립트의 경로를 **dashboard_fin + 새 run** 으로 수정한 뒤 실행해야 한다 (각 STEP에 수정 위치 명시).

---

## 1. 모델-의존 파일 vs 무관 파일 (먼저 이해)

| 구분 | 파일 | 새 모델 적용 시 |
|---|---|---|
| **모델-의존 (재생성 필수)** | dashboard_units.csv, wafer_map.csv, wafer_map_lots/, feature_importance.csv, shap_bar/beeswarm.csv, shap_unit.json, location_stats.csv, wafer_scale.json, dashboard_lot_patterns.csv, dashboard_lot_pattern_maps.json, feature_dist.csv, dashboard_lot_summary.csv, outlier_wafer.json, outlier_wafers.json, metrics.csv(val RMSE) | **재생성** |
| **원본 X-의존 (모델 무관 → 그대로)** | wafer_feat_norm.csv, feature_violin.json/csv, trend_data.csv(날짜축 고정·마지막값 자동), grade_trend.csv | **건드리지 않음** |
| **미사용/레거시 (무시)** | feature_lot_scatter.json, dashboard_dates.csv, feature_trend.csv, dist_data.csv, delta_period.csv, oof_meta.csv | **무시** |

> **트렌드 마지막 주차(8월 2주차) 값**: `trend_data.csv` 는 날짜축·과거 주 형태만 제공하고, **마지막 주차 ppm = `dashboard_units.csv` 의 reg_pred 평균에서 자동 계산**된다 (`agent/tools.py get_weekly_yield_trend`, `Overview2.jsx`). → trend_data.csv 는 수정 불필요, dashboard_units.csv만 새로 만들면 마지막 점이 자동 갱신된다.

---

## ⭐ 빠른 실행 (권장) — 원터치 오케스트레이터

경로 수정·개별 스크립트 실행 없이 **명령 하나**로 §2~§4(STEP1~7 + public 복사 + val RMSE 갱신)가 전부 자동 실행된다. 경로는 dashboard_fin 으로 고정되어 v3 오염 위험이 없다.

```powershell
cd C:/Users/Dell3571/Desktop/dashboard_fin/data_pipeline

# 0) 먼저 입력/의존성만 검증 (쓰기 없음) — 권장
python update_dashboard.py "<run_dir>" --check

# 1) 전체 실행 (processed 재생성 → public 복사 → metrics/val RMSE 갱신)
python update_dashboard.py "<run_dir>"
```
- `<run_dir>` = `best/` 를 포함한 새 모델 run 폴더
- 자동 수행: ① 모델데이터(units/wafer_map/FI/location/scale/lots) ② **SHAP 전체 die**(bar/beeswarm/**shap_unit=유닛별 SHAP**) ③ Lot 패턴 ④ dashboard_lot_summary ⑤ metrics.csv val RMSE ⑥ feature_dist·outlier(서브 생성기) ⑦ public 복사 ⑧ 하드코딩 val RMSE 기본값 갱신
- 의존성 누락 시 명확한 에러로 **중단**(부분 갱신 방지). `XS_PATH`/`MODEL_MODULES` 환경변수로 경로 override 가능
- 실행 후: `cd Dashboard && npm run build` 로 검증 → 사용자 승인 시 push

> **이 방법이 기본.** 아래 §2 STEP별 수동 절차는 오케스트레이터가 막혔을 때의 참고/디버그용이다.

---

## 2. STEP별 재생성 (수동 — 참고/디버그용)

작업 디렉터리: `C:/Users/Dell3571/Desktop/dashboard_fin/data_pipeline`
아래에서 `RUN_BEST` = `<새 run>/best` 절대경로.

### STEP 1 — 핵심 모델 데이터 (`update_model_v2.py`)
생성: **dashboard_units, wafer_map, feature_importance, shap_bar/beeswarm, shap_unit.json, location_stats, wafer_scale, wafer_map_lots/**

수정 위치 (`data_pipeline/update_model_v2.py`):
- `line 24` `SRC_BEST` → 새 run 의 `best` 경로
- `line 25` `ROOT` → `Path('C:/Users/Dell3571/Desktop/dashboard_fin')`  ← **v3로 되어 있음, 반드시 변경**
- `line 28` `XS_PATH` → `compet_xs_data.csv` 경로 확인
- `line 31-32` 모델 모듈 `sys.path` 확인

```powershell
cd C:/Users/Dell3571/Desktop/dashboard_fin/data_pipeline
python update_model_v2.py
```
> 주의: 이 스크립트는 `data/processed/wafer_map.csv` 의 메타(run_id,wafer_no,split,health,date,die_x,die_y)를 **기존 파일에서 재사용**한다. 따라서 fin 의 기존 processed 가 있어야 한다(있음). 신규 unit 집합이 동일(43,643)하다는 전제.

### STEP 2 — SHAP 재계산 (wafer_no 반영판) (`regen_shap_all.py`)
STEP 1 의 shap 을 덮어쓴다 (die 단위 wafer_no 파싱 포함, 권장).
수정 위치 (`regen_shap_all.py`):
- `line 15` `ROOT` → `dashboard_fin`  ← v3 하드코딩
- `line 16-18` `SRC_BEST`/`PUBLIC`/`XS_PATH` 확인
- `line 20-21` 모델 모듈 경로 확인
```powershell
python regen_shap_all.py
```
> STEP 1 의 SHAP 으로 충분하면 생략 가능. 단 `shap_unit.json` 의 wafer_no 기반 로직을 쓰면 이 스크립트 사용.

### STEP 3 — Lot 패턴 (`regen_pattern.py`)
생성: **dashboard_lot_patterns.csv, dashboard_lot_pattern_maps.json** (계층탐색 패턴분류 / Edge Ring 등)
수정 위치 (`regen_pattern.py`):
- `line 14` `P = Path('.../dashboard_v3/data/processed')` → `dashboard_fin`  ← **변경**
```powershell
python regen_pattern.py
```

### STEP 4 — Feature 분포 (`generate_feature_dist.py`)
생성: **feature_dist.csv** (R3 피처 정상/불량 분포 · 보고서/ProcessFactor). grade 기반이라 모델-의존.
- `VERSION` 인자로 동작: `dashboard_{VERSION}` 경로 사용 → fin 은 `fin` 전달
- `line 17` `XS` 경로 확인
```powershell
python generate_feature_dist.py fin
```

### STEP 5 — 이상치 웨이퍼 (`generate_outlier_wafer.py`)
생성: **outlier_wafer.json**(단일) + **outlier_wafers.json**(리스트, grade4 매우위험 웨이퍼만, 채우기 없음)
```powershell
python generate_outlier_wafer.py fin
```

### STEP 6 — Lot 요약 (`dashboard_lot_summary.csv`) — ★ 생성기 없음, 아래 스니펫으로 재생성
계층탐색 lot/wafer 트리의 핵심 데이터(웨이퍼별 total_dies/risk_dies/avg_ppm). wafer_map + wafer_scale 임계값에서 파생.
```powershell
cd C:/Users/Dell3571/Desktop/dashboard_fin
python -c "
import pandas as pd, json
P='data/processed'
wm=pd.read_csv(f'{P}/wafer_map.csv'); wm['pred']=pd.to_numeric(wm['pred'],errors='coerce')
wm=wm.dropna(subset=['pred'])
th=json.load(open(f'{P}/wafer_scale.json'))['threshold']
g=wm.groupby(['run_id','wafer_no']).agg(
    total_dies=('pred','size'),
    risk_dies=('pred',lambda x:(x>th).sum()),
    avg_ppm=('pred',lambda x:round(x.mean()*1e6,1))).reset_index()
g.to_csv(f'{P}/dashboard_lot_summary.csv',index=False)
print('dashboard_lot_summary.csv', len(g), 'wafers | threshold', round(th,8))
"
```
> 컬럼 순서/이름은 기존과 동일해야 한다: `run_id,wafer_no,total_dies,risk_dies,avg_ppm`.

### STEP 7 — metrics.csv val RMSE 갱신 — ★ 생성기 없음
ModelPerformanceV2 페이지가 `model=='stacking' & stage=='reg' & split=='val' & metric=='rmse'` 행을 읽는다.
새 run 의 `summary_record.json["val_rmse"]` 값으로 해당 행을 갱신(없으면 추가)한다.
```powershell
cd C:/Users/Dell3571/Desktop/dashboard_fin
python -c "
import pandas as pd, json
run_best=r'<새 run>/best'          # ← 실제 경로로 교체
v=round(float(json.load(open(run_best+'/summary_record.json'))['val_rmse']),6)
m=pd.read_csv('data/processed/metrics.csv')
mask=(m['stage']=='reg')&(m['model']=='stacking')&(m['split']=='val')&(m['metric']=='rmse')
if mask.any(): m.loc[mask,'value']=v
else: m=pd.concat([m,pd.DataFrame([{'stage':'reg','model':'stacking','split':'val','metric':'rmse','value':v}])],ignore_index=True)
m.to_csv('data/processed/metrics.csv',index=False)
print('metrics val rmse =', v)
"
```
> 보고서의 RMSE 는 metrics 가 없어도 `get_val_rmse()` 가 dashboard_units(val) 로 자동 계산하지만, **ModelPerformanceV2 화면**은 metrics.csv 를 직접 읽으므로 갱신 권장.

---

## 3. processed → public 복사

프론트엔드가 fetch 하는 파일만 `Dashboard/public/` 로 복사. (모델-의존 파일 + lots 디렉터리)
```powershell
cd C:/Users/Dell3571/Desktop/dashboard_fin
$files=@('dashboard_units.csv','wafer_map.csv','wafer_scale.json','location_stats.csv',
 'feature_importance.csv','shap_bar.csv','shap_beeswarm.csv','shap_unit.json',
 'feature_dist.csv','dashboard_lot_patterns.csv','dashboard_lot_pattern_maps.json',
 'dashboard_lot_summary.csv','metrics.csv','outlier_wafer.json','outlier_wafers.json')
foreach($f in $files){ Copy-Item "data/processed/$f" "Dashboard/public/$f" -Force }
# wafer_map_lots 디렉터리 전체 교체
Remove-Item "Dashboard/public/wafer_map_lots" -Recurse -Force -ErrorAction SilentlyContinue
Copy-Item "data/processed/wafer_map_lots" "Dashboard/public/wafer_map_lots" -Recurse -Force
```
> ※ chroma.sqlite3, `*_backup_*`, `*.full-backup.*`, `pattern_out/` 은 커밋/복사 대상 아님.

---

## 4. 하드코딩 값 갱신

### 4-1. Val RMSE  (`0.005699` → 새 `summary_record.json["val_rmse"]` 6자리)
- `agent/report.py:973`  `val_rmse = meta.get("val_rmse", "0.005699")`
- `agent/report.py:1480` 동일 (PPT/HTML 양쪽 기본값)
- `agent/agent.py:288`   fallback `_val_rmse = "0.005698"` → 새 값
- `agent/tools.py` `get_val_rmse()` fallback 리턴값(약 line 1185) `"0.005699"` → 새 값
> 모두 metrics.csv 가 있으면 그 값이 우선하지만, 기본값/표시 일관성을 위해 함께 교체.

### 4-2. 날짜  — **변경하지 않음 (현재 하드코딩 유지)**
오늘=2026-06-11, 트렌드 6월 2주차~8월 2주차 등 연출용 날짜는 **그대로 둔다**. 마지막 주차(8월 2주차)의 값만 새 모델에서 자동 반영됨(§1 참고).
（참고용 위치: `report.py:975/1484` today, `report.py:165/2241` 트렌드 시작 2026-06-08, `tools.py:75-76,108,119,128-140,983`, `agent.py:155`, `TopBar.jsx:4`, `ModelPerformanceV2.jsx:7`, `ProcessFactor.jsx:8`, `Overview2.jsx:379`, `WaferMap.jsx:11-24,214`）

### 4-3. 모델명  — **표시 안 함 (작업 없음)**
보고서 footer 등에서 모델명("Stacking Ensemble")은 이미 제거되어 화면에 노출되지 않는다. 새 데이터 적용 시 별도 작업 불필요.

### 4-4. (참고) 그 외 연출 상수 — 데이터 분포가 크게 달라지면만 점검
- 트렌드 과거 기준 `TARGET_PAST_PPM=2100`: `tools.py:715,717,923`, `Overview2.jsx:416`
- 트렌드 y축 `ppmMin=1610 / ppmMax=2500`: `Overview2.jsx:453-454` (마지막 주 ppm 이 이 범위를 벗어나면 조정)
- 표시 lot 범위 `1~28`: `DrilldownV2.jsx:832`, `Overview2.jsx:578`, `WaferMap.jsx:17` (원본 0_data lot 수와 동일하면 유지)
- 보고서 더미 생산량 `[1800,...]`: `report.py:153` (실데이터 있으면 미사용 경로)

---

## 5. 검증

```powershell
# 1) 프론트 빌드
cd C:/Users/Dell3571/Desktop/dashboard_fin/Dashboard
npm run build        # 에러 없이 "built in" 확인

# 2) 보고서 생성 + 핵심 값 점검 (백엔드)
cd C:/Users/Dell3571/Desktop/dashboard_fin/agent
python -c "
import agent, report
d=agent._build_report_data({})
import report as R
print('top unit :', R.get_top_unit_data()['serial'] if hasattr(R,'get_top_unit_data') else 'n/a')
print('val rmse :', d.get('meta',{}).get('val_rmse'))
html=report.build_html(d); print('build_html OK', len(html))
pptx=report.build_pptx(d); open(r'C:/temp/_check.pptx','wb').write(pptx); print('pptx', len(pptx))
"
```
점검 포인트:
- 대시보드 이상치 웨이퍼맵 최고 유닛 = 보고서 R1 대표 유닛 (serial 일치)
- 보고서 RMSE = ModelPerformanceV2 화면 RMSE = `metrics.csv` 값 = `summary_record.json["val_rmse"]`
- 트렌드 마지막 주차 ppm(대시보드) = 보고서 트렌드 마지막 점

---

## 6. 푸시 (사용자 승인 후에만)

fin 은 배포본이며 `github.com/seong-eun822/sk-dashboard` `main` 으로 푸시 시 실제 대시보드에 반영된다.
```powershell
cd C:/Users/Dell3571/Desktop/dashboard_fin
git add Dashboard/public agent  data_pipeline   # 데이터/코드 변경분 (chroma.sqlite3 제외 확인)
git commit -m "Update dashboard+report data to <run 이름> (val rmse <값>)"
git push origin main
```
> 푸시는 **사용자가 명시적으로 요청할 때만** 수행한다.

---

## 부록 A. 스크립트 ↔ 출력 파일 매핑

| 스크립트 | 출력 | fin 경로 수정 필요 |
|---|---|---|
| `update_model_v2.py` | dashboard_units, wafer_map, feature_importance, shap_bar/beeswarm, shap_unit, location_stats, wafer_scale, wafer_map_lots/ | ROOT(25), SRC_BEST(24), XS_PATH(28) |
| `regen_shap_all.py` | shap_bar/beeswarm/shap_unit (wafer_no 반영) | ROOT(15), SRC_BEST(16), XS_PATH(18) |
| `regen_pattern.py` | dashboard_lot_patterns, dashboard_lot_pattern_maps | P(14) |
| `generate_feature_dist.py fin` | feature_dist | VERSION 인자, XS(17) |
| `generate_outlier_wafer.py fin` | outlier_wafer, outlier_wafers | VERSION 인자 |
| (스니펫) | dashboard_lot_summary | 본문 STEP 6 |
| (스니펫) | metrics.csv (val rmse 행) | 본문 STEP 7 |

## 부록 B. 절대 건드리지 않는 파일 (모델 무관 / 미사용)
- 모델 무관(그대로): `wafer_feat_norm.csv`, `feature_violin.json/csv`, `trend_data.csv`, `grade_trend.csv`
- 미사용/레거시: `feature_lot_scatter.json`, `dashboard_dates.csv`, `feature_trend.csv`, `dist_data.csv`, `delta_period.csv`, `oof_meta.csv`
