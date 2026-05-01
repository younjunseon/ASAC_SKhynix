# Wafer Test 기반 RCC 예측 대시보드 — 디자인

> SK Hynix Wafer Test 기반 Field Health Data(RCC) 예측 결과를 모니터링하고, **통계적 근거에 기반한 의사결정**을 지원하는 인터랙티브 대시보드.

---

## 1. 한 줄 요약

**PI(Process Integration) 직무가 위험 unit을 사전 식별하고, "어떤 변수가 통계적으로 문제인지"까지 근거를 갖춘 의사결정을 내릴 수 있도록 지원하는 대시보드.** 단순 차트 시각화를 넘어 변수별 z-score / lot 간 검정 / drift 검정 같은 통계 결과를 함께 노출한다. 보고서 자동 생성과 자연어 질의는 AI Agent / RAG 챗봇으로 보조한다.

---

## 2. 사용자 (페르소나)

### 2.1 1차 사용자 — PI 엔지니어
- **사용 시나리오**: 일/주 단위로 wafer test 결과를 보고 "오늘 어떤 lot/wafer를 잡아야 하나" 결정
- **주요 행동**:
  1. Overview에서 위험 unit 비율 추이 + 알람 + **trend 검정 결과** 확인
  2. 위험 lot/wafer 클릭 → Drill-down에서 die 위치 + **SHAP 기여 변수 + 변수별 통계 근거** 확인
  3. 필요 시 "보고서 생성" → AI Agent가 PDF/요약 출력
- **의사결정 산출물**: 검사 의뢰 / 격리 / 공정팀 알림 / 보고서
- **요구하는 근거 수준**: "pred 0.05다" 가 아니라 "X392가 lot 평균 대비 +2.3σ, p<0.01로 유의"

### 2.2 2차 사용자 — 모델러 / 데이터 분석가
- Model Diagnostics에서 모델 신뢰성 (RMSE, calibration, 잔차, drift) 확인
- Data 페이지에서 raw 다운로드해 추가 분석

---

## 3. 핵심 KPI & 임계값 정책

### 3.1 메인 KPI (3개)

| KPI | 정의 | 표시 위치 |
|---|---|---|
| **위험 unit 비율** (메인) | `(예측 health > τ인 unit 수) / 전체 unit 수` | Overview KPI 카드 1번 |
| 평균 예측 health | `mean(pred)` | Overview KPI 카드 2번 |
| 처리 unit 수 | `count(unit)` | Overview KPI 카드 3번 |

> "평균 health"는 zero가 70.8%라 항상 평탄해 변화 감지가 어려움. **PI에게는 비율 기반 KPI가 actionable** → 비율을 메인으로.

각 KPI 카드 우측 하단에 **변화 chip**: 전 기간 대비 ▲/▼ %p (chip 색상: 빨강=악화, 초록=개선).

### 3.2 임계값 정책 (단계적)

**현재 (1단계)**
- 단일 임계값 — split별 `pred` 상위 5% 분위수
- 구현: [prepare_data.py](prepare_data.py) `RISK_TOP_RATIO=0.05`
- `is_risk` boolean 컬럼

**향후 (2단계, 모델 결과 분포 확정 후 적용)**
- warning / critical 2단계 — `risk_level` enum (`none`/`warning`/`critical`)
  - warning: `Y>0` 분포의 75-percentile 부근
  - critical: `Y>0` 분포의 95-percentile 부근
- 영향: `prepare_data.py`, `api/main.py`, `lib/colors.ts`, `WaferMap.tsx`, 페이지 3개
- **현 단계는 단일 유지** — 임계값을 환경변수/설정 파일로 빼서 코드 수정 없이 조정

### 3.3 Alert 트리거
- "위험 unit 비율 > X%" 인 lot 자동 강조 (X 기본값 5%)
- Overview 상단 alert 영역 + 우상단 알람 벨에서 노출
- **alert에 통계 근거 부착**: "lot L0421 위험률 12.4% — baseline(5%) 대비 χ² 검정 p<0.01" 형태

---

## 4. 정보 구조 (4페이지)

설계 원칙: **Shneiderman의 "Overview → Filter → Detail"** (자료 카탈로그: [docs/dashboard_resources_index.md](docs/dashboard_resources_index.md))

```
사이드바 (보라 그라디언트)
  ├─ Overview      ← KPI + Alert + 시계열(+ trend test) + Top + 분포
  ├─ Drill-down   ← Lot/Wafer/Unit 통합 분석 + SHAP + 변수 통계
  ├─ Model         ← 성능 진단 + drift test + calibration
  └─ Data          ← raw 테이블 + CSV
```

| 페이지 | 역할 | 사용자 행동 | 라우팅 |
|---|---|---|---|
| **Overview** | 전체 모니터링 + Alert | "오늘 무슨 lot 잡지?" | `/` |
| **Drill-down** | Lot → Wafer → Die → Unit + 변수 근거 | "이 lot/unit 왜 위험? 어떤 변수가 문제?" | `/drilldown` |
| **Model** | 모델 성능 + drift + 신뢰도 | "예측 믿어도 돼? 분포가 학습 때와 같아?" | `/model` |
| **Data** | raw 테이블 + 검색·필터·CSV | "데이터 내려받자" | `/data` |

> 구조 변경 사항: 기존 `Wafers.tsx` + `Lots.tsx` (orphan 파일) → **`Drilldown.tsx` 한 페이지로 통합**, `Model.tsx` 신규 추가.

---

## 5. 통계 기반 의사결정 지원 (핵심)

대시보드의 **차별화 가치**. 단순 차트가 아니라 "왜 위험한가"의 통계적 근거를 같은 화면에 노출한다.

### 5.1 통계 분석 카탈로그

| 분석 | 노출 위치 | 사용 통계 | 의사결정 효과 |
|---|---|---|---|
| **변수별 z-score** | Drill-down Unit Diagnosis | `(x - μ_global) / σ_global` | "X392가 평균 대비 +2.3σ — 이상값" |
| **lot 내 / lot 간 비교** | Drill-down Lot Diagnosis | unit이 같은 lot의 다른 unit 대비 outlier 정도 | "이 unit은 같은 lot에서도 튀는 값" |
| **lot vs baseline 검정** | Overview Alert + Drill-down | Welch's t-test / Mann-Whitney U | "lot L0421 위험률은 우연이 아닌 유의한 차이 (p<0.01)" |
| **Trend test** | Overview 시계열 차트 | Mann-Kendall trend test | "최근 3주 위험률 증가 추세 유의 (p=0.003)" |
| **Drift test (분포 이동)** | Model 페이지 | KS test / PSI (Population Stability Index) | "X1083 분포가 train 대비 이동 (PSI=0.31, 경계 0.25)" |
| **변수 중요도 신뢰도** | Model 페이지 | Null importance / permutation importance + std | "X739 importance는 random shuffle 대비 5σ 우위 — 진짜 신호" |
| **이상치 점수** | Drill-down Unit Diagnosis | IsolationForest score + percentile | "anomaly score 상위 0.5% — 강한 이상" |
| **예측 신뢰구간** | Drill-down Unit Diagnosis | CV fold 5개 예측의 std → ±1.96σ | "pred 0.0234 ± 0.0041 (95% CI)" |
| **위험 unit vs 정상 unit 변수 비교** | Model 페이지 | Cohen's d (effect size) + Welch's t-test | "X1083은 위험군에서 d=0.78, p<10⁻⁶ — 강한 분리력" |

### 5.2 표시 원칙

- **통계 결과는 차트 옆 작은 칩/배지로**: 거대한 분석 페이지가 아니라 시각화 옆에 1~2줄로 동반
  - 예: 시계열 차트 우측 상단에 `Mann-Kendall: p=0.003 ▲` 칩
- **유의수준 색 구분**:
  - 녹색 chip: p < 0.001 (강한 신호)
  - 주황 chip: 0.001 ≤ p < 0.05 (유의)
  - 회색 chip: p ≥ 0.05 (유의 X — 우연 가능)
- **ⓘ 호버 툴팁으로 검정 방법 설명** (PI가 통계 전문가는 아니므로)

### 5.3 산출 위치
- 통계량은 모델링 측에서 사전 계산하여 `4_output/dashboard/` 에 저장 → `prepare_data.py`가 parquet/json으로 통합 (§8.4 명세)
- Drift / trend test는 데이터 갱신 시점에 일괄 재계산

---

## 6. 페이지별 와이어프레임

### 6.1 Overview

```
┌─ PageHeader ────────────────────────────────────────────┐
│  Overview                              [Split: test ▾]   │
│  PI를 위한 위험 unit 사전 식별 모니터                     │
└──────────────────────────────────────────────────────────┘

┌─ KPI 카드 3개 (clean white + chip) ─────────────────────┐
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐    │
│ │ ⚠ 위험 unit % │ │ ▲ 평균 health │ │ 📦 처리 unit │    │
│ │  4.2%         │ │  0.00271      │ │  10,428      │    │
│ │ [▲+0.4%p ⚠]   │ │ [≈ 변동없음]   │ │ [▲ +312 ✓]   │    │
│ └──────────────┘ └──────────────┘ └──────────────┘    │
└──────────────────────────────────────────────────────────┘

┌─ Alert 영역 (통계 근거 포함) ───────────────────────────┐
│ ⚠ 위험 lot 3개:                                          │
│   [L0421 12.4% χ² p<0.01] [L0438 9.1% p=0.03] [L0445]  │
│   클릭 시 Drill-down으로 jump (해당 lot 펼친 상태)       │
└──────────────────────────────────────────────────────────┘

┌─ 메인 시계열 + Trend test 칩 ───────────────────────────┐
│  기간별 완료수량 & 예측 불량률      [일|주|월]           │
│  [Mann-Kendall: p=0.003 ▲ 증가 추세]  ← chip            │
│  ▮▮▮ 처리량 (좌축, status별 stack: 완료/대기/오늘)        │
│  ─── 평균 pred (우축, ppm 라인)                           │
└──────────────────────────────────────────────────────────┘

┌─ Top 위험 (좌우 2분할) ─────────────────────────────────┐
│  Top 10 위험 Wafer    │    Top 10 위험 Unit              │
│   [→ Drill-down]      │    [↓ CSV]                       │
└──────────────────────────────────────────────────────────┘

┌─ 예측 health 분포 ──────────────────────────────────────┐
│  zero-inflated 히스토그램 (0~0.05, 20 bins)              │
│  [Drift vs train: PSI 0.08 ✓]  ← chip                  │
└──────────────────────────────────────────────────────────┘
```

### 6.2 Drill-down

```
┌─ PageHeader ────────────────────────────────────────────┐
│  Drill-down                            [Split: test ▾]   │
│  Lot → Wafer → Die → Unit + 통계 근거 분석                │
└──────────────────────────────────────────────────────────┘

┌─ 좌측 (col-3): Lot/Wafer 트리 ─┬─ 우측 (col-9) ──────────────────┐
│  검색 [_______________]         │  ╔ Wafer Map ═══════════════╗  │
│  ─────────────────────         │  ║                          ║  │
│  ▼ L0421 ⚠12.4%[p<0.01]       │  ║  원형 wafer + die heatmap ║  │
│     #w001  8.2%                │  ║  토글: [단일] [lot 누적]  ║  │
│     #w002  15.6% ●  (선택)     │  ║  범례 (Spotfire 스케일)   ║  │
│  ▼ L0438                       │  ╚══════════════════════════╝  │
│     #w003  4.1%                │                                │
│     #w004  9.1%                │  ╔ Unit Diagnosis ══════════╗  │
│  ▶ L0445                       │  ║  ⚠ WARNING [p<0.001]      ║  │
│  ...                           │  ║  pred 0.0234 ±0.0041 (CI) ║  │
│                                │  ║  health / π / μ            ║  │
│                                │  ║  worst die (x,y)           ║  │
│                                │  ║                            ║  │
│                                │  ║  ── SHAP Top 20 ──         ║  │
│                                │  ║  X1083  +0.012  z=+2.31 ⚠ ║  │
│                                │  ║  X392   +0.008  z=+1.87    ║  │
│                                │  ║  X739   −0.005  z=−1.42    ║  │
│                                │  ║  ...                       ║  │
│                                │  ║                            ║  │
│                                │  ║  ── Anomaly ──             ║  │
│                                │  ║  IsoForest score 0.87       ║  │
│                                │  ║  rank 1,234 / 250K (top 0.5%)║  │
│                                │  ║                            ║  │
│                                │  ║  [📄 보고서 생성 →]         ║  │
│                                │  ╚══════════════════════════╝  │
└────────────────────────────────┴────────────────────────────────┘
```

선택된 lot/wafer 변경 시 우측 패널들이 모두 갱신. Unit Diagnosis는 wafer map의 die 클릭으로 unit 선택.

### 6.3 Model Diagnostics

```
┌─ PageHeader ────────────────────────────────────────────┐
│  Model Diagnostics                                       │
│  ZITboost 성능 / drift / 신뢰도                           │
└──────────────────────────────────────────────────────────┘

┌─ KPI 카드 (split별 RMSE) ──────────────────────────────┐
│  oof: 0.005703    val: 0.0XXXX    test: 0.0XXXX          │
│  vs 사내 최우수 0.005848  →  현재 2.5% 개선              │
└──────────────────────────────────────────────────────────┘

┌─ 예측 vs 실측 산점도          ┬─ 잔차 분포 ───────────┐
│  scatter + y=x 기준선 (oof)   │  histogram + density   │
│  log scale 토글               │  [Shapiro-Wilk: ...]   │
└────────────────────────────────┴─────────────────────────┘

┌─ Calibration ──────────────────┬─ Drift Test ──────────┐
│  pred 분위수 10 bin × 실측 평균│  feature별 PSI / KS    │
│  [Hosmer-Lemeshow: p=0.12]     │  Top 10 drift feature  │
└────────────────────────────────┴─────────────────────────┘

┌─ Fold별 RMSE (CV)              ┬─ Top Feature Importance┐
│  bar (5-fold) + std            │  Null Importance 비교  │
│                                │  [신뢰도: 5σ 우위 ✓]    │
└────────────────────────────────┴─────────────────────────┘

┌─ 위험 vs 정상 변수 비교 (신규) ─────────────────────────┐
│  Top 20 변수 × Cohen's d × p-value (위험군 vs 정상군)    │
│  → 어떤 변수가 위험을 가르는 진짜 신호인지 통계적 비교    │
└──────────────────────────────────────────────────────────┘
```

### 6.4 Data — 변경 없음

현 [pages/Data.tsx](frontend/src/pages/Data.tsx) 그대로 유지: status 토글 + 위험 필터 + 검색 (검사일/pred 범위 포함) + 정렬 + 페이지네이션 + CSV 다운로드. 백엔드는 prepare_data.py 산출 parquet을 메모리 로드 (zit-100 실험 기준 ~250 MB).

---

## 7. 전역 UI

### 7.1 Layout ([components/Layout.tsx](frontend/src/components/Layout.tsx))
- **사이드바** (w-60, 보라 그라디언트 `#4c1d95 → #2e1065`)
  - 프로필 영역 + nav 4개 + 하단 모델 요약
- **상단 헤더**: 페이지 제목 + Online 배지 + 알람벨

### 7.2 알람 벨 ([components/NotificationBell.tsx](frontend/src/components/NotificationBell.tsx))
- 위험 wafer/unit Top 5+5 (test split)
- alert에 **통계 근거 chip** (예: `χ² p<0.01`) 부착
- 클릭 → Drill-down jump

### 7.3 챗봇 ([components/ChatbotWidget.tsx](frontend/src/components/ChatbotWidget.tsx))
- 우하단 floating widget. 현재 mock → **RAG + LLM** 연동 (반도체/AI 도메인 RAG 데이터셋)

### 7.4 AI Agent (보고서 생성)
- Drill-down → Unit Diagnosis → "보고서 생성" 버튼
- LLM API: 위치 히트맵 + SHAP 요약 + **통계 근거 + trend** → PDF/HTML

---

## 8. 기술 스택 + 데이터 흐름

### 8.1 Frontend
| 항목 | 버전 | 비고 |
|---|---|---|
| React | 19.2 | |
| Vite | 8.0 | dev/build |
| TypeScript | 6.0 | |
| TailwindCSS | 3.4 | 커스텀 brand 팔레트 |
| Recharts | 3.8 | 차트 (line/bar/composed/scatter) |
| @tanstack/react-query | 5.10 | 서버 상태 + 캐싱 |
| react-router-dom | 7.14 | SPA 라우팅 |
| axios | 1.15 | HTTP |

### 8.2 Backend
| 항목 | 비고 |
|---|---|
| FastAPI | port 8765, parquet 메모리 로드 |
| pyarrow | parquet 읽기 |

### 8.3 데이터 흐름
```
4_output/final/zit_only/        ← zit-100 실험 결과 (optuna_jh_zit-final-100.db)
  ├ {oof,val,test}_die.csv
  └ {oof,val,test}_unit.csv
        ↓ prepare_data.py (한 번 실행)
5_dashboard/data/
  ├ die_predictions.parquet      (~9 MB, 174K rows)
  ├ unit_predictions.parquet     (~2 MB, 43K rows)
  ├ wafer_summary.parquet        (~50 KB)
  └ overview_stats.json          (split별 RMSE 등)
        ↓ FastAPI startup 시 메모리 로드
        ↓ React (TanStack Query — 캐싱 + 자동 refetch)
        ↓ 사용자 화면
```

### 8.4 모델팀 → 대시보드 추가 산출물 명세 (선택)

§5 통계 분석 (SHAP / Anomaly / drift / var-compare) 을 mock에서 실데이터로 전환할 때 필요한 파일들. 현재는 frontend에서 mock으로 표시 중이며, 모델팀이 아래 산출하면 `prepare_data.py`가 통합.

| 파일 | 컬럼 | 비고 |
|---|---|---|
| `4_output/dashboard/anomaly.parquet` | `ufs_serial`, `anomaly_score`, `anomaly_rank`, `method` | IsolationForest 등 |
| `4_output/dashboard/shap_topk.parquet` | `ufs_serial`, `rank` (1~20), `feature_name`, `shap_value`, `feature_value` | Long 포맷, top 20 |
| `4_output/dashboard/feature_importance.json` | `feature_name`, `importance_lgbm_gain`, `importance_shap_global`, `null_importance_zscore`, `rank_combined` | global. Null importance 포함 |
| `4_output/dashboard/model_diagnostics.json` | `metric_name`, `value`, `split`, `extra` (JSON: calibration bin, fold) | RMSE / calibration / fold별 |
| `4_output/dashboard/drift_test.json` | `feature_name`, `psi`, `ks_stat`, `ks_pvalue`, `verdict` | train vs test 분포 검정 |
| `4_output/dashboard/var_compare.json` | `feature_name`, `cohens_d`, `t_stat`, `p_value`, `mean_risk`, `mean_normal` | 위험군 vs 정상군 변수 비교 |

---

## 9. 디자인 시스템

### 9.1 비주얼 톤 — "TailAdmin Clean"

레퍼런스: 사용자 제시 이미지 (TailAdmin 스타일)

핵심 원칙:
- **흰 카드 + 작은 컬러 chip** (전체 보라 카드 같은 무거운 강조 X)
- **넉넉한 여백** (padding `20~24px`, gap `5~6`)
- **부드러운 그림자** (border 대신 `shadow-md`, hover `shadow-lg`)
- **차트 컬러는 보라 + 민트(#5eead4) 조합** (강한 빨강 톤 다운)
- **위험/경고/정상은 chip으로 통일** (`.chip-danger`, `.chip-warn`, `.chip-success`)

### 9.2 컬러 팔레트 ([tailwind.config.js](frontend/tailwind.config.js))

| 토큰 | hex | 용도 |
|---|---|---|
| `brand.primary` | `#4c1d95` | 사이드바 메인 |
| `brand.primaryDark` | `#2e1065` | 그라디언트 끝 |
| `brand.primaryLight` | `#7c3aed` | hover, 강조 |
| `brand.accent` | `#a78bfa` | 보조 차트 (라벤더) |
| `brand.accentSoft` | `#c4b5fd` | 옅은 라벤더 |
| `brand.mint` | `#5eead4` | **차트 보조 — 적극 사용** |
| `brand.danger` | `#e53e3e` | 위험 chip 글자 |
| `brand.warn` | `#f59e0b` | 경고 chip 글자 |
| `brand.success` | `#10b981` | Online / 정상 chip 글자 |
| `brand.bg` | `#f1f3f9` | 본문 배경 |
| `brand.surface` | `#ffffff` | 카드 배경 |
| `brand.border` | `#e2e8f0` | 구분선 |
| `brand.text` | `#1a202c` | 본문 글자 |
| `brand.textMuted` | `#64748b` | 보조 글자 |

신규 chip 유틸리티 (index.css에 추가 예정):
```css
.chip-danger { background:#fef2f2; color:#b91c1c; padding:2px 8px; border-radius:999px; font-size:11px; font-weight:600; }
.chip-warn   { background:#fffbeb; color:#b45309; ...same }
.chip-success{ background:#ecfdf5; color:#047857; ...same }
.chip-info   { background:#eff6ff; color:#1d4ed8; ...same }  /* 통계 chip */
.chip-muted  { background:#f1f5f9; color:#475569; ...same }
```

### 9.3 Wafer Map 컬러 스케일 ([lib/colors.ts](frontend/src/lib/colors.ts))
Spotfire 스타일 5-stop gradient (회색 → 옅은 노랑 → 노랑 → 주황 → 빨강).
임계값 정책 변경 시 `predColor` 함수에 `risk_threshold` 인자가 이미 있어 추후 2단계 분기 추가 용이.

### 9.4 타이포그래피
- 본문: Inter / Segoe UI / Malgun Gothic — **13px**
- 라벨: 11~12px, `text-brand-textMuted`
- KPI 값: **26px bold**, tabular-nums
- 코드/숫자: Consolas, `font-mono`, `tabular`

### 9.5 KPI 카드 패턴 (개정)
- **`tone="accent"` 폐지** (전체 보라 카드는 사이드바와 충돌해 무거움)
- 모든 KPI를 흰 카드로 통일, 강조는 **변화 chip** (▲ +0.4%p ⚠ vs 어제)
  - 비율 악화 → `chip-danger`
  - 비율 개선 → `chip-success`
  - 변동 없음 → `chip-muted`

### 9.6 컴포넌트 라이브러리

| 컴포넌트 | 용도 | 위치 |
|---|---|---|
| `Layout` | 사이드바 + 헤더 + Outlet + 알람벨 + 챗봇 | [components/Layout.tsx](frontend/src/components/Layout.tsx) |
| `PageHeader` | 제목 + 부제 + Split 토글 | [components/PageHeader.tsx](frontend/src/components/PageHeader.tsx) |
| `Panel` | 카드 컨테이너 | [components/Panel.tsx](frontend/src/components/Panel.tsx) |
| `KpiCard` | KPI 표시 (chip 기반 강조로 개정) | [components/KpiCard.tsx](frontend/src/components/KpiCard.tsx) |
| `WaferMap` | wafer SVG 히트맵 + 범례 | [components/WaferMap.tsx](frontend/src/components/WaferMap.tsx) |
| `NotificationBell` | 알람 드롭다운 (chip 부착) | [components/NotificationBell.tsx](frontend/src/components/NotificationBell.tsx) |
| `ChatbotWidget` | 우하단 챗봇 | [components/ChatbotWidget.tsx](frontend/src/components/ChatbotWidget.tsx) |
| `StatChip` (신규) | 통계 검정 결과 chip | components/StatChip.tsx (신규) |
| `ShapBar` (신규) | SHAP top-N bar (양/음 양방향) | components/ShapBar.tsx (신규) |

CSS 클래스: `.panel`, `.panel-title`, `.panel-body`, `.chip-*`, `table.spotfire`, `.btn`, `.sidebar-link` ([index.css](frontend/src/index.css))

### 9.7 차트 디자인 가이드 (Recharts)
- `CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3"` (한 단계 옅게)
- `XAxis/YAxis tick={{ fontSize: 11 }}`
- `Tooltip contentStyle={{ fontSize: 11 }}`
- 막대 색: `brand.primary` (메인) + `brand.mint` (보조 듀얼)
- 선 색: 위험률 `#f43f5e` (rose, 기존 강한 빨강 대체) / 정상치 `brand.success`
- 막대 radius: `[4, 4, 0, 0]`

---

## 10. API 엔드포인트

### 10.1 현재 ([api/main.py](api/main.py))
| Path | 설명 | 사용 페이지 |
|---|---|---|
| `GET /api/health` | 서버 상태 | — |
| `GET /api/overview` | KPI + split별 RMSE | Overview |
| `GET /api/triage` | 위험 트리아지 + 글로벌 색상 스케일 | Overview, Drill-down |
| `GET /api/wafers` / `/{key}` | wafer 리스트 / 상세 | Drill-down |
| `GET /api/units` / `/{ufs}` / `/{ufs}/report` | unit 리스트 / 상세 / 진단 | Data, Drill-down |
| `GET /api/lots` / `/{run_id}` | lot 리스트 / 상세 | Drill-down |

### 10.2 추가 예정
| Path | 설명 | 사용 페이지 |
|---|---|---|
| `GET /api/timeseries?granularity=day\|week\|month` | 시계열 (날짜 컬럼 도입 시) — 현재는 hash mock | Overview |
| `GET /api/units/{ufs}/shap` | unit SHAP top 20 + 변수값 | Drill-down |
| `GET /api/units/{ufs}/anomaly` | unit 어노멀리 점수 + 백분위 | Drill-down |
| `GET /api/lots/{run_id}/test` | lot vs baseline 검정 결과 | Drill-down, Overview alert |
| `GET /api/model-diagnostics/feature-importance` | global importance + null imp z | Model |
| `GET /api/model-diagnostics/drift` | feature별 PSI / KS | Model |
| `GET /api/model-diagnostics/var-compare` | 위험 vs 정상 변수 비교 (Cohen's d) | Model |
| `GET /api/model-diagnostics/calibration` | calibration bin + Hosmer-Lemeshow | Model |
| `GET /api/trend?granularity=...` | Mann-Kendall 등 추세 검정 | Overview chip |
| `POST /api/agent/report` | LLM 보고서 생성 | Drill-down |
| `POST /api/agent/chat` | RAG 챗봇 응답 | ChatbotWidget |

### 10.3 API 구현 원칙
- FastAPI startup 시 parquet 4개를 메모리 로드 후 재사용
- pandas DataFrame 인덱싱/필터로 처리 (현재 데이터 규모 ~250 MB 수준)
- 향후 데이터 폭증(예: 라이브 운영 데이터) 시 DuckDB lazy 쿼리로 전환 검토

---

## 11. 데이터 운용

### 11.1 현재 데이터 출처
- **모델 산출**: zit-100번 실험 ([4_output/final/zit_only/](../4_output/final/zit_only/) — `optuna_jh_zit-final-100.db` 기반 oof/val/test die·unit CSV)
- **대시보드 산출물**: `prepare_data.py`가 위 CSV 6개를 읽어 `5_dashboard/data/` 에 parquet 4개 생성
  - `die_predictions.parquet` (~9 MB, 174K 행)
  - `unit_predictions.parquet` (~2 MB, 43K 행)
  - `wafer_summary.parquet` (~50 KB)
  - `overview_stats.json` (split별 RMSE 등)

### 11.2 status 데모 분배
실 운영 환경에서는 unit별 검사일이 데이터에 포함될 것. 현 데모에서는 백엔드 startup 시 **`hash(ufs_serial) % 10`** 으로 8:1:1 분배:
- bucket 0~7 (80%) → `completed`
- bucket 8 (10%) → `today`
- bucket 9 (10%) → `pending`

`inspected_date`도 status에 따라 가짜 날짜 부여 (today=오늘, pending=어제, completed=7~60일 전 분포). 실 운영 데이터 도입 시 이 로직 폐기하고 실제 컬럼 사용.

### 11.3 향후 확장 시 고려
| 시나리오 | 대응 |
|---|---|
| 데이터 양 1000만 행 이상으로 폭증 | 메모리 parquet → DuckDB lazy 쿼리 전환 검토 |
| 실시간 운영 데이터 도입 | polling 또는 WebSocket으로 alert 자동 갱신 |
| §5 통계 카탈로그 mock → 실데이터 | 모델팀 §8.4 표준 산출물 도입 후 prepare_data.py 통합 |

---

## 12. 폴더 구조

```
5_dashboard/
├── README.md                    # 사용자용 (실행 방법)
├── design.md                    # 이 문서 (설계)
├── start.bat                    # Windows 일괄 실행
├── prepare_data.py              # 4_output/final/zit_only → data/*.parquet ETL
├── api/
│   ├── main.py                  # FastAPI (parquet 메모리 로드)
│   └── requirements.txt
├── data/
│   ├── die_predictions.parquet
│   ├── unit_predictions.parquet
│   ├── wafer_summary.parquet
│   └── overview_stats.json
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   └── src/
│       ├── App.tsx              # 4 페이지 라우팅 (Drilldown/Model 추가)
│       ├── main.tsx
│       ├── index.css            # 전역 CSS + chip 유틸
│       ├── components/          # Layout, Panel, KpiCard, WaferMap, StatChip, ShapBar...
│       ├── pages/               # Overview, Drilldown, Model, Data
│       ├── lib/                 # api, colors, format
│       └── assets/
├── design_drafts/               # 참조 이미지 6장
└── docs/                        # 라이브러리/논문/패턴 자료
    ├── 01_shadcn-ui.md ~ 09_dashboard-design-patterns.md
    ├── dashboard_resources_index.md
    └── 논문요약_React_Dashboard.md
```

---

## 13. 진행 상태 / 다음 작업

### 13.1 이미 구현됨
- [x] 사이드바 + 헤더 Layout
- [x] Overview 페이지 (KPI 2개 + 시계열 + Top + 히스토그램)
- [x] Wafers 페이지 (좌: wafer 리스트, 중: WaferMap, 우: Unit Diagnosis)
- [x] Data 페이지 (필터 + 검색 + 정렬 + CSV)
- [x] 알람 벨 + 챗봇 위젯 (mock)
- [x] FastAPI 엔드포인트 10개 (메모리 parquet 로드 방식)
- [x] prepare_data.py (zit-100 → parquet/json 산출)

### 13.2 다음 작업 (이 문서 확정 후, 우선순위 순)

**Step 1 — 디자인 시스템 코드 반영**
1. `tailwind.config.js` 보완 (변경 없음, mint 활용 확대)
2. `index.css`에 chip 유틸리티 (`.chip-danger`, `.chip-warn`, `.chip-success`, `.chip-info`, `.chip-muted`) 추가
3. `KpiCard.tsx` 개정 — `tone="accent"` 폐지, 변화 chip props 추가
4. `StatChip.tsx` 신규 — 통계 검정 결과 chip 컴포넌트
5. 차트 컬러 톤 다운 (위험률 `#f43f5e`, grid `#e2e8f0`)

**Step 2 — 페이지 골격**
6. `App.tsx` / `Layout.tsx` 갱신 — nav 4개 (Drilldown, Model 추가)
7. `Drilldown.tsx` 신규 — 기존 Wafers + Lots 통합 + lot 트리 좌측
8. `Model.tsx` 신규 — 패널들 (KPI / scatter+잔차 / calibration+drift / fold+importance / var-compare)
9. `Wafers.tsx`, `Lots.tsx` 삭제
10. `Overview.tsx` 갱신 — KPI 3개로 + Alert 영역 + trend chip 추가

**Step 3 — 통계 분석 통합 (모델팀 §8.4 산출 후)**
11. `ShapBar.tsx` 신규 — SHAP top 20 양/음 bar 컴포넌트
12. Drill-down Unit Diagnosis에 SHAP + Anomaly + 신뢰구간 통합 (mock → 실데이터)
13. Overview alert / 시계열 chip에 통계 검정 결과 부착
14. Model 페이지에 drift / var-compare 패널 구현

**Step 4 — 향후 확장 (선택)**
15. 운영 데이터에 검사일 컬럼 도입 시 hash 분배 로직 폐기
16. 데이터 폭증 시 메모리 parquet → DuckDB lazy 쿼리 전환
17. polling 또는 WebSocket으로 alert 실시간화

### 13.3 Phase 4 후반 (AI Agent / RAG)
- [ ] LLM API 연동 (Drill-down 보고서 생성, 챗봇)
- [ ] RAG 데이터셋 구성 (반도체/AI 도메인)
- [ ] 보고서에 통계 근거 narrative 자동 삽입

---

## 14. 참고 자료

> 논문/PDF 원본은 `99_학습자료/스터디자료/dashboard/papers/`, `99_학습자료/스터디자료/rag_agent/papers/`. 카탈로그: [docs/dashboard_resources_index.md](docs/dashboard_resources_index.md)

### 14.1 디자인 원칙
- **Shneiderman 1996** — "Overview → Filter → Detail" 3단 구조
- **Stephen Few** — Information Dashboard Design (정보 밀도, 포매팅)
- **Wickham** — Layered Grammar of Graphics

### 14.2 XAI 인터페이스 (Drill-down SHAP)
- Human-Centered XAI Interface Survey — XAI 8가지 설계 원칙
- Visual Analytics for XAI — 성능+drift+SHAP 통합
- SHAP and LIME Perspective — global+local 동시 시각화

### 14.3 통계 기반 의사결정 (신규 — §5의 근거)
- Mann-Kendall trend test — 시계열 추세 검정
- KS test / PSI — 분포 drift 검정
- Cohen's d — 효과 크기 (변수 분리력)
- Null importance — feature importance 통계적 유의성 검증
- Hosmer-Lemeshow — calibration goodness-of-fit

### 14.4 라이브러리 / 패턴 ([docs/](docs/))
- [docs/05_recharts.md](docs/05_recharts.md) — 차트 라이브러리 (현재 채택)
- [docs/01_shadcn-ui.md](docs/01_shadcn-ui.md) — UI 컴포넌트
- [docs/06_tanstack-table.md](docs/06_tanstack-table.md) — Data 페이지 고도화
- [docs/09_dashboard-design-patterns.md](docs/09_dashboard-design-patterns.md) — RCC 적용 가이드
- [docs/논문요약_React_Dashboard.md](docs/논문요약_React_Dashboard.md) — React 대시보드 + NL2VIS 요약

### 14.5 도메인 레퍼런스
- Wafer-Yield-Intelligence — 도메인 일치 Streamlit 앱
- Wafer Map Defect Tiny ViT — Occlusion Sensitivity Heatmap

### 14.6 디자인 영감 ([design_drafts/](design_drafts/))
참조 이미지 6장 + 사용자 제시 TailAdmin 스타일

### 14.7 발표 자료
- [SK하이닉스_AI햄스틴_중간발표.pptx](SK하이닉스_AI햄스틴_중간발표.pptx) — slide 32~45 대시보드/Agent/RAG 설계 참조