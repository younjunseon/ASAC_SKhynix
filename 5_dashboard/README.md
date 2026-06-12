# Wafer Health Dashboard

SK Hynix Wafer Test 기반 Field Health(RCC) 예측 결과를 시각화하는 인터랙티브 대시보드.

- **백엔드**: FastAPI + pandas (port 8765) — `data/` 의 parquet/json/csv 를 읽어 `/api/...` JSON 으로 서빙
- **프론트엔드**: React 19 + Vite + TypeScript + Tailwind + Recharts + ECharts (port 5173)
- **자립 패키지**: 이 `5_dashboard/` 폴더만 있으면 실행됨 — 부모 프로젝트(`0_data/`, `4_output/` 등)에 의존하지 않는다. 모델 산출물은 `data/` 에 미리 구워져 있고, 정적 CSV(모델 성능·주별 생산량)는 `frontend/public/` 에 들어 있다.

> **셸 출처**: 사이드바·기본 틀은 "용인" 대시보드, 상단 헤더는 "준선" 대시보드를 합쳐 구성. 페이지 본문은 준선 베이스 + 용인 시각화 일부 이식.

## 페이지 구성

| 사이드바 메뉴 | 라우트 | 내용 |
|---|---|---|
| **품질 불량 예측 현황** | `/` | KPI(예측 불량률·평균 health), 기간별 완료수량+불량률 듀얼축(일/주/월), Top 10 위험 wafer/unit, 예측 분포 히스토그램 |
| **데일리 유닛 현황·자재 분석** | `/data` | unit 데이터 테이블 — status 필터, 검색, 정렬, CSV 다운로드 |
| **다이 레벨 정밀 분석 > 다이 차원** | `/drilldown` | 주별 생산량 차트(이식) + Lot/Wafer 트리 → Wafer Map → Unit 진단 리포트 (3-pane drill-down) |
| **다이 레벨 정밀 분석 > 웨이퍼 차원** | `/drilldown/wafer` | (빈 페이지 — Delta-Q Map 형태로 구현 예정) |
| **다이 레벨 정밀 분석 > 로트 차원** | `/drilldown/lot` | (빈 페이지 — Delta-Q Map 형태로 구현 예정) |
| **모델 성능 분석** | `/model` | 예측 성능(RMSE 카드·Stage1 분류·모델별 RMSE·실제vs예측 산점도·예측 분포) + 변수 진단(Feature Importance·SHAP·Pearson r·Cohen's d) |

전역 UI:
- **우상단 알람 벨** — 신규 위험 wafer/unit 알림, 클릭 시 해당 페이지로 이동 (`/api/alerts/today`)
- **우하단 챗봇** — 자연어 질의 AI Agent (현재 키워드 기반 mock. 실제 Claude API 연동은 별도 작업 중 → 추후 `/api/chat` 으로 합류 예정)

## 사전 요구사항

- **Python 3.10+** — `pip install -r api/requirements.txt` (fastapi, uvicorn, pandas, pyarrow, numpy)
- **Node.js 18+** — `cd frontend && npm install`
- 데이터: `data/*.parquet`, `data/overview_stats.json`, `data/model/*` 와 `frontend/public/*.csv` 는 레포에 포함되어 있음 (별도 다운로드 불필요)

## 설치 (최초 1회)

```bash
# 백엔드 패키지
pip install -r api/requirements.txt

# 프론트엔드 패키지
cd frontend && npm install && cd ..
```

## 실행

### Windows
`start.bat` 더블클릭 → 검은 창 2개 자동 실행 (Wafer API: uvicorn :8765 / Wafer Frontend: vite :5173)

### Mac / Linux
```bash
# 터미널 1 (백엔드)
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8765
# 터미널 2 (프론트엔드)
cd frontend && npm run dev -- --host 0.0.0.0
```

### 접속
- 본인 PC: http://localhost:5173
- 같은 와이파이 다른 PC: http://<본인-IP>:5173 (`ipconfig` / `ifconfig` 로 IP 확인)

### 단일 포트 배포 (선택)
`cd frontend && npm run build` → `frontend/dist/` 생성. 이후 `uvicorn api.main:app --port 8765` 만 띄우면 FastAPI 가 `dist/` 도 서빙하므로 `http://localhost:8765` 한 곳으로 끝남.

## 폴더 구조

```
5_dashboard/                  ← 이 폴더만 공유하면 실행 가능 (부모 프로젝트 불필요)
├── api/
│   ├── main.py               # FastAPI 엔드포인트 (data/ 만 읽음)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/        # Layout(사이드바·헤더), NotificationBell, ChatbotWidget,
│   │   │                      #   WaferMap, UnitReportModal, Panel, PredictionPerfSection(이식),
│   │   │                      #   WeeklyProductionChart(이식), ComingSoon ...
│   │   ├── pages/             # Overview, Data, Drilldown, DrilldownWafer, DrilldownLot, Model
│   │   ├── hooks/useCSV.ts    # public/*.csv fetch+파싱 (papaparse)
│   │   ├── lib/               # api.ts(axios), chart.ts, colors.ts, format.ts
│   │   └── App.tsx, main.tsx
│   ├── public/                # 정적 CSV: metrics.csv, dashboard_units.csv, dashboard_dates.csv
│   ├── package.json, vite.config.ts, tailwind.config.js
│   └── .env.local            # VITE_API_BASE (비워두면 vite proxy 사용 / gitignore)
├── data/                      # 모델 산출물 (런타임 입력)
│   ├── die_predictions.parquet, unit_predictions.parquet, wafer_summary.parquet
│   ├── unit_features.parquet, normal_baseline.parquet, position_stats.parquet
│   ├── overview_stats.json
│   └── model/                # feature_importance.csv, psi.csv, var_compare.csv, fold_metrics.json, oof/blend/stack ...
├── docs/                      # 디자인·라이브러리 참고 자료
├── design.md, design_drafts/
├── start.bat
└── README.md
```

> **데이터 재생성**: `data/` 의 parquet·json 과 `frontend/public/` 의 CSV 는 모델링 산출물에서 빌드된 것이다. 빌드 스크립트(`prepare_data.py`, `build_model_artifacts.py` 등)는 이 폴더에 포함하지 않았다 — 부모 프로젝트 쪽에서 모델링이 최종 확정되면 그쪽에서 다시 굽는다.

## 주요 API 엔드포인트

| Path | 설명 |
|---|---|
| `GET /api/overview` | 전체 KPI + status별(today/pending/completed) 통계 + RMSE |
| `GET /api/units` `…/{ufs_serial}` `…/report` | unit 리스트 / 상세 / 진단 리포트 |
| `GET /api/wafers` `…/{wafer_key}` `/api/wafer-grid` | wafer 리스트 / 상세(die map) / 그리드 |
| `GET /api/lots` `…/{run_id}` `…/aggregate-map` | lot 리스트 / 상세 / 누적 맵 |
| `GET /api/triage` `/api/position_risk` | 위험 unit·wafer 트리아지 / position별 위험도 |
| `GET /api/alerts/today` | 우상단 알람 벨 데이터 |
| `GET /api/model/fold-metrics` `…/feature-importance` `…/psi` `…/feature-corr` `…/shap` `…/var-compare` | 모델 변수 진단 (Model 페이지 하단) |

`GET /api` 로 전체 엔드포인트 목록 확인 가능. (`/api/docs` — FastAPI 자동 문서)

## 트러블슈팅

- **"API 연결 실패"** — `api/` 창이 떠있는지 / `frontend/.env.local` 의 `VITE_API_BASE` 가 비어있는지 확인 / `Ctrl+Shift+R`
- **다른 PC 접속 안 됨** — Windows 방화벽에서 포트 5173·8765 허용, 같은 와이파이, 공유기 "AP 격리" off
- **포트 충돌** — API: `start.bat` 의 `--port 8765` 수정 / 프론트: `vite.config.ts` 에 `server.port` 추가
- **차트 라이브러리 2개** — 준선 페이지는 recharts, 용인 이식분(주별 생산량·예측 성능)은 echarts. 번들이 ~620KB(gzip) 정도로 커진 건 echarts 때문 — 추후 통일 시 한쪽으로 정리 가능.
