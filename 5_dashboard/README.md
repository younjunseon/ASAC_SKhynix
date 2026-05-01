# Wafer Health Dashboard

SK Hynix Wafer Test 기반 Field Health(RCC) 예측 결과를 시각화하는 인터랙티브 대시보드.

- **백엔드**: FastAPI + pandas (port 8765)
- **프론트엔드**: React + Vite + Recharts + TailwindCSS (port 5173)

## 페이지 구성

| 페이지 | 내용 |
|---|---|
| **Overview** | KPI(예측 불량률·평균 health), 기간별 완료수량+불량률 듀얼축 차트(일/주/월), Top 10 위험 wafer/unit, 예측 분포 히스토그램 |
| **Wafer Map** | wafer 리스트 → die 단위 히트맵 → unit 진단 리포트 (3-pane drill-down) |
| **Data** | unit 데이터 테이블 — split 필터, 검색, 정렬, CSV 다운로드 |

전역 UI:
- **우상단 알람 벨** — 신규 위험 wafer/unit 알림, 클릭 시 해당 페이지로 이동
- **우하단 챗봇** — 자연어 질의 AI Agent (현재 mock, 추후 LLM 연동 예정)

## 사전 요구사항

- **Python 3.10+**
- **Node.js 18+** (npm 포함)
- **데이터 파일** (`data/` 폴더에 4개 parquet/json) — 레포에 포함되어 있음

## 설치 (최초 1회)

### 1. 레포 clone
```bash
git clone <레포-URL>
cd <레포-폴더>
```

### 2. 백엔드 패키지 설치
```bash
pip install -r api/requirements.txt
```

### 3. 프론트엔드 패키지 설치
```bash
cd frontend
npm install
cd ..
```

## 실행

### Windows
`start.bat` 더블클릭 → 검은 창 2개 자동 실행
- **Wafer API** (uvicorn, port 8765)
- **Wafer Frontend** (vite, port 5173)

### Mac / Linux
터미널 2개 열고:
```bash
# 터미널 1 (백엔드)
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8765

# 터미널 2 (프론트엔드)
cd frontend
npm run dev -- --host 0.0.0.0
```

### 접속
- 본인 PC: http://localhost:5173
- 같은 와이파이 다른 PC: http://<본인-IP>:5173 (예: `http://192.168.0.53:5173`)

본인 IP 확인:
```bash
ipconfig          # Windows
ifconfig          # Mac/Linux
```

## 폴더 구조

```
5_dashboard/
├── api/                    # FastAPI 백엔드
│   ├── main.py             # API 엔드포인트
│   └── requirements.txt    # Python 패키지
├── frontend/               # React 프론트엔드
│   ├── src/                # 소스 코드 (.tsx, .ts)
│   ├── package.json        # npm 패키지 정의
│   └── vite.config.ts      # Vite 설정 (proxy, host)
├── data/                   # 모델 산출물 (parquet, json)
├── prepare_data.py         # 4_output → data/ 변환 스크립트
├── start.bat               # Windows 일괄 실행
└── README.md
```

## 개발 워크플로우

### 코드 수정
- `frontend/src/**/*.tsx` 수정 → 저장 즉시 브라우저 자동 갱신 (Vite 핫리로드)
- `api/main.py` 수정 → uvicorn `--reload` 옵션으로 자동 재시작

### Git 협업
```bash
# 작업 시작 전
git pull

# 작업 후
git add .
git commit -m "변경 내용 요약"
git push
```

## API 엔드포인트

| Path | 설명 |
|---|---|
| `GET /api/overview` | 전체 KPI + split별 RMSE |
| `GET /api/wafers` | wafer 리스트 |
| `GET /api/wafers/{wafer_key}` | wafer 상세 (die map) |
| `GET /api/units` | unit 리스트 (페이지네이션) |
| `GET /api/units/{ufs_serial}` | unit 상세 |
| `GET /api/units/{ufs_serial}/report` | unit 진단 리포트 |
| `GET /api/lots` | lot 리스트 |
| `GET /api/triage` | 위험 unit/wafer 트리아지 |

## 트러블슈팅

### "API 연결 실패" 에러
- `api/` 검은 창이 떠있는지 확인
- `frontend/.env.local`의 `VITE_API_BASE`가 비어있는지 확인 (proxy 사용)
- 브라우저 강제 새로고침: `Ctrl + Shift + R`

### 다른 PC에서 접속 안 됨
- Windows 방화벽에서 포트 5173, 8765 허용
- 같은 와이파이인지 확인
- 공유기 "AP 격리(client isolation)" 꺼져 있는지 확인

### 포트 충돌
- API 포트 변경: `start.bat`의 `--port 8765` 수정
- 프론트 포트 변경: `frontend/vite.config.ts`에 `server.port` 추가
