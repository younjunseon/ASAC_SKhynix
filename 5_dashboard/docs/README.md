# Dashboard Docs — React 디자인 참고 자료

Phase 4 (인터랙티브 대시보드 + AI Agent) 구현 시 Claude/사람이 컨텍스트로 읽어 사용하는 자료 모음. 원본은 [99_학습자료/스터디자료/dashboard/](../../99_학습자료/스터디자료/dashboard/)에 있고, 이 폴더는 5_dashboard 작업 시 바로 참조하기 위한 스냅샷이다.

## 파일

### 이론·논문 (학습자료 스냅샷)

| 파일 | 용도 |
|------|------|
| [논문요약_React_Dashboard.md](논문요약_React_Dashboard.md) | React 프론트엔드 + LLM(NL2VIS / Text-to-SQL) 논문 19편 상세 요약. 본 프로젝트 React 대시보드 + AI Agent의 직접 레퍼런스 |
| [dashboard_resources_index.md](dashboard_resources_index.md) | 전체 대시보드 자료 카탈로그(Streamlit + React 통합). 원본 `99_학습자료/스터디자료/dashboard/INDEX.md`와 동일. 다른 자료(논문 PDF, GitHub README, 블로그) 위치 찾을 때 사용 |

### 구현 도구 README (2026-04-30 추가)

학습자료에는 Streamlit README만 있어 React 노선에서 비어있던 "구현 도구" 영역을 보충. 8개 GitHub README + 디자인 패턴 카탈로그 1개.

| # | 파일 | 원본 | 역할 |
|---|------|------|------|
| 01 | [01_shadcn-ui.md](01_shadcn-ui.md) | [shadcn-ui/ui](https://github.com/shadcn-ui/ui) | UI 컴포넌트 베이스 (copy-paste, Tailwind) |
| 02 | [02_shadcn-admin.md](02_shadcn-admin.md) | [satnaing/shadcn-admin](https://github.com/satnaing/shadcn-admin) | Admin 템플릿 (Vite + shadcn) |
| 03 | [03_tremor.md](03_tremor.md) | [tremorlabs/tremor](https://github.com/tremorlabs/tremor) | 대시보드 차트 (Recharts 기반 디자인 레이어) |
| 04 | [04_vercel-ai-chatbot.md](04_vercel-ai-chatbot.md) | [vercel/ai-chatbot](https://github.com/vercel/ai-chatbot) | AI Agent UI 풀템플릿 (Next.js + shadcn) |
| 05 | [05_recharts.md](05_recharts.md) | [recharts/recharts](https://github.com/recharts/recharts) | 차트 라이브러리 (shadcn 공식 채택) |
| 06 | [06_tanstack-table.md](06_tanstack-table.md) | [TanStack/table](https://github.com/TanStack/table) | unit-level 예측 테이블 (정렬/필터/페이징) |
| 07 | [07_nivo.md](07_nivo.md) | [plouc/nivo](https://github.com/plouc/nivo) | 30+ 차트, wafer map heatmap 후보 |
| 08 | [08_tailadmin-react.md](08_tailadmin-react.md) | [TailAdmin/free-react-tailwind-admin-dashboard](https://github.com/TailAdmin/free-react-tailwind-admin-dashboard) | 7개 admin 변형 무료 템플릿 |
| 09 | [09_dashboard-design-patterns.md](09_dashboard-design-patterns.md) | [dashboarddesignpatterns.github.io](https://dashboarddesignpatterns.github.io/) (Bach 2022, IEEE VIS) | 42개 디자인 패턴 카탈로그 + RCC 적용 매핑 |

### design.md TBD 항목 → 문서 매핑

design.md §2 기술 스택 결정 시 아래 doc을 비교 자료로 사용한다.

| design.md TBD | 후보 | 참조 문서 |
|---|---|---|
| Framework (Vite / Next.js / CRA) | Vite(02), Next.js(04) | 02, 04 |
| Styling | Tailwind 합의 시 → shadcn ecosystem | 01, 02, 08 |
| 차트 라이브러리 | Recharts / Tremor / Nivo | 03, 05, 07 + 논문요약 §B-7, D-2 |
| UI 컴포넌트 | shadcn/ui 합의 → 다른 옵션 비교 불요 | 01 |
| 테이블 | TanStack Table | 06 |
| AI Agent UI (§4.5) | vercel/ai-chatbot 풀템플릿 | 04 + 논문요약 §A, §B |
| 디자인 컨셉 / 페이지 구성 | Bach 2022 패턴 카탈로그 | 09 |

## Claude와 함께 작업할 때 사용 패턴

### React 컴포넌트/페이지 설계
> "@5_dashboard/docs/논문요약_React_Dashboard.md 의 카테고리 D(Dashboard 구현 — Flask vs React, MERN+Nivo+MUI)와 design.md를 읽고, 메인 대시보드 페이지의 컴포넌트 트리 초안 작성"

### AI Agent 아키텍처 설계
> "@5_dashboard/docs/논문요약_React_Dashboard.md 의 A-1 (V-NLI 7단계 파이프라인)과 B-1 (LIDA 4단계)을 합쳐서, RCC 대시보드용 자연어 질의 → 차트 생성 Agent 아키텍처 설계"

### Text-to-SQL 구현
> "@5_dashboard/docs/논문요약_React_Dashboard.md 의 카테고리 C(Text-to-SQL) 중 DTS-SQL을 활용한 7B 모델 기반 사내 데이터 질의 모듈 스펙 작성. 프라이버시 제약 우선"

### 차트 라이브러리 결정
> "@5_dashboard/docs/논문요약_React_Dashboard.md 의 D-2 (MERN + Nivo) 와 B-7 (VegaChat — Vega-Lite) 비교 후 우리 프로젝트 차트 라이브러리 추천. design.md의 §2 기술 스택 채워줘"

## 폴더 외부 자료가 필요할 때

원본 폴더에는 PDF 논문, Streamlit GitHub README, 블로그 튜토리얼 등 더 많은 자료가 있다. 필요할 때 [dashboard_resources_index.md](dashboard_resources_index.md)에서 위치를 찾고 원본 경로([99_학습자료/스터디자료/dashboard/](../../99_학습자료/스터디자료/dashboard/))로 접근.

PDF는 본 docs/ 폴더로 복사하지 않았다 (바이너리, 용량). PDF가 직접 필요하면 원본 경로에서 읽는다.

## 동기화

원본이 갱신되면 수동으로 다시 복사:

```bash
cp "99_학습자료/스터디자료/dashboard/papers/react/논문요약_React_Dashboard.md" "5_dashboard/docs/"
cp "99_학습자료/스터디자료/dashboard/INDEX.md" "5_dashboard/docs/dashboard_resources_index.md"
```
