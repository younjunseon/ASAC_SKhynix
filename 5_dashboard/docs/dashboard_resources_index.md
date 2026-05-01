# Dashboard 자료 인덱스

Streamlit 대시보드 구축 (Phase 4) 관련 자료. Claude Code와 함께 쓰기 좋은 자료 중심.

---

## 📄 논문 (papers/)

실제 다운로드한 PDF — 전부 arXiv 또는 공개 논문

| 파일 | 핵심 내용 | 적용 |
|------|----------|------|
| `Streamlit_AI_Trust_Platform.pdf` | Streamlit 기반 AI 신뢰성 플랫폼 (arXiv 2211.12851) | Streamlit으로 ML 평가/모니터링 탭 구성 |
| `Human_Centered_XAI_Interface_Survey.pdf` | XAI 인터페이스 설계 8가지 원칙 서베이 (2403.14496) | SHAP 대시보드 UX 설계 |
| `Visual_Analytics_for_XAI.pdf` | XAI용 Visual Analytics 프레임워크 (2507.10240) | 성능+drift+SHAP 통합 설계 |
| `Harder_Better_Faster_Stronger_Interactive_Viz.pdf` | 인터랙티브 시각화가 AI 사용 격차 해소 (2404.02147) | hover/filter/linked-view 중심 설계 |
| `SHAP_and_LIME_Perspective.pdf` | SHAP vs LIME 비교 리뷰 (2305.02012) | global+local 동시 시각화 근거 |
| `Interactive_Sensor_Dashboard_Smart_Manufacturing.pdf` | 제조 센서 대시보드 (2005.05025) | WT 피처 시계열/맵 차트 선정 |
| `Wafer_Map_Defect_Tiny_ViT.pdf` | Wafer map 결함 분류 (2504.02494) | Occlusion Sensitivity Heatmap 참고 |
| `XAI_Design_Evaluation_Survey.pdf` | XAI 설계 및 평가 서베이 (1811.11839) | faithfulness 평가 기준 |
| `Shneiderman_1996_The_Eyes_Have_It.pdf` | "Overview → Filter → Detail" 원칙 | **대시보드 3단 구조의 핵심 원리** |
| `Wickham_Layered_Grammar_of_Graphics.pdf` | Grammar of Graphics 레이어드 확장 | Altair/Plotly 차트 모듈화 |
| `Few_Rich_Data_Poor_Data.pdf` | 정보 밀도 vs 가독성 (Stephen Few) | 대시보드 정보 설계 |
| `Few_Formatting_and_Layout_Matter.pdf` | 포매팅/레이아웃 디자인 (Stephen Few) | KPI 카드/섹션 배치 |

---

## 💻 GitHub READMEs (github_readmes/)

| 파일 | 용도 |
|------|------|
| **큐레이션 리스트** | |
| `best-of-streamlit.md` | 100개 Streamlit 앱 랭킹 (17K★) |
| `MarcSkovMadsen-awesome-streamlit.md` | 가장 유명한 awesome 리스트 |
| `JjongX-awesome-streamlit.md` | 주요 컴포넌트 큐레이션 |
| `Best-Ever-Streamlit-Applications.md` | 101개 ML 앱 모음 |
| **고급 컴포넌트 (★프로젝트 스택)** | |
| `streamlit-extras.md` | 20+ 시각적 확장 (metric cards, Grid) |
| `streamlit-aggrid.md` | 엑셀급 인터랙티브 테이블 (unit-level 예측표) |
| `streamlit-shadcn-ui.md` | shadcn 기반 모던 UI |
| `streamlit-option-menu.md` | 사이드바 네비게이션 |
| `streamlit-folium.md` | 웨이퍼맵 2D heatmap |
| `streamlit-echarts.md` | 고급 인터랙티브 차트 |
| **도메인 유사 오픈소스 (★★벤치마크)** | |
| `Wafer-Yield-Intelligence.md` | 웨이퍼 수율 Streamlit 앱 — **도메인 일치** |
| `Defect-Prediction-Semiconductor-Lithography.md` | 반도체 리소그래피 결함 예측 |
| `streamlit-ML-Model-Builder.md` | 공식 ML 모델 빌더 |
| **튜토리얼** | |
| `streamlit-for-datascience.md` | 데이터사이언스 튜토리얼 (dataprofessor) |
| `awesome-streamlit-themes.md` | 10개 프로페셔널 테마 |
| **Claude/Cursor Rules (★★★최우선)** | |
| `streamlit_cursor_rule.mdc` | Streamlit 전용 Cursor/Claude rules → CLAUDE.md로 변환 사용 |

---

## ✍️ 블로그/튜토리얼 (blog_articles/)

| 파일 | 출처 |
|------|------|
| `01_Streamlit_Theming_Guide.md` | Streamlit 공식 테마 설정 문서 |
| `02_Evidently_ML_Monitoring_with_Streamlit.md` | Evidently + Streamlit 모니터링 튜토리얼 |

---

## 🎯 Claude와 작업 시 활용 패턴

### 패턴 1: 초기 설계 단계
> "dashboard/papers/Shneiderman_1996_The_Eyes_Have_It.pdf 와 CLAUDE.md 를 읽고, RCC 예측 대시보드의 Overview→Filter→Detail 3단 구조 초안 작성"

### 패턴 2: 컴포넌트 선택
> "@streamlit-aggrid.md 참고해서 unit-level 예측 결과 테이블 컴포넌트 작성. ufs_serial, 예측값, risk 카테고리, SHAP top 5 피처 컬럼 포함"

### 패턴 3: 도메인 레퍼런스
> "dashboard/github_readmes/Wafer-Yield-Intelligence.md 의 wafer map 시각화 방식을 참고해서 우리 run_wf_xy 파싱 결과를 시각화하는 Streamlit 페이지 작성"

### 패턴 4: 디자인 규칙 주입
> "streamlit_cursor_rule.mdc 내용을 프로젝트 CLAUDE.md 에 통합해서, 이후 대시보드 코드 작성 시 자동으로 이 규칙을 따르게 해줘"

### 패턴 5: 모니터링 구축
> "@02_Evidently_ML_Monitoring_with_Streamlit.md 튜토리얼 따라서 우리 RCC 모델의 WT feature drift 모니터 탭 구현"

---

## 🔵 React 대시보드 & AI Agent 논문 (papers/react/)

React 프론트엔드 + LLM 기반 NL2VIS/Text-to-SQL 중심. 요약: `papers/react/논문요약_React_Dashboard.md`

### NL2VIS 서베이 (A)
| 파일 | 핵심 내용 |
|------|----------|
| `NL2VIS_Survey_Luo2022_Towards_NL_Interfaces_for_DataVis.pdf` | V-NLI 7단계 파이프라인 서베이 (IEEE TVCG 2022) — **Agent 아키텍처 뼈대** |
| `NL2VIS_Survey_2024_NL_Interfaces_Tabular_Data.pdf` | Text-to-SQL + Text-to-Vis 통합 서베이 (IEEE TKDE 2024) |
| `Chatbot_NL_VIS_2023_Scoping_Review.pdf` | 챗봇 V-NLI 20개 시스템 AINT 프레임워크 분석 (MDPI 2023) |

### LLM 기반 시각화 생성 (B)
| 파일 | 핵심 내용 |
|------|----------|
| `LLM_Viz_Dibia2023_LIDA_Auto_Visualization_LLM.pdf` | **LIDA** 4단계 파이프라인 (Microsoft, ACL 2023) — 핵심 레퍼런스 |
| `LLM_Viz_Maddigan2023_Chat2VIS_NL_Visualization.pdf` | Chat2VIS — 스키마만 전달하여 프라이버시 보존 (IEEE Access 2023) |
| `LLM_Viz_2024_Automated_DataVis_from_NL_via_LLM.pdf` | 테이블 직렬화 10가지 비교 + CoT/self-repair (ACM SIGMOD 2024) |
| `LLM_Viz_2024_WaitGPT_Conversational_LLM_DataAnalysis.pdf` | **WaitGPT** — LLM 코드를 플로우 다이어그램으로 (ACM UIST 2024) |
| `LLM_Viz_2024_Vis_Generation_LLM_Evaluation.pdf` | 6 LLM × 8 프롬프트 전략 비교. Self-Consistency 최우수 (IEEE TVCG) |
| `LLM_Viz_2024_NL4DV_LLM_Analytic_Specs.pdf` | NL4DV-LLM — 설명 가능한 분석 명세 생성 (IEEE VIS 2024) |
| `LLM_Viz_2025_VegaChat_LLM_Chart_Generation.pdf` | VegaChat — Vega-Lite 선언적 명세 + 자동 평가 (arXiv 2025) |
| `LLM_Viz_2025_DeepVIS_NL_DataVisualization.pdf` | DeepVIS — 5단계 CoT 추론으로 NL2VIS SOTA (IEEE VIS 2025) |

### Text-to-SQL (C)
| 파일 | 핵심 내용 |
|------|----------|
| `Text2SQL_2024_NL_Query_Engine_GenAI.pdf` | 벡터DB + 비즈니스 룰 통합 SQL 엔진 (IBM, arXiv 2024) |
| `Text2SQL_Survey_2024_LLM_Text_to_SQL.pdf` | ICL/Fine-tuning 패러다임 서베이 60+ 방법론 (arXiv 2024) |
| `Text2SQL_Survey_2024_Era_of_LLMs.pdf` | Text-to-SQL 전체 라이프사이클 서베이 (HKUST, arXiv 2024) |
| `Text2SQL_Review_2024_NL_to_SQL_LLM.pdf` | RAG/Graph RAG 통합 Text-to-SQL 리뷰 (arXiv 2024) |
| `Text2SQL_2024_DTS_SQL_Decomposed_Small_LLM.pdf` | **DTS-SQL** — 7B 모델로 GPT-4급 성능 (EMNLP 2024) — 프라이버시 보호 |

### 대시보드 구현 (D) + 반도체 (E)
| 파일 | 핵심 내용 |
|------|----------|
| `Dashboard_2021_Flask_vs_React_Live_Dashboard.pdf` | Flask vs React 정량 비교 — React 전면 우위 (TIJER 2021) |
| `Dashboard_2024_MERN_DataVis_Admin.pdf` | MERN 스택 대시보드 구현 (React+Nivo+MUI) (IJNRD 2024) |
| `Semiconductor_2022_Defect_Detection_Industry4.pdf` | 웨이퍼 결함 검출 + DSS (Industry 4.0) (Frontiers 2022) |

---

## 📎 다운로드 못한 유료 자료 (참고용 링크)

- **Simpson 2021 — CMOS Parametric Yield Dashboard** (IEEE, Micron) — https://ieeexplore.ieee.org/document/9435692/
- **Munzner — Visualization Analysis and Design** (CRC Press 교과서)
- **Stephen Few — Information Dashboard Design** (도서)
- **Yield prediction two-step ML** (Taylor & Francis) — https://www.tandfonline.com/doi/full/10.1080/00207543.2025.2601804
- **Parametric Wafer Map Visualization** — https://dl.acm.org/doi/abs/10.1109/38.773959
- **Industry 4.0 Visualization (Cell Patterns)** — https://www.cell.com/patterns/fulltext/S2666-3899(21)00092-1
