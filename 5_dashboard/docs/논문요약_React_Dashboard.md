# React 대시보드 & AI Agent 논문 요약

> Phase 4 (인터랙티브 대시보드 + 자연어 AI Agent) 구현을 위한 논문 모음.
> Streamlit 기반 기존 자료(`papers/`)와 별도로, React 프론트엔드 + LLM 기반 NL2VIS/Text-to-SQL 중심.

---

## 카테고리 A: 자연어 → 시각화 (NL2VIS) 서베이

### A-1. Towards Natural Language Interfaces for Data Visualization: A Survey
- **저자**: Leixian Shen, Enya Shen, Yuyu Luo, Xiaocong Yang 외
- **년도/학회**: 2022 / IEEE TVCG (VIS 2022)
- **핵심 주제**: V-NLI(시각화 지향 자연어 인터페이스) 20년 연구를 7단계 파이프라인으로 체계적 분류
- **주요 내용**:
  - 정보 시각화 파이프라인을 확장한 V-NLI 7단계 분류: Query Interpretation → Data Transformation → Visual Mapping → View Transformation → Human Interaction → Dialogue Management → Presentation
  - 57개 V-NLI 논문 + 283개 관련 논문 체계적 분석. 2018년 BERT 등장 이후 급증
  - 상용 도구(Tableau Ask Data, Power BI Q&A) vs 학술 시스템(Eviza, NL4DV, ncNet) 비교
  - 핵심 과제: 모호한 발화 처리, 대화 맥락 관리, 시각화 추천 알고리즘
- **프로젝트 적용**: AI Agent의 자연어 → 시각화 파이프라인 설계 레퍼런스. 7단계 구조를 우리 Agent 아키텍처의 뼈대로 활용
- **키워드**: V-NLI, Visualization Pipeline, Query Interpretation, Semantic Parsing, NL2VIS

### A-2. Natural Language Interfaces for Tabular Data Querying and Visualization: A Survey
- **저자**: Weixu Zhang, Yifei Wang, Yuanfeng Song 외
- **년도/학회**: 2024 / IEEE TKDE
- **핵심 주제**: Text-to-SQL과 Text-to-Vis를 semantic parsing으로 통합하는 LLM 시대 서베이
- **주요 내용**:
  - Text-to-SQL + Text-to-Vis를 공통 프레임워크(NL → 기능적 표현 → 실행)로 통합 분석
  - 접근법 4단계 진화: Rule-based → Seq2Seq → PLM(BERT/T5) → LLM(ChatGPT) + Prompt Engineering
  - Text-to-SQL은 성숙, Text-to-Vis는 초기 단계 — 모델·데이터셋 부족
  - 향후: LoRA 기반 경량 파인튜닝, 다국어 지원, adversarial robustness
- **프로젝트 적용**: AI Agent에서 SQL(데이터 질의)과 시각화 명세(차트 생성)를 통합 생성하는 아키텍처 설계 레퍼런스
- **키워드**: Text-to-SQL, Text-to-Vis, Semantic Parsing, LLM, Prompt Engineering, LoRA

### A-3. Chatbot-Based Natural Language Interfaces for Data Visualisation: A Scoping Review
- **저자**: Ecem Kavaz, Anna Puig, Inmaculada Rodriguez
- **년도/학회**: 2023 / Applied Sciences (MDPI)
- **핵심 주제**: 챗봇 기반 V-NLI 20개 시스템을 AINT 프레임워크로 분석
- **주요 내용**:
  - AINT 프레임워크 제안: A(Anthropomorphic) + I(Intelligence) + N(NLP) + T(Interactivity)
  - 20개 챗봇 V-NLI 시스템(Ava, Chat2Vis, DataBreeze, Iris 등) 상세 비교
  - 대부분 단순 테이블 + 기본 차트만 지원. follow-up 질의, 멀티모달 지원 부족
  - 대화형 가이드 전략(auto-complete, recommendation, follow-up)이 UX 핵심
- **프로젝트 적용**: 챗봇형 AI Agent 설계 시 AINT 체크리스트 활용. follow-up 질의 지원("이 wafer 불량 원인은?" → "position별로 나눠서 보여줘") 구현 참고
- **키워드**: Chatbot V-NLI, AINT Framework, Conversational Guidance, Follow-up Query

---

## 카테고리 B: LLM 기반 시각화 자동 생성

### B-1. LIDA: Automatic Generation of Grammar-Agnostic Visualizations using LLMs
- **저자**: Victor Dibia (Microsoft Research)
- **년도/학회**: 2023 / ACL Demo
- **핵심 주제**: 4단계 모듈 파이프라인(SUMMARIZER → GOAL EXPLORER → VISGENERATOR → INFOGRAPHER)으로 문법 비의존적 시각화 자동 생성
- **주요 내용**:
  - Matplotlib, Seaborn, Altair, Vega-Lite 등 다양한 문법으로 시각화 코드 생성
  - LLM 기반 자기 평가(6개 차원) + 자기 수정 메커니즘
  - 57개 데이터셋 벤치마크에서 시각화 오류율(VER) 3.5% 달성
  - SUMMARIZER가 대규모 데이터를 LLM이 이해 가능한 압축 요약으로 변환 — 핵심
- **프로젝트 적용**: 대규모 WT 데이터(1,087 피처)를 LLM에 전달할 때, SUMMARIZER 패턴으로 데이터 메타정보를 압축 요약하여 전달. 4단계 파이프라인을 Agent 아키텍처에 참고
- **키워드**: Multi-stage pipeline, Grammar-agnostic, Self-evaluation, Data summarization

### B-2. Chat2VIS: Generating Data Visualisations via Natural Language using ChatGPT, Codex and GPT-3
- **저자**: Paula Maddigan, Teo Susnjak
- **년도/학회**: 2023 / IEEE Access
- **핵심 주제**: 프롬프트 엔지니어링으로 LLM에 데이터 스키마만 전달하여 NL → Python 시각화 코드 직접 생성
- **주요 내용**:
  - 데이터 자체가 아닌 **스키마만 전송** → 보안/프라이버시 보존
  - Description Prompt(스키마, 컬럼 타입) + Code Prompt(import, df 설정) 조합
  - ChatGPT가 GPT-3, Codex 대비 가장 안정적. 오타/불완전 질의에도 합리적 추론
  - Streamlit 기반 웹 인터페이스 구현
- **프로젝트 적용**: SK Hynix 데이터 보안을 위해 실제 데이터 대신 스키마/메타데이터만 LLM에 전달하는 전략 적용. 비식별화 컬럼(X0~X1086)에 의미 있는 description 추가 시 품질 향상 기대
- **키워드**: Prompt engineering, Data privacy, End-to-end NL2VIS, Schema-only approach

### B-3. Automated Data Visualization from Natural Language via LLMs: An Exploratory Study
- **저자**: Yang Wu, Yao Wan, Hongyu Zhang 외
- **년도/학회**: 2024 / ACM SIGMOD
- **핵심 주제**: 테이블 데이터를 LLM 프롬프트로 변환하는 10가지 방법 비교 + 반복적 최적화 전략
- **주요 내용**:
  - 10가지 테이블 직렬화 방법 비교 — **Table2SQL(프로그래밍 형식)이 가장 효과적**
  - LLM이 기존 신경망 모델을 유의미하게 초과. Cross-domain에서도 강건
  - Chain-of-Thought, 역할 부여, self-repair 등 반복적 최적화 전략 검증
  - VQL 스케치(중간 표현)를 CoT로 활용 → 단계적 추론 유도
- **프로젝트 적용**: 1,087개 피처 데이터를 프롬프트에 효율적으로 포함하는 전략(스키마 기반, SQL 형식). 실패 시 CoT/self-repair 패턴은 Agent 오류 복구 메커니즘에 직접 적용
- **키워드**: Table serialization, In-context learning, Chain-of-Thought, Cross-domain, nvBench

### B-4. WaitGPT: Monitoring and Steering Conversational LLM Agent in Data Analysis
- **저자**: Liwenhan Xie, Chengbo Zheng, Haijun Xia 외
- **년도/학회**: 2024 / ACM UIST
- **핵심 주제**: LLM 생성 코드를 실시간 노드-링크 플로우 다이어그램으로 변환하여 분석 과정 모니터링/조정
- **주요 내용**:
  - LLM 코드의 문제점 식별: 불완전한 워크플로우, 잘못된 컬럼 선택, 부적절한 파라미터
  - 테이블/연산/결과 노드로 구성된 플로우 다이어그램 + 애니메이션 테이블 글리프
  - 3가지 모드: 실시간 모니터링, 회고적 검증, 세밀한 수정(파라미터 직접 편집)
  - 사용자 연구(N=12): 오류 탐지 성공률 향상, 특히 코딩 비전문가에게 효과적
- **프로젝트 적용**: AI Agent가 WT 데이터 전처리/집계 코드 생성 시, 각 단계의 데이터 변환을 플로우 다이어그램으로 시각화 → 엔지니어가 분석 논리 검증 가능
- **키워드**: Code visualization, Flow diagram, Real-time monitoring, Human-AI collaboration

### B-5. Visualization Generation with LLMs: An Evaluation
- **저자**: Xinyu Wang, Chenwei Liang 외
- **년도/학회**: 2024 / IEEE TVCG
- **핵심 주제**: 6개 오픈소스 LLM × 8가지 프롬프트 전략 체계적 비교 평가
- **주요 내용**:
  - QWEN2-7B, Llama3-8B 등 6개 LLM + Zero-shot, Few-shot, CoT, Self-Consistency 등 8전략 비교
  - **Self-Consistency 전략이 최우수** (legality 37.06%). Few-shot 2위 (25.83%)
  - 반직관적 발견: 더 많은 추론이 항상 나은 성능 X, 쉬운 차트가 항상 나은 결과 X
  - QWEN2, Llama3가 전반적으로 가장 우수
- **프로젝트 적용**: AI Agent의 시각화 생성에 Self-Consistency 전략(여러 답변 → 다수결) 적용. 프롬프트 복잡도 조절 시사점
- **키워드**: NL2VIS benchmark, Prompt strategy, Self-Consistency, Vega-Lite

### B-6. NL4DV-LLM: Generating Analytic Specifications from NL Queries using LLMs
- **저자**: Subham Sah, Rishab Mitra, Arpit Narechania 외
- **년도/학회**: 2024 / IEEE VIS
- **핵심 주제**: 분석 명세(감지된 속성 + 분석 태스크 + 시각화 추천)를 출력하여 설명 가능한 NL2VIS
- **주요 내용**:
  - NL4DV 툴킷에 GPT-4 기반 프롬프트 통합. 중간 분석 명세로 설명가능성 확보
  - 7가지 분석 태스크(Correlation, Distribution, Trend, Filter 등) 이론을 프롬프트에 명시적 주입
  - NLVCorpus 평가: NL4DV-LLM 87.02% vs 기존 NL4DV 64.05%
  - 토큰 제한 대응: 헤더 + 랜덤 10행 서브셋
- **프로젝트 적용**: "왜 이 차트를 선택했는지" 설명하는 Agent → 엔지니어 신뢰도 향상. 프롬프트에 반도체 WT 분석 태스크 정의 주입
- **키워드**: Analytic specification, Explainability, NL4DV, GPT-4

### B-7. VegaChat: A Robust Framework for LLM-Based Chart Generation and Assessment
- **저자**: Marko Hostnik, Rauf Kurbanov 외 (JetBrains)
- **년도/학회**: 2025 / arXiv
- **핵심 주제**: 선언적 Vega-Lite 명세 생성 + 표준화된 평가 메트릭(Spec Score, Vision Score)
- **주요 내용**:
  - 코드 생성 대신 **Vega-Lite JSON 명세** 생성 → 보안 위험 감소 + UI 위자드 통합 용이
  - Spec Score(결정론적 유사도) + Vision Score(멀티모달 LLM 이미지 유사도) 제안
  - 오류 수정 피드백 루프(최대 5회 재시도)로 Empty Chart Rate ≈ 0%
  - LIDA, CoML4VIS 대비 Vision Score 85.1% (NLV Corpus)
- **프로젝트 적용**: Vega-Lite 기반 선언적 방식은 React 프론트엔드와 통합에 유리. 자동 품질 평가 메커니즘으로 Agent 신뢰성 확보
- **키워드**: Vega-Lite, Declarative specification, Spec Score, Vision Score, Error correction loop

### B-8. DeepVIS: Bridging Natural Language and Data Visualization Through Step-wise Reasoning
- **저자**: Zhihao Shuai, Boyan Li 외
- **년도/학회**: 2025 / IEEE VIS
- **핵심 주제**: Chain-of-Thought 추론을 NL2VIS에 통합 — 5단계 CoT로 투명성+정확도 동시 향상
- **주요 내용**:
  - NL2VIS를 5단계 CoT로 분해: 차트 유형 결정 → 데이터 검색 → 세분화 → 정제 → 생성
  - nvBench-CoT 데이터셋 구축 (GPT-4o-mini로 품질 검증)
  - Llama3.1-8B fine-tuning으로 7개 베이스라인 초과 SOTA
  - CoT View에서 단계별 수정 가능 → 수정 효율성 부정 평가 9%(기존 38~50%)
- **프로젝트 적용**: 복잡한 WT 데이터 시각화 질의 해석 시 CoT 기반 단계적 추론으로 정확도 향상. 특정 단계만 수정 가능한 인터페이스는 실무 활용도 극대화
- **키워드**: Chain-of-Thought, nvBench-CoT, Step-wise refinement, Llama3.1 fine-tuning

---

## 카테고리 C: Text-to-SQL

### C-1. Natural Language Query Engine for Relational Databases using Generative AI
- **저자**: Steve Tueno (IBM France)
- **년도/학회**: 2024 / arXiv
- **핵심 주제**: 5단계 파이프라인(DB 스캔 → 비즈니스 룰 통합 → NL 처리 → SQL 생성/검증 → 반복 정제)
- **주요 내용**:
  - 벡터 DB(Milvus)에 스키마 + 비즈니스 룰을 임베딩으로 저장 → 의미적 유사도 검색
  - SQL 구문/의미 정확성 검증 + 자연어 응답 생성까지 포함
  - IBM watsonx.data + watsonx.ai(LLama 3, Mixtral) 프로토타입
  - BIRD Bench 50%+ 정확도, 비기술 사용자 90%+ "우수" 평가
- **프로젝트 적용**: 벡터 DB에 반도체 도메인 지식 저장 → SQL 검증 파이프라인 적용. AI Agent의 NL→SQL 아키텍처 핵심 레퍼런스
- **키워드**: Text-to-SQL, Vector Database, SQL Validation, Business Rules, watsonx

### C-2. Next-Generation Database Interfaces: A Survey of LLM-based Text-to-SQL
- **저자**: Zijin Hong, Zheng Yuan 외 (Hong Kong PolyU)
- **년도/학회**: 2024 / arXiv
- **핵심 주제**: ICL(In-context Learning)과 Fine-tuning 패러다임으로 LLM 기반 Text-to-SQL 최신 방법론 분류
- **주요 내용**:
  - ICL 4가지 전략: Decomposition(DIN-SQL), Prompt Optimization(DAIL-SQL), Reasoning Enhancement(CoT), Execution Refinement(Self-Debugging)
  - Fine-tuning: 아키텍처 개선, 데이터 증강, 멀티태스크 튜닝
  - 60개+ LLM 기반 방법론 시간순/카테고리별 비교
  - 주요 벤치마크(Spider, BIRD, WikiSQL) 및 평가 지표(EX, EM, VES) 정리
- **프로젝트 적용**: Decomposition + Execution Refinement 조합이 최효과적. Schema Linking 별도 모듈화 필수
- **키워드**: Text-to-SQL, ICL, Decomposition, Prompt Optimization, Schema Linking

### C-3. A Survey of Text-to-SQL in the Era of LLMs
- **저자**: Xinyu Liu, Shuyu Shen, Boyan Li 외 (HKUST, Tsinghua)
- **년도/학회**: 2024 / arXiv
- **핵심 주제**: Text-to-SQL 전체 라이프사이클(모델, 데이터, 평가, 오류 분석) 종합 서베이
- **주요 내용**:
  - 모듈형 설계: Pre-Processing(Schema Linking, DB Content Retrieval) → Translation → Post-Processing(Correction, Execution-Guided)
  - Schema Linking 3가지: String Matching, Neural Network, In-context Learning
  - 5단계 난이도: 토큰 인식 → 의미 이해 → 도메인 지식 → 멀티턴 → 실세계 적응
  - 비용 분석: DIN-SQL은 SQL당 3,579 토큰/10.34초 vs RESDSQL 1.91초
- **프로젝트 적용**: Schema Linking → SQL Translation → Execution-Guided Correction 모듈형 파이프라인 참고. PLM+LLM 결합으로 비용 절감
- **키워드**: Text-to-SQL Lifecycle, Modular Design, Schema Linking, Cost-Effective LLM

### C-4. From Natural Language to SQL: Review of LLM-based Text-to-SQL Systems
- **저자**: Ali Mohammadjafari, Anthony S. Maida, Raju Gottumukkala
- **년도/학회**: 2024 / arXiv
- **핵심 주제**: RAG(Retrieval Augmented Generation) 통합 Text-to-SQL의 효과 분석
- **주요 내용**:
  - RAG 2가지 방식: SQL 생성 강화(스키마/템플릿 검색) vs SQL 우회(직접 답변 검색)
  - **Graph RAG**가 기존 RAG 대비 스키마 이해/검색 정확도에서 우위
  - Chat2Data, CRUSH4SQL, FATO-SQL 등 RAG 기반 시스템 비교
  - 향후: 계산 효율성, 데이터 프라이버시, Human-in-the-Loop
- **프로젝트 적용**: DB 스키마와 도메인 지식을 벡터 DB에 저장하고 검색하는 RAG 기반 아키텍처 적용. Graph RAG로 복잡한 테이블 관계 처리
- **키워드**: RAG, Graph RAG, Text-to-SQL, Vector Database, Schema Linking

### C-5. DTS-SQL: Decomposed Text-to-SQL with Small Large Language Models
- **저자**: Mohammadreza Pourreza, Davood Rafiei (University of Alberta)
- **년도/학회**: 2024 / EMNLP Findings
- **핵심 주제**: Schema Linking + SQL Generation 2단계 분해로 7B 모델에서 GPT-4급 성능 달성
- **주요 내용**:
  - 기존: 테이블 식별 + SQL 생성 동시 학습 → 복잡도 높음. 이를 2개 독립 태스크로 분해
  - Stage 1 Schema Linking + Stage 2 SQL Generation 각각 fine-tuning
  - DeepSeek 7B로 Spider 84.4% EX (GPT-4 DAIL-SQL 86.6%에 근접)
  - Schema Linking 분리로 실행 정확도 3~7% 향상
- **프로젝트 적용**: **프라이버시 보호**(SK Hynix 데이터를 외부 API 미전송) + 비용 절감. 로컬 7B 모델에 DTS-SQL 적용 시 실용적 Text-to-SQL Agent 구현 가능
- **키워드**: Decomposed Fine-tuning, Schema Linking, Small LLM, DeepSeek 7B, Data Privacy

---

## 카테고리 D: 대시보드 구현 / 프레임워크

### D-1. Creating Live Dashboards for Data Visualization: Flask vs. React
- **저자**: Er. Pronoy Chopra
- **년도/학회**: 2021 / TIJER
- **핵심 주제**: 실시간 대시보드에서 Flask vs React 정량적 성능 비교
- **주요 내용**:
  - **React 우위**: 응답시간(80ms vs 150ms), 데이터 업데이트 지연(60ms vs 200ms), CPU(25% vs 40%), 메모리(100MB vs 150MB)
  - 실시간 갱신율: React 5Hz vs Flask 1Hz
  - UX: React가 네비게이션(4.5/5 vs 3.5/5), 반응성(4.7/5 vs 3.0/5), 만족도(4.6/5 vs 3.2/5) 모두 우수
  - 개발 시간: React 6일 vs Flask 9일
- **프로젝트 적용**: React 기반 대시보드 선택의 기술적 근거. Flask는 백엔드 API, React는 프론트엔드로 조합 최적
- **키워드**: Flask, React, Live Dashboard, Virtual DOM, Performance Comparison

### D-2. Data Visualization Admin Dashboard Using Full Stack MERN Web Application
- **저자**: R. Dhakshina Murthy, R.K. Shanmuga Priya
- **년도/학회**: 2024 / IJNRD
- **핵심 주제**: MERN 스택(MongoDB + Express + React + Node.js) 데이터 시각화 대시보드 설계/구현
- **주요 내용**:
  - MERN 각 컴포넌트 역할과 연동 방식 상세 설명
  - React + Material UI + Nivo Chart.js 시각화 구현
  - Redux Toolkit 상태 관리, Mongoose MongoDB 인터랙션
  - 구현 페이지: Admin Dashboard, Sales Overview, Daily/Monthly Charts, Category Breakdown
- **프로젝트 적용**: React + Nivo Chart + Material UI 기술 스택을 대시보드 프론트엔드에 직접 참고. 백엔드 DB는 예측 결과 저장용으로 교체
- **키워드**: MERN Stack, React.js, Nivo Chart, Material UI, Redux Toolkit

---

## 카테고리 E: 반도체 제조 / Industry 4.0

### E-1. Defect Detection on Optoelectronical Devices: A Real Industry 4.0 Case Study
- **저자**: George P. Moustris, George Kouzas 외
- **년도/학회**: 2022 / Frontiers in Manufacturing Technology
- **핵심 주제**: Industry 4.0 기반 광전자 소자 웨이퍼 레벨 결함 검출 및 의사결정 지원 시스템
- **주요 내용**:
  - 2단계 결함 검출: 저해상도 격자 이미지 → 고해상도 표면 스캔
  - 딥러닝 세그멘테이션(normal/dirt/defect 3분류) + 반자동 디바이스 매핑
  - Decision Support System: 결함 시각화/필터링 → 합격/불합격 분류
  - "Pass" 레이저가 "Fail" 대비 특정 주파수 레이징 확률 **6배** — zero-defect manufacturing 효과 입증
- **프로젝트 적용**: 웨이퍼 맵 기반 디바이스 매핑, 결함률 기반 합격/불합격 분류, DSS 인터랙티브 시각화/필터링이 우리 대시보드의 예측 결과 모니터링에 직접 적용 가능
- **키워드**: Industry 4.0, Zero-Defect Manufacturing, Wafer Defect Detection, DSS

---

## 다운로드 실패 / 수동 다운로드 필요 (3편)

| 논문 | 사유 | 링크 |
|------|------|------|
| Interactive Data Quality Dashboard (Gami et al., 2024) | IJCSE 서버 404 | [IJCSE](https://www.ijcseonline.org/pub_paper/5-IJCSE-09507.pdf) |
| Big Data Analytics for Smart Manufacturing (MDPI, 2017) | MDPI 봇 차단 | [MDPI](https://www.mdpi.com/2227-9717/5/3/39) |
| IoT Monitoring Semiconductor Manufacturing (SAGE, 2017) | Cloudflare 차단 | [SAGE](https://journals.sagepub.com/doi/10.1177/1550147717721810) |

---

## 미다운로드 유료/접근제한 논문 (참고용 링크)

| # | 논문 | 년도 | 링크 |
|---|------|------|------|
| 1 | Dashboard Design Patterns (Bach et al.) — IEEE VIS 2022 | 2022 | [IEEE](https://ieeexplore.ieee.org/document/9903550/) / [사이트](https://dashboarddesignpatterns.github.io/) |
| 2 | humanportal — A React.js Case Study — IEEE | 2020 | [IEEE](https://ieeexplore.ieee.org/document/9141070/) |
| 3 | Web Development Using ReactJS — IEEE | 2024 | [IEEE](https://ieeexplore.ieee.org/document/10541743/) |
| 4 | Performance Optimization Techniques for ReactJS — IEEE ICECCT | 2019 | [IEEE](https://ieeexplore.ieee.org/abstract/document/8869134/) |
| 5 | Methods of Improving and Optimizing React — IEEE | 2021 | [IEEE](https://ieeexplore.ieee.org/document/9596762/) |
| 6 | React Admin Dashboard Review — IJRASET | 2024 | [IJRASET](https://www.ijraset.com/research-paper/react-admin-dashboard) |
| 7 | Integrating D3.js with React (Elrom) — Springer 단행본 | 2021 | [Springer](https://link.springer.com/book/10.1007/978-1-4842-7052-3) |
| 8 | Parametric Wafer Map Visualization (Lin) — IEEE CG&A | 1999 | [IEEE](https://ieeexplore.ieee.org/document/773959) |
| 9 | Semiconductor Process Visualization — IEEE | 2003 | [IEEE](https://ieeexplore.ieee.org/document/1245765/) |
| 10 | Big Data Vis for Semiconductor Manufacturing | 2017 | [ResearchGate](https://www.researchgate.net/publication/317039175) |
| 11 | Monitoring & Control of Semiconductor Manufacturing — IEEE TSM | 1998 | [IEEE](https://ieeexplore.ieee.org/document/736011/) |
| 12 | Survey on Semiconductor Wafer Yield Prediction by AI | 2025 | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1879239125004084) |
| 13 | Wafer Map Yield Prediction Based on ML — IEEE | 2019 | [IEEE](https://ieeexplore.ieee.org/document/8856232/) |
| 14 | Yield Prediction via Spatial Modeling (ZI-Poisson) | 2007 | [Taylor&Francis](https://www.tandfonline.com/doi/full/10.1080/07408170701275335) |
| 15 | Mobile Manufacturing Dashboard — IEEE PerCom | 2014 | [IEEE](https://ieeexplore.ieee.org/document/6815180/) |
| 16 | Digitalization Platform for Quality Management — Springer | 2023 | [Springer](https://link.springer.com/article/10.1007/s10845-023-02162-9) |
| 17 | Manufacturing Dashboards on KPI Survey | 2017 | [ResearchGate](https://www.researchgate.net/publication/312076187) |
| 18 | Learning Analytics Dashboard — Springer | 2022 | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8853217/) |
| 19 | ML/DL Predictive Quality in Manufacturing — Springer | 2022 | [Springer](https://link.springer.com/article/10.1007/s10845-022-01963-8) |
| 20 | ML-Based Defect Classification in Wafers — Springer | 2024 | [Springer](https://link.springer.com/article/10.1007/s10845-024-02521-0) |
| 21 | VisEval: Benchmark for DataVis — IEEE TVCG | 2024 | [ACM](https://dl.acm.org/doi/10.1109/TVCG.2024.3456320) |
| 22 | VL2NL: NL Dataset Gen for Vis — CHI 2024 | 2024 | [ACM](https://dl.acm.org/doi/10.1145/3613904.3642943) |
| 23 | nvAgent: Collaborative Agent for NL2VIS | 2025 | [OpenReview](https://openreview.net/forum?id=KIekYDSA3F) |
| 24 | Survey on Employing LLMs for Text-to-SQL — ACM CS | 2025 | [ACM](https://dl.acm.org/doi/10.1145/3737873) |
