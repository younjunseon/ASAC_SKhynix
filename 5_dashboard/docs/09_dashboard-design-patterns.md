# Dashboard Design Patterns (Bach et al., IEEE VIS 2022)

> 42개 대시보드 디자인 패턴 카탈로그. Phase 4 대시보드 설계 시 페이지/컴포넌트 선택의 체크리스트로 사용.

- **공식 사이트**: https://dashboarddesignpatterns.github.io/
- **논문**: Bach, Freeman, Abdul-Rahman, Turkay, Khan, Nguyen, Fan, Chen — IEEE TVCG 2023 (VIS 2022)
- **arXiv**: https://arxiv.org/abs/2205.00757
- **세부 페이지**: `/patterns.html`, `/types.html`, `/tradeoffs.html`, `/processguidelines.html`

---

## Component Design Patterns

### Data Information

| Pattern | Purpose |
|---------|---------|
| Individual Values | Highlight specific data points, emphasizing importance of single values |
| Derived Values | Present abstracted information like KPIs calculated from raw data |
| Filtered Data | Display a subset of the original dataset based on selection criteria |
| Thresholds | Make explicit judgments about data using objective criteria |
| Aggregated Data | Combine multiple data points into concise summary values |
| Detailed Datasets | Offer complete presentations showing raw data with minimal interpretation |

### Meta Information

| Pattern | Purpose |
|---------|---------|
| Data Source | Identify origin of data with links and collection methodology |
| Disclaimer | Explain assumptions, limitations, and processing decisions |
| Data Description | Provide high-level summary of what the dashboard displays |
| Update Information | Indicate when data was last refreshed |
| Annotations | Highlight specific points or developments with graphical embellishments |

### Visual Representations

| Pattern | Purpose |
|---------|---------|
| Numbers | Display individual key values prominently with units or proportions |
| Trend Arrows | Show directional change in data values using up/down indicators |
| Pictograms | Use abstract symbols to illustrate concepts or designate data types |
| Gauges & Progress Bars | Represent values on a scale or range with visual indicators |
| Signature Charts | Provide quick trend understanding without detailed axis information |
| Detailed Charts | Enable precise value reading with axes, ticks, and labels |
| Tables | Organize raw data in tabular format, optionally with charts and arrows |
| Text Lists | Present non-quantitative information in ticker or list format |

### Interactions

| Pattern | Purpose |
|---------|---------|
| Exploration | Allow users to investigate data relationships through brushing and tooltips |
| Navigation | Direct users through information facets via tabs, buttons, or links |
| Personalization | Enable users to reconfigure dashboard content and layout |
| Filter & Focus | Help users locate specific data through search and filter controls |

---

## Composition Design Patterns

### Screenspace

| Pattern | Purpose |
|---------|---------|
| Screenfit | Display all information visible at once without scrolling or interaction |
| Overflow | Allow dashboard to extend beyond screen boundaries with scrolling |
| Detail-on-Demand | Reduce visible information while revealing details on user request |
| Parameterized | Let users control what content displays through filter controls |
| Multiple Page | Distribute content across several pages with navigation between them |

### Structure

| Pattern | Purpose |
|---------|---------|
| Single Page | Present all information on one consolidated view |
| Hierarchical | Organize pages in levels supporting drill-down to greater detail |
| Parallel | Arrange multiple pages as equal-level facets without hierarchy |
| Open | Structure pages with flexible relationships between information |
| Semantic | Organize pages reflecting operational structure and data semantics |

### Page Layout

| Pattern | Purpose |
|---------|---------|
| Open Layouts | Arrange widgets of varying sizes without rigid alignment rules |
| Table Layouts | Align widgets in rows and columns representing data facets |
| Stratified Layouts | Emphasize top-down ordering from general to detailed information |
| Grouped Layouts | Visibly cluster related widgets using dividers and backgrounds |
| Schematic Layouts | Position widgets informed by external properties like geography |

### Color

| Pattern | Purpose |
|---------|---------|
| Distinct | Use unique colors for different widgets or data types |
| Shared | Apply consistent palette across components for branding coherence |
| Data Encoding | Use color to represent categories or scales within datasets |
| Semantic | Map colors to real-world outcomes with meaningful connotations |
| Emotive | Leverage color aesthetically to develop emotional responses |

---

## RCC 대시보드 적용 가이드

| design.md 섹션 | 추천 패턴 |
|---|---|
| §4.1 메인 대시보드 | Derived Values(KPI), Numbers, Trend Arrows, Stratified Layout, Detail-on-Demand |
| §4.2 Unit 상세 | Filtered Data, Detailed Charts, Annotations, Hierarchical |
| §4.3 Wafer Map | Schematic Layouts(좌표 기반), Data Encoding(color scale), Exploration(brush/tooltip) |
| §4.4 모델 성능 | Thresholds(목표 RMSE 라인), Aggregated Data, Signature Charts |
| §4.5 AI Agent | Personalization, Navigation, Open Structure |
| 전체 톤 | Shared color + Semantic color (불량=red, 정상=green) |