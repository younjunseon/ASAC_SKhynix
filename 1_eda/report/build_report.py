"""
EDA 리포트 빌드 스크립트
eda.ipynb에서 markdown 인사이트 + 텍스트 출력 + 차트 이미지를 추출하여
예쁜 HTML 리포트를 생성한다.
"""
import json, base64, os, re, html

NOTEBOOK = os.path.join(os.path.dirname(__file__), '..', '01_raw_eda', 'eda.ipynb')
IMG_DIR = os.path.join(os.path.dirname(__file__), 'images')
OUTPUT = os.path.join(os.path.dirname(__file__), 'eda_report.html')
OUTPUT_STANDALONE = os.path.join(os.path.dirname(__file__), 'eda_report_standalone.html')

# 이미지 src 매핑: 파일 기반 vs 인라인(base64)
INLINE_IMAGES = {}  # fname -> base64 string

with open(NOTEBOOK, 'r', encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']

# ── Phase/Section 매핑 ──
PHASES = {
    1: {"title": "데이터 기본 파악", "icon": "📊", "sections": [3,4]},
    2: {"title": "Feature 품질 검사", "icon": "🔍", "sections": [5,6,7]},
    3: {"title": "이상치 및 스케일", "icon": "⚠️", "sections": [8,9]},
    4: {"title": "Feature-Target 관계", "icon": "🔗", "sections": [10,11,12,13,14]},
    5: {"title": "Target 심층 분석", "icon": "🎯", "sections": [15,16,17]},
    6: {"title": "공간/구조 분석", "icon": "🗺️", "sections": [18,19,20,21,22,23,24,25]},
    7: {"title": "Die→Unit 집계 전략", "icon": "📐", "sections": [26]},
}

# ── section 번호 → 제목 매핑 (markdown 셀에서 추출) ──
SECTION_TITLES = {}
SECTION_INSIGHTS = {}
SECTION_CONTENT = []  # (section_num, content_html) 리스트

def get_section_num(text):
    """## 3. 데이터 구조 확인 → 3"""
    m = re.match(r'##\s+(\d+)[\.\-]?\s*(.*)', text.strip().split('\n')[0])
    if m:
        return int(m.group(1)), m.group(2).strip()
    # ## 17-1 같은 형태
    m = re.match(r'##\s+(\d+)-(\d+)[\.\s]*(.*)', text.strip().split('\n')[0])
    if m:
        return float(f"{m.group(1)}.{m.group(2)}"), m.group(3).strip()
    return None, None

def md_to_html(text):
    """간단한 마크다운→HTML 변환"""
    lines = text.strip().split('\n')
    result = []
    in_table = False
    table_rows = []

    def flush_table():
        nonlocal in_table, table_rows
        if table_rows:
            html_str = '<div class="table-wrapper"><table>'
            for ri, row in enumerate(table_rows):
                tag = 'th' if ri == 0 else 'td'
                cols = [c.strip() for c in row.split('|')]
                cols = [c for c in cols if c]  # 빈 문자열 제거
                html_str += f'<tr>{"".join(f"<{tag}>{c}</{tag}>" for c in cols)}</tr>'
            html_str += '</table></div>'
            result.append(html_str)
        in_table = False
        table_rows = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('---'):
            flush_table()
            continue
        if stripped.startswith('|') and '|' in stripped[1:]:
            if re.match(r'^\|[\s\-:|]+\|$', stripped):
                continue  # separator row
            if not in_table:
                in_table = True
                table_rows = []
            table_rows.append(stripped)
            continue
        else:
            flush_table()

        if stripped.startswith('> '):
            result.append(f'<div class="insight-box">{html.escape(stripped[2:])}</div>')
        elif stripped.startswith('- '):
            result.append(f'<li>{html.escape(stripped[2:])}</li>')
        elif stripped.startswith('# ') and not stripped.startswith('## '):
            pass  # Phase 제목은 별도 처리
        elif stripped.startswith('## '):
            pass  # Section 제목은 별도 처리
        elif stripped:
            # 볼드 처리
            s = html.escape(stripped)
            s = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', s)
            s = re.sub(r'`(.*?)`', r'<code>\1</code>', s)
            result.append(f'<p>{s}</p>')

    flush_table()
    return '\n'.join(result)

def text_output_to_html(text):
    """텍스트 출력을 HTML pre 블록으로"""
    text = text.strip()
    if not text or text.startswith('<Figure') or text.startswith('Font '):
        return ''
    # DataFrame 출력이나 로그 출력
    lines = text.split('\n')
    # 너무 긴 출력은 줄임
    if len(lines) > 40:
        lines = lines[:35] + [f'  ... (총 {len(lines)}줄)']
    return f'<pre class="output">{html.escape(chr(10).join(lines))}</pre>'

# ── 셀 파싱 ──
current_section = None
current_section_num = None
section_contents = {}  # section_num → html parts list

for i, cell in enumerate(cells):
    ct = cell['cell_type']
    src = ''.join(cell['source'])

    if ct == 'markdown':
        snum, stitle = get_section_num(src)
        if snum is not None:
            current_section_num = snum
            SECTION_TITLES[snum] = stitle
            if snum not in section_contents:
                section_contents[snum] = []
            continue

        # 인사이트 (> 로 시작하는 blockquote)
        if current_section_num is not None:
            converted = md_to_html(src)
            if converted.strip():
                if current_section_num not in section_contents:
                    section_contents[current_section_num] = []
                section_contents[current_section_num].append(converted)

                # 첫 번째 인사이트를 요약으로 저장
                for line in src.split('\n'):
                    line = line.strip()
                    if line.startswith('> ') and current_section_num not in SECTION_INSIGHTS:
                        SECTION_INSIGHTS[current_section_num] = line[2:]

    elif ct == 'code':
        if current_section_num is None:
            continue
        if current_section_num not in section_contents:
            section_contents[current_section_num] = []

        # 코드 설명 (주석)
        first_line = src.strip().split('\n')[0] if src.strip() else ''
        if first_line.startswith('#'):
            desc = first_line.lstrip('#').strip()
            section_contents[current_section_num].append(
                f'<div class="analysis-label">{html.escape(desc)}</div>'
            )

        # 출력 처리
        for j, out in enumerate(cell.get('outputs', [])):
            # 이미지
            if 'data' in out and 'image/png' in out['data']:
                fname = f'cell{i:03d}_out{j}.png'
                fpath = os.path.join(IMG_DIR, fname)
                # base64 데이터를 항상 메모리에 저장 (standalone 빌드용)
                b64 = out['data']['image/png']
                if isinstance(b64, list):
                    b64 = ''.join(b64)
                INLINE_IMAGES[fname] = b64.strip()
                # 파일이 없으면 base64에서 디코딩하여 저장 (이미지 자동 동기화)
                if not os.path.exists(fpath):
                    try:
                        os.makedirs(IMG_DIR, exist_ok=True)
                        with open(fpath, 'wb') as imf:
                            imf.write(base64.b64decode(INLINE_IMAGES[fname]))
                    except Exception as e:
                        print(f'  [warn] failed to write {fname}: {e}')
                if os.path.exists(fpath):
                    # 파일 기반 src 사용 (standalone 빌드 시 후처리에서 치환)
                    section_contents[current_section_num].append(
                        f'<div class="chart"><img src="images/{fname}" alt="{fname}"></div>'
                    )
            # 텍스트 출력
            else:
                text = ''
                if out.get('output_type') == 'stream':
                    text = ''.join(out.get('text', []))
                elif out.get('output_type') in ('execute_result', 'display_data'):
                    if 'text/plain' in out.get('data', {}):
                        text = ''.join(out['data']['text/plain'])
                txt_html = text_output_to_html(text)
                if txt_html:
                    section_contents[current_section_num].append(txt_html)


# ── HTML 생성 ──
KEY_FINDINGS = [
    ("Zero-Inflated Target", "Y=0이 70.8%인 극단적 편향 분포. Two-Stage 모델(분류→회귀) 필수"),
    ("단일 Feature 무력", "max |r| = 0.037. 어떤 단일 feature도 target을 예측할 수 없음"),
    ("CV 집계가 핵심", "Die→Unit 변동계수(CV) max|r|=0.147로 mean(0.037)의 4배. 핵심 집계 통계"),
    ("비선형 관계 존재", "Spearman(0.087)이 Pearson(0.037)의 2.3배. MI가 별도 feature 발굴"),
    ("Feature Interaction 유효", "전체 조합 MI 분석: 단일 max 0.0314 → interaction max 0.036. 430,970개(29.3%)가 단일보다 높음"),
    ("Lot 효과 강력", "lot별 불량률 10~45% 차이. 정규화는 역효과 → lot 통계를 feature로 활용"),
    ("공간 패턴 제한적", "radial distance, zone, NNR 잔차 모두 무효. die_x/y의 집계 통계만 활용"),
    ("Y=0 내부 이질성", "K=2 클러스터링 → 72%가 Y>0과 유사 프로파일. 잠재 불량 후보"),
]

html_parts = []
html_parts.append(f'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EDA Report — SK Hynix Wafer Test RCC 예측</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {{
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #232736;
    --border: #2d3248;
    --text: #e4e6f0;
    --text-muted: #8b8fa8;
    --accent: #6366f1;
    --accent-light: #818cf8;
    --accent-bg: rgba(99,102,241,0.08);
    --green: #22c55e;
    --orange: #f59e0b;
    --red: #ef4444;
    --cyan: #06b6d4;
}}

* {{ margin: 0; padding: 0; box-sizing: border-box; }}

body {{
    font-family: 'Noto Sans KR', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.7;
    font-size: 15px;
}}

/* ── 레이아웃 ── */
.container {{ max-width: 1200px; margin: 0 auto; padding: 0 32px; }}

/* ── 헤더 ── */
header {{
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e1b4b 100%);
    padding: 64px 0 48px;
    border-bottom: 1px solid var(--border);
    position: relative;
    overflow: hidden;
}}
header::before {{
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at 30% 50%, rgba(99,102,241,0.15), transparent 70%);
}}
header .container {{ position: relative; z-index: 1; }}
header h1 {{
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 8px;
}}
header .subtitle {{
    color: var(--accent-light);
    font-size: 1.05rem;
    font-weight: 400;
    margin-bottom: 24px;
}}
header .meta {{
    display: flex; gap: 24px;
    color: var(--text-muted);
    font-size: 0.85rem;
}}
header .meta span {{ display: flex; align-items: center; gap: 6px; }}

/* ── TOC ── */
.toc {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 32px;
    margin: 40px 0;
}}
.toc h2 {{
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 20px;
    color: var(--accent-light);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
.toc-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
    gap: 16px;
}}
.toc-phase {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 20px;
    transition: border-color 0.2s;
}}
.toc-phase:hover {{ border-color: var(--accent); }}
.toc-phase h3 {{
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 8px;
    color: var(--text);
}}
.toc-phase h3 .phase-icon {{ margin-right: 8px; }}
.toc-phase ul {{ list-style: none; padding-left: 28px; }}
.toc-phase li {{
    font-size: 0.85rem;
    color: var(--text-muted);
    padding: 2px 0;
}}
.toc-phase li a {{
    color: var(--text-muted);
    text-decoration: none;
    transition: color 0.2s;
}}
.toc-phase li a:hover {{ color: var(--accent-light); }}

/* ── Key Findings ── */
.key-findings {{
    margin: 40px 0;
}}
.key-findings h2 {{
    font-size: 1.3rem;
    font-weight: 600;
    margin-bottom: 20px;
}}
.findings-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 12px;
}}
.finding-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 16px 20px;
}}
.finding-card h4 {{
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--accent-light);
    margin-bottom: 6px;
}}
.finding-card p {{
    font-size: 0.82rem;
    color: var(--text-muted);
    line-height: 1.5;
}}

/* ── Phase Section ── */
.phase-section {{
    margin: 56px 0;
}}
.phase-header {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 32px;
    padding-bottom: 12px;
    border-bottom: 2px solid var(--accent);
}}
.phase-header .icon {{ font-size: 1.5rem; }}
.phase-header h2 {{
    font-size: 1.4rem;
    font-weight: 700;
}}
.phase-header .phase-num {{
    background: var(--accent);
    color: white;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 20px;
    margin-left: auto;
}}

/* ── Section ── */
.section {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 24px;
    overflow: hidden;
}}
.section-header {{
    padding: 20px 24px;
    border-bottom: 1px solid var(--border);
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 12px;
    transition: background 0.2s;
}}
.section-header:hover {{ background: var(--surface2); }}
.section-header h3 {{
    font-size: 1.05rem;
    font-weight: 600;
    flex: 1;
}}
.section-header .section-num {{
    background: var(--accent-bg);
    color: var(--accent-light);
    font-size: 0.75rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 4px;
    min-width: 28px;
    text-align: center;
}}
.section-header .toggle {{
    color: var(--text-muted);
    transition: transform 0.3s;
    font-size: 1.2rem;
}}
.section.open .section-header .toggle {{ transform: rotate(180deg); }}

.section-insight {{
    padding: 12px 24px;
    background: var(--accent-bg);
    border-bottom: 1px solid var(--border);
    font-size: 0.88rem;
    color: var(--accent-light);
    line-height: 1.6;
}}

.section-body {{
    padding: 24px;
    display: none;
}}
.section.open .section-body {{ display: block; }}

.section-footer {{
    text-align: center;
    padding: 12px;
    margin-top: 20px;
    border-top: 1px solid var(--border);
    color: var(--text-muted);
    font-size: 0.82rem;
    cursor: pointer;
    transition: color 0.2s, background 0.2s;
    border-radius: 0 0 12px 12px;
}}
.section-footer:hover {{
    color: var(--accent-light);
    background: var(--accent-bg);
}}

/* ── 콘텐츠 스타일 ── */
.insight-box {{
    background: rgba(34,197,94,0.06);
    border-left: 3px solid var(--green);
    padding: 10px 16px;
    margin: 12px 0;
    border-radius: 0 6px 6px 0;
    font-size: 0.9rem;
    color: var(--green);
}}
.analysis-label {{
    font-size: 0.82rem;
    color: var(--cyan);
    font-weight: 500;
    margin: 20px 0 8px;
    padding-left: 12px;
    border-left: 2px solid var(--cyan);
}}
.chart {{
    margin: 16px 0;
    text-align: center;
}}
.chart img {{
    max-width: 100%;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: white;
}}
pre.output {{
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    line-height: 1.6;
    overflow-x: auto;
    color: var(--text-muted);
    margin: 12px 0;
}}
.table-wrapper {{ overflow-x: auto; margin: 12px 0; }}
table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
}}
th, td {{
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1px solid var(--border);
}}
th {{
    background: var(--surface2);
    font-weight: 600;
    color: var(--accent-light);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}}
td {{ color: var(--text-muted); }}
li {{
    margin: 4px 0;
    padding-left: 4px;
    font-size: 0.88rem;
    color: var(--text-muted);
    list-style: disc;
    margin-left: 20px;
}}
p {{
    margin: 8px 0;
    font-size: 0.88rem;
    color: var(--text-muted);
}}
strong {{ color: var(--text); font-weight: 600; }}
code {{
    background: var(--surface2);
    padding: 1px 6px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
}}

/* ── Footer ── */
footer {{
    text-align: center;
    padding: 48px 0;
    color: var(--text-muted);
    font-size: 0.8rem;
    border-top: 1px solid var(--border);
    margin-top: 64px;
}}

/* ── 반응형 ── */
@media (max-width: 768px) {{
    .container {{ padding: 0 16px; }}
    header {{ padding: 40px 0 32px; }}
    header h1 {{ font-size: 1.6rem; }}
    .toc-grid {{ grid-template-columns: 1fr; }}
    .findings-grid {{ grid-template-columns: 1fr; }}
}}

/* ── 스크롤바 ── */
::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-track {{ background: var(--bg); }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--text-muted); }}
</style>
</head>
<body>

<!-- ── HEADER ── -->
<header>
<div class="container">
    <h1>EDA Report</h1>
    <div class="subtitle">SK Hynix — Wafer Test 기반 Field Health Data(RCC) 예측</div>
    <div class="meta">
        <span>📅 2026-04-06</span>
        <span>📊 7 Phase · 20개 모듈 · 26개 분석 항목</span>
        <span>📐 174,980 dies · 43,745 units · 1,087 features</span>
    </div>
</div>
</header>

<div class="container">

<!-- ── KEY FINDINGS ── -->
<div class="key-findings">
<h2>🔑 핵심 발견</h2>
<div class="findings-grid">
''')

for title, desc in KEY_FINDINGS:
    html_parts.append(f'''<div class="finding-card">
<h4>{html.escape(title)}</h4>
<p>{html.escape(desc)}</p>
</div>''')

html_parts.append('</div></div>')

# ── TOC ──
html_parts.append('''
<div class="toc">
<h2>목차</h2>
<div class="toc-grid">
''')

for pnum, pinfo in PHASES.items():
    html_parts.append(f'''<div class="toc-phase">
<h3><span class="phase-icon">{pinfo["icon"]}</span>Phase {pnum}: {pinfo["title"]}</h3>
<ul>''')
    for snum in pinfo["sections"]:
        # sub-sections (17-1 = 17.1 등)
        matching = [s for s in SECTION_TITLES if (int(s) == snum if isinstance(s, float) else s == snum)]
        for s in sorted(matching):
            title = SECTION_TITLES.get(s, '')
            display_num = str(s).replace('.', '-') if isinstance(s, float) else str(s)
            html_parts.append(f'<li><a href="#section-{display_num}">{display_num}. {html.escape(title)}</a></li>')
    html_parts.append('</ul></div>')

html_parts.append('</div></div>')

# ── PHASE별 SECTIONS ──
for pnum, pinfo in PHASES.items():
    html_parts.append(f'''
<div class="phase-section">
<div class="phase-header">
    <span class="icon">{pinfo["icon"]}</span>
    <h2>{pinfo["title"]}</h2>
    <span class="phase-num">PHASE {pnum}</span>
</div>
''')

    all_section_nums = sorted(section_contents.keys())
    for snum in all_section_nums:
        # 이 phase에 속하는지 확인
        base = int(snum)
        if base not in pinfo["sections"]:
            continue

        title = SECTION_TITLES.get(snum, f'Section {snum}')
        display_num = str(snum).replace('.', '-') if isinstance(snum, float) else str(snum)
        insight = SECTION_INSIGHTS.get(snum, '')

        html_parts.append(f'''
<div class="section" id="section-{display_num}">
    <div class="section-header" onclick="this.parentElement.classList.toggle('open')">
        <span class="section-num">{display_num}</span>
        <h3>{html.escape(title)}</h3>
        <span class="toggle">▼</span>
    </div>''')

        if insight:
            html_parts.append(f'<div class="section-insight">💡 {html.escape(insight)}</div>')

        html_parts.append('<div class="section-body">')
        for content in section_contents[snum]:
            html_parts.append(content)
        html_parts.append(f'''<div class="section-footer" onclick="this.parentElement.parentElement.classList.remove('open'); document.getElementById('section-{display_num}').scrollIntoView({{behavior:'smooth'}})">
    <span>▲ 접기</span>
</div>''')
        html_parts.append('</div></div>')

    html_parts.append('</div>')

# ── FOOTER ──
html_parts.append('''
</div>

<footer>
    <p>Generated from eda.ipynb — 논문 30편 기반 체계적 분석</p>
    <p>SK Hynix × 기업연계프로젝트 EDA Report</p>
</footer>

<script>
// 기본적으로 모든 섹션 닫힘. 클릭으로 열기/닫기
document.querySelectorAll('.section-header').forEach(h => {
    h.addEventListener('click', () => {
        // toggle은 onclick으로 처리됨
    });
});
</script>

</body>
</html>''')

html_full = '\n'.join(html_parts)

# 파일 기반 HTML
with open(OUTPUT, 'w', encoding='utf-8') as f:
    f.write(html_full)

# Standalone HTML: <img src="images/cellXXX_outY.png"> → data URI 로 치환
def to_inline(match):
    fname = match.group(1)
    b64 = INLINE_IMAGES.get(fname)
    if b64 is None:
        # fallback: 파일에서 읽어 base64 인코딩
        fpath = os.path.join(IMG_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath, 'rb') as imf:
                b64 = base64.b64encode(imf.read()).decode('ascii')
            INLINE_IMAGES[fname] = b64
        else:
            return match.group(0)
    return f'<img src="data:image/png;base64,{b64}" alt="{fname}">'

html_standalone = re.sub(r'<img src="images/([^"]+)" alt="[^"]*">', to_inline, html_full)
with open(OUTPUT_STANDALONE, 'w', encoding='utf-8') as f:
    f.write(html_standalone)

import sys; sys.stdout.reconfigure(encoding="utf-8")
print(f"Report:            {OUTPUT}")
print(f"Standalone Report: {OUTPUT_STANDALONE}")
print(f"   섹션 수: {len(section_contents)}")
print(f"   이미지 수: {sum(1 for parts in section_contents.values() for p in parts if 'img src' in p)}")
print(f"   인라인된 이미지 수: {sum(1 for _ in re.finditer(r'data:image/png;base64', html_standalone))}")
