"""
EDA 리포트 빌드 — 전처리 전 (raw)

01_raw_eda/eda.ipynb 에서 markdown 인사이트 + 텍스트 출력 + 차트 이미지를 추출하여
다크 테마 HTML 리포트(eda_raw_report.html + standalone)를 생성한다.
공용 디자인은 report_theme.py 를 사용 (전처리 후 리포트와 통일).
"""
import json, base64, os, re, html, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report_theme as rt

NOTEBOOK = os.path.join(os.path.dirname(__file__), '..', '01_raw_eda', 'eda.ipynb')
IMG_DIR = os.path.join(os.path.dirname(__file__), 'images')
OUTPUT = os.path.join(os.path.dirname(__file__), 'eda_raw_report.html')
OUTPUT_STANDALONE = os.path.join(os.path.dirname(__file__), 'eda_raw_report_standalone.html')

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


# ── HTML 생성 (공용 테마 report_theme, 다크 팔레트) ──
KEY_FINDINGS = [
    ("Zero-Inflated Target", "Y=0이 70.8%인 극단적 편향 분포. Two-Stage 모델(분류→회귀) 필수"),
    ("단일 Feature 무력", "max |r| = 0.037. 어떤 단일 feature도 target을 예측할 수 없음"),
    ("CV 집계가 핵심", "Die→Unit 변동계수(CV) max|r|=0.147로 mean(0.037)의 4배. 핵심 집계 통계"),
    ("비선형 관계 존재", "Spearman(0.087)이 Pearson(0.037)의 2.3배. MI가 별도 feature 발굴"),
    ("Feature Interaction 유효", "전체 조합 MI 분석: 단일 max 0.019 → interaction max 0.026. 493,253개(33.6%)가 단일보다 높음"),
    ("Lot 효과 강력", "lot별 불량률 10~45% 차이. 정규화는 역효과 → lot 통계를 feature로 활용"),
    ("공간 패턴 제한적", "radial distance, zone, NNR 잔차 모두 무효. die_x/y의 집계 통계만 활용"),
    ("Y=0 내부 이질성", "K=2 클러스터링 → 72%가 Y>0과 유사 프로파일. 잠재 불량 후보"),
]

findings_html = ''.join(
    rt.finding_card(html.escape(t), html.escape(d)) for t, d in KEY_FINDINGS)

# TOC
toc_parts = []
for pnum, pinfo in PHASES.items():
    items = []
    for snum in pinfo["sections"]:
        matching = [s for s in SECTION_TITLES
                    if (int(s) == snum if isinstance(s, float) else s == snum)]
        for s in sorted(matching):
            title = SECTION_TITLES.get(s, '')
            display_num = str(s).replace('.', '-') if isinstance(s, float) else str(s)
            items.append((f'section-{display_num}', f'{display_num}. {html.escape(title)}'))
    toc_parts.append(rt.toc_phase(pinfo["icon"], f'Phase {pnum}: {pinfo["title"]}', items))
toc_html = ''.join(toc_parts)

# Phase별 Section
phase_parts = []
for pnum, pinfo in PHASES.items():
    sec_html = []
    for snum in sorted(section_contents.keys()):
        if int(snum) not in pinfo["sections"]:
            continue
        title = SECTION_TITLES.get(snum, f'Section {snum}')
        display_num = str(snum).replace('.', '-') if isinstance(snum, float) else str(snum)
        insight = SECTION_INSIGHTS.get(snum, '')
        body = '\n'.join(section_contents[snum])
        sec_html.append(rt.section(
            f'section-{display_num}', display_num, html.escape(title),
            body, html.escape(insight) if insight else ''))
    phase_parts.append(rt.phase(pinfo["icon"], pinfo["title"], f'PHASE {pnum}', '\n'.join(sec_html)))
body_html = '\n'.join(phase_parts)

meta_html = ('<span>📅 2026-04-06</span>'
             '<span>📊 7 Phase · 20개 모듈 · 26개 분석 항목</span>'
             '<span>📐 174,980 dies · 43,745 units · 1,087 features</span>')

html_full = rt.page(
    'dark',
    title='EDA Report (전처리 전) — SK Hynix Wafer Test RCC 예측',
    h1='EDA Report — 전처리 전',
    subtitle='SK Hynix — Wafer Test 기반 Field Health Data(RCC) 예측',
    meta_html=meta_html,
    findings_title='🔑 핵심 발견',
    findings_html=findings_html,
    toc_html=toc_html,
    body_html=body_html,
    footer_text='Generated from 01_raw_eda/eda.ipynb — SK Hynix × 기업연계프로젝트 EDA Report (전처리 전)',
)

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

sys.stdout.reconfigure(encoding="utf-8")
print(f"Report:            {OUTPUT}")
print(f"Standalone Report: {OUTPUT_STANDALONE}")
print(f"   섹션 수: {len(section_contents)}")
print(f"   이미지 수: {sum(1 for parts in section_contents.values() for p in parts if 'img src' in p)}")
print(f"   인라인된 이미지 수: {sum(1 for _ in re.finditer(r'data:image/png;base64', html_standalone))}")
