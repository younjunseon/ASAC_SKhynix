#!/usr/bin/env python3
"""eda.ipynb → eda_report_standalone.html 자동 생성 (dark theme, base64 inline)"""
import json, re, sys, os
sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NB_PATH    = os.path.join(SCRIPT_DIR, '..', 'eda.ipynb')
OUT_PATH   = os.path.join(SCRIPT_DIR, 'eda_report_standalone.html')

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']

# ─── Utility ────────────────────────────────────────────────
NOISE = [
    r'^\s*<Figure size', r'^setup 완료', r'^전처리 모듈 로드 완료',
    r'^\d+ EDA 모듈 로드 완료', r'^원본 데이터:', r'계산 중\.\.\.',
    r'^Xs \(die level\)', r'^Ys \(unit level\)',
    r'^Train merged shape', r'^VIF 계산:', r'^\s*제외:.*행$',
    r'^\s*\(특이행렬', r'^Downloading',
]

def is_noise(line):
    return any(re.search(p, line) for p in NOISE)

def filter_text(txt):
    lines = [l for l in txt.split('\n') if not is_noise(l)]
    result = re.sub(r'\n{3,}', '\n\n', '\n'.join(lines).strip())
    ls = result.split('\n')
    if len(ls) > 50:
        result = '\n'.join(ls[:50]) + f'\n... (생략 {len(ls)-50}줄)'
    return result

def get_texts(cell):
    out = []
    for o in cell.get('outputs', []):
        ot = o.get('output_type', '')
        if ot in ('stream', 'execute_result'):
            txt = o.get('text', o.get('data', {}).get('text/plain', []))
            if isinstance(txt, list): txt = ''.join(txt)
            txt = filter_text(txt)
            if txt.strip(): out.append(txt)
    return out

def get_imgs(cell):
    out = []
    for o in cell.get('outputs', []):
        img = o.get('data', {}).get('image/png', '')
        if img:
            out.append(img.strip() if isinstance(img, str) else ''.join(img).strip())
    return out

def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

def md2html(s):
    s = esc(s)
    s = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
    s = re.sub(r'`(.+?)`',       r'<code>\1</code>',     s)
    return s

def render_conc(src):
    """---\n> 결론\n- 포인트\n--- 형식 렌더링"""
    if not src: return ''
    parts = []
    for line in src.split('\n'):
        line = line.rstrip()
        if not line or line.strip() == '---': continue
        if line.lstrip().startswith('>'):
            parts.append(f'<p class="conc-main">{md2html(line.lstrip("> "))}</p>')
        elif re.match(r'^[\s]*[-*]\s', line):
            parts.append(f'<p class="conc-pt">• {md2html(line.lstrip("-* "))}</p>')
        else:
            h = md2html(line)
            if h.strip(): parts.append(f'<p class="conc-note">{h}</p>')
    return '\n'.join(parts)

# ─── Parse notebook structure ────────────────────────────────
PHASE_INFO = {
    1: ('📊', '#6366f1'), 2: ('🔍', '#06b6d4'), 3: ('⚠️', '#f59e0b'),
    4: ('🔗', '#10b981'), 5: ('🎯', '#ec4899'), 6: ('🗺️', '#8b5cf6'),
    7: ('📐', '#14b8a6'),
}

phases    = []
cur_phase = None
cur_sec   = None

for i, cell in enumerate(cells):
    src = ''.join(cell.get('source', []))
    ct  = cell['cell_type']

    if ct == 'markdown':
        # 1) Phase header: contains "# Phase N:"
        if '# Phase' in src:
            m = re.search(r'# Phase (\d+):\s*([^\n]+)', src)
            if m:
                n     = int(m.group(1))
                title = re.split(r'\s*모듈', m.group(2))[0].strip()
                icon, color = PHASE_INFO.get(n, ('📊', '#6366f1'))
                cur_phase = {'num': n, 'icon': icon, 'color': color,
                             'title': title, 'sections': []}
                phases.append(cur_phase)
                cur_sec = None
                continue

        # 2) Section header: ## N. or ## N-M.
        m = re.match(r'^#{2,3}\s+([\d\-]+)\.\s+(.+?)(?:\n|$)', src)
        if m and cur_phase is not None:
            raw = m.group(2).strip()
            # 제목 첫 줄 / " - 가나다" 전까지만
            title = re.split(r'\n| - [가-힣A-Za-z(]', raw)[0][:80].strip()
            cur_sec = {'num': m.group(1), 'title': title, 'blocks': []}
            cur_phase['sections'].append(cur_sec)
            continue

        # 3) Conclusion cell: ---\n> ...
        if (src.strip().startswith('---') and '\n>' in src
                and cur_sec is not None and cur_sec['blocks']):
            cur_sec['blocks'][-1]['conc'] = src
            continue

    elif ct == 'code' and cur_sec is not None:
        imgs  = get_imgs(cell)
        texts = get_texts(cell)
        if imgs or texts:
            # 첫 번째 주석을 설명으로
            desc = ''
            for line in ''.join(cell.get('source', [])).split('\n')[:6]:
                l = line.strip()
                if l.startswith('#') and len(l) > 2 and not l.startswith('#!'):
                    desc = l.lstrip('# ').rstrip()[:160]
                    break
            cur_sec['blocks'].append(
                {'ci': i, 'imgs': imgs, 'texts': texts,
                 'desc': desc, 'conc': None}
            )

# ─── HTML Rendering ──────────────────────────────────────────

def render_block(b):
    parts = []
    desc  = b['desc']

    # 텍스트만 있을 때도 desc 표시
    if texts := b['texts']:
        if desc and not b['imgs']:
            parts.append(f'<p class="graph-desc">🔢 {esc(desc)}</p>')
            desc = ''
        for txt in texts:
            parts.append(f'<pre class="output">{esc(txt)}</pre>')

    for img in b['imgs']:
        if desc:
            parts.append(f'<p class="graph-desc">📈 {esc(desc)}</p>')
            desc = ''
        parts.append(
            f'<div class="chart">'
            f'<img src="data:image/png;base64,{img}" alt="chart" loading="lazy">'
            f'</div>'
        )

    if b['conc']:
        ch = render_conc(b['conc'])
        if ch:
            parts.append(f'<div class="conc-box">{ch}</div>')

    return '\n'.join(parts)


def render_section(sec, phase_num, color):
    body = '\n'.join(render_block(b) for b in sec['blocks'])
    sid  = f"s{phase_num}-{sec['num'].replace('-', '_')}"

    # 첫 결론에서 insight 추출
    insight = ''
    for b in sec['blocks']:
        if b.get('conc'):
            for line in b['conc'].split('\n'):
                if line.lstrip().startswith('>'):
                    insight = md2html(line.lstrip('> ').strip()[:220])
                    break
        if insight: break

    insight_div = (f'<div class="sec-insight">💡 {insight}</div>'
                   if insight else '')

    return f'''\
<div class="section" id="{sid}">
  <div class="sec-hdr" onclick="toggle(this.parentElement)">
    <span class="sec-num">{sec["num"]}</span>
    <h3>{sec["title"]}</h3>
    <span class="tog">▼</span>
  </div>
  {insight_div}
  <div class="sec-body">
{body}
  </div>
  <div class="sec-foot" onclick="toggle(this.parentElement,true)">접기 ▲</div>
</div>'''


# TOC
toc_html = ''
for ph in phases:
    items = ''.join(
        f'<li><a href="#s{ph["num"]}-{s["num"].replace("-","_")}">'
        f'{s["num"]}. {s["title"]}</a></li>'
        for s in ph['sections']
    )
    toc_html += (f'<div class="toc-ph"><h3><span>{ph["icon"]}</span>'
                 f'Phase {ph["num"]}: {ph["title"]}</h3>'
                 f'<ul>{items}</ul></div>')

# Phase sections
body_html = ''
for ph in phases:
    secs = '\n'.join(render_section(s, ph['num'], ph['color'])
                     for s in ph['sections'])
    body_html += f'''\
<div class="ph-sect">
  <div class="ph-hdr" style="border-color:{ph["color"]}">
    <span class="ph-icon">{ph["icon"]}</span>
    <h2>{ph["title"]}</h2>
    <span class="ph-badge" style="background:{ph["color"]}">PHASE {ph["num"]}</span>
  </div>
  {secs}
</div>
'''

# Stats
total_secs   = sum(len(p['sections']) for p in phases)
total_imgs   = sum(len(b['imgs'])
                   for p in phases for s in p['sections'] for b in s['blocks'])

# Key Findings
KF = [
    ('star', '⭐ 단일 Feature 무력',
     'Pearson max|r|=0.037, Spearman max|r|=0.087. 단독 예측 불가 → 비선형 모델 + 9종 집계 + Feature Interaction 필수'),
    ('star', '⭐ MI ≠ Pearson (top 30 겹침 1개)',
     'MI top 30과 Pearson top 30이 1개만 공통. Feature Selection 때 Pearson·Spearman·MI 3종 병행 필수'),
    ('star', '⭐ Lot 정규화 역효과',
     '정규화 시 76.9% feature에서 target 상관 악화. F-ratio 높을수록 악화 심각 → Lot 정규화 전면 제외'),
    ('star', '⭐ NNR on health r=0.71 — Target Leakage',
     'health NNR은 강력하나 target leakage. WT feature NNR 잔차는 개선 0개(0.0%) → 공간 잔차 피처 제외'),
    ('', 'Zero-Inflated Target',
     'Y=0 70.8%, Y>0 29.2%. Two-Stage 모델 필수. all-zero 예측 RMSE ≈ 0.015이 베이스라인'),
    ('', '다중공선성 극심',
     'VIF>10: 781개/971개(80.4%), 고상관 쌍 47개. Boruta + Null Importance로 feature 정리 필수'),
    ('', 'Y=0 내부 이질성',
     'Cluster 0(72%)은 Y>0과 유사 프로파일 → 구조적 0와 확률적 0 공존. Two-Stage 분류 어렵게 만드는 요인'),
    ('', '집계 전략: 9종 균등 생성',
     '원본 CV max|r|=0.147이나 전처리 후 0.030으로 급락. 단일 집계 고정 금지 → 9종 생성 후 Feature Selection'),
]
kf_html = '\n'.join(
    f'<div class="fc {cls}"><h4>{title}</h4><p>{esc(desc)}</p></div>'
    for cls, title, desc in KF
)

# ─── Full HTML ───────────────────────────────────────────────
HTML = f'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EDA Report — SK Hynix Wafer Test RCC 예측</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{{
  --bg:#0f1117; --surf:#1a1d27; --surf2:#232736; --bdr:#2d3248;
  --txt:#e4e6f0; --muted:#8b8fa8;
  --acc:#6366f1; --accl:#818cf8; --acc-bg:rgba(99,102,241,.08);
  --grn:#22c55e; --ora:#f59e0b; --cyn:#06b6d4;
}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Noto Sans KR',-apple-system,sans-serif;background:var(--bg);color:var(--txt);line-height:1.7;font-size:15px}}
.container{{max-width:1200px;margin:0 auto;padding:0 32px}}

header{{background:linear-gradient(135deg,#1e1b4b 0%,#312e81 50%,#1e1b4b 100%);padding:64px 0 48px;border-bottom:1px solid var(--bdr);position:relative;overflow:hidden}}
header::before{{content:'';position:absolute;top:-50%;left:-50%;width:200%;height:200%;background:radial-gradient(ellipse at 30% 50%,rgba(99,102,241,.15),transparent 70%)}}
header .container{{position:relative;z-index:1}}
header h1{{font-size:2.2rem;font-weight:700;letter-spacing:-.02em;margin-bottom:8px}}
header .sub{{color:var(--accl);font-size:1.05rem;margin-bottom:24px}}
header .meta{{display:flex;gap:24px;color:var(--muted);font-size:.85rem;flex-wrap:wrap}}

.toc{{background:var(--surf);border:1px solid var(--bdr);border-radius:12px;padding:32px;margin:40px 0}}
.toc h2{{font-size:1.1rem;font-weight:600;margin-bottom:20px;color:var(--accl);text-transform:uppercase;letter-spacing:.05em}}
.toc-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}}
.toc-ph{{background:var(--surf2);border:1px solid var(--bdr);border-radius:8px;padding:16px 20px;transition:border-color .2s}}
.toc-ph:hover{{border-color:var(--acc)}}
.toc-ph h3{{font-size:.95rem;font-weight:600;margin-bottom:8px;display:flex;align-items:center;gap:8px}}
.toc-ph ul{{list-style:none;padding-left:28px}}
.toc-ph li{{font-size:.84rem;padding:2px 0}}
.toc-ph a{{color:var(--muted);text-decoration:none;transition:color .2s}}
.toc-ph a:hover{{color:var(--accl)}}

.kf{{margin:40px 0}}
.kf h2{{font-size:1.3rem;font-weight:600;margin-bottom:20px}}
.kf-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}}
.fc{{background:var(--surf);border:1px solid var(--bdr);border-left:3px solid var(--acc);border-radius:8px;padding:16px 20px}}
.fc.star{{border-left-color:var(--ora);background:rgba(245,158,11,.05)}}
.fc h4{{font-size:.85rem;font-weight:600;color:var(--accl);margin-bottom:6px}}
.fc.star h4{{color:var(--ora)}}
.fc p{{font-size:.82rem;color:var(--muted);line-height:1.5}}

.ph-sect{{margin:56px 0}}
.ph-hdr{{display:flex;align-items:center;gap:12px;margin-bottom:32px;padding-bottom:12px;border-bottom:2px solid var(--acc)}}
.ph-icon{{font-size:1.5rem}}
.ph-hdr h2{{font-size:1.4rem;font-weight:700}}
.ph-badge{{color:#fff;font-size:.75rem;font-weight:600;padding:2px 10px;border-radius:20px;margin-left:auto}}

.section{{background:var(--surf);border:1px solid var(--bdr);border-radius:12px;margin-bottom:24px;overflow:hidden}}
.sec-hdr{{padding:20px 24px;border-bottom:1px solid var(--bdr);cursor:pointer;display:flex;align-items:center;gap:12px;transition:background .2s}}
.sec-hdr:hover{{background:var(--surf2)}}
.sec-hdr h3{{font-size:1.05rem;font-weight:600;flex:1}}
.sec-num{{background:var(--acc-bg);color:var(--accl);font-size:.75rem;font-weight:600;padding:2px 8px;border-radius:4px;min-width:28px;text-align:center}}
.tog{{color:var(--muted);transition:transform .3s;font-size:1.1rem}}
.section.open .tog{{transform:rotate(180deg)}}
.sec-insight{{padding:12px 24px;background:var(--acc-bg);border-bottom:1px solid var(--bdr);font-size:.88rem;color:var(--accl);line-height:1.6}}
.sec-body{{padding:24px;display:none}}
.section.open .sec-body{{display:block}}
.sec-foot{{text-align:center;padding:10px;margin-top:16px;border-top:1px solid var(--bdr);color:var(--muted);font-size:.8rem;cursor:pointer;transition:color .2s,background .2s;display:none;border-radius:0 0 12px 12px;user-select:none}}
.section.open .sec-foot{{display:block}}
.sec-foot:hover{{color:var(--accl);background:var(--acc-bg)}}

.graph-desc{{font-size:.83rem;color:var(--cyn);margin:16px 0 6px;padding-left:12px;border-left:2px solid var(--cyn);line-height:1.5}}
.chart{{margin:4px 0 20px;text-align:center}}
.chart img{{max-width:100%;border-radius:8px;border:1px solid var(--bdr);background:#fff}}
pre.output{{background:var(--surf2);border:1px solid var(--bdr);border-radius:8px;padding:14px 18px;font-family:'JetBrains Mono',monospace;font-size:.76rem;line-height:1.6;overflow-x:auto;color:var(--muted);margin:12px 0;white-space:pre-wrap;word-break:break-all}}
.conc-box{{background:rgba(34,197,94,.05);border:1px solid rgba(34,197,94,.2);border-radius:8px;padding:16px 20px;margin-top:20px}}
.conc-main{{font-size:.9rem;color:#86efac;margin:4px 0;line-height:1.6;font-weight:500}}
.conc-pt{{font-size:.85rem;color:var(--muted);margin:4px 0;line-height:1.5;padding-left:8px}}
.conc-note{{font-size:.82rem;color:var(--muted);margin:2px 0;font-style:italic}}
strong{{color:var(--txt);font-weight:600}}
code{{background:var(--surf2);padding:1px 6px;border-radius:4px;font-family:'JetBrains Mono',monospace;font-size:.82rem;color:var(--accl)}}
footer{{text-align:center;padding:48px 0;color:var(--muted);font-size:.8rem;border-top:1px solid var(--bdr);margin-top:64px}}
@media(max-width:768px){{.container{{padding:0 16px}}.toc-grid{{grid-template-columns:1fr}}.kf-grid{{grid-template-columns:1fr}}}}
::-webkit-scrollbar{{width:8px;height:8px}}
::-webkit-scrollbar-track{{background:var(--bg)}}
::-webkit-scrollbar-thumb{{background:var(--bdr);border-radius:4px}}
::-webkit-scrollbar-thumb:hover{{background:var(--muted)}}
</style>
</head>
<body>

<header><div class="container">
  <h1>EDA Report</h1>
  <div class="sub">SK Hynix — Wafer Test 기반 Field Health Data(RCC) 예측</div>
  <div class="meta">
    <span>📅 2026-04-10</span>
    <span>📊 {len(phases)} Phase · {total_secs}개 분석 항목</span>
    <span>📐 174,980 dies · 43,745 units · 1,087 features</span>
    <span>🖼️ {total_imgs}개 그래프</span>
  </div>
</div></header>

<div class="container">

<div class="kf">
<h2>🔑 핵심 발견</h2>
<div class="kf-grid">
{kf_html}
</div>
</div>

<div class="toc">
<h2>목차</h2>
<div class="toc-grid">
{toc_html}
</div>
</div>

{body_html}

</div>

<footer>SK Hynix RCC 예측 프로젝트 — EDA Report · 2026-04-10</footer>

<script>
function toggle(sec, fromFooter) {{
  var isOpen = sec.classList.contains('open');
  if (isOpen) {{
    var hdr = sec.querySelector('.sec-hdr');
    var top = hdr.getBoundingClientRect().top + window.scrollY;
    sec.classList.remove('open');
    if (hdr.getBoundingClientRect().top < 0)
      window.scrollTo({{top: top - 16, behavior: 'instant'}});
  }} else {{
    sec.classList.add('open');
  }}
}}
document.addEventListener('DOMContentLoaded', function() {{
  var first = document.querySelector('.section');
  if (first) first.classList.add('open');
  document.querySelectorAll('.toc-ph a').forEach(function(a) {{
    a.addEventListener('click', function(e) {{
      e.preventDefault();
      var t = document.querySelector(this.getAttribute('href'));
      if (t) {{ t.classList.add('open'); t.scrollIntoView({{behavior:'smooth',block:'start'}}); }}
    }});
  }});
}});
</script>
</body>
</html>'''

with open(OUT_PATH, 'w', encoding='utf-8') as f:
    f.write(HTML)

size = os.path.getsize(OUT_PATH)
print(f'완료: {OUT_PATH}')
print(f'크기: {size/1024/1024:.1f} MB')
print(f'Phase: {len(phases)}, 섹션: {total_secs}, 이미지: {total_imgs}')
for ph in phases:
    print(f'  Phase {ph["num"]} {ph["title"]}: {len(ph["sections"])}개 섹션')
