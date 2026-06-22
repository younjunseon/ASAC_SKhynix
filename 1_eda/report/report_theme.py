"""
EDA 리포트 공용 디자인 시스템 (전처리 전/후 리포트 통일 테마)

- PALETTES : 'dark'(전처리 전) / 'light'(전처리 후) 두 색상 팔레트
- css(theme): :root 변수만 팔레트별로 바뀌고 CSS 규칙은 100% 동일 → 통일감 보장
- JS       : 섹션 접기/펼치기 + TOC 스무스 스크롤 (공통)
- page() / section() / finding_card() / toc_phase() / phase() : 마크업 헬퍼

build_raw_report.py(dark) 와 build_processed_report.py(light) 가 공유한다.
"""

# ── 색상 팔레트 (키는 underscore → CSS 변수명은 hyphen 으로 변환) ──
PALETTES = {
    'dark': {
        'bg': '#0f1117', 'surface': '#1a1d27', 'surface2': '#232736', 'border': '#2d3248',
        'text': '#e4e6f0', 'text_muted': '#8b8fa8',
        'accent': '#6366f1', 'accent_light': '#818cf8', 'accent_bg': 'rgba(99,102,241,0.10)',
        'green': '#22c55e', 'green_bg': 'rgba(34,197,94,0.08)',
        'green_border': 'rgba(34,197,94,0.25)', 'green_text': '#4ade80',
        'star': '#fbbf24', 'star_bg': 'rgba(251,191,36,0.10)',
        'cyan': '#22d3ee', 'img_bg': '#ffffff',
        'header_grad': 'linear-gradient(135deg,#1e1b4b 0%,#312e81 50%,#1e1b4b 100%)',
        'header_glow': 'rgba(99,102,241,0.15)',
        'shadow': '0 1px 3px rgba(0,0,0,0.35)',
    },
    'light': {
        'bg': '#f8f9fc', 'surface': '#ffffff', 'surface2': '#f1f3f8', 'border': '#e2e5f0',
        'text': '#1a1d2e', 'text_muted': '#5c6078',
        'accent': '#4f46e5', 'accent_light': '#4338ca', 'accent_bg': 'rgba(79,70,229,0.06)',
        'green': '#16a34a', 'green_bg': 'rgba(22,163,74,0.05)',
        'green_border': 'rgba(22,163,74,0.20)', 'green_text': '#166534',
        'star': '#b45309', 'star_bg': 'rgba(180,83,9,0.08)',
        'cyan': '#0891b2', 'img_bg': '#ffffff',
        'header_grad': 'linear-gradient(135deg,#eef2ff 0%,#e0e7ff 50%,#eef2ff 100%)',
        'header_glow': 'rgba(79,70,229,0.08)',
        'shadow': '0 1px 3px rgba(0,0,0,0.04)',
    },
}

_FONT_IMPORT = ("@import url('https://fonts.googleapis.com/css2?"
                "family=Noto+Sans+KR:wght@300;400;500;600;700&"
                "family=JetBrains+Mono:wght@400;500&display=swap');")

# CSS 규칙 — 색상은 전부 var(--xxx). 두 테마가 동일한 이 규칙을 공유한다.
_RULES = """
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Noto Sans KR',-apple-system,BlinkMacSystemFont,sans-serif;background:var(--bg);color:var(--text);line-height:1.7;font-size:15px}
.container{max-width:1200px;margin:0 auto;padding:0 32px}
/* Header */
header{background:var(--header-grad);padding:64px 0 48px;border-bottom:1px solid var(--border);position:relative;overflow:hidden}
header::before{content:'';position:absolute;top:-50%;left:-50%;width:200%;height:200%;background:radial-gradient(ellipse at 30% 50%,var(--header-glow),transparent 70%)}
header .container{position:relative;z-index:1}
header h1{font-size:2.2rem;font-weight:700;letter-spacing:-.02em;margin-bottom:8px}
header .subtitle{color:var(--accent-light);font-size:1.05rem;margin-bottom:24px}
header .meta{display:flex;gap:24px;color:var(--text-muted);font-size:.85rem;flex-wrap:wrap}
header .meta span{display:flex;align-items:center;gap:6px}
/* TOC */
.toc{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:32px;margin:40px 0;box-shadow:var(--shadow)}
.toc h2{font-size:1.1rem;font-weight:600;margin-bottom:20px;color:var(--accent-light);text-transform:uppercase;letter-spacing:.05em}
.toc-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}
.toc-phase{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:16px 20px;transition:border-color .2s}
.toc-phase:hover{border-color:var(--accent)}
.toc-phase h3{font-size:.95rem;font-weight:600;margin-bottom:8px;display:flex;align-items:center;gap:8px}
.phase-icon{font-size:1.1rem}
.toc-phase ul{list-style:none;padding-left:28px}
.toc-phase li{font-size:.85rem;padding:2px 0}
.toc-phase a{color:var(--text-muted);text-decoration:none;transition:color .2s}
.toc-phase a:hover{color:var(--accent-light)}
/* Key Findings */
.key-findings{margin:40px 0}
.key-findings h2{font-size:1.3rem;font-weight:600;margin-bottom:20px}
.findings-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}
.finding-card{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--accent);border-radius:8px;padding:16px 20px;box-shadow:var(--shadow)}
.finding-card.star{border-left-color:var(--star);background:var(--star-bg)}
.finding-card h4{font-size:.85rem;font-weight:600;color:var(--accent-light);margin-bottom:6px}
.finding-card.star h4{color:var(--star)}
.finding-card p{font-size:.82rem;color:var(--text-muted);line-height:1.5}
/* Phase */
.phase-section{margin:56px 0}
.phase-header{display:flex;align-items:center;gap:12px;margin-bottom:32px;padding-bottom:12px;border-bottom:2px solid var(--accent)}
.phase-header .icon{font-size:1.5rem}
.phase-header h2{font-size:1.4rem;font-weight:700}
.phase-header .phase-num{background:var(--accent);color:#fff;font-size:.75rem;font-weight:600;padding:2px 10px;border-radius:20px;margin-left:auto}
/* Section */
.section{background:var(--surface);border:1px solid var(--border);border-radius:12px;margin-bottom:24px;overflow:hidden;box-shadow:var(--shadow)}
.section-header{padding:20px 24px;border-bottom:1px solid var(--border);cursor:pointer;display:flex;align-items:center;gap:12px;transition:background .2s}
.section-header:hover{background:var(--surface2)}
.section-header h3{font-size:1.05rem;font-weight:600;flex:1}
.section-num{background:var(--accent-bg);color:var(--accent-light);font-size:.75rem;font-weight:600;padding:2px 8px;border-radius:4px;min-width:28px;text-align:center}
.toggle{color:var(--text-muted);transition:transform .3s;font-size:1.1rem}
.section.open .toggle{transform:rotate(180deg)}
.section-insight{padding:12px 24px;background:var(--accent-bg);border-bottom:1px solid var(--border);font-size:.88rem;color:var(--accent-light);line-height:1.6}
.section-body{padding:24px;display:none}
.section.open .section-body{display:block}
.section-footer{text-align:center;padding:10px;margin-top:16px;border-top:1px solid var(--border);color:var(--text-muted);font-size:.8rem;cursor:pointer;transition:color .2s,background .2s;border-radius:0 0 12px 12px;user-select:none;display:none}
.section.open .section-footer{display:block}
.section-footer:hover{color:var(--accent-light);background:var(--accent-bg)}
/* Content */
.graph-desc{font-size:.83rem;color:var(--text-muted);margin:16px 0 6px;padding-left:12px;border-left:2px solid var(--accent);line-height:1.5}
.analysis-label{font-size:.82rem;color:var(--cyan);font-weight:500;margin:20px 0 8px;padding-left:12px;border-left:2px solid var(--cyan)}
.insight-box{background:var(--green-bg);border-left:3px solid var(--green);padding:10px 16px;margin:12px 0;border-radius:0 6px 6px 0;font-size:.9rem;color:var(--green)}
.chart{margin:4px 0 20px;text-align:center}
.chart img{max-width:100%;border-radius:8px;border:1px solid var(--border);background:var(--img-bg)}
pre.output{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:14px 18px;font-family:'JetBrains Mono',monospace;font-size:.76rem;line-height:1.6;overflow-x:auto;color:var(--text-muted);margin:12px 0;white-space:pre-wrap;word-break:break-all}
.table-wrapper{overflow-x:auto;margin:12px 0}
table{width:100%;border-collapse:collapse;font-size:.82rem}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--border)}
th{background:var(--surface2);font-weight:600;color:var(--accent-light);font-size:.78rem;text-transform:uppercase;letter-spacing:.03em}
td{color:var(--text-muted)}
.section-body ul{margin:0;padding-left:0}
.section-body li{margin:4px 0 4px 20px;font-size:.88rem;color:var(--text-muted);list-style:disc}
.section-body p{margin:8px 0;font-size:.88rem;color:var(--text-muted)}
/* Conclusion */
.conclusion-box{background:var(--green-bg);border:1px solid var(--green-border);border-radius:8px;padding:16px 20px;margin-top:20px}
.conclusion-box p{font-size:.88rem;color:var(--green-text);margin:4px 0;line-height:1.6}
.key-insight{background:var(--star-bg);border-left:3px solid var(--star);border-radius:0 6px 6px 0;padding:10px 14px;margin:10px 0;font-size:.88rem;color:var(--star);line-height:1.6;font-weight:500}
strong{color:var(--text);font-weight:600}
code{background:var(--surface2);padding:1px 6px;border-radius:4px;font-family:'JetBrains Mono',monospace;font-size:.82rem}
footer{text-align:center;padding:48px 0;color:var(--text-muted);font-size:.8rem;border-top:1px solid var(--border);margin-top:64px}
@media(max-width:768px){.container{padding:0 16px}.toc-grid{grid-template-columns:1fr}.findings-grid{grid-template-columns:1fr}header h1{font-size:1.6rem}}
::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px}
::-webkit-scrollbar-thumb:hover{background:var(--text-muted)}
"""

# 공통 JS — 첫 섹션 자동 열림 + 접기 시 스크롤 점프 방지 + TOC 스무스 스크롤
JS = """<script>
function toggleSection(section) {
  var isOpen = section.classList.contains('open');
  if (isOpen) {
    var header = section.querySelector('.section-header');
    var headerTop = header.getBoundingClientRect().top + window.scrollY;
    section.classList.remove('open');
    if (header.getBoundingClientRect().top < 0) {
      window.scrollTo({ top: headerTop - 16, behavior: 'instant' });
    }
  } else {
    section.classList.add('open');
  }
}
document.addEventListener('DOMContentLoaded', function () {
  var first = document.querySelector('.section');
  if (first) first.classList.add('open');
  document.querySelectorAll('.toc-phase a').forEach(function (a) {
    a.addEventListener('click', function (e) {
      e.preventDefault();
      var target = document.querySelector(this.getAttribute('href'));
      if (target) {
        target.classList.add('open');
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });
});
</script>"""


def css(theme):
    """팔레트(theme)별 :root + 공통 규칙으로 <style> 반환."""
    p = PALETTES[theme]
    root = ":root{" + ";".join(f"--{k.replace('_', '-')}:{v}" for k, v in p.items()) + "}"
    return f"<style>\n{_FONT_IMPORT}\n{root}\n{_RULES}</style>"


def finding_card(title, desc):
    """핵심 발견 카드. title이 ⭐로 시작하면 star 스타일."""
    star = ' star' if title.strip().startswith('⭐') else ''
    return f'<div class="finding-card{star}"><h4>{title}</h4><p>{desc}</p></div>'


def toc_phase(icon, title, items):
    """목차 한 Phase. items = [(anchor_id, label), ...]"""
    lis = ''.join(f'<li><a href="#{a}">{label}</a></li>' for a, label in items)
    return (f'<div class="toc-phase"><h3><span class="phase-icon">{icon}</span>{title}</h3>'
            f'<ul>{lis}</ul></div>')


def phase(icon, title, label, sections_html):
    """Phase 섹션 래퍼."""
    return (f'<div class="phase-section"><div class="phase-header">'
            f'<span class="icon">{icon}</span><h2>{title}</h2>'
            f'<span class="phase-num">{label}</span></div>{sections_html}</div>')


def section(sec_id, num, title, body_html, insight_html='', footer_label='접기 ▲'):
    """접기/펼치기 섹션. insight_html 있으면 헤더 아래 인사이트 줄 표시."""
    insight = f'<div class="section-insight">💡 {insight_html}</div>' if insight_html else ''
    return (f'<div class="section" id="{sec_id}">'
            f'<div class="section-header" onclick="toggleSection(this.parentElement)">'
            f'<span class="section-num">{num}</span><h3>{title}</h3>'
            f'<span class="toggle">▼</span></div>{insight}'
            f'<div class="section-body">{body_html}</div>'
            f'<div class="section-footer" onclick="toggleSection(this.parentElement)">{footer_label}</div>'
            f'</div>')


def page(theme, *, title, h1, subtitle, meta_html, findings_title,
         findings_html, toc_html, body_html, footer_text):
    """전체 페이지 HTML 조립 (헤더 + 핵심발견 + TOC + 본문 + 푸터 + JS)."""
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
{css(theme)}
</head>
<body>
<header><div class="container">
<h1>{h1}</h1>
<div class="subtitle">{subtitle}</div>
<div class="meta">{meta_html}</div>
</div></header>
<div class="container">
<div class="key-findings"><h2>{findings_title}</h2>
<div class="findings-grid">{findings_html}</div></div>
<div class="toc"><h2>목차</h2>
<div class="toc-grid">{toc_html}</div></div>
{body_html}
</div>
<footer>{footer_text}</footer>
{JS}
</body>
</html>"""
