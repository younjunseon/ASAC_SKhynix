import json, sys, re
sys.stdout.reconfigure(encoding='utf-8')

with open('eda_test.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']

# ── 필터 유틸 ─────────────────────────────────────────────
NOISE_PATTERNS = [
    r'^\s*<Figure size',
    r'^setup 완료',
    r'^전처리 모듈 로드 완료',
    r'^19개 EDA 모듈 로드 완료',
    r'^원본 데이터:',
    r'계산 중\.\.\.',
    r'^완료!',
    r'^Xs \(die level\)',
    r'^Ys \(unit level\)',
    r'^Train merged shape',
    r'^VIF 계산:.*대상$',
    r'^\s*제외:.*행$',
    r'^\s*\(특이행렬',
]

def is_noise_line(line):
    for p in NOISE_PATTERNS:
        if re.search(p, line):
            return True
    return False

def filter_text(txt):
    """노이즈 라인 제거 + 지나치게 긴 출력 자르기"""
    lines = txt.split('\n')
    kept = []
    for line in lines:
        if not is_noise_line(line):
            kept.append(line)
    result = '\n'.join(kept).strip()
    # 연속 빈줄 정리
    result = re.sub(r'\n{3,}', '\n\n', result)
    # 너무 긴 출력은 앞 40줄만
    lines2 = result.split('\n')
    if len(lines2) > 40:
        result = '\n'.join(lines2[:40]) + f'\n... (생략 {len(lines2)-40}줄)'
    return result

def get_text_outputs(cell):
    texts = []
    for out in cell.get('outputs', []):
        otype = out.get('output_type', '')
        if otype in ('stream', 'execute_result'):
            txt = out.get('text', out.get('data', {}).get('text/plain', []))
            if isinstance(txt, list): txt = ''.join(txt)
            txt = filter_text(txt)
            if txt:
                texts.append(txt)
    return texts

def get_images(cell):
    imgs = []
    for out in cell.get('outputs', []):
        img = out.get('data', {}).get('image/png', '')
        if img:
            imgs.append(img.strip())
    return imgs

def md_to_html(src):
    src = src.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    src = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', src)
    src = re.sub(r'`(.+?)`', r'<code>\1</code>', src)
    return src

def cell_src(cell):
    return ''.join(cell['source'])

# ── 결론 셀 렌더 ─────────────────────────────────────────
def render_conc(cell_idx):
    if cell_idx is None:
        return ''
    src = cell_src(cells[cell_idx])
    lines = src.split('\n')
    html_lines = []
    for line in lines:
        line = line.lstrip('> ').rstrip()
        if not line:
            continue
        if '⭐' in line:
            inner = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)
            inner = re.sub(r'`(.+?)`', r'<code>\1</code>', inner)
            inner = inner.replace('**⭐ ', '').replace('**', '')
            html_lines.append(f'<div class="key-insight">⭐ {inner}</div>')
        else:
            inner = md_to_html(line)
            html_lines.append(f'<p>{inner}</p>')
    return '\n'.join(html_lines)

def render_conc_insight(cell_idx):
    """결론 첫 문장 (section-insight용)"""
    if cell_idx is None: return ''
    src = cell_src(cells[cell_idx])
    for line in src.split('\n'):
        line = line.lstrip('> ').strip()
        if line and '⭐' not in line and '**결론' not in line:
            return md_to_html(line[:150])
    return ''

# ── 섹션별 콘텐츠 정의 ────────────────────────────────────
# 각 item: cells_spec = [(cell_idx, [text_idx...], [img_idx...], 'graph_desc'), ...]
# text_idx: 해당 셀의 몇번째 text output만 사용 (None = 전부)
# img_idx: 해당 셀의 몇번째 이미지만 사용 (None = 전부)

SECTIONS = [
    {
        'id': 'phase0', 'phase': 'PHASE 0', 'phase_icon': '⚙️',
        'phase_title': '전처리 파이프라인',
        'items': [
            {
                'num': '0-3/0-4', 'title': '전처리 파이프라인 실행 결과',
                'conc': None,
                'content': [
                    # (셀idx, 텍스트_출력_포함여부, 이미지_인덱스_리스트, 그래프설명)
                    (6,  True,  [],   None),   # 전처리 요약 텍스트
                    (8,  True,  [0],  '로그/Robust 스케일링 비율 — log=176개, robust=801개 적용'),
                ]
            },
            {
                'num': 'Phase 0-5', 'title': 'Train/Val 분포 일치 검증 (추가)',
                'conc': 11,
                'content': [
                    (10, True,  [0], 'Train vs Val 평균 차이 분포 (좌) / 차이 큰 상위 20개 feature (우) — 0.1 std 초과 feature 0개'),
                ]
            },
        ]
    },
    {
        'id': 'phase1', 'phase': 'PHASE 1', 'phase_icon': '📊',
        'phase_title': '데이터 기본 파악',
        'items': [
            {
                'num': '3', 'title': '데이터 구조 확인',
                'conc': 18,
                'content': [
                    (17, True,  [0], 'unit당 die 수 분포 (좌) / position별 die 수 균등 분포 (우) — 모든 unit이 정확히 4개 die'),
                    (19, True,  [],  None),  # split 비율 텍스트
                ]
            },
            {
                'num': '4', 'title': 'Target 변수 (health) 분포',
                'conc': 22,
                'content': [
                    (21, True,  [0], 'Target 전체 분포 (좌) / Y>0만 확대 (중) / split별 zero 비율 (우) — zero-inflated 구조 확인'),
                ]
            },
        ]
    },
    {
        'id': 'phase2', 'phase': 'PHASE 2', 'phase_icon': '🔍',
        'phase_title': 'Feature 품질 검사',
        'items': [
            {
                'num': '6', 'title': 'Feature 분포 분석',
                'conc': None,
                'content': [
                    (25, True,  [0],  '연속형/이산형 분류 및 고유값 수 분포 히스토그램'),
                    (26, False, [0],  '연속형 feature 랜덤 12개 히스토그램 — 분포 형태 다양성 확인'),
                    (27, False, [0],  '이산형 feature 12개 bar chart — 이산값 분포 확인'),
                ]
            },
            {
                'num': '7', 'title': '분산 분석 및 중복/상수 Feature 탐지',
                'conc': 30,
                'content': [
                    (29, True,  [0], '상수/중복 feature 개수 요약 bar chart — 상수 6개, 중복 쌍 60개 확인'),
                ]
            },
        ]
    },
    {
        'id': 'phase3', 'phase': 'PHASE 3', 'phase_icon': '⚠️',
        'phase_title': '이상치 및 스케일',
        'items': [
            {
                'num': '8', 'title': 'IQR 이상치 탐지 & 스케일 분석',
                'conc': 35,
                'content': [
                    (33, True,  [0], '이상치 비율 분포 히스토그램 (좌) / 상위 6개 feature boxplot (우) — X393(45.9%), X988(40.7%) 최상위'),
                    (34, True,  [0], 'mean·range·skewness 분포 — 억 단위 스케일 차이, |skew|>2 149개 잔존'),
                ]
            },
            {
                'num': '9', 'title': '스케일링 결과 검증 (추가)',
                'conc': 38,
                'content': [
                    (37, True,  [0, 1], 'log/robust 적용 feature 샘플 분포 (좌) / 스케일링 후 왜도 분포 (우) — |skew|>2 142개 잔존(14.5%)'),
                ]
            },
        ]
    },
    {
        'id': 'phase4', 'phase': 'PHASE 4', 'phase_icon': '🔗',
        'phase_title': 'Feature-Target 관계',
        'items': [
            {
                'num': '10', 'title': 'Pearson 상관관계',
                'conc': 44,
                'content': [
                    (41, True,  [],  None),  # 상위 20개 텍스트
                    (42, False, [0], '상관계수 분포 히스토그램 (좌) / 상위 20개 bar chart (우) — max|r|=0.037, |r|>0.05 없음'),
                    (43, False, [0], '전체 기준 상위 6개 feature scatter — Y=0/Y>0 분리는 있으나 Y>0 내부 산포 없음'),
                    (43, False, [1], 'Y>0 서브셋 기준 scatter — 전체 기준 top feature의 r이 Y>0 내부에서 ≈0으로 수렴'),
                ]
            },
            {
                'num': '10-1', 'title': '전처리 전후 Feature-Target 상관 변화 (추가)',
                'conc': 47,
                'content': [
                    (46, True,  [0], '전처리 전후 |r| 비교 scatter (좌) / delta 분포 (우) — 향상 49.2%, 악화 50.8%, 평균 delta≈0'),
                ]
            },
            {
                'num': '11', 'title': 'Feature 간 상관관계 (다중공선성)',
                'conc': 51,
                'content': [
                    (49, False, [0], '상위 30개 feature 상호 상관 히트맵 — 진한 블록 다수, 다중공선성 극심'),
                    (49, True,  [],  None),  # |r|>0.95 쌍 목록
                    (50, True,  [0], 'VIF 분포 히스토그램 — VIF>10이 781개(80.4%), X609·X303 VIF~150만'),
                ]
            },
            {
                'num': '13', 'title': '비선형 상관 분석 (Spearman + MI)',
                'conc': 56,
                'content': [
                    (53, True,  [],  None),  # 요약 수치
                    (54, False, [0], 'Pearson·Spearman·MI 순위 비교 scatter 3종 — 각 방법이 잡아내는 feature 집합이 다름'),
                    (54, True,  [],  None),  # 겹침 분석 텍스트
                    (55, False, [0], 'MI 상위 8개 feature scatter — 비선형 패턴(분산 증가, 군집) 확인'),
                ]
            },
        ]
    },
    {
        'id': 'phase5', 'phase': 'PHASE 5', 'phase_icon': '🎯',
        'phase_title': 'Target 심층 분석',
        'items': [
            {
                'num': '15', 'title': 'Y=0 vs Y>0 그룹 비교',
                'conc': 63,
                'content': [
                    (59, True,  [],  None),  # 기초통계
                    (60, True,  [0], 'Cohen\'s d 분포 (좌) / KS Statistic 분포 (중) / 상하위 feature bar (우) — |d|<0.2이나 70% 유의'),
                    (61, False, [0], 'Y=0 vs Y>0 분포 비교 violin (Cohen\'s d 상위 8개) — 분포 중심 이동 확인'),
                    (62, True,  [0], 'Y>0 내부 하위10% vs 상위90% violin — Stage 2용 피처(X41, X768 등) 확인'),
                ]
            },
            {
                'num': '15-1', 'title': 'Zero 발생 메커니즘 검증 (추가)',
                'conc': 66,
                'content': [
                    (65, True,  [0], '이상 die 수 → 불량률 bar (좌) / Y=0·Y>0 이상 die 수 분포 (우) — die 이상치 수와 불량률 역방향'),
                ]
            },
        ]
    },
    {
        'id': 'phase7', 'phase': 'PHASE 7', 'phase_icon': '📐',
        'phase_title': 'Die→Unit 집계 전략',
        'items': [
            {
                'num': '26-1', 'title': 'Die 내 이질성 심화 분석 (추가)',
                'conc': 70,
                'content': [
                    (69, True,  [0], 'CV vs mean 상관 scatter (좌) / CV 집계 상위 20개 feature bar (우) — CV 신호 소멸 확인'),
                ]
            },
            {
                'num': '26', 'title': '집계 방식별 Target 상관 비교',
                'conc': 75,
                'content': [
                    (72, True,  [],  None),  # 요약 테이블
                    (73, False, [0], '집계 방식별 |r| 분포 box/violin (좌) / 히스토그램 (중) / feature별 최적집계 히트맵 (우)'),
                    (74, False, [0], 'mean vs 각 집계 방식 scatter 10종 — 대각선 위=해당 집계가 mean보다 우세'),
                    (74, True,  [],  None),  # mean 외 상위 15개
                ]
            },
            {
                'num': '26-2', 'title': '전처리 후 집계별 Feature-Target 상관 (추가)',
                'conc': 78,
                'content': [
                    (77, True,  [0], '전처리 후 집계 방식별 |r| 분포 히스토그램 — 방식 간 차이 미미, CV만 0.147→0.030 급락'),
                    (77, True,  [],  None),  # 요약 테이블
                ]
            },
        ]
    },
]

# Key findings
KEY_FINDINGS = [
    ('⭐ Pearson top ≠ Y>0 내부 피처',
     'Y=0/Y>0 구분 피처와 Y>0 내부 health 크기 피처가 다르다. Stage 1(분류)과 Stage 2(회귀) Feature Selection을 반드시 분리해야 한다.'),
    ('⭐ MI ∩ Pearson 3.3%만 겹침',
     'MI top 30과 Pearson top 30이 1개만 공통. Pearson·Spearman·MI 3종 병행 없이는 신호의 97%를 놓친다.'),
    ('⭐ 이상 die ↑ → 불량률 ↓ (예상과 반대)',
     'IQR 기준 이상 die 수가 많을수록 불량률 감소. die 이상치 수는 field health와 무관 — Two-Stage 분류기 feature로 부적합.'),
    ('⭐ 원본 CV 0.147은 상수 피처(X375) 1개 때문',
     '전처리 후 실제 CV 신호는 mean과 동급(0.030). "CV가 집계의 핵심"이라는 원본 해석 폐기 → 9종 균등 생성 후 Feature Selection으로 선별.'),
    ('Zero-Inflated Target',
     'Y=0 70.8%, Y>0 29.2%. Two-Stage 모델 필수. RMSE 기준 all-zero 예측 시 0.015 수준이 베이스라인.'),
    ('단일 Feature 무력',
     'Pearson max|r|=0.037, Spearman max|r|=0.087. 단일 피처로 예측 불가. 비선형 모델 + 9종 집계 필수.'),
    ('다중공선성 극심',
     'VIF>10: 781개/971개(80.4%). 완전 중복 쌍 60개. Boruta + Null Importance가 최우선 Feature Selection 작업.'),
    ('전처리 누수 없음',
     'Train/Val 평균 차이 >0.1 std: 0개/977개. 스케일러 적용 후 분포 이동 없음 — 전처리 정상 확인.'),
]

# ── HTML 렌더 ─────────────────────────────────────────────
def render_content_item(ci, use_text, img_indices, graph_desc):
    cell = cells[ci]
    parts = []

    if use_text:
        texts = get_text_outputs(cell)
        for txt in texts:
            esc = txt.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
            parts.append(f'<pre class="output">{esc}</pre>')

    imgs = get_images(cell)
    for idx in img_indices:
        if idx < len(imgs):
            desc_html = f'<p class="graph-desc">📈 {graph_desc}</p>' if graph_desc else ''
            parts.append(f'{desc_html}<div class="chart"><img src="data:image/png;base64,{imgs[idx]}" alt="chart"></div>')

    return '\n'.join(parts)

def render_section(item, phase_id):
    num = item['num']
    title = item['title']
    sec_id = f"section-{phase_id}-{num.replace('/','_')}"

    conc_html = render_conc(item['conc'])
    insight_text = render_conc_insight(item['conc'])
    insight_div = f'<div class="section-insight">💡 {insight_text}</div>' if insight_text else ''

    body_parts = []
    for spec in item['content']:
        ci, use_text, img_indices, graph_desc = spec
        body_parts.append(render_content_item(ci, use_text, img_indices, graph_desc))
    body_html = '\n'.join(body_parts)

    conc_div = f'<div class="conclusion-box">{conc_html}</div>' if conc_html else ''

    return f'''
<div class="section" id="{sec_id}">
  <div class="section-header" onclick="toggleSection(this.parentElement)">
    <span class="section-num">{num}</span>
    <h3>{title}</h3>
    <span class="toggle">▼</span>
  </div>
  {insight_div}
  <div class="section-body">
    {body_html}
    {conc_div}
  </div>
  <div class="section-footer" onclick="toggleSection(this.parentElement, true)">접기 ▲</div>
</div>'''

# TOC
toc_items = []
for ph in SECTIONS:
    items_li = ''.join(
        f'<li><a href="#section-{ph["id"]}-{it["num"].replace("/","_")}">{it["num"]}. {it["title"]}</a></li>'
        for it in ph['items']
    )
    toc_items.append(f'''
<div class="toc-phase">
  <h3><span class="phase-icon">{ph["phase_icon"]}</span>{ph["phase_title"]}</h3>
  <ul>{items_li}</ul>
</div>''')

kf_cards = ''.join(
    f'<div class="finding-card {"star" if t.startswith("⭐") else ""}"><h4>{t}</h4><p>{d}</p></div>'
    for t, d in KEY_FINDINGS
)

phase_sections = []
for ph in SECTIONS:
    secs = ''.join(render_section(it, ph['id']) for it in ph['items'])
    phase_sections.append(f'''
<div class="phase-section">
  <div class="phase-header">
    <span class="icon">{ph["phase_icon"]}</span>
    <h2>{ph["phase_title"]}</h2>
    <span class="phase-num">{ph["phase"]}</span>
  </div>
  {secs}
</div>''')

# ── CSS + HTML ────────────────────────────────────────────
HTML = f'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EDA Report (전처리 후) — SK Hynix RCC 예측</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root {{
  --bg:#f8f9fc; --surface:#fff; --surface2:#f1f3f8; --border:#e2e5f0;
  --text:#1a1d2e; --text-muted:#5c6078;
  --accent:#4f46e5; --accent-light:#4338ca; --accent-bg:rgba(79,70,229,.06);
  --green:#16a34a; --green-bg:rgba(22,163,74,.05); --green-border:rgba(22,163,74,.2);
  --star:#b45309; --star-bg:rgba(180,83,9,.08);
}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Noto Sans KR',-apple-system,sans-serif;background:var(--bg);color:var(--text);line-height:1.7;font-size:15px}}
.container{{max-width:1200px;margin:0 auto;padding:0 32px}}
/* Header */
header{{background:linear-gradient(135deg,#eef2ff 0%,#e0e7ff 50%,#eef2ff 100%);padding:64px 0 48px;border-bottom:1px solid var(--border);position:relative;overflow:hidden}}
header::before{{content:'';position:absolute;top:-50%;left:-50%;width:200%;height:200%;background:radial-gradient(ellipse at 30% 50%,rgba(79,70,229,.08),transparent 70%)}}
header .container{{position:relative;z-index:1}}
header h1{{font-size:2.2rem;font-weight:700;letter-spacing:-.02em;margin-bottom:8px}}
header .subtitle{{color:var(--accent-light);font-size:1.05rem;margin-bottom:24px}}
header .meta{{display:flex;gap:24px;color:var(--text-muted);font-size:.85rem;flex-wrap:wrap}}
/* TOC */
.toc{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:32px;margin:40px 0;box-shadow:0 1px 3px rgba(0,0,0,.04)}}
.toc h2{{font-size:1.1rem;font-weight:600;margin-bottom:20px;color:var(--accent-light);text-transform:uppercase;letter-spacing:.05em}}
.toc-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}}
.toc-phase{{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:16px 20px;transition:border-color .2s}}
.toc-phase:hover{{border-color:var(--accent)}}
.toc-phase h3{{font-size:.95rem;font-weight:600;margin-bottom:8px;display:flex;align-items:center;gap:8px}}
.phase-icon{{font-size:1.1rem}}
.toc-phase ul{{list-style:none;padding-left:28px}}
.toc-phase li{{font-size:.85rem;padding:2px 0}}
.toc-phase a{{color:var(--text-muted);text-decoration:none;transition:color .2s}}
.toc-phase a:hover{{color:var(--accent-light)}}
/* Key Findings */
.key-findings{{margin:40px 0}}
.key-findings h2{{font-size:1.3rem;font-weight:600;margin-bottom:20px}}
.findings-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}}
.finding-card{{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--accent);border-radius:8px;padding:16px 20px;box-shadow:0 1px 3px rgba(0,0,0,.04)}}
.finding-card.star{{border-left-color:var(--star);background:var(--star-bg)}}
.finding-card h4{{font-size:.85rem;font-weight:600;color:var(--accent-light);margin-bottom:6px}}
.finding-card.star h4{{color:var(--star)}}
.finding-card p{{font-size:.82rem;color:var(--text-muted);line-height:1.5}}
/* Phase */
.phase-section{{margin:56px 0}}
.phase-header{{display:flex;align-items:center;gap:12px;margin-bottom:32px;padding-bottom:12px;border-bottom:2px solid var(--accent)}}
.phase-header .icon{{font-size:1.5rem}}
.phase-header h2{{font-size:1.4rem;font-weight:700}}
.phase-header .phase-num{{background:var(--accent);color:#fff;font-size:.75rem;font-weight:600;padding:2px 10px;border-radius:20px;margin-left:auto}}
/* Section */
.section{{background:var(--surface);border:1px solid var(--border);border-radius:12px;margin-bottom:24px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.04)}}
.section-header{{padding:20px 24px;border-bottom:1px solid var(--border);cursor:pointer;display:flex;align-items:center;gap:12px;transition:background .2s}}
.section-header:hover{{background:var(--surface2)}}
.section-header h3{{font-size:1.05rem;font-weight:600;flex:1}}
.section-num{{background:var(--accent-bg);color:var(--accent-light);font-size:.75rem;font-weight:600;padding:2px 8px;border-radius:4px;min-width:28px;text-align:center}}
.toggle{{color:var(--text-muted);transition:transform .3s;font-size:1.1rem}}
.section.open .toggle{{transform:rotate(180deg)}}
.section-insight{{padding:12px 24px;background:var(--accent-bg);border-bottom:1px solid var(--border);font-size:.88rem;color:var(--accent-light);line-height:1.6}}
.section-body{{padding:24px;display:none}}
.section.open .section-body{{display:block}}
.section-footer{{text-align:center;padding:10px;margin-top:16px;border-top:1px solid var(--border);color:var(--text-muted);font-size:.8rem;cursor:pointer;transition:color .2s,background .2s;border-radius:0 0 12px 12px;user-select:none;display:none}}
.section.open .section-footer{{display:block}}
.section-footer:hover{{color:var(--accent-light);background:var(--accent-bg)}}
/* Content */
.graph-desc{{font-size:.83rem;color:var(--text-muted);margin:16px 0 6px;padding-left:12px;border-left:2px solid var(--accent);line-height:1.5}}
.chart{{margin:4px 0 20px;text-align:center}}
.chart img{{max-width:100%;border-radius:8px;border:1px solid var(--border);background:#fff}}
pre.output{{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:14px 18px;font-family:'JetBrains Mono',monospace;font-size:.76rem;line-height:1.6;overflow-x:auto;color:var(--text-muted);margin:12px 0;white-space:pre-wrap;word-break:break-all}}
/* Conclusion */
.conclusion-box{{background:var(--green-bg);border:1px solid var(--green-border);border-radius:8px;padding:16px 20px;margin-top:20px}}
.conclusion-box p{{font-size:.88rem;color:#166534;margin:4px 0;line-height:1.6}}
.key-insight{{background:var(--star-bg);border-left:3px solid var(--star);border-radius:0 6px 6px 0;padding:10px 14px;margin:10px 0;font-size:.88rem;color:var(--star);line-height:1.6;font-weight:500}}
strong{{color:var(--text);font-weight:600}}
code{{background:var(--surface2);padding:1px 6px;border-radius:4px;font-family:'JetBrains Mono',monospace;font-size:.82rem}}
footer{{text-align:center;padding:48px 0;color:var(--text-muted);font-size:.8rem;border-top:1px solid var(--border);margin-top:64px}}
@media(max-width:768px){{.container{{padding:0 16px}}.toc-grid{{grid-template-columns:1fr}}.findings-grid{{grid-template-columns:1fr}}}}
::-webkit-scrollbar{{width:8px;height:8px}}
::-webkit-scrollbar-track{{background:var(--bg)}}
::-webkit-scrollbar-thumb{{background:var(--border);border-radius:4px}}
</style>
</head>
<body>

<header>
<div class="container">
  <h1>EDA Report — 전처리 후</h1>
  <div class="subtitle">SK Hynix — Wafer Test 기반 Field Health Data(RCC) 예측</div>
  <div class="meta">
    <span>📅 2026-04-10</span>
    <span>📊 7 Phase · 전처리 적용 버전</span>
    <span>📐 174,980 dies · 43,745 units · 977 features (전처리 후)</span>
  </div>
</div>
</header>

<div class="container">

<div class="key-findings">
  <h2>핵심 발견 — 전처리 후 EDA</h2>
  <div class="findings-grid">{kf_cards}</div>
</div>

<div class="toc">
  <h2>목차</h2>
  <div class="toc-grid">{''.join(toc_items)}</div>
</div>

{''.join(phase_sections)}

</div>

<footer>SK Hynix RCC 예측 프로젝트 — 전처리 후 EDA Report · 2026-04-10</footer>

<script>
function toggleSection(section, fromFooter) {{
  var isOpen = section.classList.contains('open');
  if (isOpen) {{
    // 접을 때: 스크롤 점프 방지 — 헤더 위치로 유지
    var header = section.querySelector('.section-header');
    var headerTop = header.getBoundingClientRect().top + window.scrollY;
    section.classList.remove('open');
    // 헤더가 뷰포트 밖으로 나갔으면 헤더 위치로 스크롤
    var newHeaderTop = header.getBoundingClientRect().top;
    if (newHeaderTop < 0) {{
      window.scrollTo({{ top: headerTop - 16, behavior: 'instant' }});
    }}
  }} else {{
    section.classList.add('open');
  }}
}}

document.addEventListener('DOMContentLoaded', () => {{
  // 기본으로 첫 섹션만 열기
  const first = document.querySelector('.section');
  if (first) first.classList.add('open');

  // TOC 앵커 클릭 시 스크롤 점프 방지 (smooth scroll)
  document.querySelectorAll('.toc-phase a').forEach(a => {{
    a.addEventListener('click', function(e) {{
      e.preventDefault();
      var target = document.querySelector(this.getAttribute('href'));
      if (target) {{
        target.classList.add('open');
        target.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
      }}
    }});
  }});
}});
</script>
</body>
</html>'''

with open('eda_test_report.html', 'w', encoding='utf-8') as f:
    f.write(HTML)

import os
size = os.path.getsize('eda_test_report.html')
print(f'완료: eda_test_report.html ({size/1024/1024:.1f} MB)')
