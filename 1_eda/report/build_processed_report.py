import json, sys, re, os
sys.stdout.reconfigure(encoding='utf-8')

# 입력: 전처리 후 EDA 노트북 (report/ 기준 상대경로 → 어디서 실행해도 동작)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import report_theme as rt
_NOTEBOOK = os.path.join(_HERE, '..', '02_processed_eda', 'eda_after.ipynb')
with open(_NOTEBOOK, 'r', encoding='utf-8') as f:
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
                    (8,  True,  [0],  '전처리 단계별 Feature 수 변화 바차트 — 원본 → 고결측 제거 → 저분산 제거 → 최종'),
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
                    (29, True,  [0], '상수/중복 feature 개수 요약 bar chart — 상수 6개, 중복 쌍 55개 확인'),
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
                    (33, True,  [0], '이상치 비율 분포 히스토그램 (좌) / 상위 6개 feature boxplot (우) — X393(45.8%), X988(40.5%) 최상위'),
                    (34, True,  [0], 'mean·range·skewness 분포 — 천만 단위 스케일 차이, |skew|>2 179개 잔존'),
                ]
            },
            {
                'num': '9', 'title': 'Clip 결과 검증 (추가)',
                'conc': 38,
                'content': [
                    (37, True,  [0], 'Winsorization 후 왜도 분포 — clip 적용 feature 수 및 잔존 |skew|>2 확인'),
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
                    (46, True,  [0], '전처리 전후 |r| 비교 scatter (좌) / delta 분포 (우) — 향상 50.6%, 악화 46.0%, 평균 delta≈0'),
                ]
            },
            {
                'num': '11', 'title': 'Feature 간 상관관계 (다중공선성)',
                'conc': 51,
                'content': [
                    (49, False, [0], '상위 30개 feature 상호 상관 히트맵 — 진한 블록 다수, 다중공선성 극심'),
                    (49, True,  [],  None),  # |r|>0.95 쌍 목록
                    (50, True,  [0], 'VIF 분포 히스토그램 — VIF>10이 651개(67.0%), X427·X420 VIF 10^50 이상'),
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
                    (62, True,  [0], 'Y>0 내부 하위10% vs 상위90% violin — Stage 2용 피처(X41, X982 등) 확인'),
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
                    (77, True,  [0], '전처리 후 집계 방식별 |r| 분포 히스토그램 — 방식 간 차이 미미, CV만 0.147→0.031 급락'),
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
    ('⭐ MI ∩ Pearson 10%만 겹침',
     'MI top 30과 Pearson top 30이 3개만 공통. Pearson·Spearman·MI 3종 병행 없이는 신호의 90%를 놓친다.'),
    ('⭐ 이상 die ↑ → 불량률 ↓ (예상과 반대)',
     'IQR 기준 이상 die 수가 많을수록 불량률 감소. die 이상치 수는 field health와 무관 — Two-Stage 분류기 feature로 부적합.'),
    ('⭐ 원본 CV 0.147은 상수 피처(X375) 1개 때문',
     '전처리 후 실제 CV 신호는 mean과 동급(0.031). "CV가 집계의 핵심"이라는 원본 해석 폐기 → 9종 균등 생성 후 Feature Selection으로 선별.'),
    ('Zero-Inflated Target',
     'Y=0 70.8%, Y>0 29.2%. Two-Stage 모델 필수. RMSE 기준 all-zero 예측 시 0.015 수준이 베이스라인.'),
    ('단일 Feature 무력',
     'Pearson max|r|=0.037, Spearman max|r|=0.087. 단일 피처로 예측 불가. 비선형 모델 + 9종 집계 필수.'),
    ('다중공선성 극심',
     'VIF>10: 651개/971개(67.0%). 완전 중복 쌍 55개. Boruta + Null Importance가 최우선 Feature Selection 작업.'),
    ('전처리 누수 없음',
     'Train/Val 평균 차이 >0.1 std: 0개/977개. cleaning + clip 후 분포 이동 없음 — 전처리 정상 확인.'),
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

    body_parts = []
    for spec in item['content']:
        ci, use_text, img_indices, graph_desc = spec
        body_parts.append(render_content_item(ci, use_text, img_indices, graph_desc))
    body_html = '\n'.join(body_parts)
    if conc_html:
        body_html += f'<div class="conclusion-box">{conc_html}</div>'

    return rt.section(sec_id, num, title, body_html, insight_text)

# ── TOC / 핵심발견 / Phase 섹션 (공용 테마 report_theme 사용) ──
toc_items = []
for ph in SECTIONS:
    items = [(f'section-{ph["id"]}-{it["num"].replace("/", "_")}',
              f'{it["num"]}. {it["title"]}') for it in ph['items']]
    toc_items.append(rt.toc_phase(ph["phase_icon"], ph["phase_title"], items))
toc_html = ''.join(toc_items)

findings_html = ''.join(rt.finding_card(t, d) for t, d in KEY_FINDINGS)

phase_sections = []
for ph in SECTIONS:
    secs = ''.join(render_section(it, ph['id']) for it in ph['items'])
    phase_sections.append(rt.phase(ph["phase_icon"], ph["phase_title"], ph["phase"], secs))
body_html = ''.join(phase_sections)

meta_html = ('<span>📅 2026-04-10</span>'
             '<span>📊 7 Phase · 전처리 적용 버전</span>'
             '<span>📐 174,980 dies · 43,745 units · 977 features (전처리 후)</span>')

HTML = rt.page(
    'light',
    title='EDA Report (전처리 후) — SK Hynix RCC 예측',
    h1='EDA Report — 전처리 후',
    subtitle='SK Hynix — Wafer Test 기반 Field Health Data(RCC) 예측',
    meta_html=meta_html,
    findings_title='핵심 발견 — 전처리 후 EDA',
    findings_html=findings_html,
    toc_html=toc_html,
    body_html=body_html,
    footer_text='SK Hynix RCC 예측 프로젝트 — 전처리 후 EDA Report · 2026-04-10',
)

_OUTPUT = os.path.join(_HERE, 'eda_processed_report.html')
with open(_OUTPUT, 'w', encoding='utf-8') as f:
    f.write(HTML)

size = os.path.getsize(_OUTPUT)
print(f'완료: {_OUTPUT} ({size/1024/1024:.1f} MB)')
