"""
Agent 루프 핵심 로직.
Claude API tool_use를 사용하여 ①~⑦ 단계를 처리하고
SSE(Server-Sent Events)로 프론트에 스트리밍.
"""
import json
import asyncio
import os
from dotenv import load_dotenv
import anthropic
from tools import (infer_period, scan_data, analyze_features, get_importance,
                   get_pred_actual_data, get_trend_top1_data, get_top_unit_data,
                   get_position_defect_rate, get_ppm_delta, get_feature_scatter_data,
                   get_lot_trend_with_split, get_wafer_die_data, get_recent_lot_trend,
                   get_pred_ppm_trend, get_location_ppm_top, get_weekly_yield_trend,
                   get_anomaly_feature_stats, get_val_rmse, get_feat_vs_health_scatter,
                   get_lot_grade_stack, get_pred_health_hist, get_feature_dist_compare,
                   get_top_risk_units, get_lot_mean_ppm_top, get_wafer_risk_die_ratio_top,
                   get_shap_bar_top, get_unit_shap_bar, get_candidate_units)
from report import build_html

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL = "claude-sonnet-4-6"

# Claude에게 제공하는 tool 정의
TOOLS = [
    {
        "name": "infer_period",
        "description": "사용자의 기간 표현('이번 주', '11월', '최근 3일' 등)을 YYYYMMDD start/end로 변환. 기간 표현이 있을 때만 호출.",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_text": {"type": "string", "description": "사용자가 입력한 기간 표현 원문"},
            },
            "required": ["user_text"],
        },
    },
    {
        "name": "scan_data",
        "description": "데이터를 스캔하여 grade1~4 unit 수, 집중 lot/wafer를 반환. start/end로 날짜 필터 가능.",
        "input_schema": {
            "type": "object",
            "properties": {
                "start": {"type": "string", "description": "시작 날짜 YYYYMMDD (예: 20251101). 생략 시 전체."},
                "end":   {"type": "string", "description": "종료 날짜 YYYYMMDD (예: 20251130). 생략 시 전체."},
            },
            "required": [],
        },
    },
    {
        "name": "analyze_features",
        "description": "grade1 vs grade4 그룹 feature 분포를 비교하여 유의미하게 다른 feature를 반환. start/end로 날짜 필터 가능.",
        "input_schema": {
            "type": "object",
            "properties": {
                "start": {"type": "string", "description": "시작 날짜 YYYYMMDD (예: 20251101). 생략 시 전체."},
                "end":   {"type": "string", "description": "종료 날짜 YYYYMMDD (예: 20251130). 생략 시 전체."},
                "top_n": {"type": "integer", "description": "반환할 feature 수 (기본 10)"},
            },
            "required": [],
        },
    },
    {
        "name": "get_importance",
        "description": "feature_importance.csv에서 전체 feature 중요도 상위 목록을 반환.",
        "input_schema": {
            "type": "object",
            "properties": {
                "top_n": {"type": "integer", "description": "반환할 feature 수 (기본 10)"}
            },
        },
    },
]

SYSTEM_PROMPT = """당신은 SK Hynix 반도체 공정 품질 분석 AI Agent입니다.
PI(Process Integration 엔지니어)의 요청에 따라 데이터를 분석하고 품질불량예측보고서를 작성합니다.

## 생성되는 보고서 양식 (확정된 양식)

보고서는 "품질불량예측보고서" 양식으로 생성됩니다. 구조는 아래와 같습니다.

**왼쪽 — 모델링 결과**
- 모델 성능 KPI (Val RMSE / 불량 수량 / 예측 ppm)
- Feature Importance (실데이터)
- 불량 트렌드 (실데이터)

**오른쪽 — 불량 유닛 분석**
- 대표 Unit 정보 (실데이터)
- SHAP 영향도 / 피처 정상·불량 분포 (실데이터)
- 위치별 불량률 (실데이터)

## 흐름 (반드시 순서대로, tool 1개 실행 후 반드시 텍스트 출력, 그 다음 tool 실행)

**tool 실행 전후에 절대 자체 안내 문구("기간 변환 중", "분석 시작", "STEP" 등)를 출력하지 마세요. 아래 지정된 형식만 출력.**

1. 사용자가 기간 표현("이번주", "최근 N일", "N월" 등)을 명시한 경우에만 `infer_period(user_text)` 실행. 완료 즉시 **다른 tool 호출 없이** 아래 형식 **그대로** 출력하세요 (label/start/end만 결과값으로 교체):
분석 기간: {label} ({start}~{end})
<<<BUTTONS: 확인, 기간 변경>>>

   사용자가 "확인"을 누르면 step 2로 진행. "기간 변경"을 누르면 step 6으로.

   **기간 표현이 없으면 infer_period를 호출하지 말고 기간을 묻지도 마세요. 바로 step 2로 진행 (start/end 생략).**

## grade 기준 (반드시 숙지)
- grade4 = 매우위험 (HIGH, reg_pred 가장 높음, 빨강)
- grade3 = 위험
- grade2 = 조심
- grade1 = 정상 (reg_pred 가장 낮음, 초록)

2. `scan_data(start, end)` 실행. 완료 즉시 **다른 tool 호출 없이** 아래 형식으로 출력:
   📊 스캔 결과 (기간: YYYY.MM.DD ~ YYYY.MM.DD)

   - 총 unit: N개
   - 고위험: M개
   - 집중 LOT: LOT XXX (고위험 N개)
   - 집중 웨이퍼: LOT_XXX-WF_YY (고위험 N개)

3. 스캔 결과 출력 후 `get_importance` 실행. 완료 즉시 **다른 tool 호출 없이** 아래 형식으로 출력:
   📋 주요 Feature TOP5
   - 1위: X234 (gain: 0.XXX)
   - 2위: X891 (gain: 0.XXX)
   - ...

4. importance 출력 후, 아래 두 줄을 반드시 출력하세요:

이 내용대로 보고서를 생성할까요?
<<<BUTTONS: 보고서 생성, 기간 변경>>>

5. "보고서 생성" 확인을 받으면 **다른 어떤 텍스트도 출력하지 말고, 어떤 tool도 호출하지 말고**, **아래 한 줄만** 출력하세요:
<<<HTML_REPORT>>>

6. "기간 변경" 버튼을 받으면 아래 형식으로만 출력하세요:
   분석할 기간을 입력해주세요. (예: 전체 기간, 6월, 최근 30일)
   그 후 사용자가 기간을 입력하면 `infer_period(user_text)`로 변환하고, **step 2부터 다시 실행**하세요. (scan_data, get_importance 재실행)

## 절대 금지 사항 (위반 시 사용자가 직접 지적함)

- **system에 "이미 실행 완료" 표시가 된 tool은 어떤 경우에도 다시 호출하지 마세요.**
- **"보고서 생성"/"보고서 작성"/"보고서 만들어줘"/"확인"/"진행" 등의 짧은 확인 응답에는 tool을 호출하지 않습니다.** 캐시된 결과는 이미 user가 봤으므로 다시 출력하지도 마세요. 오직 `<<<HTML_REPORT>>>` 한 줄만.
- **같은 분석 내용을 두 번 출력하지 마세요.** scan/importance/analyze 결과는 각각 한 번만 출력.

## 버튼 태그 규칙 (필수)

PI에게 선택을 요청할 때는 반드시 아래 태그를 텍스트 끝에 붙이세요:
<<<BUTTONS: 버튼1, 버튼2>>>

버튼은 최대 4개, 각 버튼은 짧고 명확하게 (10자 이내).

## 날짜 필터 규칙

오늘 날짜: **2026-06-11**. "이번 주"는 06/08~06/11. 데이터 날짜 범위: **20260327 ~ 20260708** (2026년 3월~7월). 사용자가 명시한 표현이 있을 때만 `infer_period` 도구로 변환해 start/end를 도출.

## 기타 주의사항
- 수치는 구체적으로 (예: "2.3배 높음", "HIGH 그룹 평균 0.082")
- 한국어로 답변"""


def _run_tool(name: str, inputs: dict) -> str:
    """tool 이름과 입력값으로 실제 함수 실행."""
    fn_map = {
        "infer_period": infer_period,
        "scan_data": scan_data,
        "analyze_features": analyze_features,
        "get_importance": get_importance,
    }
    result = fn_map[name](**inputs)
    return json.dumps(result, ensure_ascii=False)


def _feat_dist_for_shap(shap_items):
    """선택 유닛 SHAP 차트의 1등(유효) X피처로 R3 분포차트를 동기화.
    shap_items에서 feature_dist에 존재하는 첫 X피처를 골라 분포를 계산한다.
    유효 피처가 없으면 None(→ 호출부에서 전역 기본값 사용)."""
    try:
        for _it in (shap_items or []):
            _f = str(_it.get("feature", ""))
            if _f.startswith("X") and _f[1:].isdigit():
                _fdc = get_feature_dist_compare(feature=_f)
                if _fdc.get("labels"):
                    return _fdc
    except Exception:
        pass
    return None


def _build_report_data(tool_cache: dict) -> dict:
    """tool_cache로부터 report_data를 조립 (pred_actual / trend_top1 실데이터 포함)."""
    # scan_data에서 사용된 날짜 필터 추출 (재사용)
    _scan = tool_cache.get("scan_data", {})
    analysis = {}   # analyze_features(compet_xs) 의존 제거 — pred_actual 등 비-xs 데이터만 채움

    # Pred vs Actual 실데이터 주입
    if "pred_actual" not in analysis:
        try:
            analysis["pred_actual"] = get_pred_actual_data()
        except Exception:
            analysis["pred_actual"] = []

    # 피처 top-1 트렌드 실데이터 주입
    if "trend_top1" not in analysis:
        try:
            analysis["trend_top1"] = get_trend_top1_data()
        except Exception:
            analysis["trend_top1"] = {}

    # 대표 Unit 실데이터 주입
    top_unit = {}
    try:
        top_unit = get_top_unit_data()
    except Exception:
        pass

    # 대표 유닛의 per-unit SHAP (R2 패널용) + 후보 유닛 목록(생성/편집 시 선택용)
    unit_shap = []
    try:
        if top_unit.get("serial"):
            unit_shap = get_unit_shap_bar(top_unit["serial"], 10)
    except Exception:
        pass
    candidate_units = []
    try:
        candidate_units = get_candidate_units(5)
    except Exception:
        pass

    # 포지션별 불량률 실데이터 주입
    pos_defect = {}
    try:
        pos_defect = get_position_defect_rate()
    except Exception:
        pass

    # 배너 ppm delta
    ppm_delta = {}
    try:
        ppm_delta = get_ppm_delta()
    except Exception:
        pass

    # 피처 scatter (상위 2개)
    feat_scatter = {}
    try:
        feat_scatter = get_feature_scatter_data()
    except Exception:
        pass

    # L2 전체 split 트렌드
    lot_trend_split = {}
    try:
        lot_trend_split = get_lot_trend_with_split()
    except Exception:
        pass

    # L2 보고서용 주차별 수율 트렌드 (trend_data.csv, 최근 5주)
    weekly_yield_trend = {}
    try:
        weekly_yield_trend = get_weekly_yield_trend(recent_weeks=10)
    except Exception:
        pass

    # L2 fallback용 최근 LOT 트렌드
    recent_lot_trend = {}
    try:
        recent_lot_trend = get_recent_lot_trend(recent_n=35)
    except Exception:
        pass

    # R3 예측 ppm 트렌드 (HIGH/MED 그룹 평균)
    pred_ppm_trend = {}
    try:
        pred_ppm_trend = get_pred_ppm_trend(recent_n=20)
    except Exception:
        pass

    # 웨이퍼맵 die 좌표
    wafer_die = {}
    try:
        wafer_die = get_wafer_die_data()
    except Exception:
        pass

    # 어노멀리 피처(get_anomaly_feature_stats, compet_xs 의존) 제거 — R2 패널은 SHAP로 대체됨
    anomaly_stats = []

    # Feature Importance 전체 풀 (늘리기 복원용, top_n=20 미리 로드)
    importance_all = []
    try:
        importance_all = get_importance(top_n=20).get("features", [])
    except Exception:
        pass

    # L4: 피처값 vs 예측 health scatter
    feat_vs_health = {}
    try:
        feat_vs_health = get_feat_vs_health_scatter()
    except Exception:
        pass

    # R3: 피처 정상/위험 분포 비교 히스토그램 (ProcessFactor 차트)
    # 대표 유닛 SHAP 1등 피처와 동기화, 없으면 전역 기본값
    feat_dist_compare = {}
    try:
        feat_dist_compare = _feat_dist_for_shap(unit_shap) or get_feature_dist_compare()
    except Exception:
        pass

    try:
        _val_rmse = get_val_rmse()
    except Exception:
        _val_rmse = "0.005698"

    return {
        "meta": {
            "title": "Field Health 불량 원인 분석 보고서",
            "period": tool_cache.get("infer_period", {}).get("label", ""),
            "model": "Stacking Ensemble",
            "val_rmse": _val_rmse,
        },
        "scan":             tool_cache.get("scan_data", {}),
        "importance":       tool_cache.get("get_importance", {}),
        "analysis":         analysis,
        "top_unit":         top_unit,
        "unit_shap":        unit_shap,        # 대표/선택 유닛 per-unit SHAP (R2)
        "candidate_units":  candidate_units,  # 대표 유닛 후보 5개 (선택용)
        "pos_defect":       pos_defect,
        "ppm_delta":        ppm_delta,
        "feat_scatter":     feat_scatter,
        "lot_trend_split":  lot_trend_split,
        "weekly_yield_trend": weekly_yield_trend,
        "recent_lot_trend":   recent_lot_trend,
        "pred_ppm_trend":     pred_ppm_trend,
        "wafer_die":        wafer_die,
        "importance_all":    importance_all,           # Feature Importance 전체 풀 (top20)
        "anomaly_stats":     anomaly_stats[:5],       # 표시용: top5만
        "anomaly_stats_all": list(anomaly_stats),     # 원본 전체 보존 (늘리기 복원용)
        "feat_vs_health":    feat_vs_health,
        "feat_dist_compare": feat_dist_compare,
        "actions":          [],
    }


async def run_agent(user_message: str, history: list, initial_tool_cache: dict = None):
    """
    Agent 루프 실행. SSE 이벤트를 yield.
    history: [{"role": "user"/"assistant", "content": "..."}]
    initial_tool_cache: 이미 실행된 tool 결과 (프론트에서 전달, 재실행 방지)

    yield하는 이벤트 형식:
    {"type": "text", "content": "..."}          - 일반 텍스트 메시지
    {"type": "tool_start", "tool": "..."}        - tool 실행 시작
    {"type": "tool_result", "tool": "...", ...}  - tool 실행 결과
    {"type": "confirm", "buttons": [...]}        - PI 확인 요청 (버튼)
    {"type": "report_ready", "markdown": "..."}  - 보고서 초안 완성
    {"type": "done"}                             - 완료
    """
    import re as _re   # 함수 전체에서 안전하게 사용 (조건부 import로 인한 unbound 방지)
    # history에서 단순 텍스트 메시지만 추출 (tool_use 블록 등 복잡한 구조 제거)
    clean_history = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        # 연속된 같은 role 스킵
        if clean_history and clean_history[-1]["role"] == role:
            continue
        clean_history.append({"role": role, "content": content})

    # 마지막이 assistant면 제거
    if clean_history and clean_history[-1]["role"] == "assistant":
        clean_history = clean_history[:-1]

    messages = clean_history + [{"role": "user", "content": user_message}]
    # 프론트에서 전달된 이미 실행된 tool 결과로 초기화 (재실행 방지)
    tool_cache = dict(initial_tool_cache) if initial_tool_cache else {}

    # 이미 실행된 tool 목록을 system prompt에 주입 → Claude가 재실행 안 함
    def _build_system(cache: dict) -> str:
        if not cache:
            return SYSTEM_PROMPT
        done = []
        if "scan_data" in cache:
            done.append(f"- scan_data: 완료 → total_units={cache['scan_data'].get('total_units')}, grade1_count={cache['scan_data'].get('grade1_count')}")
        if "get_importance" in cache:
            n = len(cache['get_importance'].get('features', []))
            done.append(f"- get_importance: 완료 → {n}개 feature 반환됨")
        if "analyze_features" in cache:
            n = len(cache['analyze_features'].get('top_features', []))
            done.append(f"- analyze_features: 완료 → top_features {n}개 반환됨")
        if not done:
            return SYSTEM_PROMPT
        injected = (
            "\n\n## 이미 실행 완료된 Tool (절대 재실행 금지, 결과 재출력 금지)\n"
            + "\n".join(done)
            + "\n\n위 tool들의 결과는 이미 PI에게 출력되었습니다. **절대 같은 결과를 다시 출력하지 말고**, "
              "사용자가 '보고서 생성' 등 확인 의사를 표한 경우 곧바로 `<<<HTML_REPORT>>>` 한 줄만 응답하세요."
        )
        return SYSTEM_PROMPT + injected

    import time as _time
    _t0 = _time.time()
    print(f"[TIMING] run_agent 진입 msg={repr(user_message[:30])} cache_keys={list(tool_cache.keys())}")

    # ── Python-side 기간 감지 ──────────────────────────────────────────────────
    # 보고서 요청 키워드 감지
    _report_keywords = {"보고서", "보고서 생성", "보고서생성", "보고서 작성", "보고서작성",
                        "보고서 만들어줘", "보고서 만들어", "작성", "작성해줘"}
    _is_report_request = any(kw in user_message for kw in _report_keywords)
    _needs_scan = "scan_data" not in tool_cache

    # 보고서 요청이고 scan이 아직 없으면 → 기간 확인 먼저
    if _is_report_request and _needs_scan and "infer_period" not in tool_cache:
        # 기간 표현이 있으면 추론, 없으면 전체 기간
        import re as _re
        _period_pattern = _re.compile(
            r'이번\s*주|저번\s*주|지난\s*주|다음\s*주|'
            r'최근\s*\d+\s*[일주]|'
            r'\d{1,2}월|'
            r'\d{8}\s*[~\-]\s*\d{8}|'
            r'전체\s*기간|전체'
        )
        if _period_pattern.search(user_message):
            period = infer_period(user_message)
        else:
            period = {"label": "이번 주 (06/08~06/10)", "start": "20260608", "end": "20260610"}
        tool_cache["infer_period"] = period
        print(f"[TIMING] 기간 shortcut 진입 {_time.time()-_t0:.2f}s")
        yield {"type": "tool_result", "tool": "infer_period", "result": period}
        label = period.get("label", "전체 기간")
        start = period.get("start", "")
        end   = period.get("end", "")
        date_str = f" ({start}~{end})" if start and end else ""
        yield {"type": "text", "content": f"분석 기간: {label}{date_str}"}
        yield {"type": "confirm", "buttons": ["확인", "기간 변경"]}
        yield {"type": "done"}
        return

    # "보고서 작성" shortcut — 캐시에 필수 결과가 모두 있으면 Claude 호출 없이 바로 HTML 생성
    _shortcut_phrases = {"보고서 작성", "보고서작성", "보고서 만들어줘", "보고서 만들어",
                         "보고서 생성", "보고서생성", "작성", "작성해줘", "진행", "확인"}
    _has_min_cache = "scan_data" in tool_cache and "get_importance" in tool_cache
    if user_message.strip() in _shortcut_phrases and _has_min_cache:
        report_data = _build_report_data(tool_cache)
        html = build_html(report_data)
        yield {"type": "report_ready", "html": html, "report_data": report_data}
        yield {"type": "done"}
        return

    # 유닛 후보 버튼 선택(예: 'S22474') → 해당 유닛으로 보고서 생성
    _sel = _re.match(r'^\s*(S\d{4,})\s*$', user_message.strip())
    if _sel and _has_min_cache:
        _serial = _sel.group(1)
        report_data = _build_report_data(tool_cache)
        _handle_command({"action": "change_unit", "serial": _serial}, report_data)
        html = build_html(report_data)
        yield {"type": "text", "content": f"{_serial} 유닛으로 보고서를 생성했습니다."}
        yield {"type": "report_ready", "html": html, "report_data": report_data}
        yield {"type": "done"}
        return

    # "기간 확인" shortcut — "확인" 메시지이고 분석이 안 됐으면 바로 실행
    _analysis_done = "scan_data" in tool_cache and "get_importance" in tool_cache
    if user_message.strip() == "확인" and not _analysis_done:
        period = tool_cache.get("infer_period", {"label": "이번 주 (06/08~06/10)", "start": "20260608", "end": "20260610"})
        start = period.get("start", "")
        end = period.get("end", "")

        yield {"type": "tool_start", "tool": "scan_data"}
        await asyncio.sleep(5)   # '🔍 데이터 스캔 중...' 표시 후 5초 대기 → 이후 진행
        scan_result = await asyncio.to_thread(scan_data, start=start, end=end)
        tool_cache["scan_data"] = scan_result
        yield {"type": "tool_result", "tool": "scan_data", "result": scan_result}

        s = scan_result
        date_label = f"{start[:4]}.{start[4:6]}.{start[6:]} ~ {end[:4]}.{end[4:6]}.{end[6:]}" if start and end else "전체"
        top_wafer = s.get('top_wafer', {})
        if isinstance(top_wafer, dict):
            top_wafer_str = f"LOT{top_wafer.get('lot','?')}-WF{top_wafer.get('wafer','?')}"
            top_wafer_g4  = top_wafer.get('grade4_count', 0)
        else:
            top_wafer_str = str(top_wafer)
            top_wafer_g4  = s.get('top_wafer_grade4_count', 0)
        scan_text = (
            f"📊 스캔 결과 (기간: {date_label})\n\n"
            f"- 총 unit: {s.get('total_units', s.get('total', 0))}개\n"
            f"- 고위험: {s.get('grade4_count',0)}개\n"
            f"- 집중 LOT: LOT {s.get('top_lot','N/A')} (고위험 {s.get('top_lot_grade4_count',0)}개)\n"
            f"- 집중 웨이퍼: {top_wafer_str} (고위험 {top_wafer_g4}개)"
        )
        yield {"type": "text", "content": scan_text}

        yield {"type": "tool_start", "tool": "get_importance"}
        imp_result = await asyncio.to_thread(get_importance, top_n=10)
        tool_cache["get_importance"] = imp_result
        yield {"type": "tool_result", "tool": "get_importance", "result": imp_result}

        features = imp_result.get("features", [])
        imp_lines = ["📋 주요 Feature TOP5"]
        for i, f in enumerate(features[:5], 1):
            gain = f.get('lgbm_gain', f.get('importance', 0))
            imp_lines.append(f"- {i}위: {f.get('feature','?')} (gain: {gain:.1f})")
        yield {"type": "text", "content": "\n".join(imp_lines)}

        # 대표 유닛 후보 제시 — 선택 시 그 유닛으로 보고서 생성 (R1 웨이퍼맵 + R2 SHAP)
        _cands = []
        try: _cands = get_candidate_units(5)
        except Exception: pass
        if _cands:
            _lines = ["분석을 마쳤습니다.\n\n보고서에 표시할 **대표 유닛**을 선택하세요 (예측 ppm 높은 순):"]
            for c in _cands:
                _lines.append(f"- {c['serial']} · LOT{c['lot']}-WF{c['wafer']} · {c['ppm']:,.0f} ppm")
            yield {"type": "text", "content": "\n".join(_lines)}
            yield {"type": "confirm", "buttons": [c["serial"] for c in _cands] + ["기간 변경"]}
        else:
            yield {"type": "text", "content": "분석을 마쳤습니다.\n\n이 내용대로 보고서를 생성할까요?"}
            yield {"type": "confirm", "buttons": ["보고서 생성", "기간 변경"]}
        yield {"type": "done"}
        return

    while True:
        # asyncio.Queue를 통해 스트리밍 청크를 실시간 yield
        # (client.messages.stream은 동기 블로킹이므로 별도 스레드에서 실행 후 Queue로 전달)
        queue: asyncio.Queue = asyncio.Queue()
        _SENTINEL = object()  # 스트림 종료 신호

        def _stream_to_queue(q: asyncio.Queue, loop: asyncio.AbstractEventLoop):
            """동기 스트리밍 — 스레드 안에서 실행하며 청크를 loop.call_soon_threadsafe로 Queue에 적재."""
            try:
                with client.messages.stream(
                    model=MODEL,
                    max_tokens=4096,
                    system=_build_system(tool_cache),
                    tools=TOOLS,
                    messages=messages,
                ) as stream:
                    for event in stream:
                        etype = type(event).__name__
                        if etype == "RawContentBlockDeltaEvent":
                            delta = event.delta
                            if hasattr(delta, "text") and delta.text:
                                loop.call_soon_threadsafe(q.put_nowait, ("chunk", delta.text))
                    msg = stream.get_final_message()
                    loop.call_soon_threadsafe(q.put_nowait, ("final", msg))
            except Exception as e:
                loop.call_soon_threadsafe(q.put_nowait, ("error", e))
            finally:
                loop.call_soon_threadsafe(q.put_nowait, (_SENTINEL, None))

        loop = asyncio.get_running_loop()
        import threading
        t = threading.Thread(target=_stream_to_queue, args=(queue, loop), daemon=True)
        t.start()

        # Queue에서 실시간으로 이벤트 수신
        full_text = ""
        final_msg = None
        stream_error = None
        # 태그 포함 여부 감지용: <<<가 나타난 순간부터 청크 버퍼링
        tag_started = False
        sent_text = ""  # 이미 yield한 텍스트 길이 추적

        while True:
            kind, payload = await queue.get()
            if kind is _SENTINEL:
                break
            if kind == "chunk":
                full_text += payload
                # <<<가 아직 안 나타난 순수 텍스트만 실시간 전송
                if not tag_started:
                    if "<<<" in full_text:
                        tag_started = True
                        # <<<이전 텍스트 중 아직 안 보낸 부분만 전송
                        before_tag = full_text.split("<<<", 1)[0]
                        tag_body = full_text.split("<<<", 1)[1]
                        unsent = before_tag[len(sent_text):]
                        if unsent and not tag_body.startswith("HTML_REPORT"):
                            yield {"type": "text", "content": unsent}
                            sent_text = before_tag
                    else:
                        yield {"type": "text", "content": payload}
                        sent_text += payload
            elif kind == "final":
                final_msg = payload
            elif kind == "error":
                stream_error = payload

        if stream_error:
            if isinstance(stream_error, anthropic.BadRequestError):
                yield {"type": "error", "message": f"API 오류: {getattr(stream_error, 'message', str(stream_error))}"}
            elif isinstance(stream_error, anthropic.AuthenticationError):
                yield {"type": "error", "message": "API 키가 유효하지 않습니다. .env 파일의 ANTHROPIC_API_KEY를 확인해주세요."}
            else:
                yield {"type": "error", "message": f"서버 오류: {str(stream_error)}"}
            return

        if final_msg is None:
            yield {"type": "error", "message": "스트림에서 최종 메시지를 받지 못했습니다."}
            return

        # assistant 응답을 history에 추가
        from anthropic.types import TextBlock
        clean_content = []
        for block in final_msg.content:
            if block.type == "text":
                stripped = block.text.rstrip()
                if stripped:
                    clean_content.append(TextBlock(type="text", text=stripped))
            else:
                clean_content.append(block)
        if not clean_content:
            clean_content = [TextBlock(type="text", text=".")]
        messages.append({"role": "assistant", "content": clean_content})

        # 텍스트 블록 최종 처리 (태그 감지 및 분리)
        # final block 처리: 태그(<<<...>>>)가 있는 블록만 처리
        # 태그 앞 텍스트는 청크 스트리밍에서 이미 전송됐으므로 skip
        for block in final_msg.content:
            if block.type == "text":
                text = block.text
                print(f"[DEBUG final block] len={len(text)} has_tag={'<<<' in text} preview={repr(text[-200:])}")

                # 태그 없는 순수 텍스트 → 청크로 이미 전송됨, 중복 방지
                if "<<<" not in text:
                    continue

                # 보고서 초안 완성 감지 (<<<HTML_REPORT>>> 태그)
                if "<<<HTML_REPORT>>>" in text:
                    report_data = _build_report_data(tool_cache)
                    html = build_html(report_data)
                    yield {"type": "report_ready", "html": html, "report_data": report_data}
                # 차트 수정 명령 감지: <<<UPDATE_CHART: chart=shap, top_n=5>>>
                elif "<<<UPDATE_CHART:" in text:
                    remaining = text
                    clean_text_parts = []
                    while "<<<UPDATE_CHART:" in remaining:
                        before, rest = remaining.split("<<<UPDATE_CHART:", 1)
                        if before.strip():
                            clean_text_parts.append(before.strip())
                        tag_body, remaining = rest.split(">>>", 1) if ">>>" in rest else (rest, "")
                        params = {}
                        for part in tag_body.split(","):
                            part = part.strip()
                            if "=" in part:
                                k, v = part.split("=", 1)
                                params[k.strip()] = v.strip()
                        import copy
                        patched = copy.deepcopy({
                            "meta":       {"title": "Field Health 불량 원인 분석 보고서",
                                           "model": "Stacking Ensemble"},
                            "scan":       tool_cache.get("scan_data", {}),
                            "importance": tool_cache.get("get_importance", {}),
                            "analysis":   {},
                            "actions":    [],
                            "chart_params": params,
                        })
                        html = build_html(patched)
                        yield {"type": "report_ready", "html": html, "report_data": patched}
                    if remaining.strip():
                        clean_text_parts.append(remaining.strip())
                    combined = "\n".join(clean_text_parts)
                    if combined:
                        yield {"type": "text", "content": combined}
                # 버튼 태그 감지: <<<BUTTONS: 버튼1, 버튼2>>>
                elif "<<<BUTTONS:" in text:
                    parts = text.split("<<<BUTTONS:", 1)
                    # before 텍스트는 청크로 이미 전송됨 → skip
                    btn_part = parts[1].split(">>>", 1)
                    buttons = [b.strip() for b in btn_part[0].split(",") if b.strip()]
                    if buttons:
                        yield {"type": "confirm", "buttons": buttons}
                else:
                    pass  # <<<가 없으면 이미 청크로 실시간 전송됨 → 중복 전송 방지

        # tool_use 처리
        if final_msg.stop_reason == "tool_use":
            tool_results = []
            for block in final_msg.content:
                if block.type == "tool_use":
                    yield {"type": "tool_start", "tool": block.name}
                    result_str = await asyncio.to_thread(_run_tool, block.name, block.input)
                    result_data = json.loads(result_str)
                    tool_cache[block.name] = result_data
                    yield {"type": "tool_result", "tool": block.name, "result": result_data}
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    })

            messages.append({"role": "user", "content": tool_results})
            continue

        # end_turn 또는 그 외 모든 경우 종료
        yield {"type": "done"}
        break


REPORT_EDITOR_SYSTEM = """당신은 SK Hynix 반도체 보고서 수정 전문 AI입니다.
사용자 요청을 분석하여 보고서 데이터(d)를 수정하는 Python 코드를 생성합니다.

## 보고서 레이아웃 — 품질불량예측보고서 (한 페이지 고정)
보고서는 반드시 한 페이지(1200px 너비) 안에만 존재합니다.
추가 가능한 위치: left_col(왼쪽 컬럼 하단) 또는 right_col(오른쪽 컬럼 하단)만.

### 헤더 구조
- 좌: 발행일자 / 중앙: 품질불량예측보고서 / 우: 대외비 뱃지
- 알림 배너 (전체 너비): "전 주 대비 품질불량 ▲ ppm 이상 | [피처명] 특성 불량에 대해 Inline 원인 소급 요청"

### 왼쪽 컬럼 — 모델링 현황 보고
| 섹션 | 내용 | 데이터 |
|------|------|--------|
| L1. 요약 3열 | 모델 성능(Val RMSE) / 분석 유닛 / 예측 ppm | d["meta"]["val_rmse"], d["scan"] |
| L2. 불량률 트렌드 | LOT별 HIGH 건수 라인차트 | chart_params: {"chart":"lot"} |
| L3. SHAP 분석 (2열) | 왼쪽: bee-swarm scatter (HIGH빨강/MED파랑) / 오른쪽: Pred vs Actual scatter | d["importance"], d["analysis"] |

### 오른쪽 컬럼 — 불량 유닛 분석 현황
| 섹션 | sid | 내용 | 데이터 | 상태 |
|------|-----|------|--------|------|
| R1. 웨이퍼맵 + 대표 Unit | R1_unit | 웨이퍼맵(grade 색상) + 시리얼/LOT/WAFER/예측health/생산일자 | d["top_unit"], d["wafer_die"] | ✅ 실데이터 |
| R1b. 포지션별 예측 health | (R1 내부) | P1~P4 예측 health값 | d["top_unit"]["pos_health"] | ✅ 실데이터 |
| R2. SHAP 영향도 | R2_anomaly | 평균 SHAP 절댓값 상위 5개 피처 막대 (자동 생성, 편집 불가) | (자동) | ✅ 실데이터 |
| R3. 피처 정상/불량 분포 | R3_scatter | 선택 피처의 위험(상위10%) vs 안전(하위10%) 분포 | d["feat_dist_compare"] | ✅ 실데이터 |

## 더미 섹션 처리 규칙 (절대 원칙)
현재 보고서에서 더미인 섹션은 **없습니다**. 모든 섹션에 실데이터가 연결되어 있습니다.
절대로 실데이터 섹션을 "더미"라고 말하지 마세요.

## 수정 가능 항목
- **SHAP 영향도(R2)**: 자동 생성(SHAP 상위 5개)이라 피처 변경/필터 불가. 바꿔달라는 요청이 와도 변경하지 말고 안내만 하세요.
- **피처 정상/불량 분포(R3) 피처 변경**: 사용자가 "분포를 X831로 바꿔줘" 등으로 요청하면 그 피처로 분포를 갱신하세요. ("Anomaly"라는 표현은 절대 쓰지 마세요)
- Feature Importance 개수 변경: d.setdefault("importance",{})["features"] = d.get("importance",{}).get("features",[])[:N]

## 공간 제약
각 컬럼의 남은 공간은 약 100~150px입니다.
새 섹션 높이가 150px를 초과하면 바로 추가하지 말고 먼저 물어보세요.

## 인터랙션 이벤트 처리 (통합 차트 편집 모드)
사용자가 보고서에서 영역을 선택하면 아래 형식의 컨텍스트가 앞에 붙습니다:
```
## 현재 보고서 레이아웃 (섹션 좌표)
- [L2_trend] 주차별 수율 트렌드 : x=0, y=148, w=580, h=120
- [L3_scatter] 피처 scatter : x=0, y=276, w=580, h=240
...

## 선택 영역 (반드시 이 크기 안에 맞춰 생성)
- 위치: x=20, y=160
- 크기: 300px × 80px
- 포함 섹션: L2_trend
```
요청 유형(action)별 처리:
- **modify**: 위 선택 영역에 들어있는 기존 섹션을 수정. 새 콘텐츠 크기는 선택 영역 안에 맞춰야 함.
- **add**: 빈 영역에 새 섹션을 추가. 너비·높이는 선택 영역과 동일하게.
- **remove**: 해당 섹션을 보고서에서 제거 (`d["custom_sections"]`에서 삭제 또는 `d["hidden_sections"]`에 sid 추가).
- **explain**: HTML 변경 없이 해당 영역의 데이터를 자연어로 설명만.

## 응답 말투 규칙 (절대 원칙)
- 첫 인사말, "안녕하세요", "물론입니다", "두 가지 요청을 확인했습니다" 같은 서두 절대 금지
- 바로 핵심 내용으로 시작할 것
- 추가/수정 완료 시: "X를 Y로 변경했습니다." 한 문장으로 끝낼 것
- 새 섹션 추가 요청 시 정보 부족이면: "어떤 데이터를 어떤 형태로 추가할까요?" 한 줄로 물어볼 것
- JSON code 필드에 절대 주석(#) 넣지 말 것, 실행 가능한 코드만

## 불명확한 요청 처리 (절대 원칙)
새 섹션/차트 추가 시 아래 정보 중 하나라도 없으면 반드시 먼저 질문하고 code는 "":
1. 어떤 데이터를 넣을지 (피처명, 기간, 지표 등)
2. 어떤 형태인지 — 불명확하면: <<<BUTTONS: 막대차트, 라인차트, 표, 텍스트카드>>>
3. 피처 미지정 시 → <<<BUTTONS: {top5_features}>>>

기존 섹션 수정은 내용이 명확하면 바로 실행하세요.
기간 변경 시 주수/일수 미지정이면 반드시 물어보세요.

## 모델 정책 (고정)
- 예측에 사용되는 확정 모델: **Stacking Ensemble** (이 이름을 유지, 임의로 다른 모델명으로 바꾸지 말 것)
- SHAP 영향도 / Feature Importance 출처: **ZIT_only** (zit 단일 모델 기반)
- 사용자가 모델명을 묻거나 출처를 묻는 경우 위 사실을 답변

## 보고서 데이터 구조 (d) — 완전 명세 (키 이름 오타 절대 금지)

### 전체 키 목록
d["meta"]               - 보고서 텍스트 커스터마이징
d["scan"]               - grade별 unit 수, 집중 LOT/웨이퍼
d["importance"]         - Feature Importance  ← d["importances"] 아님
d["analysis"]           - grade4 vs grade1 분포 비교
d["top_unit"]           - 대표 Unit 정보 (R1)
d["feat_dist_compare"]  - 피처 정상/불량 분포 (R3_scatter, 표시 피처는 ["feature"])
d["weekly_yield_trend"] - 주차별 불량 트렌드 (L2)
d["pred_ppm_trend"]     - LOT별 예측 ppm 트렌드 (L3)
d["feat_scatter"]       - 피처 분포 scatter (보조)
d["wafer_die"]          - 웨이퍼맵 die 좌표
d["ppm_delta"]          - 전 주 대비 ppm 변화 (배너용)
d["pos_defect"]         - 포지션별 불량률
d["chart_params"]       - 차트 파라미터 오버라이드 {"chart": str, "top_n": str}
d["table_params"]       - 테이블 컬럼 숨기기 {"shap_hide_cols": [str,...]}
d["custom_sections"]    - 커스텀 섹션 목록
d["hidden_sections"]    - 숨길 섹션 sid 목록

### 정확한 필드명 (잘못된 이름으로 접근 시 KeyError 또는 무반응)
d["scan"]         → grade4_count (not high_count), grade4_ratio (not high_ratio),
                    grade3_count, grade2_count, grade1_count,
                    top_lot, top_lot_grade4_count,
                    top_wafer: {lot, wafer, grade4_count}, unique_lots, period
d["importance"]   → features: [{feature, lgbm_rank, lgbm_gain}, ...]
d["analysis"]     → top_features: [{feature, high_mean, low_mean, ratio, pval, importance_rank}]
d["top_unit"]     → serial, run_id, wafer_no, pred_ppm, pred_health, actual_ppm, risk,
                    pos_health: {P1, P2, P3, P4},
                    pos_feat_vals: {P1: {피처명: float}, P2: {...}, ...}
d["anomaly_stats"] → [{feature, grade1_mean, grade4_mean, ratio, z_score, danger, normal}]
                     (리스트 직접 슬라이싱: d["anomaly_stats"][:N])
d["weekly_yield_trend"] → {labels:[str], production:[int], pred_yield:[float], defect_ppm:[float]}
d["pred_ppm_trend"]     → {labels:[str], high_ppm:[float], med_ppm:[float], defect_count:[int], defect_rate:[float]}
d["feat_scatter"]  → {feat1: {name:str, pts_high:[{x,y}], pts_med:[{x,y}], threshold:float}, feat2:{...}}
d["meta"]          → {report_title:str, model:str, val_rmse:str, summary_title:str, summary_sub:str,
                       alert_features:str, section_labels:{left_header,right_header,L1,L2,L3,R1}}

## 코드 작성 규칙
✅ 사용 가능: sorted, len, max, min, sum, int, float, str, list, dict, filter, map, any, all
⛔ 절대 금지: import, exec, eval, open, __import__, subprocess, globals, getattr

## 기존 차트 수정
d["chart_params"] = {"chart": "shap", "top_n": "5"}   # SHAP 막대 상위 N개
d["chart_params"] = {"chart": "lot",  "top_n": "10"}  # 불량률 트렌드 상위 N개

## 피처 정상/불량 분포 (R3) — 표시 피처 변경
# 사용자가 "분포를 X831로 바꿔줘" 처럼 요청하면 그 피처로 분포를 갱신:
d["feat_dist_compare"] = {"feature": "X831"}   # 대시보드가 위험(상위10%)/안전(하위10%) 분포를 자동 계산
# 응답은 "피처 정상/불량 분포를 X831로 변경했습니다" 처럼. ("Anomaly"라는 표현 금지)
#
## SHAP 영향도 (R2) — 편집 불가
# SHAP 상위 5개로 자동 생성됨. 피처 변경/개수 변경 요청이 와도 코드로 바꾸지 말고 안내만.

## Feature Importance (R2) — 표시 피처 수 변경
# d["importance"]는 {"features": [{feature, lgbm_rank, lgbm_gain}, ...]} 구조
# 상위 N개만 표시하려면 features 리스트를 자른다:
feats = d.get("importance", {}).get("features", [])
d.setdefault("importance", {})["features"] = feats[:5]   # 상위 5개만

## 타이틀 / 소제목 수정
# 보고서 헤더 제목 (상단 가운데)
d.setdefault("meta", {})["report_title"] = "Field Health 불량 예측 분석 보고서"
# 요약 배너 첫째 줄 (HTML 허용, span 태그로 색상 지정 가능)
d["meta"]["summary_title"] = "전 주 대비 품질 불량 &nbsp;<span style=\"color:#EF4444\">▲163,452 ppm</span>&nbsp; <span style=\"font-size:17px;font-weight:600;color:#555\">열화</span>"
# 요약 배너 둘째 줄 (HTML 허용)
d["meta"]["summary_sub"] = "원인 WT Parameter&nbsp;<span style=\"background:#fef3c7;color:#92400e;padding:1px 7px;font-size:15px;font-weight:800\">X1064, X592</span>&nbsp;이상 → inline 참원인 도출 요청"
# 원인 피처만 바꾸려면 (summary_sub 자동 재생성됨):
d["meta"]["alert_features"] = "X1064, X592"
# 좌측/우측 패널 헤더
d["meta"].setdefault("section_labels", {})["left_header"]  = "[ 모델링 결과 ]"
d["meta"]["section_labels"]["right_header"] = "[ 불량 예측 현황 ]"
# 섹션 소제목 (번호는 자동 유지, 텍스트만 교체)
d["meta"]["section_labels"]["L1"] = "모델 성능"
d["meta"]["section_labels"]["L2"] = "불량 트렌드"
d["meta"]["section_labels"]["L3"] = "Lot별 불량 개수"
d["meta"]["section_labels"]["L4"] = "주요 피처 임계값 분포"
d["meta"]["section_labels"]["R1"] = "불량 예측 현황 · 대표 불량 unit 기준"

## 주차별 트렌드 (L2/L3) — 표시 주수 변경
# d["weekly_yield_trend"]는 {"labels","ww_labels","date_labels","production",
#   "pred_yield","defect_ppm","true_ppm"} 구조.
# ⚠ 반드시 모든 list 필드를 함께 tail 한다. defect_ppm/ww_labels 를 빠뜨리면
#   ppm 값이 pred_yield에서 역산되어 대시보드와 어긋난다.
# 예) 최근 3주:
n = 3
wyt = d.get("weekly_yield_trend", {})
d["weekly_yield_trend"] = {k: (v[-n:] if isinstance(v, list) else v) for k, v in wyt.items()}

## SHAP 분석 테이블 컬럼 숨기기/복원
# 숨길 수 있는 컬럼명: "feature", "high_mean", "low_mean", "ratio", "pval"
d.setdefault("table_params", {})["shap_hide_cols"] = ["pval"]          # p-value 숨기기
d.setdefault("table_params", {})["shap_hide_cols"] = ["pval", "ratio"] # 여러 개 숨기기
d.setdefault("table_params", {})["shap_hide_cols"] = []                # 전체 복원

## 커스텀 섹션 추가 (left_col / right_col만)
d.setdefault("custom_sections", []).append({
  "title": "섹션 제목",
  "position": "left_col",      # left_col 또는 right_col만
  "chart_type": "bar",         # bar, line, table, pie, doughnut
  "labels": [...],
  "datasets": [{"label": "시리즈", "data": [...], "color": "#3B82F6"}],
  "height": 120,               # 150px 이하 권장
  "horizontal": False,
})

## 테이블 추가
d.setdefault("custom_sections", []).append({
  "title": "표 제목",
  "position": "right_col",
  "chart_type": "table",
  "columns": ["항목", "값"],
  "rows": [["Val RMSE", d["meta"].get("val_rmse", "-")], ["불량 수량", d["scan"].get("high_count", "-")]],
  "height": 120,
})

## 커맨드 방식 처리 (최우선 — 아래 작업은 반드시 action JSON 사용)

알려진 작업은 `code` 대신 `action` 필드를 사용한다. action 방식은 사전 검증된 핸들러가 실행하므로 100% 안정적이다.

| 요청 유형 | action JSON 형식 |
|----------|----------------|
| importance top-N 변경 | `{"action":"set_top_n","section":"importance","n":5,"response":"..."}` |
| SHAP 영향도(R2) top-N 변경 | `{"action":"set_top_n","section":"shap","n":8,"response":"..."}` |
| 불량 트렌드 기간/주수 변경 (L3) | `{"action":"set_top_n","section":"trend_weeks","n":4,"response":"..."}` |
| LOT ppm 트렌드 기간 변경 | `{"action":"set_top_n","section":"trend_lots","n":10,"response":"..."}` |
| 피처 정상/불량 분포(R3) 피처 변경 | `code`로 `d["feat_dist_compare"] = {"feature": "X831"}` (action 아님, 코드 사용) |
| 보고서 제목 변경 | `{"action":"set_text","key":"report_title","value":"새 제목","response":"..."}` |
| 배너 피처명 변경 | `{"action":"set_text","key":"alert_features","value":"X1064, X592","response":"..."}` |
| 섹션 소제목 변경 | `{"action":"set_text","key":"section_label_L3","value":"새 소제목","response":"..."}` |
| 배너 첫째 줄 직접 수정 | `{"action":"set_text","key":"summary_title","value":"내용","response":"..."}` |
| 배너 둘째 줄 직접 수정 | `{"action":"set_text","key":"summary_sub","value":"내용","response":"..."}` |
| 섹션 숨기기 | `{"action":"toggle_section","sid":"R2_anomaly","hide":true,"response":"..."}` |
| 섹션 복원 | `{"action":"toggle_section","sid":"R2_anomaly","hide":false,"response":"..."}` |
| 대표 유닛 변경 | `{"action":"change_unit","serial":"S38369","response":"..."}` |

section 값: `"importance"` | `"trend_weeks"` | `"trend_lots"`
sid 값: `"L1_kpi"` | `"L2_fi"` | `"L3_trend"` | `"R1_unit"` | `"R2_anomaly"` | `"R3_scatter"`
section_label key 예시: `"section_label_L1"` | `"section_label_L2"` | `"section_label_L3"` | `"section_label_R1"`

위 목록에 없는 창의적인 요청(커스텀 섹션 추가, 테이블 컬럼 숨기기 등)은 기존 `code` 방식 사용.

## 응답 형식

**중요**: 응답은 반드시 JSON 객체 하나만 출력한다. 코드블록(```) 없이, JSON 앞뒤로 설명 텍스트를 쓰지 않는다.

커맨드 방식: `{"action": "...", ...필드들..., "response": "사용자에게 전달할 한국어 답변"}`
코드 방식:   `{"response": "사용자에게 전달할 한국어 답변", "code": "Python 코드 (없으면 빈 문자열)"}`

잘못된 예시 (절대 금지):
- "네, 처리하겠습니다. {\"action\": ...}" ← JSON 앞에 텍스트 금지
- ```json\n{...}\n``` ← 코드블록 금지
- 응답 없이 JSON만: {"action": "...", "response": ""} ← response는 항상 한국어로 작성"""


def _get_chart_section_data(chart_type: str, d: dict, position: str) -> dict | None:
    """chart_type 문자열로 custom_section dict 생성."""
    if chart_type == "importance":
        features = d.get("importance", {}).get("features", [])[:10]
        total = sum(f.get("lgbm_gain", 0) or 0 for f in features) or 1
        labels = [f.get("feature", "") for f in features]
        data   = [round((f.get("lgbm_gain", 0) or 0) / total * 100, 2) for f in features]
        return {"title": "Feature Importance", "chart_type": "bar", "position": position,
                "labels": labels, "horizontal": True, "height": 150,
                "datasets": [{"label": "Gain %", "data": data, "color": "#3B82F6"}]}

    elif chart_type == "shap":
        items  = get_shap_bar_top(int(d.get("shap_top_n", 5) or 5))
        labels = [it["feature"] for it in items]
        data   = [round(it["mag"] * 1e6, 1) for it in items]   # ppm 스케일
        colors = ["#EF4444" if it.get("signed", 0) >= 0 else "#3B82F6" for it in items]
        return {"title": "SHAP 영향도", "chart_type": "bar", "position": position,
                "labels": labels, "horizontal": True, "height": 150,
                "datasets": [{"label": "평균 |SHAP| (ppm)", "data": data,
                              "color": "#3B82F6", "colors": colors}]}

    elif chart_type == "feat_dist":
        fdc = d.get("feat_dist_compare") or {}
        if not fdc.get("labels"):
            try: fdc = get_feature_dist_compare()
            except Exception: fdc = {}
        return {"title": f"피처 정상/불량 분포 · {fdc.get('feature','')}", "chart_type": "line",
                "position": position, "labels": fdc.get("labels", []), "height": 150,
                "datasets": [
                    {"label": "안전(하위10%)", "data": fdc.get("normal", []), "color": "#3B82F6"},
                    {"label": "위험(상위10%)", "data": fdc.get("danger", []), "color": "#EF4444"},
                ]}

    elif chart_type == "defect_trend":
        wyt    = d.get("weekly_yield_trend", {})
        labels = wyt.get("labels", [])
        ppm    = wyt.get("defect_ppm", [])
        return {"title": "불량 트렌드 (예측 ppm)", "chart_type": "line", "position": position,
                "labels": labels, "height": 150,
                "datasets": [{"label": "예측불량 ppm", "data": ppm, "color": "#3B82F6"}]}

    elif chart_type == "anomaly":
        stats = d.get("anomaly_stats", [])[:10]
        labels = [s.get("feature", "") for s in stats]
        danger = [s.get("danger", 0) for s in stats]
        normal = [s.get("normal", 100) for s in stats]
        return {"title": "Anomaly Feature 비교", "chart_type": "bar", "position": position,
                "labels": labels, "horizontal": False, "height": 150,
                "datasets": [{"label": "위험 %", "data": danger, "color": "#EF4444"},
                             {"label": "정상 %", "data": normal, "color": "#22C55E"}]}

    elif chart_type == "lot_trend":
        trend = d.get("lot_trend_split", {})
        labels = trend.get("labels", [])
        high   = trend.get("high", trend.get("high_count", trend.get("defect_count", [])))
        return {"title": "LOT별 HIGH 건수", "chart_type": "line", "position": position,
                "labels": labels, "horizontal": False, "height": 150,
                "datasets": [{"label": "HIGH 건수", "data": high, "color": "#F59E0B"}]}

    elif chart_type == "weekly_trend":
        wyt = d.get("weekly_yield_trend", {})
        labels = wyt.get("labels", [])
        yield_ = wyt.get("pred_yield", wyt.get("yield_pct", []))
        return {"title": "주차별 수율 트렌드", "chart_type": "line", "position": position,
                "labels": labels, "horizontal": False, "height": 150,
                "datasets": [{"label": "수율(%)", "data": yield_, "color": "#6366F1"}]}

    elif chart_type == "ppm_trend":
        ppm = d.get("pred_ppm_trend", {})
        labels = ppm.get("labels", [])
        mean_d = ppm.get("mean_ppm", ppm.get("med_ppm", []))
        top_d  = ppm.get("top_ppm",  ppm.get("high_ppm", []))
        return {"title": "LOT별 예측 ppm (평균/상위5%)", "chart_type": "line", "position": position,
                "labels": labels, "horizontal": False, "height": 150,
                "datasets": [
                    {"label": "LOT 평균",  "data": mean_d, "color": "#3B82F6"},
                    {"label": "상위 5%",   "data": top_d,  "color": "#EF4444"},
                ]}

    elif chart_type == "pos_defect":
        pd_ = d.get("pos_defect", {})
        labels = pd_.get("labels", ["P1", "P2", "P3", "P4"])
        high   = pd_.get("high_ratio", [])
        med    = pd_.get("med_ratio",  [])
        return {"title": "포지션별 불량률", "chart_type": "bar", "position": position,
                "labels": labels, "horizontal": False, "height": 150,
                "datasets": [{"label": "HIGH", "data": high, "color": "#EF4444"},
                             {"label": "MED",  "data": med,  "color": "#F59E0B"}]}

    elif chart_type == "pred_actual":
        try:
            pts = get_pred_actual_data(max_pts=100)
        except Exception:
            pts = []
        pts_sorted = sorted(pts, key=lambda p: p["x"])
        labels      = [str(i + 1) for i in range(len(pts_sorted))]
        pred_vals   = [round(p["x"] * 1e6, 2) for p in pts_sorted]
        actual_vals = [round(p["y"] * 1e6, 2) for p in pts_sorted]
        return {"title": "예측 vs 실측 health (ppm)", "chart_type": "line", "position": position,
                "labels": labels, "horizontal": False, "height": 150,
                "datasets": [
                    {"label": "예측 health",  "data": pred_vals,   "color": "#3B82F6"},
                    {"label": "실측 health",  "data": actual_vals, "color": "#EF4444"},
                ]}

    elif chart_type == "grade_dist":
        scan = d.get("scan", {})
        return {"title": "Grade 분포", "chart_type": "doughnut", "position": position,
                "labels": ["정상 (G1)", "조심 (G2)", "위험 (G3)", "매우위험 (G4)"],
                "height": 150,
                "datasets": [{"label": "unit 수",
                               "data": [scan.get("grade1_count", 0), scan.get("grade2_count", 0),
                                        scan.get("grade3_count", 0), scan.get("grade4_count", 0)],
                               "color": "#22C55E",
                               "colors": ["#22C55E", "#EAB308", "#F59E0B", "#EF4444"]}]}

    elif chart_type == "location_ppm":
        try:
            lp = get_location_ppm_top(top_n=10)
        except Exception:
            lp = {}
        return {"title": "위치별 평균 예측 ppm Top 10", "chart_type": "bar", "position": position,
                "labels": lp.get("labels", []), "horizontal": True, "height": 150,
                "datasets": [
                    {"label": "평균 ppm", "data": lp.get("ppm", []), "color": "#EF4444"},
                ]}

    elif chart_type == "lot_grade_stack":
        try:
            lgs = get_lot_grade_stack()
        except Exception:
            lgs = {}
        return {"title": "LOT별 Grade 구성", "chart_type": "bar", "position": position,
                "labels": lgs.get("labels", []), "height": 150, "stacked": True,
                "datasets": [
                    {"label": "정상 (G1)",    "data": lgs.get("g1", []), "color": "#22C55E"},
                    {"label": "조심 (G2)",    "data": lgs.get("g2", []), "color": "#EAB308"},
                    {"label": "위험 (G3)",    "data": lgs.get("g3", []), "color": "#F59E0B"},
                    {"label": "매우위험 (G4)","data": lgs.get("g4", []), "color": "#EF4444"},
                ]}

    elif chart_type == "health_hist":
        try:
            hh = get_pred_health_hist()
        except Exception:
            hh = {}
        return {"title": "예측 Health 분포", "chart_type": "bar", "position": position,
                "labels": hh.get("labels", []), "height": 150,
                "datasets": [
                    {"label": "전체 unit", "data": hh.get("counts",      []), "color": "#93C5FD"},
                    {"label": "위험 unit (상위10%)", "data": hh.get("high_counts", []), "color": "#EF4444"},
                ]}

    elif chart_type == "feat_vs_health":
        try:
            fvh = get_feat_vs_health_scatter()
        except Exception:
            fvh = {}
        return {"title": "이상 점수 vs 예측 Health", "chart_type": "scatter", "position": position,
                "labels": [], "height": 150,
                "datasets": [
                    {"label": "위험 (G3+G4)", "data": fvh.get("high_pts",   [])[:150], "color": "#EF4444"},
                    {"label": "정상 (G1+G2)", "data": fvh.get("normal_pts", [])[:150], "color": "#22C55E"},
                ]}

    elif chart_type == "feat_scatter_lot":
        # 이상 피처 분포 (LOT 순서) — pts: {x:피처값, y:reg_pred, lot:run_id}
        try:
            fs = get_feature_scatter_data()
        except Exception:
            fs = {}
        f1 = fs.get("feat1", {}); fname = f1.get("name", "Feature")
        high = f1.get("pts_high", []); med = f1.get("pts_med", [])
        # x=LOT, y=피처값
        high_data = [{"x": p.get("lot"), "y": p.get("x")} for p in high if p.get("lot") is not None]
        med_data  = [{"x": p.get("lot"), "y": p.get("x")} for p in med  if p.get("lot") is not None]
        return {"title": f"이상 피처 분포 · {fname} (LOT 순서)", "chart_type": "scatter", "position": position,
                "labels": [], "height": 150,
                "datasets": [
                    {"label": "위험(G3+G4) 피처값", "data": high_data, "color": "#EF4444"},
                    {"label": "정상(G1+G2) 피처값", "data": med_data,  "color": "#22C55E"},
                ]}

    elif chart_type == "top_risk_units":
        try:
            tru = get_top_risk_units(top_n=10)
        except Exception:
            tru = {}
        return {"title": "위험 Unit Top 10 (예측 ppm)", "chart_type": "bar", "position": position,
                "labels": tru.get("labels", []), "horizontal": True, "height": 150,
                "datasets": [
                    {"label": "예측 ppm", "data": tru.get("ppm", []), "color": "#EF4444"},
                ]}

    elif chart_type == "lot_mean_ppm":
        try:
            lmp = get_lot_mean_ppm_top(top_n=5)
        except Exception:
            lmp = {}
        return {"title": "LOT별 평균 예측 ppm Top 5", "chart_type": "bar", "position": position,
                "labels": lmp.get("labels", []), "horizontal": True, "height": 150,
                "datasets": [
                    {"label": "평균 ppm", "data": lmp.get("ppm", []), "color": "#F59E0B"},
                ]}

    elif chart_type == "wafer_risk_ratio":
        try:
            wrr = get_wafer_risk_die_ratio_top(top_n=10)
        except Exception:
            wrr = {}
        return {"title": "웨이퍼별 위험 die 비율 Top 10 (%)", "chart_type": "bar", "position": position,
                "labels": wrr.get("labels", []), "horizontal": True, "height": 150,
                "datasets": [
                    {"label": "위험 die 비율(%)", "data": wrr.get("ratio", []), "color": "#EF4444"},
                ]}

    return None


def _handle_command(cmd: dict, d: dict):
    """
    Claude가 action JSON을 반환했을 때 처리. (success: bool, err_msg: str | None) 반환.
    성공 시 err_msg=None, 실패 시 err_msg=오류 설명.
    """
    action = cmd.get("action", "")

    if action == "set_top_n":
        section = cmd.get("section", "")
        try:
            n = max(1, int(cmd.get("n", 5)))
        except (TypeError, ValueError):
            return False, "n 값이 올바르지 않습니다"
        if section == "importance":
            pool = d.get("importance_all") or d.get("importance", {}).get("features", [])
            if n > len(pool):
                # 풀이 부족하면 재조회
                try:
                    fresh = get_importance(top_n=max(n, 20)).get("features", [])
                    if fresh:  # 빈 결과면 기존 풀 유지
                        d["importance_all"] = fresh
                        pool = fresh
                except Exception:
                    pass
            if n > len(pool):
                return False, f"현재 최대 {len(pool)}개까지만 가능합니다"
            d.setdefault("importance", {})["features"] = pool[:n]
        elif section == "anomaly":
            # anomaly 섹션은 SHAP로 대체됨 (compet_xs 의존 제거) — 기존 풀만 사용
            source = d.get("anomaly_stats_all") or []
            if n > len(source):
                return False, f"현재 최대 {len(source)}개까지 가능합니다 (풀 크기 {len(source)}개)"
            d["anomaly_stats"] = source[:n]
        elif section == "shap":
            d["shap_top_n"] = n
        elif section == "trend_weeks":
            d["weekly_yield_trend"] = get_weekly_yield_trend(recent_weeks=n)
        elif section == "trend_lots":
            d["pred_ppm_trend"] = get_pred_ppm_trend(recent_n=n)
        else:
            return False, f"알 수 없는 section: {section}"
        return True, None

    elif action == "filter_anomaly":
        features = cmd.get("features", [])
        if not features:
            return False, "features 목록이 비어있습니다"
        source = d.get("anomaly_stats_all") or d.get("anomaly_stats", [])
        d["anomaly_stats"] = [s for s in source if s.get("feature") in features]
        return True, None

    elif action == "change_scatter":
        feat1 = cmd.get("feat1", "")
        feat2 = cmd.get("feat2", "")
        if not feat1 and not feat2:
            return False, "변경할 피처명(X숫자 형식)을 알려주세요."
        # 하나만 지정된 경우 기존 값 유지
        existing = d.get("feat_scatter", {})
        if not feat1:
            feat1 = existing.get("feat1", {}).get("name", "")
        if not feat2:
            feat2 = existing.get("feat2", {}).get("name", "")
        if not feat1 or not feat2:
            return False, "기존 피처 정보가 없습니다. feat1, feat2 모두 지정해 주세요."
        try:
            d["feat_scatter"] = get_feature_scatter_data(feat1=feat1, feat2=feat2)
            # R3 피처 정상/불량 분포 차트도 같은 feature(feat1)로 갱신
            d["feat_dist_compare"] = get_feature_dist_compare(feature=feat1)
        except Exception as e:
            return False, str(e)
        return True, None

    elif action == "set_dist_feature":
        # 피처 정상/불량 분포(R3) 표시 피처 변경
        feature = cmd.get("feature", "")
        if not feature:
            return False, "변경할 피처명(X숫자 형식)을 알려주세요."
        try:
            d["feat_dist_compare"] = get_feature_dist_compare(feature=feature)
        except Exception as e:
            return False, str(e)
        return True, None

    elif action == "set_text":
        key = cmd.get("key", "")
        value = str(cmd.get("value", ""))
        if key == "report_title":
            d.setdefault("meta", {})["report_title"] = value
        elif key == "alert_features":
            d.setdefault("meta", {})["alert_features"] = value
        elif key == "summary_title":
            d.setdefault("meta", {})["summary_title"] = value
        elif key == "summary_sub":
            d.setdefault("meta", {})["summary_sub"] = value
        elif key.startswith("section_label_"):
            label_key = key[len("section_label_"):]
            d.setdefault("meta", {}).setdefault("section_labels", {})[label_key] = value
        else:
            return False, f"알 수 없는 key: {key}"
        return True, None

    elif action == "toggle_section":
        sid = cmd.get("sid", "")
        if not sid:
            return False, "sid 미지정"
        hide = cmd.get("hide", True)
        hidden = d.setdefault("hidden_sections", [])
        if hide:
            if sid not in hidden:
                hidden.append(sid)
        else:
            d["hidden_sections"] = [s for s in hidden if s != sid]
        return True, None

    elif action == "change_unit":
        serial = cmd.get("serial", "")
        if not serial:
            return False, "serial 미지정"
        try:
            d["top_unit"] = get_top_unit_data(serial=serial)
            d["wafer_die"] = get_wafer_die_data(serial=serial)
            d["unit_shap"] = get_unit_shap_bar(serial, 10)   # R2 SHAP도 해당 유닛으로
            # R3 분포차트도 해당 유닛 SHAP 1등 피처로 동기화
            _fdc = _feat_dist_for_shap(d["unit_shap"])
            if _fdc:
                d["feat_dist_compare"] = _fdc
        except Exception as e:
            return False, str(e)
        return True, None

    elif action == "change_section":
        target_sid = cmd.get("target_sid", "")
        chart_type = cmd.get("chart_type", "")
        if not target_sid or not chart_type:
            return False, "target_sid, chart_type 모두 지정해야 합니다"
        left_sids = {"L1_kpi", "L2_fi", "L3_trend"}
        position = "left_col" if target_sid in left_sids else "right_col"
        sec = _get_chart_section_data(chart_type, d, position)
        if sec is None:
            return False, f"알 수 없는 차트 유형: {chart_type}"
        sec["replace_sid"] = target_sid
        # 같은 target_sid 교체 이력 제거 후 추가 (중복 방지)
        d["custom_sections"] = [s for s in d.get("custom_sections", [])
                                if s.get("replace_sid") != target_sid]
        d["custom_sections"].append(sec)
        return True, None

    else:
        return False, f"알 수 없는 action: {action}"


def _execute_editor_code(code: str, d: dict):
    """Claude가 생성한 코드를 안전하게 실행. (success, error_msg) 반환."""
    if not code.strip():
        return True, ""

    import ast as _ast
    try:
        tree = _ast.parse(code)
    except SyntaxError as e:
        return False, f"문법 오류: {e}"
    forbidden_names = {"__import__", "exec", "eval", "compile", "globals", "getattr", "open"}
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Import):
            return False, "금지된 표현: import"
        if isinstance(node, _ast.ImportFrom):
            return False, "금지된 표현: import"
        if isinstance(node, (_ast.Name, _ast.Attribute)):
            name = node.id if isinstance(node, _ast.Name) else node.attr
            if name in forbidden_names:
                return False, f"금지된 표현: {name}"
        if isinstance(node, _ast.Constant) and isinstance(node.value, str):
            if "subprocess" in node.value:
                return False, "금지된 표현: subprocess"

    safe_builtins = {
        "sorted": sorted, "len": len, "max": max, "min": min, "sum": sum,
        "int": int, "float": float, "str": str, "list": list, "dict": dict,
        "filter": filter, "map": map, "any": any, "all": all, "zip": zip,
        "enumerate": enumerate, "range": range, "round": round,
        "True": True, "False": False, "None": None,
        "print": lambda *a, **k: None,
    }
    try:
        exec(code, {"__builtins__": safe_builtins, "d": d})
        return True, ""
    except Exception as e:
        return False, str(e)


def _try_direct_action(message: str, d: dict):
    """
    사용자 메시지에서 known action을 직접 추출 시도. Claude 호출 없이 처리.
    반환: (cmd: dict | None, question: str | None)
      - (cmd, None)      : 액션이 명확히 결정됨 → 바로 실행
      - (None, question) : 파라미터가 불명확 → 질문 반환
      - (None, None)     : 패턴 불일치 → Claude에게 위임
    """
    import re as _re, json as _json

    msg = message.strip()

    # ── __direct__: prefix — iframe 액션 버튼에서 직접 전달된 명령
    if msg.startswith("__direct__:"):
        try:
            cmd_raw = _json.loads(msg[len("__direct__:"):])
        except Exception:
            return None, None
        action = cmd_raw.get("action", "")
        if action == "remove":
            sid = cmd_raw.get("sid", "")
            if sid:
                return {"action": "toggle_section", "sid": sid, "hide": True, "response": "섹션을 삭제했습니다."}, None
        elif action == "change_chart":
            target_sid = cmd_raw.get("target_sid", "")
            chart_type = cmd_raw.get("chart_type", "")
            if target_sid and chart_type:
                chart_names = {
                    "importance": "Feature Importance", "shap": "SHAP 영향도",
                    "feat_dist": "피처 정상/불량 분포", "defect_trend": "불량 트렌드",
                    "lot_trend": "LOT 트렌드", "weekly_trend": "주차별 수율",
                    "ppm_trend": "LOT ppm", "pos_defect": "포지션별 불량률",
                    "pred_actual": "예측 vs 실측", "location_ppm": "위치별 평균 ppm",
                    "health_hist": "예측 Health 분포", "top_risk_units": "위험 Unit Top 10",
                    "lot_mean_ppm": "LOT 평균 ppm Top 5", "wafer_risk_ratio": "웨이퍼 위험 die 비율",
                }
                name = chart_names.get(chart_type, chart_type)
                return {"action": "change_section", "target_sid": target_sid,
                        "chart_type": chart_type, "response": f"{name} 차트 변경 완료"}, None
        return None, None

    # ── 숫자 추출 (X피처명 제거 후 추출 — 한글이 \w라 \b가 안 먹히는 문제 우회)
    _clean = _re.sub(r'[Xx]\d+', '', msg)   # X1064 등 피처명 제거
    nums = [int(x) for x in _re.findall(r'\d+', _clean)]
    n = nums[0] if nums else None

    # ── 섹션 판별 키워드
    is_importance = bool(_re.search(r'importance|중요도|feature\s*importance', msg, _re.IGNORECASE))
    is_shap       = bool(_re.search(r'shap|영향도', msg, _re.IGNORECASE))
    is_trend_lot  = bool(_re.search(r'lot\s*수|로트\s*수|pred_ppm|R3|trend_lot', msg, _re.IGNORECASE))
    is_trend_week = bool(_re.search(r'주\s*수|주차\s*수|L2|L3|trend_week|주간|불량\s*트렌드|트렌드\s*(차트|기간|수정|변경|주)', msg, _re.IGNORECASE))
    is_period_chg = bool(_re.search(r'기간|날짜|범위|주로|주\s*로|주\s*만|주\s*보|최근\s*\d', msg, _re.IGNORECASE))

    # ── 액션 판별 키워드
    is_topn   = bool(_re.search(r'개로|개\s*(로|만|변경|바꿔|줄여|늘려)|줄여|늘려|줄이|늘리|top[\s-]?(?:\d+|n)|상위\s*\d+', msg, _re.IGNORECASE))
    is_hide   = bool(_re.search(r'숨겨|숨기|제거|빼줘|빼\s*줘|없애|안\s*보이', msg, _re.IGNORECASE))
    is_show   = bool(_re.search(r'보여|복원|다시\s*보|표시\s*해|보이게', msg, _re.IGNORECASE))
    is_serial = bool(_re.search(r'S\d{4,}', msg))

    # ── 피처명 추출 (X숫자 형식, \b 안 씀 — 한글이 \w라 뒤에 \b 안 먹힘)
    feats = [f.upper() for f in _re.findall(r'[Xx]\d+', msg)]

    # ── 트렌드 기간 변경 요청 (숫자 없이 기간 언급) → 바로 질문
    if (is_trend_week or is_trend_lot) and is_period_chg and n is None:
        return None, "불량 트렌드 차트를 최근 몇 주로 바꿔드릴까요? (예: 4주, 10주, 12주)"

    # ── 트렌드 기간 변경 요청 (숫자 있음)
    if (is_trend_week or is_trend_lot) and is_period_chg and n is not None:
        if is_trend_week and not is_trend_lot:
            return {"action": "set_top_n", "section": "trend_weeks", "n": n, "response": f"불량 트렌드를 최근 {n}주로 변경했습니다."}, None
        if is_trend_lot and not is_trend_week:
            return {"action": "set_top_n", "section": "trend_lots",  "n": n, "response": f"LOT ppm 트렌드를 최근 {n}개로 변경했습니다."}, None

    # ── SHAP 영향도 차트 Top N 변경 ("SHAP 영향도 top8로")
    if is_shap and n is not None:
        return {"action": "set_top_n", "section": "shap", "n": n,
                "response": f"SHAP 영향도 차트를 Top {n}로 변경했습니다."}, None

    # ── set_top_n 감지
    if is_topn and n is not None:
        section_count = sum([is_importance, is_trend_lot, is_trend_week])
        if section_count == 1:
            if is_importance:
                return {"action": "set_top_n", "section": "importance",   "n": n, "response": f"Feature Importance를 {n}개로 변경했습니다."}, None
            if is_trend_lot:
                return {"action": "set_top_n", "section": "trend_lots",   "n": n, "response": f"트렌드 LOT 수를 {n}개로 변경했습니다."}, None
            if is_trend_week:
                return {"action": "set_top_n", "section": "trend_weeks",  "n": n, "response": f"트렌드를 최근 {n}주로 변경했습니다."}, None
        elif section_count == 0:
            return None, f"{n}개로 변경할 섹션을 알려주세요. Feature Importance, 트렌드 주수 중 어느 쪽인가요?"
        else:
            return None, f"어느 섹션을 {n}개로 변경할지 선택해 주세요."

    # ── 피처 정상/불량 분포(R3) 피처 변경: "분포를 X831로 바꿔줘"
    is_dist = bool(_re.search(r'분포|distribution', msg, _re.IGNORECASE))
    if feats and is_dist:
        return {"action": "set_dist_feature", "feature": feats[0],
                "response": f"피처 정상/불량 분포를 {feats[0]}로 변경했습니다."}, None

    # ── change_unit 감지 (serial 번호 명시)
    if is_serial:
        serial = _re.search(r'S\d{4,}', msg).group()
        return {"action": "change_unit", "serial": serial, "response": f"{serial} 유닛으로 변경했습니다."}, None

    # ── 유닛 변경 요청인데 serial 미지정 → 후보 유닛 버튼 제시 (run_report_editor가 처리)
    is_unit = bool(_re.search(r'유닛|unit|대표', msg, _re.IGNORECASE))
    is_unit_change = bool(_re.search(r'변경|바꿔|바꾸|교체|선택|골라|다른', msg, _re.IGNORECASE))
    if is_unit and is_unit_change:
        return {"action": "__ask_unit__"}, None

    # ── toggle_section 감지
    sid_map = {
        'L1': 'L1_kpi', 'L2': 'L2_fi', 'L3': 'L3_trend',
        'R1': 'R1_unit', 'R2': 'R2_anomaly', 'R3': 'R3_scatter',
    }
    for label, sid in sid_map.items():
        if label in msg:
            if is_hide:
                return {"action": "toggle_section", "sid": sid, "hide": True,  "response": f"{label} 섹션을 숨겼습니다."}, None
            if is_show:
                return {"action": "toggle_section", "sid": sid, "hide": False, "response": f"{label} 섹션을 복원했습니다."}, None

    return None, None


async def run_report_editor(user_message: str, history: list,
                            tool_cache: dict = None, current_html: str = "",
                            current_report_data: dict = None):
    """
    보고서 수정 전용 Agent. Claude가 Python 코드를 생성하면 exec()으로 실행.
    current_report_data: 이전 수정이 누적된 보고서 데이터 (custom_sections, commentary 포함)
    """
    import copy
    cache = dict(tool_cache) if tool_cache else {}

    # current_report_data가 있으면 그걸 기반으로, 없으면 tool_cache로 초기화
    if current_report_data and isinstance(current_report_data, dict) and current_report_data:
        d = copy.deepcopy(current_report_data)
        d.setdefault("meta", {"title": "Field Health 불량 원인 분석 보고서", "model": "Stacking Ensemble"})
        if not d.get("scan"):
            d["scan"] = cache.get("scan_data", {})
        if not d.get("importance"):
            d["importance"] = cache.get("get_importance", {})
        d.setdefault("analysis", {})   # analyze_features(compet_xs) 의존 제거
        d.setdefault("actions",         [])
        d.setdefault("chart_params",    {})
        d.setdefault("custom_sections", [])
        d.setdefault("commentary",      [])
        # 신규 실데이터 키 — 없으면 새로 조회
        if not d.get("top_unit"):
            try: d["top_unit"] = get_top_unit_data()
            except Exception: d["top_unit"] = {}
        if not d.get("unit_shap"):
            try: d["unit_shap"] = get_unit_shap_bar(d.get("top_unit", {}).get("serial", ""), 10)
            except Exception: d["unit_shap"] = []
        if not d.get("candidate_units"):
            try: d["candidate_units"] = get_candidate_units(5)
            except Exception: d["candidate_units"] = []
        if not d.get("wafer_die"):
            try: d["wafer_die"] = get_wafer_die_data()
            except Exception: d["wafer_die"] = {}
        if not d.get("pos_defect"):
            try: d["pos_defect"] = get_position_defect_rate()
            except Exception: d["pos_defect"] = {}
        if not d.get("ppm_delta"):
            try: d["ppm_delta"] = get_ppm_delta()
            except Exception: d["ppm_delta"] = {}
        if not d.get("feat_scatter"):
            try: d["feat_scatter"] = get_feature_scatter_data()
            except Exception: d["feat_scatter"] = {}
        if not d.get("lot_trend_split"):
            try: d["lot_trend_split"] = get_lot_trend_with_split()
            except Exception: d["lot_trend_split"] = {}
        if not d.get("weekly_yield_trend"):
            try: d["weekly_yield_trend"] = get_weekly_yield_trend(recent_weeks=10)
            except Exception: d["weekly_yield_trend"] = {}
        if not d.get("recent_lot_trend"):
            try: d["recent_lot_trend"] = get_recent_lot_trend(recent_n=35)
            except Exception: d["recent_lot_trend"] = {}
        if not d.get("pred_ppm_trend"):
            try: d["pred_ppm_trend"] = get_pred_ppm_trend(recent_n=20)
            except Exception: d["pred_ppm_trend"] = {}
        # anomaly_stats(get_anomaly_feature_stats, compet_xs 의존) 제거 — R2 패널은 SHAP로 대체
        d.setdefault("anomaly_stats_all", [])
        d.setdefault("anomaly_stats", [])
        # importance_all: Feature Importance 전체 풀 (top20)
        if not d.get("importance_all"):
            try: d["importance_all"] = get_importance(top_n=20).get("features", [])
            except Exception: d["importance_all"] = list(d.get("importance", {}).get("features", []))
    else:
        d = _build_report_data(cache)

    # ── Pre-router: known action 직접 처리 (Claude 호출 없이) ───────
    direct_cmd, clarify_q = _try_direct_action(user_message, d)

    # 유닛 변경 요청(serial 미지정) → 후보 유닛 버튼 제시
    if direct_cmd and direct_cmd.get("action") == "__ask_unit__":
        _cands = d.get("candidate_units") or []
        if not _cands:
            try: _cands = get_candidate_units(5)
            except Exception: _cands = []
        if _cands:
            _lines = ["변경할 **대표 유닛**을 선택하세요 (예측 ppm 높은 순):"]
            for c in _cands:
                _lines.append(f"- {c['serial']} · LOT{c['lot']}-WF{c['wafer']} · {c['ppm']:,.0f} ppm")
            yield {"type": "text", "content": "\n".join(_lines)}
            yield {"type": "confirm", "buttons": [c["serial"] for c in _cands]}
        else:
            yield {"type": "text", "content": "변경할 유닛 serial(예: S22474)을 알려주세요."}
        yield {"type": "done"}
        return

    if clarify_q:
        # 파라미터 불명확 → 질문만 반환 (보고서 수정 없음)
        yield {"type": "text", "content": clarify_q}
        yield {"type": "done"}
        return

    if direct_cmd:
        # 액션 확정 → 바로 실행
        success, err_msg = _handle_command(direct_cmd, d)
        if not success:
            yield {"type": "text", "content": f"⚠️ 처리 실패: {err_msg}"}
        else:
            try:
                html = build_html(d)
            except Exception as _e:
                yield {"type": "text", "content": f"⚠️ HTML 생성 오류: {_e}"}
                yield {"type": "done"}
                return
            yield {"type": "report_ready", "html": html, "report_data": copy.deepcopy(d)}
            yield {"type": "text", "content": direct_cmd.get("response", "보고서를 수정했습니다.")}
        yield {"type": "done"}
        return
    # ─────────────────────────────────────────────────────────────────

    clean_history = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "")
        if not isinstance(content, str) or not content.strip():
            continue
        if clean_history and clean_history[-1]["role"] == role:
            continue
        clean_history.append({"role": role, "content": content.strip()})
    if clean_history and clean_history[-1]["role"] == "assistant":
        clean_history = clean_history[:-1]

    messages = clean_history + [{"role": "user", "content": user_message}]

    # 현재 d 상태를 system prompt에 주입 → Claude가 정확한 키/값으로 코드 생성
    def _build_editor_system(d: dict) -> str:
        anomaly_feats = [s.get("feature", "") for s in d.get("anomaly_stats", [])]
        import_feats = [s.get("feature", "") for s in d.get("importance", {}).get("features", [])]
        wyt_weeks = len(d.get("weekly_yield_trend", {}).get("labels", []))
        ppt_lots = len(d.get("pred_ppm_trend", {}).get("labels", []))
        cs_titles = [s.get("title", "") for s in d.get("custom_sections", [])]
        state = (
            "\n\n## 현재 d 상태 (수정 전 반드시 확인, 이미 반영된 내용 포함)\n\n"
            "| 항목 | 현재값 |\n"
            "|------|--------|\n"
            f"| importance.features | {len(import_feats)}개 → {import_feats[:5]} |\n"
            f"| anomaly_stats | {len(anomaly_feats)}개 → {anomaly_feats} |\n"
            f"| weekly_yield_trend (L2) | {wyt_weeks}주 데이터 |\n"
            f"| pred_ppm_trend (L3) | {ppt_lots}개 LOT 데이터 |\n"
            f"| 대표 unit (R1) | {d.get('top_unit', {}).get('serial', '없음')} |\n"
            f"| chart_params | {d.get('chart_params', {})} |\n"
            f"| table_params.shap_hide_cols | {d.get('table_params', {}).get('shap_hide_cols', [])} |\n"
            f"| custom_sections | {cs_titles} |\n"
            f"| hidden_sections | {d.get('hidden_sections', [])} |\n"
            f"| report_title | {d.get('meta', {}).get('report_title', '기본값 사용중')} |\n"
            f"| alert_features | {d.get('meta', {}).get('alert_features', '기본값 사용중')} |\n"
        )
        return REPORT_EDITOR_SYSTEM + state

    try:
        response = await asyncio.to_thread(
            client.messages.create,
            model=MODEL,
            max_tokens=2048,
            system=_build_editor_system(d),
            messages=messages,
        )
    except Exception as e:
        yield {"type": "error", "message": str(e)}
        return

    # 응답 파싱
    raw_text = ""
    for block in response.content:
        if block.type == "text":
            raw_text += block.text

    print(f"[editor] raw_text: {raw_text[:500]!r}")

    # JSON 파싱 (Claude가 prose + JSON 혼재 반환하는 경우도 처리)
    import re as _re_json
    cmd = None

    def _try_parse_json(s: str):
        try:
            return json.loads(s.strip())
        except Exception:
            return None

    clean = raw_text.strip()

    # 1) 코드블록 안에 JSON
    if "```" in clean:
        m = _re_json.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, _re_json.DOTALL)
        if m:
            cmd = _try_parse_json(m.group(1))
        if cmd is None:
            # 코드블록 분리 fallback
            parts = clean.split("```")
            snippet = parts[1] if len(parts) > 1 else ""
            if snippet.startswith("json"):
                snippet = snippet[4:]
            cmd = _try_parse_json(snippet)

    # 2) 텍스트가 { 로 시작하는 경우
    if cmd is None and clean.startswith("{"):
        cmd = _try_parse_json(clean)

    # 3) 텍스트 + JSON 혼재 — { ... } 블록을 찾아 파싱
    if cmd is None:
        first = clean.find("{")
        last  = clean.rfind("}")
        if first != -1 and last > first:
            cmd = _try_parse_json(clean[first:last + 1])

    # 모든 시도 실패
    if cmd is None:
        print(f"[editor] JSON 파싱 실패 — raw_text: {raw_text[:300]!r}")
        m = _re_json.search(r'"response"\s*:\s*"(.*?)"(?:,|\})', raw_text, _re_json.DOTALL)
        fallback_text = m.group(1).replace("\\n", "\n") if m else ""
        if not fallback_text:
            fallback_text = "요청을 처리하지 못했습니다. 다시 시도해 주세요."
        yield {"type": "text", "content": fallback_text}
        yield {"type": "done"}
        return

    print(f"[editor] cmd: {cmd}")
    response_text = cmd.get("response", "").strip()

    # action 커맨드 처리 (command-based 방식 — exec 없이 핸들러 직접 실행)
    if "action" in cmd:
        success, err_msg = _handle_command(cmd, d)
        if not success:
            yield {"type": "text", "content": f"⚠️ 처리 실패: {err_msg}"}
        else:
            try:
                html = build_html(d)
            except Exception as _e:
                yield {"type": "text", "content": f"⚠️ HTML 생성 오류: {_e}"}
                yield {"type": "done"}
                return
            yield {"type": "report_ready", "html": html, "report_data": copy.deepcopy(d)}
            yield {"type": "text", "content": response_text or "보고서를 수정했습니다."}
        yield {"type": "done"}
        return

    # 기존 exec() 방식 (action 없는 커스텀 요청 fallback)
    code = cmd.get("code", "").strip()
    if code:
        success, err = _execute_editor_code(code, d)
        if not success:
            response_text = (response_text + f"\n\n⚠️ 코드 실행 오류: {err}").strip()
        else:
            try:
                html = build_html(d)
            except Exception as _e:
                response_text = (response_text + f"\n\n⚠️ HTML 생성 오류: {_e}").strip()
            else:
                yield {"type": "report_ready", "html": html, "report_data": copy.deepcopy(d)}

    yield {"type": "text", "content": response_text or "처리를 완료했습니다."}
    yield {"type": "done"}
