"""
FastAPI 앱 진입점.
- POST /chat       : Agent 루프 실행 (SSE 스트리밍)
- POST /report/pptx: 마크다운 → PPTX 변환 후 다운로드
"""
import json
import asyncio
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response, FileResponse
from pydantic import BaseModel

from agent import run_agent
from report import build_pptx

load_dotenv()

app = FastAPI(title="SK Hynix AI Agent")

# ── 서버 시작 시 report_data 백그라운드 프리빌드 ──────────────
_preview_cache: dict = {}

async def _prebuild_preview():
    """서버 시작 직후 백그라운드에서 실데이터 report_data 조립."""
    global _preview_cache
    try:
        from agent import _build_report_data
        from report import build_html
        from tools import scan_data, get_importance, analyze_features, _load

        def _warmup():
            _load("dashboard_units.csv")
            _load("feature_importance.csv")
            # compet_xs(1.2GB) 프리로드 제거 — 보고서는 xs 불필요, 서버 시작 시 xs 의존 제거
            return {
                "scan_data":      scan_data(),
                "get_importance": get_importance(top_n=10),
            }

        cache = await asyncio.to_thread(_warmup)
        report_data = _build_report_data(cache)
        html = build_html(report_data)
        _preview_cache["html"] = html
        _preview_cache["report_data"] = report_data
        print("[preview] 실데이터 프리빌드 완료")
    except Exception as e:
        print(f"[preview] 프리빌드 실패: {e}")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(_prebuild_preview())

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    history: list = []
    tool_cache: dict = {}
    context: str = ""        # "report_edit" 이면 보고서 수정 모드
    current_html: str = ""   # 현재 보고서 HTML (수정 컨텍스트)
    current_report_data: dict = {}  # 현재 보고서 데이터 (수정 누적용)


class InteractRequest(BaseModel):
    """보고서 인터랙션 이벤트 (통합 차트 편집 모드)."""
    action: str            # "add" | "modify" | "remove" | "explain"
    sid: str = ""          # 단일 섹션 ID (section 모드)
    label: str = ""        # 섹션 표시명
    sids: list = []        # 박스 선택 시 포함된 섹션 ID 목록
    labels: list = []      # 박스 선택 시 포함된 섹션 라벨 목록
    position: str = ""     # "left_col" | "right_col" (빈 공간 add 시)
    bbox: dict = {}        # 선택 영역 좌표·크기 {x,y,w,h}  ※ 슬라이드 기준
    drag: dict = {}        # (legacy) 드래그 좌표
    layout: list = []      # 전체 섹션 좌표 스냅샷
    prompt: str = ""       # JS가 생성한 자연어 요청
    history: list = []
    tool_cache: dict = {}
    current_report_data: dict = {}
    current_report_data: dict = {}


class ReportRequest(BaseModel):
    report_data: dict = {}
    filename: str = "품질불량개선조치보고서.pptx"
    current_html: str | None = None  # 인라인 텍스트 편집 반영용


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Agent 루프를 실행하고 SSE로 이벤트를 스트리밍.
    프론트는 EventSource 또는 fetch + ReadableStream으로 수신.
    """
    async def event_stream():
        try:
            if req.context == "report_edit":
                from agent import run_report_editor
                gen = run_report_editor(
                    req.message, req.history, req.tool_cache,
                    req.current_html, req.current_report_data
                )
            else:
                gen = run_agent(req.message, req.history, req.tool_cache)
            async for event in gen:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            error_event = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/report/pptx")
async def generate_pptx(req: ReportRequest):
    """구조화된 report_data로 PPTX 생성. report.py 수정 즉시 반영을 위해 매번 재import."""
    import importlib, report
    importlib.reload(report)
    pptx_bytes = report.build_pptx(req.report_data, current_html=req.current_html)
    from urllib.parse import quote
    encoded = quote(req.filename, encoding="utf-8")
    return Response(
        content=pptx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{encoded}",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        },
    )


@app.options("/report/pptx")
async def pptx_preflight():
    """Preflight 요청 처리."""
    return Response(
        content="",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        },
    )


@app.get("/report/preview")
async def preview_report():
    """실데이터로 HTML + report_data JSON 반환.
    report_data는 캐시 사용(생성 비용 큼), HTML은 매 요청 재생성(코드 수정 즉시 반영)."""
    # 캐시가 아직 없으면 최대 60초 대기
    for _ in range(60):
        if _preview_cache:
            break
        await asyncio.sleep(1)
    if not _preview_cache:
        return {"html": "<p>데이터 로딩 중입니다. 잠시 후 다시 시도해주세요.</p>", "report_data": {}}
    # HTML은 매번 새로 빌드 → report.py 수정 즉시 반영
    try:
        import importlib, report
        importlib.reload(report)
        html = report.build_html(_preview_cache["report_data"])
        return {"html": html, "report_data": _preview_cache["report_data"]}
    except Exception as e:
        print(f"[preview] HTML 재빌드 실패: {e}")
        return _preview_cache


@app.post("/report/interact")
async def report_interact(req: InteractRequest):
    """
    보고서 인터랙션 이벤트 → run_report_editor로 전달.
    드래그/우클릭/hover 버튼에서 발생한 이벤트를 처리하고 SSE로 응답.
    """
    from agent import run_report_editor, REPORT_EDITOR_SYSTEM

    # 레이아웃 정보를 시스템 컨텍스트로 주입
    layout_ctx = ""
    if req.layout:
        lines = ["## 현재 보고서 레이아웃 (섹션 좌표)"]
        for s in req.layout:
            r = s.get("rect", {})
            lines.append(f"- [{s['sid']}] {s.get('label','')} : x={r.get('x')}, y={r.get('y')}, w={r.get('w')}, h={r.get('h')}")
        layout_ctx = "\n".join(lines)

    # bbox(선택 영역 크기) 처리 — drag는 legacy 호환용
    box = req.bbox or req.drag or {}
    if box:
        bw, bh = box.get('w'), box.get('h')
        layout_ctx += (
            f"\n\n## 선택 영역 (반드시 이 크기 안에 맞춰 생성)"
            f"\n- 위치: x={box.get('x')}, y={box.get('y')}"
            f"\n- 크기: {bw}px × {bh}px"
        )
        if req.sids:
            layout_ctx += f"\n- 포함 섹션: {', '.join(req.sids)}"

    # action에 따라 프롬프트 보강
    size_hint = (
        f" 생성하는 콘텐츠는 정확히 너비 {box.get('w')}px × 높이 {box.get('h')}px 박스 안에 맞춰야 합니다."
        if box else ""
    )
    action_hint = {
        "add":     "\n\n[요청 유형: 빈 영역에 새 섹션 추가]" + size_hint,
        "modify":  "\n\n[요청 유형: 기존 섹션 수정]" + size_hint + " 구체적인 수정 내용이 불명확하면 먼저 물어보세요.",
        "remove":  "\n\n[요청 유형: 섹션 삭제] 대상 섹션을 보고서에서 제거하세요.",
        "explain": "\n\n[요청 유형: 데이터 설명] 해당 영역의 데이터를 친절히 설명해주세요. (HTML 변경은 하지 마세요)",
    }.get(req.action, "")

    message = req.prompt + action_hint
    if layout_ctx:
        message = layout_ctx + "\n\n" + message

    async def event_stream():
        try:
            gen = run_report_editor(
                message, req.history, req.tool_cache,
                "", req.current_report_data
            )
            async for event in gen:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','message':str(e)}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

@app.get("/api/data/{path:path}")
async def serve_data(path: str):
    """CSV/JSON 정적 데이터 파일 서빙."""
    file_path = os.path.join(_STATIC_DIR, path)
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


@app.get("/health")
def health():
    return {"status": "ok"}
