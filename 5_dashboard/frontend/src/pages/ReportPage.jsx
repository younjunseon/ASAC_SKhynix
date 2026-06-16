import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import './ReportPage.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const INJECT_SCRIPT = `
<script>
(function() {
  var editMode = false;
  var EDITABLE_TAGS = ['H1','H2','H3','H4','H5','H6','P','SPAN','TD','TH','LI','CAPTION','FIGCAPTION','DT','DD'];
  function isEditable(el) {
    if (!el || !el.tagName) return false;
    if (el.closest('[data-no-edit]')) return false;
    if (el.closest('canvas, svg, script, style')) return false;
    return EDITABLE_TAGS.indexOf(el.tagName) !== -1;
  }
  function flashHover(el, on) {
    if (!el) return;
    el.style.outline = on ? '1px dashed #10B981' : '';
    el.style.cursor  = on ? 'text' : '';
  }
  var lastHover = null;
  document.addEventListener('mouseover', function(e) {
    if (!editMode) return;
    if (lastHover) flashHover(lastHover, false);
    if (isEditable(e.target)) { lastHover = e.target; flashHover(lastHover, true); }
  });
  document.addEventListener('mouseout', function(e) {
    if (!editMode) return;
    if (lastHover) { flashHover(lastHover, false); lastHover = null; }
  });
  document.addEventListener('dblclick', function(e) {
    if (!editMode) return;
    var el = isEditable(e.target) ? e.target : null;
    if (!el) return;
    e.preventDefault(); e.stopPropagation();
    el.setAttribute('contenteditable','true');
    el.style.background = '#FEF9C3';
    el.style.outline    = '2px solid #F59E0B';
    el.focus();
    var range = document.createRange(); range.selectNodeContents(el);
    var sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(range);
    function commit() {
      el.removeAttribute('contenteditable');
      el.style.background = '';
      el.style.outline    = '';
      el.removeEventListener('blur', commit);
      el.removeEventListener('keydown', onKey);
      window.parent.postMessage({ type: 'INLINE_EDIT', html: document.documentElement.outerHTML }, '*');
    }
    function onKey(ev) { if (ev.key === 'Escape') { ev.preventDefault(); el.blur(); } }
    el.addEventListener('blur', commit);
    el.addEventListener('keydown', onKey);
  });
  window.addEventListener('message', function(e) {
    if (!e.data) return;
    if (e.data.type === 'SET_EDIT_MODE') { editMode = !!e.data.value; }
  });
})();
</script>
`

const INIT_MSG = {
  role: 'bot',
  text: '안녕하세요! SK Hynix 보고서 AI Agent입니다.\n\n"이번 주 보고서 만들어줘" 처럼 요청하면 좌측에 보고서가 생성됩니다.\n생성 후에는 이 채팅창에서 바로 수정 요청도 할 수 있어요.',
}

export default function ReportPage() {
  // ── 단일 채팅 상태 ─────────────────────────────────
  const [messages, setMessages] = useState([INIT_MSG])
  const [buttons, setButtons]   = useState([])
  const historyRef   = useRef([])
  const toolCacheRef = useRef({})
  // 보고서 생성 버튼: 확인 단계들을 자동으로 진행
  const autoGenRef     = useRef(false)
  const pendingAutoRef = useRef(null)

  // ── 보고서 상태 ────────────────────────────────────
  const [currentHtml, setCurrentHtml]             = useState(null)
  const [currentReportData, setCurrentReportData] = useState({})
  const currentHtmlRef   = useRef(null)
  const currentReportRef = useRef({})
  const originalReportRef = useRef(null)   // 생성 직후 원본(초기화 복원용)
  const [blobUrl, setBlobUrl] = useState(null)
  const prevBlobRef = useRef(null)
  const initialLoad = useRef(true)

  // ── 편집 모드 ─────────────────────────────────────
  const [selectMode, setSelectMode] = useState(false)
  const [editMode, setEditMode]     = useState(false)
  const [selectedSection, setSelected] = useState(null)
  const [canUndo, setCanUndo] = useState(false)
  const [canRedo, setCanRedo] = useState(false)
  const undoStackRef = useRef([])
  const redoStackRef = useRef([])

  const [input, setInput]     = useState('')
  const [loading, setLoading] = useState(false)

  const iframeRef = useRef(null)
  const bottomRef = useRef(null)
  const previewRef = useRef(null)
  const [fitScale, setFitScale] = useState(1)

  // 보고서(1280×720 고정)를 미리보기 영역 너비에 맞춰 축소
  const REPORT_W = 1280, REPORT_H = 720
  useEffect(() => {
    const compute = () => {
      const el = previewRef.current
      if (!el) return
      const pad = 24
      const s = Math.min(1, (el.clientWidth - pad) / REPORT_W)
      setFitScale(s > 0 ? s : 1)
    }
    compute()
    window.addEventListener('resize', compute)
    return () => window.removeEventListener('resize', compute)
  }, [blobUrl])

  // ref 동기화
  useEffect(() => { currentHtmlRef.current = currentHtml }, [currentHtml])
  useEffect(() => { currentReportRef.current = currentReportData }, [currentReportData])

  // 스크롤
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, buttons])

  // ── HTML → iframe ─────────────────────────────────
  function _inject(raw) {
    return raw.includes('</body>') ? raw.replace('</body>', INJECT_SCRIPT + '</body>') : raw + INJECT_SCRIPT
  }

  useEffect(() => {
    if (!currentHtml) return
    const url = URL.createObjectURL(new Blob([_inject(currentHtml)], { type: 'text/html;charset=utf-8' }))
    if (prevBlobRef.current) URL.revokeObjectURL(prevBlobRef.current)
    prevBlobRef.current = url
    setBlobUrl(url)
    initialLoad.current = true
    return () => URL.revokeObjectURL(url)
  }, [])

  useEffect(() => {
    if (!currentHtml) return
    if (initialLoad.current) { initialLoad.current = false; return }
    const iframe = iframeRef.current
    if (!iframe) return
    const doc = iframe.contentDocument || iframe.contentWindow?.document
    if (!doc) return
    const scrollY = iframe.contentWindow?.scrollY || 0
    doc.open(); doc.write(_inject(currentHtml)); doc.close()
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        iframe.contentWindow?.scrollTo(0, scrollY)
        if (selectMode) iframe.contentWindow?.postMessage({ type: 'SET_CHART_EDIT_MODE', value: true }, '*')
        if (editMode)   iframe.contentWindow?.postMessage({ type: 'SET_EDIT_MODE', value: true }, '*')
      })
    })
  }, [currentHtml])

  // iframe 메시지
  useEffect(() => {
    function onMessage(e) {
      if (e.data?.type === 'INLINE_EDIT' || e.data?.type === 'CHART_NOTE') {
        if (typeof e.data.html === 'string') currentHtmlRef.current = e.data.html
        return
      }
      if (e.data?.type === 'ia_event') {
        const payload = e.data.payload
        if (!payload) return
        if (payload.action === 'remove' || payload.action === 'change_chart') {
          sendDirectAction(payload); return
        }
        const label = payload.label || (payload.labels?.length ? payload.labels.join(', ') : '선택됨')
        setSelected(label)
        sendInteract(payload)
      }
    }
    window.addEventListener('message', onMessage)
    return () => window.removeEventListener('message', onMessage)
  }, [])

  // ── undo/redo ─────────────────────────────────────
  function pushUndo() {
    undoStackRef.current.push({ html: currentHtmlRef.current, report_data: { ...currentReportRef.current } })
    setCanUndo(true)
    redoStackRef.current = []; setCanRedo(false)
  }

  function handleUndo() {
    if (!undoStackRef.current.length) return
    redoStackRef.current.push({ html: currentHtmlRef.current, report_data: { ...currentReportRef.current } })
    setCanRedo(true)
    const prev = undoStackRef.current.pop()
    setCurrentHtml(prev.html)
    if (prev.report_data) setCurrentReportData(prev.report_data)
    setCanUndo(undoStackRef.current.length > 0)
  }

  function handleRedo() {
    if (!redoStackRef.current.length) return
    undoStackRef.current.push({ html: currentHtmlRef.current, report_data: { ...currentReportRef.current } })
    setCanUndo(true)
    const next = redoStackRef.current.pop()
    setCurrentHtml(next.html)
    if (next.report_data) setCurrentReportData(next.report_data)
    setCanRedo(redoStackRef.current.length > 0)
  }

  // ── 편집 모드 토글 ────────────────────────────────
  function toggleSelectMode() {
    const next = !selectMode
    setSelectMode(next)
    if (next) setEditMode(false)
    const iw = iframeRef.current?.contentWindow
    iw?.postMessage({ type: 'SET_CHART_EDIT_MODE', value: next }, '*')
    iw?.postMessage({ type: 'SET_EDIT_MODE', value: false }, '*')
    if (!next) setSelected(null)
  }

  function toggleEditMode() {
    const next = !editMode
    setEditMode(next)
    if (next) { setSelectMode(false); setSelected(null) }
    const iw = iframeRef.current?.contentWindow
    iw?.postMessage({ type: 'SET_EDIT_MODE', value: next }, '*')
    iw?.postMessage({ type: 'SET_CHART_EDIT_MODE', value: false }, '*')
  }

  function clearSelect() {
    setSelectMode(false); setEditMode(false); setSelected(null)
    const iw = iframeRef.current?.contentWindow
    iw?.postMessage({ type: 'SET_CHART_EDIT_MODE', value: false }, '*')
    iw?.postMessage({ type: 'SET_EDIT_MODE', value: false }, '*')
  }

  function handleReset() {
    const orig = originalReportRef.current
    if (!orig) { alert('복원할 원본 보고서가 없습니다. 먼저 보고서를 생성하세요.'); return }
    if (!window.confirm('모든 수정을 폐기하고 원본 보고서로 되돌립니까?')) return
    setCurrentHtml(orig.html)
    setCurrentReportData(orig.report_data || {})
    undoStackRef.current = []; redoStackRef.current = []
    setCanUndo(false); setCanRedo(false)
    clearSelect()
  }

  // ── PPT 다운로드 ──────────────────────────────────
  async function downloadPptx() {
    let liveHtml = currentHtmlRef.current || ''
    try {
      const doc = iframeRef.current?.contentDocument
      if (doc?.documentElement) liveHtml = '<!DOCTYPE html>' + doc.documentElement.outerHTML
    } catch {}
    const res = await fetch(`${API_URL}/report/pptx`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ report_data: currentReportRef.current || {}, filename: '품질불량개선조치보고서.pptx', current_html: liveHtml }),
    })
    if (!res.ok) { alert('PPTX 생성 실패'); return }
    const blob = await res.blob()
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob); a.download = '품질불량개선조치보고서.pptx'; a.click()
  }

  // ── SSE 공통 스트리밍 파서 ────────────────────────
  function makeStreamParser({ onText, onToolStart, onToolResult, onConfirm, onReportReady, onDone, onError }) {
    let botText = ''
    let buffer = ''
    function finalize() {
      if (onText && botText) onText(botText, true)
      botText = ''
    }
    async function consume(reader) {
      const decoder = new TextDecoder()
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n'); buffer = lines.pop()
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const raw = line.slice(6).trim()
          if (!raw || raw === '[DONE]') continue
          let ev; try { ev = JSON.parse(raw) } catch { continue }
          if (ev.type === 'text') { botText += ev.content; onText?.(botText, false) }
          if (ev.type === 'tool_start') { finalize(); onToolStart?.(ev.tool) }
          if (ev.type === 'tool_result') { finalize(); onToolResult?.(ev.tool, ev.result) }
          if (ev.type === 'confirm') { finalize(); onConfirm?.(ev.buttons) }
          if (ev.type === 'report_ready' && ev.html) { onReportReady?.(ev.html, ev.report_data) }
          if (ev.type === 'done') { finalize(); onDone?.() }
          if (ev.type === 'error') { finalize(); onError?.(ev.message) }
        }
      }
      finalize()
    }
    return consume
  }

  const TOOL_LABELS = {
    scan_data: '🔍 데이터 스캔 중...', analyze_features: '📊 feature 분포 비교 중...',
    get_importance: '📋 feature importance 조회 중...', infer_period: '📅 기간 해석 중...',
  }

  function addMsg(role, text, extra = {}) {
    setMessages(prev => [...prev, { role, text, ...extra }])
  }

  function updateStreamMsg(text, final) {
    setMessages(prev => {
      const next = [...prev]
      const last = next[next.length - 1]
      if (last?.role === 'bot' && last?.streaming)
        return next.map((m, i) => i === next.length - 1 ? { ...m, text, streaming: !final } : m)
      return [...next, { role: 'bot', text, streaming: !final }]
    })
    if (final && text) historyRef.current.push({ role: 'assistant', content: text })
  }

  // ── 단일 전송 함수 (생성/수정 자동 분기) ─────────
  async function sendMsg(msg) {
    setMessages(prev => [...prev, { role: 'user', text: msg }])
    historyRef.current.push({ role: 'user', content: msg })
    setLoading(true)

    // 보고서가 이미 있으면 수정 컨텍스트, 없으면 생성
    const isEdit = !!currentHtmlRef.current
    const body = isEdit
      ? { message: msg, history: historyRef.current.slice(0, -1), tool_cache: toolCacheRef.current,
          context: 'report_edit', current_html: currentHtmlRef.current, current_report_data: currentReportRef.current }
      : { message: msg, history: historyRef.current.slice(0, -1), tool_cache: toolCacheRef.current }

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const consume = makeStreamParser({
        onText: updateStreamMsg,
        onToolStart: (tool) => addMsg('bot', TOOL_LABELS[tool] || `⚙️ ${tool}...`, { status: true }),
        onToolResult: (tool, result) => { toolCacheRef.current = { ...toolCacheRef.current, [tool]: result } },
        onConfirm: (btns) => {
          // 보고서 생성 버튼으로 시작된 경우: 확인 단계의 첫 버튼으로 자동 진행
          if (autoGenRef.current && btns?.length) {
            pendingAutoRef.current = btns[0]
            setButtons([])
          } else {
            setButtons(btns)
          }
        },
        onReportReady: (html, data) => {
          if (isEdit) {
            // 수정: undo 스택에 현재 상태 저장 후 교체
            pushUndo()
            setCurrentHtml(html)
            if (data) setCurrentReportData(data)
          } else {
            // 최초 생성: blobUrl 초기화
            const url = URL.createObjectURL(new Blob([_inject(html)], { type: 'text/html;charset=utf-8' }))
            if (prevBlobRef.current) URL.revokeObjectURL(prevBlobRef.current)
            prevBlobRef.current = url
            setBlobUrl(url)
            initialLoad.current = true
            setCurrentHtml(html)
            if (data) setCurrentReportData(data)
            originalReportRef.current = { html, report_data: data || {} }  // 원본 저장(초기화용)
            setButtons([])
            // 자동 생성 완료
            autoGenRef.current = false
            pendingAutoRef.current = null
          }
        },
        onDone: () => {
          setLoading(false)
          // 자동 진행 중이면 다음 확인 단계로 이어서 진행
          if (autoGenRef.current && pendingAutoRef.current) {
            const next = pendingAutoRef.current
            pendingAutoRef.current = null
            setButtons([])
            setTimeout(() => sendMsg(next), 50)
          }
        },
        onError: (m) => {
          addMsg('bot', `⚠️ ${m}`); setLoading(false)
          autoGenRef.current = false; pendingAutoRef.current = null
        },
      })
      await consume(res.body.getReader())
    } catch {
      addMsg('bot', '⚠️ 서버 연결 실패. FastAPI 서버(8000)가 실행 중인지 확인하세요.')
      setLoading(false)
    }
  }

  // ── 직접 액션 (삭제/차트교체) ────────────────────
  async function sendDirectAction(cmd) {
    if (loading) return
    setLoading(true)
    try {
      const res = await fetch(`${API_URL}/report/interact`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...cmd, prompt: `__direct__:${JSON.stringify(cmd)}`, history: [], tool_cache: toolCacheRef.current, current_report_data: currentReportRef.current }),
      })
      const consume = makeStreamParser({
        onReportReady: (html, data) => { pushUndo(); setCurrentHtml(html); if (data) setCurrentReportData(data) },
        onDone: () => { addMsg('bot', cmd.action === 'remove' ? '섹션 삭제 완료' : '차트 변경 완료'); setLoading(false) },
        onError: (m) => { addMsg('bot', `⚠️ ${m}`); setLoading(false) },
      })
      await consume(res.body.getReader())
    } catch {
      addMsg('bot', '⚠️ 서버 연결 실패.')
      setLoading(false)
    }
  }

  // ── interact ──────────────────────────────────────
  async function sendInteract(payload) {
    if (loading) return
    setLoading(true)
    const userMsg = payload.prompt || `[${payload.label}] 수정 요청`
    addMsg('user', userMsg)
    historyRef.current.push({ role: 'user', content: userMsg })
    try {
      const res = await fetch(`${API_URL}/report/interact`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...payload, history: historyRef.current.slice(0, -1), tool_cache: toolCacheRef.current, current_report_data: currentReportRef.current }),
      })
      const consume = makeStreamParser({
        onText: updateStreamMsg,
        onConfirm: (btns) => setMessages(prev => [...prev, { role: 'bot', text: '', buttons: btns }]),
        onReportReady: (html, data) => { pushUndo(); setCurrentHtml(html); if (data) setCurrentReportData(data) },
        onDone: () => setLoading(false),
        onError: (m) => { addMsg('bot', `⚠️ ${m}`); setLoading(false) },
      })
      await consume(res.body.getReader())
    } catch {
      addMsg('bot', '⚠️ 서버 연결 실패.')
      setLoading(false)
    }
  }

  // ── 통합 전송 ─────────────────────────────────────
  async function send(text) {
    const msg = (text || input).trim()
    if (!msg || loading) return
    setInput(''); setButtons([])
    if (msg === '기간 변경') toolCacheRef.current = {}
    await sendMsg(msg)
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
  }

  function handleNewReport() {
    setMessages([INIT_MSG])
    setButtons([])
    setCurrentHtml(null); setCurrentReportData({})
    setBlobUrl(null)
    toolCacheRef.current = {}; historyRef.current = []
    undoStackRef.current = []; redoStackRef.current = []
    setCanUndo(false); setCanRedo(false)
    clearSelect(); setInput('')
  }

  return (
    <div className="rp-page">

      {/* 좌: 보고서 미리보기 */}
      <div className="rp-preview-panel">
        {/* 툴바 (보고서 생성 후에만 표시) */}
        {currentHtml && (
          <div className="rp-toolbar">
            <button className={`rp-tool-btn ${selectMode ? 'active' : ''}`} onClick={toggleSelectMode}>✏️ 차트 편집</button>
            <button className="rp-tool-btn" onClick={handleReset}>🔄 초기화</button>
            <button className={`rp-tool-btn ${!canUndo ? 'disabled' : ''}`} onClick={handleUndo} disabled={!canUndo}>↩</button>
            <button className={`rp-tool-btn ${!canRedo ? 'disabled' : ''}`} onClick={handleRedo} disabled={!canRedo}>↪</button>
            <button className="rp-tool-btn ppt" onClick={downloadPptx}>📊 PPT</button>
          </div>
        )}

        {selectedSection && (
          <div className="rp-selection-bar">
            📌 선택됨: <strong>{selectedSection}</strong>
            <span className="rp-selection-hint">→ 우측 채팅에 수정 요청을 입력하세요</span>
          </div>
        )}

        <div ref={previewRef} className={`rp-preview ${selectMode ? 'select-mode' : ''}`}>
          {blobUrl ? (
            <div
              className="rp-scale-wrap"
              style={{
                width: REPORT_W * fitScale,
                height: REPORT_H * fitScale,
              }}
            >
              <iframe
                ref={iframeRef}
                src={blobUrl}
                title="보고서 미리보기"
                className="rp-iframe"
                style={{
                  width: REPORT_W,
                  height: REPORT_H,
                  transform: `scale(${fitScale})`,
                  transformOrigin: 'top left',
                }}
              />
            </div>
          ) : (
            <div className="rp-empty">
              <div className="rp-empty-icon">📄</div>
              <div className="rp-empty-title">보고서가 아직 없습니다</div>
              <div className="rp-empty-sub">우측 채팅창에서 보고서 생성을 요청해 보세요</div>
            </div>
          )}
        </div>
      </div>

      {/* 우: 채팅 */}
      <div className="rp-chat-panel">
        <div className="rp-chat-header">
          <span>보고서 AI Agent</span>
        </div>

        <div className="rp-chat-messages">
          {messages.map((m, i) => (
            <div key={i} className={`rp-msg ${m.role}`}>
              {m.role === 'bot' && <div className="rp-avatar">AI</div>}
              <div className="rp-msg-col">
                <div className={`rp-bubble ${m.status || m.tool ? 'status' : ''} ${m.streaming ? 'streaming' : ''}`}>
                  {m.role === 'bot' && !m.status && !m.tool
                    ? <ReactMarkdown>{m.text}</ReactMarkdown>
                    : <span>{m.text}</span>
                  }
                </div>
                {m.buttons?.length > 0 && (
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                    {m.buttons.map((btn, bi) => (
                      <button key={bi} className="rp-confirm-btn" onClick={() => send(btn)}>{btn}</button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}

          {buttons.length > 0 && !loading && (
            <div className="rp-buttons">
              {buttons.map(btn => (
                <button key={btn} className="rp-confirm-btn" onClick={() => send(btn)}>{btn}</button>
              ))}
            </div>
          )}

          {loading && (
            <div className="rp-msg bot">
              <div className="rp-avatar">AI</div>
              <div className="rp-bubble status">
                <span className="rp-typing"><span /><span /><span /></span>
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <div className="rp-input-row">
          <textarea
            className="rp-input"
            placeholder={currentHtml ? '수정 요청을 입력하세요...' : '보고서 생성을 요청하세요... (예: 이번 주 보고서 만들어줘)'}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKey}
            rows={2}
            disabled={loading}
          />
          <button className="rp-send" onClick={() => send()} disabled={loading}>전송</button>
        </div>
      </div>
    </div>
  )
}
