import { useRef, useState, useEffect, useCallback } from 'react'
import './ReportModal.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// HTML 보고서에 주입할 인터랙티브 스크립트
//  - 차트 편집 모드: report.py가 자체 구현. 부모는 'SET_CHART_EDIT_MODE'를 직접 전송.
//  - 아래는 텍스트 편집(B-2) / 차트 메모(B-3) 전용 스크립트.
const INJECT_SCRIPT = `
<script>

// ── 인라인 텍스트 편집 ─────────────────────────────────────
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
      window.parent.postMessage({
        type: 'INLINE_EDIT',
        html: document.documentElement.outerHTML,
      }, '*');
    }
    function onKey(ev) {
      if (ev.key === 'Escape') { ev.preventDefault(); el.blur(); }
      if (ev.key === 'Enter' && !ev.shiftKey && el.tagName !== 'P' && el.tagName !== 'LI') {
        ev.preventDefault(); el.blur();
      }
    }
    el.addEventListener('blur', commit);
    el.addEventListener('keydown', onKey);
  }, true);
  window.addEventListener('message', function(e) {
    if (!e.data) return;
    if (e.data.type === 'SET_EDIT_MODE') {
      editMode = !!e.data.value;
      document.body.style.cursor = editMode ? 'text' : '';
      if (!editMode && lastHover) { flashHover(lastHover, false); lastHover = null; }
    }
  });
})();

// ── 차트 캡션·주석 추가 ────────────────────────────────────
(function() {
  var noteMode = false;
  function findChart(target) {
    var cur = target;
    while (cur && cur !== document.body) {
      if (cur.tagName === 'CANVAS' || cur.tagName === 'SVG') return cur;
      if (cur.dataset && cur.dataset.section) return cur;
      cur = cur.parentElement;
    }
    return null;
  }
  function getOrCreateNoteHost(chartEl) {
    var container = chartEl.parentElement;
    if (!container) return null;
    var existing = container.querySelector(':scope > .__chart_note');
    if (existing) return existing;
    var note = document.createElement('div');
    note.className = '__chart_note';
    note.setAttribute('data-no-edit','1');
    note.style.cssText = [
      'margin-top:6px','padding:6px 10px','border-left:3px solid #F59E0B',
      'background:#FFFBEB','color:#92400E','font-size:11px',
      'border-radius:4px','line-height:1.5','white-space:pre-wrap'
    ].join(';');
    container.appendChild(note);
    return note;
  }
  function flashTarget(el, on) {
    if (!el) return;
    el.style.outline = on ? '2px dashed #F59E0B' : '';
    el.style.cursor  = on ? 'pointer' : '';
  }
  var lastHover = null;
  document.addEventListener('mouseover', function(e) {
    if (!noteMode) return;
    if (lastHover) flashTarget(lastHover, false);
    lastHover = findChart(e.target);
    if (lastHover) flashTarget(lastHover, true);
  });
  document.addEventListener('mouseout', function(e) {
    if (!noteMode) return;
    if (lastHover) { flashTarget(lastHover, false); lastHover = null; }
  });
  document.addEventListener('click', function(e) {
    if (!noteMode) return;
    var chartEl = findChart(e.target);
    if (!chartEl) return;
    e.preventDefault(); e.stopPropagation();
    var note = getOrCreateNoteHost(chartEl);
    if (!note) return;
    var existing = note.textContent || '';
    var memo = window.prompt('차트에 추가할 메모를 입력하세요 (취소하면 삭제됩니다):', existing);
    if (memo === null) {
      note.remove();
    } else if (memo.trim() === '') {
      note.remove();
    } else {
      note.textContent = '📝 ' + memo.trim();
    }
    window.parent.postMessage({
      type: 'CHART_NOTE',
      html: document.documentElement.outerHTML,
    }, '*');
  }, true);
  window.addEventListener('message', function(e) {
    if (!e.data) return;
    if (e.data.type === 'SET_NOTE_MODE') {
      noteMode = !!e.data.value;
      document.body.style.cursor = noteMode ? 'pointer' : '';
      if (!noteMode && lastHover) { flashTarget(lastHover, false); lastHover = null; }
    }
  });
})();
</script>
`

export default function ReportModal({ markdown: html, reportData, toolCache, onClose, apiUrl }) {
  const [selectMode, setSelectMode]     = useState(false)
  const [editMode, setEditMode]         = useState(false)
  const [selectedSection, setSelected] = useState(null)
  const [messages, setMessages]         = useState([
    { role: 'bot', text: '보고서에 대해 질문하거나 수정을 요청해 보세요.\n예) "SHAP 상위 피처를 설명해줘", "개선 방안을 X552_range로 바꿔줘"' }
  ])
  const [input, setInput]   = useState('')
  const [loading, setLoading] = useState(false)
  const [currentHtml, setCurrentHtml] = useState(html)
  const [currentReportData, setCurrentReportData] = useState(reportData || {})

  const iframeRef         = useRef(null)
  const bottomRef         = useRef(null)
  const historyRef        = useRef([])
  const prevBlobRef       = useRef(null)
  const initialLoad       = useRef(true)
  const currentHtmlRef    = useRef(currentHtml)
  const currentReportRef  = useRef(currentReportData)
  const undoStackRef      = useRef([])
  const redoStackRef      = useRef([])
  const [canUndo, setCanUndo] = useState(false)
  const [canRedo, setCanRedo] = useState(false)
  const [blobUrl, setBlobUrl] = useState(null)

  // ref를 항상 최신 state로 동기화
  useEffect(() => { currentHtmlRef.current = currentHtml }, [currentHtml])
  useEffect(() => { currentReportRef.current = currentReportData }, [currentReportData])

  function _inject(rawHtml) {
    return rawHtml.includes('</body>')
      ? rawHtml.replace('</body>', INJECT_SCRIPT + '</body>')
      : rawHtml + INJECT_SCRIPT
  }

  // 최초 로드: blob URL로 iframe src 세팅
  useEffect(() => {
    if (!currentHtml) return
    const newUrl = URL.createObjectURL(
      new Blob([_inject(currentHtml)], { type: 'text/html;charset=utf-8' })
    )
    if (prevBlobRef.current) URL.revokeObjectURL(prevBlobRef.current)
    prevBlobRef.current = newUrl
    setBlobUrl(newUrl)
    initialLoad.current = true   // 다음 update는 write() 방식으로
    return () => { URL.revokeObjectURL(newUrl) }
  }, [])   // 마운트 시 1회만

  // 이후 수정: src 교체 없이 document.write()로 덮어쓰기 → 스크롤 위치 유지
  useEffect(() => {
    if (initialLoad.current) { initialLoad.current = false; return }
    const iframe = iframeRef.current
    if (!iframe) return
    const doc = iframe.contentDocument || iframe.contentWindow?.document
    if (!doc) return
    const scrollY = iframe.contentWindow?.scrollY || 0
    doc.open()
    doc.write(_inject(currentHtml))
    doc.close()
    // Chart.js 렌더링 완료 후 스크롤 복원 (rAF 2중첩 + 100ms 보험)
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        iframe.contentWindow?.scrollTo(0, scrollY)
        if (selectMode) {
          iframe.contentWindow?.postMessage({ type: 'SET_CHART_EDIT_MODE', value: true }, '*')
        }
        if (editMode) {
          iframe.contentWindow?.postMessage({ type: 'SET_EDIT_MODE', value: true }, '*')
        }
      })
    })
    const t = setTimeout(() => {
      iframe.contentWindow?.scrollTo(0, scrollY)
    }, 120)
    return () => clearTimeout(t)
  }, [currentHtml])

  // iframe → 부모: 통합 차트 편집 모드 이벤트 + 인라인 편집/메모
  useEffect(() => {
    function onMessage(e) {
      // 인라인 편집 / 차트 메모: iframe DOM이 이미 갱신되었으므로 ref만 갱신
      if (e.data?.type === 'INLINE_EDIT' || e.data?.type === 'CHART_NOTE') {
        if (typeof e.data.html === 'string') {
          currentHtmlRef.current = e.data.html
        }
        return
      }
      // 통합 차트 편집: 액션 메뉴에서 선택된 액션을 /report/interact 로 전송
      if (e.data?.type === 'ia_event') {
        const payload = e.data.payload
        if (!payload) return
        if (payload.action === 'remove' || payload.action === 'change_chart') {
          sendDirectAction(payload)
          return
        }
        const displayLabel = payload.label
          || (payload.labels && payload.labels.length ? payload.labels.join(', ') : '')
          || (payload.bbox ? `드래그 영역(${payload.bbox.w}×${payload.bbox.h})` : '선택됨')
        setSelected(displayLabel)
        sendInteract(payload)
        return
      }
    }
    window.addEventListener('message', onMessage)
    return () => window.removeEventListener('message', onMessage)
  }, [])   // dep [] 고정 — 최신 state는 ref로 참조

  // ── 직접 액션 전송 (Claude 불필요한 삭제·차트교체) ──────────
  async function sendDirectAction(cmd) {
    if (loading) return
    setLoading(true)

    const prompt = `__direct__:${JSON.stringify(cmd)}`
    try {
      const res = await fetch(`${apiUrl || API_URL}/report/interact`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...cmd,
          action: cmd.action,
          prompt,
          history: [],
          tool_cache: toolCache || {},
          current_report_data: currentReportRef.current,
        }),
      })
      const reader  = res.body.getReader()
      const decoder = new TextDecoder()
      let botText = ''
      let buffer  = ''
      const finalizeStreaming = () => {
        setMessages(prev => prev.map(m => m.streaming ? { ...m, streaming: false } : m))
        if (botText) historyRef.current.push({ role: 'assistant', content: botText })
        botText = ''
      }
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop()
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const raw = line.slice(6).trim()
          if (!raw || raw === '[DONE]') continue
          let event
          try { event = JSON.parse(raw) } catch { continue }
          if (event.type === 'text') {
            botText += event.content
            const snap = botText
            setMessages(prev => {
              const next = [...prev]
              const last = next[next.length - 1]
              if (last?.role === 'bot' && last?.streaming) {
                next[next.length - 1] = { ...last, text: snap }
              } else {
                next.push({ role: 'bot', text: snap, streaming: true })
              }
              return next
            })
          }
          if (event.type === 'report_ready' && event.html) {
            pushUndo()
            setCurrentHtml(event.html)
            if (event.report_data) setCurrentReportData(event.report_data)
          }
          if (event.type === 'done') setLoading(false)
          if (event.type === 'error') { addMsg('bot', `⚠️ ${event.message}`); setLoading(false) }
        }
      }
      finalizeStreaming()
    } catch {
      addMsg('bot', '⚠️ 서버 연결에 실패했습니다.')
      setLoading(false)
    }
  }

  // ── /report/interact 전송 (ia_event 처리) ──────────────────
  async function sendInteract(payload) {
    if (loading) return
    setLoading(true)
    const userMsg = payload.prompt || `[${payload.label}] 수정 요청`
    addMsg('user', userMsg)
    historyRef.current.push({ role: 'user', content: userMsg })

    try {
      const res = await fetch(`${apiUrl || API_URL}/report/interact`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...payload,
          history: historyRef.current.slice(0, -1),
          tool_cache: toolCache || {},
          current_report_data: currentReportRef.current,
        }),
      })

      const reader  = res.body.getReader()
      const decoder = new TextDecoder()
      let botText = ''
      let buffer  = ''

      const finalizeStreaming = () => {
        setMessages(prev => prev.map(m => m.streaming ? { ...m, streaming: false } : m))
        if (botText) historyRef.current.push({ role: 'assistant', content: botText })
        botText = ''
      }

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop()

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const raw = line.slice(6).trim()
          if (!raw || raw === '[DONE]') continue
          let event
          try { event = JSON.parse(raw) } catch { continue }

          if (event.type === 'text') {
            botText += event.content
            const snap = botText
            setMessages(prev => {
              const next = [...prev]
              const last = next[next.length - 1]
              if (last?.role === 'bot' && last?.streaming) {
                next[next.length - 1] = { ...last, text: snap }
              } else {
                next.push({ role: 'bot', text: snap, streaming: true })
              }
              return next
            })
          }
          if (event.type === 'tool_start') {
            // 이전 텍스트 메시지 종료 → 다음 텍스트는 새 버블에서 시작
            finalizeStreaming()
            const label = ({
              scan_data: '🔍 데이터 스캔 중...',
              get_importance: '📋 Feature importance 조회 중...',
              analyze_features: '📊 Feature 분포 비교 중...',
              infer_period: '📅 기간 해석 중...',
            })[event.tool] || `🔧 ${event.tool} 실행 중...`
            setMessages(prev => [...prev, { role: 'bot', text: label, tool: true }])
          }
          if (event.type === 'tool_result') {
            // tool_result 이후 곧 새 텍스트가 오므로 누적 끊기
            finalizeStreaming()
          }
          if (event.type === 'report_ready' && event.html) {
            pushUndo()
            setCurrentHtml(event.html)
            if (event.report_data) setCurrentReportData(event.report_data)
          }
          if (event.type === 'confirm' && event.buttons) {
            finalizeStreaming()
            setMessages(prev => [...prev, { role: 'bot', text: '', buttons: event.buttons }])
          }
          if (event.type === 'done') setLoading(false)
          if (event.type === 'error') {
            addMsg('bot', `⚠️ ${event.message}`)
            setLoading(false)
          }
        }
      }
      finalizeStreaming()
    } catch {
      addMsg('bot', '⚠️ 서버 연결에 실패했습니다.')
      setLoading(false)
    }
  }

  // 통합 차트 편집 모드 토글
  function toggleSelectMode() {
    const next = !selectMode
    setSelectMode(next)
    if (next) { setEditMode(false) }
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
    iw?.postMessage({ type: 'SET_EDIT_MODE',       value: next  }, '*')
    iw?.postMessage({ type: 'SET_CHART_EDIT_MODE', value: false }, '*')
  }

  function clearSelect() {
    setSelectMode(false)
    setEditMode(false)
    setSelected(null)
    const iw = iframeRef.current?.contentWindow
    iw?.postMessage({ type: 'CLEAR_CHART_EDIT' }, '*')
    iw?.postMessage({ type: 'SET_CHART_EDIT_MODE', value: false }, '*')
    iw?.postMessage({ type: 'SET_EDIT_MODE',       value: false }, '*')
  }

  function handleReset() {
    if (!window.confirm('수정한 모든 내용을 폐기하고 원본 보고서로 되돌립니다. 계속할까요?')) return
    // 원본 props로 복원
    setCurrentHtml(html)
    setCurrentReportData(reportData ? JSON.parse(JSON.stringify(reportData)) : {})
    // undo/redo 스택 초기화
    undoStackRef.current = []
    redoStackRef.current = []
    setCanUndo(false)
    setCanRedo(false)
    // 선택/편집 모드 해제
    clearSelect()
  }

  // 스크롤 bottom
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  function pushUndo() {
    undoStackRef.current.push({
      html: currentHtmlRef.current,
      report_data: currentReportRef.current ? { ...currentReportRef.current } : null,
    })
    setCanUndo(true)
    // 새 작업이 들어오면 redo 스택은 비움 (분기 후 새 branch)
    redoStackRef.current = []
    setCanRedo(false)
  }

  function handleUndo() {
    if (!undoStackRef.current.length) return
    // 현재 상태를 redo 스택에 푸시
    redoStackRef.current.push({
      html: currentHtmlRef.current,
      report_data: currentReportRef.current ? { ...currentReportRef.current } : null,
    })
    setCanRedo(true)
    const prev = undoStackRef.current.pop()
    setCurrentHtml(prev.html)
    if (prev.report_data) setCurrentReportData(prev.report_data)
    setCanUndo(undoStackRef.current.length > 0)
  }

  function handleRedo() {
    if (!redoStackRef.current.length) return
    // 현재 상태를 undo 스택에 푸시
    undoStackRef.current.push({
      html: currentHtmlRef.current,
      report_data: currentReportRef.current ? { ...currentReportRef.current } : null,
    })
    setCanUndo(true)
    const next = redoStackRef.current.pop()
    setCurrentHtml(next.html)
    if (next.report_data) setCurrentReportData(next.report_data)
    setCanRedo(redoStackRef.current.length > 0)
  }

  function addMsg(role, text) {
    setMessages(prev => [...prev, { role, text }])
  }

  async function send() {
    const msg = input.trim()
    if (!msg || loading) return
    setInput('')
    clearSelect()
    addMsg('user', msg)
    historyRef.current.push({ role: 'user', content: msg })
    setLoading(true)

    try {
      const res = await fetch(`${apiUrl || API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: msg,
          history: historyRef.current.slice(0, -1),
          tool_cache: toolCache || {},
          context: 'report_edit',
          current_html: currentHtml,
          current_report_data: currentReportData,
        }),
      })

      const reader  = res.body.getReader()
      const decoder = new TextDecoder()
      let botText = ''
      let buffer  = ''

      const finalizeStreaming = () => {
        setMessages(prev => prev.map(m => m.streaming ? { ...m, streaming: false } : m))
        if (botText) historyRef.current.push({ role: 'assistant', content: botText })
        botText = ''
      }

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop()

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const raw = line.slice(6).trim()
          if (!raw || raw === '[DONE]') continue
          let event
          try { event = JSON.parse(raw) } catch { continue }

          if (event.type === 'text') {
            botText += event.content
            const snap = botText
            setMessages(prev => {
              const next = [...prev]
              const last = next[next.length - 1]
              if (last?.role === 'bot' && last?.streaming) {
                next[next.length - 1] = { ...last, text: snap }
              } else {
                next.push({ role: 'bot', text: snap, streaming: true })
              }
              return next
            })
          }
          if (event.type === 'tool_start') {
            finalizeStreaming()
            const label = ({
              scan_data: '🔍 데이터 스캔 중...',
              get_importance: '📋 Feature importance 조회 중...',
              analyze_features: '📊 Feature 분포 비교 중...',
              infer_period: '📅 기간 해석 중...',
            })[event.tool] || `🔧 ${event.tool} 실행 중...`
            setMessages(prev => [...prev, { role: 'bot', text: label, tool: true }])
          }
          if (event.type === 'tool_result') {
            finalizeStreaming()
          }
          // 수정된 HTML이 보고서로 내려오면 미리보기 갱신
          if (event.type === 'report_ready' && event.html) {
            pushUndo()
            setCurrentHtml(event.html)
            if (event.report_data) setCurrentReportData(event.report_data)
          }
          if (event.type === 'confirm' && event.buttons) {
            finalizeStreaming()
            setMessages(prev => [...prev, { role: 'bot', text: '', buttons: event.buttons }])
          }
          if (event.type === 'done') setLoading(false)
          if (event.type === 'error') {
            addMsg('bot', `⚠️ ${event.message}`)
            setLoading(false)
          }
        }
      }

      finalizeStreaming()
    } catch {
      addMsg('bot', '⚠️ 서버 연결에 실패했습니다.')
      setLoading(false)
    }
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
  }

  async function downloadPptx() {
    // 현재 iframe 내 HTML 캡처 (인라인 텍스트 편집 반영용)
    let liveHtml = currentHtmlRef.current || ''
    try {
      const iframeDoc = iframeRef.current?.contentDocument
      if (iframeDoc?.documentElement) {
        liveHtml = '<!DOCTYPE html>' + iframeDoc.documentElement.outerHTML
      }
    } catch {}
    const res = await fetch(`${apiUrl || API_URL}/report/pptx`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        report_data: currentReportRef.current || reportData || {},
        filename: '품질불량개선조치보고서.pptx',
        current_html: liveHtml,
      }),
    })
    if (!res.ok) { alert('PPTX 생성 실패'); return }
    const blob = await res.blob()
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement('a')
    a.href = url; a.download = '품질불량개선조치보고서.pptx'; a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="rm-overlay" onClick={onClose}>
      <div className="rm-container" onClick={e => e.stopPropagation()}>

        {/* 상단 툴바 */}
        <div className="rm-toolbar">
          <span className="rm-toolbar-title">📄 Field Health 보고서 미리보기</span>
          <div className="rm-toolbar-actions">
            <button
              className={`rm-tool-btn ${selectMode ? 'active' : ''}`}
              onClick={toggleSelectMode}
              title="차트/표를 클릭하거나 드래그로 영역 선택 → 액션 메뉴"
            >
              ✏️ 차트 편집
            </button>
            <button
              className={`rm-tool-btn ${editMode ? 'active' : ''}`}
              onClick={toggleEditMode}
              title="텍스트를 더블클릭해서 직접 수정"
            >
              📝 텍스트 편집
            </button>
            <button className="rm-tool-btn" onClick={handleReset} title="모든 수정사항 폐기하고 원본 보고서로 복원">
              🔄 초기화
            </button>
            <button
              className={`rm-tool-btn undo ${canUndo ? '' : 'disabled'}`}
              onClick={handleUndo}
              disabled={!canUndo}
              title="이전 보고서로 되돌리기"
            >
              ↩
            </button>
            <button
              className={`rm-tool-btn redo ${canRedo ? '' : 'disabled'}`}
              onClick={handleRedo}
              disabled={!canRedo}
              title="앞으로 가기 (되돌린 작업 재실행)"
            >
              ↪
            </button>
            <button className="rm-tool-btn ppt" onClick={downloadPptx}>
              📊 PPT
            </button>
            <button className="rm-tool-btn close" onClick={onClose}>
              ✕ 닫기
            </button>
          </div>
        </div>

        {/* 선택된 영역 알림 */}
        {selectedSection && (
          <div className="rm-selection-bar">
            <span>📌 선택됨: <strong>{selectedSection}</strong></span>
            <span className="rm-selection-hint">→ 우측 채팅에 수정 요청을 입력하세요</span>
          </div>
        )}

        {/* 본문: 좌(미리보기) + 우(챗봇) */}
        <div className="rm-body">

          {/* 좌: HTML 미리보기 */}
          <div className={`rm-preview ${selectMode ? 'select-mode' : ''}`}>
            {blobUrl && (
              <iframe
                ref={iframeRef}
                src={blobUrl}
                title="보고서 미리보기"
                className="rm-iframe"
              />
            )}
          </div>

          {/* 우: AI 수정 에이전트 */}
          <div className="rm-chat">
            <div className="rm-chat-header">
              <span>보고서 수정 툴</span>
            </div>

            <div className="rm-chat-messages">
              {messages.map((m, i) => (
                <div key={i} className={`rm-msg ${m.role}`}>
                  {m.role === 'bot' && <div className="rm-avatar">AI</div>}
                  <div style={{display:'flex',flexDirection:'column',gap:'6px'}}>
                    <div className={`rm-bubble ${m.streaming ? 'streaming' : ''} ${m.tool ? 'tool' : ''}`}>
                      {m.text}
                    </div>
                    {m.buttons && m.buttons.length > 0 && (
                      <div style={{display:'flex',flexWrap:'wrap',gap:'6px',paddingLeft:'4px'}}>
                        {m.buttons.map((btn, bi) => (
                          <button key={bi}
                            style={{background:'#EFF6FF',border:'1px solid #93C5FD',borderRadius:'6px',
                                    padding:'4px 10px',fontSize:'11px',cursor:'pointer',color:'#1D4ED8'}}
                            onClick={() => { setInput(btn); }}
                          >{btn}</button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {loading && (
                <div className="rm-msg bot">
                  <div className="rm-avatar">AI</div>
                  <div className="rm-bubble">
                    <span className="rm-typing"><span/><span/><span/></span>
                  </div>
                </div>
              )}
              <div ref={bottomRef} />
            </div>

            <div className="rm-chat-input">
              <textarea
                className="rm-input"
                placeholder={selectMode
                  ? '영역을 선택하면 자동으로 입력됩니다...'
                  : '질문하거나 수정을 요청하세요...'}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKey}
                rows={2}
                disabled={loading}
              />
              <button className="rm-send" onClick={send} disabled={loading}>
                전송
              </button>
            </div>
          </div>

        </div>
      </div>
    </div>
  )
}
