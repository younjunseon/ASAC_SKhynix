import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import './ChatBot.css'

const API_URL = ''

const INIT_ASSISTANT = {
  role: 'bot',
  text: '안녕하세요! SK Hynix Wafer Test 도메인 전문 어시스턴트입니다.\n반도체, DRAM, 공정, 모델링 관련 질문을 해주세요.',
}

export default function ChatBot({ open, onClose }) {
  const [assistMsgs, setAssistMsgs] = useState([INIT_ASSISTANT])
  const assistHistoryRef = useRef([])
  const [input, setInput]    = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [assistMsgs])

  // ── 어시스턴트 전송 (RAG, JSON 응답) ─────────────────────────
  async function sendAssistant(msg) {
    setAssistMsgs(prev => [...prev, { role: 'user', text: msg }])
    setLoading(true)

    try {
      const res = await fetch(`${API_URL}/chat/assistant`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg, history: assistHistoryRef.current }),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(err.detail || res.statusText)
      }
      const data = await res.json()
      assistHistoryRef.current = data.history || []
      setAssistMsgs(prev => [...prev, { role: 'bot', text: data.response }])
    } catch (e) {
      setAssistMsgs(prev => [...prev, {
        role: 'bot',
        text: `⚠️ ${e.message || '서버 연결에 실패했습니다. RAG 서버(8002)가 실행 중인지 확인해주세요.'}`,
        error: true,
      }])
    } finally {
      setLoading(false)
    }
  }

  async function send() {
    const msg = input.trim()
    if (!msg || loading) return
    setInput('')
    await sendAssistant(msg)
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
  }

  function handleReset() {
    setAssistMsgs([INIT_ASSISTANT])
    assistHistoryRef.current = []
    setInput('')
  }

  return (
    <div className={`chatbot-panel ${open ? 'open' : ''}`}>
      <div className="cb-header">
        <div className="cb-title">💬 AI 어시스턴트</div>
        <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
          <button className="cb-close" onClick={handleReset} title="대화 초기화">🔄</button>
          <button className="cb-close" onClick={onClose}>✕</button>
        </div>
      </div>

      <div className="cb-messages">
        {assistMsgs.map((m, i) => (
          <div key={i} className={`cb-msg ${m.role}`}>
            {m.role === 'bot' && <div className="cb-avatar">AI</div>}
            <div className={`cb-bubble ${m.error ? 'error' : ''}`}>
              {m.role === 'bot'
                ? <ReactMarkdown>{m.text}</ReactMarkdown>
                : <span>{m.text}</span>
              }
            </div>
          </div>
        ))}

        {loading && (
          <div className="cb-msg bot">
            <div className="cb-avatar">AI</div>
            <div className="cb-bubble status">
              <span className="cb-typing"><span /><span /><span /></span>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="cb-input-row">
        <textarea
          className="cb-input"
          placeholder="반도체, 공정, 모델링 관련 질문을 해주세요..."
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKey}
          rows={2}
          disabled={loading}
        />
        <button className="cb-send" onClick={send} disabled={loading}>전송</button>
      </div>
    </div>
  )
}
