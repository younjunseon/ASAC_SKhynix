import { useState, useEffect } from 'react'
import './TopBar.css'

const FIXED_DATE = '2026-06-11'

export default function TopBar({ notifOpen, setNotifOpen }) {
  const [timeStr, setTimeStr] = useState('')

  useEffect(() => {
    const update = () => {
      setTimeStr(new Date().toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', hour12: false }))
    }
    update()
    const timer = setInterval(update, 1000)
    return () => clearInterval(timer)
  }, [])

  return (
    <header className="topbar">
      <div className="topbar-left">
        <div className="topbar-title">품질 분석 시스템</div>
      </div>

      <div className="topbar-right">
        <div className="date-pill">{FIXED_DATE} · {timeStr}</div>

        <button
          className="icon-btn"
          onClick={() => setNotifOpen(v => !v)}
          title="알림"
        >
          🔔
          <span className="notif-dot" />
        </button>
      </div>
    </header>
  )
}
