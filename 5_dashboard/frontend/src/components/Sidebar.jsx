import { useState } from 'react'
import './Sidebar.css'
import skLogo from '../assets/sk_logo_nobg.png'

const MENU = [
  { id: 'overview2',      label: '불량 현황',      icon: '◈' },
  { id: 'drilldown-v2',   label: '불량 상세 분석', icon: '◉' },
  { id: 'process-factor', label: '공정 인자 진단', icon: '◈' },
  { id: 'report',         label: '보고서',         icon: '📄' },
]

function ButterflyLogo() {
  return <img src={skLogo} alt="SK hynix" width="42" height="42" style={{ objectFit: 'contain' }} />
}

function PersonIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#6B7280" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="8" r="4" />
      <path d="M4 20c0-4 3.6-7 8-7s8 3 8 7" />
    </svg>
  )
}

export default function Sidebar({ activePage, setActivePage, open, setOpen }) {
  const [logoHover, setLogoHover] = useState(false)

  return (
    <>
      {open && <div className="sidebar-overlay" onClick={() => setOpen(false)} />}

      <nav className={`sidebar ${open ? 'open' : ''}`}>
        {/* 로고 버튼 — 호버 시 토글 화살표로 교체 */}
        <button
          className="sidebar-logo-btn"
          onClick={() => setOpen(o => !o)}
          onMouseEnter={() => setLogoHover(true)}
          onMouseLeave={() => setLogoHover(false)}
          title={open ? '접기' : '펼치기'}
        >
          {logoHover
            ? <span className="sidebar-toggle-icon">{open ? '‹' : '›'}</span>
            : <ButterflyLogo />
          }
          {open && !logoHover && <span className="sidebar-logo-text">SK hynix</span>}
          {open && logoHover && <span className="sidebar-logo-text sidebar-logo-text--hint">접기</span>}
        </button>

        <div className="sidebar-divider" />

        {/* 메뉴 아이템 — 클릭해도 사이드바 유지 */}
        {MENU.map(item => (
          <div
            key={item.id}
            className={`nav-item ${activePage === item.id ? 'active' : ''}`}
            onClick={() => setActivePage(item.id)}
            title={!open ? item.label : undefined}
          >
            <span className="nav-icon">{item.icon}</span>
            {open && <span className="nav-label">{item.label}</span>}
          </div>
        ))}

        {/* 하단 프로필 */}
        <div className="sidebar-spacer" />
        <div className="sidebar-divider" />
        <div className="sidebar-profile" title={!open ? '이정훈' : undefined}>
          <div className="sidebar-avatar">
            <PersonIcon />
          </div>
          {open && <span className="sidebar-profile-name">이정훈</span>}
        </div>
      </nav>
    </>
  )
}
