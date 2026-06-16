import { useState, useMemo } from 'react'
import TopBar from './components/TopBar'
import Sidebar from './components/Sidebar'
import ChatBot from './components/ChatBot'
import Overview2 from './pages/Overview2'
import WaferMap from './pages/WaferMap'
import ModelPerformanceV2 from './pages/ModelPerformanceV2'
import ProcessFactor from './pages/ProcessFactor'
import DrilldownV2 from './pages/DrilldownV2'
import ReportPage from './pages/ReportPage'
import { useCSV } from './hooks/useCSV'
import './App.css'

export default function App() {
  const [activePage, setActivePage] = useState('overview2')
  const [notifOpen, setNotifOpen] = useState(false)
  const [chatOpen, setChatOpen] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  // 페이지 간 selection 전달용 (Overview에서 클릭한 unit → DrilldownV2로 전달)
  const [pendingSelection, setPendingSelection] = useState(null)  // { lot, wafer, unit }
  const { data: units } = useCSV('/dashboard_units.csv')

  // Overview에서 호출: DrilldownV2로 이동하면서 unit 선택
  const navigateToDrilldown = (selection) => {
    setPendingSelection(selection)
    setActivePage('drilldown-v2')
  }

  // grade별 unit 목록
  const notifByGrade = useMemo(() => {
    if (!units.length) return { grade1: [], grade2: [], grade3: [], grade4: [] }
    const result = { grade1: [], grade2: [], grade3: [], grade4: [] }
    units.forEach(u => {
      const g = u.grade
      if (result[g]) result[g].push(u.ufs_serial)
    })
    return result
  }, [units])
  function renderPageWithProps(page) {
    switch (page) {
      case 'overview':
        return <Overview onNavigateDrilldown={navigateToDrilldown} />
      case 'overview2':
        return <Overview2
          onNavigateDrilldown={(sel) => { setPendingSelection(sel); setActivePage('drilldown-v2') }}
          onNavigateProcessFactor={() => setActivePage('process-factor')}
        />
      case 'wafer-map':
        return <WaferMap />
      case 'feat-importance-v2':
        return <ModelPerformanceV2 />
      case 'process-factor':
        return <ProcessFactor />
      case 'drilldown-v2':
        return <DrilldownV2 initialSelection={pendingSelection} />
      case 'report':
        return <ReportPage />
      default:
        return <Overview onNavigateDrilldown={navigateToDrilldown} />
    }
  }

  return (
    <div className="app">
      <TopBar notifOpen={notifOpen} setNotifOpen={setNotifOpen} />

      <div className="app-body">
        <Sidebar activePage={activePage} setActivePage={(p) => { setPendingSelection(null); setActivePage(p) }} open={sidebarOpen} setOpen={setSidebarOpen} />

        <main className="main-content">
          {renderPageWithProps(activePage)}
        </main>

        <ChatBot open={chatOpen} onClose={() => setChatOpen(false)} />
      </div>


      {notifOpen && <div className="overlay" onClick={() => setNotifOpen(false)} />}
      <div className={`notif-panel ${notifOpen ? 'open' : ''}`}>
        <div className="np-header">
          <div className="np-title">🔔 위험 감지 알림</div>
          <button className="np-close" onClick={() => setNotifOpen(false)}>✕</button>
        </div>
        <div className="np-list">
          {!units.length && (
            <div style={{ padding: 16, color: '#94A3B8', fontSize: 11 }}>데이터 로딩 중…</div>
          )}
          {units.length > 0 && (() => {
            const list = notifByGrade['grade4'] ?? []
            if (!list.length) return (
              <div style={{ padding: 16, color: '#94A3B8', fontSize: 11 }}>위험 감지 유닛 없음</div>
            )
            return (
              <div className="np-grade-section">
                <div className="np-grade-header" style={{ background: '#FEE2E2', borderColor: '#EF4444' }}>
                  <span className="np-grade-label" style={{ color: '#EF4444' }}>⚠ 매우위험 (Grade 4)</span>
                  <span className="np-grade-count" style={{ color: '#EF4444' }}>{list.length.toLocaleString()}개</span>
                </div>
                <div className="np-unit-list">
                  {list.slice(0, 50).map(serial => (
                    <div key={serial} className="np-unit-item">{serial}</div>
                  ))}
                  {list.length > 50 && (
                    <div className="np-unit-more">+{(list.length - 50).toLocaleString()}개 더</div>
                  )}
                </div>
              </div>
            )
          })()}
        </div>
      </div>
    </div>
  )
}
