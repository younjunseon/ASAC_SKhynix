import { useState, useMemo, useRef, useEffect, useCallback } from 'react'
import ReactECharts from 'echarts-for-react'
import { useCSV } from '../hooks/useCSV'
import './WaferMap.css'

function lotToDate(lot) {
  const n = Math.round(parseFloat(lot))
  let base, offset
  if (n >= 201) {
    // 합성 lot: 201~326 → 2026-05-28 ~ 2026-06-10 (하루 9개)
    base = new Date('2026-05-28')
    offset = Math.floor((n - 201) / 9)
  } else if (n >= 101) {
    // 합성 lot: 101~156 → 2026-04-11 ~ 2026-06-05
    base = new Date('2026-04-11')
    offset = n - 101
  } else if (n <= 28) {
    base = new Date('2026-03-27')
    offset = Math.round((n - 1) * (45 / 27))
  } else if (n <= 56) {
    base = new Date('2026-05-12')
    offset = n - 29
  } else {
    base = new Date('2026-06-11')
    offset = n - 57
  }
  const d = new Date(base)
  d.setDate(d.getDate() + offset)
  return d.toISOString().slice(0, 10)
}

// 실제 데이터 범위에서 정규화된 좌표계로 변환 (정사각형 그리드)
// die_x: 12~66 (중심 39, 반경 27), die_y: 11~32 (중심 21.5, 반경 10.5)
// 정규화: x_norm = (die_x - 39) / 27, y_norm = (die_y - 21.5) / 10.5
// → 둘 다 [-1, 1] 범위, 원형 웨이퍼 경계 = x_norm² + y_norm² ≤ 1
const WAFER_CX = 39, WAFER_CY = 21.5, WAFER_RX = 27, WAFER_RY = 10.5
const NORM_R = 1.0

function normX(x) { return (x - WAFER_CX) / WAFER_RX }
function normY(y) { return (y - WAFER_CY) / WAFER_RY }

// dangerThresh는 데이터 로드 후 동적으로 Q3 계산 (아래 dangerThresh useMemo 참조)
const CARD = { background: '#fff', borderRadius: 12, padding: 16, boxShadow: '0 1px 4px rgba(0,0,0,0.08)', marginBottom: 16 }

export default function WaferMap() {
  const { data: dies, loading } = useCSV('/wafer_map.csv')

  const [selDate, setSelDate]   = useState(null)   // 'YYYY-MM-DD'
  const [selLot,  setSelLot]    = useState(null)   // number
  const [selWafer, setSelWafer] = useState(null)   // number
  const waferChartRef = useRef(null)

  // 위험 임계값: 전체 die pred의 Q3
  const dangerThresh = useMemo(() => {
    const allPreds = dies
      .map(d => parseFloat(d.pred))
      .filter(isFinite)
      .sort((a, b) => a - b)
    if (!allPreds.length) return 0
    return allPreds[Math.floor(allPreds.length * 0.75)] ?? 0
  }, [dies])

  const drawCircle = useCallback(() => {
    const chart = waferChartRef.current?.getEchartsInstance?.()
    if (!chart) return
    const center = chart.convertToPixel('grid', [0, 0])
    // NORM_R=1.0 기준 픽셀 반경 (PAD 제외한 실제 웨이퍼 경계)
    const edgeX  = chart.convertToPixel('grid', [NORM_R, 0])
    const rx = Math.abs(edgeX[0] - center[0])
    if (!rx || isNaN(rx)) return
    chart.setOption({ graphic: [{ type: 'circle', shape: { cx: center[0], cy: center[1], r: rx }, style: { fill: 'none', stroke: '#94A3B8', lineWidth: 2 }, z: 100 }] })
  }, [])

  useEffect(() => {
    if (!selLot || !selWafer) return
    const t = setTimeout(drawCircle, 80)
    return () => clearTimeout(t)
  }, [selLot, selWafer, drawCircle])

  // 최근 2주 날짜 목록 + 날짜별 위험 unit 수
  const dateBarOption = useMemo(() => {
    if (!dies.length) return null
    // 날짜 목록 (val + 합성 lot만, train/test 제외)
    const dateMap = {}
    dies.forEach(d => {
      const dateStr = d.date || lotToDate(Math.round(parseFloat(d.run_id)))
      if (!dateMap[dateStr]) dateMap[dateStr] = { danger: 0, total: 0 }
      dateMap[dateStr].total++
      if (parseFloat(d.pred) > dangerThresh) dateMap[dateStr].danger++
    })
    // 최근 14개 날짜만
    const sorted = Object.entries(dateMap).sort((a, b) => a[0].localeCompare(b[0])).slice(-14)
    return {
      tooltip: { trigger: 'axis', formatter: p => `${p[0].axisValue}<br/>위험 unit: ${p[0].value}개` },
      grid: { top: 10, bottom: 55, left: 50, right: 10 },
      xAxis: { type: 'category', data: sorted.map(([d]) => d), axisLabel: { fontSize: 9, rotate: 35 } },
      yAxis: { type: 'value', axisLabel: { fontSize: 10 } },
      series: [{
        type: 'bar',
        data: sorted.map(([d, v]) => ({
          value: v.danger, date: d,
          itemStyle: { color: d === selDate ? '#6366F1' : '#EF4444', borderRadius: [3, 3, 0, 0] },
        })),
        barMaxWidth: 28,
      }],
    }
  }, [dies, selDate, dangerThresh])

  // 선택 날짜의 로트별 위험 unit 수
  const lotBarOption = useMemo(() => {
    if (!selDate || !dies.length) return null
    const dayDies = dies.filter(d => (d.date || lotToDate(Math.round(parseFloat(d.run_id)))) === selDate)
    const lotMap = {}
    dayDies.forEach(d => {
      const lot = Math.round(parseFloat(d.run_id))
      if (!lotMap[lot]) lotMap[lot] = { danger: 0, total: 0 }
      lotMap[lot].total++
      if (parseFloat(d.pred) > dangerThresh) lotMap[lot].danger++
    })
    const sorted = Object.entries(lotMap).sort((a, b) => Number(a[0]) - Number(b[0]))
    return {
      tooltip: { trigger: 'axis', formatter: p => `Lot ${p[0].axisValue}<br/>위험 unit: ${p[0].value}개` },
      grid: { top: 10, bottom: 40, left: 50, right: 10 },
      xAxis: { type: 'category', data: sorted.map(([lot]) => `L${lot}`), axisLabel: { fontSize: 10 } },
      yAxis: { type: 'value', axisLabel: { fontSize: 10 } },
      series: [{
        type: 'bar',
        data: sorted.map(([lot, v]) => ({
          value: v.danger, lot: parseFloat(lot),
          itemStyle: { color: parseFloat(lot) === selLot ? '#6366F1' : '#F97316', borderRadius: [3, 3, 0, 0] },
        })),
        barMaxWidth: 36,
      }],
    }
  }, [dies, selDate, selLot, dangerThresh])

  // 선택 로트의 웨이퍼별 위험 unit 수
  const waferBarOption = useMemo(() => {
    if (!selLot || !dies.length) return null
    const lotDies = dies.filter(d => Math.round(parseFloat(d.run_id)) === selLot)
    const waferMap = {}
    lotDies.forEach(d => {
      const w = Math.round(parseFloat(d.wafer_no))
      if (!waferMap[w]) waferMap[w] = { danger: 0, total: 0 }
      waferMap[w].total++
      if (parseFloat(d.pred) > dangerThresh) waferMap[w].danger++
    })
    const sorted = Object.entries(waferMap).sort((a, b) => Number(a[0]) - Number(b[0]))
    return {
      tooltip: { trigger: 'axis', formatter: p => `Wafer ${p[0].axisValue}<br/>위험 unit: ${p[0].value}개` },
      grid: { top: 10, bottom: 40, left: 50, right: 10 },
      xAxis: { type: 'category', data: sorted.map(([w]) => `W${w}`), axisLabel: { fontSize: 10 } },
      yAxis: { type: 'value', axisLabel: { fontSize: 10 } },
      series: [{
        type: 'bar',
        data: sorted.map(([w, v]) => ({
          value: v.danger, wafer: parseFloat(w),
          itemStyle: { color: parseFloat(w) === selWafer ? '#6366F1' : '#F97316', borderRadius: [3, 3, 0, 0] },
        })),
        barMaxWidth: 28,
      }],
    }
  }, [dies, selLot, selWafer])

  // 웨이퍼맵
  const waferMapOption = useMemo(() => {
    if (!selLot || !selWafer || !dies.length) return null
    const raw = dies.filter(d =>
      Math.round(parseFloat(d.run_id)) === selLot &&
      Math.round(parseFloat(d.wafer_no)) === selWafer
    )
    if (!raw.length) return null
    const filtered = raw
    if (!filtered.length) return null
    const maxPred = Math.max(...filtered.map(d => parseFloat(d.pred)))
    // die 수에 따라 심볼 크기 동적 계산 (외곽 짤림 방지 위해 여백 확보)
    const PAD = 0.15
    // 웨이퍼 전체 x 범위 대비 die 하나의 비율로 심볼 폭 추정
    const xVals = filtered.map(d => parseFloat(d.die_x))
    const yVals = filtered.map(d => parseFloat(d.die_y))
    const xRange = Math.max(...xVals) - Math.min(...xVals) + 1
    const yRange = Math.max(...yVals) - Math.min(...yVals) + 1
    // 500px 캔버스 기준, 그리드 영역 ~380px, die 하나 폭
    const CANVAS = 380
    const symW = Math.max(3, Math.floor(CANVAS / xRange) - 1)
    const symH = Math.max(3, Math.floor(CANVAS / yRange) - 1)
    return {
      tooltip: { formatter: p => `die(${p.data.origX}, ${p.data.origY})<br/>예측 불량지수: ${parseFloat(p.data.value[2]).toFixed(6)}` },
      visualMap: { min: 0, max: maxPred || 0.01, dimension: 2, calculable: true, orient: 'horizontal', left: 'center', bottom: 8, inRange: { color: ['#166534', '#84CC16', '#FCD34D', '#F97316', '#DC2626'] }, textStyle: { fontSize: 12 } },
      grid: { top: 20, bottom: 70, left: 20, right: 20, containLabel: false },
      xAxis: { type: 'value', min: -(NORM_R + PAD), max: NORM_R + PAD, show: false, splitLine: { show: false } },
      yAxis: { type: 'value', min: -(NORM_R + PAD), max: NORM_R + PAD, show: false, splitLine: { show: false } },
      series: [{
        type: 'scatter',
        data: filtered.map(d => ({ value: [normX(parseFloat(d.die_x)), normY(parseFloat(d.die_y)), parseFloat(d.pred)], origX: parseFloat(d.die_x), origY: parseFloat(d.die_y) })),
        symbol: 'rect', symbolSize: [symW, symH], emphasis: { scale: false },
      }],
    }
  }, [dies, selLot, selWafer])

  if (loading) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#94A3B8', fontSize: 11 }}>
      데이터 로딩 중…
    </div>
  )

  return (
    <div className="wafermap-page">
      <div className="wm-header">
        <div className="wm-title">🗺 웨이퍼맵</div>
        <div className="wm-desc">날짜 → 로트 → 웨이퍼 → 웨이퍼맵 순으로 드릴다운합니다.</div>
      </div>
      <div style={{ fontSize:13, color:'#64748B', marginBottom:8 }}>
        📅 기준일: 2026-05-07 — 최근 WT 완료분 기준
      </div>

      {/* Step 1: 최근 2주 날짜별 위험 unit 수 */}
      <div style={CARD}>
        <div style={{ fontSize: 11, fontWeight: 600, color: '#1E293B', marginBottom: 8 }}>
          📅 최근 2주 날짜별 위험 unit 수 — 막대 클릭 시 로트 상세
          {selDate && <span style={{ marginLeft: 8, color: '#6366F1' }}>선택: {selDate}</span>}
        </div>
        {dateBarOption
          ? <ReactECharts option={dateBarOption} style={{ height: 220 }}
              onEvents={{ click: p => { setSelDate(p.data.date); setSelLot(null); setSelWafer(null) } }}
            />
          : <div style={{ color: '#94A3B8', fontSize: 11, textAlign: 'center', padding: 40 }}>데이터 없음</div>
        }
      </div>

      {/* Step 2: 로트별 위험 unit 수 */}
      {selDate && (
        <div style={CARD}>
          <div style={{ fontSize: 11, fontWeight: 600, color: '#1E293B', marginBottom: 8 }}>
            🏭 {selDate} — 로트별 위험 unit 수 — 막대 클릭 시 웨이퍼 상세
            {selLot && <span style={{ marginLeft: 8, color: '#6366F1' }}>선택: Lot {selLot}</span>}
          </div>
          {lotBarOption
            ? <ReactECharts option={lotBarOption} style={{ height: 220 }}
                onEvents={{ click: p => { setSelLot(p.data.lot); setSelWafer(null) } }}
              />
            : <div style={{ color: '#94A3B8', fontSize: 11, textAlign: 'center', padding: 40 }}>데이터 없음</div>
          }
        </div>
      )}

      {/* Step 3: 웨이퍼별 위험 unit 수 */}
      {selLot && (
        <div style={CARD}>
          <div style={{ fontSize: 11, fontWeight: 600, color: '#1E293B', marginBottom: 8 }}>
            🏭 Lot {selLot} ({selDate}) — 웨이퍼별 위험 unit 수 — 막대 클릭 시 웨이퍼맵
            {selWafer && <span style={{ marginLeft: 8, color: '#6366F1' }}>선택: Wafer {selWafer}</span>}
          </div>
          {waferBarOption
            ? <ReactECharts option={waferBarOption} style={{ height: 220 }}
                onEvents={{ click: p => setSelWafer(p.data.wafer) }}
              />
            : <div style={{ color: '#94A3B8', fontSize: 11, textAlign: 'center', padding: 40 }}>데이터 없음</div>
          }
        </div>
      )}

      {/* Step 4: 웨이퍼맵 */}
      {selLot && selWafer && (
        <div style={{ ...CARD, marginBottom: 0 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: '#1E293B', marginBottom: 8 }}>
            🗺 Lot {selLot} — Wafer {selWafer} 웨이퍼맵
            <span style={{ marginLeft: 8, fontSize: 13, color: '#64748B', fontWeight: 400 }}>색상: 예측 불량지수 (진초록→연두→노랑→주황→빨강)</span>
          </div>
          {waferMapOption
            ? <div style={{ display: 'flex', justifyContent: 'center' }}>
                <ReactECharts ref={waferChartRef} option={waferMapOption} style={{ width: 500, height: 500 }} onChartReady={drawCircle} />
              </div>
            : <div style={{ color: '#94A3B8', fontSize: 11, textAlign: 'center', padding: 40 }}>데이터 없음</div>
          }
        </div>
      )}
    </div>
  )
}
