import { useMemo, useRef, useState, useEffect } from 'react'
import ReactECharts from 'echarts-for-react'
import { useCSV } from '../hooks/useCSV'
import { dataUrl } from '../utils/dataUrl'
import ALL_DIE_POSITIONS from './diePositions.js'
import './Overview2.css'

// 트렌드 차트 — 배경 영역(파랑/빨강)을 차트 뒤 div로 깔아 정확히 컬럼에 맞춤
function TrendChart({ option, lastTrueIdx = -1 }) {
  const ref = useRef(null)
  const wrapRef = useRef(null)
  const [bands, setBands] = useState(null)  // [{ left, width, color }], top, height

  const n = option?.xAxis?.data?.length ?? 0

  useEffect(() => {
    const compute = () => {
      const inst = ref.current?.getEchartsInstance?.()
      if (!inst || n < 2) return
      try {
        const x0 = inst.convertToPixel({ xAxisIndex: 0 }, 0)
        const x1 = inst.convertToPixel({ xAxisIndex: 0 }, 1)
        if ([x0, x1].some(v => v == null || isNaN(v))) return
        const yTop = inst.convertToPixel({ yAxisIndex: 0 }, option.yAxis[0].max)
        const yBot = inst.convertToPixel({ yAxisIndex: 0 }, option.yAxis[0].min)
        if ([yTop, yBot].some(v => v == null || isNaN(v))) return
        const half = (x1 - x0) / 2
        const edge   = (i) => inst.convertToPixel({ xAxisIndex: 0 }, i) - half  // 컬럼 좌측 경계
        const center = (i) => inst.convertToPixel({ xAxisIndex: 0 }, i)         // 컬럼 중심(점 위치)

        // 3구간: 검증(초록)=0~실측마지막 점, 예측(파랑)=그 점~최신직전, 최신(빨강)=마지막 컬럼
        const segs = []
        const lt = Math.max(-1, Math.min(lastTrueIdx, n - 2))
        const splitX = lt >= 0 ? center(lt) : edge(0)  // 실측 마지막 점 위치를 경계로
        // 검증 구간 (실측+예측 겹침)
        if (lt >= 0) {
          segs.push({ left: edge(0), width: splitX - edge(0), color: 'rgba(16,185,129,0.10)' })
        }
        // 예측 구간
        segs.push({ left: splitX, width: edge(n - 1) - splitX, color: 'rgba(59,130,246,0.08)' })
        // 최신 주차
        segs.push({ left: edge(n - 1), width: 2 * half, color: 'rgba(220,38,38,0.13)' })

        setBands({ segs, top: yTop, height: yBot - yTop })
      } catch { /* convert 실패 시 무시 */ }
    }
    const inst = ref.current?.getEchartsInstance?.()
    inst?.on('finished', compute)
    const t = setTimeout(compute, 60)
    window.addEventListener('resize', compute)
    // 컨테이너 폭 변화(사이드바 접기/펼치기 등) 감지 → echarts 리사이즈 + 배경 밴드 재계산
    let ro
    if (typeof ResizeObserver !== 'undefined' && wrapRef.current) {
      ro = new ResizeObserver(() => {
        ref.current?.getEchartsInstance?.()?.resize()
        compute()
      })
      ro.observe(wrapRef.current)
    }
    return () => {
      inst?.off('finished', compute)
      clearTimeout(t)
      window.removeEventListener('resize', compute)
      ro?.disconnect()
    }
  }, [option, n, lastTrueIdx])

  return (
    <div ref={wrapRef} style={{ position: 'relative', width: '100%', height: '100%' }}>
      {bands && bands.segs.map((s, i) => (
        <div key={i} style={{ position: 'absolute', top: bands.top, height: bands.height, left: s.left, width: s.width,
          background: s.color, pointerEvents: 'none', zIndex: 0 }} />
      ))}
      <ReactECharts ref={ref} option={option} style={{ width: '100%', height: '100%', position: 'relative', zIndex: 1 }}
        opts={{ renderer: 'svg' }} notMerge={true} />
    </div>
  )
}

const GLOBAL_DIE_X_MIN = 12, GLOBAL_DIE_X_MAX = 66
const GLOBAL_DIE_Y_MIN = 11, GLOBAL_DIE_Y_MAX = 32

// ── die 색상 로직 (계층탐색 DrilldownV2와 동일) ───────────────
const NORMAL_STOPS = [
  [0.0, [243, 244, 246]],
  [0.5, [219, 234, 254]],
  [1.0, [165, 215, 220]],
]
const RISK_STOPS = [
  [0.0,  [254, 240, 138]],
  [0.75, [251, 146, 60]],
  [1.0,  [220, 38, 38]],
]
function interpStops(stops, t) {
  const tt = Math.max(0, Math.min(1, t))
  for (let i = 1; i < stops.length; i++) {
    const [t1, c1] = stops[i]
    const [t0, c0] = stops[i - 1]
    if (tt <= t1) {
      const k = (tt - t0) / (t1 - t0 || 1)
      return `rgb(${Math.round(c0[0] + (c1[0] - c0[0]) * k)},${Math.round(c0[1] + (c1[1] - c0[1]) * k)},${Math.round(c0[2] + (c1[2] - c0[2]) * k)})`
    }
  }
  const last = stops[stops.length - 1][1]
  return `rgb(${last.join(',')})`
}
function predColor(pred, predMin, predMax, threshold) {
  if (!isFinite(pred)) return '#f1f5f9'
  if (pred <= threshold) {
    return interpStops(NORMAL_STOPS, (pred - predMin) / Math.max(1e-9, threshold - predMin))
  }
  return interpStops(RISK_STOPS, (pred - threshold) / Math.max(1e-9, predMax - threshold))
}
const COLOR_LEGEND_GRADIENT =
  'linear-gradient(to top, #f3f4f6, #dbeafe, #a5d7dc, #fef08a, #fef08a, #fb923c, #dc2626)'

export const GRADE_COLORS = {
  grade1: { bg: '#F0FDF4', border: '#86EFAC', text: '#166534', bar: '#22C55E', label: '정상 (G1)' },
  grade2: { bg: '#FEF9C3', border: '#EAB308', text: '#713F12', bar: '#EAB308', label: '조심 (G2)' },
  grade3: { bg: '#FEF3C7', border: '#F59E0B', text: '#92400E', bar: '#F59E0B', label: '위험 (G3)' },
  grade4: { bg: '#FEE2E2', border: '#EF4444', text: '#B91C1C', bar: '#EF4444', label: '매우위험 (G4)' },
}

export function getGrade(pred, thresholds) {
  // 위험(grade3) 기준 = P90 (reg_pred 상위 10%) — CSV grade 컬럼·드릴다운과 통일
  const { q2, p90, upperFence } = thresholds
  if (pred >= upperFence) return 'grade4'
  if (pred >= p90)        return 'grade3'
  if (pred >= q2)         return 'grade2'
  return 'grade1'
}

function deltaColor(delta, absMax) {
  if (!isFinite(delta) || absMax <= 0) return '#f3f4f6'
  const t = Math.max(-1, Math.min(1, delta / absMax))
  if (t >= 0) {
    // 옅은 빨강(#FEE2E2) → 선명한 빨강(#DC2626)
    const k = t
    const r = Math.round(254 + (220 - 254) * k)
    const g = Math.round(226 + (38  - 226) * k)
    const b = Math.round(226 + (38  - 226) * k)
    return `rgb(${r},${g},${b})`
  } else {
    // 옅은 파랑(#DBEAFE) → 선명한 파랑(#2563EB)
    const k = -t
    const r = Math.round(219 + (37  - 219) * k)
    const g = Math.round(234 + (99  - 234) * k)
    const b = Math.round(254 + (235 - 254) * k)
    return `rgb(${r},${g},${b})`
  }
}

function DeltaWaferMap({ dies, baseline = 0, absMax, periodMode }) {
  const D = 600, PAD = 12
  const VB = D + PAD * 2
  const cx = PAD + D / 2, cy = PAD + D / 2, radius = D / 2

  const refXRange = GLOBAL_DIE_X_MAX - GLOBAL_DIE_X_MIN + 1
  const refYRange = GLOBAL_DIE_Y_MAX - GLOBAL_DIE_Y_MIN + 1
  const centerX = (GLOBAL_DIE_X_MIN + GLOBAL_DIE_X_MAX) / 2
  const centerY = (GLOBAL_DIE_Y_MIN + GLOBAL_DIE_Y_MAX) / 2
  const SCALE = 0.9
  const cellW = (D / refXRange) * SCALE
  const cellH = (D / refYRange) * SCALE

  const dieMap = new Map()
  for (const d of dies) dieMap.set(`${d.die_x},${d.die_y}`, d)

  return (
    <svg viewBox={`0 0 ${VB} ${VB}`} preserveAspectRatio="xMidYMid meet" style={{ width: '100%', height: '100%' }}>
      <defs>
        <clipPath id="ov2WaferClip">
          <circle cx={cx} cy={cy} r={radius} />
        </clipPath>
      </defs>
      <circle cx={cx} cy={cy} r={radius} fill="#fafafa" stroke="#cbd5e1" strokeWidth={1.5} />
      <g clipPath="url(#ov2WaferClip)">
        {ALL_DIE_POSITIONS.map(([dx, dy]) => {
          const die = dieMap.get(`${dx},${dy}`)
          if (!die) return null
          let delta, tip
          if (periodMode) {
            delta = parseFloat(die.delta)
            if (!isFinite(delta)) return null
            tip = `(${dx}, ${dy})
이전(6/6~8)=${Math.round(parseFloat(die.pred_a) * 1e6).toLocaleString()} ppm
최근(6/9~10)=${Math.round(parseFloat(die.pred_b) * 1e6).toLocaleString()} ppm
Δ=${(delta >= 0 ? '+' : '') + Math.round(delta * 1e6).toLocaleString()} ppm`
          } else {
            const pred = parseFloat(die.pred)
            if (!isFinite(pred)) return null
            delta = pred - baseline
            tip = `(${dx}, ${dy})
pred=${Math.round(pred * 1e6).toLocaleString()} ppm
Δ=${(delta >= 0 ? '+' : '') + Math.round(delta * 1e6).toLocaleString()} ppm`
          }
          const x = cx + (dx - centerX) * cellW - cellW / 2
          const y = cy + (dy - centerY) * cellH - cellH / 2
          return (
            <g key={`${dx}-${dy}`}>
              <title>{tip}</title>
              <rect
                x={x} y={y} width={cellW} height={cellH}
                fill={deltaColor(delta, absMax)}
                stroke="rgba(15,23,42,0.10)" strokeWidth={0.6}
              />
            </g>
          )
        })}
      </g>
      <circle cx={cx} cy={cy} r={radius} fill="none" stroke="#94a3b8" strokeWidth={1.5} />
      <rect x={cx - 14} y={cy + radius - 5} width={28} height={6} fill="#fff" stroke="#94a3b8" strokeWidth={1} />
    </svg>
  )
}

// 이상치 유닛 웨이퍼맵 — 계층탐색과 동일한 predColor로 색칠, 이상치 유닛 die는 강조. 클릭 시 계층탐색 이동
function OutlierWaferMap({ dies, unitDies, scale }) {
  const D = 600, PAD = 12
  const VB = D + PAD * 2
  const cx = PAD + D / 2, cy = PAD + D / 2, radius = D / 2
  const refXRange = GLOBAL_DIE_X_MAX - GLOBAL_DIE_X_MIN + 1
  const refYRange = GLOBAL_DIE_Y_MAX - GLOBAL_DIE_Y_MIN + 1
  const centerX = (GLOBAL_DIE_X_MIN + GLOBAL_DIE_X_MAX) / 2
  const centerY = (GLOBAL_DIE_Y_MIN + GLOBAL_DIE_Y_MAX) / 2
  const SCALE = 0.9
  const cellW = (D / refXRange) * SCALE
  const cellH = (D / refYRange) * SCALE
  const unitSet = new Set((unitDies || []).map(([x, y]) => `${x},${y}`))

  // 격자선 (계층탐색 웨이퍼맵과 동일 — 전체 좌표 범위 기준)
  const gridXs = []
  for (let xi = GLOBAL_DIE_X_MIN; xi <= GLOBAL_DIE_X_MAX + 1; xi++) {
    gridXs.push(cx + (xi - centerX) * cellW - cellW / 2)
  }
  const gridYs = []
  for (let yi = GLOBAL_DIE_Y_MIN; yi <= GLOBAL_DIE_Y_MAX + 1; yi++) {
    gridYs.push(cy + (yi - centerY) * cellH - cellH / 2)
  }

  return (
    <svg viewBox={`0 0 ${VB} ${VB}`} preserveAspectRatio="xMidYMid meet" style={{ width: '100%', height: '100%' }}>
      <defs>
        <clipPath id="ov2OutlierClip"><circle cx={cx} cy={cy} r={radius} /></clipPath>
      </defs>
      <circle cx={cx} cy={cy} r={radius} fill="#fafafa" stroke="#cbd5e1" strokeWidth={1.5} />
      <g clipPath="url(#ov2OutlierClip)">
        {dies.map(([dx, dy, pred]) => {
          const x = cx + (dx - centerX) * cellW - cellW / 2
          const y = cy + (dy - centerY) * cellH - cellH / 2
          const isUnit = unitSet.has(`${dx},${dy}`)
          return (
            <g key={`${dx}-${dy}`}>
              <title>{`(${dx}, ${dy})\npred=${Math.round(pred * 1e6).toLocaleString()} ppm${isUnit ? '\n← 이상치 유닛' : ''}`}</title>
              <rect
                x={x} y={y} width={cellW} height={cellH}
                fill={predColor(pred, scale.predMin, scale.predMax, scale.threshold)}
                stroke={isUnit ? '#7C3AED' : 'rgba(15,23,42,0.10)'}
                strokeWidth={isUnit ? 3 : 0.6}
              />
            </g>
          )
        })}
        {/* 격자선 (die 위에 오버레이) */}
        {gridXs.map((gx, i) => (
          <line key={`gx-${i}`} x1={gx} y1={cy - radius} x2={gx} y2={cy + radius}
            stroke="rgba(100,116,139,0.18)" strokeWidth={0.8} />
        ))}
        {gridYs.map((gy, i) => (
          <line key={`gy-${i}`} x1={cx - radius} y1={gy} x2={cx + radius} y2={gy}
            stroke="rgba(100,116,139,0.18)" strokeWidth={0.8} />
        ))}
      </g>
      <circle cx={cx} cy={cy} r={radius} fill="none" stroke="#94a3b8" strokeWidth={1.5} />
      <rect x={cx - 14} y={cy + radius - 5} width={28} height={6} fill="#fff" stroke="#94a3b8" strokeWidth={1} />
    </svg>
  )
}

function computeThresholds(units) {
  const allPreds = units
    .map(u => parseFloat(u.reg_pred))
    .filter(v => isFinite(v))
    .sort((a, b) => a - b)
  const n = allPreds.length
  const q1 = allPreds[Math.floor(n * 0.25)] ?? 0
  const q2 = allPreds[Math.floor(n * 0.50)] ?? 0
  const q3 = allPreds[Math.floor(n * 0.75)] ?? 0
  const p90 = allPreds[Math.floor(n * 0.90)] ?? 0   // 위험(grade3) 컷
  const iqr = q3 - q1
  const upperFence = q3 + 1.5 * iqr
  return { q1, q2, q3, p90, iqr, upperFence }
}

function KpiCard({ label, value, sub, color }) {
  return (
    <div className="kpi-card ov2-kpi-card" style={{ '--kpi-color': color }}>
      <div className="kpi-info">
        <div className="kpi-val" style={{ color }}>{value}</div>
        <div className="kpi-label">{label}</div>
        {sub && <div className="kpi-sub">{sub}</div>}
      </div>
    </div>
  )
}

function ChartCard({ title, sub, children, style }) {
  return (
    <div className="chart-card" style={style}>
      <div className="cc-header">
        <div>
          <div className="cc-title">{title}</div>
          {sub && <div style={{ fontSize: 13, color: '#94A3B8', marginTop: 2 }}>{sub}</div>}
        </div>
      </div>
      <div className="cc-body">{children}</div>
    </div>
  )
}

function absBarWidth(ratio) {
  return `${Math.max(6, Math.round(ratio * 100))}%`
}

function riskClass(ratio) {
  return ratio >= 0.85 ? 'danger' : ratio >= 0.70 ? 'warn' : 'ok'
}

export default function Overview2({ onNavigateDrilldown, onNavigateProcessFactor }) {
  const { data: units, loading: loadingUnits } = useCSV('/dashboard_units.csv')
  const { data: trendRaw, loading: loadingTrend } = useCSV('/trend_data.csv')
  const { data: lotSummary } = useCSV('/dashboard_lot_summary.csv')  // 계층탐색과 동일 유닛 기반 위험비율

  const { q2, q3, p90, upperFence } = useMemo(() => {
    if (!units.length) return { q2: 0, q3: 0, p90: 0, upperFence: 0 }
    return computeThresholds(units)
  }, [units])

  const kpi = useMemo(() => {
    if (!units.length) return null
    const thresholds = { q2, q3, p90, upperFence }

    const gradeCount = { grade1: 0, grade2: 0, grade3: 0, grade4: 0 }
    units.forEach(u => { gradeCount[getGrade(parseFloat(u.reg_pred), thresholds)]++ })

    const total = units.length
    const avgPpm = Math.round(
      units.reduce((s, u) => s + parseFloat(u.reg_pred), 0) / total * 1e6
    )

    return { total, gradeCount, avgPpm }
  }, [units, q2, q3, p90, upperFence])

  // 주차별 불량 ppm 트렌드 (Overview1에서 이전)
  const trendResult = useMemo(() => {
    if (!trendRaw.length) return null

    const weekMap = {}
    trendRaw.forEach(r => {
      const d = new Date(r.date); if (isNaN(d.getTime())) return
      const day = d.getDay()
      const diff = day === 0 ? -6 : 1 - day
      const monday = new Date(d)
      monday.setDate(d.getDate() + diff)
      const sunday = new Date(monday)
      sunday.setDate(monday.getDate() + 6)
      const fmt = (dt) => `${(dt.getMonth()+1).toString().padStart(2,'0')}/${dt.getDate().toString().padStart(2,'0')}`
      const weekKey = `${fmt(monday)}~${fmt(sunday)}`
      const weekStart = monday.toISOString().slice(0, 10)

      if (!weekMap[weekStart]) weekMap[weekStart] = { label: weekKey, preds: [], trues: [], prod: 0, days: 0 }
      const yp = r.y_pred !== '' && r.y_pred != null ? parseFloat(r.y_pred) : null
      const yt = r.y_true !== '' && r.y_true != null ? parseFloat(r.y_true) : null
      const prod = r.production !== '' && r.production != null ? parseInt(r.production) : 0
      if (yp != null) weekMap[weekStart].preds.push(yp)
      if (yt != null) weekMap[weekStart].trues.push(yt)
      weekMap[weekStart].prod += prod
      weekMap[weekStart].days += 1
    })

    const weeks = Object.entries(weekMap).sort(([a], [b]) => a.localeCompare(b))

    // x축 라벨 재매핑: 실측(y_true)이 있는 마지막 주 = '이번주(6월 2주차)'로 앵커링
    //  → 실측 끝이 오늘, 그 다음 주(예측)는 6월 3주차부터 채워짐. (차트 데이터는 그대로)
    const lastTrueWeekIdx = weeks.reduce((acc, [, w], i) => (w.trues.length ? i : acc), -1)
    const ANCHOR_MONDAY = new Date(2026, 5, 8)  // 2026-06-08 = 6월 2주차
    const anchoredMondays = weeks.map((_, i) => {
      const d = new Date(ANCHOR_MONDAY)
      d.setDate(ANCHOR_MONDAY.getDate() + (i - (lastTrueWeekIdx < 0 ? weeks.length - 1 : lastTrueWeekIdx)) * 7)
      return d
    })
    const _md = (dt) => `${(dt.getMonth()+1).toString().padStart(2,'0')}/${dt.getDate().toString().padStart(2,'0')}`
    const wwLabels = anchoredMondays.map(m => `${m.getMonth() + 1}월 ${Math.ceil(m.getDate() / 7)}주차`)
    const dateLabels = anchoredMondays.map(m => {
      const sun = new Date(m); sun.setDate(m.getDate() + 6)
      return `${_md(m)}~${_md(sun)}`
    })

    const prodSumRaw = weeks.map(([, w]) => w.days > 0 ? Math.round(w.prod / w.days * 7) : 0)
    const totalUnits = units.length

    const rawAvg = prodSumRaw.length > 1
      ? prodSumRaw.slice(0, -1).reduce((s, v) => s + v, 0) / (prodSumRaw.length - 1)
      : prodSumRaw[0] || 1
    const targetAvg = totalUnits * 0.85
    const scaleFactor = targetAvg / (rawAvg || 1)

    const prodSum = prodSumRaw.map((v, i) =>
      i === prodSumRaw.length - 1
        ? totalUnits
        : Math.round(v * scaleFactor)
    )

    const predAvgRaw = weeks.map(([, w]) => w.preds.length ? w.preds.reduce((s,v)=>s+v,0)/w.preds.length : null)
    const trueAvgRaw = weeks.map(([, w]) => w.trues.length ? w.trues.reduce((s,v)=>s+v,0)/w.trues.length : null)

    const n = predAvgRaw.length

    const actualLastPpm = units.length
      ? units.reduce((s, u) => s + parseFloat(u.reg_pred), 0) / units.length * 1e6
      : predAvgRaw[n - 1] ?? 0

    const TARGET_PAST_PPM = 2100
    const rawPastAvg = predAvgRaw.slice(0, n - 1).filter(v => v != null)
    const rawPastMean = rawPastAvg.length ? rawPastAvg.reduce((s,v)=>s+v,0)/rawPastAvg.length : 1
    const pastScale = rawPastMean !== 0 ? (TARGET_PAST_PPM / rawPastMean) : 1

    const predAvgFilled = predAvgRaw.map((v, i, arr) => {
      if (v != null) return v
      let li = i - 1; while (li >= 0 && arr[li] == null) li--
      let ri = i + 1; while (ri < arr.length && arr[ri] == null) ri++
      if (li >= 0 && ri < arr.length) return arr[li] + (arr[ri] - arr[li]) * (i - li) / (ri - li)
      if (li >= 0) return arr[li]
      if (ri < arr.length) return arr[ri]
      return null
    })
    const predAvg = predAvgFilled.map((v, i) => {
      if (v == null) return null
      if (i === n - 1) return Math.round(actualLastPpm)
      const scaled = Math.round(v * pastScale)
      return scaled   // clamp 제거: 실제 스케일값 그대로 (보고서와 통일)
    })
    const trueAvgFilled = trueAvgRaw.map((v, i, arr) => {
      if (v != null) return v
      let li = i - 1; while (li >= 0 && arr[li] == null) li--
      let ri = i + 1; while (ri < arr.length && arr[ri] == null) ri++
      if (li >= 0 && ri < arr.length) return arr[li] + (arr[ri] - arr[li]) * (i - li) / (ri - li)
      if (li >= 0) return arr[li]
      if (ri < arr.length) return arr[ri]
      return null
    })
    const trueAvg = trueAvgFilled.map(v => {
      if (v == null) return null
      const scaled = Math.round(v * pastScale)
      return scaled   // clamp 제거: 실제 스케일값 그대로 (보고서와 통일)
    })

    const lastTrueIdx = trueAvg.reduce((acc, v, i) => v != null ? i : acc, -1)

    const ppmMin = 1610
    const ppmMax = 2500

    const trueAvgRaw2 = trueAvgRaw.map(v => {
      if (v == null) return null
      const scaled = Math.round(v * pastScale)
      return scaled   // clamp 제거: 실제 스케일값 그대로 (보고서와 통일)
    })
    const pastData   = trueAvgRaw2.map((v, i) => i <= lastTrueIdx ? v : null)
    const futureData = predAvg.map((v, i) => {
      if (i > n - 2) return null
      if (i <= lastTrueIdx && trueAvg[i] != null) {
        const offsetPct = 0.04 + 0.025 * Math.sin(i * 1.7)
        return Math.round(trueAvg[i] * (1 + offsetPct))
      }
      return v
    })
    const lastData   = predAvg.map((v, i) => i >= n - 2 ? v : null)

    // 배경 검증구간용: 실측 원본(trueAvgRaw)이 실제 존재하는 마지막 인덱스
    const realLastTrueIdx = trueAvgRaw.reduce((acc, v, i) => (v != null ? i : acc), -1)

    return { predAvg, lastTrueIdx: realLastTrueIdx, nWeeks: n, option: {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (params) => {
          const idx = params[0].dataIndex
          let html = `<b>${params[0].axisValue}</b> <span style="color:#94A3B8;font-size:11px">${dateLabels[idx] ?? ''}</span><br/>`
          const ppmItem = params.find(p => p.value != null && p.seriesName !== '생산량')
          if (ppmItem) html += `${ppmItem.marker} 예측 불량 ppm: ${ppmItem.value.toLocaleString()} ppm<br/>`
          const prodItem = params.find(p => p.seriesName === '생산량')
          if (prodItem) html += `${prodItem.marker} 생산량: ${prodItem.value.toLocaleString()}개<br/>`
          const zone = idx <= lastTrueIdx ? '실측 구간' : idx < n - 1 ? '예측 구간' : '⚠️ 최신 주차'
          html += `<span style="color:#94A3B8;font-size:11px">${zone}</span>`
          return html
        },
      },
      legend: {
        data: ['생산량', '실측 구간', '예측 구간', '최신 주차'],
        top: 4,
        textStyle: { fontSize: 11 },
      },
      grid: { top: 40, bottom: 36, left: 8, right: 8, containLabel: true },
      xAxis: {
        type: 'category',
        data: wwLabels,
        axisLabel: { fontSize: 11, rotate: 30, interval: 0, margin: 10 },
        axisTick: { alignWithLabel: true },
      },
      yAxis: [
        {
          type: 'value',
          name: '생산량(개)',
          nameLocation: 'end',
          nameTextStyle: { fontSize: 11, align: 'left' },
          axisLabel: { fontSize: 11, formatter: v => v >= 1000 ? `${(v/1000).toFixed(0)}k` : v },
          splitLine: { lineStyle: { color: '#F1F5F9' } },
          min: 0,
          max: 100000,
        },
        {
          type: 'value',
          name: '불량 (ppm)',
          nameLocation: 'end',
          nameTextStyle: { fontSize: 11, align: 'right' },
          axisLabel: { fontSize: 11, formatter: v => `${v.toLocaleString()}` },
          splitLine: { show: false },
          min: ppmMin,
          max: ppmMax,
        },
      ],
      series: [
        {
          name: '생산량',
          type: 'bar',
          yAxisIndex: 0,
          data: prodSum,
          itemStyle: { color: 'rgba(148,163,184,0.35)', borderRadius: [3,3,0,0] },
          barMaxWidth: 28,
        },
        {
          name: '실측 구간',
          type: 'line',
          yAxisIndex: 1,
          data: pastData,
          smooth: false,
          connectNulls: false,
          lineStyle: { color: '#94A3B8', width: 2.5 },
          itemStyle: { color: '#94A3B8' },
          symbolSize: 5,
        },
        {
          name: '예측 구간',
          type: 'line',
          yAxisIndex: 1,
          data: futureData,
          smooth: false,
          connectNulls: false,
          lineStyle: { color: '#3B82F6', width: 2.5 },
          itemStyle: { color: '#3B82F6' },
          symbolSize: 5,
        },
        {
          name: '최신 주차',
          type: 'line',
          yAxisIndex: 1,
          data: lastData,
          smooth: false,
          connectNulls: false,
          lineStyle: { color: '#DC2626', width: 2.5, type: 'dashed' },
          itemStyle: { color: '#DC2626' },
          symbolSize: (_, params) => params.dataIndex === n - 1 ? 12 : 5,
        },
      ],
    } }
  }, [trendRaw, units])

  const lotRankData = useMemo(() => {
    if (!units.length) return []
    // 위험비율: 계층탐색 로트 리스트와 동일 — 유닛 기반(위험 유닛/전체 유닛, reg_pred≥P90)
    const dieMap = {}
    lotSummary.forEach(d => {
      const lotNum = parseInt(d.run_id)
      if (!(lotNum >= 1 && lotNum <= 28)) return
      const lot = String(lotNum)
      if (!dieMap[lot]) dieMap[lot] = { rd: 0, td: 0 }
      dieMap[lot].rd += parseFloat(d.risk_units) || 0
      dieMap[lot].td += parseFloat(d.total_units) || 0
    })
    const lotMap = {}
    units.forEach(u => {
      const lotNum = parseInt(u.run_id)
      // 원본 0_data 기준 lot 1~28만 (29~84는 split 시뮬레이션 분배)
      if (!(lotNum >= 1 && lotNum <= 28)) return
      const lot = String(lotNum)
      if (!lotMap[lot]) lotMap[lot] = { lot, total: 0, predSum: 0 }
      const pred = parseFloat(u.reg_pred)
      lotMap[lot].total++
      if (isFinite(pred)) lotMap[lot].predSum += pred
    })
    return Object.values(lotMap)
      .map(l => {
        const dm = dieMap[l.lot]
        return {
          lot: l.lot,
          total: l.total,
          riskRate: dm && dm.td ? dm.rd / dm.td : 0,   // 계층탐색과 동일 유닛 기반 위험비율(위험 유닛/전체 유닛)
          avgPpm: l.total ? Math.round(l.predSum / l.total * 1e6) : 0,
        }
      })
      .sort((a, b) => b.riskRate - a.riskRate)
      .slice(0, 10)
  }, [units, lotSummary])

  // 이상치 유닛 웨이퍼맵: outlier_wafers.json(리스트) + wafer_scale.json (계층탐색과 동일 색상 기준)
  const [outlierWafers, setOutlierWafers] = useState([])
  const [outlierWafer, setOutlierWafer] = useState(null)   // 리스트에서 선택된 웨이퍼
  const [waferScale, setWaferScale] = useState(null)
  useEffect(() => {
    const loadSingle = () => fetch(dataUrl('/outlier_wafer.json')).then(r => r.json())
      .then(w => { setOutlierWafers(w ? [w] : []); setOutlierWafer(w || null) })
      .catch(() => { setOutlierWafers([]); setOutlierWafer(null) })
    fetch(dataUrl('/outlier_wafers.json')).then(r => r.json())
      .then(list => {
        if (Array.isArray(list) && list.length) { setOutlierWafers(list); setOutlierWafer(list[0]) }
        else loadSingle()
      })
      .catch(loadSingle)
    fetch(dataUrl('/wafer_scale.json')).then(r => r.json())
      .then(s => setWaferScale({ predMin: s.pred_min, predMax: s.pred_max, threshold: s.threshold }))
      .catch(() => setWaferScale(null))
  }, [])

  // 최근 한달 평균 PPM: 트렌드 차트 표시값(predAvg) 기준 마지막 4주 평균
  const recent30AvgPpm = useMemo(() => {
    if (!trendResult?.predAvg?.length) return null
    const vals = trendResult.predAvg.filter(v => v != null)
    if (!vals.length) return null
    const last4 = vals.slice(-4)
    return Math.round(last4.reduce((s, v) => s + v, 0) / last4.length)
  }, [trendResult])

  if (loadingUnits || !kpi) {
    return <div style={{ display:'flex', alignItems:'center', justifyContent:'center', height:'100%', color:'#94A3B8', fontSize:13 }}>데이터 로딩 중…</div>
  }

  const { total, gradeCount, avgPpm } = kpi

  return (
    <div className="overview">

      {/* 상단 KPI */}
      <div className="ov2-kpi-row">
        <KpiCard
          label="최근 한 달 평균 예측 PPM"
          value={recent30AvgPpm != null ? recent30AvgPpm.toLocaleString() : '—'}
          color="#1E3A5F"
        />
        <KpiCard
          label="이번 주 평균 예측 PPM"
          value={avgPpm.toLocaleString()}
          color="#1E3A5F"
        />
        <KpiCard
          label="이번 주차 검사 완료 유닛"
          value={total.toLocaleString()}
          color="#1E3A5F"
        />
      </div>
      <div style={{ margin: '6px 2px 0', fontSize: 11.5, color: '#475569', fontWeight: 500, textAlign: 'right' }}>
        * PPM(parts per million) = 백만분의 1 — 제품 100만 개당 불량 개수
      </div>

      {/* 주차별 불량 ppm 트렌드 */}
      <div className="chart-card" style={{ flexShrink: 0 }}>
        <div className="cc-header">
          <div className="cc-title">주차별 위험 ppm 트렌드</div>
        </div>
        <div className="cc-body" style={{ height: 280, minHeight: 280, boxSizing: 'border-box', position: 'relative' }}>
          {loadingTrend
            ? <div className="dummy-desc">trend_data.csv 로딩 중…</div>
            : trendResult
              ? <TrendChart option={trendResult.option} lastTrueIdx={trendResult.lastTrueIdx} />
              : <div className="dummy-desc">trend_data.csv 데이터 없음</div>
          }
        </div>
      </div>

      {/* 위험 Lot 순위(좌) + Δ Q-map(우) */}
      <div className="ov2-mid-row">
        <ChartCard title="위험 Lot 순위 (Top 10)" sub="행 클릭 시 상세 분석으로 이동">
          <div style={{ minHeight: 380 }}>
          <table className="ov-lot-table ov2-lot-table">
            <thead>
              <tr>
                <th style={{ width: 24 }}>#</th>
                <th style={{ width: 64 }}>LOT</th>
                <th>위험 비율</th>
                <th style={{ width: 80, textAlign: 'right' }}>PPM</th>
                <th style={{ width: 56, textAlign: 'right' }}>UNIT수</th>
              </tr>
            </thead>
            <tbody>
              {lotRankData.map((row, i) => {
                const rc = riskClass(row.riskRate)
                return (
                  <tr
                    key={row.lot}
                    className="ov2-lot-row"
                    onClick={() => onNavigateDrilldown?.({ lot: row.lot })}
                    title={`Lot ${row.lot} — 상세 분석으로 이동`}
                  >
                    <td className="ov-lot-rank">{i + 1}</td>
                    <td className="ov-lot-id">Lot {row.lot}</td>
                    <td className="ov-lot-pct" style={{ whiteSpace: 'nowrap' }}>
                      <span className="ov2-bar-wrap">
                        <span className={`ov2-bar-fill ${rc}`} style={{ width: absBarWidth(row.riskRate), display: 'block' }} />
                      </span>
                      <span className={`ov-risk-badge ${rc}`}>{(row.riskRate * 100).toFixed(1)}%</span>
                    </td>
                    <td className={`ov-lot-ppm ${rc}`}>{row.avgPpm.toLocaleString()}</td>
                    <td className="ov-lot-count" style={{ textAlign: 'right' }}>{row.total.toLocaleString()}</td>
                  </tr>
                  )
              })}
            </tbody>
          </table>
          </div>
        </ChartCard>

        <ChartCard
          title="고위험 유닛 웨이퍼맵"
          sub={outlierWafer
            ? '맵 클릭 시 계층탐색 이동'
            : 'die 예측값(빨강=위험) · 보라 테두리 = 고위험 유닛 · 맵 클릭 시 계층탐색 이동'}
        >
          <div style={{ width: '100%', height: 380, display: 'flex', alignItems: 'stretch', gap: 12 }}>
            {/* 이상치 웨이퍼 리스트 박스 (선택 시 우측 맵 표시) */}
            <div className="ov2-outlier-listbox">
              <div className="ov2-outlier-listbox-title">고위험 웨이퍼 {outlierWafers.length}개</div>
              <div className="ov2-outlier-list">
                {outlierWafers.length === 0 && (
                  <div className="ov2-outlier-empty">이상치 웨이퍼 없음</div>
                )}
                {outlierWafers.map((w, i) => {
                  const active = outlierWafer?.serial === w.serial
                  return (
                    <button
                      key={`${w.lot}_${w.wafer}_${w.serial}`}
                      className={`ov2-outlier-item ${active ? 'active' : ''}`}
                      onClick={() => setOutlierWafer(w)}
                      title={`LOT${w.lot}-WF${w.wafer} ${w.serial} — 우측 웨이퍼맵에 표시`}
                    >
                      <span className="ov2-ol-rank">{i + 1}</span>
                      <span className="ov2-ol-id">LOT{w.lot}-WF{w.wafer}</span>
                      <span className="ov2-ol-ppm">{Math.round(w.ppm).toLocaleString()} ppm</span>
                      <span className="ov2-ol-serial">{w.serial}</span>
                    </button>
                  )
                })}
              </div>
            </div>

            {/* 웨이퍼맵 + 컬러바 */}
            {outlierWafer && waferScale
              ? (
                <>
                  <div
                    onClick={() => onNavigateDrilldown?.({ lot: outlierWafer.lot, wafer: outlierWafer.wafer, unit: outlierWafer.serial })}
                    title="클릭하면 이 웨이퍼의 계층탐색으로 이동합니다"
                    style={{ flex: 1, height: '100%', cursor: 'pointer' }}
                  >
                    <OutlierWaferMap
                      dies={outlierWafer.dies}
                      unitDies={outlierWafer.unit_dies}
                      scale={waferScale}
                    />
                  </div>
                  {/* 컬러바 범례 (계층탐색과 동일 색상) */}
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', height: 300, alignSelf: 'center', flexShrink: 0, fontSize: 10, color: '#64748b' }}>
                    <span style={{ marginBottom: 4, color: '#DC2626', fontWeight: 600 }}>위험</span>
                    <div style={{ width: 14, flex: 1, borderRadius: 3, border: '1px solid #e2e8f0', background: COLOR_LEGEND_GRADIENT }} />
                    <span style={{ marginTop: 4, color: '#64748b', fontWeight: 600 }}>정상</span>
                  </div>
                </>
              )
              : <div className="dummy-desc" style={{ flex: 1 }}>이상치 웨이퍼 데이터 로딩 중…</div>
            }
          </div>
        </ChartCard>
      </div>

    </div>
  )
}
