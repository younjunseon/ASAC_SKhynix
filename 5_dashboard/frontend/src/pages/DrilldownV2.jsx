/**
 * 계층별 정밀 분석 — 1팀 Drilldown 구조 포팅 (CSV 기반)
 *
 * 좌측  : Lot 트리 (lot → wafer 리스트, 위험률/번호 정렬)
 * 중앙  : WaferMap SVG (단일 wafer die 히트맵 or lot 누적 max)
 * 우측  : Unit 진단 (verdict + pred ppm + grade)
 *
 * 데이터:
 *   - wafer_map.csv  : ufs_serial, run_id, wafer_no, die_x, die_y, pred, health, clf_proba, split, position
 *   - dashboard_units.csv : ufs_serial, run_id, wafer_no, split, health, reg_pred, risk
 */
import { useState, useMemo, useEffect, useRef } from 'react'
import { useCSV } from '../hooks/useCSV'
import { dataUrl } from '../utils/dataUrl'
import ALL_DIE_POSITIONS from './diePositions.js'
import './DrilldownV2.css'

// ── 색상 로직 (1팀 colors.ts 포팅) ───────────────────
const NORMAL_STOPS = [
  [0.0,  [248, 250, 252]],   // #f8fafc 거의 흰색
  [0.5,  [186, 230, 253]],   // #bae6fd 연한 하늘
  [1.0,  [125, 211, 252]],   // #7dd3fc 하늘색
]
const RISK_STOPS = [
  [0.0,  [254, 249, 195]],   // #fef9c3 아주 연한 노랑
  [0.4,  [253, 224,  71]],   // #fde047 노랑
  [0.7,  [250, 173,  20]],   // #faad14 진한 노랑
  [1.0,  [249, 115,  22]],   // #f97316 주황
]

// 드릴다운 웨이퍼맵 색칠 기준 토글:
//  'unit' = 유닛 reg_pred 기준(한 유닛의 4 die를 같은 색으로 통일) — 현재
//  'die'  = die별 기여 ppm 기준(과거 버전, 백업) ← 복원하려면 이 값으로 변경
const WMAP_COLOR_BY = 'unit'

function interp(stops, t) {
  const tt = Math.max(0, Math.min(1, t))
  for (let i = 1; i < stops.length; i++) {
    const [t1, c1] = stops[i]
    const [t0, c0] = stops[i - 1]
    if (tt <= t1) {
      const k = (tt - t0) / (t1 - t0 || 1)
      const r = Math.round(c0[0] + (c1[0] - c0[0]) * k)
      const g = Math.round(c0[1] + (c1[1] - c0[1]) * k)
      const b = Math.round(c0[2] + (c1[2] - c0[2]) * k)
      return `rgb(${r},${g},${b})`
    }
  }
  const last = stops[stops.length - 1][1]
  return `rgb(${last.join(',')})`
}

function predColor(pred, predMin, predMax, threshold) {
  if (!isFinite(pred)) return '#f1f5f9'
  if (pred <= threshold) {
    const span = Math.max(1e-9, threshold - predMin)
    return interp(NORMAL_STOPS, (pred - predMin) / span)
  } else {
    const span = Math.max(1e-9, predMax - threshold)
    // 제곱근 스케일: 낮은 위험값도 색상 퍼짐
    return interp(RISK_STOPS, Math.sqrt((pred - threshold) / span))
  }
}

const COLOR_LEGEND_GRADIENT =
  'linear-gradient(to right, #f8fafc, #bae6fd, #7dd3fc, #fef9c3, #fde047, #faad14, #f97316)'

// 전체 데이터(oof+val+test)의 die 좌표 글로벌 범위 — 웨이퍼맵 격자 고정용
const GLOBAL_DIE_X_MIN = 12, GLOBAL_DIE_X_MAX = 66
const GLOBAL_DIE_Y_MIN = 11, GLOBAL_DIE_Y_MAX = 32

// ── 스케일 계산 ───────────────────────────────────────
function computeScale(allDies) {
  const preds = allDies.map(d => parseFloat(d.pred)).filter(isFinite)
  const gx = { xRange: GLOBAL_DIE_X_MAX - GLOBAL_DIE_X_MIN + 1, yRange: GLOBAL_DIE_Y_MAX - GLOBAL_DIE_Y_MIN + 1, xMin: GLOBAL_DIE_X_MIN, xMax: GLOBAL_DIE_X_MAX, yMin: GLOBAL_DIE_Y_MIN, yMax: GLOBAL_DIE_Y_MAX }
  if (!preds.length) return { predMin: 0, predMax: 0.02, threshold: 0.005, gridXRange: gx.xRange, gridYRange: gx.yRange, gridXMin: gx.xMin, gridXMax: gx.xMax, gridYMin: gx.yMin, gridYMax: gx.yMax }
  preds.sort((a, b) => a - b)
  const predMin = preds[0]
  const predMax = preds[preds.length - 1]
  const q2 = preds[Math.floor(preds.length * 0.50)] ?? predMin
  const threshold = q2
  return { predMin, predMax, threshold, gridXRange: gx.xRange, gridYRange: gx.yRange, gridXMin: gx.xMin, gridXMax: gx.xMax, gridYMin: gx.yMin, gridYMax: gx.yMax }
}

// ── WaferMap SVG 컴포넌트 (1팀 WaferMap.tsx 포팅) ────
function WaferMap({ dies, scale, selectedUnit, onSelectUnit, selectedDie, onSelectDie, mini = false, unitColorMap, unitColorScale, colorScale }) {
  const layout = useMemo(() => {
    if (!dies.length) return null
    const xs = dies.map(d => d.die_x)
    const ys = dies.map(d => d.die_y)
    const xMin = Math.min(...xs), xMax = Math.max(...xs)
    const yMin = Math.min(...ys), yMax = Math.max(...ys)
    return { xMin, xMax, yMin, yMax, xRange: xMax - xMin + 1, yRange: yMax - yMin + 1 }
  }, [dies])

  if (!layout || !dies.length) {
    return <div className="dd-wmap-empty">표시할 die가 없습니다.</div>
  }

  const dieMap = new Map()
  for (const d of dies) dieMap.set(`${d.die_x},${d.die_y}`, d)

  // ── viewBox 좌표계 ──
  const D      = 800   // 원 지름
  const PAD    = 12    // 상/좌/하/우 동일 여백 (눈금 라벨 제거로 최소화)
  const VB_W   = D + PAD * 2
  const VB_H   = D + PAD * 2
  const cx     = PAD + D / 2
  const cy     = PAD + D / 2
  const radius = D / 2

  const { xMin, xMax, yMin, yMax, xRange, yRange } = layout

  // 전체 데이터 기준 중심/범위 사용 → 웨이퍼마다 포지션이 달라도 격자가 고정됨
  const refXMin = scale.gridXMin ?? xMin
  const refXMax = scale.gridXMax ?? xMax
  const refYMin = scale.gridYMin ?? yMin
  const refYMax = scale.gridYMax ?? yMax
  const refXRange = scale.gridXRange ?? xRange
  const refYRange = scale.gridYRange ?? yRange
  const centerX = (refXMin + refXMax) / 2
  const centerY = (refYMin + refYMax) / 2

  const SCALE = 0.9
  const cellW = (D / refXRange) * SCALE
  const cellH = (D / refYRange) * SCALE

  // mask: 전체 데이터 기준 모든 die 좌표 (웨이퍼마다 격자 일정)
  const mask = ALL_DIE_POSITIONS

  // 격자선: 전체 좌표 범위 기준 (결측 포지션 있어도 격자 일정)
  const gridXs = []
  for (let xi = refXMin; xi <= refXMax + 1; xi++) {
    gridXs.push(cx + (xi - centerX) * cellW - cellW / 2)
  }
  const gridYs = []
  for (let yi = refYMin; yi <= refYMax + 1; yi++) {
    gridYs.push(cy + (yi - centerY) * cellH - cellH / 2)
  }

  return (
    <div className="dd-wmap-inner">
      <div className="dd-wmap-svg-wrap">
        <svg viewBox={`0 0 ${VB_W} ${VB_H}`} preserveAspectRatio="xMidYMid meet" style={{ overflow: 'visible' }}
          className="dd-wmap-svg">
          <defs>
            <clipPath id="waferCircle">
              <circle cx={cx} cy={cy} r={radius} />
            </clipPath>
          </defs>
          <circle cx={cx} cy={cy} r={radius} fill="#fafafa" stroke="#cbd5e1" strokeWidth={1.5} />
          <g clipPath="url(#waferCircle)">
            {/* die 색상 셀 */}
            {mask.map(([dx, dy]) => {
              const die = dieMap.get(`${dx},${dy}`)
              if (!die) return null
              const x = cx + (dx - centerX) * cellW - cellW / 2
              const y = cy + (dy - centerY) * cellH - cellH / 2
              const isUnitSel = die.ufs_serial === selectedUnit
              const isDieSel  = selectedDie && String(die.die_x) === String(selectedDie.die_x) && String(die.die_y) === String(selectedDie.die_y)
              // 색 기준 우선순위:
              //  ① unitColorMap: die의 소속 유닛 reg_pred (드릴다운, 4 die 통일)
              //  ② colorScale: die.pred 가 이미 유닛 reg_pred 스케일 (패턴맵 — 좌표별 worst 유닛)
              //  ③ die 기여 ppm 기준 (백업)
              const _urp = (unitColorMap && unitColorScale && die.ufs_serial != null)
                ? unitColorMap.get(String(die.ufs_serial)) : undefined
              let fill, _tipUnit = ''
              if (_urp != null) {
                fill = predColor(_urp, unitColorScale.predMin, unitColorScale.predMax, unitColorScale.threshold)
                _tipUnit = `\n유닛=${Math.round(_urp * 1e6)} ppm`
              } else if (colorScale) {
                fill = predColor(parseFloat(die.pred), colorScale.predMin, colorScale.predMax, colorScale.threshold)
              } else {
                fill = predColor(parseFloat(die.pred), scale.predMin, scale.predMax, scale.threshold)
              }
              return (
                <g key={`${dx}-${dy}`}>
                  <title>{`${die.ufs_serial ?? ''}
(${dx}, ${dy})
기여=${Math.round(parseFloat(die.pred) * 1e6)} ppm${_tipUnit}`}</title>
                  <rect
                    x={x} y={y}
                    width={cellW}
                    height={cellH}
                    fill={fill}
                    stroke={isDieSel ? '#7C3AED' : isUnitSel ? '#0f172a' : mini ? 'none' : 'rgba(15,23,42,0.12)'}
                    strokeWidth={isDieSel ? 4 : isUnitSel ? 3 : mini ? 0 : 0.6}
                    style={onSelectUnit ? { cursor: 'pointer' } : undefined}
                    onClick={() => {
                      if (!die.ufs_serial || !onSelectUnit) return
                      onSelectUnit(die.ufs_serial)
                      onSelectDie?.(die)
                    }}
                  />
                </g>
              )
            })}
            {/* 격자선 (die 위에 오버레이) */}
            {gridXs.map((gx, i) => (
              <line key={`gx-${i}`} x1={gx} y1={cy - radius} x2={gx} y2={cy + radius}
                stroke="rgba(100,116,139,0.18)" strokeWidth={mini ? 0.2 : 0.8} />
            ))}
            {gridYs.map((gy, i) => (
              <line key={`gy-${i}`} x1={cx - radius} y1={gy} x2={cx + radius} y2={gy}
                stroke="rgba(100,116,139,0.18)" strokeWidth={mini ? 0.2 : 0.8} />
            ))}
          </g>
          <circle cx={cx} cy={cy} r={radius} fill="none" stroke="#94a3b8" strokeWidth={1.5} />
          {/* notch */}
          <rect x={cx - 18} y={cy + radius - 6} width={36} height={8}
            fill="#fff" stroke="#94a3b8" strokeWidth={1} />
        </svg>
      </div>

    </div>  /* dd-wmap-inner */
  )
}

// ── SHAP 바 (shap_unit.json 기반 unit별 실제 SHAP, fallback: shap_bar.csv 전체 평균) ─
function ShapBar({ shapData, shapUnitMap, ufsSerial, selectedFeature, onSelectFeature }) {
  const unitShap = useMemo(() => {
    if (!shapUnitMap || !ufsSerial) return null
    const rows = shapUnitMap[ufsSerial]
    if (!rows?.length) return null
    const mapped = rows.map(r => {
      const v = Number(r.shap_value)
      return { feature: String(r.feature), val: isNaN(v) ? 0 : v, magnitude: isNaN(v) ? 0 : Math.abs(v) }
    }).filter(r => r.magnitude > 0).sort((a, b) => b.magnitude - a.magnitude).slice(0, 10)
    return mapped.length ? mapped : null
  }, [shapUnitMap, ufsSerial])

  if (!shapUnitMap && ufsSerial) {
    return <div style={{ fontSize: 11, color: '#94a3b8', padding: '8px 0' }}>SHAP 데이터 로딩 중…</div>
  }

  const allBars = unitShap ?? (
    shapData?.length
      ? shapData.slice(0, 20).map(f => {
          const v = Number(f.mean_shap ?? f.effect_norm)
          const m = Number(f.mean_abs_shap ?? f.lgbm_gain)
          return { feature: String(f.feature), val: isNaN(v) ? 0 : v, magnitude: isNaN(m) ? 0 : Math.abs(m) }
        }).filter(r => r.magnitude > 0).sort((a, b) => b.magnitude - a.magnitude)
      : []
  )
  const bars = allBars.filter(b => /^X\d+$/.test(b.feature)).slice(0, 10)

  if (!bars.length) return null
  const maxMag = Math.max(...bars.map(b => b.magnitude), 1e-9)
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 3, marginTop: 6 }}>
      {bars.map(b => {
        const w = Math.round(b.magnitude / maxMag * 100)
        const clr = b.val >= 0 ? '#ef4444' : '#3b82f6'
        const isSelected = selectedFeature === b.feature
        return (
          <div
            key={b.feature}
            onClick={() => onSelectFeature?.(isSelected ? null : b.feature)}
            style={{
              display: 'flex', alignItems: 'center', gap: 6, fontSize: 11,
              cursor: onSelectFeature ? 'pointer' : undefined,
              background: isSelected ? '#eff6ff' : 'transparent',
              borderRadius: 4, padding: '2px 2px',
              boxShadow: isSelected ? 'inset 0 0 0 1.5px #3b82f6' : 'none',
            }}
          >
            <span style={{ width: 60, textAlign: 'right', fontFamily: 'monospace', color: isSelected ? '#1d4ed8' : '#374151', flexShrink: 0, fontWeight: isSelected ? 700 : 400 }}>{b.feature}</span>
            <div style={{ flex: 1, background: '#f1f5f9', height: 10, borderRadius: 2, overflow: 'hidden' }}>
              <div style={{ width: `${w}%`, height: '100%', background: clr, borderRadius: 2 }} />
            </div>
            <span style={{ width: 52, fontSize: 11, color: clr, fontWeight: 700, textAlign: 'right' }}>{b.val >= 0 ? '+' : ''}{Math.round(b.val * 1e6).toLocaleString()}</span>
          </div>
        )
      })}
      {onSelectFeature && <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 2 }}>막대 클릭 → 웨이퍼 히트맵</div>}
    </div>
  )
}

// ── 피처 웨이퍼 히트맵 (SVG 직접 렌더링) ────────────────
function FeatureWaferMap({ feature, featNormData }) {
  const dies = useMemo(() => {
    if (!featNormData?.length || !feature) return []
    return featNormData.filter(r => r.feature === feature).map(r => ({
      die_x: parseInt(r.die_x),
      die_y: parseInt(r.die_y),
      val: parseFloat(r.feat_norm),
    }))
  }, [featNormData, feature])

  if (!feature) return null
  if (!dies.length) return (
    <div style={{ fontSize: 11, color: '#94a3b8', textAlign: 'center', padding: '8px 0' }}>
      {feature} 데이터 없음
    </div>
  )

  const D = 800, PAD = 12
  const VB = D + PAD * 2
  const cx = PAD + D / 2, cy = PAD + D / 2, radius = D / 2

  const refXMin = GLOBAL_DIE_X_MIN, refXMax = GLOBAL_DIE_X_MAX
  const refYMin = GLOBAL_DIE_Y_MIN, refYMax = GLOBAL_DIE_Y_MAX
  const centerX = (refXMin + refXMax) / 2
  const centerY = (refYMin + refYMax) / 2
  const SCALE = 0.9
  const cellW = (D / (refXMax - refXMin + 1)) * SCALE
  const cellH = (D / (refYMax - refYMin + 1)) * SCALE

  const dieMap = new Map(dies.map(d => [`${d.die_x},${d.die_y}`, d.val]))

  function featColor(v) {
    if (!isFinite(v)) return '#f1f5f9'
    // 0.5 기준 diverging: 파랑(낮음) → 흰색(중간) → 빨강(높음)
    if (v <= 0.5) {
      const t = v / 0.5  // 0~1
      const r = Math.round(59  + t * (255 - 59))
      const g = Math.round(130 + t * (255 - 130))
      const b = Math.round(246 + t * (255 - 246))
      return `rgb(${r},${g},${b})`
    } else {
      const t = (v - 0.5) / 0.5  // 0~1
      const r = Math.round(255)
      const g = Math.round(255 - t * (255 - 59))
      const b = Math.round(255 - t * (255 - 59))
      return `rgb(${r},${g},${b})`
    }
  }

  return (
    <div style={{ marginTop: 10 }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: '#1e40af', marginBottom: 4 }}>
        {feature} 웨이퍼 히트맵
      </div>
      <svg viewBox={`0 0 ${VB} ${VB}`} preserveAspectRatio="xMidYMid meet"
        style={{ width: '100%', display: 'block' }}>
        <defs>
          <clipPath id={`fc-${feature}`}>
            <circle cx={cx} cy={cy} r={radius} />
          </clipPath>
        </defs>
        <circle cx={cx} cy={cy} r={radius} fill="#fafafa" stroke="#cbd5e1" strokeWidth={1.5} />
        <g clipPath={`url(#fc-${feature})`}>
          {ALL_DIE_POSITIONS.map(([dx, dy]) => {
            const v = dieMap.get(`${dx},${dy}`)
            if (v === undefined) return null
            const x = cx + (dx - centerX) * cellW - cellW / 2
            const y = cy + (dy - centerY) * cellH - cellH / 2
            return (
              <rect key={`${dx}-${dy}`} x={x} y={y} width={cellW} height={cellH}
                fill={featColor(v)} stroke="rgba(15,23,42,0.08)" strokeWidth={0.5} />
            )
          })}
        </g>
        <circle cx={cx} cy={cy} r={radius} fill="none" stroke="#94a3b8" strokeWidth={1.5} />
        <rect x={cx - 18} y={cy + radius - 6} width={36} height={8} fill="#fff" stroke="#94a3b8" strokeWidth={1} />
      </svg>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#64748b', marginTop: 2, padding: '0 2px' }}>
        <span style={{ color: '#3b82f6' }}>낮음</span>
        <div style={{ flex: 1, height: 6, margin: '0 6px', borderRadius: 3, background: 'linear-gradient(to right, rgb(59,130,246), #fff, rgb(255,59,59))', alignSelf: 'center' }} />
        <span style={{ color: '#ef4444' }}>높음</span>
      </div>
    </div>
  )
}

// ── Unit 진단 패널 (1팀 우측 패널 포팅) ──────────────
function UnitReport({ ufsSerial, allDies, scale, onClose, shapData, shapUnitMap, unitData, featNormData }) {
  const [selectedFeature, setSelectedFeature] = useState(null)
  const dies = useMemo(() =>
    ufsSerial ? allDies.filter(d => d.ufs_serial === ufsSerial) : [],
    [ufsSerial, allDies]
  )

  // dashboard_units.csv에서 anomaly_score 조회
  const { anomalyScore } = useMemo(() => {
    if (!ufsSerial || !unitData?.length) return { anomalyScore: null }
    const row = unitData.find(u => u.ufs_serial === ufsSerial)
    if (!row) return { anomalyScore: null }
    const v = parseFloat(row.anomaly_score)
    return { anomalyScore: isFinite(v) ? v : null }
  }, [ufsSerial, unitData])

  if (!ufsSerial) return (
    <div className="dd-report-empty">
      wafer map의 die를 클릭하면 진단이 표시됩니다.
    </div>
  )

  if (!dies.length) return (
    <div className="dd-report-empty">데이터 없음</div>
  )

  // unit 예측값 = CSV reg_pred (보정된 유닛값, 보고서와 동일). 없으면 die 평균 fallback
  const _urow = unitData?.find(u => u.ufs_serial === ufsSerial)
  const _dieVals = dies.map(d => parseFloat(d.pred)).filter(isFinite)
  const _dieAvg = _dieVals.length ? _dieVals.reduce((a, b) => a + b, 0) / _dieVals.length : 0
  const _regPred = _urow ? parseFloat(_urow.reg_pred) : NaN
  const pred = isFinite(_regPred) ? _regPred : _dieAvg
  const ppm = Math.round(pred * 1e6)
  // 위험 판정은 유닛 grade(grade3=위험, grade4=매우위험) 기준 — die 임계값과 스케일 다름
  const isRisk = _urow?.grade ? ['grade3', 'grade4'].includes(_urow.grade) : pred > scale.threshold
  const waferNo = dies[0].wafer_no ?? null
  const runId   = dies[0].run_id ?? null
  // 불량 확률: 4개 die clf_proba 평균
  const clfVals = dies.map(d => d.clf_proba !== undefined ? parseFloat(d.clf_proba) : null).filter(v => v !== null && isFinite(v))
  const clfProba = clfVals.length ? clfVals.reduce((a, b) => a + b, 0) / clfVals.length : null

  // 가장 위험한 die (같은 ufs_serial 내)
  const worstDie = dies.reduce((best, d) =>
    parseFloat(d.pred) > parseFloat(best.pred) ? d : best, dies[0])

  return (
    <div className="dd-report">
      {/* verdict 헤더 */}
      <div className={`dd-verdict ${isRisk ? 'risk' : 'normal'}`}>
        <div className="dd-verdict-row">
          <span className={`dd-verdict-badge ${isRisk ? 'risk' : 'normal'}`}>
            {isRisk ? '⚠ 위험' : '✓ 정상'}
          </span>
          {onClose && <button className="dd-report-close" onClick={onClose}>✕</button>}
        </div>
        <div className="dd-verdict-serial">
          <span className="dd-verdict-serial-main">{ufsSerial}</span>
          {runId && waferNo && (
            <span className="dd-verdict-serial-sub">Lot {runId} · Wafer #{waferNo}</span>
          )}
        </div>
        <div className="dd-verdict-pred">
          {ppm.toLocaleString()} ppm
        </div>
      </div>

      {/* 주요 기여 변수 (unit별 SHAP - shap_beeswarm.csv 기반) */}
      <div className="dd-section" style={{ padding: '6px 4px 4px' }}>
        <div className="dd-section-title">주요 기여 변수 Top 10 <span style={{ fontSize: 11, color: '#64748b', fontWeight: 600 }}>unit별 SHAP</span><span style={{ fontSize: 10, color: '#94a3b8', fontWeight: 400, marginLeft: 6 }}>(ppm)</span></div>
        <ShapBar shapData={shapData} shapUnitMap={shapUnitMap} ufsSerial={ufsSerial} selectedFeature={selectedFeature} onSelectFeature={setSelectedFeature} />
        {/* 피처 웨이퍼 히트맵 (SHAP 막대 클릭 시 SHAP 바 바로 아래 표시) */}
        {selectedFeature && (
          <div style={{ marginTop: 8, borderTop: '1px solid #e2e8f0', paddingTop: 8 }}>
            <FeatureWaferMap feature={selectedFeature} featNormData={featNormData} />
          </div>
        )}
      </div>

      {/* 가장 위험한 die */}
      {worstDie && (
        <div className="dd-section-box">
          <div className="dd-detail-key" style={{ marginBottom: 4 }}>가장 위험한 die</div>
          <div className="dd-worst-die mono">
            ({worstDie.die_x}, {worstDie.die_y}) ·{' '}
            <span className="danger">{Math.round(parseFloat(worstDie.pred) * 1e6).toLocaleString()} ppm</span>
          </div>
        </div>
      )}

      {/* 보고서 생성 버튼은 AI Agent 서버 비활성으로 인해 숨김 */}
    </div>
  )
}

// ── 보고서 생성 버튼 ─────────────────────────────────
const AI_AGENT_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function ReportButton({ ufsSerial, ppm, isRisk, worstDie, grade }) {
  const [status, setStatus] = useState('idle')  // idle | loading | done | error

  async function handleReport() {
    setStatus('loading')
    try {
      const report_data = {
        unit: {
          ufs_serial: ufsSerial,
          ppm,
          risk: isRisk ? 'HIGH' : 'LOW',
          grade: grade ?? '',
          worst_die: worstDie ? `(${worstDie.die_x}, ${worstDie.die_y}) · ${Math.round(parseFloat(worstDie.pred) * 1e6).toLocaleString()} ppm` : '',
        },
        generated_at: new Date().toLocaleString('ko-KR'),
      }

      const res = await fetch(`${AI_AGENT_URL}/report/pptx`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ report_data, filename: `보고서_${ufsSerial}.pptx` }),
      })

      if (!res.ok) throw new Error(`HTTP ${res.status}`)

      const blob = await res.blob()
      const url  = URL.createObjectURL(blob)
      const a    = document.createElement('a')
      a.href     = url
      a.download = `보고서_${ufsSerial}.pptx`
      a.click()
      URL.revokeObjectURL(url)
      setStatus('done')
      setTimeout(() => setStatus('idle'), 3000)
    } catch {
      setStatus('error')
      setTimeout(() => setStatus('idle'), 4000)
    }
  }

  const label = status === 'loading' ? '⏳ 생성 중…' : status === 'done' ? '✅ 완료' : status === 'error' ? '❌ 실패 (AI Agent 서버 확인)' : '📄 보고서 생성'
  const disabled = status === 'loading'

  return (
    <button
      className={`dd-report-btn${status === 'error' ? ' dd-report-btn-err' : ''}`}
      onClick={handleReport}
      disabled={disabled}
      title="AI Agent 서버(localhost:8000)에서 PPTX 보고서 생성"
    >
      {label}
    </button>
  )
}

// ── [백업/미사용] 구버전: die 임계 기준 게이지 + 포지션별 예측 ppm 바 ──
// die pred을 die 임계(τ)와 비교해 위험/정상으로 색칠하던 버전.
// die pred은 "유닛 예측에 대한 기여분"이라 die 단위 위험 판정이 성립하지 않아
// 기여도 기반 신버전(WaferBottomPanel)으로 교체함. 필요 시 복원용으로 보존.
function WaferBottomPanel_old({ ufsSerial, allDies, scale, selectedDie, onSelectDie }) {
  const dies = useMemo(() =>
    ufsSerial ? allDies.filter(d => d.ufs_serial === ufsSerial) : [],
    [ufsSerial, allDies]
  )

  const thPpm  = Math.round(scale.threshold * 1e6)
  const maxPpm = Math.round(scale.predMax * 1e6)
  const sorted = [...dies].sort((a, b) => parseInt(a.position || 0) - parseInt(b.position || 0))

  if (!ufsSerial) {
    return (
      <div className="dd-bottom-panel">
        <div className="dd-bottom-hint">웨이퍼맵에서 유닛을 클릭하면 임계값 및 포지션별 상세가 표시됩니다.</div>
      </div>
    )
  }

  return (
    <div className="dd-bottom-panel">
      <div className="dd-bottom-thresh">
        <div className="dd-bottom-thresh-label">
          <span className="dd-bottom-thresh-title">임계값 (τ)</span>
          <span className="dd-bottom-thresh-val">{thPpm.toLocaleString()} ppm</span>
        </div>
        <div className="dd-bottom-thresh-bar-wrap">
          <div style={{ display: 'flex', width: '100%', height: 12, borderRadius: 4, overflow: 'hidden', border: '1px solid #e2e8f0' }}>
            <div style={{ width: '50%', height: '100%', background: 'linear-gradient(to right, #f3f4f6, #dbeafe, #a5d7dc)' }} />
            <div style={{ width: '50%', height: '100%', background: 'linear-gradient(to right, #fef08a, #fef08a, #fb923c, #dc2626)' }} />
          </div>
          <div className="dd-bottom-thresh-marker" style={{ left: '50%' }}>
            <div className="dd-bottom-thresh-marker-line" />
            <div className="dd-bottom-thresh-marker-label">{thPpm.toLocaleString()}</div>
          </div>
          <div className="dd-bottom-thresh-ends">
            <span>0 ppm</span>
            <span style={{ position: 'absolute', left: '50%', transform: 'translateX(-50%)', color: '#92400E', fontSize: 10 }}>← 정상 | 위험 →</span>
            <span>{maxPpm.toLocaleString()} ppm</span>
          </div>
        </div>
      </div>
      {sorted.length > 0 && (
        <div className="dd-bottom-pos">
          <div className="dd-bottom-pos-title">
            포지션별 예측 ppm
            <span className="dd-bottom-pos-serial"> · {ufsSerial}</span>
          </div>
          <div className="dd-bottom-pos-grid">
            {sorted.map(die => {
              const pred   = parseFloat(die.pred)
              const ppm    = Math.round(pred * 1e6)
              const isRisk = pred > scale.threshold
              const pos    = die.position || '?'
              const isDieSel = selectedDie &&
                String(die.die_x) === String(selectedDie.die_x) &&
                String(die.die_y) === String(selectedDie.die_y)
              const fillColor = predColor(pred, scale.predMin, scale.predMax, scale.threshold)
              const thresh = scale.threshold
              const barW = isRisk
                ? 50 + Math.round(((pred - thresh) / Math.max(1e-9, scale.predMax - thresh)) * 50)
                : Math.round((pred / Math.max(1e-9, thresh)) * 50)
              return (
                <div
                  key={`${die.die_x}-${die.die_y}`}
                  className={`dd-bpos-row ${isDieSel ? 'selected' : ''} ${isRisk ? 'risk' : ''}`}
                  onClick={() => onSelectDie?.(die)}
                >
                  <div className="dd-bpos-header">
                    <span className="dd-bpos-num">P{pos}</span>
                    <span className="dd-bpos-chip" style={{ background: fillColor }} />
                    <span className={`dd-bpos-ppm ${isRisk ? 'danger' : ''}`}>
                      {ppm.toLocaleString()} ppm
                    </span>
                    <span className="dd-bpos-coord">({die.die_x},{die.die_y})</span>
                    {isDieSel && <span className="dd-bpos-sel-arrow">◀</span>}
                  </div>
                  <div className="dd-bpos-track">
                    <div className="dd-bpos-fill"
                      style={{ width: `${barW}%`, background: fillColor }} />
                    <div className="dd-bpos-thresh-line" style={{ left: '50%' }} />
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

// ── 웨이퍼 하단(신): 유닛 판정(reg_pred vs 유닛 임계 P90) + 포지션별 기여 ppm ──
// die는 "유닛 예측에 대한 기여분"이므로 die 단위 위험 판정 없이 기여 ppm만 표시.
// 위험/정상은 유닛값(reg_pred)을 유닛 임계(P90)와 비교해 한 줄로 표기.
function WaferBottomPanel({ ufsSerial, allDies, scale, selectedDie, onSelectDie, unitRow, unitThreshold, unitColorScale }) {
  const dies = useMemo(() =>
    ufsSerial ? allDies.filter(d => d.ufs_serial === ufsSerial) : [],
    [ufsSerial, allDies]
  )

  if (!ufsSerial) {
    return (
      <div className="dd-bottom-panel">
        <div className="dd-bottom-hint">웨이퍼맵에서 유닛을 클릭하면 유닛 판정 및 포지션별 기여가 표시됩니다.</div>
      </div>
    )
  }

  const sorted   = [...dies].sort((a, b) => parseInt(a.position || 0) - parseInt(b.position || 0))
  const dieSum   = dies.reduce((s, d) => s + parseFloat(d.pred), 0)
  // 유닛 예측값 = CSV reg_pred(보정 합버전). 없으면 die 합 fallback
  const unitPred = unitRow && isFinite(parseFloat(unitRow.reg_pred)) ? parseFloat(unitRow.reg_pred) : dieSum
  const unitPpm  = Math.round(unitPred * 1e6)
  const hasThr   = isFinite(unitThreshold) && unitThreshold > 0
  const thrPpm   = hasThr ? Math.round(unitThreshold * 1e6) : null
  const isRisk   = hasThr ? unitPred >= unitThreshold
                          : (unitRow?.grade ? ['grade3', 'grade4'].includes(unitRow.grade) : false)
  // 포지션 막대 길이 기준: die 중 최대 기여로 정규화(유닛 내 상대 비교)
  const posMax   = Math.max(...dies.map(d => parseFloat(d.pred)), 1e-9)
  // die 평균 기여 = 합 / 실제 die 개수(4 고정 아님 — 결측 포지션 대비)
  const avgDie   = dies.length ? dieSum / dies.length : 0
  // 다이버징 바 정규화용: 평균 대비 최대 편차(절대값)
  const maxAbsDev = Math.max(...dies.map(d => Math.abs(parseFloat(d.pred) - avgDie)), 1e-9)
  // 색칩 = 웨이퍼맵과 동일하게 유닛 색(4 die 통일). 없으면 die별 색(백업)
  const chipColor = unitColorScale
    ? predColor(unitPred, unitColorScale.predMin, unitColorScale.predMax, unitColorScale.threshold)
    : null

  return (
    <div className="dd-bottom-panel">
      {/* ① 유닛 판정: reg_pred vs 유닛 임계(P90) */}
      <div className="dd-bottom-thresh">
        <div className="dd-bottom-thresh-label">
          <span className="dd-bottom-thresh-title">
            유닛 판정
            <span className={`dd-verdict-badge ${isRisk ? 'risk' : 'normal'}`} style={{ fontSize: 11, marginLeft: 8 }}>
              {isRisk ? '⚠ 위험' : '✓ 정상'}
            </span>
          </span>
          <span className="dd-bottom-thresh-val">
            {unitPpm.toLocaleString()} ppm
            {thrPpm !== null && <span style={{ color: '#94A3B8', fontWeight: 400 }}> / 임계 {thrPpm.toLocaleString()}</span>}
          </span>
        </div>
        <div style={{ fontSize: 10, color: '#94A3B8', marginTop: 2 }}>
          유닛 예측(보정 합) vs 유닛 위험 임계(reg_pred 상위 10% · P90). die 단위 위험 판정 없음 — 아래는 기여 ppm.
        </div>
      </div>

      {/* ② 포지션별 기여 — die 평균 기준 다이버징 바 (오른쪽=평균보다 더 기여) */}
      {sorted.length > 0 && (
        <div className="dd-bottom-pos">
          <div className="dd-bottom-pos-title">
            포지션별 기여
            <span className="dd-bottom-pos-serial"> · die 평균 {Math.round(avgDie * 1e6).toLocaleString()} ppm 기준 · {ufsSerial}</span>
          </div>
          <div className="dd-bottom-pos-grid">
            {sorted.map(die => {
              const pred   = parseFloat(die.pred)
              const ppm    = Math.round(pred * 1e6)
              const devRaw = pred - avgDie
              const dev    = Math.round(devRaw * 1e6)             // die 평균 대비 ±ppm
              const isMax  = pred >= posMax                       // 최다 기여 die
              const pos    = die.position || '?'
              const isDieSel = selectedDie &&
                String(die.die_x) === String(selectedDie.die_x) &&
                String(die.die_y) === String(selectedDie.die_y)
              const fillColor = predColor(pred, scale.predMin, scale.predMax, scale.threshold)
              const sidePct = Math.round((Math.abs(devRaw) / maxAbsDev) * 50)
              return (
                <div
                  key={`${die.die_x}-${die.die_y}`}
                  className={`dd-bpos-row ${isDieSel ? 'selected' : ''} ${isMax ? 'ismax' : ''}`}
                  onClick={() => onSelectDie?.(die)}
                  title={`(${die.die_x}, ${die.die_y})`}
                >
                  <span className="dd-bpos-num">P{pos}</span>
                  <span className="dd-bpos-chip" style={{ background: chipColor ?? fillColor }} />
                  <span className="dd-bpos-ppm">{ppm.toLocaleString()}<span className="dd-bpos-unit"> ppm</span></span>
                  <div className="dd-bdiv-track">
                    <div className="dd-bdiv-center" />
                    {devRaw >= 0
                      ? <div className="dd-bdiv-fill pos" style={{ left: '50%', width: `${sidePct}%` }} />
                      : <div className="dd-bdiv-fill neg" style={{ right: '50%', width: `${sidePct}%` }} />}
                  </div>
                  <span className={`dd-bdiv-val ${dev > 0 ? 'pos' : dev < 0 ? 'neg' : ''}`}>
                    {dev >= 0 ? '+' : ''}{dev.toLocaleString()}
                  </span>
                  {isMax && <span className="dd-bpos-max">◀ 최다</span>}
                  {isDieSel && <span className="dd-bpos-sel-arrow">◀</span>}
                </div>
              )
            })}
          </div>
          <div className="dd-bdiv-axis"><span>평균보다 적게 기여</span><span>die 평균</span><span>평균보다 많이 기여</span></div>
        </div>
      )}
    </div>
  )
}

// ── Die 진단 패널 ──────────────────────────────────────
function DieReport({ die, scale, onClose, unitAvg, unitMaxAbsDev }) {
  if (!die) return (
    <div className="dd-report-empty">웨이퍼맵에서 다이를 클릭하면 기여도가 표시됩니다.</div>
  )

  const pred    = parseFloat(die.pred)
  const ppm     = Math.round(pred * 1e6)
  const clf     = die.clf_proba !== undefined ? parseFloat(die.clf_proba) : null
  // die 평균(유닛 die 기여 평균, 같은 ppm 스케일) 대비 편차. reg_pred(유닛값)이 아님
  const _hasAvg = unitAvg != null && isFinite(unitAvg)
  const devRaw  = _hasAvg ? (pred - unitAvg) : null
  const dev     = devRaw != null ? Math.round(devRaw * 1e6) : null
  const avgPpm  = _hasAvg ? Math.round(unitAvg * 1e6) : null
  // 하단 포지션 다이버징 바와 동일 정규화(유닛 4 die 최대 편차 기준)
  const _mad    = (unitMaxAbsDev && unitMaxAbsDev > 0) ? unitMaxAbsDev : 1e-9
  const sidePct = devRaw != null ? Math.round(Math.abs(devRaw) / _mad * 50) : 0

  return (
    <div className="dd-report">
      {/* 좌표 + 기여 ppm 요약 (die 단위 위험 판정 없음 — 위험/정상은 유닛 기준) */}
      <div className="dd-die-header normal">
        <div className="dd-die-header-row">
          <span className="dd-verdict-badge normal" style={{ fontSize: 12 }}>기여 die</span>
          <span className="dd-die-coord-main">({die.die_x}, {die.die_y})</span>
          {onClose && <button className="dd-report-close" onClick={onClose}>✕</button>}
        </div>
        <div className="dd-verdict-pred" style={{ fontSize: 11 }}>{ppm.toLocaleString()} ppm 기여</div>
      </div>

      {/* 유닛 기여도 — die 평균 기준 다이버징 바 (하단 포지션 바와 동일) */}
      <div className="dd-section-box">
        <div className="dd-section-title">유닛 기여도 <span style={{ fontSize: 10, color: '#94A3B8', fontWeight: 400 }}>(die 평균 대비)</span></div>
        {dev !== null ? (
          <div style={{ margin: '12px 0 4px' }}>
            <div className="dd-bdiv-track" style={{ height: 18 }}>
              <div className="dd-bdiv-center" />
              {devRaw >= 0
                ? <div className="dd-bdiv-fill pos" style={{ left: '50%', width: `${sidePct}%` }} />
                : <div className="dd-bdiv-fill neg" style={{ right: '50%', width: `${sidePct}%` }} />}
            </div>
            <div className="dd-bdiv-axis" style={{ marginTop: 4 }}>
              <span>평균보다 적게</span>
              <span style={{ fontWeight: 700, color: '#475569' }}>
                {dev >= 0 ? '+' : ''}{dev.toLocaleString()} ppm
                <span style={{ fontWeight: 400, color: '#94A3B8' }}> · die 평균 {avgPpm?.toLocaleString()} ppm</span>
              </span>
              <span>평균보다 많이</span>
            </div>
          </div>
        ) : (
          <div style={{ fontSize: 11, color: '#94A3B8', margin: '8px 0' }}>기여 {ppm.toLocaleString()} ppm</div>
        )}
      </div>

      {/* 상세 수치 */}
      <div className="dd-section-box">
        <div className="dd-section-title">Die 상세</div>
        <div className="dd-detail-rows">
          <div className="dd-detail-row">
            <span className="dd-detail-key">좌표</span>
            <span className="dd-detail-val mono">({die.die_x}, {die.die_y})</span>
          </div>
          <div className="dd-detail-row">
            <span className="dd-detail-key">기여 ppm</span>
            <span className="dd-detail-val mono">{ppm.toLocaleString()} ppm</span>
          </div>
          {dev !== null && (
            <div className="dd-detail-row">
              <span className="dd-detail-key">die 평균 대비</span>
              <span className="dd-detail-val mono" style={{ color: dev > 0 ? '#B91C1C' : '#64748B' }}>
                {dev >= 0 ? '+' : ''}{dev.toLocaleString()} ppm
              </span>
            </div>
          )}
          {clf !== null && (
            <div className="dd-detail-row">
              <span className="dd-detail-key">위험 확률</span>
              <span className="dd-detail-val mono">{(clf * 100).toFixed(1)}%</span>
            </div>
          )}
          <div className="dd-detail-row">
            <span className="dd-detail-key">소속 Unit</span>
            <span className="dd-detail-val mono" style={{ fontSize: 11 }}>{die.ufs_serial}</span>
          </div>
          <div className="dd-detail-row">
            <span className="dd-detail-key">Lot · Wafer</span>
            <span className="dd-detail-val mono">Lot {die.run_id} · #{die.wafer_no}</span>
          </div>
        </div>
      </div>

    </div>
  )
}

// ── 메인 ─────────────────────────────────────────────
// ── Wafer 패턴 휴리스틱 분류 ──
// die 위치 + pred를 받아 Edge Ring / NearFull / Random / Normal 분류
function classifyWaferPattern(dies, threshold) {
  if (!dies.length) return 'normal'
  const xs = dies.map(d => d.die_x), ys = dies.map(d => d.die_y)
  const cx = (Math.max(...xs) + Math.min(...xs)) / 2
  const cy = (Math.max(...ys) + Math.min(...ys)) / 2
  const rMax = Math.max(...dies.map(d => Math.hypot(d.die_x - cx, (d.die_y - cy) * 2.5)))
  if (rMax === 0) return 'normal'

  let centerSum = 0, centerN = 0
  let edgeSum = 0, edgeN = 0
  let highCount = 0

  dies.forEach(d => {
    const r = Math.hypot(d.die_x - cx, (d.die_y - cy) * 2.5) / rMax
    const p = parseFloat(d.pred)
    if (!isFinite(p)) return
    if (p > threshold) highCount++
    if (r < 0.45) { centerSum += p; centerN++ }
    if (r > 0.75) { edgeSum += p; edgeN++ }
  })

  const highRatio = highCount / dies.length

  // 1) NearFull: 위험 die 비율 55% 이상 — 웨이퍼 광역 불량 (공간 편중보다 우선)
  if (highRatio >= 0.55) return 'nearfull'

  // 2) 공간 편중(Edge Ring): 외곽이 중심 대비 1.6배 이상
  if (centerN === 0 || edgeN === 0) return highRatio < 0.10 ? 'normal' : 'random'
  const centerAvg = centerSum / centerN
  const edgeAvg = edgeSum / edgeN
  if (edgeAvg > centerAvg * 1.6 && edgeAvg > threshold * 0.3) return 'edge'

  // 3) 공간 편중이 없으면 위험 die 비율로 구분 (P90 임계값 기준)
  //    10% 미만이면 정상, 이상이면 위험 산발
  if (highRatio < 0.10) return 'normal'
  return 'random'
}

const PATTERN_META = {
  edge:   { label: 'Edge Ring',      color: '#EF4444', desc: '외곽 die 위험 집중 — 식각/세정 균일성 의심' },
  nearfull: { label: 'Near Full',      color: '#991B1B', desc: '웨이퍼 광역 불량' },
  random: { label: 'Random Scatter', color: '#3B82F6', desc: '위험 산발 — 파티클/오염 가능성' },
  normal: { label: 'Normal',         color: '#22C55E', desc: '위험 die가 적은 웨이퍼' },
}

export default function DrilldownV2({ initialSelection }) {
  const { data: summaryData, loading: loadingSummary } = useCSV('/dashboard_lot_summary.csv')
  const { data: shapData } = useCSV('/shap_bar.csv')
  const { data: featNormData } = useCSV('/wafer_feat_norm.csv')
  const { data: unitData } = useCSV('/dashboard_units.csv')
  const { data: lotPatternsAll } = useCSV('/dashboard_lot_patterns.csv')
  const [lotPatternMaps, setLotPatternMaps] = useState({})
  useEffect(() => {
    fetch(dataUrl('/dashboard_lot_pattern_maps.json'))
      .then(r => r.ok ? r.json() : {})
      .then(setLotPatternMaps)
      .catch(() => {})
  }, [])

  const [globalScale, setGlobalScale] = useState(null)
  useEffect(() => {
    fetch(dataUrl('/wafer_scale.json')).then(r => r.json()).then(s => setGlobalScale({
      predMin: s.pred_min,
      predMax: s.pred_max,
      threshold: s.threshold,
      gridXRange: GLOBAL_DIE_X_MAX - GLOBAL_DIE_X_MIN + 1,
      gridYRange: GLOBAL_DIE_Y_MAX - GLOBAL_DIE_Y_MIN + 1,
      gridXMin: GLOBAL_DIE_X_MIN, gridXMax: GLOBAL_DIE_X_MAX,
      gridYMin: GLOBAL_DIE_Y_MIN, gridYMax: GLOBAL_DIE_Y_MAX,
    })).catch(() => {})
  }, [])

  const [selectedLot, setSelectedLot]   = useState(null)
  const [selectedKey, setSelectedKey]   = useState(null)
  const [selectedUnit, setSelectedUnit] = useState(null)
  const [shapUnitMap, setShapUnitMap] = useState(null)
  useEffect(() => {
    fetch(dataUrl('/shap_unit.json'))
      .then(r => { if (!r.ok) throw new Error(r.status); return r.json() })
      .then(d => setShapUnitMap(d))
      .catch(e => console.warn('[ShapUnit] 로드 실패:', e))
  }, [])
  const [selectedDie,  setSelectedDie]  = useState(null)
  const [search, setSearch]             = useState('')
  const [expandedLot, setExpandedLot]   = useState(null)

  // lot 클릭 시 해당 lot의 wafer_map만 로드
  const [loadedLot, setLoadedLot] = useState(null)
  const { data: dieData, loading: loadingDie } = useCSV(loadedLot ? `/wafer_map_lots/lot_${loadedLot}.csv` : null)

  const pendingUnitRef = useRef(null)  // 메인 outlier 네비게이션 시 선택할 unit (initialSelection 유래)

  // 외부에서 initialSelection 전달받으면 자동 선택 (Lot 즉시 → die 로드 트리거)
  useEffect(() => {
    if (!initialSelection) return
    const { lot } = initialSelection
    if (lot) {
      setSelectedLot(String(lot))
      setExpandedLot(String(lot))
      setLoadedLot(String(lot))
      setActiveTab('default')
    }
  }, [initialSelection])

  // die 데이터 도착 후 wafer/unit 선택 (selectedKey 리셋 effect보다 나중)
  useEffect(() => {
    if (!initialSelection || !dieData.length) return
    const { lot, wafer, unit } = initialSelection
    if (lot && wafer) {
      setSelectedKey(`${lot}_${wafer}`)
    }
    if (unit) {
      pendingUnitRef.current = unit   // dies 로드 후 선택 (아래 effect ⑤)
    }
  }, [initialSelection, dieData])
  const [waferSort, setWaferSort]       = useState('default') // 'default' | 'risk_desc' | 'risk_asc'
  const [lotSort, setLotSort]           = useState('risk_desc') // 'risk_desc' | 'risk_asc' | 'default'
  const [activeTab, setActiveTab]       = useState('pattern')   // 'pattern' | 'default'
  const [selectedPattern, setSelectedPattern] = useState(null)  // 'edge' | 'nearfull' | 'random' | 'normal'
  const [zoomLot, setZoomLot] = useState(null)

  const scale = useMemo(() => globalScale ?? computeScale(dieData), [globalScale, dieData])

  // grade4(매우위험) unit이 있는 wafer / lot 집합 = 이상치
  const outlierWaferKeys = useMemo(() => {
    const s = new Set()
    unitData.forEach(u => {
      if (u.grade === 'grade4') s.add(`${u.run_id}_${u.wafer_no}`)
    })
    return s
  }, [unitData])
  const outlierLots = useMemo(() => {
    const s = new Set()
    unitData.forEach(u => {
      if (u.grade === 'grade4') s.add(String(u.run_id))
    })
    return s
  }, [unitData])

  // 유닛 reg_pred(보정 합버전) 기준 lot/wafer 평균 ppm 맵
  // 계층탐색·패턴분류의 ppm을 die 평균이 아닌 유닛 예측값으로 통일
  const { unitLotPpm, unitWaferPpm } = useMemo(() => {
    const lotAgg = {}    // lot -> {sum, n}
    const waferAgg = {}  // `${lot}_${wno}` -> {sum, n}
    unitData.forEach(u => {
      const rp = parseFloat(u.reg_pred)
      if (!isFinite(rp)) return
      const lot = String(u.run_id)
      const wkey = `${lot}_${u.wafer_no}`
      if (!lotAgg[lot]) lotAgg[lot] = { sum: 0, n: 0 }
      if (!waferAgg[wkey]) waferAgg[wkey] = { sum: 0, n: 0 }
      lotAgg[lot].sum += rp; lotAgg[lot].n += 1
      waferAgg[wkey].sum += rp; waferAgg[wkey].n += 1
    })
    const lotPpm = {}, waferPpm = {}
    Object.entries(lotAgg).forEach(([k, v]) => { lotPpm[k] = v.n ? Math.round((v.sum / v.n) * 1e6) : 0 })
    Object.entries(waferAgg).forEach(([k, v]) => { waferPpm[k] = v.n ? Math.round((v.sum / v.n) * 1e6) : 0 })
    return { unitLotPpm: lotPpm, unitWaferPpm: waferPpm }
  }, [unitData])

  // 유닛 위험 임계 = reg_pred P75 (상위 25%를 위험으로 색칠 — 노란~빨간 범위 확대)
  const unitThreshold = useMemo(() => {
    const arr = unitData.map(u => parseFloat(u.reg_pred)).filter(isFinite).sort((a, b) => a - b)
    if (!arr.length) return null
    return arr[Math.floor(arr.length * 0.75)]
  }, [unitData])

  // 웨이퍼맵 유닛 기준 색칠용: serial→reg_pred 맵 + reg_pred 분포 스케일(임계=P90)
  const unitColorMap = useMemo(() => {
    const m = new Map()
    unitData.forEach(u => { const v = parseFloat(u.reg_pred); if (isFinite(v)) m.set(String(u.ufs_serial), v) })
    return m
  }, [unitData])
  const unitColorScale = useMemo(() => {
    if (unitThreshold == null || !unitColorMap.size) return null
    let mn = Infinity, mx = -Infinity
    unitColorMap.forEach(v => { if (v < mn) mn = v; if (v > mx) mx = v })
    return isFinite(mn) ? { predMin: mn, predMax: mx, threshold: unitThreshold } : null
  }, [unitColorMap, unitThreshold])

  const lotTree = useMemo(() => {
    if (!summaryData.length) return []
    const lotMap = {}
    summaryData.forEach(d => {
      const lotNum = parseInt(d.run_id)
      // 원본 0_data 기준 lot 1~28만 표시 (29~84는 split 시뮬레이션 분배)
      if (!(lotNum >= 1 && lotNum <= 28)) return
      const lot = String(d.run_id)
      const wno = String(d.wafer_no)
      const key = `${lot}_${wno}`
      if (!lotMap[lot]) lotMap[lot] = { lot, wafers: {}, totalUnits: 0, riskUnits: 0 }
      lotMap[lot].wafers[wno] = {
        wno, key,
        units: Number(d.total_units) || 0,
        riskUnits: Number(d.risk_units) || 0,
        avgPpm: unitWaferPpm[key] ?? 0,  // 유닛 reg_pred 평균 기준
      }
      lotMap[lot].totalUnits += Number(d.total_units) || 0
      lotMap[lot].riskUnits  += Number(d.risk_units) || 0
    })
    let lots = Object.values(lotMap)
    if (search.trim()) {
      const q = search.toLowerCase()
      lots = lots.filter(l =>
        l.lot.includes(q) || Object.keys(l.wafers).some(w => w.includes(q))
      )
    }
    lots.sort((a, b) => {
      const rA = a.totalUnits ? a.riskUnits / a.totalUnits : 0
      const rB = b.totalUnits ? b.riskUnits / b.totalUnits : 0
      if (lotSort === 'risk_asc') return rA - rB
      if (lotSort === 'default') return parseInt(a.lot) - parseInt(b.lot)
      return rB - rA
    })
    return lots.map(l => ({
      ...l,
      riskRatio: l.totalUnits ? l.riskUnits / l.totalUnits : 0,
      avgPpm: unitLotPpm[l.lot] ?? 0,  // 유닛 reg_pred 평균 기준
      waferList: Object.values(l.wafers).sort((a, b) => parseInt(a.wno) - parseInt(b.wno)),
    }))
  }, [summaryData, search, lotSort, unitLotPpm, unitWaferPpm])

  const selectedDies = useMemo(() => {
    if (!selectedKey) return []
    const [lot, wno] = selectedKey.split('_')
    return dieData.filter(d => String(d.run_id) === lot && String(d.wafer_no) === wno)
  }, [dieData, selectedKey])


  const lotAccumDies = useMemo(() => {
    if (!selectedLot || selectedKey) return []
    const lotDies = dieData.filter(d => String(d.run_id) === selectedLot)
    const posMap = {}
    lotDies.forEach(d => {
      const k = `${d.die_x},${d.die_y}`
      if (!posMap[k] || parseFloat(d.pred) > parseFloat(posMap[k].pred)) posMap[k] = d
    })
    return Object.values(posMap)
  }, [dieData, selectedLot, selectedKey])

  useEffect(() => { setSelectedUnit(null); setSelectedDie(null) }, [selectedKey])

  // ── 패턴 분류: 현재 로드된 lot의 wafer별 패턴 분류 ──
  const patternResult = useMemo(() => {
    if (!dieData.length || !scale) return null
    const threshold = scale.threshold

    // wafer별 die 그룹핑
    const waferMap = {}
    dieData.forEach(d => {
      const lot = String(d.run_id)
      const wno = String(d.wafer_no)
      const key = `${lot}_${wno}`
      if (!waferMap[key]) waferMap[key] = { lot, wno, dies: [] }
      waferMap[key].dies.push(d)
    })

    // 각 wafer 분류
    const classified = Object.values(waferMap).map(w => ({
      ...w,
      pattern: classifyWaferPattern(w.dies, threshold),
      riskRatio: w.dies.filter(d => parseFloat(d.pred) > threshold).length / w.dies.length,
      avgPred: w.dies.reduce((s, d) => s + parseFloat(d.pred), 0) / w.dies.length,
    }))

    // 카테고리별 집계
    const buckets = { edge: [], nearfull: [], random: [], normal: [] }
    classified.forEach(w => { if (buckets[w.pattern]) buckets[w.pattern].push(w) })
    return { wafers: classified, buckets }
  }, [dieData, scale])

  // ── 전체 Lot 패턴 집계 (사전 계산 파일 기반) ──
  const lotPatternBuckets = useMemo(() => {
    const buckets = { edge: [], nearfull: [], random: [], normal: [] }
    lotPatternsAll.forEach(r => {
      const p = String(r.pattern)
      if (buckets[p]) buckets[p].push({
        lot: String(r.lot),
        pattern: p,
        riskRatio: parseFloat(r.risk_ratio) || 0,
        avgPred: parseFloat(r.avg_pred) || 0,
        nDies: parseInt(r.n_dies) || 0,
        nWafers: parseInt(r.n_wafers) || 0,
      })
    })
    return buckets
  }, [lotPatternsAll])

  // ── Lot 통합 패턴: 선택된 Lot의 모든 wafer die를 합친 통합맵 기준 분류 ──
  const lotPattern = useMemo(() => {
    if (!selectedLot || !scale || !lotAccumDies.length) return null
    const pat = classifyWaferPattern(lotAccumDies, scale.threshold)
    const riskN = lotAccumDies.filter(d => parseFloat(d.pred) > scale.threshold).length
    return {
      pattern: pat,
      dies: lotAccumDies,
      riskRatio: riskN / lotAccumDies.length,
      avgPred: lotAccumDies.reduce((s, d) => s + parseFloat(d.pred), 0) / lotAccumDies.length,
    }
  }, [selectedLot, scale, lotAccumDies])

  // ── 기본 선택 ① 패턴 탭 진입 시 edge(Edge Ring) 자동 선택 ──
  useEffect(() => {
    if (lotPatternsAll.length && !selectedPattern) setSelectedPattern('edge')
  }, [lotPatternsAll])  // eslint-disable-line react-hooks/exhaustive-deps

  // ── 기본 선택 ② 패턴 선택 시 위험률 최고 lot을 기본으로 슬라이드인 표시 ──
  useEffect(() => {
    if (!selectedPattern) { setZoomLot(null); return }
    const bucket = lotPatternBuckets[selectedPattern]
    if (bucket?.length) {
      setZoomLot([...bucket].sort((a, b) => b.riskRatio - a.riskRatio)[0])
    } else {
      setZoomLot(null)
    }
  }, [selectedPattern, lotPatternBuckets])  // eslint-disable-line react-hooks/exhaustive-deps

  // ── 기본 선택 ③ (제거됨): 계층탐색 진입 시 lot 자동 선택 안 함 → 로트 목록만 표시 ──
  //    (사용자가 직접 lot을 클릭해야 wafer/맵이 보임)

  // ── 기본 선택 ④ (제거됨): 위험률 최고 wafer 자동 선택 안 함 ──

  // ── 기본 선택 ⑤ 수동 wafer 클릭은 unit 자동선택 안 함.
  //    단, 메인 outlier 맵에서 넘어온 경우(initialSelection.unit)에만 해당 unit 선택 ──
  useEffect(() => {
    if (!pendingUnitRef.current || !selectedDies.length) return
    const u = pendingUnitRef.current
    if (selectedDies.some(d => String(d.ufs_serial) === String(u))) {
      setSelectedUnit(u)
      pendingUnitRef.current = null
    }
  }, [selectedDies])  // eslint-disable-line react-hooks/exhaustive-deps

  // 유닛 기반 위험비율(위험 유닛/전체 유닛) 스케일
  //  분포: 로트 최대 ~35%, 웨이퍼 최대 ~57%, 중앙값 ~0%
  //  색: <10% 초록, 10~25% 노랑, 25%+ 빨강 / 바: 일반 0~100% (100%=만충)
  function absBarWidth(ratio) {
    return `${Math.min(100, Math.round(ratio * 100))}%`
  }
  function absClass(ratio) {
    return ratio >= 0.25 ? 'danger' : ratio >= 0.10 ? 'warn' : 'ok'
  }

  const currentDies = selectedKey ? selectedDies : lotAccumDies

  return (
    <div className="drilldown">

      {/* ── 탭 바 ── */}
      <div className="dd-tab-bar">
        <button
          className={`dd-tab ${activeTab === 'pattern' ? 'active' : ''}`}
          onClick={() => setActiveTab('pattern')}
        >
          패턴 분류 (Lot 통합)
        </button>
        <button
          className={`dd-tab ${activeTab === 'default' ? 'active' : ''}`}
          onClick={() => setActiveTab('default')}
        >
          계층 탐색 (Lot → Wafer → Unit)
        </button>
      </div>

      {activeTab === 'pattern' && (
        <div className="dd-pattern-view">
          {/* 전체 Lot 패턴 요약 카드 4개 */}
          {lotPatternsAll.length > 0 && (
            <div className="dd-pattern-cards">
              {['edge', 'nearfull', 'random', 'normal'].map(pat => {
                const meta = PATTERN_META[pat]
                const lots = lotPatternBuckets[pat]
                const pct = lotPatternsAll.length
                  ? (lots.length / lotPatternsAll.length * 100).toFixed(0)
                  : 0
                const isActive = selectedPattern === pat
                return (
                  <div
                    key={pat}
                    className={`dd-pattern-card ${isActive ? 'active' : ''}`}
                    style={{ borderColor: isActive ? meta.color : undefined }}
                    onClick={() => setSelectedPattern(isActive ? null : pat)}
                  >
                    <div className="dd-pattern-card-head" style={{ color: meta.color }}>
                      <span className="dd-pattern-dot" style={{ background: meta.color }} />
                      {meta.label}
                    </div>
                    <div className="dd-pattern-card-count">
                      {lots.length}<span style={{ fontSize: 12, color: '#94A3B8' }}> Lot ({pct}%)</span>
                    </div>
                    <div className="dd-pattern-card-desc">{meta.desc}</div>
                  </div>
                )
              })}
            </div>
          )}

          {/* 선택된 패턴의 Lot 썸네일 그리드 (전체 너비) */}
          {selectedPattern && lotPatternBuckets[selectedPattern].length > 0 && (
            <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start' }}>
              <div className="dd-pattern-grid">
                <div className="dd-pattern-grid-title">
                  {PATTERN_META[selectedPattern].label} — {lotPatternBuckets[selectedPattern].length} Lot
                </div>
                <div className="dd-pattern-map-grid">
                  {lotPatternBuckets[selectedPattern]
                    .sort((a, b) => b.riskRatio - a.riskRatio)
                    .map(l => {
                      const rawDies = lotPatternMaps[l.lot] || []
                      const dies = rawDies.map(([x, y, p]) => ({ die_x: x, die_y: y, pred: p }))
                      const isActive = zoomLot && zoomLot.lot === l.lot
                      return (
                        <div
                          key={l.lot}
                          className={`dd-pattern-mini ${isActive ? 'active' : ''}`}
                          onClick={() => setZoomLot(isActive ? null : l)}
                          title="클릭 → 우측에 크게 보기"
                        >
                          <div className="dd-pattern-mini-head">
                            Lot {l.lot}
                            <span className="dd-pattern-mini-risk">{(l.riskRatio * 100).toFixed(0)}%</span>
                          </div>
                          <div className="dd-pattern-mini-map">
                            {scale && dies.length > 0 && <WaferMap dies={dies} scale={scale} mini colorScale={WMAP_COLOR_BY === 'unit' ? unitColorScale : undefined} />}
                          </div>
                        </div>
                      )
                    })}
                </div>
              </div>

              {/* 슬라이드인 미리보기 패널 */}
              {zoomLot && (() => {
                const rawDies = lotPatternMaps[zoomLot.lot] || []
                const dies = rawDies.map(([x, y, p]) => ({ die_x: x, die_y: y, pred: p }))
                const meta = PATTERN_META[zoomLot.pattern]
                return (
                  <div className="dd-pattern-slidein">
                    <div className="dd-pattern-slidein-header">
                      <div>
                        <div className="dd-zoom-lot">Lot {zoomLot.lot}</div>
                        <div className="dd-zoom-pat" style={{ color: meta.color }}>
                          <span className="dd-pattern-dot" style={{ background: meta.color }} />
                          {meta.label}
                        </div>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <div className="dd-zoom-stats">
                          <div>die <b>{zoomLot.nDies}</b></div>
                          <div>wafer <b>{zoomLot.nWafers}</b></div>
                          <div>위험 <b style={{ color: '#EF4444' }}>{(zoomLot.riskRatio * 100).toFixed(1)}%</b></div>
                          <div>평균 <b>{(unitLotPpm[zoomLot.lot] ?? Math.round(zoomLot.avgPred * 1e6)).toLocaleString()} ppm</b></div>
                        </div>
                        <button className="dd-report-close" onClick={() => setZoomLot(null)}>✕</button>
                      </div>
                    </div>
                    <div className="dd-pattern-slidein-map">
                      {scale && dies.length > 0 && <WaferMap dies={dies} scale={scale} colorScale={WMAP_COLOR_BY === 'unit' ? unitColorScale : undefined} />}
                    </div>
                    <div className="dd-zoom-actions">
                      <button
                        className="dd-zoom-btn"
                        onClick={() => {
                          setSelectedLot(zoomLot.lot)
                          setExpandedLot(zoomLot.lot)
                          setLoadedLot(zoomLot.lot)
                          setSelectedKey(null)
                          setActiveTab('default')
                        }}
                      >
                        계층 탐색에서 상세 분석 →
                      </button>
                    </div>
                  </div>
                )
              })()}
            </div>
          )}

        </div>
      )}

      {activeTab === 'default' && (
      <div className="dd-body-3col">

        {/* ── 좌: 아코디언 트리 ── */}
        <div className="dd-left-panel">
          <div className="dd-tree-controls">
            <button
              className={`dd-wafer-sort-btn ${lotSort !== 'default' ? 'active' : ''}`}
              onClick={() => setLotSort(s => s === 'risk_desc' ? 'risk_asc' : s === 'risk_asc' ? 'default' : 'risk_desc')}
            >
              {lotSort === 'risk_desc' ? '▼ 위험률순' : lotSort === 'risk_asc' ? '▲ 위험률순' : '· 번호순'}
            </button>
          </div>

          {/* 컬럼 헤더 */}
          <div className="dd-tree-col-header">
            <span className="dd-col-lot">LOT</span>
            <span className="dd-col-bar">위험비율</span>
            <span className="dd-col-pct">%</span>
            <span className="dd-col-ppm">PPM</span>
          </div>

          <div className="dd-tree-list">
            {loadingSummary && <div className="dd-tree-hint">로딩 중...</div>}
            {!loadingDie && lotTree.length === 0 && (
              <div className="dd-tree-hint">결과 없음</div>
            )}
            {lotTree.map(({ lot, riskRatio, avgPpm, waferList }) => {
              const isExpanded = expandedLot === lot
              const isLotSel   = selectedLot === lot
              const riskPct    = (riskRatio * 100).toFixed(1)
              const riskClass  = absClass(riskRatio)
              const isLotOutlier = outlierLots.has(lot)

              return (
                <div key={lot} className="dd-lot-group">
                  {/* Lot 행 */}
                  <button
                    className={`dd-lot-btn ${isLotSel ? 'selected' : ''}`}
                    title={isLotOutlier ? '이상치 웨이퍼 포함 Lot (매우위험 unit)' : undefined}
                    onClick={() => {
                      const next = isExpanded ? null : lot
                      setExpandedLot(next)
                      setSelectedLot(next)
                      setSelectedKey(null)
                      setSelectedUnit(null)
                      if (next) setLoadedLot(next)
                    }}
                  >
                    <span className="dd-lot-arrow">{isExpanded ? '▾' : '▸'}</span>
                    <span className="dd-lot-name">{isLotOutlier && <span className="dd-lot-warn">⚠</span>}Lot {lot}</span>
                    <div className="dd-lot-bar-wrap">
                      <div
                        className={`dd-lot-bar-fill ${riskClass}`}
                        style={{ width: absBarWidth(riskRatio) }}
                      />
                    </div>
                    <span className={`dd-lot-pct ${riskClass}`}>{riskPct}%</span>
                    <span className="dd-lot-ppm">{avgPpm.toLocaleString()}</span>
                  </button>

                  {/* Wafer 서브 리스트 */}
                  {isExpanded && (
                    <div className="dd-wafer-list">
                      <button
                        className={`dd-wafer-sort-btn ${waferSort !== 'default' ? 'active' : ''}`}
                        onClick={() => setWaferSort(s => s === 'default' ? 'risk_desc' : s === 'risk_desc' ? 'risk_asc' : 'default')}
                      >
                        {waferSort === 'risk_desc' ? '▼ 위험률 내림차순' : waferSort === 'risk_asc' ? '▲ 위험률 오름차순' : '· 번호순'}
                      </button>
                      {[...waferList]
                        .sort((a, b) => waferSort === 'risk_desc'
                          ? (b.units ? b.riskUnits / b.units : 0) - (a.units ? a.riskUnits / a.units : 0)
                          : waferSort === 'risk_asc'
                          ? (a.units ? a.riskUnits / a.units : 0) - (b.units ? b.riskUnits / b.units : 0)
                          : parseInt(a.wno) - parseInt(b.wno)
                        )
                        .map(w => {
                        const wRatio  = w.units ? w.riskUnits / w.units : 0
                        const wClass  = absClass(wRatio)
                        const wPpm    = w.avgPpm
                        const isSel   = selectedKey === w.key
                        const isOutlier = outlierWaferKeys.has(w.key)
                        return (
                          <button
                            key={w.key}
                            className={`dd-wafer-btn ${isSel ? 'selected' : ''} ${isOutlier ? 'outlier' : ''}`}
                            title={isOutlier ? '이상치 웨이퍼 (매우위험 unit 포함)' : undefined}
                            onClick={() => {
                              setSelectedLot(lot)
                              setSelectedKey(isSel ? null : w.key)
                              setSelectedUnit(null)
                              setLoadedLot(lot)
                            }}
                          >
                            <span className="dd-wafer-no">{isOutlier ? '⚠ ' : ''}#{w.wno}</span>
                            <div className="dd-lot-bar-wrap">
                              <div
                                className={`dd-lot-bar-fill ${wClass}`}
                                style={{ width: absBarWidth(wRatio) }}
                              />
                            </div>
                            <span className={`dd-lot-pct ${wClass}`}>{(wRatio * 100).toFixed(0)}%</span>
                            <span className="dd-lot-ppm">{wPpm.toLocaleString()}</span>
                          </button>
                        )
                      })}
                    </div>
                  )}

                </div>
              )
            })}
          </div>
        </div>

        {/* ── 중: WaferMap + 하단 패널 ── */}
        <div className="dd-center-panel">
          {!selectedKey && !selectedLot && (
            <div className="dd-center-panel-inner">
              <div className="dd-panel-title">Wafer Map</div>
              <div className="dd-map-hint">← 좌측 목록에서 Lot을 클릭해 펼친 후 Wafer를 선택하세요</div>
              <WaferBottomPanel
                ufsSerial={null}
                allDies={[]}
                scale={scale}
                selectedDie={null}
                onSelectDie={setSelectedDie}
              />
            </div>
          )}

          {/* 단일 wafer */}
          {selectedKey && (
            <div className="dd-center-panel-inner">
              <div className="dd-panel-header">
                <span className="dd-panel-title">Wafer {selectedKey.replace('_', ' · #')}</span>
                <span className="dd-panel-meta">Dies {selectedDies.length}</span>
              </div>
              <div className="dd-wmap-top">
                <WaferMap
                  dies={selectedDies}
                  scale={scale}
                  selectedUnit={selectedUnit}
                  onSelectUnit={setSelectedUnit}
                  selectedDie={selectedDie}
                  onSelectDie={setSelectedDie}
                  unitColorMap={WMAP_COLOR_BY === 'unit' ? unitColorMap : undefined}
                  unitColorScale={WMAP_COLOR_BY === 'unit' ? unitColorScale : undefined}
                />
              </div>
              <WaferBottomPanel
                ufsSerial={selectedUnit}
                allDies={selectedDies}
                scale={scale}
                selectedDie={selectedDie}
                onSelectDie={setSelectedDie}
                unitRow={unitData.find(u => u.ufs_serial === selectedUnit)}
                unitThreshold={unitThreshold}
                unitColorScale={WMAP_COLOR_BY === 'unit' ? unitColorScale : undefined}
              />
            </div>
          )}

          {/* lot 누적 */}
          {!selectedKey && selectedLot && (
            <div className="dd-center-panel-inner">
              <div className="dd-panel-header">
                <span className="dd-panel-title">Lot {selectedLot} — 누적 (max)</span>
                <span className="dd-panel-meta">
                  unique positions {lotAccumDies.length}
                </span>
              </div>
              <div className="dd-map-hint sm">
                lot 내 모든 wafer를 같은 die 좌표로 겹친 max 집계 — 좌측에서 Wafer를 선택하면 단일 보기로 전환
              </div>
              <div className="dd-wmap-top">
                <WaferMap
                  dies={lotAccumDies}
                  scale={scale}
                  selectedUnit={null}
                  onSelectUnit={undefined}
                  unitColorMap={WMAP_COLOR_BY === 'unit' ? unitColorMap : undefined}
                  unitColorScale={WMAP_COLOR_BY === 'unit' ? unitColorScale : undefined}
                />
              </div>
              <WaferBottomPanel
                ufsSerial={null}
                allDies={lotAccumDies}
                scale={scale}
                selectedDie={null}
                onSelectDie={setSelectedDie}
              />
            </div>
          )}
        </div>

        {/* ── 우: Unit | Die 가로 2열 진단 ── */}
        <div className="dd-right-panel">
          {/* Unit 진단 열 */}
          <div className="dd-right-unit-col">
            <div className="dd-right-panel-title">Unit 진단</div>
            <UnitReport
              key={selectedUnit}
              ufsSerial={selectedUnit}
              allDies={selectedDies}
              scale={scale}
              shapData={shapData}
              shapUnitMap={shapUnitMap}
              unitData={unitData}
              featNormData={featNormData}
              onClose={selectedUnit ? () => { setSelectedUnit(null); setSelectedDie(null) } : undefined}
            />
          </div>
          {/* Die 진단 열 */}
          <div className="dd-right-die-col">
            <div className="dd-right-panel-title">Die 진단</div>
            <DieReport
              die={selectedDie}
              scale={scale}
              onClose={selectedDie ? () => setSelectedDie(null) : undefined}
              {...(selectedDie ? (() => {
                const ud = selectedDies.filter(d => d.ufs_serial === selectedDie.ufs_serial)
                if (!ud.length) return { unitAvg: null, unitMaxAbsDev: null }
                const avg = ud.reduce((s, d) => s + parseFloat(d.pred), 0) / ud.length
                const mad = Math.max(...ud.map(d => Math.abs(parseFloat(d.pred) - avg)), 1e-9)
                return { unitAvg: avg, unitMaxAbsDev: mad }
              })() : { unitAvg: null, unitMaxAbsDev: null })}
            />
          </div>
        </div>

      </div>
      )}
    </div>
  )
}
