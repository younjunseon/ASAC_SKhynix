import { useMemo, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import { useCSV } from '../hooks/useCSV'
import './ProcessFactor.css'

const TOP_N = 5
const BASELINE_RMSE = 0.005845
const FIXED_DATE    = '2026-06-11'

// X0~X1086 같은 WT 피처만 허용 (die_x, die_y, position 등 메타 제외)
function isXFeature(f) {
  return /^X\d+$/.test(String(f))
}

function quantile(sorted, q) {
  if (!sorted.length) return null
  const pos = (sorted.length - 1) * q
  const base = Math.floor(pos)
  const rest = pos - base
  const next = sorted[base + 1] ?? sorted[base]
  return sorted[base] + rest * (next - sorted[base])
}

function fmt(v) {
  if (v == null || !isFinite(v)) return '-'
  if (Math.abs(v) >= 1000) return v.toFixed(0)
  if (Math.abs(v) >= 10) return v.toFixed(2)
  return v.toFixed(4)
}

export default function ProcessFactor() {
  const { data: shapBarRaw }      = useCSV('/shap_bar.csv')
  const { data: shapBeeswarmRaw } = useCSV('/shap_beeswarm.csv')
  const { data: unitsRaw }        = useCSV('/dashboard_units.csv')
  const { data: featDistRaw }     = useCSV('/feature_dist.csv')
  const { data: fiRaw }           = useCSV('/feature_importance.csv')
  const { data: metricsRaw }      = useCSV('/metrics.csv')

  const [selFeat, setSelFeat] = useState(null)
  // 정렬 기준 토글: 'default'(각 차트 자기 지표) | 'shap'(둘 다 |SHAP|) | 'fi'(둘 다 LGBM gain)
  const [sortBy, setSortBy] = useState('default')

  // 피처별 점수 맵 (X 피처만)
  const gainMap = useMemo(() => {
    const m = {}
    fiRaw.forEach(r => { if (isXFeature(r.feature)) m[r.feature] = parseFloat(r.lgbm_gain || 0) })
    return m
  }, [fiRaw])
  const shapMag = useMemo(() => {
    const m = {}
    shapBarRaw.forEach(r => { if (isXFeature(r.feature)) m[r.feature] = parseFloat(r.mean_abs_shap || 0) })
    return m
  }, [shapBarRaw])

  // 각 지표별 상위 20개(높은 순) 피처
  const fiOrder = useMemo(
    () => Object.keys(gainMap).sort((a, b) => gainMap[b] - gainMap[a]).slice(0, 20),
    [gainMap])
  const shapOrder = useMemo(
    () => Object.keys(shapMag).sort((a, b) => shapMag[b] - shapMag[a]).slice(0, 20),
    [shapMag])

  // 차트별 정렬 순서:
  //  - 'default': 피처임포턴스 차트=gain순, SHAP 차트=SHAP순 (각자 자기 지표)
  //  - 'fi'     : 둘 다 gain순 / 'shap': 둘 다 SHAP순
  const fiChartFeats = useMemo(
    () => (sortBy === 'shap' ? shapOrder : fiOrder),
    [sortBy, fiOrder, shapOrder])
  const beeswarmFeats = useMemo(
    () => (sortBy === 'fi' ? fiOrder : shapOrder),
    [sortBy, fiOrder, shapOrder])

  // ── RMSE ──────────────────────────────────────────────────
  const bestVal = useMemo(() => {
    if (!metricsRaw.length) return null
    const get = (stage, model, split) => {
      const row = metricsRaw.find(r => r.stage === stage && r.model === model && r.split === split && r.metric === 'rmse')
      return row ? parseFloat(row.value) : null
    }
    return get('reg','stacking','val') ?? get('reg','ensemble','val') ?? get('reg','lgbm','val')
  }, [metricsRaw])

  // ── 피처 중요도 ───────────────────────────────────────────
  const fiOption = useMemo(() => {
    if (!fiRaw.length || !fiChartFeats.length) return null
    // 메타 피처(die_x, die_y, position)는 공정 인자가 아니므로 제외 — X 피처만
    const fiFiltered = fiRaw.filter(r => isXFeature(r.feature))
    const totalGain = fiFiltered.reduce((s, r) => s + parseFloat(r.lgbm_gain || 0), 0) || 1
    // 막대값은 LGBM Gain%, 정렬 순서는 선택한 기준(fiChartFeats)을 따름
    const items = fiChartFeats
      .map(f => ({ feature: f, pct: (gainMap[f] ?? 0) / totalGain * 100 }))
      .reverse()
    // 20개 중 상위 10개(위쪽 50%)만 보이게 — 휠/슬라이더로 나머지 스크롤, 창 크기 50% 고정
    return {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, formatter: p => `<b>${p[0].name}</b><br/>중요도: ${parseFloat(p[0].value).toFixed(2)}%` },
      grid: { top: 8, bottom: 8, left: 8, right: 78, containLabel: true },
      dataZoom: [
        { type: 'inside', yAxisIndex: 0, start: 50, end: 100, minSpan: 50, maxSpan: 50, zoomOnMouseWheel: false, moveOnMouseWheel: true, moveOnMouseMove: true },
        { type: 'slider', yAxisIndex: 0, start: 50, end: 100, minSpan: 50, maxSpan: 50, right: 6, width: 12, handleSize: 0, showDetail: false, brushSelect: false, fillerColor: 'rgba(59,130,246,0.18)', borderColor: 'transparent', backgroundColor: '#F1F5F9' },
      ],
      xAxis: { type: 'value', axisLabel: { fontSize: 10, color: '#94A3B8', formatter: '{value}%' }, splitLine: { lineStyle: { color: '#F1F5F9' } } },
      yAxis: { type: 'category', data: items.map(d => d.feature), axisLabel: { fontSize: 10, color: '#374151', fontFamily: 'monospace' }, axisTick: { show: false } },
      series: [{ type: 'bar', data: items.map(d => ({ value: +d.pct.toFixed(2), itemStyle: { color: d.feature === selFeat ? '#7C3AED' : '#3B82F6', borderRadius: [0, 4, 4, 0] } })), barMaxWidth: 26, label: { show: true, position: 'right', fontSize: 10, color: '#64748B', formatter: p => `${parseFloat(p.value).toFixed(2)}%` } }],
    }
  }, [fiRaw, gainMap, fiChartFeats, selFeat])

  // 위험/안전 구분 기준: grade 대신 reg_pred 연속 임계값(상위 10% = P90)
  const regPredMap = useMemo(() => {
    const m = {}
    unitsRaw.forEach(u => { const v = parseFloat(u.reg_pred); if (isFinite(v)) m[u.ufs_serial] = v })
    return m
  }, [unitsRaw])
  const riskThreshold = useMemo(() => {
    const vals = unitsRaw.map(u => parseFloat(u.reg_pred)).filter(v => isFinite(v)).sort((a, b) => a - b)
    return vals.length ? quantile(vals, 0.90) : null
  }, [unitsRaw])
  // 안전 그룹 경계: 하위 10% (P10) — 극단군 비교(안전10% vs 위험10%)용
  const safeThreshold = useMemo(() => {
    const vals = unitsRaw.map(u => parseFloat(u.reg_pred)).filter(v => isFinite(v)).sort((a, b) => a - b)
    return vals.length ? quantile(vals, 0.10) : null
  }, [unitsRaw])

  // SHAP top 피처 정렬
  const shapSorted = useMemo(() => {
    if (!shapBarRaw.length) return []
    return [...shapBarRaw]
      .filter(r => isXFeature(r.feature))   // die_x/die_y/position 등 메타 제외
      .sort((a, b) => parseFloat(b.mean_abs_shap) - parseFloat(a.mean_abs_shap))
  }, [shapBarRaw])

  // Feature Pareto (SHAP 기준 누적 기여도)
  const paretoData = useMemo(() => {
    if (!shapSorted.length) return null
    const TOP = 20
    const items = shapSorted.slice(0, TOP).map(r => ({
      feature: r.feature,
      shap: parseFloat(r.mean_abs_shap),
    }))
    const totalAll = shapSorted.reduce((s, r) => s + parseFloat(r.mean_abs_shap), 0)
    const rows = items.map((it, i) => {
      const cum = items.slice(0, i + 1).reduce((s, r) => s + r.shap, 0)
      return { ...it, cumPct: totalAll ? +(cum / totalAll * 100).toFixed(1) : 0 }
    })
    const top80Idx = rows.findIndex(r => r.cumPct >= 80)
    const top80Count = top80Idx >= 0 ? top80Idx + 1 : rows.length
    return { rows, totalAll, top80Count, top80Pct: rows[top80Count - 1]?.cumPct ?? 0 }
  }, [shapSorted])

  const paretoOption = useMemo(() => {
    if (!paretoData) return null
    const { rows } = paretoData
    const maxShap = Math.max(...rows.map(r => r.shap))
    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: p => {
          const bar = p.find(s => s.seriesName === '개별 영향도')
          const line = p.find(s => s.seriesName === '누적 %')
          return `<b>${p[0].axisValue}</b><br/>` +
            (bar ? `${bar.marker}SHAP: ${parseFloat(bar.value).toFixed(6)}<br/>` : '') +
            (line ? `${line.marker}누적 기여도: ${line.value}%` : '')
        },
      },
      legend: {
        show: true, top: 4, right: 8,
        data: ['개별 영향도', '누적 %'],
        textStyle: { fontSize: 12, color: '#475569' },
      },
      grid: { top: 48, bottom: 60, left: 50, right: 60, containLabel: true },
      xAxis: {
        type: 'category',
        data: rows.map(r => r.feature),
        axisLabel: { fontSize: 10, color: '#475569', rotate: 35, fontFamily: 'monospace' },
        axisTick: { alignWithLabel: true },
      },
      yAxis: [
        {
          type: 'value',
          name: 'SHAP', nameTextStyle: { fontSize: 11, color: '#94A3B8' },
          axisLabel: { fontSize: 10, color: '#94A3B8', formatter: v => v.toFixed(4) },
          splitLine: { lineStyle: { color: '#F1F5F9' } },
          max: maxShap * 1.1,
        },
        {
          type: 'value',
          axisLabel: { fontSize: 10, color: '#94A3B8', formatter: v => v + '%' },
          splitLine: { show: false },
          min: 0, max: 100,
        },
      ],
      series: [
        {
          name: '개별 영향도',
          type: 'bar',
          yAxisIndex: 0,
          data: rows.map(r => r.shap),
          itemStyle: { color: '#3B82F6', borderRadius: [4, 4, 0, 0] },
          barMaxWidth: 28,
        },
        {
          name: '누적 %',
          type: 'line',
          yAxisIndex: 1,
          data: rows.map(r => r.cumPct),
          smooth: false,
          symbol: 'circle', symbolSize: 7,
          lineStyle: { color: '#EF4444', width: 2.5 },
          itemStyle: { color: '#EF4444' },
          markLine: {
            silent: true, symbol: 'none',
            data: [{ yAxis: 80, name: '80% 기준선' }],
            lineStyle: { color: '#94A3B8', type: 'dashed', width: 1.5 },
            label: {
              show: true, position: 'insideStartTop',
              formatter: '80% 기준선',
              fontSize: 10, color: '#94A3B8',
            },
          },
        },
      ],
    }
  }, [paretoData])

  // 피처별 위험 임계값 카드 데이터 (Top N)
  const thresholdCards = useMemo(() => {
    if (!shapSorted.length || !featDistRaw.length || riskThreshold == null || safeThreshold == null) return []
    const cols = Object.keys(featDistRaw[0] || {})
      .filter(k => !['ufs_serial', 'health', 'is_defect'].includes(k))

    const cards = []
    for (const s of shapSorted) {
      if (cards.length >= TOP_N) break
      const feat = s.feature
      if (!cols.includes(feat)) continue

      const lowVals = []
      const highVals = []
      // 극단군 비교: 위험=상위10%(≥P90) vs 안전=하위10%(≤P10), 중간 80%는 제외
      for (const r of featDistRaw) {
        const x = parseFloat(r[feat])
        if (!isFinite(x)) continue
        const rp = regPredMap[r.ufs_serial]
        if (rp == null) continue
        if (rp >= riskThreshold) highVals.push(x)
        else if (rp <= safeThreshold) lowVals.push(x)
      }
      if (lowVals.length < 10 || highVals.length < 10) continue

      const lowSorted = [...lowVals].sort((a, b) => a - b)
      const highSorted = [...highVals].sort((a, b) => a - b)

      const lowQ1 = quantile(lowSorted, 0.25)
      const lowQ3 = quantile(lowSorted, 0.75)
      const highMed = quantile(highSorted, 0.5)

      // 위험 그룹의 중앙값 방향 결정
      const direction = highMed > quantile(lowSorted, 0.5) ? 'up' : 'down'

      // 전체 unit의 (값, 위험여부) 수집
      const pts = []
      let totalHighAll = 0
      for (const r of featDistRaw) {
        const x = parseFloat(r[feat])
        if (!isFinite(x)) continue
        const rp = regPredMap[r.ufs_serial]
        if (rp == null) continue
        // 극단군(위험 상위10% + 안전 하위10%)만 사용, 중간 80% 제외
        let isHigh
        if (rp >= riskThreshold) isHigh = true
        else if (rp <= safeThreshold) isHigh = false
        else continue
        if (isHigh) totalHighAll++
        pts.push({ x, isHigh })
      }
      const total = pts.length
      const baseRate = total ? totalHighAll / total : 0

      // 임계값 = "위험(빨강) 밀도가 안전(파랑) 밀도를 추월하는 교차점"
      // 분포 차트(distOption)와 동일한 binning(P1~P99, 40구간)으로 밀도 계산 → 차트 선과 정렬
      const allSorted = pts.map(p => p.x).sort((a, b) => a - b)
      const minV = quantile(allSorted, 0.01)
      const maxV = quantile(allSorted, 0.99)
      const NB = 40
      const bw = (maxV - minV) / NB || 1
      const centerX = i => minV + (i + 0.5) * bw
      const density = vals => {
        const arr = new Array(NB).fill(0)
        for (const v of vals) {
          let idx = Math.floor((v - minV) / bw)
          if (idx < 0) idx = 0
          if (idx >= NB) idx = NB - 1
          arr[idx]++
        }
        const n = vals.length || 1
        return arr.map(c => c / n)   // 그룹 내 비율
      }
      const dHigh = density(highVals)
      const dLow = density(lowVals)
      const loMed = quantile(lowSorted, 0.5)
      const lo = Math.min(loMed, highMed), hi = Math.max(loMed, highMed)

      // 두 그룹 중앙값 사이에서 밀도 부호 전환(교차) 지점 탐색
      let threshold = direction === 'up' ? highMed : loMed
      let found = false
      for (let i = 1; i < NB; i++) {
        const xc = centerX(i)
        if (xc < lo || xc > hi) continue
        const prev = dHigh[i - 1] - dLow[i - 1]
        const cur = dHigh[i] - dLow[i]
        if (direction === 'up') {
          if (prev < 0 && cur >= 0) { threshold = xc; found = true; break }  // 파랑→빨강 우세 전환
        } else {
          if (prev >= 0 && cur < 0) { threshold = xc; found = true; break }  // 빨강→파랑 우세 전환
        }
      }
      if (!found) threshold = (loMed + highMed) / 2   // 교차점 없으면 두 중앙값 중간

      // 선택된 임계값 기준 지표(초과 위험률·lift) 계산
      let overThreshold = 0, overHigh = 0
      for (const p of pts) {
        const cond = direction === 'up' ? p.x >= threshold : p.x <= threshold
        if (cond) { overThreshold++; if (p.isHigh) overHigh++ }
      }
      const condRate = overThreshold ? overHigh / overThreshold : 0
      const liftRatio = baseRate ? condRate / baseRate : 0

      cards.push({
        feature: feat,
        shap: parseFloat(s.mean_abs_shap),
        direction,
        normalLow: lowQ1,
        normalHigh: lowQ3,
        threshold,
        condRate,
        baseRate,
        liftRatio,
        nOver: overThreshold,
        nTotal: total,
      })
    }
    return cards
  }, [shapSorted, featDistRaw, regPredMap, riskThreshold, safeThreshold])

  // 피처 분포 비교 차트 (선택 피처)
  const activeFeat = useMemo(() => {
    if (!selFeat) return thresholdCards[0]?.feature ?? null
    return selFeat
  }, [selFeat, thresholdCards])

  const distOption = useMemo(() => {
    if (!featDistRaw.length || !activeFeat || riskThreshold == null || safeThreshold == null) return null
    const highVals = [], lowVals = []
    for (const r of featDistRaw) {
      const x = parseFloat(r[activeFeat])
      if (!isFinite(x)) continue
      const rp = regPredMap[r.ufs_serial]   // 극단군 비교: 위험=상위10%(≥P90), 안전=하위10%(≤P10)
      if (rp == null) continue
      if (rp >= riskThreshold) highVals.push(x)
      else if (rp <= safeThreshold) lowVals.push(x)
    }
    if (!highVals.length && !lowVals.length) return null

    // x축 범위를 P1~P99로 클립 (극단값으로 본체가 압축되는 것 방지)
    // 범위 밖 값은 양 끝 bin에 모아서 표시
    const allSorted = [...highVals, ...lowVals].sort((a, b) => a - b)
    const minV = quantile(allSorted, 0.01)
    const maxV = quantile(allSorted, 0.99)
    const BIN = 40
    const binSz = (maxV - minV) / BIN || 1
    const bins = Array.from({ length: BIN }, (_, i) => minV + i * binSz)

    const mkDensity = vals => bins.map((b, i) => {
      const lo = i === 0 ? -Infinity : b                  // 첫 칸: 하한 밖 값 흡수
      const hi = i === BIN - 1 ? Infinity : bins[i + 1]   // 끝 칸: 상한 밖 값 흡수
      const cnt = vals.filter(v => v >= lo && v < hi).length
      return vals.length ? +(cnt / vals.length * 100).toFixed(2) : 0
    })

    const xLabels = bins.map(b => fmt(b))

    // 위험 임계값(교차점) — activeFeat에 대해 직접 계산해 모든 피처에 선 표시
    let threshold = null
    if (highVals.length >= 10 && lowVals.length >= 10) {
      const hiS = [...highVals].sort((a, b) => a - b)
      const loS = [...lowVals].sort((a, b) => a - b)
      const hiMed = quantile(hiS, 0.5), loMed = quantile(loS, 0.5)
      const direction = hiMed > loMed ? 'up' : 'down'
      const dHigh = mkDensity(highVals), dLow = mkDensity(lowVals)
      const c0 = Math.min(loMed, hiMed), c1 = Math.max(loMed, hiMed)
      threshold = direction === 'up' ? hiMed : loMed
      for (let i = 1; i < BIN; i++) {
        const xc = bins[i] + binSz / 2
        if (xc < c0 || xc > c1) continue
        const prev = dHigh[i - 1] - dLow[i - 1], cur = dHigh[i] - dLow[i]
        if (direction === 'up' && prev < 0 && cur >= 0) { threshold = xc; break }
        if (direction === 'down' && prev >= 0 && cur < 0) { threshold = xc; break }
      }
    }
    const thrIdx = threshold != null ? bins.findIndex((b, i) => threshold < (i === BIN - 1 ? Infinity : bins[i + 1])) : -1

    return {
      tooltip: {
        trigger: 'axis',
        formatter: p => {
          const lines = p
            .filter(s => s.seriesType !== 'effectScatter')
            .map(s => `${s.marker}${s.seriesName}: ${s.value}%`)
          return lines.join('<br/>') + `<br/><span style="color:#94A3B8;font-size:11px">${activeFeat} ≈ ${p[0]?.axisValue ?? ''}</span>`
        },
      },
      legend: {
        show: true, top: 4, right: 8,
        data: [
          { name: '안전(하위10%)', icon: 'rect', itemStyle: { color: '#3B82F6' } },
          { name: '위험(상위10%)', icon: 'rect', itemStyle: { color: '#EF4444' } },
        ],
        textStyle: { fontSize: 12, color: '#475569' },
      },
      grid: { top: 36, bottom: 40, left: 52, right: 20 },
      xAxis: {
        type: 'category',
        data: xLabels,
        axisLabel: { fontSize: 10, color: '#94A3B8', interval: 4, rotate: 30 },
        axisTick: { show: true, alignWithLabel: true, interval: 0 },
        boundaryGap: false,
      },
      yAxis: {
        type: 'value',
        axisLabel: { fontSize: 10, color: '#94A3B8', formatter: v => v + '%' },
        splitLine: { lineStyle: { color: '#F1F5F9' } },
      },
      series: [
        {
          name: '안전(하위10%)', type: 'line', data: mkDensity(lowVals),
          smooth: true, symbol: 'none',
          lineStyle: { color: '#3B82F6', width: 2 },
          areaStyle: { color: 'rgba(59,130,246,0.12)' },
        },
        {
          name: '위험(상위10%)', type: 'line', data: mkDensity(highVals),
          smooth: true, symbol: 'none',
          lineStyle: { color: '#EF4444', width: 2 },
          areaStyle: { color: 'rgba(239,68,68,0.12)' },
          markLine: thrIdx >= 0 ? {
            silent: true, symbol: 'none',
            data: [{ xAxis: thrIdx, name: '임계값' }],
            lineStyle: { color: '#DC2626', type: 'solid', width: 2 },
            label: {
              show: true, position: 'insideEndBottom',
              distance: [0, 6],
              formatter: `임계값 ${fmt(threshold)}`,
              fontSize: 11, color: '#DC2626', fontWeight: 600,
            },
          } : undefined,
        },
      ],
    }
  }, [featDistRaw, activeFeat, regPredMap, riskThreshold, safeThreshold])

  // SHAP Beeswarm (참고용 작게)
  const normToColor = norm => {
    const t = Math.max(0, Math.min(1, norm))
    const r = Math.round(59 + (239 - 59) * t)
    const g = Math.round(130 + (68 - 130) * t)
    const b = Math.round(246 + (68 - 246) * t)
    return `rgb(${r},${g},${b})`
  }

  const beeswarmOption = useMemo(() => {
    if (!shapBeeswarmRaw.length || !beeswarmFeats.length) return null
    const featOrder = [...beeswarmFeats].reverse()
    const featIdx = Object.fromEntries(featOrder.map((f, i) => [f, i]))
    const SAMPLE = 6000
    const rows = shapBeeswarmRaw.filter(r => beeswarmFeats.includes(r.feature))
    const step = rows.length > SAMPLE ? Math.ceil(rows.length / SAMPLE) : 1
    const points = rows.filter((_, i) => i % step === 0).map(r => ({
      feature: r.feature,
      sv: parseFloat(r.shap_value),
      yi: featIdx[r.feature] + (Math.random() - 0.5) * 0.7,
      norm: parseFloat(r.feat_norm),
    }))

    const scatterData = points.map(d => ({
      value: [d.sv, d.yi],
      itemStyle: {
        color: (selFeat && d.feature !== selFeat) ? '#CBD5E1' : normToColor(isFinite(d.norm) ? d.norm : 0.5),
        opacity: (selFeat && d.feature !== selFeat) ? 0.2 : 0.7,
      },
    }))

    const xVals = points.map(d => d.sv)
    const xBound = Math.max(Math.abs(Math.min(...xVals)), Math.abs(Math.max(...xVals))) * 1.1 || 0.001

    return {
      tooltip: {
        trigger: 'item',
        formatter: p => {
          const sv = p.data.value[0]
          const feat = featOrder[Math.round(p.data.value[1])] ?? ''
          return `<b>${feat}</b><br/>SHAP: ${sv >= 0 ? '+' : ''}${sv.toFixed(6)}<br/>${sv >= 0 ? '▲ 불량 증가 방향' : '▼ 불량 감소 방향'}`
        },
      },
      grid: { top: 8, bottom: 32, left: 22, right: 80, containLabel: true },
      dataZoom: [
        // 20개 중 상위 10개(위쪽 50%)만 보이게 — 휠/슬라이더로 나머지 스크롤, 창 크기 50% 고정
        { type: 'inside', yAxisIndex: 0, start: 50, end: 100, minSpan: 50, maxSpan: 50, zoomOnMouseWheel: false, moveOnMouseWheel: true, moveOnMouseMove: true },
        { type: 'slider', yAxisIndex: 0, start: 50, end: 100, minSpan: 50, maxSpan: 50, left: 2, width: 12, handleSize: 0, showDetail: false, brushSelect: false, fillerColor: 'rgba(59,130,246,0.18)', borderColor: 'transparent', backgroundColor: '#F1F5F9' },
      ],
      xAxis: {
        type: 'value', min: -xBound, max: xBound,
        axisLabel: { fontSize: 10, color: '#94A3B8', formatter: v => v.toFixed(4) },
        splitLine: { lineStyle: { color: '#F1F5F9' } },
        axisLine: { lineStyle: { color: '#E2E8F0' } },
        name: 'SHAP value (불량 기여도)',
        nameLocation: 'center', nameGap: 24,
        nameTextStyle: { fontSize: 11, color: '#94A3B8' },
      },
      yAxis: {
        type: 'value',
        min: -0.5, max: featOrder.length - 0.5,
        interval: 1,
        axisLabel: {
          fontSize: 11, fontFamily: 'monospace',
          formatter: v => featOrder[Math.round(v)] ?? '',
          color: '#374151',
        },
        axisTick: { show: false },
        splitLine: { lineStyle: { color: '#F8FAFC', type: 'dashed' } },
      },
      series: [{
        type: 'scatter',
        data: scatterData,
        symbolSize: 5,
        markLine: {
          silent: true, symbol: 'none',
          data: [{ xAxis: 0 }],
          lineStyle: { color: '#64748B', width: 2 },
          label: { show: false },
        },
      }],
      graphic: [{
        type: 'group', right: 8, top: '15%',
        children: [
          { type: 'text', style: { text: 'High', fontSize: 10, fill: '#EF4444', fontWeight: 700 }, left: 2, top: 0 },
          {
            type: 'rect', left: 0, top: 16,
            shape: { width: 12, height: 110 },
            style: {
              fill: {
                type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
                colorStops: [
                  { offset: 0, color: 'rgb(239,68,68)' },
                  { offset: 1, color: 'rgb(59,130,246)' },
                ],
              },
            },
          },
          { type: 'text', style: { text: 'Low', fontSize: 10, fill: '#3B82F6', fontWeight: 700 }, left: 2, top: 130 },
          { type: 'text', style: { text: 'Feat\nvalue', fontSize: 9, fill: '#94A3B8' }, left: -2, top: 150 },
        ],
      }],
    }
  }, [shapBeeswarmRaw, beeswarmFeats, selFeat])

  if (!unitsRaw.length || !featDistRaw.length) {
    return (
      <div className="pf-page" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#94A3B8' }}>
        데이터 로딩 중…
      </div>
    )
  }

  return (
    <div className="pf-page">

      {/* ── 안내 배너 ── */}
      <div className="pf-banner">
        <div className="pf-banner-title">공정 인자 진단</div>
        {bestVal != null && (
          <div className="pf-banner-rmse">
            <span className="pf-banner-rmse-label">RMSE</span>
            <span className="pf-banner-rmse-val">{bestVal.toFixed(6)}</span>
          </div>
        )}
      </div>

      {/* ── Row 0: Feature Pareto (영향력 누적 기여도) ── */}
      <div className="pf-section-title">
        <span>공정 인자 영향도 — Pareto</span>
        <span className="pf-section-sub">
          {paretoData
            ? `상위 ${paretoData.top80Count}개 인자가 전체 영향력의 ${paretoData.top80Pct}%를 설명합니다.`
            : 'SHAP 데이터 기준 상위 20개 인자의 개별 영향도(막대)와 누적 기여도(선)'}
        </span>
      </div>
      <div className="pf-chart-card">
        <div className="pf-cc-body">
          {paretoOption
            ? <ReactECharts option={paretoOption} style={{ height: 320 }} />
            : <div className="pf-empty">SHAP 데이터 없음</div>}
        </div>
      </div>

      {/* ── Row 1: 위험 임계값 카드 Top 5 ── */}
      <div className="pf-section-title">
        <span>관리 대상 공정 인자 Top {TOP_N}</span>
        <span className="pf-section-sub">SHAP 기준 영향력 순. 카드 클릭 시 아래 분포가 갱신됩니다.</span>
      </div>

      {/* 계산 기준 안내 */}
      <div className="pf-criteria">
        <div className="pf-criteria-item">
          <span className="pf-criteria-key">위험 임계값</span>
          <span className="pf-criteria-desc">
위험(빨강) 분포가 안전(파랑) 분포를 추월하는 교차점 — 비교군(위험 상위10% vs 안전 하위10%) 기준
          </span>
        </div>
        <div className="pf-criteria-item">
          <span className="pf-criteria-key">임계값 초과 위험률</span>
          <span className="pf-criteria-desc">
            비교군 중 임계값을 넘은 unit에서 위험군(상위10%)이 차지하는 비율
          </span>
        </div>
        <div className="pf-criteria-item">
          <span className="pf-criteria-key">평균 대비 배수</span>
          <span className="pf-criteria-desc">
            임계값 초과 위험률 ÷ 비교군 평균 위험률 — <b>임계값을 넘으면 평균보다 몇 배 더 위험한가</b>
          </span>
        </div>
      </div>

      <div className="pf-card-row">
        {thresholdCards.length === 0
          ? <div className="pf-empty">임계값 계산 가능한 피처가 없습니다.</div>
          : thresholdCards.map((c) => (
            <div
              key={c.feature}
              className={`pf-thr-card ${activeFeat === c.feature ? 'active' : ''}`}
              onClick={() => setSelFeat(c.feature)}
            >
              <div className="pf-thr-head">
                <span className="pf-thr-feat">{c.feature}</span>
                <span className={`pf-thr-arrow ${c.direction}`}>
                  {c.direction === 'up' ? '↑ 높을수록 위험' : '↓ 낮을수록 위험'}
                </span>
              </div>
              <div className="pf-thr-range">
                <div className="pf-thr-line">
                  <span className="pf-thr-label">안전 범위</span>
                  <span className="pf-thr-val">{fmt(c.normalLow)} ~ {fmt(c.normalHigh)}</span>
                </div>
                <div className="pf-thr-line emphasis">
                  <span className="pf-thr-label">위험 임계값</span>
                  <span className="pf-thr-val danger">
                    {c.direction === 'up' ? '≥ ' : '≤ '}{fmt(c.threshold)}
                  </span>
                </div>
              </div>
              <div className="pf-thr-foot">
                <div className="pf-thr-stat">
                  <div className="pf-thr-stat-val">{(c.condRate * 100).toFixed(1)}%</div>
                  <div className="pf-thr-stat-lbl">임계값 초과 위험률</div>
                </div>
                <div className="pf-thr-stat">
                  <div className="pf-thr-stat-val lift">×{c.liftRatio.toFixed(2)}</div>
                  <div className="pf-thr-stat-lbl">평균 대비 배수</div>
                </div>
              </div>
            </div>
          ))}
      </div>

      {/* ── Row 2: 정상위험분포 / 피처중요도 / SHAP 영향도 ── */}
      <div className="pf-section-title">
        <span>피처 영향도 — 분포 · 중요도 · SHAP</span>
        <span className="pf-sort-toggle">
          <span className="pf-sort-toggle-label">정렬 기준</span>
          <button className={sortBy === 'default' ? 'active' : ''} onClick={() => setSortBy('default')}>기본</button>
          <button className={sortBy === 'shap' ? 'active' : ''} onClick={() => setSortBy('shap')}>SHAP</button>
          <button className={sortBy === 'fi' ? 'active' : ''} onClick={() => setSortBy('fi')}>피처임포턴스</button>
        </span>
      </div>
      <div className="pf-grid-3">
        <div className="pf-chart-card">
          <div className="pf-cc-header">
            <span className="pf-cc-title">안전 vs 위험 분포 — {activeFeat ?? '-'}</span>
            <span className="pf-cc-sub">위험(상위10%) vs 안전(하위10%) 분포 비교 · 빨간 선 = 위험 임계값</span>
          </div>
          <div className="pf-cc-body">
            {distOption
              ? <ReactECharts option={distOption} style={{ height: 340 }} />
              : <div className="pf-empty">분포 데이터 없음</div>}
          </div>
        </div>

        <div className="pf-chart-card">
          <div className="pf-cc-header">
            <span className="pf-cc-title">피처 임포턴스</span>
            <span className="pf-cc-sub">막대=LGBM Gain% · {sortBy === 'shap' ? 'SHAP' : '임포턴스'} 순 정렬 · 막대 클릭 시 분포 갱신</span>
          </div>
          <div className="pf-cc-body">
            {fiOption
              ? <ReactECharts
                  option={fiOption}
                  style={{ height: 340 }}
                  onEvents={{ click: p => { if (p.name) setSelFeat(f => f === p.name ? null : p.name) } }}
                />
              : <div className="pf-empty">feature_importance.csv 없음</div>}
          </div>
        </div>

        <div className="pf-chart-card">
          <div className="pf-cc-header">
            <span className="pf-cc-title">SHAP 영향도 (전역)</span>
            <span className="pf-cc-sub">색상 = 피처 값 (빨강↑ / 파랑↓) · {sortBy === 'fi' ? '임포턴스' : 'SHAP'} 순 정렬</span>
          </div>
          <div className="pf-cc-body">
            {beeswarmOption
              ? <ReactECharts
                  option={beeswarmOption}
                  style={{ height: 360 }}
                  onEvents={{
                    click: p => {
                      if (p.componentType === 'series') {
                        const feat = [...beeswarmFeats].reverse()[Math.round(p.data.value[1])]
                        if (feat) setSelFeat(feat)
                      }
                    },
                  }}
                />
              : <div className="pf-empty">SHAP 데이터 없음</div>}
          </div>
        </div>
      </div>

    </div>
  )
}
