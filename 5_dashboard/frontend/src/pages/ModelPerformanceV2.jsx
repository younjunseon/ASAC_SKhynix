import { useMemo, useState, useEffect, useRef } from 'react'
import ReactECharts from 'echarts-for-react'
import { useCSV } from '../hooks/useCSV'
import './ModelPerformanceV2.css'

const BASELINE_RMSE = 0.005845
const FIXED_DATE    = '2026-06-11'

function ChartCard({ title, tag, controls, children, scrollable }) {
  return (
    <div className="mp-chart-card">
      <div className="mp-cc-header">
        <span className="mp-cc-title">{title}</span>
        {tag && <span className="mp-cc-tag">{tag}</span>}
        {controls && <div className="mp-cc-controls">{controls}</div>}
      </div>
      <div className={scrollable ? 'mp-cc-scroll' : 'mp-cc-body'}>
        {children}
      </div>
    </div>
  )
}

const chartH = (count) => Math.max(100, count * 28)

function KpiBox({ label, value, sub }) {
  return (
    <div className="mp-kpi">
      <div className="mp-kpi-val">{value}</div>
      <div className="mp-kpi-label">{label}</div>
      {sub && <div className="mp-kpi-sub">{sub}</div>}
    </div>
  )
}

export default function ModelPerformanceV2() {
  const { data: metricsRaw }     = useCSV('/metrics.csv')
  const { data: fiRaw }          = useCSV('/feature_importance.csv')
  const { data: shapBarRaw }     = useCSV('/shap_bar.csv')
  const { data: shapBeeswarmRaw }= useCSV('/shap_beeswarm.csv')
  const { data: unitsRaw }       = useCSV('/dashboard_units.csv')

  const [selFeat, setSelFeat] = useState(null)

  // ── 예측 신뢰도 계산 (train/val/test 전체 사용) ──
  const reliability = useMemo(() => {
    if (!unitsRaw.length) return null
    const eval_units = unitsRaw.filter(u => {
      const h = parseFloat(u.health)
      const p = parseFloat(u.reg_pred)
      return isFinite(h) && isFinite(p)
    })
    if (!eval_units.length) return null

    // 위험/정상 기준: grade3·4 = 위험, grade1·2 = 정상
    let TP = 0, FP = 0, TN = 0, FN = 0
    let errSum = 0, sqErrSum = 0
    const scatterPts = []
    const ppmConv = 1e6

    eval_units.forEach(u => {
      const h = parseFloat(u.health)
      const p = parseFloat(u.reg_pred)
      const actualHigh = h > 0           // 실측 위험 = health > 0
      const predGrade = u.grade
      const predHigh = predGrade === 'grade3' || predGrade === 'grade4'

      if (actualHigh && predHigh)  TP++
      if (!actualHigh && predHigh) FP++
      if (!actualHigh && !predHigh) TN++
      if (actualHigh && !predHigh) FN++

      const err = (p - h) * ppmConv
      errSum += Math.abs(err)
      sqErrSum += err * err

      // 산점도는 최대 2000건 샘플링
      scatterPts.push({ actual: h * ppmConv, pred: p * ppmConv, grade: predGrade })
    })

    const N = eval_units.length
    const recall    = (TP + FN) ? TP / (TP + FN) : 0   // 적중률 (실제 위험 중 잡은 비율)
    const fpr       = (FP + TN) ? FP / (FP + TN) : 0   // 오탐률 (정상 중 위험으로 오인)
    const fnr       = (TP + FN) ? FN / (TP + FN) : 0   // 미탐률 (실제 위험 중 놓침)
    const precision = (TP + FP) ? TP / (TP + FP) : 0   // 정밀도
    const mae_ppm   = errSum / N
    const rmse_ppm  = Math.sqrt(sqErrSum / N)

    // 산점도 샘플링 (최대 2000건)
    const SAMPLE = 2000
    const step = scatterPts.length > SAMPLE ? Math.ceil(scatterPts.length / SAMPLE) : 1
    const sampled = scatterPts.filter((_, i) => i % step === 0)

    return {
      N, TP, FP, TN, FN,
      recall, fpr, fnr, precision,
      mae_ppm, rmse_ppm,
      scatterPts: sampled,
    }
  }, [unitsRaw])

  const scatterOption = useMemo(() => {
    if (!reliability) return null
    const pts = reliability.scatterPts
    const maxV = Math.max(...pts.map(p => Math.max(p.actual, p.pred)), 1)

    const gradeColor = {
      grade1: '#22C55E', grade2: '#EAB308',
      grade3: '#F59E0B', grade4: '#EF4444',
    }
    const points = pts.map(p => ({
      value: [p.actual, p.pred],
      itemStyle: { color: gradeColor[p.grade] ?? '#94A3B8', opacity: 0.55 },
    }))

    return {
      tooltip: {
        trigger: 'item',
        formatter: p => `실측: ${Math.round(p.data.value[0]).toLocaleString()} ppm<br/>예측: ${Math.round(p.data.value[1]).toLocaleString()} ppm<br/><span style="color:#94A3B8;font-size:11px">대각선 위 = 과대예측, 아래 = 과소예측</span>`,
      },
      grid: { top: 16, bottom: 48, left: 60, right: 16, containLabel: true },
      xAxis: {
        type: 'value',
        name: '실측 (ppm)',
        nameLocation: 'center', nameGap: 32,
        nameTextStyle: { fontSize: 12, color: '#475569' },
        axisLabel: { fontSize: 10, color: '#94A3B8', formatter: v => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v },
        splitLine: { lineStyle: { color: '#F1F5F9' } },
        min: 0, max: maxV * 1.05,
      },
      yAxis: {
        type: 'value',
        name: '예측 (ppm)',
        nameLocation: 'center', nameGap: 44,
        nameTextStyle: { fontSize: 12, color: '#475569' },
        axisLabel: { fontSize: 10, color: '#94A3B8', formatter: v => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v },
        splitLine: { lineStyle: { color: '#F1F5F9' } },
        min: 0, max: maxV * 1.05,
      },
      series: [
        {
          type: 'scatter',
          data: points,
          symbolSize: 6,
          markLine: {
            silent: true, symbol: 'none',
            data: [[{ coord: [0, 0] }, { coord: [maxV, maxV] }]],
            lineStyle: { color: '#64748B', type: 'dashed', width: 1.5 },
            label: { show: false },
          },
        },
      ],
    }
  }, [reliability])

  const confusionOption = useMemo(() => {
    if (!reliability) return null
    const { TP, FP, TN, FN } = reliability
    // 2x2 heatmap: 행=실측, 열=예측
    // data: [colIdx, rowIdx, value]
    const data = [
      [0, 1, TN, '정상→정상 (적중)'],
      [1, 1, FP, '정상→위험 (오탐)'],
      [0, 0, FN, '위험→정상 (미탐)'],
      [1, 0, TP, '위험→위험 (적중)'],
    ]
    const max = Math.max(TP, FP, TN, FN, 1)
    return {
      tooltip: {
        formatter: p => `<b>${p.data[3]}</b><br/>건수: ${p.data[2].toLocaleString()}건`,
      },
      grid: { top: 36, bottom: 56, left: 80, right: 16, containLabel: false },
      xAxis: {
        type: 'category',
        data: ['예측: 정상', '예측: 위험'],
        position: 'top',
        axisLine: { show: false }, axisTick: { show: false },
        axisLabel: { fontSize: 12, color: '#475569', fontWeight: 600 },
      },
      yAxis: {
        type: 'category',
        data: ['실측: 위험', '실측: 정상'],
        axisLine: { show: false }, axisTick: { show: false },
        axisLabel: { fontSize: 12, color: '#475569', fontWeight: 600 },
      },
      visualMap: { show: false, min: 0, max, inRange: { color: ['#F8FAFC', '#3B82F6', '#1E3A5F'] } },
      series: [{
        type: 'heatmap',
        data,
        label: {
          show: true,
          formatter: p => `${p.data[2].toLocaleString()}건`,
          fontSize: 16, fontWeight: 700,
          color: p => p.data[2] / max > 0.5 ? '#fff' : '#1E293B',
        },
        itemStyle: { borderColor: '#fff', borderWidth: 2 },
      }],
    }
  }, [reliability])

  const metrics = useMemo(() => {
    if (!metricsRaw.length) return null
    const get = (stage, model, split, metric = 'rmse') => {
      const row = metricsRaw.find(r =>
        r.stage === stage && r.model === model && r.split === split && r.metric === metric)
      return row ? parseFloat(row.value) : null
    }
    return {
      ensemble_val: get('reg', 'ensemble', 'val'),
      stacking_val: get('reg', 'stacking', 'val'),
      lgbm_val:     get('reg', 'lgbm',     'val'),
    }
  }, [metricsRaw])

  const allFiSorted = useMemo(() => {
    if (!fiRaw.length) return []
    const totalGain = fiRaw.reduce((s, r) => s + parseFloat(r.lgbm_gain || 0), 0) || 1
    return [...fiRaw]
      .sort((a, b) => parseFloat(b.lgbm_gain) - parseFloat(a.lgbm_gain))
      .map(r => ({
        ...r,
        gain_pct: parseFloat(r.lgbm_gain || 0) / totalGain * 100,
      }))
  }, [fiRaw])

  const allShapSorted = useMemo(() => {
    if (!shapBarRaw.length) return []
    return [...shapBarRaw].sort((a, b) => parseFloat(b.mean_abs_shap) - parseFloat(a.mean_abs_shap))
  }, [shapBarRaw])

  const fiDisplayItems = useMemo(() => {
    if (!allFiSorted.length) return []
    return allFiSorted.slice(0, 20)
  }, [allFiSorted])

  const shapDisplayItems = useMemo(() => {
    if (!fiDisplayItems.length || !allShapSorted.length) return []
    return fiDisplayItems
      .map(fi => allShapSorted.find(s => s.feature === fi.feature))
      .filter(Boolean)
  }, [fiDisplayItems, allShapSorted])

  const fiOption = useMemo(() => {
    if (!fiDisplayItems.length) return null
    const items = [...fiDisplayItems].reverse()
    return {
      tooltip: {
        trigger: 'axis', axisPointer: { type: 'shadow' },
        formatter: p => `<b>${p[0].name}</b><br/>중요도: ${parseFloat(p[0].value).toFixed(2)}%`,
      },
      grid: { top: 8, bottom: 8, left: 8, right: 70, containLabel: true },
      xAxis: {
        type: 'value',
        axisLabel: { fontSize: 10, color: '#94A3B8', formatter: '{value}%' },
        splitLine: { lineStyle: { color: '#F1F5F9' } },
      },
      yAxis: {
        type: 'category',
        data: items.map(d => d.feature),
        axisLabel: { fontSize: 10, color: '#374151', fontFamily: 'monospace' },
        axisTick: { show: false },
      },
      series: [{
        type: 'bar',
        data: items.map(d => ({
          value: +d.gain_pct.toFixed(2),
          itemStyle: {
            color: d.feature === selFeat ? '#7C3AED' : '#3B82F6',
            borderRadius: [0, 4, 4, 0],
          },
        })),
        barMaxWidth: 14,
        label: {
          show: true, position: 'right', fontSize: 11, color: '#64748B',
          formatter: p => `${parseFloat(p.value).toFixed(2)}%`,
        },
      }],
    }
  }, [fiDisplayItems, selFeat])

  const onFiClick = (p) => { if (p.name && p.name !== '─────────') setSelFeat(f => f === p.name ? null : p.name) }

  const fiChartHeight = useMemo(() => chartH(fiDisplayItems.length), [fiDisplayItems])

  const beeswarmFeats = useMemo(() => shapDisplayItems.map(d => d.feature), [shapDisplayItems])

  const normToColor = norm => {
    const t = Math.max(0, Math.min(1, norm))
    const r = Math.round(59  + (239 - 59)  * t)
    const g = Math.round(130 + (68  - 130) * t)
    const b = Math.round(246 + (68  - 246) * t)
    return `rgb(${r},${g},${b})`
  }

  const beeswarmBaseData = useMemo(() => {
    if (!shapBeeswarmRaw.length || !beeswarmFeats.length) return []
    const featOrder = [...beeswarmFeats].reverse()
    const featIdx   = Object.fromEntries(featOrder.map((f, i) => [f, i]))
    const SAMPLE    = 8000
    const rows      = shapBeeswarmRaw.filter(r => beeswarmFeats.includes(r.feature))
    const step      = rows.length > SAMPLE ? Math.ceil(rows.length / SAMPLE) : 1
    return rows.filter((_, i) => i % step === 0).map(r => ({
      feature: r.feature,
      sv:      parseFloat(r.shap_value),
      yi:      featIdx[r.feature] + (Math.random() - 0.5) * 0.7,
      norm:    parseFloat(r.feat_norm),
    }))
  }, [shapBeeswarmRaw, beeswarmFeats])

  const beeswarmRef = useRef(null)
  useEffect(() => {
    const chart = beeswarmRef.current?.getEchartsInstance?.()
    if (!chart || !beeswarmBaseData.length) return
    const patch = {
      series: [{
        data: beeswarmBaseData.map(d => ({
          value: [d.sv, d.yi],
          itemStyle: {
            color: (selFeat && d.feature !== selFeat) ? '#CBD5E1' : normToColor(isFinite(d.norm) ? d.norm : 0.5),
            opacity: (selFeat && d.feature !== selFeat) ? 0.2 : 0.7,
          },
          symbolSize: (selFeat && d.feature === selFeat) ? 7 : 5,
        })),
      }],
    }
    chart.setOption(patch, false)
  }, [selFeat, beeswarmBaseData])

  const shapBeeswarmOption = useMemo(() => {
    if (!beeswarmBaseData.length || !beeswarmFeats.length) return null

    const featOrder = [...beeswarmFeats].reverse()

    const scatterData = beeswarmBaseData.map(d => ({
      value: [d.sv, d.yi],
      itemStyle: {
        color: normToColor(isFinite(d.norm) ? d.norm : 0.5),
        opacity: 0.7,
      },
    }))

    const xVals  = beeswarmBaseData.map(d => d.sv)
    const xBound = Math.max(Math.abs(Math.min(...xVals)), Math.abs(Math.max(...xVals))) * 1.1 || 0.001

    return {
      tooltip: {
        trigger: 'item',
        formatter: p => {
          const sv   = p.data.value[0]
          const feat = featOrder[Math.round(p.data.value[1])] ?? ''
          return `<b>${feat}</b><br/>SHAP: ${sv >= 0 ? '+' : ''}${sv.toFixed(6)}<br/>${sv >= 0 ? '▲ 불량 증가 방향' : '▼ 불량 감소 방향'}`
        },
      },
      grid: { top: 8, bottom: 28, left: 8, right: 80, containLabel: true },
      xAxis: {
        type: 'value', min: -xBound, max: xBound,
        axisLabel: { fontSize: 10, color: '#94A3B8', formatter: v => v.toFixed(4) },
        splitLine: { lineStyle: { color: '#F1F5F9' } },
        axisLine: { lineStyle: { color: '#E2E8F0' } },
        name: 'SHAP value (impact on model output)',
        nameLocation: 'center', nameGap: 22,
        nameTextStyle: { fontSize: 10, color: '#94A3B8' },
      },
      yAxis: {
        type: 'value',
        min: -0.5, max: featOrder.length - 0.5,
        interval: 1,
        axisLabel: {
          fontSize: 10, fontFamily: 'monospace',
          formatter: v => featOrder[Math.round(v)] ?? '',
          color: '#374151',
        },
        axisTick: { show: false },
        splitLine: { lineStyle: { color: '#F8FAFC', type: 'dashed' } },
      },
      series: [{
        type: 'scatter',
        data: scatterData,
        symbolSize: d => d.symbolSize ?? 5,
        large: false,
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
            shape: { width: 12, height: 120 },
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
          { type: 'text', style: { text: 'Low', fontSize: 10, fill: '#3B82F6', fontWeight: 700 }, left: 2, top: 140 },
          { type: 'text', style: { text: 'Feature\nvalue', fontSize: 9, fill: '#94A3B8' }, left: -2, top: 158 },
        ],
      }],
    }
  }, [beeswarmBaseData, beeswarmFeats])

  if (!metrics) {
    return (
      <div className="mp-page" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#94A3B8' }}>
        데이터 로딩 중…
      </div>
    )
  }

  const bestVal     = metrics.stacking_val ?? metrics.ensemble_val ?? metrics.lgbm_val ?? 0
  const beatBase    = bestVal < BASELINE_RMSE
  const improvement = (((BASELINE_RMSE - bestVal) / BASELINE_RMSE) * 100).toFixed(1)

  const pct = (v) => `${(v * 100).toFixed(1)}%`

  return (
    <div className="mp-page">

      {/* ══════════════════════════════════════════════════════ */}
      {/* ── 🆕 품질팀용 — 예측 신뢰도 섹션 (상단)              */}
      {/* ══════════════════════════════════════════════════════ */}


      {/* ══════════════════════════════════════════════════════ */}
      {/* ── DS팀용 — 기존 모델 분석 섹션                       */}
      {/* ══════════════════════════════════════════════════════ */}

      {/* ── Row 1: RMSE KPI 2개 ── */}
      <div className="mp-kpi-row mp-kpi-row--2">
        <KpiBox
          label="앙상블 모델 검증 오차(RMSE)"
          value={bestVal.toFixed(6)}
          sub={`예측 시점 · ${FIXED_DATE}`}
        />
        <KpiBox
          label={beatBase ? '사내경진대회 대비 개선' : '사내경진대회 대비 미달'}
          value={beatBase ? `-${improvement}%` : `+${Math.abs(parseFloat(improvement))}%`}
          sub={`경진대회 기준 ${BASELINE_RMSE} · ${beatBase ? '목표 달성 ✓' : '미달성'}`}
        />
      </div>

      {/* ── FI + SHAP 합친 카드 ── */}
      <ChartCard
        title="피처 중요도 · SHAP 분석"
        scrollable
      >
        <div className="mp-fi-shap-row">
          <div className="mp-fi-shap-col">
            <div className="mp-fi-shap-label">피처 중요도 (Gain)</div>
            {fiOption
              ? <ReactECharts
                  option={fiOption}
                  style={{ height: fiChartHeight }}
                  onEvents={{ click: onFiClick }}
                />
              : <div className="mp-empty">feature_importance.csv 없음</div>}
          </div>
          <div className="mp-fi-shap-divider" />
          <div className="mp-fi-shap-col">
            <div className="mp-fi-shap-label">SHAP 분석 (Beeswarm)</div>
            {shapBeeswarmOption
              ? <ReactECharts
                  ref={beeswarmRef}
                  option={shapBeeswarmOption}
                  style={{ height: Math.max(300, beeswarmFeats.length * 28) }}
                  notMerge={false}
                  onEvents={{ click: p => { if (p.componentType === 'series') { const feat = beeswarmFeats[Math.round(p.data.value[1])]; if (feat) setSelFeat(f => f === feat ? null : feat) } } }}
                />
              : <div className="mp-empty">shap_beeswarm.csv 없음</div>}
          </div>
        </div>
      </ChartCard>

    </div>
  )
}
