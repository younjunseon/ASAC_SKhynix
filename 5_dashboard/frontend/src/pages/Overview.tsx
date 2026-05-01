/**
 * Overview — PI 운영 페이지.
 *
 * 상단에 "오늘 검사" 강조 카드 + 일반 KPI 3개 + 위험 lot alert + 시계열 + Top + 분포.
 * Status 필터 default: today.
 */
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  fetchLots,
  fetchOverview,
  fetchTriage,
  fetchUnits,
  type StatusFilter,
  type UnitItem,
} from "../lib/api";
import KpiCard from "../components/KpiCard";
import PageHeader from "../components/PageHeader";
import Panel from "../components/Panel";
import StatChip from "../components/StatChip";
import { fmtInt, fmtPct, fmtPpm, healthToPpm, TIERS, tierOfHealth } from "../lib/format";
import {
  BAR_RADIUS,
  CHART_COLORS,
  CHART_GRID,
  CHART_LEGEND_STYLE,
  CHART_TICK,
  CHART_TOOLTIP_STYLE,
  chartBox,
} from "../lib/chart";

type Granularity = "day" | "week";

// 데이터 lot이 28개라 최대 28일치 → day=28 bin, week=4 bin (각 7일).
const DAY_BINS = 28;
const WEEK_BINS = 4;
const WEEK_DAYS = 7;

function downloadCsv(rows: UnitItem[], filename: string) {
  if (rows.length === 0) return;
  const headers = Object.keys(rows[0]);
  const lines = [
    headers.join(","),
    ...rows.map((r) => headers.map((h) => String((r as any)[h] ?? "")).join(",")),
  ];
  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

/** 오늘 자정 기준 Date — 시간 부분 제거해야 daysBack 계산이 안전 */
function startOfToday(): Date {
  const d = new Date();
  d.setHours(0, 0, 0, 0);
  return d;
}

function bucketLabel(g: Granularity, binIdx: number, nBins: number): string {
  // binIdx는 왼쪽=과거, 오른쪽=현재. 왼쪽 끝(idx=0)은 (nBins-1)*step일 전, 오른쪽 끝(idx=nBins-1)은 today.
  const step = g === "day" ? 1 : WEEK_DAYS;
  const d = startOfToday();
  d.setDate(d.getDate() - (nBins - 1 - binIdx) * step);
  return `${d.getMonth() + 1}/${d.getDate()}`;
}

interface TrendBucket {
  label: string;
  completedCount: number;
  pendingCount: number;
  todayCount: number;
  totalCount: number;
  /** 평균 pred를 ppm 단위로 (pred * 1e6) */
  meanPredPpm: number;
}

function buildTrend(units: UnitItem[], g: Granularity): TrendBucket[] {
  const nBins = g === "day" ? DAY_BINS : WEEK_BINS;
  const step = g === "day" ? 1 : WEEK_DAYS;
  const today = startOfToday().getTime();
  const arr: TrendBucket[] = Array.from({ length: nBins }, (_, i) => ({
    label: bucketLabel(g, i, nBins),
    completedCount: 0,
    pendingCount: 0,
    todayCount: 0,
    totalCount: 0,
    meanPredPpm: 0,
  }));
  const predSum = new Array(nBins).fill(0);
  for (const u of units) {
    if (!u.inspected_date) continue;
    const insp = new Date(u.inspected_date);
    insp.setHours(0, 0, 0, 0);
    const daysBack = Math.floor((today - insp.getTime()) / 86_400_000);
    if (daysBack < 0 || daysBack >= nBins * step) continue;
    // daysBack=0(today) → 마지막 bin, daysBack 클수록 왼쪽 bin
    const b = nBins - 1 - Math.floor(daysBack / step);
    arr[b].totalCount += 1;
    if (u.status === "completed") arr[b].completedCount += 1;
    else if (u.status === "today") arr[b].todayCount += 1;
    else arr[b].pendingCount += 1;
    predSum[b] += u.pred;
  }
  for (let i = 0; i < nBins; i++) {
    arr[i].meanPredPpm =
      arr[i].totalCount > 0 ? (predSum[i] / arr[i].totalCount) * 1_000_000 : 0;
  }
  return arr;
}

/* ─── 분포 차트: ppm log-scale (zero spike 분리) ──────────────
 * health=0 (정상)은 별도 카운트로 분리하고, ppm > 0인 unit만
 * log10(ppm) 기준 bin에 분배. zero가 70% 이상이라 같은 축에 두면 압도됨.
 */
interface PpmBucket {
  /** bin 라벨 (예: "1~10 ppm") */
  range: string;
  count: number;
  /** bin 중심 ppm (정렬용) */
  rangeStart: number;
}
function buildPpmHist(units: UnitItem[]): { zero: number; bins: PpmBucket[] } {
  // log10(ppm) 1..6 (1ppm ~ 1,000,000ppm) 1 단위 6 bin
  const decades = [
    { from: 0, to: 1, label: "<1 ppm" },
    { from: 1, to: 10, label: "1~10 ppm" },
    { from: 10, to: 100, label: "10~100 ppm" },
    { from: 100, to: 1_000, label: "100~1k ppm" },
    { from: 1_000, to: 10_000, label: "1k~10k ppm" },
    { from: 10_000, to: 100_000, label: "10k~100k ppm" },
    { from: 100_000, to: Infinity, label: "100k+ ppm" },
  ];
  const counts = new Array(decades.length).fill(0);
  let zero = 0;
  for (const u of units) {
    const ppm = Math.max(0, u.pred) * 1_000_000;
    if (ppm <= 0) {
      zero += 1;
      continue;
    }
    const idx = decades.findIndex((d) => ppm >= d.from && ppm < d.to);
    counts[idx >= 0 ? idx : decades.length - 1] += 1;
  }
  return {
    zero,
    bins: decades.map((d, i) => ({
      range: d.label,
      rangeStart: d.from,
      count: counts[i],
    })),
  };
}

/* ─── Tier(S/A/B/C/D) 분포 ─────────────────────────────────── */
interface TierCount {
  tier: string;
  count: number;
  ratio: number;
  color: string;
}
function buildTierDist(units: UnitItem[]): TierCount[] {
  const counts: Record<string, number> = { S: 0, A: 0, B: 0, C: 0, D: 0 };
  for (const u of units) counts[tierOfHealth(u.pred)] += 1;
  const total = units.length || 1;
  return TIERS.map((t) => ({
    tier: t.tier,
    count: counts[t.tier] ?? 0,
    ratio: (counts[t.tier] ?? 0) / total,
    color: t.color,
  }));
}

/* ─── 스크리닝 시뮬레이션 ──────────────────────────────────
 * 임계 ppm 이상 unit을 차단했을 때 통과 fleet의 평균 ppm 변화 등.
 */
interface ScreenSim {
  blocked: number;
  passed: number;
  blockRatio: number;
  meanPpmAll: number;
  meanPpmPassed: number;
  ppmReduction: number;
}
function simulateScreening(units: UnitItem[], thresholdPpm: number): ScreenSim {
  if (units.length === 0) {
    return {
      blocked: 0,
      passed: 0,
      blockRatio: 0,
      meanPpmAll: 0,
      meanPpmPassed: 0,
      ppmReduction: 0,
    };
  }
  const ppms = units.map((u) => Math.max(0, u.pred) * 1_000_000);
  const meanAll = ppms.reduce((a, b) => a + b, 0) / ppms.length;
  const passed = ppms.filter((p) => p <= thresholdPpm);
  const meanPassed =
    passed.length > 0 ? passed.reduce((a, b) => a + b, 0) / passed.length : 0;
  const blocked = ppms.length - passed.length;
  return {
    blocked,
    passed: passed.length,
    blockRatio: blocked / ppms.length,
    meanPpmAll: meanAll,
    meanPpmPassed: meanPassed,
    ppmReduction: meanAll - meanPassed,
  };
}

export default function Overview() {
  const [status, setStatus] = useState<StatusFilter>("today");
  const [granularity, setGranularity] = useState<Granularity>("day");

  const overviewQ = useQuery({ queryKey: ["overview"], queryFn: fetchOverview });
  const triageQ = useQuery({
    queryKey: ["triage", status],
    queryFn: () => fetchTriage({ status, top_units: 10, top_wafers: 10 }),
  });
  const allUnitsQ = useQuery({
    queryKey: ["units-all"],
    queryFn: () => fetchUnits({ page: 1, page_size: 500, sort: "pred", order: "desc" }),
  });
  const lotsQ = useQuery({
    queryKey: ["alert-lots", status],
    queryFn: () => fetchLots({ status, sort: "risk_ratio", limit: 5 }),
  });

  // 스크리닝 시뮬레이터 임계값 (ppm). 기본 5,000 ppm — Y>0 분포의 절반 정도 위치
  const [screenThresholdPpm, setScreenThresholdPpm] = useState(5_000);

  const filteredUnits = useMemo(() => {
    if (!allUnitsQ.data) return [];
    return status === "all"
      ? allUnitsQ.data.items
      : allUnitsQ.data.items.filter((u) => u.status === status);
  }, [allUnitsQ.data, status]);

  const trend = useMemo(
    () => (allUnitsQ.data ? buildTrend(allUnitsQ.data.items, granularity) : []),
    [allUnitsQ.data, granularity]
  );
  const ppmHist = useMemo(() => buildPpmHist(filteredUnits), [filteredUnits]);
  const tierDist = useMemo(() => buildTierDist(filteredUnits), [filteredUnits]);
  const screenSim = useMemo(
    () => simulateScreening(filteredUnits, screenThresholdPpm),
    [filteredUnits, screenThresholdPpm]
  );

  if (triageQ.isLoading || overviewQ.isLoading)
    return <div className="text-[12px] text-brand-textMuted p-2">로딩 중…</div>;
  if (triageQ.error || !triageQ.data || !overviewQ.data)
    return (
      <div className="panel p-3 text-[12px] text-brand-danger">
        API 연결 실패. 서버를 확인하세요.
      </div>
    );

  const todayStats = overviewQ.data.statuses.today;
  const t = triageQ.data.summary;
  const meanPred = (() => {
    if (status === "all") {
      const arr = Object.values(overviewQ.data.statuses);
      const total = arr.reduce((a, s) => a + s.n_units, 0);
      return total > 0 ? arr.reduce((a, s) => a + s.pred_mean * s.n_units, 0) / total : 0;
    }
    return overviewQ.data.statuses[status]?.pred_mean ?? 0;
  })();
  const baseline = lotsQ.data?.baseline_risk_ratio ?? t.risk_ratio;
  const dangerLots = (lotsQ.data?.items ?? []).filter((l) => l.risk_ratio > baseline * 1.5);

  return (
    <div>
      <PageHeader
        title="Overview"
        status={status}
        onStatusChange={setStatus}
      />

      {/* KPI 4종 — 모두 회귀(연속형 ppm) 관점. 분류 비율 표현 제거 */}
      {/* change chip은 어제 대비 delta가 backend에 없어 제거 (하드코딩이었음) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 mb-4 sm:mb-5">
        <KpiCard
          label="오늘 검사 unit"
          value={fmtInt(todayStats.n_units)}
          hint="실시간 진단 대상"
          tone="warn"
          accentBar="#f59e0b"
        />
        <KpiCard
          label="평균 예측 ppm"
          value={fmtPpm(healthToPpm(meanPred))}
          hint={`fleet 평균 (status=${status})`}
          tone="info"
        />
        <KpiCard
          label="p95 ppm"
          value={fmtPpm(healthToPpm(triageQ.data.scale.risk_threshold))}
          hint="상위 5% 꼬리 위험 수준"
          tone="warn"
        />
        <KpiCard
          label="임계 초과 unit"
          value={fmtInt(t.n_risk)}
          hint={`pred > p95 (전체 ${fmtInt(t.n_units)} 중)`}
          tone="danger"
        />
      </div>

      {/* Alert 영역 */}
      {dangerLots.length > 0 && (
        <Panel title="위험 lot 알림" className="mb-4 sm:mb-5">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[11px] text-brand-textMuted">
              평균 위험률 {fmtPct(baseline)} 대비 1.5배 이상:
            </span>
            {dangerLots.map((l) => (
              <Link
                key={l.run_id}
                to={`/drilldown`}
                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-amber-50 text-amber-800 text-[11px] font-semibold hover:bg-amber-100 transition-colors"
              >
                <span className="font-mono">{l.run_id}</span>
                <span className="tabular">{fmtPct(l.risk_ratio)}</span>
              </Link>
            ))}
          </div>
        </Panel>
      )}

      {/* 메인 시계열 — X1086 기반 lot date로 분배 (28일 spread, 월별 bin은 데이터 부족으로 제거) */}
      <Panel
        title="기간별 처리량 & 평균 pred (ppm)"
        right={
          <div className="flex items-center gap-2 flex-wrap">
            <span className="chip chip-tbd" title="Mann-Kendall 추세 검정 미연결">
              <span aria-hidden>⚠</span>
              <span>추세 검정 — TBD</span>
            </span>
            <div className="inline-flex bg-brand-subtle rounded-md overflow-hidden">
              {(["day", "week"] as Granularity[]).map((g) => (
                <button
                  key={g}
                  onClick={() => setGranularity(g)}
                  className={`text-[11px] px-2.5 py-1 font-medium transition-colors ${
                    granularity === g
                      ? "bg-brand-primary text-white"
                      : "text-brand-textMuted hover:text-brand-text"
                  }`}
                >
                  {g === "day" ? "일별" : "주별"}
                </button>
              ))}
            </div>
          </div>
        }
        className="mb-4 sm:mb-5"
      >
        {status === "today" && (
          <div className="text-[11px] text-brand-textMuted mb-2 px-1">
            오늘 모드: 오늘 검사된 unit만 누적 표시 (대기/완료 막대는 0). 전체 추이는 <strong>전체</strong> 필터에서 확인.
          </div>
        )}
        <div style={chartBox(240)}>
          <ResponsiveContainer>
            <ComposedChart data={trend} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
              <CartesianGrid {...CHART_GRID} />
              <XAxis
                dataKey="label"
                tick={CHART_TICK}
                interval={granularity === "day" ? 3 : 0}
              />
              <YAxis yAxisId="left" tick={CHART_TICK} />
              <YAxis
                yAxisId="right"
                orientation="right"
                tick={CHART_TICK}
                tickFormatter={(v) => `${Math.round(v).toLocaleString()}`}
                label={{
                  value: "ppm",
                  angle: 90,
                  position: "insideRight",
                  fontSize: 11,
                  fill: "#64748b",
                }}
              />
              <Tooltip
                contentStyle={CHART_TOOLTIP_STYLE}
                formatter={(value: any, name: any) => {
                  if (name === "평균 pred") return `${Number(value).toFixed(0)} ppm`;
                  return Number(value).toLocaleString();
                }}
              />
              <Legend wrapperStyle={CHART_LEGEND_STYLE} />
              <Bar yAxisId="left" dataKey="completedCount" stackId="a" fill={CHART_COLORS.primarySoft} name="완료" />
              <Bar yAxisId="left" dataKey="pendingCount" stackId="a" fill={CHART_COLORS.primaryLight} name="대기" />
              <Bar yAxisId="left" dataKey="todayCount" stackId="a" fill={CHART_COLORS.primary} name="오늘" radius={BAR_RADIUS} />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="meanPredPpm"
                stroke={CHART_COLORS.accent}
                strokeWidth={2.2}
                dot={{ r: 3, fill: CHART_COLORS.accent }}
                name="평균 pred"
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </Panel>

      {/* Top 위험 + 분포 */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 sm:gap-5 mb-4 sm:mb-5">
        <Panel
          title="위험 wafer Top 10"
          right={
            <Link to="/drilldown" className="text-[11px] text-brand-link hover:underline">
              자세히 →
            </Link>
          }
          bodyClassName="p-0"
        >
          <div className="overflow-x-auto">
            <table className="spotfire">
              <thead>
                <tr>
                  <th>Wafer</th>
                  <th className="text-right">Units</th>
                  <th className="text-right">위험</th>
                  <th className="text-right">비율</th>
                </tr>
              </thead>
              <tbody>
                {triageQ.data.top_wafers.slice(0, 10).map((w) => (
                  <tr key={w.wafer_key}>
                    <td>
                      <Link
                        to={`/drilldown?key=${encodeURIComponent(w.wafer_key)}`}
                        className="text-brand-link hover:underline font-mono"
                      >
                        {w.wafer_key}
                      </Link>
                    </td>
                    <td className="text-right tabular">{fmtInt(w.n_units)}</td>
                    <td className="text-right tabular text-brand-danger font-semibold">
                      {fmtInt(w.n_risk)}
                    </td>
                    <td className="text-right tabular font-bold">{fmtPct(w.risk_ratio)}</td>
                  </tr>
                ))}
                {triageQ.data.top_wafers.length === 0 && (
                  <tr><td colSpan={4} className="text-center text-brand-textMuted p-3">
                    데이터 없음
                  </td></tr>
                )}
              </tbody>
            </table>
          </div>
        </Panel>

        <Panel
          title="위험 unit Top 10"
          right={
            <button
              onClick={() => downloadCsv(triageQ.data.top_units, `risk_units_${status}.csv`)}
              className="btn btn-primary text-[10px]"
              disabled={triageQ.data.top_units.length === 0}
            >
              ↓ CSV
            </button>
          }
          bodyClassName="p-0"
        >
          <div className="overflow-x-auto">
            <table className="spotfire">
              <thead>
                <tr>
                  <th>Unit</th>
                  <th className="text-right">예측 ppm</th>
                  <th>Wafer</th>
                </tr>
              </thead>
              <tbody>
                {triageQ.data.top_units.slice(0, 10).map((u) => (
                  <tr key={u.ufs_serial}>
                    <td className="font-mono">{u.ufs_serial}</td>
                    <td className="text-right tabular font-mono font-bold text-brand-danger">
                      {fmtPpm(healthToPpm(u.pred))}
                    </td>
                    <td>
                      <Link
                        to={`/drilldown?key=${u.wafer_key}`}
                        className="text-brand-link hover:underline font-mono"
                      >
                        {u.wafer_key}
                      </Link>
                    </td>
                  </tr>
                ))}
                {triageQ.data.top_units.length === 0 && (
                  <tr><td colSpan={3} className="text-center text-brand-textMuted p-3">
                    데이터 없음
                  </td></tr>
                )}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>

      {/* Tier 분포 + 스크리닝 시뮬레이터 — 회귀 결과의 운영 의사결정 도구 */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 sm:gap-5 mb-4 sm:mb-5">
        {/* Tier 등급 분포 */}
        <Panel title="Tier 등급 분포 (S/A/B/C/D)">
          <div className="space-y-2">
            <div className="flex w-full h-7 rounded-md overflow-hidden border border-brand-border/60">
              {tierDist.map((td) =>
                td.count === 0 ? null : (
                  <div
                    key={td.tier}
                    style={{ background: td.color, width: `${td.ratio * 100}%` }}
                    className="text-[10px] text-white font-semibold flex items-center justify-center"
                    title={`${td.tier}: ${td.count.toLocaleString()} (${(td.ratio * 100).toFixed(1)}%)`}
                  >
                    {td.ratio > 0.04 ? td.tier : ""}
                  </div>
                ),
              )}
            </div>
            <table className="w-full text-[11px]">
              <tbody>
                {TIERS.map((tdef, i) => {
                  const td = tierDist[i];
                  return (
                    <tr key={tdef.tier} className="border-b border-brand-border/40 last:border-0">
                      <td className="py-1 pr-2">
                        <span
                          className="inline-block w-2.5 h-2.5 rounded-sm align-middle mr-1.5"
                          style={{ background: tdef.color }}
                        />
                        <span className="font-semibold">{tdef.tier}</span>
                        <span className="text-brand-textMuted ml-1">({tdef.desc})</span>
                      </td>
                      <td className="py-1 text-right tabular">{fmtInt(td?.count ?? 0)}</td>
                      <td className="py-1 text-right tabular text-brand-textMuted w-12">
                        {fmtPct(td?.ratio ?? 0, 1)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <div className="text-[10px] text-brand-textMuted px-1">
              * 등급은 의사결정용 그루핑 — 모델은 연속 ppm을 예측하며, 임계값은 운영 기준이지 모델 출력이 아님.
            </div>
          </div>
        </Panel>

        {/* 스크리닝 시뮬레이터 */}
        <Panel
          title="스크리닝 시뮬레이터"
          right={
            <span className="text-[11px] text-brand-textMuted">
              임계 초과 unit을 출하 차단했을 때 fleet 품질 변화
            </span>
          }
        >
          <div className="space-y-3">
            <div>
              <div className="flex justify-between items-baseline mb-1">
                <label className="text-[11px] font-medium text-brand-text">
                  차단 임계 ppm
                </label>
                <span className="tabular text-[14px] font-bold text-brand-primary">
                  {fmtPpm(screenThresholdPpm)}
                </span>
              </div>
              <input
                type="range"
                min={0}
                max={50_000}
                step={100}
                value={screenThresholdPpm}
                onChange={(e) => setScreenThresholdPpm(Number(e.target.value))}
                className="w-full accent-brand-primary"
              />
              <div className="flex justify-between text-[10px] text-brand-textMuted mt-0.5">
                <span>0</span>
                <span>10k</span>
                <span>25k</span>
                <span>50k ppm</span>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2 text-[11px]">
              <div className="bg-brand-subtle rounded-md p-2">
                <div className="text-brand-textMuted">차단율</div>
                <div className="tabular text-[16px] font-bold text-brand-warn">
                  {fmtPct(screenSim.blockRatio, 2)}
                </div>
                <div className="text-[10px] text-brand-textMuted">
                  {fmtInt(screenSim.blocked)} / {fmtInt(filteredUnits.length)} unit
                </div>
              </div>
              <div className="bg-brand-subtle rounded-md p-2">
                <div className="text-brand-textMuted">통과 fleet 평균 ppm</div>
                <div className="tabular text-[16px] font-bold text-brand-primary">
                  {fmtPpm(screenSim.meanPpmPassed)}
                </div>
                <div className="text-[10px] text-brand-textMuted">
                  차단 전: {fmtPpm(screenSim.meanPpmAll)}
                </div>
              </div>
              <div className="bg-emerald-50 rounded-md p-2 col-span-2">
                <div className="text-brand-textMuted">ppm 감소량 (스크리닝 효과)</div>
                <div className="tabular text-[18px] font-bold text-emerald-600">
                  {fmtPpm(screenSim.ppmReduction)} ↓
                </div>
                <div className="tbd-block mt-1.5">
                  <span className="font-semibold">⚠ Field 영향 환산 — TBD:</span>{" "}
                  사용자 수 × 일 동작 횟수 가정값이 아직 미연결.
                  운영 가정이 정해지면 "ppm 감소량 × N"으로 하루 field error 감소량 표시 예정.
                </div>
              </div>
            </div>
          </div>
        </Panel>
      </div>

      <Panel
        title="예측 ppm 분포 (zero spike 분리)"
        right={
          <span className="chip chip-tbd" title="train↔val PSI 미연결. Model 페이지에서 측정 가능">
            <span aria-hidden>⚠</span>
            <span>분포 변화 — TBD</span>
          </span>
        }
      >
        <div className="grid grid-cols-12 gap-3">
          {/* zero spike 별도 표시 — S 등급(0 ppm) */}
          <div className="col-span-12 sm:col-span-3 flex flex-col justify-center bg-emerald-50 rounded-md p-3">
            <div className="text-[11px] text-brand-textMuted">S 등급 (0 ppm, 정상)</div>
            <div className="tabular text-[22px] font-bold text-emerald-600 leading-tight">
              {fmtInt(ppmHist.zero)}
            </div>
            <div className="text-[10px] text-brand-textMuted">
              {fmtPct(filteredUnits.length > 0 ? ppmHist.zero / filteredUnits.length : 0, 1)} of fleet
            </div>
          </div>
          {/* ppm > 0 unit의 log decade 분포 */}
          <div className="col-span-12 sm:col-span-9" style={chartBox(180)}>
            <ResponsiveContainer>
              <BarChart data={ppmHist.bins} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid {...CHART_GRID} />
                <XAxis dataKey="range" tick={{ ...CHART_TICK, fontSize: 10 }} />
                <YAxis tick={CHART_TICK} />
                <Tooltip
                  contentStyle={CHART_TOOLTIP_STYLE}
                  formatter={(value: any) => `${Number(value).toLocaleString()} unit`}
                />
                <Bar dataKey="count" fill={CHART_COLORS.primary} radius={BAR_RADIUS} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </Panel>
    </div>
  );
}
