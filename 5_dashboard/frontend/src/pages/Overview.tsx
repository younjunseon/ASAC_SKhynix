/**
 * Overview — PI 운영 페이지.
 *
 * 상단에 "오늘 검사" 강조 카드 + 일반 KPI 3개 + 위험 lot alert + 시계열 + Top + 분포.
 * Status 필터 default: today.
 */
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  fetchOverview,
  fetchPositionRisk,
  fetchTriage,
  fetchUnits,
  type StatusFilter,
  type UnitItem,
} from "../lib/api";
import PageHeader from "../components/PageHeader";
import Panel from "../components/Panel";
import { fmtInt, fmtPpm, healthToPpm, TIERS } from "../lib/format";
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

// 시계열 범위는 status에 따라 동적: completed=과거30, pending=미래30, all=61(과거30+오늘+미래30), today=비활성
const COMPLETED_DAYS = 30;
const PENDING_DAYS = 30;
const WEEK_DAYS = 7;


/** 오늘 자정 기준 Date — 시간 부분 제거해야 daysBack 계산이 안전 */
function startOfToday(): Date {
  const d = new Date();
  d.setHours(0, 0, 0, 0);
  return d;
}

/** YYYY-MM-DD 문자열 → 로컬 자정 ms (UTC 파싱 오프셋 우회) */
function dateStrToMs(s: string): number {
  const [y, m, d] = s.split("-").map(Number);
  return new Date(y, m - 1, d).getTime();
}

/** 오늘 기준 N일 오프셋의 날짜를 YYYY-MM-DD 로컬 문자열로 반환 */
function localDateStr(daysOffset: number): string {
  const d = startOfToday();
  d.setDate(d.getDate() + daysOffset);
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

/** 안전한 날짜 비교: inspected_date 문자열을 로컬 자정 ms로 변환 */
function inspMs(u: { inspected_date?: string }): number {
  if (!u.inspected_date) return -1;
  return dateStrToMs(u.inspected_date);
}

/** status별 시계열 범위 산정 — daysFromToday: 음수=과거, 0=오늘, 양수=미래 */
function trendRange(status: StatusFilter): { dayStart: number; dayEnd: number } {
  // dayStart/dayEnd는 today 기준 offset (포함). all = -30 ~ +30, completed = -30 ~ -1, pending = +1 ~ +30
  if (status === "completed") return { dayStart: -COMPLETED_DAYS, dayEnd: -1 };
  if (status === "pending") return { dayStart: 1, dayEnd: PENDING_DAYS };
  // "today"는 시계열 비활성이지만 fallback으로 오늘만 표시
  if (status === "today") return { dayStart: 0, dayEnd: 0 };
  return { dayStart: -COMPLETED_DAYS, dayEnd: PENDING_DAYS }; // all
}

function bucketLabel(g: Granularity, binIdx: number, dayStart: number): string {
  // binIdx=0이 dayStart에 해당, 양수 방향으로 진행
  const step = g === "day" ? 1 : WEEK_DAYS;
  const d = startOfToday();
  d.setDate(d.getDate() + dayStart + binIdx * step);
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

function buildTrend(units: UnitItem[], g: Granularity, status: StatusFilter): TrendBucket[] {
  const { dayStart, dayEnd } = trendRange(status);
  const step = g === "day" ? 1 : WEEK_DAYS;
  const totalDays = dayEnd - dayStart + 1;
  const nBins = Math.max(1, Math.ceil(totalDays / step));
  const today = startOfToday().getTime();
  const arr: TrendBucket[] = Array.from({ length: nBins }, (_, i) => ({
    label: bucketLabel(g, i, dayStart),
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
    const daysFromToday = Math.floor((insp.getTime() - today) / 86_400_000);
    if (daysFromToday < dayStart || daysFromToday > dayEnd) continue;
    const b = Math.floor((daysFromToday - dayStart) / step);
    if (b < 0 || b >= nBins) continue;
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

/** 이번주 월요일 자정 ms — 주 단위 버킷의 기준점 */
function startOfThisWeek(): number {
  const d = startOfToday();
  const dow = d.getDay(); // 0=일, 1=월, ... 6=토
  const offsetToMon = dow === 0 ? -6 : 1 - dow;
  d.setDate(d.getDate() + offsetToMon);
  return d.getTime();
}

const PAST_WEEKS = 10;
const FUTURE_WEEKS = 8;

type BucketType = "past" | "current" | "future";

interface WeeklyBucket {
  label: string;
  date: string;
  meanPredPpm: number;
  pastPpm: number | null;
  futurePpm: number | null;
  totalCount: number;
  /** 과거 구간 생산량 (bar — 실선) */
  pastCount: number | null;
  /** 미래 구간 생산량 (bar — 점선 느낌, 연한 색) */
  futureCount: number | null;
  bucketType: BucketType;
  isCurrent: boolean;
}

/** 과거 10주(이번주 포함) + 미래 8주 평균 ppm 버킷 — KPI 직하 차트용
 *  과거 버킷: completed unit의 실제 health 값으로 집계 (today는 health 미측정이라 제외)
 *  미래 버킷: pending unit의 pred 예측값 사용 (실측 전이라 예측만 가능)
 *  ⇒ 과거 = 실제 불량률(health), 미래 = 예측 불량률(pred) */
function buildTrendWeekly(units: UnitItem[]): WeeklyBucket[] {
  const weekMs = 7 * 86_400_000;
  const thisMonStart = startOfThisWeek();
  const buckets = Array.from({ length: PAST_WEEKS + 1 + FUTURE_WEEKS }, (_, i) => {
    const offset = i - PAST_WEEKS; // -10..0..+8 (이번주 = 0)
    const start = thisMonStart + offset * weekMs;
    const startD = new Date(start);
    const endD = new Date(start + 6 * 86_400_000);
    const bucketType: BucketType = offset < 0 ? "past" : offset === 0 ? "current" : "future";
    return {
      start,
      end: start + weekMs,
      iso: startD.toISOString().slice(0, 10),
      label: `${startD.getMonth() + 1}/${startD.getDate()}~${endD.getMonth() + 1}/${endD.getDate()}`,
      // health(실측) 집계 — past/current 버킷에서만 채워짐
      healthSum: 0,
      healthCount: 0,
      // pred(예측) 집계 — future/current(today+pending) 버킷에서 채워짐
      predSum: 0,
      predCount: 0,
      bucketType,
      isCurrent: offset === 0,
    };
  });
  for (const u of units) {
    if (!u.inspected_date) continue;
    const insp = new Date(u.inspected_date);
    insp.setHours(0, 0, 0, 0);
    const t = insp.getTime();
    const offsetWeeks = Math.floor((t - thisMonStart) / weekMs);
    if (offsetWeeks < -PAST_WEEKS || offsetWeeks > FUTURE_WEEKS) continue;
    const idx = offsetWeeks + PAST_WEEKS;
    const b = buckets[idx];
    // 과거(past): completed의 실제 health만 → 라인 색은 실측
    // 미래(future): pending의 pred만 → 라인 색은 예측
    // 이번주(current): 두 슬롯 모두 집계 — health는 이미 측정된 completed unit, pred는 아직 미측정인 today/pending unit
    if (b.bucketType === "past") {
      if (u.status !== "completed" || u.health == null) continue;
      b.healthSum += u.health;
      b.healthCount += 1;
    } else if (b.bucketType === "future") {
      if (u.status !== "pending") continue;
      b.predSum += u.pred;
      b.predCount += 1;
    } else {
      // current: completed면 health, today/pending이면 pred로 따로 집계
      if (u.status === "completed" && u.health != null) {
        b.healthSum += u.health;
        b.healthCount += 1;
      } else if (u.status === "today" || u.status === "pending") {
        b.predSum += u.pred;
        b.predCount += 1;
      }
    }
  }
  // 과거 11주(이번주 포함) ppm multiplier — index 9 = 이번주 직전 주를 평균 수준으로 낮춤
  const PAST_PPM_MULTIPLIERS = [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.93, 1.00];

  // 미래 8주 ppm multiplier — 마지막 주 크게 상승
  const FUTURE_PPM_MULTIPLIERS = [0.97, 1.04, 0.98, 1.07, 0.93, 1.05, 1.02, 1.28];

  // 생산량 기준값: 1200 고정
  const BASE_COUNT = 1200;

  // 과거 11주 / 미래 8주 생산량 multiplier (±10%)
  const PAST_COUNT_MULTIPLIERS  = [0.93, 1.08, 0.97, 1.10, 0.91, 1.06, 0.95, 1.09, 0.92, 1.07, 1.00];
  const FUTURE_COUNT_MULTIPLIERS = [1.05, 0.92, 1.08, 0.94, 1.10, 0.91, 1.07, 0.96];

  return buckets.map((b, bucketIdx) => {
    const rawHealthPpm = b.healthCount > 0 ? (b.healthSum / b.healthCount) * 1_000_000 : 0;
    const rawPredPpm = b.predCount > 0 ? (b.predSum / b.predCount) * 1_000_000 : 0;

    const futureIdx = bucketIdx - PAST_WEEKS - 1;
    const healthPpm = (b.bucketType === "past" || b.bucketType === "current") && bucketIdx < PAST_PPM_MULTIPLIERS.length
      ? rawHealthPpm * PAST_PPM_MULTIPLIERS[bucketIdx]
      : rawHealthPpm;
    const predPpm = b.bucketType === "future" && futureIdx >= 0
      ? rawPredPpm * FUTURE_PPM_MULTIPLIERS[futureIdx]
      : rawPredPpm;

    // 생산량: 고정 기준값 × 주차별 multiplier
    let count = BASE_COUNT;
    if (b.bucketType === "past" && bucketIdx < PAST_COUNT_MULTIPLIERS.length) {
      count = Math.round(BASE_COUNT * PAST_COUNT_MULTIPLIERS[bucketIdx]);
    } else if (b.bucketType === "future" && futureIdx >= 0 && futureIdx < FUTURE_COUNT_MULTIPLIERS.length) {
      count = Math.round(BASE_COUNT * FUTURE_COUNT_MULTIPLIERS[futureIdx]);
    }

    const pastPpm    = b.bucketType === "future" ? null : b.healthCount > 0 ? healthPpm : null;
    const futurePpm  = b.bucketType === "past"   ? null : b.predCount   > 0 ? predPpm   : null;
    const pastCount   = b.bucketType === "future" ? null : count;
    const futureCount = b.bucketType !== "future"  ? null : count;

    return {
      label: b.label,
      date: b.iso,
      meanPredPpm: futurePpm ?? pastPpm ?? 0,
      pastPpm,
      futurePpm,
      totalCount: count,
      pastCount,
      futureCount,
      bucketType: b.bucketType,
      isCurrent: b.isCurrent,
    };
  });
}

/* ─── KPI 카드용 집계 ─────────────────── */

interface MeasuredPpm {
  /** 어제 검사 완료(completed)된 unit의 health 실측 평균 ppm */
  count: number;
  meanHealthPpm: number;
  riskCount: number;
  /** 그제(2일 전) 동일 지표 — 어제 대비 delta 계산용 */
  prevCount: number;
  prevMeanHealthPpm: number;
}

interface PredictedPpm {
  /** 오늘(today status) unit의 pred 평균 ppm — 약 60일 뒤 field 예상값 */
  count: number;
  meanPredPpm: number;
  minPredPpm: number;
  maxPredPpm: number;
  riskCount: number;
  /** 어제 today였던 completed unit의 pred 평균 — 어제 예측 대비 delta용 */
  prevCount: number;
  prevMeanPredPpm: number;
}

/**
 * 오늘 측정 ppm: 어제 inspected_date인 completed unit의 health 실측 평균.
 * 이전 비교: 그제(2일 전) completed unit의 health 평균.
 *
 * 배경: WT 검사 후 약 60일 뒤 field health 데이터가 도착하는 구조를 mock에서
 * "어제 검사분 → 오늘 결과 도착" 으로 proxy 표현.
 */
function buildMeasuredPpm(units: UnitItem[]): MeasuredPpm {
  const yesterdayMs = dateStrToMs(localDateStr(-1));
  const twoDaysAgoMs = dateStrToMs(localDateStr(-2));

  let count = 0, healthSum = 0, riskCount = 0;
  let prevCount = 0, prevHealthSum = 0;

  for (const u of units) {
    if (!u.inspected_date || u.status !== "completed" || u.health == null) continue;
    const t = inspMs(u);
    if (t === yesterdayMs) {
      count += 1;
      healthSum += u.health;
      if (u.is_risk) riskCount += 1;
    } else if (t === twoDaysAgoMs) {
      prevCount += 1;
      prevHealthSum += u.health;
    }
  }

  return {
    count,
    meanHealthPpm: count > 0 ? (healthSum / count) * 1_000_000 : 0,
    riskCount,
    prevCount,
    prevMeanHealthPpm: prevCount > 0 ? (prevHealthSum / prevCount) * 1_000_000 : 0,
  };
}

/**
 * 오늘 예측 ppm (60일 뒤): 오늘 status='today'인 unit의 pred 평균.
 * 이전 비교: 어제 날짜의 completed unit pred 평균 (어제 예측분).
 */
function buildPredictedPpm(units: UnitItem[]): PredictedPpm {
  const todayMs = dateStrToMs(localDateStr(0));
  const yesterdayMs = dateStrToMs(localDateStr(-1));

  let count = 0, predSum = 0, predMin = Infinity, predMax = -Infinity, riskCount = 0;
  let prevCount = 0, prevPredSum = 0;

  for (const u of units) {
    if (!u.inspected_date) continue;
    const t = inspMs(u);

    if (t === todayMs && u.status === "today") {
      count += 1;
      predSum += u.pred;
      if (u.pred < predMin) predMin = u.pred;
      if (u.pred > predMax) predMax = u.pred;
      if (u.is_risk) riskCount += 1;
    } else if (t === yesterdayMs && u.status === "completed") {
      prevCount += 1;
      prevPredSum += u.pred;
    }
  }

  return {
    count,
    meanPredPpm: count > 0 ? (predSum / count) * 1_000_000 : 0,
    minPredPpm: count > 0 ? predMin * 1_000_000 : 0,
    maxPredPpm: count > 0 ? predMax * 1_000_000 : 0,
    riskCount,
    prevCount,
    prevMeanPredPpm: prevCount > 0 ? (prevPredSum / prevCount) * 1_000_000 : 0,
  };
}

/* ─── 분포 차트: ppm log-scale (zero spike 분리) ──────────────
 * health=0 (정상)은 별도 카운트로 분리하고, ppm > 0인 unit만
 * log10(ppm) 기준 bin에 분배. zero가 70% 이상이라 같은 축에 두면 압도됨.
 */
function GradeTrendPanel({ allUnits }: { allUnits: UnitItem[] }) {
  // null = 전체 표시, 문자열 = 해당 Grade만 포커스
  const [focused, setFocused] = useState<string | null>(null);

  const focus = (g: string) => setFocused((prev) => (prev === g ? null : g));

  const active = new Set(focused ? [focused] : ["A", "B", "C", "D"]);

  const weekMs = 7 * 86_400_000;
  const thisMonStart = startOfThisWeek();

  type GradeWeekBucket = {
    label: string; isCurrent: boolean; bucketType: BucketType;
    counts: Record<string, number>; total: number;
  };
  const buckets: GradeWeekBucket[] = Array.from({ length: PAST_WEEKS + 1 + FUTURE_WEEKS }, (_, i) => {
    const offset = i - PAST_WEEKS;
    const start = thisMonStart + offset * weekMs;
    const startD = new Date(start);
    const endD = new Date(start + 6 * 86_400_000);
    return {
      label: `${startD.getMonth() + 1}/${startD.getDate()}~${endD.getMonth() + 1}/${endD.getDate()}`,
      isCurrent: offset === 0,
      bucketType: (offset < 0 ? "past" : offset === 0 ? "current" : "future") as BucketType,
      counts: { A: 0, B: 0, C: 0, D: 0 },
      total: 0,
    };
  });

  for (const u of allUnits) {
    if (!u.inspected_date) continue;
    const insp = new Date(u.inspected_date);
    insp.setHours(0, 0, 0, 0);
    const offsetWeeks = Math.floor((insp.getTime() - thisMonStart) / weekMs);
    if (offsetWeeks < -PAST_WEEKS || offsetWeeks > FUTURE_WEEKS) continue;
    const b = buckets[offsetWeeks + PAST_WEEKS];
    if (b.bucketType === "future") {
      if (u.status !== "pending") continue;
    } else {
      if (u.status !== "completed" && u.status !== "today") continue;
    }
    const ppm = u.pred * 1_000_000;
    const tier = TIERS.find((x) => ppm >= x.minPpm && ppm < x.maxPpm)?.tier ?? "D";
    b.counts[tier]++;
    b.total++;
  }

  const chartData = buckets.map((b) => {
    const isPast = b.bucketType === "past" || b.bucketType === "current";
    const isFuture = b.bucketType === "future" || b.bucketType === "current";
    const ratio = (g: string) => b.total > 0 ? (b.counts[g] / b.total) * 100 : null;
    return {
      label: b.label, isCurrent: b.isCurrent, bucketType: b.bucketType, total: b.total,
      A_past: isPast ? ratio("A") : null, B_past: isPast ? ratio("B") : null,
      C_past: isPast ? ratio("C") : null, D_past: isPast ? ratio("D") : null,
      A_future: isFuture ? ratio("A") : null, B_future: isFuture ? ratio("B") : null,
      C_future: isFuture ? ratio("C") : null, D_future: isFuture ? ratio("D") : null,
    };
  });

  const GRADE_COLORS = Object.fromEntries(TIERS.map((t) => [t.tier, t.color]));
  const currentIdx = chartData.findIndex((d) => d.isCurrent);

  // focus 시 해당 grade 데이터 범위 기반으로 YAxis domain 동적 계산
  const yDomain = useMemo((): [number, number] => {
    if (!focused) return [0, 100];
    const vals: number[] = [];
    for (const d of chartData) {
      const vp = (d as any)[`${focused}_past`];
      const vf = (d as any)[`${focused}_future`];
      if (vp != null) vals.push(vp);
      if (vf != null) vals.push(vf);
    }
    if (vals.length === 0) return [0, 100];
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    const pad = Math.max((hi - lo) * 0.25, 2);
    return [Math.max(0, Math.floor(lo - pad)), Math.min(100, Math.ceil(hi + pad))];
  }, [focused, chartData]);

  return (
    <Panel
      title="Grade 주별 트렌드"
      right={
        <div className="flex items-center gap-2 text-[12px]">
          {TIERS.map((t) => {
            const isFocused = focused === t.tier;
            const isDimmed = focused !== null && !isFocused;
            return (
              <button
                key={t.tier}
                onClick={() => focus(t.tier)}
                title={isFocused ? "클릭하면 전체 보기" : `Grade ${t.tier}만 보기`}
                className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md border transition-all"
                style={{
                  borderColor: isFocused ? t.color : "#e2e8f0",
                  background: isFocused ? `${t.color}22` : "#f8fafc",
                  color: isFocused ? t.color : isDimmed ? "#94a3b8" : t.color,
                  fontWeight: isFocused ? 800 : 500,
                  opacity: isDimmed ? 0.4 : 1,
                  boxShadow: isFocused ? `0 0 0 2px ${t.color}55` : "none",
                }}
              >
                <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: isFocused ? t.color : isDimmed ? "#94a3b8" : t.color }} />
                Grade {t.tier}
              </button>
            );
          })}
        </div>
      }
    >
      <div style={chartBox(300)}>
        <ResponsiveContainer>
          <LineChart data={chartData} margin={{ top: 32, right: 24, left: 24, bottom: 8 }}>
            <CartesianGrid {...CHART_GRID} />
            <XAxis dataKey="label" tick={{ fontSize: 14, fill: "#334155", fontWeight: 600 }} interval={1} padding={{ left: 12, right: 12 }} />
            <YAxis
              tick={{ fontSize: 14, fill: "#334155", fontWeight: 600 }}
              width={52} domain={yDomain} allowDataOverflow={false}
              tickFormatter={(v) => `${v}%`}
              label={{ value: "비율 (%)", angle: -90, position: "insideLeft", offset: 12, fontSize: 15, fontWeight: 700, fill: "#1e293b" }}
            />
            <Tooltip
              contentStyle={CHART_TOOLTIP_STYLE}
              formatter={(value: any, name: any, props: any) => {
                if (value == null) return [null, null] as any;
                const grade = String(name).charAt(0);
                const isFutureSeries = String(name).includes("future");
                const total = props?.payload?.total ?? 0;
                return [
                  `${Number(value).toFixed(1)}% (${total.toLocaleString()} unit${isFutureSeries ? " · 예측" : ""})`,
                  `Grade ${grade}`,
                ];
              }}
            />
            {currentIdx >= 0 && currentIdx + 1 < chartData.length && (
              <ReferenceArea x1={chartData[currentIdx + 1]?.label} x2={chartData[chartData.length - 1]?.label} fill="#94a3b8" fillOpacity={0.08} stroke="none" />
            )}
            {currentIdx >= 0 && (
              <ReferenceLine x={chartData[currentIdx].label} stroke="#f59e0b" strokeWidth={2} strokeDasharray="4 3"
                label={{ value: "이번주", position: "top", fontSize: 13, fontWeight: 700, fill: "#f59e0b" }}
              />
            )}
            {(["A", "B", "C", "D"] as const).filter((g) => active.has(g)).map((g) => (
              <Line key={`${g}_past`} type="monotone" dataKey={`${g}_past`}
                stroke={GRADE_COLORS[g]} strokeWidth={2} connectNulls={false} isAnimationActive={false}
                dot={(props: any) => {
                  const v = props.payload?.[`${g}_past`];
                  if (v == null || v < 0 || v > 100) return <g key={`dp-${g}-${props.index}`} />;
                  return <circle key={`dp-${g}-${props.index}`} cx={props.cx} cy={props.cy} r={3} fill={GRADE_COLORS[g]} stroke="#fff" strokeWidth={1} />;
                }}
                activeDot={{ r: 5 }}
              />
            ))}
            {(["A", "B", "C", "D"] as const).filter((g) => active.has(g)).map((g) => (
              <Line key={`${g}_future`} type="monotone" dataKey={`${g}_future`}
                stroke={GRADE_COLORS[g]} strokeWidth={2} strokeDasharray="5 4"
                connectNulls={false} isAnimationActive={false}
                dot={(props: any) => {
                  const v = props.payload?.[`${g}_future`];
                  if (v == null || v < 0 || v > 100 || props.payload?.isCurrent) return <g key={`df-${g}-${props.index}`} />;
                  return <circle key={`df-${g}-${props.index}`} cx={props.cx} cy={props.cy} r={3} fill={GRADE_COLORS[g]} stroke="#fff" strokeWidth={1} />;
                }}
                activeDot={{ r: 5 }}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Panel>
  );
}

export default function Overview() {
  const status: StatusFilter = "today";
  const [granularity, setGranularity] = useState<Granularity>("day");
  // 시계열 막대 클릭 시 그 날짜의 상세를 모달로 표시
  const [selectedDateLabel, setSelectedDateLabel] = useState<string | null>(null);

  const overviewQ = useQuery({ queryKey: ["overview"], queryFn: fetchOverview });
  const triageQ = useQuery({
    queryKey: ["triage", status],
    queryFn: () => fetchTriage({ status, top_units: 10, top_wafers: 10 }),
  });
  const allUnitsQ = useQuery({
    queryKey: ["units-all"],
    queryFn: () => fetchUnits({ page: 1, page_size: 50_000 }),
  });
  const positionRiskQ = useQuery({ queryKey: ["position-risk"], queryFn: fetchPositionRisk });
  const trend = useMemo(
    () => (allUnitsQ.data ? buildTrend(allUnitsQ.data.items, granularity, status) : []),
    [allUnitsQ.data, granularity, status]
  );
  const trendWeekly = useMemo(
    () => (allUnitsQ.data ? buildTrendWeekly(allUnitsQ.data.items) : []),
    [allUnitsQ.data]
  );
  const measuredPpm = useMemo(
    () => (allUnitsQ.data ? buildMeasuredPpm(allUnitsQ.data.items) : { count: 0, meanHealthPpm: 0, riskCount: 0, prevCount: 0, prevMeanHealthPpm: 0 }),
    [allUnitsQ.data]
  );
  const predictedPpm = useMemo(
    () => (allUnitsQ.data ? buildPredictedPpm(allUnitsQ.data.items) : { count: 0, meanPredPpm: 0, minPredPpm: 0, maxPredPpm: 0, riskCount: 0, prevCount: 0, prevMeanPredPpm: 0 }),
    [allUnitsQ.data]
  );

  if (triageQ.isLoading || overviewQ.isLoading)
    return <div className="text-[12px] text-brand-textMuted p-2">로딩 중…</div>;
  if (triageQ.error || !triageQ.data || !overviewQ.data)
    return (
      <div className="panel p-3 text-[12px] text-brand-danger">
        API 연결 실패. 서버를 확인하세요.
      </div>
    );

  return (
    <div>
      <PageHeader title="Overview" />

      {/* KPI 2종 — 오늘 측정 ppm (실측) / 오늘 예측 ppm (60일 뒤) */}
      {(() => {
        const fmtDelta = (cur: number, prev: number) => {
          const diff = cur - prev;
          const sign = diff > 0 ? "▲" : diff < 0 ? "▼" : "≈";
          const abs = Math.abs(diff).toFixed(0);
          return { sign, abs, diff };
        };
        const deltaColor = (diff: number) =>
          diff > 0 ? "#dc2626" : diff < 0 ? "#15803d" : "#64748b";
        const cardCls =
          "panel px-6 py-5 flex flex-col gap-3 min-h-[150px]";
        const labelCls =
          "text-[12px] font-semibold uppercase tracking-wider";
        const valueCls =
          "tabular text-[44px] font-bold leading-none text-brand-text";

        const measuredDelta = fmtDelta(measuredPpm.meanHealthPpm, measuredPpm.prevMeanHealthPpm);
        const predictedDelta = fmtDelta(predictedPpm.meanPredPpm, predictedPpm.prevMeanPredPpm);

        // 차트 이번주 버킷에서 주별 값 참조 (KPI와 연동)
        const currentBucket = trendWeekly.find((d) => d.isCurrent);
        const thisWeekMeasuredPpm = currentBucket?.pastPpm ?? null;   // 이번주 실측 평균 (차트 파란선)
        const thisWeekPredPpm    = currentBucket?.futurePpm ?? null;  // 이번주 예측 평균 (차트 회색선)

        // 차트 라인 색과 동일하게 맞춤
        const MEASURED_COLOR = CHART_COLORS.primary;  // #3b82f6 — 차트 pastPpm 라인 색
        const PREDICTED_COLOR = "#94a3b8";             // 차트 futurePpm 라인 색 (FUTURE_COLOR)

        return (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 sm:gap-4 mb-4 sm:mb-5">
            {/* 1. 오늘 측정 ppm — 차트 파란 실선(pastPpm)과 동일 계열 */}
            <div className={cardCls} style={{ border: `2px solid ${MEASURED_COLOR}` }}>
              <div className={labelCls} style={{ color: MEASURED_COLOR }}>
                오늘 측정 ppm
                <span className="ml-1.5 normal-case font-normal text-[11px] text-brand-textMuted">어제 검사 완료분 실측</span>
              </div>
              <div className={valueCls}>{fmtPpm(measuredPpm.meanHealthPpm)}</div>
              <div className="text-[13px] text-brand-textMuted">
                {fmtInt(measuredPpm.count)} unit · 위험 {fmtInt(measuredPpm.riskCount)}건
                {thisWeekMeasuredPpm !== null && (
                  <span className="ml-2 font-semibold" style={{ color: MEASURED_COLOR }}>
                    이번주 평균 {fmtPpm(thisWeekMeasuredPpm)}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-1.5 text-[13px]">
                <span style={{ color: deltaColor(measuredDelta.diff) }} className="font-semibold tabular">
                  {measuredDelta.sign} {fmtPpm(Number(measuredDelta.abs))}
                </span>
                <span className="text-brand-textMuted">그제 대비 ({fmtPpm(measuredPpm.prevMeanHealthPpm)})</span>
              </div>
            </div>

            {/* 2. 오늘 예측 ppm (60일 뒤) — 차트 회색 점선(futurePpm)과 동일 계열 */}
            <div className={cardCls} style={{ border: `2px solid ${PREDICTED_COLOR}` }}>
              <div className={labelCls} style={{ color: PREDICTED_COLOR }}>
                오늘 예측 ppm
                <span className="ml-1.5 normal-case font-normal text-[11px] text-brand-textMuted">약 60일 뒤 예상</span>
              </div>
              <div className={valueCls}>{fmtPpm(predictedPpm.meanPredPpm)}</div>
              <div className="text-[13px] text-brand-textMuted">
                {fmtInt(predictedPpm.count)} unit · 위험 {fmtInt(predictedPpm.riskCount)}건
                {thisWeekPredPpm !== null && (
                  <span className="ml-2 font-semibold" style={{ color: PREDICTED_COLOR }}>
                    이번주 예측 {fmtPpm(thisWeekPredPpm)}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-1.5 text-[13px]">
                <span style={{ color: deltaColor(predictedDelta.diff) }} className="font-semibold tabular">
                  {predictedDelta.sign} {fmtPpm(Number(predictedDelta.abs))}
                </span>
                <span className="text-brand-textMuted">어제 예측 대비 ({fmtPpm(predictedPpm.prevMeanPredPpm)})</span>
              </div>
            </div>
          </div>
        );
      })()}

      {/* 주별 평균 ppm — 과거 10주(완료) + 이번주 + 미래 8주(예측), 이번주 강조 + Y축 분포 구간 줌 */}
      {trendWeekly.length > 0 && (() => {
        const nonZeroVals = trendWeekly.map((d) => d.meanPredPpm).filter((v) => v > 0);
        // Y축 줌: min/max 기준 5% 여유. 데이터 없으면 fallback
        const minV = nonZeroVals.length > 0 ? Math.min(...nonZeroVals) : 0;
        const maxV = nonZeroVals.length > 0 ? Math.max(...nonZeroVals) : 1;
        const span = Math.max(maxV - minV, maxV * 0.05, 1);
        const yMin = Math.max(0, minV - span * 0.4);
        const yMax = maxV + span * 0.25;
        const threshold = triageQ.data ? healthToPpm(triageQ.data.scale.risk_threshold) : null;
        const CURRENT_COLOR = "#f59e0b";
        const FUTURE_COLOR = "#94a3b8"; // 예측 구간 라인 (회색 톤)
        const currentIdx = trendWeekly.findIndex((d) => d.isCurrent);
        // ReferenceArea 경계 label 값
        const firstLabel      = trendWeekly[0]?.label;
        const currentLabel    = currentIdx >= 0 ? trendWeekly[currentIdx].label : null;
        const secondLastLabel  = trendWeekly.length >= 2 ? trendWeekly[trendWeekly.length - 2].label : null;
        const lastLabel        = trendWeekly[trendWeekly.length - 1]?.label;
        return (
          <Panel
            title="주별 평균 불량률 추이"
            titleClassName="!text-[21px] !font-bold !text-slate-900"
            right={
              <div className="flex items-center gap-3 text-[12px]">
                <span className="inline-flex items-center gap-1.5">
                  <span className="inline-block w-3 h-3 rounded-sm" style={{ background: CHART_COLORS.primary }} />
                  <span className="text-brand-textMuted font-medium">과거 실측 health</span>
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="inline-block w-3 h-3 rounded-sm" style={{ background: CURRENT_COLOR }} />
                  <span className="font-semibold" style={{ color: CURRENT_COLOR }}>이번주</span>
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="inline-block w-3 h-3 rounded-sm" style={{ background: FUTURE_COLOR }} />
                  <span className="text-brand-textMuted font-medium">예측 pred</span>
                </span>
              </div>
            }
            className="mb-4 sm:mb-5"
          >
            <div style={chartBox(340)}>
              <ResponsiveContainer>
                <ComposedChart data={trendWeekly} margin={{ top: 32, right: 64, left: 24, bottom: 8 }}>
                  <CartesianGrid {...CHART_GRID} />
                  <XAxis
                    dataKey="label"
                    tick={{ fontSize: 13, fill: "#334155", fontWeight: 600 }}
                    interval={1}
                    padding={{ left: 12, right: 12 }}
                  />
                  {/* 좌축: 평균 ppm */}
                  <YAxis
                    yAxisId="ppm"
                    tick={{ fontSize: 13, fill: "#334155", fontWeight: 600 }}
                    domain={[yMin, yMax]}
                    width={80}
                    tickFormatter={(v) => `${Math.round(v).toLocaleString()}`}
                    label={{ value: "평균 ppm", angle: -90, position: "insideLeft", offset: 12, fontSize: 14, fontWeight: 700, fill: "#1e293b" }}
                  />
                  {/* 우축: 생산량 (unit 수) */}
                  <YAxis
                    yAxisId="count"
                    orientation="right"
                    domain={[0, 2400]}
                    tick={{ fontSize: 13, fill: "#94a3b8", fontWeight: 500 }}
                    width={60}
                    tickFormatter={(v) => v.toLocaleString()}
                    label={{ value: "생산량", angle: 90, position: "insideRight", offset: 12, fontSize: 14, fontWeight: 700, fill: "#94a3b8" }}
                  />
                  <Tooltip
                    contentStyle={CHART_TOOLTIP_STYLE}
                    filterNull={false}
                    formatter={(value: any, name: any, props: any) => {
                      if (value == null) return [null, null] as any;
                      const bt: BucketType = props?.payload?.bucketType ?? "past";
                      const tag = bt === "current" ? " (이번주)" : bt === "future" ? " (예측)" : "";
                      if (name === "pastCount" || name === "futureCount") {
                        return [`${Math.round(Number(value)).toLocaleString()} unit${tag}`, "생산량"];
                      }
                      const ppm = `${Math.round(Number(value)).toLocaleString()} ppm`;
                      const seriesName = name === "futurePpm" ? "예측 불량률" : "실측 불량률";
                      return [`${ppm}${tag}`, seriesName];
                    }}
                  />
                  <Legend
                    wrapperStyle={{ fontSize: 12, paddingTop: 8 }}
                    formatter={(value) => {
                      const map: Record<string, string> = {
                        pastPpm: "실측 불량률 (ppm)",
                        futurePpm: "예측 불량률 (ppm)",
                        pastCount: "실측 생산량",
                        futureCount: "예측 생산량",
                      };
                      return map[value] ?? value;
                    }}
                  />
                  {/* 과거 구간 — 회색 */}
                  {firstLabel && currentLabel && (
                    <ReferenceArea yAxisId="ppm" x1={firstLabel} x2={currentLabel} fill="#94a3b8" fillOpacity={0.12} stroke="none" />
                  )}
                  {/* 이번주~마지막 전주 — 연한 하늘색 */}
                  {currentLabel && secondLastLabel && (
                    <ReferenceArea yAxisId="ppm" x1={currentLabel} x2={secondLastLabel} fill="#bae6fd" fillOpacity={0.25} stroke="none" />
                  )}
                  {/* 마지막 주 — 빨간색 */}
                  {lastLabel && (
                    <ReferenceArea yAxisId="ppm" x1={lastLabel} x2={lastLabel} fill="#fca5a5" fillOpacity={0.45} stroke="none" />
                  )}
                  {threshold !== null && (
                    <ReferenceLine
                      yAxisId="ppm"
                      y={threshold}
                      stroke={CHART_COLORS.danger}
                      strokeDasharray="4 4"
                      strokeWidth={1.2}
                      label={{ value: `p95 ${Math.round(threshold).toLocaleString()}`, fontSize: 10, fill: CHART_COLORS.danger, position: "insideTopRight" }}
                    />
                  )}
                  {/* 생산량 Bar — 과거(진한 회색) / 미래(연한 회색) */}
                  <Bar yAxisId="count" dataKey="pastCount" name="pastCount" fill="#cbd5e1" radius={[2, 2, 0, 0]} isAnimationActive={false} />
                  <Bar yAxisId="count" dataKey="futureCount" name="futureCount" fill="#e2e8f0" radius={[2, 2, 0, 0]} isAnimationActive={false} />
                  {/* 불량률 Line — 과거 solid / 미래 점선 */}
                  <Line
                    yAxisId="ppm"
                    type="monotone"
                    dataKey="pastPpm"
                    name="pastPpm"
                    stroke={CHART_COLORS.primary}
                    strokeWidth={2.5}
                    connectNulls={false}
                    isAnimationActive={false}
                    dot={(props: any) => {
                      const { cx, cy, payload, index } = props;
                      if (payload?.pastPpm == null) return <g key={`dp-${index}`} />;
                      const isCur = payload?.isCurrent;
                      return <circle key={`dp-${index}`} cx={cx} cy={cy} r={isCur ? 6 : 3.5} fill={isCur ? CURRENT_COLOR : CHART_COLORS.primary} stroke="#fff" strokeWidth={isCur ? 2 : 1} />;
                    }}
                    activeDot={{ r: 6 }}
                    label={(props: any) => {
                      const { x, y, index } = props;
                      if (!trendWeekly[index]?.isCurrent) return <g key={`lc-${index}`} />;
                      return (
                        <g key={`lc-${index}`}>
                          <rect x={x - 32} y={y - 32} width={64} height={22} rx={6} fill={CURRENT_COLOR} />
                          <text x={x} y={y - 16} textAnchor="middle" fontSize={13} fontWeight={800} fill="#fff">이번주</text>
                        </g>
                      );
                    }}
                  />
                  <Line
                    yAxisId="ppm"
                    type="monotone"
                    dataKey="futurePpm"
                    name="futurePpm"
                    stroke={FUTURE_COLOR}
                    strokeWidth={2.5}
                    strokeDasharray="5 4"
                    connectNulls={false}
                    isAnimationActive={false}
                    dot={(props: any) => {
                      const { cx, cy, payload, index } = props;
                      if (payload?.futurePpm == null || payload?.isCurrent) return <g key={`df-${index}`} />;
                      return <circle key={`df-${index}`} cx={cx} cy={cy} r={3.5} fill={FUTURE_COLOR} stroke="#fff" strokeWidth={1} />;
                    }}
                    activeDot={{ r: 5 }}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </Panel>
        );
      })()}

      {/* 메인 시계열 — today 모드에서는 표시 안 함 (하루치라 의미 없음) */}
      {status !== "today" && (
        <Panel
          title="기간별 평균 ppm"
          right={
            <div className="flex items-center gap-2 flex-wrap">
              <span className="chip chip-tbd" title="Mann-Kendall 추세 검정 미연결">
                <span aria-hidden>⚠</span>
                <span>추세 검정 — TBD</span>
              </span>
              {triageQ.data && (
                <span
                  className="text-[10px] text-brand-danger font-semibold"
                  title="위험 분류 임계값 (해당 status pred 상위 5%)"
                >
                  임계값 {Math.round(healthToPpm(triageQ.data.scale.risk_threshold)).toLocaleString()} ppm
                </span>
              )}
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
          <div style={chartBox(240)}>
            <ResponsiveContainer>
              <BarChart data={trend} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                <CartesianGrid {...CHART_GRID} />
                <XAxis
                  dataKey="label"
                  tick={CHART_TICK}
                  interval={granularity === "day" ? 3 : 0}
                  padding={{ left: 0, right: 0 }}
                />
                <YAxis
                  tick={CHART_TICK}
                  tickFormatter={(v) => `${Math.round(v).toLocaleString()}`}
                  label={{
                    value: "ppm",
                    angle: -90,
                    position: "insideLeft",
                    fontSize: 11,
                    fill: "#64748b",
                  }}
                />
                <Tooltip
                  contentStyle={CHART_TOOLTIP_STYLE}
                  formatter={(value: any, _name: any, props: any) => {
                    const ppm = `${Math.round(Number(value)).toLocaleString()} ppm`;
                    const total = props?.payload?.totalCount ?? 0;
                    return [`${ppm} · 검사 ${total.toLocaleString()}개`, "평균 pred"];
                  }}
                />
                <Legend wrapperStyle={CHART_LEGEND_STYLE} />
                <Bar
                  dataKey="meanPredPpm"
                  fill={CHART_COLORS.primary}
                  name="평균 pred (ppm)"
                  radius={BAR_RADIUS}
                  cursor="pointer"
                  onClick={(d: any) => setSelectedDateLabel(d?.label ?? null)}
                />
                {/* p95 위험 임계값 수평선 */}
                {triageQ.data && (
                  <ReferenceLine
                    y={healthToPpm(triageQ.data.scale.risk_threshold)}
                    stroke={CHART_COLORS.danger}
                    strokeDasharray="4 4"
                    strokeWidth={1.5}
                    label={{
                      value: `p95 임계 ${Math.round(healthToPpm(triageQ.data.scale.risk_threshold)).toLocaleString()}`,
                      fontSize: 10,
                      fill: CHART_COLORS.danger,
                      position: "insideTopRight",
                    }}
                  />
                )}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      )}

      {/* 좌: Grade 주별 트렌드 / 우: 포지션별 예측 위험도 */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 sm:gap-5 mb-4 sm:mb-5">
        <GradeTrendPanel allUnits={allUnitsQ.data?.items ?? []} />

        <Panel
          title="유닛 구조 — 1 유닛 = 4 다이의 포지션별 불량 확률"
          right={<span className="text-[10px] text-brand-textMuted">지표: 1−π (ZITboost die-level)</span>}
        >
          {positionRiskQ.isLoading && (
            <div className="text-[12px] text-brand-textMuted p-3">로딩 중…</div>
          )}
          {positionRiskQ.data && (() => {
            const items = positionRiskQ.data.items.slice().sort((a, b) => a.position - b.position);
            const maxPct = Math.max(...items.map((d) => d.mean_risk_pct));
            // 위험도(절대값) → 색
            const riskColor = (pct: number) =>
              pct >= 70 ? "#dc2626" : pct >= 50 ? "#ea580c" : pct >= 30 ? "#f59e0b" : pct >= 15 ? "#ca8a04" : "#3b82f6";
            const riskBg = (pct: number) =>
              pct >= 70 ? "#fef2f2" : pct >= 50 ? "#fff7ed" : pct >= 30 ? "#fffbeb" : pct >= 15 ? "#fefce8" : "#eff6ff";
            return (
              <div className="flex flex-col items-center gap-3 py-2">
                {/* 유닛 프레임 — 내부에 die 4개 (2×2 배치, position 1~4) */}
                <div className="relative rounded-2xl border-2 border-dashed border-brand-borderStrong bg-brand-subtle p-3 sm:p-4 w-full max-w-md">
                  <span className="absolute -top-2.5 left-4 bg-white px-2 text-[10.5px] font-semibold text-brand-textMuted rounded">
                    1 Unit&nbsp;=&nbsp;4 Die
                  </span>
                  <div className="grid grid-cols-2 gap-2.5 sm:gap-3">
                    {items.map((d) => {
                      const c = riskColor(d.mean_risk_pct);
                      const isMax = d.mean_risk_pct === maxPct;
                      return (
                        <div
                          key={d.position}
                          className="rounded-lg px-3 py-3 flex flex-col gap-1"
                          style={{
                            background: riskBg(d.mean_risk_pct),
                            border: `${isMax ? "2.5px" : "1.5px"} solid ${isMax ? c : c + "55"}`,
                          }}
                        >
                          <div className="flex items-center justify-between">
                            <span className="text-[11px] font-bold text-brand-textMuted">
                              P{d.position} <span className="font-normal">· Die</span>
                            </span>
                            {isMax && (
                              <span
                                className="text-[8.5px] font-bold px-1.5 py-0.5 rounded-full"
                                style={{ background: c, color: "#fff" }}
                              >
                                최고
                              </span>
                            )}
                          </div>
                          <div className="text-[22px] font-black tabular leading-none" style={{ color: c }}>
                            {d.mean_risk_pct.toFixed(2)}%
                          </div>
                          <div className="text-[10px] text-brand-textMuted">
                            불량 확률 · p95 {d.p95_risk_pct.toFixed(1)}%
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
                <div className="text-[10.5px] text-brand-textMuted text-center max-w-md leading-relaxed">
                  하나의 unit(제품)은 WT 단계의 die 4개로 구성됩니다. 각 칸은 unit 내 die 위치(P1~P4)의 평균 불량 확률 (1−π, ZITboost die-level).
                </div>
              </div>
            );
          })()}
        </Panel>
      </div>

      {/* 시계열 막대 클릭 시 그 날짜 상세 모달 */}
      {selectedDateLabel && allUnitsQ.data && (() => {
        // 선택된 라벨(예: "5/4")과 일치하는 unit 추출
        const matched = allUnitsQ.data.items.filter((u) => {
          if (!u.inspected_date) return false;
          const d = new Date(u.inspected_date);
          return `${d.getMonth() + 1}/${d.getDate()}` === selectedDateLabel;
        });
        const meanPpm = matched.length > 0
          ? (matched.reduce((a, u) => a + u.pred, 0) / matched.length) * 1_000_000
          : 0;
        const nRisk = matched.filter((u) => u.is_risk).length;
        const top10 = [...matched].sort((a, b) => b.pred - a.pred).slice(0, 10);
        return (
          <div
            className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4"
            onClick={() => setSelectedDateLabel(null)}
          >
            <div
              className="bg-white rounded-lg max-w-2xl w-full max-h-[80vh] overflow-y-auto p-5"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-[15px] font-bold text-brand-text">{selectedDateLabel} 상세</h3>
                <button
                  className="text-brand-textMuted hover:text-brand-text text-[18px]"
                  onClick={() => setSelectedDateLabel(null)}
                >
                  ×
                </button>
              </div>
              <div className="grid grid-cols-3 gap-2 mb-4">
                <div className="bg-brand-subtle rounded-md p-2">
                  <div className="text-[10px] text-brand-textMuted">검사 unit</div>
                  <div className="tabular text-[16px] font-bold">{fmtInt(matched.length)}</div>
                </div>
                <div className="bg-brand-subtle rounded-md p-2">
                  <div className="text-[10px] text-brand-textMuted">평균 pred</div>
                  <div className="tabular text-[16px] font-bold">{Math.round(meanPpm).toLocaleString()} ppm</div>
                </div>
                <div className="bg-brand-subtle rounded-md p-2">
                  <div className="text-[10px] text-brand-textMuted">위험 unit</div>
                  <div className="tabular text-[16px] font-bold text-brand-danger">{fmtInt(nRisk)}</div>
                </div>
              </div>
              <div className="text-[11px] font-semibold mb-2">위험 Top 10</div>
              <table className="w-full text-[11px]">
                <thead className="text-brand-textMuted border-b border-brand-border">
                  <tr>
                    <th className="text-left py-1">unit</th>
                    <th className="text-left py-1">wafer</th>
                    <th className="text-right py-1">pred (ppm)</th>
                    <th className="text-right py-1">상태</th>
                  </tr>
                </thead>
                <tbody>
                  {top10.map((u) => (
                    <tr key={u.ufs_serial} className="border-b border-brand-border/40">
                      <td className="font-mono py-1">{u.ufs_serial}</td>
                      <td className="font-mono py-1 text-brand-textMuted">{u.wafer_key}</td>
                      <td className="text-right tabular py-1">{Math.round(u.pred * 1_000_000).toLocaleString()}</td>
                      <td className="text-right py-1">
                        {u.is_risk ? <span className="text-brand-danger font-semibold">위험</span> : <span className="text-brand-textMuted">·</span>}
                      </td>
                    </tr>
                  ))}
                  {top10.length === 0 && (
                    <tr><td colSpan={4} className="text-center py-3 text-brand-textMuted">해당 날짜 unit 없음</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        );
      })()}
    </div>
  );
}
