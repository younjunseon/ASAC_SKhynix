import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  fetchUnits,
  fetchUnitReport,
  type StatusFilter,
  type UnitItem,
} from "../lib/api";
import PageHeader from "../components/PageHeader";
import Panel from "../components/Panel";
import { fmtInt, fmtPct, fmtPpm, healthToPpm } from "../lib/format";
import {
  CHART_COLORS,
  CHART_GRID,
  CHART_TICK,
  CHART_TOOLTIP_STYLE,
  chartBox,
} from "../lib/chart";

/** 오늘 자정 기준 Date — 시간 부분 제거 */
function startOfToday(): Date {
  const d = new Date();
  d.setHours(0, 0, 0, 0);
  return d;
}

type SortKey =
  | "ufs_serial"
  | "status"
  | "inspected_date"
  | "run_id"
  | "wafer_key"
  | "pred"
  | "health"
  | "is_risk";
type Order = "asc" | "desc";

// 정렬 시 기본 방향 — ID/문자/날짜는 오름차순이 자연스럽고, 숫자/위험은 내림차순(큰 값 먼저)
const ASC_DEFAULT: SortKey[] = ["ufs_serial", "status", "inspected_date", "run_id", "wafer_key"];

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

const STATUS_LABEL: Record<string, string> = {
  today: "오늘",
  pending: "대기",
  completed: "완료",
};

export default function Data() {
  const [status, setStatus] = useState<StatusFilter>("today");
  const [sort, setSort] = useState<SortKey>("pred");
  const [order, setOrder] = useState<Order>("desc");
  const [page, setPage] = useState(1);
  const pageSize = 50;
  const [selectedUnit, setSelectedUnit] = useState<string | null>(null);

  // Draft 상태 — 입력 중인 값. 검색 버튼을 눌러야 applied로 복사됨
  const [riskOnlyDraft, setRiskOnlyDraft] = useState(false);
  const [searchDraft, setSearchDraft] = useState("");
  const [dateFromDraft, setDateFromDraft] = useState<string>("");
  const [dateToDraft, setDateToDraft] = useState<string>("");
  const [predMinDraft, setPredMinDraft] = useState<string>("");
  const [predMaxDraft, setPredMaxDraft] = useState<string>("");

  // Applied 상태 — 실제 쿼리/필터에 반영되는 값
  const [riskOnly, setRiskOnly] = useState(false);
  const [search, setSearch] = useState("");
  const [dateFrom, setDateFrom] = useState<string>("");
  const [dateTo, setDateTo] = useState<string>("");
  const [predMinPpm, setPredMinPpm] = useState<string>("");
  const [predMaxPpm, setPredMaxPpm] = useState<string>("");

  const isDirty =
    riskOnlyDraft !== riskOnly ||
    searchDraft !== search ||
    dateFromDraft !== dateFrom ||
    dateToDraft !== dateTo ||
    predMinDraft !== predMinPpm ||
    predMaxDraft !== predMaxPpm;

  const { data, isLoading, error } = useQuery({
    queryKey: ["units-table", status, riskOnly, sort, order, page],
    queryFn: () =>
      fetchUnits({ status, risk_only: riskOnly, sort, order, page, page_size: pageSize }),
  });

  // Unit Diagnosis
  const reportQ = useQuery({
    queryKey: ["unit-report", selectedUnit],
    queryFn: () => fetchUnitReport(selectedUnit!),
    enabled: !!selectedUnit,
    retry: false,
  });

  // KPI용 — 전체 unit 한 번 받아 오늘 통계 집계
  const allUnitsQ = useQuery({
    queryKey: ["units-all-data"],
    queryFn: () => fetchUnits({ page: 1, page_size: 50_000 }),
  });

  const todayKpi = useMemo(() => {
    const empty = { count: 0, riskCount: 0, meanPpm: 0, maxPpm: 0 };
    if (!allUnitsQ.data) return empty;
    const todayMs = startOfToday().getTime();
    let count = 0;
    let riskCount = 0;
    let predSum = 0;
    let predMax = 0;
    for (const u of allUnitsQ.data.items) {
      if (!u.inspected_date) continue;
      const d = new Date(u.inspected_date);
      d.setHours(0, 0, 0, 0);
      if (d.getTime() !== todayMs) continue;
      count += 1;
      predSum += u.pred;
      if (u.pred > predMax) predMax = u.pred;
      if (u.is_risk) riskCount += 1;
    }
    return {
      count,
      riskCount,
      meanPpm: count > 0 ? (predSum / count) * 1_000_000 : 0,
      maxPpm: predMax * 1_000_000,
    };
  }, [allUnitsQ.data]);

  const riskRatio = todayKpi.count > 0 ? todayKpi.riskCount / todayKpi.count : 0;

  // 최근 5일(오늘 포함) 일별 불량 unit 추이
  const recent5Days = useMemo(() => {
    const todayMs = startOfToday().getTime();
    const day = 86_400_000;
    type Bucket = { dateMs: number; label: string; iso: string; riskCount: number; total: number };
    const buckets: Bucket[] = [];
    for (let i = 4; i >= 0; i--) {
      const dMs = todayMs - i * day;
      const d = new Date(dMs);
      buckets.push({
        dateMs: dMs,
        label: i === 0 ? "오늘" : `${d.getMonth() + 1}/${d.getDate()}`,
        iso: d.toISOString().slice(0, 10),
        riskCount: 0,
        total: 0,
      });
    }
    if (!allUnitsQ.data) return buckets;
    const idxByMs = new Map(buckets.map((b, i) => [b.dateMs, i]));
    for (const u of allUnitsQ.data.items) {
      if (!u.inspected_date) continue;
      const d = new Date(u.inspected_date);
      d.setHours(0, 0, 0, 0);
      const idx = idxByMs.get(d.getTime());
      if (idx === undefined) continue;
      buckets[idx].total += 1;
      if (u.is_risk) buckets[idx].riskCount += 1;
    }
    return buckets;
  }, [allUnitsQ.data]);

  const recent5Stats = useMemo(() => {
    const totalRisk = recent5Days.reduce((a, b) => a + b.riskCount, 0);
    const total = recent5Days.reduce((a, b) => a + b.total, 0);
    const avgRisk = recent5Days.length > 0 ? totalRisk / recent5Days.length : 0;
    return { totalRisk, total, avgRisk };
  }, [recent5Days]);

  // 클라이언트 필터 (search + 검사일 + pred 범위)
  const filtered = (() => {
    if (!data) return [];
    const minPred = predMinPpm ? parseFloat(predMinPpm) / 1_000_000 : undefined;
    const maxPred = predMaxPpm ? parseFloat(predMaxPpm) / 1_000_000 : undefined;
    return data.items.filter((u) => {
      if (search) {
        const lower = search.toLowerCase();
        if (
          !u.ufs_serial.toLowerCase().includes(lower) &&
          !u.wafer_key.toLowerCase().includes(lower) &&
          !u.run_id.toLowerCase().includes(lower)
        )
          return false;
      }
      if (dateFrom && u.inspected_date < dateFrom) return false;
      if (dateTo && u.inspected_date > dateTo) return false;
      if (minPred !== undefined && u.pred < minPred) return false;
      if (maxPred !== undefined && u.pred > maxPred) return false;
      return true;
    });
  })();

  function applyFilters() {
    setRiskOnly(riskOnlyDraft);
    setSearch(searchDraft);
    setDateFrom(dateFromDraft);
    setDateTo(dateToDraft);
    setPredMinPpm(predMinDraft);
    setPredMaxPpm(predMaxDraft);
    setPage(1);
  }

  function resetFilters() {
    setRiskOnlyDraft(false);
    setSearchDraft("");
    setDateFromDraft("");
    setDateToDraft("");
    setPredMinDraft("");
    setPredMaxDraft("");
    setRiskOnly(false);
    setSearch("");
    setDateFrom("");
    setDateTo("");
    setPredMinPpm("");
    setPredMaxPpm("");
    setPage(1);
  }

  const total = data?.total ?? 0;
  const totalPages = Math.ceil(total / pageSize);

  function toggleSort(k: SortKey) {
    if (sort === k) {
      setOrder(order === "desc" ? "asc" : "desc");
    } else {
      setSort(k);
      setOrder(ASC_DEFAULT.includes(k) ? "asc" : "desc");
    }
    setPage(1);
  }
  function sortIndicator(k: SortKey) {
    if (sort !== k) return "";
    return order === "desc" ? " ▼" : " ▲";
  }

  function statusChip(s: string) {
    if (s === "today") return <span className="chip chip-today chip-sm">오늘</span>;
    if (s === "pending") return <span className="chip chip-pending chip-sm">대기</span>;
    return <span className="chip chip-completed chip-sm">완료</span>;
  }

  return (
    <div>
      <PageHeader
        title="유닛 차원"
        subtitle="데일리 유닛 현황·자재 분석 — status 필터·검색·정렬·CSV 다운로드"
        status={status}
        onStatusChange={(s) => {
          setStatus(s);
          setPage(1);
        }}
      />

      {/* 오늘 KPI — 처리 유닛 / 불량 유닛 (Overview 카드 디자인 톤) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 sm:gap-4 mb-4 sm:mb-5">
        {/* 1. 오늘 처리 유닛 */}
        <div
          className="panel relative overflow-hidden px-6 py-5 flex flex-col justify-between min-h-[160px]"
          style={{ borderLeft: "4px solid #f59e0b" }}
        >
          <div className="flex items-stretch justify-between gap-4 flex-1">
            <div className="flex flex-col justify-between flex-1">
              <div>
                <div className="text-[13px] font-semibold text-brand-textMuted uppercase tracking-wider">
                  오늘 처리 유닛
                </div>
                <div
                  className="tabular text-[48px] font-bold leading-none mt-2"
                  style={{ color: "#b45309" }}
                >
                  {fmtInt(todayKpi.count)}
                </div>
                <div className="text-[14px] text-brand-textMuted mt-1">
                  {allUnitsQ.isLoading ? "로딩 중…" : "오늘 검사 완료된 unit"}
                </div>
              </div>
              <div className="flex items-center gap-2 mt-4 text-[15px]">
                <span className="text-brand-textMuted font-medium">평균 pred</span>
                <span className="font-bold tabular text-[19px]" style={{ color: "#b45309" }}>
                  {fmtPpm(todayKpi.meanPpm)}
                </span>
              </div>
            </div>
            <div className="flex flex-col items-end justify-center gap-2 pl-5 border-l border-brand-border min-w-[110px]">
              <div className="text-[13px] font-semibold text-brand-textMuted uppercase tracking-wide">
                오늘 최대
              </div>
              <div
                className="tabular text-[24px] font-bold leading-none"
                style={{ color: "#dc2626" }}
              >
                {fmtPpm(todayKpi.maxPpm)}
              </div>
              <div className="text-[11px] text-brand-textMuted">ppm</div>
            </div>
          </div>
        </div>

        {/* 2. 오늘 불량 유닛 */}
        <div
          className="panel relative overflow-hidden px-6 py-5 flex flex-col justify-between min-h-[160px]"
          style={{ borderLeft: "4px solid #dc2626" }}
        >
          <div className="flex items-stretch justify-between gap-4 flex-1">
            <div className="flex flex-col justify-between flex-1">
              <div>
                <div className="text-[13px] font-semibold text-brand-textMuted uppercase tracking-wider">
                  오늘 불량 유닛
                </div>
                <div
                  className="tabular text-[48px] font-bold leading-none mt-2"
                  style={{ color: "#dc2626" }}
                >
                  {fmtInt(todayKpi.riskCount)}
                </div>
                <div className="text-[14px] text-brand-textMuted mt-1">
                  pred 상위 5% 위험 unit
                </div>
              </div>
              <div className="flex items-center gap-2 mt-4 text-[15px]">
                <span className="text-brand-textMuted font-medium">위험 비율</span>
                <span className="font-bold tabular text-[19px]" style={{ color: "#dc2626" }}>
                  {fmtPct(riskRatio)}
                </span>
                <span className="ml-1 text-brand-textMuted text-[15px]">
                  ({fmtInt(todayKpi.riskCount)} / {fmtInt(todayKpi.count)})
                </span>
              </div>
            </div>
            <div className="flex flex-col items-end justify-center gap-2 pl-5 border-l border-brand-border min-w-[110px]">
              <div className="text-[13px] font-semibold text-brand-textMuted uppercase tracking-wide">
                상태
              </div>
              <div
                className="tabular text-[20px] font-bold leading-tight text-right"
                style={{ color: riskRatio > 0.05 ? "#dc2626" : "#15803d" }}
              >
                {riskRatio > 0.05 ? "주의" : "정상"}
              </div>
              <div className="text-[11px] text-brand-textMuted">기준 5%</div>
            </div>
          </div>
        </div>
      </div>

      {/* 최근 5일치 불량 유닛 추이 */}
      <Panel
        title="최근 5일 불량 유닛 추이"
        right={
          <div className="flex items-center gap-3 text-[11px] text-brand-textMuted">
            <span>
              5일 합계 <span className="font-bold text-brand-danger">{fmtInt(recent5Stats.totalRisk)}</span>건
            </span>
            <span>
              일평균 <span className="font-bold tabular">{recent5Stats.avgRisk.toFixed(1)}</span>건
            </span>
            <span>
              검사 unit <span className="font-mono tabular">{fmtInt(recent5Stats.total)}</span>
            </span>
          </div>
        }
        className="mb-4"
      >
        <div style={chartBox(220)}>
          <ResponsiveContainer>
            <LineChart data={recent5Days} margin={{ top: 16, right: 24, left: 16, bottom: 8 }}>
              <CartesianGrid {...CHART_GRID} />
              <XAxis
                dataKey="label"
                tick={{ ...CHART_TICK, fontSize: 13, fontWeight: 600 }}
                padding={{ left: 12, right: 12 }}
              />
              <YAxis
                tick={CHART_TICK}
                allowDecimals={false}
                width={48}
                label={{
                  value: "불량 unit (건)",
                  angle: -90,
                  position: "insideLeft",
                  offset: 6,
                  fontSize: 12,
                  fontWeight: 700,
                  fill: "#1e293b",
                }}
              />
              <Tooltip
                contentStyle={CHART_TOOLTIP_STYLE}
                formatter={(value: any, _name: any, props: any) => {
                  const total = props?.payload?.total ?? 0;
                  const ratio = total > 0 ? (Number(value) / total) * 100 : 0;
                  return [
                    `${Number(value).toLocaleString()}건 / 검사 ${total.toLocaleString()} (${ratio.toFixed(1)}%)`,
                    "불량 unit",
                  ];
                }}
                labelFormatter={(label: any, payload: any) => {
                  const iso = payload?.[0]?.payload?.iso;
                  return iso ? `${label} (${iso})` : String(label);
                }}
              />
              <ReferenceLine
                y={recent5Stats.avgRisk}
                stroke="#94a3b8"
                strokeDasharray="4 4"
                strokeWidth={1.2}
                label={{
                  value: `평균 ${recent5Stats.avgRisk.toFixed(1)}`,
                  fontSize: 10,
                  fill: "#64748b",
                  position: "insideTopRight",
                }}
              />
              <Line
                type="monotone"
                dataKey="riskCount"
                stroke={CHART_COLORS.danger}
                strokeWidth={2.5}
                isAnimationActive={false}
                dot={(props: any) => {
                  const { cx, cy, index, payload } = props;
                  const isToday = payload?.label === "오늘";
                  return (
                    <circle
                      key={`dot-${index}`}
                      cx={cx}
                      cy={cy}
                      r={isToday ? 6 : 4}
                      fill={isToday ? "#b45309" : CHART_COLORS.danger}
                      stroke="#fff"
                      strokeWidth={isToday ? 2.5 : 1.5}
                    />
                  );
                }}
                activeDot={{ r: 6 }}
                label={(props: any) => {
                  const { x, y, value, index } = props;
                  if (value == null) return <g key={`lbl-${index}`} />;
                  return (
                    <text
                      key={`lbl-${index}`}
                      x={x}
                      y={y - 10}
                      textAnchor="middle"
                      fontSize={12}
                      fontWeight={700}
                      fill={CHART_COLORS.danger}
                    >
                      {value}
                    </text>
                  );
                }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Panel>

      <Panel title="필터" className="mb-4">
        <div className="space-y-2 text-[12px]">
          <div className="flex flex-wrap items-center gap-3">
            <label className="flex items-center gap-1.5">
              <input
                type="checkbox"
                checked={riskOnlyDraft}
                onChange={(e) => setRiskOnlyDraft(e.target.checked)}
              />
              <span>위험 unit만</span>
            </label>

            <div className="flex items-center gap-1.5">
              <span className="text-brand-textMuted">검색</span>
              <input
                type="text"
                value={searchDraft}
                onChange={(e) => setSearchDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") applyFilters();
                }}
                placeholder="unit / wafer / run_id"
                className="border border-brand-border rounded bg-white text-[12px] px-2 py-1 w-56 focus:outline-none focus:border-brand-primary"
              />
            </div>

            {status !== "today" && (
              <div className="flex items-center gap-1.5">
                <span className="text-brand-textMuted">검사일</span>
                <input
                  type="date"
                  value={dateFromDraft}
                  onChange={(e) => setDateFromDraft(e.target.value)}
                  className="border border-brand-border rounded bg-white text-[12px] px-2 py-1 focus:outline-none focus:border-brand-primary"
                />
                <span className="text-brand-textMuted">~</span>
                <input
                  type="date"
                  value={dateToDraft}
                  onChange={(e) => setDateToDraft(e.target.value)}
                  className="border border-brand-border rounded bg-white text-[12px] px-2 py-1 focus:outline-none focus:border-brand-primary"
                />
              </div>
            )}

            <div className="flex items-center gap-1.5">
              <span className="text-brand-textMuted">pred (ppm)</span>
              <input
                type="number"
                value={predMinDraft}
                onChange={(e) => setPredMinDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") applyFilters();
                }}
                placeholder="min"
                className="border border-brand-border rounded bg-white text-[12px] px-2 py-1 w-20 focus:outline-none focus:border-brand-primary"
              />
              <span className="text-brand-textMuted">~</span>
              <input
                type="number"
                value={predMaxDraft}
                onChange={(e) => setPredMaxDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") applyFilters();
                }}
                placeholder="max"
                className="border border-brand-border rounded bg-white text-[12px] px-2 py-1 w-20 focus:outline-none focus:border-brand-primary"
              />
            </div>

            <div className="ml-auto flex items-center gap-1.5">
              <button
                onClick={applyFilters}
                disabled={!isDirty}
                className="btn btn-primary text-[11px]"
                title={isDirty ? "필터를 적용합니다 (Enter 가능)" : "변경사항 없음"}
              >
                🔍 검색
              </button>
              <button onClick={resetFilters} className="btn text-[11px]" title="필터 초기화">
                초기화
              </button>
            </div>
          </div>

          <div className="flex items-center gap-2 text-[11px] pt-2 border-t border-brand-border">
            <span className="text-brand-textMuted">
              총 {fmtInt(total)} · 필터 후 {filtered.length}
            </span>
            {isDirty && (
              <span className="chip chip-tbd chip-sm" title="입력값이 아직 적용되지 않음">
                <span aria-hidden>⚠</span>
                <span>검색 미적용</span>
              </span>
            )}
            <div className="ml-auto">
              <button
                onClick={() => downloadCsv(filtered, `units_${status}_p${page}.csv`)}
                className="btn btn-primary text-[10px]"
                disabled={filtered.length === 0}
              >
                ↓ CSV
              </button>
            </div>
          </div>
        </div>
      </Panel>

      <Panel title="Units" bodyClassName="p-0">
        {isLoading && <div className="text-[12px] p-3">로딩…</div>}
        {error && (
          <div className="text-[12px] text-brand-danger p-3">
            API 연결 실패. 서버를 확인하세요.
          </div>
        )}
        {data && (
          <div className="max-h-[560px] overflow-y-auto overflow-x-auto">
            <table className="spotfire">
              <thead className="sticky top-0 z-10">
                <tr>
                  <th
                    className="cursor-pointer select-none hover:bg-brand-subtle"
                    onClick={() => toggleSort("ufs_serial")}
                  >
                    ufs_serial{sortIndicator("ufs_serial")}
                  </th>
                  <th
                    className="cursor-pointer select-none hover:bg-brand-subtle"
                    onClick={() => toggleSort("status")}
                  >
                    상태{sortIndicator("status")}
                  </th>
                  {status !== "today" && (
                    <th
                      className="cursor-pointer select-none hover:bg-brand-subtle"
                      onClick={() => toggleSort("inspected_date")}
                    >
                      검사일{sortIndicator("inspected_date")}
                    </th>
                  )}
                  <th
                    className="cursor-pointer select-none hover:bg-brand-subtle"
                    onClick={() => toggleSort("run_id")}
                  >
                    run_id{sortIndicator("run_id")}
                  </th>
                  <th
                    className="cursor-pointer select-none hover:bg-brand-subtle"
                    onClick={() => toggleSort("wafer_key")}
                  >
                    wafer{sortIndicator("wafer_key")}
                  </th>
                  <th
                    className="text-right cursor-pointer select-none hover:bg-brand-subtle"
                    onClick={() => toggleSort("pred")}
                  >
                    pred{sortIndicator("pred")}
                  </th>
                  <th
                    className="text-right cursor-pointer select-none hover:bg-brand-subtle"
                    onClick={() => toggleSort("health")}
                    title="field health 실측값. 오늘/대기 unit은 측정 전이라 '—'로 표시됨 (검사 후 1~3일 소요)"
                  >
                    health{sortIndicator("health")}
                  </th>
                  {status === "completed" && (
                    <th className="text-right" title="|pred − health| · 모델 절대오차 (작을수록 정확)">
                      |오차|
                    </th>
                  )}
                  <th
                    className="text-center cursor-pointer select-none hover:bg-brand-subtle"
                    onClick={() => toggleSort("is_risk")}
                    title="위험 = 해당 status 모집단 내 pred 상위 5%"
                  >
                    위험{sortIndicator("is_risk")}
                  </th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((u) => (
                  <tr
                    key={u.ufs_serial}
                    className={`cursor-pointer ${selectedUnit === u.ufs_serial ? "active" : ""}`}
                    onClick={() => setSelectedUnit((prev) => prev === u.ufs_serial ? null : u.ufs_serial)}
                  >
                    <td className="font-mono">{u.ufs_serial}</td>
                    <td>{statusChip(u.status)}</td>
                    {status !== "today" && (
                      <td className="font-mono text-[11px]">{u.inspected_date}</td>
                    )}
                    <td className="font-mono">{u.run_id}</td>
                    <td>
                      <Link
                        to={`/drilldown?key=${encodeURIComponent(u.wafer_key)}`}
                        className="text-brand-link hover:underline font-mono"
                      >
                        {u.wafer_key}
                      </Link>
                    </td>
                    <td
                      className={`text-right tabular font-mono ${
                        u.is_risk ? "text-brand-danger font-bold" : ""
                      }`}
                    >
                      {fmtPpm(healthToPpm(u.pred))}
                    </td>
                    <td className="text-right tabular font-mono text-brand-textMuted">
                      {u.health == null ? "—" : fmtPpm(healthToPpm(u.health))}
                    </td>
                    {status === "completed" && (
                      <td className="text-right tabular font-mono text-brand-textMuted">
                        {u.health == null
                          ? "—"
                          : fmtPpm(Math.abs(healthToPpm(u.pred) - healthToPpm(u.health)))}
                      </td>
                    )}
                    <td className="text-center">
                      {u.is_risk ? (
                        <span className="text-brand-danger font-bold">⚠</span>
                      ) : (
                        <span className="text-brand-textMuted">·</span>
                      )}
                    </td>
                  </tr>
                ))}
                {filtered.length === 0 && (
                  <tr>
                    <td
                      colSpan={8 - (status === "today" ? 1 : 0) + (status === "completed" ? 1 : 0)}
                      className="text-center text-brand-textMuted p-3"
                    >
                      조건에 맞는 unit 없음. (현재 필터: {STATUS_LABEL[status] ?? status})
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      {data && totalPages > 1 && (
        <div className="flex items-center justify-between text-[11px] mt-2 px-1 pr-20">
          <div className="text-brand-textMuted">
            page {page} / {totalPages} · {fmtPct((page * pageSize) / Math.max(1, total))}
          </div>
          <div className="flex gap-1">
            <button onClick={() => setPage(1)} disabled={page === 1} className="btn text-[10px]">« 처음</button>
            <button onClick={() => setPage(Math.max(1, page - 1))} disabled={page === 1} className="btn text-[10px]">‹ 이전</button>
            <button onClick={() => setPage(Math.min(totalPages, page + 1))} disabled={page === totalPages} className="btn text-[10px]">다음 ›</button>
            <button onClick={() => setPage(totalPages)} disabled={page === totalPages} className="btn text-[10px]">마지막 »</button>
          </div>
        </div>
      )}

      {/* Unit Diagnosis 패널 — 행 클릭 시 하단에 표시 */}
      {selectedUnit && (
        <Panel
          title="Unit Diagnosis"
          className="mt-4"
          right={
            <button
              className="btn text-[11px]"
              onClick={() => setSelectedUnit(null)}
            >
              ✕ 닫기
            </button>
          }
        >
          {reportQ.isLoading && (
            <div className="text-[12px] text-brand-textMuted p-2">로딩…</div>
          )}
          {reportQ.error && (
            <div className="text-[12px] text-brand-danger p-2">
              unit 보고서를 불러올 수 없습니다.
            </div>
          )}
          {reportQ.data && (() => {
            const r = reportQ.data;
            return (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {/* 좌: 판정 + 내러티브 */}
                <div className="space-y-3">
                  <div
                    className={`px-3 py-2 rounded-md border ${
                      r.verdict === "WARNING"
                        ? "bg-red-50 border-brand-danger"
                        : "bg-green-50 border-green-600"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span
                        className={`text-[13px] font-bold ${
                          r.verdict === "WARNING" ? "text-brand-danger" : "text-green-700"
                        }`}
                      >
                        {r.verdict === "WARNING" ? "⚠ WARNING" : "✓ NORMAL"}
                      </span>
                      <span className="font-mono text-[11px] text-brand-textMuted">{r.ufs_serial}</span>
                    </div>
                  </div>
                  <ul className="space-y-1 text-[12px] text-brand-text">
                    {r.narrative.map((line, i) => (
                      <li key={i} className="flex gap-1.5 leading-snug">
                        <span className="text-brand-primary">•</span>
                        <span>{line}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                {/* 우: 수치 테이블 + worst die */}
                <div className="space-y-3">
                  <table className="spotfire">
                    <tbody>
                      <tr>
                        <th>Pred</th>
                        <td className="font-mono tabular text-right font-bold text-brand-danger">
                          {fmtPpm(r.pred * 1e6)}
                        </td>
                      </tr>
                      <tr>
                        <th>Health</th>
                        <td className="font-mono tabular text-right">
                          {r.health == null ? "—" : fmtPpm(r.health * 1e6)}
                        </td>
                      </tr>
                      <tr>
                        <th>π mean</th>
                        <td className="font-mono tabular text-right">{r.pi_mean.toFixed(4)}</td>
                      </tr>
                      <tr>
                        <th>μ mean</th>
                        <td className="font-mono tabular text-right">{fmtPpm(r.mu_mean * 1e6)}</td>
                      </tr>
                      <tr>
                        <th>Percentile</th>
                        <td className="font-mono tabular text-right">{fmtPct(r.pred_rank)}</td>
                      </tr>
                      <tr>
                        <th>Wafer</th>
                        <td className="font-mono tabular text-right text-[11px]">{r.wafer_key}</td>
                      </tr>
                    </tbody>
                  </table>
                  {r.worst_die && (
                    <div className="border border-brand-border rounded-md bg-brand-subtle px-3 py-2">
                      <div className="text-[10px] font-semibold text-brand-textMuted uppercase tracking-wider mb-1">
                        Worst Die
                      </div>
                      <div className="font-mono text-[12px]">
                        ({r.worst_die.die_x}, {r.worst_die.die_y})
                        <span className="ml-2 text-brand-textMuted">pred</span>
                        <span className="ml-1 font-bold text-brand-danger">
                          {fmtPpm(r.worst_die.pred * 1e6)}
                        </span>
                        <span className="ml-2 text-brand-textMuted">π={r.worst_die.pi.toFixed(3)}</span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            );
          })()}
        </Panel>
      )}
    </div>
  );
}
