import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import {
  fetchUnits,
  type StatusFilter,
  type UnitItem,
} from "../lib/api";
import PageHeader from "../components/PageHeader";
import Panel from "../components/Panel";
import { fmtInt, fmtPct, fmtPpm, healthToPpm } from "../lib/format";

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
        title="Data"
        status={status}
        onStatusChange={(s) => {
          setStatus(s);
          setPage(1);
        }}
      />

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
                  <th
                    className="cursor-pointer select-none hover:bg-brand-subtle"
                    onClick={() => toggleSort("inspected_date")}
                  >
                    검사일{sortIndicator("inspected_date")}
                  </th>
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
                  <th
                    className="text-center cursor-pointer select-none hover:bg-brand-subtle"
                    onClick={() => toggleSort("is_risk")}
                  >
                    위험{sortIndicator("is_risk")}
                  </th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((u) => (
                  <tr key={u.ufs_serial}>
                    <td className="font-mono">{u.ufs_serial}</td>
                    <td>{statusChip(u.status)}</td>
                    <td className="font-mono text-[11px]">{u.inspected_date}</td>
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
                    <td colSpan={8} className="text-center text-brand-textMuted p-3">
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
    </div>
  );
}
