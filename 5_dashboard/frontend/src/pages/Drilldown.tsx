/**
 * Drill-down 페이지.
 *
 * 좌측  : Lot 트리 (lot 그룹 → wafer 리스트)
 * 중앙  : Wafer Map (단일/lot 누적 토글)
 * 우측  : Unit Diagnosis (verdict + SHAP + Anomaly + 신뢰구간)
 *
 * SHAP / Anomaly / 신뢰구간 / lot 검정은 현재 산출물 미연결 — TBD placeholder로 비워둠.
 */
import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";
import {
  fetchLotMap,
  fetchLots,
  fetchTriage,
  fetchUnitReport,
  fetchWaferDetail,
  fetchWaferGrid,
  fetchWafers,
  type LotItem,
  type StatusFilter,
  type WaferSummary,
} from "../lib/api";
import WaferMap from "../components/WaferMap";
import PageHeader from "../components/PageHeader";
import Panel from "../components/Panel";
import { fmtInt, fmtNum, fmtPct } from "../lib/format";

type ViewMode = "single" | "lot";
type LotSort = "risk_ratio" | "lot_id";
type SortOrder = "asc" | "desc";

const LOT_SORT_OPTIONS: { value: LotSort; label: string }[] = [
  { value: "risk_ratio", label: "위험률" },
  { value: "lot_id", label: "lot/wafer 번호" },
];

export default function Drilldown() {
  const [searchParams, setSearchParams] = useSearchParams();
  const initialKey = searchParams.get("key");
  const [status, setStatus] = useState<StatusFilter>("today");
  const [selectedLot, setSelectedLot] = useState<string | null>(null);
  const [selectedKey, setSelectedKey] = useState<string | null>(initialKey);
  const [selectedUnit, setSelectedUnit] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("single");
  const [search, setSearch] = useState("");
  const [lotSort, setLotSort] = useState<LotSort>("risk_ratio");
  // 항목별 default 방향: 위험률은 ↓, 번호는 ↑
  const [lotOrder, setLotOrder] = useState<SortOrder>("desc");

  const triageQ = useQuery({
    queryKey: ["dd-triage-scale", status],
    queryFn: () => fetchTriage({ status, top_units: 1, top_wafers: 1 }),
  });
  const gridQ = useQuery({
    queryKey: ["wafer-grid"],
    queryFn: fetchWaferGrid,
    staleTime: Infinity,
  });
  const lotsQ = useQuery({
    queryKey: ["dd-lots", status],
    queryFn: () => fetchLots({ status, sort: "risk_ratio", limit: 50 }),
  });
  // wafer 리스트는 status 무관 — lot 트리에서 모든 wafer를 펼치도록.
  // (lot 자체는 lotsQ로 status 필터됨. wafer 단위 status는 트리 표시에서 의미 적음.)
  const wafersQ = useQuery({
    queryKey: ["dd-wafers-all"],
    queryFn: () => fetchWafers({ sort: "risk_ratio", limit: 2000 }),
    staleTime: Infinity,
  });
  const detailQ = useQuery({
    queryKey: ["dd-wafer-detail", selectedKey],
    queryFn: () => fetchWaferDetail(selectedKey!),
    enabled: !!selectedKey,
  });
  // lot 자체를 선택한 경우 (wafer 미선택) lot 누적 wafer map 표시
  const lotMapQ = useQuery({
    queryKey: ["dd-lot-map", selectedLot, status],
    queryFn: () => fetchLotMap(selectedLot!, { status, agg: "max" }),
    enabled: !!selectedLot && !selectedKey,
  });
  const reportQ = useQuery({
    queryKey: ["dd-unit-report", selectedUnit],
    queryFn: () => fetchUnitReport(selectedUnit!),
    enabled: !!selectedUnit,
    retry: false,
  });

  useEffect(() => {
    if (selectedKey) setSearchParams({ key: selectedKey });
    else setSearchParams({});
  }, [selectedKey, setSearchParams]);
  useEffect(() => {
    setSelectedUnit(null);
  }, [selectedKey]);

  const lotTree = useMemo(() => {
    if (!lotsQ.data) return [];  // wafersQ는 옵셔널 — 없으면 lot만 펼침 없이 표시
    const wafersByLot = new Map<string, WaferSummary[]>();
    if (wafersQ.data) {
      for (const w of wafersQ.data.items) {
        const arr = wafersByLot.get(w.run_id) ?? [];
        arr.push(w);
        wafersByLot.set(w.run_id, arr);
      }
    }
    const filtered = lotsQ.data.items.filter((lot: LotItem) => {
      if (!search) return true;
      const lower = search.toLowerCase();
      if (lot.run_id.toLowerCase().includes(lower)) return true;
      const ws = wafersByLot.get(lot.run_id) ?? [];
      return ws.some((w) => w.wafer_key.toLowerCase().includes(lower));
    });
    const orderSign = lotOrder === "asc" ? 1 : -1;
    const sorted = [...filtered].sort((a, b) => {
      if (lotSort === "lot_id") return orderSign * a.run_id.localeCompare(b.run_id);
      return orderSign * (a.risk_ratio - b.risk_ratio);
    });
    // 내부(wafer)는 lot 정렬과 무관하게 항상 wafer 번호 오름차순 (#1, #2, #3, ...)
    const compareWafers = (a: WaferSummary, b: WaferSummary) =>
      Number(a.wafer_no) - Number(b.wafer_no);
    return sorted.map((lot) => ({
      lot,
      wafers: (wafersByLot.get(lot.run_id) ?? []).sort(compareWafers),
    }));
  }, [lotsQ.data, wafersQ.data, search, lotSort, lotOrder]);

  useEffect(() => {
    if (selectedKey && !selectedLot && wafersQ.data) {
      const w = wafersQ.data.items.find((x) => x.wafer_key === selectedKey);
      if (w) setSelectedLot(w.run_id);
    }
  }, [selectedKey, selectedLot, wafersQ.data]);

  // 정렬 변경 시 펼쳐둔 lot 위치가 바뀌면서 화면 밖으로 사라지지 않게 스크롤
  useEffect(() => {
    if (!selectedLot) return;
    const el = document.querySelector(`[data-lot="${selectedLot}"]`);
    el?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [lotSort, lotOrder, selectedLot]);

  const scale = triageQ.data?.scale;

  return (
    <div>
      <PageHeader
        title="Drill-down"
        status={status}
        onStatusChange={setStatus}
      />

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 sm:gap-5">
        {/* 좌: Lot 트리 */}
        <div className="lg:col-span-3">
          <Panel title="Lot / Wafer" bodyClassName="p-0">
            <div className="px-3 py-2 border-b border-brand-border space-y-2">
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="lot / wafer 검색"
                className="w-full border border-brand-border rounded bg-white text-[11px] px-2 py-1.5 focus:outline-none focus:border-brand-primary"
              />
              <div className="flex items-center gap-1.5">
                <span className="text-[10px] text-brand-textMuted">정렬</span>
                <select
                  value={lotSort}
                  onChange={(e) => {
                    const next = e.target.value as LotSort;
                    setLotSort(next);
                    // 항목 바꿀 때 합리적인 default 방향 재설정
                    setLotOrder(next === "lot_id" ? "asc" : "desc");
                  }}
                  className="flex-1 border border-brand-border rounded bg-white text-[11px] px-1.5 py-1 focus:outline-none focus:border-brand-primary"
                >
                  {LOT_SORT_OPTIONS.map((o) => (
                    <option key={o.value} value={o.value}>{o.label}</option>
                  ))}
                </select>
                <button
                  type="button"
                  onClick={() => setLotOrder((o) => (o === "asc" ? "desc" : "asc"))}
                  title={lotOrder === "asc" ? "오름차순 (클릭 시 내림차순)" : "내림차순 (클릭 시 오름차순)"}
                  className="border border-brand-border rounded bg-white text-[12px] px-2 py-1 hover:bg-brand-subtle focus:outline-none focus:border-brand-primary tabular w-7 text-center"
                >
                  {lotOrder === "asc" ? "↑" : "↓"}
                </button>
              </div>
            </div>
            <div className="max-h-[480px] lg:max-h-[680px] overflow-y-auto">
              {lotsQ.isLoading && (
                <div className="p-3 text-[11px] text-brand-textMuted">로딩…</div>
              )}
              {!lotsQ.isLoading && lotTree.length === 0 && (
                <div className="p-3 text-[11px] text-brand-textMuted">데이터 없음</div>
              )}
              {lotTree.map(({ lot, wafers }) => {
                const isLotOpen = selectedLot === lot.run_id;
                return (
                  <div
                    key={lot.run_id}
                    data-lot={lot.run_id}
                    className="border-b border-brand-border/60"
                  >
                    <button
                      onClick={() => {
                        if (isLotOpen) {
                          // lot 접기 — 안에 선택돼 있던 wafer도 함께 해제해야
                          // useEffect가 selectedKey를 보고 다시 강제로 펼치지 않음
                          setSelectedLot(null);
                          setSelectedKey(null);
                          setSearchParams({});
                        } else {
                          setSelectedLot(lot.run_id);
                        }
                      }}
                      className={`w-full flex items-center justify-between px-3 py-2 text-left text-[12px] hover:bg-brand-subtle ${
                        isLotOpen ? "bg-brand-subtle font-semibold" : ""
                      }`}
                    >
                      <span className="font-mono flex items-center gap-1 truncate">
                        <span className="text-brand-textMuted">
                          {isLotOpen ? "▼" : "▶"}
                        </span>
                        {lot.run_id}
                      </span>
                      <span className="flex items-center gap-1.5 flex-shrink-0">
                        <span
                          className={`tabular text-[11px] font-bold ${
                            lot.risk_ratio > 0.1
                              ? "text-brand-danger"
                              : lot.risk_ratio > 0.05
                              ? "text-brand-warn"
                              : "text-brand-textMuted"
                          }`}
                        >
                          {fmtPct(lot.risk_ratio)}
                        </span>
                        <span
                          className="chip chip-tbd chip-sm"
                          title="lot 위험률 통계 검정 (Fisher exact 등) 미연결 — TBD"
                        >
                          <span aria-hidden>⚠</span>
                          <span>검정 TBD</span>
                        </span>
                      </span>
                    </button>
                    {isLotOpen && (
                      <div className="bg-white">
                        {wafers.map((w) => {
                          const active = selectedKey === w.wafer_key;
                          return (
                            <button
                              key={w.wafer_key}
                              onClick={() => setSelectedKey(w.wafer_key)}
                              className={`w-full flex items-center justify-between pl-7 pr-3 py-1.5 text-left text-[11px] hover:bg-brand-subtle ${
                                active
                                  ? "bg-brand-primary/10 font-semibold"
                                  : ""
                              }`}
                            >
                              <span className="font-mono">#{w.wafer_no}</span>
                              <span
                                className={`tabular ${
                                  w.risk_ratio > 0.1
                                    ? "text-brand-danger"
                                    : w.risk_ratio > 0.05
                                    ? "text-brand-warn"
                                    : ""
                                }`}
                              >
                                {fmtPct(w.risk_ratio)}
                              </span>
                            </button>
                          );
                        })}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </Panel>
        </div>

        {/* 중: Wafer Map (선택 우선순위: wafer > lot 누적 > 안내) */}
        <div className="lg:col-span-6">
          {!selectedKey && !selectedLot && (
            <Panel title="Wafer Map">
              <div className="text-[11px] text-brand-textMuted p-6 text-center">
                ← 좌측 트리에서 lot 또는 wafer를 선택하세요
              </div>
            </Panel>
          )}

          {/* wafer 단일 선택 */}
          {selectedKey && detailQ.isLoading && (
            <Panel title="Wafer Map">
              <div className="text-[11px] text-brand-textMuted p-6 text-center">로딩…</div>
            </Panel>
          )}
          {selectedKey && detailQ.data && scale && (
            <Panel
              title={`Wafer ${detailQ.data.wafer_key}`}
              right={
                <div className="flex items-center gap-2 flex-wrap">
                  <button
                    onClick={() => setSelectedKey(null)}
                    className="text-[10px] text-white/85 hover:text-white"
                    title="lot 누적 view로 전환"
                  >
                    ↺ lot 전체 보기
                  </button>
                  <div className="inline-flex bg-brand-subtle rounded-md overflow-hidden text-[10px]">
                    {(["single", "lot"] as ViewMode[]).map((m) => (
                      <button
                        key={m}
                        onClick={() => setViewMode(m)}
                        className={`px-2.5 py-1 font-medium ${
                          viewMode === m
                            ? "bg-brand-primary text-white"
                            : "text-brand-textMuted hover:text-brand-text"
                        }`}
                      >
                        {m === "single" ? "단일" : "lot 누적"}
                      </button>
                    ))}
                  </div>
                  <span className="text-[10px] text-brand-textMuted">
                    Units {fmtInt(detailQ.data.summary?.n_units)} · 위험{" "}
                    <span className="font-bold text-brand-warn">
                      {fmtInt(detailQ.data.summary?.n_risk)}
                    </span>
                  </span>
                </div>
              }
            >
              {viewMode === "lot" && (
                <div className="text-[11px] text-brand-textMuted mb-2 px-1">
                  lot 누적 보려면 좌측에서 wafer 선택 해제 (위 ↺ 버튼)
                </div>
              )}
              <WaferMap
                dies={detailQ.data.dies}
                scale={scale}
                grid={gridQ.data}
                selectedUnit={selectedUnit}
                onSelectUnit={setSelectedUnit}
              />
            </Panel>
          )}

          {/* lot 누적 view (wafer 미선택 + lot 선택) */}
          {!selectedKey && selectedLot && lotMapQ.isLoading && (
            <Panel title="Wafer Map">
              <div className="text-[11px] text-brand-textMuted p-6 text-center">로딩…</div>
            </Panel>
          )}
          {!selectedKey && selectedLot && lotMapQ.data && scale && (
            <Panel
              title={`Lot ${selectedLot} — 누적 (max)`}
              right={
                <span className="text-[10px] text-brand-textMuted">
                  wafer {fmtInt(lotMapQ.data.n_wafers)} · die 좌표{" "}
                  {fmtInt(lotMapQ.data.n_unique_positions)}개
                </span>
              }
            >
              <div className="text-[11px] text-brand-textMuted mb-2 px-1">
                lot 내 모든 wafer를 같은 die 좌표로 겹친 max 집계 — wafer 단일 보려면 좌측 트리에서 wafer 선택
              </div>
              <WaferMap
                dies={lotMapQ.data.dies}
                scale={scale}
                grid={gridQ.data}
                selectedUnit={null}
                onSelectUnit={undefined}
              />
            </Panel>
          )}
        </div>

        {/* 우: Unit 진단 */}
        <div className="lg:col-span-3">
          <Panel title="Unit 진단">
            {!selectedUnit && (
              <div className="text-[11px] text-brand-textMuted p-3">
                wafer map의 die를 클릭하면 진단이 표시됩니다.
              </div>
            )}
            {selectedUnit && reportQ.isLoading && (
              <div className="text-[11px] text-brand-textMuted p-3">로딩…</div>
            )}
            {reportQ.data && (
              <div className="space-y-3">
                <div
                  className={`px-3 py-2 rounded-lg border ${
                    reportQ.data.verdict === "WARNING"
                      ? "bg-red-50 border-brand-danger"
                      : "bg-emerald-50 border-brand-success"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span
                      className={`text-[12px] font-bold ${
                        reportQ.data.verdict === "WARNING"
                          ? "text-brand-danger"
                          : "text-brand-success"
                      }`}
                    >
                      {reportQ.data.verdict === "WARNING"
                        ? "⚠ 위험"
                        : "✓ 정상"}
                    </span>
                    <span className="font-mono text-[10px] text-brand-textMuted">
                      {reportQ.data.ufs_serial}
                    </span>
                  </div>
                  <div className="text-[11px] mt-1 font-mono flex items-center gap-2 flex-wrap">
                    <span>pred {fmtNum(reportQ.data.pred)}</span>
                    <span className="chip chip-tbd chip-sm" title="모델 예측 신뢰구간 산출 미연결 — TBD">
                      <span aria-hidden>⚠</span>
                      <span>95% 신뢰구간 TBD</span>
                    </span>
                  </div>
                </div>

                <div className="flex flex-wrap gap-1.5">
                  {reportQ.data.is_risk && (
                    <span
                      className="chip chip-danger"
                      title="pred > p95 (status 내 상위 5%)"
                    >
                      <span aria-hidden>⚠</span>
                      <span>위험 분류</span>
                    </span>
                  )}
                  <span
                    className="chip chip-info"
                    title="status 내 pred 백분위 (실측값)"
                  >
                    <span aria-hidden>·</span>
                    <span>백분위 {fmtPct(reportQ.data.pred_rank)}</span>
                  </span>
                </div>

                <div>
                  <div className="text-[11px] font-semibold text-brand-text mb-1.5 flex items-center gap-1.5">
                    <span>주요 기여 변수 Top 20</span>
                    <span className="chip chip-tbd chip-sm" title="unit별 SHAP 산출 미연결">
                      <span aria-hidden>⚠</span>
                      <span>TBD</span>
                    </span>
                  </div>
                  <div className="tbd-block">
                    unit별 SHAP feature attribution이 아직 산출 파이프라인에 연결되지 않았습니다.
                    회귀 stage(μ) / 분류 stage(π) 모델에 SHAP을 unit 단위로 계산해 채울 예정.
                  </div>
                </div>

                <div className="border border-brand-border rounded-lg px-3 py-2 bg-brand-subtle">
                  <div className="text-[11px] font-semibold text-brand-text mb-1 flex items-center gap-1.5">
                    <span>이상도 점수</span>
                    <span className="chip chip-tbd chip-sm" title="IsolationForest / Mahalanobis 미연결">
                      <span aria-hidden>⚠</span>
                      <span>TBD</span>
                    </span>
                  </div>
                  <div className="text-[11px] text-brand-textMuted">
                    다변량 이상도 산출 (IsolationForest 또는 Mahalanobis) 미연결 — TBD.
                  </div>
                </div>

                {reportQ.data.worst_die && (
                  <div className="border border-brand-border rounded-lg px-3 py-2">
                    <div className="text-[10px] text-brand-textMuted font-semibold mb-0.5">
                      가장 위험한 die
                    </div>
                    <div className="font-mono text-[11px]">
                      ({reportQ.data.worst_die.die_x},{" "}
                      {reportQ.data.worst_die.die_y}) ·{" "}
                      <span className="font-bold text-brand-danger">
                        {fmtNum(reportQ.data.worst_die.pred)}
                      </span>
                    </div>
                  </div>
                )}

                <button className="btn btn-primary w-full text-[11px]" disabled>
                  📄 보고서 생성
                </button>
              </div>
            )}
          </Panel>
        </div>
      </div>
    </div>
  );
}
