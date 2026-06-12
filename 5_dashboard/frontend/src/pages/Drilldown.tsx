/**
 * Drill-down 페이지 (= 사이드바 "다이 레벨 정밀 분석 > 다이 차원").
 *
 * 상단  : 주별 생산량 (용인 이식, public/dashboard_dates.csv)
 * 좌측  : Lot 트리 (lot 그룹 → wafer 리스트)
 * 중앙  : Wafer Map (단일/lot 누적 토글)
 * 우측  : Unit Diagnosis (verdict + SHAP + Anomaly + 신뢰구간)
 *
 * SHAP / Anomaly / 신뢰구간 / lot 검정은 현재 산출물 미연결 — TBD placeholder로 비워둠.
 * 웨이퍼 차원 / 로트 차원은 별도 라우트(/drilldown/wafer, /drilldown/lot)에 빈 페이지로 분리.
 */
import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";
import {
  fetchLots,
  fetchTriage,
  fetchUnitDetail,
  fetchUnitReport,
  fetchWaferDetail,
  fetchWaferGrid,
  fetchWafers,
  fetchUnitAnomalyFeatures,
  type LotItem,
  type StatusFilter,
  type WaferSummary,
} from "../lib/api";
import WaferMap from "../components/WaferMap";
import PageHeader from "../components/PageHeader";
import Panel from "../components/Panel";
import UnitReportModal from "../components/UnitReportModal";
import WeeklyProductionChart from "../components/WeeklyProductionChart";
import { fmtInt, fmtNum, fmtPct } from "../lib/format";

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
  const [search, setSearch] = useState("");
  const [reportOpen, setReportOpen] = useState(false);
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
  const anomalyQ = useQuery({
    queryKey: ["dd-unit-anomaly", selectedUnit],
    queryFn: () => fetchUnitAnomalyFeatures(selectedUnit!, 5),
    enabled: !!selectedUnit,
  });
  const reportQ = useQuery({
    queryKey: ["dd-unit-report", selectedUnit],
    queryFn: () => fetchUnitReport(selectedUnit!),
    enabled: !!selectedUnit,
    retry: false,
  });
  const unitDetailQ = useQuery({
    queryKey: ["dd-unit-detail", selectedUnit],
    queryFn: () => fetchUnitDetail(selectedUnit!),
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
        title="다이 차원"
        subtitle="주별 생산량 추이 + Lot/Wafer 트리 → Wafer Map → Unit 진단 (3-pane drill-down)"
        status={status}
        onStatusChange={setStatus}
      />

      {/* 주별 생산량 (용인 이식) */}
      <div className="mb-4 sm:mb-5">
        <WeeklyProductionChart />
      </div>

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
                    title="wafer 선택 해제"
                  >
                    ↺ lot 전체 보기
                  </button>
                  <span className="text-[10px] text-brand-textMuted">
                    Units {fmtInt(detailQ.data.summary?.n_units)} · 위험{" "}
                    <span className="font-bold text-brand-warn">
                      {fmtInt(detailQ.data.summary?.n_risk)}
                    </span>
                  </span>
                </div>
              }
            >
              <WaferMap
                dies={detailQ.data.dies}
                scale={scale}
                grid={gridQ.data}
                selectedUnit={selectedUnit}
                onSelectUnit={setSelectedUnit}
              />
            </Panel>
          )}

          {/* Die 단위 위험도 — wafer map 아래, 선택한 unit의 die 4개 게이지 */}
          {selectedKey && selectedUnit && unitDetailQ.data && unitDetailQ.data.dies.length > 0 && scale && (
            <Panel
              title="Die 단위 위험도"
              className="mt-4 sm:mt-5"
              right={
                <div className="flex items-center gap-3 text-[11px]">
                  <span className="font-mono text-white/85">
                    unit <span className="font-bold">{selectedUnit}</span>
                  </span>
                  <span className="text-white/70">
                    τ = {(scale.risk_threshold * 1e6).toFixed(0)} ppm
                  </span>
                </div>
              }
            >
              <div className="space-y-2">
                {[...unitDetailQ.data.dies]
                  .sort((a, b) => (a.position ?? 99) - (b.position ?? 99))
                  .map((d, i) => {
                    const ppm = d.pred * 1e6;
                    const tau = scale.risk_threshold;
                    // 3단계 분류: 정상(<0.95τ) / 주의(0.95τ~τ) / 위험(≥τ)
                    const cautionMin = tau * 0.95;
                    const tier: "normal" | "caution" | "risk" =
                      d.pred >= tau ? "risk" : d.pred >= cautionMin ? "caution" : "normal";
                    // 게이지: 정상=0~40%, 주의=40~50%, 위험=50~100% (임계값 50% 지점 고정)
                    let pct: number;
                    if (tier === "risk") {
                      const denom = Math.max(scale.pred_max - tau, 1e-9);
                      pct = 50 + Math.min(50, ((d.pred - tau) / denom) * 50);
                    } else if (tier === "caution") {
                      const denom = Math.max(tau - cautionMin, 1e-9);
                      pct = 40 + ((d.pred - cautionMin) / denom) * 10;
                    } else {
                      pct = Math.max(0, (d.pred / Math.max(cautionMin, 1e-9)) * 40);
                    }
                    const barColor =
                      tier === "risk" ? "bg-brand-danger"
                      : tier === "caution" ? "bg-amber-400"
                      : "bg-brand-primary";
                    const labelColor =
                      tier === "risk" ? "text-brand-danger"
                      : tier === "caution" ? "text-amber-600"
                      : "text-brand-text";
                    const chipClass =
                      tier === "risk" ? "bg-red-50 text-brand-danger"
                      : tier === "caution" ? "bg-amber-50 text-amber-700"
                      : "bg-emerald-50 text-brand-success";
                    const chipLabel =
                      tier === "risk" ? "위험" : tier === "caution" ? "주의" : "정상";
                    const borderColor =
                      tier === "risk" ? "#dc2626"
                      : tier === "caution" ? "#f59e0b"
                      : "#10b981";
                    const posBg =
                      tier === "risk" ? "bg-red-50"
                      : tier === "caution" ? "bg-amber-50"
                      : "bg-emerald-50";
                    return (
                      <div
                        key={`${d.die_x}-${d.die_y}-${i}`}
                        className="rounded-lg border-2 px-3 py-2.5 flex items-center gap-3"
                        style={{ borderColor }}
                      >
                        {/* 포지션 뱃지 */}
                        <div className={`shrink-0 w-12 h-12 rounded-lg flex flex-col items-center justify-center ${posBg}`} style={{ border: `2px solid ${borderColor}` }}>
                          <span className="text-[10px] font-semibold text-brand-textMuted leading-none">POS</span>
                          <span className="text-[22px] font-black leading-none" style={{ color: borderColor }}>
                            {d.position ?? "?"}
                          </span>
                        </div>

                        {/* 게이지 + 수치 */}
                        <div className="flex-1 flex flex-col gap-1 min-w-0">
                          <div className="flex items-center justify-between">
                            <span className="font-mono text-[11px] text-brand-textMuted">
                              ({d.die_x}, {d.die_y})
                            </span>
                            <div className="flex items-center gap-1.5">
                              <span className={`font-mono tabular text-[15px] font-black ${labelColor}`}>
                                {ppm.toFixed(0)} ppm
                              </span>
                              <span className={`text-[11px] px-1.5 py-0.5 rounded font-semibold ${chipClass}`}>
                                {chipLabel}
                              </span>
                            </div>
                          </div>
                          <div className="h-4 bg-brand-subtle rounded-full relative overflow-hidden border border-brand-border">
                            <div
                              className="absolute top-0 bottom-0 border-l border-dashed border-brand-danger/60 z-10"
                              style={{ left: "50%" }}
                            />
                            <div
                              className={`absolute top-0 bottom-0 left-0 rounded-full transition-all ${barColor}`}
                              style={{ width: `${pct}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    );
                  })}
              </div>
              <div className="text-[10px] text-brand-textMuted mt-2 px-1 leading-snug">
                점선 = 임계값 (게이지 50% 지점). 3단계 — <span className="text-brand-primary font-semibold">정상</span> (&lt;0.95τ) · <span className="text-amber-600 font-semibold">주의</span> (0.95τ~τ) · <span className="text-brand-danger font-semibold">위험</span> (≥τ).
              </div>
            </Panel>
          )}

          {/* lot만 선택, wafer 미선택 → wafer 선택 안내 */}
          {!selectedKey && selectedLot && (
            <Panel title={`Lot ${selectedLot}`}>
              <div className="text-[12px] text-brand-textMuted p-6 text-center">
                좌측 트리에서 <strong className="text-brand-text">wafer를 선택</strong>하면 wafer map과 비정상 변수 분석이 표시됩니다.
              </div>
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
                      className={`text-[22px] font-black leading-none ${
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

                {reportQ.data.is_risk && (
                  <div className="flex flex-wrap gap-1.5">
                    <span
                      className="chip chip-danger"
                      title="pred > p95 (status 내 상위 5%)"
                    >
                      <span aria-hidden>⚠</span>
                      <span>위험 분류</span>
                    </span>
                  </div>
                )}

                <div>
                  <div className="text-[11px] font-semibold text-brand-text mb-1">
                    정상 unit 대비 비정상 변수 Top 5
                  </div>
                  <div className="text-[16px] text-brand-textMuted mb-2 font-mono bg-brand-subtle rounded px-2 py-1">
                    z = |unit값 − 정상평균| / 정상std
                  </div>
                  {anomalyQ.isLoading && (
                    <div className="text-[11px] text-brand-textMuted p-2">로딩…</div>
                  )}
                  {anomalyQ.data && anomalyQ.data.items.length > 0 && (
                    <div className="space-y-2">
                      {anomalyQ.data.items.map((it) => {
                        const minV = Math.min(it.normal_mean, it.unit_value);
                        const maxV = Math.max(it.normal_mean, it.unit_value);
                        const range = Math.max(Math.abs(maxV - minV), Math.abs(maxV) * 0.1, 1e-9);
                        const axisMin = minV - range * 0.15;
                        const axisMax = maxV + range * 0.15;
                        const span = axisMax - axisMin || 1;
                        const normalLen = ((it.normal_mean - axisMin) / span) * 100;
                        const unitLen = ((it.unit_value - axisMin) / span) * 100;
                        const isHigher = it.unit_value > it.normal_mean;
                        const zSeverity =
                          it.z_score >= 3 ? "text-brand-danger" : it.z_score >= 2 ? "text-brand-warn" : "text-brand-textMuted";
                        return (
                          <div key={it.feature} className="border border-brand-border rounded-md p-2">
                            <div className="flex items-center justify-between mb-1.5">
                              <span className="font-mono text-[11px] font-semibold text-brand-text">{it.feature}</span>
                              <div className="flex items-center gap-1.5">
                                <span className={`text-[10px] text-brand-textMuted`}>
                                  {isHigher ? "▲ 높음" : "▼ 낮음"}
                                </span>
                                <span className={`tabular text-[18px] font-black leading-none ${zSeverity}`}>
                                  z={it.z_score.toFixed(1)}
                                </span>
                              </div>
                            </div>
                            {/* 정상 평균 막대 (위) */}
                            <div className="flex items-center gap-1.5 mb-0.5">
                              <span className="text-[9px] text-brand-textMuted w-8 shrink-0">정상</span>
                              <div className="flex-1 h-3 bg-brand-subtle rounded-sm relative">
                                <div
                                  className="absolute top-0 bottom-0 left-0 bg-emerald-500 rounded-sm"
                                  style={{ width: `${normalLen}%` }}
                                />
                              </div>
                              <span className="text-[9px] text-brand-textMuted tabular w-12 shrink-0 text-right">
                                {fmtNum(it.normal_mean, 2)}
                              </span>
                            </div>
                            {/* 이 unit 막대 (아래) */}
                            <div className="flex items-center gap-1.5">
                              <span className="text-[9px] text-brand-text font-semibold w-8 shrink-0">unit</span>
                              <div className="flex-1 h-3 bg-brand-subtle rounded-sm relative">
                                <div
                                  className={`absolute top-0 bottom-0 left-0 rounded-sm ${isHigher ? "bg-brand-danger" : "bg-brand-primary"}`}
                                  style={{ width: `${unitLen}%` }}
                                />
                              </div>
                              <span className={`text-[9px] tabular w-12 shrink-0 text-right font-semibold ${isHigher ? "text-brand-danger" : "text-brand-primary"}`}>
                                {fmtNum(it.unit_value, 2)}
                              </span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                  {anomalyQ.data && anomalyQ.data.items.length === 0 && (
                    <div className="text-[11px] text-brand-textMuted p-2">비정상 변수 없음</div>
                  )}
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


                <button
                  className="btn btn-primary w-full text-[11px]"
                  onClick={() => setReportOpen(true)}
                  disabled={!selectedUnit}
                >
                  📄 보고서 생성
                </button>
              </div>
            )}
          </Panel>
        </div>
      </div>

      {reportOpen && selectedUnit && (
        <UnitReportModal
          ufsSerial={selectedUnit}
          onClose={() => setReportOpen(false)}
        />
      )}
    </div>
  );
}
