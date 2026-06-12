/**
 * Unit 보고서 모달 — 임시 버전.
 *
 * Drilldown 페이지의 "📄 보고서 생성" 버튼 클릭 시 열린다.
 * 좌측 : 불량예측 현황 / 날짜 / 위치 스캐터 / 트렌드 / 피처 임포턴스
 * 우측 : 비정상 피처(정상 vs 이 unit) + position 비교 + wafer 내 비교
 *
 * LLM 요약 텍스트는 임시로 클라이언트에서 mock 생성. 추후
 * `/api/units/:id/llm_report` 같은 백엔드 엔드포인트로 교체.
 */
import { useEffect, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  fetchFeatureImportance,
  fetchUnitAnomalyFeatures,
  fetchUnitDetail,
  fetchUnitReport,
  fetchWaferDetail,
  fetchWaferGrid,
  type AnomalyFeature,
  type DieItem,
  type UnitReport,
  type WaferGrid,
} from "../lib/api";
import { fmtNum, fmtPct } from "../lib/format";

interface Props {
  ufsSerial: string;
  onClose: () => void;
}

export default function UnitReportModal({ ufsSerial, onClose }: Props) {
  const reportQ = useQuery({
    queryKey: ["modal-unit-report", ufsSerial],
    queryFn: () => fetchUnitReport(ufsSerial),
  });
  const anomalyQ = useQuery({
    queryKey: ["modal-unit-anomaly", ufsSerial],
    queryFn: () => fetchUnitAnomalyFeatures(ufsSerial, 6),
  });
  const unitDetailQ = useQuery({
    queryKey: ["modal-unit-detail", ufsSerial],
    queryFn: () => fetchUnitDetail(ufsSerial),
  });
  const waferKey = reportQ.data?.wafer_key;
  const waferQ = useQuery({
    queryKey: ["modal-wafer-detail", waferKey],
    queryFn: () => fetchWaferDetail(waferKey!),
    enabled: !!waferKey,
  });
  const importanceQ = useQuery({
    queryKey: ["modal-feature-importance"],
    queryFn: () => fetchFeatureImportance(8),
    staleTime: Infinity,
  });
  const gridQ = useQuery({
    queryKey: ["modal-wafer-grid"],
    queryFn: fetchWaferGrid,
    staleTime: Infinity,
  });

  // ESC 닫기
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // ── LLM 요약 (임시: 클라이언트 mock) ──
  const llmSummary = useMemo(
    () => buildMockLlmSummary(reportQ.data, anomalyQ.data?.items ?? []),
    [reportQ.data, anomalyQ.data],
  );

  const isLoading =
    reportQ.isLoading || anomalyQ.isLoading || unitDetailQ.isLoading;

  return (
    <div
      className="print-root fixed inset-0 z-50 flex items-center justify-center bg-black/40"
      onClick={onClose}
    >
      <div
        className="print-area bg-white rounded-lg shadow-2xl w-[1100px] max-w-[95vw] max-h-[92vh] overflow-hidden flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* 헤더 */}
        <div className="px-4 py-2.5 border-b border-brand-border flex items-center justify-between bg-brand-subtle">
          <div className="flex items-center gap-2">
            <span className="text-[14px] font-bold text-brand-text">
              📄 Unit 진단 보고서
            </span>
            <span className="font-mono text-[11px] text-brand-textMuted">
              {ufsSerial}
            </span>
            <span className="chip chip-tbd chip-sm" title="LLM 자동 요약 (임시 mock — 추후 API 연결)">
              <span aria-hidden>✨</span>
              <span>LLM 요약 (임시)</span>
            </span>
          </div>
          <button
            onClick={onClose}
            className="no-print text-brand-textMuted hover:text-brand-text text-[14px] leading-none px-2"
            aria-label="닫기"
          >
            ✕
          </button>
        </div>

        {/* 본문 */}
        {isLoading ? (
          <div className="flex-1 flex items-center justify-center text-[12px] text-brand-textMuted">
            보고서 생성 중…
          </div>
        ) : (
          <div className="print-body flex-1 overflow-auto p-4 grid grid-cols-2 gap-4">
            {/* 좌측 */}
            <div className="space-y-3">
              <SectionVerdict report={reportQ.data} llm={llmSummary.verdict} />
              <SectionMeta report={reportQ.data} />
              <SectionWaferLocation
                report={reportQ.data}
                wafer={waferQ.data}
                grid={gridQ.data}
                ufsSerial={ufsSerial}
              />
              <SectionImportance items={importanceQ.data?.items ?? []} />
            </div>

            {/* 우측 */}
            <div className="space-y-3">
              <SectionAnomalyCompare items={anomalyQ.data?.items ?? []} />
              <SectionPositionCompare
                ufsSerial={ufsSerial}
                detail={unitDetailQ.data}
                wafer={waferQ.data}
              />
              <SectionLlmNarrative llm={llmSummary} />
            </div>
          </div>
        )}

        {/* 푸터 */}
        <div className="no-print px-4 py-2 border-t border-brand-border flex items-center justify-between bg-brand-subtle">
          <span className="text-[10px] text-brand-textMuted">
            ※ 본 보고서는 모델 산출물 기반 자동 요약입니다 (LLM 부분은 임시 mock).
          </span>
          <div className="flex gap-2">
            <button
              className="btn btn-ghost text-[11px]"
              onClick={() => window.print()}
            >
              인쇄
            </button>
            <button className="btn btn-primary text-[11px]" onClick={onClose}>
              닫기
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ────────────── 좌측 섹션 ────────────── */

function SectionVerdict({
  report,
  llm,
}: {
  report?: UnitReport;
  llm: string;
}) {
  if (!report) return null;
  const isWarn = report.verdict === "WARNING";
  return (
    <div
      className={`rounded-lg border p-3 ${
        isWarn
          ? "bg-red-50 border-brand-danger"
          : "bg-emerald-50 border-brand-success"
      }`}
    >
      <div className="flex items-center justify-between mb-1.5">
        <span
          className={`text-[20px] font-black leading-none ${
            isWarn ? "text-brand-danger" : "text-brand-success"
          }`}
        >
          {isWarn ? "⚠ 위험" : "✓ 정상"}
        </span>
        <span className="text-[11px] font-mono text-brand-textMuted">
          pred <span className="font-bold text-brand-text">{fmtNum(report.pred)}</span>
          {" · "}
          상위 {fmtPct(report.pred_rank)}
        </span>
      </div>
      <div className="text-[11px] text-brand-text leading-relaxed">{llm}</div>
    </div>
  );
}

function SectionMeta({ report }: { report?: UnitReport }) {
  if (!report) return null;
  return (
    <div className="border border-brand-border rounded-lg p-3">
      <div className="text-[11px] font-semibold text-brand-text mb-1.5">기본 정보</div>
      <table className="spotfire">
        <tbody>
          <tr>
            <th>검사 일자</th>
            <td className="font-mono">{report.inspected_date}</td>
          </tr>
          <tr>
            <th>Wafer</th>
            <td className="font-mono">{report.wafer_key}</td>
          </tr>
          <tr>
            <th>Health (실측)</th>
            <td className="font-mono tabular">
              {report.health == null ? "—" : fmtNum(report.health)}
            </td>
          </tr>
          <tr>
            <th>Threshold</th>
            <td className="font-mono tabular">{fmtNum(report.threshold)}</td>
          </tr>
          <tr>
            <th>π / μ</th>
            <td className="font-mono tabular">
              {fmtNum(report.pi_mean, 3)} / {fmtNum(report.mu_mean)}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

function SectionWaferLocation({
  report,
  wafer,
  grid,
  ufsSerial,
}: {
  report?: UnitReport;
  wafer?: { dies: DieItem[] };
  grid?: WaferGrid;
  ufsSerial: string;
}) {
  if (!report) return null;
  const dies = wafer?.dies ?? [];

  // WaferMap과 동일한 좌표계: bounds + 정사각 cell + Y flip 없음 + 원형 클리핑
  let bounds: WaferGrid["bounds"];
  let mask: [number, number][];
  if (grid) {
    bounds = grid.bounds;
    mask = grid.mask;
  } else if (dies.length > 0) {
    const xs = dies.map((d) => d.die_x);
    const ys = dies.map((d) => d.die_y);
    bounds = {
      x_min: Math.min(...xs),
      x_max: Math.max(...xs),
      y_min: Math.min(...ys),
      y_max: Math.max(...ys),
    };
    mask = dies.map((d) => [d.die_x, d.die_y] as [number, number]);
  } else {
    return (
      <div className="border border-brand-border rounded-lg p-3">
        <div className="text-[11px] font-semibold text-brand-text mb-1.5">
          Wafer 내 위치 ({report.wafer_key})
        </div>
        <div className="text-[11px] text-brand-textMuted">데이터 없음</div>
      </div>
    );
  }

  const VB = 1000;
  const margin = 6;
  const inner = VB - margin * 2;
  const xRange = bounds.x_max - bounds.x_min + 1;
  const yRange = bounds.y_max - bounds.y_min + 1;
  const cellW = inner / xRange;
  const cellH = inner / yRange;
  const centerX = (bounds.x_min + bounds.x_max) / 2;
  const centerY = (bounds.y_min + bounds.y_max) / 2;
  const cx = VB / 2;
  const cy = VB / 2;
  const radius = inner / 2;

  const dieMap = new Map<string, DieItem>();
  for (const d of dies) dieMap.set(`${d.die_x},${d.die_y}`, d);

  const unitDies = dies.filter((d) => d.ufs_serial === ufsSerial);

  return (
    <div className="border border-brand-border rounded-lg p-3">
      <div className="text-[11px] font-semibold text-brand-text mb-1.5">
        Wafer 내 위치 ({report.wafer_key})
      </div>
      <div className="w-full max-w-[260px] mx-auto">
        <svg
          viewBox={`0 0 ${VB} ${VB}`}
          preserveAspectRatio="xMidYMid meet"
          className="w-full h-auto block bg-white rounded border border-brand-border"
        >
          <defs>
            <clipPath id={`waferCircleModal-${report.wafer_key}`}>
              <circle cx={cx} cy={cy} r={radius} />
            </clipPath>
          </defs>
          <circle cx={cx} cy={cy} r={radius} fill="#fafafa" stroke="#cbd5e1" strokeWidth={2} />
          <g clipPath={`url(#waferCircleModal-${report.wafer_key})`}>
            {mask.map(([dx, dy]) => {
              const die = dieMap.get(`${dx},${dy}`);
              const x = cx + (dx - centerX) * cellW - cellW / 2;
              const y = cy + (dy - centerY) * cellH - cellH / 2;
              const isUnit = die?.ufs_serial === ufsSerial;
              const danger = die ? die.pred >= report.threshold : false;
              const fill = !die
                ? "#f1f5f9"
                : isUnit
                ? "#dc2626"
                : danger
                ? "#f59e0b"
                : "#cbd5e1";
              return (
                <rect
                  key={`${dx}-${dy}`}
                  x={x}
                  y={y}
                  width={Math.max(0, cellW - 0.6)}
                  height={Math.max(0, cellH - 0.6)}
                  fill={fill}
                  stroke={isUnit ? "#7f1d1d" : "rgba(15,23,42,0.06)"}
                  strokeWidth={isUnit ? 2 : 0.4}
                />
              );
            })}
          </g>
          <circle cx={cx} cy={cy} r={radius} fill="none" stroke="#94a3b8" strokeWidth={2} />
          <rect
            x={cx - 18}
            y={cy + radius - 6}
            width={36}
            height={8}
            fill="#fff"
            stroke="#94a3b8"
            strokeWidth={1}
          />
        </svg>
      </div>
      <div className="text-[10px] text-brand-textMuted mt-1 flex gap-3 flex-wrap justify-center">
        <span className="text-brand-danger">■ 이 unit ({unitDies.length} die)</span>
        <span className="text-amber-600">■ 위험 die</span>
        <span>■ 정상 die</span>
      </div>
    </div>
  );
}

function SectionImportance({
  items,
}: {
  items: { feature: string; total_gain: number }[];
}) {
  if (items.length === 0) {
    return (
      <div className="border border-brand-border rounded-lg p-3">
        <div className="text-[11px] font-semibold text-brand-text mb-1">
          모델 피처 임포턴스 (전역)
        </div>
        <div className="text-[11px] text-brand-textMuted">데이터 없음</div>
      </div>
    );
  }
  const max = Math.max(...items.map((i) => i.total_gain));
  return (
    <div className="border border-brand-border rounded-lg p-3">
      <div className="text-[11px] font-semibold text-brand-text mb-1.5">
        모델 피처 임포턴스 (전역 Top {items.length})
      </div>
      <div className="space-y-1">
        {items.map((it) => (
          <div key={it.feature} className="flex items-center gap-1.5">
            <span className="font-mono text-[10px] w-12 shrink-0 text-brand-text">
              {it.feature}
            </span>
            <div className="flex-1 h-2.5 bg-brand-subtle rounded-sm relative">
              <div
                className="absolute top-0 bottom-0 left-0 bg-brand-primary rounded-sm"
                style={{ width: `${(it.total_gain / max) * 100}%` }}
              />
            </div>
            <span className="text-[10px] tabular w-10 text-right text-brand-textMuted">
              {fmtNum(it.total_gain, 0)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ────────────── 우측 섹션 ────────────── */

function SectionAnomalyCompare({ items }: { items: AnomalyFeature[] }) {
  return (
    <div className="border border-brand-border rounded-lg p-3">
      <div className="text-[11px] font-semibold text-brand-text mb-1.5">
        비정상 피처 비교 (정상 unit 평균 vs 이 unit)
      </div>
      {items.length === 0 ? (
        <div className="text-[11px] text-brand-textMuted">비정상 피처 없음</div>
      ) : (
        <div className="space-y-2">
          {items.map((it) => {
            const minV = Math.min(it.normal_mean, it.unit_value);
            const maxV = Math.max(it.normal_mean, it.unit_value);
            const range = Math.max(Math.abs(maxV - minV), Math.abs(maxV) * 0.1, 1e-9);
            const axisMin = minV - range * 0.2;
            const axisMax = maxV + range * 0.2;
            const span = axisMax - axisMin || 1;
            const normalLen = ((it.normal_mean - axisMin) / span) * 100;
            const unitLen = ((it.unit_value - axisMin) / span) * 100;
            const isHigher = it.unit_value > it.normal_mean;
            const sev =
              it.z_score >= 3
                ? "text-brand-danger"
                : it.z_score >= 2
                ? "text-brand-warn"
                : "text-brand-textMuted";
            return (
              <div key={it.feature} className="border border-brand-border rounded p-2">
                <div className="flex items-center justify-between mb-1">
                  <span className="font-mono text-[11px] font-semibold">{it.feature}</span>
                  <div className="flex items-center gap-1.5">
                    <span className="text-[10px] text-brand-textMuted">
                      {isHigher ? "▲ 높음" : "▼ 낮음"}
                    </span>
                    <span className={`tabular text-[14px] font-black ${sev}`}>
                      z={it.z_score.toFixed(1)}
                    </span>
                  </div>
                </div>
                <div className="flex items-center gap-1.5 mb-0.5">
                  <span className="text-[9px] text-brand-textMuted w-8 shrink-0">정상</span>
                  <div className="flex-1 h-2.5 bg-brand-subtle rounded-sm relative">
                    <div
                      className="absolute top-0 bottom-0 left-0 bg-emerald-500 rounded-sm"
                      style={{ width: `${normalLen}%` }}
                    />
                  </div>
                  <span className="text-[9px] tabular w-12 text-right">{fmtNum(it.normal_mean, 2)}</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="text-[9px] font-semibold w-8 shrink-0">unit</span>
                  <div className="flex-1 h-2.5 bg-brand-subtle rounded-sm relative">
                    <div
                      className={`absolute top-0 bottom-0 left-0 rounded-sm ${
                        isHigher ? "bg-brand-danger" : "bg-brand-primary"
                      }`}
                      style={{ width: `${unitLen}%` }}
                    />
                  </div>
                  <span
                    className={`text-[9px] tabular w-12 text-right font-semibold ${
                      isHigher ? "text-brand-danger" : "text-brand-primary"
                    }`}
                  >
                    {fmtNum(it.unit_value, 2)}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function SectionPositionCompare({
  ufsSerial,
  detail,
  wafer,
}: {
  ufsSerial: string;
  detail?: { dies: DieItem[] };
  wafer?: { dies: DieItem[] };
}) {
  // unit 내 die들 (position별 pred)
  const unitDies = (detail?.dies ?? [])
    .filter((d): d is DieItem & { position: number } => d.position != null)
    .slice()
    .sort((a, b) => a.position - b.position);
  // wafer의 정상 unit 평균을 position별로
  const waferDies = wafer?.dies ?? [];
  const byPos = new Map<number, number[]>();
  for (const d of waferDies) {
    if (d.position == null) continue;
    if (d.ufs_serial === ufsSerial) continue;
    const arr = byPos.get(d.position) ?? [];
    arr.push(d.pred);
    byPos.set(d.position, arr);
  }
  const positions = [1, 2, 3, 4];
  const max = Math.max(
    ...unitDies.map((d) => d.pred),
    ...Array.from(byPos.values()).flatMap((vs) => vs),
    0.001,
  );

  return (
    <div className="border border-brand-border rounded-lg p-3">
      <div className="text-[11px] font-semibold text-brand-text mb-1.5">
        Position별 비교 (이 unit vs wafer 평균, ppm)
      </div>
      {unitDies.length === 0 ? (
        <div className="text-[11px] text-brand-textMuted">데이터 없음</div>
      ) : (
        <div className="space-y-1.5">
          {positions.map((p) => {
            const u = unitDies.find((d) => d.position === p);
            const others = byPos.get(p) ?? [];
            const otherMean =
              others.length > 0 ? others.reduce((a, b) => a + b, 0) / others.length : 0;
            const uPred = u?.pred ?? 0;
            const isWorse = uPred > otherMean;
            const uPpm = uPred * 1e6;
            const meanPpm = otherMean * 1e6;
            const maxPpm = max * 1e6;
            return (
              <div key={p} className="border border-brand-border rounded p-1.5">
                <div className="flex items-center justify-between text-[10px] mb-0.5">
                  <span className="font-semibold">Position {p}</span>
                  <span className={`tabular ${isWorse ? "text-brand-danger" : "text-brand-textMuted"}`}>
                    {isWorse ? "▲" : "▼"} {(uPpm - meanPpm).toFixed(0)} ppm
                  </span>
                </div>
                <div className="flex items-center gap-1.5 mb-0.5">
                  <span className="text-[9px] w-8 text-brand-textMuted shrink-0">평균</span>
                  <div className="flex-1 h-2 bg-brand-subtle rounded-sm relative">
                    <div
                      className="absolute top-0 bottom-0 left-0 bg-emerald-500 rounded-sm"
                      style={{ width: `${(meanPpm / Math.max(maxPpm, 1e-9)) * 100}%` }}
                    />
                  </div>
                  <span className="text-[9px] tabular w-14 text-right font-mono">{meanPpm.toFixed(0)}</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="text-[9px] w-8 font-semibold shrink-0">unit</span>
                  <div className="flex-1 h-2 bg-brand-subtle rounded-sm relative">
                    <div
                      className={`absolute top-0 bottom-0 left-0 rounded-sm ${
                        isWorse ? "bg-brand-danger" : "bg-brand-primary"
                      }`}
                      style={{ width: `${(uPpm / Math.max(maxPpm, 1e-9)) * 100}%` }}
                    />
                  </div>
                  <span
                    className={`text-[9px] tabular w-14 text-right font-semibold font-mono ${
                      isWorse ? "text-brand-danger" : "text-brand-primary"
                    }`}
                  >
                    {uPpm.toFixed(0)}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function SectionLlmNarrative({
  llm,
}: {
  llm: { bullets: string[]; recommendation: string };
}) {
  return (
    <div className="border border-brand-primary/40 bg-brand-subtle rounded-lg p-3">
      <div className="text-[11px] font-semibold text-brand-text mb-1.5 flex items-center gap-1.5">
        <span>✨ LLM 종합 진단</span>
        <span className="chip chip-tbd chip-sm" title="임시 mock — 추후 백엔드 LLM API 연결">
          <span aria-hidden>⚠</span>
          <span>임시</span>
        </span>
      </div>
      <ul className="space-y-1 text-[11px] text-brand-text mb-2">
        {llm.bullets.map((b, i) => (
          <li key={i} className="flex gap-1.5 leading-snug">
            <span className="text-brand-primary">•</span>
            <span>{b}</span>
          </li>
        ))}
      </ul>
      <div className="text-[11px] text-brand-text bg-white border border-brand-border rounded p-2">
        <span className="font-semibold">권고: </span>
        {llm.recommendation}
      </div>
    </div>
  );
}

/* ────────────── LLM mock 생성기 (임시) ────────────── */

function buildMockLlmSummary(
  report: UnitReport | undefined,
  anomalies: AnomalyFeature[],
): { verdict: string; bullets: string[]; recommendation: string } {
  if (!report) {
    return { verdict: "데이터 로딩 중", bullets: [], recommendation: "—" };
  }
  const isWarn = report.verdict === "WARNING";
  const top = anomalies.slice(0, 3);
  const topNames = top.map((a) => a.feature).join(", ");
  const bullets: string[] = [];
  bullets.push(
    `예측 health 값은 ${fmtNum(report.pred)}로 임계값(${fmtNum(report.threshold)}) 대비 ${
      isWarn ? "초과" : "이내"
    }입니다 (상위 ${fmtPct(report.pred_rank)}).`,
  );
  bullets.push(
    `분류 확률 π=${fmtNum(report.pi_mean, 3)}, 회귀 추정 μ=${fmtNum(
      report.mu_mean,
    )}로 ZIT 모델 두 경로 모두 ${isWarn ? "위험 신호를 보입니다" : "안정적입니다"}.`,
  );
  if (top.length > 0) {
    bullets.push(
      `주요 비정상 피처: ${topNames}. 정상 unit 분포 대비 z-score ${top[0].z_score.toFixed(
        1,
      )}로 ${top[0].unit_value > top[0].normal_mean ? "상한 이탈" : "하한 이탈"} 패턴입니다.`,
    );
  }
  if (report.worst_die) {
    bullets.push(
      `Worst die는 (${report.worst_die.die_x}, ${report.worst_die.die_y}) 위치이며 die-level pred ${fmtNum(
        report.worst_die.pred,
      )}로 unit 평균을 견인하고 있습니다.`,
    );
  }
  const recommendation = isWarn
    ? `이 unit은 출하 전 정밀 검사를 권장합니다. 특히 ${
        topNames || "비정상 피처"
      }의 측정 재확인과 동일 wafer 내 인접 die의 패턴 점검을 진행하세요.`
    : `현재 지표상 출하 가능 수준입니다. 다만 인접 wafer/lot의 트렌드 모니터링은 유지 권장합니다.`;
  return {
    verdict: bullets[0],
    bullets,
    recommendation,
  };
}
