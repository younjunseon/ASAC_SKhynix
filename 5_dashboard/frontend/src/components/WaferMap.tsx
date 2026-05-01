import { useMemo } from "react";
import type { DieItem, WaferGrid } from "../lib/api";
import { COLOR_LEGEND_GRADIENT, predColor } from "../lib/colors";

interface Props {
  dies: DieItem[];
  scale: { pred_min: number; pred_max: number; risk_threshold: number };
  /** 모든 wafer 좌표 union — wafer 외형 마스크 */
  grid?: WaferGrid;
  selectedUnit?: string | null;
  onSelectUnit?: (ufsSerial: string) => void;
  title?: string;
}

/**
 * Wafer 히트맵.
 * - SVG viewBox 사용 → 부모 폭에 자동 scale (반응형)
 * - 모든 wafer의 die 좌표 union을 background mask로 → 빈 die도 회색 cell로 표시
 * - cell 종횡비를 데이터 비율에 맞춰 분리 (정사각형 X)
 */
export default function WaferMap({
  dies, scale, grid, selectedUnit, onSelectUnit, title,
}: Props) {
  const layout = useMemo(() => {
    let bounds: WaferGrid["bounds"];
    let mask: [number, number][];

    if (grid) {
      bounds = grid.bounds;
      mask = grid.mask;
    } else if (dies.length > 0) {
      const xs = dies.map((d) => d.die_x);
      const ys = dies.map((d) => d.die_y);
      bounds = {
        x_min: Math.min(...xs), x_max: Math.max(...xs),
        y_min: Math.min(...ys), y_max: Math.max(...ys),
      };
      mask = dies.map((d) => [d.die_x, d.die_y] as [number, number]);
    } else {
      return null;
    }

    const xRange = bounds.x_max - bounds.x_min + 1;
    const yRange = bounds.y_max - bounds.y_min + 1;
    return { bounds, mask, xRange, yRange };
  }, [dies, grid]);

  if (!layout || dies.length === 0)
    return (
      <div className="text-brand-textMuted text-[11px] p-3 text-center">
        표시할 die가 없습니다.
      </div>
    );

  const dieMap = new Map<string, DieItem>();
  for (const d of dies) dieMap.set(`${d.die_x},${d.die_y}`, d);

  // viewBox를 1000×1000 정사각형으로 고정. 모든 좌표를 그 안에 정규화.
  const VB = 1000;
  const margin = 6;
  const inner = VB - margin * 2;
  const { mask, xRange, yRange } = layout;
  const cellWidth = inner / xRange;
  const cellHeight = inner / yRange;
  const centerX = (layout.bounds.x_min + layout.bounds.x_max) / 2;
  const centerY = (layout.bounds.y_min + layout.bounds.y_max) / 2;
  const cx = VB / 2;
  const cy = VB / 2;
  const radius = inner / 2;

  return (
    <div className="w-full">
      {title && (
        <div className="text-[12px] font-semibold text-brand-text mb-1.5">{title}</div>
      )}
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto] gap-3 items-start">
        {/* SVG — 부모 폭에 맞춰 자동 scale */}
        <div className="relative w-full max-w-[560px] mx-auto">
          <svg
            viewBox={`0 0 ${VB} ${VB}`}
            preserveAspectRatio="xMidYMid meet"
            className="w-full h-auto block bg-white rounded-lg border border-brand-border"
          >
            <defs>
              <clipPath id="waferCircle">
                <circle cx={cx} cy={cy} r={radius} />
              </clipPath>
            </defs>
            <circle cx={cx} cy={cy} r={radius} fill="#fafafa" stroke="#cbd5e1" strokeWidth={1.5} />
            <g clipPath="url(#waferCircle)">
              {mask.map(([dx, dy]) => {
                const die = dieMap.get(`${dx},${dy}`);
                const x = cx + (dx - centerX) * cellWidth - cellWidth / 2;
                const y = cy + (dy - centerY) * cellHeight - cellHeight / 2;
                const isSelected = die?.ufs_serial === selectedUnit;
                const fill = die
                  ? predColor(die.pred, scale.pred_min, scale.pred_max, scale.risk_threshold)
                  : "#f1f5f9";

                return (
                  <g key={`${dx}-${dy}`}>
                    <title>
                      {die
                        ? `${die.run_wf_xy}\npred=${(die.pred * 1e6).toFixed(0)} ppm  π=${die.pi.toFixed(3)}  μ=${(die.mu * 1e6).toFixed(0)} ppm${die.ufs_serial ? `\nunit=${die.ufs_serial}` : ""}`
                        : `(${dx}, ${dy}) — die 없음`}
                    </title>
                    <rect
                      x={x}
                      y={y}
                      width={Math.max(0, cellWidth - 0.6)}
                      height={Math.max(0, cellHeight - 0.6)}
                      fill={fill}
                      stroke={isSelected ? "#0f172a" : die ? "rgba(15,23,42,0.12)" : "rgba(15,23,42,0.04)"}
                      strokeWidth={isSelected ? 3 : 0.6}
                      className={die && onSelectUnit ? "cursor-pointer" : ""}
                      onClick={() => die?.ufs_serial && onSelectUnit?.(die.ufs_serial)}
                    />
                  </g>
                );
              })}
            </g>
            <circle cx={cx} cy={cy} r={radius} fill="none" stroke="#94a3b8" strokeWidth={1.5} />
            {/* notch 표시 (하단 중앙) */}
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

        {/* 범례 — 모바일에서는 SVG 아래로, lg부터 우측에 */}
        <div className="text-[10px] text-brand-text leading-tight w-full lg:w-auto lg:min-w-[140px]">
          <div className="font-semibold mb-1.5">예측값 (ppm)</div>

          {/* 정상 영역 */}
          <div className="mb-2">
            <div className="text-[10px] text-brand-textMuted mb-0.5">정상</div>
            <div className="flex lg:flex-col gap-2 lg:gap-0.5 flex-wrap">
              {[0, 0.5, 1].map((t, i, arr) => {
                const next = arr[i + 1] ?? 1.01;
                const v = scale.pred_min + (scale.risk_threshold - scale.pred_min) * t;
                const vNext = scale.pred_min + (scale.risk_threshold - scale.pred_min) * next;
                return (
                  <div key={`n-${t}`} className="flex items-center gap-1.5">
                    <span
                      className="inline-block w-3 h-3 rounded-sm border border-slate-300"
                      style={{ background: predColor(v, scale.pred_min, scale.pred_max, scale.risk_threshold) }}
                    />
                    <span className="font-mono tabular text-[10px]">
                      {next > 1 ? `≤ ${(v * 1e6).toFixed(0)}` : `~${(vNext * 1e6).toFixed(0)}`}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* 위험 영역 */}
          <div>
            <div className="text-[10px] text-brand-danger font-semibold mb-0.5">위험 (≥ τ)</div>
            <div className="flex lg:flex-col gap-2 lg:gap-0.5 flex-wrap">
              {[0, 0.5, 1].map((t, i) => {
                const v = scale.risk_threshold + (scale.pred_max - scale.risk_threshold) * t;
                return (
                  <div key={`r-${t}`} className="flex items-center gap-1.5">
                    <span
                      className="inline-block w-3 h-3 rounded-sm border border-slate-300"
                      style={{ background: predColor(v, scale.pred_min, scale.pred_max, scale.risk_threshold) }}
                    />
                    <span className="font-mono tabular text-[10px]">
                      {i === 2 ? `≥ ${(v * 1e6).toFixed(0)}` : `${(v * 1e6).toFixed(0)}~`}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="mt-2 pt-2 border-t border-brand-border text-[10px] text-brand-textMuted leading-snug">
            <div>{dies.length} dies</div>
            <div>{xRange}×{yRange} grid</div>
            <div className="mt-1">
              τ = <span className="font-mono">{(scale.risk_threshold * 1e6).toFixed(0)} ppm</span>
            </div>
          </div>
          <div className="mt-2">
            <div
              className="w-full h-2 rounded-sm border border-brand-border"
              style={{ background: COLOR_LEGEND_GRADIENT }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}