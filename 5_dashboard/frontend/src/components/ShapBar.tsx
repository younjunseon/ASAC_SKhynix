/**
 * SHAP top-N 양/음 양방향 bar.
 * design.md §6.2 Drill-down Unit Diagnosis — SHAP top 20 표시용.
 *
 * 양수 → 빨간색(위험 기여), 음수 → 초록색(정상 방향).
 * z-score는 별도 chip으로 우측에 표시.
 */
import { CHART_COLORS } from "../lib/chart";

export interface ShapItem {
  rank: number;
  feature_name: string;
  shap_value: number;       // + or -
  feature_value?: number;   // 그 unit의 해당 feature 값
  z_score?: number;         // global 분포 대비 z-score
}

interface Props {
  items: ShapItem[];
  /** 막대 최대값 기준 (생략 시 자동 — items 절댓값 max) */
  scale?: number;
  /** 표시 개수 (기본 전체) */
  limit?: number;
}

export default function ShapBar({ items, scale, limit }: Props) {
  const slice = limit ? items.slice(0, limit) : items;
  const maxAbs =
    scale ?? Math.max(1e-9, ...slice.map((i) => Math.abs(i.shap_value)));

  if (slice.length === 0) {
    return (
      <div className="text-[11px] text-brand-textMuted p-2">SHAP 데이터 없음</div>
    );
  }

  return (
    <div className="space-y-1">
      {slice.map((it) => {
        const ratio = Math.min(100, (Math.abs(it.shap_value) / maxAbs) * 100);
        const isPos = it.shap_value > 0;
        const color = isPos ? CHART_COLORS.danger : CHART_COLORS.success;
        const zChip =
          it.z_score === undefined
            ? null
            : Math.abs(it.z_score) >= 2
            ? "chip-danger"
            : Math.abs(it.z_score) >= 1
            ? "chip-warn"
            : "chip-muted";

        return (
          <div
            key={`${it.rank}-${it.feature_name}`}
            className="grid grid-cols-[40px_1fr_auto] items-center gap-2 text-[11px]"
          >
            {/* feature 이름 */}
            <span className="font-mono text-brand-text truncate" title={it.feature_name}>
              {it.feature_name}
            </span>

            {/* 막대 (중앙 0 기준 양/음 분기) */}
            <div className="relative h-3 bg-brand-subtle rounded-sm overflow-hidden flex">
              {/* 0선 */}
              <div className="absolute left-1/2 top-0 bottom-0 w-px bg-brand-border" />
              {/* 음수 바 (왼쪽) */}
              <div className="w-1/2 flex justify-end">
                {!isPos && (
                  <div
                    style={{ width: `${ratio}%`, background: color }}
                    className="h-full rounded-l-sm"
                  />
                )}
              </div>
              {/* 양수 바 (오른쪽) */}
              <div className="w-1/2">
                {isPos && (
                  <div
                    style={{ width: `${ratio}%`, background: color }}
                    className="h-full rounded-r-sm"
                  />
                )}
              </div>
            </div>

            {/* SHAP 값 + z chip */}
            <div className="flex items-center gap-1.5 justify-end">
              <span
                className={`font-mono tabular text-[10px] ${
                  isPos ? "text-brand-danger" : "text-brand-success"
                }`}
              >
                {isPos ? "+" : ""}
                {it.shap_value.toFixed(4)}
              </span>
              {zChip && it.z_score !== undefined && (
                <span className={`chip chip-sm ${zChip}`} title="global z-score">
                  z={it.z_score >= 0 ? "+" : ""}
                  {it.z_score.toFixed(2)}
                </span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}