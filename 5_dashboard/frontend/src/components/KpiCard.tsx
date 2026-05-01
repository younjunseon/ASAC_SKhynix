import type { ReactNode } from "react";

/**
 * 컴팩트 KPI 카드 — 큰 숫자 + 작은 변화 chip + 옅은 라벨.
 * tone은 숫자 색만 영향. accent 톤은 폐기.
 */
interface Props {
  label: string;
  value: string;
  hint?: string;
  tone?: "default" | "danger" | "warn" | "info" | "accent";
  icon?: ReactNode;
  change?: {
    label: string;
    direction: "up" | "down" | "flat";
    higherIsBad?: boolean;
  };
  /** 카드 좌측 강조 색 띠 (today 카드 같은 곳에 사용) */
  accentBar?: string;
  /** TBD — 아직 실측값이 연결되지 않은 자리. 시각적으로 강조됨 */
  pending?: boolean;
  pendingHint?: string;
}

export default function KpiCard({
  label, value, hint, tone = "default", icon, change, accentBar, pending, pendingHint,
}: Props) {
  const valueClass =
    tone === "danger" ? "text-brand-danger"
    : tone === "warn" ? "text-brand-warn"
    : tone === "info" ? "text-brand-primary"
    : "text-brand-text";

  let chipClass = "chip-muted";
  let chipIcon = "≈";
  if (change) {
    const bad = change.higherIsBad ?? true;
    if (change.direction === "up") {
      chipClass = bad ? "chip-danger" : "chip-success";
      chipIcon = "▲";
    } else if (change.direction === "down") {
      chipClass = bad ? "chip-success" : "chip-danger";
      chipIcon = "▼";
    } else {
      chipClass = "chip-muted";
      chipIcon = "≈";
    }
  }

  return (
    <div className="panel relative overflow-hidden">
      {accentBar && (
        <div
          className="absolute left-0 top-0 bottom-0 w-1"
          style={{ background: accentBar }}
        />
      )}
      <div className="px-4 pt-3.5 pb-1 flex items-center justify-between gap-2">
        <div className="text-[11px] font-medium text-brand-textMuted uppercase tracking-wide">
          {label}
        </div>
        {icon && (
          <div className="w-7 h-7 rounded-md flex items-center justify-center bg-brand-subtle text-brand-primary flex-shrink-0">
            {icon}
          </div>
        )}
      </div>
      <div className="px-4 pb-4">
        <div className={`tabular text-[24px] font-bold leading-tight ${pending ? "" : valueClass}`}>
          {pending ? <span className="tbd-value">　　　</span> : value}
        </div>
        <div className="flex items-center gap-2 mt-1.5 flex-wrap">
          {pending && (
            <span className="chip chip-tbd" title="실측값 미연결 (TBD)">
              <span aria-hidden>⚠</span>
              <span>값 미연결</span>
            </span>
          )}
          {!pending && change && (
            <span className={`chip ${chipClass}`}>
              <span aria-hidden>{chipIcon}</span>
              <span>{change.label}</span>
            </span>
          )}
          {(pending ? pendingHint : hint) && (
            <div className="text-[11px] text-brand-textMuted">
              {pending ? pendingHint : hint}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}