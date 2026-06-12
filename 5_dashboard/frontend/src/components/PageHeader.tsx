import type { StatusFilter } from "../lib/api";

const STATUS_OPTIONS: { value: StatusFilter; label: string }[] = [
  { value: "today", label: "오늘" },
  { value: "all", label: "전체" },
];

/** 오늘 날짜를 "YYYY-MM-DD (요일)" 형식으로 표시 */
function formatToday(): string {
  const d = new Date();
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  const days = ["일", "월", "화", "수", "목", "금", "토"];
  return `${yyyy}-${mm}-${dd} (${days[d.getDay()]})`;
}

interface Props {
  title: string;
  subtitle?: string;
  status?: StatusFilter;
  onStatusChange?: (s: StatusFilter) => void;
  right?: React.ReactNode;
}

export default function PageHeader({ title, subtitle, status, onStatusChange, right }: Props) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-3 mb-4 sm:mb-5">
      <div className="min-w-0">
        <h1 className="text-[18px] sm:text-[20px] font-bold text-brand-text leading-tight">
          {title}
        </h1>
        {subtitle && (
          <p className="text-[12px] text-brand-textMuted mt-1">{subtitle}</p>
        )}
      </div>
      <div className="flex items-center gap-3 flex-wrap">
        <div className="text-[11px] text-brand-textMuted tabular bg-brand-subtle rounded-md px-2.5 py-1.5">
          오늘 <span className="font-semibold text-brand-text">{formatToday()}</span>
        </div>
        {right}
        {status && onStatusChange && (
          <div className="inline-flex bg-white border border-brand-border rounded-lg overflow-hidden">
            {STATUS_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                onClick={() => onStatusChange(opt.value)}
                className={`text-[12px] px-3 py-1.5 font-medium transition-colors ${
                  status === opt.value
                    ? "bg-brand-primary text-white"
                    : "text-brand-textMuted hover:bg-brand-subtle"
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
