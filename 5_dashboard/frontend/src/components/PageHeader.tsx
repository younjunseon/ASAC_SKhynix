import type { StatusFilter } from "../lib/api";

const STATUS_OPTIONS: { value: StatusFilter; label: string }[] = [
  { value: "today", label: "오늘" },
  { value: "pending", label: "대기" },
  { value: "completed", label: "완료" },
  { value: "all", label: "전체" },
];

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
