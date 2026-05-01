import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { fetchTriage } from "../lib/api";
import { fmtPpm, healthToPpm } from "../lib/format";

export default function NotificationBell() {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // 오늘 검사된 위험 항목으로 알림 구성 — PI가 가장 먼저 봐야 하는 것
  const { data } = useQuery({
    queryKey: ["alerts-today"],
    queryFn: () => fetchTriage({ status: "today", top_units: 5, top_wafers: 5 }),
  });

  useEffect(() => {
    function onClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  const wafers = data?.top_wafers ?? [];
  const units = data?.top_units ?? [];
  const totalAlerts = wafers.length + units.length;

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((v) => !v)}
        className="relative w-9 h-9 rounded-full hover:bg-brand-subtle flex items-center justify-center text-brand-textMuted transition-colors"
        aria-label="알림"
      >
        <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M14.857 17.082a23.848 23.848 0 005.454-1.31A8.967 8.967 0 0118 9.75v-.7V9A6 6 0 006 9v.75a8.967 8.967 0 01-2.312 6.022c1.733.64 3.56 1.085 5.455 1.31m5.714 0a24.255 24.255 0 01-5.714 0m5.714 0a3 3 0 11-5.714 0" />
        </svg>
        {totalAlerts > 0 && (
          <span className="absolute -top-0.5 -right-0.5 min-w-[18px] h-[18px] px-1 rounded-full bg-brand-danger text-white text-[10px] font-bold flex items-center justify-center ring-2 ring-white">
            {totalAlerts > 99 ? "99+" : totalAlerts}
          </span>
        )}
      </button>

      {open && (
        <div className="absolute right-0 top-11 w-96 bg-white rounded-lg shadow-cardHover border border-brand-border z-50 overflow-hidden">
          <div className="px-4 py-3 border-b border-brand-border bg-brand-subtle flex items-center justify-between">
            <div>
              <div className="text-[13px] font-semibold text-brand-text">오늘 알림</div>
              <div className="text-[11px] text-brand-textMuted">
                오늘 검사된 위험 항목
              </div>
            </div>
            <span className="text-[11px] text-brand-textMuted">
              총 {totalAlerts}건
            </span>
          </div>

          <div className="max-h-[420px] overflow-y-auto">
            {wafers.length > 0 && (
              <div>
                <div className="px-4 py-2 text-[10px] font-semibold text-brand-textMuted uppercase tracking-wider bg-white sticky top-0">
                  위험 Wafer
                </div>
                {wafers.map((w) => (
                  <Link
                    key={w.wafer_key}
                    to={`/drilldown?key=${encodeURIComponent(w.wafer_key)}`}
                    onClick={() => setOpen(false)}
                    className="block px-4 py-2.5 hover:bg-brand-subtle border-b border-brand-border/50"
                  >
                    <div className="flex items-start gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-brand-danger mt-1.5 flex-shrink-0"></div>
                      <div className="flex-1 min-w-0">
                        <div className="font-mono text-[12px] text-brand-text font-semibold">
                          {w.wafer_key}
                        </div>
                        <div className="text-[11px] text-brand-textMuted mt-0.5">
                          평균 <span className="font-bold text-brand-danger">{fmtPpm(healthToPpm(w.mean_pred))}</span>
                          {" · "}
                          임계 초과 {w.n_risk}/{w.n_units}
                        </div>
                      </div>
                    </div>
                  </Link>
                ))}
              </div>
            )}

            {units.length > 0 && (
              <div>
                <div className="px-4 py-2 text-[10px] font-semibold text-brand-textMuted uppercase tracking-wider bg-white sticky top-0">
                  위험 Unit
                </div>
                {units.map((u) => (
                  <Link
                    key={u.ufs_serial}
                    to={`/drilldown?key=${encodeURIComponent(u.wafer_key)}`}
                    onClick={() => setOpen(false)}
                    className="block px-4 py-2.5 hover:bg-brand-subtle border-b border-brand-border/50"
                  >
                    <div className="flex items-start gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-brand-warn mt-1.5 flex-shrink-0"></div>
                      <div className="flex-1 min-w-0">
                        <div className="font-mono text-[12px] text-brand-text font-semibold">
                          {u.ufs_serial}
                        </div>
                        <div className="text-[11px] text-brand-textMuted mt-0.5">
                          예측 <span className="font-bold text-brand-danger">{fmtPpm(healthToPpm(u.pred))}</span>
                          {" · "}
                          <span className="font-mono">{u.wafer_key}</span>
                        </div>
                      </div>
                    </div>
                  </Link>
                ))}
              </div>
            )}

            {totalAlerts === 0 && (
              <div className="px-4 py-8 text-center text-[12px] text-brand-textMuted">
                새로운 알림이 없습니다.
              </div>
            )}
          </div>

          <div className="px-4 py-2 border-t border-brand-border bg-brand-subtle text-center">
            <Link
              to="/"
              onClick={() => setOpen(false)}
              className="text-[11px] text-brand-link hover:underline font-medium"
            >
              모든 알림 보기 →
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}
