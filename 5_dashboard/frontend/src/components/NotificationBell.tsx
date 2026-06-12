import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { fetchAlertsToday } from "../lib/api";
import { fmtPpm, healthToPpm } from "../lib/format";

const POS_COLORS: Record<number, string> = { 1: "#3b82f6", 2: "#f59e0b", 3: "#10b981", 4: "#8b5cf6" };

export default function NotificationBell() {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const { data } = useQuery({
    queryKey: ["alerts-today-v2"],
    queryFn: fetchAlertsToday,
  });

  useEffect(() => {
    function onClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  const positionAlerts = data?.position_alerts ?? [];
  const lotClusters = data?.lot_cluster_alerts ?? [];
  const pointClusters = data?.point_cluster_alerts ?? [];
  const systematicLots = data?.systematic_lot_alerts ?? [];
  const totalAlerts =
    positionAlerts.length + lotClusters.length + pointClusters.length + systematicLots.length;

  const sectionHeader = (text: string, color: string) => (
    <div
      className="px-4 py-2 text-[10px] font-semibold uppercase tracking-wider sticky top-0 border-b border-red-100"
      style={{ background: `${color}14`, color }}
    >
      {text}
    </div>
  );

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
        <div className="absolute right-0 top-11 w-[420px] bg-white rounded-lg shadow-cardHover border border-brand-border z-50 overflow-hidden">
          <div className="px-4 py-3 border-b border-brand-border bg-brand-subtle flex items-center justify-between">
            <div>
              <div className="text-[13px] font-semibold text-brand-text">오늘 알림</div>
              <div className="text-[11px] text-brand-textMuted">
                연속/패턴 불량 시그널
              </div>
            </div>
            <span className="text-[11px] text-brand-textMuted">총 {totalAlerts}건</span>
          </div>

          <div className="max-h-[480px] overflow-y-auto">
            {/* 1. 같은 lot 연속 불량 (위험 unit 임계 이상) */}
            {lotClusters.length > 0 && (
              <div>
                {sectionHeader("⚠ 같은 로트에서 연속 불량 발생", "#dc2626")}
                {lotClusters.map((c) => (
                  <Link
                    key={`lot-${c.run_id}`}
                    to={`/drilldown?key=${encodeURIComponent(c.sample_wafer_key)}`}
                    onClick={() => setOpen(false)}
                    className="block px-4 py-2.5 hover:bg-brand-subtle border-b border-brand-border/50 bg-red-50/30"
                  >
                    <div className="flex items-start gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-brand-danger mt-1.5 flex-shrink-0"></div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-1.5">
                          <span className="text-[10px] font-bold uppercase tracking-wide text-brand-danger bg-red-100 px-1.5 py-0.5 rounded">
                            LOT
                          </span>
                          <span className="font-mono text-[12px] text-brand-text font-semibold">
                            {c.run_id}
                          </span>
                        </div>
                        <div className="text-[11px] text-brand-textMuted mt-0.5">
                          위험 unit{" "}
                          <span className="font-bold text-brand-danger">{c.n_risk_units}개</span>
                          {" 연속 발생 · 평균 "}
                          <span className="font-bold text-brand-danger">
                            {fmtPpm(healthToPpm(c.mean_pred))}
                          </span>
                        </div>
                      </div>
                    </div>
                  </Link>
                ))}
              </div>
            )}

            {/* 2. 같은 포인트 연속 불량 (die hotspot) */}
            {pointClusters.length > 0 && (
              <div>
                {sectionHeader("📍 같은 포인트에서 연속 불량 발생", "#d97706")}
                {pointClusters.map((c) => (
                  <Link
                    key={`pt-${c.die_x}-${c.die_y}`}
                    to={`/drilldown?key=${encodeURIComponent(c.sample_wafer_key)}`}
                    onClick={() => setOpen(false)}
                    className="block px-4 py-2.5 hover:bg-brand-subtle border-b border-brand-border/50"
                    style={{ background: "#fef3c714" }}
                  >
                    <div className="flex items-start gap-2">
                      <div className="w-1.5 h-1.5 rounded-full mt-1.5 flex-shrink-0" style={{ background: "#d97706" }}></div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-1.5">
                          <span
                            className="text-[10px] font-bold uppercase tracking-wide px-1.5 py-0.5 rounded"
                            style={{ color: "#d97706", background: "#fef3c7" }}
                          >
                            POINT
                          </span>
                          <span className="font-mono text-[12px] text-brand-text font-semibold">
                            ({c.die_x}, {c.die_y})
                          </span>
                        </div>
                        <div className="text-[11px] text-brand-textMuted mt-0.5">
                          위험 die{" "}
                          <span className="font-bold" style={{ color: "#d97706" }}>
                            {c.n_risk_dies}개
                          </span>
                          {" · "}
                          <span className="font-mono">{c.n_wafers}장 wafer</span>
                          {" 걸쳐서 · 평균 "}
                          <span className="font-bold" style={{ color: "#d97706" }}>
                            {fmtPpm(healthToPpm(c.mean_pred))}
                          </span>
                        </div>
                      </div>
                    </div>
                  </Link>
                ))}
              </div>
            )}

            {/* 3. 포지션 N번 연속 불량 (position lift) */}
            {positionAlerts.length > 0 && (
              <div>
                {sectionHeader("◆ 특정 포지션 연속 불량 발생", "#7c3aed")}
                {positionAlerts.map((p) => {
                  const c = POS_COLORS[p.position] ?? "#7c3aed";
                  return (
                    <div
                      key={`pos-${p.position}`}
                      className="block px-4 py-2.5 border-b border-brand-border/50"
                      style={{ background: "#ede9fe22" }}
                    >
                      <div className="flex items-start gap-2">
                        <div className="w-1.5 h-1.5 rounded-full mt-1.5 flex-shrink-0" style={{ background: c }}></div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-1.5">
                            <span
                              className="text-[10px] font-bold uppercase tracking-wide px-1.5 py-0.5 rounded"
                              style={{ color: c, background: `${c}1f` }}
                            >
                              POSITION
                            </span>
                            <span className="font-mono text-[12px] text-brand-text font-semibold">
                              P{p.position}
                            </span>
                          </div>
                          <div className="text-[11px] text-brand-textMuted mt-0.5">
                            위험 die{" "}
                            <span className="font-bold" style={{ color: c }}>
                              {p.n_risk_dies}/{p.n_dies}
                            </span>
                            {" ("}
                            <span className="font-bold" style={{ color: c }}>
                              {(p.risk_ratio * 100).toFixed(1)}%
                            </span>
                            {") · 평균 대비 "}
                            <span className="font-bold" style={{ color: c }}>
                              ×{p.lift.toFixed(2)}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* 4. 디펙성 lot 과다 발생 (lift) */}
            {systematicLots.length > 0 && (
              <div>
                {sectionHeader("⚡ 디펙성 불량 특정 로트 과다 발생", "#b91c1c")}
                {systematicLots.map((c) => (
                  <Link
                    key={`sys-${c.run_id}`}
                    to={`/drilldown?key=${encodeURIComponent(c.sample_wafer_key)}`}
                    onClick={() => setOpen(false)}
                    className="block px-4 py-2.5 hover:bg-brand-subtle border-b border-brand-border/50"
                    style={{ background: "#fee2e222" }}
                  >
                    <div className="flex items-start gap-2">
                      <div className="w-1.5 h-1.5 rounded-full mt-1.5 flex-shrink-0" style={{ background: "#b91c1c" }}></div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-1.5">
                          <span
                            className="text-[10px] font-bold uppercase tracking-wide px-1.5 py-0.5 rounded"
                            style={{ color: "#b91c1c", background: "#fecaca" }}
                          >
                            SYSTEMATIC
                          </span>
                          <span className="font-mono text-[12px] text-brand-text font-semibold">
                            {c.run_id}
                          </span>
                        </div>
                        <div className="text-[11px] text-brand-textMuted mt-0.5">
                          위험 비율{" "}
                          <span className="font-bold" style={{ color: "#b91c1c" }}>
                            {(c.risk_ratio * 100).toFixed(1)}%
                          </span>
                          {" ("}
                          <span className="font-mono">
                            {c.n_risk}/{c.n_units}
                          </span>
                          {") · baseline 대비 "}
                          <span className="font-bold" style={{ color: "#b91c1c" }}>
                            ×{c.lift.toFixed(2)}
                          </span>
                        </div>
                      </div>
                    </div>
                  </Link>
                ))}
              </div>
            )}

            {totalAlerts === 0 && (
              <div className="px-4 py-8 text-center text-[12px] text-brand-textMuted">
                패턴 불량 시그널이 없습니다.
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
