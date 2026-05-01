import { useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import NotificationBell from "./NotificationBell";
import ChatbotWidget from "./ChatbotWidget";

interface NavItem {
  to: string;
  label: string;
  end?: boolean;
  dot: string;
  icon: React.ReactNode;
}

const tabs: NavItem[] = [
  {
    to: "/",
    label: "Overview",
    end: true,
    dot: "#3b82f6",
    icon: (
      <svg fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
        <rect x="3" y="3" width="7" height="9" rx="1.5" />
        <rect x="14" y="3" width="7" height="5" rx="1.5" />
        <rect x="14" y="12" width="7" height="9" rx="1.5" />
        <rect x="3" y="16" width="7" height="5" rx="1.5" />
      </svg>
    ),
  },
  {
    to: "/drilldown",
    label: "Drill-down",
    dot: "#06b6d4",
    icon: (
      <svg fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
        <circle cx="11" cy="11" r="7" />
        <line x1="21" y1="21" x2="16" y2="16" strokeLinecap="round" />
        <line x1="11" y1="8" x2="11" y2="14" strokeLinecap="round" />
        <line x1="8" y1="11" x2="14" y2="11" strokeLinecap="round" />
      </svg>
    ),
  },
  {
    to: "/model",
    label: "Model",
    dot: "#10b981",
    icon: (
      <svg fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
        <line x1="4" y1="20" x2="4" y2="10" strokeLinecap="round" />
        <line x1="10" y1="20" x2="10" y2="4" strokeLinecap="round" />
        <line x1="16" y1="20" x2="16" y2="14" strokeLinecap="round" />
        <line x1="22" y1="20" x2="22" y2="8" strokeLinecap="round" />
      </svg>
    ),
  },
  {
    to: "/data",
    label: "Data",
    dot: "#f59e0b",
    icon: (
      <svg fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
        <ellipse cx="12" cy="5" rx="8" ry="2.5" />
        <path d="M4 5v6c0 1.4 3.6 2.5 8 2.5s8-1.1 8-2.5V5" />
        <path d="M4 11v6c0 1.4 3.6 2.5 8 2.5s8-1.1 8-2.5v-6" />
      </svg>
    ),
  },
];

/** SK 하이닉스 임시 브랜드 마크 (실 로고 파일이 있으면 <img>로 교체) */
function HynixMark({ size = 32 }: { size?: number }) {
  return (
    <div
      className="rounded-md flex items-center justify-center text-white font-bold flex-shrink-0"
      style={{
        width: size,
        height: size,
        background: "#EE2737", // SK hynix red
        fontSize: size * 0.42,
        letterSpacing: "-0.5px",
      }}
      title="SK hynix"
    >
      SK
    </div>
  );
}

export default function Layout() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="min-h-screen flex bg-brand-bg">
      {/* 사이드바 */}
      <aside
        className={`bg-white border-r border-brand-border flex flex-col flex-shrink-0 transition-all duration-200 ${
          collapsed ? "w-16" : "w-56"
        }`}
      >
        {/* 로고 영역 + 접기 토글 */}
        <div className="border-b border-brand-border min-h-[64px] flex items-center px-3 gap-2">
          <HynixMark size={32} />
          {!collapsed && (
            <div className="overflow-hidden flex-1 min-w-0">
              <div className="text-[13px] font-bold text-brand-text leading-tight whitespace-nowrap">
                SK hynix
              </div>
              <div className="text-[10px] text-brand-textMuted whitespace-nowrap">
                Wafer Health
              </div>
            </div>
          )}
          <button
            onClick={() => setCollapsed((v) => !v)}
            className="w-7 h-7 rounded-md flex items-center justify-center text-brand-textMuted hover:bg-brand-subtle hover:text-brand-text flex-shrink-0"
            aria-label={collapsed ? "사이드바 펴기" : "사이드바 접기"}
            title={collapsed ? "펴기" : "접기"}
          >
            <svg
              className="w-4 h-4 transition-transform"
              style={{ transform: collapsed ? "rotate(180deg)" : "none" }}
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              viewBox="0 0 24 24"
            >
              <polyline points="15 18 9 12 15 6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        </div>

        {/* 메뉴 */}
        <nav className="py-3 flex-1">
          {tabs.map((t) => (
            <NavLink
              key={t.to}
              to={t.to}
              end={t.end}
              title={collapsed ? t.label : undefined}
              className={({ isActive }) =>
                `sidebar-link ${isActive ? "active" : ""} ${collapsed ? "justify-center px-0" : ""}`
              }
            >
              {!collapsed && (
                <span
                  className="sidebar-dot"
                  style={{ background: t.dot }}
                  aria-hidden
                />
              )}
              {t.icon}
              {!collapsed && <span>{t.label}</span>}
            </NavLink>
          ))}
        </nav>

        {/* 사용자 프로필 */}
        <div className="border-t border-brand-border p-3">
          <div className={`flex items-center gap-2.5 ${collapsed ? "justify-center" : ""}`}>
            <div
              className="w-9 h-9 rounded-full flex items-center justify-center text-white font-semibold text-[13px] flex-shrink-0"
              style={{ background: "linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%)" }}
              title="이정훈 · PI"
            >
              이
            </div>
            {!collapsed && (
              <div className="min-w-0 flex-1">
                <div className="text-[12.5px] font-semibold text-brand-text leading-tight truncate">
                  이정훈
                </div>
                <div className="text-[10px] text-brand-textMuted truncate">
                  Process Integration
                </div>
              </div>
            )}
          </div>
        </div>
      </aside>

      {/* 본문 */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* 상단 바 */}
        <header className="bg-white border-b border-brand-border px-4 sm:px-5 py-3.5 flex items-center justify-between gap-4">
          <div className="text-[15px] font-semibold text-brand-text truncate">
            Dashboard
          </div>
          <div className="flex items-center gap-3 flex-shrink-0">
            <span className="hidden sm:inline-flex items-center gap-1.5 text-[11px] text-brand-textMuted">
              <span className="inline-block w-1.5 h-1.5 rounded-full bg-brand-success" />
              Online
            </span>
            <NotificationBell />
          </div>
        </header>

        <main className="flex-1 p-4 sm:p-5 lg:p-6 pb-28 sm:pb-32 overflow-x-hidden">
          <Outlet />
        </main>
      </div>

      <ChatbotWidget />
    </div>
  );
}
