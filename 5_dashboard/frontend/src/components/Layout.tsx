import { useState, type ReactNode } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import NotificationBell from "./NotificationBell";
import ChatbotWidget from "./ChatbotWidget";

interface SubItem {
  to: string;
  label: string;
  /** index 경로(부모와 동일 path)인 경우 true — exact match 로 active 판정 */
  end?: boolean;
  /** 아직 빈 페이지(델타큐맵 예정) 표시용 */
  tbd?: boolean;
}

interface NavItem {
  to: string;
  label: string;
  end?: boolean;
  dot: string;
  icon: ReactNode;
  /** 펼침 하위메뉴 — 있으면 항목 클릭 시 to 이동 + 하위메뉴 펼침, caret 클릭 시 토글 */
  children?: SubItem[];
}

const tabs: NavItem[] = [
  {
    to: "/",
    label: "품질 불량 예측 현황",
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
    // 다이만 보는 게 아니라 로트→웨이퍼→유닛→다이 4단계 계층 — 카테고리명 변경
    to: "/drilldown", // 부모 클릭 시 이동 (다이 차원 페이지). active 판정은 하위 4개 라우트 전체.
    label: "계층별 정밀 분석",
    dot: "#06b6d4",
    icon: (
      <svg fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
        <polygon points="12 2 22 8 12 14 2 8" />
        <polyline points="2 14 12 20 22 14" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
    children: [
      { to: "/drilldown/lot", label: "로트 차원", tbd: true },
      { to: "/drilldown/wafer", label: "웨이퍼 차원", tbd: true },
      { to: "/data", label: "유닛 차원" }, // = 기존 데일리 유닛 현황·자재 분석 페이지
      { to: "/drilldown", label: "다이 차원", end: true }, // = 기존 Drilldown(3-pane + 주별생산량)
    ],
  },
  {
    to: "/model",
    label: "모델 성능 분석",
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
];

/** 라우트가 해당 sub-item 에 매칭되는지 (end=true 면 exact). */
function subItemActive(c: SubItem, pathname: string): boolean {
  if (c.end) return pathname === c.to;
  return pathname === c.to || pathname.startsWith(c.to + "/");
}

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

function Caret({ open }: { open: boolean }) {
  return (
    <svg
      className="w-3 h-3 transition-transform"
      style={{ transform: open ? "rotate(90deg)" : "none" }}
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      viewBox="0 0 24 24"
      aria-hidden
    >
      <polyline points="9 6 15 12 9 18" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export default function Layout() {
  const [collapsed, setCollapsed] = useState(false);
  const { pathname } = useLocation();

  /** 부모 메뉴가 active 인가 — 하위 sub-item 중 하나라도 현재 경로면 true (하위 라우트가 부모 path 밖에 있어도 OK) */
  const parentActive = (t: NavItem) => !!t.children?.some((c) => subItemActive(c, pathname));

  // 하위메뉴 펼침 — 사용자가 명시적으로 토글한 값만 저장, 기본값은 "현재 경로가 하위면 펼침"
  const [openOverride, setOpenOverride] = useState<Record<string, boolean>>({});
  const menuOpen = (t: NavItem) => openOverride[t.to] ?? parentActive(t);
  const toggleMenu = (t: NavItem) =>
    setOpenOverride((o) => ({ ...o, [t.to]: !menuOpen(t) }));

  return (
    <div className="min-h-screen flex bg-brand-bg">
      {/* 사이드바 (틀: 용인 — 하위메뉴 펼침형) */}
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
        <nav className="py-3 flex-1 overflow-y-auto">
          {!collapsed && (
            <div className="px-5 pb-1.5 text-[10px] font-semibold uppercase tracking-wider text-brand-textMuted">
              분석 메뉴
            </div>
          )}

          {tabs.map((t) => {
            const hasChildren = !!t.children?.length;

            // 하위메뉴 없는 일반 항목
            if (!hasChildren) {
              return (
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
                    <span className="sidebar-dot" style={{ background: t.dot }} aria-hidden />
                  )}
                  {t.icon}
                  {!collapsed && <span className="flex-1 truncate">{t.label}</span>}
                </NavLink>
              );
            }

            // 하위메뉴 있는 항목 (계층별 정밀 분석 ▾ — 로트/웨이퍼/유닛/다이)
            const open = !collapsed && menuOpen(t);
            const active = parentActive(t);
            return (
              <div key={t.to}>
                <NavLink
                  to={t.to}
                  onClick={() => {
                    if (!collapsed) setOpenOverride((o) => ({ ...o, [t.to]: true }));
                  }}
                  title={collapsed ? t.label : undefined}
                  className={`sidebar-link ${active ? "active" : ""} ${collapsed ? "justify-center px-0" : ""}`}
                >
                  {!collapsed && (
                    <span className="sidebar-dot" style={{ background: t.dot }} aria-hidden />
                  )}
                  {t.icon}
                  {!collapsed && <span className="flex-1 truncate">{t.label}</span>}
                  {!collapsed && (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        toggleMenu(t);
                      }}
                      className="ml-auto -mr-1 p-1 rounded text-brand-textMuted hover:text-brand-text hover:bg-brand-subtle"
                      aria-label={open ? "하위메뉴 접기" : "하위메뉴 펼치기"}
                      title={open ? "접기" : "펼치기"}
                    >
                      <Caret open={open} />
                    </button>
                  )}
                </NavLink>

                {open && (
                  <div className="py-0.5">
                    {t.children!.map((c) => (
                      <NavLink
                        key={c.to}
                        to={c.to}
                        end={c.end}
                        className={({ isActive }) =>
                          `flex items-center gap-2 py-2 pl-[3.25rem] pr-4 text-[12.5px] cursor-pointer transition-colors border-l-[3px] ${
                            isActive
                              ? "text-brand-text font-semibold bg-brand-subtle border-brand-primary"
                              : "text-brand-textMuted border-transparent hover:text-brand-text hover:bg-brand-subtle"
                          }`
                        }
                      >
                        <span
                          className="w-1.5 h-1.5 rounded-full bg-current opacity-40 flex-shrink-0"
                          aria-hidden
                        />
                        <span className="flex-1 truncate">{c.label}</span>
                        {c.tbd && (
                          <span className="chip chip-tbd chip-sm flex-shrink-0" title="델타큐맵 페이지 준비 중">
                            예정
                          </span>
                        )}
                      </NavLink>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
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
        {/* 상단 바 (헤더: 준선) */}
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
