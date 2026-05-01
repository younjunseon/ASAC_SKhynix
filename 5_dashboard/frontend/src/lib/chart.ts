/**
 * Recharts 공통 색상/스타일 상수.
 * 단색(블루) 농도 위주 + 시안 1색 강조 + 위험은 amber/red.
 */
export const CHART_COLORS = {
  primary: "#3b82f6",     // blue-500 — 메인
  primaryDark: "#2563eb", // blue-600
  primaryLight: "#93c5fd",// blue-300
  primarySoft: "#dbeafe", // blue-100 — 배경/보조
  accent: "#06b6d4",      // cyan-500 — 강조 1색
  accentLight: "#67e8f9", // cyan-300
  // 상태
  danger: "#ef4444",      // red-500 — 위험 라인
  warn: "#f59e0b",        // amber-500 — today 강조
  success: "#10b981",     // emerald-500
  neutral: "#94a3b8",
};

export const CHART_GRID = {
  stroke: "#eef2f7",
  strokeDasharray: "3 3",
};

export const CHART_TICK = { fontSize: 11, fill: "#64748b" };

export const CHART_TOOLTIP_STYLE = {
  fontSize: 11,
  borderRadius: 8,
  border: "1px solid #eef2f7",
  boxShadow: "0 4px 12px rgba(15, 23, 42, 0.08)",
};

export const CHART_LEGEND_STYLE = { fontSize: 11 };

export const BAR_RADIUS: [number, number, number, number] = [4, 4, 0, 0];

/**
 * ResponsiveContainer wrapper용 표준 스타일.
 * minWidth:0 — CSS Grid item에서 width:100%가 부모 컬럼 폭으로 정확히 잡히도록
 * (기본 min-width: auto는 콘텐츠 크기로 부풀어 ResponsiveContainer가 width=-1 받는 원인)
 */
export const chartBox = (height: number) =>
  ({
    width: "100%",
    height,
    minWidth: 0,
  } as const);