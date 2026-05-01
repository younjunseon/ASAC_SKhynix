/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          // Primary: Indigo/Blue (이전 보라 폐기)
          primary: "#3b82f6",        // blue-500 — 메인 액센트
          primaryDark: "#2563eb",    // blue-600 — hover
          primaryLight: "#60a5fa",   // blue-400
          accent: "#06b6d4",         // cyan-500 — 강조 보조
          accentSoft: "#67e8f9",     // cyan-300
          // 메뉴 dot 인디케이터용 (메뉴별 다른 색)
          dot1: "#3b82f6",           // blue
          dot2: "#06b6d4",           // cyan
          dot3: "#f59e0b",           // amber
          dot4: "#10b981",           // emerald
          // 베이스
          surface: "#ffffff",
          bg: "#fafafa",             // 거의 white-on-white
          subtle: "#f8fafc",
          border: "#eef2f7",         // 매우 옅게
          borderStrong: "#e2e8f0",
          text: "#0f172a",           // slate-900
          textMuted: "#64748b",      // slate-500
          // 상태
          danger: "#ef4444",
          warn: "#f59e0b",
          success: "#10b981",
          info: "#3b82f6",
          link: "#3b82f6",
        },
        // 호환용 sf — 기존 코드 깨지지 않게 유지 (점진 폐기)
        sf: {
          bg: "#fafafa",
          panel: "#ffffff",
          border: "#eef2f7",
          softBorder: "#eef2f7",
          headBg: "#f8fafc",
          rowHover: "#eff6ff",
          select: "#3b82f6",
          link: "#3b82f6",
          magenta: "#3b82f6",
          blue: "#3b82f6",
          danger: "#ef4444",
          warn: "#f59e0b",
        },
      },
      fontFamily: {
        sans: ["Inter", "Segoe UI", "Malgun Gothic", "sans-serif"],
        mono: ["Consolas", "Courier New", "monospace"],
      },
      boxShadow: {
        card: "0 1px 2px rgba(15, 23, 42, 0.04)",
        cardHover: "0 4px 12px rgba(15, 23, 42, 0.08)",
        soft: "0 1px 3px rgba(15, 23, 42, 0.06)",
      },
      borderRadius: {
        DEFAULT: "8px",
      },
    },
  },
  plugins: [],
};
