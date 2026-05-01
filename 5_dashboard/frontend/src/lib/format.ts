export const fmtNum = (v: number | null | undefined, digits = 5) =>
  v === null || v === undefined || Number.isNaN(v) ? "-" : v.toFixed(digits);

export const fmtPct = (v: number | null | undefined, digits = 1) =>
  v === null || v === undefined || Number.isNaN(v)
    ? "-"
    : `${(v * 100).toFixed(digits)}%`;

export const fmtInt = (v: number | null | undefined) =>
  v === null || v === undefined || Number.isNaN(v) ? "-" : v.toLocaleString();

/* ─── PPM (Parts Per Million) ─────────────────────────────────
 * health는 동작 오류율(연속형). 1 ppm = 100만 동작 중 1건 오류.
 * health 0.001 = 1,000 ppm. health 0.0065 (Y>0 median) = 6,500 ppm.
 * 분류 임계값이 아니라 "품질 등급"으로 다룸.
 */
export const HEALTH_TO_PPM = 1_000_000;

/** health 원값 → ppm */
export const healthToPpm = (h: number | null | undefined): number =>
  h === null || h === undefined || Number.isNaN(h) ? 0 : h * HEALTH_TO_PPM;

/** ppm 정수 포맷팅 — 1234 → "1,234 ppm" */
export const fmtPpm = (
  v: number | null | undefined,
  opts: { suffix?: boolean; digits?: number } = {},
): string => {
  if (v === null || v === undefined || Number.isNaN(v)) return "-";
  const { suffix = true, digits = 0 } = opts;
  const txt =
    digits > 0 ? v.toFixed(digits) : Math.round(v).toLocaleString();
  return suffix ? `${txt} ppm` : txt;
};

/** health 원값 → "1,234 ppm" 한 번에 */
export const fmtHealthAsPpm = (
  h: number | null | undefined,
  opts: { suffix?: boolean; digits?: number } = {},
): string => fmtPpm(healthToPpm(h ?? 0), opts);

/* ─── Tier 등급 (S/A/B/C/D) ──────────────────────────────────
 * 회귀 예측 결과를 운영 의사결정용으로 그루핑.
 * 모델은 여전히 연속 ppm을 예측 — Tier는 표시·필터링용 bucket.
 */
export type Tier = "S" | "A" | "B" | "C" | "D";

export interface TierDef {
  tier: Tier;
  /** 하한 (포함) — ppm */
  minPpm: number;
  /** 상한 (배타) — ppm. 마지막 D는 Infinity */
  maxPpm: number;
  label: string;
  color: string;
  desc: string;
}

export const TIERS: TierDef[] = [
  { tier: "S", minPpm: 0, maxPpm: 1, label: "S (정상)", color: "#10b981", desc: "0 ppm" },
  { tier: "A", minPpm: 1, maxPpm: 100, label: "A (양호)", color: "#3b82f6", desc: "1~100 ppm" },
  { tier: "B", minPpm: 100, maxPpm: 1_000, label: "B (관리)", color: "#06b6d4", desc: "100~1k ppm" },
  { tier: "C", minPpm: 1_000, maxPpm: 10_000, label: "C (주의)", color: "#f59e0b", desc: "1k~10k ppm" },
  { tier: "D", minPpm: 10_000, maxPpm: Infinity, label: "D (보류 후보)", color: "#ef4444", desc: "10k+ ppm" },
];

/** ppm 값 → Tier */
export const tierOfPpm = (ppm: number): Tier => {
  for (const t of TIERS) {
    if (ppm >= t.minPpm && ppm < t.maxPpm) return t.tier;
  }
  return "D";
};

/** health 원값 → Tier */
export const tierOfHealth = (h: number | null | undefined): Tier =>
  tierOfPpm(healthToPpm(h ?? 0));

export const tierColor = (tier: Tier): string =>
  TIERS.find((t) => t.tier === tier)?.color ?? "#94a3b8";
