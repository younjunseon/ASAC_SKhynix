/**
 * Wafer map cell 색.
 * 임계값(threshold)을 기준으로 두 영역을 다른 팔레트로 분리:
 *   - threshold 미만 (정상): 옅은 회색 → 옅은 청록
 *   - threshold 이상 (위험): 노랑 → 주황 → 빨강
 */
type RGB = [number, number, number];

const NORMAL_STOPS: Array<[number, RGB]> = [
  [0.0, [243, 244, 246]],  // 거의 흰색 (gray-100)
  [0.5, [219, 234, 254]],  // 매우 옅은 파랑 (blue-100)
  [1.0, [165, 215, 220]],  // 옅은 청록
];

const RISK_STOPS: Array<[number, RGB]> = [
  [0.0, [254, 240, 138]],  // 노랑 (yellow-200) — 임계 직전 경고
  [0.5, [251, 146, 60]],   // 주황 (orange-400)
  [1.0, [220, 38, 38]],    // 빨강 (red-600)
];

function interp(stops: Array<[number, RGB]>, t: number): string {
  const tt = Math.max(0, Math.min(1, t));
  for (let i = 1; i < stops.length; i++) {
    const [t1, c1] = stops[i];
    const [t0, c0] = stops[i - 1];
    if (tt <= t1) {
      const k = (tt - t0) / (t1 - t0 || 1);
      const r = Math.round(c0[0] + (c1[0] - c0[0]) * k);
      const g = Math.round(c0[1] + (c1[1] - c0[1]) * k);
      const b = Math.round(c0[2] + (c1[2] - c0[2]) * k);
      return `rgb(${r},${g},${b})`;
    }
  }
  const [, last] = stops[stops.length - 1];
  return `rgb(${last.join(",")})`;
}

export function predColor(
  pred: number,
  predMin: number,
  predMax: number,
  threshold: number
): string {
  if (!isFinite(pred)) return "#f1f5f9";
  if (pred <= threshold) {
    // 정상 영역 — predMin~threshold를 0~1로 매핑
    const span = Math.max(1e-9, threshold - predMin);
    const t = (pred - predMin) / span;
    return interp(NORMAL_STOPS, t);
  } else {
    // 위험 영역 — threshold~predMax를 0~1로 매핑
    const span = Math.max(1e-9, predMax - threshold);
    const t = (pred - threshold) / span;
    return interp(RISK_STOPS, t);
  }
}

/** 범례 그라데이션 — 정상→경고→위험 한 번에 보여주는 띠 */
export const COLOR_LEGEND_GRADIENT =
  "linear-gradient(to right, #f3f4f6, #dbeafe, #a5d7dc, #fef08a, #fb923c, #dc2626)";