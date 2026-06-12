import { useEffect, useState } from "react";
import Papa from "papaparse";

/**
 * public/ 의 정적 CSV를 fetch + 파싱하는 훅 (용인 대시보드에서 이식).
 *
 * 용도: 모델 성능 시각화·주별 생산량 등 — FastAPI 엔드포인트로 노출하지 않고
 *      build 단계에서 굳혀둔 정적 CSV를 그대로 읽는 화면들.
 *
 * @param path  public/ 기준 경로 (예: "/metrics.csv")
 */
export function useCSV<T = Record<string, unknown>>(path: string) {
  const [data, setData] = useState<T[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetch(path)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.text();
      })
      .then((text) => {
        if (cancelled) return;
        const result = Papa.parse<T>(text, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
        });
        setData(result.data);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        const msg = err instanceof Error ? err.message : String(err);
        console.error(`[useCSV] ${path}:`, msg);
        setError(msg);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [path]);

  return { data, loading, error };
}
