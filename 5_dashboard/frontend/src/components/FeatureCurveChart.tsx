/**
 * 피처 변곡점 차트 — (기존 SHAP beeswarm 자리를 대체)
 *
 * 선택한 피처를 등빈도(quantile) 구간으로 나눠, 구간별 평균 health / 불량률을 그려
 * "어느 변수값 근처에서 관계가 꺾이는지(변곡점)" 가 드러나게 한다.
 * 데이터: public/feature_dist.csv (ufs_serial, X..., health, is_defect) — 상위 중요 피처 15개 수록.
 */
import { useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import { useCSV } from "../hooks/useCSV";
import Panel from "./Panel";

type DistRow = Record<string, number | string>;

const num = (v: unknown): number => {
  const n = typeof v === "number" ? v : parseFloat(String(v));
  return Number.isFinite(n) ? n : 0;
};

const N_BINS = 20;

export default function FeatureCurveChart() {
  const { data, loading } = useCSV<DistRow>("/feature_dist.csv");

  const featureCols = useMemo(() => {
    if (!data.length) return [];
    return Object.keys(data[0]).filter((k) => k !== "ufs_serial" && k !== "health" && k !== "is_defect");
  }, [data]);

  const [feat, setFeat] = useState<string | null>(null);
  const activeFeat = feat ?? featureCols[0] ?? null;

  const binned = useMemo(() => {
    if (!activeFeat || !data.length) return [];
    const pts = data
      .map((r) => ({ v: num(r[activeFeat]), h: num(r.health), d: num(r.is_defect) }))
      .filter((p) => Number.isFinite(p.v))
      .sort((a, b) => a.v - b.v);
    if (pts.length < N_BINS) return [];
    const per = Math.floor(pts.length / N_BINS);
    const out: { x: number; meanHealth: number; defectRate: number; n: number }[] = [];
    for (let i = 0; i < N_BINS; i++) {
      const start = i * per;
      const end = i === N_BINS - 1 ? pts.length : start + per;
      const chunk = pts.slice(start, end);
      if (chunk.length === 0) continue;
      const mid = chunk[Math.floor(chunk.length / 2)].v;
      const meanHealth = chunk.reduce((a, p) => a + p.h, 0) / chunk.length;
      const defectRate = (chunk.reduce((a, p) => a + p.d, 0) / chunk.length) * 100;
      out.push({ x: mid, meanHealth, defectRate, n: chunk.length });
    }
    return out;
  }, [data, activeFeat]);

  const option = {
    tooltip: {
      trigger: "axis",
      formatter: (p: any[]) => {
        const i = p[0].dataIndex;
        const b = binned[i];
        if (!b) return "";
        return (
          `${activeFeat} ≈ <b>${b.x.toPrecision(5)}</b><br/>` +
          `불량률: ${b.defectRate.toFixed(1)}%<br/>` +
          `평균 health: ${b.meanHealth.toExponential(3)}<br/>` +
          `구간 unit 수: ${b.n}`
        );
      },
    },
    legend: { data: ["불량률(%)", "평균 health"], bottom: 0, textStyle: { fontSize: 11, color: "#475569" } },
    grid: { top: 16, left: 56, right: 64, bottom: 40 },
    xAxis: {
      type: "value",
      name: activeFeat ?? "",
      nameLocation: "middle",
      nameGap: 24,
      nameTextStyle: { fontSize: 11, color: "#64748b", fontFamily: "Consolas, monospace" },
      axisLabel: { fontSize: 9, color: "#94a3b8", formatter: (v: number) => (Math.abs(v) >= 1000 ? v.toExponential(1) : Number(v.toPrecision(3))) },
      splitLine: { lineStyle: { color: "#f1f5f9" } },
    },
    yAxis: [
      {
        type: "value",
        name: "불량률 (%)",
        nameTextStyle: { fontSize: 10, color: "#f59e0b" },
        axisLabel: { fontSize: 9, color: "#94a3b8", formatter: (v: number) => `${v}%` },
        splitLine: { lineStyle: { color: "#f1f5f9" } },
      },
      {
        type: "value",
        name: "평균 health",
        nameTextStyle: { fontSize: 10, color: "#ef4444" },
        axisLabel: { fontSize: 9, color: "#94a3b8", formatter: (v: number) => v.toExponential(0) },
        splitLine: { show: false },
      },
    ],
    series: [
      {
        name: "불량률(%)",
        type: "line",
        data: binned.map((b) => [b.x, b.defectRate]),
        smooth: false,
        symbolSize: 5,
        lineStyle: { color: "#f59e0b", width: 2 },
        itemStyle: { color: "#f59e0b" },
        z: 2,
      },
      {
        name: "평균 health",
        type: "line",
        yAxisIndex: 1,
        data: binned.map((b) => [b.x, b.meanHealth]),
        smooth: false,
        symbolSize: 5,
        lineStyle: { color: "#ef4444", width: 2, type: "dashed" },
        itemStyle: { color: "#ef4444" },
        z: 3,
      },
    ],
  };

  return (
    <Panel
      title="피처 변곡점 — 변수값 구간별 평균 health / 불량률"
      right={
        <select
          className="text-[11px] border border-brand-border rounded px-1.5 py-0.5 bg-white text-brand-text"
          value={activeFeat ?? ""}
          onChange={(e) => setFeat(e.target.value)}
        >
          {featureCols.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
      }
    >
      {loading ? (
        <div className="py-10 text-center text-brand-textMuted text-[12px]">데이터 로딩 중… (feature_dist.csv ~4MB)</div>
      ) : binned.length === 0 ? (
        <div className="py-10 text-center text-brand-textMuted text-[12px]">데이터 없음 (feature_dist.csv)</div>
      ) : (
        <>
          <ReactECharts option={option} style={{ height: 300 }} notMerge lazyUpdate />
          <div className="text-[10.5px] text-brand-textMuted mt-1 px-1">
            등빈도 {N_BINS}구간(각 구간 ≈ {Math.floor(data.length / N_BINS).toLocaleString()} unit). <span style={{ color: "#f59e0b", fontWeight: 600 }}>주황</span> = 불량률(%, 좌축), <span style={{ color: "#ef4444", fontWeight: 600 }}>빨강 점선</span> = 평균 health(우축). 선의 기울기가 바뀌는 지점이 그 변수의 변곡점.
          </div>
        </>
      )}
    </Panel>
  );
}
