/**
 * 주별 생산량 차트 — 용인 대시보드에서 이식 (echarts + 정적 CSV).
 *
 * 데이터: public/dashboard_dates.csv (lot 단위: lot/split/total/danger/defect_rate/health_mean/pred_mean)
 * lot 7개를 1주로 묶어(주차 = floor((lot-1)/7)) 생산 수량 합 + 가중 평균 예측 health 를 듀얼축으로 표시.
 */
import { useMemo } from "react";
import ReactECharts from "echarts-for-react";
import { useCSV } from "../hooks/useCSV";
import Panel from "./Panel";

type DateRow = {
  lot?: number;
  split?: string;
  total?: number;
  danger?: number;
  defect_rate?: number;
  health_mean?: number;
  pred_mean?: number;
};

const num = (v: unknown): number => {
  const n = typeof v === "number" ? v : parseFloat(String(v));
  return Number.isFinite(n) ? n : 0;
};

const LOTS_PER_WEEK = 7;

export default function WeeklyProductionChart() {
  const { data, loading } = useCSV<DateRow>("/dashboard_dates.csv");

  const weekly = useMemo(() => {
    if (!data.length) return [];
    const buckets = new Map<number, { total: number; predW: number; predSum: number; splits: Set<string> }>();
    for (const r of data) {
      const lot = num(r.lot);
      if (lot <= 0) continue;
      const w = Math.floor((lot - 1) / LOTS_PER_WEEK);
      const total = num(r.total);
      const b = buckets.get(w) ?? { total: 0, predW: 0, predSum: 0, splits: new Set<string>() };
      b.total += total;
      b.predW += num(r.pred_mean) * total; // 수량 가중
      b.predSum += total;
      if (r.split) b.splits.add(String(r.split));
      buckets.set(w, b);
    }
    return [...buckets.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([w, b]) => ({
        label: `${w + 1}주차`,
        total: b.total,
        pred: b.predSum > 0 ? b.predW / b.predSum : 0,
        split: b.splits.size === 1 ? [...b.splits][0] : "mixed",
      }));
  }, [data]);

  const splitColor: Record<string, string> = {
    train: "rgba(100,116,139,.55)",
    val: "rgba(59,130,246,.7)",
    test: "rgba(6,182,212,.7)",
    mixed: "rgba(148,163,184,.6)",
  };

  const option = {
    tooltip: {
      trigger: "axis",
      formatter: (p: any[]) => {
        const bar = p.find((s) => s.seriesName === "생산 수량");
        const line = p.find((s) => s.seriesName === "평균 예측 health");
        const row = weekly[p[0].dataIndex];
        return (
          `<b>${p[0].axisValue}</b> · ${row?.split ?? ""}<br/>` +
          (bar ? `생산 수량: ${Number(bar.value).toLocaleString()} unit<br/>` : "") +
          (line ? `평균 예측 health: ${Number(line.value).toFixed(6)}` : "")
        );
      },
    },
    legend: { data: ["생산 수량", "평균 예측 health"], bottom: 0, textStyle: { fontSize: 11, color: "#475569" } },
    grid: { top: 18, left: 56, right: 64, bottom: 40 },
    xAxis: {
      type: "category",
      data: weekly.map((d) => d.label),
      axisLabel: { fontSize: 10, color: "#64748b", interval: weekly.length > 16 ? 1 : 0 },
    },
    yAxis: [
      {
        type: "value",
        name: "수량",
        nameTextStyle: { fontSize: 10, color: "#94a3b8" },
        axisLabel: { fontSize: 9, color: "#94a3b8", formatter: (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v) },
        splitLine: { lineStyle: { color: "#f1f5f9" } },
      },
      {
        type: "value",
        name: "예측 health",
        nameTextStyle: { fontSize: 10, color: "#94a3b8" },
        axisLabel: { fontSize: 9, color: "#94a3b8", formatter: (v: number) => v.toExponential(0) },
        splitLine: { show: false },
      },
    ],
    series: [
      {
        name: "생산 수량",
        type: "bar",
        data: weekly.map((d) => ({ value: d.total, itemStyle: { color: splitColor[d.split] ?? splitColor.mixed, borderRadius: [4, 4, 0, 0] } })),
        barMaxWidth: 30,
      },
      {
        name: "평균 예측 health",
        type: "line",
        yAxisIndex: 1,
        data: weekly.map((d) => d.pred),
        smooth: true,
        symbolSize: 5,
        lineStyle: { color: "#ef4444", width: 2 },
        itemStyle: { color: "#ef4444" },
      },
    ],
  };

  return (
    <Panel
      title="주별 생산량"
      right={<span className="text-[11px] text-brand-textMuted">lot 7개 = 1주 · 막대=수량(split색) / 선=평균 예측 health</span>}
    >
      {loading ? (
        <div className="py-10 text-center text-brand-textMuted text-[12px]">로딩 중…</div>
      ) : weekly.length === 0 ? (
        <div className="py-10 text-center text-brand-textMuted text-[12px]">데이터 없음 (dashboard_dates.csv)</div>
      ) : (
        <ReactECharts option={option} style={{ height: 240 }} notMerge lazyUpdate />
      )}
    </Panel>
  );
}
