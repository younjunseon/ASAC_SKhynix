/**
 * 주요 변수 — 트리 모델 평균 importance.
 *
 * LGBM gain 하나만 쓰지 않고 트리 모델들(LGBM gain · ExtraTrees impurity)을 각각 정규화 후 평균.
 * 데이터: public/feature_importance.csv (feature, lgbm_gain, et_impurity, enet_abs_coef, *_rank ...)
 */
import { useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import { useCSV } from "../hooks/useCSV";
import Panel from "./Panel";

type FIRow = {
  feature?: string;
  lgbm_gain?: number;
  et_impurity?: number;
  enet_abs_coef?: number;
  lgbm_rank?: number;
  et_rank?: number;
};

const num = (v: unknown): number => {
  const n = typeof v === "number" ? v : parseFloat(String(v));
  return Number.isFinite(n) ? n : 0;
};

const TOP_N = 15;

export default function TreeImportanceChart() {
  const { data, loading } = useCSV<FIRow>("/feature_importance.csv");
  const [showEnet, setShowEnet] = useState(false);

  const rows = useMemo(() => {
    if (!data.length) return [];
    const maxLgbm = Math.max(...data.map((r) => num(r.lgbm_gain)), 1e-12);
    const maxEt = Math.max(...data.map((r) => num(r.et_impurity)), 1e-12);
    const maxEnet = Math.max(...data.map((r) => Math.abs(num(r.enet_abs_coef))), 1e-12);
    return data
      .map((r) => {
        const lgbm = num(r.lgbm_gain) / maxLgbm; // 0~1
        const et = num(r.et_impurity) / maxEt; // 0~1
        const enet = Math.abs(num(r.enet_abs_coef)) / maxEnet; // 0~1
        return {
          feature: String(r.feature ?? ""),
          lgbm,
          et,
          enet,
          treeAvg: (lgbm + et) / 2, // 트리 평균
        };
      })
      .filter((r) => r.feature)
      .sort((a, b) => b.treeAvg - a.treeAvg)
      .slice(0, TOP_N);
  }, [data]);

  const features = rows.map((r) => r.feature);
  const option = {
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      formatter: (p: any[]) => {
        const i = p[0].dataIndex;
        const r = rows[i];
        return (
          `<b>${r.feature}</b><br/>` +
          `트리 평균: ${(r.treeAvg * 100).toFixed(1)}%<br/>` +
          `LGBM gain(정규화): ${(r.lgbm * 100).toFixed(1)}%<br/>` +
          `ET impurity(정규화): ${(r.et * 100).toFixed(1)}%` +
          (showEnet ? `<br/>ElasticNet |coef|(정규화): ${(r.enet * 100).toFixed(1)}%` : "")
        );
      },
    },
    legend: {
      data: showEnet ? ["LGBM gain", "ET impurity", "ElasticNet |coef|"] : ["LGBM gain", "ET impurity"],
      bottom: 0,
      textStyle: { fontSize: 11, color: "#475569" },
    },
    grid: { top: 8, left: 56, right: 24, bottom: 40 },
    xAxis: {
      type: "value",
      max: 1,
      axisLabel: { fontSize: 9, color: "#94a3b8", formatter: (v: number) => `${Math.round(v * 100)}%` },
      splitLine: { lineStyle: { color: "#f1f5f9" } },
    },
    yAxis: {
      type: "category",
      data: features,
      inverse: true, // top feature 가 위
      axisLabel: { fontSize: 11, color: "#475569", fontFamily: "Consolas, monospace" },
    },
    series: [
      { name: "LGBM gain", type: "bar", data: rows.map((r) => r.lgbm), barMaxWidth: 9, itemStyle: { color: "#3b82f6", borderRadius: [0, 3, 3, 0] } },
      { name: "ET impurity", type: "bar", data: rows.map((r) => r.et), barMaxWidth: 9, itemStyle: { color: "#06b6d4", borderRadius: [0, 3, 3, 0] } },
      ...(showEnet
        ? [{ name: "ElasticNet |coef|", type: "bar", data: rows.map((r) => r.enet), barMaxWidth: 9, itemStyle: { color: "#f59e0b", borderRadius: [0, 3, 3, 0] } }]
        : []),
    ],
  };

  return (
    <Panel
      title="주요 변수 — 트리 모델 평균 importance"
      right={
        <label className="text-[11px] text-brand-textMuted flex items-center gap-1.5 cursor-pointer">
          <input type="checkbox" checked={showEnet} onChange={(e) => setShowEnet(e.target.checked)} />
          ElasticNet도 보기
        </label>
      }
    >
      {loading ? (
        <div className="py-10 text-center text-brand-textMuted text-[12px]">로딩 중…</div>
      ) : rows.length === 0 ? (
        <div className="py-10 text-center text-brand-textMuted text-[12px]">데이터 없음 (feature_importance.csv)</div>
      ) : (
        <>
          <ReactECharts option={option} style={{ height: 360 }} notMerge lazyUpdate />
          <div className="text-[10.5px] text-brand-textMuted mt-1 px-1">
            LGBM gain과 ExtraTrees impurity를 각각 0~100%로 정규화 후 <b>평균</b>한 순으로 상위 {TOP_N}개. (LGBM 단독이 아닌 트리들의 합의)
          </div>
        </>
      )}
    </Panel>
  );
}
