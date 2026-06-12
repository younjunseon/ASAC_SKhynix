/**
 * 예측 성능 섹션 — 용인 대시보드 ModelPerformance 시각화 이식 (echarts + 정적 CSV).
 *
 * 데이터: public/metrics.csv (stage/model/split/metric/value),
 *        public/dashboard_units.csv (ufs_serial/split/health/reg_pred/risk)
 *  ⚠ 준선 /api/model/* (변수 진단) 와는 별개 실험 스냅샷 — 모델링 최종본 확정 시 통일 예정.
 *
 * 박스플롯은 의도적으로 제외.
 */
import { useMemo } from "react";
import ReactECharts from "echarts-for-react";
import { useCSV } from "../hooks/useCSV";
import Panel from "./Panel";

const BASELINE_RMSE = 0.015; // 사내 최우수 RMSE 기준 (참고용)

type MetricRow = { stage?: string; model?: string; split?: string; metric?: string; value?: number };
type UnitRow = { ufs_serial?: string; split?: string; health?: number; reg_pred?: number; risk?: string };

const num = (v: unknown): number => {
  const n = typeof v === "number" ? v : parseFloat(String(v));
  return Number.isFinite(n) ? n : 0;
};

function MetricCard({
  label,
  value,
  color,
  sub,
  highlight,
}: {
  label: string;
  value: string;
  color: string;
  sub?: string;
  highlight?: boolean;
}) {
  return (
    <div
      className="panel px-5 py-4 flex flex-col gap-1.5"
      style={{ borderTop: `3px solid ${color}`, background: highlight ? color + "0a" : undefined }}
    >
      <div className="text-[22px] font-black tabular leading-none" style={{ color }}>
        {value}
      </div>
      <div className="text-[12px] text-brand-text font-medium">{label}</div>
      {sub && <div className="text-[10.5px] text-brand-textMuted">{sub}</div>}
    </div>
  );
}

export default function PredictionPerfSection() {
  const { data: metricsRaw, loading: mLoading } = useCSV<MetricRow>("/metrics.csv");
  const { data: unitsRaw, loading: uLoading } = useCSV<UnitRow>("/dashboard_units.csv");

  const metrics = useMemo(() => {
    if (!metricsRaw.length) return null;
    const get = (stage: string, model: string, split: string, metric = "rmse"): number | null => {
      const row = metricsRaw.find(
        (r) => r.stage === stage && r.model === model && r.split === split && r.metric === metric
      );
      return row && row.value != null ? num(row.value) : null;
    };
    return {
      lgbm_val: get("reg", "lgbm", "val"),
      lgbm_test: get("reg", "lgbm", "test"),
      et_val: get("reg", "et", "val"),
      et_test: get("reg", "et", "test"),
      enet_val: get("reg", "enet", "val"),
      enet_test: get("reg", "enet", "test"),
      ensemble_oof: get("reg", "ensemble", "oof"),
      ensemble_val: get("reg", "ensemble", "val"),
      clf_auc_val: get("clf", "soft_vote", "val", "auc"),
      clf_auc_test: get("clf", "soft_vote", "test", "auc"),
      clf_recall_val: get("clf", "soft_vote", "val", "recall"),
      clf_ap_val: get("clf", "soft_vote", "val", "ap"),
    };
  }, [metricsRaw]);

  // 산점도 데이터 — split별 분리 + 다운샘플 (train 600, val 300, test 200)
  const scatter = useMemo(() => {
    const buckets: Record<"train" | "val" | "test", [number, number][]> = { train: [], val: [], test: [] };
    for (const r of unitsRaw) {
      const sp = (r.split as "train" | "val" | "test") ?? "train";
      if (buckets[sp]) buckets[sp].push([num(r.health), num(r.reg_pred)]);
    }
    const sample = (arr: [number, number][], n: number) =>
      arr.length <= n ? arr : arr.filter((_, i) => i % Math.ceil(arr.length / n) === 0).slice(0, n);
    return { train: sample(buckets.train, 600), val: sample(buckets.val, 300), test: sample(buckets.test, 200) };
  }, [unitsRaw]);

  const predVals = useMemo(() => unitsRaw.map((r) => num(r.reg_pred)).filter((v) => Number.isFinite(v)), [unitsRaw]);

  if (mLoading || uLoading || !metrics) {
    return (
      <div className="panel px-6 py-10 text-center text-brand-textMuted text-[13px] mb-4 sm:mb-5">
        예측 성능 데이터 로딩 중…
      </div>
    );
  }

  const bestVal = metrics.ensemble_val ?? metrics.lgbm_val ?? 0;
  const beatBase = bestVal > 0 && bestVal < BASELINE_RMSE;
  const improvement = bestVal > 0 ? (((BASELINE_RMSE - bestVal) / BASELINE_RMSE) * 100).toFixed(1) : "—";

  // ── 모델별 RMSE 비교 바차트 ──
  const barModels = ["LGBM", "ET", "ElasticNet", "Ensemble"];
  const valRmse = [metrics.lgbm_val, metrics.et_val, metrics.enet_val, metrics.ensemble_val];
  const testRmse = [metrics.lgbm_test, metrics.et_test, metrics.enet_test, null];
  const barOpt = {
    tooltip: {
      trigger: "axis",
      formatter: (p: any[]) =>
        `<b>${p[0].axisValue}</b><br/>` +
        p.filter((s) => s.value != null).map((s) => `${s.seriesName}: ${Number(s.value).toFixed(6)}`).join("<br/>"),
    },
    legend: { data: ["Val RMSE", "Test RMSE"], bottom: 0, textStyle: { fontSize: 11, color: "#475569" } },
    grid: { top: 16, left: 60, right: 20, bottom: 44 },
    xAxis: { type: "category", data: barModels, axisLabel: { fontSize: 12, color: "#475569", fontWeight: 600 } },
    yAxis: {
      type: "value",
      name: "RMSE",
      nameTextStyle: { fontSize: 10, color: "#94a3b8" },
      axisLabel: { fontSize: 9, color: "#94a3b8", formatter: (v: number) => v.toFixed(4) },
      splitLine: { lineStyle: { color: "#f1f5f9" } },
      min: 0,
      // 기준선(0.015)이 차트 안에 보이도록 상한 확보
      max: (v: { max: number }) => parseFloat((Math.max(v.max, BASELINE_RMSE) * 1.12).toFixed(6)),
    },
    series: [
      {
        name: "Val RMSE",
        type: "bar",
        data: valRmse,
        barMaxWidth: 36,
        itemStyle: {
          color: (p: any) => (p.dataIndex === 3 ? "#3b82f6" : "rgba(59,130,246,.45)"),
          borderRadius: [4, 4, 0, 0],
        },
        label: { show: true, position: "top", fontSize: 9, formatter: (p: any) => (p.value != null ? Number(p.value).toFixed(5) : ""), color: "#475569" },
        markLine: {
          silent: true,
          symbol: "none",
          data: [
            {
              yAxis: BASELINE_RMSE,
              lineStyle: { color: "#ef4444", type: "dashed", width: 2 },
              label: { formatter: `기준 ${BASELINE_RMSE}`, color: "#ef4444", fontSize: 10, position: "end" },
            },
          ],
        },
      },
      {
        name: "Test RMSE",
        type: "bar",
        data: testRmse,
        barMaxWidth: 36,
        itemStyle: { color: "rgba(249,115,22,.6)", borderRadius: [4, 4, 0, 0] },
        label: { show: true, position: "top", fontSize: 9, formatter: (p: any) => (p.value != null ? Number(p.value).toFixed(5) : ""), color: "#475569" },
      },
    ],
  };

  // ── 실제 vs 예측 산점도 ──
  const allScatter = [...scatter.train, ...scatter.val, ...scatter.test];
  const maxVal = Math.max(...allScatter.map((d) => Math.max(d[0], d[1])), 0.01);
  const scatterOpt = {
    tooltip: { formatter: (p: any) => `실제: ${Number(p.data[0]).toFixed(6)}<br/>예측: ${Number(p.data[1]).toFixed(6)}<br/>${p.seriesName}` },
    legend: { data: ["Train(불량)", "Train(정상)", "Val", "Test"], bottom: 0, textStyle: { fontSize: 10, color: "#475569" } },
    grid: { top: 20, left: 64, right: 20, bottom: 44 },
    xAxis: { type: "value", name: "실제 health", nameTextStyle: { fontSize: 10, color: "#94a3b8" }, axisLabel: { fontSize: 9, color: "#94a3b8" }, splitLine: { lineStyle: { color: "#f1f5f9" } }, max: maxVal },
    yAxis: { type: "value", name: "예측 health", nameTextStyle: { fontSize: 10, color: "#94a3b8" }, axisLabel: { fontSize: 9, color: "#94a3b8" }, splitLine: { lineStyle: { color: "#f1f5f9" } }, max: maxVal },
    series: [
      { type: "line", data: [[0, 0], [maxVal, maxVal]], lineStyle: { color: "#94a3b8", type: "dashed", width: 1 }, symbol: "none", silent: true },
      { name: "Train(불량)", type: "scatter", data: scatter.train.filter((d) => d[0] > 0), symbolSize: 5, itemStyle: { color: "rgba(239,68,68,.6)" } },
      { name: "Train(정상)", type: "scatter", data: scatter.train.filter((d) => d[0] === 0), symbolSize: 4, itemStyle: { color: "rgba(59,130,246,.22)" } },
      { name: "Val", type: "scatter", data: scatter.val, symbolSize: 5, itemStyle: { color: "rgba(139,92,246,.7)" } },
      { name: "Test", type: "scatter", data: scatter.test, symbolSize: 5, itemStyle: { color: "rgba(6,182,212,.7)" } },
    ],
  };

  // ── 예측값 분포 히스토그램 ──
  const maxPred = Math.max(...predVals, 1e-6);
  const BIN_N = 24;
  const binSize = maxPred / BIN_N;
  const bins = Array.from({ length: BIN_N }, (_, i) => i * binSize);
  const histCounts = bins.map((b, i) => {
    const next = i === BIN_N - 1 ? Infinity : bins[i + 1];
    return predVals.filter((v) => v >= b && v < next).length;
  });
  const histOpt = {
    tooltip: { trigger: "axis", formatter: (p: any[]) => `구간 ${Number(p[0].name).toFixed(5)}<br/>건수: ${p[0].value}` },
    grid: { top: 10, left: 56, right: 20, bottom: 40 },
    xAxis: { type: "category", data: bins.map((b) => b.toFixed(4)), axisLabel: { fontSize: 8, color: "#94a3b8", interval: 5 } },
    yAxis: { type: "value", axisLabel: { fontSize: 10, color: "#94a3b8" }, splitLine: { lineStyle: { color: "#f1f5f9" } } },
    series: [
      {
        type: "bar",
        data: histCounts,
        barWidth: "98%",
        itemStyle: { color: (p: any) => (p.dataIndex === 0 ? "rgba(34,197,94,.75)" : "rgba(59,130,246,.6)"), borderRadius: [3, 3, 0, 0] },
      },
    ],
  };

  const clfAuc = metrics.clf_auc_val;
  const clfQual = clfAuc == null ? "—" : clfAuc > 0.65 ? "양호" : clfAuc > 0.58 ? "보통" : "랜덤에 가까움";

  return (
    <div className="space-y-4 sm:space-y-5 mb-5 sm:mb-6">
      {/* RMSE 카드 */}
      <div className="grid grid-cols-2 xl:grid-cols-4 gap-3">
        <MetricCard label="Ensemble Val RMSE" value={bestVal ? bestVal.toFixed(6) : "—"} color="#3b82f6" sub="최종 앙상블" highlight />
        <MetricCard label="LGBM Val RMSE" value={metrics.lgbm_val != null ? metrics.lgbm_val.toFixed(6) : "—"} color="#8b5cf6" sub="LightGBM 단독" />
        <MetricCard label="ExtraTrees Val RMSE" value={metrics.et_val != null ? metrics.et_val.toFixed(6) : "—"} color="#06b6d4" sub="ExtraTrees 단독" />
        <MetricCard
          label={beatBase ? "기준 대비 개선" : "사내 최우수 기준"}
          value={beatBase ? `-${improvement}%` : BASELINE_RMSE.toFixed(4)}
          color={beatBase ? "#16a34a" : "#ef4444"}
          sub={beatBase ? `기준 ${BASELINE_RMSE} 대비 (참고)` : "RMSE (참고용)"}
        />
      </div>

      {/* Stage 1 분류 성능 */}
      {clfAuc != null && (
        <Panel title="Stage 1 분류 성능" right={<span className="text-[11px] text-brand-textMuted">clf_proba — soft_vote · val 기준</span>}>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-3">
            {[
              { label: "AUC (Val)", value: metrics.clf_auc_val, color: "#3b82f6" },
              { label: "AUC (Test)", value: metrics.clf_auc_test, color: "#06b6d4" },
              { label: "Recall (Val)", value: metrics.clf_recall_val, color: "#ef4444", warn: (metrics.clf_recall_val ?? 1) < 0.1 },
              { label: "AP (Val)", value: metrics.clf_ap_val, color: "#f97316" },
            ].map((s) => (
              <div key={s.label} className="rounded-lg px-3.5 py-3" style={{ background: "#f8fafc", border: `1px solid ${s.color}33`, borderTop: `3px solid ${s.color}` }}>
                <div className="text-[18px] font-bold tabular" style={{ color: s.color }}>
                  {s.value != null ? s.value.toFixed(4) : "—"}
                </div>
                <div className="text-[11px] text-brand-textMuted mt-0.5">{s.label}</div>
                {s.warn && <div className="text-[10px] text-brand-danger font-semibold mt-0.5">⚠ 매우 낮음</div>}
              </div>
            ))}
          </div>
          <div className="tbd-block">
            💡 Recall이 낮을수록 Stage 1이 실제 불량 unit을 잘 못 잡는다는 의미 → Two-Stage 구조의 한계.
            AUC {clfAuc.toFixed(3)} = {clfQual}.
          </div>
        </Panel>
      )}

      {/* 모델 비교 + 산점도 */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 sm:gap-5">
        <Panel title="모델별 RMSE 비교" right={<span className="text-[11px] text-brand-textMuted">Val / Test · 빨간선=기준</span>}>
          <ReactECharts option={barOpt} style={{ height: 250 }} notMerge lazyUpdate />
        </Panel>
        <Panel title="실제 vs 예측값 산점도" right={<span className="text-[11px] text-brand-textMuted">split별 다운샘플</span>}>
          <div className="text-[11px] text-brand-textMuted mb-1.5">
            <span style={{ color: "#ef4444" }}>●</span> Train 불량(health&gt;0) &nbsp;
            <span style={{ color: "#3b82f6" }}>●</span> Train 정상(health=0) &nbsp;·&nbsp; 점선 = 완벽 예측
          </div>
          <ReactECharts option={scatterOpt} style={{ height: 222 }} notMerge lazyUpdate />
        </Panel>
      </div>

      {/* 예측값 분포 히스토그램 */}
      <Panel title="예측값 분포" right={<span className="text-[11px] text-brand-textMuted">reg_pred · Zero-inflated 확인</span>}>
        <div className="text-[11px] text-brand-textMuted mb-1.5">
          <span style={{ color: "#22c55e", fontWeight: 600 }}>초록</span> = 정상(y≈0) 구간 &nbsp;·&nbsp;
          <span style={{ color: "#3b82f6", fontWeight: 600 }}>파랑</span> = 양수 예측 구간
        </div>
        <ReactECharts option={histOpt} style={{ height: 210 }} notMerge lazyUpdate />
      </Panel>

      {/* 모델별 RMSE 상세 테이블 */}
      <Panel title="모델별 RMSE 상세" right={<span className="text-[11px] text-brand-textMuted">metrics.csv</span>}>
        <table className="spotfire">
          <thead>
            <tr>
              <th>모델</th>
              <th className="text-right">Val RMSE</th>
              <th className="text-right">Test RMSE</th>
              <th className="text-right">기준 대비</th>
            </tr>
          </thead>
          <tbody>
            {[
              { name: "LGBM", val: metrics.lgbm_val, test: metrics.lgbm_test, color: "#8b5cf6" },
              { name: "ExtraTrees", val: metrics.et_val, test: metrics.et_test, color: "#06b6d4" },
              { name: "ElasticNet", val: metrics.enet_val, test: metrics.enet_test, color: "#f97316" },
              { name: "Ensemble ★", val: metrics.ensemble_val, test: null, color: "#3b82f6" },
            ].map((m, i) => {
              const v = m.val ?? 0;
              const beat = v > 0 && v < BASELINE_RMSE;
              const diff = v > 0 ? (((BASELINE_RMSE - v) / BASELINE_RMSE) * 100).toFixed(1) : "—";
              return (
                <tr key={m.name} style={{ background: i === 3 ? "rgba(59,130,246,.05)" : undefined }}>
                  <td style={{ fontWeight: 700, color: m.color }}>{m.name}</td>
                  <td className="text-right tabular" style={{ color: beat ? "#16a34a" : "#ef4444", fontWeight: 600 }}>
                    {m.val != null ? m.val.toFixed(6) : "—"}
                  </td>
                  <td className="text-right tabular text-brand-textMuted">{m.test != null ? m.test.toFixed(6) : "—"}</td>
                  <td className="text-right tabular" style={{ fontWeight: 700, color: beat ? "#16a34a" : "#ef4444" }}>
                    {v > 0 ? (beat ? `-${diff}%` : `+${Math.abs(parseFloat(diff))}%`) : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Panel>
    </div>
  );
}
