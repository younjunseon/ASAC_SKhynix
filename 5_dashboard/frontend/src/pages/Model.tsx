/**
 * Model 페이지 — 모델 성능 진단.
 *
 * 산출물 (build_model_artifacts.py):
 *   - fold_metrics.json     → Fold별 RMSE
 *   - feature_importance.csv → 5-fold 평균 LGBM gain (mu/pi)
 *   - psi.csv               → train ↔ val 분포 변화
 *   - var_compare.csv       → 위험 vs 정상 unit 변수 비교 (Cohen's d, p-value)
 */
import { useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  fetchFeatureImportance,
  fetchFoldMetrics,
  fetchPsi,
  fetchVarCompare,
} from "../lib/api";
import KpiCard from "../components/KpiCard";
import PageHeader from "../components/PageHeader";
import Panel from "../components/Panel";
import StatChip from "../components/StatChip";
import { fmtPpm, healthToPpm } from "../lib/format";
import {
  BAR_RADIUS,
  CHART_COLORS,
  CHART_GRID,
  CHART_TICK,
  CHART_TOOLTIP_STYLE,
  chartBox,
} from "../lib/chart";

export default function Model() {
  const foldQ = useQuery({ queryKey: ["model", "fold"], queryFn: fetchFoldMetrics });
  const fiQ = useQuery({
    queryKey: ["model", "fi"],
    queryFn: () => fetchFeatureImportance(10),
  });
  const psiQ = useQuery({ queryKey: ["model", "psi"], queryFn: () => fetchPsi(10) });
  const varQ = useQuery({
    queryKey: ["model", "var"],
    queryFn: () => fetchVarCompare(10),
  });

  const folds = foldQ.data?.folds ?? [];
  const meanRmse = foldQ.data?.mean_rmse;
  const stdRmse = foldQ.data?.std_rmse;
  const fi = fiQ.data?.items ?? [];
  const psi = psiQ.data?.items ?? [];
  const varCmp = varQ.data?.items ?? [];

  // RMSE를 ppm 단위로 (health × 1e6)
  const meanRmsePpm = meanRmse !== undefined ? healthToPpm(meanRmse) : null;
  const stdRmsePpm = stdRmse !== undefined ? healthToPpm(stdRmse) : null;

  // Fold RMSE를 ppm으로 변환한 차트 데이터
  const foldChartData = folds.map((f) => ({
    fold: `Fold ${f.fold}`,
    rmse_ppm: healthToPpm(f.rmse),
    n_units: f.n_units,
  }));
  const foldRmseDomain = (() => {
    if (foldChartData.length === 0) return undefined;
    const vals = foldChartData.map((d) => d.rmse_ppm);
    const lo = Math.floor(Math.min(...vals) * 0.9);
    const hi = Math.ceil(Math.max(...vals) * 1.1);
    return [lo, hi] as [number, number];
  })();

  return (
    <div>
      <PageHeader title="Model" subtitle="모델 성능 / 분포 변화 / 신뢰도 (실측 산출물 기반)" />

      {/* RMSE KPI — 모두 ppm 단위. 비교 KPI 2개는 사내 최우수 RMSE 값이 아직 미입력이라 TBD */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-4 mb-4 sm:mb-5">
        <KpiCard
          label="평균 RMSE"
          value={meanRmsePpm !== null ? fmtPpm(meanRmsePpm) : "—"}
          tone="info"
          hint={`5-fold OOF (std ${stdRmsePpm !== null ? fmtPpm(stdRmsePpm) : "—"})`}
        />
        <KpiCard
          label="vs 사내 최우수"
          value=""
          pending
          pendingHint="사내 최우수 RMSE 기준값 미입력"
        />
        <KpiCard
          label="목표"
          value=""
          pending
          pendingHint="목표 RMSE 기준값 미입력"
        />
      </div>

      {/* Fold별 RMSE + 잔차 분포 */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 sm:gap-5 mb-4 sm:mb-5">
        <Panel
          title="Fold별 RMSE (CV)"
          right={
            <span className="text-[11px] text-brand-textMuted">
              fold 1은 outlier 영향으로 높음
            </span>
          }
        >
          <div style={chartBox(220)}>
            <ResponsiveContainer>
              <BarChart data={foldChartData}>
                <CartesianGrid {...CHART_GRID} />
                <XAxis dataKey="fold" tick={CHART_TICK} />
                <YAxis
                  tick={CHART_TICK}
                  domain={foldRmseDomain}
                  tickFormatter={(v) => `${Math.round(v).toLocaleString()}`}
                  label={{ value: "ppm", angle: -90, position: "insideLeft", fontSize: 11, fill: "#64748b" }}
                />
                <Tooltip
                  contentStyle={CHART_TOOLTIP_STYLE}
                  formatter={(v: any) => `${Math.round(Number(v)).toLocaleString()} ppm`}
                />
                <Bar dataKey="rmse_ppm" fill={CHART_COLORS.primary} radius={BAR_RADIUS}>
                  {foldChartData.map((d, i) => (
                    <Cell
                      key={i}
                      fill={
                        d.rmse_ppm > (meanRmsePpm ?? 0) * 1.5
                          ? CHART_COLORS.danger
                          : CHART_COLORS.primary
                      }
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        {/* PSI (train ↔ val) */}
        <Panel
          title="분포 변화 PSI Top 10 (train ↔ val)"
          right={
            <StatChip
              label="안정성"
              value={psi[0]?.psi ?? 0}
              threshold={0.25}
              decimals={3}
              hint="τ=0.25 초과 시 분포 이동"
            />
          }
        >
          <div style={chartBox(220)}>
            <ResponsiveContainer>
              <BarChart data={psi} margin={{ top: 10, right: 10, bottom: 5, left: 10 }}>
                <CartesianGrid {...CHART_GRID} />
                <XAxis dataKey="feature" tick={{ ...CHART_TICK, fontSize: 10 }} />
                <YAxis tick={CHART_TICK} tickFormatter={(v) => v.toFixed(3)} />
                <Tooltip
                  contentStyle={CHART_TOOLTIP_STYLE}
                  formatter={(v: any) => Number(v).toFixed(4)}
                />
                <ReferenceLine
                  y={0.25}
                  stroke={CHART_COLORS.danger}
                  strokeDasharray="3 3"
                  label={{ value: "τ=0.25", fontSize: 10, fill: CHART_COLORS.danger, position: "right" }}
                />
                <Bar dataKey="psi" fill={CHART_COLORS.primary} radius={BAR_RADIUS} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="text-[10px] text-brand-textMuted mt-1 px-1">
            PSI 최대 {(psi[0]?.psi ?? 0).toFixed(3)} — 모든 변수 안정 구간 (PSI &lt; 0.1)
          </div>
        </Panel>
      </div>

      {/* Feature Importance (mu, pi 분리 표시) */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 sm:gap-5 mb-4 sm:mb-5">
        <Panel
          title="주요 변수 Top 10 — μ (회귀 stage)"
          right={
            <span className="text-[11px] text-brand-textMuted">
              health &gt; 0인 unit의 ppm 크기 결정
            </span>
          }
        >
          <div style={chartBox(280)}>
            <ResponsiveContainer>
              <BarChart data={fi} layout="vertical" margin={{ left: 30 }}>
                <CartesianGrid {...CHART_GRID} />
                <XAxis type="number" tick={CHART_TICK} />
                <YAxis
                  type="category"
                  dataKey="feature"
                  tick={{ ...CHART_TICK, fontSize: 10 }}
                  width={50}
                />
                <Tooltip
                  contentStyle={CHART_TOOLTIP_STYLE}
                  formatter={(v: any) => Math.round(Number(v)).toLocaleString()}
                />
                <Bar dataKey="mu_gain" fill={CHART_COLORS.accent} radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel
          title="주요 변수 Top 10 — π (분류 stage)"
          right={
            <span className="text-[11px] text-brand-textMuted">
              health=0 vs 0+ 분리 신호
            </span>
          }
        >
          <div style={chartBox(280)}>
            <ResponsiveContainer>
              <BarChart data={fi} layout="vertical" margin={{ left: 30 }}>
                <CartesianGrid {...CHART_GRID} />
                <XAxis type="number" tick={CHART_TICK} />
                <YAxis
                  type="category"
                  dataKey="feature"
                  tick={{ ...CHART_TICK, fontSize: 10 }}
                  width={50}
                />
                <Tooltip
                  contentStyle={CHART_TOOLTIP_STYLE}
                  formatter={(v: any) => Math.round(Number(v)).toLocaleString()}
                />
                <Bar dataKey="pi_gain" fill={CHART_COLORS.primary} radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      </div>

      {/* 위험 vs 정상 변수 비교 (실데이터) */}
      <Panel title="위험군 vs 정상군 변수 비교 (Welch's t-test + Cohen's d)">
        <div className="overflow-x-auto">
          <table className="spotfire">
            <thead>
              <tr>
                <th>변수</th>
                <th className="text-right">효과 크기 (d)</th>
                <th className="text-right">p-value</th>
                <th className="text-right">위험 평균</th>
                <th className="text-right">정상 평균</th>
                <th>유의수준</th>
                <th>해석</th>
              </tr>
            </thead>
            <tbody>
              {varCmp.map((v) => (
                <tr key={v.feature}>
                  <td className="font-mono">{v.feature}</td>
                  <td className="text-right tabular font-mono font-semibold">
                    {v.cohens_d.toFixed(2)}
                  </td>
                  <td className="text-right tabular font-mono">
                    {v.p_value < 1e-6 ? "<1e-6" : v.p_value.toExponential(1)}
                  </td>
                  <td className="text-right tabular font-mono text-brand-textMuted">
                    {v.mean_risk.toFixed(2)}
                  </td>
                  <td className="text-right tabular font-mono text-brand-textMuted">
                    {v.mean_norm.toFixed(2)}
                  </td>
                  <td>
                    <StatChip label="t-test" pValue={v.p_value} stat="" />
                  </td>
                  <td className="text-[11px] text-brand-textMuted">
                    {Math.abs(v.cohens_d) > 0.8
                      ? "강한 분리 신호"
                      : Math.abs(v.cohens_d) > 0.5
                      ? "중간 분리 신호"
                      : "약한 분리"}
                  </td>
                </tr>
              ))}
              {varCmp.length === 0 && (
                <tr>
                  <td colSpan={7} className="text-center text-brand-textMuted p-3">
                    산출물 없음 — `python 5_dashboard/build_model_artifacts.py` 실행 필요
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}
