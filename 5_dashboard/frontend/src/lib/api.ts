import axios from "axios";

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE ?? "",
  timeout: 15000,
});

/**
 * Status — PI 운영 관점 unit 분류.
 * - completed : health 실측 끝난 unit (어제까지 검사 완료)
 * - pending   : WT 끝났지만 health 미측정 (예측만 있음)
 * - today     : 오늘 검사된 unit (가장 강조 — 의사결정 우선)
 */
export type Status = "completed" | "pending" | "today";
/** 필터 값: "all"은 전체 표시 */
export type StatusFilter = Status | "all";

export interface StatusStats {
  n_units: number;
  n_risk: number;
  risk_ratio: number;
  pred_mean: number;
  pred_max: number;
  health_zero_ratio: number;
  rmse?: number;  // completed에만 있음 (실측 health 있어야 계산 가능)
}

export interface Overview {
  risk_top_ratio: number;
  statuses: Record<Status, StatusStats>;
  totals: { n_units: number; n_dies: number; n_wafers: number };
}

export interface WaferSummary {
  wafer_key: string;
  run_id: string;
  wafer_no: string;
  status: Status;
  n_units: number;
  mean_pred: number;
  max_pred: number;
  n_risk: number;
  mean_health: number;
  risk_ratio: number;
}

export interface UnitItem {
  ufs_serial: string;
  pred: number;
  health: number;
  status: Status;
  inspected_date: string;  // ISO date "YYYY-MM-DD"
  run_id: string;
  wafer_no: string;
  wafer_key: string;
  pi_mean: number;
  mu_mean: number;
  is_risk: boolean;
}

export interface DieItem {
  run_wf_xy: string;
  die_x: number;
  die_y: number;
  position?: number;
  pi: number;
  one_minus_pi: number;
  mu: number;
  pred: number;
  health?: number;
  ufs_serial?: string;
  status?: Status;
}



export const fetchOverview = () =>
  api.get<Overview>("/api/overview").then((r) => r.data);

export interface WaferGrid {
  bounds: { x_min: number; x_max: number; y_min: number; y_max: number };
  width: number;
  height: number;
  /** 모든 wafer를 겹친 unique (die_x, die_y) 좌표 — wafer 진짜 외형 */
  mask: [number, number][];
}

export const fetchWaferGrid = () =>
  api.get<WaferGrid>("/api/wafer-grid").then((r) => r.data);

export interface AnomalyFeature {
  feature: string;
  unit_value: number;
  normal_mean: number;
  normal_std: number;
  z_score: number;
}

export interface AnomalyFeaturesResponse {
  ufs_serial: string;
  top_n: number;
  items: AnomalyFeature[];
}

export const fetchUnitAnomalyFeatures = (ufsSerial: string, topN: number = 10) =>
  api
    .get<AnomalyFeaturesResponse>(`/api/units/${ufsSerial}/anomaly_features`, {
      params: { top_n: topN },
    })
    .then((r) => r.data);

export interface GlobalAnomalyItem {
  feature: string;
  risk_mean: number;
  normal_mean: number;
  normal_std: number;
  z_score: number;
}

export interface GlobalAnomalyResponse {
  status: string;
  n_risk_units: number;
  top_n: number;
  items: GlobalAnomalyItem[];
}

export const fetchGlobalAnomalyFeatures = (params: { status?: StatusFilter; top_n?: number }) =>
  api
    .get<GlobalAnomalyResponse>("/api/anomaly_features_global", { params })
    .then((r) => r.data);

export const fetchWafers = (params: {
  status?: StatusFilter;
  sort?: string;
  limit?: number;
}) =>
  api
    .get<{ count: number; items: WaferSummary[] }>("/api/wafers", { params })
    .then((r) => r.data);

export const fetchWaferDetail = (waferKey: string) =>
  api
    .get<{
      wafer_key: string;
      summary: WaferSummary;
      dies: DieItem[];
      units: UnitItem[];
    }>(`/api/wafers/${waferKey}`)
    .then((r) => r.data);

export const fetchUnits = (params: {
  status?: StatusFilter;
  risk_only?: boolean;
  sort?: string;
  order?: "asc" | "desc";
  page?: number;
  page_size?: number;
}) =>
  api
    .get<{
      total: number;
      page: number;
      page_size: number;
      items: UnitItem[];
    }>("/api/units", { params })
    .then((r) => r.data);

export const fetchUnitDetail = (ufsSerial: string) =>
  api
    .get<{ unit: UnitItem; dies: DieItem[] }>(`/api/units/${ufsSerial}`)
    .then((r) => r.data);

export interface TriageSummary {
  n_units: number;
  n_risk: number;
  risk_ratio: number;
  estimated_loss: number;
  unit_cost: number;
  rmse: number;
}

export interface ScaleInfo {
  pred_min: number;
  pred_max: number;
  risk_threshold: number;
}

export interface TriageResponse {
  status: StatusFilter;
  summary: TriageSummary;
  scale: ScaleInfo;
  top_wafers: WaferSummary[];
  top_units: UnitItem[];
}

export interface LotItem {
  run_id: string;
  n_units: number;
  n_wafers: number;
  n_risk: number;
  mean_pred: number;
  max_pred: number;
  mean_health: number;
  risk_ratio: number;
  lift: number;
}

export interface LotsResponse {
  status: StatusFilter;
  baseline_risk_ratio: number;
  n_lots_total: number;
  items: LotItem[];
}

export interface LotDetail {
  run_id: string;
  status: StatusFilter;
  summary: {
    n_units: number;
    n_wafers: number;
    n_risk: number;
    risk_ratio: number;
    mean_pred: number;
    max_pred: number;
  };
  wafers: {
    wafer_key: string;
    wafer_no: string;
    n_units: number;
    n_risk: number;
    mean_pred: number;
    max_pred: number;
    risk_ratio: number;
  }[];
}

export const fetchLots = (params: { status?: StatusFilter; sort?: string; limit?: number }) =>
  api.get<LotsResponse>("/api/lots", { params }).then((r) => r.data);

export const fetchLotDetail = (runId: string, status: StatusFilter) =>
  api.get<LotDetail>(`/api/lots/${runId}`, { params: { status } }).then((r) => r.data);

export interface LotMap {
  run_id: string;
  agg: "mean" | "max";
  n_wafers: number;
  n_dies_total: number;
  n_unique_positions: number;
  dies: DieItem[];
}

export const fetchLotMap = (runId: string, params: { status?: StatusFilter; agg?: "mean" | "max" } = {}) =>
  api
    .get<LotMap>(`/api/lots/${runId}/aggregate-map`, { params })
    .then((r) => r.data);

export const fetchTriage = (params: {
  status?: StatusFilter;
  top_units?: number;
  top_wafers?: number;
  unit_cost?: number;
}) =>
  api.get<TriageResponse>("/api/triage", { params }).then((r) => r.data);

export interface UnitReport {
  ufs_serial: string;
  verdict: "WARNING" | "NORMAL";
  is_risk: boolean;
  pred: number;
  health: number | null;
  status: Status;
  inspected_date: string;
  wafer_key: string;
  pred_rank: number;
  threshold: number;
  pi_mean: number;
  mu_mean: number;
  worst_die: {
    run_wf_xy: string;
    die_x: number;
    die_y: number;
    pi: number;
    mu: number;
    pred: number;
  } | null;
  narrative: string[];
}

export const fetchUnitReport = (ufsSerial: string) =>
  api
    .get<UnitReport>(`/api/units/${ufsSerial}/report`)
    .then((r) => r.data);


/* ─── Model 산출물 (build_model_artifacts.py) ─────────────── */
export interface FoldMetric {
  fold: number;
  rmse: number;
  n_units: number;
}

export interface FoldMetrics {
  n_folds: number;
  seed: number;
  folds: FoldMetric[];
  mean_rmse: number;
  std_rmse: number;
  n_total_units: number;
}

export const fetchFoldMetrics = () =>
  api.get<FoldMetrics>("/api/model/fold-metrics").then((r) => r.data);

export interface FeatureImportance {
  feature: string;
  mu_gain: number;
  pi_gain: number;
  total_gain: number;
}

export const fetchFeatureImportance = (top = 10) =>
  api
    .get<{ items: FeatureImportance[]; n_total: number }>(
      "/api/model/feature-importance",
      { params: { top } },
    )
    .then((r) => r.data);

export interface PsiItem {
  feature: string;
  psi: number;
}

export const fetchPsi = (top = 10) =>
  api
    .get<{ items: PsiItem[] }>("/api/model/psi", { params: { top } })
    .then((r) => r.data);

export interface FeatureCorrItem {
  feature: string;
  r: number;
  abs_r: number;
}

export interface ShapItem {
  feature: string;
  shap: number;
  abs_shap: number;
  total_gain: number;
  cohens_d: number;
  mean_risk: number;
  mean_norm: number;
}

export const fetchShap = (top = 10) =>
  api
    .get<{ items: ShapItem[]; n_total: number; max_abs_shap: number }>(
      "/api/model/shap",
      { params: { top } },
    )
    .then((r) => r.data);

export const fetchFeatureCorr = (top = 10) =>
  api
    .get<{ items: FeatureCorrItem[]; n_total: number; max_abs_r: number }>(
      "/api/model/feature-corr",
      { params: { top } },
    )
    .then((r) => r.data);

export interface VarCompareItem {
  feature: string;
  cohens_d: number;
  p_value: number;
  mean_risk: number;
  mean_norm: number;
  n_risk: number;
  n_norm: number;
}

export const fetchVarCompare = (top = 10) =>
  api
    .get<{ items: VarCompareItem[] }>("/api/model/var-compare", { params: { top } })
    .then((r) => r.data);

export interface PositionRiskItem {
  position: number;
  n_dies: number;
  mean_risk_prob: number;
  mean_risk_pct: number;
  p75_risk_pct: number;
  p95_risk_pct: number;
}

export const fetchPositionRisk = () =>
  api.get<{ items: PositionRiskItem[] }>("/api/position_risk").then((r) => r.data);


/* ─── 알림 (오늘 데이터 기반 4종 패턴) ─────────────── */
export interface PositionAlert {
  position: number;
  n_dies: number;
  n_risk_dies: number;
  risk_ratio: number;
  lift: number;
  mean_pred: number;
}

export interface LotClusterAlert {
  run_id: string;
  n_risk_units: number;
  mean_pred: number;
  sample_wafer_key: string;
}

export interface PointClusterAlert {
  die_x: number;
  die_y: number;
  n_risk_dies: number;
  n_wafers: number;
  mean_pred: number;
  sample_wafer_key: string;
}

export interface SystematicLotAlert {
  run_id: string;
  n_units: number;
  n_risk: number;
  risk_ratio: number;
  lift: number;
  mean_pred: number;
  sample_wafer_key: string;
}

export interface AlertsTodayResponse {
  baseline?: { die_risk_threshold: number; unit_risk_ratio: number };
  position_alerts: PositionAlert[];
  lot_cluster_alerts: LotClusterAlert[];
  point_cluster_alerts: PointClusterAlert[];
  systematic_lot_alerts: SystematicLotAlert[];
}

export const fetchAlertsToday = () =>
  api.get<AlertsTodayResponse>("/api/alerts/today").then((r) => r.data);

