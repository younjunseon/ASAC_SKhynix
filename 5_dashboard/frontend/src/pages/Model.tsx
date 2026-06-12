/**
 * Model 페이지 — 모델 성능 분석.
 *
 * 구성 (모두 정적 CSV + echarts 기반):
 *  1) 예측 성능 — PredictionPerfSection
 *     RMSE 카드 / Stage1 분류 성능 / 모델별 RMSE 비교 / 실제vs예측 산점도 / 예측 분포 히스토그램 / 상세 테이블
 *     ※ 박스플롯은 의도적으로 제외.
 *  2) 주요 변수 — TreeImportanceChart  (LGBM gain · ET impurity 정규화 평균 — LGBM 단독 아님)
 *  3) 피처 변곡점 — FeatureCurveChart   (구 SHAP beeswarm 자리 대체 — 변수값 구간별 평균 health/불량률)
 *
 * 데이터: public/metrics.csv, dashboard_units.csv, feature_importance.csv, feature_dist.csv
 *  ⚠ 준선 /api/model/* 의 변수 진단(Feature Importance·SHAP·Pearson r·Cohen's d) 섹션은 제거함.
 */
import PageHeader from "../components/PageHeader";
import PredictionPerfSection from "../components/PredictionPerfSection";
import TreeImportanceChart from "../components/TreeImportanceChart";
import FeatureCurveChart from "../components/FeatureCurveChart";

export default function Model() {
  return (
    <div>
      <PageHeader title="모델 성능 분석" subtitle="예측 성능(RMSE·분류·잔차) + 주요 변수(트리 평균 importance) + 피처 변곡점" />

      {/* ───────── 1) 예측 성능 ───────── */}
      <div className="flex items-center gap-2 mb-3">
        <span className="text-[13px] font-bold text-brand-text">예측 성능</span>
        <span className="text-[10.5px] text-brand-textMuted">RMSE · Stage1 분류 · 실제vs예측 · 예측 분포</span>
      </div>
      <PredictionPerfSection />

      {/* ───────── 2) 주요 변수 — 트리 평균 importance ───────── */}
      <div className="flex items-center gap-2 mb-3 pt-1">
        <span className="text-[13px] font-bold text-brand-text">주요 변수</span>
        <span className="text-[10.5px] text-brand-textMuted">트리 모델(LGBM·ET) importance 정규화 평균</span>
      </div>
      <div className="mb-5 sm:mb-6">
        <TreeImportanceChart />
      </div>

      {/* ───────── 3) 피처 변곡점 ───────── */}
      <div className="flex items-center gap-2 mb-3">
        <span className="text-[13px] font-bold text-brand-text">피처 변곡점</span>
        <span className="text-[10.5px] text-brand-textMuted">선택한 변수의 값 구간별 평균 health·불량률 — 관계가 꺾이는 지점</span>
      </div>
      <FeatureCurveChart />
    </div>
  );
}
