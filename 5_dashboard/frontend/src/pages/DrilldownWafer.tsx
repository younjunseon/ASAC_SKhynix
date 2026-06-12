import ComingSoon from "../components/ComingSoon";

/** 드릴다운 > 웨이퍼 차원 — 빈 페이지 (Delta-Q Map 형태로 구현 예정) */
export default function DrilldownWafer() {
  return (
    <ComingSoon
      title="웨이퍼 차원 분석"
      subtitle="개별 웨이퍼 단위 예측·잔차 패턴 — Delta-Q Map 형태로 구현 예정"
      plans={[
        "웨이퍼 맵 위 die별 예측 health / 잔차 히트맵",
        "공정 조건(로트·장비) 기준선 대비 편차(ΔQ) 시각화",
        "위험 die 클러스터 자동 탐지 및 하이라이트",
      ]}
    />
  );
}
