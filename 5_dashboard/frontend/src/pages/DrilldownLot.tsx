import ComingSoon from "../components/ComingSoon";

/** 드릴다운 > 로트 차원 — 빈 페이지 (Delta-Q Map 형태로 구현 예정) */
export default function DrilldownLot() {
  return (
    <ComingSoon
      title="로트 차원 분석"
      subtitle="작업(로트) 단위 위험도 비교 — Delta-Q Map 형태로 구현 예정"
      plans={[
        "로트별 예측 불량률·평균 health 랭킹",
        "로트 내 웨이퍼 간 편차(ΔQ) 비교",
        "전체 평균 대비 이상 로트 하이라이트",
      ]}
    />
  );
}
