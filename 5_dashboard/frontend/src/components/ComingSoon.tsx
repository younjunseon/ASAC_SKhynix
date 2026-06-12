import PageHeader from "./PageHeader";

interface Props {
  title: string;
  subtitle?: string;
  /** 구현 예정 항목 리스트 */
  plans?: string[];
}

/**
 * 아직 구현되지 않은 페이지용 placeholder.
 * 드릴다운 하위(웨이퍼 차원 / 로트 차원)는 추후 Delta-Q Map 형태로 채울 예정.
 */
export default function ComingSoon({ title, subtitle, plans }: Props) {
  return (
    <div>
      <PageHeader title={title} subtitle={subtitle} />
      <div className="panel max-w-3xl">
        <div className="panel-title">
          <span>준비 중</span>
          <span className="chip chip-tbd chip-sm">Delta-Q Map 예정</span>
        </div>
        <div className="panel-body">
          <div className="tbd-block mb-3">
            이 페이지는 아직 구현되지 않았습니다. 공정 조건(로트·장비) 기준선 대비 편차를 보여주는
            Delta-Q Map 형태로 채울 예정입니다.
          </div>
          {plans && plans.length > 0 && (
            <ul className="text-[12.5px] text-brand-textMuted list-disc pl-5 space-y-1.5">
              {plans.map((p) => (
                <li key={p}>{p}</li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}
