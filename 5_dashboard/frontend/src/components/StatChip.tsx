/**
 * 통계 검정 결과 chip.
 * design.md §5.2 표시 원칙: p-value에 따라 색 구분.
 *
 * 사용 예:
 *   <StatChip label="Mann-Kendall" pValue={0.003} hint="시계열 추세 검정" />
 *   <StatChip label="χ² lot vs baseline" pValue={0.008} />
 *   <StatChip label="PSI" value={0.31} threshold={0.25} hint="분포 drift" />
 */
interface PValueProps {
  label: string;
  pValue: number;
  hint?: string;
  /** 통계량 값을 함께 표시 (예: t=2.34) */
  stat?: string;
}

interface ThresholdProps {
  label: string;
  value: number;
  threshold: number;
  /** value가 threshold 이상일 때 위험으로 간주할지 (기본 true) */
  higherIsBad?: boolean;
  hint?: string;
  /** 표시할 소수점 자리수 (기본 3) */
  decimals?: number;
}

type Props = PValueProps | ThresholdProps;

function isPValue(p: Props): p is PValueProps {
  return "pValue" in p;
}

export default function StatChip(props: Props) {
  let toneClass = "chip-muted";
  let body = "";
  let icon = "";

  if (isPValue(props)) {
    const { pValue, label, stat } = props;
    if (pValue < 0.001) {
      toneClass = "chip-success";
      icon = "✓";
    } else if (pValue < 0.05) {
      toneClass = "chip-warn";
      icon = "⚠";
    } else {
      toneClass = "chip-muted";
      icon = "·";
    }
    const pStr =
      pValue < 0.001 ? "p<0.001" : pValue < 0.01 ? `p<0.01` : `p=${pValue.toFixed(3)}`;
    body = `${label}: ${stat ? stat + " " : ""}${pStr}`;
  } else {
    const { value, threshold, higherIsBad = true, label, decimals = 3 } = props;
    const bad = higherIsBad ? value >= threshold : value <= threshold;
    toneClass = bad ? "chip-danger" : "chip-success";
    icon = bad ? "⚠" : "✓";
    body = `${label} ${value.toFixed(decimals)} (τ=${threshold.toFixed(decimals)})`;
  }

  return (
    <span className={`chip ${toneClass}`} title={props.hint}>
      <span aria-hidden>{icon}</span>
      <span>{body}</span>
    </span>
  );
}