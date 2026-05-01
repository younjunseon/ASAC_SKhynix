import type { ReactNode } from "react";

interface Props {
  title: string;
  right?: ReactNode;
  children: ReactNode;
  bodyClassName?: string;
  className?: string;
}

export default function Panel({ title, right, children, bodyClassName = "", className = "" }: Props) {
  return (
    <div className={`panel flex flex-col ${className}`}>
      <div className="panel-title">
        <span>{title}</span>
        {right}
      </div>
      <div className={`panel-body flex-1 ${bodyClassName}`}>{children}</div>
    </div>
  );
}
