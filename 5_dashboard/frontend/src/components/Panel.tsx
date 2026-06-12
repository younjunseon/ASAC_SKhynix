import type { ReactNode } from "react";

interface Props {
  title: string;
  right?: ReactNode;
  children: ReactNode;
  bodyClassName?: string;
  className?: string;
  titleClassName?: string;
}

export default function Panel({ title, right, children, bodyClassName = "", className = "", titleClassName = "" }: Props) {
  return (
    <div className={`panel flex flex-col ${className}`}>
      <div className="panel-title">
        <span className={titleClassName || undefined}>{title}</span>
        {right}
      </div>
      <div className={`panel-body flex-1 ${bodyClassName}`}>{children}</div>
    </div>
  );
}
