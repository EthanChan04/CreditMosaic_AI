"use client";

import { cn } from "@/lib/utils";

interface RiskLevelBadgeProps {
  level: string;
  score?: number;
  className?: string;
}

const levelStyles: Record<string, string> = {
  Low: "bg-[var(--risk-low)]/15 text-[var(--risk-low)] border-[var(--risk-low)]/30",
  Medium: "bg-[var(--risk-medium)]/15 text-[var(--risk-medium)] border-[var(--risk-medium)]/30",
  High: "bg-[var(--risk-high)]/15 text-[var(--risk-high)] border-[var(--risk-high)]/30",
  Critical: "bg-[var(--risk-critical)]/15 text-[var(--risk-critical)] border-[var(--risk-critical)]/30",
  Unknown: "bg-[var(--muted)] text-[var(--muted-foreground)] border-[var(--border)]",
};

export function RiskLevelBadge({ level, score, className }: RiskLevelBadgeProps) {
  const style = levelStyles[level] || levelStyles.Unknown;

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold border",
        style,
        className
      )}
    >
      {level}
      {score !== undefined && (
        <span className="opacity-70">
          ({typeof score === "number" ? score.toFixed(3) : score})
        </span>
      )}
    </span>
  );
}
