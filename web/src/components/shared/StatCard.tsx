"use client";

import { Skeleton } from "./Skeleton";
import { cn } from "@/lib/utils";

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: string;
  variant?: "default" | "success" | "warning" | "danger";
  loading?: boolean;
}

const variants: Record<string, string> = {
  default: "border-l-[var(--chart-1)]",
  success: "border-l-[var(--risk-low)]",
  warning: "border-l-[var(--risk-medium)]",
  danger: "border-l-[var(--risk-critical)]",
};

export function StatCard({
  title,
  value,
  subtitle,
  icon,
  variant = "default",
  loading = false,
}: StatCardProps) {
  if (loading) {
    return (
      <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-4">
        <Skeleton variant="text" className="w-20 mb-2" />
        <Skeleton variant="text" className="w-32 h-8 mb-1" />
        <Skeleton variant="text" className="w-24" />
      </div>
    );
  }

  return (
    <div
      className={cn(
        "bg-[var(--card)] border border-[var(--border)] border-l-4 rounded-lg p-4",
        variants[variant]
      )}
    >
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs font-medium text-[var(--muted-foreground)] uppercase tracking-wide">
          {title}
        </span>
        {icon && <span className="text-lg">{icon}</span>}
      </div>
      <div className="text-2xl font-bold text-[var(--foreground)]">{value}</div>
      {subtitle && (
        <div className="text-xs text-[var(--muted-foreground)] mt-1">{subtitle}</div>
      )}
    </div>
  );
}
