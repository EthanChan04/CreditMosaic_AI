"use client";

import { cn } from "@/lib/utils";

interface SkeletonProps {
  className?: string;
  variant?: "text" | "card" | "chart" | "table-row" | "circle";
}

export function Skeleton({ className, variant = "text" }: SkeletonProps) {
  const base = "skeleton rounded";

  const variants: Record<string, string> = {
    text: "h-4 w-full",
    card: "h-32 w-full rounded-lg",
    chart: "h-[300px] w-full rounded-lg",
    "table-row": "h-10 w-full rounded",
    circle: "h-10 w-10 rounded-full",
  };

  return <div className={cn(base, variants[variant], className)} />;
}

export function ChartSkeleton() {
  return <Skeleton variant="chart" />;
}

export function TableSkeleton({ rows = 5 }: { rows?: number }) {
  return (
    <div className="space-y-2">
      <Skeleton variant="table-row" />
      {Array.from({ length: rows }).map((_, i) => (
        <Skeleton key={i} variant="table-row" className="opacity-70" />
      ))}
    </div>
  );
}

export function CardSkeleton() {
  return <Skeleton variant="card" />;
}

export function DashboardSkeleton() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <CardSkeleton key={i} />
        ))}
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ChartSkeleton />
        <TableSkeleton rows={5} />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ChartSkeleton />
        <ChartSkeleton />
      </div>
    </div>
  );
}
