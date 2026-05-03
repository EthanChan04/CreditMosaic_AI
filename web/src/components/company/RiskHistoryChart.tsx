"use client";

import useSWR from "swr";
import { EChartsWrapper } from "@/components/charts/EChartsWrapper";
import { useTheme } from "@/hooks/useTheme";
import type { RiskScoreResponse } from "@/components/shared/api/types";

interface Props {
  ticker: string;
}

export function RiskHistoryChart({ ticker }: Props) {
  const { theme } = useTheme();
  const isDark = theme === "dark";

  const { data, error, isLoading } = useSWR(
    `/api/risk/scores/${ticker}?days=90`,
    { dedupingInterval: 30000 }
  );

  const history: RiskScoreResponse[] = data?.history || [];

  const option = {
    tooltip: {
      trigger: "axis" as const,
      formatter: (params: unknown) => {
        const arr = Array.isArray(params) ? params : [params];
        const item = arr[0] as { value: [string, number] } | undefined;
        if (!item?.value) return "";
        return `Risk: ${(item.value[1] * 100).toFixed(1)}%`;
      },
    },
    xAxis: {
      type: "category" as const,
      data: history.map((h) => h.date),
      axisLabel: { color: isDark ? "#94a3b8" : "#64748b", fontSize: 10 },
    },
    yAxis: {
      type: "value" as const,
      min: 0,
      max: 1,
      axisLabel: { formatter: (v: number) => `${(v * 100).toFixed(0)}%`, color: isDark ? "#94a3b8" : "#64748b" },
    },
    series: [{
      type: "line" as const,
      data: history.map((h) => h.risk_score),
      smooth: true,
      color: "var(--chart-1)",
      areaStyle: { color: "var(--chart-1)", opacity: 0.1 },
      markLine: {
        silent: true,
        lineStyle: { color: "var(--risk-high)", type: "dashed" as const },
        data: [
          { yAxis: 0.25, label: { formatter: "Medium" } },
          { yAxis: 0.50, label: { formatter: "High" } },
          { yAxis: 0.75, label: { formatter: "Critical" } },
        ],
      },
    }],
    grid: { left: "8%", right: "4%", top: 20, bottom: 30 },
  };

  return (
    <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-4">
      <h2 className="text-sm font-semibold text-[var(--muted-foreground)] uppercase mb-3">Risk History (90 days)</h2>
      <EChartsWrapper
        option={option}
        height={300}
        loading={isLoading}
        empty={!isLoading && history.length === 0}
        emptyMessage="No risk history available"
        theme={theme}
      />
      {error && (
        <p className="text-xs text-[var(--risk-critical)] text-center mt-2">Failed to load risk history</p>
      )}
    </div>
  );
}
