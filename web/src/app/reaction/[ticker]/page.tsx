"use client";

import useSWR from "swr";
import { useParams } from "next/navigation";
import { StatCard } from "@/components/shared/StatCard";
import { EChartsWrapper } from "@/components/charts/EChartsWrapper";
import { DashboardSkeleton } from "@/components/shared/Skeleton";
import { EmptyState } from "@/components/shared/EmptyState";
import { useTheme } from "@/hooks/useTheme";
import type { ReactionItem, ReactionAnalysisResponse, LagAnalysisResponse, CompareResponse } from "@/components/shared/api/types";
import { formatDate } from "@/lib/utils";
import { startOfDay, subDays } from "date-fns";

function buildTimelineOption(reactions: ReactionItem[], isDark: boolean) {
  return {
    tooltip: { trigger: "item" as const },
    xAxis: {
      type: "time" as const,
      name: "Date",
      axisLabel: { color: isDark ? "#94a3b8" : "#64748b" },
    },
    yAxis: {
      type: "value" as const,
      name: "Cumulative Abnormal Return",
      axisLabel: { color: isDark ? "#94a3b8" : "#64748b", formatter: (v: number) => `${(v * 100).toFixed(1)}%` },
    },
    series: [{
      type: "scatter" as const,
      data: reactions.map((r) => {
        const car = r.windows?.["0_1"]?.cumulative_abnormal_return || 0;
        return { value: [r.event_date, car], name: `#${r.news_id}` };
      }),
      symbolSize: 10,
      color: "var(--chart-1)",
    }],
    grid: { left: "10%", right: "4%", top: 20, bottom: 30 },
  };
}

function buildLeadLagOption(crossCorrelation: Record<string, unknown> | undefined, isDark: boolean) {
  const lagged = crossCorrelation?.lagged_correlations as Record<string, { lag_correlations: Record<string, number | null> }> | undefined;
  const first = lagged ? Object.values(lagged)[0] : undefined;
  const correlations = first?.lag_correlations;
  if (!correlations) return {};
  const entries = Object.entries(correlations)
    .filter(([, value]) => value !== null)
    .sort((a, b) => Number(a[0]) - Number(b[0])) as [string, number][];
  return {
    tooltip: { trigger: "axis" as const },
    xAxis: { type: "category" as const, data: entries.map(([k]) => `Lag ${k}`), axisLabel: { color: isDark ? "#94a3b8" : "#64748b", fontSize: 9 } },
    yAxis: { type: "value" as const, name: "Correlation", axisLabel: { color: isDark ? "#94a3b8" : "#64748b" } },
    series: [{
      type: "bar" as const,
      data: entries.map(([, v]) => ({
        value: v,
        itemStyle: { color: v >= 0 ? "var(--risk-low)" : "var(--risk-critical)" },
      })),
    }],
    grid: { left: "10%", right: "4%", top: 20, bottom: 30 },
  };
}

function buildConfusionOption(compare: CompareResponse) {
  const flows = compare.confusion_flow || {};
  const fromTypes = ["equity_leading", "credit_leading", "two_market_shock", "low_impact"];
  const toTypes = ["equity_leading", "credit_leading", "two_market_shock", "low_impact"];
  const data = fromTypes.flatMap((f, fi) =>
    toTypes.map((t, ti) => [fi, ti, flows[`${f}->${t}`] || 0])
  );

  return {
    tooltip: {},
    xAxis: { type: "category" as const, data: fromTypes.map((f) => f.replace(/_/g, " ")), axisLabel: { rotate: 30, fontSize: 9 } },
    yAxis: { type: "category" as const, data: toTypes.map((t) => t.replace(/_/g, " ")) },
    visualMap: { min: 0, max: Math.max(...data.map((d) => d[2] as number), 1), calculable: true, orient: "horizontal" as const, left: "center" },
    series: [{
      type: "heatmap" as const,
      data,
      label: { show: true },
    }],
    grid: { left: "15%", right: "4%", top: 40, bottom: 30 },
  };
}

export default function ReactionPage() {
  const params = useParams();
  const ticker = params.ticker as string;
  const { theme } = useTheme();
  const isDark = theme === "dark";

  const endDate = new Date();
  const startDate = subDays(endDate, 180);

  const body = {
    tickers: [ticker],
    start_date: startOfDay(startDate).toISOString().split("T")[0],
    end_date: startOfDay(endDate).toISOString().split("T")[0],
  };

  const { data: analysis, isLoading: aLoad, error: aErr } = useSWR<ReactionAnalysisResponse>(
    `reaction-analyze-${ticker}`,
    async () => {
      const res = await fetch("/api/reaction/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error("Failed");
      return res.json();
    },
    { dedupingInterval: 60000 }
  );

  const { data: lag, isLoading: lLoad } = useSWR<LagAnalysisResponse>(
    `reaction-lag-${ticker}`,
    async () => {
      const res = await fetch("/api/reaction/lag", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error("Failed");
      return res.json();
    },
    { dedupingInterval: 60000 }
  );

  const { data: compare } = useSWR<CompareResponse>(
    `reaction-compare-${ticker}`,
    async () => {
      const res = await fetch(`/api/reaction/compare?tickers=${ticker}&days=90`, {
        method: "POST",
      });
      if (!res.ok) throw new Error("Failed");
      return res.json();
    },
    { dedupingInterval: 60000 }
  );

  if (aLoad || lLoad) return <DashboardSkeleton />;
  if (aErr) return <EmptyState icon="⚠️" title="Failed to load reaction data" message={aErr.message} />;

  const reactions = analysis?.reactions || [];
  const agreement = analysis?.agreement || {};
  const overallAgreement = typeof agreement?.overall_agreement === "number"
    ? `${(agreement.overall_agreement as number * 100).toFixed(1)}%`
    : "N/A";

  return (
    <div className="max-w-[1400px] mx-auto space-y-6">
      <h1 className="text-2xl font-bold text-[var(--foreground)]">
        Cross-Market Reaction: {ticker}
      </h1>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatCard title="Total Events" value={analysis?.total_events || 0} icon="📅" variant="default" />
        <StatCard title="Agreement Rate" value={overallAgreement} subtitle="LLM vs Observed" variant="success" />
        <StatCard title="Lead-Lag Events" value={lag?.event_count || 0} icon="⏱️" variant="default" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-4">
          <h2 className="text-sm font-semibold text-[var(--muted-foreground)] uppercase mb-3">Reaction Timeline</h2>
          <EChartsWrapper option={buildTimelineOption(reactions, isDark)} height={300} empty={reactions.length === 0} emptyMessage="No reactions" theme={theme} />
        </div>
        <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-4">
          <h2 className="text-sm font-semibold text-[var(--muted-foreground)] uppercase mb-3">Lead-Lag Correlation</h2>
          <EChartsWrapper option={buildLeadLagOption(lag?.cross_correlation, isDark)} height={300} empty={!lag} emptyMessage="No lag data" theme={theme} />
        </div>
      </div>

      {compare && compare.total_events > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-4">
            <h2 className="text-sm font-semibold text-[var(--muted-foreground)] uppercase mb-3">
              Confusion Matrix (Predicted → Observed)
            </h2>
            <EChartsWrapper option={buildConfusionOption(compare)} height={300} theme={theme} />
          </div>
          <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-4">
            <h2 className="text-sm font-semibold text-[var(--muted-foreground)] uppercase mb-3">Recent Reactions</h2>
            <div className="max-h-[300px] overflow-y-auto space-y-2">
              {reactions.slice(0, 10).map((r) => (
                <div key={r.news_id} className="flex items-center justify-between py-2 px-3 rounded-md bg-[var(--muted)] text-sm">
                  <span className="font-semibold text-[var(--foreground)]">#{r.news_id}</span>
                  <span className="text-[var(--muted-foreground)]">{formatDate(r.event_date)}</span>
                  <span className="capitalize text-xs">{r.event_type?.replace(/_/g, " ") || "N/A"}</span>
                  <span className="text-xs">
                    <span className="text-[var(--chart-1)]">{r.llm_market_impact?.replace(/_/g, " ") || "N/A"}</span>
                    {" → "}
                    <span className="text-[var(--chart-4)]">{r.observed_impact_type?.replace(/_/g, " ") || "N/A"}</span>
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
