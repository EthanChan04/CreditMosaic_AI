"use client";

import useSWR from "swr";
import { StatCard } from "@/components/shared/StatCard";
import { RiskLevelBadge } from "@/components/shared/RiskLevelBadge";
import { DataTable, type Column } from "@/components/shared/DataTable";
import { DashboardSkeleton } from "@/components/shared/Skeleton";
import { EmptyState } from "@/components/shared/EmptyState";
import { EChartsWrapper } from "@/components/charts/EChartsWrapper";
import { useTheme } from "@/hooks/useTheme";
import type { SignalResponse } from "@/components/shared/api/types";
import { formatDate } from "@/lib/utils";
import Link from "next/link";

function buildEventDistributionOption(signals: SignalResponse[], isDark: boolean) {
  const counts: Record<string, number> = {};
  signals.forEach((s) => { counts[s.event_type] = (counts[s.event_type] || 0) + 1; });
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  return {
    tooltip: { trigger: "axis" as const },
    xAxis: {
      type: "category" as const,
      data: entries.map(([k]) => k.replace(/_/g, " ")),
      axisLabel: { rotate: 45, fontSize: 10, color: isDark ? "#94a3b8" : "#64748b" },
    },
    yAxis: { type: "value" as const, name: "Count" },
    series: [{ type: "bar" as const, data: entries.map(([, v]) => v), color: "var(--chart-1)" }],
    grid: { left: "10%", right: "4%", top: 10, bottom: 60 },
  };
}

export default function DashboardPage() {
  const { theme } = useTheme();
  const isDark = theme === "dark";

  const { data: companiesData, error: coError, isLoading: coLoading } = useSWR(
    "/api/companies?page=1&page_size=200"
  );
  const { data: sectorsData } = useSWR("/api/companies/sectors");
  const { data: signalsData, error: sigError, isLoading: sigLoading } = useSWR(
    "/api/signals?limit=100"
  );

  const companies = companiesData?.items || [];
  const signals: SignalResponse[] = signalsData || [];
  const totalCo = companies.length;
  const highRisk = signals.filter((s) => s.credit_risk_score >= 70).length;
  const avgRisk = signals.length
    ? signals.reduce((a, s) => a + s.credit_risk_score, 0) / signals.length / 100
    : 0;

  const tickerRisk: Record<string, { ticker: string; maxRisk: number; count: number }> = {};
  signals.forEach((s) => {
    if (!tickerRisk[s.ticker]) tickerRisk[s.ticker] = { ticker: s.ticker, maxRisk: s.credit_risk_score, count: 1 };
    else {
      tickerRisk[s.ticker].maxRisk = Math.max(tickerRisk[s.ticker].maxRisk, s.credit_risk_score);
      tickerRisk[s.ticker].count++;
    }
  });
  const topRisky = Object.values(tickerRisk).sort((a, b) => b.maxRisk - a.maxRisk).slice(0, 10);

  const riskColumns: Column<(typeof topRisky)[0]>[] = [
    { key: "ticker", header: "Ticker", accessor: (r) => (
      <Link href={`/company/${r.ticker}`} className="text-[var(--chart-1)] font-semibold hover:underline">{r.ticker}</Link>
    )},
    { key: "risk", header: "Max Risk", accessor: (r) => (
      <RiskLevelBadge level={r.maxRisk >= 70 ? "High" : r.maxRisk >= 50 ? "Medium" : "Low"} score={r.maxRisk} />
    )},
    { key: "count", header: "Signals", accessor: (r) => r.count },
  ];

  if (coLoading || sigLoading) return <DashboardSkeleton />;
  if (coError && sigError) return <EmptyState icon="⚠️" title="Failed to load dashboard" message={coError?.message || "Check backend connection"} />;

  return (
    <div className="space-y-6 max-w-[1400px] mx-auto">
      <h1 className="text-2xl font-bold text-[var(--foreground)]">Dashboard</h1>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard title="Tracked Companies" value={totalCo} icon="🏢" variant="default" />
        <StatCard title="High Risk Alerts" value={highRisk} subtitle={`${signals.length} total signals`} variant="danger" />
        <StatCard title="Avg Risk Score" value={avgRisk.toFixed(3)} subtitle="0-1 scale" variant="warning" />
        <StatCard title="Sectors" value={sectorsData?.sectors?.length || 0} icon="📊" variant="default" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-4">
          <h2 className="text-sm font-semibold text-[var(--muted-foreground)] uppercase mb-3">Top Risk Companies</h2>
          <DataTable columns={riskColumns} data={topRisky} empty={topRisky.length === 0} emptyMessage="No risk signals found" />
        </div>
        <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-4">
          <h2 className="text-sm font-semibold text-[var(--muted-foreground)] uppercase mb-3">Recent Signals</h2>
          <div className="max-h-[300px] overflow-y-auto space-y-2">
            {signals.slice(0, 10).map((s) => (
              <Link
                key={s.signal_id}
                href={`/news/${s.news_id}`}
                className="flex items-center justify-between py-2 px-3 rounded-md hover:bg-[var(--muted)] transition-colors"
              >
                <div className="flex items-center gap-3">
                  <span className="font-semibold text-sm text-[var(--foreground)]">{s.ticker}</span>
                  <span className="text-xs text-[var(--muted-foreground)]">{s.event_type?.replace(/_/g, " ") || "N/A"}</span>
                </div>
                <div className="flex items-center gap-2">
                  <RiskLevelBadge level={s.credit_risk_score >= 70 ? "High" : s.credit_risk_score >= 50 ? "Medium" : "Low"} score={s.credit_risk_score} />
                  <span className="text-xs text-[var(--muted-foreground)]">{formatDate(s.extracted_at)}</span>
                </div>
              </Link>
            ))}
            {signals.length === 0 && <EmptyState icon="📰" title="No recent signals" />}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-4">
          <h2 className="text-sm font-semibold text-[var(--muted-foreground)] uppercase mb-3">Event Distribution</h2>
          <EChartsWrapper
            option={buildEventDistributionOption(signals, isDark)}
            height={350}
            empty={signals.length === 0}
            emptyMessage="No signals data available"
          />
        </div>
        <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-4">
          <h2 className="text-sm font-semibold text-[var(--muted-foreground)] uppercase mb-3">Quick Actions</h2>
          <div className="grid grid-cols-2 gap-3">
            <Link href="/portfolio" className="p-4 rounded-lg border border-[var(--border)] hover:bg-[var(--muted)] transition-colors text-center">
              <div className="text-2xl mb-1">⊡</div>
              <div className="text-sm font-medium text-[var(--foreground)]">Portfolio Analysis</div>
            </Link>
            <Link href="/company/AAPL" className="p-4 rounded-lg border border-[var(--border)] hover:bg-[var(--muted)] transition-colors text-center">
              <div className="text-2xl mb-1">📈</div>
              <div className="text-sm font-medium text-[var(--foreground)]">Company Lookup</div>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
