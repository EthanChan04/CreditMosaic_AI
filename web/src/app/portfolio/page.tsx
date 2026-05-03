"use client";

import { useState } from "react";
import useSWR, { mutate } from "swr";
import { StatCard } from "@/components/shared/StatCard";
import { DataTable, type Column } from "@/components/shared/DataTable";
import { RiskLevelBadge } from "@/components/shared/RiskLevelBadge";
import { EChartsWrapper } from "@/components/charts/EChartsWrapper";
import { useTheme } from "@/hooks/useTheme";
import type { HoldingRiskDetail, PortfolioAnalyzeResponse, PortfolioSummary } from "@/components/shared/api/types";

interface StressScenario {
  scenario: string;
  base_portfolio_risk: number;
  shocked_portfolio_risk: number;
  risk_increase_pct: number;
}

interface CorrelationResult {
  tickers: string[];
  correlation_matrix: number[][];
  portfolio_volatility_annual: number;
  diversification_ratio: number;
  holdings_risk: (HoldingRiskDetail & { volatility_annual: number; risk_contribution_pct: number; marginal_ctr: number })[];
  risk_level: string;
}

interface StressResult {
  base_portfolio_risk: number;
  scenarios: StressScenario[];
}

export default function PortfolioPage() {
  const { theme } = useTheme();
  const isDark = theme === "dark";

  const [tickerInput, setTickerInput] = useState("");
  const [holdings, setHoldings] = useState<{ ticker: string; weight: number }[]>([]);
  const [name, setName] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const [diversificationScore, setDiversificationScore] = useState<string | null>(null);
  const [result, setResult] = useState<PortfolioAnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"analyze" | "saved">("analyze");

  // Part 7: Advanced analytics state
  const [correlationData, setCorrelationData] = useState<CorrelationResult | null>(null);
  const [stressData, setStressData] = useState<StressResult | null>(null);
  const [deepAnalyzing, setDeepAnalyzing] = useState(false);
  const [reportGenerating, setReportGenerating] = useState(false);
  const [reportMarkdown, setReportMarkdown] = useState<string | null>(null);
  const [showDeepAnalysis, setShowDeepAnalysis] = useState(false);

  const { data: savedData } = useSWR("/api/portfolios");
  const savedPortfolios: PortfolioSummary[] = savedData?.portfolios || [];

  const addHolding = () => {
    if (!tickerInput.trim()) return;
    const ticker = tickerInput.trim().toUpperCase();
    if (holdings.some((h) => h.ticker === ticker)) {
      setTickerInput("");
      return;
    }
    const next = [...holdings, { ticker, weight: 1 }];
    const equalWeight = 1 / next.length;
    setHoldings(next.map((h) => ({ ...h, weight: equalWeight })));
    setTickerInput("");
  };

  const updateWeight = (index: number, weight: number) => {
    const next = [...holdings];
    next[index] = { ...next[index], weight };
    setHoldings(next);
  };

  const removeHolding = (index: number) => {
    setHoldings(holdings.filter((_, i) => i !== index));
  };

  const normalizeWeights = () => {
    const total = holdings.reduce((s, h) => s + h.weight, 0);
    if (total === 0) return;
    setHoldings(holdings.map((h) => ({ ...h, weight: h.weight / total })));
  };

  const analyze = async () => {
    setError(null);
    setAnalyzing(true);
    setCorrelationData(null);
    setStressData(null);
    setShowDeepAnalysis(false);
    try {
      const res = await fetch("/api/portfolio/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: name || undefined,
          holdings: holdings.map((h) => ({ ticker: h.ticker, weight: h.weight })),
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err?.error?.message || "Analysis failed");
      }
      const data: PortfolioAnalyzeResponse = await res.json();
      setResult(data);
      setDiversificationScore(data.diversification_score != null ? data.diversification_score.toFixed(3) : null);
      mutate("/api/portfolios");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setAnalyzing(false);
    }
  };

  const runDeepAnalysis = async () => {
    if (holdings.length < 2) return;
    setDeepAnalyzing(true);
    try {
      const [corrRes, stressRes] = await Promise.all([
        fetch("/api/portfolio/correlation", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ holdings: holdings.map((h) => ({ ticker: h.ticker, weight: h.weight })), days: 60 }),
        }).then((r) => r.json()),
        fetch("/api/portfolio/stress-test", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ holdings: holdings.map((h) => ({ ticker: h.ticker, weight: h.weight })) }),
        }).then((r) => r.json()),
      ]);
      if (!corrRes.error) setCorrelationData(corrRes);
      if (!stressRes.error) setStressData(stressRes);
      setShowDeepAnalysis(true);
    } catch {
      // Ignore deep analysis errors
    } finally {
      setDeepAnalyzing(false);
    }
  };

  const generatePortfolioReport = async () => {
    setReportGenerating(true);
    try {
      const res = await fetch("/api/portfolio/report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          holdings: holdings.map((h) => ({ ticker: h.ticker, weight: h.weight })),
          include_correlations: true,
          include_stress: true,
        }),
      });
      if (!res.ok) {
        const payload = await res.json().catch(() => ({}));
        throw new Error(payload?.detail || payload?.error?.message || "Report generation failed");
      }
      const data = await res.json();
      setReportMarkdown(data.markdown_content);
    } catch {
      setError("Failed to generate portfolio report");
    } finally {
      setReportGenerating(false);
    }
  };

  const downloadReport = () => {
    if (!reportMarkdown) return;
    const blob = new Blob([reportMarkdown], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `portfolio_risk_report_${new Date().toISOString().slice(0, 10)}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const deletePortfolio = async (id: number) => {
    await fetch(`/api/portfolio/${id}`, { method: "DELETE" });
    mutate("/api/portfolios");
  };

  const holdingColumns: Column<HoldingRiskDetail>[] = [
    { key: "ticker", header: "Ticker", accessor: (r) => <span className="font-semibold text-[var(--foreground)]">{r.ticker}</span> },
    { key: "name", header: "Company", accessor: (r) => r.company_name || "-" },
    { key: "weight", header: "Weight", accessor: (r) => `${(r.weight * 100).toFixed(1)}%` },
    { key: "risk_score", header: "Risk", accessor: (r) => <RiskLevelBadge level={r.risk_level} score={r.risk_score} /> },
    { key: "contribution", header: "Contribution", accessor: (r) => r.risk_contribution.toFixed(4) },
  ];

  const treemapOption = result ? {
    tooltip: {},
    series: [{
      type: "treemap" as const,
      data: [...result.holdings_risk]
        .sort((a, b) => b.risk_contribution - a.risk_contribution)
        .map((h) => ({
          name: h.ticker,
          value: Math.max(h.risk_contribution * 100, 0.01),
        })),
      label: { show: true, formatter: "{b}" },
      itemStyle: { borderColor: isDark ? "#1e293b" : "#fff" },
    }],
  } : {};

  // Part 7: Correlation heatmap option
  const heatmapOption = correlationData ? {
    tooltip: {},
    xAxis: {
      type: "category" as const,
      data: correlationData.tickers,
      axisLabel: { fontSize: 10, color: isDark ? "#94a3b8" : "#64748b" },
    },
    yAxis: {
      type: "category" as const,
      data: correlationData.tickers,
      axisLabel: { fontSize: 10, color: isDark ? "#94a3b8" : "#64748b" },
    },
    visualMap: {
      min: -1, max: 1,
      inRange: { color: ["var(--risk-low)", "#ffffff", "var(--risk-critical)"] },
      calculable: true,
      orient: "horizontal" as const,
      left: "center",
      bottom: 0,
    },
    series: [{
      type: "heatmap" as const,
      data: correlationData.tickers.flatMap((t, i) =>
        correlationData.tickers.map((_, j) => [i, j, correlationData.correlation_matrix[i]?.[j] ?? 0])
      ),
      label: { show: true, fontSize: 9, formatter: (p: unknown) => {
        const v = (p as { value: [number, number, number] }).value;
        return Array.isArray(v) && v.length > 2 ? v[2].toFixed(2) : "";
      }},
    }],
    grid: { left: "8%", right: "4%", top: 10, bottom: 60 },
  } : {};

  // Part 7: Stress test bar chart
  const stressOption = stressData ? {
    tooltip: { trigger: "axis" as const },
    xAxis: {
      type: "category" as const,
      data: stressData.scenarios.map((s) => s.scenario),
      axisLabel: { rotate: 0, fontSize: 11, color: isDark ? "#e2e8f0" : "#1a1a2e" },
    },
    yAxis: {
      type: "value" as const,
      name: "Risk Increase %",
      axisLabel: { formatter: (v: number) => `+${v}%`, color: isDark ? "#94a3b8" : "#64748b" },
    },
    series: [{
      type: "bar" as const,
      data: stressData.scenarios.map((s) => s.risk_increase_pct),
      color: "var(--risk-critical)",
    }],
    grid: { left: "8%", right: "4%", top: 20, bottom: 30 },
  } : {};

  const savedColumns: Column<PortfolioSummary>[] = [
    { key: "name", header: "Name", accessor: (r) => <span className="font-semibold text-[var(--foreground)]">{r.name}</span> },
    { key: "count", header: "Holdings", accessor: (r) => r.holdings_count },
    { key: "risk", header: "Risk", accessor: (r) => r.risk_level ? <RiskLevelBadge level={r.risk_level} score={r.total_risk_score ?? undefined} /> : "-" },
    { key: "actions", header: "", accessor: (r) => (
      <button
        onClick={(e) => { e.stopPropagation(); deletePortfolio(r.portfolio_id); }}
        className="text-xs text-[var(--risk-critical)] hover:underline"
      >
        Delete
      </button>
    )},
  ];

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold text-[var(--foreground)]">Portfolio Analysis</h1>

      <div className="flex gap-1 border-b border-[var(--border)]">
        {(["analyze", "saved"] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors capitalize ${
              activeTab === tab
                ? "border-[var(--chart-1)] text-[var(--chart-1)]"
                : "border-transparent text-[var(--muted-foreground)] hover:text-[var(--foreground)]"
            }`}
          >
            {tab === "analyze" ? "Analyze" : "Saved Portfolios"}
          </button>
        ))}
      </div>

      {activeTab === "analyze" && (
        <div className="space-y-6">
          {/* Form */}
          <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-4 space-y-3">
            <div className="flex gap-2">
              <input
                type="text"
                value={tickerInput}
                onChange={(e) => setTickerInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && addHolding()}
                placeholder="Add ticker (e.g. AAPL)"
                className="flex-1 px-3 py-2 text-sm rounded-lg border border-[var(--border)] bg-[var(--background)] text-[var(--foreground)] focus:outline-none focus:ring-2 focus:ring-[var(--chart-1)]/50"
              />
              <button onClick={addHolding} className="px-4 py-2 text-sm font-medium rounded-lg bg-[var(--chart-1)] text-white hover:opacity-90">
                Add
              </button>
            </div>

            {holdings.length > 0 && (
              <div className="space-y-2">
                {holdings.map((h, i) => (
                  <div key={i} className="flex items-center gap-3">
                    <span className="font-semibold text-sm w-16 text-[var(--foreground)]">{h.ticker}</span>
                    <input
                      type="range"
                      min={0}
                      max={100}
                      value={h.weight * 100}
                      onChange={(e) => updateWeight(i, Number(e.target.value) / 100)}
                      className="flex-1"
                    />
                    <span className="text-sm w-14 text-right text-[var(--foreground)]">{(h.weight * 100).toFixed(0)}%</span>
                    <button onClick={() => removeHolding(i)} className="text-[var(--risk-critical)] text-sm">Remove</button>
                  </div>
                ))}
                <div className="flex gap-2">
                  <button onClick={normalizeWeights} className="text-xs text-[var(--chart-1)] hover:underline">
                    Normalize to 100%
                  </button>
                  <span className="text-xs text-[var(--muted-foreground)]">
                    Total: {(holdings.reduce((s, h) => s + h.weight, 0) * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            )}

            <div className="flex gap-2 items-end">
              <div className="flex-1">
                <label className="text-xs text-[var(--muted-foreground)]">Portfolio Name (optional)</label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="My Portfolio"
                  className="w-full px-3 py-2 text-sm rounded-lg border border-[var(--border)] bg-[var(--background)] text-[var(--foreground)] focus:outline-none focus:ring-2 focus:ring-[var(--chart-1)]/50"
                />
              </div>
              <button
                onClick={analyze}
                disabled={holdings.length === 0 || analyzing}
                className="px-6 py-2 text-sm font-medium rounded-lg bg-[var(--chart-1)] text-white hover:opacity-90 disabled:opacity-50 transition-opacity"
              >
                {analyzing ? "Analyzing..." : "Analyze"}
              </button>
            </div>
          </div>

          {/* Error */}
          {error && <div className="bg-[var(--risk-critical)]/10 text-[var(--risk-critical)] text-sm p-3 rounded-lg">{error}</div>}

          {/* Results */}
          {result && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                <StatCard title="Portfolio Risk" value={result.total_risk_score.toFixed(3)} variant={result.risk_level === "High" || result.risk_level === "Critical" ? "danger" : "default"} />
                <StatCard title="Risk Level" value={result.risk_level} variant={result.risk_level === "High" || result.risk_level === "Critical" ? "danger" : result.risk_level === "Medium" ? "warning" : "success"} />
                <StatCard title="Holdings" value={result.holdings_risk.length} variant="default" />
                {diversificationScore && <StatCard title="Diversification" value={diversificationScore} variant="default" />}
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-4">
                  <h2 className="text-sm font-semibold text-[var(--muted-foreground)] uppercase mb-3">Holdings Risk</h2>
                  <DataTable
                    columns={holdingColumns}
                    data={[...result.holdings_risk].sort((a, b) => b.risk_contribution - a.risk_contribution)}
                    empty={result.holdings_risk.length === 0}
                    emptyMessage="No holdings"
                  />
                </div>
                <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-4">
                  <h2 className="text-sm font-semibold text-[var(--muted-foreground)] uppercase mb-3">Risk Breakdown</h2>
                  <EChartsWrapper option={treemapOption} height={300} empty={result.holdings_risk.length === 0} emptyMessage="No holdings" theme={theme} />
                </div>
              </div>

              {result.recommendation && (
                <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-4">
                  <h2 className="text-sm font-semibold text-[var(--muted-foreground)] uppercase mb-2">Recommendation</h2>
                  <p className="text-sm text-[var(--foreground)]">{result.recommendation}</p>
                </div>
              )}

              {/* Part 7: Deep Analysis & Report buttons */}
              <div className="flex flex-wrap gap-3">
                {holdings.length >= 2 && (
                  <button
                    onClick={runDeepAnalysis}
                    disabled={deepAnalyzing}
                    className="px-4 py-2 text-sm font-medium rounded-lg border border-[var(--border)] hover:bg-[var(--muted)] transition-colors"
                  >
                    {deepAnalyzing ? "Running..." : "Deep Analysis (Correlations + Stress)"}
                  </button>
                )}
                <button
                  onClick={generatePortfolioReport}
                  disabled={reportGenerating}
                  className="px-4 py-2 text-sm font-medium rounded-lg bg-[var(--chart-1)] text-white hover:opacity-90 disabled:opacity-50"
                >
                  {reportGenerating ? "Generating..." : "Generate Portfolio Report"}
                </button>
              </div>

              {/* Part 7: Deep Analysis results */}
              {showDeepAnalysis && (correlationData || stressData) && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  {correlationData && (
                    <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-4">
                      <h2 className="text-sm font-semibold text-[var(--muted-foreground)] uppercase mb-3">Return Correlation Matrix</h2>
                      <EChartsWrapper option={heatmapOption} height={320} empty={false} theme={theme} />
                      <div className="text-xs text-[var(--muted-foreground)] mt-2 space-y-1">
                        <div>Annual Vol: {(correlationData.portfolio_volatility_annual * 100).toFixed(2)}%</div>
                        <div>Diversification Ratio: {correlationData.diversification_ratio.toFixed(3)}</div>
                      </div>
                    </div>
                  )}
                  {stressData && (
                    <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-4">
                      <h2 className="text-sm font-semibold text-[var(--muted-foreground)] uppercase mb-3">Stress Test Scenarios</h2>
                      <EChartsWrapper option={stressOption} height={320} empty={false} theme={theme} />
                      <div className="text-xs text-[var(--muted-foreground)] mt-2">
                        Base risk: {stressData.base_portfolio_risk.toFixed(4)}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Part 7: Portfolio report markdown viewer */}
              {reportMarkdown && (
                <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <h2 className="text-sm font-semibold text-[var(--muted-foreground)] uppercase">Generated Portfolio Report</h2>
                    <button
                      onClick={downloadReport}
                      className="px-3 py-1.5 text-xs font-medium rounded-lg bg-[var(--chart-1)]/10 text-[var(--chart-1)] hover:bg-[var(--chart-1)]/20"
                    >
                      Download .md
                    </button>
                  </div>
                  <pre className="text-xs text-[var(--foreground)] whitespace-pre-wrap max-h-96 overflow-y-auto bg-[var(--background)] rounded-lg p-3 border border-[var(--border)]">
                    {reportMarkdown}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {activeTab === "saved" && (
        <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-4">
          <h2 className="text-sm font-semibold text-[var(--muted-foreground)] uppercase mb-3">Saved Portfolios</h2>
          <DataTable columns={savedColumns} data={savedPortfolios} empty={savedPortfolios.length === 0} emptyMessage="No saved portfolios yet" />
        </div>
      )}
    </div>
  );
}
