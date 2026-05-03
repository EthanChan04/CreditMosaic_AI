"use client";

import { useState } from "react";

interface Props {
  ticker: string;
}

export function GenerateReportButton({ ticker }: Props) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generate = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/report/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker, report_type: "company_risk" }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data?.detail || data?.error?.message || "Report generation failed");
      }
      window.location.reload();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-2">
      <button
        onClick={generate}
        disabled={loading}
        className="block w-full text-center px-3 py-2 text-xs font-medium rounded-lg bg-[var(--chart-1)] text-white hover:opacity-90 disabled:opacity-50"
      >
        {loading ? "Generating..." : "+ Generate New"}
      </button>
      {error && <p className="text-xs text-[var(--risk-critical)]">{error}</p>}
    </div>
  );
}
