"use client";

import useSWR from "swr";
import { EmptyState } from "@/components/shared/EmptyState";
import { LoadingSpinner } from "@/components/shared/LoadingSpinner";

interface Props {
  ticker: string;
  newsId: number;
}

export function FinBERTComparison({ ticker, newsId }: Props) {
  const { data, error, isLoading } = useSWR(
    `/api/compare-finbert?ticker=${ticker}&news_id=${newsId}`
  );

  if (isLoading) return <LoadingSpinner />;
  if (error) return null;

  const comparison = data?.comparison;

  return (
    <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-4">
      <h3 className="text-sm font-semibold text-[var(--muted-foreground)] uppercase mb-3">FinBERT Baseline Comparison</h3>
      {comparison ? (
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <div className="text-xs text-[var(--muted-foreground)] mb-1">LLM Sentiment</div>
            <span className="font-semibold text-[var(--foreground)]">{typeof comparison.llm_sentiment === "number" ? (comparison.llm_sentiment as number).toFixed(3) : String(comparison.llm_sentiment || "N/A")}</span>
          </div>
          <div>
            <div className="text-xs text-[var(--muted-foreground)] mb-1">FinBERT Sentiment</div>
            <span className="font-semibold text-[var(--foreground)]">{typeof comparison.finbert_sentiment === "number" ? (comparison.finbert_sentiment as number).toFixed(3) : String(comparison.finbert_sentiment || "N/A")}</span>
          </div>
          <div className="col-span-2">
            <div className="text-xs text-[var(--muted-foreground)] mb-1">Agreement</div>
            <span className="font-semibold text-[var(--chart-1)]">{comparison.agreement || "N/A"}</span>
          </div>
        </div>
      ) : (
        <EmptyState icon="🤖" title="No comparison available" />
      )}
    </div>
  );
}
