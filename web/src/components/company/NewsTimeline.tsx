"use client";

import useSWR from "swr";
import Link from "next/link";
import { RiskLevelBadge } from "@/components/shared/RiskLevelBadge";
import { EmptyState } from "@/components/shared/EmptyState";
import { TableSkeleton } from "@/components/shared/Skeleton";
import { formatDate, truncate } from "@/lib/utils";
import type { NewsItemWithSignal } from "@/components/shared/api/types";

interface Props {
  ticker: string;
}

export function NewsTimeline({ ticker }: Props) {
  const { data, error, isLoading } = useSWR(
    `/api/companies/${ticker}/news?limit=50`,
    { dedupingInterval: 30000 }
  );

  const news: NewsItemWithSignal[] = data?.news || [];

  if (isLoading) {
    return <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-4"><TableSkeleton rows={5} /></div>;
  }

  return (
    <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-4">
      <h2 className="text-sm font-semibold text-[var(--muted-foreground)] uppercase mb-3">Recent News</h2>
      {news.length === 0 ? (
        <EmptyState icon="📰" title="No news articles" message="No recent news for this company" />
      ) : (
        <div className="space-y-3 max-h-[500px] overflow-y-auto">
          {news.map((n) => (
            <Link
              key={n.news_id}
              href={`/news/${n.news_id}`}
              className="block p-3 rounded-lg border border-[var(--border)] hover:bg-[var(--muted)] transition-colors"
            >
              <div className="flex items-start justify-between gap-2 mb-1">
                <h3 className="text-sm font-medium text-[var(--foreground)] line-clamp-2">
                  {truncate(n.title, 100)}
                </h3>
                {n.credit_risk_score != null && (
                  <RiskLevelBadge
                    level={n.credit_risk_score >= 70 ? "High" : n.credit_risk_score >= 50 ? "Medium" : "Low"}
                    score={n.credit_risk_score}
                  />
                )}
              </div>
              <div className="flex items-center gap-3 text-xs text-[var(--muted-foreground)]">
                <span>{formatDate(n.published_at)}</span>
                {n.source && <span>{n.source}</span>}
                {n.event_type && <span className="capitalize">{n.event_type.replace(/_/g, " ")}</span>}
                {n.market_impact_type && (
                  <span className="capitalize text-[var(--chart-1)]">{n.market_impact_type.replace(/_/g, " ")}</span>
                )}
              </div>
            </Link>
          ))}
        </div>
      )}
      {error && <p className="text-xs text-[var(--risk-critical)] text-center mt-2">Failed to load news</p>}
    </div>
  );
}
