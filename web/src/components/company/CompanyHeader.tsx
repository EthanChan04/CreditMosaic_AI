import type { CompanyDetailResponse } from "@/components/shared/api/types";
import { RiskLevelBadge } from "@/components/shared/RiskLevelBadge";
import { formatMarketCap, formatPercentChange } from "@/lib/formatters";

interface CompanyHeaderProps {
  company: CompanyDetailResponse;
}

export function CompanyHeader({ company }: CompanyHeaderProps) {
  const priceChangeIsPositive = (company.price_change_5d ?? 0) >= 0;

  return (
    <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-4 lg:p-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-[var(--foreground)]">
            {company.company_name}
            <span className="text-[var(--muted-foreground)] text-lg ml-2 font-normal">
              {company.ticker}
            </span>
          </h1>
          <div className="flex flex-wrap gap-x-4 gap-y-1 mt-1 text-sm text-[var(--muted-foreground)]">
            {company.sector && <span>{company.sector}</span>}
            {company.industry && <span>· {company.industry}</span>}
            {company.exchange && <span>· {company.exchange}</span>}
            {company.country && <span>· {company.country}</span>}
          </div>
        </div>
        {company.latest_risk_score != null && (
          <RiskLevelBadge level={company.risk_level || "Unknown"} score={company.latest_risk_score} className="text-sm px-4 py-1.5 self-start" />
        )}
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-4 pt-4 border-t border-[var(--border)]">
        <div>
          <div className="text-xs text-[var(--muted-foreground)] uppercase">Market Cap</div>
          <div className="text-sm font-semibold text-[var(--foreground)]">{formatMarketCap(company.market_cap)}</div>
        </div>
        <div>
          <div className="text-xs text-[var(--muted-foreground)] uppercase">Latest Price</div>
          <div className="text-sm font-semibold text-[var(--foreground)]">
            {company.latest_price != null ? `$${company.latest_price.toFixed(2)}` : "N/A"}
          </div>
        </div>
        <div>
          <div className="text-xs text-[var(--muted-foreground)] uppercase">5-Day Change</div>
          <div className={`text-sm font-semibold ${priceChangeIsPositive ? "text-[var(--risk-low)]" : "text-[var(--risk-critical)]"}`}>
            {company.price_change_5d != null ? formatPercentChange(company.price_change_5d) : "N/A"}
          </div>
        </div>
        <div>
          <div className="text-xs text-[var(--muted-foreground)] uppercase">News (30d)</div>
          <div className="text-sm font-semibold text-[var(--foreground)]">
            {company.news_count_30d}
            {company.high_risk_news_count_30d > 0 && (
              <span className="text-[var(--risk-high)] ml-1">({company.high_risk_news_count_30d} high risk)</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
