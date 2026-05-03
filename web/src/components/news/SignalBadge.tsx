import type { LLMSignal } from "@/components/shared/api/types";
import { RiskLevelBadge } from "@/components/shared/RiskLevelBadge";

interface SignalBadgeProps {
  signal: LLMSignal;
}

export function SignalBadge({ signal }: SignalBadgeProps) {
  const sentimentLabel = signal.sentiment_score > 0.3 ? "Positive" : signal.sentiment_score < -0.3 ? "Negative" : "Neutral";
  const sentimentColor = signal.sentiment_score > 0.3 ? "var(--risk-low)" : signal.sentiment_score < -0.3 ? "var(--risk-critical)" : "var(--muted-foreground)";

  return (
    <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-4 space-y-3">
      <h3 className="text-sm font-semibold text-[var(--muted-foreground)] uppercase">LLM Risk Signal</h3>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div>
          <div className="text-xs text-[var(--muted-foreground)]">Risk Score</div>
          <RiskLevelBadge level={signal.credit_risk_score >= 70 ? "High" : signal.credit_risk_score >= 50 ? "Medium" : "Low"} score={signal.credit_risk_score} />
        </div>
        <div>
          <div className="text-xs text-[var(--muted-foreground)]">Sentiment</div>
          <span className="text-sm font-semibold" style={{ color: sentimentColor }}>
            {sentimentLabel} ({signal.sentiment_score.toFixed(2)})
          </span>
        </div>
        <div>
          <div className="text-xs text-[var(--muted-foreground)]">Event Type</div>
          <span className="text-sm font-semibold text-[var(--foreground)] capitalize">
            {signal.event_type?.replace(/_/g, " ") || "N/A"}
          </span>
        </div>
        <div>
          <div className="text-xs text-[var(--muted-foreground)]">Risk Horizon</div>
          <span className="text-sm font-semibold text-[var(--foreground)]">{signal.risk_horizon || "N/A"}</span>
        </div>
        <div>
          <div className="text-xs text-[var(--muted-foreground)]">Market Impact</div>
          <span className="text-sm font-semibold text-[var(--foreground)] capitalize">
            {signal.market_impact_type?.replace(/_/g, " ") || "N/A"}
          </span>
        </div>
        <div>
          <div className="text-xs text-[var(--muted-foreground)]">Confidence</div>
          <span className="text-sm font-semibold text-[var(--foreground)]">
            {(signal.confidence * 100).toFixed(1)}%
          </span>
        </div>
        <div>
          <div className="text-xs text-[var(--muted-foreground)]">Model</div>
          <span className="text-sm font-semibold text-[var(--foreground)]">{signal.llm_model || "N/A"}</span>
        </div>
        <div>
          <div className="text-xs text-[var(--muted-foreground)]">Signal ID</div>
          <span className="text-sm font-semibold text-[var(--foreground)]">#{signal.signal_id}</span>
        </div>
      </div>
    </div>
  );
}
