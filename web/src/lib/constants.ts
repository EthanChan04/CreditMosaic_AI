export const API_BASE = "/api";

export const RISK_LEVELS = {
  LOW: { threshold: 0.25, label: "Low", color: "var(--risk-low)" },
  MEDIUM: { threshold: 0.50, label: "Medium", color: "var(--risk-medium)" },
  HIGH: { threshold: 0.75, label: "High", color: "var(--risk-high)" },
  CRITICAL: { threshold: 1.00, label: "Critical", color: "var(--risk-critical)" },
} as const;

export const EVENT_TYPES = [
  "liquidity_pressure",
  "debt_refinancing",
  "earnings_deterioration",
  "litigation",
  "regulatory",
  "rating_change",
  "management_change",
  "supply_chain",
  "fraud_or_accounting",
  "neutral_or_irrelevant",
] as const;

export const MARKET_IMPACT_TYPES = [
  "equity_leading",
  "credit_leading",
  "two_market_shock",
  "low_impact",
] as const;

export const RISK_HORIZONS = ["1w", "1m", "3m", "12m"] as const;

export const REACTION_WINDOWS = ["0_1", "1_3", "3_5", "5_20"] as const;

export const CHART_COLORS = [
  "var(--chart-1)",
  "var(--chart-2)",
  "var(--chart-3)",
  "var(--chart-4)",
  "var(--chart-5)",
  "var(--chart-6)",
  "var(--chart-7)",
  "var(--chart-8)",
] as const;

export const PAGE_SIZE_DEFAULT = 20;

export const EVENT_TYPE_LABELS: Record<string, string> = {
  liquidity_pressure: "Liquidity Pressure",
  debt_refinancing: "Debt Refinancing",
  earnings_deterioration: "Earnings Deterioration",
  litigation: "Litigation",
  regulatory: "Regulatory",
  rating_change: "Rating Change",
  management_change: "Management Change",
  supply_chain: "Supply Chain",
  fraud_or_accounting: "Fraud / Accounting",
  neutral_or_irrelevant: "Neutral",
};
