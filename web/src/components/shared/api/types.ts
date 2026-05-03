export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface CompanyResponse {
  ticker: string;
  company_name: string;
  sector: string | null;
  industry: string | null;
  exchange: string | null;
  market_cap: number | null;
  country: string | null;
  founded_year: number | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface CompanyDetailResponse extends CompanyResponse {
  latest_risk_score: number | null;
  risk_level: string | null;
  news_count_30d: number;
  high_risk_news_count_30d: number;
  latest_price: number | null;
  price_change_5d: number | null;
}

export interface CompanyNewsResponse {
  ticker: string;
  total: number;
  news: NewsItemWithSignal[];
}

export interface NewsItemWithSignal {
  news_id: number;
  ticker: string;
  title: string;
  body: string;
  source: string;
  url: string;
  published_at: string;
  is_processed: boolean;
  sentiment_score: number | null;
  credit_risk_score: number | null;
  event_type: string | null;
  market_impact_type: string | null;
  confidence: number | null;
}

export interface NewsItemResponse {
  news_id: number;
  ticker: string;
  title: string;
  body: string;
  source: string;
  url: string;
  published_at: string;
  is_processed: boolean;
}

export interface NewsDetailResponse extends NewsItemResponse {
  signal: LLMSignal | null;
  reaction: Record<string, unknown> | null;
}

export interface LLMSignal {
  signal_id: number;
  sentiment_score: number;
  credit_risk_score: number;
  event_type: string;
  risk_horizon: string;
  market_impact_type: string;
  evidence_spans: string[];
  confidence: number;
  llm_model: string;
}

export interface SignalResponse {
  signal_id: number;
  news_id: number;
  ticker: string;
  sentiment_score: number;
  credit_risk_score: number;
  event_type: string;
  risk_horizon: string;
  market_impact_type: string;
  evidence_spans: string[];
  confidence: number;
  extracted_at: string;
  llm_model: string;
}

export interface NewsExtractResponse {
  news_id: number;
  signal: Record<string, unknown> | null;
  error: string | null;
  processing_time: number;
}

export interface RiskScoreResponse {
  ticker: string;
  date: string;
  risk_score: number;
  risk_level: string;
  model_version: string;
  top_features: { feature: string; importance: number }[] | null;
}

export interface RiskHistoryResponse {
  ticker: string;
  history: RiskScoreResponse[];
}

export interface ReactionAnalysisResponse {
  total_events: number;
  agreement: Record<string, unknown>;
  reactions: ReactionItem[];
}

export interface ReactionItem {
  news_id: number;
  ticker: string;
  event_date: string;
  event_type: string;
  llm_market_impact: string;
  observed_impact_type: string;
  windows: Record<string, Record<string, number>>;
}

export interface LagAnalysisResponse {
  lead_lag_analysis: Record<string, unknown>;
  cross_correlation: Record<string, unknown>;
  event_count: number;
}

export interface CompareResponse {
  total_events: number;
  agreement: Record<string, unknown>;
  confusion_flow: Record<string, number>;
}

export interface SectorItem {
  sector: string;
  company_count: number;
  avg_market_cap: number;
}

export interface HoldingItem {
  ticker: string;
  weight: number;
}

export interface HoldingRiskDetail {
  ticker: string;
  company_name: string | null;
  weight: number;
  risk_score: number;
  risk_level: string;
  risk_contribution: number;
  top_drivers: { feature: string; importance: number }[] | null;
}

export interface PortfolioAnalyzeResponse {
  portfolio_id: number | null;
  name: string | null;
  total_risk_score: number;
  risk_level: string;
  holdings_risk: HoldingRiskDetail[];
  top_contributors: Record<string, unknown>[];
  diversification_score: number | null;
  recommendation: string | null;
}

export interface PortfolioSummary {
  portfolio_id: number;
  name: string;
  description: string | null;
  holdings_count: number;
  total_risk_score: number | null;
  risk_level: string | null;
  created_at: string | null;
}

export interface PortfolioDetailResponse {
  portfolio_id: number;
  name: string;
  description: string | null;
  holdings: Record<string, unknown>[];
  latest_analysis: Record<string, unknown> | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface ReportResponse {
  report_id: number;
  ticker: string;
  report_type: string;
  title: string;
  markdown_content: string;
  summary: Record<string, unknown> | null;
  model_used: string | null;
  generated_at: string;
}

export interface ReportSummaryItem {
  report_id: number;
  ticker: string;
  title: string;
  report_type: string;
  generated_at: string;
}

export interface HealthResponse {
  status: string;
  version?: string;
  db_connected: boolean;
  llm_providers: number;
  finbert_initialized: boolean;
}

export interface ErrorDetail {
  code: string;
  message: string;
  detail?: string;
}

export interface ErrorResponse {
  error: ErrorDetail;
}

export interface SuccessMessage {
  status: string;
  message: string;
}
