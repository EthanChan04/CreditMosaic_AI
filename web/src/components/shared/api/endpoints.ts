import { api } from "./client";
import type {
  PaginatedResponse,
  CompanyResponse,
  CompanyDetailResponse,
  CompanyNewsResponse,
  NewsDetailResponse,
  SignalResponse,
  RiskScoreResponse,
  RiskHistoryResponse,
  ReactionAnalysisResponse,
  LagAnalysisResponse,
  CompareResponse,
  SectorItem,
  PortfolioAnalyzeResponse,
  PortfolioSummary,
  PortfolioDetailResponse,
  ReportResponse,
  ReportSummaryItem,
  HealthResponse,
  SuccessMessage,
} from "./types";

// Companies
export function getCompanies(params?: Record<string, string>) {
  return api.get<PaginatedResponse<CompanyResponse>>("/api/companies", params);
}
export function getCompanySectors() {
  return api.get<{ sectors: SectorItem[] }>("/api/companies/sectors");
}
export function searchCompanies(params: Record<string, string>) {
  return api.get<CompanyResponse[]>("/api/companies/search", params);
}
export function getCompany(ticker: string) {
  return api.get<CompanyDetailResponse>(`/api/companies/${ticker}`);
}
export function getCompanyNews(ticker: string, limit = 50) {
  return api.get<CompanyNewsResponse>(`/api/companies/${ticker}/news`, {
    limit: String(limit),
  });
}
export function getCompanyRiskHistory(ticker: string, days = 90) {
  return api.get<RiskHistoryResponse>(`/api/risk/scores/${ticker}`, {
    days: String(days),
  });
}

// News
export function getNewsList(params?: Record<string, string>) {
  return api.get("/api/news", params);
}
export function getNewsDetail(newsId: number) {
  return api.get<NewsDetailResponse>(`/api/news/${newsId}`);
}
export function getSignals(params?: Record<string, string>) {
  return api.get<SignalResponse[]>("/api/signals", params);
}
export function compareFinBERT(ticker: string, newsId: number) {
  return api.get("/api/compare-finbert", {
    ticker,
    news_id: String(newsId),
  });
}

// Risk
export function getRiskScores(tickers: string) {
  return api.get<RiskScoreResponse[]>("/api/risk/scores", { tickers });
}
export function getRiskScoreHistory(ticker: string, days = 90) {
  return api.get<RiskHistoryResponse>(`/api/risk/scores/${ticker}`, {
    days: String(days),
  });
}
export function getModelEvaluation() {
  return api.get<Record<string, unknown>>("/api/risk/models/evaluation");
}

// Reactions
export function analyzeReactions(body: Record<string, unknown>) {
  return api.post<ReactionAnalysisResponse>("/api/reaction/analyze", body);
}
export function analyzeLag(body: Record<string, unknown>) {
  return api.post<LagAnalysisResponse>("/api/reaction/lag", body);
}
export function getReactionByNews(newsId: number) {
  return api.get<Record<string, unknown>>(`/api/reaction/news/${newsId}`);
}
export function getReactionByTicker(ticker: string, days = 90) {
  return api.get<{ ticker: string; reactions: Record<string, unknown>[] }>(
    `/api/reaction/ticker/${ticker}`,
    { days: String(days) }
  );
}
export function compareLLMvsActual(tickers: string, days = 90) {
  return api.post<CompareResponse>(
    `/api/reaction/compare?tickers=${tickers}&days=${days}`
  );
}
export function getReactionSummary(tickers: string, days = 90) {
  return api.get<Record<string, unknown>>("/api/reaction/summary", {
    tickers,
    days: String(days),
  });
}

// Portfolios
export function analyzePortfolio(body: Record<string, unknown>) {
  return api.post<PortfolioAnalyzeResponse>("/api/portfolio/analyze", body);
}
export function getPortfolios() {
  return api.get<{ total: number; portfolios: PortfolioSummary[] }>(
    "/api/portfolios"
  );
}
export function getPortfolio(portfolioId: number) {
  return api.get<PortfolioDetailResponse>(`/api/portfolio/${portfolioId}`);
}
export function deletePortfolio(portfolioId: number) {
  return api.delete<SuccessMessage>(`/api/portfolio/${portfolioId}`);
}

// Reports
export function generateReport(body: Record<string, unknown>) {
  return api.post<ReportResponse>("/api/report/generate", body);
}
export function getReport(reportId: number) {
  return api.get<ReportResponse>(`/api/report/${reportId}`);
}
export function getReports(params?: Record<string, string>) {
  return api.get<{ total: number; reports: ReportSummaryItem[] }>(
    "/api/reports",
    params
  );
}
export function getCompanyLatestReport(ticker: string) {
  return api.get<ReportResponse>(`/api/report/company/${ticker}`);
}
export function deleteReport(reportId: number) {
  return api.delete<SuccessMessage>(`/api/report/${reportId}`);
}
export function getReportDownloadUrl(reportId: number) {
  return `/api/report/${reportId}/download`;
}

// System
export function getHealth() {
  return api.get<HealthResponse>("/health");
}
