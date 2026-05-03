-- Migration 001: Portfolios and Reports tables
-- Part 5: Backend API Service Layer

-- Portfolio configurations
CREATE TABLE IF NOT EXISTS portfolios (
    portfolio_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    holdings JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Portfolio risk analysis snapshots
CREATE TABLE IF NOT EXISTS portfolio_analyses (
    analysis_id SERIAL PRIMARY KEY,
    portfolio_id INTEGER REFERENCES portfolios(portfolio_id) ON DELETE CASCADE,
    total_risk_score DECIMAL(10, 6),
    risk_level VARCHAR(20),
    holdings_data JSONB NOT NULL DEFAULT '[]',
    top_contributors JSONB NOT NULL DEFAULT '[]',
    diversification_score DECIMAL(10, 6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- AI-generated risk reports
CREATE TABLE IF NOT EXISTS risk_reports (
    report_id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) REFERENCES companies(ticker),
    report_type VARCHAR(50) NOT NULL DEFAULT 'company_risk',
    title VARCHAR(500),
    markdown_content TEXT NOT NULL,
    summary JSONB,
    model_used VARCHAR(100),
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_reports_ticker ON risk_reports(ticker, generated_at DESC);
CREATE INDEX IF NOT EXISTS idx_portfolio_analyses_portfolio ON portfolio_analyses(portfolio_id, created_at DESC);
