-- CreditMosaic AI PostgreSQL schema.
-- This file is mounted by docker-compose and can also be run manually with psql.

CREATE TABLE IF NOT EXISTS companies (
    ticker VARCHAR(20) PRIMARY KEY,
    company_name VARCHAR(255) NOT NULL,
    sector VARCHAR(100),
    industry VARCHAR(100),
    exchange VARCHAR(50),
    market_cap DECIMAL(20, 2),
    country VARCHAR(100),
    founded_year INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS news_items (
    news_id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) REFERENCES companies(ticker),
    title TEXT NOT NULL,
    body TEXT,
    source VARCHAR(255),
    url TEXT,
    published_at TIMESTAMP NOT NULL,
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_processed BOOLEAN DEFAULT FALSE,
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS llm_news_signals (
    signal_id SERIAL PRIMARY KEY,
    news_id INTEGER REFERENCES news_items(news_id),
    ticker VARCHAR(20) REFERENCES companies(ticker),
    sentiment_score DECIMAL(5, 4),
    credit_risk_score INTEGER CHECK (credit_risk_score >= 0 AND credit_risk_score <= 100),
    event_type VARCHAR(50),
    risk_horizon VARCHAR(10),
    market_impact_type VARCHAR(50),
    evidence_spans TEXT[],
    confidence DECIMAL(5, 4),
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    llm_model VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS daily_market_data (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) REFERENCES companies(ticker),
    date DATE NOT NULL,
    open_price DECIMAL(15, 4),
    high_price DECIMAL(15, 4),
    low_price DECIMAL(15, 4),
    close_price DECIMAL(15, 4),
    volume BIGINT,
    adjusted_close DECIMAL(15, 4),
    volatility_5d DECIMAL(10, 6),
    volatility_20d DECIMAL(10, 6),
    returns_1d DECIMAL(10, 6),
    returns_5d DECIMAL(10, 6),
    returns_20d DECIMAL(10, 6),
    volume_ma_5d DECIMAL(15, 2),
    volume_ma_20d DECIMAL(15, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date)
);

CREATE TABLE IF NOT EXISTS financial_fundamentals (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) REFERENCES companies(ticker),
    report_date DATE NOT NULL,
    filing_type VARCHAR(20),
    fiscal_year INTEGER,
    fiscal_quarter INTEGER,
    total_assets DECIMAL(20, 2),
    total_liabilities DECIMAL(20, 2),
    total_equity DECIMAL(20, 2),
    long_term_debt DECIMAL(20, 2),
    current_assets DECIMAL(20, 2),
    current_liabilities DECIMAL(20, 2),
    cash_and_equivalents DECIMAL(20, 2),
    revenue DECIMAL(20, 2),
    net_income DECIMAL(20, 2),
    operating_cash_flow DECIMAL(20, 2),
    ebitda DECIMAL(20, 2),
    debt_to_assets DECIMAL(10, 6),
    current_ratio DECIMAL(10, 6),
    quick_ratio DECIMAL(10, 6),
    gross_margin DECIMAL(10, 6),
    operating_margin DECIMAL(10, 6),
    net_margin DECIMAL(10, 6),
    roa DECIMAL(10, 6),
    roe DECIMAL(10, 6),
    revenue_growth_yoy DECIMAL(10, 6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, report_date, filing_type)
);

CREATE TABLE IF NOT EXISTS credit_proxy_data (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    hyg_price DECIMAL(15, 4),
    hyg_yield DECIMAL(10, 6),
    lqd_price DECIMAL(15, 4),
    lqd_yield DECIMAL(10, 6),
    vix DECIMAL(10, 6),
    ted_spread DECIMAL(10, 6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date)
);

CREATE TABLE IF NOT EXISTS risk_labels (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) REFERENCES companies(ticker),
    date DATE NOT NULL,
    abnormal_negative_return_1d BOOLEAN DEFAULT FALSE,
    abnormal_negative_return_5d BOOLEAN DEFAULT FALSE,
    abnormal_negative_return_20d BOOLEAN DEFAULT FALSE,
    abnormal_volume_spike_1d BOOLEAN DEFAULT FALSE,
    abnormal_volume_spike_5d BOOLEAN DEFAULT FALSE,
    volatility_jump_5d BOOLEAN DEFAULT FALSE,
    volatility_jump_20d BOOLEAN DEFAULT FALSE,
    credit_proxy_widening_5d BOOLEAN DEFAULT FALSE,
    credit_proxy_widening_20d BOOLEAN DEFAULT FALSE,
    distress_news_followup_30d BOOLEAN DEFAULT FALSE,
    distress_news_followup_90d BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date)
);

CREATE TABLE IF NOT EXISTS risk_scores (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) REFERENCES companies(ticker),
    date DATE NOT NULL,
    risk_score DECIMAL(10, 6),
    risk_level VARCHAR(20),
    model_version VARCHAR(50),
    top_features JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date, model_version)
);

CREATE TABLE IF NOT EXISTS portfolios (
    portfolio_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    holdings JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

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

CREATE INDEX IF NOT EXISTS idx_news_ticker_date ON news_items(ticker, published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_processed ON news_items(is_processed);
CREATE INDEX IF NOT EXISTS idx_market_data_ticker_date ON daily_market_data(ticker, date DESC);
CREATE INDEX IF NOT EXISTS idx_financials_ticker_date ON financial_fundamentals(ticker, report_date DESC);
CREATE INDEX IF NOT EXISTS idx_risk_labels_ticker_date ON risk_labels(ticker, date DESC);
CREATE INDEX IF NOT EXISTS idx_risk_scores_ticker_date ON risk_scores(ticker, date DESC);
CREATE INDEX IF NOT EXISTS idx_reports_ticker ON risk_reports(ticker, generated_at DESC);
CREATE INDEX IF NOT EXISTS idx_portfolio_analyses_portfolio ON portfolio_analyses(portfolio_id, created_at DESC);

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_companies_updated_at ON companies;
CREATE TRIGGER update_companies_updated_at
    BEFORE UPDATE ON companies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_portfolios_updated_at ON portfolios;
CREATE TRIGGER update_portfolios_updated_at
    BEFORE UPDATE ON portfolios
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
