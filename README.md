# CreditMosaic AI

**CreditMosaic AI** is a research-oriented financial technology MVP for news-driven, cross-market credit risk monitoring. The project investigates whether structured signals extracted from firm-level news by large language models can improve the identification of short-horizon credit risk deterioration, and whether equity-market and credit-proxy reactions exhibit measurable lead-lag patterns after firm-specific news events.

The system is designed for academic research, teaching, and financial technology prototyping. It does **not** provide investment advice, trading recommendations, or credit rating opinions.

## 1. Research Motivation

Public firm news contains information about liquidity pressure, refinancing risk, litigation, regulatory exposure, earnings deterioration, management changes, and other credit-relevant events. Traditional sentiment models often compress such information into coarse positive, neutral, or negative scores. CreditMosaic AI instead treats news as a source of structured credit risk factors.

The central research question is:

> Can LLM-derived credit risk signals from firm-level news improve short-horizon corporate risk monitoring and help explain asynchronous reactions across equity markets and credit-proxy markets?

The project is inspired by research themes in machine learning asset pricing, corporate bond return prediction, financial text analysis, interpretable factor modeling, and cross-market information transmission.

## 2. MVP Scope

The first version is intentionally scoped as an interpretable risk monitoring platform rather than a general-purpose financial chatbot.

The MVP supports the following workflow:

1. Input a firm ticker or a weighted portfolio.
2. Collect firm news, equity-market data, financial fundamentals, and credit-proxy variables.
3. Extract structured news risk signals using an OpenAI-compatible LLM provider.
4. Construct daily firm-level features and proxy risk labels.
5. Train baseline and tree-based risk models.
6. Analyze market reactions and lead-lag patterns.
7. Display firm risk, news evidence, portfolio risk contribution, and one-page Markdown risk reports.

## 3. Repository Structure

```text
creditmosaic-ai/
  apps/api/                 FastAPI backend application
  db/                       PostgreSQL schema and migrations
  models/                   Trained model artifacts and evaluation reports
  pipelines/ingestion/      News, market data, fundamentals, and proxy data ingestion
  pipelines/risk/           Feature engineering, label generation, model training, scoring
  pipelines/reaction/       Cross-market event reaction and lead-lag analysis
  services/                 LLM provider, FinBERT baseline, extraction and model services
  web/                      Next.js frontend application
  docker-compose.yml        Local multi-service deployment entry point
```

## 4. Technical Stack

| Layer | Implementation |
| --- | --- |
| Frontend | Next.js, React, TypeScript, ECharts |
| Backend | FastAPI, Python |
| Operational database | PostgreSQL |
| Analytical storage | DuckDB |
| LLM provider | Longcat through an OpenAI-compatible interface |
| Extensible LLM interfaces | OpenAI, Qwen, DeepSeek-compatible provider classes |
| NLP baseline | FinBERT |
| Risk models | Logistic Regression, LightGBM, XGBoost |
| Interpretability | Feature importance and SHAP-compatible tree explanations |
| Deployment | Docker Compose for local MVP deployment |

## 5. Core Data Objects

The project follows the data objects defined in the execution plan:

- `Company`: firm identifiers, sector, exchange, market capitalization, and financial metrics.
- `NewsItem`: firm-level news text with source, publication time, and URL.
- `LLMNewsSignal`: structured sentiment, credit risk score, event type, horizon, evidence spans, and confidence.
- `MarketReaction`: abnormal return, abnormal volume, volatility change, credit-proxy movement, and reaction lag.
- `RiskAlert`: alert-level risk summary and recommended review action.

These objects are persisted through PostgreSQL tables defined in `db/schema.sql`.

## 6. Environment Variables

Copy `.env.example` to `.env` and place local secrets only in `.env` or in the shell environment.

```powershell
Copy-Item .env.example .env
```

Minimal configuration:

```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=creditmosaic
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password

API_PORT=8000
WEB_PORT=3000

LONGCAT_API_KEY=
```

The default LLM provider is `longcat`. The key is injected through `${LONGCAT_API_KEY}` in `config.yaml`; raw API keys should never be written into repository files.

## 7. Running with Docker Compose

If Docker is installed, the complete local stack can be started with:

```powershell
docker compose up --build
```

Service endpoints:

- Frontend: <http://localhost:3000>
- Backend API: <http://localhost:8000>
- OpenAPI documentation: <http://localhost:8000/docs>
- PostgreSQL: `localhost:5432`

The PostgreSQL service mounts and executes `db/schema.sql` during first initialization. To recreate the database from scratch, only after confirming that local data can be discarded:

```powershell
docker compose down -v
docker compose up --build
```

## 8. Local Development

### Backend

```powershell
cd F:\Fintech\creditmosaic-ai
.venv\Scripts\python.exe -m uvicorn apps.api.app.main:app --reload --host 127.0.0.1 --port 8000
```

Health check:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/health
```

### Frontend

```powershell
cd F:\Fintech\creditmosaic-ai\web
npm run dev -- --hostname 127.0.0.1 --port 3000
```

The frontend proxies `/api/*` requests to `NEXT_PUBLIC_API_URL`. If the variable is not set, it defaults to `http://localhost:8000`.

## 9. Database and Data Pipeline

The PostgreSQL schema is defined in `db/schema.sql`. Core tables include:

- `companies`
- `news_items`
- `llm_news_signals`
- `daily_market_data`
- `financial_fundamentals`
- `credit_proxy_data`
- `risk_labels`
- `risk_scores`
- `portfolios`
- `portfolio_analyses`
- `risk_reports`

The ingestion layer in `pipelines/ingestion/` is responsible for:

- constructing the default ticker universe,
- collecting news and market data,
- collecting financial fundamentals,
- collecting credit-proxy variables such as HYG, LQD, and VIX,
- performing company entity matching,
- writing structured records into PostgreSQL.

## 10. LLM News Signal Extraction

The LLM extraction service converts unstructured news into structured credit risk signals. The target output includes:

- `sentiment_score`,
- `credit_risk_score`,
- `event_type`,
- `risk_horizon`,
- `market_impact_type`,
- `evidence_spans`,
- `confidence`.

The project currently uses Longcat as the default OpenAI-compatible provider. Provider abstraction is implemented so that OpenAI, Qwen, and DeepSeek-compatible endpoints can be configured with the same service interface.

FinBERT is used as a traditional financial NLP baseline for sentiment comparison.

## 11. Risk Modeling

Risk modeling code is located in `pipelines/risk/`:

- `risk_labeler.py` constructs proxy labels such as abnormal negative returns, abnormal volume spikes, volatility jumps, credit-proxy widening, and distress-news follow-up indicators.
- `feature_engineer.py` merges market features, financial fundamentals, LLM news signals, and FinBERT sentiment features into daily firm-level features.
- `model_trainer.py` trains Logistic Regression, LightGBM, and XGBoost models.
- `risk_scorer.py` produces firm-level daily risk scores and top feature drivers.

Model artifacts are stored in `models/`, and the current evaluation summary is stored in `models/evaluation_report.json`.

## 12. Cross-Market Reaction Analysis

The reaction analysis module studies how news events are reflected in:

- equity returns,
- trading volume,
- realized volatility,
- credit-proxy variables such as HYG, LQD, and VIX.

The lead-lag module estimates whether equity variables lead credit-proxy variables, whether credit-proxy variables lead equity variables, or whether both markets react simultaneously.

## 13. API Overview

| Endpoint | Purpose |
| --- | --- |
| `POST /api/news/extract` | Extract structured LLM news risk signals |
| `GET /api/company/{ticker}/risk` | Retrieve firm-level risk overview |
| `GET /api/company/{ticker}/signals` | Retrieve firm-level news signals |
| `POST /api/risk/labels/generate` | Generate proxy risk labels |
| `POST /api/risk/models/train` | Train risk models |
| `POST /api/risk/scores/generate` | Generate risk scores |
| `POST /api/reaction/analyze` | Analyze cross-market reactions |
| `POST /api/reaction/lag` | Analyze lead-lag relationships |
| `POST /api/portfolio/analyze` | Analyze portfolio risk contribution |
| `POST /api/portfolio/report` | Generate a portfolio Markdown report |
| `POST /api/report/generate` | Generate a firm-level Markdown report |
| `GET /api/alerts` | Retrieve risk alerts |

## 14. Frontend Pages

| Route | Function |
| --- | --- |
| `/` | Search and project landing interface |
| `/company/[ticker]` | Firm-level risk dashboard |
| `/news/[news_id]` | News signal evidence page |
| `/reaction/[ticker]` | Cross-market reaction page |
| `/portfolio` | Portfolio risk analysis and portfolio report generation |
| `/report/[ticker]` | Firm-level Markdown risk report |

## 15. Reproducibility Checks

The following commands were used as lightweight project-level checks:

```powershell
cd F:\Fintech\creditmosaic-ai
python -m compileall apps services pipelines -q

@'
from apps.api.app.main import app
schema = app.openapi()
required = [
    "/api/news/extract",
    "/api/company/{ticker}/risk",
    "/api/company/{ticker}/signals",
    "/api/portfolio/analyze",
    "/api/report/generate",
    "/api/alerts",
]
print({"paths": len(schema["paths"]), "missing": [p for p in required if p not in schema["paths"]]})
'@ | .venv\Scripts\python.exe -

cd F:\Fintech\creditmosaic-ai\web
npm run lint
npm run build
```

## 16. Demonstration Workflow

A typical MVP demonstration follows this sequence:

```text
Input a ticker or portfolio
-> inspect news-derived credit risk signals
-> inspect cross-market reactions
-> inspect risk scores and top drivers
-> generate a one-page Markdown risk report
```

Suggested demonstration tickers include `AAPL`, `TSLA`, `NVDA`, `JPM`, and `BA`.

## 17. Security and Secret Management

- Do not commit `.env`, raw API keys, database passwords, or access tokens.
- Longcat credentials should be supplied only through `LONGCAT_API_KEY`.
- `.gitignore` and `.dockerignore` exclude `.env`, `.venv`, `web/node_modules`, `web/.next`, local database files, and cache directories.
- `config.yaml` uses variable references rather than raw secrets.
- Before publishing, scan the repository for `sk-...`, tokens, and other private credentials.

## 18. Current Limitations

- Free news sources may have incomplete coverage, delayed availability, or missing full text.
- The MVP uses credit proxies rather than TRACE, Bloomberg, Refinitiv, or CDS data.
- LLM outputs require JSON validation and manual sampling for research-quality assurance.
- Risk scores are research and monitoring signals, not investment recommendations.
- Full academic validation still requires a larger sample, at least ten case studies, and formal Model A-E comparisons.

## 19. Future Work

1. Expand to at least 50 firms with six months of queryable news and market data.
2. Complete Model A-E experiments comparing traditional variables, FinBERT, LLM signals, and combined models.
3. Produce a structured case study report for at least ten firms.
4. Add Prefect-based scheduled data pipelines and richer data quality reports.
5. Extend report export to PDF and add watchlist-based risk monitoring.
