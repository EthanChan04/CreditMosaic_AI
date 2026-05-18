# CreditMosaic AI

**CreditMosaic AI** is a research-oriented financial technology platform for news-driven corporate risk monitoring and cross-market reaction analysis. The project examines whether structured credit-risk signals extracted from firm-level news by large language models (LLMs) can improve short-horizon risk assessment beyond conventional market, fundamental, and sentiment-based features.

The repository is designed as an **academic project and reproducible research prototype**. It integrates data ingestion, structured text extraction, feature engineering, risk-label construction, predictive modelling, ablation analysis, and an application layer for interactive inspection. It is **not** intended to provide investment advice, trading recommendations, or formal credit ratings.

## 1. Research Context and Objectives

Corporate news frequently contains information relevant to liquidity conditions, refinancing pressure, litigation exposure, earnings deterioration, management instability, regulatory intervention, and other mechanisms associated with deteriorating credit quality. Traditional financial-text pipelines often reduce such information to coarse sentiment scores, which may fail to distinguish between generic negative tone and events with specific credit implications.

CreditMosaic AI is motivated by the proposition that news should be represented as a source of **structured credit-risk information**, rather than as sentiment alone. The project therefore addresses the following research questions:

1. Can LLM-derived credit-risk signals improve short-horizon corporate risk monitoring relative to models based only on market and fundamental variables?
2. Do LLM-derived signals provide incremental explanatory value beyond a traditional financial NLP baseline such as FinBERT?
3. Following firm-specific news events, do equity-market variables and credit-proxy variables exhibit measurable lead-lag relationships?

These questions connect the project to research in financial text analysis, machine learning for asset pricing, interpretable predictive modelling, and cross-market information transmission.

## 2. Methodological Framework

The current system implements an end-to-end research pipeline:

1. **Data acquisition**: firm-level news, equity-market data, financial fundamentals, and market-wide credit proxies are collected and stored in PostgreSQL.
2. **Structured text extraction**: an OpenAI-compatible LLM provider transforms unstructured news into explicit variables such as event type, credit-risk score, risk horizon, evidence spans, and confidence.
3. **Baseline comparison**: FinBERT is used as a conventional financial-sentiment benchmark.
4. **Feature construction**: daily firm-level features combine market variables, fundamentals, credit proxies, FinBERT outputs, and LLM-derived signals.
5. **Leakage control**: features that directly proxy label definitions are removed or lagged to reduce target leakage, and model evaluation is performed with date-aware walk-forward splits.
6. **Risk modelling**: Logistic Regression, LightGBM, and XGBoost are trained for short-horizon risk prediction.
7. **Ablation analysis**: Model A-E experiments quantify the incremental contribution of traditional variables, FinBERT, LLM features, and the full feature set.
8. **Reaction analysis**: abnormal returns, volume changes, volatility responses, and credit-proxy movements are examined after firm-specific news events.

### 2.1 Ablation Design

The repository includes a structured ablation workflow in `pipelines/risk/ablation_runner.py`:

| Model | Feature Set |
| --- | --- |
| Model A | Market and fundamental variables |
| Model B | Model A + FinBERT sentiment |
| Model C | Model A + LLM-derived credit-risk signals |
| Model D | Model A + FinBERT + LLM signals |
| Model E | Full model, including credit-proxy and cross-sectional features |

The ablation module records AUC, Brier score, precision-oriented metrics, bootstrap confidence intervals, and comparative model summaries. A dedicated experiment-tracking utility is provided in `pipelines/risk/experiment_tracker.py` for explicit run logging and subsequent comparison.

## 3. System Architecture

```text
creditmosaic-ai/
  apps/api/                 FastAPI application and REST endpoints
  db/                       PostgreSQL schema and database definitions
  pipelines/ingestion/      Data acquisition and persistence workflows
  pipelines/risk/           Label generation, feature engineering, training, scoring, ablation
  pipelines/reaction/       Event-response and lead-lag analysis
  services/                 LLM, FinBERT, extraction, and model orchestration services
  tests/                    Unit and integration test suites
  web/                      Next.js frontend application
  docker-compose.yml        Local multi-service deployment configuration
```

### 3.1 Principal Components

| Component | Role |
| --- | --- |
| Frontend | Interactive Next.js interface for company, portfolio, report, and reaction views |
| Backend API | FastAPI service exposing research and application workflows |
| Operational database | PostgreSQL for structured application and research records |
| Analytical storage | DuckDB-backed analytical workflows where appropriate |
| NLP layer | LLM-based structured extraction with FinBERT as a baseline comparator |
| Modelling layer | Logistic Regression, LightGBM, and XGBoost |
| Deployment layer | Docker Compose for reproducible local orchestration |

## 4. Data Model

The project is organised around several central data objects:

- `Company`: firm identifiers, sector information, exchange metadata, and financial characteristics.
- `NewsItem`: source text, publication time, linked firm, and document-level metadata.
- `LLMNewsSignal`: structured event interpretation, risk score, horizon, evidence spans, and extraction confidence.
- `MarketReaction`: abnormal return, abnormal volume, volatility response, credit-proxy movement, and reaction lag.
- `RiskAlert`: alert-level summary intended for monitoring and review.

These entities are persisted through the PostgreSQL schema defined in `db/schema.sql`.

## 5. Installation and Configuration

### 5.1 Environment Variables

Create a local environment file from the provided template:

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

Secrets should remain in `.env` or the shell environment and must not be committed to version control. The LLM provider is configured through an OpenAI-compatible interface, with Longcat currently used as the default provider.

### 5.2 Running the Full Stack with Docker Compose

```powershell
docker compose up --build
```

Default service endpoints:

| Service | Address |
| --- | --- |
| Frontend | `http://localhost:3000` |
| Backend API | `http://localhost:8000` |
| OpenAPI documentation | `http://localhost:8000/docs` |
| PostgreSQL | `localhost:5432` |

To recreate the database from a clean local state:

```powershell
docker compose down -v
docker compose up --build
```

### 5.3 Local Development

Backend:

```powershell
cd F:\Fintech\creditmosaic-ai
python -m uvicorn apps.api.app.main:app --reload --host 127.0.0.1 --port 8000
```

Frontend:

```powershell
cd F:\Fintech\creditmosaic-ai\web
npm run dev -- --hostname 127.0.0.1 --port 3000
```

Health check:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/health
```

## 6. Functional Capabilities

### 6.1 API Endpoints

| Endpoint | Purpose |
| --- | --- |
| `POST /api/news/extract` | Extract structured LLM-based credit-risk signals from news |
| `GET /api/company/{ticker}/risk` | Retrieve firm-level risk overview |
| `GET /api/company/{ticker}/signals` | Retrieve firm-level news signals |
| `POST /api/risk/labels/generate` | Construct proxy risk labels |
| `POST /api/risk/models/train` | Train predictive risk models |
| `POST /api/risk/scores/generate` | Generate firm-level risk scores |
| `POST /api/reaction/analyze` | Analyse cross-market event reactions |
| `POST /api/reaction/lag` | Estimate lead-lag relationships |
| `POST /api/portfolio/analyze` | Analyse portfolio-level risk contribution |
| `POST /api/portfolio/report` | Generate a portfolio risk report |
| `POST /api/report/generate` | Generate a firm-level Markdown report |
| `GET /api/alerts` | Retrieve risk alerts |

### 6.2 Frontend Views

| Route | Function |
| --- | --- |
| `/` | Landing page and ticker search |
| `/company/[ticker]` | Firm-level risk dashboard |
| `/news/[news_id]` | News interpretation and evidence page |
| `/reaction/[ticker]` | Cross-market reaction analysis |
| `/portfolio` | Portfolio analysis and report generation |
| `/report/[ticker]` | Firm-level Markdown report view |

## 7. Reproducibility and Evaluation

The repository includes unit tests, integration tests, model-comparison utilities, and a dedicated experiment-tracking utility. Typical project-level checks include:

```powershell
python -m compileall apps services pipelines -q
python -m pytest
```

For frontend verification:

```powershell
cd web
npm run lint
npm run build
```

Generated model artefacts, evaluation outputs, and experiment logs are expected to be produced during execution rather than treated as immutable source files. The modelling workflow is designed to support:

- time-aware walk-forward validation,
- comparison across multiple model families,
- explicit control of target leakage,
- ablation-based contribution analysis,
- confidence-interval estimation and comparative model summaries.

## 8. Illustrative Research Workflow

```text
Select a firm or portfolio
-> collect news, market, fundamental, and proxy data
-> extract structured credit-risk signals from news
-> construct daily features and proxy labels
-> train and compare predictive models
-> inspect event reactions and lead-lag behaviour
-> generate a concise risk report for review
```

Suggested demonstration tickers include `AAPL`, `TSLA`, `NVDA`, `JPM`, and `BA`.

## 9. Security and Research Governance

- Secret values such as API keys and database credentials must not be committed.
- `.gitignore` and `.dockerignore` exclude local secrets, virtual environments, dependency caches, and local database artefacts.
- The API restricts CORS origins through the `CORS_ORIGINS` environment variable rather than permitting unrestricted production access.
- The containerised API is configured to run as a non-root user.
- LLM outputs should be validated, sampled, and reviewed before being used in formal research conclusions.

## 10. Current Limitations

The project remains a research prototype and is subject to several limitations:

1. Public or low-cost news sources may provide incomplete coverage, delayed publication, or uneven document quality.
2. The system currently relies on market-based credit proxies rather than proprietary bond, CDS, or institutional fixed-income datasets.
3. Proxy labels are useful for experimentation but do not constitute ground-truth credit events.
4. LLM-derived signals require further validation for stability, calibration, and sensitivity to prompting or provider choice.
5. Large-sample empirical validation, broader firm coverage, and formal case-study design remain necessary before strong academic claims can be made.

## 11. Future Research Directions

Future development may proceed along the following lines:

1. Expand the firm universe and historical observation window.
2. Complete large-sample Model A-E comparisons under a unified experimental protocol.
3. Conduct structured case studies on firms experiencing distinct classes of credit-relevant events.
4. Integrate richer fixed-income data sources when available.
5. Extend experiment tracking, data-quality monitoring, and report export capabilities.
6. Evaluate calibration, robustness, and interpretability under alternative LLM providers and prompt specifications.

## 12. Intended Use

CreditMosaic AI is intended for:

- academic research,
- methodological experimentation,
- financial technology prototyping,
- classroom or thesis-oriented demonstrations of applied machine learning in finance.

It should not be interpreted as a production credit-rating system, an investment decision engine, or a substitute for professional financial judgement.
