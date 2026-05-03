"""
CreditMosaic AI — FastAPI Application
Backend API service layer for company, news, portfolio, and report management.

Provides:
  - Company API: listing, search, detail with risk enrichment
  - News API: LLM signal extraction, FinBERT comparison
  - Reaction API: cross-market reaction analysis, lead-lag detection
  - Risk API: label generation, model training, risk scoring
  - Portfolio API: portfolio risk analysis, diversification metrics
  - Report API: AI-powered risk report generation and export

Usage:
    uvicorn apps.api.app.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from apps.api.app.dependencies import init_app, get_container
from apps.api.app.schemas.common import HealthResponse, ErrorResponse, ErrorDetail

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("creditmosaic.api")

description = """
## CreditMosaic AI — Backend API

Risk intelligence platform combining LLM-driven news analysis with quantitative
risk modeling for cross-market credit risk assessment.

### Modules

| Module | Description |
|--------|-------------|
| **Companies** | Company directory, search, and risk-enriched profiles |
| **News** | News ingestion, LLM signal extraction, FinBERT baseline comparison |
| **Reactions** | Cross-market reaction analysis, lead-lag detection |
| **Risk** | Risk labels, model training, risk scoring with top-5 drivers |
| **Portfolios** | Portfolio risk analysis, diversification metrics, optimization |
| **Reports** | AI-powered markdown risk report generation and download |

### Architecture

All endpoints are RESTful. The API uses dependency injection for service lifecycle
management. LLM integration supports OpenAI-compatible, Qwen, and DeepSeek providers.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    logger.info("Starting CreditMosaic AI API server...")
    init_app()
    container = get_container()
    logger.info("LLM providers: %s", container.llm_manager.list_providers())
    logger.info("Database: PostgreSQL connected=%s", container.db.postgres_conn is not None)
    yield
    logger.info("Shutting down CreditMosaic AI API server...")
    container.close()


app = FastAPI(
    title="CreditMosaic AI API",
    description=description,
    version="1.0.0",
    contact={
        "name": "CreditMosaic AI Team",
        "url": "https://github.com/creditmosaic",
    },
    license_info={
        "name": "MIT",
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_request_timing(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    response.headers["X-Process-Time"] = f"{elapsed:.4f}"
    if elapsed > 1.0:
        logger.warning("Slow request: %s %s — %.2fs", request.method, request.url.path, elapsed)
    return response


# Global exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error=ErrorDetail(code="BAD_REQUEST", message=str(exc))
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=ErrorDetail(code="INTERNAL_ERROR", message="An unexpected error occurred")
        ).model_dump(),
    )


# Health check
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """System health check: database connectivity, LLM providers, FinBERT."""
    try:
        container = get_container()
        return HealthResponse(
            status="healthy",
            db_connected=container.db.postgres_conn is not None,
            llm_providers=len(container.llm_manager.list_providers()),
            finbert_initialized=container.finbert.initialized if container.finbert else False,
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            db_connected=False,
            llm_providers=0,
        )


# Register routers
from apps.api.app.api.news import router as news_router
from apps.api.app.api.risk import router as risk_router
from apps.api.app.api.reaction import router as reaction_router
from apps.api.app.api.company import router as company_router
from apps.api.app.api.portfolio import router as portfolio_router
from apps.api.app.api.report import router as report_router
from apps.api.app.api.alerts import router as alerts_router

app.include_router(company_router, prefix="/api")
app.include_router(news_router, prefix="/api")
app.include_router(risk_router, prefix="/api")
app.include_router(reaction_router, prefix="/api")
app.include_router(portfolio_router, prefix="/api")
app.include_router(report_router, prefix="/api")
app.include_router(alerts_router, prefix="/api")


# Static file serving for frontend (when built)
# from fastapi.staticfiles import StaticFiles
# import os
# static_dir = os.path.join(os.path.dirname(__file__), "..", "..", "web", "out")
# if os.path.isdir(static_dir):
#     app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
