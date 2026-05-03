"""
Company API Router
Endpoints for company directory, search, and risk-enriched profiles.
"""

from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends, Query

from apps.api.app.dependencies import get_company_service, get_risk_model_service, get_db
from apps.api.app.schemas.company import (
    CompanyResponse, CompanyDetailResponse, CompanyListResponse,
    CompanySearchRequest, SectorsResponse, CompanyRiskHistoryResponse,
    CompanyNewsResponse, CompanyUpsertRequest,
)
from apps.api.app.schemas.common import PaginatedResponse, SuccessMessage, ErrorResponse, ErrorDetail

import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Companies"])


@router.get(
    "/companies",
    response_model=PaginatedResponse[CompanyResponse],
    summary="List all companies",
    description="Paginated company directory with optional sector filter and text search.",
)
def list_companies(
    sector: Optional[str] = Query(default=None, description="Filter by sector (e.g., Technology)"),
    search: Optional[str] = Query(default=None, description="Search by ticker or company name"),
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(default=20, ge=1, le=200, description="Items per page"),
    svc=Depends(get_company_service),
):
    result = svc.list_companies(sector=sector, search=search, page=page, page_size=page_size)
    return result


@router.get(
    "/companies/sectors",
    response_model=SectorsResponse,
    summary="List sectors",
    description="Get all unique sectors with company counts and average market cap.",
)
def list_sectors(svc=Depends(get_company_service)):
    sectors = svc.get_companies_by_sector()
    return SectorsResponse(sectors=sectors)


@router.get(
    "/companies/search",
    response_model=List[CompanyResponse],
    summary="Search companies",
    description="Quick-search companies by name or ticker with optional sector filter.",
)
def search_companies(
    q: str = Query(..., min_length=1, description="Search query"),
    sector: Optional[str] = Query(default=None, description="Filter by sector"),
    limit: int = Query(default=20, ge=1, le=100),
    svc=Depends(get_company_service),
):
    return svc.search_companies(q, sector, limit)


@router.get(
    "/companies/{ticker}",
    response_model=CompanyDetailResponse,
    summary="Company detail",
    description="Full company profile enriched with latest risk score, news counts, and market data.",
)
def get_company(ticker: str, svc=Depends(get_company_service)):
    company = svc.get_company_detail(ticker)
    if not company:
        raise HTTPException(status_code=404, detail=f"Company '{ticker}' not found")
    return company


@router.post(
    "/companies",
    response_model=CompanyResponse,
    status_code=201,
    summary="Create or update company",
    description="Upsert a company record by ticker.",
)
def upsert_company(request: CompanyUpsertRequest, svc=Depends(get_company_service)):
    try:
        return svc.upsert_company(request.model_dump())
    except Exception as e:
        logger.error("Company upsert failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/companies/{ticker}",
    response_model=SuccessMessage,
    summary="Delete company",
    description="Remove a company and all related data from the database.",
)
def delete_company(ticker: str, svc=Depends(get_company_service)):
    if not svc.delete_company(ticker):
        raise HTTPException(status_code=404, detail=f"Company '{ticker}' not found")
    return SuccessMessage(message=f"Company '{ticker}' deleted")


@router.get(
    "/companies/{ticker}/news",
    response_model=CompanyNewsResponse,
    summary="Company news",
    description="Recent news for a company with associated LLM risk signals.",
)
def get_company_news(
    ticker: str,
    limit: int = Query(default=50, ge=1, le=200),
    svc=Depends(get_company_service),
):
    news_list = svc.get_company_news(ticker, limit)
    return CompanyNewsResponse(ticker=ticker, total=len(news_list), news=news_list)


@router.get(
    "/companies/{ticker}/risk-history",
    response_model=CompanyRiskHistoryResponse,
    summary="Risk score history",
    description="Historical risk scores for a company over a configurable time window.",
)
def get_company_risk_history(
    ticker: str,
    days: int = Query(default=90, ge=1, le=730, description="Days of history"),
    risk_svc=Depends(get_risk_model_service),
):
    history = risk_svc.get_company_risk_history(ticker, days)
    return CompanyRiskHistoryResponse(ticker=ticker, history=history)


@router.get(
    "/company/{ticker}/risk",
    summary="MVP compatibility: company risk summary",
    description="Compatibility endpoint for GET /api/company/{ticker}/risk.",
)
def get_company_risk_summary(ticker: str, days: int = Query(default=90, ge=1, le=730), risk_svc=Depends(get_risk_model_service)):
    summary = risk_svc.get_risk_summary([ticker])
    history = risk_svc.get_company_risk_history(ticker, days)
    latest = summary[0] if summary else None
    return {"ticker": ticker, "latest": latest, "history": history}


@router.get(
    "/company/{ticker}/signals",
    summary="MVP compatibility: company LLM signals",
    description="Compatibility endpoint for GET /api/company/{ticker}/signals.",
)
def get_company_signals(
    ticker: str,
    limit: int = Query(default=100, ge=1, le=500),
    db=Depends(get_db),
):
    sql = """
        SELECT signal_id, news_id, ticker, sentiment_score, credit_risk_score,
               event_type, risk_horizon, market_impact_type, evidence_spans,
               confidence, extracted_at, llm_model
        FROM llm_news_signals
        WHERE ticker = %s
        ORDER BY extracted_at DESC
        LIMIT %s
    """
    with db.cursor() as cur:
        cur.execute(sql, (ticker, limit))
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]
