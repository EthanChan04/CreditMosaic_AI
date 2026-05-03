"""
Portfolio API Router
Endpoints for portfolio risk analysis, management, and diversification metrics.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query

from apps.api.app.dependencies import get_portfolio_service, get_risk_model_service
from apps.api.app.schemas.portfolio import (
    PortfolioAnalyzeRequest, PortfolioAnalyzeResponse,
    PortfolioListResponse, PortfolioDetailResponse, PortfolioSummary,
)
from apps.api.app.schemas.common import SuccessMessage

import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Portfolios"])


@router.post(
    "/portfolio/analyze",
    response_model=PortfolioAnalyzeResponse,
    summary="Analyze portfolio risk",
    description=(
        "Analyze portfolio risk by computing weighted risk scores, "
        "identifying top contributors, and calculating diversification metrics. "
        "Optionally saves the portfolio for later reference."
    ),
)
def analyze_portfolio(request: PortfolioAnalyzeRequest, svc=Depends(get_portfolio_service)):
    try:
        holdings = svc.normalize_holdings([h.model_dump() for h in request.holdings])
        result = svc.analyze_portfolio(
            holdings=holdings,
            name=request.name,
            description=request.description,
        )
        return result
    except Exception as e:
        logger.error("Portfolio analysis failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/portfolios",
    response_model=PortfolioListResponse,
    summary="List saved portfolios",
    description="Get all previously saved portfolio configurations with their latest risk analysis.",
)
def list_portfolios(svc=Depends(get_portfolio_service)):
    portfolios = svc.list_portfolios()
    return PortfolioListResponse(total=len(portfolios), portfolios=portfolios)


@router.get(
    "/portfolio/{portfolio_id}",
    response_model=PortfolioDetailResponse,
    summary="Portfolio detail",
    description="Get a saved portfolio with holdings and latest risk analysis snapshot.",
)
def get_portfolio(portfolio_id: int, svc=Depends(get_portfolio_service)):
    portfolio = svc.get_portfolio(portfolio_id)
    if not portfolio:
        raise HTTPException(status_code=404, detail=f"Portfolio {portfolio_id} not found")
    return portfolio


@router.delete(
    "/portfolio/{portfolio_id}",
    response_model=SuccessMessage,
    summary="Delete portfolio",
    description="Delete a saved portfolio and its analysis history.",
)
def delete_portfolio(portfolio_id: int, svc=Depends(get_portfolio_service)):
    if not svc.delete_portfolio(portfolio_id):
        raise HTTPException(status_code=404, detail=f"Portfolio {portfolio_id} not found")
    return SuccessMessage(message=f"Portfolio {portfolio_id} deleted")


# ------------------------------------------------------------------
# Part 7: Advanced portfolio analytics
# ------------------------------------------------------------------

from pydantic import BaseModel, Field


class CorrelationRequest(BaseModel):
    holdings: List[dict] = Field(..., min_length=2, max_length=50)
    days: int = Field(default=60, ge=20, le=252)


class StressTestRequest(BaseModel):
    holdings: List[dict] = Field(..., min_length=1, max_length=50)
    scenarios: Optional[List[dict]] = None


class PortfolioReportRequest(BaseModel):
    holdings: List[dict] = Field(..., min_length=1, max_length=50)
    include_correlations: bool = True
    include_stress: bool = True


class PortfolioReportResponse(BaseModel):
    ticker: str = "portfolio"
    report_type: str = "portfolio_summary"
    title: str
    markdown_content: str
    report_id: Optional[int] = None


@router.post(
    "/portfolio/correlation",
    summary="Correlation-based risk decomposition",
    description=(
        "Deep portfolio analysis using daily return correlations. "
        "Returns the pairwise correlation matrix, portfolio volatility, "
        "diversification ratio, and marginal risk contributions per holding."
    ),
)
def analyze_correlations(request: CorrelationRequest, svc=Depends(get_portfolio_service)):
    try:
        return svc.analyze_with_correlations(svc.normalize_holdings(request.holdings), request.days)
    except Exception as e:
        logger.error("Correlation analysis failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/portfolio/stress-test",
    summary="Stress test portfolio",
    description=(
        "Run stress scenarios (credit spread shock, equity crash, liquidity crisis) "
        "against the portfolio and report risk increases per scenario."
    ),
)
def stress_test_portfolio(request: StressTestRequest, svc=Depends(get_portfolio_service)):
    try:
        return svc.stress_test(svc.normalize_holdings(request.holdings), request.scenarios)
    except Exception as e:
        logger.error("Stress test failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/portfolio/report",
    response_model=PortfolioReportResponse,
    summary="Generate portfolio risk report",
    description=(
        "Generate a one-page markdown risk report for an entire portfolio. "
        "Combines holdings summary, risk contribution ranking, correlation analysis, "
        "and stress test results into a single document. Returns markdown ready for download."
    ),
)
def generate_portfolio_report(
    request: PortfolioReportRequest,
    portfolio_svc=Depends(get_portfolio_service),
):
    try:
        # Get report service via DI container
        from apps.api.app.dependencies import get_report_service, get_container
        container = get_container()
        report_svc = container.report_service

        holdings = portfolio_svc.normalize_holdings(request.holdings)

        correlation_data = None
        if request.include_correlations and len(holdings) >= 2:
            correlation_data = portfolio_svc.analyze_with_correlations(holdings)

        stress_data = None
        if request.include_stress:
            stress_data = portfolio_svc.stress_test(holdings)

        markdown = portfolio_svc.generate_portfolio_report_markdown(
            holdings, correlation_data, stress_data
        )

        # Also run basic analysis to get the risk score/drivers for the summary
        analysis = portfolio_svc.analyze_portfolio(holdings)

        title = f"Portfolio Risk Report — {len(holdings)} Holdings"
        summary = {
            "total_risk": analysis.get("total_risk_score"),
            "risk_level": analysis.get("risk_level"),
            "holdings_count": len(holdings),
            "diversification_score": analysis.get("diversification_score"),
        }

        report_id = report_svc._save_report(
            None, "portfolio_summary", title, markdown, summary, "template"
        )

        return PortfolioReportResponse(
            ticker="portfolio",
            report_type="portfolio_summary",
            title=title,
            markdown_content=markdown,
            report_id=report_id,
        )
    except Exception as e:
        logger.error("Portfolio report generation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
