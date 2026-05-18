"""
Market Reaction API Module
Provides REST endpoints for cross-market reaction analysis and lead-lag detection.

Uses dependency injection for service lifecycle management.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, date, timedelta
import logging

from apps.api.app.dependencies import get_market_reaction_service, get_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reaction", tags=["Reactions"])


# Request / Response models
class ReactionWindowMetrics(BaseModel):
    cumulative_abnormal_return: float = 0.0
    abnormal_volume_ratio: float = 1.0
    volatility_change_ratio: float = 1.0
    hyg_yield_change: Optional[float] = None
    lqd_yield_change: Optional[float] = None
    vix_change: Optional[float] = None


class ReactionItem(BaseModel):
    news_id: int
    ticker: str
    event_date: str
    event_type: str
    llm_market_impact: str
    observed_impact_type: str
    windows: Dict[str, Dict[str, Any]]


class ReactionAnalysisResponse(BaseModel):
    total_events: int
    agreement: Dict[str, Any]
    reactions: List[ReactionItem]


class LagAnalysisResponse(BaseModel):
    lead_lag_analysis: Dict[str, Any]
    cross_correlation: Dict[str, Any]
    event_count: int


class CompareResponse(BaseModel):
    total_events: int
    agreement: Dict[str, Any]
    confusion_flow: Dict[str, int]


class ReactionRequest(BaseModel):
    tickers: List[str] = Field(..., description="Company tickers")
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    min_credit_risk_score: int = Field(default=0, description="Minimum credit risk score filter")


class LagRequest(BaseModel):
    tickers: List[str] = Field(..., description="Company tickers")
    start_date: Optional[date] = None
    end_date: Optional[date] = None


@router.post("/analyze", response_model=ReactionAnalysisResponse, summary="Analyze market reactions")
def analyze_reactions(
    request: ReactionRequest,
    svc=Depends(get_market_reaction_service),
):
    """Compute post-news market reactions across equity and credit markets over 4 time windows."""
    try:
        result = svc.analyze_reactions(
            request.tickers, request.start_date, request.end_date, request.min_credit_risk_score
        )
        if "error" in result:
            return ReactionAnalysisResponse(total_events=0, agreement={}, reactions=[])
        return ReactionAnalysisResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reaction analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/lag", response_model=LagAnalysisResponse, summary="Analyze lead-lag patterns")
def analyze_lag(
    request: LagRequest,
    svc=Depends(get_market_reaction_service),
):
    """Analyze equity/credit lead-lag patterns: which market moves first after news events."""
    try:
        result = svc.analyze_lag(request.tickers, request.start_date, request.end_date)
        return LagAnalysisResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lag analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/news/{news_id}", summary="Reaction by news")
def get_reaction_by_news(
    news_id: int,
    svc=Depends(get_market_reaction_service),
):
    """Get cross-market reaction metrics for a specific news event."""
    try:
        result = svc.get_reaction_by_news(news_id)
        if result is None:
            raise HTTPException(status_code=404, detail="News event not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get reaction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ticker/{ticker}", summary="Reactions by ticker")
def get_reaction_by_ticker(
    ticker: str,
    days: int = Query(default=90, ge=1, le=365, description="Days of history"),
    svc=Depends(get_market_reaction_service),
):
    """Get recent market reactions for a specific ticker."""
    try:
        reactions = svc.get_reaction_by_ticker(ticker, days)
        return {"ticker": ticker, "reactions": reactions}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get ticker reactions failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare", response_model=CompareResponse, summary="Compare LLM vs observed")
def compare_llm_vs_actual(
    tickers: List[str] = Query(..., description="Company tickers"),
    days: int = Query(default=90, ge=1, le=365, description="Days of history"),
    svc=Depends(get_market_reaction_service),
):
    """Compare LLM-predicted market_impact_type against observed market reactions."""
    try:
        result = svc.compare_llm_vs_actual(tickers, days)
        if "error" in result:
            return CompareResponse(total_events=0, agreement={}, confusion_flow={})
        return CompareResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Compare analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary", summary="Reaction summary")
def get_reaction_summary(
    tickers: str = Query(..., description="Comma-separated tickers"),
    days: int = Query(default=90, ge=1, le=365, description="Days of history"),
    svc=Depends(get_market_reaction_service),
):
    """Get a combined reaction and lag analysis summary for a set of tickers."""
    try:
        ticker_list = [t.strip() for t in tickers.split(",") if t.strip()]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        reactions_result = svc.analyze_reactions(ticker_list, start_date, end_date)
        lag_result = svc.analyze_lag(ticker_list, start_date, end_date)

        return {
            "reaction_analysis": reactions_result.get("agreement", {}),
            "lag_analysis": lag_result.get("lead_lag_analysis", {}),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summary failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
