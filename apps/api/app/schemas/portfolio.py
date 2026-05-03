"""Portfolio-related Pydantic schemas."""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class HoldingItem(BaseModel):
    ticker: str = Field(..., description="Company ticker")
    weight: float = Field(..., gt=0, le=1.0, description="Portfolio weight (0-1)")


class PortfolioAnalyzeRequest(BaseModel):
    name: Optional[str] = Field(default=None, description="Portfolio name for saving")
    description: Optional[str] = Field(default=None, description="Portfolio description")
    holdings: List[HoldingItem] = Field(..., min_length=1, max_length=50, description="Portfolio holdings")


class HoldingRiskDetail(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    weight: float
    risk_score: float
    risk_level: str
    risk_contribution: float
    top_drivers: Optional[List[Dict[str, Any]]] = None


class PortfolioAnalyzeResponse(BaseModel):
    portfolio_id: Optional[int] = None
    name: Optional[str] = None
    total_risk_score: float
    risk_level: str
    holdings_risk: List[HoldingRiskDetail]
    top_contributors: List[Dict[str, Any]]
    diversification_score: Optional[float] = None
    recommendation: Optional[str] = None


class PortfolioSummary(BaseModel):
    portfolio_id: int
    name: str
    description: Optional[str] = None
    holdings_count: int
    total_risk_score: Optional[float] = None
    risk_level: Optional[str] = None
    created_at: Optional[datetime] = None


class PortfolioListResponse(BaseModel):
    total: int
    portfolios: List[PortfolioSummary]


class PortfolioDetailResponse(BaseModel):
    portfolio_id: int
    name: str
    description: Optional[str] = None
    holdings: List[Dict[str, Any]]
    latest_analysis: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
