"""Company-related Pydantic schemas."""

from typing import Optional, List, Dict, Any
from datetime import date, datetime
from pydantic import BaseModel, Field


class CompanyBase(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=20, description="Stock ticker symbol")
    company_name: str = Field(..., min_length=1, max_length=255)
    sector: Optional[str] = None
    industry: Optional[str] = None
    exchange: Optional[str] = None
    market_cap: Optional[float] = None
    country: Optional[str] = None
    founded_year: Optional[int] = None


class CompanyResponse(CompanyBase):
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class CompanyDetailResponse(CompanyResponse):
    """Company detail enriched with latest risk and market data."""
    latest_risk_score: Optional[float] = None
    risk_level: Optional[str] = None
    news_count_30d: int = 0
    high_risk_news_count_30d: int = 0
    latest_price: Optional[float] = None
    price_change_5d: Optional[float] = None


class CompanyListResponse(BaseModel):
    total: int
    companies: List[CompanyResponse]


class CompanySearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query for company name or ticker")
    sector: Optional[str] = None
    limit: int = Field(default=20, ge=1, le=100)


class SectorsResponse(BaseModel):
    sectors: List[Dict[str, Any]]


class CompanyRiskHistoryResponse(BaseModel):
    ticker: str
    history: List[Dict[str, Any]]


class CompanyNewsResponse(BaseModel):
    ticker: str
    total: int
    news: List[Dict[str, Any]]


class CompanyUpsertRequest(CompanyBase):
    """Request model for creating or updating a company."""
    pass
