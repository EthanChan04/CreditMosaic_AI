"""Report-related Pydantic schemas."""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class ReportGenerateRequest(BaseModel):
    ticker: str = Field(..., description="Company ticker")
    report_type: str = Field(
        default="company_risk",
        description="Report type: company_risk, sector_comparison, portfolio_summary"
    )
    provider: Optional[str] = Field(default=None, description="LLM provider for report generation")


class ReportResponse(BaseModel):
    report_id: int
    ticker: str
    report_type: str
    title: str
    markdown_content: str
    summary: Optional[Dict[str, Any]] = None
    model_used: Optional[str] = None
    generated_at: datetime


class ReportSummaryItem(BaseModel):
    report_id: int
    ticker: str
    title: str
    report_type: str
    generated_at: datetime


class ReportListResponse(BaseModel):
    total: int
    reports: List[ReportSummaryItem]
