"""News-related Pydantic schemas."""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class NewsExtractRequest(BaseModel):
    ticker: str = Field(..., description="Company ticker")
    title: str = Field(..., description="News title")
    body: str = Field(..., description="News body")
    source: str = Field(default="", description="News source")
    url: str = Field(default="", description="News URL")
    published_at: datetime = Field(default_factory=datetime.now, description="Published time")
    provider: Optional[str] = Field(default=None, description="LLM provider name")
    model: Optional[str] = Field(default=None, description="LLM model name")


class NewsExtractResponse(BaseModel):
    news_id: int
    signal: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: float


class NewsItemResponse(BaseModel):
    news_id: int
    ticker: str
    title: str
    body: str
    source: str
    url: str
    published_at: datetime
    is_processed: bool


class NewsDetailResponse(NewsItemResponse):
    """News detail with LLM signal and market reaction."""
    signal: Optional[Dict[str, Any]] = None
    reaction: Optional[Dict[str, Any]] = None


class SignalResponse(BaseModel):
    signal_id: int
    news_id: int
    ticker: str
    sentiment_score: float
    credit_risk_score: int
    event_type: str
    risk_horizon: str
    market_impact_type: str
    evidence_spans: List[str]
    confidence: float
    extracted_at: datetime
    llm_model: str


class BatchExtractRequest(BaseModel):
    news_items: List[Dict[str, Any]]
    provider: Optional[str] = Field(default=None, description="LLM provider name")
    max_concurrent: int = Field(default=5, description="Max concurrency")


class BatchExtractResponse(BaseModel):
    total: int
    successful: int
    failed: int
    results: List[Dict[str, Any]]


class CompareFinBERTResponse(BaseModel):
    news_id: int
    ticker: str
    comparison: Dict[str, Any]
