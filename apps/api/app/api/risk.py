"""
Risk API Module
Provides REST endpoints for risk scores, model evaluation, and risk label generation.

Uses dependency injection for service lifecycle management.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, date
import logging

from apps.api.app.dependencies import get_risk_model_service, get_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/risk", tags=["Risk"])


# Request / Response models
class RiskScoreResponse(BaseModel):
    ticker: str
    date: date
    risk_score: float
    risk_level: str
    model_version: str
    top_features: Optional[List[Dict[str, Any]]] = None


class RiskHistoryResponse(BaseModel):
    ticker: str
    history: List[RiskScoreResponse]


class LabelGenerationRequest(BaseModel):
    tickers: List[str] = Field(..., description="Company tickers to generate labels for")
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class LabelGenerationResponse(BaseModel):
    status: str
    rows_generated: int
    tickers_processed: List[str]


class TrainRequest(BaseModel):
    tickers: List[str] = Field(..., description="Company tickers for training")
    end_date: Optional[date] = None
    n_splits: int = Field(default=5, description="Cross-validation folds")


class TrainResponse(BaseModel):
    status: str
    models: Dict[str, Any]


class ScoreRequest(BaseModel):
    tickers: List[str] = Field(..., description="Company tickers to score")
    model_name: Optional[str] = Field(default=None, description="Model to use (defaults to best)")


class ScoreResponse(BaseModel):
    status: str
    scores: List[RiskScoreResponse]


@router.post("/labels/generate", response_model=LabelGenerationResponse, summary="Generate risk labels")
def generate_risk_labels(
    request: LabelGenerationRequest,
    svc=Depends(get_risk_model_service),
):
    """Generate multi-label risk indicators from market data and LLM signals."""
    try:
        start = request.start_date or datetime.now().date()
        end = request.end_date or datetime.now().date()
        labels = svc.generate_risk_labels(request.tickers, start, end)
        return LabelGenerationResponse(
            status="ok",
            rows_generated=len(labels),
            tickers_processed=request.tickers,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Label generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/train", response_model=TrainResponse, summary="Train risk models")
def train_risk_models(
    request: TrainRequest,
    svc=Depends(get_risk_model_service),
):
    """Train Logistic Regression, LightGBM, and XGBoost risk models using walk-forward CV."""
    try:
        results = svc.train_models(request.tickers, request.end_date, request.n_splits)
        return TrainResponse(status="ok", models=results)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scores/generate", response_model=ScoreResponse, summary="Generate risk scores")
def generate_risk_scores(
    request: ScoreRequest,
    svc=Depends(get_risk_model_service),
):
    """Score companies using the best available model, persisting results with top-5 drivers."""
    try:
        scores_df = svc.score_companies(request.tickers, request.model_name)
        scores = []
        if not scores_df.empty:
            for _, row in scores_df.iterrows():
                scores.append(RiskScoreResponse(
                    ticker=row['ticker'],
                    date=row.get('date', datetime.now().date()),
                    risk_score=float(row['risk_score']),
                    risk_level=row['risk_level'],
                    model_version=row['model_version'],
                    top_features=row.get('top_features'),
                ))
        return ScoreResponse(status="ok", scores=scores)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Risk scoring failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scores", response_model=List[RiskScoreResponse], summary="Get risk scores")
def get_risk_scores(
    tickers: str = Query(..., description="Comma-separated tickers"),
    model_name: Optional[str] = None,
    svc=Depends(get_risk_model_service),
):
    """Get latest risk scores with top-5 drivers for one or more companies."""
    try:
        ticker_list = [t.strip() for t in tickers.split(",") if t.strip()]
        results = svc.get_risk_summary(ticker_list)
        return [
            RiskScoreResponse(
                ticker=r['ticker'],
                date=r.get('date', datetime.now().date()),
                risk_score=float(r.get('risk_score', 0)),
                risk_level=r.get('risk_level', 'Low'),
                model_version=r.get('model_version', ''),
                top_features=r.get('top_features'),
            )
            for r in results
        ]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get risk scores: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scores/{ticker}", response_model=RiskHistoryResponse, summary="Risk score history")
def get_risk_history(
    ticker: str,
    days: int = Query(default=90, ge=1, le=365, description="Days of history"),
    svc=Depends(get_risk_model_service),
):
    """Get risk score history for a single company over a configurable time window."""
    try:
        history = svc.get_company_risk_history(ticker, days)
        return RiskHistoryResponse(
            ticker=ticker,
            history=[
                RiskScoreResponse(
                    ticker=h['ticker'],
                    date=h.get('date', datetime.now().date()),
                    risk_score=float(h.get('risk_score', 0)),
                    risk_level=h.get('risk_level', 'Low'),
                    model_version=h.get('model_version', ''),
                    top_features=h.get('top_features'),
                )
                for h in history
            ],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get risk history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/evaluation", summary="Model evaluation report")
def get_model_evaluation(svc=Depends(get_risk_model_service)):
    """Get the latest model evaluation report (AUC, Precision@K, Recall@K, top features)."""
    try:
        report = svc.get_model_evaluation()
        if report is None:
            return {"status": "no_models_trained", "models": {}}
        return {"status": "ok", **report}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
