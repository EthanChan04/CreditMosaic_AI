"""
Risk Model Service
Service layer that integrates feature engineering, model inference, and risk scoring.

Provides methods for:
  - Generating risk labels from market data
  - Training models on historical data
  - Scoring companies and retrieving latest risk scores
  - Explaining risk drivers
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskModelService:
    """Unified service for risk model training and inference."""

    def __init__(self, db_connection, model_dir: str = "models"):
        self.db = db_connection
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def generate_risk_labels(
        self,
        tickers: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Generate and save multi-label risk indicators."""
        from pipelines.risk.risk_labeler import RiskLabeler

        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365)

        labeler = RiskLabeler(self.db)
        labels = labeler.generate_all_labels(tickers, start_date, end_date)
        labeler.save_labels_to_db(labels)
        return labels

    def train_models(
        self,
        tickers: List[str],
        end_date: Optional[datetime] = None,
        n_splits: int = 5
    ) -> Dict[str, Any]:
        """Train all risk models (LR, LightGBM, XGBoost)."""
        from pipelines.risk.feature_engineer import FeatureEngineer
        from pipelines.risk.model_trainer import RiskModelTrainer

        if end_date is None:
            end_date = datetime.now()

        engineer = FeatureEngineer(self.db)
        X, y, _ = engineer.build_feature_matrix(tickers, end_date)

        if X.empty:
            logger.error("No features available for training")
            return {'error': 'No features available'}

        trainer = RiskModelTrainer(str(self.model_dir))
        results = trainer.train_all(X, y, n_splits)

        return results

    def score_companies(
        self,
        tickers: List[str],
        model_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Score companies using the best available model."""
        from pipelines.risk.feature_engineer import FeatureEngineer
        from pipelines.risk.risk_scorer import RiskScorer

        engineer = FeatureEngineer(self.db)
        X, meta = engineer.build_prediction_features(tickers, datetime.now())

        if X.empty:
            logger.warning("No features for scoring")
            return pd.DataFrame()

        scorer = RiskScorer(self.db, str(self.model_dir))
        scores = scorer.score_companies(
            X,
            tickers=meta['ticker'] if 'ticker' in meta.columns else None,
            dates=meta['date'] if 'date' in meta.columns else None,
            model_name=model_name
        )
        scorer.save_to_db(scores)
        return scores

    def get_risk_summary(self, tickers: List[str]) -> List[Dict[str, Any]]:
        """Get latest risk scores with top-5 drivers for given tickers."""
        from pipelines.risk.risk_scorer import RiskScorer

        scorer = RiskScorer(self.db, str(self.model_dir))
        return scorer.get_latest_scores(tickers)

    def get_company_risk_history(
        self,
        ticker: str,
        days: int = 90
    ) -> List[Dict[str, Any]]:
        """Get risk score history for a single company."""
        sql = """
            SELECT ticker, date, risk_score, risk_level, model_version, top_features
            FROM risk_scores
            WHERE ticker = %s AND date >= %s
            ORDER BY date DESC
        """
        with self.db.cursor() as cur:
            cur.execute(sql, (ticker, (datetime.now() - timedelta(days=days)).date()))
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            return [dict(zip(columns, row)) for row in rows]

    def get_model_evaluation(self) -> Optional[Dict[str, Any]]:
        """Get the latest model evaluation report."""
        import json
        path = self.model_dir / "evaluation_report.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None
