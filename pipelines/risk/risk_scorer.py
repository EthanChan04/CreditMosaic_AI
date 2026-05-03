"""
Risk Scorer
Produces daily company risk scores using trained models and explains top-5 drivers.

Reads from PostgreSQL, scores with the best available model, and writes risk_scores
back to the database with risk_level and top_features (JSONB driver list).
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskScorer:
    """Daily risk scoring engine using trained ML models."""

    RISK_LEVELS = {
        'Low': (0.0, 0.25),
        'Medium': (0.25, 0.50),
        'High': (0.50, 0.75),
        'Critical': (0.75, 1.01),
    }

    def __init__(self, db_connection, model_dir: str = "models"):
        self.db = db_connection
        self.model_dir = Path(model_dir)
        self.trainer = None

    def _load_trainer(self):
        from pipelines.risk.model_trainer import RiskModelTrainer
        self.trainer = RiskModelTrainer(str(self.model_dir))
        self.trainer.feature_names = self._load_feature_names()

    def _load_feature_names(self) -> List[str]:
        report_path = self.model_dir / "evaluation_report.json"
        if report_path.exists():
            import json
            with open(report_path) as f:
                report = json.load(f)
            names = report.get('feature_names', [])
            if names:
                return names
        self.trainer.feature_names = []
        return []

    def _select_best_model(self) -> Optional[str]:
        if self.trainer is None:
            self._load_trainer()

        best_name = None
        best_auc = -1

        for name in ['lightgbm', 'xgboost', 'logistic_regression']:
            model = self.trainer.load_model(name)
            if model is not None:
                if name in self.trainer.evaluation_results:
                    auc = self.trainer.evaluation_results[name].get('auc', 0)
                elif (self.model_dir / "evaluation_report.json").exists():
                    import json
                    with open(self.model_dir / "evaluation_report.json") as f:
                        report = json.load(f)
                    auc = report.get('models', {}).get(name, {}).get('metrics', {}).get('auc', 0)
                else:
                    auc = 0

                if auc > best_auc:
                    best_auc = auc
                    best_name = name

        return best_name

    def score_companies(
        self,
        X: pd.DataFrame,
        tickers: pd.Series = None,
        dates: pd.Series = None,
        model_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Score companies and return DataFrame with risk_score, risk_level, top_features."""
        if model_name is None:
            model_name = self._select_best_model()

        if model_name is None or self.trainer is None:
            logger.error("No trained model available for scoring")
            return pd.DataFrame()

        model = self.trainer.models.get(model_name)
        if model is None:
            model = self.trainer.load_model(model_name)
        if model is None:
            logger.error(f"Model {model_name} could not be loaded")
            return pd.DataFrame()

        feature_names = self.trainer.feature_names
        if feature_names:
            X = X.reindex(columns=feature_names, fill_value=0)
        else:
            X = X.select_dtypes(include=[np.number])
            expected = getattr(model, "n_features_in_", None)
            if expected is not None and X.shape[1] != expected:
                logger.warning(
                    "Model was trained with %s features but scoring data has %s; "
                    "using positional fallback because feature_names are missing",
                    expected, X.shape[1]
                )
                if X.shape[1] > expected:
                    X = X.iloc[:, :expected]
                else:
                    for i in range(X.shape[1], expected):
                        X[f"missing_feature_{i}"] = 0
            self.trainer.feature_names = list(X.columns)

        X_filled = X.fillna(0).replace([np.inf, -np.inf], 0)
        probs = model.predict_proba(X_filled.values)[:, 1]

        results = pd.DataFrame({
            'risk_score': probs,
            'risk_level': [self._score_to_level(p) for p in probs],
            'model_version': model_name,
        })

        if tickers is not None:
            results['ticker'] = tickers.values
        if dates is not None:
            results['date'] = dates.values

        top_features = self._compute_top_features(model, model_name, X_filled.values)
        results['top_features'] = top_features if top_features else [None] * len(results)

        return results

    def _score_to_level(self, score: float) -> str:
        for level, (low, high) in self.RISK_LEVELS.items():
            if low <= score < high:
                return level
        return 'Critical' if score >= 0.75 else 'Low'

    def _compute_top_features(
        self,
        model,
        model_name: str,
        X: np.ndarray
    ) -> Optional[List[Dict]]:
        """Get top-5 feature drivers for each sample."""
        feature_names = self.trainer.feature_names
        if not feature_names:
            return None

        if model_name == 'logistic_regression':
            coef = model.coef_[0]
            top_indices = np.argsort(np.abs(coef))[::-1][:5]
            per_sample = []
            for i in range(len(X)):
                drivers = []
                for idx in top_indices:
                    feat_name = feature_names[idx] if idx < len(feature_names) else f'f{idx}'
                    drivers.append({
                        'feature': feat_name,
                        'importance': round(float(np.abs(coef[idx])), 6)
                    })
                per_sample.append(drivers)
            return per_sample

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[::-1][:5]
            per_sample = []
            for i in range(len(X)):
                drivers = []
                for idx in top_indices:
                    feat_name = feature_names[idx] if idx < len(feature_names) else f'f{idx}'
                    drivers.append({
                        'feature': feat_name,
                        'importance': round(float(importances[idx]), 6)
                    })
                per_sample.append(drivers)
            return per_sample

        return None

    def save_to_db(self, scores: pd.DataFrame):
        """Save risk scores to PostgreSQL."""
        if scores.empty:
            return

        import json
        with self.db.cursor() as cur:
            for _, row in scores.iterrows():
                top_features_json = json.dumps(row.get('top_features', [])) if row.get('top_features') is not None else None
                cur.execute("""
                    INSERT INTO risk_scores (
                        ticker, date, risk_score, risk_level, model_version, top_features
                    ) VALUES (
                        %(ticker)s, %(date)s, %(risk_score)s, %(risk_level)s, %(model_version)s, %(top_features)s
                    )
                    ON CONFLICT (ticker, date, model_version) DO UPDATE SET
                        risk_score = EXCLUDED.risk_score,
                        risk_level = EXCLUDED.risk_level,
                        top_features = EXCLUDED.top_features
                """, {
                    'ticker': row['ticker'],
                    'date': row['date'].date() if hasattr(row['date'], 'date') else row['date'],
                    'risk_score': float(row['risk_score']),
                    'risk_level': row['risk_level'],
                    'model_version': row['model_version'],
                    'top_features': top_features_json,
                })
            self.db.commit()

        logger.info(f"Saved {len(scores)} risk scores to database")

    def get_latest_scores(self, tickers: List[str]) -> List[Dict[str, Any]]:
        """Retrieve the most recent risk scores for given tickers."""
        sql = """
            SELECT DISTINCT ON (ticker) ticker, date, risk_score, risk_level,
                   model_version, top_features
            FROM risk_scores
            WHERE ticker = ANY(%s)
            ORDER BY ticker, date DESC
        """
        with self.db.cursor() as cur:
            cur.execute(sql, (tickers,))
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            return [dict(zip(columns, row)) for row in rows]
