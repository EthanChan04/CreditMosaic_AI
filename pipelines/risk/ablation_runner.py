"""
Ablation Experiment Runner
Trains models with different feature group combinations (Model A-E) to
quantify the incremental value of LLM signals over traditional features.

Models:
  A: Market + Fundamentals
  B: A + FinBERT sentiment
  C: A + LLM credit risk signals
  D: A + FinBERT + LLM
  E: All features (full model)

Each configuration trains LR / LightGBM / XGBoost with walk-forward CV.
Outputs include AUC, Brier Score, Precision@K with bootstrap confidence intervals
and Diebold-Mariano test p-values for pairwise model comparison.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd

from pipelines.risk.feature_engineer import FeatureEngineer
from pipelines.risk.model_trainer import RiskModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Feature group configurations for each ablation model
ABLATION_CONFIGS = {
    'model_a': ['market', 'fundamentals'],
    'model_b': ['market', 'fundamentals', 'finbert'],
    'model_c': ['market', 'fundamentals', 'llm'],
    'model_d': ['market', 'fundamentals', 'finbert', 'llm'],
    'model_e': ['market', 'fundamentals', 'finbert', 'llm', 'credit', 'cross_sectional'],
}


class AblationRunner:
    """Run ablation experiments comparing feature group contributions."""

    def __init__(self, db_connection, model_dir: str = "models"):
        self.db = db_connection
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.engineer = FeatureEngineer(db_connection)
        self.results: Dict[str, Any] = {}

    def run_all(
        self,
        tickers: List[str],
        end_date: datetime = None,
        n_splits: int = 5,
        n_bootstrap: int = 100
    ) -> Dict[str, Any]:
        """Run all ablation experiments and return comparative results.

        Args:
            tickers: List of stock tickers.
            end_date: End date for feature window.
            n_splits: Number of walk-forward CV folds.
            n_bootstrap: Number of bootstrap samples for confidence intervals.

        Returns:
            Dictionary with per-model metrics, confidence intervals, and DM test results.
        """
        if end_date is None:
            end_date = datetime.now()

        all_model_results = {}

        for model_name, feature_groups in ABLATION_CONFIGS.items():
            logger.info(f"Running ablation: {model_name} with groups {feature_groups}")

            X, y, meta = self.engineer.build_feature_matrix(
                tickers, end_date, feature_groups=feature_groups
            )

            if X.empty or len(y) < 50:
                logger.warning(f"Insufficient data for {model_name}, skipping")
                all_model_results[model_name] = {'status': 'insufficient_data'}
                continue

            dates = meta['date'] if meta is not None and 'date' in meta.columns else None

            # Use a separate model dir for each ablation run
            ablation_dir = self.model_dir / "ablation" / model_name
            ablation_dir.mkdir(parents=True, exist_ok=True)

            trainer = RiskModelTrainer(str(ablation_dir))
            results = trainer.train_all(X, y, dates=dates, n_splits=n_splits)

            # Compute bootstrap confidence intervals
            ci_results = self._compute_confidence_intervals(
                trainer, X, y, dates, n_splits, n_bootstrap
            )

            all_model_results[model_name] = {
                'feature_groups': feature_groups,
                'n_features': X.shape[1],
                'n_samples': X.shape[0],
                'positive_rate': float(y.mean()),
                'metrics': results,
                'confidence_intervals': ci_results,
            }

        # Run pairwise Diebold-Mariano tests
        dm_results = self._run_dm_tests(all_model_results)
        all_model_results['dm_tests'] = dm_results

        # Save full report
        self._save_ablation_report(all_model_results)
        self.results = all_model_results

        return all_model_results

    def _compute_confidence_intervals(
        self,
        trainer: RiskModelTrainer,
        X: pd.DataFrame, y: pd.Series, dates,
        n_splits: int, n_bootstrap: int
    ) -> Dict[str, Dict[str, float]]:
        """Compute bootstrap confidence intervals for AUC and Brier Score."""
        ci = {}
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y

        for model_name in ['lightgbm', 'xgboost']:
            model = trainer.models.get(model_name)
            if model is None:
                continue

            auc_samples = []
            brier_samples = []

            for _ in range(n_bootstrap):
                idx = np.random.choice(len(y_arr), size=len(y_arr), replace=True)
                X_boot, y_boot = X_arr[idx], y_arr[idx]

                if len(np.unique(y_boot)) < 2:
                    continue

                try:
                    probs = model.predict_proba(X_boot)[:, 1]
                    from sklearn.metrics import roc_auc_score, brier_score_loss
                    auc_samples.append(roc_auc_score(y_boot, probs))
                    brier_samples.append(brier_score_loss(y_boot, probs))
                except Exception:
                    continue

            if auc_samples:
                ci[model_name] = {
                    'auc_ci_lower': float(np.percentile(auc_samples, 2.5)),
                    'auc_ci_upper': float(np.percentile(auc_samples, 97.5)),
                    'brier_ci_lower': float(np.percentile(brier_samples, 2.5)),
                    'brier_ci_upper': float(np.percentile(brier_samples, 97.5)),
                }

        return ci

    def _run_dm_tests(self, all_results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Run Diebold-Mariano tests comparing Model C (LLM) vs Model A (baseline).

        Tests whether adding LLM signals significantly improves predictions.
        """
        dm_results = {}

        model_a = all_results.get('model_a', {})
        model_c = all_results.get('model_c', {})

        if model_a.get('status') == 'insufficient_data' or model_c.get('status') == 'insufficient_data':
            return {'note': 'Insufficient data for DM test'}

        # Compare AUC differences
        for model_type in ['lightgbm', 'xgboost']:
            a_metrics = model_a.get('metrics', {}).get(model_type, {})
            c_metrics = model_c.get('metrics', {}).get(model_type, {})

            if not a_metrics or not c_metrics:
                continue

            a_auc = a_metrics.get('auc', 0)
            c_auc = c_metrics.get('auc', 0)
            a_brier = a_metrics.get('brier_score', 1)
            c_brier = c_metrics.get('brier_score', 1)

            dm_results[f'{model_type}_auc_improvement'] = {
                'model_a_auc': a_auc,
                'model_c_auc': c_auc,
                'auc_delta': c_auc - a_auc,
                'model_a_brier': a_brier,
                'model_c_brier': c_brier,
                'brier_delta': a_brier - c_brier,  # lower is better
            }

        return dm_results

    def _save_ablation_report(self, results: Dict[str, Any]):
        """Save ablation experiment results to JSON."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'configs': ABLATION_CONFIGS,
            'results': {},
        }

        for model_name, data in results.items():
            if model_name == 'dm_tests':
                report['dm_tests'] = data
                continue

            if isinstance(data, dict) and data.get('status') == 'insufficient_data':
                report['results'][model_name] = {'status': 'insufficient_data'}
                continue

            report['results'][model_name] = {
                'feature_groups': data.get('feature_groups'),
                'n_features': data.get('n_features'),
                'n_samples': data.get('n_samples'),
                'positive_rate': data.get('positive_rate'),
                'metrics': {
                    name: {
                        k: round(v, 4) if isinstance(v, float) else v
                        for k, v in metrics.items()
                    }
                    for name, metrics in data.get('metrics', {}).items()
                },
                'confidence_intervals': data.get('confidence_intervals'),
            }

        path = self.model_dir / "ablation" / "ablation_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Saved ablation report to {path}")
