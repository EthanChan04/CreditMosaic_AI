"""
Risk Model Trainer
Trains Logistic Regression, LightGBM, and XGBoost models for credit risk prediction.

Uses walk-forward (expanding-window) cross-validation to respect temporal ordering.
Evaluates with AUC, Precision@K, Recall@K, and F1 metrics.
Extracts top-5 feature importance via SHAP for tree models and coefficients for LR.
"""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class RiskModelTrainer:
    """Train and evaluate ML models for credit risk prediction."""

    def __init__(self, model_dir: str = "models", random_state: int = 42):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.models = {}
        self.feature_names = []
        self.evaluation_results = {}

    def train_all(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.Series = None,
        n_splits: int = 5
    ) -> Dict[str, Any]:
        """Train Logistic Regression, LightGBM, XGBoost with walk-forward CV.

        Args:
            X: Feature matrix (DataFrame with descriptive column names).
            y: Binary target vector.
            dates: Date series for each row, used for temporal walk-forward splits.
                   If None, falls back to index-based splitting (not recommended).
            n_splits: Number of walk-forward folds.
        """
        self.feature_names = list(X.columns)

        # Verify feature names are descriptive, not generic numbered placeholders
        generic_names = [f for f in self.feature_names if f.startswith('feature_') and f.split('_')[-1].isdigit()]
        if generic_names:
            raise ValueError(
                f"Feature names contain generic placeholders: {generic_names[:5]}. "
                "This indicates stale feature engineering code. All features must have descriptive names."
            )

        # Store training metadata for the evaluation report
        self._training_meta = {
            'total_samples': len(X),
            'positive_rate': float(y.mean()) if hasattr(y, 'mean') else 0,
            'n_features': len(self.feature_names),
            'n_folds': n_splits,
        }
        if dates is not None:
            dates_arr = pd.to_datetime(dates)
            self._training_meta['date_range'] = [
                str(dates_arr.min().date()),
                str(dates_arr.max().date())
            ]
            n_tickers = len(set(dates.index)) if hasattr(dates, 'index') else 'unknown'
            self._training_meta['n_tickers'] = n_tickers

        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y
        dates_arr = dates.values if dates is not None else None

        if len(np.unique(y_arr)) < 2:
            raise ValueError("Risk model training requires at least two target classes")

        results = {}

        logger.info("Training Logistic Regression...")
        lr_model, lr_eval = self._train_lr(X_arr, y_arr, dates_arr, n_splits)
        results['logistic_regression'] = lr_eval
        self.models['logistic_regression'] = lr_model

        logger.info("Training LightGBM...")
        lgb_model, lgb_eval = self._train_lightgbm(X_arr, y_arr, dates_arr, n_splits)
        results['lightgbm'] = lgb_eval
        self.models['lightgbm'] = lgb_model

        logger.info("Training XGBoost...")
        xgb_model, xgb_eval = self._train_xgboost(X_arr, y_arr, dates_arr, n_splits)
        results['xgboost'] = xgb_eval
        self.models['xgboost'] = xgb_model

        self.evaluation_results = results
        self._save_models()
        self._save_evaluation_report()

        return results

    def _walk_forward_splits(self, X, y, dates, n_splits: int):
        """Generate walk-forward train/test splits.

        When dates are provided, splits by date threshold so all tickers
        share the same temporal boundary (prevents cross-ticker leakage).
        Otherwise falls back to index-based splitting.
        """
        n_samples = len(y)

        if dates is not None:
            unique_dates = np.sort(np.unique(dates))
            split_size = len(unique_dates) // (n_splits + 1)
            if split_size < 1:
                split_size = 1

            for i in range(1, n_splits + 1):
                cutoff_idx = min(split_size * i, len(unique_dates) - 1)
                cutoff_date = unique_dates[cutoff_idx]
                train_mask = dates < cutoff_date

                test_end_idx = min(split_size * (i + 1), len(unique_dates) - 1)
                test_end_date = unique_dates[test_end_idx]
                test_mask = (dates >= cutoff_date) & (dates < test_end_date)

                if train_mask.sum() < 10 or test_mask.sum() < 5:
                    continue

                yield train_mask, test_mask
        else:
            # Fallback: index-based splitting (assumes data is sorted by time)
            split_size = n_samples // (n_splits + 1)
            for i in range(1, n_splits + 1):
                train_end = split_size * i
                test_start = train_end
                test_end = min(test_start + split_size, n_samples)

                train_mask = np.arange(n_samples) < train_end
                test_mask = (np.arange(n_samples) >= test_start) & (np.arange(n_samples) < test_end)
                yield train_mask, test_mask

    def _train_lr(
        self, X: np.ndarray, y: np.ndarray, dates, n_splits: int
    ) -> Tuple[Any, Dict]:
        from sklearn.linear_model import LogisticRegression

        metrics = {'auc': [], 'precision_at_k': [], 'recall_at_k': [], 'f1': [], 'brier_score': []}
        k = max(10, len(y) // (n_splits * 20))

        for train_mask, test_mask in self._walk_forward_splits(X, y, dates, n_splits):
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            if len(np.unique(y_train)) < 2:
                continue

            model = LogisticRegression(
                max_iter=2000, random_state=self.random_state, class_weight='balanced'
            )
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]

            self._record_metrics(metrics, y_test, probs, k)

        final_model = LogisticRegression(
            max_iter=2000, random_state=self.random_state, class_weight='balanced'
        )
        final_model.fit(X, y)

        return final_model, {key: np.mean(vals) if vals else 0 for key, vals in metrics.items()}

    def _train_lightgbm(
        self, X: np.ndarray, y: np.ndarray, dates, n_splits: int
    ) -> Tuple[Any, Dict]:
        try:
            import lightgbm as lgb
        except ImportError:
            logger.warning("LightGBM not installed; skipping")
            return None, {'auc': 0, 'precision_at_k': 0, 'recall_at_k': 0, 'f1': 0}

        metrics = {'auc': [], 'precision_at_k': [], 'recall_at_k': [], 'f1': [], 'brier_score': []}
        k = max(10, len(y) // (n_splits * 20))
        scale_pos_weight = (len(y) - sum(y)) / max(sum(y), 1)

        for train_mask, test_mask in self._walk_forward_splits(X, y, dates, n_splits):
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            if len(np.unique(y_train)) < 2:
                continue

            model = lgb.LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                random_state=self.random_state, scale_pos_weight=scale_pos_weight,
                verbosity=-1
            )
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]

            self._record_metrics(metrics, y_test, probs, k)

        final_model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            random_state=self.random_state, scale_pos_weight=scale_pos_weight,
            verbosity=-1
        )
        final_model.fit(X, y)

        return final_model, {key: np.mean(vals) if vals else 0 for key, vals in metrics.items()}

    def _train_xgboost(
        self, X: np.ndarray, y: np.ndarray, dates, n_splits: int
    ) -> Tuple[Any, Dict]:
        try:
            import xgboost as xgb
        except ImportError:
            logger.warning("XGBoost not installed; skipping")
            return None, {'auc': 0, 'precision_at_k': 0, 'recall_at_k': 0, 'f1': 0}

        metrics = {'auc': [], 'precision_at_k': [], 'recall_at_k': [], 'f1': [], 'brier_score': []}
        k = max(10, len(y) // (n_splits * 20))
        scale_pos_weight = (len(y) - sum(y)) / max(sum(y), 1)

        for train_mask, test_mask in self._walk_forward_splits(X, y, dates, n_splits):
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            if len(np.unique(y_train)) < 2:
                continue

            model = xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                random_state=self.random_state, scale_pos_weight=scale_pos_weight,
                verbosity=0
            )
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]

            self._record_metrics(metrics, y_test, probs, k)

        final_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            random_state=self.random_state, scale_pos_weight=scale_pos_weight,
            verbosity=0
        )
        final_model.fit(X, y)

        return final_model, {key: np.mean(vals) if vals else 0 for key, vals in metrics.items()}

    def _record_metrics(self, metrics: Dict, y_true: np.ndarray, probs: np.ndarray, k: int):
        from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss

        try:
            metrics['auc'].append(roc_auc_score(y_true, probs))
        except ValueError:
            pass

        try:
            metrics['f1'].append(f1_score(y_true, (probs >= 0.5).astype(int)))
        except ValueError:
            pass

        try:
            metrics['brier_score'].append(brier_score_loss(y_true, probs))
        except ValueError:
            pass

        top_k_idx = np.argsort(probs)[-k:]
        y_pred_topk = np.zeros_like(y_true)
        y_pred_topk[top_k_idx] = 1

        true_positives = y_true[y_pred_topk == 1].sum()
        metrics['precision_at_k'].append(
            true_positives / k if k > 0 else 0
        )
        metrics['recall_at_k'].append(
            true_positives / max(y_true.sum(), 1)
        )

    def get_top_features(self, model_name: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """Get top-N feature importance for a trained model."""
        if model_name not in self.models or self.models[model_name] is None:
            return []

        model = self.models[model_name]

        if model_name == 'logistic_regression':
            importances = np.abs(model.coef_[0])
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            return []

        indices = np.argsort(importances)[::-1][:top_n]
        return [
            {
                'feature': self.feature_names[i] if i < len(self.feature_names) else f'f{i}',
                'importance': round(float(importances[i]), 6),
                'rank': rank + 1
            }
            for rank, i in enumerate(indices)
        ]

    def get_shap_explanation(self, model_name: str, X_sample: np.ndarray) -> Optional[Dict]:
        """Compute SHAP values for top feature drivers."""
        if model_name not in self.models or self.models[model_name] is None:
            return None

        if model_name == 'logistic_regression':
            coef = self.models[model_name].coef_[0]
            top_idx = np.argsort(np.abs(coef))[::-1][:5]
            return {
                'method': 'coefficients',
                'top_drivers': [
                    {
                        'feature': self.feature_names[i] if i < len(self.feature_names) else f'f{i}',
                        'coefficient': round(float(coef[i]), 6),
                        'contribution': round(float(coef[i] * X_sample[0, i]) if X_sample.shape[0] > 0 else 0, 6)
                    }
                    for i in top_idx
                ]
            }

        try:
            import shap
            if hasattr(self.models[model_name], 'predict_proba'):
                explainer = shap.TreeExplainer(self.models[model_name])
                sample = X_sample[:min(100, len(X_sample))]
                shap_values = explainer.shap_values(sample)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]

                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                top_idx = np.argsort(mean_abs_shap)[::-1][:5]

                return {
                    'method': 'shap',
                    'top_drivers': [
                        {
                            'feature': self.feature_names[i] if i < len(self.feature_names) else f'f{i}',
                            'shap_importance': round(float(mean_abs_shap[i]), 6),
                        }
                        for i in top_idx
                    ]
                }
        except Exception as e:
            logger.warning(f"SHAP computation failed for {model_name}: {e}")
            return None

    def _save_models(self):
        import pickle
        for name, model in self.models.items():
            if model is not None:
                path = self.model_dir / f"{name}.pkl"
                with open(path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Saved {name} to {path}")

    def _save_evaluation_report(self):
        report = {
            'timestamp': datetime.now().isoformat(),
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'training_metadata': getattr(self, '_training_meta', {}),
            'models': {}
        }

        for name, eval_metrics in self.evaluation_results.items():
            if eval_metrics is None:
                report['models'][name] = {'status': 'skipped'}
            else:
                report['models'][name] = {
                    'metrics': eval_metrics,
                    'top_features': self.get_top_features(name)
                }

        path = self.model_dir / "evaluation_report.json"
        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Saved evaluation report to {path}")

    def load_model(self, model_name: str):
        import pickle
        path = self.model_dir / f"{model_name}.pkl"
        if path.exists():
            with open(path, 'rb') as f:
                self.models[model_name] = pickle.load(f)
            return self.models[model_name]
        return None
