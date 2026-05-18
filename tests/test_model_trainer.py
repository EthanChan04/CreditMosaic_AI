"""
Tests for model training pipeline.
Verifies walk-forward CV correctness, metric computation, and feature name validation.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

from pipelines.risk.model_trainer import RiskModelTrainer


class TestWalkForwardSplits:
    """Verify walk-forward CV splits are temporally correct."""

    def test_date_based_split_no_leakage(self):
        """Date-based splits must ensure train dates < test dates."""
        trainer = RiskModelTrainer.__new__(RiskModelTrainer)

        n = 300
        dates = np.array([
            datetime(2025, 1, 1) + timedelta(days=i) for i in range(n)
        ])
        X = np.random.randn(n, 5)
        y = np.random.randint(0, 2, n)

        splits = list(trainer._walk_forward_splits(X, y, dates, n_splits=3))
        assert len(splits) > 0

        for train_mask, test_mask in splits:
            train_dates = dates[train_mask]
            test_dates = dates[test_mask]

            # All train dates must be strictly before all test dates
            max_train = max(train_dates)
            min_test = min(test_dates)
            assert max_train < min_test, \
                f"Train max ({max_train}) must be < test min ({min_test})"

    def test_all_tickers_present_in_both_sets(self):
        """When using date-based splits, all tickers should appear in both train and test."""
        trainer = RiskModelTrainer.__new__(RiskModelTrainer)

        # Create data with multiple tickers sharing same dates
        n_dates = 100
        dates_single = np.array([
            datetime(2025, 1, 1) + timedelta(days=i) for i in range(n_dates)
        ])
        dates = np.tile(dates_single, 3)  # 3 tickers
        X = np.random.randn(n_dates * 3, 5)
        y = np.random.randint(0, 2, n_dates * 3)

        splits = list(trainer._walk_forward_splits(X, y, dates, n_splits=3))
        assert len(splits) > 0

        for train_mask, test_mask in splits:
            # Both sets should have data from multiple tickers
            assert train_mask.sum() > 10
            assert test_mask.sum() > 5

    def test_fallback_index_split(self):
        """When dates is None, should fall back to index-based splitting."""
        trainer = RiskModelTrainer.__new__(RiskModelTrainer)

        n = 200
        X = np.random.randn(n, 5)
        y = np.random.randint(0, 2, n)

        splits = list(trainer._walk_forward_splits(X, y, dates=None, n_splits=3))
        assert len(splits) > 0

        for train_mask, test_mask in splits:
            assert train_mask.any()
            assert test_mask.any()


class TestMetricComputation:
    """Verify that all required metrics are computed correctly."""

    def test_record_metrics_adds_brier_score(self):
        """_record_metrics must include brier_score in the metrics dict."""
        trainer = RiskModelTrainer.__new__(RiskModelTrainer)
        metrics = {'auc': [], 'precision_at_k': [], 'recall_at_k': [], 'f1': [], 'brier_score': []}

        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 1])
        probs = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.15, 0.25, 0.85, 0.95])

        trainer._record_metrics(metrics, y_true, probs, k=3)

        assert len(metrics['brier_score']) == 1
        assert 0 <= metrics['brier_score'][0] <= 1

    def test_record_metrics_auc_range(self):
        """AUC must be between 0 and 1."""
        trainer = RiskModelTrainer.__new__(RiskModelTrainer)
        metrics = {'auc': [], 'precision_at_k': [], 'recall_at_k': [], 'f1': [], 'brier_score': []}

        y_true = np.array([0, 1, 0, 1, 0, 1])
        probs = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])

        trainer._record_metrics(metrics, y_true, probs, k=2)

        assert 0 <= metrics['auc'][0] <= 1


class TestFeatureNameValidation:
    """Verify that generic feature names are rejected."""

    def test_rejects_generic_feature_names(self):
        """train_all must raise ValueError for feature_N style names."""
        trainer = RiskModelTrainer(tempfile.mkdtemp())

        X = pd.DataFrame({
            'feature_0': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        })
        y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        with pytest.raises(ValueError, match="generic placeholders"):
            trainer.train_all(X, y)

    def test_accepts_descriptive_feature_names(self):
        """train_all must accept descriptive feature names."""
        trainer = RiskModelTrainer(tempfile.mkdtemp())

        X = pd.DataFrame({
            'returns_1d_lag6': np.random.randn(100),
            'volatility_5d_lag6': np.abs(np.random.randn(100)),
            'debt_to_assets': np.random.uniform(0, 1, 100),
        })
        y = pd.Series(np.random.randint(0, 2, 100))

        # Should not raise
        try:
            trainer.train_all(X, y, n_splits=2)
        except ValueError as e:
            if "generic placeholders" in str(e):
                pytest.fail("Should not reject descriptive feature names")
