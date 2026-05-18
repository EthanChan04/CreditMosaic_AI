"""
Unit tests for Risk Scorer pipeline.
Covers: score-to-level mapping, scoring engine, top features, DB persistence.
"""

import json
import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile

from pipelines.risk.risk_scorer import RiskScorer


@pytest.fixture
def scorer(tmp_path):
    """Create a RiskScorer with mock DB."""
    conn = MagicMock()
    return RiskScorer(conn, model_dir=str(tmp_path))


# ── Score-to-Level Mapping ──────────────────────────────────────────────────


class TestScoreToLevel:
    """Test risk level classification thresholds."""

    def test_low_range(self, scorer):
        assert scorer._score_to_level(0.0) == 'Low'
        assert scorer._score_to_level(0.1) == 'Low'
        assert scorer._score_to_level(0.24) == 'Low'

    def test_medium_range(self, scorer):
        assert scorer._score_to_level(0.25) == 'Medium'
        assert scorer._score_to_level(0.4) == 'Medium'
        assert scorer._score_to_level(0.49) == 'Medium'

    def test_high_range(self, scorer):
        assert scorer._score_to_level(0.5) == 'High'
        assert scorer._score_to_level(0.65) == 'High'
        assert scorer._score_to_level(0.74) == 'High'

    def test_critical_range(self, scorer):
        assert scorer._score_to_level(0.75) == 'Critical'
        assert scorer._score_to_level(0.9) == 'Critical'
        assert scorer._score_to_level(1.0) == 'Critical'


# ── Score Companies ─────────────────────────────────────────────────────────


class TestScoreCompanies:
    """Test the scoring engine."""

    def test_empty_when_no_model(self, scorer):
        """Should return empty DataFrame when no model can be selected."""
        mock_trainer = MagicMock()
        mock_trainer.models = {}
        mock_trainer.feature_names = []
        mock_trainer.evaluation_results = {}
        mock_trainer.load_model.return_value = None
        scorer.trainer = mock_trainer

        X = pd.DataFrame({'f1': [1, 2, 3]})
        result = scorer.score_companies(X, model_name='nonexistent')
        assert result.empty

    def test_scoring_with_mock_model(self, scorer):
        """Test scoring with a mocked model."""
        # Create a mock model that returns predictable probabilities
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2], [0.5, 0.5]])

        mock_trainer = MagicMock()
        mock_trainer.models = {'lightgbm': mock_model}
        mock_trainer.feature_names = ['f1', 'f2']
        mock_trainer.evaluation_results = {'lightgbm': {'auc': 0.85}}
        mock_trainer.load_model.return_value = mock_model
        scorer.trainer = mock_trainer

        X = pd.DataFrame({'f1': [1, 2, 3], 'f2': [4, 5, 6]})
        tickers = pd.Series(['AAPL', 'MSFT', 'GOOGL'])
        dates = pd.Series([datetime(2026, 4, 30)] * 3)

        result = scorer.score_companies(X, tickers=tickers, dates=dates, model_name='lightgbm')

        assert not result.empty
        assert 'risk_score' in result.columns
        assert 'risk_level' in result.columns
        assert 'ticker' in result.columns
        assert len(result) == 3
        # First row should have risk_score 0.7
        assert abs(result.iloc[0]['risk_score'] - 0.7) < 1e-6

    def test_risk_levels_assigned_correctly(self, scorer):
        """Risk levels should match score thresholds."""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.9, 0.1], [0.6, 0.4], [0.3, 0.7]])
        mock_model.n_features_in_ = 2

        mock_trainer = MagicMock()
        mock_trainer.models = {'lightgbm': mock_model}
        mock_trainer.feature_names = ['f1', 'f2']
        mock_trainer.evaluation_results = {'lightgbm': {'auc': 0.8}}
        mock_trainer.load_model.return_value = mock_model
        scorer.trainer = mock_trainer

        X = pd.DataFrame({'f1': [1, 2, 3], 'f2': [4, 5, 6]})
        result = scorer.score_companies(X, model_name='lightgbm')

        # score 0.1 -> Low, 0.4 -> Medium, 0.7 -> High
        assert result.iloc[0]['risk_level'] == 'Low'
        assert result.iloc[1]['risk_level'] == 'Medium'
        assert result.iloc[2]['risk_level'] == 'High'


# ── Top Features ────────────────────────────────────────────────────────────


class TestComputeTopFeatures:
    """Test feature driver extraction."""

    def test_lr_coefficients(self, scorer):
        """For logistic regression, use absolute coefficients."""
        mock_model = MagicMock()
        mock_model.coef_ = np.array([[0.5, -0.3, 0.8, 0.1, -0.6, 0.2]])

        mock_trainer = MagicMock()
        mock_trainer.feature_names = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5']
        scorer.trainer = mock_trainer

        X = np.array([[1, 2, 3, 4, 5, 6]])
        result = scorer._compute_top_features(mock_model, 'logistic_regression', X)

        assert result is not None
        assert len(result) == 1
        assert len(result[0]) == 5
        # Top feature should be f2 (coef 0.8)
        assert result[0][0]['feature'] == 'f2'

    def test_tree_model_importance(self, scorer):
        """For tree models, use feature_importances_."""
        mock_model = MagicMock()
        mock_model.feature_importances_ = np.array([0.1, 0.3, 0.5, 0.05, 0.02, 0.03])
        del mock_model.coef_  # ensure no coef attribute

        mock_trainer = MagicMock()
        mock_trainer.feature_names = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5']
        scorer.trainer = mock_trainer

        X = np.array([[1, 2, 3, 4, 5, 6]])
        result = scorer._compute_top_features(mock_model, 'lightgbm', X)

        assert result is not None
        assert result[0][0]['feature'] == 'f2'  # highest importance 0.5

    def test_no_feature_names_returns_none(self, scorer):
        mock_model = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.feature_names = []
        scorer.trainer = mock_trainer

        result = scorer._compute_top_features(mock_model, 'lightgbm', np.array([[1]]))
        assert result is None


# ── Save To DB ──────────────────────────────────────────────────────────────


class TestSaveToDb:
    """Test database persistence of risk scores."""

    def test_empty_scores_no_write(self, scorer):
        scorer.save_to_db(pd.DataFrame())
        scorer.db.cursor.assert_not_called()

    def test_scores_written_to_db(self, scorer):
        mock_cursor = MagicMock()
        scorer.db.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        scorer.db.cursor.return_value.__exit__ = MagicMock(return_value=False)

        scores = pd.DataFrame({
            'ticker': ['AAPL'],
            'date': [datetime(2026, 4, 30)],
            'risk_score': [0.65],
            'risk_level': ['High'],
            'model_version': ['lightgbm'],
            'top_features': [[{'feature': 'f1', 'importance': 0.5}]],
        })
        scorer.save_to_db(scores)
        mock_cursor.execute.assert_called_once()
        scorer.db.commit.assert_called_once()


# ── Get Latest Scores ───────────────────────────────────────────────────────


class TestGetLatestScores:
    """Test retrieval of latest risk scores."""

    def test_returns_formatted_dicts(self, scorer):
        mock_cursor = MagicMock()
        mock_cursor.description = [
            ('ticker',), ('date',), ('risk_score',), ('risk_level',),
            ('model_version',), ('top_features',),
        ]
        mock_cursor.fetchall.return_value = [
            ('AAPL', datetime(2026, 4, 30).date(), 0.65, 'High', 'lightgbm', '[]'),
        ]
        scorer.db.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        scorer.db.cursor.return_value.__exit__ = MagicMock(return_value=False)

        result = scorer.get_latest_scores(['AAPL'])
        assert len(result) == 1
        assert result[0]['ticker'] == 'AAPL'
        assert result[0]['risk_score'] == 0.65
