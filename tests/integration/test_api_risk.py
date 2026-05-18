"""
Integration tests for Risk API endpoints.
"""

import pytest
from unittest.mock import MagicMock, patch
import pandas as pd


class TestGenerateLabels:
    """Test POST /api/risk/labels/generate."""

    def test_generate_labels_returns_200(self, client, mock_risk_service):
        mock_risk_service.generate_risk_labels.return_value = pd.DataFrame(
            {'ticker': ['AAPL'], 'date': ['2026-04-30']}
        )
        response = client.post("/api/risk/labels/generate", json={
            "tickers": ["AAPL"],
        })
        assert response.status_code == 200

    def test_generate_labels_response_structure(self, client, mock_risk_service):
        mock_risk_service.generate_risk_labels.return_value = pd.DataFrame(
            {'ticker': ['AAPL'] * 10, 'date': ['2026-04-30'] * 10}
        )
        response = client.post("/api/risk/labels/generate", json={
            "tickers": ["AAPL"],
        })
        data = response.json()
        assert "status" in data
        assert "rows_generated" in data
        assert "tickers_processed" in data

    def test_generate_labels_empty_tickers_accepted(self, client, mock_risk_service):
        import pandas as pd
        mock_risk_service.generate_risk_labels.return_value = pd.DataFrame()
        response = client.post("/api/risk/labels/generate", json={
            "tickers": [],
        })
        # Empty tickers list may be accepted (returns empty result) or rejected
        assert response.status_code in (200, 422)


class TestTrainModels:
    """Test POST /api/risk/models/train."""

    def test_train_returns_200(self, client):
        response = client.post("/api/risk/models/train", json={
            "tickers": ["AAPL", "MSFT"],
        })
        assert response.status_code == 200

    def test_train_response_has_models(self, client):
        response = client.post("/api/risk/models/train", json={
            "tickers": ["AAPL"],
        })
        data = response.json()
        assert "status" in data


class TestGenerateScores:
    """Test POST /api/risk/scores/generate."""

    def test_generate_scores_returns_200(self, client, mock_risk_service):
        mock_risk_service.score_companies.return_value = pd.DataFrame({
            'ticker': ['AAPL'], 'risk_score': [0.5], 'risk_level': ['Medium'],
            'model_version': ['lightgbm'], 'date': ['2026-04-30'],
            'top_features': [[]],
        })
        response = client.post("/api/risk/scores/generate", json={
            "tickers": ["AAPL"],
        })
        assert response.status_code == 200


class TestGetScores:
    """Test GET /api/risk/scores."""

    def test_get_scores_returns_200(self, client):
        response = client.get("/api/risk/scores?tickers=AAPL,MSFT")
        assert response.status_code == 200

    def test_get_scores_returns_list(self, client):
        response = client.get("/api/risk/scores?tickers=AAPL")
        data = response.json()
        assert isinstance(data, list)

    def test_get_scores_has_required_fields(self, client):
        response = client.get("/api/risk/scores?tickers=AAPL")
        data = response.json()
        if data:
            item = data[0]
            assert "ticker" in item
            assert "risk_score" in item
            assert "risk_level" in item


class TestGetScoreHistory:
    """Test GET /api/risk/scores/{ticker}."""

    def test_history_returns_200(self, client):
        response = client.get("/api/risk/scores/AAPL")
        assert response.status_code == 200

    def test_history_has_ticker(self, client):
        response = client.get("/api/risk/scores/AAPL")
        data = response.json()
        assert "ticker" in data
        assert "history" in data


class TestModelEvaluation:
    """Test GET /api/risk/models/evaluation."""

    def test_evaluation_returns_200(self, client):
        response = client.get("/api/risk/models/evaluation")
        assert response.status_code == 200

    def test_evaluation_has_models(self, client):
        response = client.get("/api/risk/models/evaluation")
        data = response.json()
        assert "models" in data
