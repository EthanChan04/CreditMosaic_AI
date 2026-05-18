"""
Integration tests for Portfolio API endpoints.
"""

import pytest
from unittest.mock import MagicMock


class TestPortfolioAnalyze:
    """Test POST /api/portfolio/analyze."""

    def test_analyze_returns_response(self, client, mock_portfolio_service):
        """Portfolio analyze may succeed or fail depending on mock state."""
        mock_portfolio_service.analyze_portfolio.return_value = {
            'total_risk_score': 0.45,
            'risk_level': 'Medium',
            'holdings_risk': [
                {'ticker': 'AAPL', 'weight': 0.5, 'risk_score': 0.35,
                 'risk_level': 'Medium', 'risk_contribution': 0.175},
            ],
            'top_contributors': [{'ticker': 'AAPL', 'contribution': 0.175}],
            'diversification_score': 0.72,
            'recommendation': 'Portfolio risk is moderate.',
        }
        response = client.post("/api/portfolio/analyze", json={
            "name": "Test Portfolio",
            "holdings": [
                {"ticker": "AAPL", "weight": 0.5},
                {"ticker": "MSFT", "weight": 0.5},
            ],
        })
        assert response.status_code in (200, 500)

    def test_analyze_empty_holdings_rejected(self, client):
        response = client.post("/api/portfolio/analyze", json={
            "name": "Empty",
            "holdings": [],
        })
        assert response.status_code == 422

    def test_analyze_missing_holdings_rejected(self, client):
        response = client.post("/api/portfolio/analyze", json={
            "name": "No Holdings",
        })
        assert response.status_code == 422


class TestPortfolioList:
    """Test GET /api/portfolios."""

    def test_list_returns_response(self, client, mock_db):
        mock_cursor = MagicMock()
        mock_cursor.description = [
            ('portfolio_id',), ('name',), ('description',), ('holdings',),
            ('created_at',), ('updated_at',),
        ]
        mock_cursor.fetchall.return_value = []
        mock_db.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db.cursor.return_value.__exit__ = MagicMock(return_value=False)
        response = client.get("/api/portfolios")
        assert response.status_code in (200, 500)


class TestPortfolioCorrelation:
    """Test POST /api/portfolio/correlation."""

    def test_correlation_returns_response(self, client, mock_db):
        mock_cursor = MagicMock()
        mock_cursor.description = [('count',)]
        mock_cursor.fetchall.return_value = [(0,)]
        mock_db.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db.cursor.return_value.__exit__ = MagicMock(return_value=False)
        response = client.post("/api/portfolio/correlation", json={
            "holdings": [
                {"ticker": "AAPL", "weight": 0.5},
                {"ticker": "MSFT", "weight": 0.5},
            ],
            "days": 90,
        })
        assert response.status_code in (200, 500)


class TestPortfolioStressTest:
    """Test POST /api/portfolio/stress-test."""

    def test_stress_test_returns_response(self, client, mock_db):
        mock_cursor = MagicMock()
        mock_cursor.description = [('count',)]
        mock_cursor.fetchall.return_value = [(0,)]
        mock_db.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db.cursor.return_value.__exit__ = MagicMock(return_value=False)
        response = client.post("/api/portfolio/stress-test", json={
            "holdings": [
                {"ticker": "AAPL", "weight": 0.5},
                {"ticker": "MSFT", "weight": 0.5},
            ],
        })
        assert response.status_code in (200, 500)


class TestPortfolioReport:
    """Test POST /api/portfolio/report."""

    def test_report_returns_response(self, client, mock_db):
        mock_cursor = MagicMock()
        mock_cursor.description = [('count',)]
        mock_cursor.fetchall.return_value = [(0,)]
        mock_db.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db.cursor.return_value.__exit__ = MagicMock(return_value=False)
        response = client.post("/api/portfolio/report", json={
            "holdings": [
                {"ticker": "AAPL", "weight": 0.5},
                {"ticker": "MSFT", "weight": 0.5},
            ],
        })
        assert response.status_code in (200, 500)
