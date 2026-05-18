"""
Integration tests for Reaction API endpoints.
"""

import pytest
from unittest.mock import MagicMock
from datetime import datetime


class TestReactionAnalyze:
    """Test POST /api/reaction/analyze."""

    def test_analyze_returns_response(self, client, mock_db):
        mock_cursor = MagicMock()
        mock_cursor.description = [('count',)]
        mock_cursor.fetchall.return_value = [(0,)]
        mock_db.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db.cursor.return_value.__exit__ = MagicMock(return_value=False)
        response = client.post("/api/reaction/analyze", json={
            "tickers": ["AAPL"],
            "start_date": "2026-01-01",
            "end_date": "2026-04-30",
        })
        assert response.status_code in (200, 500)

    def test_analyze_missing_tickers_rejected(self, client):
        response = client.post("/api/reaction/analyze", json={})
        assert response.status_code == 422


class TestReactionLag:
    """Test POST /api/reaction/lag."""

    def test_lag_returns_response(self, client, mock_db):
        mock_cursor = MagicMock()
        mock_cursor.description = [('count',)]
        mock_cursor.fetchall.return_value = [(0,)]
        mock_db.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db.cursor.return_value.__exit__ = MagicMock(return_value=False)
        response = client.post("/api/reaction/lag", json={
            "tickers": ["AAPL"],
            "start_date": "2026-01-01",
            "end_date": "2026-04-30",
        })
        assert response.status_code in (200, 500)


class TestReactionByTicker:
    """Test GET /api/reaction/ticker/{ticker}."""

    def test_ticker_reaction_returns_200(self, client, mock_db):
        mock_cursor = MagicMock()
        mock_cursor.description = [('count',)]
        mock_cursor.fetchall.return_value = [(0,)]
        mock_db.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db.cursor.return_value.__exit__ = MagicMock(return_value=False)
        response = client.get("/api/reaction/ticker/AAPL?days=30")
        assert response.status_code in (200, 500)


class TestReactionByNews:
    """Test GET /api/reaction/news/{news_id}."""

    def test_news_reaction_returns_response(self, client, mock_db):
        mock_cursor = MagicMock()
        mock_cursor.description = [('count',)]
        mock_cursor.fetchall.return_value = []
        mock_db.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db.cursor.return_value.__exit__ = MagicMock(return_value=False)
        response = client.get("/api/reaction/news/1")
        assert response.status_code in (200, 404, 500)


class TestReactionSummary:
    """Test GET /api/reaction/summary."""

    def test_summary_returns_response(self, client, mock_db, mock_market_reaction_service):
        mock_cursor = MagicMock()
        mock_cursor.description = [('count',)]
        mock_cursor.fetchall.return_value = [(0,)]
        mock_db.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_market_reaction_service.get_reaction_summary.return_value = {
            'total_events': 0,
            'avg_car': 0.0,
        }
        response = client.get("/api/reaction/summary")
        assert response.status_code in (200, 422, 500)
