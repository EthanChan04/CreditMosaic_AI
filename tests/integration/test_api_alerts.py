"""
Integration tests for Alerts API endpoints.
"""

import pytest
from unittest.mock import MagicMock


class TestListAlerts:
    """Test GET /api/alerts."""

    def test_alerts_returns_200(self, client, mock_db):
        mock_cursor = MagicMock()
        mock_cursor.description = [
            ('alert_id',), ('ticker',), ('risk_level',), ('main_driver',),
            ('recommended_review_action',), ('created_at',),
        ]
        mock_cursor.fetchall.return_value = []
        mock_db.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db.cursor.return_value.__exit__ = MagicMock(return_value=False)
        response = client.get("/api/alerts")
        assert response.status_code == 200

    def test_alerts_returns_list(self, client, mock_db):
        mock_cursor = MagicMock()
        mock_cursor.description = [
            ('alert_id',), ('ticker',), ('risk_level',), ('main_driver',),
            ('recommended_review_action',), ('created_at',),
        ]
        mock_cursor.fetchall.return_value = []
        mock_db.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db.cursor.return_value.__exit__ = MagicMock(return_value=False)
        response = client.get("/api/alerts")
        data = response.json()
        assert isinstance(data, list)

    def test_alerts_with_custom_params(self, client, mock_db):
        mock_cursor = MagicMock()
        mock_cursor.description = [
            ('alert_id',), ('ticker',), ('risk_level',), ('main_driver',),
            ('recommended_review_action',), ('created_at',),
        ]
        mock_cursor.fetchall.return_value = []
        mock_db.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db.cursor.return_value.__exit__ = MagicMock(return_value=False)
        response = client.get("/api/alerts?limit=10&min_risk_score=0.7")
        assert response.status_code == 200

    def test_alerts_limit_boundary(self, client, mock_db):
        mock_cursor = MagicMock()
        mock_cursor.description = [
            ('alert_id',), ('ticker',), ('risk_level',), ('main_driver',),
            ('recommended_review_action',), ('created_at',),
        ]
        mock_cursor.fetchall.return_value = []
        mock_db.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db.cursor.return_value.__exit__ = MagicMock(return_value=False)
        # limit=1 (min)
        response = client.get("/api/alerts?limit=1")
        assert response.status_code == 200
        # limit=500 (max)
        response = client.get("/api/alerts?limit=500")
        assert response.status_code == 200

    def test_alerts_limit_out_of_range_rejected(self, client):
        # limit > 500 should be rejected
        response = client.get("/api/alerts?limit=501")
        assert response.status_code == 422

    def test_alerts_min_risk_score_boundary(self, client, mock_db):
        mock_cursor = MagicMock()
        mock_cursor.description = [
            ('alert_id',), ('ticker',), ('risk_level',), ('main_driver',),
            ('recommended_review_action',), ('created_at',),
        ]
        mock_cursor.fetchall.return_value = []
        mock_db.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db.cursor.return_value.__exit__ = MagicMock(return_value=False)
        response = client.get("/api/alerts?min_risk_score=0.0")
        assert response.status_code == 200
        response = client.get("/api/alerts?min_risk_score=1.0")
        assert response.status_code == 200

    def test_alerts_min_risk_score_out_of_range_rejected(self, client):
        response = client.get("/api/alerts?min_risk_score=1.5")
        assert response.status_code == 422
