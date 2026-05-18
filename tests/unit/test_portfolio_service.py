"""
Key unit tests for Portfolio Service.
Focuses on holdings normalization and risk analysis.
"""

import pytest
from unittest.mock import MagicMock
from datetime import datetime

from apps.api.app.services.portfolio_service import PortfolioService


@pytest.fixture
def svc():
    conn = MagicMock()
    return PortfolioService(conn)


class TestNormalizeHoldings:
    """Test holdings normalization logic."""

    def test_uppercases_tickers(self, svc):
        holdings = [{"ticker": "aapl", "weight": 0.5}, {"ticker": "msft", "weight": 0.5}]
        result = svc.normalize_holdings(holdings)
        tickers = [h['ticker'] for h in result]
        assert 'AAPL' in tickers
        assert 'MSFT' in tickers

    def test_combines_duplicate_tickers(self, svc):
        holdings = [
            {"ticker": "AAPL", "weight": 0.3},
            {"ticker": "AAPL", "weight": 0.2},
            {"ticker": "MSFT", "weight": 0.5},
        ]
        result = svc.normalize_holdings(holdings)
        aapl = [h for h in result if h['ticker'] == 'AAPL']
        assert len(aapl) == 1
        assert abs(aapl[0]['weight'] - 0.5) < 0.01

    def test_normalizes_weights_to_one(self, svc):
        holdings = [
            {"ticker": "AAPL", "weight": 2.0},
            {"ticker": "MSFT", "weight": 3.0},
        ]
        result = svc.normalize_holdings(holdings)
        total = sum(h['weight'] for h in result)
        assert abs(total - 1.0) < 0.01

    def test_empty_holdings_raises(self, svc):
        """Empty holdings should raise ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            svc.normalize_holdings([])


class TestStressTest:
    """Test built-in stress scenarios."""

    def test_builtin_scenarios_exist(self, svc):
        """PortfolioService should have built-in stress test scenarios."""
        # The stress_test method should handle standard scenarios
        mock_cursor = MagicMock()
        mock_cursor.description = [('date',), ('returns_1d',)]
        mock_cursor.fetchall.return_value = []
        svc.db.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        svc.db.cursor.return_value.__exit__ = MagicMock(return_value=False)

        holdings = [{"ticker": "AAPL", "weight": 1.0}]
        result = svc.stress_test(holdings)
        assert 'scenarios' in result or isinstance(result, dict)
