"""
Key unit tests for Lag Analyzer.
Focuses on lead-lag detection and distribution analysis.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from pipelines.reaction.lag_analyzer import LagAnalyzer


@pytest.fixture
def analyzer():
    conn = MagicMock()
    return LagAnalyzer(conn)


class TestLeadLagByEvent:
    """Test event-level lead-lag detection."""

    def _make_data(self, n=30):
        dates = pd.bdate_range(end=datetime(2026, 4, 30), periods=n)
        market = pd.DataFrame({
            'ticker': ['AAPL'] * n,
            'date': dates,
            'returns_1d': np.random.normal(0, 0.02, n),
        })
        credit = pd.DataFrame({
            'date': dates,
            'hyg_yield': [5.0] * 20 + [5.3] * 10,
            'lqd_yield': [4.0] * n,
        })
        return market, credit

    def test_equity_leads_credit(self, analyzer):
        """When equity moves before credit, leading_market should be 'equity'."""
        market, credit = self._make_data()
        # Make equity move on day 22 (first day after event)
        market.loc[22, 'returns_1d'] = -0.05
        # Make credit move on day 25
        credit.loc[25, 'hyg_yield'] = 5.5

        events = [{
            'news_id': 1, 'ticker': 'AAPL',
            'event_date': market.loc[21, 'date'],
            'event_type': 'earnings',
            'llm_market_impact': 'equity_leading',
        }]

        result = analyzer.analyze_lead_lag_by_event(events, market, credit, max_lag=10)
        assert len(result) == 1
        assert result.iloc[0]['leading_market'] in ('equity', 'simultaneous', 'none')

    def test_no_movement_returns_none(self, analyzer):
        """When neither market moves, leading_market should be 'none'."""
        market = pd.DataFrame({
            'ticker': ['AAPL'] * 30,
            'date': pd.bdate_range(end=datetime(2026, 4, 30), periods=30),
            'returns_1d': [0.001] * 30,
        })
        credit = pd.DataFrame({
            'date': pd.bdate_range(end=datetime(2026, 4, 30), periods=30),
            'hyg_yield': [5.0] * 30,
        })

        events = [{
            'news_id': 1, 'ticker': 'AAPL',
            'event_date': market.loc[20, 'date'],
            'event_type': 'neutral',
            'llm_market_impact': 'low_impact',
        }]

        result = analyzer.analyze_lead_lag_by_event(events, market, credit, max_lag=10)
        assert result.iloc[0]['leading_market'] == 'none'

    def test_empty_events(self, analyzer):
        result = analyzer.analyze_lead_lag_by_event([], pd.DataFrame(), pd.DataFrame())
        assert result.empty


class TestLeadLagDistribution:
    """Test distribution summary of lead-lag patterns."""

    def test_distribution_summary(self, analyzer):
        results = pd.DataFrame({
            'leading_market': ['equity', 'equity', 'credit', 'simultaneous'],
            'llm_market_impact': ['equity_leading', 'equity_leading', 'credit_leading', 'two_market_shock'],
            'lead_lag_days': [-2, -1, 3, 0],
        })
        dist = analyzer.get_lead_lag_distribution(results)
        assert dist['total_events'] == 4
        assert dist['leading_market_distribution']['equity'] == 2
        assert dist['negative_means_equity_leads'] is True

    def test_empty_results(self, analyzer):
        dist = analyzer.get_lead_lag_distribution(pd.DataFrame())
        assert dist['total_events'] == 0
