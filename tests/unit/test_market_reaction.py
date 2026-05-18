"""
Unit tests for Market Reaction Analyzer.
Covers: window metrics, movement detection, impact classification,
agreement rate, and market model CAR computation.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from pipelines.reaction.market_reaction import (
    MarketReactionAnalyzer,
    MarketReaction,
    REACTION_WINDOWS,
    PRE_NEWS_BASELINE,
)


@pytest.fixture
def analyzer():
    conn = MagicMock()
    return MarketReactionAnalyzer(conn)


# ── Window Metrics ──────────────────────────────────────────────────────────


class TestWindowMetrics:
    """Test computation of reaction metrics per time window."""

    def test_normal_pre_post_data(self, analyzer):
        """With pre and post data, metrics should be computed."""
        dates = pd.bdate_range(end=datetime(2026, 4, 30), periods=30)
        pre = pd.DataFrame({
            'returns_1d': [0.001] * 20,
            'volume': [1000000] * 20,
            'volatility_5d': [0.02] * 20,
        })
        post = pd.DataFrame({
            'returns_1d': [-0.01, -0.02, 0.005],
            'volume': [2000000, 3000000, 1500000],
            'volatility_5d': [0.04, 0.05, 0.03],
        })
        credit = pd.DataFrame({
            'date': dates,
            'hyg_yield': [5.0] * 30,
            'lqd_yield': [4.0] * 30,
            'vix': [20.0] * 30,
        })
        event_date = dates[20]

        metrics = analyzer._compute_window_metrics(pre, post, credit, event_date)

        assert 'cumulative_abnormal_return' in metrics
        assert 'abnormal_volume_ratio' in metrics
        assert 'volatility_change_ratio' in metrics
        assert metrics['abnormal_volume_ratio'] > 1.0  # post volume > pre volume

    def test_empty_pre_data(self, analyzer):
        """Empty pre-data should return zeroed metrics."""
        post = pd.DataFrame({'returns_1d': [0.01], 'volume': [1000], 'volatility_5d': [0.02]})
        metrics = analyzer._compute_window_metrics(pd.DataFrame(), post, pd.DataFrame(), datetime(2026, 4, 30))
        assert metrics['cumulative_abnormal_return'] == 0.0
        assert metrics['abnormal_volume_ratio'] == 1.0

    def test_empty_post_data(self, analyzer):
        """Empty post-data should return zeroed metrics."""
        pre = pd.DataFrame({'returns_1d': [0.01], 'volume': [1000], 'volatility_5d': [0.02]})
        metrics = analyzer._compute_window_metrics(pre, pd.DataFrame(), pd.DataFrame(), datetime(2026, 4, 30))
        assert metrics['cumulative_abnormal_return'] == 0.0

    def test_both_empty(self, analyzer):
        metrics = analyzer._compute_window_metrics(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), datetime(2026, 4, 30))
        assert metrics['cumulative_abnormal_return'] == 0.0
        assert metrics['abnormal_volume_ratio'] == 1.0
        assert metrics['volatility_change_ratio'] == 1.0

    def test_credit_proxy_changes(self, analyzer):
        """Credit proxy changes should be computed when data is available."""
        dates = pd.bdate_range(end=datetime(2026, 4, 30), periods=40)
        pre = pd.DataFrame({'returns_1d': [0.001] * 20, 'volume': [1000] * 20, 'volatility_5d': [0.02] * 20})
        post = pd.DataFrame({'returns_1d': [0.01] * 3, 'volume': [1000] * 3, 'volatility_5d': [0.02] * 3})
        credit = pd.DataFrame({
            'date': dates,
            'hyg_yield': [5.0] * 35 + [5.2] * 5,
            'lqd_yield': [4.0] * 40,
            'vix': [20.0] * 40,
        })
        event_date = dates[20]
        metrics = analyzer._compute_window_metrics(pre, post, credit, event_date)
        assert 'hyg_yield_change' in metrics
        assert 'hyg_yield_change_pct' in metrics


# ── Movement Detection ──────────────────────────────────────────────────────


class TestMovementDetection:
    """Test equity/credit movement detection logic."""

    def test_equity_moved_credit_not(self, analyzer):
        reaction = MarketReaction(
            news_id=1, ticker='TEST', event_date=datetime(2026, 4, 30),
            event_type='earnings', llm_market_impact='equity_leading',
            windows={
                '0_1': {
                    'cumulative_abnormal_return': -0.05,
                    'abnormal_volume_ratio': 1.0,
                    'volatility_change_ratio': 1.0,
                    'hyg_yield_change': 0.0,
                    'lqd_yield_change': 0.0,
                },
                '5_20': {'hyg_yield_change': 0.0},
            },
        )
        eq, cr = analyzer._detect_movement(reaction)
        assert eq is True
        assert cr is False

    def test_credit_moved_equity_not(self, analyzer):
        reaction = MarketReaction(
            news_id=1, ticker='TEST', event_date=datetime(2026, 4, 30),
            event_type='rating_change', llm_market_impact='credit_leading',
            windows={
                '0_1': {
                    'cumulative_abnormal_return': 0.001,
                    'abnormal_volume_ratio': 1.0,
                    'volatility_change_ratio': 1.0,
                    'hyg_yield_change': 0.005,
                    'lqd_yield_change': 0.0,
                },
                '5_20': {'hyg_yield_change': 0.0},
            },
        )
        eq, cr = analyzer._detect_movement(reaction)
        assert eq is False
        assert cr is True

    def test_both_moved(self, analyzer):
        reaction = MarketReaction(
            news_id=1, ticker='TEST', event_date=datetime(2026, 4, 30),
            event_type='crisis', llm_market_impact='two_market_shock',
            windows={
                '0_1': {
                    'cumulative_abnormal_return': -0.08,
                    'abnormal_volume_ratio': 2.0,
                    'volatility_change_ratio': 1.0,
                    'hyg_yield_change': 0.005,
                    'lqd_yield_change': 0.002,
                },
                '5_20': {'hyg_yield_change': 0.0},
            },
        )
        eq, cr = analyzer._detect_movement(reaction)
        assert eq is True
        assert cr is True

    def test_neither_moved(self, analyzer):
        reaction = MarketReaction(
            news_id=1, ticker='TEST', event_date=datetime(2026, 4, 30),
            event_type='neutral', llm_market_impact='low_impact',
            windows={
                '0_1': {
                    'cumulative_abnormal_return': 0.001,
                    'abnormal_volume_ratio': 1.0,
                    'volatility_change_ratio': 1.0,
                    'hyg_yield_change': 0.0,
                    'lqd_yield_change': 0.0,
                },
                '5_20': {'hyg_yield_change': 0.0},
            },
        )
        eq, cr = analyzer._detect_movement(reaction)
        assert eq is False
        assert cr is False


# ── Impact Classification ───────────────────────────────────────────────────


class TestImpactClassification:
    """Test observed impact classification logic."""

    def test_low_impact(self, analyzer):
        result = analyzer._classify_observed_impact(False, False, {}, {})
        assert result == "low_impact"

    def test_equity_leading(self, analyzer):
        result = analyzer._classify_observed_impact(True, False, {}, {})
        assert result == "equity_leading"

    def test_credit_leading(self, analyzer):
        result = analyzer._classify_observed_impact(False, True, {}, {})
        assert result == "credit_leading"

    def test_two_market_shock_both_significant(self, analyzer):
        w_short = {'cumulative_abnormal_return': 0.05, 'hyg_yield_change': 0.005}
        w_long = {'cumulative_abnormal_return': 0.02, 'hyg_yield_change': 0.002}
        result = analyzer._classify_observed_impact(True, True, w_short, w_long)
        assert result in ("two_market_shock", "equity_leading", "credit_leading")

    def test_equity_dominates(self, analyzer):
        w_short = {'cumulative_abnormal_return': 0.1, 'hyg_yield_change': 0.0001}
        w_long = {'cumulative_abnormal_return': 0.05, 'hyg_yield_change': 0.0001}
        result = analyzer._classify_observed_impact(True, True, w_short, w_long)
        assert result == "equity_leading"

    def test_credit_dominates(self, analyzer):
        w_short = {'cumulative_abnormal_return': 0.0001, 'hyg_yield_change': 0.05}
        w_long = {'cumulative_abnormal_return': 0.0001, 'hyg_yield_change': 0.05}
        result = analyzer._classify_observed_impact(True, True, w_short, w_long)
        assert result == "credit_leading"


# ── Agreement Rate ──────────────────────────────────────────────────────────


class TestAgreementRate:
    """Test LLM-predicted vs observed agreement computation."""

    def test_perfect_agreement(self, analyzer):
        reactions = [
            MarketReaction(news_id=i, ticker='TEST', event_date=datetime(2026, 4, 30),
                           event_type='e', llm_market_impact='equity_leading',
                           observed_impact_type='equity_leading')
            for i in range(5)
        ]
        result = analyzer.compute_agreement_rate(reactions)
        assert result['overall_agreement'] == 1.0

    def test_no_agreement(self, analyzer):
        reactions = [
            MarketReaction(news_id=i, ticker='TEST', event_date=datetime(2026, 4, 30),
                           event_type='e', llm_market_impact='equity_leading',
                           observed_impact_type='credit_leading')
            for i in range(5)
        ]
        result = analyzer.compute_agreement_rate(reactions)
        assert result['overall_agreement'] == 0.0

    def test_empty_reactions(self, analyzer):
        result = analyzer.compute_agreement_rate([])
        assert result == {}

    def test_by_type_agreement(self, analyzer):
        reactions = [
            MarketReaction(news_id=1, ticker='TEST', event_date=datetime(2026, 4, 30),
                           event_type='e', llm_market_impact='equity_leading',
                           observed_impact_type='equity_leading'),
            MarketReaction(news_id=2, ticker='TEST', event_date=datetime(2026, 4, 30),
                           event_type='e', llm_market_impact='credit_leading',
                           observed_impact_type='equity_leading'),
        ]
        result = analyzer.compute_agreement_rate(reactions)
        assert result['equity_leading_agreement'] == 1.0
        assert result['credit_leading_agreement'] == 0.0


# ── Market Model CAR ────────────────────────────────────────────────────────


class TestMarketModelCAR:
    """Test CAPM-style CAR computation."""

    def test_estimates_reasonable_beta(self, analyzer):
        np.random.seed(42)
        n = 200
        dates = pd.bdate_range(end=datetime(2026, 4, 30), periods=n)
        market = pd.Series(np.random.normal(0.0005, 0.01, n), index=dates)
        alpha, beta = 0.001, 1.2
        ticker = pd.Series(alpha + beta * market.values + np.random.normal(0, 0.005, n), index=dates)

        result = analyzer.compute_market_model_car(ticker, market, event_idx=150, estimation_window=100, gap=10)
        assert 'alpha' in result
        assert 'beta' in result
        assert 0.8 < result['beta'] < 1.6

    def test_car_near_zero_no_event(self, analyzer):
        np.random.seed(99)
        n = 200
        dates = pd.bdate_range(end=datetime(2026, 4, 30), periods=n)
        market = pd.Series(np.random.normal(0, 0.01, n), index=dates)
        ticker = pd.Series(np.random.normal(0, 0.01, n), index=dates)
        result = analyzer.compute_market_model_car(ticker, market, event_idx=150)
        assert abs(result['car']) < 0.1

    def test_returns_t_stat_and_pvalue(self, analyzer):
        np.random.seed(42)
        n = 100
        dates = pd.bdate_range(end=datetime(2026, 4, 30), periods=n)
        market = pd.Series(np.random.normal(0, 0.01, n), index=dates)
        ticker = pd.Series(0.001 + 1.2 * market.values + np.random.normal(0, 0.005, n), index=dates)
        result = analyzer.compute_market_model_car(ticker, market, event_idx=80)
        assert 'car_t_stat' in result
        assert 'car_p_value' in result
        assert 0 <= result['car_p_value'] <= 1

    def test_insufficient_data_returns_error(self, analyzer):
        dates = pd.bdate_range(end=datetime(2026, 4, 30), periods=10)
        ticker = pd.Series(np.random.randn(10), index=dates)
        market = pd.Series(np.random.randn(10), index=dates)
        result = analyzer.compute_market_model_car(ticker, market, event_idx=5)
        assert 'error' in result

    def test_significance_flag(self, analyzer):
        np.random.seed(42)
        n = 200
        dates = pd.bdate_range(end=datetime(2026, 4, 30), periods=n)
        market = pd.Series(np.random.normal(0, 0.01, n), index=dates)
        ticker = pd.Series(0.001 + 1.2 * market.values + np.random.normal(0, 0.005, n), index=dates)
        result = analyzer.compute_market_model_car(ticker, market, event_idx=150)
        assert 'significant_at_5pct' in result
        # numpy bool_ is a subclass of bool in practice, but check explicitly
        assert bool(result['significant_at_5pct']) == result['significant_at_5pct']


# ── Reaction Summary ────────────────────────────────────────────────────────


class TestReactionSummary:
    """Test reaction-to-DataFrame conversion."""

    def test_summary_has_all_fields(self, analyzer):
        reactions = [
            MarketReaction(
                news_id=1, ticker='AAPL', event_date=datetime(2026, 4, 30),
                event_type='earnings', llm_market_impact='equity_leading',
                observed_impact_type='equity_leading',
                windows={'0_1': {'cumulative_abnormal_return': -0.03}},
            ),
        ]
        df = analyzer.get_reaction_summary(reactions)
        assert len(df) == 1
        assert 'news_id' in df.columns
        assert 'ticker' in df.columns
        assert '0_1_cumulative_abnormal_return' in df.columns

    def test_empty_reactions(self, analyzer):
        df = analyzer.get_reaction_summary([])
        assert df.empty
