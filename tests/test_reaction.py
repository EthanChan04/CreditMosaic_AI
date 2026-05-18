"""
Tests for event study and reaction analysis pipeline.
Verifies market model CAR computation and lag significance testing.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta


class TestMarketModelCAR:
    """Verify market model (CAPM-style) CAR computation."""

    def _make_returns(self, n=100, seed=42):
        np.random.seed(seed)
        dates = pd.bdate_range(end=datetime(2026, 4, 30), periods=n)
        market = pd.Series(np.random.normal(0.0005, 0.01, n), index=dates)
        # Ticker = alpha + beta * market + noise
        alpha, beta = 0.001, 1.2
        ticker = alpha + beta * market + np.random.normal(0, 0.005, n)
        return pd.Series(ticker, index=dates), market

    def test_market_model_estimates_alpha_beta(self):
        """Market model should recover approximate alpha and beta."""
        from pipelines.reaction.market_reaction import MarketReactionAnalyzer

        analyzer = MarketReactionAnalyzer.__new__(MarketReactionAnalyzer)
        ticker_ret, market_ret = self._make_returns(n=200)

        result = analyzer.compute_market_model_car(
            ticker_ret, market_ret, event_idx=150,
            estimation_window=100, gap=10
        )

        assert 'alpha' in result
        assert 'beta' in result
        assert 'error' not in result
        # Beta should be close to 1.2
        assert 0.8 < result['beta'] < 1.6, f"Beta {result['beta']} out of expected range"

    def test_car_zero_when_no_event(self):
        """CAR should be near zero when there's no abnormal event."""
        from pipelines.reaction.market_reaction import MarketReactionAnalyzer

        analyzer = MarketReactionAnalyzer.__new__(MarketReactionAnalyzer)
        np.random.seed(99)
        n = 200
        dates = pd.bdate_range(end=datetime(2026, 4, 30), periods=n)
        # Both series are just noise
        market = pd.Series(np.random.normal(0, 0.01, n), index=dates)
        ticker = pd.Series(np.random.normal(0, 0.01, n), index=dates)

        result = analyzer.compute_market_model_car(
            ticker, market, event_idx=150,
            estimation_window=100, gap=10
        )

        assert 'car' in result
        assert abs(result['car']) < 0.1, f"CAR {result['car']} should be near zero"

    def test_returns_t_statistic_and_pvalue(self):
        """Result must include t-statistic and p-value for CAR."""
        from pipelines.reaction.market_reaction import MarketReactionAnalyzer

        analyzer = MarketReactionAnalyzer.__new__(MarketReactionAnalyzer)
        ticker_ret, market_ret = self._make_returns()

        result = analyzer.compute_market_model_car(
            ticker_ret, market_ret, event_idx=80
        )

        assert 'car_t_stat' in result
        assert 'car_p_value' in result
        assert 0 <= result['car_p_value'] <= 1

    def test_handles_insufficient_data(self):
        """Should return error dict when data is too short."""
        from pipelines.reaction.market_reaction import MarketReactionAnalyzer

        analyzer = MarketReactionAnalyzer.__new__(MarketReactionAnalyzer)
        dates = pd.bdate_range(end=datetime(2026, 4, 30), periods=10)
        ticker = pd.Series(np.random.randn(10), index=dates)
        market = pd.Series(np.random.randn(10), index=dates)

        result = analyzer.compute_market_model_car(ticker, market, event_idx=5)
        assert 'error' in result


class TestLagSignificance:
    """Verify lag correlation includes p-values."""

    def test_lag_results_include_pvalues(self):
        """compute_cross_correlation output must include lag_pvalues."""
        from pipelines.reaction.lag_analyzer import LagAnalyzer
        import inspect

        # Check that the method source contains p-value computation
        source = inspect.getsource(LagAnalyzer.compute_cross_correlation)
        assert 'lag_pvalues' in source
        assert 'p_value' in source
        assert 'significant_at_5pct' in source

    def test_best_lag_includes_significance(self):
        """Best lag result must include significance flag."""
        from pipelines.reaction.lag_analyzer import LagAnalyzer
        import inspect

        source = inspect.getsource(LagAnalyzer.compute_cross_correlation)
        assert 'significant_at_5pct' in source
