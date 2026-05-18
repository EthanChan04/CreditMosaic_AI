"""
Unit tests for Risk Labeler pipeline.
Covers: abnormal returns, volume spikes, volatility jumps, credit widening,
distress news labeling, and full label generation flow.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from pipelines.risk.risk_labeler import RiskLabeler


@pytest.fixture
def labeler():
    """Create a RiskLabeler with a mock DB connection."""
    conn = MagicMock()
    return RiskLabeler(conn)


@pytest.fixture
def normal_market_data():
    """Generate market data with normal behavior (no anomalies)."""
    np.random.seed(42)
    n = 100
    dates = pd.bdate_range(end=datetime(2026, 4, 30), periods=n)
    rows = []
    for i, date in enumerate(dates):
        rows.append({
            'ticker': 'TEST',
            'date': date,
            'close_price': 100 + np.random.normal(0, 1),
            'volume': int(np.random.lognormal(15, 0.3)),
            'returns_1d': np.random.normal(0.0005, 0.01),
            'returns_5d': np.random.normal(0.002, 0.03),
            'returns_20d': np.random.normal(0.008, 0.06),
            'volatility_5d': abs(np.random.normal(0.015, 0.003)),
            'volatility_20d': abs(np.random.normal(0.02, 0.004)),
            'volume_ma_5d': int(np.random.lognormal(15, 0.3)),
            'volume_ma_20d': int(np.random.lognormal(15, 0.3)),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def extreme_market_data():
    """Generate market data with extreme events embedded."""
    np.random.seed(99)
    n = 100
    dates = pd.bdate_range(end=datetime(2026, 4, 30), periods=n)
    rows = []
    for i, date in enumerate(dates):
        if i == 50:
            # Extreme negative return
            ret_1d = -0.15
            vol = int(1e9)  # huge volume
            v5 = 0.08  # high volatility
        else:
            ret_1d = np.random.normal(0.0005, 0.01)
            vol = int(np.random.lognormal(15, 0.3))
            v5 = abs(np.random.normal(0.015, 0.003))

        rows.append({
            'ticker': 'TEST',
            'date': date,
            'close_price': 100 + i * 0.1,
            'volume': vol,
            'returns_1d': ret_1d,
            'returns_5d': np.random.normal(0.002, 0.03),
            'returns_20d': np.random.normal(0.008, 0.06),
            'volatility_5d': v5,
            'volatility_20d': abs(np.random.normal(0.02, 0.004)),
            'volume_ma_5d': int(np.random.lognormal(15, 0.3)),
            'volume_ma_20d': int(np.random.lognormal(15, 0.3)),
        })
    return pd.DataFrame(rows)


# ── Abnormal Returns ────────────────────────────────────────────────────────


class TestAbnormalReturns:
    """Test abnormal negative return labeling."""

    def test_normal_returns_not_flagged(self, labeler, normal_market_data):
        df = normal_market_data.copy()
        labeler._label_abnormal_returns(df)
        # Most normal returns should not be flagged
        flag_rate = df['abnormal_negative_return_1d'].mean()
        assert flag_rate < 0.2, f"Too many normal returns flagged: {flag_rate}"

    def test_extreme_negative_return_flagged(self, labeler, extreme_market_data):
        df = extreme_market_data.copy()
        labeler._label_abnormal_returns(df)
        # The extreme day (index 50) should be flagged
        assert df.iloc[50]['abnormal_negative_return_1d'] == True

    def test_nan_handled_gracefully(self, labeler):
        df = pd.DataFrame({
            'ticker': ['A'] * 10,
            'date': pd.bdate_range(end=datetime(2026, 4, 30), periods=10),
            'returns_1d': [np.nan] * 10,
            'returns_5d': [np.nan] * 10,
            'returns_20d': [np.nan] * 10,
        })
        labeler._label_abnormal_returns(df)
        assert df['abnormal_negative_return_1d'].fillna(False).sum() == 0

    def test_missing_column_handled(self, labeler):
        df = pd.DataFrame({
            'ticker': ['A'],
            'date': [datetime(2026, 4, 30)],
        })
        labeler._label_abnormal_returns(df)
        # Should not raise; column simply not added

    def test_5d_returns_flagged(self, labeler):
        df = pd.DataFrame({
            'ticker': ['A'] * 80,
            'date': pd.bdate_range(end=datetime(2026, 4, 30), periods=80),
            'returns_5d': [0.0] * 79 + [-0.5],
        })
        labeler._label_abnormal_returns(df)
        assert df.iloc[-1]['abnormal_negative_return_5d'] == True


# ── Volume Spikes ───────────────────────────────────────────────────────────


class TestVolumeSpikes:
    """Test volume spike labeling."""

    def test_normal_volume_not_flagged(self, labeler, normal_market_data):
        df = normal_market_data.copy()
        labeler._label_volume_spikes(df)
        assert df['abnormal_volume_spike_1d'].sum() < len(df) * 0.1

    def test_extreme_volume_flagged(self, labeler):
        df = pd.DataFrame({
            'ticker': ['A'] * 30,
            'date': pd.bdate_range(end=datetime(2026, 4, 30), periods=30),
            'volume': [1000] * 29 + [100000],
            'volume_ma_20d': [1000] * 30,
            'volume_ma_5d': [1000] * 30,
        })
        labeler._label_volume_spikes(df)
        assert df.iloc[-1]['abnormal_volume_spike_1d'] == True

    def test_zero_ma_not_flagged(self, labeler):
        df = pd.DataFrame({
            'ticker': ['A'] * 5,
            'date': pd.bdate_range(end=datetime(2026, 4, 30), periods=5),
            'volume': [1000] * 5,
            'volume_ma_20d': [0] * 5,
            'volume_ma_5d': [0] * 5,
        })
        labeler._label_volume_spikes(df)
        assert df['abnormal_volume_spike_1d'].sum() == 0

    def test_missing_columns_default_false(self, labeler):
        df = pd.DataFrame({
            'ticker': ['A'],
            'date': [datetime(2026, 4, 30)],
        })
        labeler._label_volume_spikes(df)
        assert df['abnormal_volume_spike_1d'].iloc[0] == False

    def test_5d_spike_threshold(self, labeler):
        """5d spike threshold is 3x volume_ma_5d."""
        df = pd.DataFrame({
            'ticker': ['A'] * 10,
            'date': pd.bdate_range(end=datetime(2026, 4, 30), periods=10),
            'volume': [1000] * 9 + [4000],
            'volume_ma_5d': [1000] * 10,
            'volume_ma_20d': [1000] * 10,
        })
        labeler._label_volume_spikes(df)
        assert df.iloc[-1]['abnormal_volume_spike_5d'] == True


# ── Volatility Jumps ────────────────────────────────────────────────────────


class TestVolatilityJumps:
    """Test volatility jump labeling."""

    def test_normal_volatility_not_flagged(self, labeler, normal_market_data):
        df = normal_market_data.copy()
        labeler._label_volatility_jumps(df)
        flag_rate = df['volatility_jump_5d'].mean()
        assert flag_rate < 0.2

    def test_volatility_jump_flagged(self, labeler):
        df = pd.DataFrame({
            'ticker': ['A'] * 80,
            'date': pd.bdate_range(end=datetime(2026, 4, 30), periods=80),
            'volatility_5d': [0.02] * 79 + [0.06],
            'volatility_20d': [0.02] * 80,
        })
        labeler._label_volatility_jumps(df)
        assert df.iloc[-1]['volatility_jump_5d'] == True

    def test_zero_baseline_not_flagged(self, labeler):
        df = pd.DataFrame({
            'ticker': ['A'] * 10,
            'date': pd.bdate_range(end=datetime(2026, 4, 30), periods=10),
            'volatility_5d': [0.05] * 10,
            'volatility_20d': [0.0] * 10,
        })
        labeler._label_volatility_jumps(df)
        assert df['volatility_jump_5d'].sum() == 0

    def test_missing_columns_default_false(self, labeler):
        df = pd.DataFrame({
            'ticker': ['A'],
            'date': [datetime(2026, 4, 30)],
        })
        labeler._label_volatility_jumps(df)
        assert df['volatility_jump_5d'].iloc[0] == False
        assert df['volatility_jump_20d'].iloc[0] == False


# ── Credit Widening ─────────────────────────────────────────────────────────


class TestCreditWidening:
    """Test credit proxy widening labeling."""

    def test_empty_credit_data_defaults_false(self, labeler):
        df = pd.DataFrame({
            'ticker': ['A'],
            'date': [datetime(2026, 4, 30)],
        })
        labeler._label_credit_widening(df, pd.DataFrame())
        assert df['credit_proxy_widening_5d'].iloc[0] == False

    def test_credit_widening_with_yield_data(self, labeler):
        """When HYG yield spikes, flag should be set."""
        dates = pd.bdate_range(end=datetime(2026, 4, 30), periods=30)
        credit = pd.DataFrame({
            'date': dates,
            'hyg_yield': [5.0] * 29 + [5.5],  # big jump
            'lqd_yield': [4.0] * 30,
        })
        df = pd.DataFrame({
            'ticker': ['A'] * 5,
            'date': dates[-5:],
        })
        labeler._label_credit_widening(df, credit)
        # The spike at day 29 should propagate to nearby days
        assert df['credit_proxy_widening_5d'].any()


# ── Distress News ───────────────────────────────────────────────────────────


class TestDistressNews:
    """Test distress news followup labeling."""

    def test_empty_signals_defaults_false(self, labeler):
        df = pd.DataFrame({
            'ticker': ['A'],
            'date': [datetime(2026, 4, 30)],
        })
        labeler._label_distress_news(df, pd.DataFrame())
        assert df['distress_news_followup_30d'].iloc[0] == False
        assert df['distress_news_followup_90d'].iloc[0] == False

    def test_no_high_risk_signals(self, labeler):
        df = pd.DataFrame({
            'ticker': ['A'],
            'date': [datetime(2026, 4, 30)],
        })
        signals = pd.DataFrame({
            'ticker': ['A'],
            'credit_risk_score': [30],  # below 70
            'date': [datetime(2026, 4, 28)],
        })
        labeler._label_distress_news(df, signals)
        assert df['distress_news_followup_30d'].iloc[0] == False

    def test_high_risk_signal_within_30d(self, labeler):
        """Distress news followup checks if high-risk signal exists in the FUTURE window.
        The labeler checks (dist_d - d).days in range(0, 31), meaning the distress
        date must be >= current date and within 30 days AFTER.
        """
        # Event date is AFTER the label dates (future distress signal)
        event_date = datetime(2026, 4, 28)
        df = pd.DataFrame({
            'ticker': ['A'] * 3,
            'date': pd.bdate_range(end=datetime(2026, 4, 25), periods=3),
        })
        signals = pd.DataFrame({
            'ticker': ['A'],
            'credit_risk_score': [85],
            'date': [event_date],
        })
        labeler._label_distress_news(df, signals)
        # Dates within 30 days before the event should be flagged
        assert df['distress_news_followup_30d'].any()

    def test_high_risk_signal_outside_window(self, labeler):
        df = pd.DataFrame({
            'ticker': ['A'],
            'date': [datetime(2026, 1, 1)],
        })
        signals = pd.DataFrame({
            'ticker': ['A'],
            'credit_risk_score': [90],
            'date': [datetime(2026, 4, 30)],
        })
        labeler._label_distress_news(df, signals)
        assert df['distress_news_followup_30d'].iloc[0] == False


# ── Full Label Generation ───────────────────────────────────────────────────


class TestGenerateAllLabels:
    """Test the full label generation pipeline."""

    def test_empty_market_data_returns_empty(self, labeler, monkeypatch):
        monkeypatch.setattr(labeler, '_load_market_data', lambda *a, **kw: pd.DataFrame())
        monkeypatch.setattr(labeler, '_load_credit_proxy_data', lambda *a, **kw: pd.DataFrame())
        monkeypatch.setattr(labeler, '_load_llm_signals', lambda *a, **kw: pd.DataFrame())
        result = labeler.generate_all_labels(['AAPL'], datetime(2026, 1, 1), datetime(2026, 4, 30))
        assert result.empty

    def test_labels_have_expected_columns(self, labeler, monkeypatch, normal_market_data):
        monkeypatch.setattr(labeler, '_load_market_data', lambda *a, **kw: normal_market_data)
        monkeypatch.setattr(labeler, '_load_credit_proxy_data', lambda *a, **kw: pd.DataFrame())
        monkeypatch.setattr(labeler, '_load_llm_signals', lambda *a, **kw: pd.DataFrame())
        result = labeler.generate_all_labels(['TEST'], datetime(2025, 1, 1), datetime(2026, 4, 30))
        expected_cols = [
            'ticker', 'date',
            'abnormal_negative_return_1d', 'abnormal_negative_return_5d',
            'abnormal_negative_return_20d',
            'abnormal_volume_spike_1d', 'abnormal_volume_spike_5d',
            'volatility_jump_5d', 'volatility_jump_20d',
            'credit_proxy_widening_5d', 'credit_proxy_widening_20d',
            'distress_news_followup_30d', 'distress_news_followup_90d',
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"
