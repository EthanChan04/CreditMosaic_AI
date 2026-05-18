"""
Shared test fixtures for CreditMosaic AI test suite.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch


@pytest.fixture
def sample_market_data():
    """Generate synthetic daily market data for testing."""
    np.random.seed(42)
    n = 250  # ~1 trading year
    dates = pd.bdate_range(end=datetime(2026, 4, 30), periods=n)
    tickers = ['AAPL', 'MSFT', 'GOOGL']

    rows = []
    for ticker in tickers:
        price = 100.0
        for i, date in enumerate(dates):
            ret = np.random.normal(0.0005, 0.02)
            price *= (1 + ret)
            rows.append({
                'ticker': ticker,
                'date': date,
                'open_price': price * (1 + np.random.uniform(-0.01, 0.01)),
                'high_price': price * (1 + abs(np.random.normal(0, 0.01))),
                'low_price': price * (1 - abs(np.random.normal(0, 0.01))),
                'close_price': price,
                'volume': int(np.random.lognormal(15, 0.5)),
                'adjusted_close': price,
                'returns_1d': ret,
                'returns_5d': np.random.normal(0.002, 0.05),
                'returns_20d': np.random.normal(0.008, 0.1),
                'volatility_5d': abs(np.random.normal(0.02, 0.005)),
                'volatility_20d': abs(np.random.normal(0.025, 0.008)),
                'volume_ma_5d': int(np.random.lognormal(15, 0.5)),
                'volume_ma_20d': int(np.random.lognormal(15, 0.5)),
            })

    return pd.DataFrame(rows)


@pytest.fixture
def sample_risk_labels():
    """Generate synthetic risk labels."""
    np.random.seed(123)
    n = 250
    dates = pd.bdate_range(end=datetime(2026, 4, 30), periods=n)
    tickers = ['AAPL', 'MSFT', 'GOOGL']

    rows = []
    for ticker in tickers:
        for date in dates:
            rows.append({
                'ticker': ticker,
                'date': date,
                'abnormal_negative_return_1d': np.random.random() < 0.05,
                'abnormal_negative_return_5d': np.random.random() < 0.08,
                'abnormal_volume_spike_1d': np.random.random() < 0.03,
                'volatility_jump_5d': np.random.random() < 0.04,
                'credit_proxy_widening_5d': np.random.random() < 0.02,
                'distress_news_followup_30d': np.random.random() < 0.01,
            })

    return pd.DataFrame(rows)


@pytest.fixture
def sample_llm_signals():
    """Generate synthetic LLM news signals."""
    np.random.seed(456)
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    rows = []
    for ticker in tickers:
        for i in range(20):
            rows.append({
                'ticker': ticker,
                'date': datetime(2026, 4, 1) + timedelta(days=i),
                'sentiment_score': np.random.uniform(-1, 1),
                'credit_risk_score': int(np.random.uniform(0, 100)),
                'event_type': np.random.choice([
                    'earnings_deterioration', 'debt_refinancing',
                    'neutral_or_irrelevant', 'regulatory'
                ]),
                'confidence': np.random.uniform(0.5, 1.0),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def mock_db_connection():
    """Create a mock database connection."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn
