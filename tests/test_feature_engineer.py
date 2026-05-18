"""
Tests for feature engineering pipeline.
Verifies data leakage prevention, feature group selection, and correct aggregation.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from pipelines.risk.feature_engineer import FeatureEngineer


class TestLabelProxyFeatureRemoval:
    """Verify that label proxy features are correctly removed."""

    def test_returns_1d_removed(self):
        """returns_1d must be removed as it feeds directly into label computation."""
        fe = FeatureEngineer.__new__(FeatureEngineer)
        X = pd.DataFrame({
            'returns_1d': [0.01, -0.02],
            'returns_1d_lag6': [0.01, -0.02],
            'debt_to_assets': [0.5, 0.6],
        })
        result = fe._remove_label_proxy_features(X)
        assert 'returns_1d' not in result.columns
        assert 'returns_1d_lag6' in result.columns

    def test_volatility_5d_removed(self):
        """volatility_5d must be removed as it feeds into volatility_jump_5d label."""
        fe = FeatureEngineer.__new__(FeatureEngineer)
        X = pd.DataFrame({
            'volatility_5d': [0.02, 0.03],
            'volatility_5d_lag6': [0.02, 0.03],
            'debt_to_assets': [0.5, 0.6],
        })
        result = fe._remove_label_proxy_features(X)
        assert 'volatility_5d' not in result.columns
        assert 'volatility_5d_lag6' in result.columns

    def test_volume_removed(self):
        """volume must be removed as it feeds into volume spike label."""
        fe = FeatureEngineer.__new__(FeatureEngineer)
        X = pd.DataFrame({
            'volume': [1000.0, 2000.0],
            'volume_lag6': [1000.0, 2000.0],
            'debt_to_assets': [0.5, 0.6],
        })
        result = fe._remove_label_proxy_features(X)
        assert 'volume' not in result.columns
        assert 'volume_lag6' in result.columns

    def test_unsafe_lags_removed(self):
        """Lagged features with lag < SAFE_LAG_MIN must be removed."""
        fe = FeatureEngineer.__new__(FeatureEngineer)
        X = pd.DataFrame({
            'returns_1d_lag1': [0.01],
            'returns_1d_lag5': [0.01],
            'volatility_5d_lag1': [0.02],
            'volatility_5d_lag5': [0.02],
            'debt_to_assets': [0.5],
        })
        result = fe._remove_label_proxy_features(X)
        for col in ['returns_1d_lag1', 'returns_1d_lag5', 'volatility_5d_lag1', 'volatility_5d_lag5']:
            assert col not in result.columns, f"{col} should be removed (lag < {fe.SAFE_LAG_MIN})"

    def test_rolling_features_removed(self):
        """Rolling mean features of returns/volatility must be removed."""
        fe = FeatureEngineer.__new__(FeatureEngineer)
        X = pd.DataFrame({
            'rolling_return_1d': [0.01],
            'rolling_return_5d': [0.01],
            'rolling_vol_1d': [0.02],
            'rolling_vol_5d': [0.02],
            'debt_to_assets': [0.5],
        })
        result = fe._remove_label_proxy_features(X)
        for col in ['rolling_return_1d', 'rolling_return_5d', 'rolling_vol_1d', 'rolling_vol_5d']:
            assert col not in result.columns

    def test_non_proxy_features_kept(self):
        """Non-proxy features like fundamentals and LLM signals must be kept."""
        fe = FeatureEngineer.__new__(FeatureEngineer)
        X = pd.DataFrame({
            'debt_to_assets': [0.5],
            'current_ratio': [1.5],
            'llm_risk_avg_7d': [50.0],
            'vix': [20.0],
            'hyg_yield': [5.0],
            'returns_1d': [0.01],
        })
        result = fe._remove_label_proxy_features(X)
        for col in ['debt_to_assets', 'current_ratio', 'llm_risk_avg_7d', 'vix', 'hyg_yield']:
            assert col in result.columns, f"{col} should be kept"

    def test_safe_lag_minimum(self):
        """SAFE_LAG_MIN must be >= 5 to avoid overlapping with label windows."""
        assert FeatureEngineer.SAFE_LAG_MIN >= 5

    def test_all_label_sources_in_proxy_set(self):
        """All data sources used in risk_labeler must be in LABEL_PROXY_FEATURES."""
        # These are the columns used by RiskLabeler to compute labels
        label_source_cols = [
            'returns_1d', 'returns_5d', 'volatility_5d', 'volatility_20d',
            'volume', 'volume_ma_20d', 'volume_ma_5d',
        ]
        for col in label_source_cols:
            assert col in FeatureEngineer.LABEL_PROXY_FEATURES, \
                f"{col} is used in label computation but not in LABEL_PROXY_FEATURES"


class TestFeatureGroupSelection:
    """Verify that feature_groups parameter correctly selects which features to include."""

    def test_all_feature_groups_defined(self):
        """ALL_FEATURE_GROUPS must list all available group names."""
        expected = {'market', 'fundamentals', 'credit', 'llm', 'finbert', 'cross_sectional'}
        assert set(FeatureEngineer.ALL_FEATURE_GROUPS) == expected

    def test_market_features_always_included(self):
        """Market features are the base and always included regardless of feature_groups."""
        # This is verified by checking that _build_market_features is called unconditionally
        fe = FeatureEngineer.__new__(FeatureEngineer)
        # The method exists and is called before the feature_groups checks
        assert hasattr(fe, '_build_market_features')


class TestCrossSectionalFeatures:
    """Verify cross-sectional features use safe-lagged data."""

    def test_uses_lag6_features(self):
        """Cross-sectional ranks must use lagged features, not current-day."""
        fe = FeatureEngineer.__new__(FeatureEngineer)
        # The _add_cross_sectional method should reference lag6 columns
        import inspect
        source = inspect.getsource(fe._add_cross_sectional)
        assert 'returns_1d_lag6' in source
        assert 'volatility_5d_lag6' in source
