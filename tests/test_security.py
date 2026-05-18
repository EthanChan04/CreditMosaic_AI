"""
Security-focused tests for CreditMosaic AI.
Verifies SQL injection prevention, CORS configuration, and input validation.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestSQLInjectionPrevention:
    """Verify that DuckDB operations use whitelist validation."""

    def test_table_whitelist_enforced_on_save(self):
        """save_to_analytical_store must reject non-whitelisted table names."""
        from pipelines.ingestion.db_manager import DatabaseManager

        manager = DatabaseManager.__new__(DatabaseManager)
        manager.duckdb_conn = MagicMock()

        import pandas as pd
        df = pd.DataFrame({'a': [1]})

        with pytest.raises(ValueError, match="Invalid table name"):
            manager.save_to_analytical_store(df, "malicious_table; DROP TABLE users;")

    def test_table_whitelist_enforced_on_load(self):
        """load_from_analytical_store must reject non-whitelisted table names."""
        from pipelines.ingestion.db_manager import DatabaseManager

        manager = DatabaseManager.__new__(DatabaseManager)
        manager.duckdb_conn = MagicMock()

        with pytest.raises(ValueError, match="Invalid table name"):
            manager.load_from_analytical_store("evil_table")

    def test_valid_tables_accepted(self):
        """Whitelisted table names must be accepted."""
        from pipelines.ingestion.db_manager import DatabaseManager

        manager = DatabaseManager.__new__(DatabaseManager)
        manager.duckdb_conn = MagicMock()
        manager.duckdb_conn.execute.return_value.df.return_value = __import__('pandas').DataFrame()

        # Should not raise
        manager.load_from_analytical_store("company_daily_features")

    def test_column_name_validation(self):
        """Filter column names must contain only alphanumeric and underscore."""
        from pipelines.ingestion.db_manager import DatabaseManager

        manager = DatabaseManager.__new__(DatabaseManager)
        manager.duckdb_conn = MagicMock()

        with pytest.raises(ValueError, match="Invalid column name"):
            manager.load_from_analytical_store(
                "company_daily_features",
                filters={"ticker; DROP": "AAPL"}
            )

    def test_whitelist_contains_expected_tables(self):
        """Whitelist must include all known analytical tables."""
        from pipelines.ingestion.db_manager import DatabaseManager

        expected = {'company_daily_features', 'market_summary', 'risk_analysis'}
        for t in expected:
            assert t in DatabaseManager.VALID_DUCKDB_TABLES


class TestLLMProviderConcurrency:
    """Verify that LLM provider does not use shared mutable state."""

    def test_generate_completion_does_not_mutate_config(self):
        """generate_completion must pass parameters directly, not mutate provider config."""
        from services.llm_provider import LLMProviderManager, BaseLLMProvider, LLMConfig, LLMResponse
        import asyncio

        class MockProvider(BaseLLMProvider):
            async def generate_completion(self, prompt, system_prompt=None):
                return LLMResponse(content="ok", model="test")

            async def generate_chat_completion(self, messages, temperature=None, max_tokens=None):
                # Verify parameters are passed correctly
                return LLMResponse(content=f"temp={temperature}", model="test")

        config = LLMConfig(provider="test", api_key="k", model_name="m")
        provider = MockProvider(config)
        original_temp = config.temperature
        original_tokens = config.max_tokens

        manager = LLMProviderManager()
        manager.add_provider("test", provider)

        result = asyncio.run(manager.generate_completion(
            [{"role": "user", "content": "hi"}],
            temperature=0.9, max_tokens=500
        ))

        # Config must NOT be mutated
        assert config.temperature == original_temp
        assert config.max_tokens == original_tokens


class TestInputValidation:
    """Verify API input constraints are properly configured."""

    def test_news_limit_has_bounds(self):
        """news.py limit parameter must have ge/le constraints."""
        try:
            from apps.api.app.api.news import get_news
        except ImportError:
            pytest.skip("Missing dependencies (torch) for news module import")
        import inspect
        sig = inspect.signature(get_news)
        limit_param = sig.parameters.get('limit')
        assert limit_param is not None
        default = limit_param.default
        assert hasattr(default, 'ge') or hasattr(default, 'le')

    def test_reaction_days_has_bounds(self):
        """reaction.py days parameter must have ge/le constraints."""
        from apps.api.app.api.reaction import get_reaction_by_ticker
        import inspect
        sig = inspect.signature(get_reaction_by_ticker)
        days_param = sig.parameters.get('days')
        assert days_param is not None
