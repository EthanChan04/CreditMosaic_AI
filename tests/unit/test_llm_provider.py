"""
Unit tests for LLM Provider module.
Covers: LLMResponseValidator, LLMProviderFactory, LLMProviderManager, OpenAIProvider.
"""

import asyncio
import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from services.llm_provider import (
    LLMConfig,
    LLMResponse,
    LLMResponseValidator,
    LLMProviderFactory,
    LLMProviderManager,
    OpenAIProvider,
    QwenProvider,
    DeepSeekProvider,
    BaseLLMProvider,
)


# ── LLMResponseValidator._extract_json ──────────────────────────────────────


class TestExtractJson:
    """Test JSON extraction from LLM text output."""

    def test_pure_json_object(self):
        text = '{"sentiment_score": -0.5, "credit_risk_score": 60}'
        result = LLMResponseValidator._extract_json(text)
        data = json.loads(result)
        assert data["sentiment_score"] == -0.5

    def test_json_in_markdown_code_fence(self):
        text = '```json\n{"sentiment_score": 0.3, "credit_risk_score": 20}\n```'
        result = LLMResponseValidator._extract_json(text)
        data = json.loads(result)
        assert data["credit_risk_score"] == 20

    def test_json_in_plain_code_fence(self):
        text = '```\n{"sentiment_score": 0.0}\n```'
        result = LLMResponseValidator._extract_json(text)
        data = json.loads(result)
        assert "sentiment_score" in data

    def test_json_buried_in_text(self):
        text = 'Here is my analysis:\n{"sentiment_score": -0.8}\nDone.'
        result = LLMResponseValidator._extract_json(text)
        data = json.loads(result)
        assert data["sentiment_score"] == -0.8

    def test_empty_string(self):
        result = LLMResponseValidator._extract_json("")
        assert result == ""

    def test_whitespace_only(self):
        result = LLMResponseValidator._extract_json("   \n  ")
        assert result.strip() == ""


# ── LLMResponseValidator.validate_credit_risk_signal ─────────────────────────


class TestValidateCreditRiskSignal:
    """Test structured signal validation."""

    VALID_SIGNAL = json.dumps({
        "sentiment_score": -0.72,
        "credit_risk_score": 84,
        "event_type": "debt_refinancing",
        "risk_horizon": "3m",
        "market_impact_type": "credit_leading",
        "evidence_spans": ["The company warned that refinancing remains uncertain."],
        "confidence": 0.86,
    })

    def test_valid_signal_accepted(self):
        result = LLMResponseValidator.validate_credit_risk_signal(self.VALID_SIGNAL)
        assert result is not None
        assert result["sentiment_score"] == -0.72
        assert result["credit_risk_score"] == 84
        assert result["event_type"] == "debt_refinancing"

    def test_missing_required_field_rejected(self):
        data = json.loads(self.VALID_SIGNAL)
        del data["confidence"]
        result = LLMResponseValidator.validate_credit_risk_signal(json.dumps(data))
        assert result is None

    def test_sentiment_score_out_of_range(self):
        data = json.loads(self.VALID_SIGNAL)
        data["sentiment_score"] = 1.5
        result = LLMResponseValidator.validate_credit_risk_signal(json.dumps(data))
        assert result is None

    def test_sentiment_score_below_range(self):
        data = json.loads(self.VALID_SIGNAL)
        data["sentiment_score"] = -1.5
        result = LLMResponseValidator.validate_credit_risk_signal(json.dumps(data))
        assert result is None

    def test_credit_risk_score_not_integer(self):
        data = json.loads(self.VALID_SIGNAL)
        data["credit_risk_score"] = 50.5
        result = LLMResponseValidator.validate_credit_risk_signal(json.dumps(data))
        assert result is None

    def test_credit_risk_score_above_100(self):
        data = json.loads(self.VALID_SIGNAL)
        data["credit_risk_score"] = 101
        result = LLMResponseValidator.validate_credit_risk_signal(json.dumps(data))
        assert result is None

    def test_credit_risk_score_below_0(self):
        data = json.loads(self.VALID_SIGNAL)
        data["credit_risk_score"] = -1
        result = LLMResponseValidator.validate_credit_risk_signal(json.dumps(data))
        assert result is None

    def test_invalid_event_type_rejected(self):
        data = json.loads(self.VALID_SIGNAL)
        data["event_type"] = "bankruptcy_imminent"
        result = LLMResponseValidator.validate_credit_risk_signal(json.dumps(data))
        assert result is None

    def test_invalid_risk_horizon_rejected(self):
        data = json.loads(self.VALID_SIGNAL)
        data["risk_horizon"] = "6m"
        result = LLMResponseValidator.validate_credit_risk_signal(json.dumps(data))
        assert result is None

    def test_invalid_market_impact_type_rejected(self):
        data = json.loads(self.VALID_SIGNAL)
        data["market_impact_type"] = "unknown"
        result = LLMResponseValidator.validate_credit_risk_signal(json.dumps(data))
        assert result is None

    def test_empty_evidence_spans_rejected(self):
        data = json.loads(self.VALID_SIGNAL)
        data["evidence_spans"] = []
        result = LLMResponseValidator.validate_credit_risk_signal(json.dumps(data))
        assert result is None

    def test_confidence_out_of_range(self):
        data = json.loads(self.VALID_SIGNAL)
        data["confidence"] = 1.5
        result = LLMResponseValidator.validate_credit_risk_signal(json.dumps(data))
        assert result is None

    def test_boundary_values_accepted(self):
        data = json.loads(self.VALID_SIGNAL)
        data["sentiment_score"] = -1.0
        data["credit_risk_score"] = 0
        data["confidence"] = 0.0
        result = LLMResponseValidator.validate_credit_risk_signal(json.dumps(data))
        assert result is not None

    def test_upper_boundary_values_accepted(self):
        data = json.loads(self.VALID_SIGNAL)
        data["sentiment_score"] = 1.0
        data["credit_risk_score"] = 100
        data["confidence"] = 1.0
        result = LLMResponseValidator.validate_credit_risk_signal(json.dumps(data))
        assert result is not None

    def test_malformed_json_rejected(self):
        result = LLMResponseValidator.validate_credit_risk_signal("not json at all")
        assert result is None

    def test_all_valid_event_types(self):
        valid_types = [
            "liquidity_pressure", "debt_refinancing", "earnings_deterioration",
            "litigation", "regulatory", "rating_change", "management_change",
            "supply_chain", "fraud_or_accounting", "neutral_or_irrelevant",
        ]
        for etype in valid_types:
            data = json.loads(self.VALID_SIGNAL)
            data["event_type"] = etype
            result = LLMResponseValidator.validate_credit_risk_signal(json.dumps(data))
            assert result is not None, f"event_type={etype} should be accepted"

    def test_all_valid_risk_horizons(self):
        for horizon in ["1w", "1m", "3m", "12m"]:
            data = json.loads(self.VALID_SIGNAL)
            data["risk_horizon"] = horizon
            result = LLMResponseValidator.validate_credit_risk_signal(json.dumps(data))
            assert result is not None, f"risk_horizon={horizon} should be accepted"

    def test_all_valid_market_impact_types(self):
        for mtype in ["equity_leading", "credit_leading", "two_market_shock", "low_impact"]:
            data = json.loads(self.VALID_SIGNAL)
            data["market_impact_type"] = mtype
            result = LLMResponseValidator.validate_credit_risk_signal(json.dumps(data))
            assert result is not None, f"market_impact_type={mtype} should be accepted"

    def test_signal_in_markdown_fence(self):
        wrapped = f"```json\n{self.VALID_SIGNAL}\n```"
        result = LLMResponseValidator.validate_credit_risk_signal(wrapped)
        assert result is not None

    def test_signal_with_surrounding_text(self):
        wrapped = f"Here is my analysis:\n{self.VALID_SIGNAL}\nEnd of report."
        result = LLMResponseValidator.validate_credit_risk_signal(wrapped)
        assert result is not None


# ── LLMProviderFactory ──────────────────────────────────────────────────────


class TestLLMProviderFactory:
    """Test provider factory creation."""

    def test_create_openai_provider(self):
        config = LLMConfig(provider="openai", api_key="sk-test", model_name="gpt-4")
        provider = LLMProviderFactory.create_provider("openai", config)
        assert isinstance(provider, OpenAIProvider)

    def test_create_qwen_provider(self):
        config = LLMConfig(provider="qwen", api_key="sk-test", model_name="qwen-turbo")
        provider = LLMProviderFactory.create_provider("qwen", config)
        assert isinstance(provider, QwenProvider)

    def test_create_deepseek_provider(self):
        config = LLMConfig(provider="deepseek", api_key="sk-test", model_name="deepseek-chat")
        provider = LLMProviderFactory.create_provider("deepseek", config)
        assert isinstance(provider, DeepSeekProvider)

    def test_case_insensitive_provider(self):
        config = LLMConfig(provider="OpenAI", api_key="sk-test", model_name="gpt-4")
        provider = LLMProviderFactory.create_provider("OpenAI", config)
        assert isinstance(provider, OpenAIProvider)

    def test_unsupported_provider_raises(self):
        config = LLMConfig(provider="unknown", api_key="sk-test", model_name="m")
        with pytest.raises(ValueError, match="Unsupported"):
            LLMProviderFactory.create_provider("unknown", config)


# ── LLMProviderManager ──────────────────────────────────────────────────────


class TestLLMProviderManager:
    """Test provider manager lifecycle."""

    def _make_mock_provider(self, name="test"):
        class MockProvider(BaseLLMProvider):
            async def generate_completion(self, prompt, system_prompt=None):
                return LLMResponse(content="ok", model=name)

            async def generate_chat_completion(self, messages, temperature=None, max_tokens=None):
                return LLMResponse(content="ok", model=name)

        config = LLMConfig(provider="test", api_key="k", model_name=name)
        return MockProvider(config)

    def test_add_and_get_provider(self):
        mgr = LLMProviderManager()
        provider = self._make_mock_provider("p1")
        mgr.add_provider("p1", provider)
        assert mgr.get_provider("p1") is provider

    def test_default_provider_set_on_first_add(self):
        mgr = LLMProviderManager()
        provider = self._make_mock_provider()
        mgr.add_provider("p1", provider)
        assert mgr.default_provider == "p1"

    def test_set_default_provider(self):
        mgr = LLMProviderManager()
        p1 = self._make_mock_provider("p1")
        p2 = self._make_mock_provider("p2")
        mgr.add_provider("p1", p1)
        mgr.add_provider("p2", p2, set_as_default=True)
        assert mgr.default_provider == "p2"

    def test_set_default_raises_for_unknown(self):
        mgr = LLMProviderManager()
        with pytest.raises(ValueError, match="not registered"):
            mgr.set_default_provider("nonexistent")

    def test_list_providers(self):
        mgr = LLMProviderManager()
        mgr.add_provider("a", self._make_mock_provider("a"))
        mgr.add_provider("b", self._make_mock_provider("b"))
        assert set(mgr.list_providers()) == {"a", "b"}

    def test_get_provider_returns_none_for_missing(self):
        mgr = LLMProviderManager()
        assert mgr.get_provider("missing") is None

    def test_get_provider_returns_default(self):
        mgr = LLMProviderManager()
        p = self._make_mock_provider()
        mgr.add_provider("p1", p)
        assert mgr.get_provider() is p

    def test_generate_completion_calls_provider(self):
        mgr = LLMProviderManager()
        p = self._make_mock_provider()
        mgr.add_provider("p1", p)
        resp = asyncio.run(mgr.generate_completion([{"role": "user", "content": "hi"}]))
        assert resp.content == "ok"

    def test_generate_completion_raises_for_missing(self):
        mgr = LLMProviderManager()
        with pytest.raises(RuntimeError, match="No LLM provider"):
            asyncio.run(mgr.generate_completion([{"role": "user", "content": "hi"}]))


# ── OpenAIProvider ───────────────────────────────────────────────────────────


class TestOpenAIProvider:
    """Test OpenAI-compatible provider with mocked HTTP."""

    def test_headers_contain_bearer_token(self):
        config = LLMConfig(provider="openai", api_key="sk-test-key", model_name="gpt-4")
        provider = OpenAIProvider(config)
        assert provider.headers["Authorization"] == "Bearer sk-test-key"

    def test_base_url_defaults_to_openai(self):
        config = LLMConfig(provider="openai", api_key="sk-test", model_name="gpt-4")
        provider = OpenAIProvider(config)
        assert provider.base_url == "https://api.openai.com/v1"

    def test_base_url_override(self):
        config = LLMConfig(provider="openai", api_key="sk-test", model_name="gpt-4",
                           base_url="https://custom.api.com/v1")
        provider = OpenAIProvider(config)
        assert provider.base_url == "https://custom.api.com/v1"

    def test_qwen_default_base_url(self):
        config = LLMConfig(provider="qwen", api_key="sk-test", model_name="qwen-turbo")
        provider = QwenProvider(config)
        assert "dashscope" in provider.base_url

    def test_deepseek_default_base_url(self):
        config = LLMConfig(provider="deepseek", api_key="sk-test", model_name="deepseek-chat")
        provider = DeepSeekProvider(config)
        assert "deepseek" in provider.base_url


# ── LLMConfig ───────────────────────────────────────────────────────────────


class TestLLMConfig:
    """Test config dataclass defaults."""

    def test_default_values(self):
        config = LLMConfig()
        assert config.temperature == 0.1
        assert config.max_tokens == 1000
        assert config.timeout == 30

    def test_custom_values(self):
        config = LLMConfig(provider="openai", api_key="sk-test", model_name="gpt-4",
                           temperature=0.5, max_tokens=2000)
        assert config.temperature == 0.5
        assert config.max_tokens == 2000


# ── LLMResponse ─────────────────────────────────────────────────────────────


class TestLLMResponse:
    """Test response dataclass."""

    def test_successful_response(self):
        resp = LLMResponse(content="hello", model="gpt-4", tokens_used=100)
        assert resp.error is None
        assert resp.tokens_used == 100

    def test_error_response(self):
        resp = LLMResponse(content="", model="gpt-4", error="rate limited")
        assert resp.error == "rate limited"
        assert resp.content == ""
