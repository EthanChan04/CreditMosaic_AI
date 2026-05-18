"""
Key unit tests for News Signal Extractor.
Focuses on signal extraction, consistency check, and quality analysis.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio

from services.news_signal_extractor import NewsSignalExtractor, NewsItem, LLMNewsSignal
from services.llm_provider import LLMProviderManager, LLMResponse


@pytest.fixture
def extractor():
    mgr = MagicMock(spec=LLMProviderManager)
    return NewsSignalExtractor(mgr)


@pytest.fixture
def sample_news():
    return NewsItem(
        news_id="n1",
        ticker="AAPL",
        title="Apple Reports 30% Profit Decline",
        body="Apple Inc. reported a 30% decline in quarterly profits.",
        source="Reuters",
        url="https://example.com",
        published_at=datetime(2026, 4, 28),
    )


VALID_SIGNAL_JSON = '''{
    "sentiment_score": -0.72,
    "credit_risk_score": 84,
    "event_type": "earnings_deterioration",
    "risk_horizon": "1m",
    "market_impact_type": "equity_leading",
    "evidence_spans": ["30% decline in quarterly profits"],
    "confidence": 0.88
}'''


class TestExtractSignal:
    """Test single news signal extraction."""

    def test_valid_signal_extracted(self, extractor, sample_news):
        """Valid LLM response should produce a signal."""
        extractor.llm_manager.generate_completion = AsyncMock(
            return_value=LLMResponse(content=VALID_SIGNAL_JSON, model="test")
        )
        signal = asyncio.run(extractor.extract_signal(sample_news))
        assert signal is not None
        assert signal.credit_risk_score == 84
        assert signal.event_type == "earnings_deterioration"

    def test_invalid_json_returns_none(self, extractor, sample_news):
        """Invalid JSON should return None."""
        extractor.llm_manager.generate_completion = AsyncMock(
            return_value=LLMResponse(content="not json", model="test")
        )
        signal = asyncio.run(extractor.extract_signal(sample_news))
        assert signal is None

    def test_out_of_range_score_returns_none(self, extractor, sample_news):
        """Out-of-range values should be rejected."""
        bad_json = VALID_SIGNAL_JSON.replace('"credit_risk_score": 84', '"credit_risk_score": 150')
        extractor.llm_manager.generate_completion = AsyncMock(
            return_value=LLMResponse(content=bad_json, model="test")
        )
        signal = asyncio.run(extractor.extract_signal(sample_news))
        assert signal is None


class TestConsistencyCheck:
    """Test dual-extraction consistency check."""

    def test_consistent_signals_return_higher_confidence(self, extractor, sample_news):
        """Two consistent signals should return the one with higher confidence."""
        signal1 = LLMNewsSignal(
            news_id="n1", ticker="AAPL", sentiment_score=-0.7,
            credit_risk_score=80, event_type="earnings_deterioration",
            risk_horizon="1m", market_impact_type="equity_leading",
            evidence_spans=["decline"], confidence=0.85,
            extracted_at=datetime.now(), llm_model="test",
        )
        signal2 = LLMNewsSignal(
            news_id="n1", ticker="AAPL", sentiment_score=-0.65,
            credit_risk_score=78, event_type="earnings_deterioration",
            risk_horizon="1m", market_impact_type="equity_leading",
            evidence_spans=["decline"], confidence=0.90,
            extracted_at=datetime.now(), llm_model="test",
        )
        score = extractor._calculate_consistency(signal1, signal2)
        assert score >= 0.8

    def test_inconsistent_signals_lower_score(self, extractor):
        """Two inconsistent signals should get a low consistency score."""
        signal1 = LLMNewsSignal(
            news_id="n1", ticker="AAPL", sentiment_score=-0.8,
            credit_risk_score=90, event_type="earnings_deterioration",
            risk_horizon="1m", market_impact_type="equity_leading",
            evidence_spans=["decline"], confidence=0.9,
            extracted_at=datetime.now(), llm_model="test",
        )
        signal2 = LLMNewsSignal(
            news_id="n1", ticker="AAPL", sentiment_score=0.5,
            credit_risk_score=20, event_type="neutral_or_irrelevant",
            risk_horizon="12m", market_impact_type="low_impact",
            evidence_spans=["growth"], confidence=0.7,
            extracted_at=datetime.now(), llm_model="test",
        )
        score = extractor._calculate_consistency(signal1, signal2)
        assert score < 0.5


class TestSignalQualityAnalysis:
    """Test signal quality analysis."""

    def test_empty_signals(self, extractor):
        result = extractor.analyze_signal_quality([])
        assert "error" in result

    def test_quality_distribution(self, extractor):
        signals = [
            LLMNewsSignal(
                news_id=f"n{i}", ticker="AAPL", sentiment_score=-0.5,
                credit_risk_score=score, event_type="earnings_deterioration",
                risk_horizon="1m", market_impact_type="equity_leading",
                evidence_spans=["text"], confidence=conf,
                extracted_at=datetime.now(), llm_model="test",
            )
            for i, (score, conf) in enumerate([
                (85, 0.9), (30, 0.7), (55, 0.5), (90, 0.95), (10, 0.8),
            ])
        ]
        result = extractor.analyze_signal_quality(signals)
        assert result["total_signals"] == 5
        # high confidence >= 0.8: 0.85, 0.9, 0.95 = 3
        assert result["confidence_distribution"]["high"] == 3
        assert result["risk_distribution"]["high"] == 2  # 85, 90
