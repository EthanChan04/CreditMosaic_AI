"""
Key unit tests for FinBERT baseline module.
Focuses on sentiment analysis, batch processing, and comparison logic.
"""

import sys
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

# Skip if torch not available (heavy dependency)
pytestmark = pytest.mark.skipif('torch' not in sys.modules, reason="torch not installed")

try:
    from services.finbert_baseline import FinBERTModel, FinBERTResult, FinBERTComparator
except ImportError:
    pytest.skip("Cannot import finbert_baseline (torch missing)", allow_module_level=True)


@pytest.fixture
def finbert():
    """Create a FinBERT model (not initialized)."""
    return FinBERTModel("ProsusAI/finbert")


class TestFinBERTModel:
    """Test FinBERT model behavior."""

    def test_not_initialized_returns_defaults(self, finbert):
        """Uninitialized model should return neutral defaults."""
        result = finbert.analyze_sentiment("Apple reports strong earnings")
        assert result.sentiment_label == "neutral"
        assert result.sentiment_score == 0.0
        assert result.confidence == 1.0

    def test_not_initialized_batch_returns_defaults(self, finbert):
        """Uninitialized batch should return neutral defaults for all texts."""
        results = finbert.analyze_batch(["text1", "text2", "text3"])
        assert len(results) == 3
        for r in results:
            assert r.sentiment_label == "neutral"

    def test_initialized_flag(self, finbert):
        """initialized flag should be False by default."""
        assert finbert.initialized is False

    def test_sentiment_score_range(self, finbert):
        """Sentiment score should be in [-1, 1]."""
        result = finbert.analyze_sentiment("test")
        assert -1.0 <= result.sentiment_score <= 1.0

    def test_probabilities_sum_to_one(self, finbert):
        """Probabilities should sum to approximately 1."""
        result = finbert.analyze_sentiment("test")
        total = result.positive_score + result.negative_score + result.neutral_score
        assert abs(total - 1.0) < 0.01


class TestFinBERTComparator:
    """Test FinBERT vs LLM comparison logic."""

    def test_high_agreement(self, finbert):
        """Close sentiment scores should be high agreement."""
        comparator = FinBERTComparator(finbert)
        agreement = comparator._calculate_sentiment_agreement(-0.7, -0.65)
        assert agreement == "high"

    def test_medium_agreement(self, finbert):
        comparator = FinBERTComparator(finbert)
        agreement = comparator._calculate_sentiment_agreement(-0.7, -0.4)
        assert agreement == "medium"

    def test_low_agreement(self, finbert):
        comparator = FinBERTComparator(finbert)
        agreement = comparator._calculate_sentiment_agreement(-0.8, 0.5)
        assert agreement == "low"

    def test_comparison_without_llm_signal(self, finbert):
        """Comparison without LLM signal should only have FinBERT data."""
        comparator = FinBERTComparator(finbert)
        result = comparator.compare_with_llm("Apple reports earnings")
        assert "finbert" in result
        assert "llm" not in result

    def test_comparison_report_conclusion(self, finbert):
        """Report conclusion should reflect agreement quality."""
        comparator = FinBERTComparator(finbert)
        comparisons = [
            {"comparison": {"sentiment_difference": 0.1, "sentiment_agreement": "high"}},
            {"comparison": {"sentiment_difference": 0.15, "sentiment_agreement": "high"}},
            {"comparison": {"sentiment_difference": 0.2, "sentiment_agreement": "medium"}},
        ]
        report = comparator.generate_comparison_report(comparisons)
        assert report["total_samples"] == 3
        assert "conclusion" in report


class TestFinBERTResult:
    """Test FinBERTResult dataclass."""

    def test_result_fields(self):
        result = FinBERTResult(
            text="test", positive_score=0.7, negative_score=0.1,
            neutral_score=0.2, sentiment_label="positive",
            sentiment_score=0.7, confidence=0.7,
        )
        assert result.text == "test"
        assert result.sentiment_label == "positive"
        assert result.confidence == 0.7
