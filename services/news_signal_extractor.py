"""
News Signal Extractor
Transforms unstructured news text into structured credit risk signals using LLMs
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import re

from services.llm_provider import (
    LLMProviderManager, LLMConfig, LLMResponseValidator,
    LLMResponse, OpenAIProvider
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    news_id: str
    ticker: str
    title: str
    body: str
    source: str
    url: str
    published_at: datetime


@dataclass
class LLMNewsSignal:
    news_id: str
    ticker: str
    sentiment_score: float
    credit_risk_score: int
    event_type: str
    risk_horizon: str
    market_impact_type: str
    evidence_spans: List[str]
    confidence: float
    extracted_at: datetime
    llm_model: str


class NewsSignalExtractor:

    def __init__(self, llm_manager: LLMProviderManager):
        self.llm_manager = llm_manager

        self.event_types = [
            "liquidity_pressure",
            "debt_refinancing",
            "earnings_deterioration",
            "litigation",
            "regulatory",
            "rating_change",
            "management_change",
            "supply_chain",
            "fraud_or_accounting",
            "neutral_or_irrelevant"
        ]

        self.risk_horizons = ["1w", "1m", "3m", "12m"]

        self.market_impact_types = [
            "equity_leading",
            "credit_leading",
            "two_market_shock",
            "low_impact"
        ]

        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        return f"""You are a professional financial news analyst specializing in extracting credit risk signals from corporate news.

**Task Requirements:**
1. Analyze news content and identify information related to credit risk
2. Evaluate the sentiment intensity and credit risk level
3. Determine the event type, risk time horizon, and market impact pattern
4. Extract key evidence snippets from the original text to support your judgment

**Event Types (choose one):**
{', '.join(self.event_types)}

**Risk Time Horizons (choose one):**
{', '.join(self.risk_horizons)}

**Market Impact Types (choose one):**
{', '.join(self.market_impact_types)}

**Output Format:**
Output a JSON object with the following fields:
- sentiment_score: News sentiment score, range -1 to 1 (-1 = extremely negative, 1 = extremely positive)
- credit_risk_score: Credit risk intensity, range 0 to 100 (0 = no risk, 100 = extremely high risk)
- event_type: Event type, must be one of the enumerated values above
- risk_horizon: Risk time horizon, must be one of the enumerated values above
- market_impact_type: Market impact type, must be one of the enumerated values above
- evidence_spans: List of original text snippets supporting the judgment, must come from the source text
- confidence: Confidence level, range 0 to 1, indicating certainty of the analysis

**Judgment Principles:**
1. Distinguish between general negative sentiment and genuine credit risk
2. For the same news item, you may run twice and compare consistency
3. If news is weakly related to credit risk, output neutral_or_irrelevant
4. Evidence snippets must be accurately quoted from the original text
5. High-risk news must include non-empty evidence_spans

**Example:**
Input: "The company warned that refinancing remains uncertain due to deteriorating market conditions. Management expects free cash flow to remain negative next quarter."

Output:
{{
  "sentiment_score": -0.72,
  "credit_risk_score": 84,
  "event_type": "debt_refinancing",
  "risk_horizon": "3m",
  "market_impact_type": "credit_leading",
  "evidence_spans": [
    "The company warned that refinancing remains uncertain due to deteriorating market conditions.",
    "Management expects free cash flow to remain negative next quarter."
  ],
  "confidence": 0.86
}}
"""

    def _build_user_prompt(self, news_item: NewsItem) -> str:
        return f"""Analyze the following news for credit risk:

**Company Ticker:** {news_item.ticker}

**News Title:** {news_item.title}

**News Body:** {news_item.body}

**Published At:** {news_item.published_at}

**News Source:** {news_item.source}

Please output the credit risk analysis in JSON format. Ensure:
1. Sentiment score is between -1 and 1
2. Credit risk score is between 0 and 100
3. Confidence is between 0 and 1
4. All enum fields use the specified values
5. Evidence snippets must come from the original text
"""

    async def extract_signal(
        self,
        news_item: NewsItem,
        provider_name: Optional[str] = None,
        model: Optional[str] = None
    ) -> Optional[LLMNewsSignal]:
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self._build_user_prompt(news_item)}
            ]

            response = await self.llm_manager.generate_completion(
                messages,
                provider_name=provider_name,
                temperature=0.3,
                max_tokens=1500
            )

            signal_data = LLMResponseValidator.validate_credit_risk_signal(response.content)

            if not signal_data:
                logger.error(f"Signal validation failed for news {news_item.news_id}")
                return None

            signal = LLMNewsSignal(
                news_id=news_item.news_id,
                ticker=news_item.ticker,
                sentiment_score=signal_data["sentiment_score"],
                credit_risk_score=signal_data["credit_risk_score"],
                event_type=signal_data["event_type"],
                risk_horizon=signal_data["risk_horizon"],
                market_impact_type=signal_data["market_impact_type"],
                evidence_spans=signal_data["evidence_spans"],
                confidence=signal_data["confidence"],
                extracted_at=datetime.now(),
                llm_model=response.model
            )

            logger.info(
                f"Signal extracted: {news_item.ticker} | "
                f"risk_score: {signal.credit_risk_score} | "
                f"event: {signal.event_type} | "
                f"confidence: {signal.confidence:.2f}"
            )

            return signal

        except Exception as e:
            logger.error(f"Signal extraction failed for {news_item.news_id}: {e}")
            return None

    async def extract_batch_signals(
        self,
        news_items: List[NewsItem],
        provider_name: Optional[str] = None,
        max_concurrent: int = 5
    ) -> List[LLMNewsSignal]:
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []

        async def extract_with_semaphore(news_item: NewsItem):
            async with semaphore:
                signal = await self.extract_signal(news_item, provider_name)
                if signal:
                    results.append(signal)

        tasks = [extract_with_semaphore(item) for item in news_items]
        await asyncio.gather(*tasks)

        logger.info(f"Batch extraction complete: {len(results)}/{len(news_items)} succeeded")
        return results

    async def extract_with_consistency_check(
        self,
        news_item: NewsItem,
        provider_name: Optional[str] = None,
        consistency_threshold: float = 0.8
    ) -> Optional[LLMNewsSignal]:
        try:
            signal1 = await self.extract_signal(news_item, provider_name)
            if not signal1:
                return None

            await asyncio.sleep(1)

            signal2 = await self.extract_signal(news_item, provider_name)
            if not signal2:
                return signal1

            consistency_score = self._calculate_consistency(signal1, signal2)

            if consistency_score >= consistency_threshold:
                logger.info(
                    f"Consistency check passed: {news_item.news_id} | "
                    f"score: {consistency_score:.2f}"
                )
                return signal1 if signal1.confidence >= signal2.confidence else signal2
            else:
                logger.warning(
                    f"Consistency check failed: {news_item.news_id} | "
                    f"score: {consistency_score:.2f}"
                )
                return signal1 if signal1.confidence >= signal2.confidence else signal2

        except Exception as e:
            logger.error(f"Consistency extraction failed for {news_item.news_id}: {e}")
            return None

    def _calculate_consistency(self, signal1: LLMNewsSignal, signal2: LLMNewsSignal) -> float:
        scores = []

        sentiment_diff = abs(signal1.sentiment_score - signal2.sentiment_score)
        sentiment_consistency = 1.0 if sentiment_diff < 0.2 else 0.5 if sentiment_diff < 0.3 else 0.0
        scores.append(sentiment_consistency)

        risk_diff = abs(signal1.credit_risk_score - signal2.credit_risk_score)
        risk_consistency = 1.0 if risk_diff < 15 else 0.5 if risk_diff < 25 else 0.0
        scores.append(risk_consistency)

        event_consistency = 1.0 if signal1.event_type == signal2.event_type else 0.3
        scores.append(event_consistency)

        impact_consistency = 1.0 if signal1.market_impact_type == signal2.market_impact_type else 0.3
        scores.append(impact_consistency)

        evidence1 = set(' '.join(signal1.evidence_spans).split())
        evidence2 = set(' '.join(signal2.evidence_spans).split())
        if evidence1 or evidence2:
            intersection = len(evidence1 & evidence2)
            union = len(evidence1 | evidence2)
            evidence_consistency = intersection / union if union > 0 else 1.0
        else:
            evidence_consistency = 1.0
        scores.append(evidence_consistency)

        return sum(scores) / len(scores)

    def analyze_signal_quality(self, signals: List[LLMNewsSignal]) -> Dict[str, Any]:
        total = len(signals)

        if total == 0:
            return {"error": "No signals to analyze"}

        high_confidence = len([s for s in signals if s.confidence >= 0.8])
        medium_confidence = len([s for s in signals if 0.6 <= s.confidence < 0.8])
        low_confidence = len([s for s in signals if s.confidence < 0.6])

        event_distribution = {}
        for signal in signals:
            event_type = signal.event_type
            event_distribution[event_type] = event_distribution.get(event_type, 0) + 1

        risk_scores = [s.credit_risk_score for s in signals]
        high_risk = len([s for s in risk_scores if s >= 70])
        medium_risk = len([s for s in risk_scores if 40 <= s < 70])
        low_risk = len([s for s in risk_scores if s < 40])

        avg_evidence = sum(len(s.evidence_spans) for s in signals) / total

        return {
            "total_signals": total,
            "confidence_distribution": {
                "high": high_confidence,
                "medium": medium_confidence,
                "low": low_confidence
            },
            "risk_distribution": {
                "high": high_risk,
                "medium": medium_risk,
                "low": low_risk
            },
            "event_type_distribution": event_distribution,
            "avg_evidence_per_signal": round(avg_evidence, 2),
            "model_usage": {}
        }

    def get_extraction_stats(self, db_connection) -> Dict[str, Any]:
        try:
            stats_sql = """
                SELECT llm_model, COUNT(*) as count,
                       AVG(credit_risk_score) as avg_risk_score,
                       AVG(confidence) as avg_confidence
                FROM llm_news_signals
                GROUP BY llm_model
            """
            stats = db_connection.execute(stats_sql).fetchall()

            total_sql = "SELECT COUNT(*) as total FROM llm_news_signals"
            total = db_connection.execute(total_sql).fetchone()['total']

            return {
                "total_signals": total,
                "model_stats": [dict(row) for row in stats]
            }

        except Exception as e:
            logger.error(f"Failed to get extraction stats: {e}")
            return {}


class FinBERTBaseline:

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.initialized = False

    def initialize(self):
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            if self.model_path:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

            self.model.eval()
            self.initialized = True
            logger.info("FinBERT model initialized successfully")

        except Exception as e:
            logger.error(f"FinBERT initialization failed: {e}")
            self.initialized = False

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        if not self.initialized:
            logger.warning("FinBERT not initialized, returning defaults")
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

        try:
            import torch

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)

            sentiment_scores = {
                "positive": probabilities[0][0].item(),
                "negative": probabilities[0][1].item(),
                "neutral": probabilities[0][2].item()
            }

            return sentiment_scores

        except Exception as e:
            logger.error(f"FinBERT sentiment analysis failed: {e}")
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

    def get_sentiment_score(self, text: str) -> float:
        scores = self.analyze_sentiment(text)
        return scores["positive"] - scores["negative"]


class SignalComparisonAnalyzer:

    def __init__(self, llm_extractor: NewsSignalExtractor, finbert_baseline: FinBERTBaseline):
        self.llm_extractor = llm_extractor
        self.finbert_baseline = finbert_baseline

    async def compare_analysis(self, news_item: NewsItem) -> Dict[str, Any]:
        llm_signal = await self.llm_extractor.extract_signal(news_item)
        if not llm_signal:
            return {"error": "LLM analysis failed"}

        finbert_score = self.finbert_baseline.get_sentiment_score(
            f"{news_item.title} {news_item.body}"
        )

        comparison = {
            "news_id": news_item.news_id,
            "ticker": news_item.ticker,
            "llm_analysis": {
                "sentiment_score": llm_signal.sentiment_score,
                "credit_risk_score": llm_signal.credit_risk_score,
                "event_type": llm_signal.event_type,
                "confidence": llm_signal.confidence
            },
            "finbert_baseline": {
                "sentiment_score": finbert_score
            },
            "sentiment_comparison": {
                "llm_sentiment": llm_signal.sentiment_score,
                "finbert_sentiment": finbert_score,
                "difference": abs(llm_signal.sentiment_score - finbert_score),
                "agreement": "high" if abs(llm_signal.sentiment_score - finbert_score) < 0.3 else "low"
            }
        }

        return comparison

    def batch_compare_analysis(self, news_items: List[NewsItem]) -> List[Dict[str, Any]]:
        comparisons = []

        for news_item in news_items:
            try:
                comparison = asyncio.run(self.compare_analysis(news_item))
                comparisons.append(comparison)
            except Exception as e:
                logger.error(f"Comparison analysis failed for {news_item.news_id}: {e}")

        return comparisons


if __name__ == "__main__":
    async def main():
        llm_manager = LLMProviderManager()

        openai_config = LLMConfig(
            provider="openai",
            api_key="your_api_key_here",
            model_name="gpt-4-turbo-preview"
        )
        llm_manager.add_provider("openai", OpenAIProvider(openai_config))

        extractor = NewsSignalExtractor(llm_manager)

        test_news = NewsItem(
            news_id="test_001",
            ticker="AAPL",
            title="Apple Reports 30% Profit Decline, Announces Massive Layoffs",
            body="Apple Inc. today reported a 30% decline in quarterly profits, significantly below analyst expectations. The company also announced plans to lay off 10% of its workforce due to economic challenges and reduced consumer spending.",
            source="Financial Times",
            url="https://example.com/apple-earnings",
            published_at=datetime.now()
        )

        signal = await extractor.extract_signal(test_news)

        if signal:
            print("Extraction successful:")
            print(f"  Risk score: {signal.credit_risk_score}")
            print(f"  Event type: {signal.event_type}")
            print(f"  Confidence: {signal.confidence}")
            print(f"  Evidence: {signal.evidence_spans}")
        else:
            print("Extraction failed")

    asyncio.run(main())
