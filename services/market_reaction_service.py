"""
Market Reaction Service
Unified service for cross-market reaction analysis and lead-lag detection.

Integrates MarketReactionAnalyzer and LagAnalyzer to provide:
  - Post-news reaction computation across equity and credit markets
  - Lead-lag pattern analysis
  - Agreement validation between LLM-predicted and observed market impact
  - Reaction summary by ticker, event_type, and market_impact_type
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketReactionService:
    """Service layer for market reaction analysis."""

    def __init__(self, db_connection):
        self.db = db_connection

    def analyze_reactions(
        self,
        tickers: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        min_credit_risk_score: int = 0
    ) -> Dict[str, Any]:
        """Run full reaction analysis: compute reactions + agreement + summary."""
        from pipelines.reaction.market_reaction import MarketReactionAnalyzer

        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=180)

        analyzer = MarketReactionAnalyzer(self.db)
        reactions = analyzer.analyze_news_events(
            tickers, start_date, end_date, min_credit_risk_score
        )

        if not reactions:
            return {"error": "No reactions computed", "reactions": [], "agreement": {}}

        agreement = analyzer.compute_agreement_rate(reactions)
        summary_df = analyzer.get_reaction_summary(reactions)

        summary = []
        for _, row in summary_df.iterrows():
            item = {
                'news_id': int(row['news_id']),
                'ticker': row['ticker'],
                'event_date': str(row['event_date']),
                'event_type': row.get('event_type', ''),
                'llm_market_impact': row.get('llm_market_impact', ''),
                'observed_impact_type': row.get('observed_impact_type', ''),
                'windows': {}
            }
            for wname in ['0_1', '1_3', '3_5', '5_20']:
                prefix = f'{wname}_'
                item['windows'][wname] = {
                    k.replace(prefix, ''): v
                    for k, v in row.items()
                    if k.startswith(prefix) and pd.notna(v)
                }
            summary.append(item)

        return {
            "total_events": len(reactions),
            "agreement": agreement,
            "reactions": summary,
        }

    def analyze_lag(
        self,
        tickers: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Run lead-lag analysis between equity and credit markets."""
        from pipelines.reaction.lag_analyzer import LagAnalyzer

        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=180)

        analyzer = LagAnalyzer(self.db)
        return analyzer.compute_reaction_lag_stats(tickers, start_date, end_date)

    def get_reaction_by_ticker(
        self,
        ticker: str,
        days: int = 90
    ) -> List[Dict[str, Any]]:
        """Get recent market reactions for a single ticker."""
        from pipelines.reaction.market_reaction import MarketReactionAnalyzer

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        analyzer = MarketReactionAnalyzer(self.db)
        reactions = analyzer.analyze_news_events([ticker], start_date, end_date)

        summary = analyzer.get_reaction_summary(reactions)
        return summary.to_dict('records') if not summary.empty else []

    def get_reaction_by_news(self, news_id: int) -> Optional[Dict[str, Any]]:
        """Get market reaction for a specific news event."""
        sql = """
            SELECT ni.news_id, ni.ticker, ni.published_at, ni.title,
                   lns.sentiment_score, lns.credit_risk_score,
                   lns.event_type, lns.market_impact_type, lns.confidence
            FROM news_items ni
            JOIN llm_news_signals lns ON ni.news_id = lns.news_id
            WHERE ni.news_id = %s
        """
        with self.db.cursor() as cur:
            cur.execute(sql, (news_id,))
            columns = [desc[0] for desc in cur.description]
            row = cur.fetchone()
            if not row:
                return None
            news_data = dict(zip(columns, row))

        from pipelines.reaction.market_reaction import MarketReactionAnalyzer
        start = news_data['published_at'] - timedelta(days=30)
        end = news_data['published_at'] + timedelta(days=30)

        analyzer = MarketReactionAnalyzer(self.db)
        reactions = analyzer.analyze_news_events(
            [news_data['ticker']], start, end, min_credit_risk_score=0
        )

        for r in reactions:
            if r.news_id == news_id:
                windows_dict = r.windows
                return {
                    'news_id': r.news_id,
                    'ticker': r.ticker,
                    'event_date': str(r.event_date),
                    'event_type': r.event_type,
                    'llm_market_impact': r.llm_market_impact,
                    'observed_impact_type': r.observed_impact_type,
                    'windows': {k: v for k, v in windows_dict.items()},
                    'news_title': news_data.get('title', ''),
                    'credit_risk_score': news_data.get('credit_risk_score'),
                }

        return None

    def compare_llm_vs_actual(
        self,
        tickers: List[str],
        days: int = 90
    ) -> Dict[str, Any]:
        """Compare LLM-predicted market_impact_type against observed reactions."""
        from pipelines.reaction.market_reaction import MarketReactionAnalyzer

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        analyzer = MarketReactionAnalyzer(self.db)
        reactions = analyzer.analyze_news_events(tickers, start_date, end_date)

        if not reactions:
            return {"error": "No reactions to compare"}

        agreement = analyzer.compute_agreement_rate(reactions)

        confusion = {}
        for r in reactions:
            key = f"{r.llm_market_impact}->{r.observed_impact_type}"
            confusion[key] = confusion.get(key, 0) + 1

        return {
            "total_events": len(reactions),
            "agreement": agreement,
            "confusion_flow": confusion,
        }
