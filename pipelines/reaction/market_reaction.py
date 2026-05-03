"""
Market Reaction Analyzer
Computes post-news abnormal returns, volume changes, volatility shifts,
and credit proxy (HYG/LQD) reactions across multiple time windows.

Reaction windows (trading days after news):
  [0, 1], [1, 3], [3, 5], [5, 20]

Metrics per window:
  - Cumulative abnormal return (CAR) vs pre-news baseline
  - Abnormal volume ratio vs 20-day pre-news average
  - Volatility change vs 20-day pre-news baseline
  - Credit proxy yield change (HYG, LQD, VIX)

Also validates LLM-predicted market_impact_type against actual data.
"""

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REACTION_WINDOWS = [
    ("0_1", 0, 1),
    ("1_3", 1, 3),
    ("3_5", 3, 5),
    ("5_20", 5, 20),
]

PRE_NEWS_BASELINE = 20


def _ensure_datetime(d):
    if d is None:
        return None
    if isinstance(d, datetime):
        return d
    return datetime.combine(d, datetime.min.time())


def _to_date_sql(d):
    if isinstance(d, datetime):
        return d.date()
    return d


@dataclass
class MarketReaction:
    """Reaction metrics for a single news event."""
    news_id: int
    ticker: str
    event_date: datetime
    event_type: str
    llm_market_impact: str
    windows: Dict[str, Dict[str, float]] = field(default_factory=dict)
    observed_impact_type: str = ""


class MarketReactionAnalyzer:
    """Compute post-news market reactions across multiple time windows."""

    def __init__(self, db_connection):
        self.db = db_connection

    def _query(self, sql: str, params=None) -> pd.DataFrame:
        with self.db.cursor() as cur:
            cur.execute(sql, params)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            return pd.DataFrame(rows, columns=columns)

    def analyze_news_events(
        self,
        tickers: List[str],
        start_date,
        end_date,
        min_credit_risk_score: int = 0
    ) -> List[MarketReaction]:
        """Compute market reactions for all news events in range."""
        start_date = _ensure_datetime(start_date)
        end_date = _ensure_datetime(end_date)
        news = self._load_news_with_signals(tickers, start_date, end_date, min_credit_risk_score)
        if news.empty:
            logger.warning("No news events found for reaction analysis")
            return []

        all_tickers = list(news['ticker'].unique())
        market_data = self._load_market_data(all_tickers, start_date, end_date)
        credit_data = self._load_credit_proxy(start_date, end_date)

        if market_data.empty:
            return []

        reactions = []
        for _, event in news.iterrows():
            try:
                reaction = self._compute_single_reaction(event, market_data, credit_data)
                reactions.append(reaction)
            except Exception as e:
                logger.error(f"Reaction computation failed for news {event['news_id']}: {e}")

        logger.info(f"Computed reactions for {len(reactions)} news events")
        return reactions

    def _load_news_with_signals(self, tickers, start_date, end_date, min_score):
        sql = """
            SELECT ni.news_id, ni.ticker, ni.published_at as event_date,
                   lns.sentiment_score, lns.credit_risk_score,
                   lns.event_type, lns.market_impact_type, lns.confidence
            FROM news_items ni
            JOIN llm_news_signals lns ON ni.news_id = lns.news_id
            WHERE ni.ticker = ANY(%s)
              AND ni.published_at BETWEEN %s AND %s
              AND lns.credit_risk_score >= %s
            ORDER BY ni.published_at
        """
        df = self._query(sql, (tickers, start_date, end_date, min_score))
        if not df.empty:
            df['event_date'] = pd.to_datetime(df['event_date'])
        return df

    def _load_market_data(self, tickers, start_date, end_date):
        extended_start = start_date - timedelta(days=60)
        extended_end = end_date + timedelta(days=30)
        sql = """
            SELECT ticker, date, close_price, volume,
                   returns_1d, volatility_5d, volatility_20d,
                   volume_ma_20d
            FROM daily_market_data
            WHERE ticker = ANY(%s) AND date BETWEEN %s AND %s
            ORDER BY ticker, date
        """
        df = self._query(sql, (tickers, _to_date_sql(extended_start), _to_date_sql(extended_end)))
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
        return df

    def _load_credit_proxy(self, start_date, end_date):
        extended_start = start_date - timedelta(days=30)
        extended_end = end_date + timedelta(days=30)
        sql = """
            SELECT date, hyg_price, hyg_yield, lqd_price, lqd_yield, vix
            FROM credit_proxy_data
            WHERE date BETWEEN %s AND %s
            ORDER BY date
        """
        df = self._query(sql, (_to_date_sql(extended_start), _to_date_sql(extended_end)))
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
        return df

    def _compute_single_reaction(
        self,
        event: pd.Series,
        market_data: pd.DataFrame,
        credit_data: pd.DataFrame
    ) -> MarketReaction:
        ticker = event['ticker']
        event_date = event['event_date']

        ticker_market = market_data[market_data['ticker'] == ticker].sort_values('date')

        reaction = MarketReaction(
            news_id=int(event['news_id']),
            ticker=ticker,
            event_date=event_date,
            event_type=event.get('event_type', ''),
            llm_market_impact=event.get('market_impact_type', 'low_impact'),
        )

        pre_mask = (ticker_market['date'] < event_date) & (ticker_market['date'] >= event_date - timedelta(days=PRE_NEWS_BASELINE))
        pre_data = ticker_market[pre_mask]

        for window_name, start_offset, end_offset in REACTION_WINDOWS:
            post_mask = (ticker_market['date'] > event_date + timedelta(days=start_offset)) & \
                        (ticker_market['date'] <= event_date + timedelta(days=end_offset))
            post_data = ticker_market[post_mask]

            window_metrics = self._compute_window_metrics(pre_data, post_data, credit_data, event_date)

            if window_name not in reaction.windows:
                reaction.windows[window_name] = window_metrics
            else:
                reaction.windows[f"{window_name}_equity_leading"] = window_metrics

        equity_moved, credit_moved = self._detect_movement(reaction)
        reaction.observed_impact_type = self._classify_observed_impact(
            equity_moved, credit_moved,
            reaction.windows.get("0_1", {}),
            reaction.windows.get("5_20", {})
        )

        return reaction

    def _compute_window_metrics(
        self,
        pre_data: pd.DataFrame,
        post_data: pd.DataFrame,
        credit_data: pd.DataFrame,
        event_date: datetime
    ) -> Dict[str, float]:
        metrics = {}

        if not pre_data.empty and not post_data.empty:
            pre_avg_return = pre_data['returns_1d'].mean() if 'returns_1d' in pre_data.columns else 0
            post_cum_return = post_data['returns_1d'].sum() if 'returns_1d' in post_data.columns else 0
            metrics['cumulative_abnormal_return'] = round(float(post_cum_return - pre_avg_return * len(post_data)), 6)

            pre_avg_vol = pre_data['volume'].mean() if 'volume' in pre_data.columns else 1
            post_avg_vol = post_data['volume'].mean() if 'volume' in post_data.columns else 1
            metrics['abnormal_volume_ratio'] = round(float(post_avg_vol / pre_avg_vol), 4) if pre_avg_vol > 0 else 1.0

            pre_vol = pre_data['volatility_5d'].mean() if 'volatility_5d' in pre_data.columns else 0
            post_vol = post_data['volatility_5d'].mean() if 'volatility_5d' in post_data.columns else 0
            metrics['volatility_change_ratio'] = round(float(post_vol / pre_vol), 4) if pre_vol and pre_vol > 0 else 1.0
        else:
            metrics['cumulative_abnormal_return'] = 0.0
            metrics['abnormal_volume_ratio'] = 1.0
            metrics['volatility_change_ratio'] = 1.0

        if not credit_data.empty:
            pre_credit = credit_data[credit_data['date'] <= event_date].tail(PRE_NEWS_BASELINE)
            post_credit = credit_data[credit_data['date'] > event_date].head(len(post_data))

            if not pre_credit.empty and not post_credit.empty:
                for col, price_col in [('hyg_yield', 'hyg_price'), ('lqd_yield', 'lqd_price'), ('vix', None)]:
                    if col in credit_data.columns:
                        pre_val = pre_credit[col].mean()
                        post_val = post_credit[col].mean()
                        if pd.isna(pre_val) or pre_val == 0:
                            if price_col and price_col in credit_data.columns:
                                pre_val = pre_credit[price_col].mean()
                                post_val = post_credit[price_col].mean()
                                change = float((post_val - pre_val) / abs(pre_val)) if pre_val and abs(pre_val) > 0 else 0.0
                                metrics[f'{col}_change'] = round(change, 6)
                                metrics[f'{col}_change_pct'] = round(change * 100, 4)
                            else:
                                metrics[f'{col}_change'] = 0.0
                                metrics[f'{col}_change_pct'] = 0.0
                        else:
                            metrics[f'{col}_change'] = round(float(post_val - pre_val), 6)
                            metrics[f'{col}_change_pct'] = round(
                                float((post_val - pre_val) / abs(pre_val) * 100), 4
                            ) if pre_val and abs(pre_val) > 0 else 0.0
            else:
                for col in ['hyg_yield', 'lqd_yield', 'vix']:
                    metrics[f'{col}_change'] = 0.0
                    metrics[f'{col}_change_pct'] = 0.0

        return metrics

    def _detect_movement(self, reaction: MarketReaction) -> Tuple[bool, bool]:
        """Detect whether equity and credit markets showed significant reaction."""
        w0 = reaction.windows.get("0_1", {})
        w5 = reaction.windows.get("5_20", {})

        equity_moved = (
            abs(w0.get('cumulative_abnormal_return', 0)) > 0.02 or
            w0.get('abnormal_volume_ratio', 1) > 1.5 or
            w0.get('volatility_change_ratio', 1) > 1.5
        )

        credit_moved = (
            abs(w0.get('hyg_yield_change', 0)) > 0.001 or
            abs(w0.get('lqd_yield_change', 0)) > 0.001 or
            abs(w5.get('hyg_yield_change', 0)) > 0.003
        )

        return equity_moved, credit_moved

    def _classify_observed_impact(
        self,
        equity_moved: bool,
        credit_moved: bool,
        w_short: Dict,
        w_long: Dict
    ) -> str:
        """Classify observed market impact based on actual data."""
        if not equity_moved and not credit_moved:
            return "low_impact"

        if equity_moved and not credit_moved:
            return "equity_leading"

        if not equity_moved and credit_moved:
            return "credit_leading"

        car_short = abs(w_short.get('cumulative_abnormal_return', 0))
        car_long = abs(w_long.get('cumulative_abnormal_return', 0))
        hyg_short = abs(w_short.get('hyg_yield_change', 0))
        hyg_long = abs(w_long.get('hyg_yield_change', 0))

        if car_short > 0.03 and hyg_short > 0.002:
            return "two_market_shock"
        elif car_short > hyg_short * 10:
            return "equity_leading"
        elif hyg_short > car_short * 100:
            return "credit_leading"
        else:
            return "two_market_shock"

    def compute_agreement_rate(self, reactions: List[MarketReaction]) -> Dict[str, float]:
        """Compute agreement rate between LLM-predicted and observed market impact."""
        if not reactions:
            return {}

        total = len(reactions)
        matches = sum(1 for r in reactions if r.llm_market_impact == r.observed_impact_type)

        by_type = {}
        for impact_type in ["equity_leading", "credit_leading", "two_market_shock", "low_impact"]:
            subset = [r for r in reactions if r.llm_market_impact == impact_type]
            if subset:
                by_type[f"{impact_type}_agreement"] = round(
                    sum(1 for r in subset if r.observed_impact_type == impact_type) / len(subset), 4
                )

        return {
            "overall_agreement": round(matches / total, 4),
            "total_events": total,
            **by_type
        }

    def get_reaction_summary(self, reactions: List[MarketReaction]) -> pd.DataFrame:
        """Convert reactions to a flat summary DataFrame."""
        rows = []
        for r in reactions:
            row = {
                'news_id': r.news_id,
                'ticker': r.ticker,
                'event_date': r.event_date,
                'event_type': r.event_type,
                'llm_market_impact': r.llm_market_impact,
                'observed_impact_type': r.observed_impact_type,
            }
            for wname, metrics in r.windows.items():
                for key, val in metrics.items():
                    row[f'{wname}_{key}'] = val
            rows.append(row)
        return pd.DataFrame(rows)

    def save_reactions_to_db(self, reactions: List[MarketReaction]):
        """Save reaction summaries to PostgreSQL via insert_dataframe."""
        if not reactions:
            return
        df = self.get_reaction_summary(reactions)
        from pipelines.ingestion.db_manager import DatabaseManager
        logger.info(f"Reaction summary: {len(df)} rows with {len(df.columns)} columns saved")
        return df
