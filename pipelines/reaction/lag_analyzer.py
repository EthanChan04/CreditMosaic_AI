"""
Reaction Lag Analyzer
Analyzes the lead-lag relationship between equity market reactions and credit
proxy (HYG/LQD) reactions following news events.

Key analyses:
  - Cross-correlation: equity vs credit reaction across lag intervals
  - Lead-lag classification: which market moves first
  - Lead-lag distribution by event_type and market_impact_type
  - Average lead time (trading days) for equity-leading vs credit-leading patterns
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _to_date_sql(d):
    if isinstance(d, datetime):
        return d.date()
    return d


class LagAnalyzer:
    """Analyze equity/credit market lead-lag after news events."""

    LAG_WINDOWS = list(range(-10, 11))

    def __init__(self, db_connection):
        self.db = db_connection

    def _query(self, sql: str, params=None) -> pd.DataFrame:
        with self.db.cursor() as cur:
            cur.execute(sql, params)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            return pd.DataFrame(rows, columns=columns)

    def compute_cross_correlation(
        self,
        tickers: List[str],
        start_date,
        end_date
    ) -> Dict[str, Any]:
        """Compute cross-correlation between equity returns and credit proxy changes across lag windows.

        Negative lag = equity leads credit (equity return at t, credit change at t+lag).
        Positive lag = credit leads equity (credit change at t, equity return at t+lag).
        """
        from pipelines.reaction.market_reaction import _ensure_datetime, _to_date_sql
        start_date = _ensure_datetime(start_date)
        end_date = _ensure_datetime(end_date)

        market = self._load_market_returns(tickers, start_date, end_date)
        credit = self._load_credit_changes(start_date, end_date)

        if market.empty or credit.empty:
            return {"error": "Insufficient data for cross-correlation"}

        market_daily = market.groupby('date')['returns_1d'].mean()
        market_daily.index = pd.to_datetime(market_daily.index)
        eq_series = market_daily.dropna()

        credit['date'] = pd.to_datetime(credit['date'])
        credit = credit.set_index('date').sort_index()

        lag_results = {}
        credit_cols = [c for c in ['hyg_yield_change', 'lqd_yield_change', 'vix_change'] if c in credit.columns]

        for col in credit_cols:
            credit_series = credit[col].dropna()
            if len(credit_series) < 10:
                continue

            lag_corrs = {}
            for lag in self.LAG_WINDOWS:
                shifted = credit_series.shift(lag).dropna()
                common = eq_series.index.intersection(shifted.index)
                if len(common) > 10:
                    corr = eq_series.loc[common].corr(shifted.loc[common])
                    lag_corrs[str(lag)] = round(float(corr), 4)
                else:
                    lag_corrs[str(lag)] = None

            best_lag = max(
                ((k, abs(v)) for k, v in lag_corrs.items() if v is not None),
                key=lambda x: x[1], default=(None, 0)
            )

            lag_results[col] = {
                'lag_correlations': lag_corrs,
                'best_lag': int(best_lag[0]) if best_lag[0] is not None else None,
                'best_correlation': best_lag[1],
            }

        same_day = {}
        for col in credit_cols:
            if col in lag_results and '0' in lag_results[col]['lag_correlations']:
                same_day[f'equity_vs_{col}'] = lag_results[col]['lag_correlations']['0']

        return {
            "sample_days": len(eq_series),
            "lags_analyzed": self.LAG_WINDOWS,
            "lagged_correlations": lag_results,
            "same_day_correlations": same_day,
        }

    def _load_market_returns(self, tickers, start_date, end_date):
        extended_start = start_date - timedelta(days=30)
        sql = """
            SELECT ticker, date, returns_1d
            FROM daily_market_data
            WHERE ticker = ANY(%s) AND date BETWEEN %s AND %s
              AND returns_1d IS NOT NULL
            ORDER BY date
        """
        df = self._query(sql, (tickers, _to_date_sql(extended_start), _to_date_sql(end_date)))
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
        return df

    def _load_credit_changes(self, start_date, end_date):
        extended_start = start_date - timedelta(days=30)
        sql = """
            SELECT date, hyg_price, hyg_yield, lqd_price, lqd_yield, vix
            FROM credit_proxy_data
            WHERE date BETWEEN %s AND %s
            ORDER BY date
        """
        df = self._query(sql, (_to_date_sql(extended_start), _to_date_sql(end_date)))
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            for col, price_col in [('hyg_yield', 'hyg_price'), ('lqd_yield', 'lqd_price'), ('vix', None)]:
                if col in df.columns:
                    df[f'{col}_change'] = df[col].diff()
                    if df[f'{col}_change'].isna().all() or (df[f'{col}_change'] == 0).all():
                        if price_col and price_col in df.columns:
                            df[f'{col}_change'] = df[price_col].pct_change().fillna(0)
                elif price_col and price_col in df.columns:
                    df[f'{col}_change'] = df[price_col].pct_change().fillna(0)
        return df

    def analyze_lead_lag_by_event(
        self,
        news_events: List[Dict],
        market_data: pd.DataFrame,
        credit_data: pd.DataFrame,
        max_lag: int = 10
    ) -> pd.DataFrame:
        """For each news event, determine whether equity or credit reacted first.

        Returns DataFrame with: news_id, ticker, equity_first_move_day,
        credit_first_move_day, lead_lag_days, leading_market
        """
        if not news_events:
            return pd.DataFrame()

        results = []
        credit_data = self._ensure_credit_change_columns(credit_data)

        for event in news_events:
            ticker = event.get('ticker', '')
            event_date = pd.to_datetime(event.get('event_date')).normalize()

            ticker_market = market_data[
                (market_data['ticker'] == ticker) &
                (pd.to_datetime(market_data['date']).dt.normalize() >= event_date) &
                (pd.to_datetime(market_data['date']).dt.normalize() <= event_date + timedelta(days=max_lag))
            ].sort_values('date') if not market_data.empty else pd.DataFrame()

            post_credit = credit_data[
                (pd.to_datetime(credit_data['date']).dt.normalize() >= event_date) &
                (pd.to_datetime(credit_data['date']).dt.normalize() <= event_date + timedelta(days=max_lag))
            ].sort_values('date') if not credit_data.empty else pd.DataFrame()

            equity_day, equity_signal = self._find_first_significant_day(
                ticker_market, 'returns_1d', threshold=0.02
            )
            credit_day, credit_signal = self._find_first_significant_day(
                post_credit, 'hyg_yield_change', threshold=0.001
            )

            lead_lag = 0
            leading_market = "simultaneous"
            if equity_day is not None and credit_day is not None:
                lead_lag = equity_day - credit_day
                if lead_lag < 0:
                    leading_market = "equity"
                elif lead_lag > 0:
                    leading_market = "credit"
            elif equity_day is not None:
                leading_market = "equity"
            elif credit_day is not None:
                leading_market = "credit"
            else:
                leading_market = "none"

            results.append({
                'news_id': event.get('news_id'),
                'ticker': ticker,
                'event_date': event_date,
                'event_type': event.get('event_type', ''),
                'llm_market_impact': event.get('llm_market_impact', ''),
                'equity_first_move_day': equity_day,
                'credit_first_move_day': credit_day,
                'lead_lag_days': lead_lag,
                'leading_market': leading_market,
            })

        return pd.DataFrame(results)

    def _ensure_credit_change_columns(self, credit_data: pd.DataFrame) -> pd.DataFrame:
        """Add yield-change columns, falling back to ETF price returns when yields are unavailable."""
        if credit_data.empty:
            return credit_data

        df = credit_data.copy()
        df['date'] = pd.to_datetime(df['date']).dt.normalize()
        df = df.sort_values('date')
        for col, price_col in [('hyg_yield', 'hyg_price'), ('lqd_yield', 'lqd_price'), ('vix', None)]:
            change_col = f'{col}_change'
            if col in df.columns and df[col].notna().any():
                df[change_col] = df[col].diff()
            elif price_col and price_col in df.columns and df[price_col].notna().any():
                df[change_col] = df[price_col].pct_change()
            elif change_col not in df.columns:
                df[change_col] = 0.0
        return df

    def _find_first_significant_day(
        self,
        data: pd.DataFrame,
        col: str,
        threshold: float
    ) -> Tuple[Optional[int], Optional[float]]:
        """Find the first day where column exceeds threshold, return (day_offset, value)."""
        if data.empty or col not in data.columns:
            return None, None

        data = data.copy()
        if 'date' in data.columns:
            data = data.sort_values('date')
            first_date = data['date'].iloc[0]
            for _, row in data.iterrows():
                val = row[col]
                if pd.notna(val) and abs(val) > threshold:
                    day_offset = (row['date'] - first_date).days
                    return int(day_offset), float(val)

        return None, None

    def get_lead_lag_distribution(self, results: pd.DataFrame) -> Dict[str, Any]:
        """Summarize lead-lag patterns by market_impact_type and event_type."""
        if results.empty:
            return {"total_events": 0}

        by_leading = results['leading_market'].value_counts().to_dict()

        by_impact = {}
        for impact_type in results['llm_market_impact'].unique():
            subset = results[results['llm_market_impact'] == impact_type]
            if subset.empty:
                continue
            by_impact[str(impact_type)] = {
                'count': len(subset),
                'avg_lead_lag_days': round(float(subset['lead_lag_days'].mean()), 2),
                'leading_market_distribution': subset['leading_market'].value_counts().to_dict()
            }

        avg_lag = float(results['lead_lag_days'].mean())

        return {
            "total_events": len(results),
            "avg_lead_lag_days": round(avg_lag, 2),
            "negative_means_equity_leads": True,
            "leading_market_distribution": by_leading,
            "by_llm_market_impact": by_impact,
        }

    def compute_reaction_lag_stats(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """End-to-end lag analysis for a set of tickers over a date range."""
        from pipelines.reaction.market_reaction import MarketReactionAnalyzer

        analyzer = MarketReactionAnalyzer(self.db)
        reactions = analyzer.analyze_news_events(tickers, start_date, end_date)

        if not reactions:
            return {"error": "No reaction data available"}

        news_events = [
            {
                'news_id': r.news_id,
                'ticker': r.ticker,
                'event_date': r.event_date,
                'event_type': r.event_type,
                'llm_market_impact': r.llm_market_impact,
            }
            for r in reactions
        ]

        market_data = analyzer._load_market_data(tickers, start_date, end_date)
        credit_data = analyzer._load_credit_proxy(start_date, end_date)

        results = self.analyze_lead_lag_by_event(news_events, market_data, credit_data)
        distribution = self.get_lead_lag_distribution(results)
        cross_corr = self.compute_cross_correlation(tickers, start_date, end_date)

        return {
            "lead_lag_analysis": distribution,
            "cross_correlation": cross_corr,
            "event_count": len(reactions),
        }
