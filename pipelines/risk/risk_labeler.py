"""
Risk Label Generator
Computes multi-label proxy risk indicators from daily market data.

Each label is a boolean flag derived from market behavior thresholds:
  - Abnormal negative returns: daily return < -2 std below 60-day rolling mean
  - Abnormal volume spike: volume > 5x 20-day moving average
  - Volatility jump: 5-day realized vol > 2x 20-day baseline
  - Credit proxy widening: HYG/LQD yield increase > 1.5 std over 20-day
  - Distress news followup: high-risk LLM signal (score >= 70) within window
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import psycopg2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _ensure_datetime(d):
    """Normalize date or datetime to datetime."""
    if d is None:
        return None
    if isinstance(d, datetime):
        return d
    return datetime.combine(d, datetime.min.time())


class RiskLabeler:
    """Compute multi-label risk indicators from market data and LLM signals."""

    def __init__(self, db_connection):
        self.db = db_connection

    def _query(self, sql: str, params=None) -> pd.DataFrame:
        with self.db.cursor() as cur:
            cur.execute(sql, params)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            return pd.DataFrame(rows, columns=columns)

    @staticmethod
    def _to_date_sql(d):
        """Return date() for SQL, works with both datetime and date."""
        if isinstance(d, datetime):
            return d.date()
        return d

    def generate_all_labels(
        self,
        tickers: List[str],
        start_date,
        end_date
    ) -> pd.DataFrame:
        """Generate all risk labels for given tickers and date range."""
        start_date = _ensure_datetime(start_date)
        end_date = _ensure_datetime(end_date)
        market_data = self._load_market_data(tickers, start_date, end_date)
        credit_data = self._load_credit_proxy_data(start_date, end_date)
        llm_signals = self._load_llm_signals(tickers, start_date, end_date)

        if market_data.empty:
            logger.warning("No market data found for label generation")
            return pd.DataFrame()

        labels = self._compute_labels(market_data, credit_data, llm_signals)
        return labels

    def _load_market_data(self, tickers, start_date, end_date) -> pd.DataFrame:
        sql = """
            SELECT ticker, date, close_price, volume,
                   returns_1d, returns_5d, returns_20d,
                   volatility_5d, volatility_20d,
                   volume_ma_5d, volume_ma_20d
            FROM daily_market_data
            WHERE ticker = ANY(%s) AND date BETWEEN %s AND %s
            ORDER BY ticker, date
        """
        return self._query(sql, (tickers, self._to_date_sql(start_date), self._to_date_sql(end_date)))

    def _load_credit_proxy_data(self, start_date, end_date) -> pd.DataFrame:
        sql = """
            SELECT date, hyg_price, lqd_price, hyg_yield, lqd_yield, vix
            FROM credit_proxy_data
            WHERE date BETWEEN %s AND %s
            ORDER BY date
        """
        return self._query(sql, (self._to_date_sql(start_date), self._to_date_sql(end_date)))

    def _load_llm_signals(self, tickers, start_date, end_date) -> pd.DataFrame:
        sql = """
            SELECT lns.ticker, lns.credit_risk_score, ni.published_at as date
            FROM llm_news_signals lns
            JOIN news_items ni ON lns.news_id = ni.news_id
            WHERE lns.ticker = ANY(%s)
              AND ni.published_at BETWEEN %s AND %s
        """
        return self._query(sql, (tickers, start_date, end_date))

    def _compute_labels(
        self,
        market: pd.DataFrame,
        credit: pd.DataFrame,
        llm_signals: pd.DataFrame
    ) -> pd.DataFrame:
        market = market.copy()
        market['date'] = pd.to_datetime(market['date'])

        results = []

        for ticker in market['ticker'].unique():
            ticker_data = market[market['ticker'] == ticker].sort_values('date').copy()
            if ticker_data.empty:
                continue

            self._label_abnormal_returns(ticker_data)
            self._label_volume_spikes(ticker_data)
            self._label_volatility_jumps(ticker_data)
            self._label_credit_widening(ticker_data, credit)
            self._label_distress_news(ticker_data, llm_signals)

            results.append(ticker_data)

        if not results:
            return pd.DataFrame()

        labels = pd.concat(results, ignore_index=True)
        label_cols = [
            'ticker', 'date',
            'abnormal_negative_return_1d', 'abnormal_negative_return_5d',
            'abnormal_negative_return_20d',
            'abnormal_volume_spike_1d', 'abnormal_volume_spike_5d',
            'volatility_jump_5d', 'volatility_jump_20d',
            'credit_proxy_widening_5d', 'credit_proxy_widening_20d',
            'distress_news_followup_30d', 'distress_news_followup_90d'
        ]
        return labels[label_cols]

    def _label_abnormal_returns(self, df: pd.DataFrame):
        """Flag returns below -2 std of rolling 60-day returns."""
        if 'returns_1d' in df.columns:
            mean_ret = df['returns_1d'].rolling(60, min_periods=20).mean()
            std_ret = df['returns_1d'].rolling(60, min_periods=20).std()
            threshold = mean_ret - 2 * std_ret
            df['abnormal_negative_return_1d'] = df['returns_1d'] < threshold
            df['abnormal_negative_return_1d'] = df['abnormal_negative_return_1d'].fillna(False)

        if 'returns_5d' in df.columns:
            mean_ret5 = df['returns_5d'].rolling(60, min_periods=20).mean()
            std_ret5 = df['returns_5d'].rolling(60, min_periods=20).std()
            threshold5 = mean_ret5 - 2 * std_ret5
            df['abnormal_negative_return_5d'] = df['returns_5d'] < threshold5
            df['abnormal_negative_return_5d'] = df['abnormal_negative_return_5d'].fillna(False)

        if 'returns_20d' in df.columns:
            mean_ret20 = df['returns_20d'].rolling(60, min_periods=20).mean()
            std_ret20 = df['returns_20d'].rolling(60, min_periods=20).std()
            threshold20 = mean_ret20 - 2 * std_ret20
            df['abnormal_negative_return_20d'] = df['returns_20d'] < threshold20
            df['abnormal_negative_return_20d'] = df['abnormal_negative_return_20d'].fillna(False)

    def _label_volume_spikes(self, df: pd.DataFrame):
        """Flag volume spikes > 5x 20-day MA."""
        if 'volume' in df.columns and 'volume_ma_20d' in df.columns:
            df['abnormal_volume_spike_1d'] = (
                (df['volume_ma_20d'] > 0) &
                (df['volume'] > df['volume_ma_20d'] * 5)
            )
            df['abnormal_volume_spike_1d'] = df['abnormal_volume_spike_1d'].fillna(False)
        else:
            df['abnormal_volume_spike_1d'] = False

        if 'volume' in df.columns and 'volume_ma_5d' in df.columns:
            df['abnormal_volume_spike_5d'] = (
                (df['volume_ma_5d'] > 0) &
                (df['volume'] > df['volume_ma_5d'] * 3)
            )
            df['abnormal_volume_spike_5d'] = df['abnormal_volume_spike_5d'].fillna(False)
        else:
            df['abnormal_volume_spike_5d'] = False

    def _label_volatility_jumps(self, df: pd.DataFrame):
        """Flag volatility jumps: 5-day vol > 2x 20-day vol baseline."""
        if 'volatility_5d' in df.columns and 'volatility_20d' in df.columns:
            df['volatility_jump_5d'] = (
                (df['volatility_20d'] > 0) &
                (df['volatility_5d'] > df['volatility_20d'] * 2.0)
            )
            df['volatility_jump_5d'] = df['volatility_jump_5d'].fillna(False)

            df['volatility_jump_20d'] = (
                (df['volatility_20d'] > 0) &
                (df['volatility_20d'] > df['volatility_20d'].rolling(60).mean() * 1.5)
            )
            df['volatility_jump_20d'] = df['volatility_jump_20d'].fillna(False)
        else:
            df['volatility_jump_5d'] = False
            df['volatility_jump_20d'] = False

    def _label_credit_widening(self, df: pd.DataFrame, credit: pd.DataFrame):
        """Flag credit proxy widening from yields, or ETF price drops when yields are unavailable."""
        if credit.empty:
            df['credit_proxy_widening_5d'] = False
            df['credit_proxy_widening_20d'] = False
            return

        credit = credit.copy()
        credit['date'] = pd.to_datetime(credit['date'])
        credit = credit.sort_values('date')

        proxy_flags = []
        for col in ['hyg_yield', 'lqd_yield']:
            if col in credit.columns and credit[col].notna().any():
                credit[f'{col}_change'] = credit[col].diff()
                credit[f'{col}_widening'] = credit[f'{col}_change'] > (
                    credit[f'{col}_change'].rolling(20, min_periods=5).std() * 1.5
                )
                proxy_flags.append(f'{col}_widening')

        for col in ['hyg_price', 'lqd_price']:
            if col in credit.columns and credit[col].notna().any():
                credit[f'{col}_return'] = credit[col].pct_change()
                baseline = credit[f'{col}_return'].rolling(20, min_periods=5).mean()
                spread = credit[f'{col}_return'].rolling(20, min_periods=5).std()
                credit[f'{col}_widening'] = credit[f'{col}_return'] < (baseline - 1.5 * spread)
                proxy_flags.append(f'{col}_widening')

        has_widening = pd.Series(False, index=credit.index)
        for col in proxy_flags:
            if col in credit.columns:
                has_widening = has_widening | credit[col].fillna(False)

        df['credit_proxy_widening_5d'] = False
        df['credit_proxy_widening_20d'] = False

        if not has_widening.empty and has_widening.any():
            widen_dates = set(credit.loc[has_widening, 'date'])
            for i, row in df.iterrows():
                d = row['date']
                df.at[i, 'credit_proxy_widening_5d'] = any(
                    (d - timedelta(days=offset)) in widen_dates
                    for offset in range(6)
                )
                df.at[i, 'credit_proxy_widening_20d'] = any(
                    (d - timedelta(days=offset)) in widen_dates
                    for offset in range(21)
                )

    def _label_distress_news(self, df: pd.DataFrame, llm_signals: pd.DataFrame):
        """Flag distress news: high-risk LLM signal (score >= 70) in followup window."""
        if llm_signals.empty:
            df['distress_news_followup_30d'] = False
            df['distress_news_followup_90d'] = False
            return

        distress = llm_signals[llm_signals['credit_risk_score'] >= 70].copy()
        if distress.empty:
            df['distress_news_followup_30d'] = False
            df['distress_news_followup_90d'] = False
            return

        distress['date'] = pd.to_datetime(distress['date'])
        distress_dates = distress[['ticker', 'date']].drop_duplicates()

        df['distress_news_followup_30d'] = False
        df['distress_news_followup_90d'] = False

        for _, row in df.iterrows():
            ticker, d = row['ticker'], row['date']
            ticker_distress = distress_dates[distress_dates['ticker'] == ticker]
            if ticker_distress.empty:
                continue
            df.at[row.name, 'distress_news_followup_30d'] = any(
                (dist_d - d).days in range(0, 31)
                for dist_d in ticker_distress['date']
            )
            df.at[row.name, 'distress_news_followup_90d'] = any(
                (dist_d - d).days in range(0, 91)
                for dist_d in ticker_distress['date']
            )

    def save_labels_to_db(self, labels: pd.DataFrame):
        """Save computed risk labels to PostgreSQL."""
        if labels.empty:
            return

        with self.db.cursor() as cur:
            for _, row in labels.iterrows():
                cur.execute("""
                    INSERT INTO risk_labels (
                        ticker, date,
                        abnormal_negative_return_1d, abnormal_negative_return_5d,
                        abnormal_negative_return_20d,
                        abnormal_volume_spike_1d, abnormal_volume_spike_5d,
                        volatility_jump_5d, volatility_jump_20d,
                        credit_proxy_widening_5d, credit_proxy_widening_20d,
                        distress_news_followup_30d, distress_news_followup_90d
                    ) VALUES (
                        %(ticker)s, %(date)s,
                        %(abnormal_negative_return_1d)s, %(abnormal_negative_return_5d)s,
                        %(abnormal_negative_return_20d)s,
                        %(abnormal_volume_spike_1d)s, %(abnormal_volume_spike_5d)s,
                        %(volatility_jump_5d)s, %(volatility_jump_20d)s,
                        %(credit_proxy_widening_5d)s, %(credit_proxy_widening_20d)s,
                        %(distress_news_followup_30d)s, %(distress_news_followup_90d)s
                    )
                    ON CONFLICT (ticker, date) DO UPDATE SET
                        abnormal_negative_return_1d = EXCLUDED.abnormal_negative_return_1d,
                        abnormal_negative_return_5d = EXCLUDED.abnormal_negative_return_5d,
                        abnormal_negative_return_20d = EXCLUDED.abnormal_negative_return_20d,
                        abnormal_volume_spike_1d = EXCLUDED.abnormal_volume_spike_1d,
                        abnormal_volume_spike_5d = EXCLUDED.abnormal_volume_spike_5d,
                        volatility_jump_5d = EXCLUDED.volatility_jump_5d,
                        volatility_jump_20d = EXCLUDED.volatility_jump_20d,
                        credit_proxy_widening_5d = EXCLUDED.credit_proxy_widening_5d,
                        credit_proxy_widening_20d = EXCLUDED.credit_proxy_widening_20d,
                        distress_news_followup_30d = EXCLUDED.distress_news_followup_30d,
                        distress_news_followup_90d = EXCLUDED.distress_news_followup_90d
                """, {
                    'ticker': row['ticker'],
                    'date': row['date'].date() if hasattr(row['date'], 'date') else row['date'],
                    'abnormal_negative_return_1d': bool(row['abnormal_negative_return_1d']),
                    'abnormal_negative_return_5d': bool(row['abnormal_negative_return_5d']),
                    'abnormal_negative_return_20d': bool(row['abnormal_negative_return_20d']),
                    'abnormal_volume_spike_1d': bool(row['abnormal_volume_spike_1d']),
                    'abnormal_volume_spike_5d': bool(row['abnormal_volume_spike_5d']),
                    'volatility_jump_5d': bool(row['volatility_jump_5d']),
                    'volatility_jump_20d': bool(row['volatility_jump_20d']),
                    'credit_proxy_widening_5d': bool(row['credit_proxy_widening_5d']),
                    'credit_proxy_widening_20d': bool(row['credit_proxy_widening_20d']),
                    'distress_news_followup_30d': bool(row['distress_news_followup_30d']),
                    'distress_news_followup_90d': bool(row['distress_news_followup_90d']),
                })
            self.db.commit()

        logger.info(f"Saved {len(labels)} risk labels to database")
