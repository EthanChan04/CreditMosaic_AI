"""
Feature Engineering Pipeline
Builds daily feature vectors for ML model training from all data sources.

Features are organized into five groups:
  1. Market features: returns, volatility, volume (with lags t-1, t-5, t-20)
  2. Fundamentals: debt ratios, margins, growth rates
  3. Credit proxy: HYG/LQD yields, VIX levels + changes
  4. LLM signal features: aggregated credit risk scores, sentiment
  5. Derived features: cross-sectional ranks, interactions, technical indicators
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class FeatureEngineer:
    """Build daily feature matrix from PostgreSQL data sources."""

    MARKET_LAG_PERIODS = [1, 5, 20]
    LOOKBACK_DAYS = 365

    def __init__(self, db_connection):
        self.db = db_connection

    def _query(self, sql: str, params=None) -> pd.DataFrame:
        with self.db.cursor() as cur:
            cur.execute(sql, params)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            return pd.DataFrame(rows, columns=columns)

    def build_feature_matrix(
        self,
        tickers: List[str],
        end_date: datetime,
        lookback_days: int = None
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Build feature matrix X, label vector y, and metadata for ML training.

        Returns (X, y, meta) where meta contains ticker/date for each row.
        y is a binary target: 1 if any risk event within next 5 trading days.
        """
        if lookback_days is None:
            lookback_days = self.LOOKBACK_DAYS

        start_date = end_date - timedelta(days=lookback_days)

        market = self._load_market_data(tickers, start_date, end_date)
        fundamentals = self._load_fundamentals(tickers)
        credit = self._load_credit_proxy(start_date, end_date)
        signals = self._load_llm_signals(tickers, start_date, end_date)
        labels = self._load_risk_labels(tickers, start_date, end_date)

        if market.empty:
            logger.warning("No market data for feature building")
            return pd.DataFrame(), pd.Series(), pd.DataFrame()

        features = self._build_market_features(market)
        features = self._merge_fundamentals(features, fundamentals)
        features = self._merge_credit_proxy(features, credit)
        features = self._merge_llm_signals(features, signals)
        features = self._add_cross_sectional(features)

        X, y, meta = self._create_target(features, labels)

        logger.info(f"Built feature matrix: {X.shape[0]} rows x {X.shape[1]} features")
        return X, y, meta

    def _load_market_data(self, tickers, start_date, end_date) -> pd.DataFrame:
        sql = """
            SELECT ticker, date,
                   open_price, high_price, low_price, close_price,
                   volume, adjusted_close,
                   volatility_5d, volatility_20d,
                   returns_1d, returns_5d, returns_20d,
                   volume_ma_5d, volume_ma_20d
            FROM daily_market_data
            WHERE ticker = ANY(%s) AND date BETWEEN %s AND %s
            ORDER BY ticker, date
        """
        df = self._query(sql, (tickers, _to_date_sql(start_date), _to_date_sql(end_date)))
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
        return df

    def _load_fundamentals(self, tickers) -> pd.DataFrame:
        sql = """
            SELECT DISTINCT ON (ticker) ticker,
                   debt_to_assets, current_ratio, quick_ratio,
                   gross_margin, operating_margin, net_margin,
                   roa, roe, revenue_growth_yoy,
                   total_assets, total_liabilities,
                   long_term_debt, cash_and_equivalents
            FROM financial_fundamentals
            WHERE ticker = ANY(%s)
            ORDER BY ticker, report_date DESC
        """
        return self._query(sql, (tickers,))

    def _load_credit_proxy(self, start_date, end_date) -> pd.DataFrame:
        sql = """
            SELECT date, hyg_price, hyg_yield, lqd_price, lqd_yield, vix, ted_spread
            FROM credit_proxy_data
            WHERE date BETWEEN %s AND %s
            ORDER BY date
        """
        df = self._query(sql, (_to_date_sql(start_date), _to_date_sql(end_date)))
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
        return df

    def _load_llm_signals(self, tickers, start_date, end_date) -> pd.DataFrame:
        sql = """
            SELECT lns.ticker, ni.published_at as date,
                   lns.sentiment_score, lns.credit_risk_score,
                   lns.event_type, lns.confidence
            FROM llm_news_signals lns
            JOIN news_items ni ON lns.news_id = ni.news_id
            WHERE lns.ticker = ANY(%s)
              AND ni.published_at BETWEEN %s AND %s
        """
        df = self._query(sql, (tickers, start_date, end_date))
        if not df.empty:
            df['date'] = pd.to_datetime(df['date']).dt.normalize()
        return df

    def _load_risk_labels(self, tickers, start_date, end_date) -> pd.DataFrame:
        sql = """
            SELECT ticker, date,
                   abnormal_negative_return_1d, abnormal_negative_return_5d,
                   abnormal_volume_spike_1d, volatility_jump_5d,
                   credit_proxy_widening_5d, distress_news_followup_30d
            FROM risk_labels
            WHERE ticker = ANY(%s) AND date BETWEEN %s AND %s
            ORDER BY ticker, date
        """
        df = self._query(sql, (tickers, _to_date_sql(start_date), _to_date_sql(end_date)))
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
        return df

    def _build_market_features(self, market: pd.DataFrame) -> pd.DataFrame:
        df = market.sort_values(['ticker', 'date']).copy()

        base_features = [
            'returns_1d', 'returns_5d', 'returns_20d',
            'volatility_5d', 'volatility_20d', 'volume'
        ]

        result_rows = []
        for ticker in df['ticker'].unique():
            tdf = df[df['ticker'] == ticker].sort_values('date').copy()

            for lag in self.MARKET_LAG_PERIODS:
                for feat in base_features:
                    if feat in tdf.columns:
                        tdf[f'{feat}_lag{lag}'] = tdf[feat].shift(lag)

            for lag in [1, 5]:
                if 'returns_1d' in tdf.columns:
                    tdf[f'rolling_return_{lag}d'] = (
                        tdf['returns_1d'].rolling(lag).mean()
                    )
                if 'volatility_5d' in tdf.columns:
                    tdf[f'rolling_vol_{lag}d'] = (
                        tdf['volatility_5d'].rolling(lag).mean()
                    )

            if 'high_price' in tdf.columns and 'low_price' in tdf.columns and 'close_price' in tdf.columns:
                tdf['daily_range'] = (tdf['high_price'] - tdf['low_price']) / tdf['close_price']

            if 'volume' in tdf.columns and 'volume_ma_20d' in tdf.columns:
                tdf['volume_ratio'] = tdf['volume'] / tdf['volume_ma_20d'].replace(0, np.nan)

            result_rows.append(tdf)

        result = pd.concat(result_rows, ignore_index=True)
        return result

    def _merge_fundamentals(self, features: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
        if fundamentals.empty:
            return features

        fund_cols = [
            'ticker', 'debt_to_assets', 'current_ratio', 'quick_ratio',
            'gross_margin', 'operating_margin', 'net_margin',
            'roa', 'roe', 'revenue_growth_yoy'
        ]
        available = [c for c in fund_cols if c in fundamentals.columns]
        return features.merge(
            fundamentals[available],
            on='ticker', how='left', suffixes=('', '_fund')
        )

    def _merge_credit_proxy(self, features: pd.DataFrame, credit: pd.DataFrame) -> pd.DataFrame:
        if credit.empty:
            return features

        credit = credit.sort_values('date').copy()
        for col in ['hyg_price', 'lqd_price', 'hyg_yield', 'lqd_yield', 'vix', 'ted_spread']:
            if col in credit.columns and credit[col].notna().any():
                credit[f'{col}_d1'] = credit[col].diff()
                credit[f'{col}_d5'] = credit[col].diff(5)

        for col in ['hyg_price', 'lqd_price']:
            if col in credit.columns and credit[col].notna().any():
                credit[f'{col}_return_1d'] = credit[col].pct_change()
                credit[f'{col}_return_5d'] = credit[col].pct_change(5)

        features['date'] = pd.to_datetime(features['date'])
        credit['date'] = pd.to_datetime(credit['date'])

        return features.merge(credit, on='date', how='left', suffixes=('', '_credit'))

    def _merge_llm_signals(self, features: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        if signals.empty:
            for col in ['llm_risk_avg_7d', 'llm_risk_max_7d', 'llm_risk_count_7d',
                        'llm_sentiment_avg_7d', 'llm_high_risk_count_7d']:
                features[col] = 0.0
            return features

        signals = signals.copy()
        signals['date'] = pd.to_datetime(signals['date']).dt.normalize()
        signals = signals.sort_values(['ticker', 'date'])

        agg_rows = []
        for ticker in features['ticker'].dropna().unique():
            feature_dates = (
                features.loc[features['ticker'] == ticker, 'date']
                .pipe(pd.to_datetime)
                .dt.normalize()
                .drop_duplicates()
                .sort_values()
            )
            if feature_dates.empty:
                continue

            tdf = signals[signals['ticker'] == ticker].copy()
            if tdf.empty:
                agg_rows.append(pd.DataFrame({
                    'date': feature_dates,
                    'ticker': ticker,
                    'llm_risk_avg_7d': 0.0,
                    'llm_risk_max_7d': 0.0,
                    'llm_risk_count_7d': 0.0,
                    'llm_sentiment_avg_7d': 0.0,
                    'llm_high_risk_count_7d': 0.0,
                }))
                continue

            daily = tdf.groupby('date').agg(
                daily_risk_avg=('credit_risk_score', 'mean'),
                daily_risk_max=('credit_risk_score', 'max'),
                daily_risk_count=('credit_risk_score', 'count'),
                daily_sentiment_avg=('sentiment_score', 'mean'),
                daily_high_risk_count=('credit_risk_score', lambda x: (x >= 70).sum()),
            ).sort_index()

            full_index = feature_dates
            daily = daily.reindex(full_index, fill_value=0.0)
            rolling = daily.rolling('7D', min_periods=1)
            agg_rows.append(pd.DataFrame({
                'date': full_index,
                'ticker': ticker,
                'llm_risk_avg_7d': rolling['daily_risk_avg'].mean().values,
                'llm_risk_max_7d': rolling['daily_risk_max'].max().values,
                'llm_risk_count_7d': rolling['daily_risk_count'].sum().values,
                'llm_sentiment_avg_7d': rolling['daily_sentiment_avg'].mean().values,
                'llm_high_risk_count_7d': rolling['daily_high_risk_count'].sum().values,
            }))

        if agg_rows:
            all_agg = pd.concat(agg_rows, ignore_index=True)
            features = features.copy()
            features['date'] = pd.to_datetime(features['date']).dt.normalize()
            merged = features.merge(all_agg, on=['ticker', 'date'], how='left')
            for col in ['llm_risk_avg_7d', 'llm_risk_max_7d', 'llm_risk_count_7d',
                        'llm_sentiment_avg_7d', 'llm_high_risk_count_7d']:
                merged[col] = merged[col].fillna(0.0)
            return merged

        return features

    def _add_cross_sectional(self, features: pd.DataFrame) -> pd.DataFrame:
        df = features.copy()

        for col in ['returns_1d', 'volatility_5d', 'volume_ratio']:
            if col in df.columns:
                df[f'{col}_rank'] = df.groupby('date')[col].rank(pct=True)

        if 'debt_to_assets' in df.columns and 'debt_to_assets_rank' not in df.columns:
            df['debt_to_assets_rank'] = df['debt_to_assets'].rank(pct=True)
            df['high_leverage'] = (df['debt_to_assets'] > df['debt_to_assets'].median()).astype(int)

        if 'returns_1d' in df.columns and 'volatility_5d' in df.columns:
            df['return_vol_ratio'] = df['returns_1d'] / df['volatility_5d'].replace(0, np.nan)
            df['abs_return_vol_ratio'] = np.abs(df['return_vol_ratio'])

        return df

    def _create_target(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Create binary target and return (X, y, meta) where meta has ticker/date."""
        meta = features[['ticker', 'date']].copy() if 'ticker' in features.columns and 'date' in features.columns else pd.DataFrame()

        if labels.empty:
            features['target'] = 0
        else:
            labels['date'] = pd.to_datetime(labels['date'])
            labels['has_risk'] = (
                labels['abnormal_negative_return_1d'].astype(bool) |
                labels['abnormal_volume_spike_1d'].astype(bool) |
                labels['volatility_jump_5d'].astype(bool) |
                labels['credit_proxy_widening_5d'].astype(bool) |
                labels['distress_news_followup_30d'].astype(bool)
            ).astype(int)

            features = features.merge(
                labels[['ticker', 'date', 'has_risk']],
                on=['ticker', 'date'], how='left'
            )
            features['has_risk'] = features['has_risk'].fillna(0)

            features = features.sort_values(['ticker', 'date'])
            features['target'] = features.groupby('ticker')['has_risk'].shift(-5)
            features['target'] = features['target'].fillna(0).astype(int)

        y = features['target']
        non_feature_cols = ['ticker', 'date', 'target', 'has_risk',
                           'open_price', 'high_price', 'low_price', 'close_price',
                           'adjusted_close']
        X = features.drop(columns=[c for c in non_feature_cols if c in features.columns])

        X = X.select_dtypes(include=[np.number])
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)

        return X, y, meta

    def build_prediction_features(
        self,
        tickers: List[str],
        end_date: datetime
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Build features for prediction with ticker/date metadata.

        Returns (X, meta) where meta has ticker and date columns.
        """
        X, _, meta = self.build_feature_matrix(tickers, end_date)
        if X.empty:
            return X, meta

        if not meta.empty:
            latest_mask = meta.groupby('ticker')['date'].transform('max') == meta['date']
            X_latest = X.loc[latest_mask.index[latest_mask.values]]
            meta_latest = meta.loc[latest_mask.index[latest_mask.values]]
        else:
            X_latest = X
            meta_latest = meta

        return X_latest, meta_latest
