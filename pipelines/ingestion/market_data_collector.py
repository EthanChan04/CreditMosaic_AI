"""
市场数据采集器
采集股票价格、成交量、波动率等市场数据
"""

import asyncio
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketDataPoint:
    """市场数据点"""
    ticker: str
    date: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    adjusted_close: float

class MarketDataCollector:
    """市场数据采集器"""

    def __init__(self, tickers: List[str]):
        self.tickers = tickers

    def collect_historical_data(
        self,
        start_date: datetime,
        end_date: datetime,
        include_technicals: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """采集历史市场数据"""
        all_data = {}

        for ticker in self.tickers:
            try:
                logger.info(f"开始采集 {ticker} 的市场数据...")

                # 获取股票数据
                stock = yf.Ticker(ticker)
                hist_data = stock.history(
                    start=start_date,
                    end=end_date,
                    interval='1d'
                )

                if hist_data.empty:
                    logger.warning(f"{ticker} 没有获取到数据")
                    continue

                # 重置索引，将Date变为列
                hist_data.reset_index(inplace=True)
                hist_data.rename(columns={'Date': 'date'}, inplace=True)

                # 计算技术指标
                if include_technicals:
                    hist_data = self._calculate_technicals(hist_data)

                all_data[ticker] = hist_data
                logger.info(f"成功采集 {ticker}: {len(hist_data)} 条记录")

            except Exception as e:
                logger.error(f"采集 {ticker} 失败: {e}")

        return all_data

    def _calculate_technicals(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        # 确保数据按日期排序
        df = df.sort_values('date')

        # 计算收益率
        df['returns_1d'] = df['Close'].pct_change()
        df['returns_5d'] = df['Close'].pct_change(5)
        df['returns_20d'] = df['Close'].pct_change(20)

        # 计算波动率
        df['volatility_5d'] = df['returns_1d'].rolling(window=5).std()
        df['volatility_20d'] = df['returns_1d'].rolling(window=20).std()

        # 计算成交量移动平均
        df['volume_ma_5d'] = df['Volume'].rolling(window=5).mean()
        df['volume_ma_20d'] = df['Volume'].rolling(window=20).mean()

        # 计算异常成交量（超过2个标准差）
        df['abnormal_volume_5d'] = df['Volume'] > (df['Volume'].rolling(window=60).mean() + 2 * df['Volume'].rolling(window=60).std())

        # 计算最大回撤
        df['cumulative_returns'] = (1 + df['returns_1d']).cumprod()
        df['running_max'] = df['cumulative_returns'].expanding().max()
        df['drawdown'] = (df['cumulative_returns'] - df['running_max']) / df['running_max']

        # 计算布林带
        df['bb_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

        # 计算RSI
        df['rsi'] = self._calculate_rsi(df['returns_1d'])

        return df

    def _calculate_rsi(self, returns: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        gains = returns.where(returns > 0, 0)
        losses = -returns.where(returns < 0, 0)

        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()

        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def collect_fundamentals(self, tickers: List[str]) -> Dict[str, Dict]:
        """采集基本面数据"""
        fundamentals = {}

        for ticker in tickers:
            try:
                logger.info(f"采集 {ticker} 的基本面数据...")
                stock = yf.Ticker(ticker)
                info = stock.info

                # 提取关键财务指标
                fund_data = {
                    'market_cap': info.get('marketCap'),
                    'enterprise_value': info.get('enterpriseValue'),
                    'total_assets': info.get('totalAssets'),
                    'total_liabilities': info.get('totalLiabilities'),
                    'total_equity': info.get('totalEquity'),
                    'long_term_debt': info.get('longTermDebt'),
                    'current_assets': info.get('currentAssets'),
                    'current_liabilities': info.get('currentLiabilities'),
                    'cash_and_equivalents': info.get('cash'),
                    'revenue': info.get('totalRevenue'),
                    'net_income': info.get('netIncome'),
                    'operating_cash_flow': info.get('operatingCashflow'),
                    'ebitda': info.get('ebitda'),
                    'debt_to_assets': info.get('debtToAssets'),
                    'current_ratio': info.get('currentRatio'),
                    'quick_ratio': info.get('quickRatio'),
                    'gross_margin': info.get('grossMargins'),
                    'operating_margin': info.get('operatingMargins'),
                    'net_margin': info.get('profitMargins'),
                    'roa': info.get('returnOnAssets'),
                    'roe': info.get('returnOnEquity'),
                    'revenue_growth_yoy': info.get('revenueGrowth'),
                    'sector': info.get('sector'),
                    'industry': info.get('industry'),
                    'exchange': info.get('exchange'),
                    'company_name': info.get('longName'),
                    'country': info.get('country'),
                    'founded_year': info.get('foundedYear')
                }

                fundamentals[ticker] = fund_data
                logger.info(f"成功采集 {ticker} 基本面数据")

            except Exception as e:
                logger.error(f"采集 {ticker} 基本面失败: {e}")

        return fundamentals

class CreditProxyCollector:
    """信用代理变量采集器"""

    def __init__(self):
        # 信用代理ETF和指标
        self.credit_proxies = {
            'HYG': 'iShares iBoxx $ High Yield Corporate Bond ETF',
            'LQD': 'iShares iBoxx $ Investment Grade Corporate Bond ETF',
            'IEF': 'iShares 7-10 Year Treasury Bond ETF'
        }

    def collect_proxy_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """采集信用代理数据"""
        all_proxy_data = []

        # 采集ETF数据
        for ticker, description in self.credit_proxies.items():
            try:
                logger.info(f"采集信用代理 {ticker}...")
                etf = yf.Ticker(ticker)
                hist_data = etf.history(start=start_date, end=end_date, interval='1d')

                if not hist_data.empty:
                    hist_data.reset_index(inplace=True)
                    hist_data['proxy_type'] = ticker
                    hist_data['proxy_description'] = description

                    # 计算收益率
                    hist_data['returns'] = hist_data['Close'].pct_change()

                    all_proxy_data.append(hist_data)

            except Exception as e:
                logger.error(f"采集 {ticker} 失败: {e}")

        # 合并所有代理数据
        if all_proxy_data:
            combined_df = pd.concat(all_proxy_data, ignore_index=True)
            return combined_df
        else:
            return pd.DataFrame()

    def collect_vix_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """采集VIX波动率指数数据"""
        try:
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(start=start_date, end=end_date, interval='1d')

            if not vix_data.empty:
                vix_data.reset_index(inplace=True)
                vix_data['proxy_type'] = 'VIX'
                vix_data['proxy_description'] = 'CBOE Volatility Index'

                return vix_data

        except Exception as e:
            logger.error(f"采集VIX数据失败: {e}")
            return pd.DataFrame()

class DataQualityChecker:
    """Data quality checker using psycopg2 connection"""

    def __init__(self, db_connection):
        self.db = db_connection

    def _query(self, sql: str, params: tuple = None):
        with self.db.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def _query_one(self, sql: str, params: tuple = None):
        with self.db.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            if row:
                columns = [desc[0] for desc in cur.description]
                return dict(zip(columns, row))
            return None

    def check_data_completeness(self, tickers: List[str], start_date: datetime, end_date: datetime) -> Dict[str, Dict]:
        results = {}

        for ticker in tickers:
            try:
                market_sql = """
                    SELECT COUNT(*) as total_days,
                           COUNT(close_price) as non_null_close,
                           MIN(date) as start_date,
                           MAX(date) as end_date
                    FROM daily_market_data
                    WHERE ticker = %s AND date BETWEEN %s AND %s
                """
                market_result = self._query_one(market_sql, (ticker, start_date, end_date))

                news_sql = """
                    SELECT COUNT(*) as total_news,
                           COUNT(body) as non_empty_body,
                           MIN(published_at) as start_date,
                           MAX(published_at) as end_date
                    FROM news_items
                    WHERE ticker = %s AND published_at BETWEEN %s AND %s
                """
                news_result = self._query_one(news_sql, (ticker, start_date, end_date))

                results[ticker] = {
                    'market_data': market_result or {},
                    'news_data': news_result or {}
                }

            except Exception as e:
                logger.error(f"Data completeness check failed for {ticker}: {e}")

        return results

    def check_outliers(self, ticker: str) -> Dict[str, List]:
        outliers = {}

        try:
            price_sql = """
                SELECT date, close_price, returns_1d
                FROM daily_market_data
                WHERE ticker = %s
                AND (returns_1d > 0.5 OR returns_1d < -0.5)
                ORDER BY ABS(returns_1d) DESC
                LIMIT 20
            """
            price_outliers = self._query(price_sql, (ticker,))
            outliers['price_spikes'] = [dict(zip(['date', 'close_price', 'returns_1d'], row)) for row in price_outliers]

            volume_sql = """
                SELECT date, volume, volume_ma_20d
                FROM daily_market_data
                WHERE ticker = %s
                AND volume > volume_ma_20d * 5
                ORDER BY volume DESC
                LIMIT 20
            """
            volume_outliers = self._query(volume_sql, (ticker,))
            outliers['volume_spikes'] = [dict(zip(['date', 'volume', 'volume_ma_20d'], row)) for row in volume_outliers]

        except Exception as e:
            logger.error(f"Outlier check failed for {ticker}: {e}")

        return outliers

# 使用示例
if __name__ == "__main__":
    # 配置
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    start_date = datetime.now() - timedelta(days=180)  # 6个月
    end_date = datetime.now()

    # 创建采集器
    market_collector = MarketDataCollector(tickers)

    # 采集市场数据
    market_data = market_collector.collect_historical_data(start_date, end_date)

    # 采集基本面数据
    fundamentals = market_collector.collect_fundamentals(tickers)

    # 采集信用代理数据
    proxy_collector = CreditProxyCollector()
    proxy_data = proxy_collector.collect_proxy_data(start_date, end_date)
    vix_data = proxy_collector.collect_vix_data(start_date, end_date)

    # 打印结果
    print(f"采集完成:")
    for ticker, df in market_data.items():
        print(f"{ticker}: {len(df)} 条市场数据记录")

    print(f"基本面数据: {len(fundamentals)} 家公司")
    print(f"信用代理数据: {len(proxy_data)} 条记录")
    print(f"VIX数据: {len(vix_data)} 条记录")