"""
Data Pipeline Coordinator
Main entry point that orchestrates the full data collection and processing flow
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import argparse
import yaml
import pandas as pd
from pathlib import Path

from pipelines.ingestion.news_collector import BatchNewsCollector, NewsCollectorFactory
from pipelines.ingestion.market_data_collector import MarketDataCollector, CreditProxyCollector, DataQualityChecker
from pipelines.ingestion.entity_matcher import EntityMatcher, TickerListManager
from pipelines.ingestion.db_manager import DatabaseManager, DatabaseConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineCoordinator:

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.db = None
        self.news_collectors = []
        self.entity_matcher = None

    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _initialize_components(self):
        postgres_config = self.config['postgresql']
        duckdb_path = self.config['duckdb']['path']

        self.db = DatabaseManager(postgres_config, duckdb_path)
        self.db.connect()

        logger.info("Creating database tables...")
        schema_path = Path(__file__).parent.parent.parent / 'db' / 'schema.sql'
        self.db.create_postgres_tables(str(schema_path))

        logger.info("Initializing news collectors...")
        for collector_config in self.config['news_collectors']:
            source = collector_config['source']
            api_key = collector_config.get('api_key')
            collector = NewsCollectorFactory.create_collector(source, api_key)
            self.news_collectors.append(collector)

        logger.info("Initializing entity matcher...")
        ticker_list_path = Path(self.config['ticker_list']['path'])
        if ticker_list_path.exists():
            companies = TickerListManager.load_ticker_list(str(ticker_list_path))
        else:
            companies = TickerListManager.create_default_ticker_list()

        self.db.upsert_companies(companies)
        self.entity_matcher = EntityMatcher(companies)

    async def run_initial_setup(self):
        logger.info("Starting initial setup...")

        try:
            self._initialize_components()

            tickers = [comp['ticker'] for comp in self.entity_matcher.companies[:50]]

            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)

            logger.info(f"Collecting data for {len(tickers)} companies, range: {start_date.date()} to {end_date.date()}")

            # 1. Collect news data
            logger.info("Collecting news data...")
            news_batch_collector = BatchNewsCollector(self.news_collectors, self.db.postgres_conn)
            news_results = await news_batch_collector.collect_batch(tickers, start_date, end_date)
            logger.info(f"News collection complete: {news_results}")

            # 2. Collect market data
            logger.info("Collecting market data...")
            market_collector = MarketDataCollector(tickers)
            market_data = market_collector.collect_historical_data(start_date, end_date)

            for ticker, df in market_data.items():
                df = df.reset_index()
                df['ticker'] = ticker
                df['created_at'] = datetime.now()

                required_columns = [
                    'ticker', 'date', 'open_price', 'high_price', 'low_price',
                    'close_price', 'volume', 'adjusted_close', 'volatility_5d',
                    'volatility_20d', 'returns_1d', 'returns_5d', 'returns_20d',
                    'volume_ma_5d', 'volume_ma_20d', 'created_at'
                ]

                for col in required_columns:
                    if col not in df.columns:
                        if col in ['open_price', 'high_price', 'low_price', 'close_price']:
                            df[col] = df['Close']
                        elif col in ['adjusted_close']:
                            df[col] = df['Close']
                        elif col in ['volume']:
                            df[col] = df['Volume']
                        elif col in ['date']:
                            df[col] = df['Date']
                        else:
                            df[col] = None

                df_to_save = df[required_columns].copy()
                df_to_save['date'] = pd.to_datetime(df_to_save['date']).dt.date

                self.db.insert_dataframe_postgres(df_to_save, 'daily_market_data')

            logger.info("Market data saved")

            # 3. Collect fundamentals
            logger.info("Collecting fundamentals...")
            fundamentals = market_collector.collect_fundamentals(tickers)

            fund_schema_columns = [
                'ticker', 'report_date', 'filing_type', 'fiscal_year', 'fiscal_quarter',
                'total_assets', 'total_liabilities', 'total_equity', 'long_term_debt',
                'current_assets', 'current_liabilities', 'cash_and_equivalents',
                'revenue', 'net_income', 'operating_cash_flow', 'ebitda',
                'debt_to_assets', 'current_ratio', 'quick_ratio', 'gross_margin',
                'operating_margin', 'net_margin', 'roa', 'roe', 'revenue_growth_yoy',
                'created_at'
            ]

            for ticker, fund_data in fundamentals.items():
                if fund_data:
                    fund_data['ticker'] = ticker
                    fund_data['report_date'] = datetime.now().date()
                    fund_data['created_at'] = datetime.now()
                    df = pd.DataFrame([fund_data])
                    for col in fund_schema_columns:
                        if col not in df.columns:
                            df[col] = None
                    df = df[fund_schema_columns]
                    self.db.insert_dataframe_postgres(df, 'financial_fundamentals')

            logger.info("Fundamentals saved")

            # 4. Collect credit proxy data
            logger.info("Collecting credit proxy data...")
            proxy_collector = CreditProxyCollector()
            proxy_data = proxy_collector.collect_proxy_data(start_date, end_date)
            vix_data = proxy_collector.collect_vix_data(start_date, end_date)

            self._save_credit_proxy_data(proxy_data, vix_data)
            logger.info("Credit proxy data saved")

            # 5. Data quality check
            logger.info("Running data quality checks...")
            quality_checker = DataQualityChecker(self.db.postgres_conn)
            completeness_results = quality_checker.check_data_completeness(tickers[:10], start_date, end_date)

            for ticker, results in completeness_results.items():
                logger.info(f"{ticker} data completeness: {results}")

            # 6. Create DuckDB analytical views
            logger.info("Creating DuckDB analytical views...")
            duckdb_ok = self.db.create_analytical_tables_duckdb()
            if not duckdb_ok:
                logger.warning("DuckDB analytical view was NOT created — PostgreSQL may not be running")

            logger.info("Initial setup complete!")

        except Exception as e:
            logger.error(f"Initial setup failed: {e}")
            raise
        finally:
            if self.db:
                self.db.close()

    def _save_credit_proxy_data(self, proxy_data: pd.DataFrame, vix_data: pd.DataFrame):
        """Transform raw yfinance credit proxy data to schema columns and save."""
        schema_rows = []

        if not proxy_data.empty:
            proxy_data = proxy_data.copy()
            proxy_data['Date'] = pd.to_datetime(proxy_data['Date']).dt.date
            dates = sorted(proxy_data['Date'].unique())

            for d in dates:
                row = {'date': d, 'created_at': datetime.now()}
                day_data = proxy_data[proxy_data['Date'] == d]

                for _, r in day_data.iterrows():
                    proxy_type = r.get('proxy_type', '')
                    close_val = r.get('Close')
                    if proxy_type == 'HYG':
                        row['hyg_price'] = close_val
                    elif proxy_type == 'LQD':
                        row['lqd_price'] = close_val
                    elif proxy_type == 'IEF':
                        pass

                schema_rows.append(row)

        if not vix_data.empty:
            vix_data = vix_data.copy()
            vix_data['Date'] = pd.to_datetime(vix_data['Date']).dt.date

            for _, r in vix_data.iterrows():
                d = r['Date']
                existing = next((x for x in schema_rows if x['date'] == d), None)
                if existing:
                    existing['vix'] = r.get('Close')
                else:
                    schema_rows.append({
                        'date': d,
                        'vix': r.get('Close'),
                        'created_at': datetime.now()
                    })

        if schema_rows:
            df = pd.DataFrame(schema_rows)
            df = df.sort_values('date')
            # Compute yield proxy as daily price return (inverse: price up -> yield down)
            if 'hyg_price' in df.columns:
                df['hyg_yield'] = df['hyg_price'].pct_change().fillna(0)
            if 'lqd_price' in df.columns:
                df['lqd_yield'] = df['lqd_price'].pct_change().fillna(0)
            for col in ['hyg_price', 'hyg_yield', 'lqd_price', 'lqd_yield', 'vix', 'ted_spread']:
                if col not in df.columns:
                    df[col] = None
            df = df[['date', 'hyg_price', 'hyg_yield', 'lqd_price', 'lqd_yield', 'vix', 'ted_spread', 'created_at']]
            self.db.insert_dataframe_postgres(df, 'credit_proxy_data')

    async def run_news_collection(self, tickers: List[str] = None, days: int = 7):
        logger.info(f"Starting news collection for last {days} days...")

        try:
            if not self.db:
                self._initialize_components()

            if tickers is None:
                companies = self.db.get_company_list()
                tickers = [comp['ticker'] for comp in companies]

            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            news_batch_collector = BatchNewsCollector(self.news_collectors, self.db.postgres_conn)
            news_results = await news_batch_collector.collect_batch(tickers, start_date, end_date)

            logger.info(f"News collection complete: {news_results}")
            return news_results

        except Exception as e:
            logger.error(f"News collection failed: {e}")
            raise

    def run_entity_matching(self, batch_size: int = 100):
        logger.info("Starting entity matching...")

        try:
            if not self.db:
                self._initialize_components()

            pending_news = self.db.get_pending_news(limit=batch_size * 10)

            if not pending_news:
                logger.info("No pending news to process")
                return

            processed_count = 0
            matched_count = 0

            for news in pending_news:
                try:
                    text_to_match = f"{news['title']} {news.get('body', '')}"
                    matches = self.entity_matcher.match_from_text(text_to_match)

                    matched_ticker = None
                    if matches:
                        best_match = matches[0]
                        matched_ticker = best_match.ticker
                        matched_count += 1
                        logger.debug(f"Matched: {news['title'][:50]}... -> {best_match.ticker}")

                    if matched_ticker:
                        self.db.execute_postgres(
                            "UPDATE news_items SET ticker = %s WHERE news_id = %s",
                            (matched_ticker, news['news_id'])
                        )

                    self.db.mark_news_as_processed([news['news_id']])
                    processed_count += 1

                except Exception as e:
                    logger.error(f"Failed processing news {news['news_id']}: {e}")

            logger.info(f"Entity matching complete: {processed_count} processed, {matched_count} matched")

        except Exception as e:
            logger.error(f"Entity matching failed: {e}")
            raise


async def main():
    parser = argparse.ArgumentParser(description='CreditMosaic AI Data Pipeline')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--mode', choices=['setup', 'news', 'entity'], default='setup',
                       help='Run mode: setup (full), news (news only), entity (entity matching only)')
    parser.add_argument('--tickers', nargs='+', help='Specify ticker list')
    parser.add_argument('--days', type=int, default=7, help='Days of news to collect')

    args = parser.parse_args()

    coordinator = PipelineCoordinator(args.config)

    try:
        if args.mode == 'setup':
            await coordinator.run_initial_setup()
        elif args.mode == 'news':
            await coordinator.run_news_collection(args.tickers, args.days)
        elif args.mode == 'entity':
            coordinator.run_entity_matching()

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline run failed: {e}")
    finally:
        if coordinator.db:
            coordinator.db.close()


if __name__ == "__main__":
    asyncio.run(main())
