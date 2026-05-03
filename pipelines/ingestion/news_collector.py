"""
News data collectors
Supports GDELT, Yahoo Finance News, NewsAPI data sources
"""

import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json
import logging
from dataclasses import dataclass
import yfinance as yf
from newsapi import NewsApiClient
import psycopg2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    ticker: str
    title: str
    body: str
    source: str
    url: str
    published_at: datetime
    collected_at: datetime = None

    def __post_init__(self):
        if self.collected_at is None:
            self.collected_at = datetime.now()


class BaseNewsCollector:

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            self.session = None

    async def _ensure_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def collect_news(self, ticker: str, start_date: datetime, end_date: datetime) -> List[NewsItem]:
        raise NotImplementedError


class YahooFinanceCollector(BaseNewsCollector):

    async def collect_news(self, ticker: str, start_date: datetime, end_date: datetime) -> List[NewsItem]:
        try:
            stock = yf.Ticker(ticker)
            news_data = stock.news

            news_items = []
            for news in news_data:
                published_at = datetime.fromtimestamp(news['providerPublishTime'])

                if start_date <= published_at <= end_date:
                    news_item = NewsItem(
                        ticker=ticker,
                        title=news.get('title', ''),
                        body=news.get('summary', ''),
                        source=news.get('publisher', ''),
                        url=news.get('link', ''),
                        published_at=published_at
                    )
                    news_items.append(news_item)

            logger.info(f"Yahoo Finance: collected {len(news_items)} news for {ticker}")
            return news_items

        except Exception as e:
            logger.error(f"Yahoo Finance collection failed: {e}")
            return []


class NewsAPICollector(BaseNewsCollector):

    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = NewsApiClient(api_key=api_key)

    async def collect_news(self, ticker: str, start_date: datetime, end_date: datetime) -> List[NewsItem]:
        try:
            company_info = yf.Ticker(ticker).info
            company_name = company_info.get('longName', ticker)

            from_date = start_date.strftime('%Y-%m-%d')
            to_date = end_date.strftime('%Y-%m-%d')

            articles = self.client.get_everything(
                q=f"{ticker} OR {company_name}",
                from_param=from_date,
                to=to_date,
                language='en',
                sort_by='publishedAt',
                page_size=100
            )

            news_items = []
            for article in articles.get('articles', []):
                published_at = datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')

                news_item = NewsItem(
                    ticker=ticker,
                    title=article.get('title', ''),
                    body=article.get('description', '') or '',
                    source=article.get('source', {}).get('name', ''),
                    url=article.get('url', ''),
                    published_at=published_at
                )
                news_items.append(news_item)

            logger.info(f"NewsAPI: collected {len(news_items)} news for {ticker}")
            return news_items

        except Exception as e:
            logger.error(f"NewsAPI collection failed: {e}")
            return []


class GDELTCollector(BaseNewsCollector):

    async def collect_news(self, ticker: str, start_date: datetime, end_date: datetime) -> List[NewsItem]:
        try:
            await self._ensure_session()

            company_info = yf.Ticker(ticker).info
            company_name = company_info.get('longName', ticker)

            from_date = start_date.strftime('%Y%m%d')
            to_date = end_date.strftime('%Y%m%d')

            base_url = "https://api.gdeltproject.org/api/v2/doc/doc"

            params = {
                'query': f"{ticker} OR {company_name}",
                'mode': 'artlist',
                'format': 'json',
                'start': from_date,
                'end': to_date,
                'maxrecords': 100
            }

            async with self.session.get(base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    news_items = []
                    for article in data.get('articles', []):
                        published_at_str = article.get('seendate', '')
                        if published_at_str:
                            published_at = datetime.strptime(published_at_str, '%Y%m%d%H%M%S')
                        else:
                            published_at = datetime.now()

                        news_item = NewsItem(
                            ticker=ticker,
                            title=article.get('title', ''),
                            body=article.get('socialimage', '') or '',
                            source=article.get('source', {}).get('name', ''),
                            url=article.get('url', ''),
                            published_at=published_at
                        )
                        news_items.append(news_item)

                    logger.info(f"GDELT: collected {len(news_items)} news for {ticker}")
                    return news_items

                else:
                    logger.error(f"GDELT API error: {response.status}")
                    return []

        except Exception as e:
            logger.error(f"GDELT collection failed: {e}")
            return []


class NewsCollectorFactory:

    @staticmethod
    def create_collector(source: str, api_key: Optional[str] = None) -> BaseNewsCollector:
        source_lower = source.lower()
        if source_lower == 'yahoo':
            return YahooFinanceCollector()
        elif source_lower == 'newsapi':
            if not api_key:
                raise ValueError("NewsAPI requires an API key")
            return NewsAPICollector(api_key)
        elif source_lower == 'gdelt':
            return GDELTCollector()
        else:
            raise ValueError(f"Unsupported news source: {source}")


class BatchNewsCollector:

    def __init__(self, collectors: List[BaseNewsCollector], db_connection):
        self.collectors = collectors
        self.db = db_connection

    async def collect_batch(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
        max_concurrent: int = 5
    ) -> Dict[str, int]:
        semaphore = asyncio.Semaphore(max_concurrent)
        results = {ticker: 0 for ticker in tickers}

        async def collect_for_ticker(ticker: str):
            async with semaphore:
                all_news = []

                for collector in self.collectors:
                    try:
                        news_items = await collector.collect_news(ticker, start_date, end_date)
                        all_news.extend(news_items)
                    except Exception as e:
                        logger.error(f"{collector.__class__.__name__} failed for {ticker}: {e}")

                unique_news = self._deduplicate_news(all_news)

                count = self._save_to_db(ticker, unique_news)
                results[ticker] = count

                logger.info(f"Completed {ticker}: saved {count} news items")

        tasks = [collect_for_ticker(ticker) for ticker in tickers]
        await asyncio.gather(*tasks)

        return results

    def _deduplicate_news(self, news_items: List[NewsItem]) -> List[NewsItem]:
        seen = set()
        unique_news = []

        for news in news_items:
            key = f"{news.title}_{news.published_at.strftime('%Y%m%d')}"
            if key not in seen:
                seen.add(key)
                unique_news.append(news)

        return unique_news

    def _save_to_db(self, ticker: str, news_items: List[NewsItem]) -> int:
        count = 0

        with self.db.cursor() as cur:
            for news in news_items:
                try:
                    cur.execute(
                        "SELECT news_id FROM news_items WHERE ticker = %s AND title = %s AND published_at = %s",
                        (ticker, news.title, news.published_at)
                    )
                    existing = cur.fetchone()

                    if not existing:
                        cur.execute(
                            """INSERT INTO news_items (ticker, title, body, source, url, published_at, collected_at)
                               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                            (ticker, news.title, news.body, news.source, news.url,
                             news.published_at, news.collected_at)
                        )
                        count += 1

                except Exception as e:
                    logger.error(f"Failed saving news for {ticker}: {e}")

            self.db.commit()

        return count


if __name__ == "__main__":
    async def main():
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        start_date = datetime.now() - timedelta(days=180)
        end_date = datetime.now()

        collectors = [
            NewsCollectorFactory.create_collector('yahoo'),
            NewsCollectorFactory.create_collector('gdelt')
        ]

        db = psycopg2.connect(
            host="localhost",
            database="creditmosaic",
            user="postgres",
            password="password"
        )

        async with asyncio.TaskGroup() as tg:
            for collector in collectors:
                tg.create_task(collector.__aenter__())

            batch_collector = BatchNewsCollector(collectors, db)
            results = await batch_collector.collect_batch(tickers, start_date, end_date)

            for collector in collectors:
                tg.create_task(collector.__aexit__(None, None, None))

        print("Collection complete:")
        for ticker, count in results.items():
            print(f"{ticker}: {count} news items")

        db.close()

    asyncio.run(main())
