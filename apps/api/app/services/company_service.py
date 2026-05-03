"""
Company Service
Business logic for company queries, enrichment, and sector analysis.
"""

import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CompanyService:
    """Service for company data access and enrichment."""

    def __init__(self, db):
        self.db = db

    def list_companies(
        self,
        sector: Optional[str] = None,
        search: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        conditions = []
        params = []

        if sector:
            conditions.append("sector = %s")
            params.append(sector)
        if search:
            conditions.append("(ticker ILIKE %s OR company_name ILIKE %s)")
            params.extend([f"%{search}%", f"%{search}%"])

        where = " AND ".join(conditions) if conditions else "1=1"

        count_sql = f"SELECT COUNT(*) FROM companies WHERE {where}"
        total = 0
        with self.db.cursor() as cur:
            cur.execute(count_sql, params)
            row = cur.fetchone()
            total = row[0] if row else 0

        offset = (page - 1) * page_size
        sql = f"""
            SELECT ticker, company_name, sector, industry, exchange,
                   market_cap, country, founded_year, created_at, updated_at
            FROM companies
            WHERE {where}
            ORDER BY market_cap DESC NULLS LAST
            LIMIT %s OFFSET %s
        """
        with self.db.cursor() as cur:
            cur.execute(sql, params + [page_size, offset])
            columns = [desc[0] for desc in cur.description]
            rows = [dict(zip(columns, row)) for row in cur.fetchall()]

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": max(1, (total + page_size - 1) // page_size),
            "items": rows,
        }

    def get_company_detail(self, ticker: str) -> Optional[Dict[str, Any]]:
        sql = """
            SELECT ticker, company_name, sector, industry, exchange,
                   market_cap, country, founded_year, created_at, updated_at
            FROM companies WHERE ticker = %s
        """
        with self.db.cursor() as cur:
            cur.execute(sql, (ticker,))
            row = cur.fetchone()
            if not row:
                return None
            columns = [desc[0] for desc in cur.description]
            company = dict(zip(columns, row))

        latest_risk = self._get_latest_risk(ticker)
        company["latest_risk_score"] = latest_risk["risk_score"] if latest_risk else None
        company["risk_level"] = latest_risk["risk_level"] if latest_risk else None
        company["news_count_30d"] = self._get_news_count(ticker, 30)
        company["high_risk_news_count_30d"] = self._get_high_risk_news_count(ticker, 30)
        company["latest_price"] = self._get_latest_price(ticker)
        company["price_change_5d"] = self._get_price_change(ticker, 5)

        return company

    def get_companies_by_sector(self) -> List[Dict[str, Any]]:
        sql = """
            SELECT sector, COUNT(*) as company_count,
                   AVG(market_cap) as avg_market_cap
            FROM companies
            WHERE sector IS NOT NULL
            GROUP BY sector
            ORDER BY company_count DESC
        """
        with self.db.cursor() as cur:
            cur.execute(sql)
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    def search_companies(self, query: str, sector: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        conditions = ["(ticker ILIKE %s OR company_name ILIKE %s)"]
        params = [f"%{query}%", f"%{query}%"]
        if sector:
            conditions.append("sector = %s")
            params.append(sector)

        where = " AND ".join(conditions)
        sql = f"""
            SELECT ticker, company_name, sector, industry, exchange, market_cap
            FROM companies
            WHERE {where}
            ORDER BY
                CASE WHEN ticker ILIKE %s THEN 0 ELSE 1 END,
                market_cap DESC NULLS LAST
            LIMIT %s
        """
        params_with_order = params + [f"%{query}%", limit]
        with self.db.cursor() as cur:
            cur.execute(sql, params_with_order)
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    def upsert_company(self, data: Dict[str, Any]) -> Dict[str, Any]:
        with self.db.cursor() as cur:
            cur.execute("""
                INSERT INTO companies (ticker, company_name, sector, industry, exchange, market_cap, country, founded_year)
                VALUES (%(ticker)s, %(company_name)s, %(sector)s, %(industry)s, %(exchange)s, %(market_cap)s, %(country)s, %(founded_year)s)
                ON CONFLICT (ticker) DO UPDATE SET
                    company_name = EXCLUDED.company_name,
                    sector = EXCLUDED.sector,
                    industry = EXCLUDED.industry,
                    exchange = EXCLUDED.exchange,
                    market_cap = EXCLUDED.market_cap,
                    country = EXCLUDED.country,
                    founded_year = EXCLUDED.founded_year
            """, data)
            self.db.commit()
        return self.get_company_detail(data["ticker"])

    def delete_company(self, ticker: str) -> bool:
        with self.db.cursor() as cur:
            cur.execute("DELETE FROM companies WHERE ticker = %s", (ticker,))
            self.db.commit()
            return cur.rowcount > 0

    def get_company_news(self, ticker: str, limit: int = 50) -> List[Dict[str, Any]]:
        sql = """
            SELECT ni.news_id, ni.ticker, ni.title, ni.body, ni.source, ni.url,
                   ni.published_at, ni.is_processed,
                   lns.sentiment_score, lns.credit_risk_score, lns.event_type,
                   lns.market_impact_type, lns.confidence
            FROM news_items ni
            LEFT JOIN llm_news_signals lns ON ni.news_id = lns.news_id
            WHERE ni.ticker = %s
            ORDER BY ni.published_at DESC
            LIMIT %s
        """
        with self.db.cursor() as cur:
            cur.execute(sql, (ticker, limit))
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    def _get_latest_risk(self, ticker: str) -> Optional[Dict[str, Any]]:
        sql = """
            SELECT risk_score, risk_level
            FROM risk_scores
            WHERE ticker = %s
            ORDER BY date DESC LIMIT 1
        """
        with self.db.cursor() as cur:
            cur.execute(sql, (ticker,))
            row = cur.fetchone()
            if row:
                return {"risk_score": float(row[0]), "risk_level": row[1]}
        return None

    def _get_news_count(self, ticker: str, days: int) -> int:
        sql = """
            SELECT COUNT(*)
            FROM news_items
            WHERE ticker = %s AND published_at >= %s
        """
        cutoff = datetime.now() - timedelta(days=days)
        with self.db.cursor() as cur:
            cur.execute(sql, (ticker, cutoff))
            row = cur.fetchone()
            return row[0] if row else 0

    def _get_high_risk_news_count(self, ticker: str, days: int) -> int:
        sql = """
            SELECT COUNT(*)
            FROM news_items ni
            JOIN llm_news_signals lns ON ni.news_id = lns.news_id
            WHERE ni.ticker = %s
              AND ni.published_at >= %s
              AND lns.credit_risk_score >= 70
        """
        cutoff = datetime.now() - timedelta(days=days)
        with self.db.cursor() as cur:
            cur.execute(sql, (ticker, cutoff))
            row = cur.fetchone()
            return row[0] if row else 0

    def _get_latest_price(self, ticker: str) -> Optional[float]:
        sql = """
            SELECT close_price FROM daily_market_data
            WHERE ticker = %s
            ORDER BY date DESC LIMIT 1
        """
        with self.db.cursor() as cur:
            cur.execute(sql, (ticker,))
            row = cur.fetchone()
            return float(row[0]) if row else None

    def _get_price_change(self, ticker: str, days: int) -> Optional[float]:
        sql = """
            SELECT close_price FROM daily_market_data
            WHERE ticker = %s AND date >= %s
            ORDER BY date ASC LIMIT 1
        """
        cutoff = (datetime.now() - timedelta(days=days + 2)).date()
        with self.db.cursor() as cur:
            cur.execute(sql, (ticker, cutoff))
            row = cur.fetchone()
            if not row:
                return None
            old_price = float(row[0])

        latest = self._get_latest_price(ticker)
        if latest and old_price:
            return (latest - old_price) / old_price
        return None
