"""
Report Service
Business logic for AI-powered risk report generation and management.

Uses the LLM provider to synthesize company risk data, news signals,
and market reactions into a structured markdown report.
"""

import json
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ReportService:
    """Service for generating and managing AI risk reports."""

    def __init__(self, db, llm_manager=None):
        self.db = db
        self.llm_manager = llm_manager

    def generate_report(
        self,
        ticker: str,
        report_type: str = "company_risk",
        provider_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        context = self._gather_report_context(ticker)
        if not context.get("company"):
            raise ValueError(f"Company '{ticker}' not found")

        markdown = self._render_report(context)
        title = f"Risk Report: {context['company']['company_name']} ({ticker})"

        summary = {
            "ticker": ticker,
            "risk_score": context.get("latest_risk"),
            "risk_level": context.get("risk_level"),
            "event_type_distribution": context.get("event_distribution", {}),
            "market_impact_summary": context.get("impact_summary", {}),
        }

        if self.llm_manager:
            try:
                markdown = self._generate_with_llm(context, provider_name)
                model_used = provider_name or "default"
            except Exception as e:
                logger.warning(f"LLM report generation failed, using template: {e}")
                model_used = "template"
        else:
            model_used = "template"

        report_id = self._save_report(ticker, report_type, title, markdown, summary, model_used)

        return {
            "report_id": report_id,
            "ticker": ticker,
            "report_type": report_type,
            "title": title,
            "markdown_content": markdown,
            "summary": summary,
            "model_used": model_used,
            "generated_at": datetime.now(),
        }

    def get_report(self, report_id: int) -> Optional[Dict[str, Any]]:
        sql = "SELECT * FROM risk_reports WHERE report_id = %s"
        with self.db.cursor() as cur:
            cur.execute(sql, (report_id,))
            columns = [desc[0] for desc in cur.description]
            row = cur.fetchone()
            if not row:
                return None
            return dict(zip(columns, row))

    def list_reports(self, ticker: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        conditions = []
        params = []
        if ticker:
            conditions.append("ticker = %s")
            params.append(ticker)

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"""
            SELECT report_id, ticker, title, report_type, generated_at
            FROM risk_reports
            WHERE {where}
            ORDER BY generated_at DESC
            LIMIT %s
        """
        params.append(limit)
        with self.db.cursor() as cur:
            cur.execute(sql, params)
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    def get_company_latest_report(self, ticker: str) -> Optional[Dict[str, Any]]:
        sql = """
            SELECT * FROM risk_reports
            WHERE ticker = %s
            ORDER BY generated_at DESC LIMIT 1
        """
        with self.db.cursor() as cur:
            cur.execute(sql, (ticker,))
            columns = [desc[0] for desc in cur.description]
            row = cur.fetchone()
            return dict(zip(columns, row)) if row else None

    def delete_report(self, report_id: int) -> bool:
        with self.db.cursor() as cur:
            cur.execute("DELETE FROM risk_reports WHERE report_id = %s", (report_id,))
            self.db.commit()
            return cur.rowcount > 0

    def _gather_report_context(self, ticker: str) -> Dict[str, Any]:
        context = {}
        context["company"] = self._get_company(ticker)
        if not context["company"]:
            return context

        context["latest_risk"] = self._get_latest_risk(ticker)
        context["risk_level"] = self._get_risk_level(context["latest_risk"])
        context["risk_history"] = self._get_risk_history(ticker, 90)
        context["recent_news"] = self._get_recent_news(ticker, 30)
        context["recent_signals"] = self._get_recent_signals(ticker, 30)
        context["event_distribution"] = self._get_event_distribution(ticker, 90)
        context["impact_summary"] = self._get_impact_summary(ticker, 90)
        context["market_snapshot"] = self._get_market_snapshot(ticker)

        return context

    def _get_company(self, ticker: str) -> Optional[Dict[str, Any]]:
        sql = "SELECT * FROM companies WHERE ticker = %s"
        with self.db.cursor() as cur:
            cur.execute(sql, (ticker,))
            columns = [desc[0] for desc in cur.description]
            row = cur.fetchone()
            return dict(zip(columns, row)) if row else None

    def _get_latest_risk(self, ticker: str) -> Optional[float]:
        sql = """
            SELECT risk_score FROM risk_scores
            WHERE ticker = %s ORDER BY date DESC LIMIT 1
        """
        with self.db.cursor() as cur:
            cur.execute(sql, (ticker,))
            row = cur.fetchone()
            return float(row[0]) if row else None

    def _get_risk_level(self, score: Optional[float]) -> str:
        if score is None:
            return "Unknown"
        if score < 0.25:
            return "Low"
        elif score < 0.50:
            return "Medium"
        elif score < 0.75:
            return "High"
        return "Critical"

    def _get_risk_history(self, ticker: str, days: int) -> List[Dict[str, Any]]:
        sql = """
            SELECT date, risk_score, risk_level
            FROM risk_scores
            WHERE ticker = %s AND date >= %s
            ORDER BY date DESC
        """
        cutoff = (datetime.now() - timedelta(days=days)).date()
        with self.db.cursor() as cur:
            cur.execute(sql, (ticker, cutoff))
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    def _get_recent_news(self, ticker: str, days: int) -> List[Dict[str, Any]]:
        sql = """
            SELECT ni.news_id, ni.title, ni.published_at, ni.source,
                   lns.credit_risk_score, lns.event_type, lns.market_impact_type, lns.confidence
            FROM news_items ni
            LEFT JOIN llm_news_signals lns ON ni.news_id = lns.news_id
            WHERE ni.ticker = %s AND ni.published_at >= %s
            ORDER BY ni.published_at DESC
            LIMIT 20
        """
        cutoff = datetime.now() - timedelta(days=days)
        with self.db.cursor() as cur:
            cur.execute(sql, (ticker, cutoff))
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    def _get_recent_signals(self, ticker: str, days: int) -> List[Dict[str, Any]]:
        sql = """
            SELECT signal_id, credit_risk_score, event_type,
                   market_impact_type, confidence, extracted_at
            FROM llm_news_signals
            WHERE ticker = %s AND extracted_at >= %s
            ORDER BY extracted_at DESC
            LIMIT 20
        """
        cutoff = datetime.now() - timedelta(days=days)
        with self.db.cursor() as cur:
            cur.execute(sql, (ticker, cutoff))
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    def _get_event_distribution(self, ticker: str, days: int) -> Dict[str, int]:
        sql = """
            SELECT event_type, COUNT(*) as count
            FROM llm_news_signals
            WHERE ticker = %s AND extracted_at >= %s
            GROUP BY event_type
            ORDER BY count DESC
        """
        cutoff = datetime.now() - timedelta(days=days)
        with self.db.cursor() as cur:
            cur.execute(sql, (ticker, cutoff))
            return {row[0]: row[1] for row in cur.fetchall()}

    def _get_impact_summary(self, ticker: str, days: int) -> Dict[str, int]:
        sql = """
            SELECT market_impact_type, COUNT(*) as count
            FROM llm_news_signals
            WHERE ticker = %s AND extracted_at >= %s
            GROUP BY market_impact_type
            ORDER BY count DESC
        """
        cutoff = datetime.now() - timedelta(days=days)
        with self.db.cursor() as cur:
            cur.execute(sql, (ticker, cutoff))
            return {row[0]: row[1] for row in cur.fetchall()}

    def _get_market_snapshot(self, ticker: str) -> Optional[Dict[str, Any]]:
        sql = """
            SELECT date, close_price, volume, returns_1d, returns_5d, returns_20d,
                   volatility_5d, volatility_20d
            FROM daily_market_data
            WHERE ticker = %s
            ORDER BY date DESC LIMIT 5
        """
        with self.db.cursor() as cur:
            cur.execute(sql, (ticker,))
            columns = [desc[0] for desc in cur.description]
            rows = [dict(zip(columns, row)) for row in cur.fetchall()]
        return rows[0] if rows else None

    def _render_report(self, context: Dict[str, Any]) -> str:
        c = context["company"]
        risk = context.get("latest_risk")
        rl = context.get("risk_level", "Unknown")
        news_list = context.get("recent_news", [])
        signals = context.get("recent_signals", [])
        ms = context.get("market_snapshot") or {}
        market_cap = c.get("market_cap")
        market_cap_text = f"${float(market_cap):,.0f}" if market_cap is not None else "N/A"

        lines = [
            f"# Risk Report: {c.get('company_name', 'N/A')} ({c.get('ticker', 'N/A')})",
            "",
            f"**Sector:** {c.get('sector', 'N/A')}  ",
            f"**Industry:** {c.get('industry', 'N/A')}  ",
            f"**Market Cap:** {market_cap_text}  ",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
            "",
            "---",
            "",
            "## Risk Summary",
            "",
            f"**Current Risk Score:** {risk:.4f}" if risk else "**Current Risk Score:** N/A",
            f"**Risk Level:** {rl}  ",
            "",
            "### Risk Level Guide",
            "",
            "| Level | Range | Description |",
            "|-------|-------|-------------|",
            "| Low | 0.00 - 0.25 | Minimal risk indicators |",
            "| Medium | 0.25 - 0.50 | Moderate risk; monitor closely |",
            "| High | 0.50 - 0.75 | Elevated risk; consider hedging |",
            "| Critical | 0.75 - 1.00 | Severe risk; immediate action recommended |",
            "",
            "---",
            "",
            "## Market Snapshot",
            "",
            f"- **Latest Price:** ${ms.get('close_price', 'N/A')}",
            f"- **1-Day Return:** {ms.get('returns_1d', 0):.4%}" if ms.get('returns_1d') is not None else "- **1-Day Return:** N/A",
            f"- **5-Day Return:** {ms.get('returns_5d', 0):.4%}" if ms.get('returns_5d') is not None else "- **5-Day Return:** N/A",
            f"- **20-Day Volatility:** {ms.get('volatility_20d', 0):.4%}" if ms.get('volatility_20d') is not None else "- **20-Day Volatility:** N/A",
            "",
            "---",
            "",
            "## Recent News Signals",
            "",
        ]

        if news_list:
            lines.append("| Date | Title | Risk Score | Event Type | Impact |")
            lines.append("|------|-------|-----------|------------|--------|")
            for n in news_list[:10]:
                date_str = n.get("published_at", "")
                if hasattr(date_str, "strftime"):
                    date_str = date_str.strftime("%Y-%m-%d")
                title = (n.get("title") or "")[:60]
                rs = n.get("credit_risk_score", "-")
                et = n.get("event_type", "-")
                mi = n.get("market_impact_type", "-")
                lines.append(f"| {date_str} | {title} | {rs} | {et} | {mi} |")
        else:
            lines.append("*No recent news signals available*")

        lines.extend([
            "",
            "---",
            "",
            "## Event Type Distribution (Last 90 Days)",
            "",
        ])
        ed = context.get("event_distribution", {})
        if ed:
            lines.append("| Event Type | Count |")
            lines.append("|------------|-------|")
            for et, cnt in sorted(ed.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"| {et} | {cnt} |")
        else:
            lines.append("*No event data available*")

        lines.extend([
            "",
            "---",
            "",
            "## Market Impact Summary",
            "",
        ])
        im = context.get("impact_summary", {})
        if im:
            lines.append("| Predicted Impact | Count |")
            lines.append("|------------------|-------|")
            for mi, cnt in im.items():
                lines.append(f"| {mi} | {cnt} |")
        else:
            lines.append("*No impact data available*")

        lines.extend([
            "",
            "---",
            "",
            "*Report generated by CreditMosaic AI Risk Engine*",
        ])
        return "\n".join(lines)

    def _generate_with_llm(self, context: Dict[str, Any], provider_name: Optional[str] = None) -> str:
        import asyncio

        c = context["company"]
        risk = context.get("latest_risk")
        rl = context.get("risk_level", "Unknown")
        news = context.get("recent_news", [])
        signals = context.get("recent_signals", [])

        news_summary = "\n".join(
            f"- [{n.get('event_type', 'N/A')}] {n.get('title', '')} (Risk: {n.get('credit_risk_score', 'N/A')})"
            for n in (news or [])[:10]
        ) or "No recent news"

        signal_summary = f"Total signals (90d): {len(signals)}, "
        if signals:
            high_risk = sum(1 for s in signals if s.get("credit_risk_score", 0) >= 70)
            signal_summary += f"High risk: {high_risk}, "
            signal_summary += f"Avg confidence: {sum(s.get('confidence', 0) for s in signals) / len(signals):.3f}"

        risk_text = f"{risk:.4f}" if risk is not None else "N/A"
        market_cap = c.get("market_cap")
        market_cap_text = f"${float(market_cap):,.0f}" if market_cap is not None else "N/A"

        user_prompt = f"""Generate a professional credit risk analysis report in markdown for:

**Company:** {c.get('company_name')} ({c.get('ticker')})
**Sector:** {c.get('sector', 'N/A')}
**Industry:** {c.get('industry', 'N/A')}
**Market Cap:** {market_cap_text}

**Risk Score:** {risk_text}
**Risk Level:** {rl}

**Recent News:**
{news_summary}

**Signal Stats:** {signal_summary}

Please structure the report with:
1. Executive Summary (2-3 sentences)
2. Risk Assessment with key drivers
3. News Signal Analysis
4. Market Reaction Patterns
5. Forward-Looking Risk Outlook
6. Recommended Actions

Keep it concise and data-driven. Use markdown formatting with headers and tables where appropriate."""

        system_prompt = (
            "You are a senior credit risk analyst. Generate professional, "
            "data-driven risk reports in markdown format. Be concise and specific. "
            "Use the data provided, do not fabricate numbers. "
            "When data is unavailable, clearly state it rather than guessing."
        )

        async def _call_llm():
            return await self.llm_manager.generate_completion(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                provider_name=provider_name,
            )

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        response = loop.run_until_complete(_call_llm())
        return response.content if response and response.content else self._render_report(context)

    def _save_report(
        self,
        ticker: Optional[str],
        report_type: str,
        title: str,
        markdown: str,
        summary: Dict[str, Any],
        model_used: str,
    ) -> int:
        with self.db.cursor() as cur:
            cur.execute(
                """INSERT INTO risk_reports (ticker, report_type, title, markdown_content, summary, model_used)
                   VALUES (%s, %s, %s, %s, %s, %s) RETURNING report_id""",
                (ticker, report_type, title, markdown, json.dumps(summary), model_used),
            )
            self.db.commit()
            return cur.fetchone()[0]
