"""
Portfolio Service
Business logic for portfolio risk analysis, management, and optimization suggestions.
"""

import json
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


class PortfolioService:
    """Service for portfolio risk analysis and management."""

    def __init__(self, db):
        self.db = db

    def normalize_holdings(self, holdings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Uppercase tickers, combine duplicates, and normalize weights to 1.0."""
        combined: Dict[str, float] = {}
        for holding in holdings:
            ticker = str(holding.get("ticker", "")).strip().upper()
            weight = float(holding.get("weight", 0) or 0)
            if not ticker or weight <= 0:
                continue
            combined[ticker] = combined.get(ticker, 0.0) + weight

        total = sum(combined.values())
        if total <= 0:
            raise ValueError("Portfolio holdings must contain at least one positive weight")

        return [
            {"ticker": ticker, "weight": weight / total}
            for ticker, weight in combined.items()
        ]

    def analyze_portfolio(
        self,
        holdings: List[Dict[str, Any]],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        holdings = self.normalize_holdings(holdings)
        tickers = [h["ticker"] for h in holdings]
        weight_map = {h["ticker"]: h["weight"] for h in holdings}

        scores = self._fetch_risk_scores(tickers)
        score_map = {s["ticker"]: s for s in scores}

        holdings_risk = []
        total_risk = 0.0
        contributions = []

        for h in holdings:
            t = h["ticker"]
            w = h["weight"]
            s = score_map.get(t, {})
            rs = float(s.get("risk_score", 0))
            rl = s.get("risk_level", "Unknown")

            contribution = w * rs
            total_risk += contribution

            holdings_risk.append({
                "ticker": t,
                "company_name": s.get("company_name"),
                "weight": w,
                "risk_score": rs,
                "risk_level": rl,
                "risk_contribution": contribution,
                "top_drivers": s.get("top_features"),
            })
            contributions.append({"ticker": t, "contribution": contribution, "weight": w})

        risk_level = self._classify_risk_level(total_risk)
        top_contributors = sorted(contributions, key=lambda x: x["contribution"], reverse=True)[:5]

        div_score = self._compute_diversification(holdings_risk)

        recommendation = self._generate_recommendation(total_risk, risk_level, top_contributors, score_map)

        portfolio_id = None
        if name:
            portfolio_id = self._save_portfolio(name, description, holdings)
            self._save_analysis(portfolio_id, total_risk, risk_level, holdings_risk, top_contributors, div_score)

        return {
            "portfolio_id": portfolio_id,
            "name": name,
            "total_risk_score": round(total_risk, 6),
            "risk_level": risk_level,
            "holdings_risk": holdings_risk,
            "top_contributors": top_contributors,
            "diversification_score": round(div_score, 4) if div_score else None,
            "recommendation": recommendation,
        }

    def list_portfolios(self) -> List[Dict[str, Any]]:
        sql = """
            SELECT p.portfolio_id, p.name, p.description, p.holdings, p.created_at,
                   pa.total_risk_score, pa.risk_level
            FROM portfolios p
            LEFT JOIN LATERAL (
                SELECT total_risk_score, risk_level
                FROM portfolio_analyses
                WHERE portfolio_id = p.portfolio_id
                ORDER BY created_at DESC LIMIT 1
            ) pa ON TRUE
            ORDER BY p.created_at DESC
        """
        with self.db.cursor() as cur:
            cur.execute(sql)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

        result = []
        for row in rows:
            d = dict(zip(columns, row))
            holdings = d.get("holdings")
            if isinstance(holdings, str):
                holdings = json.loads(holdings)
            result.append({
                "portfolio_id": d["portfolio_id"],
                "name": d["name"],
                "description": d.get("description"),
                "holdings_count": len(holdings) if holdings else 0,
                "total_risk_score": float(d["total_risk_score"]) if d.get("total_risk_score") else None,
                "risk_level": d.get("risk_level"),
                "created_at": d.get("created_at"),
            })
        return result

    def get_portfolio(self, portfolio_id: int) -> Optional[Dict[str, Any]]:
        sql = "SELECT * FROM portfolios WHERE portfolio_id = %s"
        with self.db.cursor() as cur:
            cur.execute(sql, (portfolio_id,))
            columns = [desc[0] for desc in cur.description]
            row = cur.fetchone()
            if not row:
                return None
            d = dict(zip(columns, row))

        holdings = d.get("holdings")
        if isinstance(holdings, str):
            holdings = json.loads(holdings)
        d["holdings"] = holdings

        latest = self._get_latest_analysis(portfolio_id)
        d["latest_analysis"] = latest
        return d

    def delete_portfolio(self, portfolio_id: int) -> bool:
        with self.db.cursor() as cur:
            cur.execute("DELETE FROM portfolios WHERE portfolio_id = %s", (portfolio_id,))
            self.db.commit()
            return cur.rowcount > 0

    def _fetch_risk_scores(self, tickers: List[str]) -> List[Dict[str, Any]]:
        if not tickers:
            return []
        placeholders = ", ".join(["%s"] * len(tickers))
        sql = f"""
            SELECT DISTINCT ON (rs.ticker)
                rs.ticker, rs.date, rs.risk_score, rs.risk_level,
                rs.model_version, rs.top_features,
                c.company_name
            FROM risk_scores rs
            LEFT JOIN companies c ON rs.ticker = c.ticker
            WHERE rs.ticker IN ({placeholders})
            ORDER BY rs.ticker, rs.date DESC
        """
        with self.db.cursor() as cur:
            cur.execute(sql, tickers)
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    def _classify_risk_level(self, score: float) -> str:
        if score < 0.25:
            return "Low"
        elif score < 0.50:
            return "Medium"
        elif score < 0.75:
            return "High"
        return "Critical"

    def _compute_diversification(self, holdings_risk: List[Dict[str, Any]]) -> Optional[float]:
        if len(holdings_risk) < 2:
            return None

        max_single = max(h["weight"] for h in holdings_risk)
        concentration_penalty = max_single * 0.5 + (1 - 1 / len(holdings_risk)) * 0.5

        risk_values = [h["risk_score"] for h in holdings_risk]
        if max(risk_values) - min(risk_values) > 0:
            risk_spread = 1 - (max(risk_values) - min(risk_values))
        else:
            risk_spread = 0.5
        risk_spread = max(0, min(1, risk_spread))

        return round(1 - (concentration_penalty * 0.6 + (1 - risk_spread) * 0.4), 4)

    def _generate_recommendation(
        self,
        total_risk: float,
        risk_level: str,
        top_contributors: List[Dict[str, Any]],
        score_map: Dict[str, Any],
    ) -> Optional[str]:
        if not top_contributors:
            return None

        top = top_contributors[0]
        if risk_level in ("High", "Critical"):
            return (
                f"Portfolio risk is {risk_level}. "
                f"The top contributor is {top['ticker']} "
                f"(weight: {top['weight']:.0%}, risk: {score_map.get(top['ticker'], {}).get('risk_score', 0):.3f}). "
                f"Consider reducing exposure to {top['ticker']} or hedging with credit protection."
            )
        elif risk_level == "Medium":
            return (
                f"Portfolio risk is moderate. {top['ticker']} is the largest risk contributor. "
                f"Monitor news signals for {top['ticker']} closely."
            )
        return "Portfolio risk is well-managed. No immediate action recommended."

    def _save_portfolio(self, name: str, description: Optional[str], holdings: List[Dict[str, Any]]) -> int:
        with self.db.cursor() as cur:
            cur.execute(
                """INSERT INTO portfolios (name, description, holdings)
                   VALUES (%s, %s, %s) RETURNING portfolio_id""",
                (name, description, json.dumps(holdings, default=str)),
            )
            self.db.commit()
            return cur.fetchone()[0]

    def _save_analysis(
        self,
        portfolio_id: int,
        total_risk_score: float,
        risk_level: str,
        holdings_risk: List[Dict[str, Any]],
        top_contributors: List[Dict[str, Any]],
        diversification_score: Optional[float],
    ):
        with self.db.cursor() as cur:
            cur.execute(
                """INSERT INTO portfolio_analyses
                   (portfolio_id, total_risk_score, risk_level, holdings_data, top_contributors, diversification_score)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (
                    portfolio_id, total_risk_score, risk_level,
                    json.dumps(holdings_risk, default=str),
                    json.dumps(top_contributors, default=str),
                    diversification_score,
                ),
            )
            self.db.commit()

    def _get_latest_analysis(self, portfolio_id: int) -> Optional[Dict[str, Any]]:
        sql = """
            SELECT * FROM portfolio_analyses
            WHERE portfolio_id = %s
            ORDER BY created_at DESC LIMIT 1
        """
        with self.db.cursor() as cur:
            cur.execute(sql, (portfolio_id,))
            columns = [desc[0] for desc in cur.description]
            row = cur.fetchone()
            if not row:
                return None
            d = dict(zip(columns, row))
            for json_col in ("holdings_data", "top_contributors"):
                if isinstance(d.get(json_col), str):
                    d[json_col] = json.loads(d[json_col])
            return d

    # ------------------------------------------------------------------
    # Part 7: Advanced risk analytics & stress testing
    # ------------------------------------------------------------------

    def analyze_with_correlations(
        self,
        holdings: List[Dict[str, Any]],
        days: int = 60,
    ) -> Dict[str, Any]:
        """Deep portfolio analysis with return correlations and risk decomposition.

        Computes:
          - Pairwise return correlation matrix from daily market data
          - Systematic vs idiosyncratic risk decomposition
          - Diversification ratio
          - Marginal risk contribution per holding
        """
        holdings = self.normalize_holdings(holdings)
        tickers = [h["ticker"] for h in holdings]
        weights = {h["ticker"]: h["weight"] for h in holdings}

        returns_df = self._fetch_returns_matrix(tickers, days)
        if returns_df.empty or returns_df.shape[1] < 2:
            return {"error": "Insufficient market data for correlation analysis", "holdings_risk": []}

        portfolio_weights = pd.Series(weights)
        aligned_tickers = [t for t in tickers if t in returns_df.columns]
        if len(aligned_tickers) < 2:
            return {"error": "Need at least 2 tickers with overlapping data"}

        w = portfolio_weights[aligned_tickers]
        w = w / w.sum()

        rets = returns_df[aligned_tickers].dropna()
        cov = rets.cov() * 252
        corr = rets.corr()

        portfolio_vol = float((w @ cov @ w) ** 0.5)
        individual_vols = pd.Series({t: float((cov.loc[t, t]) ** 0.5) for t in aligned_tickers})

        # Marginal contribution to risk
        mctr = (cov @ w) / portfolio_vol if portfolio_vol > 0 else pd.Series(0, index=w.index)
        risk_contributions = (w * mctr).to_dict()

        # Diversification ratio
        weighted_avg_vol = float((w * individual_vols).sum())
        div_ratio = float(portfolio_vol / weighted_avg_vol) if weighted_avg_vol > 0 else 1.0

        # Correlation matrix as list-of-lists for frontend heatmap
        corr_matrix = corr.values.tolist()

        # Build per-holding detail
        scores = {s["ticker"]: s for s in self._fetch_risk_scores(aligned_tickers)}
        holdings_detail = []
        for t in aligned_tickers:
            s = scores.get(t, {})
            holdings_detail.append({
                "ticker": t,
                "company_name": s.get("company_name"),
                "weight": float(w[t]),
                "risk_score": float(s.get("risk_score", 0)),
                "risk_level": s.get("risk_level", "Unknown"),
                "volatility_annual": float(individual_vols[t]),
                "risk_contribution_pct": float(risk_contributions.get(t, 0)),
                "marginal_ctr": float(mctr[t]) if t in mctr else 0.0,
                "top_drivers": s.get("top_features"),
            })

        return {
            "tickers": aligned_tickers,
            "correlation_matrix": corr_matrix,
            "portfolio_volatility_annual": portfolio_vol,
            "diversification_ratio": round(div_ratio, 4),
            "total_risk_score": round(float(w @ pd.Series({t: float(scores.get(t, {}).get("risk_score", 0)) for t in aligned_tickers})), 6),
            "risk_level": self._classify_risk_level(float(w @ pd.Series({t: float(scores.get(t, {}).get("risk_score", 0)) for t in aligned_tickers}))),
            "holdings_risk": holdings_detail,
            "risk_contributions": risk_contributions,
        }

    def _fetch_returns_matrix(self, tickers: List[str], days: int) -> pd.DataFrame:
        placeholders = ", ".join(["%s"] * len(tickers))
        sql = f"""
            SELECT ticker, date, returns_1d
            FROM daily_market_data
            WHERE ticker IN ({placeholders})
              AND date >= CURRENT_DATE - (%s * INTERVAL '1 day')
            ORDER BY date
        """
        with self.db.cursor() as cur:
            cur.execute(sql, tickers + [days])
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=columns)
        df["returns_1d"] = pd.to_numeric(df["returns_1d"], errors="coerce")
        return df.pivot(index="date", columns="ticker", values="returns_1d")

    def stress_test(
        self,
        holdings: List[Dict[str, Any]],
        scenarios: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run stress scenarios against the portfolio.

        Built-in scenarios:
          - credit_spread_shock: HYG +2%, LQD +1%, equity -5%
          - equity_crash: equities -20%, vol spike 3x
          - liquidity_crisis: volumes down 50%, spreads widen
          - rates_hike: rates +100bp, duration impact
        """
        holdings = self.normalize_holdings(holdings)

        if scenarios is None:
            scenarios = [
                {
                    "name": "Credit Spread Shock",
                    "equity_shock": -0.05,
                    "credit_widening_pct": 0.02,
                    "vol_multiplier": 2.0,
                },
                {
                    "name": "Equity Crash",
                    "equity_shock": -0.20,
                    "credit_widening_pct": 0.015,
                    "vol_multiplier": 3.0,
                },
                {
                    "name": "Liquidity Crisis",
                    "equity_shock": -0.08,
                    "credit_widening_pct": 0.03,
                    "vol_multiplier": 2.5,
                },
            ]

        tickers = [h["ticker"] for h in holdings]
        weight_map = {h["ticker"]: h["weight"] for h in holdings}
        scores = self._fetch_risk_scores(tickers)
        score_map = {s["ticker"]: s for s in scores}

        base_risk = sum(
            weight_map.get(t, 0) * float(score_map.get(t, {}).get("risk_score", 0))
            for t in tickers
        )

        results = []
        for sc in scenarios:
            shocked_risk = 0.0
            items = []
            for h in holdings:
                t = h["ticker"]
                base_rs = float(score_map.get(t, {}).get("risk_score", 0))
                eq_shock_effect = abs(float(sc.get("equity_shock", 0))) * 0.3
                credit_effect = float(sc.get("credit_widening_pct", 0)) * 0.5
                shocked_rs = min(1.0, base_rs + eq_shock_effect + credit_effect)
                shocked_risk += h["weight"] * shocked_rs
                items.append({
                    "ticker": t,
                    "base_risk": base_rs,
                    "shocked_risk": round(shocked_rs, 4),
                    "delta": round(shocked_rs - base_rs, 4),
                })

            results.append({
                "scenario": sc["name"],
                "base_portfolio_risk": round(base_risk, 6),
                "shocked_portfolio_risk": round(shocked_risk, 6),
                "risk_increase_pct": round((shocked_risk - base_risk) / base_risk * 100, 2) if base_risk > 0 else 0,
                "holdings": items,
            })

        return {
            "tickers": tickers,
            "base_portfolio_risk": round(base_risk, 6),
            "scenarios": results,
        }

    def generate_portfolio_report_markdown(
        self,
        holdings: List[Dict[str, Any]],
        correlation_data: Optional[Dict[str, Any]] = None,
        stress_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a one-page portfolio risk report in markdown format."""
        holdings = self.normalize_holdings(holdings)
        tickers = [h["ticker"] for h in holdings]
        weight_map = {h["ticker"]: h["weight"] for h in holdings}
        scores = self._fetch_risk_scores(tickers)
        score_map = {s["ticker"]: s for s in scores}

        lines = [
            "# Portfolio Risk Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Holdings:** {len(holdings)} positions",
            "",
            "---",
            "",
            "## Holdings Summary",
            "",
            "| Ticker | Company | Weight | Risk Score | Risk Level | Risk Contribution |",
            "|--------|---------|--------|------------|------------|------------------|",
        ]

        total_risk = 0.0
        for h in holdings:
            t = h["ticker"]
            s = score_map.get(t, {})
            rs = float(s.get("risk_score", 0))
            contrib = h["weight"] * rs
            total_risk += contrib
            lines.append(
                f"| {t} | {s.get('company_name', t)} | {h['weight']:.1%} | "
                f"{rs:.4f} | {s.get('risk_level', 'N/A')} | {contrib:.4f} |"
            )

        risk_level = self._classify_risk_level(total_risk)
        lines.extend([
            f"| **Total** | | **100%** | | **{risk_level}** | **{total_risk:.4f}** |",
            "",
            "---",
            "",
            "## Risk Contribution Ranking",
            "",
            "| Rank | Ticker | Risk Contribution | % of Total |",
            "|------|--------|------------------|------------|",
        ])

        ranked = sorted(holdings, key=lambda h: h["weight"] * float(score_map.get(h["ticker"], {}).get("risk_score", 0)), reverse=True)
        for i, h in enumerate(ranked):
            t = h["ticker"]
            contrib = h["weight"] * float(score_map.get(t, {}).get("risk_score", 0))
            pct = (contrib / total_risk * 100) if total_risk > 0 else 0
            lines.append(f"| {i + 1} | {t} | {contrib:.4f} | {pct:.1f}% |")

        if correlation_data and "error" not in correlation_data:
            lines.extend([
                "",
                "---",
                "",
                "## Risk Decomposition",
                "",
                f"- **Portfolio Volatility (Annual):** {correlation_data.get('portfolio_volatility_annual', 0):.2%}",
                f"- **Diversification Ratio:** {correlation_data.get('diversification_ratio', 0):.3f}",
                f"- **Interpretation:** {'Well-diversified' if correlation_data.get('diversification_ratio', 1) < 0.7 else 'Concentrated risk'} portfolio",
            ])

        if stress_data and "scenarios" in stress_data:
            lines.extend([
                "",
                "---",
                "",
                "## Stress Test Results",
                "",
                "| Scenario | Base Risk | Shocked Risk | Increase |",
                "|----------|-----------|-------------|----------|",
            ])
            for sc in stress_data["scenarios"]:
                lines.append(
                    f"| {sc['scenario']} | {sc['base_portfolio_risk']:.4f} | "
                    f"{sc['shocked_portfolio_risk']:.4f} | +{sc['risk_increase_pct']:.1f}% |"
                )

        lines.extend([
            "",
            "---",
            "",
            "## Recommendation",
            "",
            self._generate_recommendation(
                total_risk, risk_level,
                [{"ticker": h["ticker"], "contribution": h["weight"] * float(score_map.get(h["ticker"], {}).get("risk_score", 0)), "weight": h["weight"]}
                 for h in ranked[:5]],
                score_map,
            ) or "No specific recommendation.",
            "",
            "---",
            "",
            "*Report generated by CreditMosaic AI Risk Engine*",
        ])

        return "\n".join(lines)
