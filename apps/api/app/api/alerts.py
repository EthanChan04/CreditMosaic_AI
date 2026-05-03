"""Risk alert API router."""

from fastapi import APIRouter, Depends, Query

from apps.api.app.dependencies import get_db

router = APIRouter(tags=["Alerts"])


@router.get("/alerts", summary="List risk alerts")
def list_alerts(
    limit: int = Query(default=100, ge=1, le=500),
    min_risk_score: float = Query(default=0.5, ge=0.0, le=1.0),
    db=Depends(get_db),
):
    """Return recent risk alerts derived from persisted risk scores and LLM signals."""
    sql = """
        SELECT
            rs.id AS alert_id,
            rs.ticker,
            rs.risk_level,
            COALESCE(lns.event_type, 'model_score') AS main_driver,
            CASE
                WHEN rs.risk_level IN ('High', 'Critical') THEN 'Review news evidence and credit proxy reaction'
                WHEN lns.credit_risk_score >= 70 THEN 'Review high-risk news signal'
                ELSE 'Monitor next refresh'
            END AS recommended_review_action,
            rs.created_at
        FROM risk_scores rs
        LEFT JOIN LATERAL (
            SELECT event_type, credit_risk_score
            FROM llm_news_signals
            WHERE ticker = rs.ticker
            ORDER BY extracted_at DESC
            LIMIT 1
        ) lns ON TRUE
        WHERE rs.risk_score >= %s
        ORDER BY rs.created_at DESC
        LIMIT %s
    """
    with db.cursor() as cur:
        cur.execute(sql, (min_risk_score, limit))
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]
