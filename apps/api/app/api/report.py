"""
Report API Router
Endpoints for AI-powered risk report generation, retrieval, and download.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import PlainTextResponse

from apps.api.app.dependencies import get_report_service
from apps.api.app.schemas.report import (
    ReportGenerateRequest, ReportResponse, ReportListResponse,
)
from apps.api.app.schemas.common import SuccessMessage

import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Reports"])


@router.post(
    "/report/generate",
    response_model=ReportResponse,
    summary="Generate risk report",
    description=(
        "Generate an AI-powered credit risk report for a company. "
        "Synthesizes risk scores, news signals, event distributions, and market "
        "reaction patterns into a structured markdown report. Optionally uses LLM "
        "for natural language generation; falls back to a data-driven template."
    ),
)
def generate_report(
    request: ReportGenerateRequest,
    svc=Depends(get_report_service),
):
    try:
        result = svc.generate_report(
            ticker=request.ticker,
            report_type=request.report_type,
            provider_name=request.provider,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Report generation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/report/{report_id}",
    response_model=ReportResponse,
    summary="Get report",
    description="Retrieve a previously generated report by ID.",
)
def get_report(report_id: int, svc=Depends(get_report_service)):
    report = svc.get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
    return report


@router.get(
    "/report/{report_id}/download",
    response_class=PlainTextResponse,
    summary="Download report",
    description="Download a report as raw markdown text.",
)
def download_report(report_id: int, svc=Depends(get_report_service)):
    report = svc.get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
    return PlainTextResponse(
        content=report["markdown_content"],
        media_type="text/markdown",
        headers={
            "Content-Disposition": f'attachment; filename="risk_report_{report_id}.md"'
        },
    )


@router.get(
    "/reports",
    response_model=ReportListResponse,
    summary="List reports",
    description="List generated reports, optionally filtered by ticker.",
)
def list_reports(
    ticker: Optional[str] = Query(default=None, description="Filter by company ticker"),
    limit: int = Query(default=50, ge=1, le=200),
    svc=Depends(get_report_service),
):
    reports = svc.list_reports(ticker=ticker, limit=limit)
    return ReportListResponse(total=len(reports), reports=reports)


@router.get(
    "/report/company/{ticker}",
    response_model=ReportResponse,
    summary="Latest company report",
    description="Get the most recent risk report generated for a company.",
)
def get_company_report(ticker: str, svc=Depends(get_report_service)):
    report = svc.get_company_latest_report(ticker)
    if not report:
        raise HTTPException(status_code=404, detail=f"No reports found for '{ticker}'")
    return report


@router.delete(
    "/report/{report_id}",
    response_model=SuccessMessage,
    summary="Delete report",
    description="Delete a previously generated report.",
)
def delete_report(report_id: int, svc=Depends(get_report_service)):
    if not svc.delete_report(report_id):
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
    return SuccessMessage(message=f"Report {report_id} deleted")
