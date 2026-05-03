import { getCompanyLatestReport, getReports } from "@/components/shared/api/endpoints";
import type { ReportResponse } from "@/components/shared/api/types";
import { MarkdownRenderer } from "@/components/report/MarkdownRenderer";
import { GenerateReportButton } from "@/components/report/GenerateReportButton";
import { EmptyState } from "@/components/shared/EmptyState";
import { formatDateTime } from "@/lib/utils";
import type { Metadata } from "next";

interface Props {
  params: Promise<{ ticker: string }>;
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { ticker } = await params;
  return { title: `Report: ${ticker} — CreditMosaic AI` };
}

export default async function ReportPage({ params }: Props) {
  const { ticker } = await params;

  let report: ReportResponse | null = null;
  let reports: { report_id: number; ticker: string; title: string; report_type: string; generated_at: string }[] = [];
  try {
    report = await getCompanyLatestReport(ticker);
  } catch {
    // No report yet
  }

  try {
    const listData = await getReports({ ticker, limit: "20" });
    reports = listData.reports || [];
  } catch {
    // Ignore
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold text-[var(--foreground)]">
        Risk Report: {ticker}
      </h1>

      <div className="flex gap-6">
        {/* Sidebar */}
        <aside className="hidden lg:block w-56 shrink-0">
          <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-3 sticky top-4">
            <h3 className="text-xs font-semibold text-[var(--muted-foreground)] uppercase mb-3">
              Reports for {ticker}
            </h3>

            <div className="mb-3">
              <GenerateReportButton ticker={ticker} />
            </div>

            <div className="space-y-1 max-h-[60vh] overflow-y-auto">
              {reports.length === 0 && (
                <p className="text-xs text-[var(--muted-foreground)] py-2">No reports yet</p>
              )}
              {reports.map((r) => (
                <a
                  key={r.report_id}
                  href={`/api/report/${r.report_id}`}
                  className={`block p-2 rounded text-xs hover:bg-[var(--muted)] transition-colors ${
                    report?.report_id === r.report_id
                      ? "bg-[var(--chart-1)]/10 border-l-2 border-[var(--chart-1)]"
                      : ""
                  }`}
                >
                  <div className="font-medium text-[var(--foreground)] truncate">{r.title}</div>
                  <div className="text-[var(--muted-foreground)] mt-0.5">{formatDateTime(r.generated_at)}</div>
                </a>
              ))}
            </div>
          </div>
        </aside>

        {/* Main Content */}
        <div className="flex-1 min-w-0">
          {report ? (
            <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-4 lg:p-6 space-y-4">
              {/* Report metadata */}
              <div className="flex flex-wrap items-center gap-3 pb-4 border-b border-[var(--border)]">
                <span className="text-xs text-[var(--muted-foreground)]">
                  Generated: {formatDateTime(report.generated_at)}
                </span>
                {report.model_used && (
                  <span className="text-xs px-2 py-0.5 rounded-full bg-[var(--muted)] text-[var(--muted-foreground)]">
                    {report.model_used}
                  </span>
                )}
                <span className="text-xs px-2 py-0.5 rounded-full bg-[var(--muted)] text-[var(--muted-foreground)]">
                  {report.report_type}
                </span>
                <a
                  href={`/api/report/${report.report_id}/download`}
                  download
                  className="ml-auto text-xs px-3 py-1.5 rounded-lg bg-[var(--chart-1)]/10 text-[var(--chart-1)] hover:bg-[var(--chart-1)]/20 transition-colors font-medium"
                >
                  Download .md
                </a>
              </div>

              {/* Summary */}
              {report.summary && (
                <div className="text-sm text-[var(--muted-foreground)] bg-[var(--muted)] rounded-lg p-3">
                  <strong>Summary:</strong> Risk Score: {report.summary?.risk_score as string || "N/A"} | Level: {report.summary?.risk_level as string || "N/A"}
                </div>
              )}

              {/* Markdown content */}
              <MarkdownRenderer content={report.markdown_content} />
            </div>
          ) : (
            <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-4 lg:p-6">
              <EmptyState
                icon="📄"
                title="No report available"
                message={`No risk report has been generated for ${ticker} yet. Generate one from the sidebar or via the API.`}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
