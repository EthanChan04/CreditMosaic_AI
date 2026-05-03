import { getCompany } from "@/components/shared/api/endpoints";
import { CompanyHeader } from "@/components/company/CompanyHeader";
import { RiskHistoryChart } from "@/components/company/RiskHistoryChart";
import { NewsTimeline } from "@/components/company/NewsTimeline";
import { EmptyState } from "@/components/shared/EmptyState";
import Link from "next/link";
import type { Metadata } from "next";

interface Props {
  params: Promise<{ ticker: string }>;
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { ticker } = await params;
  return {
    title: `${ticker} — CreditMosaic AI`,
    description: `Credit risk analysis for ${ticker}`,
  };
}

export default async function CompanyPage({ params }: Props) {
  const { ticker } = await params;

  let company;
  let error: string | null = null;
  try {
    company = await getCompany(ticker);
  } catch (e) {
    error = (e as Error).message;
  }

  if (error || !company) {
    return (
      <div className="max-w-5xl mx-auto py-8">
        <EmptyState
          icon="🏢"
          title="Company not found"
          message={error || `No data for ticker "${ticker}"`}
          action={{ label: "Back to Dashboard", onClick: () => {} }}
        />
        <div className="text-center mt-4">
          <Link href="/" className="text-sm text-[var(--chart-1)] hover:underline">
            Back to Dashboard
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <CompanyHeader company={company} />

      <RiskHistoryChart ticker={ticker} />

      <NewsTimeline ticker={ticker} />
    </div>
  );
}
