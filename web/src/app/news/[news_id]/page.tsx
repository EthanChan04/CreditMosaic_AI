import { getNewsDetail } from "@/components/shared/api/endpoints";
import { SignalBadge } from "@/components/news/SignalBadge";
import { FinBERTComparison } from "@/components/news/FinBERTComparison";
import { EmptyState } from "@/components/shared/EmptyState";
import { formatDateTime } from "@/lib/utils";
import Link from "next/link";
import type { Metadata } from "next";

interface Props {
  params: Promise<{ news_id: string }>;
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { news_id } = await params;
  return { title: `News #${news_id} — CreditMosaic AI` };
}

export default async function NewsDetailPage({ params }: Props) {
  const { news_id } = await params;
  const id = parseInt(news_id, 10);

  let news;
  let error: string | null = null;
  try {
    news = await getNewsDetail(id);
  } catch (e) {
    error = (e as Error).message;
  }

  if (error || !news) {
    return (
      <div className="max-w-4xl mx-auto py-8">
        <EmptyState icon="📰" title="News article not found" message={error || `No news with ID ${id}`} />
        <div className="text-center mt-4">
          <Link href="/" className="text-sm text-[var(--chart-1)] hover:underline">Back to Dashboard</Link>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Breadcrumb */}
      <div className="text-sm text-[var(--muted-foreground)]">
        <Link href="/" className="hover:text-[var(--chart-1)]">Dashboard</Link>
        {" / "}
        <Link href={`/company/${news.ticker}`} className="hover:text-[var(--chart-1)]">{news.ticker}</Link>
        {" / "}
        <span className="text-[var(--foreground)]">News #{id}</span>
      </div>

      {/* Article */}
      <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-4 lg:p-6">
        <h1 className="text-xl font-bold text-[var(--foreground)] mb-2">{news.title}</h1>
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-sm text-[var(--muted-foreground)] mb-4">
          <Link href={`/company/${news.ticker}`} className="font-semibold text-[var(--chart-1)] hover:underline">{news.ticker}</Link>
          {news.source && <span>{news.source}</span>}
          <span>{formatDateTime(news.published_at)}</span>
        </div>
        <div className="prose prose-sm max-w-none text-[var(--foreground)] whitespace-pre-wrap leading-relaxed">
          {news.body || "No body content"}
        </div>
      </div>

      {/* LLM Signal */}
      {news.signal && <SignalBadge signal={news.signal} />}

      {/* Evidence Spans */}
      {news.signal?.evidence_spans && news.signal.evidence_spans.length > 0 && (
        <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg p-4">
          <h3 className="text-sm font-semibold text-[var(--muted-foreground)] uppercase mb-3">Evidence Spans</h3>
          <ul className="space-y-2">
            {news.signal.evidence_spans.map((span: string, i: number) => (
              <li key={i} className="text-sm text-[var(--foreground)] border-l-2 border-[var(--chart-1)] pl-3 py-1 bg-[var(--muted)] rounded-r">
                {span}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* FinBERT Comparison */}
      <FinBERTComparison ticker={news.ticker} newsId={id} />
    </div>
  );
}
