"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface MarkdownRendererProps {
  content: string;
}

export function MarkdownRenderer({ content }: MarkdownRendererProps) {
  return (
    <div className="prose prose-sm max-w-none text-[var(--foreground)] prose-headings:text-[var(--foreground)] prose-a:text-[var(--chart-1)] prose-strong:text-[var(--foreground)] prose-code:text-[var(--chart-4)] prose-code:bg-[var(--muted)] prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-pre:bg-[var(--muted)] prose-pre:border prose-pre:border-[var(--border)] prose-table:border-[var(--border)] prose-th:bg-[var(--muted)] prose-th:px-3 prose-th:py-2 prose-td:px-3 prose-td:py-2 prose-tr:border-[var(--border)]">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>
        {content}
      </ReactMarkdown>
    </div>
  );
}
