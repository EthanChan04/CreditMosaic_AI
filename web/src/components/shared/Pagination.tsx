"use client";

interface PaginationProps {
  page: number;
  pageSize: number;
  total: number;
  totalPages: number;
  onPageChange: (page: number) => void;
}

export function Pagination({
  page,
  pageSize,
  total,
  totalPages,
  onPageChange,
}: PaginationProps) {
  const start = (page - 1) * pageSize + 1;
  const end = Math.min(page * pageSize, total);

  return (
    <div className="flex items-center justify-between px-3 py-2 text-sm text-[var(--muted-foreground)]">
      <span>
        Showing {start}–{end} of {total}
      </span>
      <div className="flex items-center gap-1">
        <button
          onClick={() => onPageChange(page - 1)}
          disabled={page <= 1}
          className="px-2 py-1 rounded border border-[var(--border)] disabled:opacity-30 hover:bg-[var(--muted)] transition-colors"
        >
          Prev
        </button>
        {Array.from({ length: Math.min(totalPages, 7) }).map((_, i) => {
          const p = i + 1;
          return (
            <button
              key={p}
              onClick={() => onPageChange(p)}
              className={`px-2 py-1 rounded border transition-colors ${
                p === page
                  ? "bg-[var(--chart-1)] text-white border-[var(--chart-1)]"
                  : "border-[var(--border)] hover:bg-[var(--muted)]"
              }`}
            >
              {p}
            </button>
          );
        })}
        <button
          onClick={() => onPageChange(page + 1)}
          disabled={page >= totalPages}
          className="px-2 py-1 rounded border border-[var(--border)] disabled:opacity-30 hover:bg-[var(--muted)] transition-colors"
        >
          Next
        </button>
      </div>
    </div>
  );
}
