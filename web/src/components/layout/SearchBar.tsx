"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { useDebounce } from "@/hooks/useDebounce";
import { searchCompanies } from "@/components/shared/api/endpoints";
import type { CompanyResponse } from "@/components/shared/api/types";

export function SearchBar() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<CompanyResponse[]>([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const debouncedQuery = useDebounce(query, 300);
  const router = useRouter();
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!debouncedQuery || debouncedQuery.length < 1) {
      return;
    }

    let cancelled = false;
    const id = window.setTimeout(() => {
      setLoading(true);
      searchCompanies({ q: debouncedQuery, limit: "10" })
        .then((data) => {
          if (!cancelled) {
            setResults(data);
            setOpen(true);
          }
        })
        .catch(() => { if (!cancelled) setResults([]); })
        .finally(() => { if (!cancelled) setLoading(false); });
    }, 0);

    return () => {
      cancelled = true;
      window.clearTimeout(id);
    };
  }, [debouncedQuery]);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <div ref={containerRef} className="relative w-full max-w-md">
      <input
        type="text"
        value={query}
        onChange={(e) => {
          const value = e.target.value;
          setQuery(value);
          if (!value.trim()) {
            setResults([]);
            setOpen(false);
          }
        }}
        placeholder="Search companies..."
        className="w-full px-3 py-2 text-sm rounded-lg border border-[var(--border)] bg-[var(--background)] text-[var(--foreground)] placeholder:text-[var(--muted-foreground)] focus:outline-none focus:ring-2 focus:ring-[var(--chart-1)]/50"
      />
      {loading && (
        <div className="absolute right-3 top-2.5">
          <div className="h-4 w-4 animate-spin rounded-full border-2 border-[var(--border)] border-t-[var(--chart-1)]" />
        </div>
      )}
      {open && results.length > 0 && (
        <div className="absolute top-full mt-1 w-full bg-[var(--card)] border border-[var(--border)] rounded-lg shadow-lg z-50 max-h-80 overflow-y-auto">
          {results.map((c) => (
            <button
              key={c.ticker}
              onClick={() => {
                router.push(`/company/${c.ticker}`);
                setOpen(false);
                setQuery("");
              }}
              className="w-full px-3 py-2 text-left hover:bg-[var(--muted)] transition-colors flex items-center justify-between"
            >
              <div>
                <span className="text-sm font-semibold text-[var(--foreground)]">{c.ticker}</span>
                <span className="text-sm text-[var(--muted-foreground)] ml-2">{c.company_name}</span>
              </div>
              {c.sector && (
                <span className="text-xs text-[var(--muted-foreground)]">{c.sector}</span>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
