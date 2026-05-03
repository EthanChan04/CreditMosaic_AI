"use client";

import { EmptyState } from "@/components/shared/EmptyState";

export default function ErrorPage({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="max-w-[1400px] mx-auto py-12">
      <EmptyState
        icon="⚠️"
        title="Something went wrong"
        message={error.message || "An unexpected error occurred"}
        action={{ label: "Try Again", onClick: reset }}
      />
    </div>
  );
}
