import Link from "next/link";

export default function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] text-center">
      <span className="text-6xl mb-4">404</span>
      <h2 className="text-xl font-semibold text-[var(--foreground)] mb-2">
        Page Not Found
      </h2>
      <p className="text-sm text-[var(--muted-foreground)] mb-6 max-w-sm">
        The page you are looking for does not exist or has been moved.
      </p>
      <Link
        href="/"
        className="px-4 py-2 text-sm font-medium rounded-lg bg-[var(--chart-1)] text-white hover:opacity-90 transition-opacity"
      >
        Back to Dashboard
      </Link>
    </div>
  );
}
