import { ApiError } from "./client";

export const swrFetcher = async <T>(url: string): Promise<T> => {
  const BASE_URL = typeof window !== "undefined" ? "" : "http://localhost:8000";
  const res = await fetch(`${BASE_URL}${url}`, {
    headers: { "Content-Type": "application/json" },
  });

  if (!res.ok) {
    let code = "UNKNOWN_ERROR";
    let message = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      code = body?.error?.code || code;
      message = body?.error?.message || message;
    } catch { /* not JSON */ }
    throw new ApiError(res.status, code, message);
  }

  return res.json() as Promise<T>;
};
