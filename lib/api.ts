// API base resolution: /api-base.json (written at deploy with the tunnel URL) ->
// env override -> same-origin (Next rewrites /v1/* to the API, so no CORS in dev either).
// A `"base": ""` means same-origin and is the default.
let cached = "";

export async function apiBase(): Promise<string> {
  if (cached) return cached;
  if (process.env.NEXT_PUBLIC_API_BASE) {
    cached = process.env.NEXT_PUBLIC_API_BASE;
    return cached;
  }
  try {
    const r = await fetch("/api-base.json", { cache: "no-store" });
    if (r.ok) {
      const j = await r.json();
      cached = typeof j?.base === "string" ? j.base : "";
      return cached;
    }
  } catch { /* file absent — same-origin below */ }
  cached = "";
  return cached;
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const base = await apiBase();
  const r = await fetch(`${base}${path}`, init);
  if (!r.ok) {
    let detail = `${r.status}`;
    try { const j = await r.json(); if (j.detail) detail = j.detail; } catch { /* not problem+json */ }
    throw new Error(detail);
  }
  return r.json() as Promise<T>;
}

export async function upload(file: File, source = "upload"): Promise<{ id: string; case_id: string; events_url: string }> {
  const base = await apiBase();
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${base}/v1/analyses?source=${source}`, { method: "POST", body: fd });
  if (!r.ok) {
    let detail = `${r.status}`;
    try { const j = await r.json(); if (j.detail) detail = j.detail; } catch { /* */ }
    throw new Error(detail);
  }
  return r.json();
}
