// Typed SSE client for /v1/analyses/{id}/events (blueprint §6.4).
// Replays stored events (the server sends the log first), then streams live ones.
import type { SSEEvent } from "./types";
import { apiBase } from "./api";

export function streamEvents(
  id: string,
  onEvent: (ev: SSEEvent) => void,
  onDone?: () => void,
): () => void {
  let stopped = false;
  let es: EventSource | null = null;
  (async () => {
    const base = await apiBase();
    if (stopped) return;
    es = new EventSource(`${base}/v1/analyses/${id}/events`);
    for (const kind of ["stage", "branch", "verdict", "annotation", "gen_ready", "error", "end"]) {
      es.addEventListener(kind, (e) => {
        if (stopped) return;
        try {
          onEvent({ event: kind, data: JSON.parse((e as MessageEvent).data) } as SSEEvent);
        } catch { /* malformed frame — skip */ }
        if (kind === "end" || kind === "error") { es?.close(); onDone?.(); }
      });
    }
    es.onerror = () => { /* EventSource auto-reconnects; server keeps the log so replay is safe */ };
  })();
  return () => { stopped = true; es?.close(); onDone?.(); };
}
