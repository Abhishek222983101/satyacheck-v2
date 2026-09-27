"use client";
/**
 * Chain Theatre (§7.3) — ForwardChain of one clip: six slots fill as generations finish,
 * each analysed live. The headline's letters keep losing width as the gens advance.
 */
import { useState } from "react";
import { api } from "@/lib/api";
import { streamEvents } from "@/lib/sse";
import { useLang } from "@/lib/i18n";
import type { SSEEvent, VerdictLabel } from "@/lib/types";
import AnalysisPicker from "@/components/AnalysisPicker";

type Slot = { gen: number; url?: string; label?: VerdictLabel; p?: number; params?: Record<string, unknown> };
const SEAL: Record<string, string> = { REAL: "var(--seal-real)", SUSPECT: "var(--seal-suspect)", FAKE: "var(--seal-fake)" };

export default function ChainPage() {
  const { t } = useLang();
  const [mediaId, setMediaId] = useState("");
  const [slots, setSlots] = useState<Slot[]>([]);
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [gen, setGen] = useState(0);
  const [chainId, setChainId] = useState<string | null>(null);

  const startChain = async () => {
    setErr(null); setSlots([]); setRunning(true);
    try {
      const j = await api<{ chain_id: string; events_url: string }>("/v1/chains", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ media_id: mediaId.trim(), gens: 5 }),
      });
      setChainId(j.chain_id);
      streamEvents(j.chain_id, (ev: SSEEvent) => {
        if (ev.event === "gen_ready") {
          const g = ev.data.gen;
          setSlots((s) => [...s, { gen: g as number, url: String(ev.data.media_url), params: ev.data.params }]);
          setGen(g as number);
        } else if (ev.event === "branch" && (ev.data as { branch?: string }).branch === "verdict") {
          const d = ev.data as { gen: number; result: { label: VerdictLabel; p_fake: number } };
          setSlots((s) => s.map((x) => (x.gen === d.gen ? { ...x, label: d.result.label, p: d.result.p_fake } : x)));
        } else if (ev.event === "error") {
          setErr(ev.data.detail); setRunning(false);
        } else if (ev.event === "end") {
          setRunning(false);
        }
      });
      // Reconciler: the server holds full gen state, so polling converges even if SSE drops
      // (or the tab refreshes mid-run). This is the source of truth; SSE is just the live tick.
      const poll = setInterval(async () => {
        try {
          const st = await api<{ status: string; gens: { gen: number; media_url: string; label: VerdictLabel | null; p: number | null; p_fake: number | null }[] }>(`/v1/chains/${j.chain_id}`);
          setSlots((prev) => {
            const next = [...prev];
            for (const g of st.gens) {
              const i = next.findIndex((x) => x.gen === g.gen);
              const slot: Slot = {
                gen: g.gen, url: g.media_url,
                label: g.label ?? undefined, p: g.p_fake ?? undefined,
              };
              if (i >= 0) next[i] = { ...next[i], ...slot };
              else next.push(slot);
            }
            return next.sort((a, b2) => a.gen - b2.gen);
          });
          const maxGen = Math.max(0, ...st.gens.map((g) => g.gen));
          if (maxGen) setGen(maxGen);
          if (st.status === "done") { setRunning(false); clearInterval(poll); }
        } catch { /* SSE still covers the happy path */ }
      }, 5000);
      setTimeout(() => clearInterval(poll), 15 * 60 * 1000);
    } catch (e) {
      setErr(e instanceof Error ? e.message : t("error_generic")); setRunning(false);
    }
  };

  return (
    <div className="pt-6">
      <h1 className="t-hero" style={{
        fontWeight: 700 - gen * 55,
        filter: `blur(${gen * 0.22}px) contrast(${1 - gen * 0.05})`,
        transition: "filter 400ms, font-weight 400ms",
      }}>
        Forward it. Again. And again.
      </h1>
      <p className="t-body mt-2 max-w-[70ch]">
        WhatsApp re-compresses every forward. Watch the same fake degrade across five generations —
        and watch which checks keep catching it. Each frame below is a fresh re-encode, analysed live.
      </p>

      <div className="mt-5 grid max-w-[720px] gap-3">
        <AnalysisPicker value={mediaId} onPick={(m) => setMediaId(m.media_id)} />
        <div className="flex flex-wrap items-center gap-3">
          <input
            className="panel px-3 py-2 t-small t-num w-[380px] max-w-full"
            placeholder="…or paste a media id"
            value={mediaId} onChange={(e) => setMediaId(e.target.value)} disabled={running}
          />
          <button className="btn primary" onClick={startChain} disabled={running || mediaId.trim().length < 8}>
            {running ? "forwarding…" : "Run the chain"}
          </button>
        </div>
      </div>
      <p className="t-small mt-1">Pick a clip analysed on the Desk (or paste its id). Gen 1–5 run on the server.</p>
      {err ? <p className="t-small mt-2" style={{ color: "var(--seal-fake)" }}>{err}</p> : null}

      <div className="mt-6 grid grid-cols-2 gap-4 md:grid-cols-5">
        {[1, 2, 3, 4, 5].map((g) => {
          const s = slots.find((x) => x.gen === g);
          return (
            <div key={g} className="panel overflow-hidden">
              <div className="relative grid place-items-center" style={{ background: "#0c0f10", aspectRatio: "9/14", border: s?.url ? undefined : "3px dashed var(--rule)" }}>
                {s?.url ? <video src={s.url} muted controls preload="metadata" className="h-full w-full object-cover" /> : (
                  <div className="px-3 text-center">
                    <div className="font-display text-[34px]" style={{ color: "#5a6367" }}>G{g}</div>
                    <div className="t-small" style={{ color: "#8aa" }}>
                      {running ? (g === 1 || slots.length === g - 1 ? "re-encoding + analysing…" : "queued behind earlier gens") : "waiting to run"}
                    </div>
                  </div>
                )}
                {s?.label ? (
                  <span className="absolute right-1.5 top-1.5 rounded px-1.5 py-0.5 text-[11px] font-bold"
                    style={{ background: SEAL[s.label], color: "#f7f4ee" }}>
                    {s.label}{s.p != null ? ` ${s.p.toFixed(2)}` : ""}
                  </span>
                ) : null}
              </div>
              <div className="px-2 py-1.5 t-small t-num">Gen {g}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
