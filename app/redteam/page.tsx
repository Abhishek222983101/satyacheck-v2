"use client";
/**
 * Red team (§7.3) — make a fake, then scan it. Pick a clip, choose words to remove/swap,
 * the server splices the audio (P1.5 factory), and the attack can be sent back through
 * the pipeline. Judges do this flow in under a minute.
 */
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api, upload } from "@/lib/api";
import AnalysisPicker from "@/components/AnalysisPicker";
import { useLang } from "@/lib/i18n";

type Word = { w: string; t0: number; t1: number; conf?: number };

export default function RedTeamPage() {
  const { t } = useLang();
  const router = useRouter();
  const [mediaId, setMediaId] = useState("");
  const [words, setWords] = useState<Word[] | null>(null);
  const [wordSource, setWordSource] = useState<"whisper" | "pauses" | null>(null);
  const [picked, setPicked] = useState<number[]>([]);
  const [op, setOp] = useState<"delete" | "swap" | "duplicate">("delete");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [attackUrl, setAttackUrl] = useState<string | null>(null);
  const [truth, setTruth] = useState<number[]>([]);
  const [scanning, setScanning] = useState(false);

  useEffect(() => {
    // prefill from the most recent Desk analysis, if any (deferred: no sync setState in effect)
    const id = setTimeout(() => setMediaId(localStorage.getItem("satya:lastMediaId") ?? ""), 0);
    return () => clearTimeout(id);
  }, []);

  const loadWords = async () => {
    setErr(null); setWords(null); setPicked([]);
    try {
      const j = await api<{ words?: Word[]; source?: "whisper" | "pauses" }>(`/v1/redteam/words/${mediaId.trim()}`);
      const ws = (j.words ?? []).filter((w) => w.w.trim());
      if (!ws.length) throw new Error("no speech found in this clip — try one where someone is clearly talking.");
      setWords(ws);
      setWordSource(j.source ?? null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : t("error_generic"));
    }
  };

  const makeFake = async () => {
    setBusy(true); setErr(null);
    try {
      const ids = picked.length ? picked : [Math.floor(words!.length / 2)];
      const span = words!.filter((_, i) => ids.includes(i));
      const t0 = Math.min(...span.map((w) => w.t0));
      const t1 = Math.max(...span.map((w) => w.t1));
      const j = await api<{ media_url: string; truth: { boundaries: number[] } }>("/v1/redteam/splice", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ media_id: mediaId.trim(), op, t0, t1 }),
      });
      setAttackUrl(j.media_url); setTruth(j.truth.boundaries);
    } catch (e) {
      setErr(e instanceof Error ? e.message : t("error_generic"));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="pt-6">
      <h1 className="t-hero">Break it yourself.</h1>
      <p className="t-body mt-2 max-w-[70ch]">
        The fastest way to trust a detector is to try to fool it. Pick words to cut or shuffle —
        the server makes the tampered clip, then you can scan it on the Desk and see whether it gets caught.
      </p>

      <div className="mt-5 grid max-w-[720px] gap-3">
        <AnalysisPicker value={mediaId} onPick={(m) => { setMediaId(m.media_id); setWords(null); setAttackUrl(null); }} />
        <div className="flex flex-wrap items-center gap-3">
        <input
          className="panel px-3 py-2 t-small t-num w-[380px] max-w-full"
          placeholder="…or paste a media id"
          value={mediaId}
          onChange={(e) => {
            setMediaId(e.target.value); setWords(null); setAttackUrl(null);
            try { localStorage.setItem("satya:lastMediaId", e.target.value); } catch { /* private mode */ }
          }}
        />
        <button className="btn" onClick={loadWords} disabled={mediaId.trim().length < 8}>Load words</button>
        </div>
      </div>
      {err ? <p className="t-small mt-2" style={{ color: "var(--seal-fake)" }}>{err}</p> : null}

      {words ? (
        <>
          {wordSource === "pauses" ? (
            <p className="t-small mt-3" style={{ color: "var(--seal-suspect)" }}>
              no clear words came through — these are speech chunks between pauses. Forging still works; pick any.
            </p>
          ) : null}
          <div className="panel mt-4 flex flex-wrap gap-2 p-4">
            {words.map((w, i) => (
              <button
                key={i}
                className="btn t-num"
                style={{
                  padding: "4px 10px",
                  borderColor: picked.includes(i) ? "var(--signal-splice)" : "var(--rule)",
                  background: picked.includes(i) ? "color-mix(in srgb, var(--signal-splice) 15%, transparent)" : undefined,
                  textDecoration: picked.includes(i) && op === "delete" ? "line-through" : undefined,
                }}
                onClick={() => setPicked((p) => (p.includes(i) ? p.filter((x) => x !== i) : [...p, i]))}
              >
                {w.w}
              </button>
            ))}
          </div>
          <div className="mt-4 flex flex-wrap items-center gap-3">
            <select className="panel px-3 py-2 t-small" value={op} onChange={(e) => setOp(e.target.value as typeof op)}>
              <option value="delete">cut the selected words</option>
              <option value="swap">swap two neighbours</option>
              <option value="duplicate">repeat a word (deepfake stitch)</option>
            </select>
            <button className="btn primary" onClick={makeFake} disabled={busy || !words.length}>
              {busy ? "forging…" : `Make the fake (${picked.length || "auto"} words)`}
            </button>
          </div>
        </>
      ) : null}

      {attackUrl ? (
        <div className="mt-6 grid gap-4 md:grid-cols-12">
          <div className="panel p-3 md:col-span-7">
            <video src={attackUrl} controls className="w-full" />
            <p className="t-small mt-2 t-num">truth: {truth.length} splice seam(s) at {truth.map((x) => x.toFixed(2)).join(" s, ")} s</p>
          </div>
          <div className="panel p-4 md:col-span-5">
            <p className="t-h3">Now scan it.</p>
            <p className="t-small mt-1">Send the tampered clip back through the full pipeline — does the splice branch catch the seam?</p>
            <button
              className="btn primary mt-3"
              disabled={scanning}
              onClick={async () => {
                setScanning(true); setErr(null);
                try {
                  const b = await (await fetch(attackUrl)).blob();
                  const j = await upload(new File([b], "attack.mp4", { type: "video/mp4" }), "redteam");
                  router.push(`/case/${j.id}`);
                } catch (e) {
                  setErr(e instanceof Error ? e.message : t("error_generic")); setScanning(false);
                }
              }}
            >
              {scanning ? "sending…" : "Scan it ▸"}
            </button>
            <p className="t-small mt-2">The tampered clip goes through the full pipeline as a new case.</p>
          </div>
        </div>
      ) : null}
    </div>
  );
}
