"use client";

/**
 * Desk — the empty light table (blueprint §7.3).
 * One bold thing: the headline's letters lose width as a forwarded clip loses generations.
 * Drop a clip -> upload -> SSE stage rail -> navigate to the case board.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useLang } from "../lib/i18n";
import { upload } from "../lib/api";
import { streamEvents } from "../lib/sse";
import type { SSEEvent } from "../lib/types";

const STAGES = ["decode", "landmarks", "age", "face", "sync", "splice", "verdict"] as const;

// Labels carry measured live verdicts (see P8.2 + tape measurement runs) — tapes are evidence, not props.
const TAPES = [
  { src: "/samples/real-man.mp4", thumb: "/samples/thumb-real.jpg", en: "A real clip — judges REAL (p≈0.15)", hi: "असली क्लिप — REAL (p≈0.15)" },
  { src: "/samples/tape_dub_caught.mp4", thumb: "", en: "Lip-sync dub — caught: FAKE (p≈0.94)", hi: "लिप-सिंक डब — पकड़ी गई: FAKE (p≈0.94)" },
  { src: "/samples/tape_dub_forwarded5x.mp4", thumb: "", en: "Same dub, 5 forwards later — still FAKE (p≈0.81)", hi: "वही डब, 5 फॉरवर्ड बाद — अब भी FAKE (p≈0.81)" },
  { src: "/samples/elephant.mp4", thumb: "/samples/thumb-ele.jpg", en: "No face, no speech — the honest 'not sure'", hi: "न चेहरा, न आवाज़ — ईमानदार 'पता नहीं'" },
] as const;

function StageRail({ done, active, ms, error }: { done: string[]; active: string | null; ms: Record<string, number>; error?: string }) {
  return (
    <div className="panel px-4 py-3 flex flex-wrap items-center gap-x-5 gap-y-2" aria-live="polite">
      {STAGES.map((s) => {
        const isDone = done.includes(s) || s === "verdict" && done.length >= 6;
        const isActive = active === s;
        return (
          <span key={s} className={`rail-step ${isDone ? "done" : ""} ${isActive ? "active" : ""}`}>
            <span className="tick" />
            {s === "verdict" ? "verdict" : s}
            {ms[s] ? <span className="t-num t-small">{ms[s]} ms</span> : null}
          </span>
        );
      })}
      {error ? <span style={{ color: "var(--seal-fake)" }}>{error}</span> : null}
    </div>
  );
}

export default function Desk() {
  const { t, lang } = useLang();
  const router = useRouter();
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [rail, setRail] = useState<{ done: string[]; active: string | null; ms: Record<string, number> }>({ done: [], active: null, ms: {} });
  const [gen, setGen] = useState(0); // headline width demo: drag a slider, letters degrade
  const inputRef = useRef<HTMLInputElement>(null);
  const stopRef = useRef<(() => void) | null>(null);

  useEffect(() => () => stopRef.current?.(), []);

  const runFlow = useCallback((file: File, source: string) => {
    setErr(null); setBusy(true); setRail({ done: [], active: "decode", ms: {} });
    (async () => {
      try {
        const j = await upload(file, source);
        const watchdog = setTimeout(() => {
          setErr("The analysis stream went quiet for 3 minutes — the clip or the server may be stuck. Refresh and retry; nothing was lost.");
          setBusy(false);
          stopRef.current?.();
        }, 180000);
        stopRef.current = streamEvents(j.id, (ev: SSEEvent) => {
          if (ev.event === "stage") {
            const name = ev.data.stage as (typeof STAGES)[number];
            setRail((r) => {
              const done = [...new Set([...r.done.filter((d) => d !== name), ...STAGES.slice(0, STAGES.indexOf(name))])] as string[];
              return { done, active: name, ms: { ...r.ms, [name]: ev.data.t_ms } };
            });
          } else if (ev.event === "verdict") {
            setRail((r) => ({ done: STAGES.slice(0, 6) as unknown as string[], active: "verdict", ms: { ...r.ms, verdict: ev.data.timings_ms?.total ?? 0 } }));
          } else if (ev.event === "error") {
            clearTimeout(watchdog);
            setErr(ev.data.detail); setBusy(false);
          } else if (ev.event === "end") {
            clearTimeout(watchdog);
            router.push(`/case/${j.id}`);
          }
        });
      } catch (e) {
        setErr(e instanceof Error ? e.message : t("error_generic"));
        setBusy(false);
      }
    })();
  }, [router, t]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setDrag(false);
    const f = e.dataTransfer.files?.[0];
    if (f) runFlow(f, "upload");
  }, [runFlow]);

  // compression made visible in the letters: weight drops, edges blur (Gen 0 crisp -> Gen 5 mush)
  const heroStyle = {
    fontWeight: 700 - gen * 55,
    filter: `blur(${gen * 0.22}px) contrast(${1 - gen * 0.05})`,
    transition: "filter 240ms, font-weight 240ms",
  };

  return (
    <div className="pt-6">
      <h1 className="t-hero" style={heroStyle}>
        {t("desk_q")}
      </h1>
      <p className="t-body mt-2 max-w-[70ch]">{t("desk_sub")}</p>

      <div className="mt-4 flex items-center gap-3 t-small" aria-hidden>
        <span>Gen 0</span>
        <input type="range" min={0} max={5} value={gen} onChange={(e) => setGen(+e.target.value)} className="w-44 accent-[var(--signal-sync)]" />
        <span>Gen 5 — forwards squeeze the letters, the evidence too</span>
      </div>

      <div
        className={`panel mt-6 grid place-items-center px-6 py-14 text-center ${drag ? "outline-2 outline-dashed" : ""}`}
        style={{ outlineColor: "var(--signal-sync)" }}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={onDrop}
        onClick={() => !busy && inputRef.current?.click()}
        role="button"
        aria-label={t("drop_here")}
      >
        <input
          ref={inputRef} type="file" accept="video/*,audio/*" hidden
          onChange={(e) => { const f = e.target.files?.[0]; if (f) runFlow(f, "upload"); }}
        />
        <div className="t-h3" style={{ fontWeight: 650 }}>{busy ? t("uploading") : t("drop_here")}</div>
        <div className="t-small mt-1">{t("drop_hint")}</div>
      </div>

      {busy || rail.done.length > 0 ? (
        <div className="mt-3">
          <StageRail done={rail.done} active={rail.active} ms={rail.ms} error={err ?? undefined} />
        </div>
      ) : null}
      {err && !busy ? <p className="t-small mt-2" style={{ color: "var(--seal-fake)" }}>{err}</p> : null}

      <div className="mt-8 flex flex-wrap items-center gap-3">
        <RecordButton onFile={(f) => runFlow(f, "webcam")} label={t("record")} />
      </div>

      <h2 className="t-h3 mt-10" style={{ fontWeight: 650 }}>{t("samples")}</h2>
      <div className="mt-3 grid grid-cols-2 gap-4 md:grid-cols-4">
        {TAPES.map((tape) => (
          <button
            key={tape.src}
            className="panel overflow-hidden text-left"
            onClick={() => { fetch(tape.src).then((r) => r.blob()).then((b) => runFlow(new File([b], tape.src.split("/").pop() ?? "sample.mp4", { type: "video/mp4" }), "sample")); }}
          >
            {tape.thumb ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={tape.thumb} alt="" className="h-24 w-full object-cover" />
            ) : (
              <video src={tape.src} muted className="h-24 w-full object-cover" preload="metadata" />
            )}
            <div className="px-3 py-2 t-small" style={{ color: "var(--panel-ink)" }}>{lang === "hi" ? tape.hi : tape.en}</div>
          </button>
        ))}
      </div>
    </div>
  );
}

function RecordButton({ onFile, label }: { onFile: (f: File) => void; label: string }) {
  const [rec, setRec] = useState(false);
  const [left, setLeft] = useState(0);
  const [denied, setDenied] = useState(false);
  const ref = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const ivRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const start = async () => {
    setDenied(false);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: true });
      streamRef.current = stream;
      const chunks: BlobPart[] = [];
      const mr = new MediaRecorder(stream);
      ref.current = mr;
      mr.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
      mr.onstop = () => {
        stream.getTracks().forEach((tr) => tr.stop());
        streamRef.current = null;
        if (chunks.length) onFile(new File(chunks, "my-clip.webm", { type: "video/webm" }));
      };
      mr.start(500); setRec(true); setLeft(8);
      // the <video> element only mounts after setRec renders — attach on the next frame
      requestAnimationFrame(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(() => {});
        }
      });
      ivRef.current = setInterval(() => setLeft((l) => {
        if (l <= 1) {
          if (ivRef.current) clearInterval(ivRef.current);
          try { if (mr.state !== "inactive") mr.stop(); } catch { /* ended */ }
          setRec(false); return 0;
        }
        return l - 1;
      }), 1000);
    } catch {
      setDenied(true);
    }
  };

  useEffect(() => () => {
    if (ivRef.current) clearInterval(ivRef.current);
    streamRef.current?.getTracks().forEach((tr) => tr.stop());
  }, []);

  return (
    <div className="w-full">
      <button className="btn" onClick={start} disabled={rec}>
        {rec ? `● recording — ${left}s (speak normally!)` : `◐ ${label}`}
      </button>
      {denied ? (
        <p className="t-small mt-2" style={{ color: "var(--seal-fake)" }}>
          Camera/microphone blocked — allow access in the address-bar icon, then try again. Or just drop a file above.
        </p>
      ) : null}
      {rec ? (
        <div className="panel mx-auto mt-3 max-w-[560px] overflow-hidden">
          <div className="flex items-center gap-2 border-b-[3px] border-[color:var(--ink)] px-3 py-2">
            <span className="inline-block h-3 w-3 animate-pulse rounded-full" style={{ background: "var(--seal-fake)" }} />
            <b className="font-display text-[22px] tabular-nums">0:0{left}</b>
            <span className="t-small">look at the camera and speak continuously</span>
            <button
              className="btn ml-auto"
              style={{ padding: "4px 12px" }}
              onClick={() => {
                if (ivRef.current) clearInterval(ivRef.current);
                try { if (ref.current && ref.current.state !== "inactive") ref.current.stop(); } catch { /* ended */ }
                setRec(false);
              }}
            >
              ■ Stop & analyse
            </button>
          </div>
          <video ref={videoRef} muted playsInline className="aspect-video w-full bg-black" style={{ transform: "scaleX(-1)" }} />
        </div>
      ) : null}
    </div>
  );
}
