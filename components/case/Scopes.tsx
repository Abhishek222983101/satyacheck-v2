"use client";import { useEffect, useRef } from "react";
import type { AnalysisResult } from "@/lib/types";

/** Tamper-evident seal across the clip. Colours: real/suspect/fake tokens from §7.2. */
export const SEAL_COLOUR: Record<string, string> = {
  REAL: "var(--seal-real)", SUSPECT: "var(--seal-suspect)", FAKE: "var(--seal-fake)",
};

export function Seal({ label, hi, landed }: { label: string; hi: string; landed: boolean }) {
  if (!landed) return null;
  return (
    <div className="pointer-events-none absolute inset-0 grid place-items-center overflow-hidden">
      <div className="seal-tape" style={{ ["--seal-col" as string]: SEAL_COLOUR[label] }} role="img" aria-label={`verdict ${label}`}>
        {label} · {hi}
      </div>
    </div>
  );
}

/** Sync scope: out-of-sync spans over the assessed window + the offset verdict chip. */
export function SyncScope({ r }: { r: AnalysisResult }) {
  const b = r.branches.sync;
  const dur = r.file.assessed_s || r.file.duration_s || 1;
  const W = 100; // percent-based spans keep it responsive
  return (
    <div className="panel px-4 py-3">
      <div className="chip" style={{ color: "var(--signal-sync)" }}><span className="dot" style={{ background: "var(--signal-sync)" }} />Sync — does the mouth match the words?</div>
      {b.status !== "ok" ? (
        <p className="t-small mt-2">Not checked: {b.reason}. The clip can still be judged on the other cues.</p>
      ) : (
        <>
          <div className="relative mt-3 h-9 rounded" style={{ background: "color-mix(in srgb, var(--signal-sync) 12%, transparent)" }}>
            <span className="absolute inset-y-0 left-2 flex items-center t-small t-num">0 s</span>
            <span className="absolute inset-y-0 right-2 flex items-center t-num t-small">{dur.toFixed(1)} s</span>
            {(b.spans ?? []).map((sp, i) => (
              <div key={i} className="absolute inset-y-0 rounded-sm"
                style={{
                  left: `${(sp.t0 / dur) * W}%`, width: `${Math.max(1.2, ((sp.t1 - sp.t0) / dur) * W)}%`,
                  background: "var(--signal-sync)", opacity: 0.25 + 0.75 * sp.score,
                }}
                title={`out of sync ${sp.t0.toFixed(1)}–${sp.t1.toFixed(1)} s`} />
            ))}
          </div>
          <p className="t-small mt-2">
            Mouth/audio offset <b className="t-num">{b.offset_ms} ms</b> · out-of-sync spans{" "}
            <b className="t-num">{(b.spans ?? []).length}</b> · p(fake) <b className="t-num">{b.p_fake?.toFixed(3)}</b>
            {b.offset_ms ? <> — the audio sounds {Math.abs(b.offset_ms) < 80 ? "in step" : b.offset_ms > 0 ? "late" : "early"} against the lips</> : null}
          </p>
        </>
      )}
    </div>
  );
}

/** Splice scope: audio waveform (decoded live) + splice boundary pins + synthetic-voice heat. */
export function SpliceScope({ src, r }: { src: string; r: AnalysisResult }) {
  const b = r.branches.splice;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (b.status !== "ok" || !src) return;
    let ctx: AudioContext | null = null;
    let cancelled = false;
    (async () => {
      try {
        ctx = new AudioContext();
        const buf = await ctx.decodeAudioData(await (await fetch(src)).arrayBuffer());
        if (cancelled) return;
        const ch = buf.getChannelData(0);
        const cv = canvasRef.current; if (!cv) return;
        const g = cv.getContext("2d"); if (!g) return;
        const { width: W, height: H } = cv;
        g.clearRect(0, 0, W, H);
        g.strokeStyle = "var(--rule)";
        g.beginPath(); g.moveTo(0, H / 2); g.lineTo(W, H / 2); g.stroke();
        // waveform: min/max per column
        const col = Math.floor(ch.length / W);
        g.strokeStyle = getComputedStyle(cv).color || "#888";
        g.beginPath();
        for (let x = 0; x < W; x++) {
          let mn = 1, mx = -1;
          for (let i = x * col; i < (x + 1) * col; i += 16) { mn = Math.min(mn, ch[i]); mx = Math.max(mx, ch[i]); }
          g.moveTo(x, H / 2 + mn * H / 2); g.lineTo(x, H / 2 + mx * H / 2);
        }
        g.stroke();
      } catch { /* audio decode failed — the pins below still render */ }
    })();
    return () => { cancelled = true; ctx?.close(); };
  }, [src, b.status]);
  const dur = r.file.assessed_s || r.file.duration_s || 1;
  return (
    <div className="panel px-4 py-3">
      <div className="chip" style={{ color: "var(--signal-splice)" }}><span className="dot" style={{ background: "var(--signal-splice)" }} />Splice — was the audio cut or synthesised?</div>
      {b.status !== "ok" ? (
        <p className="t-small mt-2">Not checked: {b.reason}.</p>
      ) : (
        <>
          <canvas ref={canvasRef} width={640} height={72} className="mt-3 w-full" style={{ color: "var(--panel-ink)" }} />
          <div className="relative mt-1 h-5">
            {(b.boundaries ?? []).map((bd, i) => (
              <div key={i} className="absolute top-0 h-5 w-[2px]" style={{ left: `${(bd.t / dur) * 100}%`, background: "var(--signal-splice)", opacity: 0.4 + 0.6 * bd.strength }}
                title={`cut at ${bd.t.toFixed(2)} s`} />
            ))}
          </div>
          <p className="t-small mt-1">
            p(spliced) <b className="t-num">{b.p_spliced?.toFixed(3)}</b> · p(synthetic voice) <b className="t-num">{b.p_synthetic?.toFixed(3)}</b> · cuts detected <b className="t-num">{(b.boundaries ?? []).length}</b>
          </p>
        </>
      )}
    </div>
  );
}

/** ForwardAge dial: 0-5 generations, ramp graphite->rule (age is context, not alarm). */
export function AgeDial({ r }: { r: AnalysisResult }) {
  const a = r.branches.age;
  if (a.status !== "ok") return null;
  const pct = (a.gen_hat / 5) * 100;
  return (
    <div className="panel px-4 py-3">
      <div className="chip"><span className="dot" style={{ background: "var(--rule)" }} />Forward age</div>
      <div className="mt-3 flex items-baseline gap-2">
        <span className="t-h2 t-num">{a.gen_hat.toFixed(1)}</span>
        <span className="t-small">/ 5 generations — {a.gen_hat < 1.5 ? "fresh export" : a.gen_hat < 3.5 ? "shared a few times" : "forwarded a lot"}</span>
      </div>
      <div className="mt-2 h-2 rounded" style={{ background: "linear-gradient(90deg, var(--graphite), var(--rule))" }}>
        <div className="h-2 rounded" style={{ width: `${pct}%`, background: "transparent", borderTop: "2px solid var(--seal-fake)", position: "relative", top: "-2px" }} />
      </div>
      <div className="mt-1 flex justify-between t-small t-num">{[0, 1, 2, 3, 4, 5].map((g) => <span key={g}>{g}</span>)}</div>
      <p className="t-small mt-1">confident the clip was re-saved many times? {a.gen_hat >= 3.5 ? "yes — quality is visibly degraded" : "quality is mostly fine"}</p>
    </div>
  );
}

/** FaceTrace strip: per-frame blend probability over the clip. */
export function FaceTraceStrip({ r }: { r: AnalysisResult }) {
  const b = r.branches.face;
  if (b.status !== "ok") return null;
  const frames = b.frames ?? [];
  return (
    <div className="panel px-4 py-3">
      <div className="chip" style={{ color: "var(--signal-face)" }}><span className="dot" style={{ background: "var(--signal-face)" }} />Face trace — blending artefacts</div>
      <div className="mt-3 flex h-8 items-end gap-px">
        {frames.map((v, i) => (
          <div key={i} className="flex-1" style={{ height: `${Math.max(4, v * 100)}%`, background: "var(--signal-face)", opacity: 0.35 + 0.65 * v }} title={`${(v * 100).toFixed(0)}%`} />
        ))}
      </div>
      <p className="t-small mt-1">clip p(fake) <b className="t-num">{b.p_fake?.toFixed(3)}</b> — spikes are frames where the face shows synthesis artefacts</p>
    </div>
  );
}

/** Decision trace: what the fusion actually concluded, and how honestly. */
export function DecisionTrace({ r }: { r: AnalysisResult }) {
  const v = r.verdict;
  return (
    <div className="panel px-4 py-3">
      <div className="chip"><span className="dot" style={{ background: SEAL_COLOUR[v.label] }} />How the decision was made</div>
      <ul className="t-small mt-2 space-y-1">
        <li>Fused p(fake) <b className="t-num">{v.p_fake.toFixed(3)}</b> → <b>{v.label}</b></li>
        <li>Honest set at α={v.alpha.toFixed(2)}: <b>{v.conformal_set.join(" or ")}</b>
          {v.conformal_set.length === 2 ? " — the evidence genuinely supports both, so we refuse to guess" : " — evidence points one way"}</li>
        <li>Clip generation bucket <b>{v.bucket.replace("gen_", "Gen ")}</b>{v.reason ? ` · note: ${v.reason.replace("_", " ")}` : ""}</li>
        <li>Models: {Object.entries(r.models).map(([k, h]) => `${k}#${String(h).slice(0, 6)}`).join(" · ")}</li>
      </ul>
    </div>
  );
}
