"use client";
/**
 * P9.8 offline tier — when the API is unreachable, still give an honest answer:
 * v1 heuristics (lib/analyze.ts, on-device) + FaceTrace INT8 via onnxruntime-web WASM
 * on sampled face frames. Branches that need the server are labelled as such.
 */
import { useEffect, useRef, useState } from "react";
import { analyzeVideo, type AnalysisResult as V1Result } from "@/lib/analyze";
import type { Tensor } from "onnxruntime-web";
import { useLang } from "@/lib/i18n";

type Offline = {
  v1: V1Result | null;
  face?: { p: number; frames: number };
  error?: string;
};

export default function OfflineFallback() {
  const { t } = useLang();
  const fileRef = useRef<HTMLInputElement>(null);
  const [state, setState] = useState<Offline>({ v1: null });
  const [busy, setBusy] = useState(false);

  useEffect(() => { fileRef.current?.click(); }, []);

  const run = async (file: File) => {
    setBusy(true); setState({ v1: null });
    try {
      const v1 = await analyzeVideo(file, "auto");
      let face: Offline["face"];
      try { face = await faceTraceOnnxBrowser(file); } catch { /* wasm/model load can fail offline — v1 still answers */ }
      setState({ v1, face });
    } catch (e) {
      setState({ v1: null, error: e instanceof Error ? e.message : "could not read this file" });
    } finally {
      setBusy(false);
    }
  };

  const v1 = state.v1;
  const faceP = state.face?.p;
  const label = !v1 ? "…" : faceP != null ? (faceP > 0.6 ? "FAKE" : faceP < 0.35 ? "REAL" : "SUSPECT") : v1.overall > 55 ? "SUSPECT" : v1.overall < 25 ? "REAL" : "SUSPECT";
  const colour: string = label === "REAL" ? "var(--seal-real)" : label === "FAKE" ? "var(--seal-fake)" : "var(--seal-suspect)";

  return (
    <div className="pt-6">
      <div className="panel px-5 py-3" style={{ borderColor: "var(--seal-suspect)" }}>
        <p className="t-small">{t("offline_banner")}</p>
      </div>
      <h1 className="t-h2 mt-6">On-device check</h1>
      <p className="t-small mt-1">This runs entirely in your browser: compression forensics (v1 heuristics) plus the FaceTrace network compiled to WebAssembly. No upload happens.</p>
      <input ref={fileRef} type="file" accept="video/*" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) run(f); }} />
      <button className="btn mt-4" onClick={() => fileRef.current?.click()} disabled={busy}>{busy ? "checking…" : "pick a clip"}</button>

      {v1 ? (
        <div className="mt-6 grid gap-4 md:grid-cols-12">
          <div className="panel px-5 py-4 md:col-span-5" style={{ borderColor: colour }}>
            <span className="t-h1" style={{ color: colour }}>{label}</span>
            <p className="t-small mt-2 t-num">compression score {v1.overall.toFixed(0)}/100{faceP != null ? ` · face p(fake) ${faceP.toFixed(3)}` : ""}</p>
            <p className="t-small mt-1">Offline mode judges fewer cues — the sync and splice scopes need the server.</p>
          </div>
          <div className="panel px-5 py-4 md:col-span-7">
            <div className="chip"><span className="dot" style={{ background: "var(--rule)" }} />What ran on-device</div>
            <ul className="t-small mt-2 space-y-1">
              <li>✓ codec/quality forensics (v1 heuristics)</li>
              <li>{faceP != null ? "✓" : "○"} FaceTrace WASM {faceP != null ? `(${state.face?.frames} faces sampled)` : "(model unavailable offline)"}</li>
              <li>○ sync scope — needs the server</li>
              <li>○ splice scope — needs the server</li>
              <li>○ forwarding age — needs the server</li>
            </ul>
          </div>
        </div>
      ) : null}
      {state.error ? <p className="t-small mt-3" style={{ color: "var(--seal-fake)" }}>{state.error}</p> : null}
    </div>
  );
}

/** FaceTrace INT8 in the browser: sample frames, no detection model (centre-crop fallback),
 * run the exported ONNX via onnxruntime-web WASM. Same ImageNet normalisation as serving. */
async function faceTraceOnnxBrowser(file: File): Promise<{ p: number; frames: number }> {
  const ort = await import("onnxruntime-web");
  const video = document.createElement("video");
  video.src = URL.createObjectURL(file);
  video.muted = true;
  await new Promise((res) => { video.onloadeddata = res; });
  const S = 8;
  const session = await ort.InferenceSession.create("/models/facetrace-int8.onnx", { executionProviders: ["wasm"] });
  const ps: number[] = [];
  for (let i = 1; i <= S; i++) {
    video.currentTime = (video.duration / (S + 1)) * i;
    await new Promise((res) => { video.onseeked = res; });
    const cv = document.createElement("canvas");
    cv.width = 224; cv.height = 224;
    const g = cv.getContext("2d");
    if (!g) continue;
    // centre crop to the likely face region (0.5 scale box); real detection is server-side
    const cw = video.videoWidth * 0.6, chh = video.videoHeight * 0.8;
    g.drawImage(video, (video.videoWidth - cw) / 2, (video.videoHeight - chh) / 2.2, cw, chh, 0, 0, 224, 224);
    const d = g.getImageData(0, 0, 224, 224).data;
    const mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
    const data = new Float32Array(3 * 224 * 224);
    for (let px = 0; px < 224 * 224; px++) {
      for (let c = 0; c < 3; c++) data[c * 224 * 224 + px] = (d[px * 4 + c] / 255 - mean[c]) / std[c];
    }
    const out = await session.run({ x: new ort.Tensor("float32", data, [1, 3, 224, 224]) });
    ps.push(1 / (1 + Math.exp(-((out.p as Tensor).data as Float32Array)[0])));
  }
  URL.revokeObjectURL(video.src);
  if (!ps.length) throw new Error("no frames sampled");
  ps.sort((a, b) => a - b);
  const top = ps.slice(-Math.max(1, Math.floor(ps.length / 4))); // same top-quartile rule as serving
  return { p: top.reduce((a, b) => a + b, 0) / top.length, frames: ps.length };
}
