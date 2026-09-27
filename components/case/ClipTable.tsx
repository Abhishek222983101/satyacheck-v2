"use client";
/**
 * ClipTable — the clip on the light table with a MediaPipe landmark mesh overlay.
 * Overlay is optional (model loads lazily from /models/face_landmarker.task); if it
 * can't load, the clip plays plain — the verdict never depends on the overlay.
 */
import { useEffect, useRef, useState } from "react";
import type { AnalysisResult } from "@/lib/types";
import { Seal } from "./Scopes";

export default function ClipTable({ src, r, landed }: { src: string; r: AnalysisResult; landed: boolean }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [overlay, setOverlay] = useState<"loading" | "on" | "off">("loading");

  useEffect(() => {
    let stopped = false;
    let raf = 0;
    let lastTs = 0; // MediaPipe VIDEO mode needs strictly-increasing timestamps; currentTime goes backwards on seek/pause
    let landmarker: { detectForVideo: (v: HTMLVideoElement, t: number) => { faceLandmarks: { x: number; y: number }[][] }; close?: () => void } | null = null;
    (async () => {
      try {
        const vision = await import("@mediapipe/tasks-vision");
        const files = await vision.FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10/wasm");
        const lm = await vision.FaceLandmarker.createFromOptions(files, {
          baseOptions: { modelAssetPath: "/models/face_landmarker.task", delegate: "GPU" },
          runningMode: "VIDEO", numFaces: 1,
        });
        if (stopped) { lm.close(); return; }
        landmarker = lm as unknown as typeof landmarker;
        setOverlay("on");
        const draw = () => {
          raf = requestAnimationFrame(draw);
          const v = videoRef.current, cv = canvasRef.current;
          if (!v || !cv || v.paused || v.ended || !landmarker) return;
          if (v.videoWidth === 0) return;
          lastTs = Math.max(lastTs + 1, Math.floor(v.currentTime * 1000));
          const res = landmarker.detectForVideo(v, lastTs);          const g = cv.getContext("2d");
          cv.width = v.videoWidth; cv.height = v.videoHeight;
          if (!g) return;
          g.clearRect(0, 0, cv.width, cv.height);
          const lms = res.faceLandmarks?.[0];
          if (!lms) return;
          // tesselation-lite: draw every landmark as a dot + connect i to i+1 within small index gaps.
          g.fillStyle = "rgba(18,166,196,0.85)";
          for (const p of lms) g.fillRect(p.x * cv.width - 0.8, p.y * cv.height - 0.8, 1.6, 1.6);
        };
        draw();
      } catch {
        if (!stopped) setOverlay("off");
      }
    })();
    return () => { stopped = true; cancelAnimationFrame(raf); landmarker?.close?.(); };
  }, []);

  return (
    <div className="relative panel overflow-hidden" style={{ background: "#0c0f10" }}>
      <video ref={videoRef} src={src} controls className="w-full" playsInline />
      <canvas ref={canvasRef} className="pointer-events-none absolute left-0 top-0 h-full w-full" style={{ opacity: overlay === "on" ? 0.9 : 0 }} />
      <Seal label={r.verdict.label} hi={r.verdict.headline_hi.split("—")[0]} landed={landed} />
      {overlay === "loading" ? <span className="absolute bottom-2 left-2 t-small" style={{ color: "#cfd8d5" }}>loading landmark overlay…</span> : null}
    </div>
  );
}
