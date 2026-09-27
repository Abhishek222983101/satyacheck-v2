"use client";
/**
 * P9.4 — FaceMeshHeat: the 478 landmarks as a rotating 3D point-cloud face, vertex
 * colour = per-frame blend probability at that moment of the clip. Auto-rotates slowly;
 * drag to orbit. Lazy-loaded from the case board so the Lab stays light.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { FilesetResolver, FaceLandmarker } from "@mediapipe/tasks-vision";

type Props = { src: string; frames?: number[] };

const HEAT = (t: number) => new THREE.Color().setHSL(0.12 - 0.12 * t, 0.75, 0.45 + 0.15 * t); // yellow -> red

export default function FaceMeshHeat({ src, frames = [] }: Props) {
  const [ready, setReady] = useState(false);
  return (
    <div className="panel relative overflow-hidden" style={{ height: 320 }}>
      <Canvas camera={{ position: [0, 0, 2.2], fov: 40 }} dpr={[1, 2]}>
        <ambientLight intensity={1.1} />
        <Mesh src={src} frames={frames} setReady={setReady} />
        <OrbitControls enablePan={false} autoRotate autoRotateSpeed={0.8} minDistance={1.4} maxDistance={3.5} />
      </Canvas>
      {!ready ? (
        <span className="absolute bottom-2 left-3 t-small">
          loading the 3D face… (needs WebGL + the landmark model — the strip above still carries the verdict)
        </span>
      ) : null}
    </div>
  );
}

function Mesh({ src, frames = [], setReady }: Props & { setReady: (b: boolean) => void }) {
  const pointsRef = useRef<THREE.Points>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const lmRef = useRef<FaceLandmarker | null>(null);
  const ready = true;

  const geo = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(478 * 3), 3));
    g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(478 * 3), 3));
    return g;
  }, []);

  useEffect(() => {
    let alive = true;
    const v = document.createElement("video");
    v.src = src; v.muted = true; v.loop = true; v.playsInline = true;
    videoRef.current = v;
    (async () => {
      try {
        const vision = await import("@mediapipe/tasks-vision");
        const files = await vision.FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10/wasm");
        const lm = await vision.FaceLandmarker.createFromOptions(files, {
          baseOptions: { modelAssetPath: "/models/face_landmarker.task", delegate: "GPU" },
          runningMode: "VIDEO", numFaces: 1,
        });
        if (!alive) { lm.close(); return; }
        lmRef.current = lm;
        await v.play();
        setReady(true);
      } catch { /* overlay model unavailable — panel stays empty but harmless */ }
    })();
    return () => { alive = false; lmRef.current?.close(); v.pause(); };
  }, [src]);

  // sample landmarks on a timer, lift z for depth, colour by heat
  useEffect(() => {
    if (!ready) return;
    const iv = setInterval(() => {
      const v = videoRef.current, lm = lmRef.current;
      if (!v || !lm || v.readyState < 2 || v.paused) return;
      const res = lm.detectForVideo(v, v.currentTime * 1000);
      const lms = res.faceLandmarks?.[0];
      if (!lms) return;
      const pos = geo.getAttribute("position") as THREE.BufferAttribute;
      const col = geo.getAttribute("color") as THREE.BufferAttribute;
      const t = frames.length ? (frames[Math.floor((v.currentTime / (v.duration || 1)) * frames.length)] ?? 0) : 0.35;
      const heat = HEAT(t);
      for (let i = 0; i < Math.min(lms.length, 478); i++) {
        const p = lms[i];
        pos.setXYZ(i, (p.x - 0.5) * 1.6, -(p.y - 0.5) * 1.6, (0.5 - p.z) * 1.6);
        col.setXYZ(i, heat.r, heat.g, heat.b);
      }
      pos.needsUpdate = true; col.needsUpdate = true;
    }, 100);
    return () => clearInterval(iv);
  }, [ready, geo, frames]);

  return (
    <points ref={pointsRef} geometry={geo}>
      <pointsMaterial size={0.028} vertexColors sizeAttenuation />
    </points>
  );
}
