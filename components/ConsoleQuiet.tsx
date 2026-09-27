"use client";
// MediaPipe's WASM/TFLite stack logs INFO-level chatter (XNNPACK delegate, feedback
// manager) through console.error — thousands of lines that bury real errors and trip
// error-counting extensions. This filter drops exactly those known-harmless patterns
// and passes everything else through untouched.
const NOISE = [
  "XNNPACK delegate",
  "TensorFlow Lite",
  "inference_feedback_manager",
  "face_landmarker_graph",
  "Created TensorFlow",
  "GPU delegate",
];

function quieted(orig: (...a: unknown[]) => void) {
  return (...args: unknown[]) => {
    const first = String(args[0] ?? "");
    if (NOISE.some((p) => first.includes(p))) return;
    orig(...args);
  };
}

if (typeof window !== "undefined" && !(window as unknown as { __satyaQuiet?: boolean }).__satyaQuiet) {
  (window as unknown as { __satyaQuiet?: boolean }).__satyaQuiet = true;
  // eslint-disable-next-line no-console
  console.error = quieted(console.error.bind(console));
  // eslint-disable-next-line no-console
  console.warn = quieted(console.warn.bind(console));
}

export default function ConsoleQuiet() {
  return null;
}
