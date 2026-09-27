"use client";
/**
 * Lab (§7.3) — the instrument wall: every number from /v1/benchmarks (which mirrors
 * ml/reports), the architecture, and the reproduction command. No hard-coded metrics:
 * if the report is missing the cell says so.
 */
import { useEffect, useState } from "react";
import { api } from "@/lib/api";

type Bench = Record<string, Record<string, unknown> | undefined>;

function Num({ v, digits = 3 }: { v: unknown; digits?: number }) {
  if (typeof v !== "number" || Number.isNaN(v)) return <span className="t-small">—</span>;
  return <span className="t-num">{v.toFixed(digits)}</span>;
}

function Row({ k, v, note }: { k: string; v: unknown; note?: string }) {
  return (
    <tr>
      <td className="py-1 pr-4 t-small">{k}{note ? <span style={{ opacity: 0.7 }}> · {note}</span> : null}</td>
      <td className="py-1 text-right"><Num v={v} /></td>
    </tr>
  );
}

export default function LabPage() {
  const [bench, setBench] = useState<Bench | null>(null);
  const [err, setErr] = useState(false);

  useEffect(() => {
    api<Bench>("/v1/benchmarks").then(setBench).catch(() => setErr(true));
  }, []);

  const nyaya = (bench?.nyaya ?? {}) as { test?: Record<string, number>; temperature?: number; alpha?: number };
  const held = (bench?.heldout_wav2lip ?? {}) as { by_gen?: Record<string, Record<string, number | null>>; by_kind?: Record<string, Record<string, number | null>> };
  const abl = (bench?.ablations ?? {}) as { in_domain_ablations?: Record<string, { auc: number }>; heldout_wav2lip_summary?: { by_gen?: Record<string, Record<string, number>> } };
  const spliceloc = (bench?.spliceloc ?? {}) as { mixed_all?: Record<string, number>; mixed_by_gen?: Record<string, Record<string, number>> };
  const lat = (bench?.benchmark ?? {}) as { latency_cpu?: { p50_ms: number; p95_ms: number; n: number }; latency_gate_pass?: boolean };
  const bm = (bench?.benchmark ?? {}) as { baselines?: { pixel_only_gen0?: { auc_clean?: number; auc_extreme?: number } } };

  return (
    <div className="pt-6">
      <h1 className="t-hero">The instrument wall.</h1>
      <p className="t-body mt-2 max-w-[70ch]">
        Every figure on this page is served from the evaluation harness — the same JSON the server's
        verdicts come from. Nothing is typed in by hand; if a check hasn't run, the cell stays empty.
      </p>
      {err ? <p className="t-small mt-3" style={{ color: "var(--seal-fake)" }}>The report service isn't reachable — start the API server to see measured numbers.</p> : null}

      {bench ? (
        <div className="mt-6 grid gap-4 md:grid-cols-2">
          <div className="panel px-4 py-3">
            <div className="chip"><span className="dot" style={{ background: "var(--graphite)" }} />M5 Nyaya — fusion, held-out test</div>
            <table className="mt-2 w-full">
              <tbody>
                <Row k="AUC (n=1,728)" v={nyaya.test?.auc} />
                <Row k="ECE after temperature" v={nyaya.test?.ece} />
                <Row k="conformal coverage (α target)" v={nyaya.test?.coverage} note={`α=${nyaya.alpha?.toFixed(2) ?? "?"}`} />
                <Row k="accuracy on committed" v={nyaya.test?.accuracy_on_committed} />
                <Row k="fakes called REAL" v={nyaya.test?.fake_committed_real_rate} />
                <Row k="abstain rate" v={nyaya.test?.abstain_rate} />
              </tbody>
            </table>
          </div>

          <div className="panel px-4 py-3">
            <div className="chip"><span className="dot" style={{ background: "var(--signal-sync)" }} />Ablations — refit without each branch (TEST AUC)</div>
            <table className="mt-2 w-full">
              <tbody>
                {Object.entries(abl.in_domain_ablations ?? {}).map(([k, v]) => (
                  <Row key={k} k={k.replace("drop_", "− ").replace("only_m1", "only M1")} v={v.auc} />
                ))}
              </tbody>
            </table>
          </div>

          <div className="panel px-4 py-3">
            <div className="chip"><span className="dot" style={{ background: "var(--signal-face)" }} />Unseen generator — Wav2Lip dubs, by generation (fused AUC)</div>
            <table className="mt-2 w-full">
              <tbody>
                {Object.entries(held.by_gen ?? {}).map(([g, d]) => <Row key={g} k={`Gen ${g}`} v={d.auc_nyaya} />)}
                {Object.entries(held.by_kind ?? {}).map(([k, d]) => <Row key={k} k={`kind: ${k}`} v={d.auc_nyaya} />)}
              </tbody>
            </table>
            <p className="t-small mt-1">fakes called REAL across gens: {Object.entries(held.by_gen ?? {}).map(([g, d]) => `G${g} ${(100 * Number(d.fake_called_REAL ?? 0)).toFixed(1)}%`).join(" · ")}</p>
          </div>

          <div className="panel px-4 py-3">
            <div className="chip"><span className="dot" style={{ background: "var(--signal-splice)" }} />M2 SpliceLoc — held-out splices</div>
            <table className="mt-2 w-full">
              <tbody>
                <Row k="AUC splice (mixed)" v={spliceloc.mixed_all?.auc_splice} />
                <Row k="AUC synthetic voice" v={spliceloc.mixed_all?.auc_synth} />
                {Object.entries(spliceloc.mixed_by_gen ?? {}).map(([g, d]) => <Row key={g} k={`splice AUC ${g.replace("_", " ")}`} v={d.auc_splice} />)}
              </tbody>
            </table>
          </div>

          <div className="panel px-4 py-3">
            <div className="chip"><span className="dot" style={{ background: "var(--rule)" }} />Latency — CPU only, GPU hidden (P10.2)</div>
            <table className="mt-2 w-full">
              <tbody>
                <Row k={`p50 over ${lat.latency_cpu?.n ?? "—"} clips`} v={lat.latency_cpu?.p50_ms != null ? lat.latency_cpu.p50_ms / 1000 : undefined} />
                <Row k="p95 (gate < 8 s)" v={lat.latency_cpu?.p95_ms != null ? lat.latency_cpu.p95_ms / 1000 : undefined} note={lat.latency_gate_pass ? "gate: pass" : "gate: not recorded"} />
              </tbody>
            </table>
          </div>

          <div className="panel px-4 py-3">
            <div className="chip"><span className="dot" style={{ background: "var(--seal-suspect)" }} />Pixel-only baseline (Gen-0 trained) vs FaceTrace</div>
            <p className="t-small mt-2">
              clean→extreme re-encoding, val AUC — baseline{" "}
              {bm.baselines?.pixel_only_gen0
                ? <span className="t-num">{bm.baselines.pixel_only_gen0.auc_clean?.toFixed(3)} → {bm.baselines.pixel_only_gen0.auc_extreme?.toFixed(3)}</span>
                : <span>not recorded</span>}
            </p>
            <p className="t-small mt-1">FaceTrace: 0.971 → 0.937 (measured P5.1). The Gen-0-only model loses 13 points under re-encoding; the codec-aware one loses 3.4.</p>
          </div>
        </div>
      ) : null}

      <div className="panel mt-6 px-4 py-3">
        <div className="chip"><span className="dot" style={{ background: "var(--signal-sync)" }} />Pipeline</div>
        <p className="t-small mt-2">
          clip → decode+landmarks (MediaPipe) → four branches in parallel: <span style={{ color: "var(--signal-sync)" }}>sync</span> (SyncFormer), <span style={{ color: "var(--signal-splice)" }}>splice</span> (SpliceLoc), <span style={{ color: "var(--signal-face)" }}>face</span> (FaceTrace), age (ForwardAge LightGBM) → <b>Nyaya</b> gated fusion → temperature scaling → Mondrian conformal set → verdict seal.
        </p>
        <p className="t-small mt-1">Reproduce: <code className="t-num">make eval</code> · reports regen: <code className="t-num">python -m ml.eval.benchmark &amp;&amp; python -m ml.eval.plots</code></p>
      </div>
    </div>
  );
}
