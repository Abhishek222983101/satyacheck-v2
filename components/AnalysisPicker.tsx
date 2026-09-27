"use client";
// Shared picker: "which clip?" — lists recent analyses from the server so the user
// never has to copy-paste a media id. Failures degrade to the manual input box.
import { useEffect, useState } from "react";
import { api } from "@/lib/api";

export type Picked = { media_id: string; analysis_id: string; name: string; label: string | null };
type Row = { analysis_id: string; media_id: string; status: string; label: string | null; name: string };

export default function AnalysisPicker({ value, onPick }: { value: string; onPick: (m: Picked) => void }) {
  const [rows, setRows] = useState<Row[]>([]);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let alive = true;
    api<{ analyses: Row[] }>("/v1/analyses?limit=30")
      .then((j) => { if (alive) setRows(j.analyses); })
      .catch(() => { if (alive) setFailed(true); });
    return () => { alive = false; };
  }, []);

  if (failed && !rows.length) return null; // caller keeps its manual input as fallback description

  return (
    <div className="panel px-3 py-2 t-small">
      <div className="mb-1" style={{ opacity: 0.75 }}>recent analyses{failed ? " (list unavailable — paste id below)" : ""}</div>
      <div className="flex max-h-40 flex-col gap-1 overflow-y-auto">
        {rows.length === 0 ? <span style={{ opacity: 0.6 }}>none yet — analyse a clip on the Desk first</span> : null}
        {rows.map((r) => (
          <button
            key={r.analysis_id}
            className="btn text-left"
            style={{
              padding: "3px 8px", fontWeight: value === r.media_id ? 750 : 400,
              borderColor: value === r.media_id ? "var(--signal-sync)" : "var(--rule)",
            }}
            onClick={() => onPick({ media_id: r.media_id, analysis_id: r.analysis_id, name: r.name, label: r.label })}
          >
            <span className="t-num">{r.name.slice(0, 28)}</span>
            <span style={{ opacity: 0.65 }}> · {r.status}{r.label ? ` · ${r.label}` : ""}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
