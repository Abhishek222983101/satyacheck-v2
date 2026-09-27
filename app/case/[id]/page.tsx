"use client";
import { use, useEffect, useState } from "react";
import Link from "next/link";
import { api, apiBase } from "../../../lib/api";
import { useLang } from "../../../lib/i18n";
import type { AnalysisResult } from "../../../lib/types";
import dynamic from "next/dynamic";
import ClipTable from "../../../components/case/ClipTable";
import { AgeDial, DecisionTrace, FaceTraceStrip, SEAL_COLOUR, SpliceScope, SyncScope } from "../../../components/case/Scopes";
import OfflineFallback from "../../../components/case/OfflineFallback";

const FaceMeshHeat = dynamic(() => import("../../../components/case/FaceMeshHeat"), {
  ssr: false, loading: () => <div className="panel" style={{ height: 320 }} />,
});

export default function CasePage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const { t, lang } = useLang();
  const [r, setR] = useState<AnalysisResult | null>(null);
  const [status, setStatus] = useState<string>("queued");
  const [dead, setDead] = useState(false);   // server unreachable -> offline tier
  const [base, setBase] = useState("");
  const [source, setSource] = useState<string>("");

  useEffect(() => {
    let alive = true;
    apiBase().then((b) => alive && setBase(b));
    api<{ source: string }>(`/v1/analyses/${id}/meta`).then((m) => alive && setSource(m.source)).catch(() => {});
    const poll = async () => {
      try {
        const j = await api<AnalysisResult & { status?: string }>(`/v1/analyses/${id}`);
        if (!alive) return;
        if (j.schema_version === "2.0") { setR(j); setStatus("done"); }
        else { setStatus(j.status ?? "queued"); setTimeout(poll, 700); }
      } catch {
        if (alive) { setDead(true); setStatus("done"); }
      }
    };
    poll();
    return () => { alive = false; };
  }, [id]);

  if (dead) return <OfflineFallback />;

  if (!r) {
    return (
      <div className="pt-16 text-center">
        <p className="t-h3">{status === "error" ? t("error_generic") : "Working on the light table…"}</p>
        <p className="t-small mt-2">analysis {id.slice(0, 8)}</p>
        {status === "error" ? <Link href="/" className="btn mt-4 inline-block">◂ {t("nav_desk")}</Link> : null}
      </div>
    );
  }

  const v = r.verdict;
  const mediaUrl = `${base}/v1/analyses/${r.id}/media`;

  return (
    <div className="pt-4">
      <div className="flex flex-wrap items-baseline gap-3">
        <Link href="/" className="t-small">◂ {t("nav_desk")}</Link>
        <h1 className="t-h2 truncate">{r.file.name}</h1>
        <span className="t-small t-num">{r.file.assessed_s.toFixed(1)} s · {r.file.width}×{r.file.height} · {r.mode === "mock" ? "demo run" : r.mode}</span>
      </div>

      {source === "webcam" ? (
        <div className="panel mt-3 px-4 py-2 t-small" style={{ borderColor: "var(--seal-real)", borderWidth: 3 }}>
          <b>◉ Recorded in-app</b> — trusted capture: this clip came straight from your camera through this page,
          so its <i>origin</i> is known-genuine. The forensic scopes below still report exactly what they see;
          mic codecs (Opus) can read as artefacts, so treat low confidence here as "unknown", not "fake".
        </div>
      ) : null}

      <div className="mt-4 grid gap-4 md:grid-cols-12">
        {/* clip 7 cols, verdict 5 cols (§7.2 layout) */}
        <div className="md:col-span-7"><ClipTable src={mediaUrl} r={r} landed /></div>
        <div className="flex flex-col gap-4 md:col-span-5">
          <div className="panel px-5 py-4" style={{ borderColor: SEAL_COLOUR[v.label] }}>
            <div className="flex items-baseline justify-between">
              <span className="t-h1" style={{ color: SEAL_COLOUR[v.label] }}>
                {v.label === "REAL" ? t("verdict_real") : v.label === "FAKE" ? t("verdict_fake") : t("verdict_suspect")}
              </span>
              <span className="t-num t-h3">{v.integrity}/100 {t("integrity")}</span>
            </div>
            <p className="t-body mt-2">{lang === "hi" ? v.headline_hi : v.headline_en}</p>
            <p className="t-small mt-2 t-num">p(fake) {v.p_fake.toFixed(3)} · honest set: {v.conformal_set.join(" or ")} (α={v.alpha.toFixed(2)})</p>
          </div>
          <DecisionTrace r={r} />
          <DossierActions id={r.id} />
        </div>

        <div className="md:col-span-7"><SyncScope r={r} /></div>
        <div className="md:col-span-5"><AgeDial r={r} /></div>
        <div className="md:col-span-7"><SpliceScope src={mediaUrl} r={r} /></div>
        <div className="md:col-span-5"><FaceTraceStrip r={r} /></div>
        {r.branches.face.status === "ok" ? (
          <div className="md:col-span-12">
            <div className="chip mb-2"><span className="dot" style={{ background: "var(--signal-face)" }} />Face mesh — drag to orbit, colour = blend probability at this moment</div>
            <FaceMeshHeat src={mediaUrl} frames={r.branches.face.frames ?? []} />
          </div>
        ) : null}
      </div>
    </div>
  );
}

function DossierActions({ id }: { id: string }) {
  const { t } = useLang();
  return (
    <div className="flex flex-wrap gap-3">
      <a className="btn primary" href={`/v1/analyses/${id}/dossier.pdf`}>{t("dossier")}</a>
      <a className="btn" href={`/v1/analyses/${id}/card.png?lang=hi`}>{t("card_hi")}</a>
    </div>
  );
}
