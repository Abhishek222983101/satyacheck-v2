// AnalysisResult v2 — mirrors contracts/analysis.schema.json exactly. Single source: the contract.
export type VerdictLabel = "REAL" | "SUSPECT" | "FAKE";
export type Abstain = "no_speech" | "no_face" | "too_short" | "uncertain" | "branch_conflict" | "heavy_degradation";
export type Mode = "live" | "mock" | "offline_fallback" | "chain_gen_k";

export interface Span { t0: number; t1: number; score: number }
export interface Boundary { t: number; strength: number }

export interface SyncBranch {
  status: "ok" | "abstained" | "error" | "skipped";
  reason?: Abstain;
  p_fake?: number; offset_ms?: number; curve?: number[]; spans?: Span[];
}
export interface SpliceBranch {
  status: "ok" | "abstained" | "error" | "skipped";
  reason?: Abstain;
  p_spliced?: number; p_synthetic?: number; frame_spoof?: number[]; boundaries?: Boundary[];
}
export interface FaceBranch {
  status: "ok" | "abstained" | "error" | "skipped";
  reason?: Abstain;
  p_fake?: number; frames?: number[]; cam_refs?: string[];
}
export interface AgeBranch {
  status: "ok" | "abstained" | "error" | "skipped";
  gen_hat: number; gen_dist: number[]; platform: Record<string, number>; why: string[];
}

export interface AnalysisResult {
  schema_version: "2.0";
  id: string;
  file: { name: string; sha256: string; duration_s: number; assessed_s: number; width?: number; height?: number; has_audio: boolean };
  verdict: {
    label: VerdictLabel; p_fake: number; integrity: number;
    conformal_set: ("REAL" | "FAKE")[]; alpha: number;
    bucket: "gen_0_1" | "gen_2_3" | "gen_4_5";
    reason: Abstain | null; headline_en: string; headline_hi: string;
  };
  branches: { sync: SyncBranch; splice: SpliceBranch; face: FaceBranch; age: AgeBranch };
  timeline?: { fps?: number; speech?: number[]; mouth_open?: number[] };
  timings_ms: Record<string, number>;
  models: Record<string, string>;
  mode: Mode;
}

// ---- SSE events (blueprint §6.4) ----
export type SSEEvent =
  | { event: "stage"; data: { stage: string; t_ms: number; progress: number | null } }
  | { event: "branch"; data: { branch: string; result: unknown; gen?: number } }
  | { event: "verdict"; data: AnalysisResult["verdict"] & { timings_ms: Record<string, number> } }
  | { event: "annotation"; data: { words: { w: string; t0: number; t1: number }[] } }
  | { event: "gen_ready"; data: { gen: number; media_url: string; params: Record<string, unknown> } }
  | { event: "error"; data: { type: string; title: string; detail: string } }
  | { event: "end"; data: Record<string, never> };

export const SIGNAL_COLOUR: Record<string, string> = {
  sync: "#12A6C4", splice: "#C2338F", face: "#D9B21B", age: "#B9C3C0",
};
