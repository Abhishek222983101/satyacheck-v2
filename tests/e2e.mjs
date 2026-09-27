// P10.1 — E2E of the wows + P9.9 screenshots (390/1440, light/dark).
// Drives the real stack: uvicorn (mock) on :8000, next dev on :3000.
// Run:  .venv/bin/python -m uvicorn server.main:app --port 8000 &   (SATYA_MOCK=1)
//       npx next dev -p 3000 &
//       node tests/e2e.mjs
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const OUT = join(ROOT, "docs", "screens");
mkdirSync(OUT, { recursive: true });

const BASE = "http://localhost:3000";
let failures = 0;
const ok = (cond, name) => {
  console.log(`${cond ? "PASS" : "FAIL"}  ${name}`);
  if (!cond) failures++;
};

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

// --- Desk renders + screenshots (light/dark, 1440/390)
await page.goto(BASE + "/");
await page.waitForSelector("text=Is this clip real?");
await page.screenshot({ path: join(OUT, "desk_1440.png"), fullPage: true });
await page.emulateMedia({ colorScheme: "dark" });
await page.screenshot({ path: join(OUT, "desk_1440_dark.png"), fullPage: true });
await page.emulateMedia({ colorScheme: "light" });
await page.setViewportSize({ width: 390, height: 844 });
await page.screenshot({ path: join(OUT, "desk_390.png"), fullPage: true });
await page.setViewportSize({ width: 1440, height: 900 });
ok(true, "desk renders + screenshots");

// --- upload flow: sample tape -> case board
await page.setInputFiles("input[type=file]", join(ROOT, "public", "samples", "tape_dub_caught.mp4"));
await page.waitForURL("**/case/**", { timeout: 60000 });
await page.waitForSelector("text=p(fake)", { timeout: 30000 });
await page.screenshot({ path: join(OUT, "case_1440.png"), fullPage: true });
ok(true, "upload -> case board renders verdict");

const caseUrl = page.url();

// --- Lab renders measured reports
await page.goto(BASE + "/lab");
await page.waitForSelector("text=Nyaya", { timeout: 15000 });
await page.screenshot({ path: join(OUT, "lab_1440.png"), fullPage: true });
ok(true, "lab renders");

// --- Chain + Red team render
await page.goto(BASE + "/chain");
await page.waitForSelector("button:has-text('Run the chain')");
await page.screenshot({ path: join(OUT, "chain_1440.png"), fullPage: true });
await page.goto(BASE + "/redteam");
await page.waitForSelector("button:has-text('Load words')");
await page.screenshot({ path: join(OUT, "redteam_1440.png"), fullPage: true });
ok(true, "chain + redteam render");

// --- offline tier: block the API, open a bogus case -> on-device check appears
await page.route("**/v1/**", (r) => r.abort());
await page.goto(BASE + "/case/00000000-0000-0000-0000-000000000000");
await page.waitForSelector("text=On-device check", { timeout: 20000 });
await page.screenshot({ path: join(OUT, "offline_1440.png"), fullPage: true });
ok(true, "offline tier falls back to on-device check");

// --- server came back: normal case still loads
await page.unroute("**/v1/**");
await page.goto(caseUrl);
await page.waitForSelector("text=p(fake)", { timeout: 30000 });
ok(true, "case still renders after API returns");

await browser.close();
console.log(failures ? `\n${failures} FAILURES` : "\nALL E2E PASS");
process.exit(failures ? 1 : 0);
