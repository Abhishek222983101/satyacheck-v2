"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useLang } from "../lib/i18n";

const TABS = [
  { href: "/", key: "nav_desk" },
  { href: "/chain", key: "nav_chain" },
  { href: "/redteam", key: "nav_redteam" },
  { href: "/lab", key: "nav_lab" },
] as const;

export default function Nav() {
  const path = usePathname();
  const { lang, setLang, t } = useLang();
  return (
    <header className="sticky top-0 z-50" style={{ background: "var(--topbar-bg)", borderBottom: "3px solid var(--ink)" }}>
      <div className="mx-auto flex w-full max-w-[1200px] flex-wrap items-center gap-4 px-5 py-3" style={{ color: "var(--topbar-ink)" }}>
        <Link
          href="/"
          className="font-display inline-block bg-[color:var(--seal-fake)] px-3 py-1 text-[18px] text-white shadow-[4px_4px_0_#000] outline outline-[3px] outline-white/90"
          style={{ transform: "rotate(-2deg)" }}
        >
          SatyaCheck
        </Link>
        <nav className="flex gap-1">
          {TABS.map((tab) => {
            const active = tab.href === "/" ? path === "/" || path.startsWith("/case") : path.startsWith(tab.href);
            return (
              <Link
                key={tab.href}
                href={tab.href}
                className="px-2.5 py-1 text-[14px] font-bold"
                style={{
                  background: active ? "var(--signal-face)" : "transparent",
                  color: active ? "var(--ink)" : "#fff",
                  border: "2px solid #fff",
                  boxShadow: active ? "3px 3px 0 #000" : "none",
                }}
              >
                {t(tab.key)}
              </Link>
            );
          })}
        </nav>
        <div className="ml-auto flex items-center gap-1" role="group" aria-label="Language">
          {(["en", "hi"] as const).map((l) => (
            <button
              key={l}
              className="font-bold"
              style={{
                padding: "3px 10px",
                fontSize: 13,
                border: "2px solid #fff",
                background: lang === l ? "var(--signal-face)" : "transparent",
                color: lang === l ? "var(--ink)" : "#fff",
                cursor: "pointer",
              }}
              onClick={() => setLang(l)}
            >
              {l === "en" ? "EN" : "हिं"}
            </button>
          ))}
        </div>
      </div>
    </header>
  );
}
