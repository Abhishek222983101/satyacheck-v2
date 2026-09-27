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
    <header className="mx-auto flex w-full max-w-[1200px] items-baseline gap-7 px-5 pt-6 pb-4">
      <Link href="/" className="t-h3" style={{ fontWeight: 750 }}>
        Satya<span style={{ color: "var(--signal-sync)" }}>Check</span>
      </Link>
      <nav className="flex gap-5 text-[15px]">
        {TABS.map((tab) => {
          const active = tab.href === "/" ? path === "/" || path.startsWith("/case") : path.startsWith(tab.href);
          return (
            <Link
              key={tab.href}
              href={tab.href}
              className="border-b-2 pb-0.5"
              style={{ borderColor: active ? "var(--signal-sync)" : "transparent", fontWeight: active ? 700 : 500 }}
            >
              {t(tab.key)}
            </Link>
          );
        })}
      </nav>
      <div className="ml-auto flex items-center gap-1 t-small" role="group" aria-label="Language">
        {(["en", "hi"] as const).map((l) => (
          <button
            key={l}
            className="btn"
            style={{
              padding: "2px 9px",
              fontWeight: lang === l ? 750 : 400,
              borderColor: lang === l ? "var(--graphite)" : "var(--rule)",
            }}
            onClick={() => setLang(l)}
          >
            {l === "en" ? "EN" : "हिं"}
          </button>
        ))}
      </div>
    </header>
  );
}
