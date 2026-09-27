import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";
import { LangProvider } from "../lib/i18n";
import Nav from "../components/Nav";

// Anek (Ek Type) variable fonts, bundled locally so the wdth axis works offline too.
const anekLatin = localFont({
  src: "./fonts/AnekLatin.ttf",
  variable: "--font-anek-latin",
  weight: "100 800",
});
const anekDeva = localFont({
  src: "./fonts/AnekDevanagari.ttf",
  variable: "--font-anek-deva",
  weight: "100 800",
});

export const metadata: Metadata = {
  title: "SatyaCheck — is this forwarded clip real?",
  description:
    "Forensic checks for forwarded videos: lip-sync, splice, face artefacts and forwarding age — with an honest 'not sure' instead of a bluff.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${anekLatin.variable} ${anekDeva.variable}`}>
      <body>
        <LangProvider>
          <Nav />
          <main className="mx-auto w-full max-w-[1200px] px-5 pb-16">{children}</main>
          <footer className="mx-auto max-w-[1200px] px-5 pb-8 t-small">
            SatyaCheck v2 — verdicts are probabilistic evidence, not proof. Clips auto-delete after 24 h.
          </footer>
        </LangProvider>
      </body>
    </html>
  );
}
