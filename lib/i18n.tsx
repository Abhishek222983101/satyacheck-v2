// Minimal i18n: en/hi dictionaries + a language context. No framework — the surface is two languages.
"use client";
import { createContext, useContext, useState, type ReactNode } from "react";

export type Lang = "en" | "hi";
const DICT = {
  en: {
    nav_desk: "Desk", nav_chain: "Chain", nav_redteam: "Red team", nav_lab: "Lab",
    desk_q: "Is this clip real?",
    desk_sub: "Drop a forwarded video. I'll check the mouth, the audio, the face and how many times it's been re-saved — and I'll tell you honestly, even when I'm not sure.",
    drop_here: "Drop the clip here, or click to pick one",
    drop_hint: "MP4 · WebM · MOV · up to 100 MB — nothing is stored for more than 24 hours",
    record: "Record yourself (8 s)",
    samples: "Evidence tapes",
    rail: "Working…",
    verdict_real: "REAL", verdict_suspect: "NOT SURE", verdict_fake: "FAKE",
    integrity: "integrity", why: "why",
    play_resync: "Play re-synced",
    dossier: "Download dossier", card_hi: "Share card (हिंदी)",
    offline_banner: "The server isn't reachable — showing the on-device check instead. Branches marked 'needs the server' weren't run.",
    error_generic: "Something went wrong — sorry! Please try again.",
    uploading: "Sending your clip…",
  },
  hi: {
    nav_desk: "डेस्क", nav_chain: "चेन", nav_redteam: "रेड टीम", nav_lab: "लैब",
    desk_q: "क्या यह क्लिप असली है?",
    desk_sub: "फॉरवर्ड किया हुआ वीडियो यहाँ डालें। मैं मुँह, आवाज़, चेहरा और यह भी देखूँगी कि क्लिप कितनी बार दोबारा सेव हुई — और ईमानदारी से बताऊँगी, यहाँ तक कि अनिश्चित होने पर भी।",
    drop_here: "क्लिप यहाँ डालें, या चुनने के लिए क्लिक करें",
    drop_hint: "MP4 · WebM · MOV · 100 MB तक — कुछ भी 24 घंटे से ज़्यादा नहीं रहता",
    record: "खुद रिकॉर्ड करें (8 सेकंड)",
    samples: "सबूत टेप",
    rail: "काम चल रहा है…",
    verdict_real: "असली", verdict_suspect: "पक्का नहीं", verdict_fake: "नकली",
    integrity: "अखंडता", why: "कारण",
    play_resync: "री-सिंक करके चलाएँ",
    dossier: "डॉसिएर डाउनलोड", card_hi: "शेयर कार्ड (हिंदी)",
    offline_banner: "सर्वर नहीं मिल रहा — डिवाइस पर जाँच दिखा रहे हैं। 'सर्वर चाहिए' वाली जाँचें नहीं हुईं।",
    error_generic: "कुछ गड़बड़ हो गई — क्षमा करें! फिर कोशिश करें।",
    uploading: "क्लिप भेज रहे हैं…",
  },
} as const;
export type DictKey = keyof (typeof DICT)["en"];

const Ctx = createContext<{ lang: Lang; setLang: (l: Lang) => void; t: (k: DictKey) => string }>({
  lang: "en", setLang: () => {}, t: (k) => DICT.en[k],
});

export function LangProvider({ children }: { children: ReactNode }) {
  const [lang, setLang] = useState<Lang>("en");
  return <Ctx.Provider value={{ lang, setLang, t: (k) => DICT[lang][k] ?? DICT.en[k] }}>{children}</Ctx.Provider>;
}
export const useLang = () => useContext(Ctx);
