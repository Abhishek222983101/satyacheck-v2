"""P8.5 — Dossier: A4 PDF report + 1080x1350 share card, generated from a finished analysis.
PDF via WeasyPrint (system HTML/CSS, no network), card via Pillow + qrcode."""
from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi.responses import FileResponse
from sqlmodel import Session

from server.db import Analysis, Artifact, Media, engine, j

OUT = Path("var/out")
SEAL = {"REAL": "#0e9f6e", "SUSPECT": "#d97706", "FAKE": "#dc2626"}
_VERDICT_EN = {"REAL": "Appears genuine", "SUSPECT": "Not sure — treat with care", "FAKE": "Likely manipulated"}
_VERDICT_HI = {"REAL": "असली प्रतीत होती है", "SUSPECT": "अनिश्चित — सावधानी से देखें", "FAKE": "संभवतः बदली गई"}


def _load(aid: str) -> dict:
    with Session(engine) as s:
        a = s.get(Analysis, aid)
        if not a or a.status != "done":
            from fastapi import HTTPException
            raise HTTPException(404, "No finished analysis for that id.")
        return json.loads(a.result_json)


def _strip(html: str) -> str:
    return (html.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&#39;"))


def _store(aid: str, name: str, path: Path, mime: str) -> None:
    with Session(engine) as s:
        s.add(Artifact(id=str(uuid.uuid4()), analysis_id=aid, name=name, path=str(path), mime=mime))
        s.commit()


def render_dossier(aid: str) -> FileResponse:
    r = _load(aid)
    OUT.mkdir(parents=True, exist_ok=True)
    pdf = OUT / f"dossier_{aid[:8]}.pdf"
    v = r["verdict"]
    colour = SEAL[v["label"]]
    rows = "".join(
        f"<tr><td>{name}</td><td class='{b['status']}'>{_strip(str(b.get('p_fake', b.get('p_spliced') or b.get('gen_hat', '—')))[:6])}</td>"
        f"<td>{b['status']}</td></tr>"
        for name, b in r["branches"].items())
    html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
      body {{ font-family: sans-serif; color: #111; margin: 40px; }}
      .seal {{ display: inline-block; padding: 6px 18px; border: 3px solid {colour}; color: {colour};
               border-radius: 8px; font-size: 26px; font-weight: 700; letter-spacing: 2px; }}
      h1 {{ font-size: 20px; }} table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
      td, th {{ border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 13px; }}
      .ok {{ color: #0e9f6e; }} .abstained {{ color: #999; }} .muted {{ color: #666; font-size: 12px; }}
    </style></head><body>
      <span class="seal">{v['label']}</span>
      <h1>{_strip(v['headline_en'])}</h1>
      <p class="muted">p(fake) = {v['p_fake']:.3f} · integrity {v['integrity']}/100 · bucket {v['bucket']}
         · conformal set {', '.join(v['conformal_set'])} at α={v['alpha']:.2f}</p>
      <table><tr><th>Branch</th><th>Score</th><th>Status</th></tr>{rows}</table>
      <p class="muted">File {_strip(r['file']['name'])} · assessed {r['file']['assessed_s']} s ·
         mode {r['mode']} · {_strip(datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'))}</p>
      <p class="muted">SatyaCheck v2 — scores are probabilistic, not proof. An 'abstained' branch means the clip
         could not be judged on that cue.</p>
    </body></html>"""
    from weasyprint import HTML
    HTML(string=html).write_pdf(str(pdf))
    _store(aid, "dossier.pdf", pdf, "application/pdf")
    return FileResponse(pdf, media_type="application/pdf", filename=f"satyacheck_{aid[:8]}.pdf")


def render_card(aid: str, lang: str = "en") -> FileResponse:
    r = _load(aid)
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"card_{aid[:8]}_{lang}.png"
    v = r["verdict"]
    import qrcode
    from PIL import Image, ImageDraw
    W, H = 1080, 1350
    colour = {"REAL": (14, 159, 110), "SUSPECT": (217, 119, 6), "FAKE": (220, 38, 38)}[v["label"]]
    img = Image.new("RGB", (W, H), (250, 247, 242))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 14], fill=colour)
    d.text((60, 80), "SatyaCheck", fill=(60, 55, 50))
    d.text((60, 200), v["label"], fill=colour, font_size=120)
    headline = v["headline_hi"] if lang == "hi" else v["headline_en"]
    d.text((60, 400), headline, fill=(40, 38, 35), font_size=40)
    d.text((60, 520), f"p(fake) {v['p_fake']:.3f}   integrity {v['integrity']}/100", fill=(90, 85, 80), font_size=30)
    # thumbnail frame from the clip's first second
    thumb = OUT / f"thumb_{aid[:8]}.png"
    with Session(engine) as s:
        m = s.get(Media, aid)
    if m and not thumb.exists():
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "1", "-i", m.path, "-frames:v", "1",
                        "-vf", "scale=720:-2", str(thumb)], check=False)
    if thumb.exists():
        img.paste(Image.open(thumb).convert("RGB").resize((720, 405)), (60, 620))
    d.text((60, 1080), f"case {aid[:8]} · {_strip(datetime.now(timezone.utc).strftime('%d %b %Y'))}",
           fill=(120, 115, 110), font_size=26)
    qr = qrcode.make(f"satyacheck://case/{aid}").resize((160, 160))
    img.paste(qr.convert("RGB"), (W - 220, H - 220))
    img.save(png)
    _store(aid, f"card_{lang}.png", png, "image/png")
    return FileResponse(png, media_type="image/png")
