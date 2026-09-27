# P10.5 — rehearsal & failure drills (finale day)

Print this. Three timed run-throughs before the judges walk in. Target: **≤ 5:00** each.

## Pre-flight (do 30 min before, not during)

- [ ] Laptop on charger; **sleep disabled** (`systemctl suspend` blocked / settings); notifications off
- [ ] `SATYA_MOCK` **unset** (live models), `SATYA_RUNTIME=ort` if GPU misbehaves — decide once, stick to it
- [ ] API up: `curl localhost:8000/v1/health` → `{"ok":true,...}`
- [ ] Web up: `localhost:3000` renders the Desk in Anek (letters must not be serif)
- [ ] Tunnel URL live on a **phone on 4G** (not the demo Wi-Fi)
- [ ] Browser zoom 100%, bookmarks bar hidden, clean profile
- [ ] Six pre-analysed case URLs bookmarked (1 real, 2 fake, 1 Gen-5, 1 splice attack, 1 SUSPECT)
- [ ] Terminal open on `logs/` for the "we measure everything" moment

## The 5-minute script

| t | Action | What to say |
|---|---|---|
| 0:00 | Desk, drag the Gen slider 0→5 | "Forwards squeeze the letters — and the evidence. Same idea as the whole system: compression is the twist." |
| 0:30 | Drop the AI-anchor tape | rail ticks live — name the stages as they light |
| 1:30 | Verdict lands: seal + honest set | "It gives p(fake), and a set — if both labels survive, it says NOT SURE instead of bluffing." |
| 2:15 | SyncScope + FaceTrace strip | "Every colour is a branch — cyan sync, magenta splice, yellow face." |
| 2:45 | Red team: load words, cut one, make the fake, scan it | "We try to fool ourselves before a judge does." |
| 3:30 | Chain Theatre (pre-loaded case id) | "Five generations, analysed live — the face branch fades, sync holds. That's the finding." |
| 4:15 | Lab wall | "Nothing here is typed in — regenerate with `make eval`. p95 5.5 s on CPU only." |
| 4:45 | Dossier download (Hindi card) | close on the artefact a fact-checker would actually forward |

## Failure drills (do each once)

1. **API down mid-demo** → kill uvicorn. Case page must show the on-device check banner; Desk upload shows the friendly problem card. Recovery: restart uvicorn, refresh — the same case URL still works (SQLite persisted).
2. **Wi-Fi down** → the tunnel dies; switch to `localhost:3000` on the laptop and keep talking.
3. **A live clip misbehaves** (no face / no speech) → the branches show "not checked: no face" and the verdict explains itself. **Do not** apologise for the abstention — that IS the product.
4. **GPU dies** → restart API with `SATYA_RUNTIME=ort`; verdicts continue on CPU (parity measured, p95 5.5 s).
5. **Judge asks for a number you don't remember** → Lab page. Never quote from memory; point at the wall ("it's the same JSON our verdicts come from").

## Hard rules

- Never run `vercel --prod` live. Deploy happens **before** rehearsal, once.
- Never promise "real WhatsApp tested" — T-WA is pending; say "simulated forwards, real-phone validation is next".
- If total time is slipping, cut Chain Theatre live run and show the bookmarked completed chain instead.
