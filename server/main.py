"""SatyaCheck v2 inference server. FastAPI + SSE + SQLite per blueprint §6.2–6.4.

Run: .venv/bin/python -m uvicorn server.main:app --port 8000
Mock pipeline (no GPU/weights needed): SATYA_MOCK=1
"""
from __future__ import annotations

import asyncio

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


def _problem(status: int, title: str, detail: str) -> JSONResponse:
    return JSONResponse({"type": "about:blank", "title": title, "detail": detail, "status": status},
                        status_code=status, media_type="application/problem+json")

from server.db import init_db, janitor
from server.routes.analyses import router as analyses
from server.routes.chain import router as chain
from server.routes.meta import router as meta
from server.routes.redteam import router as redteam

app = FastAPI(title="SatyaCheck v2", version="2.0", docs_url="/v1/docs", openapi_url="/v1/openapi.json")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost(:\d+)?|127\.0\.0\.1(:\d+)?|.*\.vercel\.app)$",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    return _problem(500, "Internal error", "Something broke on our side — sorry! Nothing was lost; please retry.")


@app.exception_handler(HTTPException)
async def http_exc(request: Request, exc: HTTPException):
    titles = {404: "Not found", 413: "Too large", 415: "Unsupported file", 422: "Invalid request"}
    return _problem(exc.status_code, titles.get(exc.status_code, "Request rejected"), str(exc.detail))


@app.exception_handler(RequestValidationError)
async def validation_exc(request: Request, exc: RequestValidationError):
    return _problem(422, "Invalid request", "I couldn't read that request — check the fields and try again.")


@app.on_event("startup")
async def startup():
    init_db()

    def warm_whisper():
        try:
            from ml.redteam.words import _model
            _model()  # 80 s cold load on CPU; do it now, not on the user's first click
            print("[startup] whisper warmed")
        except Exception as e:
            print(f"[startup] whisper warmup skipped: {e.__class__.__name__}")

    import threading
    threading.Thread(target=warm_whisper, daemon=True).start()

    async def sweep():  # blueprint: janitor on start + hourly
        while True:
            await asyncio.sleep(3600)
            await asyncio.get_running_loop().run_in_executor(None, janitor)

    asyncio.create_task(sweep())


app.include_router(analyses)
app.include_router(chain)
app.include_router(redteam)
app.include_router(meta)
