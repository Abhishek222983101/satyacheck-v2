"""API integration tests (P8.3/P8.4) — run the server fully in mock mode (SATYA_MOCK=1):
no GPU, no weights. Covers upload validation, SSE stream, result contract, resync,
chain and red-team splice."""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import tempfile
import time

import pytest

_TMP = pathlib.Path(tempfile.mkdtemp(prefix="satya_test_", dir="/tmp/opencode"))
os.environ["SATYA_MOCK"] = "1"
os.environ["SATYA_VAR"] = str(_TMP)
os.environ.setdefault("SATYA_DEVICE", "cpu")

from fastapi.testclient import TestClient  # noqa: E402

from server.main import app  # noqa: E402

CLIP = pathlib.Path(_TMP / "test_clip.mp4")


def _make_clip() -> None:
    """Synthetic 4 s clip (color + sine) so tests never depend on outside files."""
    import av
    import numpy as np
    with av.open(str(CLIP), "w") as c:
        vs = c.add_stream("libx264", rate=10)
        vs.width, vs.height, vs.pix_fmt = 64, 64, "yuv420p"
        ast = c.add_stream("aac", rate=16000)
        ast.layout = "mono"
        for i in range(40):
            fr = av.VideoFrame(64, 64, "rgb24").reformat(format="yuv420p")
            for pkt in vs.encode(fr):
                c.mux(pkt)
        for pkt in vs.encode(None):
            c.mux(pkt)
        t = np.arange(64000) / 16000
        a = av.AudioFrame.from_ndarray((np.sin(2 * np.pi * 440 * t)[None] * 8000).astype(np.int16),
                                       format="s16", layout="mono")
        a.sample_rate = 16000
        for pkt in ast.encode(a):
            c.mux(pkt)
        for pkt in ast.encode(None):
            c.mux(pkt)


@pytest.fixture(scope="module", autouse=True)
def _init_db():
    import server.db as db
    db.init_db()
    if not CLIP.exists():
        _make_clip()
    yield


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _upload(client, name="clip.mp4"):
    with open(CLIP, "rb") as f:
        r = client.post("/v1/analyses", files={"file": (name, f, "video/mp4")})
    assert r.status_code == 202, r.text
    return r.json()


def _wait_done(client, aid, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = client.get(f"/v1/analyses/{aid}").json()
        if r.get("schema_version"):  # done: the full contract, no status key
            return r
        if r.get("status") == "error":
            pytest.fail(f"analysis errored: {r.get('error')}")
        time.sleep(0.3)
    pytest.fail("analysis timed out")


def test_health(client):
    h = client.get("/v1/health").json()
    assert h["ok"] and h["mode"] == "mock"


def test_upload_result_contract(client):
    j = _upload(client)
    r = _wait_done(client, j["id"])
    result = client.get(f"/v1/analyses/{j['id']}").json()
    assert result["schema_version"] == "2.0"
    assert result["verdict"]["label"] in ("REAL", "SUSPECT", "FAKE")
    assert set(result["branches"]) == {"sync", "splice", "face", "age"}
    assert "headline_en" in result["verdict"] and "headline_hi" in result["verdict"]
    _validate_contract(result)


def _validate_contract(result):
    import jsonschema
    repo = pathlib.Path(__file__).resolve().parents[1]
    schema = json.loads((repo / "contracts" / "analysis.schema.json").read_text())
    jsonschema.validate(result, schema)


def test_sse_stream(client):
    j = _upload(client)
    _wait_done(client, j["id"])
    with client.stream("GET", f"/v1/analyses/{j['id']}/events") as r:
        assert r.status_code == 200
        body = b"".join(r.iter_bytes()).decode()
    assert "event: verdict" in body and "event: end" in body


def test_reject_bad_upload(client):
    r = client.post("/v1/analyses", files={"file": ("evil.exe", b"MZ\x90\x00", "application/octet-stream")})
    assert r.status_code == 415
    assert r.headers["content-type"].startswith("application/problem+json")


def test_reject_oversize_id(client):
    assert client.get("/v1/analyses/..%2F..%2Fetc").status_code in (400, 404)
    assert client.get("/v1/analyses/x/artifacts/..%2Fsecrets").status_code in (400, 404)


def test_resync(client):
    j = _upload(client)
    _wait_done(client, j["id"])
    r = client.post(f"/v1/analyses/{j['id']}/resync", json={"offset_ms": 120})
    assert r.status_code == 200, r.text
    url = r.json()["media_url"]
    art = client.get(url)
    assert art.status_code == 200


def test_chain(client):
    j = _upload(client)
    _wait_done(client, j["id"])
    r = client.post("/v1/chains", json={"media_id": j["id"], "gens": 2})
    assert r.status_code == 202, r.text
    cid = r.json()["chain_id"]
    with client.stream("GET", f"/v1/analyses/{cid}/events") as resp:
        body = b"".join(resp.iter_bytes()).decode()
    assert body.count("event: gen_ready") == 2
    assert "event: end" in body


def test_redteam_splice(client):
    j = _upload(client)
    _wait_done(client, j["id"])
    r = client.post("/v1/redteam/splice", json={"media_id": j["id"], "op": "delete", "t0": 0.5, "t1": 1.2})
    assert r.status_code == 200, r.text
    out = r.json()
    assert len(out["truth"]["boundaries"]) >= 1
    art = client.get(out["media_url"])
    assert art.status_code == 200


def test_redteam_splice_validation(client):
    j = _upload(client)
    _wait_done(client, j["id"])
    assert client.post("/v1/redteam/splice", json={"media_id": j["id"], "op": "nuke", "t0": 0, "t1": 1}).status_code == 422
    assert client.post("/v1/redteam/splice", json={"media_id": j["id"], "op": "delete", "t0": 5, "t1": 1}).status_code == 422


def test_dossier_and_card(client):
    j = _upload(client)
    _wait_done(client, j["id"])
    pdf = client.get(f"/v1/analyses/{j['id']}/dossier.pdf")
    assert pdf.status_code == 200 and pdf.headers["content-type"] == "application/pdf"
    card = client.get(f"/v1/analyses/{j['id']}/card.png?lang=hi")
    assert card.status_code == 200 and card.headers["content-type"] == "image/png"
    import io
    from PIL import Image
    assert Image.open(io.BytesIO(card.content)).size == (1080, 1350)
