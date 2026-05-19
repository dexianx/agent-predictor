import pytest
import json
from httpx import AsyncClient, ASGITransport
from api.main import app


@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/health")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_featured_personas_includes_ng():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/personas/featured")
    assert r.status_code == 200
    locales = {p["locale"] for p in r.json()}
    assert "NG" in locales


@pytest.mark.asyncio
async def test_ws_simulate_full_flow():
    import websockets

    payload = json.dumps({
        "user_id": "NG_KONGA_2ae66f90",
        "raw_message": "Write a review for Sony headphones",
    })
    events = []
    async with websockets.connect("ws://localhost:8000/ws/simulate") as ws:
        await ws.send(payload)
        while True:
            ev = json.loads(await ws.recv())
            events.append(ev)
            if ev["type"] in ("final", "error"):
                break
    types = {e["type"] for e in events}
    assert "step" in types and "token" in types and "final" in types
    assert "error" not in types
