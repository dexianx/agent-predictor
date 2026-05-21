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
async def test_featured_personas_includes_us_amazon():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/personas/featured")
    assert r.status_code == 200
    locales = {p["locale"] for p in r.json()}
    assert "US" in locales


@pytest.mark.asyncio
async def test_ws_simulate_full_flow():
    import websockets

    from tests.conftest import TEST_USER_ID, TEST_MESSAGE

    payload = json.dumps({
        "user_id": TEST_USER_ID,
        "item_name": "Apple AirPods Pro 2 Wireless Earbuds",
        "price_ngn": 0.0,
        "price_usd": 249.99,
        "raw_message": TEST_MESSAGE,
        "item_category": "Home Audio & Theater",
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
