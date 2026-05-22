import json
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from task_a.inference.persona_loader import (
    load_persona,
    load_featured_personas,
    search_personas,
)
from task_a.inference.task_a import SimulateRequest, simulate_review
from api.models import PersonaDetail, CategoryBias, SimulateResponse

import sqlite3

SQLITE_PATH = os.environ.get("SQLITE_PATH", "./personas.db")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="Kora Task A API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ── Persona endpoints ─────────────────────────────────────────────────────────

def _to_detail(p: dict) -> PersonaDetail:
    cat_bias = [
        CategoryBias(category=cat, mean=v["mean"], count=v["count"])
        for cat, v in p.get("cat_bias", {}).items()
    ]
    return PersonaDetail(
        user_id=p["user_id"],
        locale=p.get("locale"),
        platform=p.get("platform"),
        review_count=p.get("review_count"),
        rating_mean=p.get("rating_mean"),
        rating_std=p.get("rating_std"),
        rating_skew=p.get("rating_skew"),
        top_categories=p.get("top_categories") or [],
        avg_sent_len=p.get("avg_sent_len"),
        vocab_richness=p.get("vocab_richness"),
        exclamation_rate=p.get("exclamation_rate"),
        caps_rate=p.get("caps_rate"),
        pidgin_freq=p.get("pidgin_freq"),
        emoji_rate=p.get("emoji_rate"),
        question_rate=p.get("question_rate"),
        naira_mention_rate=p.get("naira_mention_rate"),
        price_sensitivity=p.get("price_sensitivity"),
        authenticity_concern=p.get("authenticity_concern"),
        cat_bias=cat_bias,
    )


@app.get("/personas/featured", response_model=list[PersonaDetail])
def get_featured_personas():
    rows = load_featured_personas(SQLITE_PATH, limit=8)
    return [_to_detail(r) for r in rows]


@app.get("/personas", response_model=list[PersonaDetail])
def list_personas(
    search: str = "",
    locale: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    rows = search_personas(SQLITE_PATH, query=search, locale=locale,
                           limit=min(limit, 200), offset=offset)
    return [_to_detail(r) for r in rows]


@app.get("/personas/{user_id}", response_model=PersonaDetail)
def get_persona(user_id: str):
    p = load_persona(SQLITE_PATH, user_id)
    if p is None:
        raise HTTPException(status_code=404, detail="Persona not found")
    return _to_detail(p)


# ── WebSocket simulate ────────────────────────────────────────────────────────

@app.websocket("/ws/simulate")
async def ws_simulate(websocket: WebSocket):
    await websocket.accept()
    try:
        raw = await websocket.receive_text()
        data = json.loads(raw)
        req = SimulateRequest(**data)

        async def emit(event: dict):
            await websocket.send_text(json.dumps(event))

        await simulate_review(req, emit)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ── REST fallback ─────────────────────────────────────────────────────────────

@app.post("/simulate", response_model=SimulateResponse)
async def rest_simulate(req: SimulateRequest):
    events: list[dict] = []

    async def collect(ev: dict):
        events.append(ev)

    await simulate_review(req, collect)

    final = next((e for e in events if e.get("type") == "final"), None)
    if final is None:
        error = next((e for e in events if e.get("type") == "error"), None)
        msg = error["message"] if error else "Simulation failed"
        raise HTTPException(status_code=500, detail=msg)

    return SimulateResponse(
        user_id=final["user_id"],
        item_id=final["item_id"],
        review_text=final["review_text"],
        predicted_rating=final["predicted_rating"],
        locale=final.get("locale"),
        retrieval_branch=final["retrieval_branch"],
    )
