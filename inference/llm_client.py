import json
import os
from typing import AsyncIterator, TypeVar

import httpx
from google import genai
from google.genai import types
from pydantic import BaseModel

# Providers: gemini (default) | groq
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").strip().lower()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

_gemini_client: genai.Client | None = None
_groq_http: httpx.AsyncClient | None = None

T = TypeVar("T", bound=BaseModel)


def llm_display_name() -> str:
    if LLM_PROVIDER == "groq":
        return f"Groq · {active_model()}"
    return f"Gemini · {active_model()}"


def active_model() -> str:
    override = os.environ.get("LLM_MODEL", "").strip()
    if override:
        return override
    return GROQ_MODEL if LLM_PROVIDER == "groq" else GEMINI_MODEL


def get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
    return _gemini_client


def get_groq_client() -> httpx.AsyncClient:
    global _groq_http
    if _groq_http is None:
        _groq_http = httpx.AsyncClient(base_url=GROQ_BASE_URL, timeout=120.0)
    return _groq_http


def _groq_headers() -> dict[str, str]:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set (LLM_PROVIDER=groq)")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _json_schema_instruction(model: type[BaseModel]) -> str:
    return (
        "Respond with a single JSON object only, no markdown, matching this schema:\n"
        f"{json.dumps(model.model_json_schema())}"
    )


# ── Structured output schemas ─────────────────────────────────────────────────

class RetrievalQuery(BaseModel):
    cypher: str  # READ-only Cypher query (MATCH/CALL/RETURN only)


class ReviewRating(BaseModel):
    rating: int  # 1–5


# ── Gemini backends ───────────────────────────────────────────────────────────

async def _gemini_review_stream(system: str, user_prompt: str) -> AsyncIterator[str]:
    client = get_gemini_client()
    async for chunk in await client.aio.models.generate_content_stream(
        model=active_model(),
        contents=user_prompt,
        config=types.GenerateContentConfig(system_instruction=system),
    ):
        try:
            if chunk.text:
                yield chunk.text
        except Exception:
            pass


async def _gemini_structured(prompt: str, schema: type[T]) -> T | None:
    client = get_gemini_client()
    response = await client.aio.models.generate_content(
        model=active_model(),
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    try:
        return response.parsed
    except Exception:
        return None


# ── Groq backends (OpenAI-compatible chat completions) ────────────────────────

async def _groq_review_stream(system: str, user_prompt: str) -> AsyncIterator[str]:
    client = get_groq_client()
    payload = {
        "model": active_model(),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        "stream": True,
    }
    async with client.stream(
        "POST",
        "/chat/completions",
        json=payload,
        headers=_groq_headers(),
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                text = chunk["choices"][0]["delta"].get("content")
                if text:
                    yield text
            except (json.JSONDecodeError, KeyError, IndexError):
                continue


async def _groq_structured(
    user_prompt: str,
    schema: type[T],
    *,
    system: str | None = None,
) -> T | None:
    client = get_groq_client()
    system_content = system or _json_schema_instruction(schema)
    payload = {
        "model": active_model(),
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    resp = await client.post(
        "/chat/completions",
        json=payload,
        headers=_groq_headers(),
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    try:
        return schema.model_validate_json(content)
    except Exception:
        return None


# ── Public API ────────────────────────────────────────────────────────────────

async def call_review_stream(system: str, user_prompt: str):
    """Async generator yielding token strings."""
    if LLM_PROVIDER == "groq":
        async for token in _groq_review_stream(system, user_prompt):
            yield token
    else:
        async for token in _gemini_review_stream(system, user_prompt):
            yield token


async def call_rating(prompt: str) -> int | None:
    """Structured output: returns int 1-5 or None."""
    if LLM_PROVIDER == "groq":
        parsed = await _groq_structured(prompt, ReviewRating)
    else:
        parsed = await _gemini_structured(prompt, ReviewRating)
    if parsed is None:
        return None
    try:
        return max(1, min(5, int(parsed.rating)))
    except Exception:
        return None


async def call_generate_cypher(prompt: str) -> RetrievalQuery:
    """Structured output: returns a Cypher query string."""
    if LLM_PROVIDER == "groq":
        parsed = await _groq_structured(prompt, RetrievalQuery)
    else:
        parsed = await _gemini_structured(prompt, RetrievalQuery)
    if parsed is not None:
        return parsed
    return RetrievalQuery(cypher="")
