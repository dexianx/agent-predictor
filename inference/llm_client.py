import json
import os
import asyncio
from typing import AsyncIterator, TypeVar

import httpx
from google import genai
from google.genai import types
from pydantic import BaseModel

# Providers: gemini (default) | groq | cerebras
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").strip().lower()
_VALID_PROVIDERS = frozenset({"gemini", "groq", "cerebras"})
_OPENAI_COMPAT_PROVIDERS = frozenset({"groq", "cerebras"})

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
CEREBRAS_MODEL = os.environ.get("CEREBRAS_MODEL", "gpt-oss-120b")
CEREBRAS_BASE_URL = os.environ.get("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")

# Shared retry policy for OpenAI-compatible providers (Groq, Cerebras)
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", os.environ.get("GROQ_MAX_RETRIES", "6")))
LLM_RETRY_BASE_SECONDS = float(
    os.environ.get("LLM_RETRY_BASE_SECONDS", os.environ.get("GROQ_RETRY_BASE_SECONDS", "2.0"))
)

_gemini_client: genai.Client | None = None
_openai_http: httpx.AsyncClient | None = None
_openai_http_provider: str | None = None

T = TypeVar("T", bound=BaseModel)


def _ensure_valid_provider() -> None:
    if LLM_PROVIDER not in _VALID_PROVIDERS:
        raise RuntimeError(
            f"Unknown LLM_PROVIDER={LLM_PROVIDER!r}; use one of: {', '.join(sorted(_VALID_PROVIDERS))}"
        )


def _uses_openai_compat() -> bool:
    _ensure_valid_provider()
    return LLM_PROVIDER in _OPENAI_COMPAT_PROVIDERS


def _openai_compat_config() -> dict[str, str]:
    if LLM_PROVIDER == "cerebras":
        return {
            "base_url": CEREBRAS_BASE_URL,
            "api_key_env": "CEREBRAS_API_KEY",
            "default_model": CEREBRAS_MODEL,
            "label": "Cerebras",
        }
    return {
        "base_url": GROQ_BASE_URL,
        "api_key_env": "GROQ_API_KEY",
        "default_model": GROQ_MODEL,
        "label": "Groq",
    }


def llm_display_name() -> str:
    if _uses_openai_compat():
        cfg = _openai_compat_config()
        return f"{cfg['label']} · {active_model()}"
    return f"Gemini · {active_model()}"


def active_model() -> str:
    override = os.environ.get("LLM_MODEL", "").strip()
    if override:
        return override
    if _uses_openai_compat():
        return _openai_compat_config()["default_model"]
    return GEMINI_MODEL


def get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
    return _gemini_client


def get_openai_compat_client() -> httpx.AsyncClient:
    global _openai_http, _openai_http_provider
    cfg = _openai_compat_config()
    if _openai_http is None or _openai_http_provider != LLM_PROVIDER:
        _openai_http = httpx.AsyncClient(base_url=cfg["base_url"], timeout=120.0)
        _openai_http_provider = LLM_PROVIDER
    return _openai_http


def _openai_compat_headers() -> dict[str, str]:
    cfg = _openai_compat_config()
    key = os.environ.get(cfg["api_key_env"], "")
    if not key:
        raise RuntimeError(f"{cfg['api_key_env']} is not set (LLM_PROVIDER={LLM_PROVIDER})")
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


class CalibratedReviewRating(BaseModel):
    rating: int  # 1–5
    confidence: float = 0.5
    evidence_used: list[str] = []
    calibration_reason: str = ""


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


# ── OpenAI-compatible backends (Groq, Cerebras) ───────────────────────────────

async def _openai_compat_review_stream(system: str, user_prompt: str) -> AsyncIterator[str]:
    client = get_openai_compat_client()
    payload = {
        "model": active_model(),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        "stream": True,
    }
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            async with client.stream(
                "POST",
                "/chat/completions",
                json=payload,
                headers=_openai_compat_headers(),
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
                return
        except httpx.HTTPStatusError as exc:
            if not _should_retry_http(exc) or attempt >= LLM_MAX_RETRIES:
                raise
            await _sleep_for_retry(attempt, exc.response)
        except (httpx.TimeoutException, httpx.TransportError):
            if attempt >= LLM_MAX_RETRIES:
                raise
            await _sleep_for_retry(attempt, None)


async def _openai_compat_structured(
    user_prompt: str,
    schema: type[T],
    *,
    system: str | None = None,
) -> T | None:
    client = get_openai_compat_client()
    system_content = system or _json_schema_instruction(schema)
    payload = {
        "model": active_model(),
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    resp = await _openai_compat_post_with_retry(client, payload)
    content = resp.json()["choices"][0]["message"]["content"]
    try:
        return schema.model_validate_json(content)
    except Exception:
        return None


# ── Public API ────────────────────────────────────────────────────────────────

async def call_review_stream(system: str, user_prompt: str):
    """Async generator yielding token strings."""
    if _uses_openai_compat():
        async for token in _openai_compat_review_stream(system, user_prompt):
            yield token
    else:
        async for token in _gemini_review_stream(system, user_prompt):
            yield token


async def call_rating(prompt: str) -> int | None:
    """Structured output: returns int 1-5 or None."""
    if _uses_openai_compat():
        parsed = await _openai_compat_structured(prompt, ReviewRating)
    else:
        parsed = await _gemini_structured(prompt, ReviewRating)
    if parsed is None:
        return None
    try:
        return max(1, min(5, int(parsed.rating)))
    except Exception:
        return None


async def call_calibrated_rating(prompt: str) -> dict | None:
    """Structured rating with evidence for calibration/debugging."""
    if _uses_openai_compat():
        parsed = await _openai_compat_structured(prompt, CalibratedReviewRating)
    else:
        parsed = await _gemini_structured(prompt, CalibratedReviewRating)
    if parsed is None:
        return None
    try:
        rating = max(1, min(5, int(parsed.rating)))
    except Exception:
        return None
    return {
        "rating": rating,
        "confidence": max(0.0, min(1.0, float(parsed.confidence or 0.5))),
        "evidence_used": list(parsed.evidence_used or []),
        "calibration_reason": parsed.calibration_reason or "",
    }


async def call_generate_cypher(prompt: str) -> RetrievalQuery:
    """Structured output: returns a Cypher query string."""
    if _uses_openai_compat():
        parsed = await _openai_compat_structured(prompt, RetrievalQuery)
    else:
        parsed = await _gemini_structured(prompt, RetrievalQuery)
    if parsed is not None:
        return parsed
    return RetrievalQuery(cypher="")


async def _openai_compat_post_with_retry(client: httpx.AsyncClient, payload: dict) -> httpx.Response:
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            resp = await client.post(
                "/chat/completions",
                json=payload,
                headers=_openai_compat_headers(),
            )
            resp.raise_for_status()
            return resp
        except httpx.HTTPStatusError as exc:
            if not _should_retry_http(exc) or attempt >= LLM_MAX_RETRIES:
                raise
            await _sleep_for_retry(attempt, exc.response)
        except (httpx.TimeoutException, httpx.TransportError):
            if attempt >= LLM_MAX_RETRIES:
                raise
            await _sleep_for_retry(attempt, None)
    raise RuntimeError("OpenAI-compatible LLM retry loop exhausted")


def _should_retry_http(exc: httpx.HTTPStatusError) -> bool:
    return exc.response.status_code == 429 or 500 <= exc.response.status_code < 600


async def _sleep_for_retry(attempt: int, response: httpx.Response | None) -> None:
    retry_after = None
    if response is not None:
        retry_after = response.headers.get("retry-after")
    try:
        delay = float(retry_after) if retry_after else None
    except ValueError:
        delay = None
    if delay is None:
        delay = LLM_RETRY_BASE_SECONDS * (2 ** attempt)
    await asyncio.sleep(min(delay, 60.0))
