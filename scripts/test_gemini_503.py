#!/usr/bin/env python3
"""Probe Gemini 503 vs input size (small / medium / large)."""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from google import genai
from google.genai import types

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
API_KEY = os.environ.get("GEMINI_API_KEY", "")

SMALL_SYSTEM = "You are a helpful assistant. Reply in one short sentence."
SMALL_USER = "Say hello and confirm you received this."

MEDIUM_USER = "Summarize this in one sentence:\n" + ("The product was okay. " * 200)

# Rough upper bound of a heavy simulate_review ICL prompt
LARGE_REVIEW = (
    "I bought this item last month. Delivery was fast but packaging was damaged. "
    "Quality is decent for the price in Lagos. Would recommend with reservations. "
)
LARGE_USER = (
    "Write a short product review in the voice described below.\n\n"
    + "## Style exemplars\n"
    + "\n".join(f"Review {i}: {LARGE_REVIEW * 8}" for i in range(1, 9))
    + "\n\n## Content anchors\n"
    + "\n".join(f"Anchor {i}: {LARGE_REVIEW * 6}" for i in range(1, 11))
    + "\n\nProduct: Wireless earbuds. Category: Electronics. Price: ₦45000."
)


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


async def one_call(
    client: genai.Client,
    label: str,
    system: str,
    user: str,
) -> dict:
    in_tok = approx_tokens(system) + approx_tokens(user)
    t0 = time.perf_counter()
    try:
        response = await client.aio.models.generate_content(
            model=MODEL,
            contents=user,
            config=types.GenerateContentConfig(system_instruction=system),
        )
        elapsed = time.perf_counter() - t0
        text = (response.text or "")[:120]
        usage = getattr(response, "usage_metadata", None)
        prompt_tok = getattr(usage, "prompt_token_count", None) if usage else None
        total_tok = getattr(usage, "total_token_count", None) if usage else None
        return {
            "label": label,
            "ok": True,
            "elapsed_s": round(elapsed, 2),
            "approx_input_tokens": in_tok,
            "prompt_token_count": prompt_tok,
            "total_token_count": total_tok,
            "preview": text.replace("\n", " "),
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        err = str(exc)
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        return {
            "label": label,
            "ok": False,
            "elapsed_s": round(elapsed, 2),
            "approx_input_tokens": in_tok,
            "error_type": type(exc).__name__,
            "code": code,
            "error": err[:500],
        }


async def main() -> int:
    if not API_KEY:
        print("GEMINI_API_KEY not set", file=sys.stderr)
        return 1

    client = genai.Client(api_key=API_KEY)
    print(f"Model: {MODEL}")
    print("Running 3 non-streaming generate_content calls...\n")

    cases = [
        ("small", SMALL_SYSTEM, SMALL_USER),
        ("medium", SMALL_SYSTEM, MEDIUM_USER),
        ("large_icl_like", SMALL_SYSTEM, LARGE_USER),
    ]

    for label, system, user in cases:
        result = await one_call(client, label, system, user)
        status = "OK" if result["ok"] else "FAIL"
        print(f"── {label} [{status}] ──")
        print(f"  approx input tokens: ~{result['approx_input_tokens']:,}")
        if result["ok"]:
            print(f"  API prompt tokens:   {result.get('prompt_token_count')}")
            print(f"  elapsed:           {result['elapsed_s']}s")
            print(f"  preview:           {result['preview']!r}")
        else:
            print(f"  elapsed:           {result['elapsed_s']}s")
            print(f"  {result['error_type']}: {result['error']}")
        print()
        await asyncio.sleep(1.5)

    # Stream path (same as call_review_stream)
    print("── stream (large_icl_like) ──")
    t0 = time.perf_counter()
    try:
        n = 0
        async for chunk in await client.aio.models.generate_content_stream(
            model=MODEL,
            contents=LARGE_USER,
            config=types.GenerateContentConfig(system_instruction=SMALL_SYSTEM),
        ):
            if chunk.text:
                n += len(chunk.text)
        print(f"  OK — received ~{n} chars in {round(time.perf_counter() - t0, 2)}s")
    except Exception as exc:
        print(f"  FAIL in {round(time.perf_counter() - t0, 2)}s")
        print(f"  {type(exc).__name__}: {str(exc)[:500]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
