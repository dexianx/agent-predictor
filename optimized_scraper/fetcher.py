from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from urllib.request import Request, urlopen

from .config import DEFAULT_REQUEST_DELAY, DEFAULT_TIMEOUT_SECONDS

DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-NG,en;q=0.9",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
}


class AsyncFetcher:
    def __init__(
        self,
        concurrency: int,
        delay: float = DEFAULT_REQUEST_DELAY,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ):
        self._sem = asyncio.Semaphore(concurrency)
        self._delay = delay
        self._timeout = timeout
        self._last_request: dict[str, float] = defaultdict(float)
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    def _blocking_get_text(self, url: str) -> str:
        request = Request(url, headers=DEFAULT_HEADERS)
        with urlopen(request, timeout=self._timeout) as response:
            return response.read().decode("utf-8", "ignore")

    async def get_text(self, url: str, retries: int = 3) -> str:
        host = url.split("/")[2] if "://" in url else ""
        async with self._sem:
            async with self._locks[host]:
                elapsed = time.monotonic() - self._last_request[host]
                if elapsed < self._delay:
                    await asyncio.sleep(self._delay - elapsed)
                self._last_request[host] = time.monotonic()

            last_error: Exception | None = None
            for attempt in range(retries):
                try:
                    return await asyncio.to_thread(self._blocking_get_text, url)
                except Exception as exc:
                    last_error = exc
                    await asyncio.sleep(1.5 * (attempt + 1))
            if last_error:
                raise last_error
            return ""
