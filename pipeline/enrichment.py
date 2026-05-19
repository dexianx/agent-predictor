from __future__ import annotations

import re
from typing import Iterable

from .config import MIN_WORDS_FOR_STYLOMETRY, PIDGIN_TOKENS
from .models import ReviewRecord


def compute_stylometry(text: str) -> dict[str, float | int]:
    words = text.split()
    chars = max(len(text), 1)
    n_words = max(len(words), 1)
    sentences = [part.strip() for part in re.split(r"[.!?]+", text) if part.strip()]
    n_sentences = max(len(sentences), 1)
    tokens = [word.lower().strip(".,!?\"'():;[]{}") for word in words]
    lowered = text.lower()

    return {
        "avg_sent_len": round(n_words / n_sentences, 2),
        "vocab_richness": round(len(set(tokens)) / n_words, 3),
        "exclamation_rate": round(text.count("!") / chars, 4),
        "caps_rate": round(sum(char.isupper() for char in text) / chars, 4),
        "pidgin_freq": round(sum(token in PIDGIN_TOKENS for token in tokens) / n_words, 4),
        "emoji_rate": round(len(re.findall(r"[\U00010000-\U0010ffff]", text)) / n_words, 4),
        "question_rate": round(text.count("?") / chars, 4),
        "avg_word_len": round(sum(len(word) for word in words) / n_words, 2),
        "uppercase_word_rate": round(sum(word.isupper() and len(word) > 1 for word in words) / n_words, 4),
        "naira_mention": int("₦" in text or "naira" in lowered),
        "price_sensitivity": int(
            any(
                phrase in lowered
                for phrase in ("overpriced", "too expensive", "not worth", "value for money", "worth the money")
            )
        ),
        "authenticity_concern": int(
            any(phrase in lowered for phrase in ("original", "fake", "tokunbo", "confirmed"))
        ),
    }


class Embedder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Embedding requires `sentence-transformers`. Install requirements.txt before running the pipeline."
            ) from exc
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return [embedding.tolist() for embedding in embeddings]


class HashingEmbedder(Embedder):
    """Tiny deterministic test embedder used only when explicitly requested."""

    def __init__(self, dimensions: int):
        self.dimensions = dimensions

    def encode(self, texts: list[str]) -> list[list[float]]:
        import hashlib
        import math

        vectors: list[list[float]] = []
        for text in texts:
            values = [0.0] * self.dimensions
            for token in text.lower().split():
                digest = hashlib.sha1(token.encode("utf-8")).digest()
                values[int.from_bytes(digest[:4], "big") % self.dimensions] += 1.0
            norm = math.sqrt(sum(value * value for value in values)) or 1.0
            vectors.append([round(value / norm, 6) for value in values])
        return vectors


def enrich_records(records: list[ReviewRecord], embedder: Embedder, batch_size: int) -> list[ReviewRecord]:
    for record in records:
        record.word_count = len(record.text.split())
        record.stylometry = (
            compute_stylometry(record.text)
            if record.word_count >= MIN_WORDS_FOR_STYLOMETRY
            else {}
        )

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        texts = [
            record.text
            if record.word_count >= 2
            else (record.title or record.item_title or "no review text")
            for record in batch
        ]
        for record, embedding in zip(batch, embedder.encode(texts)):
            record.embedding = embedding
        if len(records) >= 10_000 and (start + len(batch)) % max(batch_size * 100, 1) == 0:
            print(f"Embedded {start + len(batch):,}/{len(records):,} reviews", flush=True)
    return records
