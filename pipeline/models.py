from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any


@dataclass
class ReviewRecord:
    review_id: str
    user_id: str
    item_id: str
    rating: float
    text: str
    title: str
    timestamp: int
    locale: str
    platform: str
    category: str
    sub_category: str
    item_title: str
    item_brand: str
    price_ngn: float | None
    item_avg_rating: float | None
    word_count: int = 0
    embedding: list[float] = field(default_factory=list)
    stylometry: dict[str, Any] = field(default_factory=dict)

    def as_neo4j_row(self) -> dict[str, Any]:
        row = self.__dict__.copy()
        row["stylometry_json"] = json.dumps(self.stylometry, sort_keys=True)
        return row
