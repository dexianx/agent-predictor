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
    price_usd: float | None = None
    item_avg_rating: float | None = None
    item_rating_count: int | None = None
    parent_asin: str = ""
    child_asin: str = ""
    item_store: str = ""
    item_features: list[str] = field(default_factory=list)
    item_description: list[str] = field(default_factory=list)
    item_image_url: str = ""
    item_images: list[str] = field(default_factory=list)
    item_categories: list[str] = field(default_factory=list)
    item_details: dict[str, Any] = field(default_factory=dict)
    bought_together: list[str] = field(default_factory=list)
    verified_purchase: bool | None = None
    helpful_vote: int | None = None
    review_images: list[str] = field(default_factory=list)
    word_count: int = 0
    embedding: list[float] = field(default_factory=list)
    stylometry: dict[str, Any] = field(default_factory=dict)

    def as_neo4j_row(self) -> dict[str, Any]:
        row = self.__dict__.copy()
        row["stylometry_json"] = json.dumps(self.stylometry, sort_keys=True)
        row["item_details_json"] = json.dumps(self.item_details, sort_keys=True)
        row["bought_together_count"] = len(self.bought_together)
        return row
