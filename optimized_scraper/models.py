from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ProductRef:
    url: str
    title: str | None = None
    sku: str | None = None
    brand: str | None = None
    price_ngn: float | None = None
    rating: float | None = None
    review_count: int | None = None
    categories: list[str] = field(default_factory=list)


@dataclass
class RawReview:
    rating: float | None
    title: str | None
    text: str
    reviewer_name: str | None = None
    date_raw: str | None = None
    verified_purchase: bool = False
    helpful_vote: int = 0


@dataclass
class RawProduct:
    url: str
    sku: str
    title: str
    brand: str | None
    seller: str | None
    price_ngn: float | None
    average_rating: float | None
    rating_number: int | None
    categories: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    description: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProductBundle:
    product: RawProduct
    reviews: list[RawReview]


@dataclass
class ReviewRecord:
    rating: float
    title: str
    text: str
    asin: str
    parent_asin: str
    user_id: str
    timestamp: int | None
    helpful_vote: int
    verified_purchase: bool
    platform: str
    main_category: str
    sub_category: str
    price_ngn: float | None
    brand: str | None
    seller: str | None
    city_signal: str | None
    language_signal: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MetaRecord:
    parent_asin: str
    asin: str
    main_category: str
    sub_category: str
    title: str
    brand: str | None
    average_rating: float | None
    rating_number: int | None
    price_ngn: float | None
    features: list[str]
    description: list[str]
    categories: list[str]
    details: dict[str, Any]
    seller: str | None
    platform: str
    url: str
    scraped_at: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
