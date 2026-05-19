from __future__ import annotations
from typing import Any, Literal, Union
from pydantic import BaseModel, Field


class SimulateRequest(BaseModel):
    user_id: str
    item_name: str
    price_ngn: float
    item_id: str = ""            # optional; when set, uses graph item node
    item_description: str | None = None
    raw_message: str | None = None   # natural language → LLM Cypher retrieval


class SimulateResponse(BaseModel):
    user_id: str
    item_id: str
    review_text: str
    predicted_rating: int
    locale: str | None
    retrieval_branch: str


class CategoryBias(BaseModel):
    category: str
    mean: float
    count: int


class PersonaDetail(BaseModel):
    user_id: str
    locale: str | None
    platform: str | None
    review_count: int | None
    rating_mean: float | None
    rating_std: float | None
    rating_skew: float | None
    top_categories: list[str] = Field(default_factory=list)
    avg_sent_len: float | None
    vocab_richness: float | None
    exclamation_rate: float | None
    caps_rate: float | None
    pidgin_freq: float | None
    emoji_rate: float | None
    question_rate: float | None
    naira_mention_rate: float | None
    price_sensitivity: float | None
    authenticity_concern: float | None
    cat_bias: list[CategoryBias] = Field(default_factory=list)


class GraphNode(BaseModel):
    id: str
    type: str
    label: str
    text: str | None = None
    age: int | None = None
    score: float | None = None
    isTarget: bool = False


class GraphEdge(BaseModel):
    source: str
    target: str
    label: str
    score: float | None = None


class GraphData(BaseModel):
    branch: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]


# WebSocket event union
class StepEvent(BaseModel):
    type: Literal["step"]
    id: str
    label: str
    detail: str | None = None


class TokenEvent(BaseModel):
    type: Literal["token"]
    text: str


class GraphDataEvent(BaseModel):
    type: Literal["graph_data"]
    branch: str
    nodes: list[dict]
    edges: list[dict]


class FinalEvent(BaseModel):
    type: Literal["final"]
    user_id: str
    item_id: str
    review_text: str
    predicted_rating: int
    locale: str | None
    retrieval_branch: str


class ErrorEvent(BaseModel):
    type: Literal["error"]
    message: str


WsEvent = Union[StepEvent, TokenEvent, GraphDataEvent, FinalEvent, ErrorEvent]
