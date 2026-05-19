from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_COMBINED_ROOT = PROJECT_ROOT / "output" / "combined"
DEFAULT_NIGERIAN_REVIEWS = PROJECT_ROOT / "optimized_scraper" / "nigeria_reviews.csv"
DEFAULT_NIGERIAN_PLACES = PROJECT_ROOT / "optimized_scraper" / "nigeria_places.csv"
DEFAULT_SQLITE_PATH = PROJECT_ROOT / "personas.db"

DEFAULT_CATEGORIES = (
    "Home_and_Kitchen",
    "Electronics",
    "Cell_Phones_and_Accessories",
)

DEFAULT_SAMPLE_SIZE = 5_000
DEFAULT_AMAZON_ROWS = 2_500
DEFAULT_LOCAL_ROWS = 2_500
DEFAULT_NG_ECOMMERCE_ROWS = 1_500
DEFAULT_GOOGLE_MAPS_ROWS = 1_000

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "yourpassword")

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "64"))
NEO4J_WRITE_BATCH = int(os.getenv("NEO4J_WRITE_BATCH", "500"))
MIN_WORDS_FOR_STYLOMETRY = 5
MIN_REVIEWS_FOR_PERSONA = 3

AMAZON_DATASET = "McAuley-Lab/Amazon-Reviews-2023"
AMAZON_CONFIG_PREFIX = "raw_review_"

PIDGIN_TOKENS = frozenset(
    {
        "na",
        "sha",
        "dey",
        "abeg",
        "abi",
        "oga",
        "wahala",
        "nawa",
        "sabi",
        "chop",
        "wey",
        "dem",
        "una",
        "comot",
        "wetin",
        "sef",
        "joor",
    }
)

