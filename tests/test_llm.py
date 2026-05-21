import pytest
from inference.llm_client import call_rating, call_generate_cypher


@pytest.mark.asyncio
async def test_rating_returns_valid_integer():
    rating = await call_rating("Rate this review on 1-5: Great product, fast shipping.")
    assert rating is not None and 1 <= rating <= 5


@pytest.mark.asyncio
async def test_generate_cypher_is_read_only():
    from inference.retrieval import generate_retrieval_query

    cypher = await generate_retrieval_query(
        "Write a review for Apple AirPods Pro 2 wireless earbuds",
        "US",
        item_category="Home Audio & Theater",
    )
    assert cypher
    first_token = cypher.strip().split()[0].upper()
    assert first_token in ("MATCH", "CALL"), f"got: {first_token}"


@pytest.mark.asyncio
async def test_generate_cypher_from_raw_message():
    result = await call_generate_cypher(
        'Write a READ-ONLY Cypher MATCH query to find reviews about "iPhone 15 Pro". '
        "Return text, rating, locale, item_name, category. LIMIT 5."
    )
    assert result.cypher
    assert result.cypher.strip().upper().startswith(("MATCH", "CALL"))
