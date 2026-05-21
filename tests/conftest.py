import asyncio
import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Amazon Reviews 2023 persona (US locale, amazon_us) — see evaluation/amazon_fixtures.py
TEST_USER_ID = "AEIIRIHLIYKQGI7ZOCIJTRDF5NPQ"
TEST_MESSAGE = "Write a review for Apple AirPods Pro 2 wireless earbuds"
