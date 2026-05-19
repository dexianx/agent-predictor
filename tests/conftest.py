import asyncio
import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


TEST_USER_ID = "NG_KONGA_2ae66f90"
TEST_MESSAGE = "Write a review for Sony WH-1000XM4 wireless headphones"
