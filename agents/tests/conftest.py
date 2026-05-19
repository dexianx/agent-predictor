import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENTS_ROOT = ROOT / "agents"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(AGENTS_ROOT))


def pytest_configure(config):
    config.addinivalue_line("markers", "live: tests that call live provider APIs")
