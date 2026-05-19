from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
AGENTS_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = AGENTS_ROOT.parent
OUTPUT_ROOT = PROJECT_ROOT / "output"
