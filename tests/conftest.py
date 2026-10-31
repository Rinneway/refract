from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from refract.config import Settings
from refract.knowledge import DataStore

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "demo"
NOW = datetime(2026, 10, 31, 9, 0, tzinfo=timezone.utc)


@pytest.fixture
def store() -> DataStore:
    s = DataStore.from_dir(DEMO)
    s.today = date(2026, 10, 31)
    return s


@pytest.fixture
def settings() -> Settings:
    return Settings(
        api_key="scripted",
        language="ru",
        cache_dir=Path(".cache/test-llm"),
        log_path=Path(".cache/test-llm/log.jsonl"),
    )
