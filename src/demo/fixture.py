from __future__ import annotations

import json
from pathlib import Path

from demo.contracts import DemoFixture


def load_david_fixture() -> DemoFixture:
    """Load the repository-controlled, immutable-at-runtime demo profile."""
    path = Path(__file__).with_name("fixtures") / "david-v1.json"
    return DemoFixture.model_validate(json.loads(path.read_text(encoding="utf-8")))
