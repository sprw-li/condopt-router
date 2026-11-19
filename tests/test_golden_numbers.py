"""A3: archived numbers are registered and the checker can go red on demand."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_golden_numbers.py"


def _module():
    spec = importlib.util.spec_from_file_location("check_golden_numbers", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_registered_numbers_are_present_in_their_documents():
    report = _module().check()
    assert report["problems"] == [], report["problems"]
    assert report["checked"] >= 5


def test_checker_detects_a_corrupted_number_in_memory():
    """The negative control runs without touching any file on disk."""
    report = _module().self_test()
    assert report["failures"] == [], report["failures"]
    assert report["entries"] >= 5


def test_registry_entries_are_well_formed():
    registry = json.loads((ROOT / "docs" / "GOLDEN_NUMBERS.json").read_text(encoding="utf-8"))
    for entry in registry["entries"]:
        for key in ("id", "value", "tolerance", "doc", "line_hint", "number_text"):
            assert key in entry, (entry.get("id"), key)
        assert (ROOT / entry["doc"]).exists(), entry["doc"]
