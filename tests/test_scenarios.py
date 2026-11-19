"""The executable scenario matrix must stay green and complete."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "scenario_matrix.py"
_spec = importlib.util.spec_from_file_location("scenario_matrix", SCRIPT)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
# dataclasses need the module registered before exec_module
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)


def test_scenario_matrix_matches_documented_defaults():
    rows = _module.evaluate()
    failures = [row for row in rows if not row["ok"]]
    assert failures == [], failures
    assert len(rows) >= 19


def test_every_scenario_has_existing_evidence():
    for scenario in _module.SCENARIOS:
        assert scenario.evidence, scenario.name
        assert (ROOT / scenario.evidence).exists(), scenario.evidence
        assert scenario.problem, scenario.name


def test_scenarios_doc_lists_evidence_scripts():
    text = (ROOT / "docs" / "SCENARIOS.md").read_text(encoding="utf-8")
    for scenario in _module.SCENARIOS:
        assert f"`{scenario.evidence}`" in text, scenario.evidence


def test_scenarios_doc_lists_every_scenario():
    """Every scenario needs its own row with its key, core title and default algorithm.

    Checking only ``f"{key}." in text`` passed even when two scenarios reused the same
    key letter (the "G" collision), so the check now matches key + title + algorithm.
    """
    lines = (ROOT / "docs" / "SCENARIOS.md").read_text(encoding="utf-8").splitlines()
    for scenario in _module.SCENARIOS:
        key, rest = scenario.name.split(". ", 1)
        core = rest.split("（")[0]
        prefix = f"| {key}. {core}"
        matches = [
            line for line in lines
            if line.startswith(prefix) and f"`{scenario.expected}`" in line
        ]
        assert matches, (
            f"docs/SCENARIOS.md needs a routing row '{prefix}…' carrying "
            f"`{scenario.expected}` for {scenario.name}"
        )


def test_scenarios_doc_header_count_matches_the_matrix():
    text = (ROOT / "docs" / "SCENARIOS.md").read_text(encoding="utf-8")
    match = re.search(r"共 (\d+) 个场景", text)
    assert match is not None, "SCENARIOS.md lost its scenario count line"
    assert int(match.group(1)) == len(_module.SCENARIOS)


def test_scenarios_doc_documents_warm_start_decisions():
    text = (ROOT / "docs" / "SCENARIOS.md").read_text(encoding="utf-8")
    assert "## 暖启动决策（transfer）" in text
    for token in ("trust", "suspect", "avoid", "budget_charged", "多保真"):
        assert token in text, f"warm-start table lost {token!r}"

