"""The decision matrix must hold and its generated doc must stay current."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    script = ROOT / "scripts" / "decision_matrix.py"
    spec = importlib.util.spec_from_file_location("decision_matrix", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_every_decision_rule_holds():
    rows = _module().evaluate()
    failed = [row["name"] for row in rows if not row["ok"]]
    assert not failed, failed
    assert len(rows) >= 15


def test_decision_doc_matches_the_generator():
    module = _module()
    doc = ROOT / "docs" / "DECISIONS.md"
    assert doc.exists(), "run scripts/decision_matrix.py --write"
    assert doc.read_text(encoding="utf-8").strip() == module.build_doc(module.evaluate()).strip()


def test_every_rule_cites_evidence():
    for rule in _module().RULES:
        assert str(rule.evidence).strip(), f"{rule.name} has no evidence pointer"
        assert str(rule.expected).strip(), f"{rule.name} has no expected value"

def test_every_suppression_holds():
    rows = _module().evaluate_suppressions()
    failed = [row["name"] for row in rows if not row["ok"]]
    assert not failed, failed
    assert len(rows) >= 8

