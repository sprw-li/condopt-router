"""Tests for the literature -> router experience aggregation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "literature_experience_report.py"
spec = importlib.util.spec_from_file_location("literature_experience_report", SCRIPT)
report = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(report)


def labels() -> list[dict]:
    rows = []
    for index in range(6):
        rows.append({
            "paper_id": f"p{index}",
            "space_type": "categorical_heavy",
            "problem_type": "reaction",
            "objectives": "single",
            "constraints": False,
            "noisy": False,
            "automation": "HTE",
            "literature_method": "BO",
            "router_recommendation": "bayes",
            "confidence": "medium",
            "evidence": "five categorical variables, Bayesian optimization",
        })
    for index in range(3):
        rows.append({
            "paper_id": f"c{index}",
            "space_type": "continuous_heavy",
            "problem_type": "process",
            "objectives": "single",
            "constraints": False,
            "noisy": True,
            "automation": "flow",
            "literature_method": "BO",
            "router_recommendation": "bayes",
            "confidence": "high",
        })
    return rows


def test_normalise_and_crosstab():
    cleaned = report.normalise(labels())
    assert len(cleaned) == 9
    assert cleaned[0]["space_type"] == "categorical_heavy"
    table = report.crosstab(cleaned, "space_type")
    assert table["categorical_heavy"]["bayes"] == 6


def test_method_agreement():
    cleaned = report.normalise(labels())
    agreement = report.method_agreement(cleaned)
    assert agreement["n"] == 9
    assert agreement["agreement"] == 1.0


def test_proposals_and_markdown():
    experience = report.summarise(report.normalise(labels()), [])
    assert experience["n_labels"] == 9
    rules = {item["rule"] for item in experience["proposals"]}
    assert "categorical_heavy -> prefer bayes" in rules
    text = report.markdown(experience)
    assert "Router experience from literature labels" in text
    assert "categorical_heavy -> prefer bayes" in text
