"""Tests for the reusable literature-scan pipeline."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "literature_scan.py"
spec = importlib.util.spec_from_file_location("literature_scan", SCRIPT)
literature_scan = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(literature_scan)


def sample_record() -> dict:
    return {
        "paper_id": "10.1000/test",
        "title": "Bayesian optimization of a Suzuki coupling",
        "year": 2026,
        "journal": "Test J",
        "doi": "10.1000/test",
        "abstract": (
            "We optimize a reaction with five categorical variables (solvents, ligands) "
            "and three continuous variables. Bayesian optimization is used with two "
            "objectives and a constraint on temperature. Replicates show measurement noise. "
            "A self-driving laboratory runs the experiments."
        ),
        "query_tags": ["Bayesian optimization"],
    }


def test_extract_features_from_abstract():
    paper = literature_scan.extract_features(sample_record())
    assert "Bayesian optimization" in paper["algorithm_terms"]
    assert "self-driving lab" in paper["automation_tags"]
    assert paper["space"]["categorical"] == 5
    assert paper["space"]["continuous"] == 3
    assert paper["multi_objective"] is True
    assert paper["n_objectives"] == 2
    assert paper["constraints"] is True
    assert paper["noisy"] is True
    assert paper["evidence"]


def test_aggregate_counts_subsets():
    papers = [literature_scan.extract_features(sample_record())]
    stats = literature_scan.aggregate(papers)
    assert stats["n_papers"] == 1
    assert stats["algorithms"]["Bayesian optimization"] == 1
    assert stats["subsets"]["categorical_heavy"]["n"] == 1
    assert stats["subsets"]["multi_objective"]["n"] == 1
    assert stats["subsets"]["constraints"]["n"] == 1


def test_chunk_and_read_jsonl(tmp_path):
    config = {
        "output_dir": tmp_path,
        "chunk_size": 1,
        "queries": [],
        "page_size": 100,
        "max_per_query": 10,
        "sleep_seconds": 0,
        "since": "2024-01-01",
        "until": "2026-12-31",
        "open_access": True,
    }
    papers = [literature_scan.extract_features(sample_record())]
    (tmp_path / "papers.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in papers) + "\n", encoding="utf-8"
    )
    payload = literature_scan.chunk(config)
    assert payload["n_chunks"] == 1
    chunk_file = tmp_path / "chunks" / "chunk_00.jsonl"
    assert chunk_file.exists()
    assert literature_scan._read_jsonl(chunk_file)[0]["paper_id"] == "10.1000/test"
