"""README quotes concrete example results; keep them pinned to reality."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_example(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_multiobjective_example_numbers_match_readme():
    """README: cqehvi 23 次观测 / 18 条可行 / Pareto 4 点 / HV 143.03（可交付 80.94）。"""
    module = _load_example("mo_example_snapshot", "examples/multiobjective_executor.py")
    report = module.run(budget=24, batch_size=4, seed=0)
    assert report["ok"] is True
    assert report["algorithm"] == "cqehvi"
    assert report["n_observations"] == 23
    assert report["n_feasible"] == 18
    assert report["pareto_size"] == 4
    assert round(report["hypervolume"], 2) == 143.03
    assert round(report["feasible_hypervolume"], 2) == 80.94
    # the README documents this run; both numbers must appear side by side
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    compact = readme.replace(" ", "")
    assert "23次观测" in compact and "80.94" in compact


def test_constrained_mf_example_numbers_match_readme():
    """README: mfbo 21 次观测 / 成本 7.5 / 3 条可行全保真 / best 34.227。"""
    module = _load_example(
        "mf_example_snapshot", "examples/constrained_mf_executor.py"
    )
    report = module.run_campaign(cost_budget=10.0, budget=24, seed=0)
    assert report["ok"] is True
    assert report["algorithm"] == "mfbo"
    assert report["n_observations"] == 21
    assert round(report["cost_spent"], 2) == 7.5
    assert report["n_feasible_full_fidelity"] == 3
    assert round(report["best_feasible_full_fidelity"], 3) == 34.227
