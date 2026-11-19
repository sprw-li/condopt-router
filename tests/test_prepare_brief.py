"""The generated prepare brief must stay in sync with prepare's real output."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_prepare_brief.py"
_spec = importlib.util.spec_from_file_location("build_prepare_brief", SCRIPT)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)


def test_prepare_brief_doc_matches_generator(tmp_path):
    _module.OUT_ROOT = tmp_path / "brief"
    cases = [_module.build_case(case) for case in _module.CASES]
    rendered = _module.render(cases)
    doc = (ROOT / "docs" / "PREPARE_BRIEF.md").read_text(encoding="utf-8")
    assert doc == rendered


def test_prepare_brief_advice_is_evidence_based(tmp_path):
    _module.OUT_ROOT = tmp_path / "brief"
    cases = {case["key"]: case for case in _module.CASES}
    built = {case["key"]: _module.build_case(case) for case in cases.values()}
    assert built["A"]["stopping"]["recommended"] == "stagnation_first"  # budget 24
    assert built["B"]["stopping"]["recommended"] == "budget_first"  # budget 120
    assert "69/120" in built["B"]["stopping"]["why"]
    assert built["C"]["feasibility"]["risk"] == "high"  # 3 constraints / 3 dims
    assert "写进变量范围" in built["C"]["feasibility"]["advice"]
    assert "HARD_FEASIBILITY" in built["C"]["feasibility"]["reference"]
    assert built["D"]["prior"]["has_prior"] is True
    assert "transfer-audit" in built["D"]["prior"]["advice"]

def test_prepare_fields_match_documented_guide(tmp_path):
    from condopt.prepare import FIELD_GUIDE, template, write_problem

    result = write_problem(template(), tmp_path / "fields")
    assert set(result) == set(FIELD_GUIDE), (
        f"write_problem keys {sorted(result)} != FIELD_GUIDE {sorted(FIELD_GUIDE)}"
    )
    doc = (ROOT / "docs" / "PREPARE_BRIEF.md").read_text(encoding="utf-8")
    for field in (
        "recommendation", "stopping", "feasibility", "prior", "deliverable", "noise_plan"
    ):
        assert f"`{field}`" in doc, f"PREPARE_BRIEF.md lost the {field!r} row"


def test_multiobjective_digest_reports_the_trajectory(tmp_path):
    import json

    from condopt.base import Objective
    from condopt.space import SearchSpace
    from condopt.study import Study

    space = SearchSpace.from_dict([
        {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}
    ])
    objective = Objective(
        name="yield", direction="max",
        objectives=("yield", "cost"), directions=("max", "min"),
    )
    study = Study.create(
        tmp_path / "traj", space, objective, algorithm="qehvi",
        budget=6, batch_size=3, seed=0,
    )
    study.import_observations([
        {"x": {"x": 0.1}, "y": 0.9, "objectives": {"cost": 0.1}},
        {"x": {"x": 0.2}, "y": 0.8, "objectives": {"cost": 0.2}},
    ])
    (tmp_path / "traj" / "progress.jsonl").write_text(
        "\n".join(json.dumps(row) for row in [
            {"round": 3, "n_front": 1, "hypervolume": 10.0, "n_feasible": 3},
            {"round": 6, "n_front": 2, "hypervolume": 12.5, "n_feasible": 6},
        ]) + "\n",
        encoding="utf-8",
    )
    report = study.report()
    assert "进展:" in report
    assert "10.0" in report and "12.5" in report
    assert study.progress_trajectory()[0]["round"] == 3


def test_constrained_multiobjective_digest_shows_deliverable_front(tmp_path):
    import json

    from condopt.base import Objective
    from condopt.space import SearchSpace
    from condopt.study import Study

    space = SearchSpace.from_dict([
        {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}
    ])
    objective = Objective(
        name="yield", direction="max",
        objectives=("yield", "cost"), directions=("max", "min"),
        constraints=("c",), constraint_limits={"c": {"op": "<=", "value": 0.5}},
    )
    study = Study.create(
        tmp_path / "deliverable", space, objective, algorithm="cqehvi",
        budget=6, batch_size=2, seed=0,
    )
    study.import_observations([
        {"x": {"x": 0.1}, "y": 0.9, "objectives": {"cost": 0.1}, "constraints": {"c": 0.1}},
        {"x": {"x": 0.2}, "y": 0.8, "objectives": {"cost": 0.2}, "constraints": {"c": 0.2}},
        {"x": {"x": 0.9}, "y": 0.99, "objectives": {"cost": 0.01}, "constraints": {"c": 0.9}},
    ])
    (tmp_path / "deliverable" / "progress.jsonl").write_text(
        "\n".join(json.dumps(row) for row in [
            {"round": 2, "n_front": 2, "hypervolume": 70.0, "n_feasible": 2,
             "hypervolume_feasible": 30.0, "feasible_fraction": 0.67},
            {"round": 3, "n_front": 3, "hypervolume": 80.0, "n_feasible": 2,
             "hypervolume_feasible": 35.0, "feasible_fraction": 0.67},
        ]) + "\n",
        encoding="utf-8",
    )
    report = study.report()
    assert "可交付前沿" in report
    assert "可行 HV" in report
    assert "可行 HV 30 → 35" in report

