"""Release-gate helpers: the README test count must not drift from reality."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "release_check.py"
_spec = importlib.util.spec_from_file_location("release_check", SCRIPT)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)


def test_readme_test_count_matches_collected_tests():
    collected = _module.collected_test_count()
    declared = _module.readme_test_count()
    assert collected > 0
    assert declared == collected, (
        f"README declares {declared} tests but pytest collects {collected}"
    )


def test_tail_helper_keeps_last_lines():
    assert _module._tail("a\nb\nc\n", lines=2) == "b | c"

def test_docs_links_are_not_broken():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "check_docs_links.py"
    spec = importlib.util.spec_from_file_location("check_docs_links", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    broken = module.check()
    assert broken == [], f"broken markdown links: {broken}"

def test_e2e_drill_passes():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "e2e_drill.py"
    spec = importlib.util.spec_from_file_location("e2e_drill", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    report = module.run_drill("sphere2d", target_budget=10)
    assert report["ok"] is True, report
    assert report["integrity_ok"] is True
    assert report["pending_left"] == 0
    assert report["duplicate_ids"] == 0
    assert report["digest_written"] is True

def test_all_cli_commands_are_documented():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "check_commands_documented.py"
    spec = importlib.util.spec_from_file_location("check_commands_documented", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    report = module.check()
    assert report["n_commands"] >= 25
    assert report["missing"] == {}, report["missing"]

def test_core_public_api_is_documented():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "check_api_docs.py"
    spec = importlib.util.spec_from_file_location("check_api_docs", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    report = module.check()
    assert report["core_missing"] == [], report["core_missing"]
    assert report["n_public"] >= 25

def test_ab_router_weights_script_runs(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    from condopt.calibration import RunRecord, set_algorithm_weights

    records_path = tmp_path / "records.json"
    records_path.write_text(
        json.dumps([
            {"problem": "p1", "algorithm": "bayes", "score": 1.0,
             "context": {"n_dims": 2, "n_continuous": 2, "budget_total": 10}, "metric_name": "true_best"},
            {"problem": "p1", "algorithm": "random", "score": 0.5,
             "context": {"n_dims": 2, "n_continuous": 2, "budget_total": 10}, "metric_name": "true_best"},
            {"problem": "p2", "algorithm": "random", "score": 2.0,
             "context": {"n_dims": 2, "n_continuous": 2, "budget_total": 10}, "metric_name": "true_best"},
            {"problem": "p2", "algorithm": "bayes", "score": 1.0,
             "context": {"n_dims": 2, "n_continuous": 2, "budget_total": 10}, "metric_name": "true_best"},
        ]), encoding="utf-8")
    weights_path = tmp_path / "weights.json"
    weights_path.write_text(json.dumps({
        "weights": {"bayes": 1.1, "random": 0.9},
        "lopo_metrics": {"top1": 0.5, "top2": 1.0, "mean_regret": 0.1, "n_problems": 2},
    }), encoding="utf-8")

    script = Path(__file__).resolve().parents[1] / "scripts" / "ab_router_weights.py"
    spec = importlib.util.spec_from_file_location("ab_router_weights", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    records = module._load_records(records_path)
    assert len(records) == 4
    lopo = module.default_lopo(records)
    assert lopo["n_problems"] == 2
    set_algorithm_weights(None)

def test_status_snapshot_is_current():
    """docs/STATUS.md must match what build_status.py generates."""
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "build_status.py"
    spec = importlib.util.spec_from_file_location("build_status", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    doc = root / "docs" / "STATUS.md"
    assert doc.exists(), "run scripts/build_status.py"
    assert doc.read_text(encoding="utf-8").strip() == module.build().strip()
    live = module.counts()
    assert live["algorithms"] >= 18
    assert live["scenarios"] >= 19
    assert live["claims"] >= 7
    assert live["negative"] >= 12

def test_skill_documents_the_end_to_end_workflow():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    skill = (root / "skill" / "SKILL.md").read_text(encoding="utf-8")
    for token in (
        "condopt prepare",
        "condopt init",
        "condopt step",
        "condopt digest",
        "deliverable",
        "delivery_checklist",
        # decision-layer policy must stay documented in the skill
        "budget_first",
        "effective_exploration_budget",
        "交付自查",
        "reliability_grade",
        "DECISIONS.md",
    ):
        assert token in skill, f"SKILL.md lost {token!r}"

def test_default_evidence_registry_satisfies_the_two_dataset_rule():
    import importlib.util
    import json as _json
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "check_evidence_policy.py"
    spec = importlib.util.spec_from_file_location("check_evidence_policy_registry", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    report = module.check_registry()
    assert report["problems"] == [], report["problems"]
    assert report["entries"] >= 5
    assert report["changes"] >= 1
    payload = _json.loads((root / "docs" / "DEFAULT_EVIDENCE.json").read_text(encoding="utf-8"))
    for entry in payload["decisions"]:
        if entry["status"] == "change":
            assert len(entry["datasets"]) >= 2, entry["key"]

def test_default_evidence_registry_tracks_reviews():
    import importlib.util
    import json as _json
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "check_evidence_policy.py"
    spec = importlib.util.spec_from_file_location("check_evidence_policy_review", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert module.current_round() >= 200  # parsed from research/ralph/LOG.md
    report = module.check_registry()
    assert report["problems"] == []
    assert report["current_round"] == module.current_round()
    payload = _json.loads(
        (root / "docs" / "DEFAULT_EVIDENCE.json").read_text(encoding="utf-8")
    )
    for entry in payload["decisions"]:
        assert entry["last_reviewed_round"] > 0, entry["key"]
        assert entry["review_every_rounds"] > 0, entry["key"]

def test_evidence_policy_documents_measurement_artefacts():
    """The policy must keep the "check the measurement itself" section."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    text = (root / "research" / "EVIDENCE_POLICY.md").read_text(encoding="utf-8")
    assert "先检验测量本身" in text
    # the three retracted conclusions are named so readers do not re-use them
    for token in ("2/5", "range(60)", "推荐条件"):
        assert token in text, f"policy lost the case: {token}"

def test_benchmark_loops_scale_with_the_budget():
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "check_benchmark_hygiene.py"
    spec = importlib.util.spec_from_file_location("check_benchmark_hygiene", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    report = module.check()
    assert report["problems"] == [], report["problems"]
    assert report["checked"] >= 5

def test_documented_skill_workflow_actually_runs(tmp_path):
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "verify_skill_workflow.py"
    spec = importlib.util.spec_from_file_location("verify_skill_workflow", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    report = module.run_workflow(tmp_path / "skill_flow")
    assert report["ok"] is True, report["problems"]
    assert report["suggestions"] == 3
    # the spec-window leg must run too (prepare -> init --spec -> recommend)
    assert report["spec_suggestions"] == 3
    # multi-objective + constrained leg (Pareto / deliverable sections in the digest)
    assert report["moo_suggestions"] == 3
    # multi-fidelity leg (fidelity per suggestion, digest keeps the fidelity info)
    assert report["mf_suggestions"] == 3
    # transfer leg (audit + impact + import --from-run all return usable payloads)
    assert report["imported"] >= 1

def test_algorithm_reference_agrees_with_the_router():
    """skill/references/ALGORITHM_CHOICE.md must not contradict the router."""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from condopt.router import OptimizationContext, route

    reference = (root / "skill" / "references" / "ALGORITHM_CHOICE.md").read_text(
        encoding="utf-8"
    )
    # multi-fidelity + 2 objectives: the router picks mfparego; the doc used to say momfbo
    context = OptimizationContext(
        n_dims=4, n_continuous=4, budget_total=40, batch_size=4,
        n_objectives=2, multi_fidelity=True, noise="medium",
    )
    chosen = route(context).algorithm
    assert chosen == "mfparego"
    assert f"默认 `{chosen}`" in reference, f"reference lost the default {chosen}"
    assert "router 仍默认 momfbo" not in reference, "stale default in the reference"
    # cmaes caveat and the tiny-budget wording must stay
    assert "地貌依赖" in reference
    assert "建模下限" in reference

def test_parameter_guide_agrees_with_the_defaults():
    """The parameter reference must quote the same defaults the code uses."""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from condopt.prepare import stopping_advice
    from condopt.router import OptimizationContext
    from condopt.study import default_confirm_n

    guide = (root / "skill" / "references" / "PARAMETER_GUIDE.md").read_text(
        encoding="utf-8"
    )
    # stopping threshold and the confirmation default are the two most load-bearing numbers
    assert stopping_advice(OptimizationContext(n_dims=4, n_continuous=4, budget_total=120))[
        "recommended"
    ] == "budget_first"
    assert "≥60" in guide
    assert default_confirm_n(12) == 3
    assert "confirm_n" in guide
    assert "effective_exploration_budget" in guide
    assert "underpowered" in guide

def test_release_notes_match_the_generator():
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "build_release_notes.py"
    spec = importlib.util.spec_from_file_location("build_release_notes", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    doc = root / "RELEASE_NOTES.md"
    assert doc.exists(), "run scripts/build_release_notes.py"
    assert doc.read_text(encoding="utf-8").strip() == module.build().strip()
    live = module.counts()
    assert live["algorithms"] >= 18
    assert live["gates"] >= 17
    assert live["tests"] >= 400

def test_readme_has_no_hard_coded_gate_count():
    """Gate counts belong to docs/STATUS.md (generated); README must not drift."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "release_check.py" in readme
    stale = re.findall(r"\b1[0-9]/1[0-9]\b", readme)
    assert not stale, f"README carries a hard-coded gate count: {stale}"

def test_doc_counts_match_the_code():
    """Numbers quoted in the user-facing docs must equal the live counts."""
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "check_doc_counts.py"
    spec = importlib.util.spec_from_file_location("check_doc_counts", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    report = module.check()
    assert report["problems"] == [], report["problems"]
    assert report["counts"]["algorithms"] >= 18
    assert report["counts"]["gates"] >= 18

def test_user_docs_state_the_descriptor_pairing_rule():
    """Encoding guidance must not be readable as "descriptors improve the optimum"."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    catalog = (root / "docs" / "ALGORITHM_CATALOG.md").read_text(encoding="utf-8")
    routing = (root / "docs" / "ROUTING_POLICY.md").read_text(encoding="utf-8")
    for name, text in (("ALGORITHM_CATALOG.md", catalog), ("ROUTING_POLICY.md", routing)):
        assert "描述符" in text, name
        assert "覆盖" in text, f"{name} lost the coverage-not-quality framing"
        assert "约束感知" in text, f"{name} lost the required pairing with a constraint-aware optimizer"

def test_readme_version_matches_the_package_metadata():
    """One source of truth: the version in pyproject.toml."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M)
    assert match is not None, "pyproject.toml lost its version"
    declared = match.group(1)
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert f"（v{declared}）" in readme, (
        f"README must state the package version v{declared}"
    )
    stale = [
        found for found in re.findall(r"（v(\d+\.\d+)）", readme)
        if found != declared
    ]
    assert stale == [], f"README carries stale versions: {stale}"

