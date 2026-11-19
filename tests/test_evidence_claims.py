"""Evidence claims stay runnable and self-describing (no heavy benchmarks in CI)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_evidence.py"
_spec = importlib.util.spec_from_file_location("verify_evidence", SCRIPT)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)


def test_evidence_claims_reference_existing_scripts():
    claims = _module.CLAIMS
    assert len(claims) >= 7
    for claim in claims:
        script = Path(claim.command[1])
        assert (ROOT / script).exists(), f"{claim.name}: missing {script}"
        assert claim.description, claim.name
        assert claim.check, claim.name
        # Benchmark claims must pin seeds for reproducibility; example-driven
        # claims (like the multi-objective executor demo) need not.
        if "benchmark" in script.name:
            assert any(flag.startswith("--seeds") for flag in claim.command), claim.name


def test_evidence_audit_doc_lists_claims():
    text = (ROOT / "research" / "EVIDENCE_AUDIT.md").read_text(encoding="utf-8")
    for claim in _module.CLAIMS:
        assert claim.name in text, f"EVIDENCE_AUDIT.md missing {claim.name}"

def test_every_claim_maps_to_a_registered_decision():
    import importlib.util
    import json as _json
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "check_evidence_policy_links", root / "scripts" / "check_evidence_policy.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    payload = _json.loads(
        (root / "docs" / "DEFAULT_EVIDENCE.json").read_text(encoding="utf-8")
    )
    keys = {entry["key"] for entry in payload["decisions"]}
    report = module.check_claim_links(keys)
    assert report["problems"] == [], report["problems"]
    assert report["claims"] >= 10
    # one claim may back several decisions, so links can exceed the claim count
    assert report["linked"] >= report["claims"]

