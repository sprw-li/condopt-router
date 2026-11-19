"""The defaults-provenance table must stay complete and link to real evidence."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "DEFAULTS.md"


def test_defaults_doc_covers_key_decisions():
    text = DOC.read_text(encoding="utf-8")
    for token in (
        "constrained",
        "cqehvi",
        "qehvi_nd",
        "mfbo",
        "mfparego",
        "cmaes",
        "patience",
        "retune_for_exploration",
        "prior_reliability",
        "成本预算",
        "NEGATIVE_RESULTS",
    ):
        assert token in text, f"defaults doc lost the {token!r} row"
    # every referenced research/docs file must exist
    for line in text.splitlines():
        if "research/" not in line and "docs/" not in line:
            continue
        for chunk in line.replace("`", " ").split():
            candidate = chunk.strip(" |,()")
            if candidate.endswith(".md"):
                assert (ROOT / candidate).exists(), f"missing evidence file {candidate}"

def test_documented_defaults_match_code():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "check_defaults_sync.py"
    spec = importlib.util.spec_from_file_location("check_defaults_sync", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    report = module.check()
    assert report["mismatches"] == [], report["mismatches"]
    assert report["spec_defaults_checked"] >= 8
    assert report["route_results"]["mf+2obj"] == "mfparego"

