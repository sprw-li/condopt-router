"""Negative-results register: the doc must stay alive and cover known lessons."""

from __future__ import annotations

from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "research" / "NEGATIVE_RESULTS.md"


def test_negative_results_doc_covers_known_findings():
    text = DOC.read_text(encoding="utf-8")
    for topic in (
        "cqnehvi",
        "Sobol",
        "mqehvi_nd",
        "TPE",
        "Spearman",
        "hypervolume_contribution",
    ):
        assert topic in text, f"negative-results doc lost the {topic!r} entry"
    assert text.count("\n## ") >= 7

def test_skill_and_docs_do_not_duplicate_rules():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "check_doc_duplication.py"
    spec = importlib.util.spec_from_file_location("check_doc_duplication", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    matches = module.find_duplicates(0.75)
    assert len(matches) <= 2, f"skill/docs rule duplication: {matches}"

def test_negative_results_index_lists_every_entry_in_order():
    import re
    from pathlib import Path

    doc = Path(__file__).resolve().parents[1] / "research" / "NEGATIVE_RESULTS.md"
    text = doc.read_text(encoding="utf-8")
    assert "## 索引" in text
    headings = [int(m.group(1)) for m in re.finditer(r"^## (\d+)\.", text, re.MULTILINE)]
    assert headings == sorted(headings), f"entries out of order: {headings}"
    index_block = text.split("## 索引", 1)[1].split("\n## ", 1)[0]
    for number in headings:
        assert f"| {number} |" in index_block, f"index lost entry {number}"
    assert len(headings) >= 11

def test_negative_results_index_is_regenerable():
    """The index is generated; running the builder must be a no-op."""
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "build_negative_index.py"
    spec = importlib.util.spec_from_file_location("build_negative_index", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    current = (root / "research" / "NEGATIVE_RESULTS.md").read_text(encoding="utf-8")
    assert current.strip() == module.build().strip()


def test_negative_results_has_exactly_one_index_block():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "research" / "NEGATIVE_RESULTS.md").read_text(
        encoding="utf-8"
    )
    assert text.count("## 索引") == 1

