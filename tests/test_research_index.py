"""The research index must list every note in ``research/``."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_research_index.py"
_spec = importlib.util.spec_from_file_location("build_research_index", SCRIPT)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)


def test_research_index_lists_every_note():
    notes = sorted(
        path.name for path in (ROOT / "research").glob("*.md") if path.name != "README.md"
    )
    index = (ROOT / "research" / "README.md").read_text(encoding="utf-8")
    missing = [name for name in notes if f"]({name})" not in index]
    assert missing == [], f"research index is missing: {missing}"


def test_research_index_is_current():
    target = ROOT / "research" / "README.md"
    assert target.read_text(encoding="utf-8").strip() == _module.build().strip(), (
        "research/README.md is stale; run scripts/build_research_index.py"
    )

def test_research_index_overview_counts():
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "build_research_index.py"
    spec = importlib.util.spec_from_file_location("build_research_index", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    counts = module._counts()
    text = (root / "research" / "README.md").read_text(encoding="utf-8")
    assert "## 概览" in text
    overview = text.split("## 概览", 1)[1].split("## 核心结论", 1)[0]
    for key, value in counts.items():
        assert str(value) in overview, f"overview lost {key}={value}"
    assert f"活跃 {counts['active']}" in overview
    assert f"历史归档 {counts['archived']}" in overview
    assert counts["negative"] >= 11
    assert counts["claims"] >= 7
    assert counts["scenarios"] >= 19
    assert counts["archived"] >= 10
    assert counts["active"] + counts["archived"] == counts["notes"]
    assert "历史归档（不再维护）" in text

