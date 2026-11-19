"""Naming precision: literature names must not overstate what is implemented.

An external review (2026-11-11) showed the catalog implied literature equivalence for
algorithms that are simplified in this project.  These assertions keep the boundary
statements in place.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_catalog_states_the_implementation_boundary():
    catalog = (ROOT / "docs" / "ALGORITHM_CATALOG.md").read_text(encoding="utf-8")
    assert "实现边界与命名精确性" in catalog
    for token in (
        "无 tree/条件密度",
        "非联合 q 点 EHVI",
        "无 evolution paths",
        "模型后验",
    ):
        assert token in catalog, f"catalog lost the boundary statement: {token}"


def test_tpe_documents_that_it_is_independent_per_dimension():
    source = (ROOT / "src" / "condopt" / "algorithms" / "tpe.py").read_text(encoding="utf-8")
    catalog = (ROOT / "docs" / "ALGORITHM_CATALOG.md").read_text(encoding="utf-8")
    assert "independent" in source.lower(), "tpe.py must state it is independent per-dimension"
    assert "独立逐维" in catalog, "catalog must not call tpe tree-structured without qualification"


def test_readme_algorithm_bullets_carry_their_qualifier():
    """Each algorithm bullet in the README must state what it is *not*.

    (Checking for the absence of a phrase is too blunt: "非完整 CMA-ES" is correct
    usage.  So the check requires the qualifier on the same bullet line.)
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    requirements = {
        "cmaes": "无 evolution paths",
        "tpe": "独立逐维",
        "qehvi": "单点候选",
        "qnehvi": "带噪观测",
    }
    lines = readme.splitlines()
    for key, qualifier in requirements.items():
        prefix = f"`{key}`"
        line = next(
            (
                item for item in lines
                if item.lstrip().startswith(f"- {prefix}")
            ),
            None,
        )
        assert line is not None, f"README lost the {key} bullet"
        assert qualifier in line, (
            f"README bullet for {key} must carry the qualifier {qualifier!r}: {line[:120]}"
        )
