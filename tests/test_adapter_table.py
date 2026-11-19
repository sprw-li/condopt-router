"""The generated long-run table stays structurally correct."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_adapter_table.py"
_spec = importlib.util.spec_from_file_location("build_adapter_table", SCRIPT)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)


def test_adapter_table_doc_structure():
    doc = (ROOT / "research" / "ADAPTER_LONG_RUN.md").read_text(encoding="utf-8")
    assert _module.START in doc and _module.END in doc
    block = doc.split(_module.START, 1)[1].split(_module.END, 1)[0]
    for column in _module.COLUMNS:
        assert column in block, f"table lost the {column!r} column"
    for label, problem, _budget, _batch in _module.CONFIGS:
        assert label in block, f"table lost the {label!r} row"
        assert f"`{problem}`" in block, f"table lost the {problem!r} problem"


def test_adapter_table_render_is_well_formed():
    rows = [
        {
            "label": "demo",
            "problem": "mf_constrained",
            "algorithm": "mfbo",
            "n_observations": 16,
            "attempts": 18,
            "failures": 2,
            "cost_spent": 10.0,
            "wall_seconds": 1.5,
            "n_feasible_full_fidelity": 5,
            "feasible_hypervolume_curve": None,
            "duplicate_suggestion_ids": 0,
        },
        {
            "label": "demo-moo",
            "problem": "mf_moo4",
            "algorithm": "mfparego",
            "n_observations": 16,
            "attempts": 16,
            "failures": 0,
            "cost_spent": 9.0,
            "wall_seconds": 2.0,
            "n_feasible_full_fidelity": 8,
            "feasible_hypervolume_curve": [1.0, 2.0],
            "final_hypervolume": 3.0,
            "duplicate_suggestion_ids": 0,
        },
    ]
    table = _module.render(rows)
    assert table.startswith(_module.START)
    assert table.rstrip().endswith("口径检查）。")
    assert "1.00 → 2.00" in table
    assert "3.00" in table
    assert "—" in table  # missing hypervolume renders as a dash
