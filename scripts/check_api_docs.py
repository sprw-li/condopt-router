"""Check that the public ``Study`` API is documented somewhere.

The skill and docs are the only way an agent learns the API, so a public method
that no document mentions is effectively invisible.  A curated core list must
be documented; the rest is reported for information.

Run:
    py -3 scripts/check_api_docs.py
    py -3 scripts/check_api_docs.py --json
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt.study import Study  # noqa: E402

DOC_GLOBS = ("README.md", "docs/*.md", "research/*.md", "skill/**/*.md")

CORE_API = (
    "create",
    "load",
    "ask",
    "tell",
    "next_action",
    "recommend",
    "best",
    "summary",
    "report",
    "predict",
    "import_observations",
    "transfer_from",
    "assess_transfer",
    "simulate_transfer",
    "retune_for_exploration",
    "check_integrity",
    "fidelity_summary",
    "pareto_summary",
    "plate_effects",
    "cost_spent",
)


def _docs_text() -> str:
    parts = []
    for pattern in DOC_GLOBS:
        for path in ROOT.glob(pattern):
            if path.is_file():
                parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def check() -> dict:
    public = sorted(
        name
        for name, value in inspect.getmembers(Study, predicate=inspect.isfunction)
        if not name.startswith("_")
    )
    text = _docs_text()
    documented = [name for name in public if name in text]
    missing = [name for name in public if name not in text]
    core_missing = [name for name in CORE_API if name not in text]
    return {
        "n_public": len(public),
        "documented": documented,
        "missing": missing,
        "core_missing": core_missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = check()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for name in report["core_missing"]:
            print(f"UNDOCUMENTED CORE API: Study.{name}")
        if report["missing"]:
            print("undocumented (non-core): " + ", ".join(report["missing"]))
        print(
            f"api_docs: {len(report['documented'])}/{report['n_public']} public methods mentioned, "
            f"core gaps={len(report['core_missing'])}"
        )
    return 1 if report["core_missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())