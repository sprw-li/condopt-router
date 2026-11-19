"""Doc-count drift check: numbers quoted in docs must match the code.

Numbers of this kind rot silently (README once said "15/15 gates" while the real
number was 18).  This scans the user-facing docs for the countable phrases we care
about and compares them with live counts.

Run:
    py -3 scripts/check_doc_counts.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

def _utf8_child_env() -> dict:
    """Environment for child processes so their stdio is UTF-8, not the codepage."""
    import os

    return {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DOCS = [
    ROOT / "README.md",
    ROOT / "RELEASE_NOTES.md",
    ROOT / "docs" / "STATUS.md",
    ROOT / "docs" / "ARCHITECTURE.md",
    ROOT / "docs" / "ROUTING_POLICY.md",
    ROOT / "docs" / "ALGORITHM_CATALOG.md",
    ROOT / "research" / "EVIDENCE_POLICY.md",
]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def live_counts() -> dict[str, int]:
    from condopt import registry

    checks = _load("release_check_for_counts", ROOT / "scripts" / "release_check.py")
    scenarios = _load("scenario_matrix_for_counts", ROOT / "scripts" / "scenario_matrix.py")
    decision = _load("decision_matrix_for_counts", ROOT / "scripts" / "decision_matrix.py")
    evidence = _load("verify_evidence_for_counts", ROOT / "scripts" / "verify_evidence.py")
    negative = (ROOT / "research" / "NEGATIVE_RESULTS.md").read_text(encoding="utf-8", errors="replace")
    gates = set(
        re.findall(
            r'record\(\s*\n?\s*"([a-z0-9_]+)"',
            (ROOT / "scripts" / "release_check.py").read_text(encoding="utf-8", errors="replace"),
        )
    )
    return {
        "algorithms": len(registry.names()),
        "gates": len(gates),
        "scenarios": len(scenarios.SCENARIOS),
        "decision_checks": len(decision.RULES) + len(decision.evaluate_suppressions()),
        "claims": len(evidence.CLAIMS),
        "negatives": len(re.findall(r"^## \d+\.", negative, re.MULTILINE)),
    }


#: phrase -> live key (the number must equal the live count on every occurrence)
PHRASES = [
    (r"(\d+)\s*个优化算法", "algorithms"),
    (r"(\d+)\s*个\s*ask/tell", "algorithms"),
    (r"(\d+)\s*项门禁", "gates"),
    (r"(\d+)\s*个场景断言", "scenarios"),
    (r"(\d+)\s*条决策检查", "decision_checks"),
    (r"(\d+)\s*条可重跑声明", "claims"),
    (r"(\d+)\s*条负结果", "negatives"),
]


def check() -> dict:
    live = live_counts()
    problems: list[str] = []
    checked = 0
    for path in DOCS:
        if not path.exists():
            continue
        checked += 1
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if "历史" in line or "曾经" in line or "旧" in line:
                continue  # historical statements are allowed to quote old numbers
            for pattern, key in PHRASES:
                for match in re.finditer(pattern, line):
                    value = int(match.group(1))
                    if value != live[key]:
                        problems.append(
                            f"{path.relative_to(ROOT)}:{number}: "
                            f"{match.group(0)!r} (live {key}={live[key]})"
                        )
    return {"docs": checked, "counts": live, "problems": problems, "ok": not problems}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = check()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for problem in report["problems"]:
            print("PROBLEM", problem)
        print(
            f"doc_counts: docs={report['docs']} "
            f"problems={len(report['problems'])}"
        )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())