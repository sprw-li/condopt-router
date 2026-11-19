"""Golden-number gate with a negative control (backlog A3).

The independent review found that ``docs/DEFAULTS.md`` and
``docs/DEFAULT_EVIDENCE.json`` could quote two different numbers for the same claim
(``-18.1`` vs ``-2.6995``) with every gate still green.  Free-text prose cannot be
compared automatically, so this gate works from a machine-readable registry
(``docs/GOLDEN_NUMBERS.json``) and checks two things:

1. **presence**: every registered number really appears on the hinted line of the
   document it claims -- so the archive cannot quietly drift away from the value we
   ship; and
2. **negative control** (``--self-test``): the checker is fed deliberately corrupted
   copies of those lines and *must* flag every one.  A gate that cannot go red on
   demand proves nothing.

Optionally (``--claims``) the linked ``verify_evidence`` claim is executed and its
number compared with the golden value, which is the strongest form of the check.

Run:
    py -3 scripts/check_golden_numbers.py
    py -3 scripts/check_golden_numbers.py --self-test
    py -3 scripts/check_golden_numbers.py --claims
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs" / "GOLDEN_NUMBERS.json"


def _load_registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def _number_on_line(line: str, number_text: str) -> bool:
    """True when the *number* (not an arbitrary substring) is present."""
    escaped = re.escape(number_text)
    # accept the plain form and both minus-sign spellings
    patterns = [rf"(?<![\d.]){escaped}(?![\d])"]
    if number_text.startswith("-"):
        patterns.append(rf"(?<![\d.]){re.escape(number_text[1:])}(?![\d])")
    return any(re.search(pattern, line) for pattern in patterns)


def _find_line(entry: dict, doc_text: str | None = None) -> str | None:
    text = doc_text
    if text is None:
        path = ROOT / entry["doc"]
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if entry["line_hint"] in line:
            return line
    return None


def check(registry: dict | None = None, docs: dict[str, str] | None = None) -> dict:
    """Return ``{"checked", "problems"}``; ``docs`` overrides file contents (for tests)."""
    registry = registry or _load_registry()
    problems: list[str] = []
    checked = 0
    for entry in registry["entries"]:
        checked += 1
        doc_text = (docs or {}).get(entry["doc"])
        line = _find_line(entry, doc_text)
        if line is None:
            if doc_text is None and not (ROOT / entry["doc"]).exists():
                problems.append(f"{entry['id']}: missing document {entry['doc']}")
            else:
                problems.append(
                    f"{entry['id']}: no line containing {entry['line_hint']!r} in {entry['doc']}"
                )
            continue
        if not _number_on_line(line, entry["number_text"]):
            problems.append(
                f"{entry['id']}: {entry['doc']} line for {entry['line_hint']!r} does not "
                f"contain {entry['number_text']!r}"
            )
        if entry.get("claim"):
            claim_path = ROOT / "scripts" / "verify_evidence.py"
            if not claim_path.exists():
                problems.append(f"{entry['id']}: claim {entry['claim']} cannot be resolved")
    return {"checked": checked, "problems": problems, "ok": not problems}


def self_test() -> dict:
    """Corrupt each registered number and require the checker to notice."""
    registry = _load_registry()
    failures: list[str] = []
    for entry in registry["entries"]:
        path = ROOT / entry["doc"]
        if not path.exists():
            failures.append(f"{entry['id']}: document missing, cannot self-test")
            continue
        original = path.read_text(encoding="utf-8")
        lines = original.splitlines(keepends=True)
        corrupted = []
        for line in lines:
            if entry["line_hint"] in line and _number_on_line(line, entry["number_text"]):
                corrupted.append(
                    line.replace(entry["number_text"], "999999", 1)
                )
            else:
                corrupted.append(line)
        if "".join(corrupted) == original:
            failures.append(
                f"{entry['id']}: could not build a corrupted variant "
                f"(hint {entry['line_hint']!r} / number {entry['number_text']!r})"
            )
            continue
        result = check(registry, {entry["doc"]: "".join(corrupted)})
        if result["ok"]:
            failures.append(
                f"{entry['id']}: checker accepted a corrupted number -- it cannot go red"
            )
    return {
        "entries": len(registry["entries"]),
        "failures": failures,
        "ok": not failures,
    }


def claim_check(registry: dict | None = None) -> dict:
    """Execute each linked claim and compare its number with the golden value."""
    registry = registry or _load_registry()
    spec = importlib.util.spec_from_file_location(
        "verify_evidence", ROOT / "scripts" / "verify_evidence.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["verify_evidence"] = module
    spec.loader.exec_module(module)

    by_name = {claim.name: claim for claim in module.CLAIMS}
    problems: list[str] = []
    executed = 0
    for entry in registry["entries"]:
        name = entry.get("claim")
        if not name:
            continue
        claim = by_name.get(name)
        if claim is None:
            problems.append(f"{entry['id']}: unknown claim {name!r}")
            continue
        row = module.evaluate_claim(claim)
        executed += 1
        if not row["ok"]:
            problems.append(f"{entry['id']}: claim {name} failed ({row.get('detail')})")
            continue
        detail = str(row.get("detail", ""))
        numbers = [
            float(match) for match in re.findall(r"-?\d+\.\d+", detail)
        ]
        target = float(entry["value"])
        tolerance = float(entry["tolerance"])
        if numbers and not any(abs(value - target) <= tolerance for value in numbers):
            problems.append(
                f"{entry['id']}: claim {name} reported {numbers}, expected "
                f"{target} ± {tolerance}"
            )
    return {"executed": executed, "problems": problems, "ok": not problems}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--claims", action="store_true")
    args = parser.parse_args()

    report: dict = {}
    if args.self_test:
        report["self_test"] = self_test()
    if args.claims:
        report["claims"] = claim_check()
    report["presence"] = check()

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        presence = report["presence"]
        for problem in presence["problems"]:
            print("PROBLEM", problem)
        print(
            f"golden_numbers: entries={presence['checked']} "
            f"problems={len(presence['problems'])}"
        )
        if "self_test" in report:
            result = report["self_test"]
            for failure in result["failures"]:
                print("PROBLEM", failure)
            print(
                f"golden_numbers self-test: entries={result['entries']} "
                f"failures={len(result['failures'])}"
            )
        if "claims" in report:
            result = report["claims"]
            for problem in result["problems"]:
                print("PROBLEM", problem)
            print(f"golden_numbers claims: executed={result['executed']}")

    ok = report["presence"]["ok"]
    ok = ok and report.get("self_test", {"ok": True})["ok"]
    ok = ok and report.get("claims", {"ok": True})["ok"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
