"""Every default must cite evidence that exists, and default *changes* need breadth.

Two rules from ``research/EVIDENCE_POLICY.md`` are machine-checkable:

1. each row of ``docs/DEFAULTS.md`` references at least one existing file;
2. the evidence table does not cite a file that has been deleted/renamed.

Run:
    py -3 scripts/check_evidence_policy.py
    py -3 scripts/check_evidence_policy.py --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = ROOT / "docs" / "DEFAULTS.md"
POLICY = ROOT / "research" / "EVIDENCE_POLICY.md"

#: paths that are referenced but are not expected to exist (external checkouts)
EXTERNAL_PREFIXES = ("http://", "https://")


def _rows() -> list[tuple[int, str]]:
    rows = []
    for number, line in enumerate(DEFAULTS.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("|") and not re.match(r"^\|[\s:-]+\|", stripped):
            rows.append((number, stripped))
    return rows


REGISTRY = ROOT / "docs" / "DEFAULT_EVIDENCE.json"


def check_benchmark_hygiene() -> dict:
    """Benchmark loops must scale with the budget (see EVIDENCE_POLICY.md §7)."""
    spec = _load_script("check_benchmark_hygiene", ROOT / "scripts" / "check_benchmark_hygiene.py")
    return spec.check()


def _load_script(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def current_round() -> int:
    """Latest round number recorded in the ralph log (0 when unknown)."""
    log = ROOT / "research" / "ralph" / "LOG.md"
    if not log.exists():
        return 0
    numbers = [
        int(match.group(1))
        for match in re.finditer(
            r"^#+ Round (\d+)", log.read_text(encoding="utf-8"), re.MULTILINE
        )
    ]
    return max(numbers) if numbers else 0


def check_claim_links(registry_keys: set[str] | None = None) -> dict:
    """Every re-runnable claim must point at a registered decision."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "verify_evidence_for_policy", ROOT / "scripts" / "verify_evidence.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    problems: list[str] = []
    links: dict[str, list[str]] = {}
    for claim in module.CLAIMS:
        decisions = [
            part.strip()
            for part in str(getattr(claim, "decision", "") or "").split(",")
            if part.strip()
        ]
        if not decisions:
            problems.append(f"claim {claim.name}: no decision key")
            continue
        for decision in decisions:
            links.setdefault(decision, []).append(claim.name)
            if registry_keys is not None and decision not in registry_keys:
                problems.append(f"claim {claim.name}: unknown decision {decision!r}")
    if registry_keys is not None:
        for key in sorted(registry_keys):
            if key not in links:
                problems.append(f"decision {key}: no re-runnable claim backs it")
    return {
        "claims": len(module.CLAIMS),
        "linked": sum(len(names) for names in links.values()),
        "problems": problems,
    }


def check_registry() -> dict:
    """Validate docs/DEFAULT_EVIDENCE.json against the two-dataset rule."""
    if not REGISTRY.exists():
        return {"entries": 0, "problems": ["docs/DEFAULT_EVIDENCE.json is missing"]}
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    problems: list[str] = []
    warnings: list[str] = []
    round_number = current_round()
    entries = payload.get("decisions", [])
    if not isinstance(entries, list) or not entries:
        problems.append("registry has no decisions")
        entries = []
    for entry in entries:
        key = entry.get("key", "?")
        datasets = entry.get("datasets") or []
        status = entry.get("status")
        if status not in ("change", "keep"):
            problems.append(f"{key}: status must be change/keep")
        if not datasets:
            problems.append(f"{key}: no datasets recorded")
        for dataset in datasets:
            if not str(dataset.get("name", "")).strip():
                problems.append(f"{key}: a dataset has no name")
            if not str(dataset.get("result", "")).strip():
                problems.append(f"{key}: dataset {dataset.get('name')} has no result")
        consistent = bool(entry.get("direction_consistent"))
        if status == "change" and (len(datasets) < 2 or not consistent):
            problems.append(
                f"{key}: a *change* needs >=2 datasets with a consistent direction"
            )
        if status == "keep" and not consistent and not str(entry.get("tradeoff", "")).strip():
            problems.append(f"{key}: inconsistent evidence kept must document a tradeoff")
        for path in entry.get("evidence", []) or []:
            if not (ROOT / str(path)).exists():
                problems.append(f"{key}: evidence file missing -> {path}")
        reviewed = entry.get("last_reviewed_round")
        every = int(entry.get("review_every_rounds", 0) or 0)
        if reviewed is None or every <= 0:
            warnings.append(f"{key}: review metadata missing")
        elif round_number and round_number - int(reviewed) > every:
            warnings.append(
                f"{key}: 已 {round_number - int(reviewed)} 轮未复核"
                f"（周期 {every} 轮，上次第 {reviewed} 轮）"
            )
    return {
        "entries": len(entries),
        "changes": sum(1 for entry in entries if entry.get("status") == "change"),
        "keeps": sum(1 for entry in entries if entry.get("status") == "keep"),
        "current_round": round_number,
        "warnings": warnings,
        "problems": problems,
    }


def check() -> dict:
    problems: list[str] = []
    citations = 0
    rows = _rows()
    for number, row in rows:
        if "实测依据" in row or "证据文件" in row:
            continue  # header row of any provenance table
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        if len(cells) < 4:
            continue  # not a provenance row
        evidence_cell = cells[-1]
        paths = re.findall(r"`([^`]+)`", evidence_cell)
        paths = [p for p in paths if not p.startswith(EXTERNAL_PREFIXES)]
        if not paths:
            problems.append(f"line {number}: no evidence path cited")
            continue
        citations += len(paths)
        for candidate in paths:
            target = ROOT / candidate.split("#")[0].split(" ")[0]
            if not target.exists():
                problems.append(f"line {number}: cited file missing -> {candidate}")

    if not POLICY.exists():
        problems.append("research/EVIDENCE_POLICY.md is missing")
    text = DEFAULTS.read_text(encoding="utf-8")
    for required in ("实测依据", "证据文件"):
        if required not in text:
            problems.append(f"docs/DEFAULTS.md lost its '{required}' column")
    hygiene = check_benchmark_hygiene()
    problems.extend(hygiene["problems"])
    doc_counts = _load_script(
        "check_doc_counts", ROOT / "scripts" / "check_doc_counts.py"
    ).check()
    problems.extend(doc_counts["problems"])
    registry = check_registry()
    problems.extend(registry["problems"])
    payload = json.loads(REGISTRY.read_text(encoding="utf-8")) if REGISTRY.exists() else {}
    registry_keys = {entry.get("key") for entry in payload.get("decisions", [])}
    claims = check_claim_links(registry_keys)
    problems.extend(claims["problems"])
    warnings = list(registry.get("warnings", []))
    return {
        "rows": len(rows),
        "citations": citations,
        "registry_entries": registry["entries"],
        "registry_changes": registry.get("changes", 0),
        "registry_keeps": registry.get("keeps", 0),
        "claims": claims["claims"],
        "benchmark_scripts": hygiene["checked"],
        "docs_checked": doc_counts["docs"],
        "current_round": registry.get("current_round", 0),
        "warnings": warnings,
        "problems": problems,
        "ok": not problems,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--strict", action="store_true",
        help="treat stale review metadata as a failure too",
    )
    args = parser.parse_args()
    report = check()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for problem in report["problems"]:
            print("PROBLEM", problem)
        for warning in report["warnings"]:
            print("WARN", warning)
        print(
            f"evidence_policy: rows={report['rows']} citations={report['citations']} "
            f"registry={report['registry_entries']} "
            f"(change={report['registry_changes']} keep={report['registry_keeps']}) "
            f"claims_linked={report['claims']} "
            f"benchmarks={report['benchmark_scripts']} docs={report['docs_checked']} "
            f"round={report.get('current_round')} "
            f"warnings={len(report['warnings'])} problems={len(report['problems'])}"
        )
    if not report["ok"]:
        return 1
    if args.strict and report["warnings"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())