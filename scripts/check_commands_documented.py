"""Check that every CLI subcommand is documented in README and the skill.

The command list is user-facing API: if a subcommand exists but the docs never
mention it (or vice versa after a rename), agents cannot discover it.

Run:
    py -3 scripts/check_commands_documented.py
    py -3 scripts/check_commands_documented.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt.cli import build_parser  # noqa: E402

DOCS = ("README.md", "skill/SKILL.md")


def subcommands() -> list[str]:
    parser = build_parser()
    for action in parser._actions:  # type: ignore[attr-defined]
        choices = getattr(action, "choices", None)
        if choices and getattr(action, "dest", "") == "command":
            return sorted(choices)
    return []


def check() -> dict:
    commands = subcommands()
    missing: dict[str, list[str]] = {}
    for relative in DOCS:
        path = ROOT / relative
        if not path.exists():
            missing[relative] = commands
            continue
        text = path.read_text(encoding="utf-8")
        absent = [name for name in commands if name not in text]
        if absent:
            missing[relative] = absent
    return {"commands": commands, "n_commands": len(commands), "missing": missing}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = check()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for relative, absent in report["missing"].items():
            for name in absent:
                print(f"UNDOCUMENTED {name} (not found in {relative})")
        print(
            f"commands_documented: {report['n_commands']} commands, "
            f"{len(report['missing'])} files with gaps"
        )
    return 1 if report["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())