"""Aggregate literature labels + lessons into router experience.

Reads:
    research/literature_scan/labels/labels_*.jsonl   (written by subagents)
    research/literature_scan/lessons/lesson_*.json

Writes:
    research/literature_scan/ROUTER_EXPERIENCE.md
    research/literature_scan/router_experience.json

The output is deliberately *evidence first*: it reports label counts and
crosstabs, then a small number of candidate router rules with the supporting
label counts.  It does not silently change the router.

Usage:
    py -3 scripts/literature_experience_report.py --input research/literature_scan
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHOD_TO_ROUTER = {
    "BO": "bayes",
    "TPE": "tpe",
    "DoE": "lhs",
    "evolutionary": "random",
    "RL": "random",
}
SPACE_LABELS = ["categorical_heavy", "continuous_heavy", "mixed", "unknown"]
RECOMMENDATIONS = ["bayes", "tpe", "parego", "qehvi", "qnehvi", "constrained", "mfbo",
                   "sobol", "lhs", "grid", "random", "none"]


def _read_jsonl_dir(directory: Path) -> list[dict]:
    rows = []
    if not directory.exists():
        return rows
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append({**json.loads(line), "_file": path.name})
            except json.JSONDecodeError:
                continue
    return rows


def load_lessons(directory: Path) -> list[dict]:
    lessons = []
    if not directory.exists():
        return lessons
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        payload["_file"] = path.name
        lessons.append(payload)
    return lessons


def normalise(labels: list[dict]) -> list[dict]:
    seen = set()
    cleaned = []
    for row in labels:
        paper_id = str(row.get("paper_id") or "").strip()
        if not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        item = {
            "paper_id": paper_id,
            "space_type": row.get("space_type") if row.get("space_type") in SPACE_LABELS else "unknown",
            "problem_type": row.get("problem_type") or "other",
            "objectives": row.get("objectives") if row.get("objectives") in ("single", "multi", "unknown") else "unknown",
            "constraints": row.get("constraints") if isinstance(row.get("constraints"), bool) else None,
            "noisy": row.get("noisy") if isinstance(row.get("noisy"), bool) else None,
            "automation": row.get("automation") or "unknown",
            "literature_method": row.get("literature_method") or "unknown",
            "router_recommendation": row.get("router_recommendation") if row.get("router_recommendation") in RECOMMENDATIONS else "none",
            "confidence": row.get("confidence") if row.get("confidence") in ("low", "medium", "high") else "low",
            "categorical_count": _int_or_none(row.get("categorical_count")),
            "continuous_count": _int_or_none(row.get("continuous_count")),
            "evidence": str(row.get("evidence") or "")[:200],
            "rationale": str(row.get("rationale") or "")[:240],
            "_file": row.get("_file", ""),
        }
        cleaned.append(item)
    return cleaned


def _int_or_none(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def crosstab(labels: list[dict], key: str) -> dict[str, dict]:
    table: dict[str, Counter] = defaultdict(Counter)
    for row in labels:
        group = row.get(key)
        if isinstance(group, bool):
            group = "true" if group else "false"
        elif group is None:
            group = "unknown"
        table[str(group)][row["router_recommendation"]] += 1
    return {group: dict(counter.most_common()) for group, counter in table.items()}


def method_agreement(labels: list[dict]) -> dict:
    known = [row for row in labels if row["literature_method"] in METHOD_TO_ROUTER]
    if not known:
        return {"n": 0, "agreement": None}
    hits = 0
    for row in known:
        expected = METHOD_TO_ROUTER[row["literature_method"]]
        # DoE is a family: lhs/sobol/grid all count as agreement.
        if expected == "lhs" and row["router_recommendation"] in ("lhs", "sobol", "grid"):
            hits += 1
        elif row["router_recommendation"] == expected:
            hits += 1
    return {"n": len(known), "agreement": round(hits / len(known), 3)}


def proposals(labels: list[dict]) -> list[dict]:
    out = []
    high = [row for row in labels if row["confidence"] in ("medium", "high")]
    by_space = crosstab(high, "space_type")
    categorical = by_space.get("categorical_heavy", {})
    if categorical:
        top = max(categorical, key=categorical.get)
        total = sum(categorical.values())
        if total >= 5:
            actions = {
                "bayes": "raise bayes bonus / lower tpe bonus for categorical-heavy spaces",
                "tpe": "keep tpe preference for categorical-heavy spaces",
                "grid": "small factorial categorical spaces: prefer grid/DoE over model-based BO",
                "parego": "categorical-heavy multi-objective: prefer parego over single-objective BO",
            }
            out.append({
                "rule": f"categorical_heavy -> prefer {top}",
                "evidence_n": total,
                "support": categorical,
                "action": actions.get(top, "inspect: literature preference is mixed"),
                "confidence": "medium" if total >= 15 else "low",
            })
    continuous = by_space.get("continuous_heavy", {})
    if continuous:
        top = max(continuous, key=continuous.get)
        total = sum(continuous.values())
        if total >= 5:
            out.append({
                "rule": f"continuous_heavy -> prefer {top}",
                "evidence_n": total,
                "support": continuous,
                "action": "keep bayes/GP preference for continuous spaces" if top == "bayes" else "inspect",
                "confidence": "medium" if total >= 15 else "low",
            })
    by_constraints = crosstab(high, "constraints").get("true", {})
    if by_constraints:
        top = max(by_constraints, key=by_constraints.get)
        out.append({
            "rule": f"constraints -> prefer {top}",
            "evidence_n": sum(by_constraints.values()),
            "support": by_constraints,
            "action": "keep constrained routing (PoF/EI_c) / check if BO with penalty dominates",
            "confidence": "medium" if sum(by_constraints.values()) >= 10 else "low",
        })
    by_objectives = crosstab(high, "objectives").get("multi", {})
    if by_objectives:
        top = max(by_objectives, key=by_objectives.get)
        out.append({
            "rule": f"multi_objective -> prefer {top}",
            "evidence_n": sum(by_objectives.values()),
            "support": by_objectives,
            "action": "align multi-objective router (parego/qehvi/qnehvi) with literature practice",
            "confidence": "medium" if sum(by_objectives.values()) >= 10 else "low",
        })
    return out


def summarise(labels: list[dict], lessons: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "n_labels": len(labels),
        "n_files": len({row.get("_file") for row in labels if row.get("_file")}),
        "space_type": dict(Counter(row["space_type"] for row in labels).most_common()),
        "problem_type": dict(Counter(row["problem_type"] for row in labels).most_common()),
        "literature_method": dict(Counter(row["literature_method"] for row in labels).most_common()),
        "router_recommendation": dict(Counter(row["router_recommendation"] for row in labels).most_common()),
        "automation": dict(Counter(row["automation"] for row in labels).most_common()),
        "method_agreement": method_agreement(labels),
        "crosstabs": {
            "space_type": crosstab(labels, "space_type"),
            "problem_type": crosstab(labels, "problem_type"),
            "objectives": crosstab(labels, "objectives"),
            "constraints": crosstab(labels, "constraints"),
            "automation": crosstab(labels, "automation"),
        },
        "proposals": proposals(labels),
        "lessons": [
            {"source": lesson.get("_file"), "lessons": lesson.get("lessons", []),
             "patterns": lesson.get("patterns", [])}
            for lesson in lessons
        ],
    }


def markdown(experience: dict) -> str:
    lines = [
        "# Router experience from literature labels",
        "",
        f"Generated: {experience['generated_at']}",
        f"Labels: {experience['n_labels']} (from {experience['n_files']} subagent files)",
        "",
        "## Space type",
        "",
        *[f"- {key}: {value}" for key, value in experience["space_type"].items()],
        "",
        "## Literature method",
        "",
        *[f"- {key}: {value}" for key, value in experience["literature_method"].items()],
        "",
        "## Router recommendation",
        "",
        *[f"- {key}: {value}" for key, value in experience["router_recommendation"].items()],
        "",
        f"## Method agreement: {experience['method_agreement']['agreement']} "
        f"(n={experience['method_agreement']['n']})",
        "",
        "## Candidate rules",
        "",
    ]
    if not experience["proposals"]:
        lines.append("- (not enough high-confidence labels yet)")
    for item in experience["proposals"]:
        support = ", ".join(f"{k}={v}" for k, v in item["support"].items())
        lines += [
            f"### {item['rule']} (n={item['evidence_n']}, {item['confidence']})",
            "",
            f"- evidence: {support}",
            f"- action: {item['action']}",
            "",
        ]
    lines.append("## Cross-tabs (router recommendation)")
    lines.append("")
    for key, table in experience["crosstabs"].items():
        lines.append(f"### by {key}")
        lines.append("")
        for group, counts in table.items():
            support = ", ".join(f"{k}={v}" for k, v in counts.items())
            lines.append(f"- {group}: {support}")
        lines.append("")
    lines += [
        "## Subagent lessons",
        "",
    ]
    for item in experience["lessons"]:
        for lesson in item["lessons"]:
            lines.append(f"- ({item['source']}) {lesson}")
    lines += [
        "",
        "## Caveat",
        "",
        "These labels come from titles/abstracts and LLM judgement, not from",
        "re-running experiments. They are router *priors*, to be tested on",
        "benchmarks/proxy campaigns before changing defaults.",
        "",
        "Reproduce:",
        "",
        "```powershell",
        "py -3 scripts/literature_experience_report.py --input research/literature_scan",
        "```",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="research/literature_scan")
    args = parser.parse_args()
    root = Path(args.input)
    if not root.is_absolute():
        root = ROOT / root
    labels = normalise(_read_jsonl_dir(root / "labels"))
    # attach source file for provenance
    for row in labels:
        row.setdefault("_file", "")
    lessons = load_lessons(root / "lessons")
    experience = summarise(labels, lessons)
    (root / "router_experience.json").write_text(
        json.dumps(experience, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (root / "ROUTER_EXPERIENCE.md").write_text(markdown(experience), encoding="utf-8")
    print(json.dumps({
        "labels": experience["n_labels"],
        "method_agreement": experience["method_agreement"],
        "proposals": experience["proposals"],
    }, ensure_ascii=False, indent=2)[:5000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
