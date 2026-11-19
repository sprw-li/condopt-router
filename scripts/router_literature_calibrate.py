"""Turn literature labels into candidate router weights, with honest validation.

Input : research/literature_scan/labels/*.jsonl (subagent routing labels)
        runs/calibration_180/records.json (benchmark matrix, optional)
Output: configs/router_weights_literature.json
        research/literature_scan/LITERATURE_WEIGHTS_REPORT.md

Method
------
1. Build an approximate OptimizationContext from each label (space type, variable
   counts, objectives, constraints, noise, automation).
2. Target = the label's router_recommendation (grid/lhs/sobol treated as one DoE family).
3. Random-search multiplicative weights to maximise label agreement on a
   train split, then report held-out agreement (split by paper hash).
4. Re-evaluate the same weights on the benchmark calibration records with a
   leave-one-problem-out protocol, so literature priors are checked against
   simulated tasks before being shipped.

The default router is NOT changed. Use the candidate explicitly:

    py -3 scripts/route?  (actual CLI below)
    py -3 scripts/run_condopt.py route --weights configs/router_weights_literature.json ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from condopt.calibration import evaluate_weights  # noqa: E402
from condopt.router import OptimizationContext, algorithm_weights, route, set_algorithm_weights  # noqa: E402

def _utf8_child_env() -> dict:
    """Environment for child processes so their stdio is UTF-8, not the codepage."""
    import os

    return {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


DOE_FAMILY = {"grid", "lhs", "sobol"}
ALGORITHMS = ["bayes", "constrained", "grid", "lhs", "mfbo", "parego", "qehvi",
              "qnehvi", "random", "sobol", "tpe"]


def load_labels(path: Path) -> list[dict]:
    rows = []
    for file in sorted(path.glob("labels_*.jsonl")):
        for line in file.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("confidence") in ("medium", "high") and payload.get("router_recommendation") not in (None, "none"):
                rows.append(payload)
    return rows


def _int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_context(label: dict) -> OptimizationContext:
    space = label.get("space_type", "unknown")
    cat = _int(label.get("categorical_count"))
    cont = _int(label.get("continuous_count"))
    if space == "categorical_heavy":
        cat = cat if cat is not None else 8
        cont = cont if cont is not None else 0
    elif space == "continuous_heavy":
        cat = cat if cat is not None else 0
        cont = cont if cont is not None else 6
    elif space == "mixed":
        cat = cat if cat is not None else 4
        cont = cont if cont is not None else 4
    else:
        cat, cont = (cat or 4), (cont or 3)
    automation = str(label.get("automation") or "unknown")
    budget = 60 if automation in ("self-driving lab", "HTE", "flow") else 40
    return OptimizationContext(
        n_dims=cat + cont,
        n_continuous=cont,
        n_categorical=cat,
        feature_dim=cat * 3 + cont if space != "continuous_heavy" else cont,
        budget_total=budget,
        batch_size=4 if automation in ("self-driving lab", "HTE", "flow") else 1,
        noise="high" if label.get("noisy") is True else "medium",
        n_objectives=2 if label.get("objectives") == "multi" else 1,
        n_constraints=1 if label.get("constraints") is True else 0,
        domain="chemistry" if label.get("problem_type") in ("reaction",) else "materials",
        smoothness="mixed",
    )


def matches(label: dict, chosen: str) -> bool:
    target = label.get("router_recommendation")
    if target in DOE_FAMILY:
        return chosen in DOE_FAMILY
    return chosen == target


def evaluate_label_weights(labels: list[dict], weights: dict | None) -> dict:
    set_algorithm_weights(weights)
    try:
        hits = 0
        by_space: dict[str, list[int]] = {}
        for label in labels:
            context = build_context(label)
            chosen = route(context).algorithm
            ok = int(matches(label, chosen))
            hits += ok
            by_space.setdefault(label.get("space_type", "unknown"), []).append(ok)
    finally:
        set_algorithm_weights(None)
    return {
        "n": len(labels),
        "accuracy": round(hits / max(len(labels), 1), 4),
        "by_space": {key: round(sum(values) / len(values), 3) for key, values in by_space.items()},
    }


def split(labels: list[dict]) -> tuple[list[dict], list[dict]]:
    train, test = [], []
    for label in labels:
        digest = hashlib.sha1(str(label.get("paper_id", "")).encode("utf-8")).hexdigest()
        (test if int(digest[:2], 16) < 51 else train).append(label)  # ~20% test
    return train, test


def random_search(labels: list[dict], iterations: int = 400, seed: int = 0) -> dict:
    rng = random.Random(seed)
    baseline = evaluate_label_weights(labels, None)
    best_weights = {name: 1.0 for name in ALGORITHMS}
    best_score = baseline["accuracy"]
    for _ in range(iterations):
        weights = {name: round(rng.uniform(0.55, 1.8), 3) for name in ALGORITHMS}
        metrics = evaluate_label_weights(labels, weights)
        # mild regularisation towards 1.0 to avoid overfitting the 250 labels
        penalty = 0.002 * sum(abs(math.log(value)) for value in weights.values())
        score = metrics["accuracy"] - penalty
        if score > best_score:
            best_score = score
            best_weights = weights
    return {"weights": best_weights, "train": evaluate_label_weights(labels, best_weights), "baseline": baseline}


def benchmark_lopo(records_path: Path, weights: dict) -> dict:
    if not records_path.exists():
        return {"error": f"missing {records_path}"}
    from condopt.calibration import RunRecord, best_algorithm_per_problem, evaluate_weights as eval_bench

    raw = json.loads(records_path.read_text(encoding="utf-8", errors="replace"))
    records = [RunRecord(**item) for item in raw]
    problems = sorted({record.problem for record in records})
    top1 = top2 = 0
    for held_out in problems:
        test = [record for record in records if record.problem == held_out]
        metrics = eval_bench(test, weights)
        top1 += metrics["top1"]
        top2 += metrics["top2"]
    return {
        "n_problems": len(problems),
        "top1": round(top1 / max(len(problems), 1), 3),
        "top2": round(top2 / max(len(problems), 1), 3),
        "baseline_top1": round(eval_bench(records, None)["top1"], 3),
        "baseline_top2": round(eval_bench(records, None)["top2"], 3),
    }


def markdown(result: dict, labels: list[dict]) -> str:
    lines = [
        "# Literature → router weights (candidate)",
        "",
        f"Labels used (medium/high confidence, non-none): **{len(labels)}**",
        "",
        "## Label agreement (approximate contexts)",
        "",
        f"- baseline (default rules): {result['baseline']['accuracy']:.3f}",
        f"- candidate weights: {result['train']['accuracy']:.3f}",
        f"- held-out split: {result['test_accuracy']:.3f}",
        "",
        "### By space type",
        "",
    ]
    for key, value in result["train"]["by_space"].items():
        lines.append(f"- {key}: {value:.3f}")
    lines += [
        "",
        "## Benchmark LOPO check (runs/calibration_180/records.json)",
        "",
        f"- baseline top1/top2: {result['benchmark']['baseline_top1']} / {result['benchmark']['baseline_top2']}",
        f"- candidate top1/top2: {result['benchmark']['top1']} / {result['benchmark']['top2']}",
        "",
        "## Candidate weights",
        "",
        "```json",
        json.dumps(result["weights"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## How to use",
        "",
        "```powershell",
        "py -3 scripts/run_condopt.py route --space runs/prepared/space.json --weights configs/router_weights_literature.json --json",
        "```",
        "",
        "Not enabled by default: label contexts are approximate (abstract-only), so",
        "this is a prior to test, not a replacement for benchmark/default rules.",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="research/literature_scan/labels")
    parser.add_argument("--records", default="runs/calibration_180/records.json")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--out", default="configs/router_weights_literature.json")
    args = parser.parse_args()
    labels_dir = Path(args.labels)
    if not labels_dir.is_absolute():
        labels_dir = ROOT / labels_dir
    labels = load_labels(labels_dir)
    train, test = split(labels)
    result = random_search(train, iterations=args.iterations)
    test_metrics = evaluate_label_weights(test, result["weights"])
    result["test_accuracy"] = test_metrics["accuracy"]
    result["test_metrics"] = test_metrics
    result["benchmark"] = benchmark_lopo(ROOT / args.records, result["weights"])
    result["provenance"] = {
        "labels_dir": str(labels_dir.relative_to(ROOT)),
        "n_labels": len(labels),
        "n_train": len(train),
        "n_test": len(test),
    }
    out = ROOT / args.out
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8", errors="replace")
    report_dir = ROOT / "research/literature_scan"
    (report_dir / "LITERATURE_WEIGHTS_REPORT.md").write_text(markdown(result, labels), encoding="utf-8", errors="replace")
    print(json.dumps({
        "labels": len(labels),
        "baseline_accuracy": result["baseline"]["accuracy"],
        "candidate_accuracy": result["train"]["accuracy"],
        "held_out": result["test_accuracy"],
        "benchmark": result["benchmark"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
