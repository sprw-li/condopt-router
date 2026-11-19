"""Router weight calibration helpers (LOPO-style, lightweight).

Given a small benchmark matrix (problem × algorithm × seed best scores), we
label the empirical best algorithm per problem and search per-algorithm score
multipliers that maximize the router's top-1/top-2 agreement.  Weights are
*multiplicative* on the final rule score (see ``router.set_algorithm_weights``)
and always fall back to the hand-tuned defaults when empty.

This is intentionally tiny (no sklearn, no heavy grid): the full protocol lives
in research/ROUTER_CALIBRATION_HARNESS.md; this module is the executable MVP.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

from .feasibility import feasibility_state as _feasibility_state
from .router import OptimizationContext, algorithm_weights, route, set_algorithm_weights


@dataclass
class RunRecord:
    problem: str
    algorithm: str
    score: float  # higher is better (already signed by the objective direction)
    context: dict
    #: which metric produced ``score`` (records v2: true_best / true_hv / ...)
    metric_name: str = "best_score"


def _true_hv_ratio(problem, vectors) -> float | None:
    """Signed-hypervolume ratio vs the benchmark reference front (MOO records)."""
    import numpy as np

    from .benchmarks import moo_reference_front
    from .pareto import hypervolume_2d, non_dominated_mask

    try:
        names = list(problem.objective.objectives)
        directions = list(problem.objective.directions)
        signs = np.asarray([1.0 if d == "max" else -1.0 for d in directions])
        points = np.asarray([[float(v[n]) for n in names] for v in vectors], dtype=float)
        points = points * signs
        ref = problem.objective.ref_point
        if ref is None or len(ref) != len(names):
            return None
        ref_signed = np.asarray(list(ref), dtype=float) * signs
        reference = np.asarray(moo_reference_front(), dtype=float)  # already signed
        ref_hv = hypervolume_2d(reference, ref_signed)
        if ref_hv <= 0:
            return None
        front = points[non_dominated_mask(points)]
        ratio = float(hypervolume_2d(front, ref_signed) / ref_hv)
        return ratio if 0.0 <= ratio <= 1.5 else None
    except Exception:
        return None


def _run_multifidelity(problem, algorithm: str, budget: int, batch_size: int, seed: int) -> dict:
    """Run a multi-fidelity campaign and return cost-aware metrics."""
    from .benchmarks import cost_aware_best, mf_suzuki
    from .dataset import Observation
    from .registry import build

    optimizer = build(
        algorithm,
        problem.space,
        problem.objective,
        params={"n_initial": min(4, max(2, budget))},
        seed=seed,
    )
    evaluate = getattr(problem, "fidelity_fn", None) or mf_suzuki
    constraints_fn = getattr(problem, "constraints_fn", None)
    observations: list[Observation] = []
    done = 0
    while done < budget:
        n = min(batch_size, budget - done)
        pairs = optimizer.ask_with_fidelity(n)
        for config, fidelity in pairs:
            observation = Observation(
                x=config,
                y=evaluate(config, fidelity),
                constraints=constraints_fn(config) if constraints_fn else {},
                fidelity=fidelity,
                cost=fidelity,
            )
            observations.append(observation)
            optimizer.tell([observation])
            done += 1
    stats = cost_aware_best(observations)
    if problem.objective.constraints:
        best_feasible = None
        for observation in observations:
            if observation.fidelity < 1.0 - 1e-9:
                continue
            # Only a *measured and satisfied* observation can be the feasible
            # baseline; an unmeasured constraint is UNKNOWN, not a violation (A12).
            state = _feasibility_state(observation, problem.objective)["state"]
            if state == "true":
                best_feasible = (
                    observation.y if best_feasible is None else max(best_feasible, observation.y)
                )
        stats["best_feasible_full_fidelity"] = best_feasible
    return stats


def best_algorithm_per_problem(records: Sequence[RunRecord]) -> dict[str, str]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for record in records:
        grouped.setdefault(record.problem, {}).setdefault(record.algorithm, []).append(record.score)
    labels = {}
    for problem, per_algorithm in grouped.items():
        means = {algorithm: sum(values) / len(values) for algorithm, values in per_algorithm.items()}
        labels[problem] = max(means, key=means.get)
    return labels


def _context(record: RunRecord) -> OptimizationContext:
    return OptimizationContext.from_dict(record.context)


def evaluate_weights(records: Sequence[RunRecord], weights: Mapping[str, float] | None) -> dict:
    """Router accuracy/regret under ``weights`` (``None`` = default rules)."""
    labels = best_algorithm_per_problem(records)
    contexts: dict[str, tuple[OptimizationContext, list[str]]] = {}
    for record in records:
        context_key = record.problem
        if context_key not in contexts:
            contexts[context_key] = (_context(record), [])
        contexts[context_key][1].append(record.algorithm)
    best_scores: dict[str, dict[str, float]] = {}
    for record in records:
        key = (record.problem, record.algorithm)
        best_scores.setdefault(record.problem, {})[record.algorithm] = max(
            best_scores.setdefault(record.problem, {}).get(record.algorithm, -math.inf), record.score
        )

    set_algorithm_weights(weights)
    try:
        top1 = top2 = 0
        regrets = []
        for problem, (context, algorithms) in contexts.items():
            recommendation = route(context)
            chosen = recommendation.algorithm
            label = labels[problem]
            scores = best_scores[problem]
            best = max(scores.values())
            chosen_score = scores.get(chosen)
            # Ties count as correct: a label picked by max() must not make an
            # equally-good algorithm a "failure" (review v0.9 P1 tie/regret).
            is_best = chosen_score is not None and chosen_score >= best - 1e-9
            top1 += int(is_best)
            alternatives = [a["algorithm"] for a in recommendation.alternatives]
            tied = any(
                scores.get(algorithm) is not None and scores[algorithm] >= best - 1e-9
                for algorithm in [chosen, *alternatives]
            )
            top2 += int(tied)
            if chosen_score is not None:
                span = max(abs(best), 1e-9)
                regrets.append(max(0.0, (best - chosen_score) / span))
    finally:
        set_algorithm_weights(None)
    n = max(len(contexts), 1)
    return {
        "top1": top1 / n,
        "top2": top2 / n,
        "mean_regret": sum(regrets) / max(len(regrets), 1),
        "n_problems": len(contexts),
    }


def random_search_weights(
    records: Sequence[RunRecord],
    iterations: int = 60,
    seed: int = 0,
    low: float = 0.6,
    high: float = 1.6,
) -> dict:
    """Random-search per-algorithm multipliers; returns ``{"weights", "metrics"}``."""
    rng = random.Random(seed)
    algorithms = sorted({record.algorithm for record in records})
    default_metrics = evaluate_weights(records, None)
    best_weights: dict[str, float] = {}
    best_metrics = dict(default_metrics)
    for _ in range(max(0, iterations)):
        weights = {name: round(rng.uniform(low, high), 3) for name in algorithms}
        metrics = evaluate_weights(records, weights)
        better = (metrics["top1"], metrics["top2"], -metrics["mean_regret"]) > (
            best_metrics["top1"],
            best_metrics["top2"],
            -best_metrics["mean_regret"],
        )
        if better:
            best_metrics, best_weights = metrics, weights
    if not best_weights:
        best_weights = {name: 1.0 for name in algorithms}
    return {
        "weights": best_weights,
        "metrics": best_metrics,
        "default_metrics": default_metrics,
        "n_iterations": iterations,
    }


def lopo_validate(
    records: Sequence[RunRecord],
    iterations: int = 60,
    seed: int = 0,
    low: float = 0.6,
    high: float = 1.6,
) -> dict:
    """Leave-one-problem-out validation of the weight calibration.

    Each fold trains multipliers on all OTHER problems and evaluates on the
    held-out problem, so the reported accuracy is out-of-sample evidence (the
    in-sample ``random_search_weights`` metrics alone are over-optimistic).
    """
    problems = sorted({record.problem for record in records})
    if len(problems) < 2:
        return {
            "top1": 0.0,
            "top2": 0.0,
            "mean_regret": 0.0,
            "n_problems": len(problems),
            "folds": [],
            "note": "need >=2 problems for LOPO",
        }
    fold_metrics = []
    folds = []
    for held_out in problems:
        train = [record for record in records if record.problem != held_out]
        held = [record for record in records if record.problem == held_out]
        result = random_search_weights(
            train, iterations=iterations, seed=seed, low=low, high=high
        )
        metrics = evaluate_weights(held, result["weights"])
        fold_metrics.append(metrics)
        folds.append({"held_out": held_out, "weights": result["weights"], "metrics": metrics})
    n = len(fold_metrics)
    return {
        "top1": sum(metric["top1"] for metric in fold_metrics) / n,
        "top2": sum(metric["top2"] for metric in fold_metrics) / n,
        "mean_regret": sum(metric["mean_regret"] for metric in fold_metrics) / n,
        "n_problems": n,
        "folds": folds,
    }


def collect_records(
    problem_names: Sequence[str],
    algorithms: Sequence[str],
    budget: int = 10,
    batch_size: int = 2,
    seeds: Sequence[int] = (0,),
    noise: str = "medium",
    smoothness: str = "mixed",
    multi_fidelity: bool = False,
) -> list[RunRecord]:
    """Run a small benchmark matrix and return calibration records."""
    from .benchmarks import builtin_problems, mf_suzuki_problem, run_algorithm
    from . import registry

    problems = builtin_problems()
    problems["mf_suzuki"] = mf_suzuki_problem()
    records: list[RunRecord] = []
    for name in problem_names:
        problem = problems[name]
        # Context metadata is derived from the problem (never fabricated by
        # the caller): a benchmark with observation noise is a noisy problem.
        problem_noise = "high" if getattr(problem, "noise_std", 0.0) > 0 else noise
        required: set[str] = set()
        if problem.objective.is_multi_objective:
            required.add("multi_objective")
        if "multi_fidelity" in getattr(problem, "tags", ()):
            required.add("multi_fidelity")
        if problem.objective.constraints:
            required.add("constraints")
        context = OptimizationContext.from_space(
            problem.space,
            budget_total=budget,
            batch_size=batch_size,
            n_objectives=problem.objective.n_objectives,
            n_constraints=len(problem.objective.constraints),
            noise=problem_noise,
            smoothness=smoothness,
            multi_fidelity=multi_fidelity or "multi_fidelity" in getattr(problem, "tags", ()),
        )
        for algorithm in algorithms:
            capabilities = set(registry.spec(algorithm).capabilities)
            if required and not required <= capabilities:
                continue  # e.g. single-objective algorithm on a MOO problem
            special = capabilities & {"multi_objective", "constraints", "multi_fidelity"}
            if special and not special <= required:
                # e.g. qehvi on a plain single-objective problem.  A
                # multi-fidelity algorithm may legitimately carry the extra
                # constraints capability while the problem has none.
                if not ("multi_fidelity" in required and "multi_fidelity" in capabilities):
                    continue
            for seed in seeds:
                if "multi_fidelity" in getattr(problem, "tags", ()):
                    stats = _run_multifidelity(problem, algorithm, budget, batch_size, seed)
                    records.append(
                        RunRecord(
                            problem=name,
                            algorithm=algorithm,
                            score=(
                                float(
                                    stats.get("best_feasible_full_fidelity")
                                    if problem.objective.constraints
                                    else stats["best_full_fidelity"]
                                )
                                if (
                                    stats.get("best_feasible_full_fidelity")
                                    if problem.objective.constraints
                                    else stats["best_full_fidelity"]
                                )
                                is not None
                                else -math.inf
                            ),
                            context=context.to_dict(),
                            metric_name=(
                                "cost_aware_feasible_best"
                                if problem.objective.constraints
                                else "cost_aware_best"
                            ),
                        )
                    )
                    continue
                result = run_algorithm(
                    problem,
                    algorithm,
                    budget=budget,
                    batch_size=batch_size,
                    seed=seed,
                    noise_std=float(getattr(problem, "noise_std", 0.0) or 0.0),
                )
                score = float(result.best_score)
                metric_name = "true_best"
                if problem.objective.constraints:
                    feasible_best = result.best_feasible_score
                    score = float(feasible_best) if feasible_best is not None else -math.inf
                    metric_name = "feasible_best"
                if problem.objective.is_multi_objective and result.objective_vectors:
                    hv_ratio = _true_hv_ratio(problem, result.objective_vectors)
                    if hv_ratio is not None:
                        score, metric_name = hv_ratio, "true_hv"
                records.append(
                    RunRecord(
                        problem=name,
                        algorithm=algorithm,
                        score=score,
                        context=context.to_dict(),
                        metric_name=metric_name,
                    )
                )
    return records
