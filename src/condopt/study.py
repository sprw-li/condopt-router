"""A persistent optimization study: the unit the skill actually drives.

A ``Study`` is one directory that holds everything needed to resume an
optimization campaign on any machine:

```
run/
  problem.json      search space, objective, seed, budget
  config.json       algorithm, params, router context, auto-route settings
  dataset.jsonl     every measured observation (append only)
  pending.json      proposals waiting for measurements (plate/well/batch included)
  rounds.json       one record per ask/tell round (for stall detection)
  decisions.jsonl   router decisions and AI/user overrides with reasons
  opt_state.json    optimizer-internal state (sequence counters, GP data)
  suggestions.csv   latest proposals as a human/robot friendly table
```

The study is the boundary between the *hard algorithm* layer and the
*agent / AI* layer: the agent only calls ``ask``/``tell``/``override`` and
never edits the algorithm code.  All overrides are logged, so an AI can be
flexible without making the campaign unauditable.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from . import registry
from .base import Objective
from .dataset import Dataset, Observation, SCHEMA_VERSION, default_actor
from .pareto import (
    non_dominated_mask,
    observation_values,
    summarise as pareto_summarise,
)
from .router import (
    OptimizationContext,
    Recommendation,
    context_from_space,
    route,
    route_round,
)
from .space import SearchSpace
from .spec import Spec

PROBLEM_FILE = "problem.json"
CONFIG_FILE = "config.json"
DATASET_FILE = "dataset.jsonl"
PENDING_FILE = "pending.json"
ROUNDS_FILE = "rounds.json"
DECISIONS_FILE = "decisions.jsonl"
STATE_FILE = "opt_state.json"
#: Warm-start bookkeeping lives in its own file: ``_save_all`` rewrites state from
#: the optimizer and would drop a field stored there.
WARM_START_FILE = "warm_start.json"
SUGGESTIONS_FILE = "suggestions.csv"

#: Default plate geometry used to label proposals (8x12 = a 96-well plate).
DEFAULT_PLATE_ROWS = 8
DEFAULT_PLATE_COLUMNS = 12

#: Hex characters of the configuration fingerprint written into audit records (A9).
CONFIG_HASH_LENGTH = 12


def _canonical_hash(payload: Mapping) -> str:
    """First :data:`CONFIG_HASH_LENGTH` hex chars of sha256 over canonical JSON.

    ``sort_keys`` plus compact separators make the digest independent of dict
    insertion order, so the same configuration always yields the same hash --
    that equality is what answers "which config version did this decision
    belong to" after the fact.
    """
    canonical = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:CONFIG_HASH_LENGTH]


class PendingError(RuntimeError):
    """Raised when new proposals are requested before the last batch was told."""


def default_confirm_n(budget: int) -> int:
    """Default number of confirmation replicates.

    Measured on a small campaign (branin, budget 12, batch 3): cutting the batch
    from 3 to 2 saves one exploration evaluation but does **not** improve the
    result (20 seeds: best median 4.08 with 3 replicates vs 3.33 with 2; means
    4.19 vs 3.34, one-sided in favour of 3).  So the default stays at 3 and the
    budget is only used for reporting the share — campaigns that cannot afford it
    can override via ``meta["confirm_n"]``.
    """
    return 3


class Study:
    """One optimization campaign persisted in a directory."""

    def __init__(
        self,
        run_dir: str | Path,
        space: SearchSpace,
        objective: Objective,
        algorithm: str,
        params: Mapping | None,
        seed: int = 0,
        context: OptimizationContext | None = None,
        budget: int | None = None,
        batch_size: int = 1,
        auto_route: bool = True,
        patience: int = 3,
        manual_override: bool = False,
        meta: Mapping | None = None,
        spec: Spec | None = None,
    ) -> None:
        self.dir = Path(run_dir)
        self.space = space
        self.objective = objective
        self.spec = spec
        self.algorithm = algorithm
        self.params = registry.validate_params(algorithm, params)
        self.seed = int(seed)
        self.context = context or context_from_space(space, n_objectives=objective.n_objectives)
        self.budget = int(budget if budget is not None else self.context.budget_total)
        self.batch_size = int(batch_size)
        self.auto_route = bool(auto_route)
        # Sweep evidence (scripts/sweep_patience.py, rosenbrock10, budget 60):
        # patience=1 stops so early that even retuning only reaches -358 vs
        # -205 at patience=3, while patience>=5 almost never fires. Clamp the
        # pathological value instead of silently wasting the campaign.
        requested_patience = int(patience)
        self.patience = max(2, requested_patience)
        if requested_patience < 2:
            meta["patience_clamped_from"] = requested_patience
        #: Once the user/AI overrides the algorithm, auto-rerouting stays off
        #: until ``resume_auto_route`` is called; otherwise the next ``ask``
        #: silently reverts the override (review P0-4).
        self.manual_override = bool(manual_override)
        self.meta = dict(meta or {})
        self._rounds: list[dict] = []
        self._pending: list[dict] = []

    # -- creation / loading ----------------------------------------------
    @classmethod
    def create(
        cls,
        run_dir: str | Path,
        space: SearchSpace,
        objective: Objective | None = None,
        spec: Spec | None = None,
        algorithm: str = "auto",
        params: Mapping | None = None,
        seed: int = 0,
        context: OptimizationContext | None = None,
        budget: int | None = None,
        batch_size: int | None = None,
        auto_route: bool = True,
        patience: int = 3,
        stop_on_stagnation: bool = True,
        manual_override: bool = False,
        cost_budget: float | None = None,
        meta: Mapping | None = None,
    ) -> "Study":
        run_dir = Path(run_dir)
        meta = dict(meta or {})
        if cost_budget is not None:
            meta["cost_budget"] = float(cost_budget)
        meta["stop_on_stagnation"] = bool(stop_on_stagnation)
        if not stop_on_stagnation:
            meta["stopping_policy"] = "budget_first"
        if objective is None and spec is not None:
            objective = spec.scalar_objective()
        objective = objective or Objective()
        context = context or context_from_space(
            space,
            n_objectives=objective.n_objectives,
            n_constraints=len(objective.constraints),
            budget_total=budget or 30,
            batch_size=batch_size or 1,
        )
        if budget is not None:
            context.budget_total = int(budget)
        if batch_size is not None:
            context.batch_size = int(batch_size)
        batch_size = int(batch_size or context.batch_size)
        rec: Recommendation | None = None
        explicit_algorithm = algorithm != "auto"
        if algorithm == "auto":
            rec = route(context)
            algorithm = rec.algorithm
            params = {**rec.params, **(params or {})}
        study = cls(
            run_dir=run_dir,
            space=space,
            objective=objective,
            spec=spec,
            algorithm=algorithm,
            params=params,
            seed=seed,
            context=context,
            budget=context.budget_total,
            batch_size=batch_size,
            auto_route=auto_route,
            patience=patience,
            manual_override=manual_override,
            meta=meta,
        )
        study._save_all(create=True)
        if explicit_algorithm:
            # Explicitly requested algorithms must be pinned; otherwise the
            # first ask's auto_route silently replaces them (review v0.7 P0-1).
            study.manual_override = True
            study.log_decision(
                action="manual_selection",
                algorithm=study.algorithm,
                params=study.params,
                reason="显式指定算法，已钉住（auto_route 暂停；condopt auto 可恢复）",
            )
            study._save_all()
        if rec is not None:
            study.log_decision(
                action="initial_route",
                algorithm=rec.algorithm,
                params=rec.params,
                reason="；".join(rec.rationale[:3]) or "router 初始推荐",
                payload={"confidence": rec.confidence, "warnings": rec.warnings},
            )
        return study

    def reset_campaign(
        self,
        algorithm: str = "auto",
        params: Mapping | None = None,
        reason: str = "重新建战役（保留已测数据作为先验）",
    ) -> dict:
        """Re-aim an existing campaign *without* silently discarding its history.

        Re-running ``init`` on a populated run used to overwrite ``config.json``
        (algorithm and manual_override) while keeping ``dataset.jsonl`` and writing no
        decision record -- measured: algorithm ``tpe`` -> ``sobol`` with no audit trail,
        which contradicts the project's own rule that every override is logged.
        This keeps the observations, moves them into the prior slot (so the plateau
        logic treats them exactly like an imported warm start) and logs the change.
        """
        before = {
            "algorithm": self.algorithm,
            "params": dict(self.params or {}),
            "manual_override": bool(self.manual_override),
        }
        self.algorithm = algorithm or "auto"
        self.params = dict(params or {})
        self.manual_override = False
        self._pending = []
        self.log_decision(
            action="campaign_reset",
            algorithm=self.algorithm,
            params=self.params,
            reason=reason,
            payload={"before": before, "observations_kept": len(self._dataset())},
        )
        # Observations stay, but they are history rather than a fresh baseline: the
        # warm-start grace marker keeps plateau detection from firing immediately.
        self._priors_imported_at_round = len(self._rounds)
        (self.dir / WARM_START_FILE).write_text(
            json.dumps({"priors_imported_at_round": len(self._rounds)}, ensure_ascii=False),
            encoding="utf-8",
        )
        self._save_all()
        return {
            "action": "campaign_reset",
            "before": before,
            "after": {"algorithm": self.algorithm, "params": self.params},
            "observations_kept": len(self._dataset()),
            "warm_start_marker": self._priors_imported_at_round,
        }

    @classmethod
    def exists(cls, run_dir: str | Path) -> bool:
        """True when ``run_dir`` already holds a campaign (``problem.json``)."""
        return (Path(run_dir) / PROBLEM_FILE).exists()

    @classmethod
    def load(cls, run_dir: str | Path) -> "Study":
        run_dir = Path(run_dir)
        problem_path = run_dir / PROBLEM_FILE
        if not problem_path.exists():
            raise FileNotFoundError(f"{problem_path} does not exist; run `condopt init` first")
        problem = json.loads(problem_path.read_text(encoding="utf-8-sig"))
        config = json.loads((run_dir / CONFIG_FILE).read_text(encoding="utf-8-sig"))
        space = SearchSpace.from_dict(problem["space"])
        objective = Objective.from_dict(problem.get("objective"))
        spec = Spec.from_dict(problem.get("spec"))
        study = cls(
            run_dir=run_dir,
            space=space,
            objective=objective,
            spec=spec,
            algorithm=config["algorithm"],
            params=config.get("params"),
            seed=problem.get("seed", 0),
            context=OptimizationContext.from_dict(config["context"]),
            budget=problem.get("budget"),
            batch_size=config.get("batch_size", 1),
            auto_route=config.get("auto_route", True),
            patience=config.get("patience", 3),
            manual_override=config.get("manual_override", False),
            meta=problem.get("meta"),
        )
        study._load_state()
        return study

    # -- persistence helpers ---------------------------------------------
    def _save_all(self, create: bool = False) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        problem = {
            "space": self.space.to_dict(),
            "objective": self.objective.to_dict(),
            "spec": self.spec.to_dict() if self.spec is not None else None,
            "seed": self.seed,
            "budget": self.budget,
            "meta": self.meta,
        }
        (self.dir / PROBLEM_FILE).write_text(
            json.dumps(problem, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        config = {
            "algorithm": self.algorithm,
            "params": self.params,
            "context": self.context.to_dict(),
            "batch_size": self.batch_size,
            "auto_route": self.auto_route,
            "patience": self.patience,
            "manual_override": self.manual_override,
        }
        (self.dir / CONFIG_FILE).write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if not (self.dir / DATASET_FILE).exists():
            (self.dir / DATASET_FILE).touch()
        (self.dir / PENDING_FILE).write_text(
            json.dumps(self._pending, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (self.dir / ROUNDS_FILE).write_text(
            json.dumps(self._rounds, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _load_state(self) -> None:
        self._rounds = _read_json(self.dir / ROUNDS_FILE, [])
        self._pending = _read_json(self.dir / PENDING_FILE, [])
        # The warm-start grace marker has to survive a reload: the CLI is one process
        # per command, so an in-memory-only value is lost between `import` and the
        # next `step` (the plateau rule would then fire early).
        state = _read_json(self.dir / WARM_START_FILE, {}) or {}
        marker = state.get("priors_imported_at_round")
        if marker is not None:
            self._priors_imported_at_round = int(marker)

    def _dataset(self) -> Dataset:
        return Dataset.from_jsonl(self.dir / DATASET_FILE)

    def _stagnation_guard_rounds(self) -> int:
        """Rounds that must elapse after a warm start before plateaus may stop the run.

        Imported priors can be *better* than anything the target measures for a while,
        so the "best-so-far" window looks flat right after a transfer and the campaign
        stops early (measured: warm-start arms stopped at 16-19 of 30 evaluations and
        ended far worse than cold starts).  A prior should not count as progress the
        target has to beat from round 1.
        """
        return int(getattr(self, "_priors_imported_at_round", 0) or 0)

    def _recent_best_scores(self, delay: int = 0) -> list[float]:
        """Best-so-far per round, preferring the *local* series after a warm start."""
        has_priors = any(
            observation.meta.get("transferred_from")
            for observation in self._dataset()
        )
        key = "local_best" if has_priors else "best"
        scores = [
            float(round_.get(key, round_.get("best")))
            for round_ in self._rounds
            if round_.get(key, round_.get("best")) is not None
        ]
        return scores[: len(scores) - delay] if delay else scores

    # -- plate / batch bookkeeping ---------------------------------------
    def _plate_geometry(self) -> tuple[int, int]:
        """Rows x columns of the physical plate (meta override, default 8x12)."""
        rows = int(self.meta.get("plate_rows") or DEFAULT_PLATE_ROWS)
        columns = int(self.meta.get("plate_columns") or DEFAULT_PLATE_COLUMNS)
        return max(1, rows), max(1, columns)

    def _used_plate_ids(self) -> list[str]:
        ids = [
            str(observation.meta["plate"])
            for observation in self._dataset()
            if observation.meta.get("plate")
        ]
        ids += [str(item["plate"]) for item in self._pending if item.get("plate")]
        return ids

    def _next_plate_id(self) -> str:
        used = self._used_plate_ids()
        numbers = [
            int(match.group(1))
            for match in (re.fullmatch(r"[Pp](\d+)", plate) for plate in used)
            if match is not None
        ]
        return f"P{max(numbers) + 1 if numbers else len(set(used)) + 1}"

    def _resolve_plate(self, plate: str | None) -> str:
        if plate is None or str(plate).strip() == "":
            return self._next_plate_id()
        return str(plate).strip()

    def _assign_wells(self, plate: str, count: int) -> list[str]:
        """Pick the first free wells on ``plate`` (skipping already-used ones)."""
        rows, columns = self._plate_geometry()
        used = {
            str(observation.meta["plate_well"])
            for observation in self._dataset()
            if str(observation.meta.get("plate") or "") == plate
            and observation.meta.get("plate_well")
        }
        used |= {
            str(item["well"])
            for item in self._pending
            if str(item.get("plate") or "") == plate and item.get("well")
        }
        free = [
            f"{_row_label(row)}{column + 1}"
            for row in range(rows)
            for column in range(columns)
            if f"{_row_label(row)}{column + 1}" not in used
        ]
        if len(free) < int(count):
            raise ValueError(
                f"孔板 {plate} 剩余孔位不足：需要 {int(count)} 个，只剩 {len(free)} 个"
                f"（板规格 {rows}x{columns}）；请换一块新板（去掉 --plate）或调大 "
                "meta.plate_rows / meta.plate_columns"
            )
        return free[: int(count)]

    # -- read-only views --------------------------------------------------
    @property
    def dataset(self) -> Dataset:
        return self._dataset()

    @property
    def pending(self) -> list[dict]:
        return list(self._pending)

    def cancel_pending(self, ids: Sequence[str], reason: str = "") -> list[str]:
        """Drop pending suggestions that cannot be measured (e.g. repeated failures).

        Keeps the queue honest: without this an unattended loop keeps re-measuring
        a broken condition until ``max_empty_batches`` stops the campaign.
        """
        wanted = {str(item) for item in ids}
        cancelled = [item["id"] for item in self._pending if item["id"] in wanted]
        if not cancelled:
            return []
        self._pending = [item for item in self._pending if item["id"] not in wanted]
        self.log_decision(
            action="cancel_pending",
            algorithm=self.algorithm,
            params=self.params,
            reason=reason or f"取消 {len(cancelled)} 条无法完成的建议",
            payload={"cancelled": cancelled, "remaining_pending": len(self._pending)},
        )
        self._save_all()
        return cancelled

    @property
    def rounds(self) -> list[dict]:
        return list(self._rounds)

    def best(self) -> Observation | None:
        dataset = self._dataset()
        full = [observation for observation in dataset if observation.fidelity >= 1.0 - 1e-9]
        # Multi-fidelity studies must report the best FULL-fidelity observation;
        # biased low-fidelity values must not win the "best condition".
        candidates = Dataset(full) if full else dataset
        return candidates.best(self.objective.name, self.objective.direction)

    def remaining(self) -> int:
        return max(0, self.budget - len(self._dataset()) - len(self._pending))

    def cost_spent(self) -> float:
        """Total recorded resource cost of the campaign (multi-fidelity etc.)."""
        return float(sum(observation.cost for observation in self._dataset()))

    def remaining_cost(self) -> float | None:
        """Remaining cost budget when ``meta["cost_budget"]`` is set."""
        budget = self.meta.get("cost_budget")
        if budget is None:
            return None
        return max(0.0, float(budget) - self.cost_spent())

    def current_context(self) -> OptimizationContext:
        dataset = self._dataset()
        ctx = OptimizationContext.from_dict(self.context.to_dict())
        ctx.n_observations = len(dataset)
        ctx.budget_remaining = self.remaining()
        ctx.budget_total = self.budget
        ctx.batch_size = self.batch_size
        return ctx

    # -- decisions --------------------------------------------------------
    def config_hash(self) -> str:
        """Fingerprint of the configuration the next decision belongs to (A9).

        Covers space + objective (+ spec) and everything persisted in
        ``problem.json``/``config.json`` that can change where the search looks
        (seed, budget, algorithm, params, context): two decisions with the same
        hash were taken under one configuration version.
        """
        return _canonical_hash(
            {
                "space": self.space.to_dict(),
                "objective": self.objective.to_dict(),
                "spec": self.spec.to_dict() if self.spec is not None else None,
                "seed": self.seed,
                "budget": self.budget,
                "algorithm": self.algorithm,
                "params": self.params,
                "context": self.context.to_dict(),
                "batch_size": self.batch_size,
                "auto_route": self.auto_route,
                "patience": self.patience,
            }
        )

    def log_decision(
        self,
        action: str,
        algorithm: str,
        params: Mapping | None = None,
        reason: str = "",
        payload: Mapping | None = None,
        actor: str | None = None,
    ) -> dict:
        """Append one audit record and return it.

        ``seq`` is derived from the file rather than an instance counter: the CLI
        runs one process per command, so an in-memory counter would restart at 1
        and break "monotonic within a run" across invocations.
        """
        record = {
            "t": time.time(),
            "seq": self._next_decision_seq(),
            "action": action,
            "algorithm": algorithm,
            "params": dict(params or {}),
            "reason": reason,
            "payload": dict(payload or {}),
            "config_hash": self.config_hash(),
            "actor": str(actor or default_actor()),
        }
        with (self.dir / DECISIONS_FILE).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def _next_decision_seq(self) -> int:
        path = self.dir / DECISIONS_FILE
        if not path.exists():
            return 1
        existing = sum(
            1 for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()
        )
        return existing + 1

    def decisions(self) -> list[dict]:
        """Every decision record, oldest first, with audit defaults for old lines.

        Pre-A9 files carry no ``seq``/``config_hash``/``actor``: they read as
        ``seq`` = position in the file and ``config_hash`` = ``None``.  ``None`` is
        deliberate -- the version those records belonged to was never written down,
        so filling in (say) today's hash would be a false audit claim.  Same for
        ``actor``: a legacy line reads as ``"unknown"``, not as whoever happens to
        have ``CONDOPT_ACTOR`` exported now.
        """
        path = self.dir / DECISIONS_FILE
        if not path.exists():
            return []
        records: list[dict] = []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            record.setdefault("seq", len(records) + 1)
            record.setdefault("config_hash", None)
            record.setdefault("actor", "unknown")
            records.append(record)
        return records

    def audit_trail(self) -> dict:
        """Compact, checkable audit summary of a run (A9).

        Answers the post-hoc questions an unattributed run cannot: which schema
        wrote these records, how much evidence there is, under which
        configuration, and who produced it.
        """
        dataset = self._dataset()
        decisions = self.decisions()
        actors = sorted(
            {str(observation.actor) for observation in dataset}
            | {str(record.get("actor") or "unknown") for record in decisions}
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "n_observations": len(dataset),
            "n_decisions": len(decisions),
            "config_hash": self.config_hash(),
            "actors": actors,
        }

    # -- core loop --------------------------------------------------------
    def ask(self, n: int | None = None, plate: str | None = None) -> list[dict]:
        """Propose the next batch of conditions.

        Every proposal is stamped with a ``plate``, a ``well`` and the round
        ``batch`` so the physical execution is traceable and batch/plate
        effects can be diagnosed later.  Pass ``plate="P3"`` to keep filling
        an existing plate; otherwise a fresh plate id is generated.
        """
        if self._pending:
            raise PendingError(
                f"{len(self._pending)} suggestions are still pending; call `tell` with their "
                "measured results before asking for new ones (or `condopt tell` with the results file)"
            )
        n = int(n or self.batch_size)
        if n < 1:
            raise ValueError("n must be >= 1")
        if n > self.remaining():
            raise ValueError(f"only {self.remaining()} evaluations left in the budget")
        dataset = self._dataset()
        observations = dataset.observations
        ctx = self.current_context()

        algorithm, params = self.algorithm, dict(self.params)
        skip_reroute = False
        # An exploration restart (route_round) lasts exactly one batch; remember
        # the main algorithm in meta so the next ask can return to it instead
        # of rerouting to sobol forever.
        resume_main = self.meta.pop("_exploration_main", None)
        cooldown = int(self.meta.pop("_reroute_cooldown", 0) or 0)
        if resume_main and self.auto_route and not self.manual_override:
            rec = route(ctx)
            algorithm, params = rec.algorithm, rec.params
            self.algorithm, self.params = algorithm, dict(params)
            self.log_decision(
                action="resume_route",
                algorithm=algorithm,
                params=params,
                reason=f"探索批次结束，恢复模型引导搜索（原主算法 {resume_main}）",
                payload={"from": resume_main},
            )
            skip_reroute = True
            cooldown = self.patience
        elif cooldown > 0:
            # Give the main algorithm `patience` consecutive batches before the
            # next exploration restart, otherwise half the budget is wasted on
            # alternating sobol batches.
            skip_reroute = True
            cooldown -= 1
        if self.auto_route and not self.manual_override and not skip_reroute:
            rec = route_round(ctx, self._recent_best_scores(), patience=self.patience)
            if rec.algorithm != algorithm:
                self.log_decision(
                    action="auto_reroute",
                    algorithm=rec.algorithm,
                    params=rec.params,
                    reason="；".join(rec.rationale),
                    payload={"from": algorithm, "scores": rec.candidate_scores},
                )
                algorithm, params = rec.algorithm, rec.params
                design = rec.design or {}
                main = design.get("main") or next(
                    (item.get("algorithm") for item in rec.alternatives
                     if item.get("why", "").startswith("原推荐")),
                    None,
                )
                if main:
                    # exploration batch: do not pin sobol as the study algorithm
                    self.meta["_exploration_main"] = main
                    cooldown = self.patience
                else:
                    self.algorithm, self.params = algorithm, dict(params)
        if cooldown > 0:
            self.meta["_reroute_cooldown"] = cooldown

        optimizer = registry.build(
            algorithm,
            self.space,
            self.objective,
            params=params,
            # Advance the RNG stream per round; reusing the same seed would make
            # every ask replay the same random sequence (review P1-8).
            seed=self.seed + len(observations),
            prior=observations,
        )
        state_path = self.dir / STATE_FILE
        optimizer.cost_remaining = self.remaining_cost()
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8-sig"))
                if state.get("algorithm") == algorithm:
                    optimizer.load_state_dict(state.get("state", {}))
                    # A state snapshot is stale for algorithms whose state is derived
                    # from the observations (the Study records data itself and never
                    # calls tell).  Replay the history so the loaded snapshot cannot
                    # freeze them on their initial mean.
                    if getattr(optimizer, "replay_history_on_load", False) and observations:
                        optimizer.tell(list(observations))
            except (json.JSONDecodeError, ValueError):
                pass

        if getattr(optimizer, "supports_fidelity", False):
            pairs = optimizer.ask_with_fidelity(n)
        else:
            pairs = [(config, 1.0) for config in optimizer.ask(n)]
        # Hard cost cap: the algorithm plans with its own cost model, but the
        # campaign must not spend more than the recorded budget.  The default
        # cost model is cost == fidelity, so trim the tail of the batch with it.
        cost_cap = self.remaining_cost()
        if cost_cap is not None and pairs:
            kept: list[tuple[dict, float]] = []
            planned = 0.0
            for config, fidelity in pairs:
                estimate = float(fidelity)
                if planned + estimate > cost_cap + 1e-9:
                    break
                kept.append((config, fidelity))
                planned += estimate
            if len(kept) < len(pairs):
                self.log_decision(
                    action="cost_budget_trim",
                    algorithm=algorithm,
                    params=params,
                    reason=(
                        f"剩余成本预算 {cost_cap:.3f}，按 cost≈fidelity 截断 "
                        f"{len(pairs)} 个提议为 {len(kept)} 个"
                    ),
                    payload={
                        "requested": n,
                        "returned": len(kept),
                        "remaining_cost": round(cost_cap, 6),
                        "planned_cost": round(planned, 6),
                    },
                )
                pairs = kept
        # Suggestions carry the algorithm + fidelity for auditability.
        trimmed = int(getattr(optimizer, "last_cost_trimmed", 0) or 0)
        if trimmed:
            self.log_decision(
                action="cost_trim",
                algorithm=algorithm,
                params=params,
                reason=f"成本预算不足，{n} 个提议裁剪为 {len(pairs)} 个（保留可用低保真档）",
                payload={
                    "requested": n,
                    "returned": len(pairs),
                    "trimmed": trimmed,
                    "full_requested": int(getattr(optimizer, "last_full_requested", 0) or 0),
                    "full_returned": int(getattr(optimizer, "last_full_returned", 0) or 0),
                },
            )
        batch_id = len(self._rounds) + 1
        plate_id = self._resolve_plate(plate)
        wells = self._assign_wells(plate_id, len(pairs))
        pending = []
        for index, ((config, fidelity), well) in enumerate(zip(pairs, wells)):
            pending.append(
                {
                    "id": f"r{batch_id}-{index + 1}",
                    "batch": batch_id,
                    "plate": plate_id,
                    "well": well,
                    "algorithm": algorithm,
                    "fidelity": float(fidelity),
                    "x": config,
                }
            )
        self._pending = pending
        state_path.write_text(
            json.dumps({"algorithm": algorithm, "state": optimizer.state_dict()}, ensure_ascii=False),
            encoding="utf-8",
        )
        self._save_all()
        self._write_suggestions(pending)
        return pending

    def tell(self, results: Sequence[Mapping | Observation]) -> list[Observation]:
        """Record measurements for the current pending batch.

        Supports *partial* and *repeated* tells: only suggestions that appear
        in ``results`` leave the pending queue, and a suggestion id that was
        already recorded is skipped instead of being appended twice (so a
        retried network/result upload is idempotent).
        """
        if not results:
            return []
        dataset = self._dataset()
        existing_ids = {
            obs.meta.get("suggestion_id")
            for obs in dataset
            if obs.meta.get("suggestion_id")
        }
        pending_by_id = {item["id"]: item for item in self._pending}
        observations: list[Observation] = []
        matched: set[str] = set()
        for item in results:
            sugg_id: str | None = None
            payload_x = None
            if isinstance(item, Observation):
                sugg_id = item.meta.get("suggestion_id")
                payload_x = item.x
            else:
                payload = dict(item)
                sugg_id = payload.get("id")
                payload_x = payload.get("x")
            if sugg_id:
                if sugg_id in existing_ids:
                    continue  # idempotent replay
                if sugg_id not in pending_by_id:
                    raise ValueError(f"result references unknown pending id {sugg_id!r}")
                matched.add(sugg_id)
            elif payload_x is not None and self._pending:
                # Match by configuration value when no id was supplied.
                key = self._encode_key(payload_x)
                for pending_id, pending_item in pending_by_id.items():
                    if self._encode_key(pending_item["x"]) == key:
                        sugg_id = pending_id
                        matched.add(pending_id)
                        break
            observations.append(self._to_observation(item, pending_by_id, suggestion_id=sugg_id))

        if observations:
            dataset.append_jsonl(self.dir / DATASET_FILE, observations)
            dataset_snapshot = Dataset.from_jsonl(self.dir / DATASET_FILE)
            best = dataset_snapshot.best_score(
                self.objective.name, self.objective.direction
            )
            # Plateau detection must look at what *this campaign* achieved: an
            # imported prior can be better than anything the target can reach, and
            # then the global best looks flat from round 1 (measured: warm starts
            # stopped at 16-19 of 30 evaluations and ended far worse than cold runs).
            local_scores = [
                observation.score(self.objective.name, self.objective.direction)
                for observation in dataset_snapshot.observations
                if not observation.meta.get("transferred_from")
            ]
            local_best = max(local_scores) if local_scores else best
            self._rounds.append(
                {
                    "round": len(self._rounds) + 1,
                    "t": time.time(),
                    "n": len(observations),
                    "algorithm": observations[0].meta.get("algorithm", self.algorithm),
                    "best": best,
                    "local_best": local_best,
                }
            )
        if matched:
            self._pending = [item for item in self._pending if item["id"] not in matched]
        self._save_all()
        return observations

    def _encode_key(self, config: Mapping) -> tuple:
        validated = self.space.validate_config(config)
        return tuple(float(x) for x in self.space.encode(validated).round(9))

    def import_observations(self, results: Sequence[Mapping | Observation]) -> list[Observation]:
        """Append historical measurements as prior data (not part of a round).

        Use this to warm-start a study from an existing campaign: the
        observations count against ``budget`` and are visible to the model,
        but they are not attributed to any proposal of this study.
        """
        observations = []
        existing = {
            (self._encode_key(obs.x), round(float(obs.fidelity), 6))
            for obs in self._dataset()
        }
        for item in results:
            obs = self._to_observation(item, {})
            key = (self._encode_key(obs.x), round(float(obs.fidelity), 6))
            if key in existing:
                continue
            existing.add(key)
            observations.append(obs)
        if observations:
            dataset = self._dataset()
            dataset.append_jsonl(self.dir / DATASET_FILE, observations)
            # Warm starts need a grace period before plateau detection is trusted.
            # The marker must survive a reload: the CLI runs one process per command,
            # so an in-memory-only marker silently disappears (probe: the guard was
            # lost after `Study.load`, and the plateau rule then fires early).
            self._priors_imported_at_round = len(self._rounds)
            self._save_all()
            (self.dir / WARM_START_FILE).write_text(
                json.dumps(
                    {"priors_imported_at_round": int(self._priors_imported_at_round)},
                    ensure_ascii=False, indent=2,
                ),
                encoding="utf-8",
            )
        return observations

    def _to_observation(
        self,
        item: Mapping | Observation,
        pending_by_id: Mapping[str, dict],
        suggestion_id: str | None = None,
    ) -> Observation:
        if isinstance(item, Observation):
            obs = item
            if suggestion_id:
                pending_item = pending_by_id[suggestion_id]
                obs.meta.setdefault("algorithm", pending_item["algorithm"])
                obs.meta.setdefault("suggestion_id", suggestion_id)
                obs.meta.setdefault("batch", pending_item.get("batch"))
                obs.meta.setdefault("plate", pending_item.get("plate"))
                obs.meta.setdefault("plate_well", pending_item.get("well"))
                if pending_item.get("purpose"):
                    obs.meta.setdefault("purpose", pending_item["purpose"])
        else:
            payload = dict(item)
            sugg_id = suggestion_id or payload.get("id")
            if "x" not in payload:
                if sugg_id is None:
                    raise ValueError(
                        "observation needs an 'x' condition (or a pending suggestion id)"
                    )
                if sugg_id not in pending_by_id:
                    raise ValueError(f"result references unknown pending id {sugg_id!r}")
                payload["x"] = pending_by_id[sugg_id]["x"]
            # Validation happens exactly once, at the end of this function, where
            # out-of-range values are either rejected with an actionable message or
            # clipped *with* an audit record (A10).  Validating here as well would
            # raise before that guard could run.
            x = dict(payload["x"])
            meta = dict(payload.get("meta", {}))
            if sugg_id in pending_by_id:
                pending_item = pending_by_id[sugg_id]
                meta.setdefault("algorithm", pending_item["algorithm"])
                meta.setdefault("batch", pending_item.get("batch"))
                meta.setdefault("plate", pending_item.get("plate"))
                meta.setdefault("plate_well", pending_item.get("well"))
                if pending_item.get("purpose"):
                    meta.setdefault("purpose", pending_item["purpose"])
            else:
                meta.setdefault("algorithm", self.algorithm)
            if sugg_id:
                meta.setdefault("suggestion_id", sugg_id)
            # Execution metadata may be overridden explicitly (measured on a
            # different plate than the proposal, re-arrayed well, ...).
            if "plate" in payload:
                meta["plate"] = payload["plate"]
            if "well" in payload:
                meta["plate_well"] = payload["well"]
            if "batch" in payload:
                meta["batch"] = payload["batch"]
            fidelity = payload.get("fidelity")
            if fidelity is None:
                fidelity = (
                    pending_by_id[sugg_id].get("fidelity", 1.0)
                    if sugg_id in pending_by_id
                    else 1.0
                )
            std = payload.get("objective_std", payload.get("std", payload.get("y_std", 0.0)))
            replicates = payload.get(
                "n_replicates", payload.get("k", payload.get("replicates", 1))
            )
            cost = payload.get("cost")
            if cost is None:
                # Multi-fidelity default cost model: every run costs ~fidelity
                # (a full-fidelity run costs 1.0); explicit costs always win.
                cost = float(fidelity)
            obs = Observation(
                x=x,
                y=payload.get("y"),
                objectives=payload.get("objectives", {}),
                constraints=payload.get("constraints", {}),
                fidelity=float(fidelity),
                cost=float(cost),
                objective_std=float(std or 0.0),
                n_replicates=int(replicates or 1),
                source=payload.get("source", "manual"),
                meta=meta,
            )
        if self.spec is not None:
            values = dict(obs.objectives)
            values.update(obs.constraints)
            if all(goal.metric in values for goal in self.spec.goals):
                obs.y = self.spec.score(values)
                obs.meta["spec"] = self.spec.compliance(values)
        try:
            self.space.validate_config(obs.x)
        except ValueError as exc:
            if not self.meta.get("clip_out_of_bounds"):
                raise ValueError(
                    f"{exc}\n"
                    "  越界值默认被拒绝：下游所有 decode 都会静默夹到边界，"
                    "记录下来的就不是算法实际看到的条件。\n"
                    "  确实要夹到边界并留痕：init 时用 --clip-out-of-bounds，"
                    '或 meta 里设 {"clip_out_of_bounds": true}。'
                ) from exc
            clipped, adjustments = self.space.coerce_bounds(obs.x)
            obs.x = self.space.validate_config(clipped)
            obs.meta["bounds_clipped"] = {
                name: {"original": original, "clipped": value}
                for name, (original, value) in adjustments.items()
            }
            self.log_decision(
                action="clip_out_of_bounds",
                algorithm=self.algorithm,
                params={},
                reason="显式开启 clip_out_of_bounds，越界测量被夹到边界",
                payload={"adjustments": obs.meta["bounds_clipped"]},
            )
        return obs

    def _spec_summary(self, best: Observation | None, decision: dict | None = None) -> dict | None:
        if self.spec is None:
            return None
        dataset = self._dataset()
        in_spec = 0
        for observation in dataset:
            values = self.spec.observation_values(observation)
            if values and all(goal.metric in values for goal in self.spec.goals):
                if self.spec.compliance(values)["in_spec"]:
                    in_spec += 1
        best_compliance = None
        if best is not None:
            values = self.spec.observation_values(best)
            if all(goal.metric in values for goal in self.spec.goals):
                best_compliance = self.spec.compliance(values)
        recommended = decision
        return {
            "goals": [goal.to_dict() for goal in self.spec.goals],
            "constraints": self.spec.constraints,
            "n": len(dataset),
            "in_spec": in_spec,
            "best": best_compliance,
            "recommended": recommended,
        }

    def predict(self, config: Mapping) -> dict:
        """Surrogate prediction at a *new* condition (no experiment needed).

        Fits one frozen-hyperparameter GP per objective metric on the recorded
        observations and returns mean/std in raw units, plus predicted
        constraints, feasibility probability and (when a spec exists) the
        predicted spec score.  Useful for "what-if" checks and for keeping a
        human in the loop before committing lab time.
        """
        import numpy as np
        from sklearn.gaussian_process import GaussianProcessRegressor

        from .algorithms.bayes import _make_kernel

        dataset = self._dataset()
        if len(dataset) < 3:
            raise ValueError("predict needs at least 3 observations")
        condition = self.space.validate_config(config)
        features = self.space.features_many([condition])
        X = self.space.features_many([obs.x for obs in dataset])

        def fit(values: np.ndarray):
            usable = np.isfinite(values)
            target = values[usable]
            if int(np.sum(usable)) < 3:
                return None
            noise = np.asarray([
                (obs.objective_std ** 2) / obs.n_replicates if obs.objective_std > 0 else np.nan
                for obs, ok in zip(dataset, usable) if ok
            ], dtype=float)
            base = 1e-4
            if np.any(np.isfinite(noise) & (noise > 0)):
                y_var = max(float(np.var(target)), 1e-12)
                alpha = np.clip(
                    np.where(np.isfinite(noise) & (noise > 0), noise / y_var, base),
                    1e-12, 0.5,
                )
            else:
                alpha = base
            model = GaussianProcessRegressor(
                kernel=_make_kernel(
                    "matern52", 0.3, feature_dim=self.space.feature_dim, ard=True
                ),
                alpha=alpha, normalize_y=True, optimizer=None,
                random_state=self.meta.get("seed", 0),
            )
            return model.fit(X[usable], target)

        metric_names = (
            list(self.spec.metrics) if self.spec is not None
            else list(self.objective.objectives) or [self.objective.name]
        )
        predictions: dict[str, dict] = {}
        for name in metric_names:
            values = np.asarray([
                obs.objectives.get(name, obs.y if name == self.objective.name else np.nan)
                if obs.objectives else (obs.y if name == self.objective.name else np.nan)
                for obs in dataset
            ], dtype=float)
            model = fit(values)
            if model is None:
                continue
            mean, std = model.predict(features, return_std=True)
            predictions[name] = {"mean": float(mean[0]), "std": float(std[0])}

        feasible_probs: dict[str, dict] = {}
        feasibility = 1.0
        limits = (
            self.spec.constraints if self.spec is not None and self.spec.constraints
            else self.objective.constraint_limits or {}
        )
        for name, limit in (limits or {}).items():
            values = np.asarray(
                [obs.constraints.get(name, np.nan) for obs in dataset], dtype=float
            )
            model = fit(values)
            if model is None:
                continue
            mean, std = model.predict(features, return_std=True)
            from scipy.stats import norm

            threshold = float((limit or {}).get("value", 0.0))
            op = str((limit or {}).get("op", "<="))
            z = (threshold - mean[0]) / max(std[0], 1e-9) if op == "<=" else (
                mean[0] - threshold
            ) / max(std[0], 1e-9)
            probability = float(norm.cdf(z))
            feasible_probs[name] = {
                "mean": float(mean[0]), "std": float(std[0]),
                "threshold": threshold, "op": op, "p_feasible": probability,
            }
            feasibility *= probability

        spec_score = None
        if self.spec is not None and all(
            goal.metric in predictions for goal in self.spec.goals
        ):
            spec_score = float(self.spec.score({
                goal.metric: predictions[goal.metric]["mean"] for goal in self.spec.goals
            }))
        return {
            "run_dir": str(self.dir),
            "condition": condition,
            "predictions": predictions,
            "constraints": feasible_probs,
            "p_feasible": feasibility if feasible_probs else None,
            "spec_score": spec_score,
            "n_observations": len(dataset),
        }

    def simulate_transfer(
        self,
        source: str | Path,
        top_k: int | None = None,
        prior_reliability: float = 1.0,
    ) -> dict:
        """Preview how a transfer would change the current recommendation.

        Works on a throwaway copy of the run directory, so the live study is
        untouched.  Useful as a final check before committing priors.
        """
        import shutil
        import tempfile

        before = self.recommend()
        before_condition = None if before is None else dict(before["condition"])
        report: dict = {
            "before": before_condition,
            "dry_run": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            clone = Path(tmp) / "clone"
            shutil.copytree(self.dir, clone)
            preview = Study.load(clone)
            transfer = preview.transfer_from(
                source, top_k=top_k, prior_reliability=prior_reliability,
                reason="simulate_transfer preview",
            )
            after = preview.recommend()
            after_condition = None if after is None else dict(after["condition"])
            report.update(
                {
                    "after": after_condition,
                    "transferred": transfer.get("imported"),
                    "changed": before_condition != after_condition,
                    "n_before": len(self._dataset()),
                    "n_after": len(preview.dataset.observations),
                }
            )
            if before_condition is not None and after_condition is not None:
                names = [name for name in self.space.names]
                same = sum(
                    1 for name in names if before_condition.get(name) == after_condition.get(name)
                )
                report["unchanged_fields"] = same
                report["n_fields"] = len(names)
        report["live_run_untouched"] = len(self._dataset()) == report["n_before"]
        return report

    def assess_transfer(self, source: str | Path) -> dict:
        """Estimate whether a source campaign agrees with this study's evidence.

        Fits the target surrogate on the observations already collected, predicts
        the source observation conditions, and compares the predictions with the
        source labels (Spearman rank correlation + normalised MAE).  The result
        suggests a ``prior_reliability`` for :meth:`transfer_from`, turning the
        manual risk knob into a data-driven default.
        """
        # Accept either a run directory or an already-loaded Study.
        source_study = source if isinstance(source, Study) else Study.load(source)
        target_observations = self._dataset()
        source_observations = list(source_study.dataset.observations)
        if len(target_observations) < 3 or len(source_observations) < 3:
            return {
                "ready": False,
                "reason": "need at least 3 observations on both sides",
                "n_target": len(target_observations),
                "n_source": len(source_observations),
                "suggested_reliability": None,
            }
        metric_names = (
            list(self.spec.metrics) if self.spec is not None
            else list(self.objective.objectives) or [self.objective.name]
        )
        metric = self.objective.name
        if self.objective.is_multi_objective:
            metric = metric_names[0]
        predicted: list[float] = []
        actual: list[float] = []
        for observation in source_observations[:20]:
            try:
                condition = self.space.validate_config(observation.x)
            except ValueError:
                continue
            try:
                prediction = self.predict(condition)
            except ValueError:
                continue
            if metric not in prediction["predictions"]:
                continue
            value = observation.objectives.get(metric)
            if value is None and metric == self.objective.name:
                value = observation.y
            if value is None:
                continue
            predicted.append(float(prediction["predictions"][metric]["mean"]))
            actual.append(float(value))
        import numpy as np

        if len(predicted) < 3:
            return {
                "ready": False,
                "reason": "not enough comparable source conditions",
                "n_target": len(target_observations),
                "n_source": len(source_observations),
                "suggested_reliability": None,
            }
        rank_a = np.argsort(np.argsort(actual))
        rank_b = np.argsort(np.argsort(predicted))
        if float(np.std(rank_a)) < 1e-12 or float(np.std(rank_b)) < 1e-12:
            rho = None
        else:
            rho = float(np.corrcoef(rank_a, rank_b)[0, 1])
        target_scores = [
            float(obs.score(self.objective.name, self.objective.direction))
            for obs in target_observations
        ]
        span = max(target_scores) - min(target_scores) if len(target_scores) > 1 else 1.0
        mae = float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted)))) / max(
            abs(span), 1e-12
        )
        # Bootstrap the rank correlation first: both the robustness estimate and
        # the confidence flag need it.  Small samples made this estimator badly
        # wrong on the real ART set (3 target / 3 source points -> suggested 0.3
        # while the warm start rediscovered the real best 2/2 times).
        ci_low = ci_high = None
        samples: list[float] = []
        n_pairs = len(predicted)
        if n_pairs >= 3:
            rng = np.random.default_rng(0)  # deterministic: same data, same CI
            actual_array = np.asarray(actual, dtype=float)
            predicted_array = np.asarray(predicted, dtype=float)
            for _ in range(200):
                index = rng.integers(0, n_pairs, size=n_pairs)
                a = actual_array[index]
                b = predicted_array[index]
                rank_a_boot = np.argsort(np.argsort(a))
                rank_b_boot = np.argsort(np.argsort(b))
                if float(np.std(rank_a_boot)) < 1e-12 or float(np.std(rank_b_boot)) < 1e-12:
                    continue
                samples.append(float(np.corrcoef(rank_a_boot, rank_b_boot)[0, 1]))
            if samples:
                ci_low = round(float(np.percentile(samples, 5)), 3)
                ci_high = round(float(np.percentile(samples, 95)), 3)
        # Use the bootstrap median as the primary estimate: on tiny samples the
        # single point estimate swings wildly (real ART set: point rho gave 0.3
        # while the warm start won 2/2).
        rho_point = rho
        rho_robust = None
        if samples:
            rho_robust = float(np.median(samples))
        rho_used = rho_robust if rho_robust is not None else rho_point
        if rho_used is None:
            suggested = None
        elif rho_used >= 0.6:
            suggested = 1.0
        elif rho_used >= 0.2:
            suggested = 0.6
        else:
            suggested = 0.3
        suggested_point = None
        if rho_point is not None:
            suggested_point = (
                1.0 if rho_point >= 0.6 else 0.6 if rho_point >= 0.2 else 0.3
            )
        smallest = min(len(target_observations), n_pairs)
        wide = ci_low is not None and ci_high is not None and (ci_high - ci_low) > 1.0
        confidence = "low" if (smallest < 5 or wide) else "normal"
        # Calibration (``research/AUDIT_CALIBRATION.md``): with few target observations a
        # *biased* source still scored 0.65 (=> full trust) while it cost -0.54, and an
        # identical source scored below a noisy one.  A low-confidence audit therefore
        # may not recommend full trust; cap it at 0.6 and say why.
        capped = False
        if confidence == "low" and suggested is not None and suggested > 0.6:
            suggested = 0.6
            capped = True
        return {
            "ready": True,
            "metric": metric,
            "n_target": len(target_observations),
            "n_source": len(source_observations),
            "n_compared": len(predicted),
            "confidence": confidence,
            "spearman_ci": (
                None if ci_low is None else {"low": ci_low, "high": ci_high, "level": 0.9}
            ),
            "sample_warning": (
                (
                    f"只比较了 {smallest} 个条件：建议值噪声大，先补到 5 条以上再当依据"
                    if smallest < 5
                    else f"bootstrap 置信区间过宽（{ci_low} ~ {ci_high}）：source 与目标关系不稳定"
                )
                if confidence == "low"
                else None
            ),
            "spearman": None if rho_point is None else round(rho_point, 3),
            "spearman_robust": None if rho_robust is None else round(rho_robust, 3),
            "normalized_mae": round(mae, 4),
            "suggested_reliability": suggested,
            "suggested_reliability_point": suggested_point,
            "capped_by_low_confidence": capped,
            "estimate": "bootstrap_median" if rho_robust is not None else "point",
            "reason": (
                "源与目标证据一致，可完全信任先验"
                if suggested == 1.0
                else "一致性一般：建议折价先验"
                if suggested is not None
                else "秩相关无法估计（源标签几乎无变化）"
            )
            + (
                f"（bootstrap 中位数 {rho_robust:.3f}）"
                if rho_robust is not None
                else ""
            )
            + (
                "；低置信度：建议值已封顶 0.6（校准实验里有偏源拿到 0.65 却带来负收益）"
                if capped
                else ""
            ),
        }

    def check_integrity(self) -> dict:
        """Self-check a run directory before/after unattended campaigns.

        Verifies that the persisted files are mutually consistent: unique
        suggestion ids, pending/dataset disjointness, parseable optimizer state
        that matches the configured algorithm, and readable rounds/problem
        metadata.  Returns ``{"ok": bool, "issues": [...], "counts": {...}}``.
        """
        issues: list[str] = []
        counts: dict[str, int] = {}

        try:
            dataset = self._dataset()
        except Exception as exc:  # noqa: BLE001 - integrity report, not a crash
            return {
                "ok": False,
                "issues": [f"dataset unreadable: {type(exc).__name__}: {exc}"],
                "counts": {},
                "run_dir": str(self.dir),
            }
        counts["observations"] = len(dataset)

        dataset_ids = [
            str(obs.meta["suggestion_id"])
            for obs in dataset
            if obs.meta.get("suggestion_id")
        ]
        if len(set(dataset_ids)) != len(dataset_ids):
            issues.append("dataset contains duplicate suggestion_id values")

        pending = list(self.pending)
        counts["pending"] = len(pending)
        pending_ids = [str(item.get("id")) for item in pending]
        if len(set(pending_ids)) != len(pending_ids):
            issues.append("pending contains duplicate ids")
        overlap = set(pending_ids) & set(dataset_ids)
        if overlap:
            issues.append(f"ids present in both pending and dataset: {sorted(overlap)[:3]}")

        state_path = self.dir / STATE_FILE
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8-sig"))
            except json.JSONDecodeError as exc:
                issues.append(f"opt_state.json is not valid JSON: {exc}")
            else:
                stored = state.get("algorithm")
                if stored and stored != self.algorithm:
                    issues.append(
                        f"opt_state algorithm {stored!r} != config algorithm {self.algorithm!r}"
                    )
                if not isinstance(state.get("state", {}), dict):
                    issues.append("opt_state.json 'state' is not an object")

        for name in (CONFIG_FILE, PROBLEM_FILE, ROUNDS_FILE):
            path = self.dir / name
            if not path.exists():
                issues.append(f"missing file: {name}")
                continue
            try:
                json.loads(path.read_text(encoding="utf-8-sig"))
            except json.JSONDecodeError as exc:
                issues.append(f"{name} is not valid JSON: {exc}")

        counts["rounds"] = len(self.rounds)
        return {
            "ok": not issues,
            "issues": issues,
            "counts": counts,
            "run_dir": str(self.dir),
            "algorithm": self.algorithm,
        }

    def fidelity_summary(self) -> dict:
        """Low- vs full-fidelity consistency diagnostics.

        Observations that share the same encoded condition at different
        fidelities give an empirical bias estimate ``E[y_full - y_low]``; the
        normalised bias divides it by the observed metric span so it is
        comparable across scales.  ``None`` means "not enough paired data".
        """
        dataset = self._dataset()
        levels: dict[float, list[float]] = {}
        for observation in dataset:
            if observation.y is None:
                continue
            key = round(float(observation.fidelity), 6)
            levels.setdefault(key, []).append(float(observation.y))
        per_level = {
            f"{level:g}": {"n": len(values), "mean": sum(values) / len(values)}
            for level, values in sorted(levels.items())
        }
        groups: dict[tuple, list] = {}
        for observation in dataset:
            try:
                encoded = tuple(float(value) for value in self.space.encode(observation.x))
            except Exception:
                continue
            key = tuple(round(value, 6) for value in encoded)
            groups.setdefault(key, []).append(observation)
        diffs: list[float] = []
        for group in groups.values():
            full = [obs for obs in group if float(obs.fidelity) >= 1.0 - 1e-9 and obs.y is not None]
            low = [obs for obs in group if float(obs.fidelity) < 1.0 - 1e-9 and obs.y is not None]
            if full and low:
                diffs.append(
                    sum(float(obs.y) for obs in full) / len(full)
                    - sum(float(obs.y) for obs in low) / len(low)
                )
        mean_bias = sum(diffs) / len(diffs) if diffs else None
        values = [float(obs.y) for obs in dataset if obs.y is not None]
        span = (max(values) - min(values)) if len(values) >= 2 else 0.0
        normalized = (
            mean_bias / span if mean_bias is not None and span > 1e-12 else None
        )
        return {
            "levels": per_level,
            "n_full_fidelity": sum(
                1 for obs in dataset if float(obs.fidelity) >= 1.0 - 1e-9
            ),
            "paired": {"n": len(diffs), "mean_bias": mean_bias},
            "normalized_bias": normalized,
        }

    def plate_summary(self) -> dict:
        """Per-plate/batch statistics plus a batch-effect warning.

        A plate (``meta["plate"]``) is one physical batch of wells; wells of
        the same plate share a pipetting/reagent lot, so a systematic shift
        between plates ("batch effect") can be mistaken for chemistry signal.
        The check is a one-way ANOVA on the signed objective score across
        plates with >=2 observations each; a detected shift is reported as a
        warning, never silently corrected.
        """
        dataset = self._dataset()
        groups: dict[str, list[Observation]] = {}
        unassigned = 0
        for observation in dataset:
            plate = observation.meta.get("plate")
            if plate is None or str(plate) == "":
                unassigned += 1
                continue
            groups.setdefault(str(plate), []).append(observation)
        plates: list[dict] = []
        for name in sorted(groups, key=_plate_sort_key):
            observations = groups[name]
            scores = np.asarray(
                [o.score(self.objective.name, self.objective.direction) for o in observations],
                dtype=float,
            )
            entry: dict = {
                "plate": name,
                "n": len(observations),
                "mean_score": float(scores.mean()),
                "min_score": float(scores.min()),
                "max_score": float(scores.max()),
                "range_score": float(scores.max() - scores.min()),
                "std_score": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
            }
            values = [
                value
                for value in (self._native_value(observation) for observation in observations)
                if value is not None
            ]
            if values:
                arr = np.asarray(values, dtype=float)
                entry.update(
                    {
                        "mean_value": float(arr.mean()),
                        "min_value": float(arr.min()),
                        "max_value": float(arr.max()),
                        "range_value": float(arr.max() - arr.min()),
                        "std_value": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                    }
                )
            plates.append(entry)
        return {
            "n_plates": len(plates),
            "plates": plates,
            "unassigned": unassigned,
            "batch_effect": self._batch_effect(groups),
        }

    def _native_value(self, observation: Observation) -> float | None:
        if self.objective.name in observation.objectives:
            return float(observation.objectives[self.objective.name])
        return None if observation.y is None else float(observation.y)

    def _batch_effect(self, groups: Mapping[str, Sequence[Observation]]) -> dict:
        usable = []
        for name in sorted(groups, key=_plate_sort_key):
            scores = np.asarray(
                [
                    o.score(self.objective.name, self.objective.direction)
                    for o in groups[name]
                ],
                dtype=float,
            )
            if len(scores) >= 2:
                usable.append((name, scores))
        base: dict = {
            "detected": False,
            "plates_compared": len(usable),
            "n_used": int(sum(len(scores) for _, scores in usable)),
            "spread": None,
            "relative_spread": None,
            "pooled_std": None,
            "f": None,
            "p_value": None,
        }
        if len(usable) < 2:
            base["message"] = (
                "孔板/批次样本不足（需 ≥2 块板且每板 ≥2 次评价），暂不做批次效应检验"
            )
            return base
        means = np.asarray([scores.mean() for _, scores in usable], dtype=float)
        all_scores = np.concatenate([scores for _, scores in usable])
        overall = float(all_scores.mean())
        spread = float(means.max() - means.min())
        relative = spread / max(abs(overall), 1e-12)
        within_ss = float(sum(((scores - scores.mean()) ** 2).sum() for _, scores in usable))
        df_within = int(sum(len(scores) - 1 for _, scores in usable))
        between_ss = float(
            sum(len(scores) * (scores.mean() - overall) ** 2 for _, scores in usable)
        )
        k = len(usable)
        pooled_var = within_ss / df_within if df_within else 0.0
        pooled_std = pooled_var ** 0.5
        ms_between = between_ss / (k - 1)
        if pooled_var > 0:
            f_stat = ms_between / pooled_var
            p_value = _f_sf(f_stat, k - 1, df_within)
        else:
            f_stat = float("inf") if ms_between > 0 else 0.0
            p_value = 0.0 if ms_between > 0 else 1.0
        min_relative = float(self.meta.get("plate_effect_min_relative") or 0.02)
        detected = bool(p_value < 0.05 and relative >= min_relative)
        base.update(
            {
                "detected": detected,
                "spread": spread,
                "relative_spread": relative,
                "pooled_std": pooled_std,
                "f": f_stat,
                "p_value": p_value,
            }
        )
        if detected:
            base["message"] = (
                f"检测到批次/孔板效应：板间均值差 {spread:.4g}"
                f"（相对 {relative:.1%}，F={f_stat:.3g}，p={p_value:.3g}）；"
                "建议同批内比较，并把 plate 作为随机效应或特征记录"
            )
        else:
            base["message"] = (
                f"未检测到显著批次/孔板效应（板间均值差 {spread:.4g}，"
                f"相对 {relative:.1%}，p={p_value:.3g}）"
            )
        return base

    def plate_effects(self, metric: str | None = None) -> dict:
        """One-way random-effects estimates of plate/batch shifts.

        ``metric=None`` uses the scalar objective score; a metric name uses the
        raw objective/constraint value, which lets the recommendation correct
        every spec metric, not just the composite score. Returns BLUP-style
        shrunken plate effects plus variance components.
        """
        dataset = self._dataset()
        groups: dict[str, list[float]] = {}
        for observation in dataset:
            plate = observation.meta.get("plate")
            if plate is None or str(plate) == "":
                continue
            if metric is None:
                value = observation.y
            else:
                value = observation.objectives.get(metric)
                if value is None:
                    value = observation.constraints.get(metric)
            if value is None:
                continue
            groups.setdefault(str(plate), []).append(float(value))
        groups = {key: values for key, values in groups.items() if len(values) >= 2}
        base = {
            "detected": False,
            "effects": {},
            "sigma_between": 0.0,
            "within_var": 0.0,
            "f": None,
            "p_value": None,
            "relative_spread": None,
            "message": "plate samples are insufficient (need >=2 plates with >=2 observations each)",
        }
        if len(groups) < 2:
            return base
        sizes = np.asarray([len(values) for values in groups.values()], dtype=float)
        means = np.asarray([float(np.mean(values)) for values in groups.values()], dtype=float)
        all_values = np.concatenate([np.asarray(values, dtype=float) for values in groups.values()])
        grand = float(all_values.mean())
        n_total = len(all_values)
        k = len(groups)
        within_ss = float(
            sum(((np.asarray(values) - np.mean(values)) ** 2).sum() for values in groups.values())
        )
        df_within = int(n_total - k)
        within_var = within_ss / df_within if df_within > 0 else 0.0
        between_ss = float(sum(len(values) * (np.mean(values) - grand) ** 2 for values in groups.values()))
        ms_between = between_ss / (k - 1)
        if within_var > 0:
            f_stat = ms_between / within_var
            p_value = _f_sf(f_stat, k - 1, df_within) if df_within > 0 else 1.0
        else:
            f_stat = float("inf") if ms_between > 0 else 0.0
            p_value = 0.0 if ms_between > 0 else 1.0
        spread = float(means.max() - means.min())
        relative = spread / max(abs(grand), 1e-12)
        min_relative = float(self.meta.get("plate_effect_min_relative") or 0.02)
        detected = bool(p_value < 0.05 and relative >= min_relative)
        n0 = (n_total - float((sizes ** 2).sum()) / n_total) / (k - 1)
        sigma_between = max(0.0, (ms_between - within_var) / n0) if n0 > 0 else 0.0
        base.update({
            "detected": detected,
            "sigma_between": float(sigma_between),
            "within_var": float(within_var),
            "f": float(f_stat),
            "p_value": float(p_value),
            "relative_spread": float(relative),
            "message": (
                f"batch/plate effect detected: spread {spread:.4g} ({relative:.1%}, p={p_value:.3g})"
                if detected else
                f"no significant batch/plate effect (spread {spread:.4g}, p={p_value:.3g})"
            ),
        })
        if sigma_between > 0:
            for name, values in groups.items():
                n_i = len(values)
                denominator = n_i * sigma_between + within_var
                shrink = (n_i * sigma_between) / denominator if denominator > 0 else 0.0
                base["effects"][name] = float(shrink * (float(np.mean(values)) - grand))
        return base

    def _delivery_incumbent(self):
        """The observation a delivery is judged on.

        ``confirm()`` replicates the **recommended condition** at full fidelity, so the
        checklist must judge that same condition (not the raw best, and not another
        fidelity channel).  Otherwise multi-fidelity campaigns confirm one condition
        while the checklist looks at another one and reports "unconfirmed" forever.
        """
        try:
            condition = dict(self.recommend().get("condition") or {})
        except Exception:  # recommendation is advisory; fall back to the raw best
            condition = {}
        dataset = [
            obs
            for obs in self._dataset()
            if obs.y is not None and float(obs.fidelity) >= 1.0 - 1e-9
        ]
        if condition:
            key = self._encode_key(condition)
            at_condition = [obs for obs in dataset if self._encode_key(obs.x) == key]
            if at_condition:
                return max(
                    at_condition,
                    key=lambda obs: obs.score(
                        self.objective.name, self.objective.direction
                    ),
                )
        if dataset:
            return max(
                dataset,
                key=lambda obs: obs.score(self.objective.name, self.objective.direction),
            )
        return self.best()

    def _confirmation_stats(self, observation) -> dict | None:
        dataset = self._dataset()
        key = self._encode_key(observation.x)
        fidelity = float(getattr(observation, "fidelity", 1.0))
        scores = [
            float(item.y)
            for item in dataset
            if item.meta.get("purpose") == "confirmation"
            and item.y is not None
            and abs(float(item.fidelity) - fidelity) < 1e-9
            and self._encode_key(item.x) == key
        ]
        if not scores:
            return None
        mean = sum(scores) / len(scores)
        variance = sum((value - mean) ** 2 for value in scores) / max(len(scores) - 1, 1)
        return {"n": len(scores), "mean": mean, "std": variance ** 0.5, "scores": scores}

    def should_stop(
        self,
        patience: int | None = None,
        min_improvement: float = 1e-6,
        target: float | None = None,
        min_rounds: int = 3,
        max_rounds: int | None = None,
    ) -> tuple[bool, str]:
        """Decision-layer stopping rule (budget / target / plateau / max rounds)."""
        if self.remaining() == 0 and not self._pending:
            return True, "预算已用尽"
        best = self.best()
        threshold = target if target is not None else self.meta.get("target_score")
        if threshold is not None and best is not None:
            confirmation = self._confirmation_stats(best)
            score = confirmation["mean"] if confirmation is not None else best.y
            # A raw ``>=`` comparison only works for maximisation.  For a minimised
            # objective "reaching the target" means going *below* it; using the raw
            # value made an unattended loop stop immediately with the wrong reason
            # (measured: direction=min, target=0.1, best=5.0 -> "达到目标分数").
            direction = self.objective.direction_for(self.objective.name)
            reached = (
                score <= float(threshold) if direction == "min"
                else score >= float(threshold)
            )
            if reached:
                source = "确认复孔均值" if confirmation is not None else "当前最优"
                verb = "低于" if direction == "min" else "达到"
                return True, f"{verb}目标分数 {float(threshold):.6g}（{source} {score:.6g}）"
        limit = max_rounds if max_rounds is not None else self.meta.get("max_rounds")
        if limit is not None and len(self._rounds) >= int(limit):
            return True, f"达到最大轮次 {int(limit)}"
        scores = self._recent_best_scores()
        patience = self.patience if patience is None else int(patience)
        if self.meta.get("stop_on_stagnation", True) is False:
            # Budget-first mode: plateau is advisory (use retune_for_exploration to
            # keep going), not a stopping condition.
            return False, "继续优化"
        guard = self._stagnation_guard_rounds()
        if guard and len(self._rounds) < guard + patience + 1:
            return False, "继续优化（先验导入后的宽限期：先积累本地观测再判断平台期）"
        if scores and len(scores) >= max(min_rounds, patience + 1):
            window = scores[-(patience + 1):]
            if window[-1] - window[0] <= min_improvement:
                return True, f"最近 {patience} 轮无显著改进（{window[0]:.6g} → {window[-1]:.6g}）"
        return False, "继续优化"

    def _is_feasible(self, observation) -> bool:
        """True **only** when every declared constraint was measured and satisfied.

        Delegates to :mod:`condopt.feasibility` so a missing measurement is classified
        as ``UNKNOWN`` (and therefore not feasible) by exactly the same rule used by
        the benchmarks, the calibration baseline and the reports (A12).
        """
        from .feasibility import ConstraintState, feasibility_state

        report = feasibility_state(observation, self.objective)
        return report["state"] == ConstraintState.TRUE.value

    def feasibility_report(self) -> dict:
        """Three-state feasibility summary of the recorded dataset."""
        from .feasibility import coverage_note, feasible_count

        dataset = self._dataset()
        summary = feasible_count(dataset, self.objective)
        summary["coverage"] = coverage_note(dataset, self.objective)
        return summary

    @staticmethod
    def _stop_category(reason: str) -> str:
        """Coarse machine-readable class for a stop reason (agent-facing)."""
        if not reason:
            return "continue"
        if "成本预算" in reason:
            return "cost_exhausted"
        if "预算已用尽" in reason:
            return "budget_exhausted"
        if "达到目标分数" in reason:
            return "target_reached"
        if "最大轮次" in reason:
            return "max_rounds"
        if "无显著改进" in reason:
            return "stagnation"
        if reason == "继续优化":
            return "continue"
        return "other"

    def _pareto_front_observations(self) -> list:
        dataset = self._dataset()
        if len(dataset) == 0:
            return []
        result = pareto_summarise(dataset, self.objective)
        return [dataset[index] for index in result.indices]

    def _objective_plate_effects(self) -> dict:
        detected: dict[str, dict] = {}
        for name in self.objective.objectives:
            effects = self.plate_effects(name)
            if effects.get("detected") and effects.get("effects"):
                detected[name] = effects
        return detected

    def _adjusted_objective_values(self, observation, metric_effects: dict) -> dict:
        plate = str(observation.meta.get("plate") or "")
        values: dict[str, float] = {}
        for name, value in observation.objectives.items():
            if value is None:
                continue
            effect = metric_effects.get(name)
            shift = float(effect["effects"].get(plate, 0.0)) if effect else 0.0
            values[name] = float(value) - shift
        return values

    def _adjusted_all_values(self, observation, metric_effects: dict) -> dict:
        values = self._adjusted_objective_values(observation, metric_effects)
        plate = str(observation.meta.get("plate") or "")
        for name, value in observation.constraints.items():
            if value is None:
                continue
            effect = self.plate_effects(name)
            shift = float(effect["effects"].get(plate, 0.0)) if effect.get("detected") else 0.0
            values[name] = float(value) - shift
        return values

    def _multiobjective_recommendation(self, raw_best) -> dict:
        """Recommend a Pareto knee; apply plate correction when detected."""
        pareto = self.pareto_summary()
        metric_effects = self._objective_plate_effects()
        names = self.objective.objectives
        signs = np.asarray(
            [1.0 if self.objective.direction_for(name) == "max" else -1.0 for name in names]
        )
        dataset = self._dataset()
        rows = []
        row_observations = []
        row_values = []
        for observation in dataset:
            if metric_effects:
                adjusted = self._adjusted_objective_values(observation, metric_effects)
            else:
                adjusted = {name: observation.objectives.get(name) for name in names}
            if any(adjusted.get(name) is None for name in names):
                continue
            rows.append([float(adjusted[name]) for name in names])
            row_observations.append(observation)
            row_values.append(adjusted)
        # Multi-fidelity: biased low-fidelity points must not win the Pareto
        # recommendation, mirroring ``Study.best`` (full-fidelity incumbents).
        fidelity_filter = "all"
        full_positions = [
            index for index, observation in enumerate(row_observations)
            if float(observation.fidelity) >= 1.0 - 1e-9
        ]
        if full_positions:
            rows = [rows[index] for index in full_positions]
            row_observations = [row_observations[index] for index in full_positions]
            row_values = [row_values[index] for index in full_positions]
            fidelity_filter = "full"
        if not rows:
            stop, reason = self.should_stop()
            return {
                "run_dir": str(self.dir),
                "condition": dict(raw_best.x),
                "score": raw_best.y,
                "best_score": raw_best.y,
                "selection_basis": "raw",
                "selection_reason": "按原始分数选最优条件",
                "raw_best": {"condition": dict(raw_best.x), "score": raw_best.y},
                "plate_effects": {},
                "objectives": dict(raw_best.objectives),
                "constraints": dict(raw_best.constraints),
                "spec": None,
                "confirmation": self._confirmation_stats(raw_best),
                "should_stop": stop,
                "stop_reason": reason,
                "pareto": pareto,
                "fidelity_filter": "all",
            }
        matrix = np.asarray(rows, dtype=float) * signs
        mask = non_dominated_mask(matrix)
        front_indices = np.flatnonzero(mask)
        front_scores = matrix[mask]
        spans = np.maximum(front_scores.max(axis=0) - front_scores.min(axis=0), 1e-12)
        normalized = (front_scores - front_scores.min(axis=0)) / spans
        configured = self.meta.get("knee_weights") or {}
        weight_vector = np.asarray([float(configured.get(name, 1.0)) for name in names])
        distances = np.sum(weight_vector * ((1.0 - normalized) ** 2), axis=1)
        best_position = int(np.argmin(distances))
        selected_index = int(front_indices[best_position])
        selected = row_observations[selected_index]
        selected_adjusted = row_values[selected_index]
        basis = "pareto_knee_plate_adjusted" if metric_effects else "pareto_knee"
        values = self._adjusted_all_values(selected, metric_effects)
        compliance = None

        # If a spec exists, prefer a front member that satisfies it; among
        # in-spec members choose the best spec score (the knee is the fallback).
        if self.spec is not None:
            candidates = []
            for index in front_indices:
                observation = row_observations[int(index)]
                candidate_values = self._adjusted_all_values(observation, metric_effects)
                if not all(goal.metric in candidate_values for goal in self.spec.goals):
                    continue
                candidate_compliance = self.spec.compliance(candidate_values)
                candidate_score = self.spec.score(candidate_values)
                candidates.append(
                    (candidate_score, candidate_compliance["in_spec"], int(index), candidate_values)
                )
            if candidates:
                in_spec = [candidate for candidate in candidates if candidate[1]]
                pool = in_spec or candidates
                chosen = max(pool, key=lambda candidate: candidate[0])
                selected_index = chosen[2]
                selected = row_observations[selected_index]
                selected_adjusted = row_values[selected_index]
                values = chosen[3]
                basis = "pareto_spec_plate_adjusted" if metric_effects else "pareto_spec"

        if self.spec is not None and all(goal.metric in values for goal in self.spec.goals):
            compliance = self.spec.compliance(values)
        raw_values = dict(selected.objectives)
        raw_values.update(selected.constraints)
        compliance_raw = None
        if self.spec is not None and all(goal.metric in raw_values for goal in self.spec.goals):
            compliance_raw = self.spec.compliance(raw_values)
        stop, reason = self.should_stop()
        reason_text = {
            "pareto_knee": "Pareto 前沿上离 utopia 最近的 knee",
            "pareto_knee_plate_adjusted": "板效应校正后的 Pareto knee",
            "pareto_spec": "Pareto 前沿上 spec 最优",
            "pareto_spec_plate_adjusted": "板效应校正后的 Pareto 前沿上 spec 最优",
        }.get(basis, "Pareto 前沿推荐")
        if fidelity_filter == "full":
            reason_text += "（仅用全保真点，低保真点不参与推荐）"
        return {
            "run_dir": str(self.dir),
            "condition": dict(selected.x),
            "score": selected.y,
            "best_score": raw_best.y,
            "selection_basis": basis,
            "selection_reason": reason_text,
            "raw_best": {
                "condition": dict(raw_best.x),
                "score": raw_best.y,
            },
            "plate_effects": {
                "detected": bool(metric_effects),
                "metrics": metric_effects,
            },
            "objectives": dict(selected.objectives),
            "objectives_adjusted": selected_adjusted if metric_effects else None,
            "constraints": dict(selected.constraints),
            "spec": compliance,
            "spec_raw": compliance_raw,
            "confirmation": self._confirmation_stats(selected),
            "should_stop": stop,
            "stop_reason": reason,
            "pareto": pareto,
            "adjusted_pareto_n_front": int(len(front_indices)),
            "fidelity_filter": fidelity_filter,
        }

    def _detected_metric_effects(self) -> dict:
        names = list(self.spec.metrics) + list(self.spec.constraints) if self.spec is not None else []
        effects: dict[str, dict] = {}
        for name in names:
            metric_effect = self.plate_effects(name)
            if metric_effect.get("detected") and metric_effect.get("effects"):
                effects[name] = metric_effect
        return effects

    def _adjusted_score(self, observation, metric_effects: dict) -> float:
        """Plate-corrected score: recompute from adjusted metrics when possible."""
        if self.spec is not None and metric_effects:
            values = self._adjusted_all_values(observation, metric_effects)
            if all(goal.metric in values for goal in self.spec.goals):
                return float(self.spec.score(values))
        effect = self.plate_effects()
        plate = str(observation.meta.get("plate") or "")
        shift = float(effect["effects"].get(plate, 0.0)) if effect.get("detected") else 0.0
        return float(observation.y) - shift

    def recommend(self) -> dict:
        """Executable recommendation: condition, spec compliance, confirmation.

        If a plate/batch effect is detected, select by the plate-adjusted score
        (random-effects BLUP); spec compliance is then evaluated on
        plate-adjusted metric values, and the raw compliance is kept for audit.
        """
        raw_best = self.best()
        if raw_best is None:
            raise ValueError("no observation to recommend yet")
        if self.objective.is_multi_objective:
            return self._multiobjective_recommendation(raw_best)
        effects = self.plate_effects()
        metric_effects = self._detected_metric_effects()
        selected = raw_best
        selected_adjusted = None
        basis = "raw"
        if (effects["detected"] and effects["effects"]) or metric_effects:
            dataset = self._dataset()
            full = [obs for obs in dataset if obs.fidelity >= 1.0 - 1e-9]
            candidates = full or dataset
            ranked = []
            for observation in candidates:
                if observation.y is None:
                    continue
                adjusted = self._adjusted_score(observation, metric_effects)
                ranked.append((adjusted, observation))
            if ranked:
                selected_adjusted, selected = max(ranked, key=lambda item: item[0])
                basis = "plate_adjusted"

        raw_values = dict(selected.objectives)
        raw_values.update(selected.constraints)
        compliance_raw = None
        if self.spec is not None and all(goal.metric in raw_values for goal in self.spec.goals):
            compliance_raw = self.spec.compliance(raw_values)
        compliance = compliance_raw
        if basis == "plate_adjusted" and self.spec is not None:
            adjusted_values = dict(raw_values)
            plate = str(selected.meta.get("plate") or "")
            for name in list(self.spec.metrics) + list(self.spec.constraints):
                if name not in raw_values:
                    continue
                metric_effects = self.plate_effects(name)
                shift = float(metric_effects["effects"].get(plate, 0.0))
                adjusted_values[name] = float(raw_values[name]) - shift
            if all(goal.metric in adjusted_values for goal in self.spec.goals):
                compliance = self.spec.compliance(adjusted_values)

        stats = self._confirmation_stats(selected)
        stop, reason = self.should_stop()
        reported_score = stats["mean"] if stats is not None else selected.y
        selection_reason = (
            "按逐指标板效应校正后的 spec_score 选最优条件"
            if basis == "plate_adjusted"
            else "按原始分数选最优条件"
        )
        return {
            "run_dir": str(self.dir),
            "condition": dict(selected.x),
            "score": reported_score,
            "best_score": selected.y,
            "adjusted_score": selected_adjusted,
            "selection_basis": basis,
            "selection_reason": selection_reason,
            "raw_best": {
                "condition": dict(raw_best.x),
                "score": raw_best.y,
                "plate": raw_best.meta.get("plate"),
            },
            "plate_effects": effects,
            "objectives": dict(selected.objectives),
            "constraints": dict(selected.constraints),
            "spec": compliance,
            "spec_raw": compliance_raw,
            "confirmation": stats,
            "should_stop": stop,
            "stop_reason": reason,
        }

    def confirm(self, n: int = 3) -> list[dict]:
        """Create a confirmation batch that replicates the current best condition."""
        if self._pending:
            raise PendingError("confirm() needs an empty pending batch; call tell first")
        if n < 1:
            raise ValueError("n must be >= 1")
        if n > self.remaining():
            raise ValueError(f"only {self.remaining()} evaluations left in the budget")
        # Confirmation runs are full-fidelity, so a cost budget caps them too
        # (otherwise the campaign can overshoot the recorded cost budget).
        cost_cap = self.remaining_cost()
        if cost_cap is not None:
            affordable = int(cost_cap // 1.0 + 1e-9)
            if affordable <= 0:
                self.log_decision(
                    action="cost_budget_trim",
                    algorithm=self.algorithm,
                    reason=(
                        f"剩余成本预算 {cost_cap:.3f} 不足一次全保真确认，跳过 confirm"
                    ),
                    payload={"requested": n, "returned": 0},
                )
                return []
            if affordable < n:
                self.log_decision(
                    action="cost_budget_trim",
                    algorithm=self.algorithm,
                    reason=(
                        f"剩余成本预算 {cost_cap:.3f} 只够 {affordable} 次全保真确认，"
                        f"confirm 从 {n} 缩减为 {affordable}"
                    ),
                    payload={"requested": n, "returned": affordable},
                )
                n = affordable
        # Confirm the *recommended* condition, not the raw best: after a
        # plate-effect correction the recommendation may be a different well.
        recommendation = self.recommend()
        condition = dict(recommendation["condition"])
        confirmation_basis = "recommendation"
        # Constrained + multi-fidelity: if every full-fidelity run so far was
        # infeasible, confirming the (possibly cheap) recommendation again may
        # repeat that mistake.  Prefer the best condition that is at least known
        # to be feasible, so the expensive confirmation buys usable evidence.
        if getattr(self.objective, "constraints", ()):
            dataset = self._dataset()
            has_feasible_full = any(
                observation.fidelity >= 1.0 - 1e-9 and self._is_feasible(observation)
                for observation in dataset
            )
            if not has_feasible_full:
                feasible_observations = [
                    observation for observation in dataset if self._is_feasible(observation)
                ]
                if feasible_observations:
                    best_feasible = max(
                        feasible_observations,
                        key=lambda observation: observation.score(
                            self.objective.name, self.objective.direction
                        ),
                    )
                    condition = dict(best_feasible.x)
                    confirmation_basis = "best_feasible_observed"
        batch_id = len(self._rounds) + 1
        plate_id = self._resolve_plate(None)
        wells = self._assign_wells(plate_id, n)
        pending = []
        for index, well in enumerate(wells):
            pending.append({
                "id": f"r{batch_id}-{index + 1}",
                "batch": batch_id,
                "plate": plate_id,
                "well": well,
                "algorithm": "confirmation",
                "purpose": "confirmation",
                "x": dict(condition),
            })
        self._pending = pending
        self._save_all()
        self._write_suggestions(pending)
        self.log_decision(
            action="confirmation_batch",
            algorithm=self.algorithm,
            params=self.params,
            reason=f"对当前最优条件做 {n} 次确认复孔（规格层要求）",
            payload={
                "n": n,
                "condition": dict(condition),
                "confirmation_basis": confirmation_basis,
                "selection_basis": recommendation.get("selection_basis"),
                "best_score": recommendation.get("best_score"),
            },
        )
        self._save_all()
        return pending

    # -- automatic loop ---------------------------------------------------
    def next_action(self) -> dict:
        """One machine-readable decision for an agent-driven optimization loop.

        Combines pending-state, :meth:`should_stop`, :meth:`recommend` and
        :meth:`confirm` into a single ``action`` so the skill never has to
        guess "what now".  Actions:

        ``tell``      results are missing for the current batch;
        ``confirm``   the best condition should be replicated before declaring
                      a result (target/plateau reached or budget nearly spent);
        ``ask``       continue optimizing;
        ``stop``      budget/target/plateau reached and (if applicable) the
                      best condition has already been confirmed.

        ``meta["auto_confirm"] = False`` disables the confirmation detour;
        ``meta["confirm_n"]`` sets how many replicates are requested (default 3).
        """
        stop, stop_reason = self.should_stop()
        stop_category = self._stop_category(stop_reason)
        remaining_cost = self.remaining_cost()
        cost_exhausted = remaining_cost is not None and remaining_cost <= 0
        if not stop and cost_exhausted:
            stop, stop_reason = True, "成本预算已用尽（remaining_cost <= 0）"
            stop_category = "cost_exhausted"
        best = self.best()
        recommendation = self.recommend() if best is not None else None
        confirmation = recommendation["confirmation"] if recommendation else None
        remaining = self.remaining()
        # Scale the confirmation batch with the campaign size: spending 3 of 12
        # evaluations (25%) on replicates starves exploration, while a large
        # campaign can afford the full default of 3.
        requested_confirm = self.meta.get("confirm_n")
        if requested_confirm:
            base_confirm = int(requested_confirm)
        else:
            base_confirm = default_confirm_n(self.budget)
        confirm_n = max(1, min(base_confirm, remaining))
        auto_confirm = bool(self.meta.get("auto_confirm", True))
        # Reserve budget for the confirmation replicates so an ask batch cannot
        # consume the whole remaining budget before the incumbent is confirmed.
        # Reserve the confirmation budget for whatever the *final* incumbent turns
        # out to be.  Keying the reserve on "the current best has no confirmation"
        # was not enough: after confirming an earlier incumbent the loop could spend
        # the last batch on a new (better) point, which then ended the campaign
        # unconfirmed -- the delivery checklist caught exactly that in the e2e drill.
        # Keep a slot to confirm whatever the next batch produces: while the current
        # incumbent is unconfirmed that is the full confirmation batch, and once it is
        # confirmed a single replicate is enough to validate a *new* winner
        # (total >= 2 measurements + >= 1 tagged replicate).
        # In budget-first campaigns the loop will keep exploring until the budget is
        # gone, so a single slot is not enough: a *new* winner would only get one
        # replicate.  Reserve the full confirmation batch there, otherwise those runs
        # ended with an unconfirmed delivery (measured: delivery 0/2 at budget 90/120).
        budget_first = self.meta.get("stop_on_stagnation", True) is False
        delivery_slots = confirm_n if confirmation is None else 1
        reserve = (
            delivery_slots if (auto_confirm and best is not None and not stop) else 0
        )
        # Keep the reserve unless nothing can be done with it: zeroing it as soon as
        # exploration space ran out made budget-first campaigns burn the confirmation
        # batch on one more exploration point (delivery 0/2 at budget 90/120).
        if remaining <= 1:
            reserve = 0  # nothing left to protect: confirm now or finish
        # A single lucky measurement is not a result: replicate the incumbent
        # before stopping, and validate it when the budget is about to run out.
        # A confirmation batch runs at full fidelity; if the remaining cost
        # budget cannot pay for even one such run, asking for it would trim to an
        # empty batch and the caller would spin forever.
        remaining_cost = self.remaining_cost()
        cost_allows_confirmation = (
            remaining_cost is None or float(remaining_cost) + 1e-9 >= 1.0
        )
        needs_confirmation = bool(
            auto_confirm
            and best is not None
            and confirmation is None
            and best.n_replicates < 2
            and remaining >= 1
            and cost_allows_confirmation
            and (stop or remaining <= confirm_n or remaining - reserve <= confirm_n)
        )
        # Delivery rule: never spend a batch that could *improve* the incumbent when
        # the remaining budget could not then confirm the new winner.  The final best
        # stayed unconfirmed in the e2e drill exactly because of such a last batch.
        end_game_floor = 1
        if (
            auto_confirm
            and best is not None
            and not stop
            and remaining <= end_game_floor
            and confirmation is not None
        ):
            stop = True
            stop_reason = reason = (
                f"仅剩 {remaining} 次，不足以「探索一轮 + 确认新最优」，直接收尾交付"
            )
            stop_category = self._stop_category(stop_reason)
        if self._pending:
            action = "tell"
            reason = f"有 {len(self._pending)} 条建议待回填实测结果，先 tell 再继续"
        elif stop and (not needs_confirmation or cost_exhausted):
            action, reason = "stop", stop_reason
        elif (
            auto_confirm
            and best is not None
            and confirmation is None
            and not cost_allows_confirmation
        ):
            action = "stop"
            stop = True
            stop_reason = reason = "成本预算不足以完成全保真确认"
            stop_category = self._stop_category(stop_reason)
        elif needs_confirmation:
            action = "confirm"
            reason = (
                f"收尾前确认最优条件（{confirm_n} 次复孔）：{stop_reason}"
                if stop
                else f"预算仅剩 {remaining} 次，先确认当前最优条件（{confirm_n} 次复孔）"
            )
        else:
            action, reason = "ask", "继续优化"
        dataset = self._dataset()
        fidelity_info = self.fidelity_summary()
        plate_effect = self.plate_effects()
        batch_effect = self.plate_summary()["batch_effect"]
        spec_in_spec_count = (
            sum(1 for obs in dataset if (obs.meta.get("spec") or {}).get("in_spec"))
            if self.spec is not None
            else None
        )
        metric_effects = self._detected_metric_effects() if self.spec is not None else {}
        recommendation_plate = bool(metric_effects) or (
            bool((recommendation.get("plate_effects") or {}).get("detected"))
            if recommendation is not None
            else False
        )
        spec_in_spec_adjusted = None
        if self.spec is not None:
            adjusted_count = 0
            for observation in dataset:
                values = self._adjusted_all_values(observation, metric_effects)
                if all(goal.metric in values for goal in self.spec.goals):
                    if self.spec.compliance(values)["in_spec"]:
                        adjusted_count += 1
            spec_in_spec_adjusted = adjusted_count
        plate_counts: dict[str, int] = {}
        for observation in dataset:
            plate_name = observation.meta.get("plate")
            if plate_name:
                plate_counts[str(plate_name)] = plate_counts.get(str(plate_name), 0) + 1
        plate_groups_ready = sum(1 for count in plate_counts.values() if count >= 2)
        diagnostics = {
            "batch_effect": batch_effect,
            "plate_groups_ready": plate_groups_ready,
            "plate_effect": plate_effect,
            "plate_effect_detected": recommendation_plate or bool(plate_effect.get("detected")),
            "selection_basis": recommendation["selection_basis"] if recommendation else None,
            "recommendation_spec": recommendation["spec"] if recommendation else None,
            "knee_weights": dict(self.meta.get("knee_weights") or {}),
            "forked_from": self.meta.get("forked_from"),
            "fork_spec_mode": self.meta.get("fork_spec_mode"),
            "spec_in_spec_count": spec_in_spec_count,
            "spec_in_spec_adjusted": spec_in_spec_adjusted,
            "spec_calibration_ready": bool(self.spec is not None and len(dataset) >= 3),
            "multi_fidelity": "multi_fidelity" in registry.spec(self.algorithm).capabilities,
            "multi_fidelity_context": bool(getattr(self.context, "multi_fidelity", False)),
            "multi_fidelity_supported": (
                "multi_fidelity" in registry.spec(self.algorithm).capabilities
            ),
            "fidelity_levels": sorted(fidelity_info["levels"]),
            "fidelity_paired_n": fidelity_info["paired"]["n"],
            "fidelity_normalized_bias": fidelity_info["normalized_bias"],
            "n_full_fidelity": fidelity_info["n_full_fidelity"],
            "observed_feasible_fraction": (
                round(
                    sum(1 for observation in dataset if self._is_feasible(observation))
                    / len(dataset),
                    4,
                )
                if (getattr(self.objective, "constraints", ()) and len(dataset) > 0)
                else None
            ),
            "n_feasible_full_fidelity": sum(
                1
                for observation in dataset
                if observation.fidelity >= 1.0 - 1e-9 and self._is_feasible(observation)
            ),
            "n_descriptor_vars": sum(
                1 for variable in self.space.variables if variable.descriptors
            ),
            "constraints_supported": (
                "constraints" in registry.spec(self.algorithm).capabilities
                if getattr(self.objective, "constraints", ())
                else None
            ),
            "cost_spent": self.cost_spent(),
            "remaining_cost": self.remaining_cost(),
            "stop_reason": stop_reason,
            "stop_category": stop_category,
            "remaining_evals": remaining,
        }
        if recommendation is not None:
            raw_spec = recommendation.get("spec_raw")
            adjusted_spec = recommendation.get("spec")
            raw_in_spec = None if raw_spec is None else raw_spec.get("in_spec")
            adjusted_in_spec = None if adjusted_spec is None else adjusted_spec.get("in_spec")
            diagnostics["spec_check"] = {
                "raw_in_spec": raw_in_spec,
                "adjusted_in_spec": adjusted_in_spec,
                "changed_by_plate": (
                    raw_in_spec is not None
                    and adjusted_in_spec is not None
                    and raw_in_spec != adjusted_in_spec
                ),
            }
        suggested: list[str] = []
        if self._pending:
            suggested.append("tell")
        feasible_fraction = diagnostics.get("observed_feasible_fraction")
        if (
            feasible_fraction is not None
            and feasible_fraction < 0.05
            and len(dataset) >= 10
        ):
            # Evidence: with a ~0.4% feasible box, constraint-aware search found a
            # feasible point in only 1/5 runs; encoding the same limits as variable
            # bounds succeeded 5/5. Recommend the engineering fix early.
            suggested.append("encode_limits_as_bounds")
        if (
            diagnostics.get("n_full_fidelity", 0) > 0
            and diagnostics.get("n_feasible_full_fidelity") == 0
            and getattr(self.objective, "constraints", ())
        ):
            # Expensive runs happened, but none satisfied the constraints, so the
            # delivered answer has no full-fidelity feasible evidence.
            suggested.append("collect_feasible_full_fidelity")
        if (
            action == "stop"
            and stop_category == "stagnation"
            and remaining >= max(1, self.batch_size)
        ):
            # The optimizer stopped because the last rounds plateaued, but the
            # budget is not spent: an operator/agent may prefer to keep exploring
            # (possibly with a different algorithm) instead of declaring victory.
            suggested.append("consider_exploration_restart")
        if plate_effect.get("detected") or recommendation_plate:
            suggested += ["replicate_on_new_plate", "use_plate_adjusted_recommendation"]
        if diagnostics["spec_calibration_ready"]:
            suggested.append("calibrate_spec")
            calibrated, report = self.spec.calibrate_observations(dataset.observations)
            diagnostics["spec_calibration"] = {
                "report": report,
                "suggested_goals": [goal.to_dict() for goal in calibrated.goals],
            }
        if self.spec is not None and plate_groups_ready >= 2:
            suggested.append("calibrate_by_plate")
        if (
            spec_in_spec_count == 0
            and (spec_in_spec_adjusted or 0) == 0
            and self.spec is not None
            and len(dataset) >= 5
        ):
            # No observed point satisfies the spec: likely infeasible / mis-specified.
            suggested.append("review_spec")
            records = [self.spec.observation_values(observation) for observation in dataset]
            diagnostics["spec_review"] = self.spec.achievability(records)
            review_groups = (
                [str(observation.meta.get("plate") or "?") for observation in dataset]
                if plate_groups_ready >= 2
                else None
            )
            suggested_spec, suggestion = self.spec.suggest_spec(records, groups=review_groups)
            diagnostics["spec_review"]["suggested_spec"] = suggested_spec.to_dict()
            diagnostics["spec_review"]["suggestion"] = suggestion
        remaining_cost = diagnostics.get("remaining_cost")
        if remaining_cost is not None and remaining_cost < 1.0:
            suggested.append("review_cost_budget")
        if diagnostics.get("constraints_supported") is False:
            suggested.append("switch_to_constraint_aware")
        if (
            diagnostics.get("multi_fidelity_context")
            and not diagnostics.get("multi_fidelity_supported")
        ):
            suggested.append("switch_to_multi_fidelity")
        if (
            diagnostics.get("multi_fidelity_context")
            and len(dataset) >= 4
            and int(diagnostics.get("n_full_fidelity") or 0) == 0
        ):
            suggested.append("collect_full_fidelity")
        normalized_bias = diagnostics.get("fidelity_normalized_bias")
        if (
            normalized_bias is not None
            and abs(float(normalized_bias)) >= 0.5
            and diagnostics.get("fidelity_paired_n", 0) >= 2
        ):
            suggested.append("review_fidelity_channel")
        if action == "ask":
            suggested.append("ask_next_batch")
        ask_n = (
            min(int(self.batch_size), max(1, remaining - reserve)) if action == "ask" else None
        )
        action_reasons: dict[str, str] = {}
        if "tell" in suggested:
            action_reasons["tell"] = f"{len(self._pending)} 条建议待回填实测结果"
        if "replicate_on_new_plate" in suggested:
            action_reasons["replicate_on_new_plate"] = "检测到板/批次效应，建议换板复孔以分离批次"
        if "use_plate_adjusted_recommendation" in suggested:
            action_reasons["use_plate_adjusted_recommendation"] = (
                f"推荐依据为 {diagnostics.get('selection_basis')}，已按板效应校正"
            )
        if "calibrate_spec" in suggested:
            action_reasons["calibrate_spec"] = f"已有 {len(dataset)} 条观测，可用历史校准 scale/tolerance"
        if "calibrate_by_plate" in suggested:
            action_reasons["calibrate_by_plate"] = (
                f"{plate_groups_ready} 个板各有 ≥2 条观测，可做板内校准以剔除板间偏移"
            )
        if "review_spec" in suggested:
            action_reasons["review_spec"] = f"{len(dataset)} 条观测仍 0 个 in-spec，规格可能不可达或写错"
        if (
            int(diagnostics.get("n_descriptor_vars") or 0) > 0
            and self.algorithm in ("tpe",)
        ):
            suggested.append("review_algorithm_for_descriptors")
        if "encode_limits_as_bounds" in suggested:
            action_reasons["encode_limits_as_bounds"] = (
                f"已观测可行率仅 {feasible_fraction:.1%}：可行域过小，"
                "建议把已知硬限制写进变量范围/固定因子，而不是继续用约束采集硬找"
                "（实测 ~0.4% 可行域：约束搜索 1/5 命中，写进范围 5/5；"
                "见 research/HARD_FEASIBILITY.md）"
            )
        if "collect_feasible_full_fidelity" in suggested:
            action_reasons["collect_feasible_full_fidelity"] = (
                "已有全保真观测但都不满足约束：交付缺少可行的全保真证据。"
                "建议对当前可行的低保真最优条件做全保真确认（或 confirm 该条件），"
                "而不是继续在半可行区域消耗昂贵实验"
            )
        checklist = self.delivery_checklist()
        missing_delivery = [item for item in checklist if not item["ok"]]
        diagnostics["delivery_missing"] = [
            {"item": item["item"], "detail": item["detail"]} for item in missing_delivery
        ]
        if missing_delivery and not stop:
            suggested.append("complete_delivery_evidence")
            action_reasons["complete_delivery_evidence"] = (
                "交付证据不完整："
                + "；".join(f"{item['item']}（{item['detail']}）" for item in missing_delivery)
                + "。先把缺的证据补齐再谈交付"
            )

        if "consider_exploration_restart" in suggested:
            action_reasons["consider_exploration_restart"] = (
                f"因停滞停止（{stop_reason}）但仍剩 {remaining} 次预算。"
                "实测（rosenbrock10，budget 60，3 seeds）：`retune_for_exploration()` −205.4 "
                "优于原样继续 −319.8，优于 sobol 重启 −373.7（与停止相同）；"
                "优先级：调参重调 > 继续当前算法 > 探索重启"
            )
        if "review_algorithm_for_descriptors" in suggested:
            action_reasons["review_algorithm_for_descriptors"] = (
                "空间含描述符特征但当前算法是 tpe；Tier C 实测 TPE 在描述符下明显变差，"
                "建议改用 bayes/constrained/parego 等 GP 类算法（或移除描述符）"
            )
        if "collect_full_fidelity" in suggested:
            action_reasons["collect_full_fidelity"] = (
                "已有多条观测但没有任何全保真点：推荐与结论缺少全保真证据，"
                "建议至少补一次全保真（或提高 min_full_fraction）"
            )
        if "review_fidelity_channel" in suggested:
            action_reasons["review_fidelity_channel"] = (
                f"同一条件下全保真与低保真平均偏差为观测跨度的 "
                f"{float(normalized_bias):.0%}；低保真可能带系统偏置，建议加测全保真或改用全保真"
            )
        if "switch_to_multi_fidelity" in suggested:
            action_reasons["switch_to_multi_fidelity"] = (
                f"上下文声明多保真，但当前算法 {self.algorithm} 未声明 multi_fidelity 能力；"
                "建议 override 到 mfbo 或关闭多保真上下文"
            )
        if "switch_to_constraint_aware" in suggested:
            action_reasons["switch_to_constraint_aware"] = (
                f"当前算法 {self.algorithm} 未声明 constraints 能力，硬约束可能被忽略；"
                "建议 override 到 constrained/cqehvi"
            )
        if "review_cost_budget" in suggested:
            action_reasons["review_cost_budget"] = (
                f"资源预算仅剩 {remaining_cost:g}，建议降低保真度或追加成本预算"
            )
        if "ask_next_batch" in suggested:
            action_reasons["ask_next_batch"] = f"剩余预算 {remaining} 次，继续优化"
        payload: dict = {
            "action": action,
            "reason": reason,
            "should_stop": stop,
            "stop_reason": stop_reason,
            "used": len(dataset),
            "remaining": remaining,
            "pending": [item["id"] for item in self._pending],
            "confirm_n": confirm_n if action == "confirm" else None,
            "ask_n": ask_n,
            "recommendation": recommendation,
            "batch_effect": batch_effect,
            "diagnostics": diagnostics,
            "suggested_actions": suggested,
            "suggested_action_reasons": action_reasons,
        }
        payload["next_command"] = self._command_for(action, confirm_n, ask_n)
        return payload

    def _command_for(self, action: str, confirm_n: int, ask_n: int | None = None) -> str:
        run = self.dir
        if action == "tell":
            return f"condopt tell --run {run} --results <results.json|results.jsonl>"
        if action == "confirm":
            return f"condopt confirm --run {run} --n {confirm_n}"
        if action == "ask":
            n = ask_n if ask_n is not None else self.batch_size
            return f"condopt ask --run {run} --n {n}"
        return f"condopt recommend --run {run} --json"

    def transfer_staged(
        self,
        source: str | Path,
        start_top_k: int = 1,
        wide_top_k: int = 5,
        after_batches: int = 2,
        min_reliability: float = 0.6,
        prior_reliability: float = 1.0,
        force: bool = False,
        reason: str = "staged transfer",
    ) -> dict:
        """Import a few priors first, widen only after the fresh local evidence agrees.

        Measured (``research/TRANSFER_ADAPTIVE.md``): importing 1 prior, running two
        local batches, then re-running :meth:`assess_transfer` and widening only when it
        does not veto gives +0.71 on a biased source where a fixed ``top_k=5`` gives
        -0.33 and ``top_k=3`` gives -1.54; on a good source it matches the fixed policy.
        ``min_reliability`` is the veto line (0.4-0.6 measured safe; below 0.4 the
        protection disappears, at >=0.8 even good sources are rejected).

        Call it on every iteration; it acts once, when enough local batches exist.
        """
        source_study = source if isinstance(source, Study) else Study.load(source)
        source_path = source_study.dir if isinstance(source, Study) else Path(source)
        imported_at = int(getattr(self, "_priors_imported_at_round", 0) or 0)
        local_batches = len(self._rounds) - imported_at
        already = any(
            observation.meta.get("transferred_from") for observation in self._dataset()
        )
        report: dict = {
            "action": "transfer_staged",
            "source": str(source_path),
            "local_batches_since_import": local_batches,
            "needed_batches": after_batches,
        }
        if not force:
            if local_batches < after_batches and not already:
                # First call: the small first look.
                import_report = self.transfer_from(
                    source_path, top_k=start_top_k,
                    prior_reliability=prior_reliability,
                    reason=f"{reason}: first look (top_k={start_top_k})",
                )
                report.update({
                    "stage": "first_look",
                    "imported": import_report.get("imported"),
                    "widened": False,
                })
                return report
            if local_batches < after_batches:
                report.update({
                    "stage": "waiting",
                    "imported": 0,
                    "widened": False,
                    "ask": f"再跑 {after_batches - local_batches} 批本地观测再评估是否放大",
                })
                return report

        audit = self.assess_transfer(source_path)
        suggestion = audit.get("suggested_reliability")
        report["audit"] = {
            "ready": audit.get("ready"),
            "suggested_reliability": suggestion,
            "confidence": audit.get("confidence"),
            "reason": audit.get("reason"),
        }
        if not force and (
            not audit.get("ready") or suggestion is None or suggestion < min_reliability
        ):
            report.update({
                "stage": "vetoed",
                "imported": 0,
                "widened": False,
                "reason": (
                    f"审计未通过（建议 {suggestion} < 否决线 {min_reliability}）"
                    if suggestion is not None
                    else "审计不可用，保持小 top_k"
                ),
            })
            return report
        import_report = self.transfer_from(
            source_path, top_k=wide_top_k,
            prior_reliability=prior_reliability,
            reason=f"{reason}: widen to top_k={wide_top_k}",
        )
        report.update({
            "stage": "widened",
            "imported": import_report.get("imported"),
            "widened": True,
        })
        return report

    def transfer_from(
        self,
        source: str | Path,
        top_k: int | None = None,
        include_low_fidelity: bool = False,
        reason: str = "",
        prior_reliability: float = 1.0,
    ) -> dict:
        """Warm-start from another study after strict compatibility checks.

        Only the same search space + objective (including constraint limits) are
        accepted; observations are deduplicated by configuration and recorded
        with ``source="transfer"``. ``top_k`` keeps only the best-scoring prior
        points, a cheap guard against negative transfer from a distant campaign.

        ``prior_reliability`` in (0, 1] scales how strongly the priors are
        trusted: 1.0 keeps the recorded noise, while lower values add an
        observation-noise floor of ``(1 - reliability) * span / 2`` so the
        target GP can discount a doubtful source instead of overriding it.
        """
        if not (0.0 < float(prior_reliability) <= 1.0 + 1e-12):
            raise ValueError("prior_reliability must be in (0, 1]")
        if top_k is not None and int(top_k) < 0:
            raise ValueError("transfer_from: top_k must be >= 0 (0 imports nothing)")
        # Accept either a run directory or an already-loaded Study.
        source_study = source if isinstance(source, Study) else Study.load(source)
        if self.space.to_dict() != source_study.space.to_dict():
            raise ValueError("transfer_from: search spaces differ; refusing to transfer")
        if self.objective.to_dict() != source_study.objective.to_dict():
            raise ValueError("transfer_from: objectives/constraints differ; refusing to transfer")
        observations = list(source_study.dataset.observations)
        if not include_low_fidelity:
            observations = [obs for obs in observations if obs.fidelity >= 1.0 - 1e-9]
        selection = "primary_score"
        if top_k is not None and int(top_k) == 0:
            # explicit "import nothing" (e.g. a dry run); previously top_k=0 was
            # falsy and silently behaved like "import everything"
            observations = []
            selection = "none"
        elif top_k is not None and top_k > 0:
            if self.objective.is_multi_objective:
                # Keep trade-off points: a primary-objective ranking would
                # discard the other end of the Pareto front before the target
                # campaign even starts.
                complete = []
                skipped = []
                for observation in observations:
                    try:
                        observation_values(observation, self.objective)
                    except ValueError:
                        skipped.append(observation)
                        continue
                    complete.append(observation)
                if complete:
                    import numpy as np

                    matrix = np.asarray(
                        [observation_values(obs, self.objective) for obs in complete],
                        dtype=float,
                    ) * np.asarray(
                        [
                            1.0 if direction == "max" else -1.0
                            for direction in self.objective.directions
                        ]
                    )
                    mask = non_dominated_mask(matrix)
                    front = [
                        obs for obs, keep in zip(complete, mask) if bool(keep)
                    ]
                    front.sort(
                        key=lambda obs: obs.score(
                            self.objective.name, self.objective.direction
                        ),
                        reverse=True,
                    )
                    rest = sorted(
                        [obs for obs in complete if obs not in front] + skipped,
                        key=lambda obs: obs.score(
                            self.objective.name, self.objective.direction
                        ),
                        reverse=True,
                    )
                    observations = (front + rest)[: int(top_k)]
                    selection = "pareto_front"
                else:
                    observations = sorted(
                        observations,
                        key=lambda obs: obs.score(
                            self.objective.name, self.objective.direction
                        ),
                        reverse=True,
                    )[: int(top_k)]
            else:
                observations = sorted(
                    observations,
                    key=lambda obs: obs.score(self.objective.name, self.objective.direction),
                    reverse=True,
                )[: int(top_k)]
        reliability = float(prior_reliability)
        span = 0.0
        if reliability < 1.0 and observations:
            scores = [
                float(obs.score(self.objective.name, self.objective.direction))
                for obs in observations
            ]
            span = max(scores) - min(scores) if len(scores) > 1 else abs(scores[0]) or 1.0
        noise_floor = 0.5 * (1.0 - reliability) * span
        payload = []
        for observation in observations:
            meta = dict(observation.meta)
            # Source suggestion ids belong to the source round numbering; keep
            # them for provenance but drop the live id so it cannot collide
            # with this study's own pending ids (which would make `tell` treat
            # a fresh result as an idempotent replay and leave it pending).
            original_id = meta.pop("suggestion_id", None)
            if original_id is not None:
                meta["original_suggestion_id"] = original_id
            meta["transferred_from"] = str(source_study.dir)
            meta["source_n_replicates"] = float(observation.n_replicates)
            # Round-scoped bookkeeping from the source campaign must not leak into
            # this study: an imported prior inherited purpose="confirmation" from
            # the source, which made the target believe its incumbent was already
            # confirmed and skip the confirmation replicate entirely.
            for key in ("purpose", "plate", "plate_well", "batch"):
                if key in meta:
                    meta.setdefault("source_meta", {})[key] = meta.pop(key)
            meta["transfer_reason"] = reason or "warm-start"
            payload.append(
                Observation(
                    x=dict(observation.x),
                    y=observation.y,
                    objectives=dict(observation.objectives),
                    constraints=dict(observation.constraints),
                    fidelity=observation.fidelity,
                    cost=observation.cost,
                    objective_std=max(float(observation.objective_std), noise_floor),
                    # A prior is a *single* historical measurement as far as the
                    # target is concerned: the target has not replicated it, so the
                    # confirmation logic must still ask for its own replicate.
                    # The source-side count is kept for transparency.
                    n_replicates=1,
                    source="transfer",
                    meta=meta,
                )
            )
        imported = self.import_observations(payload)
        warnings: list[str] = []
        if top_k is None or int(top_k) > 4:
            # Measured (rosenbrock10, sweep_transfer.py): with a bad source the
            # damage grows monotonically with top_k (-369 at k=1 vs -1190 at
            # k=8); a reliable source still does not need many priors.
            warnings.append(
                "迁移点数偏多：实测源不可靠时负迁移随 top_k 增大而加剧"
                "（k=1 -369 -> k=8 -1190）；建议 top_k 1–3，并先确认源战役可信"
            )
        source_domain = str(getattr(source_study.context, "domain", "generic"))
        target_domain = str(getattr(self.context, "domain", "generic"))
        if source_domain != target_domain:
            warnings.append(
                f"跨领域迁移（domain {source_domain!r} -> {target_domain!r}）："
                "负迁移风险更高，建议先用小额 top_k 试探"
            )
        source_size = len(source_study.dataset.observations)
        if source_size < 5:
            warnings.append(
                f"源战役只有 {source_size} 条观测，先验统计价值有限；"
                "建议先跑一小段源战役或显式降低对先验的依赖"
            )
        warning = warnings[0] if warnings else None
        summary = {
            "source": str(source_study.dir),
            "candidates": len(payload),
            "imported": len(imported),
            "skipped": len(payload) - len(imported),
            "top_k": top_k,
            "selection": selection,
            "prior_reliability": reliability,
            "warning": warning,
            "warnings": warnings,
            "include_low_fidelity": bool(include_low_fidelity),
        }
        # Warm start is not free: imported observations count against the target
        # campaign's evaluation budget, so report what it actually consumed.
        summary["budget_charged"] = len(imported)
        summary["cost_charged"] = round(
            float(sum(getattr(obs, "cost", None) or float(obs.fidelity) for obs in imported)),
            6,
        )
        summary["budget_total"] = self.budget
        summary["remaining_after_transfer"] = self.remaining()
        self.log_decision(
            action="transfer_import",
            algorithm=self.algorithm,
            params=self.params,
            reason=f"从 {source_study.dir} 迁移 {len(imported)} 条历史观测（{reason or 'warm-start'}）",
            payload=summary,
        )
        self._save_all()
        return summary

    # -- overrides --------------------------------------------------------
    def override(self, algorithm: str, params: Mapping | None = None, reason: str = "") -> dict:
        """Switch algorithm/params, recording who changed what and why."""
        if not reason.strip():
            raise ValueError("override requires a non-empty reason (audit trail)")
        merged = registry.validate_params(algorithm, params)
        previous = {"algorithm": self.algorithm, "params": dict(self.params)}
        self.algorithm, self.params = algorithm, merged
        self.manual_override = True
        self.log_decision(
            action="override",
            algorithm=algorithm,
            params=merged,
            reason=reason,
            payload={"previous": previous},
        )
        self._save_all()
        return {"algorithm": algorithm, "params": merged, "reason": reason}

    def retune_for_exploration(self, reason: str = "") -> dict:
        """Shift an exploration-related parameter of the current algorithm.

        Used when a campaign stalls (see ``next_action().diagnostics.stop_category
        == "stagnation"``): flipping the acquisition mode or raising ``kappa``
        keeps the same model but changes where it looks, which measured better
        than either plain continuation or a model-free restart
        (rosenbrock10, budget 60: retune -205.4 vs continue -319.8 vs
        sobol-restart -373.7; see research/NEGATIVE_RESULTS.md #9).
        """
        params = dict(self.params)
        spec = registry.spec(self.algorithm)  # type: ignore[attr-defined]
        changed: list[str] = []
        for parameter in spec.params:
            choices = getattr(parameter, "choices", None)
            if parameter.name == "acquisition" and choices:
                current = params.get("acquisition", choices[0])
                if current in choices:
                    params["acquisition"] = choices[(list(choices).index(current) + 1) % len(choices)]
                    changed.append("acquisition")
                break
        if not changed:
            if "kappa" in params:
                params["kappa"] = min(10.0, max(1.0, float(params["kappa"]) * 2.0))
                changed.append("kappa")
            elif "local_sigma" in params:
                params["local_sigma"] = min(1.0, float(params["local_sigma"]) * 1.5)
                changed.append("local_sigma")
        if not changed:
            return {
                "algorithm": self.algorithm,
                "params": params,
                "changed": [],
                "reason": "没有可自动调整的探索参数",
            }
        method = getattr(self, "override", None)
        if method is None:  # pragma: no cover - Study always has override
            self.params = registry.validate_params(self.algorithm, params)
        else:
            method(
                self.algorithm,
                params=params,
                reason=reason or f"停滞重调参：{'/'.join(changed)}",
            )
        return {
            "algorithm": self.algorithm,
            "params": dict(self.params),
            "changed": changed,
            "reason": reason or f"停滞重调参：{'/'.join(changed)}",
        }

    def resume_auto_route(self) -> dict:
        """Re-enable the router after a manual override (logged)."""
        self.manual_override = False
        self.auto_route = True
        self.log_decision(
            action="resume_auto_route",
            algorithm=self.algorithm,
            params=self.params,
            reason="恢复 router 自动重路由",
        )
        self._save_all()
        return {"auto_route": True, "manual_override": False, "algorithm": self.algorithm}

    def set_knee_weights(self, weights: Mapping[str, float], reason: str = "") -> dict:
        """Set per-objective weights for the multi-objective knee selection."""
        names = set(self.objective.objectives)
        cleaned: dict[str, float] = {}
        for name, value in weights.items():
            if name not in names:
                raise ValueError(f"unknown objective {name!r}; expected one of {sorted(names)}")
            weight = float(value)
            if weight <= 0:
                raise ValueError(f"weight for {name!r} must be > 0")
            cleaned[name] = weight
        self.meta["knee_weights"] = cleaned
        self.log_decision(
            action="knee_weights",
            algorithm=self.algorithm,
            params=self.params,
            reason=reason or "set knee weights",
            payload={"weights": cleaned},
        )
        self._save_all()
        return cleaned

    # -- reporting --------------------------------------------------------
    def pareto_summary(self) -> dict:
        """Non-dominated front for multi-objective studies (empty for single)."""
        if not self.objective.is_multi_objective:
            return {"n_front": 0, "front": [], "hypervolume": None, "reference_point": None}
        dataset = self._dataset()
        if len(dataset) == 0:
            return {"n_front": 0, "front": [], "hypervolume": None, "reference_point": None}
        # Multi-fidelity: prefer full-fidelity points so biased low-fidelity
        # values neither enter the front nor inflate the hypervolume.
        full = [obs for obs in dataset if float(obs.fidelity) >= 1.0 - 1e-9]
        fidelity_filter = "full" if full else "all"
        used = full if full else dataset
        excluded = len(dataset) - len(used)
        result = pareto_summarise(used, self.objective)
        front = []
        for index in result.indices:
            observation = used[index]
            front.append(
                {
                    "x": observation.x,
                    "y": observation.y,
                    "objectives": observation.objectives,
                    "source": observation.source,
                }
            )
        return {
            "n_front": result.n_front,
            "front": front,
            "hypervolume": result.hypervolume,
            "reference_point": None
            if result.reference_point is None
            else [float(x) for x in result.reference_point],
            "fidelity_filter": fidelity_filter,
            "n_low_fidelity_excluded": excluded,
        }

    def _decision_summary(self) -> dict | None:
        if self.best() is None or len(self._dataset()) == 0:
            return None
        recommendation = self.recommend()
        spec = recommendation.get("spec")
        raw_spec = recommendation.get("spec_raw")
        plate = recommendation.get("plate_effects") or {}
        raw_in_spec = None if raw_spec is None else raw_spec.get("in_spec")
        adjusted_in_spec = None if spec is None else spec.get("in_spec")
        return {
            "selection_basis": recommendation.get("selection_basis"),
            "selection_reason": recommendation.get("selection_reason"),
            "score": recommendation.get("score"),
            "in_spec": adjusted_in_spec,
            "raw_in_spec": raw_in_spec,
            "plate_effect_detected": bool(plate.get("detected")),
            "spec_check": {
                "raw_in_spec": raw_in_spec,
                "adjusted_in_spec": adjusted_in_spec,
                "changed_by_plate": (
                    raw_in_spec is not None
                    and adjusted_in_spec is not None
                    and raw_in_spec != adjusted_in_spec
                ),
            },
            "knee_weights": dict(self.meta.get("knee_weights") or {}),
            "forked_from": self.meta.get("forked_from"),
        }

    def decision(self, explain: bool = False) -> dict | None:
        """Public, lightweight recommendation verdict (same as summary.decision).

        ``explain=True`` adds an ``explain`` list of human-readable reasons
        (data volume, plate correction, spec judgement, confirmation, stop).
        """
        payload = self._decision_summary()
        if payload is None or not explain:
            return payload
        dataset = self._dataset()
        recommendation = self.recommend()
        lines = [f"观测 {len(dataset)} 条，剩余预算 {self.remaining()} 次"]
        basis = payload.get("selection_basis")
        lines.append(f"推荐依据: {basis}")
        if payload.get("plate_effect_detected"):
            effects = (recommendation.get("plate_effects") or {}).get("metrics") or {}
            plates = ", ".join(sorted(effects)) or "主目标"
            lines.append(f"检测到板效应（{plates}），推荐值已按板效应校正")
        spec_check = payload.get("spec_check") or {}
        if spec_check.get("changed_by_plate"):
            raw = "✓" if spec_check.get("raw_in_spec") else "✗"
            adjusted = "✓" if spec_check.get("adjusted_in_spec") else "✗"
            lines.append(f"规格判定被板效应改变: 原始 {raw} / 校正后 {adjusted}")
        elif payload.get("in_spec") is not None:
            lines.append(f"规格符合性: {'✓ in-spec' if payload['in_spec'] else '✗ out-of-spec'}")
        stats = recommendation.get("confirmation")
        if stats is not None:
            lines.append(
                f"确认复孔: n={stats['n']} mean={stats['mean']:.6g} std={stats['std']:.3g}"
            )
        if payload.get("knee_weights"):
            weights = payload["knee_weights"]
            lines.append("knee 权重: " + ", ".join(f"{k}={v:g}" for k, v in weights.items()))
        if payload.get("forked_from"):
            lines.append(f"来自 fork: {payload['forked_from']}")
        stop, reason = self.should_stop()
        if stop:
            lines.append(f"停止条件: {reason}")
        enriched = dict(payload)
        enriched["explain"] = lines
        return enriched

    def summary(self) -> dict:
        dataset = self._dataset()
        best = self.best()
        ctx = self.current_context()
        recommendation = route(ctx)
        decision = self._decision_summary()
        return {
            "run_dir": str(self.dir),
            "algorithm": self.algorithm,
            "params": self.params,
            "budget": self.budget,
            "used": len(dataset),
            "pending": len(self._pending),
            "cost_spent": self.cost_spent(),
            "remaining_cost": self.remaining_cost(),
            "manual_override": self.manual_override,
            "remaining": self.remaining(),
            "best": None if best is None else {
                "x": best.x,
                "y": best.y,
                "objectives": best.objectives,
                "source": best.source,
            },
            "best_score": dataset.best_score(self.objective.name, self.objective.direction),
            "spec": self._spec_summary(best, decision),
            "decision": decision,
            "pareto": self.pareto_summary(),
            "plates": self.plate_summary(),
            "rounds": list(self._rounds),
            "router": {
                "next": recommendation.to_dict(),
                "scores": recommendation.candidate_scores,
            },
            "decisions": self.decisions()[-5:],
        }

    def feasible_pareto_summary(self) -> dict:
        """Feasible-only front, using the *same* fidelity policy as ``pareto_summary``.

        Mixing fidelities would let biased low-fidelity values inflate the
        deliverable hypervolume above the raw one (measured: 712 > 702 before this
        fix), so the full-fidelity preference is applied first.
        """
        empty = {
            "n_front": 0,
            "front": [],
            "hypervolume": None,
            "reference_point": None,
            "n_feasible": 0,
            "fidelity_filter": None,
        }
        if not self.objective.is_multi_objective:
            return empty
        dataset = self._dataset()
        if not dataset:
            return empty
        full = [obs for obs in dataset if float(obs.fidelity) >= 1.0 - 1e-9]
        fidelity_filter = "full" if full else "all"
        used = full if full else dataset
        feasible = [obs for obs in used if self._is_feasible(obs)]
        if not feasible:
            return {**empty, "fidelity_filter": fidelity_filter}
        result = pareto_summarise(feasible, self.objective)
        return {
            "n_front": result.n_front,
            "front": [
                {
                    "x": used[index].x,
                    "y": used[index].y,
                    "objectives": used[index].objectives,
                    "source": used[index].source,
                }
                for index in result.indices
            ],
            "hypervolume": result.hypervolume,
            "reference_point": result.reference_point,
            "n_feasible": len(feasible),
            "fidelity_filter": fidelity_filter,
        }

    def delivery_checklist(self) -> list[dict]:
        """Check the run against the evidence a delivery must carry.

        Mirrors ``prepare.deliverable_advice``: constraints need a feasible
        final-fidelity observation, multi-fidelity needs full-fidelity evidence,
        single-objective needs a confirmed incumbent, multi-objective needs a
        Pareto front (and its progress record).
        """
        dataset = self._dataset()
        constraints = tuple(getattr(self.objective, "constraints", ()) or ())
        checks: list[dict] = []
        full = [obs for obs in dataset if float(obs.fidelity) >= 1.0 - 1e-9]

        if float(getattr(self.context, "n_fidelities", 1) or 1) > 1:
            checks.append({
                "item": "至少一条全保真观测",
                "ok": bool(full),
                "detail": f"全保真 {len(full)} 条 / 共 {len(dataset)} 条",
            })
        if constraints:
            feasible_full = [obs for obs in full if self._is_feasible(obs)]
            checks.append({
                "item": "至少一条可行的最终保真观测",
                "ok": bool(feasible_full),
                "detail": f"可行全保真 {len(feasible_full)} 条",
            })
        if self.objective.is_multi_objective:
            summary = (
                self.feasible_pareto_summary() if constraints else self.pareto_summary()
            )
            checks.append({
                "item": "非支配前沿存在",
                "ok": bool(summary["n_front"]),
                "detail": f"前沿 {summary['n_front']} 点"
                + (
                    f"，可交付 HV={summary['hypervolume']:.4g}"
                    if summary.get("hypervolume") is not None
                    else ""
                ),
            })
            checks.append({
                "item": "逐批进展记录",
                "ok": bool(self.progress_trajectory()),
                "detail": f"{len(self.progress_trajectory())} 个记录点",
            })
        else:
            best = self._delivery_incumbent()
            total_at_best = 0
            confirmation_count = 0
            if best is not None:
                key = self._encode_key(best.x)
                fidelity = float(getattr(best, "fidelity", 1.0))
                total_at_best = sum(
                    1
                    for obs in dataset
                    if self._encode_key(obs.x) == key
                    and abs(float(obs.fidelity) - fidelity) < 1e-9
                )
                confirmation = self._confirmation_stats(best)
                confirmation_count = int(confirmation["n"]) if confirmation else 0
            checks.append({
                "item": "推荐条件已复孔确认",
                # the incumbent must have been measured more than once, and at least
                # one of those measurements must be a tagged confirmation replicate
                "ok": total_at_best >= 2 and confirmation_count >= 1,
                "detail": f"该条件共 {total_at_best} 次测量（其中确认复孔 {confirmation_count} 次）",
            })
        return checks

    def record_progress(self) -> dict | None:
        """Append one progress row (multi-objective only) and return it.

        The adapter calls this after every batch; hand-written loops can call it
        themselves so a delivery still carries the "逐批进展记录" that the
        checklist expects.
        """
        if not self.objective.is_multi_objective:
            return None
        constraints = tuple(getattr(self.objective, "constraints", ()) or ())
        summary = (
            self.feasible_pareto_summary() if constraints else self.pareto_summary()
        )
        dataset = self._dataset()
        feasible = sum(1 for obs in dataset if self._is_feasible(obs))
        row = {
            "t": time.time(),
            "round": len(dataset),
            "n_front": summary.get("n_front", 0),
            "hypervolume": summary.get("hypervolume"),
            "n_feasible": feasible,
        }
        if constraints:
            row["feasible_fraction"] = round(feasible / max(1, len(dataset)), 4)
            row["hypervolume_feasible"] = summary.get("hypervolume")
        path = self.dir / "progress.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def progress_trajectory(self) -> list[dict]:
        """Per-batch Pareto/hypervolume records written by the executor adapter."""
        path = self.dir / "progress.jsonl"
        if not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def report(self) -> str:
        summary = self.summary()
        lines = [
            f"Study: {self.dir}",
            f"算法: {summary['algorithm']}  参数: {json.dumps(self.params, ensure_ascii=False)}",
            f"预算: {summary['used']}/{self.budget} 已评价，{summary['remaining']} 剩余"
            + (f"（{summary['pending']} 个待测建议）" if summary["pending"] else ""),
        ]
        forked_from = self.meta.get("forked_from")
        if forked_from:
            lines.append(
                f"来源: fork from {forked_from}（spec_mode={self.meta.get('fork_spec_mode', '?')}）"
            )
            changed = self.meta.get("fork_changed") or []
            if changed:
                lines.append(f"  规格改动: {json.dumps(changed, ensure_ascii=False)}")
        log_path = self.dir / "execution_log.jsonl"
        if log_path.exists():
            from .executor import failure_summary

            failures = failure_summary(log_path)
            if failures["total"]:
                detail = "、".join(
                    f"{kind} {count}"
                    for kind, count in sorted(
                        failures["by_kind"].items(), key=lambda item: -item[1]
                    )
                )
                lines.append(
                    f"执行日志：{failures['total']} 次尝试，{failures['failures']} 次失败"
                    + (f"（{detail}）" if detail else "")
                )
        best = summary["best"]
        if best is None:
            lines.append("尚无观测结果。")
        else:
            lines.append(f"当前最优 ({self.objective.name}, {self.objective.direction}):")
            for name, value in best["x"].items():
                lines.append(f"  {name} = {value}")
            if best["y"] is not None:
                lines.append(f"  {self.objective.name} = {best['y']:.6g}")
        pareto = summary["pareto"]
        if self.objective.is_multi_objective:
            if pareto["n_front"] == 0:
                lines.append("Pareto 前沿：暂无观测数据。")
            else:
                hv = pareto["hypervolume"]
                hv_text = f"，HV={hv:.4g}" if hv is not None else ""
                reference = pareto.get("reference_point")
                ref_text = (
                    f"，参考点={tuple(round(float(x), 4) for x in reference)}"
                    if reference
                    else ""
                )
                lines.append(f"Pareto 前沿：{pareto['n_front']} 个非支配点{hv_text}{ref_text}")
                if self.objective.constraints:
                    # Raw HV can include infeasible points; what ships is the
                    # feasible front (measured: raw 143.0 vs feasible 80.9).
                    feasible = self.feasible_pareto_summary()
                    if feasible["n_feasible"]:
                        feasible_hv = feasible["hypervolume"]
                        hv_part = (
                            f"，可行 HV={feasible_hv:.4g}" if feasible_hv is not None else ""
                        )
                        lines.append(
                            f"  可交付前沿：{feasible['n_front']} 个可行非支配点"
                            f"{hv_part}（可行观测 {feasible['n_feasible']} 条，"
                            f"口径 {feasible['fidelity_filter']}）"
                        )
                progress = self.progress_trajectory()
                if progress:
                    first = progress[0]
                    last = progress[-1]
                    increases = sum(
                        1
                        for previous, current in zip(progress, progress[1:])
                        if previous["hypervolume"] is not None
                        and current["hypervolume"] is not None
                        and current["hypervolume"] > previous["hypervolume"]
                    )
                    feasible_series = [
                        row.get("hypervolume_feasible")
                        for row in progress
                        if row.get("hypervolume_feasible") is not None
                    ]
                    def _fmt_volume(value: float) -> str:
                        # A single feasible point has no hypervolume; print 0
                        # instead of 2.5e-21.
                        return "0" if abs(value) < 1e-12 else f"{value:.4g}"

                    feasible_text = (
                        f"，可行 HV {_fmt_volume(feasible_series[0])} → "
                        f"{_fmt_volume(feasible_series[-1])}"
                        if feasible_series
                        else ""
                    )
                    lines.append(
                        "  进展: "
                        f"{len(progress)} 个记录点，HV "
                        f"{first['hypervolume'] if first['hypervolume'] is not None else '—'}"
                        f" → {last['hypervolume'] if last['hypervolume'] is not None else '—'}"
                        f"{feasible_text}"
                        f"（提升 {increases} 次）"
                    )
                for item in pareto["front"][:5]:
                    objectives = ", ".join(f"{k}={v:.4g}" for k, v in item["objectives"].items())
                    lines.append(f"  - {objectives}")
        if self._pending:
            lines.append("待测建议：")
            for item in self._pending:
                cond = ", ".join(f"{k}={v}" for k, v in item["x"].items())
                where = ""
                if item.get("plate"):
                    where = f"  @{item['plate']}/{item.get('well', '?')}"
                lines.append(f"  [{item['id']}] {cond}{where}  (算法 {item['algorithm']})")
        if self._rounds:
            lines.append("轮次记录：")
            for round_ in self._rounds:
                lines.append(
                    f"  round {round_['round']}: n={round_['n']} best={round_.get('best')} "
                    f"algo={round_.get('algorithm')}"
                )
        plates = summary["plates"]
        if plates["n_plates"]:
            lines.append("孔板/批次：")
            for entry in plates["plates"]:
                parts = [f"n={entry['n']}"]
                if "mean_value" in entry:
                    parts.append(f"均值={entry['mean_value']:.6g}")
                    parts.append(f"极差={entry['range_value']:.4g}")
                else:
                    parts.append(f"均值(score)={entry['mean_score']:.6g}")
                lines.append(f"  {entry['plate']}: " + "  ".join(parts))
            effect = plates["batch_effect"]
            prefix = "  ⚠ " if effect["detected"] else "  "
            lines.append(prefix + "批次效应：" + effect["message"])
        action = self.next_action()
        lines.append(f"下一步行动：{action['action']}（{action['reason']}）")
        lines.append(f"  命令：{action['next_command']}")
        recommendation = action.get("recommendation")
        if recommendation is not None:
            basis = recommendation.get("selection_basis", "raw")
            lines.append(f"推荐条件（{basis}）:")
            for name, value in recommendation["condition"].items():
                lines.append(f"  {name} = {value}")
            if recommendation.get("confirmation") is not None:
                stats = recommendation["confirmation"]
                lines.append(
                    f"  确认复孔: n={stats['n']} mean={stats['mean']:.6g} std={stats['std']:.3g}"
                )
            else:
                score = recommendation.get("score")
                score_text = "n/a" if score is None else f"{float(score):.6g}"
                lines.append(f"  score = {score_text}")
            if basis == "plate_adjusted":
                raw = recommendation.get("raw_best") or {}
                if raw:
                    raw_score = raw.get("score")
                    raw_text = "n/a" if raw_score is None else f"{float(raw_score):.6g}"
                    lines.append(
                        f"  (raw best={raw_text} @ {raw.get('plate')}; 因批次效应未选)"
                    )
            spec = recommendation.get("spec")
            if spec is not None:
                lines.append(f"  规格符合性: {'✓ in-spec' if spec['in_spec'] else '✗ out-of-spec'}")
            spec_check = (action.get("diagnostics") or {}).get("spec_check") or {}
            if spec_check.get("changed_by_plate"):
                raw_ok = "✓" if spec_check.get("raw_in_spec") else "✗"
                adjusted_ok = "✓" if spec_check.get("adjusted_in_spec") else "✗"
                lines.append(f"  规格判定: 原始 {raw_ok} / 板校正后 {adjusted_ok}")
            diagnostics = action.get("diagnostics") or {}
            if diagnostics.get("knee_weights"):
                weights = diagnostics["knee_weights"]
                lines.append(
                    "  knee 权重: " + ", ".join(f"{name}={value:g}" for name, value in weights.items())
                )
            if diagnostics.get("spec_calibration_ready"):
                lines.append("  可运行 calibrate-spec 校准 tolerance/scale")
        next_route = summary["router"]["next"]
        lines.append(
            f"下一轮建议：{next_route['algorithm']}（置信度 {next_route['confidence']}）"
        )
        for reason in next_route["rationale"][:3]:
            lines.append(f"  - {reason}")
        checklist = self.delivery_checklist()
        if checklist:
            lines.append("")
            lines.append("交付自查:")
            for item in checklist:
                mark = "OK " if item["ok"] else "缺 "
                lines.append(f"  [{mark}] {item['item']}（{item['detail']}）")
        return "\n".join(lines)

    def _write_suggestions(self, pending: Sequence[dict]) -> None:
        path = self.dir / SUGGESTIONS_FILE
        header = ["id", "batch", "algorithm", "fidelity", "plate", "well"] + self.space.names
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for item in pending:
                row = [
                    item["id"],
                    item["batch"],
                    item["algorithm"],
                    item.get("fidelity", 1.0),
                    item.get("plate", ""),
                    item.get("well", ""),
                ]
                row.extend(item["x"][name] for name in self.space.names)
                writer.writerow(row)


def _plate_sort_key(name: str) -> tuple:
    """Natural ordering: P2 before P10, non-standard names last."""
    match = re.fullmatch(r"[Pp](\d+)", name)
    return (0, int(match.group(1)), "") if match else (1, 0, name)


def _f_sf(f_stat: float, dfn: int, dfd: int) -> float:
    """Upper-tail probability of the F distribution (ANOVA p-value)."""
    try:
        from scipy.stats import f as f_distribution

        return float(f_distribution.sf(f_stat, dfn, dfd))
    except ImportError:  # pragma: no cover - scipy is a hard dependency
        return 0.0 if f_stat > 1.0 else 1.0


def _row_label(index: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA (Excel-style row labels)."""
    letters = ""
    index += 1
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return default

