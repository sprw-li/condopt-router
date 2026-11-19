"""Minimal executor adapter: plug a real instrument into a ``Study`` loop.

A ``Study`` produces conditions/ask-confirm-tell actions; this module supplies the
other half — the thing that actually runs the experiment.  Users implement one
method (:meth:`Executor.measure`) and :func:`run_campaign` drives everything else,
including partial failures (a failed run stays pending and is logged, so it can be
retried instead of silently shrinking the budget).
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


@dataclass
class ExecutionRequest:
    """One condition the study wants measured."""

    id: str
    x: Mapping[str, Any]
    fidelity: float = 1.0
    note: str = ""


@dataclass
class Measurement:
    """Result of a single experiment.

    Set ``failed=True`` (and leave ``y`` empty) when the run could not be
    executed: the suggestion stays pending so it can be retried, and the attempt
    is written to the execution log.
    """

    id: str
    y: float | None = None
    fidelity: float = 1.0
    constraints: Mapping[str, float] = field(default_factory=dict)
    cost: float | None = None
    failed: bool = False
    note: str = ""
    #: Multi-objective studies: measured values for the *other* objectives
    #: (``y`` carries the primary one, or the first objective).
    objectives: Mapping[str, float] = field(default_factory=dict)

    def payload(self) -> dict:
        if self.failed or self.y is None:
            raise ValueError("a failed measurement has no usable payload")
        payload: dict = {"id": self.id, "y": float(self.y), "fidelity": float(self.fidelity)}
        if self.objectives:
            payload["objectives"] = {key: float(value) for key, value in self.objectives.items()}
        if self.constraints:
            payload["constraints"] = dict(self.constraints)
        if self.cost is not None:
            payload["cost"] = float(self.cost)
        return payload


def _safe_measure(
    executor: Executor, request: ExecutionRequest, timeout: float | None
) -> Measurement:
    """Run one measurement, converting errors/timeouts into failed results."""
    try:
        return _measure_one(executor, request, timeout)
    except Exception as exc:  # an instrument error must not kill the campaign
        return Measurement(
            id=request.id,
            fidelity=request.fidelity,
            failed=True,
            note=f"{type(exc).__name__}: {exc}",
        )


class Executor(ABC):
    """Subclass this and implement :meth:`measure`."""

    @abstractmethod
    def measure(self, request: ExecutionRequest) -> Measurement:  # pragma: no cover
        raise NotImplementedError


def classify_failure(note: str, abandoned: bool = False) -> str:
    """Coarse cause for an unusable measurement (for post-mortem summaries)."""
    text = (note or "").strip()
    if abandoned:
        return "abandoned"
    if text.startswith("timeout"):
        return "timeout"
    if ":" in text and text.split(":", 1)[0].endswith("Error"):
        return f"exception:{text.split(':', 1)[0].strip()}"
    # Anything else is the instrument reporting a failed run itself.
    return "instrument_error"


def failure_summary(path: str | Path) -> dict:
    """Aggregate ``execution_log.jsonl`` into failure counts by cause."""
    log_path = Path(path)
    if log_path.is_dir():
        log_path = log_path / "execution_log.jsonl"
    counts: dict[str, int] = {}
    total = 0
    if not log_path.exists():
        return {"total": 0, "failures": 0, "by_kind": counts, "path": str(log_path)}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        total += 1
        if not row.get("failed"):
            continue
        kind = row.get("failure_kind") or classify_failure(
            str(row.get("note") or ""), bool(row.get("abandoned"))
        )
        counts[kind] = counts.get(kind, 0) + 1
    return {
        "total": total,
        "failures": sum(counts.values()),
        "by_kind": counts,
        "path": str(log_path),
    }


def _append_progress(path: Path, study) -> dict | None:
    """Record the Pareto/hypervolume trajectory after each batch (multi-objective)."""
    try:
        row = study.record_progress()
    except Exception:
        return None
    if row is None:
        return None
    if Path(path) != study.dir / "progress.jsonl":
        # a custom log location was requested: mirror the row there as well
        _append_log(path, [row])
    return row


def _append_log(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _measure_one(
    executor: Executor, request: ExecutionRequest, timeout: float | None
) -> Measurement:
    if timeout is None:
        return executor.measure(request)

    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

    # NOTE: do *not* use the context manager here -- its __exit__ joins the worker
    # thread, which would block for the full duration of a hung instrument.  We
    # shut the pool down without waiting so the campaign can move on.
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(executor.measure, request)
    try:
        result = future.result(timeout=timeout)
    except FutureTimeout:
        pool.shutdown(wait=False)
        return Measurement(
            id=request.id,
            fidelity=request.fidelity,
            failed=True,
            note=f"timeout after {timeout}s",
        )
    pool.shutdown(wait=False)
    return result


def run_campaign(
    study,
    executor: Executor,
    max_steps: int = 200,
    log_path: str | Path | None = None,
    on_event: Callable[[str, Mapping], None] | None = None,
    max_empty_batches: int = 3,
    max_workers: int = 1,
    timeout: float | None = None,
    max_retries: int = 2,
    prior_source: str | Path | None = None,
    staged_transfer: bool = True,
) -> dict:
    """Drive ``study`` until it stops, using ``executor`` to measure conditions.

    ``max_empty_batches`` guards against a broken instrument: if that many
    consecutive tell rounds produce no usable measurement the campaign stops with
    ``stopped_reason="no_progress"`` instead of retrying forever.

    ``max_workers > 1`` measures the batch in parallel (your instrument must be
    thread-safe); ``timeout`` (seconds, per request) turns a hung run into a
    failed measurement instead of blocking the campaign.

    Re-running on the same study resumes naturally: whatever stayed pending is
    measured first.

    ``prior_source`` wires the validated staged-prior recipe into the loop
    (``research/TRANSFER_ADAPTIVE.md``): a small first look, two local batches,
    then ``Study.transfer_staged`` decides once whether to widen -- a biased source
    costs -1.54/-0.33 under a fixed ``top_k`` of 3/5 versus +0.71 here.  Pass
    ``staged_transfer=False`` to disable it (e.g. when the caller already imported).
    """
    log_file = Path(log_path) if log_path else study.dir / "execution_log.jsonl"
    progress_file = study.dir / "progress.jsonl"
    counts = {"ask": 0, "confirm": 0, "tell": 0, "tell_actions": 0, "stop": 0}
    failures = 0
    attempts = 0
    steps = 0
    empty_batches = 0
    stopped_reason = None
    attempts_by_id: dict[str, int] = {}
    cancelled = 0
    transfer_settled = False
    transfer_events: list[dict] = []
    resumed_pending = len(study.pending)

    def emit(kind: str, payload: Mapping) -> None:
        if on_event is not None:
            on_event(kind, payload)

    while steps < max_steps:
        steps += 1
        action = study.next_action()
        kind = action["action"]
        if kind in ("ask", "confirm", "stop"):
            counts[kind] += 1
        elif kind == "tell":
            # ``tell`` counts batches that actually recorded results; every tell
            # *request* (including empty ones) is tracked separately.
            counts["tell_actions"] += 1
        if kind == "stop":
            stopped_reason = action.get("stop_reason")
            emit("stop", {"reason": stopped_reason})
            break
        if kind == "ask":
            if prior_source is not None and staged_transfer and not transfer_settled:
                report = study.transfer_staged(str(prior_source))
                transfer_events.append(report)
                emit("transfer_staged", report)
                if report.get("stage") in {"widened", "vetoed"} or report.get("force"):
                    transfer_settled = True
            study.ask(int(action.get("ask_n") or 1))
            continue
        if kind == "confirm":
            study.confirm(int(action.get("confirm_n") or 1))
            continue
        if kind != "tell":
            continue

        pending = list(study.pending)
        requests = [
            ExecutionRequest(
                id=item["id"],
                x=item["x"],
                fidelity=float(item.get("fidelity", 1.0) or 1.0),
            )
            for item in pending
        ]
        started = time.time()
        if max_workers > 1 and len(requests) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                outcomes = list(
                    pool.map(lambda req: _safe_measure(executor, req, timeout), requests)
                )
        else:
            outcomes = [_safe_measure(executor, request, timeout) for request in requests]
        results: list[dict] = []
        log_rows: list[dict] = []
        exhausted: list[str] = []
        for request, measurement in zip(requests, outcomes):
            attempts += 1
            attempts_by_id[request.id] = attempts_by_id.get(request.id, 0) + 1
            row = {
                "t": started,
                "batch_elapsed": round(time.time() - started, 3),
                "id": measurement.id,
                "x": dict(request.x),
                "fidelity": float(measurement.fidelity),
                "failed": bool(measurement.failed),
                "note": measurement.note,
                "y": None if measurement.y is None else float(measurement.y),
                "cost": measurement.cost,
                "constraints": dict(measurement.constraints),
                "objectives": dict(measurement.objectives),
            }
            log_rows.append(row)
            if measurement.failed or measurement.y is None:
                failures += 1
                if max_retries >= 0 and attempts_by_id[request.id] > max_retries:
                    row["abandoned"] = True
                    exhausted.append(request.id)
                row["failure_kind"] = classify_failure(
                    measurement.note, bool(row.get("abandoned"))
                )
                continue
            results.append(measurement.payload())
        if exhausted:
            cancelled += len(study.cancel_pending(
                exhausted,
                reason=f"重试 {max_retries} 次仍失败，放弃这些建议（执行器故障？）",
            ))
        if results:
            study.tell(results)
            counts["tell"] += 1
        # Flush the log *before* any early break, otherwise the failed attempts that
        # triggered the stall guard would never be recorded.
        _append_log(log_file, log_rows)
        if results:
            empty_batches = 0
        else:
            empty_batches += 1
            if max_empty_batches and empty_batches >= max_empty_batches:
                stopped_reason = (
                    f"连续 {empty_batches} 批没有可用的实测结果（执行器故障？）"
                )
                emit("stalled", {"empty_batches": empty_batches})
                break
        if getattr(study.objective, "is_multi_objective", False):
            _append_progress(progress_file, study)
        emit("batch", {"measured": len(results), "failed": len(log_rows) - len(results)})

    cost_budget = study.meta.get("cost_budget")
    report = {
        "steps": steps,
        "actions": counts,
        "attempts": attempts,
        "failures": failures,
        "n_observations": len(study.dataset.observations),
        "still_pending": len(study.pending),
        "cost_spent": round(float(study.cost_spent()), 6),
        "cost_budget": cost_budget,
        "execution_log": str(log_file),
        "failure_kinds": failure_summary(log_file)["by_kind"],
        "cancelled_after_retries": cancelled,
        "max_retries": max_retries,
        "resumed_pending": resumed_pending,
        "stopped_reason": stopped_reason or "达到 max_steps",
        "integrity_ok": bool(study.check_integrity()["ok"]),
        "transfer_staged": transfer_events,
    }
    if getattr(study.objective, "is_multi_objective", False):
        last_summary = study.pareto_summary()
        report["final_hypervolume"] = last_summary.get("hypervolume")
        report["final_n_front"] = last_summary.get("n_front")
        report["progress_points"] = (
            len(progress_file.read_text(encoding="utf-8").splitlines())
            if progress_file.exists()
            else 0
        )
    report["ok"] = bool(
        report["integrity_ok"]
        and (cost_budget is None or report["cost_spent"] <= float(cost_budget) + 1e-9)
        # A campaign that measured nothing usable is a failure, not a success.
        and (report["n_observations"] > 0 or report["attempts"] == 0)
    )
    return report
