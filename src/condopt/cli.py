"""Command line interface for condopt.

The skill layer drives these commands.  Everything is JSON-friendly so an
agent can pipe results into its own reasoning without parsing Chinese text.

错误契约（A7）
-------------
已知错误不再抛 traceback，而是在 **stderr 上写一行 JSON** 并返回稳定退出码::

    {"ok": false, "error": {"type": "StateError", "message": "...", "hint": "..."}}

退出码表：``0`` 成功 ｜ ``2`` 输入错误（``InputError``：参数/JSON/字段名或类型）
｜ ``3`` 状态错误（``StateError``：run 未 init、pending id 未知、批次状态冲突）
｜ ``4`` 环境错误（``EnvironmentError_``：输入文件缺失/依赖不可用）。
表与错误类的对应关系见 :mod:`condopt.errors`。

排障时加 ``--debug``（每个子命令都支持）会在 JSON 之后再打完整 traceback；
不加时 stderr 只有那一行 JSON，agent 可以直接 ``json.loads``。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Sequence

from . import registry
from .base import Objective
from .errors import (
    CondoptError,
    EnvironmentError_,
    InputError,
    StateError,
    error_payload,
    exit_code_for,
)
from .router import OptimizationContext, route
from .schema import validate_answers
from .space import SearchSpace
from .spec import Spec
from .study import Study


def _load_study(run: str) -> Study:
    """``Study.load``，但把"run 不存在"报成状态错误（退出码 3）而不是环境错误。

    缺文件一律是 4，只有"该 run 还没 init"才有独立语义：agent 的正确动作是
    先 ``init``（而不是去修路径），所以这里换成 :class:`StateError`。
    """
    try:
        return Study.load(run)
    except FileNotFoundError as exc:
        raise StateError(
            str(exc), hint=f"先创建战役: condopt init --run {run} --space <space.json>"
        ) from exc


def _load_json_arg(value: str) -> dict:
    """Accept an inline JSON object or a path to a JSON file."""
    text = value.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise InputError(
                f"内联 JSON 解析失败：{exc}",
                hint="Windows PowerShell 可能吞掉内联 JSON 的引号；"
                "请把 JSON 写入文件后用文件路径传入（例如 --params params.json）",
            ) from exc
    path = Path(text)
    if not path.exists():
        raise EnvironmentError_(f"file not found: {value}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _parse_objective(value: str) -> Objective:
    """Parse ``yield:max`` shorthand, an inline JSON object, or a JSON file."""
    text = value.strip()
    if text.startswith("{") or Path(text).exists():
        return Objective.from_dict(_load_json_arg(text))
    if ":" in text:
        name, direction = text.split(":", 1)
        return Objective(name=name.strip(), direction=direction.strip())
    raise InputError(
        "objective 格式应为 'name:direction'（如 yield:max）、JSON 字符串或 JSON 文件路径"
    )


def _dump(payload, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, indent=2))


def _context_from_args(args) -> OptimizationContext:
    payload: dict | None = None
    explicit_remaining = False
    if getattr(args, "context", None):
        payload = dict(_load_json_arg(args.context))
        explicit_remaining = payload.get("budget_remaining") is not None
        if getattr(args, "space", None):
            # Merge: space derives dimensions/counts, explicit context fields win.
            base = OptimizationContext.from_space(SearchSpace.from_json(args.space)).to_dict()
            payload = {**base, **payload}
    elif not getattr(args, "space", None):
        raise InputError("route/init requires --space FILE (or --context FILE)")
    else:
        payload = OptimizationContext.from_space(SearchSpace.from_json(args.space)).to_dict()

    flags = {
        "budget_total": getattr(args, "budget", None),
        "batch_size": getattr(args, "batch", None),
        "noise": getattr(args, "noise", None),
        "smoothness": getattr(args, "smoothness", None),
        "compute": getattr(args, "compute", None),
        "domain": getattr(args, "domain", None),
        "n_objectives": getattr(args, "objectives", None),
        "n_constraints": getattr(args, "constraints", None),
        "n_observations": getattr(args, "prior", None),
        "notes": getattr(args, "notes", None),
    }
    for key, value in flags.items():
        if value is not None:
            payload[key] = value
    if getattr(args, "multi_fidelity", False):
        payload["multi_fidelity"] = True

    ctx = OptimizationContext.from_dict(payload)
    if not explicit_remaining:
        ctx.budget_remaining = max(0, ctx.budget_total - ctx.n_observations)
    return ctx


def _cmd_spec(args) -> int:
    if args.algorithm:
        payload = registry.describe(args.algorithm)
        if args.json:
            _dump(payload, True)
        else:
            spec = registry.spec(args.algorithm)
            print(f"{spec.name}  [{spec.family}]")
            print(spec.summary)
            print(f"何时用: {spec.when}")
            print("参数:")
            for param in spec.params:
                bits = [f"{param.name}={param.default}"]
                if param.choices:
                    bits.append(f"可选 {list(param.choices)}")
                if param.low is not None or param.high is not None:
                    bits.append(f"范围 [{param.low}, {param.high}]")
                if param.description:
                    bits.append(param.description)
                print("  - " + "  ".join(bits))
        return 0
    payload = registry.describe_all()
    if args.json:
        _dump(payload, True)
    else:
        for item in payload:
            print(f"{item['name']:<10} [{item['family']}] {item['summary']}")
    return 0


def _apply_router_weights(value: str | None) -> None:
    """Load calibration multipliers from a JSON file (``{"weights": {...}}``)."""
    if not value:
        return
    from .router import set_algorithm_weights

    path = Path(value)
    if not path.exists():
        raise EnvironmentError_(f"weights file not found: {value}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    weights = payload.get("weights", payload) if isinstance(payload, dict) else payload
    try:
        set_algorithm_weights(weights)
    except (ValueError, TypeError) as exc:
        raise InputError(f"invalid router weights: {exc}") from exc


def _cmd_route(args) -> int:
    _apply_router_weights(getattr(args, "weights", None))
    ctx = _context_from_args(args)
    rec = route(ctx)
    if args.json:
        _dump(rec.to_dict(), True)
    else:
        print(rec.to_text())
    return 0


def _cmd_init(args) -> int:
    _apply_router_weights(getattr(args, "weights", None))
    # Guard: initialising over a populated run used to silently reset the algorithm and
    # manual_override while keeping the observations, with no decision record.  Now it
    # is an explicit choice (optuna-style ``load_if_exists``).
    policy = getattr(args, "if_exists", "error")
    if Study.exists(args.run):
        if policy == "error":
            raise InputError(
                f"run 目录已存在战役：{args.run}",
                hint="继续（保留数据、可换算法并留痕）: --if-exists=keep；"
                "丢弃配置重建: --if-exists=reset",
            )
        if policy == "keep":
            study = _load_study(args.run)
            report = study.reset_campaign(
                algorithm=args.algorithm or "auto",
                params=_load_json_arg(args.params) if args.params else None,
            )
            if getattr(args, "json", False):
                _dump(report, True)
            else:
                print(
                    f"保留已测数据（{report['observations_kept']} 条）并重设算法："
                    f"{report['before']['algorithm']} -> {report['after']['algorithm']}"
                    f"（已写入 decisions.jsonl）"
                )
            return 0
        if policy == "reset":
            shutil.rmtree(args.run, ignore_errors=True)
    if not args.space:
        # 提前报错：原先 SearchSpace.from_json(None) 会以 TypeError traceback 收场。
        raise InputError(
            "init requires --space FILE",
            hint="先 condopt prepare 生成 space.json，再 init --space <dir>/space.json",
        )
    space = SearchSpace.from_json(args.space)
    spec = None
    spec_path = getattr(args, "spec", None)
    if spec_path:
        payload = _load_json_arg(spec_path)
        spec = Spec.from_dict(payload)
    if args.objective:
        objective = _parse_objective(args.objective)
    elif spec is not None:
        objective = spec.scalar_objective()
    else:
        objective = Objective()
    context = _context_from_args(args)
    if len(objective.constraints) > context.n_constraints:
        context.n_constraints = len(objective.constraints)
    params = _load_json_arg(args.params) if args.params else None
    study = Study.create(
        run_dir=args.run,
        space=space,
        objective=objective,
        spec=spec,
        algorithm=args.algorithm or "auto",
        params=params,
        seed=args.seed or 0,
        context=context,
        budget=args.budget,
        batch_size=args.batch,
        meta={"clip_out_of_bounds": True}
        if getattr(args, "clip_out_of_bounds", False)
        else None,
        auto_route=not args.no_auto_route,
        cost_budget=args.cost_budget,
        patience=args.patience if args.patience is not None else 3,
        stop_on_stagnation=not getattr(args, "budget_first", False),
    )
    payload = {
        "run_dir": str(study.dir),
        "algorithm": study.algorithm,
        "params": study.params,
        "budget": study.budget,
        "batch_size": study.batch_size,
        "context": study.context.to_dict(),
    }
    if args.json:
        _dump(payload, True)
    else:
        print(f"已创建 study: {study.dir}")
        print(f"算法: {study.algorithm}")
        print(json.dumps(study.params, ensure_ascii=False, indent=2))
        print(f"预算: {study.budget}，批量: {study.batch_size}")
        print("下一步: condopt ask --run <dir>")
    return 0


def _cmd_ask(args) -> int:
    study = _load_study(args.run)
    pending = study.ask(args.n, plate=args.plate)
    if args.json:
        _dump(pending, True)
    else:
        where = f"，孔板 {pending[0]['plate']}" if pending[0].get("plate") else ""
        print(f"待测建议（{len(pending)} 条{where}，算法 {pending[0]['algorithm']}）：")
        for item in pending:
            cond = ", ".join(f"{k}={v}" for k, v in item["x"].items())
            well = f" [{item['plate']}/{item['well']}]" if item.get("plate") else ""
            print(f"  [{item['id']}]{well} {cond}")
        print(f"建议表: {study.dir / 'suggestions.csv'}")
        print("测完后: condopt tell --run <dir> --results results.json")
    return 0


def _coerce(value):
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if number.is_integer() and "." not in text and "e" not in lowered else number


def _results_from_csv(text: str) -> list[dict]:
    """Parse measurement CSV.

    Column conventions (all optional except the condition/measurement itself):

    * ``x.<name>`` / ``x:<name>``  -> condition variable;
    * ``obj.<name>`` / ``obj:<name>`` -> objective value;
    * ``con.<name>`` / ``con:<name>`` -> constraint value;
    * ``y`` / ``fidelity`` / ``plate`` / ``well`` / ``id`` / ``source`` / ``cost``;
    * ``objectives`` / ``constraints`` columns may hold JSON objects;
    * plain columns with no prefixes are treated as condition variables.
    """
    import csv
    import io

    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    rows = []
    for raw in reader:
        record: dict = {}
        config: dict = {}
        objectives: dict = {}
        constraints: dict = {}
        prefixed = any(
            str(key).lower().startswith(("x.", "x:", "obj.", "obj:", "con.", "con:"))
            for key in raw
        )
        for key, value in raw.items():
            if key is None:
                continue
            name = str(key).strip()
            lower = name.lower()
            cooked = _coerce(value)
            if lower == "x." or lower == "x:":
                continue
            if lower.startswith(("x.", "x:")):
                config[name[2:]] = cooked
            elif lower.startswith(("obj.", "obj:")):
                objectives[name[4:]] = cooked
            elif lower.startswith(("con.", "con:")):
                constraints[name[4:]] = cooked
            elif lower in ("objectives", "objective"):
                parsed = cooked if isinstance(cooked, dict) else json.loads(str(value))
                objectives.update(parsed)
            elif lower in ("constraints", "constraint"):
                parsed = cooked if isinstance(cooked, dict) else json.loads(str(value))
                constraints.update(parsed)
            elif lower in ("y", "score", "value", "target"):
                record["y"] = cooked
            elif lower in ("fidelity", "f"):
                record["fidelity"] = cooked
            elif lower in ("plate",):
                record["plate"] = cooked
            elif lower in ("well",):
                record["well"] = cooked
            elif lower in ("id", "suggestion_id"):
                record["id"] = cooked
            elif lower in ("source",):
                record["source"] = cooked
            elif lower in ("cost",):
                record["cost"] = cooked
            elif not prefixed:
                config[name] = cooked
        record["x"] = config
        if objectives:
            record["objectives"] = objectives
        if constraints:
            record["constraints"] = constraints
        if "meta" not in record and (record.get("plate") or record.get("well")):
            meta = {}
            if record.get("plate"):
                meta["plate"] = record["plate"]
            if record.get("well"):
                meta["plate_well"] = record["well"]
            record["meta"] = meta
        record.pop("plate", None)
        record.pop("well", None)
        rows.append(record)
    return rows


def _normalize_results(rows: list[dict], study) -> list[dict]:
    """Move plain objective/constraint columns out of the condition dict.

    Lab CSV exports often carry a plain ``yield`` column.  ``_read_results``
    cannot know the study, so objective/constraint/spec metric names are
    re-classified here before space validation.
    """
    objective = study.objective
    metric_names = list(objective.objectives) or [objective.name]
    if study.spec is not None:
        metric_names = list(study.spec.metrics)
    constraint_names = list(objective.constraints)
    if study.spec is not None and study.spec.constraints:
        constraint_names = list(study.spec.constraints)
    for row in rows:
        config = row.get("x") or {}
        if row.get("y") is None and objective.name in config:
            row["y"] = config.pop(objective.name)
        for name in metric_names:
            if name in config:
                row.setdefault("objectives", {})[name] = config.pop(name)
        for name in constraint_names:
            if name in config:
                row.setdefault("constraints", {})[name] = config.pop(name)
    return rows


def _read_results(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise EnvironmentError_(f"results file not found: {path}")
    text = p.read_text(encoding="utf-8-sig")
    stripped = text.strip()
    first_line = stripped.splitlines()[0] if stripped.splitlines() else ""
    looks_json = stripped.startswith(("[", "{"))
    delimited_suffix = p.suffix.lower() in (".csv", ".tsv", ".txt")
    has_separator = any(separator in first_line for separator in (",", ";", "\t"))
    if not looks_json and (delimited_suffix or has_separator):
        return _results_from_csv(text)
    if stripped.startswith("["):
        return json.loads(stripped)
    return [json.loads(line) for line in stripped.splitlines() if line.strip()]


def _cmd_pending(args) -> int:
    study = _load_study(args.run)
    items = list(study.pending)
    if args.json:
        _dump({"run_dir": str(study.dir), "n_pending": len(items), "pending": items}, True)
        return 0
    if not items:
        print("no pending suggestions")
        return 0
    space_names = list(study.space.names)
    print(f"{len(items)} pending suggestion(s) in {study.dir}")
    for item in items:
        condition = ", ".join(
            f"{name}={item['x'].get(name)}" for name in space_names
        )
        print(
            f"  {item['id']}  plate={item.get('plate', '-')} well={item.get('well', '-')}  "
            f"{condition}"
        )
    return 0


def _cmd_tell(args) -> int:
    study = _load_study(args.run)
    results = _normalize_results(_read_results(args.results), study)
    requested_ids = {str(item.get("id")) for item in results if item.get("id")}
    try:
        observations = study.tell(results)
    except ValueError as exc:
        # study.tell 对未知 id 抛 ValueError（已记录的 id 走幂等跳过，不会到这里）；
        # 这是"当前待测批次状态"问题，不是输入格式问题，所以升级为退出码 3。
        if "unknown pending id" in str(exc):
            raise StateError(
                str(exc),
                hint="用 condopt pending --run <dir> --json 查当前待回填 id；"
                "或先 condopt ask --run <dir> 生成建议",
            ) from exc
        raise
    still_pending = [item["id"] for item in study.pending]
    if still_pending:
        study.log_decision(
            action="execution_partial",
            algorithm=study.algorithm,
            reason=(
                f"提交了 {len(requested_ids)} 个 id，仍有 {len(still_pending)} 个 pending"
                "（本轮未提交/结果缺失/被幂等跳过）；未测的孔位可补测后重跑 tell"
            ),
            payload={"submitted_ids": sorted(requested_ids), "still_pending": still_pending},
        )
    payload = {
        "n": len(observations),
        "still_pending": still_pending,
        "best": None
        if study.best() is None
        else {"x": study.best().x, "y": study.best().y},
        "remaining": study.remaining(),
        "rounds": study.rounds,
    }
    if args.json:
        _dump(payload, True)
    else:
        print(f"已记录 {len(observations)} 条实验结果，剩余预算 {study.remaining()} 次。")
        if still_pending:
            print(
                f"注意：{len(still_pending)} 个 id 仍未记录："
                f"{', '.join(str(item) for item in still_pending[:5])}"
                "（缺少测量列或被幂等跳过；补测后重跑 tell 即可）"
            )
        best = study.best()
        if best is not None:
            cond = ", ".join(f"{k}={v}" for k, v in best.x.items())
            print(f"当前最优: {cond}  ->  {best.y}")
        print("下一步: condopt ask --run <dir>")
    return 0


def _cmd_fork(args) -> int:
    old = _load_study(args.run)
    if old.spec is None:
        raise StateError("fork needs a study with a spec", hint="用 prepare 生成 spec 后重新 init")
    records = [old.spec.observation_values(obs) for obs in old.dataset.observations]
    spec_mode = args.spec_mode
    if args.spec_file:
        from .spec import Spec

        payload = json.loads(Path(args.spec_file).read_text(encoding="utf-8-sig"))
        new_spec = Spec.from_dict(payload)
        if new_spec is None:
            raise InputError("empty spec file")
        report = {"changed": [], "spec_file": str(args.spec_file)}
        spec_mode = "file"
    elif args.spec_mode == "calibrated":
        new_spec, report = old.spec.calibrate_observations(
            old.dataset.observations, by_plate=bool(getattr(args, "by_plate", False))
        )
    else:
        groups = (
            [str(obs.meta.get("plate") or "?") for obs in old.dataset.observations]
            if getattr(args, "by_plate", False)
            else None
        )
        new_spec, report = old.spec.suggest_spec(records, groups=groups)
    new_dir = Path(args.new_run)
    if new_dir.exists():
        # 保持 SystemExit：现有契约测试（tests/test_fork.py）以此判定"目标已存在"。
        raise SystemExit(f"new run already exists: {new_dir}")
    meta = dict(old.meta)
    meta.update({
        "forked_from": str(old.dir),
        "fork_spec_mode": spec_mode,
        "fork_changed": report.get("changed", []),
    })
    study = Study.create(
        new_dir,
        old.space,
        spec=new_spec,
        context=old.context,
        budget=int(args.budget or old.budget or 20),
        batch_size=int(args.batch_size or old.batch_size),
        seed=int(args.seed if args.seed is not None else old.seed),
        meta=meta,
    )
    imported = None
    if args.top_k and args.top_k > 0:
        imported = study.transfer_from(old.dir, top_k=args.top_k, reason="fork warm start")
    (new_dir / "fork_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (new_dir / "spec_fork.json").write_text(
        json.dumps(new_spec.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    payload = {
        "old_run": str(old.dir),
        "new_run": str(new_dir),
        "spec_mode": spec_mode,
        "changed": report.get("changed", []),
        "imported": None if imported is None else imported["imported"],
        "used": len(study.dataset),
        "remaining": study.remaining(),
    }
    if args.json:
        _dump(payload, True)
    else:
        print(f"forked {old.dir} -> {new_dir} ({args.spec_mode}); imported={payload['imported']}")
    return 0


def _cmd_import(args) -> int:
    study = _load_study(args.run)
    if args.from_run:
        summary = study.transfer_from(
            args.from_run,
            top_k=args.top_k,
            reason=args.reason or "CLI warm-start",
            prior_reliability=getattr(args, "prior_reliability", 1.0),
        )
        payload = {
            **summary,
            "used": len(study.dataset),
            "remaining": study.remaining(),
            "best": None if study.best() is None else {
                "x": study.best().x, "y": study.best().y,
            },
        }
        if args.json:
            _dump(payload, True)
        else:
            print(f"从 {args.from_run} 迁移 {summary['imported']} 条观测（跳过 {summary['skipped']}）")
        return 0
    results = _normalize_results(_read_results(args.results), study)
    observations = study.import_observations(results)
    payload = {
        "n": len(observations),
        "used": len(study.dataset),
        "remaining": study.remaining(),
        "best": None if study.best() is None else {"x": study.best().x, "y": study.best().y},
    }
    if args.json:
        _dump(payload, True)
    else:
        print(f"导入 {len(observations)} 条历史观测")
    return 0


def _cmd_knee_weights(args) -> int:
    study = _load_study(args.run)
    weights = {}
    for part in str(args.weights).split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise InputError(f"bad weight {part!r}; expected name=value")
        name, value = part.split("=", 1)
        weights[name.strip()] = float(value)
    cleaned = study.set_knee_weights(weights, reason=args.reason or "")
    payload = {"run_dir": str(study.dir), "weights": cleaned}
    if args.json:
        _dump(payload, True)
    else:
        print("knee weights: " + ", ".join(f"{k}={v:g}" for k, v in cleaned.items()))
    return 0


def _cmd_decision(args) -> int:
    study = _load_study(args.run)
    decision = study.decision(explain=args.explain)
    payload = {"run_dir": str(study.dir), "decision": decision}
    (study.dir / "decision.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.json:
        _dump(payload, True)
    else:
        if decision is None:
            print("no observation yet")
        else:
            print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


def _cmd_report(args) -> int:
    study = _load_study(args.run)
    if args.json:
        _dump(study.summary(), True)
    else:
        print(study.report())
    return 0


def _cmd_override(args) -> int:
    study = _load_study(args.run)
    params = _load_json_arg(args.params) if args.params else None
    result = study.override(args.algorithm, params, args.reason)
    _dump(result, args.json)
    return 0


def _cmd_auto(args) -> int:
    study = _load_study(args.run)
    _dump(study.resume_auto_route(), args.json)
    return 0


def _cmd_prepare(args) -> int:
    from .prepare import FIELD_GUIDE, stopping_advice, feasibility_advice, prior_advice
    from .prepare import template, write_problem

    if args.fields:
        _dump(FIELD_GUIDE, True)
        return 0
    if args.template:
        _dump(template(), True)
        return 0
    if not args.answers:
        raise InputError(
            "prepare 需要 --answers FILE（或用 --template 查看模板）",
            hint="py -3 scripts/run_condopt.py prepare --template",
        )
    answers = _load_json_arg(args.answers)
    # A8: 未知键/类型/必填项在这里报错（带 difflib 候选键），不再静默忽略拼错的键。
    validate_answers(answers)
    result = write_problem(answers, args.out)
    if args.json:
        _dump(result, True)
    else:
        print(f"已生成问题文件：{result['out_dir']}")
        for key in ("space", "context", "objective"):
            print(f"  {key}: {result[key]}")
        recommendation = result["recommendation"]
        print(f"router 推荐：{recommendation['algorithm']}（置信度 {recommendation['confidence']}）")
        for reason in recommendation["rationale"][:3]:
            print(f"  - {reason}")
        stopping = result["stopping"]
        print(f"停止口径建议：{stopping['recommended']}（{stopping['why']}）")
        feasibility = result["feasibility"]
        print(
            f"可行率风险：{feasibility['risk']}"
            + (
                f"（估计 {feasibility['estimated_feasible_fraction']:.1%}）"
                if feasibility.get("estimated_feasible_fraction") is not None
                else ""
            )
        )
        if feasibility["risk"] == "high":
            print(f"  - {feasibility['advice']}")
        prior = result["prior"]
        if prior.get("has_prior"):
            print(f"先验数据：{prior['advice']}")
        deliverable = result["deliverable"]
        print(f"交付口径：{deliverable['metric']} — {deliverable['definition']}")
        for item in deliverable["must_include"]:
            print(f"  - 必须包含：{item}")
        noise_plan = result["noise_plan"]
        print(f"噪声处置（{noise_plan['level']}）：{noise_plan['advice']}")
        print("下一步: condopt init --run <run> --space ... --objective ...")
    return 0


def _cmd_render(args) -> int:
    from .render import render_study

    result = render_study(args.run, out_dir=args.out, make_png=not args.no_png)
    if args.json:
        _dump(result, True)
    else:
        print(f"看板: {result['dashboard']}")
        if result.get("convergence_png"):
            print(f"PNG : {result['convergence_png']}")
        for warning in result.get("warnings", []):
            print(f"警告: {warning}")
    return 0


def _cmd_recommend(args) -> int:
    study = _load_study(args.run)
    payload = study.recommend()
    if args.json:
        _dump(payload, True)
    else:
        print(f"推荐条件: {json.dumps(payload['condition'], ensure_ascii=False)}")
        print(f"规格得分: {payload['score']}")
        if payload["spec"] is not None:
            print(f"是否在规格内: {payload['spec']['in_spec']}")
        if payload["confirmation"] is not None:
            stats = payload["confirmation"]
            print(f"确认复孔: n={stats['n']} mean={stats['mean']:.6g} std={stats['std']:.3g}")
        print(f"停止: {payload['should_stop']}（{payload['stop_reason']}）")
    return 0


def _cmd_confirm(args) -> int:
    study = _load_study(args.run)
    pending = study.confirm(args.n)
    if args.json:
        _dump(pending, True)
    else:
        print(f"确认批次（{len(pending)} 条，复制当前最优条件）:")
        for item in pending:
            print(f"  [{item['id']}] {json.dumps(item['x'], ensure_ascii=False)}")
    return 0


def _cmd_decisions(args) -> int:
    study = _load_study(args.run)
    decisions = study.decisions()
    if args.action:
        decisions = [item for item in decisions if item.get("action") == args.action]
    if args.limit is not None:
        decisions = decisions[-int(args.limit):]
    payload = {"run_dir": str(study.dir), "n": len(decisions), "decisions": decisions}
    if args.json:
        _dump(payload, True)
    else:
        for item in decisions:
            print(f"[{item.get('ts', '')}] {item.get('action')} "
                  f"{item.get('algorithm', '')} :: {item.get('reason', '')}")
    return 0


def _cmd_retune(args) -> int:
    study = _load_study(args.run)
    report = study.retune_for_exploration(reason=args.reason or "CLI 停滞重调参")
    if args.json:
        _dump(report, True)
    else:
        print(
            f"{report['algorithm']} params updated ({', '.join(report['changed']) or 'none'}): "
            f"{report['params']}"
        )
    return 0


def _cmd_transfer_impact(args) -> int:
    study = _load_study(args.run)
    report = study.simulate_transfer(
        args.from_run,
        top_k=args.top_k,
        prior_reliability=getattr(args, "prior_reliability", 1.0),
    )
    if args.json:
        _dump(report, True)
    else:
        if not report.get("changed"):
            print("迁移不会改变当前推荐（before == after）")
        else:
            print(f"迁移后推荐变化：{report['before']} -> {report['after']}")
        print(f"导入 {report.get('transferred')} 条；live run 未被改动: {report['live_run_untouched']}")
    return 0


def _cmd_transfer_staged(args) -> int:
    study = _load_study(args.run)
    report = study.transfer_staged(
        args.from_run,
        start_top_k=args.start_top_k,
        wide_top_k=args.top_k,
        after_batches=args.after_batches,
        min_reliability=args.min_reliability,
        prior_reliability=args.prior_reliability,
        force=args.force,
    )
    if args.json:
        _dump(report, True)
    else:
        stage = report.get("stage")
        if stage == "first_look":
            print(f"分阶段迁移：先导入 {report['imported']} 条（top_k={args.start_top_k}）")
            print(f"跑满 {args.after_batches} 批本地观测后再执行本命令，决定是否放大")
        elif stage == "waiting":
            print(f"还差 {args.after_batches - report['local_batches_since_import']} 批本地观测")
        elif stage == "vetoed":
            print(f"审计未通过，保持小 top_k：{report.get('reason')}")
        else:
            print(f"审计通过，已放大到 top_k={args.top_k}（新导入 {report['imported']} 条）")
    return 0


def _cmd_transfer_audit(args) -> int:
    study = _load_study(args.run)
    report = study.assess_transfer(args.from_run)
    if args.json:
        _dump(report, True)
    else:
        if not report.get("ready"):
            print(f"transfer audit not ready: {report.get('reason')}")
        else:
            from .prepare import reliability_grade

            suggestion = report["suggested_reliability"]
            grade = reliability_grade(suggestion, report.get("confidence", "normal"))
            print(
                f"spearman={report['spearman']} mae={report['normalized_mae']} "
                f"-> prior_reliability={suggestion}"
            )
            print(f"评级 {grade['grade']}：{grade['advice']}")
            ci = report.get("spearman_ci")
            if ci:
                print(f"bootstrap 90% CI: {ci['low']} ~ {ci['high']}")
            if report.get("sample_warning"):
                print(f"注意：{report['sample_warning']}")
    return 0


def _cmd_doctor(args) -> int:
    study = _load_study(args.run)
    report = study.check_integrity()
    if args.json:
        _dump(report, True)
    else:
        status = "OK" if report["ok"] else "ISSUES"
        print(f"[{status}] {report['run_dir']}  algorithm={report.get('algorithm')}")
        print(f"  counts: {report['counts']}")
        for issue in report["issues"]:
            print(f"  - {issue}")
    return 0 if report["ok"] else 1


def _cmd_predict(args) -> int:
    study = _load_study(args.run)
    try:
        config = json.loads(args.config)
    except json.JSONDecodeError as exc:
        raise InputError(f"--config must be JSON: {exc}") from exc
    if not isinstance(config, dict):
        raise InputError("--config must be a JSON object of condition variables")
    payload = study.predict(config)
    if args.json:
        _dump(payload, True)
    else:
        condition = ", ".join(f"{k}={v}" for k, v in payload["condition"].items())
        print(f"条件 {condition}")
        for name, stats in payload["predictions"].items():
            print(f"  {name}: {stats['mean']:.4g} ± {stats['std']:.3g}")
        if payload["p_feasible"] is not None:
            print(f"  P(feasible) = {payload['p_feasible']:.3f}")
        if payload["spec_score"] is not None:
            print(f"  spec score = {payload['spec_score']:.4g}")
    return 0


def _cmd_pareto(args) -> int:
    import csv as csv_module

    study = _load_study(args.run)
    summary = study.pareto_summary()
    front = summary["front"]
    space_names = list(study.space.names)
    objective_names = list(study.objective.objectives)
    constraint_names = list(study.objective.constraints)
    header = (
        [f"x.{name}" for name in space_names]
        + [f"obj.{name}" for name in objective_names]
        + [f"con.{name}" for name in constraint_names]
        + ["y", "source"]
    )
    out = Path(args.out) if args.out else study.dir / "pareto_front.csv"
    if front:
        with out.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv_module.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            for item in front:
                row = {f"x.{name}": item["x"].get(name, "") for name in space_names}
                for name in objective_names:
                    row[f"obj.{name}"] = item.get("objectives", {}).get(name, "")
                for name in constraint_names:
                    row[f"con.{name}"] = ""
                row["y"] = item.get("y")
                row["source"] = item.get("source", "")
                writer.writerow(row)
    payload = {
        "run_dir": str(study.dir),
        "out": str(out) if front else None,
        "n_front": summary["n_front"],
        "hypervolume": summary["hypervolume"],
        "reference_point": summary.get("reference_point"),
    }
    if args.json:
        _dump(payload, True)
    else:
        print(f"Pareto 前沿 {summary['n_front']} 个点；HV={summary['hypervolume']}")
        if front:
            print(f"front -> {out}")
    return 0


def _cmd_export(args) -> int:
    import csv as csv_module

    study = _load_study(args.run)
    observations = list(study.dataset.observations)
    objective_names = sorted({name for obs in observations for name in obs.objectives})
    constraint_names = sorted({name for obs in observations for name in obs.constraints})
    space_names = list(study.space.names)
    header = (
        ["id", "batch", "algorithm", "purpose", "source", "plate", "well",
         "fidelity", "cost", "n_replicates"]
        + [f"x.{name}" for name in space_names]
        + [f"obj.{name}" for name in objective_names]
        + [f"con.{name}" for name in constraint_names]
        + ["y"]
    )
    out = Path(args.out) if args.out else study.dir / "dataset_export.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv_module.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for observation in observations:
            row = {
                "id": observation.meta.get("suggestion_id", ""),
                "batch": observation.meta.get("batch", ""),
                "algorithm": observation.meta.get("algorithm", ""),
                "purpose": observation.meta.get("purpose", ""),
                "source": observation.source,
                "plate": observation.meta.get("plate", ""),
                "well": observation.meta.get("plate_well", ""),
                "fidelity": observation.fidelity,
                "cost": observation.cost,
                "n_replicates": observation.n_replicates,
                "y": observation.y,
            }
            for name in space_names:
                row[f"x.{name}"] = observation.x.get(name, "")
            for name in objective_names:
                row[f"obj.{name}"] = observation.objectives.get(name, "")
            for name in constraint_names:
                row[f"con.{name}"] = observation.constraints.get(name, "")
            writer.writerow(row)
    payload = {"run_dir": str(study.dir), "out": str(out), "n_rows": len(observations),
               "columns": header}
    if args.json:
        _dump(payload, True)
    else:
        print(f"exported {len(observations)} rows -> {out}")
    return 0


def _cmd_template(args) -> int:
    import csv as csv_module

    study = _load_study(args.run)
    if not study.pending:
        raise StateError("no pending suggestions; run `ask` first")
    space_names = list(study.space.names)
    objective = study.objective
    if study.spec is not None:
        objective_columns = [f"obj.{goal.metric}" for goal in study.spec.goals]
        constraint_columns = [f"con.{name}" for name in study.spec.constraints]
    elif objective.is_multi_objective:
        objective_columns = [f"obj.{name}" for name in objective.objectives]
        constraint_columns = [f"con.{name}" for name in objective.constraints]
    else:
        objective_columns = ["y"]
        constraint_columns = [f"con.{name}" for name in objective.constraints]
    header = ["id", "plate", "well"]
    from . import registry as _registry

    multi_fidelity = "multi_fidelity" in _registry.spec(study.algorithm).capabilities
    if multi_fidelity:
        header.append("fidelity")
    header += [f"x.{name}" for name in space_names] + objective_columns + constraint_columns
    out = Path(args.out) if args.out else study.dir / "results_template.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv_module.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for item in study.pending:
            row = {"id": item["id"], "plate": item.get("plate", ""), "well": item.get("well", "")}
            if multi_fidelity:
                row["fidelity"] = item.get("fidelity", "")
            for name in space_names:
                row[f"x.{name}"] = item["x"].get(name, "")
            for column in objective_columns + constraint_columns:
                row[column] = ""
            writer.writerow(row)
    payload = {"run_dir": str(study.dir), "out": str(out), "n_pending": len(study.pending),
               "columns": header}
    if args.json:
        _dump(payload, True)
    else:
        print(f"results template -> {out}（{len(study.pending)} 条待测）")
    return 0


def _cmd_digest(args) -> int:
    study = _load_study(args.run)
    decision = study.decision(explain=True)
    action = study.next_action()
    (study.dir / "step.json").write_text(
        json.dumps(action, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    if decision is not None:
        (study.dir / "decision.json").write_text(
            json.dumps({"run_dir": str(study.dir), "decision": decision},
                       ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    calibration = None
    calibration_path = study.dir / "spec_calibration.json"
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8-sig"))
    lines = [f"# condopt 摘要：{study.dir.name}", "", "## 报告", "", study.report(), ""]
    lines += ["## 推荐判定", "", "```json",
              json.dumps(decision, ensure_ascii=False, indent=2, default=str), "```", ""]
    lines += ["## 下一步行动", "", f"- action: `{action['action']}`",
              f"- reason: {action['reason']}", f"- command: `{action['next_command']}`"]
    for code in action.get("suggested_actions", []):
        reason = (action.get("suggested_action_reasons") or {}).get(code)
        lines.append(f"- `{code}`" + (f" — {reason}" if reason else ""))
    lines.append("")
    if calibration is not None:
        lines += ["## 规格校准", "", "```json",
                  json.dumps(calibration, ensure_ascii=False, indent=2), "```", ""]
    if getattr(args, "format", "md") == "html":
        import html as html_module

        def block(title: str, content: str) -> str:
            return (f"<section><h2>{html_module.escape(title)}</h2>"
                    f"<pre>{html_module.escape(content)}</pre></section>")

        actions_html = "".join(
            f"<li><code>{html_module.escape(str(code))}</code>"
            + (f" — {html_module.escape(str((action.get('suggested_action_reasons') or {}).get(code)))}"
               if (action.get("suggested_action_reasons") or {}).get(code) else "")
            + "</li>"
            for code in action.get("suggested_actions", [])
        )
        document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>condopt 摘要 · {html_module.escape(study.dir.name)}</title>
<style>body{{font:14px/1.6 system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem}}
pre{{background:#f6f8fa;padding:12px;border-radius:8px;overflow:auto}}code{{background:#eef1f4;padding:1px 4px}}</style>
</head><body>
<h1>condopt 摘要：{html_module.escape(study.dir.name)}</h1>
{block("报告", study.report())}
{block("推荐判定", json.dumps(decision, ensure_ascii=False, indent=2, default=str))}
<section><h2>下一步行动</h2><p><b>{html_module.escape(str(action['action']))}</b> — {html_module.escape(str(action['reason']))}</p>
<p>命令：<code>{html_module.escape(str(action['next_command']))}</code></p><ul>{actions_html}</ul></section>
{block("规格校准", json.dumps(calibration, ensure_ascii=False, indent=2)) if calibration is not None else ""}
</body></html>"""
        out = Path(args.out) if args.out else study.dir / "digest.html"
        out.write_text(document, encoding="utf-8")
    else:
        out = Path(args.out) if args.out else study.dir / "digest.md"
        out.write_text("\n".join(lines), encoding="utf-8")
    payload = {"run_dir": str(study.dir), "out": str(out),
               "action": action["action"],
               "selection_basis": None if decision is None else decision["selection_basis"]}
    if args.json:
        _dump(payload, True)
    else:
        print(f"digest -> {out}")
    return 0


def _cmd_step(args) -> int:
    study = _load_study(args.run)
    action = study.next_action()
    (study.dir / "step.json").write_text(
        json.dumps(action, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    if args.json:
        _dump(action, True)
        return 0
    print(f"下一步行动：{action['action']}（{action['reason']}）")
    print(f"命令：{action['next_command']}")
    print(f"停止判断：{action['should_stop']}（{action['stop_reason']}）")
    if action["pending"]:
        print(f"待回填建议：{', '.join(action['pending'])}")
    recommendation = action.get("recommendation")
    if recommendation is not None:
        print(f"当前推荐条件：{json.dumps(recommendation['condition'], ensure_ascii=False)}")
        print(f"推荐分数：{recommendation['score']}")
        confirmation = recommendation.get("confirmation")
        if confirmation is not None:
            print(
                f"确认复孔：n={confirmation['n']} mean={confirmation['mean']:.6g} "
                f"std={confirmation['std']:.3g}"
            )
    effect = action.get("batch_effect") or {}
    if effect.get("detected"):
        print(f"⚠ {effect['message']}")
    return 0


def _cmd_suggest_spec(args) -> int:
    study = _load_study(args.run)
    if study.spec is None:
        raise StateError("this study has no spec; create it with --spec or prepare spec first")
    suggested, report = study.spec.suggest_spec(
        [study.spec.observation_values(obs) for obs in study.dataset.observations],
        min_samples=args.min_samples,
        tolerance_sigmas=args.tolerance_sigmas,
    )
    out = Path(args.out) if args.out else study.dir / "spec_suggested.json"
    out.write_text(json.dumps(suggested.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    payload = {"run_dir": str(study.dir), "out": str(out),
               "spec": suggested.to_dict(), "report": report}
    if args.json:
        _dump(payload, True)
    else:
        print(f"suggested spec -> {out}")
        print(json.dumps(report.get("changed", []), ensure_ascii=False, indent=2))
    return 0


def _cmd_calibrate_spec(args) -> int:
    study = _load_study(args.run)
    if study.spec is None:
        raise StateError("this study has no spec; create it with --spec or prepare spec first")
    calibrated, report = study.spec.calibrate_observations(
        study.dataset.observations,
        by_plate=bool(getattr(args, "by_plate", False)),
        min_samples=args.min_samples,
        tolerance_sigmas=args.tolerance_sigmas,
    )
    out = Path(args.out) if args.out else study.dir / "spec_calibrated.json"
    out.write_text(json.dumps(calibrated.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (study.dir / "spec_calibration.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    payload = {"run_dir": str(study.dir), "out": str(out),
               "spec": calibrated.to_dict(), "report": report}
    if args.json:
        _dump(payload, True)
    else:
        print(f"calibrated spec -> {out}")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="condopt",
        description="条件优化算法库 + 算法 router（AI 自动实验室前置技术）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_spec = sub.add_parser("spec", help="列出/查看算法与参数 schema")
    p_spec.add_argument("--algorithm", help="算法名（bayes/sobol/...）")
    p_spec.add_argument("--json", action="store_true")
    p_spec.set_defaults(func=_cmd_spec)

    def add_context_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--space", help="搜索空间 JSON 文件")
        p.add_argument("--context", help="完整 OptimizationContext JSON 文件")
        p.add_argument("--budget", type=int, help="总实验预算（次）")
        p.add_argument("--batch", type=int, help="批量大小（并行实验数）")
        p.add_argument("--noise", choices=["none", "low", "medium", "high"])
        p.add_argument("--smoothness", choices=["smooth", "mixed", "rough"])
        p.add_argument("--compute", choices=["low", "medium", "high"])
        p.add_argument("--domain", choices=["chemistry", "materials", "process", "generic"])
        p.add_argument("--objectives", type=int, help="目标个数")
        p.add_argument("--constraints", type=int, help="约束个数")
        p.add_argument("--prior", type=int, help="已有先验数据条数")
        p.add_argument("--multi-fidelity", action="store_true")
        p.add_argument("--notes", help="自由备注，供 router/AI 参考")

    p_route = sub.add_parser("route", help="根据问题上下文推荐算法与参数")
    add_context_args(p_route)
    p_route.add_argument("--weights", help="校准权重 JSON（{\"weights\": {\"bayes\": 1.2}}）")
    p_route.add_argument("--json", action="store_true")
    p_route.set_defaults(func=_cmd_route)

    p_init = sub.add_parser("init", help="创建 study（algorithm=auto 时自动路由）")
    p_init.add_argument(
        "--clip-out-of-bounds", action="store_true",
        help="把越界测量显式夹到变量边界并写入 decisions.jsonl（默认拒绝越界值）",
    )
    p_init.add_argument(
        "--if-exists", choices=("error", "keep", "reset"), default="error",
        help="run 目录已有战役时：error（默认拒绝）/ keep（保留数据、换算法并留痕）/ reset（丢弃配置重建）",
    )
    p_init.add_argument("--run", required=True, help="study 目录")
    p_init.add_argument("--objective", help="目标：yield:max，或 JSON，或 JSON 文件")
    p_init.add_argument("--algorithm", default="auto")
    p_init.add_argument("--params", help="算法参数 JSON 或文件")
    p_init.add_argument("--seed", type=int, default=0)
    p_init.add_argument("--no-auto-route", action="store_true")
    p_init.add_argument("--weights", help="校准权重 JSON（auto 路由时生效）")
    p_init.add_argument("--cost-budget", type=float, help="成本预算（多保真 study 的 cost_budget）")
    p_init.add_argument("--spec", help="规格 JSON（goals/constraints，替代 --objective）")
    p_init.add_argument(
        "--patience", type=int, default=None,
        help="停滞判定轮数（默认 3，最小 2）",
    )
    p_init.add_argument(
        "--budget-first", action="store_true",
        help="预算优先：停滞不再触发停止（stop_on_stagnation=False）",
    )
    add_context_args(p_init)
    p_init.add_argument("--json", action="store_true")
    p_init.set_defaults(func=_cmd_init)

    p_ask = sub.add_parser("ask", help="生成下一批待测条件")
    p_ask.add_argument("--run", required=True)
    p_ask.add_argument("--n", type=int, default=None)
    p_ask.add_argument("--plate", help="孔板编号（默认自动 P1、P2…；同一板续测时显式指定）")
    p_ask.add_argument("--json", action="store_true")
    p_ask.set_defaults(func=_cmd_ask)

    p_tell = sub.add_parser("tell", help="记录实验结果（JSON 或 JSONL 文件）")
    p_tell.add_argument("--run", required=True)
    p_tell.add_argument("--results", required=True)
    p_tell.add_argument("--json", action="store_true")
    p_tell.set_defaults(func=_cmd_tell)

    p_fork = sub.add_parser("fork", help="用校准/建议后的 spec 从旧 study fork 新 campaign")
    p_fork.add_argument("--run", required=True)
    p_fork.add_argument("--new-run", required=True)
    p_fork.add_argument("--spec-mode", choices=("calibrated", "suggested"), default="suggested")
    p_fork.add_argument("--spec-file", help="直接使用外部 spec JSON（优先于 --spec-mode）")
    p_fork.add_argument("--top-k", type=int, default=5, help="迁移旧观测数（0 表示不迁移；建议 1–5）")
    p_fork.add_argument("--budget", type=int, help="新 campaign 预算（默认沿用旧 budget）")
    p_fork.add_argument("--batch-size", type=int, help="新 campaign batch（默认沿用旧 batch）")
    p_fork.add_argument("--seed", type=int, help="新 campaign seed（默认沿用旧 seed）")
    p_fork.add_argument("--by-plate", action="store_true",
                        help="按板内尺度校准/建议（忽略板间偏移）")
    p_fork.add_argument("--json", action="store_true")
    p_fork.set_defaults(func=_cmd_fork)

    p_import = sub.add_parser("import", help="导入历史实验数据作为先验（计入预算）")
    p_import.add_argument("--run", required=True)
    import_source = p_import.add_mutually_exclusive_group(required=True)
    import_source.add_argument("--results", help="results JSON/JSONL 文件")
    import_source.add_argument("--from-run", help="从另一个兼容的 study 迁移观测")
    p_import.add_argument(
        "--top-k",
        type=int,
        help="只迁移得分最高的 k 条（--from-run）；建议 1–3，源不可靠时负迁移与 k 同增",
    )
    p_import.add_argument(
        "--prior-reliability",
        type=float,
        default=1.0,
        help="先验可信度 (0,1]；<1 会抬高迁移点的观测噪声，避免坏源压倒新证据",
    )
    p_import.add_argument("--reason", help="迁移/导入理由（审计）")
    p_import.add_argument("--json", action="store_true")
    p_import.set_defaults(func=_cmd_import)

    p_knee = sub.add_parser("knee-weights", help="设置多目标 knee 的目标权重")
    p_knee.add_argument("--run", required=True)
    p_knee.add_argument("--weights", required=True, help="逗号分隔，如 yield=2,cost=1")
    p_knee.add_argument("--reason", help="设置理由（审计）")
    p_knee.add_argument("--json", action="store_true")
    p_knee.set_defaults(func=_cmd_knee_weights)

    p_decision = sub.add_parser("decision", help="查看推荐判定（selection_basis/spec/板效应）")
    p_decision.add_argument("--run", required=True)
    p_decision.add_argument("--explain", action="store_true", help="附推荐理由链")
    p_decision.add_argument("--json", action="store_true")
    p_decision.set_defaults(func=_cmd_decision)

    p_report = sub.add_parser("report", help="查看 study 进度与下一轮建议")
    p_report.add_argument("--run", required=True)
    p_report.add_argument("--json", action="store_true")
    p_report.set_defaults(func=_cmd_report)

    p_override = sub.add_parser("override", help="人工/AI 覆写算法（必须给理由并留痕）")
    p_override.add_argument("--run", required=True)
    p_override.add_argument("--algorithm", required=True)
    p_override.add_argument("--params", help="算法参数 JSON 或文件")
    p_override.add_argument("--reason", required=True)
    p_override.add_argument("--json", action="store_true")
    p_override.set_defaults(func=_cmd_override)

    p_auto = sub.add_parser("auto", help="覆写后恢复 router 自动重路由")
    p_auto.add_argument("--run", required=True)
    p_auto.add_argument("--json", action="store_true")
    p_auto.set_defaults(func=_cmd_auto)

    p_prepare = sub.add_parser("prepare", help="把 skill 问卷答案转成 space/context/objective JSON")
    p_prepare.add_argument("--answers", help="answers.json（问卷答案）")
    p_prepare.add_argument("--out", default="workspace", help="输出目录")
    p_prepare.add_argument("--template", action="store_true", help="打印答案模板")
    p_prepare.add_argument(
        "--fields", action="store_true",
        help="打印 prepare 输出字段说明（agent 自描述用）",
    )
    p_prepare.add_argument("--json", action="store_true")
    p_prepare.set_defaults(func=_cmd_prepare)

    p_recommend = sub.add_parser("recommend", help="给出可执行推荐（规格符合性/确认复孔/停止）")
    p_recommend.add_argument("--run", required=True)
    p_recommend.add_argument("--json", action="store_true")
    p_recommend.set_defaults(func=_cmd_recommend)

    p_suggest_spec = sub.add_parser("suggest-spec", help="用历史观测建议可达的 spec（target/range 越界时调整）")
    p_suggest_spec.add_argument("--run", required=True)
    p_suggest_spec.add_argument("--out", help="输出 spec JSON（默认 run/spec_suggested.json）")
    p_suggest_spec.add_argument("--min-samples", type=int, default=3)
    p_suggest_spec.add_argument("--tolerance-sigmas", type=float, default=2.0)
    p_suggest_spec.add_argument("--json", action="store_true")
    p_suggest_spec.set_defaults(func=_cmd_suggest_spec)

    p_calibrate_spec = sub.add_parser("calibrate-spec", help="用历史观测校准 spec 的 scale/tolerance")
    p_calibrate_spec.add_argument("--run", required=True)
    p_calibrate_spec.add_argument("--out", help="输出 spec JSON（默认 run/spec_calibrated.json）")
    p_calibrate_spec.add_argument("--min-samples", type=int, default=3)
    p_calibrate_spec.add_argument("--tolerance-sigmas", type=float, default=2.0)
    p_calibrate_spec.add_argument("--by-plate", action="store_true",
                                  help="按板内尺度校准（忽略板间偏移）")
    p_calibrate_spec.add_argument("--json", action="store_true")
    p_calibrate_spec.set_defaults(func=_cmd_calibrate_spec)

    p_confirm = sub.add_parser("confirm", help="复制当前最优条件做确认复孔")
    p_confirm.add_argument("--run", required=True)
    p_confirm.add_argument("--n", type=int, default=3)
    p_confirm.add_argument("--json", action="store_true")
    p_confirm.set_defaults(func=_cmd_confirm)

    p_decisions = sub.add_parser("decisions", help="查看审计日志（router/override/transfer 等）")
    p_decisions.add_argument("--run", required=True)
    p_decisions.add_argument("--action", help="只显示某类 action")
    p_decisions.add_argument("--limit", type=int, help="只显示最后 N 条")
    p_decisions.add_argument("--json", action="store_true")
    p_decisions.set_defaults(func=_cmd_decisions)

    p_retune = sub.add_parser(
        "retune", help="停滞时调整探索相关参数（acquisition/kappa），保持同一算法"
    )
    p_retune.add_argument("--run", required=True)
    p_retune.add_argument("--reason", default="")
    p_retune.add_argument("--json", action="store_true")
    p_retune.set_defaults(func=_cmd_retune)

    p_impact = sub.add_parser(
        "transfer-impact", help="演练迁移对当前推荐的影响（在副本上，不改动 live run）"
    )
    p_impact.add_argument("--run", required=True)
    p_impact.add_argument("--from-run", required=True)
    p_impact.add_argument("--top-k", type=int, default=None)
    p_impact.add_argument("--prior-reliability", type=float, default=1.0)
    p_impact.add_argument("--json", action="store_true")
    p_impact.set_defaults(func=_cmd_transfer_impact)

    p_audit = sub.add_parser(
        "transfer-audit", help="用目标已测数据评估源战役可信度（建议 prior_reliability）"
    )
    p_audit.add_argument("--run", required=True, help="目标 study（需已有 >=3 条观测）")
    p_audit.add_argument("--from-run", required=True, help="源 study")
    p_audit.add_argument("--json", action="store_true")
    p_staged = sub.add_parser(
        "transfer-staged",
        help="分阶段迁移：先小 top_k，跑几批后按审计决定是否放大（research/TRANSFER_ADAPTIVE.md）",
    )
    p_staged.add_argument("--run", required=True)
    p_staged.add_argument("--from-run", required=True)
    p_staged.add_argument("--start-top-k", type=int, default=1)
    p_staged.add_argument("--top-k", type=int, default=5, help="放大后的 top_k")
    p_staged.add_argument("--after-batches", type=int, default=2)
    p_staged.add_argument(
        "--min-reliability", type=float, default=0.6,
        help="否决线：审计建议低于此值就不放大（实测 0.4–0.6 安全）",
    )
    p_staged.add_argument("--prior-reliability", type=float, default=1.0)
    p_staged.add_argument("--force", action="store_true", help="跳过否决线（需自行留痕）")
    p_staged.add_argument("--json", action="store_true")
    p_staged.set_defaults(func=_cmd_transfer_staged)
    p_audit.set_defaults(func=_cmd_transfer_audit)

    p_pending = sub.add_parser("pending", help="列出未回填的 pending 建议（执行器轮询用）")
    p_pending.add_argument("--run", required=True)
    p_pending.add_argument("--json", action="store_true")
    p_pending.set_defaults(func=_cmd_pending)

    p_doctor = sub.add_parser("doctor", help="自检 run 目录一致性（文件/ID/状态）")
    p_doctor.add_argument("--run", required=True)
    p_doctor.add_argument("--json", action="store_true")
    p_doctor.set_defaults(func=_cmd_doctor)

    p_predict = sub.add_parser("predict", help="对新条件做代理预测（mean/std/P(feasible)/spec）")
    p_predict.add_argument("--run", required=True)
    p_predict.add_argument("--config", required=True, help="条件 JSON，例如 '{\"T\": 80}'")
    p_predict.add_argument("--json", action="store_true")
    p_predict.set_defaults(func=_cmd_predict)

    p_pareto = sub.add_parser("pareto", help="导出 Pareto 前沿 CSV / 查看 HV")
    p_pareto.add_argument("--run", required=True)
    p_pareto.add_argument("--out", help="输出 CSV（默认 run/pareto_front.csv）")
    p_pareto.add_argument("--json", action="store_true")
    p_pareto.set_defaults(func=_cmd_pareto)

    p_export = sub.add_parser("export", help="导出观测数据为 CSV（含 x/obj/con/plate）")
    p_export.add_argument("--run", required=True)
    p_export.add_argument("--out", help="输出 CSV（默认 run/dataset_export.csv）")
    p_export.add_argument("--json", action="store_true")
    p_export.set_defaults(func=_cmd_export)

    p_template = sub.add_parser("template", help="为当前 pending 生成实测回填 CSV 模板")
    p_template.add_argument("--run", required=True)
    p_template.add_argument("--out", help="输出 CSV（默认 run/results_template.csv）")
    p_template.add_argument("--json", action="store_true")
    p_template.set_defaults(func=_cmd_template)

    p_digest = sub.add_parser("digest", help="导出单页摘要（报告 + 推荐判定 + 下一步 + 校准）")
    p_digest.add_argument("--run", required=True)
    p_digest.add_argument("--out", help="输出文件（默认 run/digest.md 或 digest.html）")
    p_digest.add_argument("--format", choices=("md", "html"), default="md")
    p_digest.add_argument("--json", action="store_true")
    p_digest.set_defaults(func=_cmd_digest)

    p_step = sub.add_parser("step", help="自动循环下一步（tell/confirm/ask/stop 机器可读决策）")
    p_step.add_argument("--run", required=True)
    p_step.add_argument("--json", action="store_true")
    p_step.set_defaults(func=_cmd_step)

    p_render = sub.add_parser("render", help="生成离线看板 dashboard.html + convergence.png")
    p_render.add_argument("--run", required=True)
    p_render.add_argument("--out", help="输出目录；默认写 run 目录")
    p_render.add_argument("--no-png", action="store_true")
    p_render.add_argument("--json", action="store_true")
    p_render.set_defaults(func=_cmd_render)

    # 每个子命令都支持 --debug：agent 先看一行 JSON，人类排障时再要 traceback。
    for subparser in sub.choices.values():
        subparser.add_argument(
            "--debug",
            action="store_true",
            help="出错时在 JSON 之后打印完整 traceback（排障用）",
        )

    return parser


def _write_stderr(text: str) -> None:
    """把错误输出固定写成 UTF-8 字节。

    Windows 控制台/重定向流的默认编码可能是 cp936，而错误包是给 agent
    解析的 JSON 契约——字节必须是确定的 UTF-8，不能随代码页漂移。
    pytest 的 capsys 等文本替身没有 ``buffer``，退回文本写。
    """
    buffer = getattr(sys.stderr, "buffer", None)
    if buffer is None:
        sys.stderr.write(text)
        return
    sys.stderr.flush()
    buffer.write(text.encode("utf-8"))
    buffer.flush()


def _emit_error(exc: BaseException, debug: bool = False) -> int:
    """已知错误 → stderr 一行 JSON + 退出码；``--debug`` 时追加完整 traceback。"""
    chunks = [json.dumps(error_payload(exc), ensure_ascii=False) + "\n"]
    if debug:
        chunks.append(traceback.format_exc())
    _write_stderr("".join(chunks))
    return exit_code_for(exc)


def _force_utf8_stdio() -> None:
    """Emit UTF-8 on stdout/stderr regardless of the Windows console codepage.

    The console here is cp936; an agent or pipeline that reads our JSON as UTF-8 got a
    ``UnicodeDecodeError`` and ``None`` instead of the payload (measured:
    ``tests/test_benchmark_smoke.py::test_noise_floor_script_runs``).
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover - exotic streams
            pass


def main(argv: Sequence[str] | None = None) -> int:
    _force_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    try:
        return int(args.func(args))
    except CondoptError as exc:
        return _emit_error(exc, debug=bool(getattr(args, "debug", False)))
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        # 库层（study/space/objective/spec）抛出的已知错误同样收敛成结构化错误包；
        # 未知异常保持原样抛出，避免把真 bug 伪装成"用户输入问题"。
        return _emit_error(exc, debug=bool(getattr(args, "debug", False)))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
