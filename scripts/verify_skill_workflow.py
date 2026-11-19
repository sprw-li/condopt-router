"""Doc-executable test: run the workflow exactly as ``skill/SKILL.md`` describes.

The skill tells the agent to: ask the questionnaire -> ``prepare`` -> ``init`` ->
``step`` -> ``tell`` -> ``digest``.  This script performs that sequence in a temp
workspace and asserts every artifact the skill mentions actually appears, so the
documentation cannot drift away from the CLI.

Run:
    py -3 scripts/verify_skill_workflow.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

def _utf8_child_env() -> dict:
    """Environment for child processes so their stdio is UTF-8, not the codepage."""
    import os

    return {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "run_condopt.py"


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT,
    )


def _answers() -> dict:
    return {
        "variables": [
            {"name": "temp", "kind": "continuous", "low": 40.0, "high": 100.0},
            {"name": "time", "kind": "continuous", "low": 1.0, "high": 24.0},
            {"name": "solvent", "kind": "categorical",
             "choices": ["THF", "DMF", "MeCN", "toluene"]},
        ],
        "objective": {"name": "yield", "direction": "max"},
        "budget": 12,
        "batch": 3,
        "noise": "medium",
        "smoothness": "smooth",
    }


def run_workflow(workspace: Path) -> dict:
    workspace.mkdir(parents=True, exist_ok=True)
    answers_path = workspace / "answers.json"
    answers_path.write_text(json.dumps(_answers(), ensure_ascii=False), encoding="utf-8", errors="replace")

    problems: list[str] = []
    prepare = _run(["prepare", "--answers", str(answers_path), "--out", str(workspace)])
    if prepare.returncode != 0:
        problems.append(f"prepare failed: {prepare.stderr.strip()[:300]}")
    for name in ("space.json", "context.json", "objective.json"):
        if not (workspace / name).exists():
            problems.append(f"prepare did not write {name}")
    # the skill promises stopping / feasibility / deliverable advice in the output
    for token in ("停止", "交付口径"):
        if token not in prepare.stdout:
            problems.append(f"prepare output lost {token!r}")

    run_dir = workspace / "run"
    init = _run([
        "init", "--run", str(run_dir),
        "--space", str(workspace / "space.json"),
        "--objective", str(workspace / "objective.json"),
        "--context", str(workspace / "context.json"),
        "--budget", "12", "--batch", "3", "--seed", "0",
    ])
    if init.returncode != 0:
        problems.append(f"init failed: {init.stderr.strip()[:300]}")

    step = _run(["step", "--run", str(run_dir), "--json"])
    action = {}
    try:
        action = json.loads(step.stdout)
    except json.JSONDecodeError:
        problems.append(f"step --json did not return JSON: {step.stdout[:200]}")
    if action.get("action") not in {"ask", "tell", "confirm", "stop", None}:
        problems.append(f"unexpected step action: {action.get('action')}")

    # one ask + tell round, using obviously synthetic values
    ask = _run(["ask", "--run", str(run_dir), "--n", "3"])
    if ask.returncode != 0:
        problems.append(f"ask failed: {ask.stderr.strip()[:300]}")
    pending_path = run_dir / "pending.json"
    pending = []
    if pending_path.exists():
        pending = json.loads(pending_path.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(pending, list):
            pending = pending.get("pending", [])
    if len(pending) != 3:
        problems.append(f"ask produced {len(pending)} pending suggestions (expected 3)")
    results_path = workspace / "results.json"
    rows = [
        {"id": item.get("id"), "y": 1.0, "fidelity": 1.0}
        for item in pending[:3]
        if item.get("id")
    ]
    results_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8", errors="replace")
    tell = _run(["tell", "--run", str(run_dir), "--results", str(results_path)])
    if tell.returncode != 0:
        problems.append(f"tell failed: {tell.stderr.strip()[:300]}")
    dataset_lines = [
        line for line in (run_dir / "dataset.jsonl").read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ] if (run_dir / "dataset.jsonl").exists() else []
    if len(dataset_lines) < 3:
        problems.append(f"tell recorded {len(dataset_lines)} observations (expected 3)")

    digest_path = workspace / "digest.md"
    digest = _run(["digest", "--run", str(run_dir), "--out", str(digest_path)])
    if digest.returncode != 0:
        problems.append(f"digest failed: {digest.stderr.strip()[:300]}")
    if not digest_path.exists():
        problems.append("digest.md was not written")
    else:
        text = digest_path.read_text(encoding="utf-8", errors="replace")
        for token in ("交付自查", "预算"):
            if token not in text:
                problems.append(f"digest lost {token!r}")

    for name in ("dataset.jsonl", "decisions.jsonl"):
        if not (run_dir / name).exists():
            problems.append(f"run directory lost {name}")

    # 第 6 步文档里的 spec 窗口流程：prepare 写出 spec.json，init --spec 可用，recommend 给出符合性
    spec_answers = _answers()
    spec_answers["spec"] = {
        "goals": [{"metric": "yield", "mode": "target", "target": 80.0, "tolerance": 2.0, "scale": 5.0}]
    }
    spec_answers_path = workspace / "answers_spec.json"
    spec_answers_path.write_text(
        json.dumps(spec_answers, ensure_ascii=False), encoding="utf-8", errors="replace"
    )
    spec_dir = workspace / "spec_case"
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_prepare = _run(
        ["prepare", "--answers", str(spec_answers_path), "--out", str(spec_dir)]
    )
    if spec_prepare.returncode != 0:
        problems.append(f"prepare(spec) failed: {spec_prepare.stderr.strip()[:300]}")
    if not (spec_dir / "spec.json").exists():
        problems.append("prepare(spec) did not write spec.json")
    spec_run = spec_dir / "run"
    spec_init = _run([
        "init", "--run", str(spec_run),
        "--space", str(spec_dir / "space.json"),
        "--objective", str(spec_dir / "objective.json"),
        "--context", str(spec_dir / "context.json"),
        "--spec", str(spec_dir / "spec.json"),
        "--budget", "9", "--batch", "3", "--seed", "0",
    ])
    if spec_init.returncode != 0:
        problems.append(f"init --spec failed: {spec_init.stderr.strip()[:300]}")
    spec_ask = _run(["ask", "--run", str(spec_run), "--n", "3"])
    if spec_ask.returncode != 0:
        problems.append(f"ask(spec) failed: {spec_ask.stderr.strip()[:300]}")
    spec_pending = []
    if (spec_run / "pending.json").exists():
        spec_pending = json.loads((spec_run / "pending.json").read_text(encoding="utf-8", errors="replace"))
        if not isinstance(spec_pending, list):
            spec_pending = spec_pending.get("pending", [])
    if len(spec_pending) != 3:
        problems.append(f"ask(spec) produced {len(spec_pending)} suggestions (expected 3)")
    spec_results = workspace / "results_spec.json"
    spec_results.write_text(
        json.dumps(
            [
                {"id": item.get("id"), "y": 80.0, "fidelity": 1.0}
                for item in spec_pending[:3]
                if item.get("id")
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8", errors="replace",
    )
    _run(["tell", "--run", str(spec_run), "--results", str(spec_results)])
    recommend = _run(["recommend", "--run", str(spec_run), "--json"])
    if recommend.returncode != 0:
        problems.append(f"recommend(spec) failed: {recommend.stderr.strip()[:300]}")
    else:
        try:
            payload = json.loads(recommend.stdout)
        except json.JSONDecodeError:
            payload = {}
            problems.append("recommend --json did not return JSON")
        if payload and not any(
            key in json.dumps(payload, ensure_ascii=False) for key in ("spec", "符合")
        ):
            problems.append("recommend output lost the spec conformance info")

    # 多目标 + 约束：文档里给了 Pareto/可行前沿的表格，这里验证它真能跑出来
    moo_answers = {
        "variables": [
            {"name": "temp", "kind": "continuous", "low": 40.0, "high": 100.0},
            {"name": "time", "kind": "continuous", "low": 1.0, "high": 24.0},
        ],
        "objective": {
            "name": "yield",
            "direction": "max",
            "objectives": ["yield", "cost"],
            "directions": ["max", "min"],
            "constraints": ["impurity"],
            "constraint_limits": {"impurity": {"op": "<=", "value": 2.0}},
        },
        "budget": 9,
        "batch": 3,
        "noise": "medium",
    }
    moo_answers_path = workspace / "answers_moo.json"
    moo_answers_path.write_text(
        json.dumps(moo_answers, ensure_ascii=False), encoding="utf-8", errors="replace"
    )
    moo_dir = workspace / "moo_case"
    moo_dir.mkdir(parents=True, exist_ok=True)
    moo_prepare = _run(["prepare", "--answers", str(moo_answers_path), "--out", str(moo_dir)])
    if moo_prepare.returncode != 0:
        problems.append(f"prepare(moo) failed: {moo_prepare.stderr.strip()[:300]}")
    moo_run = moo_dir / "run"
    moo_init = _run([
        "init", "--run", str(moo_run),
        "--space", str(moo_dir / "space.json"),
        "--objective", str(moo_dir / "objective.json"),
        "--context", str(moo_dir / "context.json"),
        "--budget", "9", "--batch", "3", "--seed", "0",
    ])
    if moo_init.returncode != 0:
        problems.append(f"init(moo) failed: {moo_init.stderr.strip()[:300]}")
    _run(["ask", "--run", str(moo_run), "--n", "3"])
    moo_pending = []
    if (moo_run / "pending.json").exists():
        moo_pending = json.loads((moo_run / "pending.json").read_text(encoding="utf-8", errors="replace"))
        if not isinstance(moo_pending, list):
            moo_pending = moo_pending.get("pending", [])
    if len(moo_pending) != 3:
        problems.append(f"ask(moo) produced {len(moo_pending)} suggestions (expected 3)")
    moo_results = workspace / "results_moo.json"
    moo_results.write_text(
        json.dumps(
            [
                {
                    "id": item.get("id"),
                    "y": 50.0 + index,
                    "objectives": {"yield": 50.0 + index, "cost": 3.0 - 0.5 * index},
                    "constraints": {"impurity": 1.0 + 0.2 * index},
                    "fidelity": 1.0,
                }
                for index, item in enumerate(moo_pending[:3])
                if item.get("id")
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8", errors="replace",
    )
    moo_tell = _run(["tell", "--run", str(moo_run), "--results", str(moo_results)])
    if moo_tell.returncode != 0:
        problems.append(f"tell(moo) failed: {moo_tell.stderr.strip()[:300]}")
    moo_digest_path = workspace / "digest_moo.md"
    _run(["digest", "--run", str(moo_run), "--out", str(moo_digest_path)])
    moo_digest = (
        moo_digest_path.read_text(encoding="utf-8", errors="replace") if moo_digest_path.exists() else ""
    )
    for token in ("Pareto", "可交付"):
        if token not in moo_digest:
            problems.append(f"moo digest lost {token!r}")

    # 多保真：文档要求 ask 返回 fidelity、tell 带上 fidelity/cost、digest 有全保真口径
    mf_answers = {
        "variables": [
            {"name": "temp", "kind": "continuous", "low": 40.0, "high": 100.0},
            {"name": "time", "kind": "continuous", "low": 1.0, "high": 24.0},
        ],
        "objective": {"name": "yield", "direction": "max"},
        "budget": 9,
        "batch": 3,
        "noise": "medium",
        "multi_fidelity": True,
    }
    mf_answers_path = workspace / "answers_mf.json"
    mf_answers_path.write_text(json.dumps(mf_answers, ensure_ascii=False), encoding="utf-8", errors="replace")
    mf_dir = workspace / "mf_case"
    mf_dir.mkdir(parents=True, exist_ok=True)
    mf_prepare = _run(["prepare", "--answers", str(mf_answers_path), "--out", str(mf_dir)])
    if mf_prepare.returncode != 0:
        problems.append(f"prepare(mf) failed: {mf_prepare.stderr.strip()[:300]}")
    mf_context = {}
    if (mf_dir / "context.json").exists():
        mf_context = json.loads((mf_dir / "context.json").read_text(encoding="utf-8", errors="replace"))
    if not mf_context.get("multi_fidelity"):
        problems.append("prepare(mf) did not mark the context multi_fidelity")
    mf_run = mf_dir / "run"
    mf_init = _run([
        "init", "--run", str(mf_run),
        "--space", str(mf_dir / "space.json"),
        "--objective", str(mf_dir / "objective.json"),
        "--context", str(mf_dir / "context.json"),
        "--multi-fidelity",
        "--budget", "9", "--batch", "3", "--seed", "0",
    ])
    if mf_init.returncode != 0:
        problems.append(f"init(mf) failed: {mf_init.stderr.strip()[:300]}")
    _run(["ask", "--run", str(mf_run), "--n", "3"])
    mf_pending = []
    if (mf_run / "pending.json").exists():
        mf_pending = json.loads((mf_run / "pending.json").read_text(encoding="utf-8", errors="replace"))
        if not isinstance(mf_pending, list):
            mf_pending = mf_pending.get("pending", [])
    if len(mf_pending) != 3:
        problems.append(f"ask(mf) produced {len(mf_pending)} suggestions (expected 3)")
    if mf_pending and not all("fidelity" in item for item in mf_pending):
        problems.append("ask(mf) suggestions lost the fidelity field")
    mf_results = workspace / "results_mf.json"
    mf_results.write_text(
        json.dumps(
            [
                {
                    "id": item.get("id"),
                    "y": 40.0 + index * 5,
                    "fidelity": float(item.get("fidelity", 1.0)),
                    "cost": float(item.get("fidelity", 1.0)),
                }
                for index, item in enumerate(mf_pending[:3])
                if item.get("id")
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8", errors="replace",
    )
    mf_tell = _run(["tell", "--run", str(mf_run), "--results", str(mf_results)])
    if mf_tell.returncode != 0:
        problems.append(f"tell(mf) failed: {mf_tell.stderr.strip()[:300]}")
    mf_digest_path = workspace / "digest_mf.md"
    _run(["digest", "--run", str(mf_run), "--out", str(mf_digest_path)])
    mf_digest = (
        mf_digest_path.read_text(encoding="utf-8", errors="replace") if mf_digest_path.exists() else ""
    )
    if "全保真" not in mf_digest and "保真" not in mf_digest:
        problems.append("mf digest lost the fidelity information")

    # 迁移：文档承诺 import --from-run / transfer-audit / transfer-impact 都要能用，
    # 且迁移的观测要计入目标预算（budget_charged）
    source_run = workspace / "transfer_source"
    source_init = _run([
        "init", "--run", str(source_run),
        "--space", str(workspace / "space.json"),
        "--objective", str(workspace / "objective.json"),
        "--context", str(workspace / "context.json"),
        "--budget", "6", "--batch", "3", "--seed", "11",
    ])
    if source_init.returncode != 0:
        problems.append(f"init(source) failed: {source_init.stderr.strip()[:300]}")
    _run(["ask", "--run", str(source_run), "--n", "3"])
    source_pending = []
    if (source_run / "pending.json").exists():
        source_pending = json.loads((source_run / "pending.json").read_text(encoding="utf-8", errors="replace"))
        if not isinstance(source_pending, list):
            source_pending = source_pending.get("pending", [])
    source_results = workspace / "results_source.json"
    source_results.write_text(
        json.dumps(
            [
                {"id": item.get("id"), "y": 30.0 + index * 7, "fidelity": 1.0}
                for index, item in enumerate(source_pending[:3])
                if item.get("id")
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8", errors="replace",
    )
    _run(["tell", "--run", str(source_run), "--results", str(source_results)])

    target_run = workspace / "transfer_target"
    _run([
        "init", "--run", str(target_run),
        "--space", str(workspace / "space.json"),
        "--objective", str(workspace / "objective.json"),
        "--context", str(workspace / "context.json"),
        "--budget", "12", "--batch", "3", "--seed", "0",
    ])
    # the audit needs >=3 comparable observations on the target side
    for _ in range(2):
        _run(["ask", "--run", str(target_run), "--n", "3"])
        target_pending = []
        if (target_run / "pending.json").exists():
            target_pending = json.loads(
                (target_run / "pending.json").read_text(encoding="utf-8", errors="replace")
            )
            if not isinstance(target_pending, list):
                target_pending = target_pending.get("pending", [])
        target_results = workspace / "results_target.json"
        target_results.write_text(
            json.dumps(
                [
                    {"id": item.get("id"), "y": 20.0 + index, "fidelity": 1.0}
                    for index, item in enumerate(target_pending[:3])
                    if item.get("id")
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8", errors="replace",
        )
        _run(["tell", "--run", str(target_run), "--results", str(target_results)])

    audit = _run(["transfer-audit", "--run", str(target_run), "--from-run", str(source_run), "--json"])
    if audit.returncode != 0:
        problems.append(f"transfer-audit failed: {audit.stderr.strip()[:300]}")
    else:
        try:
            audit_payload = json.loads(audit.stdout)
        except json.JSONDecodeError:
            audit_payload = {}
            problems.append("transfer-audit --json did not return JSON")
        if audit_payload and "suggested_reliability" not in json.dumps(audit_payload):
            problems.append("transfer-audit lost suggested_reliability")

    impact = _run(["transfer-impact", "--run", str(target_run), "--from-run", str(source_run), "--json"])
    if impact.returncode != 0:
        problems.append(f"transfer-impact failed: {impact.stderr.strip()[:300]}")

    imported = _run([
        "import", "--run", str(target_run), "--from-run", str(source_run),
        "--top-k", "2", "--prior-reliability", "0.8", "--json",
    ])
    if imported.returncode != 0:
        problems.append(f"import --from-run failed: {imported.stderr.strip()[:300]}")
    else:
        try:
            import_payload = json.loads(imported.stdout)
        except json.JSONDecodeError:
            import_payload = {}
            problems.append("import --json did not return JSON")
        if import_payload and not import_payload.get("imported"):
            problems.append("import reported no transferred observations")

    return {
        "problems": problems,
        "moo_suggestions": len(moo_pending),
        "mf_suggestions": len(mf_pending),
        "imported": import_payload.get("imported") if "import_payload" in dir() else 0,
        "action": action.get("action"),
        "suggestions": len(pending),
        "spec_suggestions": len(spec_pending),
        "ok": not problems,
    }


def main() -> int:
    workspace = Path(tempfile.mkdtemp(prefix="skill_workflow_"))
    try:
        report = run_workflow(workspace)
        for problem in report["problems"]:
            print("PROBLEM", problem)
        print(
            f"skill_workflow: action={report['action']} "
            f"suggestions={report['suggestions']} spec_suggestions={report['spec_suggestions']} "
            f"moo_suggestions={report['moo_suggestions']} mf_suggestions={report['mf_suggestions']} "
            f"imported={report['imported']} "
            f"problems={len(report['problems'])}"
        )
        return 0 if report["ok"] else 1
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())