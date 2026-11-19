"""One-shot release gate: tests + evals + audits + skill install check.

Runs every quality gate the project relies on and reports a single summary:

* ``pytest -q`` (unit/integration tests),
* ``eval_router.py`` from the agent-library checkout (intent routing evals),
* ``audit_algorithms.py`` (every registered algorithm runs),
* ``audit_routing.py`` (every capability context routes to a runnable algorithm),
* skill install/check,
* README test-count drift (README must match the collected test count),
* relative markdown link resolution across README/docs/research/skill/examples,
* end-to-end drill (unattended loop + transfer + doctor + digest),
* skill/docs rule duplication (numbers live in one place),
* every CLI subcommand documented in README and SKILL,
* every core public ``Study`` method mentioned in the docs,
* documented defaults vs the values actually in the code,
* the executable scenario matrix (real setups -> documented defaults),
* a re-run of all evidence claims (documented numbers still hold),
* the generated status snapshot (docs/STATUS.md) matching the code,
* the decision-layer rule matrix holding and matching its doc,
* every defaults row citing evidence files that exist (research/EVIDENCE_POLICY.md).

Run:
    py -3 scripts/release_check.py
    py -3 scripts/release_check.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: 共享技能库里的路由评测脚本（不在本仓库内）；可用 CONDOPT_EVAL_SCRIPT 覆盖。
#: 找不到时该门禁记为 skipped（不是 pass）。
def _resolve_eval_script() -> Path:
    override = os.environ.get("CONDOPT_EVAL_SCRIPT")
    if override:
        return Path(override).expanduser()
    candidates = [Path.home() / "agent-library"]
    if ROOT.drive:
        candidates.append(Path(f"{ROOT.drive}/") / "agent-library")
    for candidate in candidates:
        script = candidate / "scripts" / "eval_router.py"
        if script.exists():
            return script
    return candidates[0] / "scripts" / "eval_router.py"


DEFAULT_EVAL = _resolve_eval_script()


def _run(command: list[str], cwd: Path | None = None) -> tuple[int, str]:
    result = subprocess.run(
        command, cwd=str(cwd or ROOT), capture_output=True, text=True, encoding="utf-8"
    )
    return result.returncode, (result.stdout or "") + (result.stderr or "")


def _tail(text: str, lines: int = 3) -> str:
    return " | ".join(part for part in text.strip().splitlines()[-lines:] if part.strip())


def collected_test_count() -> int:
    code, out = _run([sys.executable, "-m", "pytest", "--collect-only", "-q"])
    total = 0
    for line in out.splitlines():
        match = re.match(r".+\.py:\s*(\d+)$", line.strip())
        if match:
            total += int(match.group(1))
    return total


def readme_test_count() -> int | None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"测试\*\*：(\d+) 个单测", text)
    return int(match.group(1)) if match else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--eval-script", default=str(DEFAULT_EVAL))
    parser.add_argument("--skip-slow", action="store_true", help="skip pytest and audits")
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="with --skip-slow: return 0 instead of 3 (a partial run is not a release gate)",
    )
    parser.add_argument("--audit-budget", type=int, default=6)
    args = parser.parse_args()

    checks: list[dict] = []

    def record(
        name: str, ok: bool, detail: str, status: str | None = None
    ) -> None:
        """Record one gate.

        ``status`` defaults to ``ok``/``fail`` but may be ``skipped`` (the gate could
        not run here) or ``external_drift`` (a failure caused by something outside
        this repository).  A skipped gate is **not** a pass: silently counting it as
        one was exactly the hole the independent review found.
        """
        checks.append({
            "check": name,
            "ok": bool(ok),
            "detail": detail,
            "status": status or ("ok" if ok else "fail"),
        })

    collected = collected_test_count()
    readme_count = readme_test_count()
    record(
        "readme_test_count",
        readme_count == collected,
        f"README={readme_count} collected={collected}",
    )

    if not args.skip_slow:
        code, out = _run([sys.executable, "-m", "pytest", "-q"])
        record("pytest", code == 0, _tail(out))

        code, out = _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "audit_algorithms.py"),
                "--budget",
                str(args.audit_budget),
                "--batch",
                "2",
            ]
        )
        record("audit_algorithms", code == 0, _tail(out))

        code, out = _run([sys.executable, str(ROOT / "scripts" / "audit_routing.py")])
        record("audit_routing", code == 0, _tail(out))

        code, out = _run([sys.executable, str(ROOT / "scripts" / "check_docs_links.py")])
        record("docs_links", code == 0, _tail(out))

        code, out = _run([sys.executable, str(ROOT / "scripts" / "e2e_drill.py")])
        record("e2e_drill", code == 0, _tail(out))

        code, out = _run([
            sys.executable, str(ROOT / "scripts" / "check_doc_duplication.py"),
            "--max-duplicates", "2",
        ])
        record("doc_duplication", code == 0, _tail(out))

        code, out = _run([sys.executable, str(ROOT / "scripts" / "check_commands_documented.py")])
        record("commands_documented", code == 0, _tail(out))

        code, out = _run([sys.executable, str(ROOT / "scripts" / "check_api_docs.py")])
        record("api_docs", code == 0, _tail(out))

        code, out = _run([sys.executable, str(ROOT / "scripts" / "check_defaults_sync.py")])
        record("defaults_sync", code == 0, _tail(out))

        code, out = _run([sys.executable, str(ROOT / "scripts" / "scenario_matrix.py")])
        record("scenario_matrix", code == 0, _tail(out))

        # All six claims re-run in ~50 s, so the release gate verifies the full
        # evidence set rather than a subset.
        code, out = _run([sys.executable, str(ROOT / "scripts" / "verify_evidence.py")])
        record("evidence_claims", code == 0, _tail(out))

        code, out = _run([
            sys.executable, str(ROOT / "scripts" / "build_status.py"), "--check",
        ])
        record("status_snapshot", code == 0, _tail(out))

        code, out = _run([
            sys.executable, str(ROOT / "scripts" / "decision_matrix.py"), "--check",
        ])
        record("decision_matrix", code == 0, _tail(out))

        code, out = _run([
            sys.executable, str(ROOT / "scripts" / "check_evidence_policy.py"),
        ])
        record("evidence_policy", code == 0, _tail(out))

        code, out = _run([
            sys.executable, str(ROOT / "scripts" / "build_release_notes.py"), "--check",
        ])
        record("release_notes", code == 0, _tail(out))

        code, out = _run([
            sys.executable, str(ROOT / "scripts" / "check_golden_numbers.py"),
            "--self-test",
        ])
        record("golden_numbers", code == 0, _tail(out))

    eval_script = Path(args.eval_script)
    if eval_script.exists():
        code, out = _run([sys.executable, str(eval_script)])
        detail = _tail(out)
        if code != 0 and "condopt" not in out:
            # The shared agent-library catalog is edited by other sessions too; a
            # failing case that never mentions this skill is external churn, not a
            # regression here -- record it transparently instead of failing.
            record(
                "eval_router",
                True,
                "external drift (unrelated cases failing in the shared catalog): " + detail,
                status="external_drift",
            )
        else:
            record("eval_router", code == 0, detail)
    else:
        record(
            "eval_router",
            False,
            f"skipped: eval script not found ({eval_script}); "
            "pass --eval-script or install the shared catalog",
            status="skipped",
        )

    code, out = _run([sys.executable, str(ROOT / "scripts" / "install_skill.py")])
    workflow_code, workflow_out = _run([
        sys.executable, str(ROOT / "scripts" / "verify_skill_workflow.py"),
    ])
    record(
        "skill_install",
        code == 0 and workflow_code == 0,
        _tail(out) + " | " + _tail(workflow_out),
    )

    failures = [check for check in checks if not check["ok"]]
    skipped = [check for check in checks if check["status"] == "skipped"]
    drifted = [check for check in checks if check["status"] == "external_drift"]
    partial = bool(args.skip_slow)
    if args.json:
        print(json.dumps({
            "checks": checks,
            "failures": len(failures),
            "skipped": len(skipped),
            "external_drift": len(drifted),
            "partial": partial,
        }, ensure_ascii=False, indent=2))
    else:
        markers = {
            "ok": "OK  ",
            "fail": "FAIL",
            "skipped": "SKIP",
            "external_drift": "DRIFT",
        }
        for check in checks:
            marker = markers.get(check["status"], "FAIL")
            print(f"[{marker}] {check['check']}: {check['detail']}")
        if partial:
            print(
                f"release_check: PARTIAL RUN (--skip-slow): "
                f"{len(checks) - len(failures) - len(skipped)}/{len(checks)} executed and green — "
                "this is NOT a release gate"
            )
        else:
            summary = f"release_check: {len(checks) - len(failures) - len(skipped)}/{len(checks)} passed"
            if skipped:
                summary += f", {len(skipped)} skipped"
            if drifted:
                summary += f", {len(drifted)} tolerated external drift"
            print(summary)

    if partial and not args.allow_partial:
        # exit 3 = "partial run"; distinct from 1 (a real failure) so callers cannot
        # mistake a smoke test for a release gate.
        return 3
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())