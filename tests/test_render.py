"""Tests for the static dashboard renderer (P0+P1+P2).

The test studies are built through the public ``Study`` API (no fixture files),
then the renderer must produce a self-contained offline HTML (+ PNG) without
touching the run directory.  P2 covers the multi-objective Pareto block, the
parameter-importance block and the read-only ``scripts/serve.py`` server.
"""

import html as html_module
import http.client
import json
import re
import threading
from pathlib import Path

import pytest

from condopt import Objective, SearchSpace, Study
from condopt.benchmarks import builtin_problems
from condopt.render import CONVERGENCE_PNG, DASHBOARD_FILE, render_study


def _space() -> SearchSpace:
    return SearchSpace.from_dict(
        [
            {"name": "T", "kind": "continuous", "low": 40.0, "high": 120.0, "unit": "degC"},
            {"name": "time", "kind": "continuous", "low": 5.0, "high": 120.0, "unit": "min"},
            {"name": "solvent", "kind": "categorical", "choices": ["MeCN", "EtOH", "THF"]},
            {"name": "equiv", "kind": "integer", "low": 1, "high": 4},
        ]
    )


def make_study(tmp_path: Path, budget: int = 10, batch_size: int = 2) -> Study:
    """Create a small study: 2 told rounds + 1 pending round + one override."""
    study = Study.create(
        tmp_path / "run",
        _space(),
        Objective(name="yield", direction="max"),
        algorithm="sobol",
        budget=budget,
        batch_size=batch_size,
        seed=0,
        auto_route=False,
    )
    for round_index in range(2):
        batch = study.ask()
        study.tell(
            [
                {"id": item["id"], "y": 60.0 + 3.0 * round_index + index, "source": "simulation"}
                for index, item in enumerate(batch)
            ]
        )
    study.ask()  # third batch stays pending -> suggestion table has rows
    study.override("sobol", {"scramble": True}, reason="人工复核：保持空间填充，验证 override 审计展示")
    return study


def test_render_generates_offline_dashboard(tmp_path):
    study = make_study(tmp_path)
    result = render_study(study.dir, make_png=True)

    dashboard = Path(result["dashboard"])
    assert dashboard.exists()
    raw = dashboard.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "dashboard must be UTF-8 without BOM"
    document = raw.decode("utf-8")

    # required sections and study data
    assert "<!doctype html>" in document
    assert "收敛曲线" in document
    assert "空间覆盖" in document
    assert "建议批次" in document
    assert "yield" in document
    assert "degC" in document
    assert "r3-1" in document, "pending suggestion id must be listed"
    assert "人工复核" in document, "override reason must be shown"
    assert "—" in document, "missing predictions must render as an em dash, not fake values"

    # offline guarantee: no external resources at all
    assert not re.search(r"https?://", document, re.IGNORECASE)
    assert "<script src=" not in document
    assert "<link " not in document

    png = Path(result["convergence_png"])
    assert png.exists()
    assert png.stat().st_size > 1000
    assert result["n_observations"] == 4
    assert result["n_rounds"] == 2
    assert result["n_pending"] == 2
    assert result["warnings"] == []


def test_render_no_png_and_custom_out_dir(tmp_path):
    study = make_study(tmp_path)
    out_dir = tmp_path / "deliverables"
    result = render_study(study.dir, out_dir=out_dir, make_png=False)

    assert Path(result["dashboard"]).exists()
    assert result["convergence_png"] is None
    assert not (out_dir / CONVERGENCE_PNG).exists()
    assert (out_dir / DASHBOARD_FILE).read_text(encoding="utf-8").startswith("<!doctype html>")
    # the run directory must stay untouched (renderer is read-only)
    assert not (study.dir / DASHBOARD_FILE).exists()


def test_render_empty_study_shows_placeholder(tmp_path):
    study = Study.create(
        tmp_path / "empty",
        _space(),
        Objective(name="yield"),
        algorithm="sobol",
        budget=6,
        batch_size=2,
        seed=0,
        auto_route=False,
    )
    result = render_study(study.dir, make_png=False)
    document = Path(result["dashboard"]).read_text(encoding="utf-8")
    assert "暂无观测数据" in document
    assert result["n_observations"] == 0


def test_render_missing_run_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        render_study(tmp_path / "does-not-exist", make_png=False)


# ---------------------------------------------------------------------------
# P2: multi-objective Pareto block
# ---------------------------------------------------------------------------


def make_moo_study(tmp_path: Path, rounds: int = 2) -> Study:
    problem = builtin_problems()["moo_yield_cost"]
    study = Study.create(
        tmp_path / "moo",
        problem.space,
        problem.objective,
        budget=8,
        batch_size=2,
        seed=0,
    )
    for _ in range(rounds):
        batch = study.ask(2)
        study.tell(
            [
                {
                    "id": item["id"],
                    "objectives": problem.objectives_fn(item["x"]),
                    "source": "simulation",
                }
                for item in batch
            ]
        )
    return study


def test_render_multiobjective_pareto_block(tmp_path):
    study = make_moo_study(tmp_path)
    result = render_study(study.dir, make_png=False)
    document = Path(result["dashboard"]).read_text(encoding="utf-8")
    text = html_module.unescape(document)

    assert "Pareto 前沿" in text
    assert "非支配" in text
    assert "参考点" in text and "HV=" in text
    assert "yield" in text and "cost" in text
    # 4 observations < 20 -> importance must say so instead of inventing numbers
    assert "样本不足（<20）" in text
    assert not re.search(r"https?://", document, re.IGNORECASE)

    assert result["n_objectives"] == 2
    assert result["n_pareto_front"] >= 1
    assert result["hypervolume"] is not None and result["hypervolume"] > 0
    # renderer must match Study's own Pareto/HV definition (incl. reference point)
    summary = study.pareto_summary()
    assert summary["n_front"] == result["n_pareto_front"]
    assert summary["hypervolume"] == pytest.approx(result["hypervolume"], rel=1e-12)


# ---------------------------------------------------------------------------
# P2: parameter importance
# ---------------------------------------------------------------------------


def test_render_importance_with_enough_samples(tmp_path):
    space = SearchSpace.from_dict(
        [
            {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0},
            {"name": "z", "kind": "continuous", "low": 0.0, "high": 1.0},
        ]
    )
    study = Study.create(
        tmp_path / "importance",
        space,
        Objective(name="score", direction="max"),
        algorithm="sobol",
        budget=25,
        batch_size=5,
        seed=0,
        auto_route=False,
    )
    for round_index in range(4):
        batch = study.ask()
        study.tell(
            [
                {
                    "id": item["id"],
                    "y": 2.0 * item["x"]["x"] + 0.1 * round_index,
                    "source": "simulation",
                }
                for item in batch
            ]
        )
    result = render_study(study.dir, make_png=False)
    text = html_module.unescape(Path(result["dashboard"]).read_text(encoding="utf-8"))

    assert result["importance_n"] == 20
    assert "参数重要性" in text
    assert "样本不足" not in text
    assert "|ρ|" in text


def test_render_importance_placeholder_below_20(tmp_path):
    study = Study.create(
        tmp_path / "small",
        _space(),
        Objective(name="yield"),
        algorithm="sobol",
        budget=6,
        batch_size=2,
        seed=0,
        auto_route=False,
    )
    batch = study.ask(2)
    study.tell(
        [
            {"id": item["id"], "y": 1.0 + index, "source": "simulation"}
            for index, item in enumerate(batch)
        ]
    )
    result = render_study(study.dir, make_png=False)
    text = html_module.unescape(Path(result["dashboard"]).read_text(encoding="utf-8"))

    assert result["importance_n"] == 2
    assert "样本不足（<20）" in text
    assert "参数重要性" in text


# ---------------------------------------------------------------------------
# P2: read-only local server (scripts/serve.py)
# ---------------------------------------------------------------------------


def load_serve_module():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "serve.py"
    spec = importlib.util.spec_from_file_location("_condopt_serve_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _request(port: int, path: str, method: str = "GET", body: bytes | None = None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        headers = {"Content-Type": "application/json"} if body is not None else {}
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def test_serve_is_readonly_get_only(tmp_path):
    study = make_study(tmp_path)
    serve = load_serve_module()
    server = serve.create_server(study.dir, port=0, out_dir=tmp_path / "serve_out")
    host, port = server.server_address[:2]
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    try:
        assert host == "127.0.0.1", "server must bind loopback only"

        status, body = _request(port, "/")
        assert status == 200
        document = body.decode("utf-8")
        assert "condopt 看板" in document
        assert 'http-equiv="refresh"' in document
        assert not (study.dir / DASHBOARD_FILE).exists(), "server must not write the run dir"

        status, body = _request(port, "/api/state")
        assert status == 200
        payload = json.loads(body.decode("utf-8"))
        assert payload["run"] == study.dir.name
        assert payload["used"] == 4
        assert payload["pending"] == 2
        assert payload["refresh_seconds"] == 10

        # write methods are refused, not silently ignored
        assert _request(port, "/", method="POST", body=b'{"evil": true}')[0] == 405
        assert _request(port, "/", method="PUT", body=b"x")[0] == 405
        assert _request(port, "/", method="DELETE")[0] == 405
        assert _request(port, "/", method="HEAD")[0] == 405

        # traversal and unknown paths are rejected
        assert _request(port, "/../problem.json")[0] in (400, 404)
        assert _request(port, "/%2e%2e/problem.json")[0] in (400, 404)
        assert _request(port, "/does-not-exist")[0] == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

def test_render_shows_decision_card(tmp_path, capsys):
    from condopt.cli import main

    study = make_study(tmp_path)
    assert main(["decision", "--run", str(study.dir), "--json"]) == 0
    capsys.readouterr()
    assert (study.dir / "decision.json").exists()
    result = render_study(study.dir, out_dir=tmp_path / "dashboard", make_png=False)
    document = Path(result["dashboard"]).read_text(encoding="utf-8")
    assert "推荐判定" in document

def test_render_shows_calibration_card(tmp_path, capsys):
    from condopt.cli import main
    from condopt.spec import Goal, Spec

    space = SearchSpace.from_dict(
        [{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]
    )
    spec = Spec(goals=(Goal("y", mode="target", target=80.0, tolerance=1.0, scale=5.0),))
    study = Study.create(
        tmp_path / "calibration_run",
        space,
        spec=spec,
        algorithm="lhs",
        budget=8,
        batch_size=2,
        seed=0,
    )
    study.import_observations([
        {"x": {"x": 0.1}, "objectives": {"y": 79.0}},
        {"x": {"x": 0.5}, "objectives": {"y": 80.0}},
        {"x": {"x": 0.9}, "objectives": {"y": 81.0}},
    ])
    assert main(["calibrate-spec", "--run", str(study.dir), "--json"]) == 0
    capsys.readouterr()
    assert (study.dir / "spec_calibration.json").exists()
    result = render_study(study.dir, out_dir=tmp_path / "calibration_dashboard", make_png=False)
    document = Path(result["dashboard"]).read_text(encoding="utf-8")
    assert "规格校准" in document

def test_render_shows_step_card(tmp_path, capsys):
    from condopt.cli import main

    study = make_study(tmp_path)
    assert main(["step", "--run", str(study.dir), "--json"]) == 0
    capsys.readouterr()
    assert (study.dir / "step.json").exists()
    result = render_study(study.dir, out_dir=tmp_path / "step_dashboard", make_png=False)
    document = Path(result["dashboard"]).read_text(encoding="utf-8")
    assert "下一步行动" in document

def test_digest_writes_markdown(tmp_path, capsys):
    from condopt.cli import main

    study = make_study(tmp_path)
    assert main(["digest", "--run", str(study.dir), "--json"]) == 0
    capsys.readouterr()
    digest = study.dir / "digest.md"
    assert digest.exists()
    document = digest.read_text(encoding="utf-8")
    assert "# condopt 摘要" in document
    assert "推荐判定" in document
    assert "下一步行动" in document

def test_digest_html_export(tmp_path, capsys):
    from condopt.cli import main

    study = make_study(tmp_path)
    assert main(["digest", "--run", str(study.dir), "--format", "html", "--json"]) == 0
    capsys.readouterr()
    document = (study.dir / "digest.html").read_text(encoding="utf-8")
    assert "<!doctype html>" in document
    assert "推荐判定" in document
    assert "下一步行动" in document

