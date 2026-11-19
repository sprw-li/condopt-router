"""End-to-end integration: spec + plates + next_action loop + confirmation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from condopt import Study

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "full_campaign_demo.py"
spec = importlib.util.spec_from_file_location("full_campaign_demo", SCRIPT)
demo = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(demo)


def test_full_campaign_reaches_confirmed_stop(tmp_path):
    payload = demo.run_campaign(tmp_path / "campaign", budget=10, batch_size=2, seed=0, render=False)
    actions = [item["action"] for item in payload["actions"]]
    assert "confirm" in actions
    assert actions[-1] == "stop"
    recommendation = payload["recommendation"]
    assert recommendation["confirmation"] is not None
    assert recommendation["confirmation"]["n"] >= 1
    assert recommendation["should_stop"] is True
    assert payload["summary_spec"]["n"] > 0
    assert payload["spec_calibration"]["metrics"]
    full_report = Study.load(Path(payload["run_dir"])).report()
    assert "推荐条件" in full_report
    assert payload["run_dir"].endswith("campaign")
