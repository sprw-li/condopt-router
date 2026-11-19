"""Tests for plate/batch bookkeeping and the batch-effect warning (priority b)."""

import csv
import json

import pytest

from condopt import Objective, SearchSpace, Study
from condopt.cli import main


def simple_space() -> SearchSpace:
    return SearchSpace.from_dict(
        [
            {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0},
            {"name": "y", "kind": "continuous", "low": 0.0, "high": 1.0},
        ]
    )


def make_study(tmp_path, budget=20, batch_size=3, **kwargs) -> Study:
    return Study.create(
        tmp_path / "run",
        simple_space(),
        Objective(name="yield", direction="max"),
        budget=budget,
        batch_size=batch_size,
        seed=0,
        **kwargs,
    )


def import_plate_group(study: Study, plate: str, values: list[float], start: int = 0) -> None:
    """Import one observation per value, tagged with ``plate``."""
    results = [
        {
            "x": {"x": round(0.01 + 0.01 * (start + index), 4), "y": 0.5},
            "y": value,
            "plate": plate,
            "well": f"A{index + 1}",
        }
        for index, value in enumerate(values)
    ]
    study.import_observations(results)


def test_ask_stamps_plate_well_and_batch(tmp_path):
    study = make_study(tmp_path)
    pending = study.ask(3)
    assert [item["plate"] for item in pending] == ["P1", "P1", "P1"]
    assert [item["well"] for item in pending] == ["A1", "A2", "A3"]
    assert [item["batch"] for item in pending] == [1, 1, 1]
    # persisted so a different machine/robot can execute the plate
    on_disk = json.loads((tmp_path / "run" / "pending.json").read_text(encoding="utf-8"))
    assert [(item["plate"], item["well"]) for item in on_disk] == [
        ("P1", "A1"),
        ("P1", "A2"),
        ("P1", "A3"),
    ]
    with (tmp_path / "run" / "suggestions.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [row["plate"] for row in rows] == ["P1", "P1", "P1"]
    assert [row["well"] for row in rows] == ["A1", "A2", "A3"]


def test_each_ask_opens_a_new_plate(tmp_path):
    study = make_study(tmp_path)
    first = study.ask(2)
    study.tell([{"id": item["id"], "y": 1.0} for item in first])
    second = study.ask(2)
    assert {item["plate"] for item in second} == {"P2"}
    assert [item["well"] for item in second] == ["A1", "A2"]
    assert [item["batch"] for item in second] == [2, 2]


def test_explicit_plate_continues_filling_used_plate(tmp_path):
    study = make_study(tmp_path, meta={"plate_columns": 2})
    first = study.ask(2)
    assert [item["well"] for item in first] == ["A1", "A2"]
    study.tell([{"id": item["id"], "y": 1.0} for item in first])
    # same physical plate, next row; used wells must not be re-issued
    second = study.ask(2, plate="P1")
    assert [item["well"] for item in second] == ["B1", "B2"]


def test_tell_propagates_plate_and_allows_explicit_override(tmp_path):
    study = make_study(tmp_path)
    pending = study.ask(2)
    observations = study.tell(
        [
            {"id": pending[0]["id"], "y": 1.0},
            {
                "id": pending[1]["id"],
                "y": 2.0,
                "plate": "P9",
                "well": "C7",
                "batch": 99,
            },
        ]
    )
    assert observations[0].meta["plate"] == "P1"
    assert observations[0].meta["plate_well"] == "A1"
    assert observations[0].meta["batch"] == 1
    assert observations[1].meta["plate"] == "P9"
    assert observations[1].meta["plate_well"] == "C7"
    assert observations[1].meta["batch"] == 99
    # the override survives a reload (it lives in dataset.jsonl meta)
    reloaded = Study.load(tmp_path / "run")
    assert reloaded.dataset[1].meta["plate"] == "P9"


def test_plate_summary_reports_stats_without_warning(tmp_path):
    study = make_study(tmp_path)
    import_plate_group(study, "P1", [50.0, 51.0, 49.0])
    import_plate_group(study, "P2", [50.5, 49.5, 50.0], start=3)
    summary = study.plate_summary()
    assert summary["n_plates"] == 2
    assert [entry["plate"] for entry in summary["plates"]] == ["P1", "P2"]
    first = summary["plates"][0]
    assert first["n"] == 3
    assert first["mean_value"] == pytest.approx(50.0)
    assert first["range_value"] == pytest.approx(2.0)
    effect = summary["batch_effect"]
    assert effect["detected"] is False
    assert effect["plates_compared"] == 2
    assert effect["p_value"] > 0.05
    assert "未检测到显著批次" in effect["message"]


def test_batch_effect_is_detected_on_plate_shift(tmp_path):
    study = make_study(tmp_path)
    import_plate_group(study, "P1", [40.0, 41.0, 40.5])
    import_plate_group(study, "P2", [60.0, 61.0, 60.5], start=3)
    effect = study.plate_summary()["batch_effect"]
    assert effect["detected"] is True
    assert effect["p_value"] < 0.05
    assert effect["spread"] == pytest.approx(20.0)
    assert effect["relative_spread"] == pytest.approx(20.0 / 50.5)
    assert "检测到批次/孔板效应" in effect["message"]
    assert "plate" in effect["message"]


def test_tiny_relative_shift_is_not_flagged(tmp_path):
    # Statistically separable but practically irrelevant (<2% of the mean).
    study = make_study(tmp_path)
    import_plate_group(study, "P1", [100.0, 100.0])
    import_plate_group(study, "P2", [100.5, 100.5], start=2)
    effect = study.plate_summary()["batch_effect"]
    assert effect["detected"] is False
    assert "未检测到显著批次" in effect["message"]


def test_batch_effect_needs_two_replicates_per_plate(tmp_path):
    study = make_study(tmp_path)
    import_plate_group(study, "P1", [10.0])
    import_plate_group(study, "P2", [90.0], start=1)
    effect = study.plate_summary()["batch_effect"]
    assert effect["detected"] is False
    assert effect["plates_compared"] == 0
    assert "样本不足" in effect["message"]


def test_unassigned_observations_are_counted(tmp_path):
    study = make_study(tmp_path)
    import_plate_group(study, "P1", [10.0, 11.0])
    study.import_observations([{"x": {"x": 0.9, "y": 0.1}, "y": 12.0}])
    summary = study.plate_summary()
    assert summary["unassigned"] == 1
    assert summary["n_plates"] == 1


def test_summary_and_report_include_plate_section(tmp_path):
    study = make_study(tmp_path)
    import_plate_group(study, "P1", [40.0, 41.0])
    import_plate_group(study, "P2", [60.0, 61.0], start=2)
    summary = study.summary()
    assert summary["plates"]["n_plates"] == 2
    assert summary["plates"]["batch_effect"]["detected"] is True
    text = study.report()
    assert "孔板/批次" in text
    assert "P1" in text and "P2" in text
    assert "批次效应" in text
    assert "检测到批次/孔板效应" in text


def test_confirm_batch_is_plate_stamped(tmp_path):
    study = make_study(tmp_path, budget=10)
    pending = study.ask(2)
    study.tell([{"id": item["id"], "y": 1.0} for item in pending])
    confirmation = study.confirm(2)
    assert {item["plate"] for item in confirmation} == {"P2"}
    assert [item["well"] for item in confirmation] == ["A1", "A2"]
    assert all(item["purpose"] == "confirmation" for item in confirmation)
    observations = study.tell([{"id": item["id"], "y": 2.0} for item in confirmation])
    assert all(obs.meta["purpose"] == "confirmation" for obs in observations)
    assert all(obs.meta["plate"] == "P2" for obs in observations)


def test_plate_capacity_is_enforced(tmp_path):
    study = make_study(tmp_path, meta={"plate_rows": 1, "plate_columns": 2})
    pending = study.ask(2)
    assert [item["well"] for item in pending] == ["A1", "A2"]
    study.tell([{"id": item["id"], "y": 1.0} for item in pending])
    with pytest.raises(ValueError, match="剩余孔位不足"):
        study.ask(1, plate="P1")
    # a new plate still works
    assert [item["well"] for item in study.ask(2)] == ["A1", "A2"]


def test_cli_ask_accepts_explicit_plate(tmp_path):
    study = make_study(tmp_path)
    assert main(["ask", "--run", str(study.dir), "--n", "2", "--plate", "P5", "--json"]) == 0
    on_disk = json.loads((tmp_path / "run" / "pending.json").read_text(encoding="utf-8"))
    assert {item["plate"] for item in on_disk} == {"P5"}
    assert [item["well"] for item in on_disk] == ["A1", "A2"]

def test_recommend_uses_plate_adjusted_score(tmp_path):
    from condopt import Objective, SearchSpace, Study

    space = SearchSpace.from_dict(
        [{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]
    )
    study = Study.create(
        tmp_path / "plate_adjust",
        space,
        Objective(name="y", direction="max"),
        algorithm="lhs",
        budget=12,
        batch_size=3,
        seed=0,
        meta={"plate_effect_min_relative": 0.02},
    )
    first = study.ask(3, plate="P1")
    study.tell([{"id": item["id"], "y": value} for item, value in zip(first, [10.0, 12.0, 14.0])])
    second = study.ask(3, plate="P2")
    study.tell([{"id": item["id"], "y": value} for item, value in zip(second, [20.0, 19.0, 21.0])])
    effects = study.plate_effects()
    assert effects["detected"] is True
    assert set(effects["effects"]) == {"P1", "P2"}
    recommend = study.recommend()
    assert recommend["selection_basis"] == "plate_adjusted"
    assert recommend["raw_best"]["score"] == 21.0
    assert recommend["score"] == 14.0
    assert recommend["condition"] in [item["x"] for item in first]
    confirm = study.confirm(2)
    assert all(item["x"] == recommend["condition"] for item in confirm)
    assert confirm[0]["x"] != recommend["raw_best"]["condition"]
    action = study.next_action()
    assert action["diagnostics"]["plate_effect"]["detected"] is True
    assert "use_plate_adjusted_recommendation" in action["suggested_actions"]


def test_plate_adjusted_spec_compliance_uses_metric_effects(tmp_path):
    from condopt import SearchSpace, Study
    from condopt.spec import Goal, Spec

    space = SearchSpace.from_dict(
        [{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]
    )
    spec = Spec(goals=(Goal("E", mode="target", target=80.0, tolerance=1.0, scale=5.0),))
    study = Study.create(
        tmp_path / "plate_spec",
        space,
        spec=spec,
        algorithm="lhs",
        budget=12,
        batch_size=3,
        seed=0,
    )
    first = study.ask(3, plate="P1")
    study.tell([
        {"id": item["id"], "objectives": {"E": value}}
        for item, value in zip(first, [78.0, 80.0, 82.0])
    ])
    second = study.ask(3, plate="P2")
    study.tell([
        {"id": item["id"], "objectives": {"E": value}}
        for item, value in zip(second, [90.0, 89.0, 91.0])
    ])
    assert study.plate_effects("E")["detected"] is True
    recommend = study.recommend()
    assert recommend["selection_basis"] == "plate_adjusted"
    assert "板效应" in recommend["selection_reason"]
    # the selected E is chosen from plate-corrected values, not raw values
    adjusted_value = recommend["spec"]["goals"]["E"]["value"]
    raw_value = recommend["spec_raw"]["goals"]["E"]["value"]
    assert adjusted_value != raw_value
    assert recommend["condition"] in [item["x"] for item in first]
    action = study.next_action()
    check = action["diagnostics"]["spec_check"]
    assert check["raw_in_spec"] is False
    assert check["adjusted_in_spec"] is False
    assert "calibrate_by_plate" in action["suggested_actions"]
    assert action["diagnostics"]["plate_groups_ready"] >= 2
    summary_spec = study.summary()["spec"]["recommended"]
    assert summary_spec["selection_basis"] == "plate_adjusted"
    assert summary_spec["plate_effect_detected"] is True
    assert summary_spec["in_spec"] is False

def test_worst_case_plate_adjusted_selection_uses_metric_effects(tmp_path):
    from condopt import SearchSpace, Study
    from condopt.spec import Goal, Spec

    space = SearchSpace.from_dict(
        [{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]
    )
    spec = Spec(
        goals=(
            Goal("a", mode="target", target=10.0, tolerance=1.0, scale=1.0),
            Goal("b", mode="target", target=10.0, tolerance=1.0, scale=1.0),
        ),
        aggregation="worst_case",
    )
    study = Study.create(
        tmp_path / "plate_worst_case",
        space,
        spec=spec,
        algorithm="lhs",
        budget=20,
        batch_size=4,
        seed=0,
    )
    first = study.ask(3, plate="P1")
    study.tell([{"id": item["id"], "objectives": {"a": 10.0, "b": 2.0}} for item in first])
    second = study.ask(2, plate="P2")
    study.tell([{"id": item["id"], "objectives": {"a": 4.0, "b": 7.0}} for item in second])

    assert study._detected_metric_effects()
    recommendation = study.recommend()
    assert recommendation["selection_basis"] == "plate_adjusted"
    # raw worst_case best is on P2; after per-metric plate correction P1 wins
    assert recommendation["raw_best"]["plate"] == "P2"
    assert recommendation["condition"] in [item["x"] for item in first]
    assert recommendation["spec"]["goals"]["a"]["value"] != recommendation["spec_raw"]["goals"]["a"]["value"]

def test_review_spec_is_skipped_when_plate_adjusted_values_are_in_spec(tmp_path):
    from condopt import SearchSpace, Study
    from condopt.spec import Goal, Spec

    space = SearchSpace.from_dict(
        [{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]
    )
    spec = Spec(goals=(Goal("E", mode="target", target=80.0, tolerance=1.0, scale=5.0),))
    study = Study.create(
        tmp_path / "plate_spec_count",
        space,
        spec=spec,
        algorithm="lhs",
        budget=12,
        batch_size=3,
        seed=0,
    )
    study.import_observations(
        [
            {"x": {"x": index / 10.0}, "objectives": {"E": value},
             "meta": {"plate": plate}}
            for index, (value, plate) in enumerate([
                (74.0, "P1"), (74.2, "P1"), (73.8, "P1"),
                (86.0, "P2"), (86.2, "P2"), (85.8, "P2"),
            ])
        ]
    )
    assert study.plate_effects("E")["detected"] is True
    action = study.next_action()
    assert action["diagnostics"]["spec_in_spec_count"] == 0
    assert action["diagnostics"]["spec_in_spec_adjusted"] > 0
    assert "review_spec" not in action["suggested_actions"]
    assert "use_plate_adjusted_recommendation" in action["suggested_actions"]

