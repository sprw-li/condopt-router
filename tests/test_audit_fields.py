"""A9 审计字段：观测/决策的版本与来源信息。

被审计的不是"跑出了什么"，而是"这条记录属于哪个配置版本、由谁产生"。
所以这里同时钉住两件事：新记录必须带全字段，旧 JSONL（没有这些字段）必须
仍能读出来——否则升级一次 schema 就把历史 run 变成不可读的孤儿。
"""

from __future__ import annotations

import json
import re

import pytest

from condopt import Objective, SearchSpace, Study
from condopt.dataset import SCHEMA_VERSION, Dataset, Observation


def simple_space() -> SearchSpace:
    return SearchSpace.from_dict(
        [
            {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0},
            {"name": "y", "kind": "continuous", "low": 0.0, "high": 1.0},
        ]
    )


def make_study(tmp_path, name: str = "run", budget: int = 10, batch_size: int = 2) -> Study:
    # 显式算法：跑 router 会让 config_hash 随上下文变化，掩盖本任务要钉的字段。
    return Study.create(
        tmp_path / name,
        simple_space(),
        Objective(name="yield", direction="max"),
        algorithm="lhs",
        budget=budget,
        batch_size=batch_size,
        seed=0,
    )


# -- ① 新写观测含 schema_version / actor --------------------------------
def test_new_observation_carries_schema_version_and_actor(monkeypatch):
    monkeypatch.delenv("CONDOPT_ACTOR", raising=False)
    observation = Observation(x={"x": 0.3, "y": 0.4}, y=1.0)
    payload = observation.to_dict()
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["actor"] == "unknown"  # 未设环境变量时的默认值

    explicit = Observation(x={"x": 0.3, "y": 0.4}, y=1.0, actor="robot-7")
    assert explicit.to_dict()["actor"] == "robot-7"
    # 显式传入的值必须原样保留，不能被环境变量或默认值覆盖
    assert explicit.actor == "robot-7"


def test_observation_actor_defaults_to_environment_variable(monkeypatch):
    monkeypatch.setenv("CONDOPT_ACTOR", "agent-42")
    assert Observation(x={"x": 0.1, "y": 0.2}, y=1.0).actor == "agent-42"
    # 空字符串不是身份：退回默认值，而不是把 "" 写进审计记录
    monkeypatch.setenv("CONDOPT_ACTOR", "   ")
    assert Observation(x={"x": 0.1, "y": 0.2}, y=1.0).actor == "unknown"


def test_observation_roundtrip_keeps_audit_fields():
    original = Observation(x={"x": 0.2, "y": 0.8}, y=2.0, actor="robot-1")
    restored = Observation.from_dict(original.to_dict())
    assert restored.schema_version == original.schema_version
    assert restored.actor == "robot-1"
    assert restored.to_dict() == original.to_dict()


def test_new_observation_jsonl_line_contains_audit_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("CONDOPT_ACTOR", "agent-9")
    path = tmp_path / "dataset.jsonl"
    Dataset().append_jsonl(path, [Observation(x={"x": 0.5, "y": 0.5}, y=1.5)])
    raw = json.loads(path.read_text(encoding="utf-8").strip())
    assert raw["schema_version"] == SCHEMA_VERSION
    assert raw["actor"] == "agent-9"


# -- ② 旧格式 JSONL 能读（缺失字段给默认值，而不是报错） -----------------
def test_legacy_observation_jsonl_without_audit_fields_is_readable(tmp_path, monkeypatch):
    """旧行只有 x/y/fidelity，没有任何 A9 字段；load 必须成功并补默认值。"""
    monkeypatch.setenv("CONDOPT_ACTOR", "agent-now")
    path = tmp_path / "dataset.jsonl"
    path.write_text(
        json.dumps({"x": {"x": 0.25, "y": 0.75}, "y": 3.5, "fidelity": 1.0}) + "\n",
        encoding="utf-8",
    )
    dataset = Dataset.from_jsonl(path)
    assert len(dataset) == 1
    observation = dataset[0]
    assert observation.y == 3.5
    assert observation.schema_version == SCHEMA_VERSION
    # 旧行没有记录生产者：即使当前环境设了 CONDOPT_ACTOR，也不能把旧测量算到它头上
    assert observation.actor == "unknown"


def test_legacy_observation_without_measurement_is_still_rejected():
    """边界说明：连 y/objectives 都没有的行从来就不是合法观测（A9 之前也是）。

    这条不是向后兼容的让步，而是把边界写下来：审计字段可以缺省，测量值不行。
    """
    with pytest.raises(ValueError, match="needs either y or objectives"):
        Observation.from_dict({"x": {"x": 0.1, "y": 0.2}, "fidelity": 1.0})


def test_mixed_legacy_and_new_dataset_lines_load_together(tmp_path):
    """同一个文件里新老行混存（增量升级的真实形态）也要能读。"""
    path = tmp_path / "dataset.jsonl"
    path.write_text(
        json.dumps({"x": {"x": 0.1, "y": 0.2}, "y": 1.0, "fidelity": 1.0}) + "\n"
        + json.dumps(
            {
                "x": {"x": 0.8, "y": 0.9},
                "y": 2.0,
                "fidelity": 1.0,
                "schema_version": SCHEMA_VERSION,
                "actor": "agent-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = Dataset.from_jsonl(path)
    assert [observation.actor for observation in dataset] == ["unknown", "agent-1"]
    assert len(dataset) == 2


# -- ③ log_decision 的 seq 单调递增、config_hash 稳定 ---------------------
def test_log_decision_seq_is_monotonic_and_config_hash_is_stable(tmp_path):
    study = make_study(tmp_path)
    first = study.log_decision(action="probe_a", algorithm=study.algorithm, reason="第一次")
    second = study.log_decision(action="probe_b", algorithm=study.algorithm, reason="第二次")

    assert second["seq"] > first["seq"]
    # 探针记在同一次 create 的 manual_selection 之后，seq 必须连续无洞
    seqs = [record["seq"] for record in study.decisions()]
    assert seqs == list(range(1, len(seqs) + 1))
    assert seqs == sorted(seqs)
    # 配置没变 -> 两次决策指向同一个配置版本
    assert first["config_hash"] == second["config_hash"] == study.config_hash()
    assert re.fullmatch(r"[0-9a-f]{12}", first["config_hash"])


def test_config_hash_changes_when_config_changes(tmp_path):
    study = make_study(tmp_path)
    before = study.config_hash()
    study.log_decision(action="probe", algorithm=study.algorithm, reason="改之前")

    study.override("random", {"dedupe": True}, reason="审计测试：换算法")
    after = study.config_hash()
    assert after != before

    records = study.decisions()
    assert records[-1]["action"] == "override"
    assert records[-1]["config_hash"] == after  # 决策记的是改完之后的配置版本
    assert records[-2]["config_hash"] == before


def test_config_hash_is_stable_across_studies_with_the_same_config(tmp_path):
    """hash 是配置的函数，不是进程/目录的函数（跨进程复现的前提）。"""
    first = make_study(tmp_path, name="run_a")
    second = make_study(tmp_path, name="run_b")
    assert first.config_hash() == second.config_hash()
    assert first.dir != second.dir


def test_new_decisions_carry_actor_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("CONDOPT_ACTOR", "agent-7")
    study = make_study(tmp_path)
    record = study.log_decision(action="probe", algorithm=study.algorithm, reason="带身份")
    assert record["actor"] == "agent-7"
    # 显式 actor 优先于环境变量
    explicit = study.log_decision(
        action="probe", algorithm=study.algorithm, reason="显式", actor="human-1"
    )
    assert explicit["actor"] == "human-1"


# -- 旧 decisions.jsonl（6 字段）继续可读 ---------------------------------
def test_decisions_reads_legacy_lines_without_audit_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("CONDOPT_ACTOR", "agent-now")
    study = make_study(tmp_path)
    legacy = {
        "t": 1700000000.0,
        "action": "initial_route",
        "algorithm": "lhs",
        "params": {"scramble": True},
        "reason": "旧格式记录",
        "payload": {},
    }
    (study.dir / "decisions.jsonl").write_text(
        json.dumps(legacy, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    records = study.decisions()
    assert len(records) == 1
    assert records[0]["action"] == "initial_route"
    assert records[0]["seq"] == 1
    # 旧记录从未写版本/身份：必须显式留空，不能拿当前配置或当前环境冒充
    assert records[0]["config_hash"] is None
    assert records[0]["actor"] == "unknown"

    # 旧行不影响后续写入：下一条从 2 开始，用当前身份
    fresh = study.log_decision(action="probe", algorithm=study.algorithm, reason="新记录")
    assert fresh["seq"] == 2
    assert fresh["actor"] == "agent-now"
    assert [record["seq"] for record in study.decisions()] == [1, 2]


# -- ④ audit_trail 字段齐全 ----------------------------------------------
def test_audit_trail_reports_schema_counts_hash_and_actors(tmp_path, monkeypatch):
    monkeypatch.setenv("CONDOPT_ACTOR", "agent-3")
    study = make_study(tmp_path)
    pending = study.ask(2)
    study.tell([{"id": item["id"], "y": float(index)} for index, item in enumerate(pending)])

    trail = study.audit_trail()
    assert set(trail) == {
        "schema_version",
        "n_observations",
        "n_decisions",
        "config_hash",
        "actors",
    }
    assert trail["schema_version"] == SCHEMA_VERSION
    assert trail["n_observations"] == 2
    assert trail["n_decisions"] == len(study.decisions())
    assert trail["config_hash"] == study.config_hash()
    assert re.fullmatch(r"[0-9a-f]{12}", trail["config_hash"])
    assert trail["actors"] == ["agent-3"]


def test_audit_trail_on_an_empty_or_legacy_run_still_returns_every_field(tmp_path):
    study = make_study(tmp_path)
    (study.dir / "decisions.jsonl").write_text("", encoding="utf-8")

    trail = study.audit_trail()
    assert trail["n_observations"] == 0
    assert trail["n_decisions"] == 0
    assert trail["actors"] == []
    assert trail["schema_version"] == SCHEMA_VERSION
    assert re.fullmatch(r"[0-9a-f]{12}", trail["config_hash"])


def test_audit_trail_separates_observation_and_decision_actors(tmp_path, monkeypatch):
    """观测与决策可能由不同角色产生（机器人测量、AI 决策），两者都要出现。"""
    monkeypatch.setenv("CONDOPT_ACTOR", "agent-3")
    study = make_study(tmp_path)
    pending = study.ask(2)
    monkeypatch.setenv("CONDOPT_ACTOR", "robot-1")
    study.tell([{"id": item["id"], "y": float(index)} for index, item in enumerate(pending)])

    trail = study.audit_trail()
    assert trail["actors"] == ["agent-3", "robot-1"]
