"""``prepare`` 的 answers JSON Schema 与纯 stdlib 校验器。

背景：``answers.json`` 里拼错的键以前会被 ``prepare.parse_answers`` 静默忽略
（例如 ``constaint`` 既不报错也不生效），agent 拿到的是"看起来成功"的错误结果。
这里给出显式的白名单 + 类型校验，未知键直接报错并**列出最相近的合法键**
（:mod:`difflib`），错误走 :class:`condopt.errors.InputError`（退出码 2）。

白名单依据 :mod:`condopt.prepare` 实际读取的键：``parse_answers`` 的
``ALIASES`` / ``CONTEXT_FIELDS``、``noise_advice`` / ``feasibility_advice`` /
``prior_advice``，以及 ``write_problem`` 的 ``spec_calibrate_*``。
``objective`` 只收 :meth:`condopt.base.Objective.from_dict` 认识的键；
``spec`` 交给 :meth:`condopt.spec.Spec.from_dict` 自己校验，这里只查类型，
以免 schema 与 spec 演化脱节。
"""

from __future__ import annotations

import difflib
from typing import Any, Mapping

from .errors import InputError
from .space import VALID_KINDS

#: ``answers.space`` 与 ``answers.variables`` 共用的变量 schema。
#: 键取自 :meth:`condopt.space.Variable.from_dict`。
VARIABLE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["name"],
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string"},
        "kind": {"type": "string", "enum": list(VALID_KINDS)},
        "low": {"type": "number"},
        "high": {"type": "number"},
        "choices": {"type": "array", "items": {"type": "string"}},
        "unit": {"type": "string"},
        "description": {"type": "string"},
        "descriptors": {"type": "object"},
        "descriptor_names": {"type": "array", "items": {"type": "string"}},
    },
}

#: ``answers.objective`` 的键取自 :meth:`condopt.base.Objective.from_dict`。
OBJECTIVE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string"},
        "direction": {"type": "string", "enum": ["max", "min"]},
        "objectives": {"type": "array", "items": {"type": "string"}},
        "directions": {"type": "array", "items": {"type": "string"}},
        "constraints": {"type": "array", "items": {"type": "string"}},
        "constraint_limits": {"type": "object"},
        "ref_point": {"type": "array", "items": {"type": "number"}},
        "objective_bounds": {"type": "object"},
    },
}

#: ``answers.json`` 顶层 schema（``additionalProperties: false`` 即未知键白名单）。
ANSWERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["variables"],
    "additionalProperties": False,
    "properties": {
        "variables": {"type": "array", "minItems": 1, "items": VARIABLE_SCHEMA},
        "space": {
            "type": "object",
            "required": ["variables"],
            "additionalProperties": False,
            "properties": {
                "variables": {"type": "array", "minItems": 1, "items": VARIABLE_SCHEMA}
            },
        },
        "objective": OBJECTIVE_SCHEMA,
        "spec": {"type": "object"},
        # 别名（prepare.ALIASES）+ 直接可用的 OptimizationContext 字段
        "budget": {"type": "integer", "minimum": 1},
        "budget_total": {"type": "integer", "minimum": 1},
        "budget_remaining": {"type": "integer", "minimum": 0},
        "batch": {"type": "integer", "minimum": 1},
        "batch_size": {"type": "integer", "minimum": 1},
        "prior": {"type": "integer", "minimum": 0},
        "n_observations": {"type": "integer", "minimum": 0},
        "constraints": {"type": "integer", "minimum": 0},
        "n_constraints": {"type": "integer", "minimum": 0},
        "n_objectives": {"type": "integer", "minimum": 1},
        "noise": {"type": "string", "enum": ["none", "low", "medium", "high"]},
        "smoothness": {"type": "string", "enum": ["smooth", "mixed", "rough"]},
        "compute": {"type": "string", "enum": ["low", "medium", "high"]},
        "domain": {"type": "string", "enum": ["chemistry", "materials", "process", "generic"]},
        "multi_fidelity": {"type": "boolean"},
        "fidelity": {"type": "boolean"},
        "notes": {"type": "string"},
        # prepare 的建议函数读取
        "feasible_fraction_estimate": {"type": "number", "minimum": 0, "maximum": 1},
        # write_problem 的 spec 校准开关
        "spec_calibrate_from": {"type": "string"},
        "spec_calibrate_by_plate": {"type": "boolean"},
        "spec_calibrate_min_samples": {"type": "integer", "minimum": 1},
        "spec_calibrate_tolerance_sigmas": {"type": "number", "minimum": 0},
    },
}

#: 顶层合法键（agent 自描述 / 错误提示用）。
KNOWN_ANSWERS_KEYS: tuple[str, ...] = tuple(ANSWERS_SCHEMA["properties"])


def _kind(value: Any) -> str:
    """给错误消息用的、比 ``type(value).__name__`` 更好读的类型名。"""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return type(value).__name__


def _matches_type(value: Any, name: str) -> bool:
    if name == "object":
        return isinstance(value, Mapping)
    if name == "array":
        return isinstance(value, (list, tuple))
    if name == "string":
        return isinstance(value, str)
    if name == "boolean":
        return isinstance(value, bool)
    # bool 是 int 的子类，但 "true" 不是合法的预算值。
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    raise InputError(f"内部错误：answers schema 使用了未知类型 {name!r}")


def _unknown_keys_error(
    unknown: list[str], properties: Mapping[str, Any], path: str
) -> InputError:
    """未知键错误：用 difflib 给出最相近的合法键，而不是让 agent 盲猜。"""
    allowed = sorted(properties)
    parts = []
    for key in unknown:
        close = difflib.get_close_matches(str(key), allowed, n=3, cutoff=0.6)
        if close:
            parts.append(f"{path}.{key}（是否想写 {', '.join(close)}？）")
        else:
            parts.append(f"{path}.{key}（无相近字段）")
    return InputError(
        f"未知字段：" + "；".join(parts) + f"。合法字段：{allowed}",
        hint="删掉或改正这些字段后重跑 prepare；完整模板见 condopt prepare --template",
    )


def _validate(value: Any, schema: Mapping[str, Any], path: str) -> None:
    """按 schema 子集（type/enum/minimum/maximum/items/properties/required）校验。"""
    expected = schema.get("type")
    if expected is not None:
        names = [expected] if isinstance(expected, str) else list(expected)
        if not any(_matches_type(value, name) for name in names):
            raise InputError(
                f"{path} 的类型应为 {'/'.join(names)}，实际是 {_kind(value)}",
                hint="对照 condopt prepare --template 修正类型",
            )
    if "enum" in schema and value not in schema["enum"]:
        raise InputError(
            f"{path}={value!r} 不在允许取值 {list(schema['enum'])} 内",
            hint="改用允许取值之一",
        )
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise InputError(f"{path}={value!r} 小于下限 {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise InputError(f"{path}={value!r} 大于上限 {schema['maximum']}")
    properties = schema.get("properties")
    if isinstance(value, Mapping) and properties is not None:
        unknown = [str(key) for key in value if key not in properties]
        if unknown and schema.get("additionalProperties") is False:
            raise _unknown_keys_error(unknown, properties, path)
        for key in schema.get("required", ()):
            if key not in value:
                raise InputError(
                    f"{path} 缺少必填字段 {key!r}",
                    hint=f"合法字段：{sorted(properties)}",
                )
        for key, sub_schema in properties.items():
            if key in value:
                _validate(value[key], sub_schema, f"{path}.{key}")
    if isinstance(value, (list, tuple)) and isinstance(schema.get("items"), Mapping):
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise InputError(f"{path} 至少需要 {schema['minItems']} 个元素，实际 {len(value)} 个")
        for index, item in enumerate(value):
            _validate(item, schema["items"], f"{path}[{index}]")


def validate_answers(answers: Any) -> Mapping:
    """校验 ``prepare`` 的 answers，通过则原样返回（便于链式调用）。

    ``space.variables`` 是 ``variables`` 的等价写法（``parse_answers`` 支持），
    所以先按这个等价关系归一化再校验，避免把合法输入误判成缺必填项；
    调用方拿到的仍是原始对象。
    """
    normalized = answers
    if isinstance(answers, Mapping) and "variables" not in answers:
        nested = answers.get("space")
        if isinstance(nested, Mapping) and nested.get("variables"):
            normalized = {**answers, "variables": nested["variables"]}
    _validate(normalized, ANSWERS_SCHEMA, "answers")
    return answers
