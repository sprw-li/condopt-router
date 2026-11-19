"""结构化 CLI 错误：稳定退出码 + 一行 JSON 错误包。

agent（skill 层）靠 ``stderr`` 的 JSON 决定下一步，而不是从 traceback 里猜恢复
动作，所以这里同时固定两件事：

* **错误包字段**（:func:`error_payload`，字段名固定，不再增减）::

      {"ok": false,
       "error": {"type": "...", "message": "...", "hint": "..."}}

* **退出码表**：

  ======  ================================================================
  代码    含义
  ======  ================================================================
  0       成功
  1       未分类（``doctor`` 自检发现不一致等；未知异常仍打 traceback）
  2       输入错误（``InputError``）：参数/JSON/字段名或类型不合法
  3       状态错误（``StateError``）：run 未 init、pending id 未知、
          待测批次或预算状态冲突
  4       环境错误（``EnvironmentError_``）：输入文件缺失、可选依赖不可用
  ======  ================================================================

退出码与错误类一一对应，新的异常必须继承 :class:`CondoptError` 并声明
``exit_code``，否则 agent 无法稳定分支。

``error.type`` 用**具体异常类名**：CLI 自己抛的是 ``InputError``/``StateError``/
``EnvironmentError_``；库层（study/space/objective/spec）透传的 ``ValueError`` /
``FileNotFoundError`` / ``JSONDecodeError`` 保留原名，方便定位到抛出点。要按族
分支就用退出码（由 :func:`exit_code_for` 归一化），要定位细节就读 ``type``。
"""

from __future__ import annotations

import json
from typing import Any

#: 成功（与 ``argparse`` 的 0 一致）。
EXIT_OK = 0
#: 输入错误：参数、JSON、字段名/类型/取值不合法。
EXIT_INPUT = 2
#: 状态错误：run 未初始化、pending id 未知、批次/预算状态冲突。
EXIT_STATE = 3
#: 环境错误：输入文件缺失、可选依赖不可用。
EXIT_ENVIRONMENT = 4


class CondoptError(Exception):
    """所有已知 CLI 错误的基类。

    Parameters
    ----------
    message:
        面向 agent/用户的一句话描述（会原样进 JSON 的 ``error.message``）。
    hint:
        下一步怎么修；缺省用子类的 ``default_hint``，永远不为 ``None``，
        这样 agent 不需要处理字段缺失。
    """

    exit_code: int = EXIT_INPUT
    default_hint: str = ""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = str(message)
        self.hint = self.default_hint if hint is None else str(hint)


class InputError(CondoptError):
    """输入错误（退出码 2）：拼错的键、错误的类型、越界的取值。"""

    exit_code = EXIT_INPUT
    default_hint = "检查参数与 JSON 字段名/类型；prepare 的合法字段见 condopt prepare --template"


class StateError(CondoptError):
    """状态错误（退出码 3）：未 init 就 ask、pending id 未知、批次状态冲突。"""

    exit_code = EXIT_STATE
    default_hint = "先 condopt init --run <dir>；当前待回填 id 见 condopt pending --run <dir>"


class EnvironmentError_(CondoptError):
    """环境错误（退出码 4）：文件缺失/不可读、可选依赖不可用。

    名字带下划线是为了不遮蔽内置 ``EnvironmentError``（``OSError`` 的别名）。
    """

    exit_code = EXIT_ENVIRONMENT
    default_hint = "确认文件/目录存在且可读；路径含空格时加引号"


#: 非 CondoptError 的已知异常 → 提示语（退出码见 :func:`exit_code_for`）。
_DEFAULT_HINTS: tuple[tuple[type[BaseException], str], ...] = (
    (
        FileNotFoundError,
        "确认路径存在（run 目录需要先 condopt init 创建）",
    ),
    (
        json.JSONDecodeError,
        "文件必须是合法 JSON（注意 BOM、尾逗号、单引号与 Windows 转义）",
    ),
    (
        ValueError,
        "检查取值/字段是否合法（类型、范围、枚举）",
    ),
)


def exit_code_for(exc: BaseException) -> int:
    """已知异常 → 退出码（表见模块 docstring）。"""
    if isinstance(exc, CondoptError):
        return exc.exit_code
    if isinstance(exc, FileNotFoundError):
        return EXIT_ENVIRONMENT
    return EXIT_INPUT


def hint_for(exc: BaseException) -> str:
    """已知异常 → 修复提示；未知类型返回空串而不是 ``None``。"""
    if isinstance(exc, CondoptError):
        return exc.hint
    for exc_type, hint in _DEFAULT_HINTS:
        if isinstance(exc, exc_type):
            return hint
    return ""


def error_payload(exc: BaseException) -> dict[str, Any]:
    """把异常转成固定的错误包（agent 只依赖这三个字段）。"""
    return {
        "ok": False,
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "hint": hint_for(exc),
        },
    }
