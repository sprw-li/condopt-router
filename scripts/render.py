"""Render a static, offline dashboard for a condopt study.

Usage (Windows, from the repository root)::

    py -3 scripts/render.py --run runs\\demo
    py -3 scripts/render.py --run runs\\demo --out build --no-png

The renderer only reads the run directory; it never changes the campaign.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RENDER_PATH = SRC / "condopt" / "render.py"


def _load_render_study():
    """Load ``render.py`` by path instead of ``import condopt.render``.

    Importing the ``condopt`` package initialises ``condopt.algorithms``,
    which imports scikit-learn (~3 s import graph, ~8-16 s cold process on
    this machine).  The renderer is pure stdlib, so a static dashboard should
    not pay that cost.  Fall back to the normal import if the direct load
    fails for any reason.
    """
    try:
        spec = importlib.util.spec_from_file_location("_condopt_render_static", RENDER_PATH)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load renderer from {RENDER_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.render_study
    except Exception:
        if str(SRC) not in sys.path:
            sys.path.insert(0, str(SRC))
        from condopt.render import render_study as _render_study

        return _render_study


render_study = _load_render_study()


def _size_kb(path: str | None) -> str:
    if not path:
        return "—"
    try:
        return f"{(Path(path).stat().st_size / 1024):.1f} KB"
    except OSError:
        return "?"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="render",
        description="生成 condopt 静态看板（单文件 HTML + 可选 convergence.png）",
    )
    parser.add_argument("--run", required=True, help="study 目录（含 problem.json/config.json/dataset.jsonl）")
    parser.add_argument("--out", default=None, help="输出目录（默认写到 run 目录下）")
    parser.add_argument("--no-png", action="store_true", help="只生成 HTML，跳过 matplotlib PNG")
    args = parser.parse_args(argv)

    try:
        result = render_study(args.run, out_dir=args.out, make_png=not args.no_png)
    except FileNotFoundError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    print("看板已生成：")
    print(f"  dashboard : {result['dashboard']}  ({_size_kb(result['dashboard'])})")
    if result["convergence_png"]:
        print(f"  png       : {result['convergence_png']}  ({_size_kb(result['convergence_png'])})")
    else:
        print("  png       : 跳过（--no-png 或 matplotlib 不可用）")
    print(
        f"  数据      : {result['n_observations']} 观测 / {result['n_rounds']} 轮 / "
        f"{result['n_pending']} 待测 / {result['n_decisions']} 决策"
    )
    for warning in result.get("warnings") or []:
        print(f"  警告      : {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
