"""Read-only local dashboard server for a condopt study.

Serves an auto-refreshing view of ``runs/<study>/dashboard.html`` on
``http://127.0.0.1:<port>/``.

Safety contract (enforced here, not by convention):

* binds to ``127.0.0.1`` only -- never ``0.0.0.0``;
* only ``GET`` is served; every other method gets ``405`` (no POST/PUT/DELETE);
* the run directory is only read.  Rendering output (``dashboard.html`` and
  optionally ``convergence.png``) goes to ``--out`` or a throw-away temp dir;
* path traversal attempts are rejected with ``400``; unknown paths get ``404``;
* no CDN / external request is ever referenced by the served page.

Usage (Windows, repository root)::

    py -3 scripts/serve.py --run runs/demo
    py -3 scripts/serve.py --run runs/demo_moo --port 8765 --out build/serve --png

Programmatic use (tests)::

    from serve import create_server  # loaded by path
    httpd = create_server(run_dir, port=0)
    httpd.serve_forever()  # or in a background thread
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import threading
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

def _utf8_child_env() -> dict:
    """Environment for child processes so their stdio is UTF-8, not the codepage."""
    import os

    return {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RENDER_PATH = SRC / "condopt" / "render.py"
REFRESH_SECONDS = 10
BIND_HOST = "127.0.0.1"


def _load_render_module():
    """Load ``render.py`` without importing the ``condopt`` package.

    The package initialisation pulls scikit-learn through ``condopt.algorithms``
    (seconds of import time); the renderer itself is stdlib-only.  Fall back to
    the normal import if the direct load fails.
    """
    try:
        spec = importlib.util.spec_from_file_location("_condopt_render_static", RENDER_PATH)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load renderer from {RENDER_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:  # pragma: no cover - defensive fallback
        if str(SRC) not in sys.path:
            sys.path.insert(0, str(SRC))
        from condopt import render as package_render

        return package_render


_RENDER = _load_render_module()
render_study = _RENDER.render_study
load_run = _RENDER.load_run


def _state_payload(root: Path) -> dict:
    info = load_run(root)
    config = info["config"] or {}
    context = config.get("context") or {}
    objective = (info["problem"].get("objective") or {}).get("objectives") or []
    budget = int(info["problem"].get("budget") or context.get("budget_total") or 0)
    return {
        "run": root.name,
        "algorithm": config.get("algorithm"),
        "objectives": objective,
        "budget": budget,
        "used": len(info["dataset"]),
        "pending": len(info["pending"]),
        "rounds": len(info["rounds"]),
        "decisions": len(info["decisions"]),
        "refresh_seconds": REFRESH_SECONDS,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


class _QuietThreadingHTTPServer(ThreadingHTTPServer):
    """Threaded server that stays quiet when a browser/client resets a socket."""

    def handle_error(self, request, client_address) -> None:  # noqa: ANN001 - stdlib signature
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


def create_server(
    run_dir: str | Path,
    port: int = 0,
    out_dir: str | Path | None = None,
    make_png: bool = False,
    verbose: bool = False,
) -> ThreadingHTTPServer:
    """Create (but do not start) the read-only dashboard server.

    ``port=0`` asks the OS for a free port; read ``server.server_address`` after
    creation.  The initial render is done eagerly so ``/convergence.png`` is
    available when ``make_png=True``; ``/`` re-renders HTML on every request so
    the page tracks new ``tell`` results.
    """
    root = Path(run_dir).resolve()
    if not (root / "problem.json").exists():
        raise FileNotFoundError(
            f"{root} does not look like a condopt run (problem.json missing)"
        )
    output = Path(out_dir).resolve() if out_dir is not None else Path(
        tempfile.mkdtemp(prefix="condopt-serve-")
    )
    output.mkdir(parents=True, exist_ok=True)
    render_study(root, out_dir=output, make_png=make_png)

    render_lock = threading.Lock()
    refresh_meta = f'<meta http-equiv="refresh" content="{REFRESH_SECONDS}">'

    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "condopt-render/0.1"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:  # noqa: A003 - stdlib name
            if verbose:
                sys.stderr.write(f"{self.address_string()} {fmt % args}\n")

        # -- response helpers -------------------------------------------------
        def _send(self, status: int, content_type: str, body: bytes, extra: dict | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _text(self, status: int, text: str, extra: dict | None = None) -> None:
            self._send(status, "text/plain; charset=utf-8", text.encode("utf-8"), extra)

        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False, indent=1).encode("utf-8")
            self._send(status, "application/json; charset=utf-8", body)

        def _method_not_allowed(self) -> None:
            self._text(
                HTTPStatus.METHOD_NOT_ALLOWED,
                "405 method not allowed：这是只读看板，只接受 GET\n",
                {"Allow": "GET"},
            )

        def _safe_path(self) -> str | None:
            raw_path = urlsplit(self.path).path
            try:
                decoded = unquote(raw_path, errors="strict")
            except UnicodeDecodeError:
                return None
            if "\\" in decoded or "\x00" in decoded or not decoded.startswith("/"):
                return None
            if any(part == ".." for part in decoded.split("/")):
                return None
            return decoded

        # -- routes -----------------------------------------------------------
        def _serve_dashboard(self) -> None:
            with render_lock:
                result = render_study(root, out_dir=output, make_png=False)
            document = Path(result["dashboard"]).read_text(encoding="utf-8", errors="replace")
            document = document.replace("</head>", refresh_meta + "</head>", 1)
            self._send(HTTPStatus.OK, "text/html; charset=utf-8", document.encode("utf-8"))

        def _serve_png(self) -> None:
            png = output / "convergence.png"
            if not png.exists():
                self._text(HTTPStatus.NOT_FOUND, "404 not found：PNG 未生成（加 --png 启动）\n")
                return
            self._send(HTTPStatus.OK, "image/png", png.read_bytes())

        def do_GET(self) -> None:  # noqa: N802 - stdlib name
            path = self._safe_path()
            if path is None:
                self._text(HTTPStatus.BAD_REQUEST, "400 bad path\n")
                return
            if path in ("/", "/dashboard.html"):
                self._serve_dashboard()
            elif path == "/api/state":
                self._json(HTTPStatus.OK, _state_payload(root))
            elif path == "/convergence.png":
                self._serve_png()
            else:
                self._text(HTTPStatus.NOT_FOUND, "404 not found\n")

        # every non-GET method is refused, including HEAD (GET only by design)
        def do_HEAD(self) -> None:  # noqa: N802
            self._method_not_allowed()

        def do_POST(self) -> None:  # noqa: N802
            self._method_not_allowed()

        def do_PUT(self) -> None:  # noqa: N802
            self._method_not_allowed()

        def do_DELETE(self) -> None:  # noqa: N802
            self._method_not_allowed()

        def do_PATCH(self) -> None:  # noqa: N802
            self._method_not_allowed()

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._method_not_allowed()

    httpd = _QuietThreadingHTTPServer((BIND_HOST, int(port)), DashboardHandler)
    httpd.daemon_threads = True
    httpd.render_output = output  # type: ignore[attr-defined]
    return httpd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="serve",
        description="condopt 只读看板服务（127.0.0.1，GET only，不修改 run 目录）",
    )
    parser.add_argument("--run", required=True, help="study 目录（含 problem.json/config.json/dataset.jsonl）")
    parser.add_argument("--port", type=int, default=8765, help="监听端口（默认 8765；0=自动分配）")
    parser.add_argument("--out", default=None, help="渲染输出目录（默认临时目录）")
    parser.add_argument("--png", action="store_true", help="启动时同时生成 convergence.png")
    parser.add_argument("--verbose", action="store_true", help="打印访问日志")
    args = parser.parse_args(argv)

    try:
        httpd = create_server(
            args.run,
            port=args.port,
            out_dir=args.out,
            make_png=args.png,
            verbose=args.verbose,
        )
    except FileNotFoundError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"错误：无法监听 {BIND_HOST}:{args.port}（{exc}）", file=sys.stderr)
        return 1

    host, port = httpd.server_address[:2]
    print("condopt 只读看板已启动：")
    print(f"  URL      : http://{host}:{port}/")
    print(f"  run      : {Path(args.run).resolve()}")
    print(f"  渲染输出 : {httpd.render_output}")  # type: ignore[attr-defined]
    print(f"  规则     : 只绑 {BIND_HOST}、只接受 GET、不修改 run 目录；页面每 {REFRESH_SECONDS}s 自动刷新")
    print("  Ctrl+C 停止")
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
