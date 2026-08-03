"""Command-line launcher for the loopback-only RotaryCAM web application."""

from __future__ import annotations

import argparse
import threading
import webbrowser
from collections.abc import Sequence

LOOPBACK_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65_535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def build_parser() -> argparse.ArgumentParser:
    """Create the small launcher parser without importing optional web packages."""

    parser = argparse.ArgumentParser(
        prog="rotarycam-web",
        description="Open the local, loopback-only RotaryCAM web workspace.",
    )
    parser.add_argument("--port", type=_port, default=DEFAULT_PORT)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="start the server without opening the default browser",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run Uvicorn on loopback and optionally open the exact local URL."""

    arguments = build_parser().parse_args(argv)
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "RotaryCAM web dependencies are not installed; install 'rotarycam[web]'."
        ) from exc

    from rotarycam.web.app import create_app

    url = f"http://{LOOPBACK_HOST}:{arguments.port}"
    if not arguments.no_browser:
        opener = threading.Timer(0.75, webbrowser.open, args=(url,))
        opener.daemon = True
        opener.start()
    uvicorn.run(
        create_app(),
        host=LOOPBACK_HOST,
        port=arguments.port,
        access_log=False,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
