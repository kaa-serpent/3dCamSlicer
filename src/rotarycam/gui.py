"""Lazy entry point for the optional RotaryCAM desktop interface."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def create_main_window(
    *,
    plotter: Any | None = None,
    engine: Any | None = None,
    tool_library_path: Path | None = None,
    machine_library_path: Path | None = None,
) -> Any:
    """Create the window while keeping optional UI imports out of core imports."""

    try:
        from rotarycam.viewer.window import MainWindow
    except ModuleNotFoundError as exc:
        if exc.name in {"PySide6", "pyvista", "pyvistaqt", "vtk"}:
            raise RuntimeError(
                "RotaryCAM UI dependencies are not installed; install 'rotarycam[ui]'."
            ) from exc
        raise
    return MainWindow(
        plotter=plotter,
        engine=engine,
        tool_library_path=tool_library_path,
        machine_library_path=machine_library_path,
    )


def main() -> int:
    """Launch the Qt application for the ``rotarycam-gui`` console script."""

    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:
        print(
            "RotaryCAM UI dependencies are not installed; install 'rotarycam[ui]'.",
            file=sys.stderr,
        )
        return 2

    application = QApplication.instance() or QApplication(sys.argv)
    from rotarycam.machine.library import default_machine_library_path
    from rotarycam.tools.library import default_tool_library_path

    window = create_main_window(
        tool_library_path=default_tool_library_path(),
        machine_library_path=default_machine_library_path(),
    )
    window.show()
    return int(application.exec())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
