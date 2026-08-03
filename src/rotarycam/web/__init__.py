"""Optional local web application for RotaryCAM."""

from __future__ import annotations

from concurrent.futures import Executor
from pathlib import Path
from typing import TYPE_CHECKING

from rotarycam.web.models import (
    GenerationJobSnapshot,
    JobState,
    SceneManifest,
    StockSceneRecord,
    ToolpathSceneRecord,
    WebWorkspaceDocument,
)

if TYPE_CHECKING:
    from fastapi import FastAPI


def create_app(
    project_root: Path | None = None,
    tool_library_path: Path | None = None,
    machine_library_path: Path | None = None,
    executor: Executor | None = None,
) -> FastAPI:
    """Create the local web app while retaining a lazy optional dependency boundary."""

    from rotarycam.web.app import create_app as create_web_app

    return create_web_app(
        project_root=project_root,
        tool_library_path=tool_library_path,
        machine_library_path=machine_library_path,
        executor=executor,
    )

__all__ = [
    "GenerationJobSnapshot",
    "JobState",
    "SceneManifest",
    "StockSceneRecord",
    "ToolpathSceneRecord",
    "WebWorkspaceDocument",
    "create_app",
]
