from __future__ import annotations

from concurrent.futures import Executor, Future
from pathlib import Path
from typing import Any

import pytest

from rotarycam.machine.profiles import makera_z1_community_profile
from rotarycam.machine.validation import ToolpathValidationReport
from rotarycam.project import CylindricalStockConfig, ToolConfig
from rotarycam.web.jobs import (
    GenerationAlreadyRunningError,
    GenerationJobManager,
)
from rotarycam.web.models import JobState, SceneManifest, WebWorkspaceDocument, utc_now
from rotarycam.web.store import ProjectStore


class ImmediateExecutor(Executor):
    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Future[Any]:
        future: Future[Any] = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:  # pragma: no cover - mirrors Executor behavior.
            future.set_exception(exc)
        return future


class PendingExecutor(Executor):
    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Future[Any]:
        return Future()


class FakeSimulation:
    stock = None


class FakeEngine:
    def __init__(self, calls: list[str], *, fail: bool = False) -> None:
        self.calls = calls
        self.fail = fail
        self.mesh = None
        self.stock = None
        self.supports: list[Any] = []
        self.operations: list[Any] = []
        self.machine = makera_z1_community_profile()

    def build_target(self) -> None:
        self.calls.append("sampling")

    def generate_plan(self) -> None:
        self.calls.append("planning")
        if self.fail:
            raise RuntimeError("planner exploded")

    def simulate(self) -> FakeSimulation:
        self.calls.append("simulation")
        return FakeSimulation()

    def validate(self) -> ToolpathValidationReport:
        self.calls.append("validation")
        return ToolpathValidationReport()


def _complete_workspace(store: ProjectStore) -> WebWorkspaceDocument:
    created = store.create("Job", makera_z1_community_profile())
    directory = store.project_directory(created.project_id)
    (directory / "assets" / "target.stl").write_bytes(b"stl")
    tool = ToolConfig(
        number=1,
        name="Flat",
        type="flat",
        diameter=2,
        cutting_length=5,
        flute_length=5,
        overall_length=20,
        shank_diameter=2,
        max_stepdown=1,
        stepover=1,
        feed=100,
        plunge_feed=50,
        spindle_rpm=10_000,
    )
    complete = WebWorkspaceDocument.model_validate(
        created.model_copy(
            update={
                "mesh_asset": "target.stl",
                "stock": CylindricalStockConfig(length=10, diameter=12),
                "tools": (tool,),
            }
        ).model_dump(by_alias=True)
    )
    store.save(complete)
    return complete


def test_job_runs_five_stages_and_publishes_current_result(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    workspace = _complete_workspace(store)
    calls: list[str] = []

    def preview(output_directory: Path, **kwargs: Any) -> SceneManifest:
        calls.append("preview")
        output_directory.mkdir(parents=True)
        return SceneManifest(revision=int(kwargs["revision"]))

    manager = GenerationJobManager(
        store,
        executor=ImmediateExecutor(),
        engine_factory=lambda _project: FakeEngine(calls),  # type: ignore[arg-type,return-value]
        preview_builder=preview,  # type: ignore[arg-type]
    )

    manager.submit(workspace.project_id)

    assert calls == ["sampling", "planning", "simulation", "validation", "preview"]
    assert manager.snapshot(workspace.project_id).state is JobState.COMPLETED  # type: ignore[union-attr]
    assert manager.result(workspace.project_id) is not None


def test_job_discards_result_when_workspace_revision_changes(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    workspace = _complete_workspace(store)

    def stale_preview(output_directory: Path, **kwargs: Any) -> SceneManifest:
        current = store.load(workspace.project_id)
        store.save(
            WebWorkspaceDocument.model_validate(
                current.model_copy(
                    update={"revision": current.revision + 1, "updated_at": utc_now()}
                ).model_dump(by_alias=True)
            )
        )
        return SceneManifest(revision=int(kwargs["revision"]))

    manager = GenerationJobManager(
        store,
        executor=ImmediateExecutor(),
        engine_factory=lambda _project: FakeEngine([]),  # type: ignore[arg-type,return-value]
        preview_builder=stale_preview,  # type: ignore[arg-type]
    )
    manager.submit(workspace.project_id)

    snapshot = manager.snapshot(workspace.project_id)
    assert snapshot is not None
    assert snapshot.state is JobState.COMPLETED
    assert "discarded" in snapshot.message
    assert manager.result(workspace.project_id) is None


def test_job_rejects_second_active_generation_and_records_failures(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    workspace = _complete_workspace(store)
    pending = GenerationJobManager(store, executor=PendingExecutor())
    pending.submit(workspace.project_id)
    with pytest.raises(GenerationAlreadyRunningError):
        pending.submit(workspace.project_id)

    failed = GenerationJobManager(
        store,
        executor=ImmediateExecutor(),
        engine_factory=lambda _project: FakeEngine([], fail=True),  # type: ignore[arg-type,return-value]
    )
    failed.submit(workspace.project_id)
    snapshot = failed.snapshot(workspace.project_id)
    assert snapshot is not None
    assert snapshot.state is JobState.FAILED
    assert snapshot.error == "planner exploded"
