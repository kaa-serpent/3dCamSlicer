"""Single-project background generation with revision-safe results."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

import trimesh

from rotarycam.engine import RotaryCamEngine
from rotarycam.errors import RotaryCamError
from rotarycam.geometry.models import RotaryGrid
from rotarycam.machine.validation import ToolpathValidationReport
from rotarycam.planning import MachiningOperation
from rotarycam.project import RotaryCamProject
from rotarycam.stock import Stock
from rotarycam.supports import Support
from rotarycam.web.models import GenerationJobSnapshot, JobState, SceneManifest
from rotarycam.web.store import ProjectStore


class GenerationJobError(RotaryCamError):
    """Base class for expected background generation failures."""


class GenerationAlreadyRunningError(GenerationJobError):
    """Raised when a project already has a queued or running job."""


class GenerationNotReadyError(GenerationJobError):
    """Raised when no current successful result is available."""


class PreviewBuilder(Protocol):
    """Callable boundary implemented by the optional Three.js preview adapter."""

    def __call__(
        self,
        output_directory: Path,
        *,
        revision: int,
        target_mesh: trimesh.Trimesh | None,
        residual_grid: RotaryGrid | None,
        stock: Stock | None,
        supports: Sequence[Support],
        operations: Sequence[MachiningOperation],
        url_prefix: str,
    ) -> SceneManifest: ...


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """In-memory generated state, valid only for its exact workspace revision."""

    project_id: UUID
    revision: int
    engine: RotaryCamEngine
    validation: ToolpathValidationReport
    manifest: SceneManifest


def _default_preview_builder(
    output_directory: Path,
    *,
    revision: int,
    target_mesh: trimesh.Trimesh | None,
    residual_grid: RotaryGrid | None,
    stock: Stock | None,
    supports: Sequence[Support],
    operations: Sequence[MachiningOperation],
    url_prefix: str,
) -> SceneManifest:
    from rotarycam.web.preview import build_scene_assets

    # The adapter owns stricter concrete geometry types; this lazy wrapper keeps
    # the web service importable while optional preview code is assembled.
    return build_scene_assets(
        output_directory,
        revision=revision,
        target_mesh=target_mesh,
        residual_grid=residual_grid,
        stock=stock,
        supports=supports,
        operations=operations,
        url_prefix=url_prefix,
    )


class GenerationJobManager:
    """Run at most one job per project and discard stale revision results."""

    def __init__(
        self,
        store: ProjectStore,
        *,
        executor: Executor | None = None,
        engine_factory: Callable[[RotaryCamProject], RotaryCamEngine] | None = None,
        preview_builder: PreviewBuilder | None = None,
    ) -> None:
        self.store = store
        self._owns_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="rotarycam-web",
        )
        self._engine_factory = engine_factory or RotaryCamEngine.from_project
        self._preview_builder = preview_builder or _default_preview_builder
        self._lock = threading.RLock()
        self._snapshots: dict[UUID, GenerationJobSnapshot] = {}
        self._results: dict[UUID, GenerationResult] = {}
        self._futures: dict[UUID, Future[None]] = {}

    def close(self) -> None:
        """Release the internally owned worker without controlling injected executors."""

        if self._owns_executor:
            assert isinstance(self._executor, ThreadPoolExecutor)
            self._executor.shutdown(wait=False, cancel_futures=True)

    def submit(self, project_id: UUID | str) -> GenerationJobSnapshot:
        """Queue a complete workspace unless that project already has active work."""

        workspace = self.store.load(project_id)
        if not workspace.is_complete:
            raise GenerationNotReadyError(
                "Mesh, stock, and at least one tool are required before generation."
            )
        parsed = workspace.project_id
        with self._lock:
            existing = self._snapshots.get(parsed)
            if existing is not None and existing.state in {JobState.QUEUED, JobState.RUNNING}:
                raise GenerationAlreadyRunningError(
                    "Generation is already running for this project."
                )
            snapshot = GenerationJobSnapshot(
                job_id=uuid4(),
                project_id=parsed,
                revision=workspace.revision,
                state=JobState.QUEUED,
                message="Waiting to start",
            )
            self._snapshots[parsed] = snapshot
            self._results.pop(parsed, None)
            self._futures[parsed] = self._executor.submit(self._run, snapshot)
            return snapshot

    def snapshot(self, project_id: UUID | str) -> GenerationJobSnapshot | None:
        """Return the latest immutable job status for a project."""

        parsed = self.store.parse_project_id(project_id)
        with self._lock:
            return self._snapshots.get(parsed)

    def result(self, project_id: UUID | str) -> GenerationResult | None:
        """Return a result only when it still matches the persisted revision."""

        workspace = self.store.load(project_id)
        with self._lock:
            result = self._results.get(workspace.project_id)
        if result is None or result.revision != workspace.revision:
            return None
        return result

    def require_result(self, project_id: UUID | str) -> GenerationResult:
        """Return a current generation or raise an expected domain error."""

        result = self.result(project_id)
        if result is None:
            raise GenerationNotReadyError("Generate the current project before continuing.")
        return result

    def _set_progress(
        self,
        initial: GenerationJobSnapshot,
        *,
        step: int,
        message: str,
    ) -> None:
        with self._lock:
            self._snapshots[initial.project_id] = initial.model_copy(
                update={
                    "state": JobState.RUNNING,
                    "step": step,
                    "message": message,
                    "error": None,
                }
            )

    def _run(self, initial: GenerationJobSnapshot) -> None:
        try:
            workspace = self.store.load(initial.project_id)
            if workspace.revision != initial.revision:
                raise GenerationNotReadyError("Project inputs changed before generation started.")
            project = workspace.to_project(self.store.project_directory(initial.project_id))
            engine = self._engine_factory(project)

            self._set_progress(initial, step=1, message="Sampling radial target")
            engine.build_target()

            self._set_progress(initial, step=2, message="Planning toolpaths")
            engine.generate_plan()

            self._set_progress(initial, step=3, message="Simulating material removal")
            simulation = engine.simulate()

            self._set_progress(initial, step=4, message="Validating machine safety")
            validation = engine.validate()

            self._set_progress(initial, step=5, message="Preparing browser preview")
            output_directory = (
                self.store.project_directory(initial.project_id)
                / "scene"
                / str(initial.revision)
            )
            manifest = self._preview_builder(
                output_directory,
                revision=initial.revision,
                target_mesh=engine.mesh,
                residual_grid=simulation.stock,
                stock=engine.stock,
                supports=engine.supports,
                operations=engine.operations,
                url_prefix=(
                    f"/projects/{initial.project_id}/scene/assets/{initial.revision}"
                ),
            )

            current = self.store.load(initial.project_id)
            with self._lock:
                if current.revision == initial.revision:
                    self._results[initial.project_id] = GenerationResult(
                        project_id=initial.project_id,
                        revision=initial.revision,
                        engine=engine,
                        validation=validation,
                        manifest=manifest,
                    )
                    message = "Generation complete"
                else:
                    self._results.pop(initial.project_id, None)
                    message = "Inputs changed; generated results were discarded"
                self._snapshots[initial.project_id] = initial.model_copy(
                    update={
                        "state": JobState.COMPLETED,
                        "step": initial.total,
                        "message": message,
                        "error": None,
                    }
                )
        except Exception as exc:
            with self._lock:
                current_snapshot = self._snapshots.get(initial.project_id, initial)
                self._results.pop(initial.project_id, None)
                self._snapshots[initial.project_id] = current_snapshot.model_copy(
                    update={
                        "state": JobState.FAILED,
                        "message": "Generation failed",
                        "error": str(exc) or type(exc).__name__,
                    }
                )


__all__ = [
    "GenerationAlreadyRunningError",
    "GenerationJobError",
    "GenerationJobManager",
    "GenerationNotReadyError",
    "GenerationResult",
    "PreviewBuilder",
]
