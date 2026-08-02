"""Qt thread-pool worker used for mesh loading and later CAM operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal, Slot


class WorkerSignals(QObject):
    result = Signal(object)
    error = Signal(str)
    progress = Signal(object)
    finished = Signal()


@dataclass(frozen=True, slots=True)
class WorkerProgress:
    """A user-facing phase within a background operation."""

    step: int
    total: int
    message: str

    def __post_init__(self) -> None:
        if self.total <= 0 or not 1 <= self.step <= self.total:
            raise ValueError("progress step must be within the positive total")
        if not self.message.strip():
            raise ValueError("progress message must not be empty")


ProgressReporter = Callable[[WorkerProgress], None]


class FunctionWorker(QRunnable):
    """Run a zero-argument callable and forward its result to the UI thread."""

    def __init__(self, function: Callable[[], Any]) -> None:
        super().__init__()
        self.function = function
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = self.function()
        except Exception as exc:  # UI boundary reports all loader/backend errors
            self.signals.error.emit(str(exc))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


class ProgressFunctionWorker(QRunnable):
    """Run a callable that reports coarse, meaningful UI phases."""

    def __init__(self, function: Callable[[ProgressReporter], Any]) -> None:
        super().__init__()
        self.function = function
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = self.function(self.signals.progress.emit)
        except Exception as exc:  # UI boundary reports all backend errors
            self.signals.error.emit(str(exc))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()
