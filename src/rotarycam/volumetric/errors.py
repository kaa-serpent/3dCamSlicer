"""Expected failures raised by the volumetric geometry domain."""

from __future__ import annotations

from rotarycam.errors import RotaryCamError


class VolumetricGeometryError(RotaryCamError):
    """Base class for deterministic volumetric-geometry failures."""


class InvalidVolumeError(VolumetricGeometryError):
    """Raised when a volume or lattice violates a numerical invariant."""


class MemoryBudgetExceeded(VolumetricGeometryError):
    """Raised instead of silently coarsening a requested lattice."""

    def __init__(self, required_bytes: int, budget_bytes: int) -> None:
        self.required_bytes = required_bytes
        self.budget_bytes = budget_bytes
        super().__init__(
            f"volumetric lattice requires {required_bytes} bytes, "
            f"exceeding the {budget_bytes}-byte memory budget"
        )


class InvalidSolidMeshError(VolumetricGeometryError):
    """Raised when triangle geometry cannot define a closed solid."""
