"""UI state independent of Qt and PyVista."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path

import trimesh

from rotarycam.geometry.transforms import xyz_to_xar
from rotarycam.stock import Stock
from rotarycam.supports.models import CylindricalSupport, RectangularSupport, Support
from rotarycam.tools.models import Tool

DEFAULT_SUPPORT_DIAMETER_MM = 5.0
DEFAULT_SUPPORT_THICKNESS_MM = 1.0
DEFAULT_SUPPORT_TRANSITION_MM = 0.5


class SceneLayer(StrEnum):
    TARGET = "target"
    STOCK = "stock"
    MACHINE = "machine"
    SUPPORTS = "supports"
    TOOLPATHS = "toolpaths"
    RESIDUAL = "residual"


@dataclass(frozen=True, slots=True)
class SupportPick:
    """Surface point selected for creation of a material-retaining support."""

    x: float
    y: float
    z: float
    angle_deg: float
    radius: float
    definition: Support | None = field(default=None, compare=False, repr=False)

    @classmethod
    def from_xyz(cls, x: float, y: float, z: float) -> SupportPick:
        support_x, angle_deg, radius = xyz_to_xar(x, y, z)
        definition = CylindricalSupport(
            x=support_x,
            angle_deg=angle_deg,
            thickness=DEFAULT_SUPPORT_THICKNESS_MM,
            transition=DEFAULT_SUPPORT_TRANSITION_MM,
            diameter=DEFAULT_SUPPORT_DIAMETER_MM,
        )
        return cls(support_x, y, z, angle_deg, radius, definition)

    @classmethod
    def from_support(cls, support: Support, *, radius: float = 0.0) -> SupportPick:
        """Create UI state for a persisted support definition."""

        angle_rad = math.radians(support.angle_deg)
        return cls(
            support.x,
            radius * math.cos(angle_rad),
            radius * math.sin(angle_rad),
            support.angle_deg,
            radius,
            support,
        )

    @property
    def size_mm(self) -> float:
        """Return the editable maximum footprint dimension."""

        if isinstance(self.definition, CylindricalSupport):
            return self.definition.diameter
        if isinstance(self.definition, RectangularSupport):
            return max(self.definition.length_x, self.definition.width_surface)
        return DEFAULT_SUPPORT_DIAMETER_MM

    def resized(self, size_mm: float) -> SupportPick:
        """Return a copy with a resized footprint, preserving shape and identity."""

        if not math.isfinite(size_mm) or size_mm <= 0.0:
            raise ValueError("support size must be finite and positive")
        definition = self.definition
        if isinstance(definition, CylindricalSupport):
            cylinder = definition.model_copy(update={"diameter": size_mm})
            return replace(self, definition=cylinder)
        if isinstance(definition, RectangularSupport):
            scale = size_mm / self.size_mm
            rectangle = definition.model_copy(
                update={
                    "length_x": definition.length_x * scale,
                    "width_surface": definition.width_surface * scale,
                }
            )
            return replace(self, definition=rectangle)
        cylinder = CylindricalSupport(
            x=self.x,
            angle_deg=self.angle_deg,
            thickness=DEFAULT_SUPPORT_THICKNESS_MM,
            transition=DEFAULT_SUPPORT_TRANSITION_MM,
            diameter=size_mm,
        )
        return replace(self, definition=cylinder)


@dataclass(slots=True)
class ProjectUiState:
    """Small state machine used by the desktop shell."""

    mesh_path: Path | None = None
    mesh: trimesh.Trimesh | None = None
    machine_profile_verified: bool = False
    revision: int = 0
    generated_revision: int | None = None
    simulation_revision: int | None = None
    validation_errors: tuple[str, ...] = ()
    invalidation_reason: str | None = None
    support_picks: list[SupportPick] = field(default_factory=list)
    tools: list[Tool] = field(default_factory=list)
    stock: Stock | None = None
    visibility: dict[SceneLayer, bool] = field(
        default_factory=lambda: {layer: True for layer in SceneLayer}
    )

    def set_mesh(self, path: Path, mesh: trimesh.Trimesh) -> None:
        self.mesh_path = path
        self.mesh = mesh
        self.invalidate_derived("Mesh changed")

    def add_support_pick(self, point: tuple[float, float, float]) -> SupportPick:
        support = SupportPick.from_xyz(*point)
        self.support_picks.append(support)
        self.invalidate_derived("Supports changed")
        return support

    def set_supports(self, supports: list[Support]) -> None:
        """Replace support UI records from persisted domain definitions."""

        self.support_picks = [SupportPick.from_support(support) for support in supports]
        self.invalidate_derived("Supports changed")

    def resize_support_pick(self, index: int, size_mm: float) -> SupportPick:
        """Resize one support footprint and invalidate generated results."""

        current = self.support_picks[index]
        resized = current.resized(size_mm)
        self.support_picks[index] = resized
        self.invalidate_derived("Supports changed")
        return resized

    def remove_support_pick(self, index: int) -> SupportPick:
        """Remove one support and invalidate generated results."""

        removed = self.support_picks.pop(index)
        self.invalidate_derived("Supports changed")
        return removed

    def set_tools(self, tools: list[Tool]) -> None:
        """Replace the UI tool library and invalidate derived operations."""

        self.tools = list(tools)
        self.invalidate_derived("Tool library changed")

    def set_stock(self, stock: Stock | None) -> None:
        """Replace the stock definition and invalidate derived operations."""

        self.stock = stock
        self.invalidate_derived("Stock changed")

    def invalidate_derived(self, reason: str) -> None:
        self.revision += 1
        self.generated_revision = None
        self.simulation_revision = None
        self.invalidation_reason = reason

    def mark_generated(self) -> None:
        if self.mesh is None:
            raise RuntimeError("a mesh must be loaded before generation")
        self.generated_revision = self.revision
        self.invalidation_reason = None

    def mark_simulated(self) -> None:
        if self.generated_revision != self.revision:
            raise RuntimeError("current toolpaths must be generated before simulation")
        self.simulation_revision = self.revision

    def set_machine_profile_verified(self, verified: bool) -> None:
        if self.machine_profile_verified != verified:
            self.machine_profile_verified = verified
            self.invalidate_derived("Machine profile changed")

    def set_validation_errors(self, errors: tuple[str, ...]) -> None:
        self.validation_errors = tuple(errors)

    @property
    def has_current_generation(self) -> bool:
        return self.generated_revision == self.revision

    @property
    def can_export(self) -> bool:
        return (
            self.machine_profile_verified
            and self.has_current_generation
            and not self.validation_errors
        )
