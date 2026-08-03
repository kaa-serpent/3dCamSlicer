"""Application service joining web inputs to typed RotaryCAM domain records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO
from uuid import UUID

from rotarycam.config import MachineDefinition, MachiningSettings
from rotarycam.errors import RotaryCamError
from rotarycam.geometry.transforms import xyz_to_xar
from rotarycam.machine.library import load_machine_library
from rotarycam.machine.profiles import makera_z1_community_profile
from rotarycam.project import (
    CylindricalStockConfig,
    RectangularStockConfig,
    StockDefinition,
    ToolConfig,
)
from rotarycam.supports import CylindricalSupport, RectangularSupport, Support
from rotarycam.tools.library import ToolRecord, load_tool_library
from rotarycam.tools.models import Tool
from rotarycam.web.models import WebWorkspaceDocument, utc_now
from rotarycam.web.store import ProjectStore, WebStoreError


class WebServiceError(RotaryCamError):
    """Base class for expected web application input failures."""


class LibrarySelectionError(WebServiceError):
    """Raised when a requested tool or machine is not in the configured library."""


class SupportNotFoundError(WebServiceError):
    """Raised when a workspace does not contain the requested support."""


def _tool_config(tool: Tool) -> ToolConfig:
    record = ToolRecord.from_tool(tool)
    return ToolConfig.model_validate(record.model_dump(by_alias=True))


class WebBackendService:
    """Apply validated mutations and invalidate generated work by revision."""

    def __init__(
        self,
        store: ProjectStore,
        *,
        tool_library_path: Path | None = None,
        machine_library_path: Path | None = None,
    ) -> None:
        self.store = store
        self.tool_library_path = tool_library_path
        self.machine_library_path = machine_library_path
        self.tools = self._load_tools(tool_library_path)
        self.machines = self._load_machines(machine_library_path)

    @staticmethod
    def _load_tools(path: Path | None) -> list[Tool]:
        if path is None or not path.is_file():
            return []
        try:
            return load_tool_library(path)
        except (OSError, ValueError) as exc:
            raise WebStoreError(f"Unable to load tool library: {exc}") from exc

    @staticmethod
    def _load_machines(path: Path | None) -> list[MachineDefinition]:
        if path is None or not path.is_file():
            return [makera_z1_community_profile()]
        try:
            profiles = load_machine_library(path)
        except (OSError, ValueError) as exc:
            raise WebStoreError(f"Unable to load machine library: {exc}") from exc
        return profiles or [makera_z1_community_profile()]

    @property
    def tool_configs(self) -> tuple[ToolConfig, ...]:
        """Return the current personal library in persistence-schema form."""

        return tuple(_tool_config(tool) for tool in self.tools)

    def create_project(self, name: str) -> WebWorkspaceDocument:
        """Create a project with a safe, unverified bundled machine default."""

        return self.store.create(name, makera_z1_community_profile())

    def _mutate(self, current: WebWorkspaceDocument, **updates: Any) -> WebWorkspaceDocument:
        changed = current.model_copy(
            update={
                **updates,
                "revision": current.revision + 1,
                "updated_at": utc_now(),
            }
        )
        # model_copy intentionally skips validation; round-trip through the schema.
        validated = WebWorkspaceDocument.model_validate(changed.model_dump(by_alias=True))
        self.store.save(validated)
        return validated

    def upload_mesh(
        self,
        project_id: UUID | str,
        filename: str,
        source: BinaryIO,
    ) -> WebWorkspaceDocument:
        """Store a validated source plus the deterministic aligned STL target."""

        current = self.store.load(project_id)
        source_name, mesh_name = self.store.save_mesh(project_id, filename, source)
        return self._mutate(
            current,
            source_asset=source_name,
            mesh_asset=mesh_name,
        )

    def set_stock(
        self,
        project_id: UUID | str,
        values: Mapping[str, Any],
    ) -> WebWorkspaceDocument:
        """Validate and store a cylindrical or rectangular stock envelope."""

        stock_type = str(values.get("type", values.get("stock_type", ""))).lower()
        if stock_type == "cylinder":
            stock: StockDefinition = CylindricalStockConfig.model_validate(values)
        elif stock_type == "rectangle":
            stock = RectangularStockConfig.model_validate(values)
        else:
            raise WebServiceError("Stock type must be 'cylinder' or 'rectangle'.")
        return self._mutate(self.store.load(project_id), stock=stock)

    def set_tools(
        self,
        project_id: UUID | str,
        tool_numbers: Sequence[int | str],
    ) -> WebWorkspaceDocument:
        """Select tools only from the injected, validated personal library."""

        requested = [int(number) for number in tool_numbers]
        if len(requested) != len(set(requested)):
            raise LibrarySelectionError("Tool numbers must be unique.")
        available = {tool.number: tool for tool in self.tools}
        missing = [number for number in requested if number not in available]
        if missing:
            raise LibrarySelectionError(f"Unknown tool number(s): {', '.join(map(str, missing))}")
        selected = tuple(_tool_config(available[number]) for number in requested)
        return self._mutate(self.store.load(project_id), tools=selected)

    def set_settings(
        self,
        project_id: UUID | str,
        values: Mapping[str, Any],
    ) -> WebWorkspaceDocument:
        """Replace all public machining settings with validated values."""

        settings = MachiningSettings.model_validate(values)
        return self._mutate(self.store.load(project_id), settings=settings)

    def set_machine(
        self,
        project_id: UUID | str,
        machine_name: str,
    ) -> WebWorkspaceDocument:
        """Select an existing profile without permitting verification edits."""

        matches = [machine for machine in self.machines if machine.name == machine_name]
        if len(matches) != 1:
            raise LibrarySelectionError("Select a machine from the configured library.")
        return self._mutate(self.store.load(project_id), machine=matches[0])

    def add_support(
        self,
        project_id: UUID | str,
        values: Mapping[str, Any],
    ) -> WebWorkspaceDocument:
        """Add one validated support from explicit X/A surface coordinates."""

        support_type = str(values.get("type", values.get("support_type", ""))).lower()
        if support_type == "cylinder":
            support: Support = CylindricalSupport.model_validate(values)
        elif support_type == "rectangle":
            support = RectangularSupport.model_validate(values)
        else:
            raise WebServiceError("Support type must be 'cylinder' or 'rectangle'.")
        current = self.store.load(project_id)
        return self._mutate(current, supports=(*current.supports, support))

    def add_picked_support(
        self,
        project_id: UUID | str,
        *,
        x: float,
        y: float,
        z: float,
        diameter: float = 5.0,
        thickness: float = 1.0,
        transition: float = 0.5,
    ) -> WebWorkspaceDocument:
        """Convert a browser XYZ raycast into the canonical X/A support frame."""

        support_x, angle_deg, _radius = xyz_to_xar(x, y, z)
        support = CylindricalSupport(
            x=support_x,
            angle_deg=angle_deg,
            diameter=diameter,
            thickness=thickness,
            transition=transition,
        )
        current = self.store.load(project_id)
        return self._mutate(current, supports=(*current.supports, support))

    def remove_support(
        self,
        project_id: UUID | str,
        support_id: UUID | str,
    ) -> WebWorkspaceDocument:
        """Remove exactly one support by stable UUID."""

        try:
            parsed = support_id if isinstance(support_id, UUID) else UUID(support_id)
        except (TypeError, ValueError) as exc:
            raise SupportNotFoundError("Support ID must be a valid UUID.") from exc
        current = self.store.load(project_id)
        supports = tuple(support for support in current.supports if support.id != parsed)
        if len(supports) == len(current.supports):
            raise SupportNotFoundError(f"Unknown support ID: {parsed}")
        return self._mutate(current, supports=supports)


__all__ = [
    "LibrarySelectionError",
    "SupportNotFoundError",
    "WebBackendService",
    "WebServiceError",
]
