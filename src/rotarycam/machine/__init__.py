"""Machine profiles, measured assemblies, validation, and coordinate mapping.

Configuration imports the persistence-safe assembly records, so modules which
depend on :mod:`rotarycam.config` are exposed lazily to avoid an import cycle.
"""

from __future__ import annotations

from typing import Any

from rotarycam.machine.assemblies import (
    AssemblyRole,
    AxisDynamics,
    Box,
    Cylinder,
    FrameKind,
    Frustum,
    MachineAssembly,
    MachineCapabilities,
)

__all__ = [
    "AssemblyRole",
    "AxisDynamics",
    "Box",
    "Cylinder",
    "FrameKind",
    "Frustum",
    "MachineAssembly",
    "MachineCapabilities",
    "ToolpathValidationReport",
    "XYZAKinematics",
    "inverse_transform_points",
    "makera_z1_community_profile",
    "transform_points",
    "validate_operations",
]


def __getattr__(name: str) -> Any:
    if name == "makera_z1_community_profile":
        from rotarycam.machine.profiles import makera_z1_community_profile

        return makera_z1_community_profile
    if name in {"ToolpathValidationReport", "validate_operations"}:
        from rotarycam.machine.validation import ToolpathValidationReport, validate_operations

        return {
            "ToolpathValidationReport": ToolpathValidationReport,
            "validate_operations": validate_operations,
        }[name]
    if name in {"XYZAKinematics", "inverse_transform_points", "transform_points"}:
        from rotarycam.machine.kinematics import (
            XYZAKinematics,
            inverse_transform_points,
            transform_points,
        )

        return {
            "XYZAKinematics": XYZAKinematics,
            "inverse_transform_points": inverse_transform_points,
            "transform_points": transform_points,
        }[name]
    raise AttributeError(name)
