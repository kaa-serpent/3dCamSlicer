"""Compatibility exports for project persistence models."""

from rotarycam.project.schema import (
    PROJECT_SCHEMA_VERSION,
    CylindricalStockConfig,
    Matrix4x4,
    RectangularStockConfig,
    RotaryCamProject,
    StockDefinition,
    StockType,
    ToolConfig,
    ToolType,
    identity_transform,
)

__all__ = [
    "PROJECT_SCHEMA_VERSION",
    "CylindricalStockConfig",
    "Matrix4x4",
    "RectangularStockConfig",
    "RotaryCamProject",
    "StockDefinition",
    "StockType",
    "ToolConfig",
    "ToolType",
    "identity_transform",
]
