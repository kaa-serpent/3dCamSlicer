"""Compatibility exports for project persistence models."""

from rotarycam.project.schema import (
    PROJECT_SCHEMA_VERSION,
    CylindricalStockConfig,
    Matrix4x4,
    PipelineMode,
    RectangularStockConfig,
    RotaryCamProject,
    RotaryCamProjectV1,
    StockDefinition,
    StockType,
    ToolConfig,
    ToolHolderConfig,
    ToolType,
    identity_transform,
    migrate_project_v1,
)

__all__ = [
    "PROJECT_SCHEMA_VERSION",
    "CylindricalStockConfig",
    "Matrix4x4",
    "PipelineMode",
    "RectangularStockConfig",
    "RotaryCamProject",
    "RotaryCamProjectV1",
    "StockDefinition",
    "StockType",
    "ToolConfig",
    "ToolHolderConfig",
    "ToolType",
    "identity_transform",
    "migrate_project_v1",
]
