"""Geometry-domain exceptions."""

from rotarycam.errors import RotaryCamError


class GeometryError(RotaryCamError, ValueError):
    """Base exception for invalid geometry input."""


class MeshLoadError(GeometryError):
    """Raised when a supported mesh cannot be loaded or normalized."""


class RadialCompatibilityError(GeometryError):
    """Raised when a mesh cannot be certified as a single radial solid."""
