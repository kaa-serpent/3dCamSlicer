"""Domain exceptions shared by RotaryCAM modules."""


class RotaryCamError(Exception):
    """Base class for expected RotaryCAM domain errors."""


class MeshError(RotaryCamError):
    """Base class for mesh import and validation failures."""


class MeshLoadError(MeshError):
    """Raised when a supported mesh file cannot be loaded."""


class UnsupportedMeshFormatError(MeshLoadError):
    """Raised when the input format is not STL or OBJ."""


class EmptyMeshError(MeshLoadError):
    """Raised when loading or normalization yields an empty mesh."""


class InvalidMeshError(MeshError):
    """Raised when a mesh violates a critical geometric invariant."""


class ToolpathValidationError(RotaryCamError):
    """Raised when a toolpath fails a critical machine-safety check."""


class UnverifiedMachineProfileError(RotaryCamError):
    """Raised when real G-code export is requested for an unverified profile."""
