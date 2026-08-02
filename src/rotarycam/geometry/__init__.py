"""Public geometry API for RotaryCAM."""

from rotarycam.geometry.errors import GeometryError, MeshLoadError, RadialCompatibilityError
from rotarycam.geometry.interpolation import interpolate_radius
from rotarycam.geometry.mesh_loader import load_mesh, normalize_mesh
from rotarycam.geometry.mesh_validation import MeshValidationReport, validate_mesh
from rotarycam.geometry.models import RotaryGrid, UndercutStatus
from rotarycam.geometry.radial_sampler import (
    RadialSamplingResult,
    sample_mesh_radially,
    sample_mesh_radially_with_report,
)
from rotarycam.geometry.transforms import (
    ValidationResult,
    align_longest_axis_to_x,
    apply_rotation,
    apply_translation,
    apply_uniform_scale,
    center_mesh_on_rotary_axis,
    unwrap_angles,
    validate_mesh_inside_stock,
    xar_to_xyz,
    xyz_array_to_xar,
    xyz_to_xar,
)

__all__ = [
    "GeometryError",
    "MeshLoadError",
    "MeshValidationReport",
    "RadialCompatibilityError",
    "RadialSamplingResult",
    "RotaryGrid",
    "UndercutStatus",
    "ValidationResult",
    "align_longest_axis_to_x",
    "apply_rotation",
    "apply_translation",
    "apply_uniform_scale",
    "center_mesh_on_rotary_axis",
    "interpolate_radius",
    "load_mesh",
    "normalize_mesh",
    "sample_mesh_radially",
    "sample_mesh_radially_with_report",
    "unwrap_angles",
    "validate_mesh",
    "validate_mesh_inside_stock",
    "xar_to_xyz",
    "xyz_array_to_xar",
    "xyz_to_xar",
]
