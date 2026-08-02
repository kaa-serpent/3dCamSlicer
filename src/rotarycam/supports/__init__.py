"""Material-retaining support definitions and masks."""

from rotarycam.supports.cylinder import cylindrical_support_mask
from rotarycam.supports.masks import apply_supports, support_mask
from rotarycam.supports.models import (
    CylindricalSupport,
    CylindricalSurfaceRetention,
    RectangularSupport,
    RectangularSurfaceRetention,
    RetentionType,
    RetentionVolume,
    Support,
    SupportType,
    migrate_support_to_retention,
)
from rotarycam.supports.rectangle import rectangular_support_mask

__all__ = [
    "CylindricalSupport",
    "CylindricalSurfaceRetention",
    "RectangularSupport",
    "RectangularSurfaceRetention",
    "RetentionType",
    "RetentionVolume",
    "Support",
    "SupportType",
    "apply_supports",
    "cylindrical_support_mask",
    "migrate_support_to_retention",
    "rectangular_support_mask",
    "support_mask",
]
