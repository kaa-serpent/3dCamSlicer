"""Material-retaining support definitions and masks."""

from rotarycam.supports.cylinder import cylindrical_support_mask
from rotarycam.supports.masks import apply_supports, support_mask
from rotarycam.supports.models import (
    CylindricalSupport,
    RectangularSupport,
    Support,
    SupportType,
)
from rotarycam.supports.rectangle import rectangular_support_mask

__all__ = [
    "CylindricalSupport",
    "RectangularSupport",
    "Support",
    "SupportType",
    "apply_supports",
    "cylindrical_support_mask",
    "rectangular_support_mask",
    "support_mask",
]
