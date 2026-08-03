"""Volumetric surface accessibility analysis."""

from rotarycam.accessibility.analyzer import (
    AccessibilityAnalyzer,
    CollisionCallback,
    analyze_accessibility,
)
from rotarycam.accessibility.digest import digest_accessibility_context
from rotarycam.accessibility.models import (
    AccessibilityReport,
    AccessibilitySettings,
    CandidatePose,
    InaccessibilityReason,
    InaccessibleRegion,
)

__all__ = [
    "AccessibilityAnalyzer",
    "AccessibilityReport",
    "AccessibilitySettings",
    "CandidatePose",
    "CollisionCallback",
    "InaccessibilityReason",
    "InaccessibleRegion",
    "analyze_accessibility",
    "digest_accessibility_context",
]
