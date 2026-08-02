"""Common radial stock contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Stock(Protocol):
    """A stock centered on Y/Z and extending from X=0 to ``length``."""

    @property
    def length(self) -> float:
        """Return the longitudinal stock length in millimetres."""

        ...

    def radius_at(self, x: float, angle_deg: float) -> float:
        """Return the radial boundary in millimetres, or zero outside X."""

        ...
