"""Continuous swept-collision validation."""

from rotarycam.collisions.checker import SweptCollisionChecker
from rotarycam.collisions.models import (
    CollisionBudgetExceeded,
    CollisionEvent,
    CollisionKind,
    CollisionReport,
    SweptCollisionSettings,
    ToolComponent,
)
from rotarycam.collisions.obstacles import (
    CollisionObstacle,
    PrimitiveSDF,
    SignedDistanceGeometry,
    SparseVolumeSDF,
    primitive_obstacle,
    stock_obstacle,
)

__all__ = [
    "CollisionBudgetExceeded",
    "CollisionEvent",
    "CollisionKind",
    "CollisionObstacle",
    "CollisionReport",
    "PrimitiveSDF",
    "SignedDistanceGeometry",
    "SparseVolumeSDF",
    "SweptCollisionChecker",
    "SweptCollisionSettings",
    "ToolComponent",
    "primitive_obstacle",
    "stock_obstacle",
]
