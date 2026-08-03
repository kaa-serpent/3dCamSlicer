from __future__ import annotations

from rotarycam.collisions.models import (
    CollisionEvent,
    CollisionKind,
    CollisionReport,
    ToolComponent,
)
from rotarycam.motion import MachinePose


def test_collision_report_exposes_first_event() -> None:
    event = CollisionEvent(
        CollisionKind.FIXTURE,
        "clamp",
        ToolComponent.HOLDER,
        0.5,
        MachinePose(1.0, 2.0, 3.0, 45.0),
        -0.1,
    )

    report = CollisionReport((event,))

    assert not report.collision_free
    assert report.first_event is event
