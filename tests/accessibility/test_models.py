from __future__ import annotations

import numpy as np
import pytest

from rotarycam.accessibility import (
    AccessibilityReport,
    CandidatePose,
    InaccessibilityReason,
    InaccessibleRegion,
)
from rotarycam.motion import MachinePose


def test_report_copies_masks_sorts_records_and_is_immutable() -> None:
    accessible = np.zeros((2, 1, 1), dtype=np.bool_)
    inaccessible = np.zeros_like(accessible)
    accessible[0, 0, 0] = True
    inaccessible[1, 0, 0] = True
    later = CandidatePose(
        (0, 0, 0),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        2,
        MachinePose(0, 0, 0, 90),
        0.1,
    )
    earlier = CandidatePose(
        (0, 0, 0),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        1,
        MachinePose(0, 0, 0, 0),
        0.1,
    )
    region = InaccessibleRegion(
        ((1, 0, 0),),
        1,
        0.2,
        (InaccessibilityReason.TRAVEL, InaccessibilityReason.CUTTER_REACH),
    )

    report = AccessibilityReport(
        (later, earlier), accessible, inaccessible, (region,), "a" * 64
    )
    accessible[:] = False

    assert report.candidates == (earlier, later)
    assert not report.complete
    assert report.residual_voxels == 1
    assert report.max_error == 0.2
    assert report.reasons == (
        InaccessibilityReason.CUTTER_REACH,
        InaccessibilityReason.TRAVEL,
    )
    assert bool(report.accessible_mask[0, 0, 0])
    with pytest.raises(ValueError, match="read-only"):
        report.inaccessible_mask[1, 0, 0] = False


def test_region_normalizes_indices_and_reasons() -> None:
    region = InaccessibleRegion(
        ((1, 0, 0), (0, 0, 0)),
        2,
        0.0,
        (InaccessibilityReason.TRAVEL, InaccessibilityReason.TRAVEL),
    )

    assert region.indices == ((0, 0, 0), (1, 0, 0))
    assert region.reasons == (InaccessibilityReason.TRAVEL,)
