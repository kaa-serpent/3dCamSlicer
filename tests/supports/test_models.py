from uuid import UUID

import pytest
from pydantic import ValidationError

from rotarycam.supports import CylindricalSupport, RectangularSupport, SupportType


def test_support_normalizes_angle_and_generates_id() -> None:
    support = RectangularSupport(
        x=2.0,
        angle_deg=-1.0,
        length_x=4.0,
        width_surface=3.0,
        thickness=1.0,
        transition=0.5,
    )

    assert isinstance(support.id, UUID)
    assert support.angle_deg == pytest.approx(359.0)
    assert support.support_type is SupportType.RECTANGLE


def test_support_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValidationError):
        CylindricalSupport(
            x=0.0,
            angle_deg=0.0,
            diameter=0.0,
            thickness=1.0,
            transition=0.0,
        )


def test_support_json_uses_type_discriminator() -> None:
    support = CylindricalSupport(
        x=1.0,
        angle_deg=45.0,
        diameter=2.0,
        thickness=0.5,
        transition=0.25,
    )

    assert support.model_dump(by_alias=True)["type"] == "cylinder"
