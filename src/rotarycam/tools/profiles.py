"""Conservative tool-interface envelopes supplied as editable UI defaults."""

from rotarycam.tools.models import ToolHolder

MAKERA_Z1_QUICK_CHANGE_ENVELOPE = ToolHolder(diameter=16.0, length=23.0)
"""Measured spindle-nose/quick-change envelope, not a complete tool assembly."""

__all__ = ["MAKERA_Z1_QUICK_CHANGE_ENVELOPE"]
