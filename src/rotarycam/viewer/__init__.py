"""Optional desktop visualization package.

Only dependency-free state types are imported here. Qt/PyVista adapters live in
their explicit modules so importing the RotaryCAM core never requires UI extras.
"""

from rotarycam.viewer.state import ProjectUiState, SceneLayer, SupportPick

__all__ = ["ProjectUiState", "SceneLayer", "SupportPick"]
