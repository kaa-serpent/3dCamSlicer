"""Canonical SHA-256 fingerprints for accessibility acknowledgements."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from enum import Enum

import numpy as np
from pydantic import BaseModel

from rotarycam.accessibility.models import AccessibilitySettings
from rotarycam.config import MachineDefinition
from rotarycam.tools import ToolAssembly
from rotarycam.volumetric import SolidVolume


def _canonical(value: object) -> object:
    if isinstance(value, BaseModel):
        return _canonical(value.model_dump(mode="json", exclude_none=False))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, np.ndarray):
        return {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported digest value: {type(value).__name__}")


def digest_accessibility_context(
    target: SolidVolume,
    tools: Sequence[ToolAssembly],
    machine: MachineDefinition,
    settings: AccessibilitySettings,
    plan_fingerprint: str = "",
) -> str:
    """Fingerprint every input that invalidates an inaccessible-residue decision."""

    if not isinstance(plan_fingerprint, str):
        raise TypeError("plan_fingerprint must be a string")
    ordered_tools = sorted(tools, key=lambda assembly: assembly.tool.number)
    tool_numbers = [assembly.tool.number for assembly in ordered_tools]
    if len(tool_numbers) != len(set(tool_numbers)):
        raise ValueError("tool assembly numbers must be unique")
    occupation = np.ascontiguousarray(target.to_dense(), dtype=np.bool_)
    payload = {
        "target": {
            "lattice": dataclasses.asdict(target.lattice),
            "brick_shape": target.brick_shape,
            "occupation": occupation,
            "conservative_guard": target.conservative_guard,
        },
        "tools": ordered_tools,
        "machine": machine,
        "settings": settings,
        "plan_fingerprint": plan_fingerprint,
    }
    encoded = json.dumps(
        _canonical(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["digest_accessibility_context"]
