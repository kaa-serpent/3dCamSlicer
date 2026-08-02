"""Small deterministic helpers shared by versioned JSON stores."""

from __future__ import annotations

import shutil
from pathlib import Path


def atomic_write_text(destination: Path, content: str) -> None:
    """Replace ``destination`` atomically with UTF-8 ``content``."""

    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def preserve_v1_source(source: Path) -> Path | None:
    """Create the one-time byte-for-byte ``.v1.bak`` sibling if needed."""

    source = source.resolve()
    if not source.exists():
        return None
    backup = source.with_name(source.name + ".v1.bak")
    if backup.exists():
        return backup
    temporary = backup.with_name(backup.name + ".tmp")
    try:
        shutil.copy2(source, temporary)
        temporary.replace(backup)
    finally:
        if temporary.exists():
            temporary.unlink()
    return backup
