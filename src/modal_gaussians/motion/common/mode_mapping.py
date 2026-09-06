"""Exact mapping between exported modes and original frequency slots."""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import numpy as np

def resolve_source_mode_slots(
    modes: Sequence[Mapping[str, Any]], source_modes: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    """Validate local-to-original slots without sorting or guessing by frequency."""

    if not modes or not source_modes:
        raise ValueError("Mode mapping requires non-empty local and source modes")
    mapped = any("source_mode_slot" in mode for mode in modes)
    if not mapped:
        if list(modes) != list(source_modes):
            raise ValueError("Unmapped modes must exactly match the complete source prefix")
        return np.arange(len(modes), dtype=np.int64)
    slots: list[int] = []
    for local_slot, mode in enumerate(modes):
        source_slot = mode.get("source_mode_slot")
        if (
            isinstance(source_slot, bool)
            or not isinstance(source_slot, int)
            or not 0 <= source_slot < len(source_modes)
            or (slots and source_slot <= slots[-1])
        ):
            raise ValueError("Source mode slots must be unique, increasing and in range")
        parent = source_modes[source_slot]
        if parent.get("mode_slot") != source_slot or dict(mode) != {
            **parent, "mode_slot": local_slot, "source_mode_slot": source_slot
        }:
            raise ValueError("Mode mapping differs from the original candidate/frequency")
        slots.append(source_slot)
    return np.asarray(slots, dtype=np.int64)
