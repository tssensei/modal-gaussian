"""Current component fields and fixed-mode banks."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import numpy as np
from modal_gaussians.common.scene_store import resolve_path
COMPLETED_MODES_FORMAT = "modal_gaussians.completed_modes"
@dataclass(frozen=True)
class CompletedModesArtifact:
    """Represent one saved full-foreground completed modal-field artifact."""

    path: Path
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]
    rotation: np.ndarray | None = None
    control_displacement: np.ndarray | None = None

def load_completed_modes(path):
    root = resolve_path(path, strict=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != COMPLETED_MODES_FORMAT:
        raise ValueError("Expected completed modes")
    if manifest.get("version") == 17:
        from modal_gaussians.coordinates.preparation import load_mode_bank
        return load_mode_bank(root)
    if manifest.get("version") != 18:
        raise ValueError("Unsupported mode version; rebuild with the mainline pipeline")
    from modal_gaussians.motion.artifacts import load_neural_completed_modes
    value = load_neural_completed_modes(root)
    return CompletedModesArtifact(root, value.manifest, value.arrays, value.rotation, value.control_displacement)
