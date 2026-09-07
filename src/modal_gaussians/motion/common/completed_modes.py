"""Stable full-foreground motion interface, independent of the fitting method.

Method-specific validators are imported lazily. Historical format strings and
strict version/identity validation remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import numpy as np

COMPLETED_MODES_FORMAT = "modal_gaussians.completed_modes"
COMPLETED_MODES_FILENAME = "completed_modes.npz"
SEQUENTIAL_COMPLETED_MODES_VERSION = 2
COMPLETED_MODES_VERSION = 3  # Historical shared-basis version, not the latest format.
MOTION_BASIS_COMPLETION_METHOD = "shared_motion_basis_blend"

@dataclass(frozen=True)
class CompletedModesArtifact:
    """Represent one validated full-foreground completed modal-field artifact."""

    path: Path
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]


def load_completed_modes(path: str | Path) -> CompletedModesArtifact:
    """Load each supported completed-mode method through its strict validator."""

    root = Path(path).expanduser().resolve(strict=True)
    manifest_path = root / "manifest.json"
    arrays_path = root / COMPLETED_MODES_FILENAME
    if not manifest_path.is_file() or not arrays_path.is_file():
        raise FileNotFoundError(f"Incomplete completed-mode artifact: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != COMPLETED_MODES_FORMAT:
        raise ValueError("Unsupported completed-mode format")
    version = manifest.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("Completed-mode version must be an integer")
    if version == 9:
        from modal_gaussians.motion.neural.fragment_propagation import load_fragment_modes

        derived = load_fragment_modes(root)
        return CompletedModesArtifact(path=derived.path, manifest=derived.manifest, arrays=derived.arrays)
    if version in (13, 15):
        from modal_gaussians.motion.neural.observation_refinement import load_refined_modes
        refined = load_refined_modes(root)
        return CompletedModesArtifact(refined.path, refined.manifest, refined.arrays)
    if version in (8, 10, 11, 12, 14, 16):
        method = {8: "neural_complex_displacement_field", 10: "neural_field_with_training_fragment_fill",
                  11: "neural_field_with_surface_attachments", 12: "neural_field_with_pointwise_displacement_fill",
                  14: "neural_field_with_guarded_neighbor_residuals",
                  16: "neural_component_field_with_stable_donors"}[version]
        if manifest.get("completion_method") != method:
            raise ValueError("Completed-mode neural method is unsupported")
        from modal_gaussians.motion.neural.neural_modes import load_neural_completed_modes

        neural_artifact = load_neural_completed_modes(root)
        return CompletedModesArtifact(
            path=neural_artifact.path,
            manifest=neural_artifact.manifest,
            arrays=neural_artifact.arrays,
        )
    if version in (5, 7):
        if manifest.get("completion_method") != "fixed_observation_green_basis_refinement":
            raise ValueError(f"Completed-mode v{version} method is unsupported")
        from modal_gaussians.motion.rigid.motion_basis_green import load_green_refined_motion_basis_modes

        green_artifact = load_green_refined_motion_basis_modes(root)
        return CompletedModesArtifact(
            path=green_artifact.path,
            manifest=green_artifact.manifest,
            arrays=green_artifact.arrays,
        )
    if version in (4, 6):
        if manifest.get("completion_method") != "per_frequency_motion_basis_blend":
            raise ValueError(f"Completed-mode v{version} method is unsupported")
        from modal_gaussians.motion.rigid.motion_basis_frequency import load_frequency_motion_basis_modes

        frequency_artifact = load_frequency_motion_basis_modes(root)
        return CompletedModesArtifact(
            path=frequency_artifact.path,
            manifest=frequency_artifact.manifest,
            arrays=frequency_artifact.arrays,
        )
    if version == COMPLETED_MODES_VERSION:
        if manifest.get("completion_method") != MOTION_BASIS_COMPLETION_METHOD:
            raise ValueError("Completed-mode v3 method is unsupported")
        # Keep schema validation in the method-specific module while exposing
        # one stable artifact interface to every downstream stage.
        from modal_gaussians.motion.rigid.motion_basis import load_motion_basis_modes

        basis_artifact = load_motion_basis_modes(root)
        return CompletedModesArtifact(
            path=basis_artifact.path,
            manifest=basis_artifact.manifest,
            arrays=basis_artifact.arrays,
        )
    if version not in (1, SEQUENTIAL_COMPLETED_MODES_VERSION):
        raise ValueError("Unsupported completed-mode version")
    from modal_gaussians.motion.rigid.motion_fill import load_sequential_completed_modes

    return load_sequential_completed_modes(root)
