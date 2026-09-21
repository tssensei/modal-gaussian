"""Identity-bound modal previews without invented video coordinates."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from modal_gaussians.scene_store import resolve_path
import tempfile
from typing import Any

from modal_gaussians.iteration_cache import atomic_json, identity, publish_directory
from modal_gaussians.motion.common.completed_modes import load_completed_modes
from modal_gaussians.rendered_design import load_rendered_modal_design
from modal_gaussians.static import load_static_scene
from modal_gaussians.motion.neural.prepared import load_prepared
from modal_gaussians.motion.neural.neural_modes import _source_identity

FORMAT = "modal_gaussians.modal_preview"


@dataclass
class ModalPreviewArtifact:
    path: Path
    manifest: dict[str, Any]
    scene: Any
    completed_modes: Any
    rendered_design: Any
    prepared: Any
    is_preview: bool = True
    coordinates: None = None


def load_preview(path: str | Path, *, validate: bool = False) -> ModalPreviewArtifact:
    root = resolve_path(path, strict=True)
    m = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if m.get("format") != FORMAT or m.get("version") not in (1, 2):
        raise ValueError("Unsupported modal preview")
    if validate and identity({k: v for k, v in m.items() if k != "preview_identity"}) != m.get("preview_identity"):
        raise ValueError("Preview identity differs")
    prepared = load_prepared(m["prepared"]) if m["version"] == 1 else None
    scene = load_static_scene(m["scene"], "cpu")
    completed = load_completed_modes(m["completed_modes"])
    if m["version"] == 2 and completed.manifest.get("version") != 17:
        raise ValueError("Multi-frequency preview requires a mode bank")
    design = load_rendered_modal_design(m["rendered_design"])
    if validate:
        _validate_bindings(m, scene, completed, design, prepared)
    return ModalPreviewArtifact(root, m, scene, completed, design, prepared)


def _validate_bindings(m, scene, completed, design, prepared):
    checks = {
        "static_scene_identity": scene.manifest["static_scene_identity"],
        "completed_modes_identity": completed.manifest["completed_modes_identity"],
        "rendered_design_identity": design.manifest["rendered_design_identity"],
    }
    if prepared is not None:
        checks["prepared_identity"] = prepared.manifest["prepared_identity"]
    if any(m.get(k) != v for k, v in checks.items()):
        raise ValueError("Preview linked source identity differs")
    if prepared is not None and _source_identity(completed.manifest) != prepared.manifest["source_identity"]:
        raise ValueError("Preview prepared source differs")
    if (completed.manifest["static_scene_identity"] != checks["static_scene_identity"]
            or design.manifest["static_scene_identity"] != checks["static_scene_identity"]
            or design.manifest["completed_modes_identity"] != checks["completed_modes_identity"]
            or m["modes"] != completed.manifest["modes"] or m["modes"] != design.manifest["modes"]
            or m["views"] != design.manifest["views"]):
        raise ValueError("Preview source domains/modes/views differ")


def build_preview(*, prepared_dir=None, scene_dir, completed_modes_dir, rendered_design_dir, output_dir,
                  prepared=None, completed=None, design=None):
    """Publish references to stage outputs without repeating validation."""
    destination = resolve_path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    prepared = prepared if prepared is not None else (load_prepared(prepared_dir) if prepared_dir is not None else None)
    completed = completed if completed is not None else load_completed_modes(completed_modes_dir)
    if prepared is None and completed.manifest.get("version") != 17:
        raise ValueError("Preview without a prepared snapshot requires a mode bank")
    design = design if design is not None else load_rendered_modal_design(rendered_design_dir)
    for artifact, path in ((prepared, prepared_dir), (completed, completed_modes_dir), (design, rendered_design_dir)):
        if artifact is None:
            continue
        if artifact.path.resolve() != resolve_path(path):
            raise ValueError("Reused preview input belongs to a different path")
    scene = load_static_scene(scene_dir, "cpu")
    m = {"format": FORMAT, "version": 1 if prepared is not None else 2,
         "scene": str(resolve_path(scene_dir)), "static_scene_identity": completed.manifest["static_scene_identity"],
         "completed_modes": str(completed.path), "completed_modes_identity": completed.manifest["completed_modes_identity"],
         "rendered_design": str(design.path), "rendered_design_identity": design.manifest["rendered_design_identity"],
         "modes": completed.manifest["modes"], "views": design.manifest["views"],
         "playback": "manual_oscillator_only", "quality_gate": {"status": "preview_candidate_unapproved"}}
    if prepared is not None:
        m.update(prepared=str(prepared.path), prepared_identity=prepared.manifest["prepared_identity"])
    if (scene.manifest["static_scene_identity"] != m["static_scene_identity"]
            or design.manifest["completed_modes_identity"] != m["completed_modes_identity"]
            or design.manifest["modes"] != m["modes"]):
        raise ValueError("Preview scene/design differs from saved modes")
    m["preview_identity"] = identity(m)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    atomic_json(temporary / "manifest.json", m)
    validated = ModalPreviewArtifact(temporary, m, scene, completed, design, prepared)
    publish_directory(temporary, destination)
    validated.path = destination
    return validated
