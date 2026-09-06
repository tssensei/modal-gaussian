"""Identity-bound materialization of one complete modal Gaussian result."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from modal_gaussians import __version__
from modal_gaussians.direct_coordinates import (
    DIRECT_COORDINATES_FORMAT,
    DirectModalCoordinatesArtifact,
    load_direct_modal_coordinates,
)
from modal_gaussians.motion.common.completed_modes import CompletedModesArtifact, load_completed_modes
from modal_gaussians.physics_coordinates import (
    PHYSICS_COORDINATES_FORMAT,
    PhysicsModalCoordinatesArtifact,
    load_physics_modal_coordinates,
)
from modal_gaussians.rendered_design import (
    RenderedModalDesignArtifact,
    load_rendered_modal_design,
)
from modal_gaussians.static import ForegroundBackgroundScene, load_static_scene


MODAL_RESULT_FORMAT = "modal_gaussians.modal_result"
MODAL_RESULT_VERSION = 1
MANIFEST_FILENAME = "manifest.json"

STORAGE_CONVENTION = {
    "layout": "linked_immutable_artifacts",
    "large_arrays_copied": False,
    "source_paths": "absolute",
    "validation": "reload_and_verify_every_source_identity",
}

DEFORMATION_CONVENTION = {
    "foreground": "means(t)=means_static+sum_k real(q_view(t,k)*phi(k))",
    "background": "static",
    "phi": "completed_modes.phi[K,G_fg,3]_complex64",
    "q": "coordinates[sum(T_view),K]_complex64",
    "coordinate_index": "views[view].frame_offset+local_frame_index",
    "flow_comparison": "q(t)-q(reference_frame_index)",
    "gaussian_index": "static_scene_foreground_order",
    "units": "normalized_scene_coordinates",
}


CoordinateArtifact = DirectModalCoordinatesArtifact | PhysicsModalCoordinatesArtifact


@dataclass(frozen=True)
class ModalResultArtifact:
    """Expose one validated result and its already verified linked sources."""

    path: Path
    manifest: dict[str, Any]
    scene: ForegroundBackgroundScene
    completed_modes: CompletedModesArtifact
    coordinates: CoordinateArtifact
    rendered_design: RenderedModalDesignArtifact
    direct_coordinates: DirectModalCoordinatesArtifact | None


def _canonical_json(value: Any) -> bytes:
    """Encode one path-independent identity payload deterministically."""

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Select the immutable scientific bindings and playback convention."""

    return {
        "format": MODAL_RESULT_FORMAT,
        "version": MODAL_RESULT_VERSION,
        "storage": manifest["storage"],
        "static_scene_identity": manifest["static_scene_identity"],
        "foreground_identity": manifest["foreground_identity"],
        "background_identity": manifest["background_identity"],
        "completed_modes_identity": manifest["completed_modes_identity"],
        "rendered_design_identity": manifest["rendered_design_identity"],
        "coordinate_source": manifest["coordinate_source"],
        "modes": manifest["modes"],
        "views": manifest["views"],
        "counts": manifest["counts"],
        "deformation": manifest["deformation"],
        "quality_gate": manifest["quality_gate"],
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    """Read one JSON object from an artifact directory."""

    manifest_path = path / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Artifact manifest does not exist: {manifest_path}")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Artifact manifest must be a JSON object: {manifest_path}")
    return value


def _source_path(sources: Mapping[str, Any], name: str) -> Path:
    """Resolve one required absolute linked-source path from a result manifest."""

    record = sources.get(name)
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise ValueError(f"Modal result source {name!r} is invalid")
    path = Path(record["path"]).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise FileNotFoundError(f"Modal result source is not a directory: {path}")
    return path


def _load_coordinate_artifact(path: Path) -> tuple[str, CoordinateArtifact]:
    """Auto-detect and strictly load direct or physics modal coordinates."""

    artifact_format = _read_manifest(path).get("format")
    if artifact_format == DIRECT_COORDINATES_FORMAT:
        return "direct", load_direct_modal_coordinates(path)
    if artifact_format == PHYSICS_COORDINATES_FORMAT:
        return "physics", load_physics_modal_coordinates(path)
    raise ValueError(
        "Result coordinates must be a direct- or physics-coordinate artifact"
    )


def _require_equal(name: str, actual: Any, expected: Any) -> None:
    """Raise one focused error when an identity or ordered contract differs."""

    if actual != expected:
        raise ValueError(f"Modal result {name} differs across linked artifacts")


def _coordinate_view_records(
    coordinate_views: Sequence[Mapping[str, Any]],
    design_views: Sequence[Mapping[str, Any]],
    completed_views: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate the view chain and retain only runtime frame/camera metadata."""

    if not (
        len(coordinate_views) == len(design_views) == len(completed_views)
    ):
        raise ValueError("Modal result linked artifacts have different view counts")
    records: list[dict[str, Any]] = []
    frame_offset = 0
    for index, (coordinate, design, completed) in enumerate(
        zip(coordinate_views, design_views, completed_views)
    ):
        for field in ("index", "label", "flow_identity", "shape_hw"):
            _require_equal(
                f"view {index} {field}", design.get(field), completed.get(field)
            )
            _require_equal(
                f"coordinate view {index} {field}",
                coordinate.get(field),
                design.get(field),
            )
        for coordinate_name, design_name in (
            ("frame_count", "frame_count"),
            ("fps_hz", "fps_hz"),
            ("reference_frame_name", "flow_reference_frame_name"),
            ("reference_frame_index", "flow_reference_frame_index"),
            ("sample_count", "sample_count"),
        ):
            _require_equal(
                f"coordinate view {index} {coordinate_name}",
                coordinate.get(coordinate_name),
                design.get(design_name),
            )
        _require_equal(
            f"coordinate view {index} frame offset",
            coordinate.get("frame_offset"),
            frame_offset,
        )
        frame_count = int(coordinate["frame_count"])
        record = {
            "index": index,
            "label": coordinate["label"],
            "camera_name": design["camera_name"],
            "camera_identity": design["camera_identity"],
            "flow_identity": coordinate["flow_identity"],
            "shape_hw": list(coordinate["shape_hw"]),
            "fps_hz": float(coordinate["fps_hz"]),
            "frame_offset": frame_offset,
            "frame_count": frame_count,
            "frame_names": list(coordinate["frame_names"]),
            "reference_frame_name": coordinate["reference_frame_name"],
            "reference_frame_index": int(coordinate["reference_frame_index"]),
        }
        records.append(record)
        frame_offset += frame_count
    return records


def _load_sources(
    *,
    scene_dir: str | Path,
    completed_modes_dir: str | Path,
    coordinates_dir: str | Path,
) -> tuple[
    Path,
    ForegroundBackgroundScene,
    CompletedModesArtifact,
    str,
    CoordinateArtifact,
    RenderedModalDesignArtifact,
    DirectModalCoordinatesArtifact | None,
    list[dict[str, Any]],
]:
    """Load and cross-check the complete static/mode/design/coordinate chain."""

    scene_path = Path(scene_dir).expanduser().resolve(strict=True)
    completed_path = Path(completed_modes_dir).expanduser().resolve(strict=True)
    coordinate_path = Path(coordinates_dir).expanduser().resolve(strict=True)
    scene = load_static_scene(scene_path, "cpu")
    completed = load_completed_modes(completed_path)
    coordinate_kind, coordinates = _load_coordinate_artifact(coordinate_path)
    scene_manifest = scene.manifest
    if scene_manifest is None:
        raise ValueError("Static scene has no manifest")

    _require_equal(
        "completed static scene identity",
        completed.manifest.get("static_scene_identity"),
        scene_manifest["static_scene_identity"],
    )
    _require_equal(
        "completed foreground identity",
        completed.manifest.get("foreground_identity"),
        scene_manifest["foreground_identity"],
    )
    _require_equal(
        "coordinate completed-mode identity",
        coordinates.manifest.get("completed_modes_identity"),
        completed.manifest["completed_modes_identity"],
    )
    _require_equal(
        "coordinate mode order",
        coordinates.manifest.get("modes"),
        completed.manifest["modes"],
    )

    design_source = coordinates.manifest.get("rendered_design")
    if not isinstance(design_source, str) or not design_source:
        raise ValueError("Coordinate artifact does not name its rendered design")
    design = load_rendered_modal_design(design_source)
    _require_equal(
        "rendered-design identity",
        coordinates.manifest.get("rendered_design_identity"),
        design.manifest["rendered_design_identity"],
    )
    for name, expected in (
        ("static_scene_identity", scene_manifest["static_scene_identity"]),
        ("foreground_identity", scene_manifest["foreground_identity"]),
        ("completed_modes_identity", completed.manifest["completed_modes_identity"]),
        ("modes", completed.manifest["modes"]),
    ):
        _require_equal(f"rendered-design {name}", design.manifest.get(name), expected)
    _require_equal(
        "rendered-design foreground count",
        design.manifest["counts"].get("foreground_gaussians"),
        scene.foreground.count,
    )

    direct: DirectModalCoordinatesArtifact | None = None
    if coordinate_kind == "physics":
        direct_source = coordinates.manifest.get("direct_coordinates")
        if not isinstance(direct_source, str) or not direct_source:
            raise ValueError("Physics coordinates do not name their direct source")
        direct = load_direct_modal_coordinates(direct_source)
        _require_equal(
            "physics direct-coordinate identity",
            coordinates.manifest.get("direct_coordinates_identity"),
            direct.manifest["direct_coordinates_identity"],
        )
        for name in ("rendered_design_identity", "completed_modes_identity", "modes"):
            _require_equal(
                f"physics/direct {name}",
                coordinates.manifest.get(name),
                direct.manifest.get(name),
            )
        direct_views = direct.manifest["views"]
        coordinate_views = coordinates.manifest["views"]
        if len(coordinate_views) != len(direct_views):
            raise ValueError("Physics/direct coordinate view counts differ")
        runtime_fields = (
            "index",
            "label",
            "flow_identity",
            "shape_hw",
            "fps_hz",
            "frame_offset",
            "frame_count",
            "frame_names",
            "reference_frame_name",
            "reference_frame_index",
            "sample_count",
        )
        for index, (physics_view, direct_view) in enumerate(
            zip(coordinate_views, direct_views)
        ):
            for name in runtime_fields:
                _require_equal(
                    f"physics/direct view {index} {name}",
                    physics_view.get(name),
                    direct_view.get(name),
                )

    views = _coordinate_view_records(
        coordinates.manifest["views"],
        design.manifest["views"],
        completed.manifest["views"],
    )
    expected_shape = (
        sum(record["frame_count"] for record in views),
        len(completed.manifest["modes"]),
    )
    _require_equal("coordinate array shape", coordinates.coordinates.shape, expected_shape)
    phi = completed.arrays["phi"]
    _require_equal(
        "completed phi shape",
        phi.shape,
        (len(completed.manifest["modes"]), scene.foreground.count, 3),
    )
    return (
        scene_path,
        scene,
        completed,
        coordinate_kind,
        coordinates,
        design,
        direct,
        views,
    )


def _source_record(path: Path, identity_name: str, identity: str) -> dict[str, str]:
    """Describe one linked artifact without putting its path in result identity."""

    return {
        "path": str(path.resolve()),
        "identity_name": identity_name,
        "identity": identity,
    }


def load_modal_result(path: str | Path) -> ModalResultArtifact:
    """Load a materialized result and revalidate every linked source artifact."""

    root = Path(path).expanduser().resolve(strict=True)
    manifest = _read_manifest(root)
    if manifest.get("format") != MODAL_RESULT_FORMAT:
        raise ValueError("Unsupported modal-result format")
    if manifest.get("version") != MODAL_RESULT_VERSION:
        raise ValueError("Unsupported modal-result version")
    if manifest.get("storage") != STORAGE_CONVENTION:
        raise ValueError("Unsupported modal-result storage convention")
    if manifest.get("deformation") != DEFORMATION_CONVENTION:
        raise ValueError("Unsupported modal-result deformation convention")
    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("Modal result sources are invalid")
    scene_path = _source_path(sources, "static_scene")
    completed_path = _source_path(sources, "completed_modes")
    coordinate_path = _source_path(sources, "coordinates")
    rendered_design_path = _source_path(sources, "rendered_design")
    (
        _,
        scene,
        completed,
        coordinate_kind,
        coordinates,
        design,
        direct,
        views,
    ) = _load_sources(
        scene_dir=scene_path,
        completed_modes_dir=completed_path,
        coordinates_dir=coordinate_path,
    )
    scene_manifest = scene.manifest
    if scene_manifest is None:
        raise ValueError("Static scene has no manifest")
    _require_equal(
        "rendered-design path",
        design.path.resolve(strict=True),
        rendered_design_path,
    )
    coordinate_identity_name = (
        "direct_coordinates_identity"
        if coordinate_kind == "direct"
        else "physics_coordinates_identity"
    )
    coordinate_identity = coordinates.manifest[coordinate_identity_name]
    expected_coordinate_source = {
        "kind": coordinate_kind,
        "format": coordinates.manifest["format"],
        "identity_name": coordinate_identity_name,
        "identity": coordinate_identity,
        "direct_coordinates_identity": (
            direct.manifest["direct_coordinates_identity"]
            if direct is not None
            else None
        ),
    }
    expected_counts = {
        "foreground_gaussians": scene.foreground.count,
        "background_gaussians": scene.background.count,
        "modes": len(completed.manifest["modes"]),
        "views": len(views),
        "frames": int(coordinates.coordinates.shape[0]),
    }
    expected_quality_gate = {
        "required": True,
        "status": "modal_result_candidate_unapproved",
        "inherited_from": coordinates.manifest["quality_gate"]["status"],
    }
    expected_fields = {
        "static_scene_identity": scene_manifest["static_scene_identity"],
        "foreground_identity": scene_manifest["foreground_identity"],
        "background_identity": scene_manifest["background_identity"],
        "completed_modes_identity": completed.manifest["completed_modes_identity"],
        "rendered_design_identity": design.manifest["rendered_design_identity"],
        "coordinate_source": expected_coordinate_source,
        "modes": completed.manifest["modes"],
        "views": views,
        "counts": expected_counts,
        "quality_gate": expected_quality_gate,
    }
    for name, expected in expected_fields.items():
        _require_equal(name, manifest.get(name), expected)
    source_expectations = {
        "static_scene": (
            "static_scene_identity",
            scene_manifest["static_scene_identity"],
        ),
        "completed_modes": (
            "completed_modes_identity",
            completed.manifest["completed_modes_identity"],
        ),
        "rendered_design": (
            "rendered_design_identity",
            design.manifest["rendered_design_identity"],
        ),
        "coordinates": (coordinate_identity_name, coordinate_identity),
    }
    for name, (identity_name, identity) in source_expectations.items():
        record = sources.get(name)
        if (
            not isinstance(record, dict)
            or record.get("identity_name") != identity_name
            or record.get("identity") != identity
        ):
            raise ValueError(f"Modal result linked {name} identity differs")
    expected_identity = hashlib.sha256(
        _canonical_json(_identity_payload(manifest))
    ).hexdigest()
    if manifest.get("modal_result_identity") != expected_identity:
        raise ValueError("Modal-result identity differs from its contents")
    return ModalResultArtifact(
        root,
        manifest,
        scene,
        completed,
        coordinates,
        design,
        direct,
    )


def materialize_modal_result(
    *,
    scene_dir: str | Path,
    completed_modes_dir: str | Path,
    coordinates_dir: str | Path,
    output_dir: str | Path,
    command: Sequence[str] = (),
) -> ModalResultArtifact:
    """Atomically publish a lightweight result binding without copying tensors."""

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Modal-result output already exists: {destination}")
    (
        scene_path,
        scene,
        completed,
        coordinate_kind,
        coordinates,
        design,
        direct,
        views,
    ) = _load_sources(
        scene_dir=scene_dir,
        completed_modes_dir=completed_modes_dir,
        coordinates_dir=coordinates_dir,
    )
    scene_manifest = scene.manifest
    if scene_manifest is None:
        raise ValueError("Static scene has no manifest")
    coordinate_identity_name = (
        "direct_coordinates_identity"
        if coordinate_kind == "direct"
        else "physics_coordinates_identity"
    )
    coordinate_identity = coordinates.manifest[coordinate_identity_name]
    sources = {
        "static_scene": _source_record(
            scene_path,
            "static_scene_identity",
            scene_manifest["static_scene_identity"],
        ),
        "completed_modes": _source_record(
            completed.path,
            "completed_modes_identity",
            completed.manifest["completed_modes_identity"],
        ),
        "rendered_design": _source_record(
            design.path,
            "rendered_design_identity",
            design.manifest["rendered_design_identity"],
        ),
        "coordinates": _source_record(
            coordinates.path, coordinate_identity_name, coordinate_identity
        ),
    }
    coordinate_source = {
        "kind": coordinate_kind,
        "format": coordinates.manifest["format"],
        "identity_name": coordinate_identity_name,
        "identity": coordinate_identity,
        "direct_coordinates_identity": (
            direct.manifest["direct_coordinates_identity"]
            if direct is not None
            else None
        ),
    }
    manifest = {
        "format": MODAL_RESULT_FORMAT,
        "version": MODAL_RESULT_VERSION,
        "producer": {
            "project_version": __version__,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "command": list(command),
        },
        "storage": dict(STORAGE_CONVENTION),
        "sources": sources,
        "static_scene_identity": scene_manifest["static_scene_identity"],
        "foreground_identity": scene_manifest["foreground_identity"],
        "background_identity": scene_manifest["background_identity"],
        "completed_modes_identity": completed.manifest["completed_modes_identity"],
        "rendered_design_identity": design.manifest["rendered_design_identity"],
        "coordinate_source": coordinate_source,
        "modes": [dict(mode) for mode in completed.manifest["modes"]],
        "views": views,
        "counts": {
            "foreground_gaussians": scene.foreground.count,
            "background_gaussians": scene.background.count,
            "modes": len(completed.manifest["modes"]),
            "views": len(views),
            "frames": int(coordinates.coordinates.shape[0]),
        },
        "deformation": dict(DEFORMATION_CONVENTION),
        "quality_gate": {
            "required": True,
            "status": "modal_result_candidate_unapproved",
            "inherited_from": coordinates.manifest["quality_gate"]["status"],
        },
    }
    manifest["modal_result_identity"] = hashlib.sha256(
        _canonical_json(_identity_payload(manifest))
    ).hexdigest()

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
    )
    try:
        (temporary / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        load_modal_result(temporary)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Modal-result output already exists: {destination}")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_modal_result(destination)


__all__ = [
    "ModalResultArtifact",
    "load_modal_result",
    "materialize_modal_result",
]
