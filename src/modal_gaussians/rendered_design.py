"""Full-foreground rasterized modal projection design for temporal coordinates."""

from __future__ import annotations

from modal_gaussians.motion.common.projection import (
    RenderedDesignConfig, projection_jacobian as _projection_jacobian,
    candidate_pixels as _candidate_pixels, sample_feature_render as _sample_feature_render,
)

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from modal_gaussians.progress import Progress
import torch

from modal_gaussians.numpy_io import save_named_arrays

from modal_gaussians import __version__
from modal_gaussians.flow.artifact import (
    FlowAnalysisArtifact,
    flow_artifact_identity,
    load_flow_analysis_artifact,
)
from modal_gaussians.motion.common.completed_modes import (
    CompletedModesArtifact,
    load_completed_modes,
)
from modal_gaussians.static import (
    Camera,
    ForegroundBackgroundScene,
    cameras_from_scene_manifest,
    load_static_scene,
)


RENDERED_DESIGN_FORMAT = "modal_gaussians.rendered_modal_design"
RENDERED_DESIGN_VERSION = 1
DESIGN_FILENAME = "design.npy"
SAMPLES_FILENAME = "samples.npz"
DESIGN_DTYPE = np.dtype(np.float32)
FINITE_BLOCK_SAMPLES = 65_536
PACKING_MAX_ABS_TOLERANCE = 1.0e-3
PACKING_RELATIVE_L2_TOLERANCE = 1.0e-3

SAMPLE_DTYPES = {
    "view_shapes_hw": np.dtype(np.int64),
    "view_sample_offsets": np.dtype(np.int64),
    "sample_view_index": np.dtype(np.int64),
    "sample_pixels_xy": np.dtype(np.int64),
    "sample_foreground_alpha": np.dtype(np.float32),
}

DESIGN_CONVENTION = {
    "source_field": "completed_complex_3d_foreground_gaussian_displacement",
    "projection": "pinhole_pixel_jacobian_at_static_gaussian_mean",
    "rasterization": "single_full_foreground_gsplat_3dgs_depth_order",
    "background_features": "zero",
    "normalization": "divide_by_rendered_foreground_alpha",
    "packing": "design[p,:,2k]=real(J_phi); design[p,:,2k+1]=-imag(J_phi)",
    "coordinate_action": "real(q_k * J_phi_k)",
    "background_gaussians": "excluded",
    "unresolved_modes": "included_as_explicit_zero_phi",
}

DISTORTED_DESIGN_CONVENTION = {
    **DESIGN_CONVENTION,
    "projection": "simple_radial_pixel_jacobian_at_static_gaussian_mean",
    "rasterization": "full_foreground_gsplat_with_simple_radial_image_warp_v1",
}


@dataclass(frozen=True)
class RenderedDesignViewInput:
    """Bind one ordered completed-mode view to its exact flow artifact."""

    label: str
    flow_artifact: Path


@dataclass(frozen=True)
class RenderedModalDesignArtifact:
    """Represent one validated mmap-backed rendered modal design."""

    path: Path
    manifest: dict[str, Any]
    design: np.ndarray
    samples: dict[str, np.ndarray]


def _canonical_json(value: Any) -> bytes:
    """Encode one path-independent identity payload deterministically."""

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    """Hash one artifact file in bounded chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arrays_identity(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash sample-array names, dtypes, shapes, and values canonically."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Select scientific inputs, conventions, diagnostics, and output hashes."""

    return {
        "format": RENDERED_DESIGN_FORMAT,
        "version": manifest["version"],
        "static_scene_identity": manifest["static_scene_identity"],
        "foreground_identity": manifest["foreground_identity"],
        "completed_modes_identity": manifest["completed_modes_identity"],
        "modes": manifest["modes"],
        "views": manifest["views"],
        "settings": manifest["settings"],
        "convention": manifest["convention"],
        "quality_gate": manifest["quality_gate"],
        "counts": manifest["counts"],
        "design": manifest["design"],
        "samples": {
            "arrays": manifest["samples"]["arrays"],
            "arrays_identity": manifest["samples"]["arrays_identity"],
        },
    }


def _validate_modes(modes: Any) -> None:
    """Validate the ordered greedy mode prefix copied from motion completion."""

    if not isinstance(modes, list) or not modes:
        raise ValueError("Rendered design must contain ordered modes")
    candidates: list[int] = []
    for expected_slot, mode in enumerate(modes):
        if not isinstance(mode, dict) or mode.get("mode_slot") != expected_slot:
            raise ValueError("Rendered-design mode slots must be contiguous")
        candidate = mode.get("candidate_index")
        frequency = float(mode.get("frequency_hz", np.nan))
        if (
            isinstance(candidate, bool)
            or not isinstance(candidate, int)
            or candidate < 0
            or not math.isfinite(frequency)
            or frequency <= 0.0
        ):
            raise ValueError("Rendered-design mode metadata is invalid")
        candidates.append(candidate)
    if len(set(candidates)) != len(candidates):
        raise ValueError("Rendered-design candidate indices must be unique")


def _validate_samples(
    samples: Mapping[str, np.ndarray],
    *,
    view_count: int,
    sample_count: int,
    config: RenderedDesignConfig,
) -> None:
    """Validate view partitions, stride-grid pixels, and sampled alpha values."""

    if set(samples) != set(SAMPLE_DTYPES):
        raise ValueError("Rendered-design sample fields are invalid")
    for name, dtype in SAMPLE_DTYPES.items():
        if samples[name].dtype != dtype:
            raise ValueError(f"Rendered-design {name} must be {dtype.name}")
    shapes = samples["view_shapes_hw"]
    offsets = samples["view_sample_offsets"]
    sample_views = samples["sample_view_index"]
    pixels = samples["sample_pixels_xy"]
    alpha = samples["sample_foreground_alpha"]
    if shapes.shape != (view_count, 2) or np.any(shapes < 3):
        raise ValueError("Rendered-design view shapes are invalid")
    if (
        offsets.shape != (view_count + 1,)
        or offsets[0] != 0
        or offsets[-1] != sample_count
        or np.any(np.diff(offsets) <= 0)
    ):
        raise ValueError("Rendered-design view sample offsets are invalid")
    if sample_views.shape != (sample_count,):
        raise ValueError("Rendered-design sample_view_index shape is invalid")
    if pixels.shape != (sample_count, 2):
        raise ValueError("Rendered-design sample_pixels_xy shape is invalid")
    if alpha.shape != (sample_count,) or not np.isfinite(alpha).all():
        raise ValueError("Rendered-design sampled alpha is invalid")
    if np.any(alpha < config.alpha_minimum - 1.0e-6) or np.any(
        alpha > 1.0 + 1.0e-5
    ):
        raise ValueError("Rendered-design sampled alpha leaves its declared range")
    for view_index in range(view_count):
        lower, upper = int(offsets[view_index]), int(offsets[view_index + 1])
        if not np.all(sample_views[lower:upper] == view_index):
            raise ValueError("Rendered-design samples are not contiguous by view")
        height, width = (int(value) for value in shapes[view_index])
        view_pixels = pixels[lower:upper]
        x, y = view_pixels[:, 0], view_pixels[:, 1]
        if (
            np.any(x < 1)
            or np.any(x >= width - 1)
            or np.any(y < 1)
            or np.any(y >= height - 1)
            or np.any((x - 1) % config.pixel_sample_stride)
            or np.any((y - 1) % config.pixel_sample_stride)
            or np.unique(view_pixels, axis=0).shape[0] != len(view_pixels)
        ):
            raise ValueError("Rendered-design pixels violate the declared stride grid")


def _validate_finite_design(design: np.ndarray) -> None:
    """Scan the mmap-backed design without copying the complete array."""

    for lower in range(0, len(design), FINITE_BLOCK_SAMPLES):
        if not np.isfinite(design[lower : lower + FINITE_BLOCK_SAMPLES]).all():
            raise ValueError("Rendered modal design contains NaN or Inf")


def load_rendered_modal_design(path: str | Path) -> RenderedModalDesignArtifact:
    """Load and strictly validate one rendered modal design directory."""

    root = Path(path).expanduser().resolve(strict=True)
    manifest_path = root / "manifest.json"
    design_path = root / DESIGN_FILENAME
    samples_path = root / SAMPLES_FILENAME
    if not manifest_path.is_file() or not design_path.is_file() or not samples_path.is_file():
        raise FileNotFoundError(f"Incomplete rendered modal design: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != RENDERED_DESIGN_FORMAT:
        raise ValueError("Unsupported rendered-design format")
    if manifest.get("version") not in (RENDERED_DESIGN_VERSION, 2):
        raise ValueError("Unsupported rendered-design version")
    expected_convention = DISTORTED_DESIGN_CONVENTION if manifest["version"] == 2 else DESIGN_CONVENTION
    if manifest.get("convention") != expected_convention:
        raise ValueError("Rendered-design convention is unsupported")
    if manifest.get("quality_gate") != {
        "required": True,
        "status": "rendered_design_candidate_unapproved",
        "inherited_from": "completion_candidate_unapproved",
    }:
        raise ValueError("Rendered design must inherit the completion quality gate")
    for name in (
        "static_scene_identity",
        "foreground_identity",
        "completed_modes_identity",
    ):
        if not isinstance(manifest.get(name), str) or not manifest[name]:
            raise ValueError(f"Rendered-design {name} is invalid")
    _validate_modes(manifest.get("modes"))
    settings = manifest.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("Rendered-design settings are invalid")
    config = RenderedDesignConfig(
        pixel_sample_stride=int(settings["pixel_sample_stride"]),
        alpha_minimum=float(settings["alpha_minimum"]),
        mask_erosion_iterations=int(settings["mask_erosion_iterations"]),
        modes_per_batch=int(settings["modes_per_batch"]),
    )
    config.validate()
    if settings != config.to_dict():
        raise ValueError("Rendered-design settings contain unsupported fields")
    counts = manifest.get("counts")
    views = manifest.get("views")
    if not isinstance(counts, dict) or not isinstance(views, list) or not views:
        raise ValueError("Rendered-design counts or views are invalid")
    view_count = int(counts.get("views", -1))
    mode_count = int(counts.get("modes", -1))
    sample_count = int(counts.get("samples", -1))
    if view_count != len(views) or mode_count != len(manifest["modes"]) or sample_count <= 0:
        raise ValueError("Rendered-design counts disagree with metadata")
    labels: list[str] = []
    for index, view in enumerate(views):
        if not isinstance(view, dict) or view.get("index") != index:
            raise ValueError("Rendered-design views must be contiguous and ordered")
        label = view.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError("Rendered-design view label is invalid")
        labels.append(label)
        for name in ("camera_identity", "flow_identity"):
            if not isinstance(view.get(name), str) or not view[name]:
                raise ValueError(f"Rendered-design {name} for {label!r} is invalid")
    if len(set(labels)) != len(labels):
        raise ValueError("Rendered-design view labels must be unique")
    design_record = manifest.get("design")
    samples_record = manifest.get("samples")
    if not isinstance(design_record, dict) or not isinstance(samples_record, dict):
        raise ValueError("Rendered-design file metadata is invalid")
    if (
        design_record.get("file") != DESIGN_FILENAME
        or design_record.get("dtype") != DESIGN_DTYPE.name
        or design_record.get("shape") != [sample_count, 2, 2 * mode_count]
        or design_record.get("sha256") != _sha256_file(design_path)
        or samples_record.get("file") != SAMPLES_FILENAME
        or samples_record.get("sha256") != _sha256_file(samples_path)
    ):
        raise ValueError("Rendered-design file metadata or SHA-256 is invalid")
    design = np.load(design_path, mmap_mode="r", allow_pickle=False)
    if design.dtype != DESIGN_DTYPE or design.shape != (sample_count, 2, 2 * mode_count):
        raise ValueError("Rendered-design array shape or dtype is invalid")
    with np.load(samples_path, allow_pickle=False) as archive:
        samples = {name: archive[name] for name in archive.files}
    sample_metadata = {
        name: {"dtype": value.dtype.name, "shape": list(value.shape)}
        for name, value in samples.items()
    }
    if samples_record.get("arrays") != sample_metadata:
        raise ValueError("Rendered-design sample metadata differs")
    if samples_record.get("arrays_identity") != _arrays_identity(samples):
        raise ValueError("Rendered-design sample identity differs")
    _validate_samples(
        samples,
        view_count=view_count,
        sample_count=sample_count,
        config=config,
    )
    offsets = samples["view_sample_offsets"]
    shapes = samples["view_shapes_hw"]
    for index, view in enumerate(views):
        lower, upper = int(offsets[index]), int(offsets[index + 1])
        if (
            view.get("shape_hw") != shapes[index].astype(int).tolist()
            or view.get("sample_offset") != lower
            or view.get("sample_count") != upper - lower
        ):
            raise ValueError("Rendered-design view/sample metadata differs")
    _validate_finite_design(design)
    expected_identity = hashlib.sha256(
        _canonical_json(_identity_payload(manifest))
    ).hexdigest()
    if manifest.get("rendered_design_identity") != expected_identity:
        raise ValueError("Rendered-design identity differs from its contents")
    return RenderedModalDesignArtifact(root, manifest, design, samples)


def _view_diagnostics(
    design: np.ndarray,
    alpha: np.ndarray,
    *,
    packing_max_abs_error: float,
    packing_relative_l2_error: float,
) -> dict[str, Any]:
    """Compute bounded-memory Gram diagnostics for one view's design columns."""

    column_count = design.shape[-1]
    gram = np.zeros((column_count, column_count), dtype=np.float64)
    row_count = 0
    for lower in range(0, len(design), FINITE_BLOCK_SAMPLES):
        block = np.asarray(
            design[lower : lower + FINITE_BLOCK_SAMPLES], dtype=np.float64
        ).reshape(-1, column_count)
        gram += block.T @ block
        row_count += len(block)
    eigenvalues = np.linalg.eigvalsh(gram)
    singular_values = np.sqrt(np.maximum(eigenvalues, 0.0))[::-1]
    tolerance = (
        float(singular_values[0])
        * max(row_count, column_count)
        * np.finfo(np.float64).eps
        if singular_values.size
        else 0.0
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    condition = None
    if singular_values.size and singular_values[-1] > tolerance:
        condition = float(singular_values[0] / singular_values[-1])
    alpha64 = alpha.astype(np.float64)
    return {
        "alpha": {
            "min": float(alpha64.min()),
            "p10": float(np.percentile(alpha64, 10.0)),
            "median": float(np.median(alpha64)),
            "p90": float(np.percentile(alpha64, 90.0)),
            "max": float(alpha64.max()),
            "mean": float(alpha64.mean()),
        },
        "design_column_rms": np.sqrt(
            np.maximum(np.diag(gram), 0.0) / max(row_count, 1)
        ).tolist(),
        "singular_values": singular_values.tolist(),
        "numerical_rank": rank,
        "condition_number": condition,
        "packing_verification": {
            "max_abs_error": packing_max_abs_error,
            "relative_l2_error": packing_relative_l2_error,
        },
    }


def _packing_errors(
    design: np.ndarray,
    coordinates: np.ndarray,
    direct_values: np.ndarray,
) -> tuple[float, float]:
    """Compare packed and direct renders without materializing a float64 design."""

    maximum = 0.0
    squared_error = 0.0
    squared_signal = 0.0
    coordinates64 = coordinates.astype(np.float64)
    for lower in range(0, len(design), FINITE_BLOCK_SAMPLES):
        upper = min(lower + FINITE_BLOCK_SAMPLES, len(design))
        packed = np.einsum(
            "pdc,c->pd",
            np.asarray(design[lower:upper], dtype=np.float64),
            coordinates64,
            optimize=True,
        )
        direct = np.asarray(direct_values[lower:upper], dtype=np.float64)
        difference = packed - direct
        maximum = max(maximum, float(np.max(np.abs(difference))))
        squared_error += float(np.sum(difference * difference))
        squared_signal += float(np.sum(direct * direct))
    relative = math.sqrt(squared_error) / max(
        math.sqrt(squared_signal), np.finfo(np.float64).eps
    )
    return maximum, relative


def _load_sources(
    scene_dir: str | Path,
    completed_modes_dir: str | Path,
    views: Sequence[RenderedDesignViewInput],
    device: torch.device,
) -> tuple[
    Path,
    ForegroundBackgroundScene,
    CompletedModesArtifact,
    tuple[Camera, ...],
    tuple[FlowAnalysisArtifact, ...],
    list[dict[str, Any]],
]:
    """Load the new-project identity chain and ordered fixed-view inputs."""

    if not views:
        raise ValueError("At least one rendered-design view is required")
    labels = [view.label.strip() for view in views]
    if any(not label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("Rendered-design view labels must be non-empty and unique")
    scene_path = Path(scene_dir).expanduser().resolve(strict=True)
    scene = load_static_scene(scene_path, device)
    if scene.manifest is None:
        raise ValueError("Static scene has no manifest")
    completed = load_completed_modes(completed_modes_dir)
    for name, actual, expected in (
        (
            "static scene",
            completed.manifest.get("static_scene_identity"),
            scene.manifest["static_scene_identity"],
        ),
        (
            "foreground",
            completed.manifest.get("foreground_identity"),
            scene.manifest["foreground_identity"],
        ),
    ):
        if actual != expected:
            raise ValueError(f"Completed-mode {name} identity differs")
    completed_views = completed.manifest["views"]
    if len(views) != len(completed_views):
        raise ValueError("Rendered-design and completed-mode view counts differ")
    reference_cameras = {
        camera.label: camera
        for camera in cameras_from_scene_manifest(scene.manifest)
        if camera.role == "reference" and camera.label is not None
    }
    cameras: list[Camera] = []
    flow_artifacts: list[FlowAnalysisArtifact] = []
    view_records: list[dict[str, Any]] = []
    for index, (label, source, completed_view) in enumerate(
        zip(labels, views, completed_views)
    ):
        if completed_view.get("index") != index or completed_view.get("label") != label:
            raise ValueError("Rendered-design view order differs from completed modes")
        camera = reference_cameras.get(label)
        if camera is None:
            raise ValueError(f"Static scene has no reference camera for {label!r}")
        camera_identity = camera.to_manifest_record()["camera_identity"]
        if completed_view.get("camera_identity") != camera_identity:
            raise ValueError(f"Completed-mode camera identity for {label!r} differs")
        flow = load_flow_analysis_artifact(
            Path(source.flow_artifact).expanduser().resolve(strict=True)
        )
        flow_identity = flow_artifact_identity(flow)
        if completed_view.get("flow_identity") != flow_identity:
            raise ValueError(f"Completed-mode flow identity for {label!r} differs")
        expected_shape = [camera.height, camera.width]
        if completed_view.get("shape_hw") != expected_shape:
            raise ValueError(f"Completed-mode image shape for {label!r} differs")
        if list(flow.arrays.mask_union.shape) != expected_shape:
            raise ValueError(f"Flow mask shape for {label!r} differs from its camera")
        cameras.append(camera.to(device))
        flow_artifacts.append(flow)
        view_records.append(
            {
                "index": index,
                "label": label,
                "camera_name": camera.name,
                "camera_identity": camera_identity,
                "flow_identity": flow_identity,
                "shape_hw": expected_shape,
                "flow_reference_frame_name": flow.manifest[
                    "reference_frame_name"
                ],
                "flow_reference_frame_index": int(
                    flow.manifest["reference_frame_index"]
                ),
                "frame_count": int(flow.arrays.flow.shape[0]),
                "fps_hz": float(flow.manifest["fps_hz"]),
            }
        )
    return (
        scene_path,
        scene,
        completed,
        tuple(cameras),
        tuple(flow_artifacts),
        view_records,
    )


def build_rendered_modal_design_artifact(
    *,
    scene_dir: str | Path,
    completed_modes_dir: str | Path,
    views: Sequence[RenderedDesignViewInput],
    output_dir: str | Path,
    config: RenderedDesignConfig | None = None,
    command: Sequence[str] = (),
) -> RenderedModalDesignArtifact:
    """Render completed full-foreground modes and publish the compact C16 artifact."""

    settings = config or RenderedDesignConfig()
    settings.validate()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Rendered-design output already exists: {destination}")
    if not torch.cuda.is_available():
        raise RuntimeError("Rendered modal design construction requires CUDA")
    device = torch.device("cuda")
    (
        scene_path,
        scene,
        completed,
        cameras,
        flows,
        view_records,
    ) = _load_sources(scene_dir, completed_modes_dir, views, device)
    scene_manifest = scene.manifest
    if scene_manifest is None:
        raise ValueError("Static scene has no manifest")
    mode_records = [dict(mode) for mode in completed.manifest["modes"]]
    _validate_modes(mode_records)
    phi = np.asarray(completed.arrays["phi"])
    mode_count, foreground_count, coordinate_count = phi.shape
    if (
        phi.dtype != np.complex64
        or coordinate_count != 3
        or mode_count != len(mode_records)
        or foreground_count != scene.foreground.count
        or not np.isfinite(phi).all()
    ):
        raise ValueError("Completed modes do not match the static foreground")

    scene.eval()
    means = (
        scene.foreground.active()["means"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    pixels_by_view: list[np.ndarray] = []
    alpha_by_view: list[np.ndarray] = []
    jacobian_by_view: list[np.ndarray] = []
    sample_offsets = [0]
    dummy = torch.zeros((foreground_count, 1), device=device, dtype=torch.float32)
    with torch.no_grad():
        for camera, flow, record in zip(cameras, flows, view_records):
            _, alpha_tensor = scene.render_features(
                camera, dummy, composition="foreground"
            )
            alpha_image = alpha_tensor.detach().cpu().float().numpy()
            if not np.isfinite(alpha_image).all():
                raise RuntimeError(f"Rendered alpha for {record['label']!r} is non-finite")
            pixels, sampled_alpha = _candidate_pixels(
                flow.arrays.mask_union, alpha_image, settings
            )
            K = camera.K.detach().cpu().numpy().astype(np.float64)
            w2c = (
                camera.world_to_camera.detach().cpu().numpy().astype(np.float64)
            )
            jacobian, visible = _projection_jacobian(means, K, w2c, camera.radial_distortion)
            if not np.any(visible):
                raise ValueError(
                    f"All foreground Gaussians lie behind view {record['label']!r}"
                )
            record["sample_offset"] = sample_offsets[-1]
            record["sample_count"] = len(pixels)
            record["foreground_gaussians_in_front"] = int(np.count_nonzero(visible))
            sample_offsets.append(sample_offsets[-1] + len(pixels))
            pixels_by_view.append(pixels)
            alpha_by_view.append(sampled_alpha)
            jacobian_by_view.append(jacobian)

    sample_count = sample_offsets[-1]
    view_shapes = np.asarray(
        [[camera.height, camera.width] for camera in cameras], dtype=np.int64
    )
    sample_view_index = np.concatenate(
        [
            np.full(len(pixels), index, dtype=np.int64)
            for index, pixels in enumerate(pixels_by_view)
        ]
    )
    samples = {
        "view_shapes_hw": view_shapes,
        "view_sample_offsets": np.asarray(sample_offsets, dtype=np.int64),
        "sample_view_index": sample_view_index,
        "sample_pixels_xy": np.concatenate(pixels_by_view, axis=0),
        "sample_foreground_alpha": np.concatenate(alpha_by_view, axis=0),
    }
    _validate_samples(
        samples,
        view_count=len(cameras),
        sample_count=sample_count,
        config=settings,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
    )
    try:
        design_path = temporary / DESIGN_FILENAME
        design = np.lib.format.open_memmap(
            design_path,
            mode="w+",
            dtype=DESIGN_DTYPE,
            shape=(sample_count, 2, 2 * mode_count),
        )
        rng = np.random.default_rng(1729)
        progress = Progress("rendered design", len(cameras) * mode_count, unit="view-modes")
        with torch.no_grad():
            for view_index, (camera, pixels, sampled_alpha, jacobian) in enumerate(
                zip(cameras, pixels_by_view, alpha_by_view, jacobian_by_view)
            ):
                lower, upper = sample_offsets[view_index : view_index + 2]
                jacobian_tensor = torch.as_tensor(
                    jacobian, device=device, dtype=torch.float32
                )
                for start in range(0, mode_count, settings.modes_per_batch):
                    stop = min(start + settings.modes_per_batch, mode_count)
                    phi_real = torch.as_tensor(
                        np.ascontiguousarray(np.real(phi[start:stop])),
                        device=device,
                        dtype=torch.float32,
                    )
                    phi_imag = torch.as_tensor(
                        np.ascontiguousarray(np.imag(phi[start:stop])),
                        device=device,
                        dtype=torch.float32,
                    )
                    projected_real = torch.einsum(
                        "gij,kgj->kgi", jacobian_tensor, phi_real
                    )
                    projected_imag = torch.einsum(
                        "gij,kgj->kgi", jacobian_tensor, phi_imag
                    )
                    features = torch.stack(
                        (
                            projected_real[..., 0],
                            projected_real[..., 1],
                            -projected_imag[..., 0],
                            -projected_imag[..., 1],
                        ),
                        dim=-1,
                    ).permute(1, 0, 2).reshape(foreground_count, -1).contiguous()
                    values = _sample_feature_render(
                        scene, camera, features, pixels, sampled_alpha
                    ).reshape(len(pixels), stop - start, 4)
                    for local_mode, mode_slot in enumerate(range(start, stop)):
                        design[lower:upper, 0, 2 * mode_slot] = values[
                            :, local_mode, 0
                        ]
                        design[lower:upper, 1, 2 * mode_slot] = values[
                            :, local_mode, 1
                        ]
                        design[lower:upper, 0, 2 * mode_slot + 1] = values[
                            :, local_mode, 2
                        ]
                        design[lower:upper, 1, 2 * mode_slot + 1] = values[
                            :, local_mode, 3
                        ]
                    progress.update(
                        view_index * mode_count + stop,
                        f"camera={camera.name} modes={stop}/{mode_count}",
                    )

                packed = rng.standard_normal(2 * mode_count).astype(np.float32)
                packed /= np.sqrt(np.mean(packed * packed, dtype=np.float64))
                q_real, q_imag = packed[0::2], packed[1::2]
                direct_phi = np.einsum(
                    "k,kgj->gj", q_real, np.real(phi), optimize=True
                ) - np.einsum(
                    "k,kgj->gj", q_imag, np.imag(phi), optimize=True
                )
                direct_features = torch.einsum(
                    "gij,gj->gi",
                    jacobian_tensor,
                    torch.as_tensor(
                        np.ascontiguousarray(direct_phi),
                        device=device,
                        dtype=torch.float32,
                    ),
                )
                direct_values = _sample_feature_render(
                    scene, camera, direct_features, pixels, sampled_alpha
                )
                max_abs_error, relative_l2_error = _packing_errors(
                    design[lower:upper], packed, direct_values
                )
                if (
                    max_abs_error > PACKING_MAX_ABS_TOLERANCE
                    and relative_l2_error > PACKING_RELATIVE_L2_TOLERANCE
                ):
                    raise RuntimeError(
                        f"Rendered-design packing failed for {view_records[view_index]['label']!r}: "
                        f"max={max_abs_error:.6g}, relative_l2={relative_l2_error:.6g}"
                    )
                view_records[view_index]["diagnostics"] = _view_diagnostics(
                    design[lower:upper],
                    sampled_alpha,
                    packing_max_abs_error=max_abs_error,
                    packing_relative_l2_error=relative_l2_error,
                )
        design.flush()
        del design
        samples_path = temporary / SAMPLES_FILENAME
        save_named_arrays(samples_path, samples)
        design_record = {
            "file": DESIGN_FILENAME,
            "dtype": DESIGN_DTYPE.name,
            "shape": [sample_count, 2, 2 * mode_count],
            "sha256": _sha256_file(design_path),
        }
        samples_record = {
            "file": SAMPLES_FILENAME,
            "arrays": {
                name: {"dtype": value.dtype.name, "shape": list(value.shape)}
                for name, value in samples.items()
            },
            "arrays_identity": _arrays_identity(samples),
            "sha256": _sha256_file(samples_path),
        }
        manifest = {
            "format": RENDERED_DESIGN_FORMAT,
            "version": 2 if any(c.distortion_applied for c in cameras) else RENDERED_DESIGN_VERSION,
            "producer": {
                "project_version": __version__,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "command": list(command),
            },
            "static_scene": str(scene_path),
            "static_scene_identity": scene_manifest["static_scene_identity"],
            "foreground_identity": scene_manifest["foreground_identity"],
            "completed_modes": str(completed.path),
            "completed_modes_identity": completed.manifest[
                "completed_modes_identity"
            ],
            "flow_artifacts": [str(flow.path.resolve()) for flow in flows],
            "modes": mode_records,
            "views": view_records,
            "settings": settings.to_dict(),
            "convention": dict(DISTORTED_DESIGN_CONVENTION if any(c.distortion_applied for c in cameras) else DESIGN_CONVENTION),
            "quality_gate": {
                "required": True,
                "status": "rendered_design_candidate_unapproved",
                "inherited_from": "completion_candidate_unapproved",
            },
            "counts": {
                "views": len(cameras),
                "modes": mode_count,
                "foreground_gaussians": foreground_count,
                "samples": sample_count,
                "columns": 2 * mode_count,
            },
            "design": design_record,
            "samples": samples_record,
        }
        manifest["rendered_design_identity"] = hashlib.sha256(
            _canonical_json(_identity_payload(manifest))
        ).hexdigest()
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        load_rendered_modal_design(temporary)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"Rendered-design output already exists: {destination}"
            )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_rendered_modal_design(destination)


__all__ = [
    "RenderedDesignConfig",
    "RenderedDesignViewInput",
    "RenderedModalDesignArtifact",
    "build_rendered_modal_design_artifact",
    "load_rendered_modal_design",
]
