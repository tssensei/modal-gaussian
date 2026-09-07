"""Artifact-native Viser playback for a completed modal Gaussian result."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor
import viser
import viser.transforms as vtf

from modal_gaussians.result import ModalResultArtifact, load_modal_result
from modal_gaussians.motion.common.mode_mapping import resolve_source_mode_slots
from modal_gaussians.motion.rigid.rigid import load_rigid_modes
from modal_gaussians.static import (
    Camera,
    _load_gsplat_rasterization,
    cameras_from_scene_manifest,
)
from modal_gaussians.motion.rigid.structure_graph import load_observed_structure_graph
from modal_gaussians.topology import load_observation_topology
from modal_gaussians.vis.playback_panel import add_gui_playback_group
from modal_gaussians.vis.render_panel import populate_render_tab
from modal_gaussians.vis.spectrum import (
    ModalSpectrumPanel,
    SpectrumComparisonController,
)


DRIVE_FLOW = "flow-derived coordinates"
DRIVE_MANUAL = "manual oscillator"
COLOR_RGB = "rgb"
COLOR_PHASE = "modal phase"
COLOR_OBSERVATIONS = "obs count"

LEGACY_SUPPORT_DISPLAY_NAMES = (
    "anchor",
    "filled",
    "unobserved",
)
BASIS_SUPPORT_DISPLAY_NAMES = (
    "measurement-supported",
    "graph-propagated",
    "zero-fallback",
)
NEURAL_SUPPORT_DISPLAY_NAMES = (
    "directly-supervised",
    "structure-inferred",
    "unresolved",
)
SUPPORT_COLORS = np.asarray(
    (
        (0.05, 0.55, 1.0),
        (0.1, 0.85, 0.3),
        (0.65, 0.35, 1.0),
        (1.0, 0.75, 0.05),
    ),
    dtype=np.float32,
)


def _stable_uniform_indices(count: int, maximum: int) -> np.ndarray:
    """Select a deterministic uniform subset without reordering its indices."""

    if count < 0 or maximum < 0:
        raise ValueError("Uniform-subset sizes must be non-negative")
    visible = min(int(count), int(maximum))
    if visible == 0:
        return np.empty((0,), dtype=np.int64)
    if visible == count:
        return np.arange(count, dtype=np.int64)
    return np.floor(
        np.linspace(0, count, visible, endpoint=False, dtype=np.float64)
    ).astype(np.int64)


def _component_colors(
    component_index: np.ndarray,
    *,
    trusted_component_index: np.ndarray | None = None,
) -> np.ndarray:
    """Color graph components deterministically, optionally graying untrusted ones."""

    components = np.asarray(component_index)
    if components.ndim != 1 or not np.issubdtype(components.dtype, np.integer):
        raise ValueError("component_index must be a 1-D integer array")
    if np.any(components < 0):
        raise ValueError("component_index must be non-negative")
    hue = np.mod(components.astype(np.float64) * 0.6180339887498949, 1.0)
    h6 = hue * 6.0
    sector = np.floor(h6).astype(np.int64)
    fraction = h6 - sector
    saturation = 0.72
    value = 0.95
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * fraction)
    t = value * (1.0 - saturation * (1.0 - fraction))
    rgb = np.empty((len(components), 3), dtype=np.float64)
    choices = (
        (value, t, p),
        (q, value, p),
        (p, value, t),
        (p, q, value),
        (t, p, value),
        (value, p, q),
    )
    for index, channels in enumerate(choices):
        mask = sector == index
        if np.any(mask):
            for column, channel in enumerate(channels):
                if isinstance(channel, np.ndarray):
                    rgb[mask, column] = channel[mask]
                else:
                    rgb[mask, column] = float(channel)
    if trusted_component_index is not None:
        trusted = np.asarray(trusted_component_index)
        if (
            trusted.ndim != 1
            or not np.issubdtype(trusted.dtype, np.integer)
            or np.any(trusted < 0)
        ):
            raise ValueError("trusted_component_index must be non-negative graph IDs")
        rgb[~np.isin(components, trusted)] = 0.55
    return rgb.astype(np.float32)


def _component_graph_color_frames(
    edge_components: np.ndarray,
    completed_manifest: dict[str, Any],
    completed_arrays: dict[str, np.ndarray],
    rigid_manifest: dict[str, Any],
    rigid_arrays: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray | None]:
    """Keep legacy trust static; color v6/v7 active bases in greedy mode order."""

    retained = np.asarray(rigid_arrays["component_retained_mask"])
    graph_ids = np.asarray(rigid_arrays["component_graph_index"])
    if retained.dtype != np.bool_ or retained.ndim != 2:
        raise ValueError("Component trust must be boolean [K_source,C]")
    if graph_ids.shape != (retained.shape[1],):
        raise ValueError("Rigid-to-graph component mapping has the wrong shape")
    if completed_manifest.get("version") not in (6, 7):
        trusted_graph = graph_ids[np.all(retained, axis=0)]
        return _component_colors(edge_components, trusted_component_index=trusted_graph), None
    if completed_manifest.get("motion_basis", {}).get("basis_selection_policy") != "trusted_per_mode":
        raise ValueError("Per-mode graph colors require trusted_per_mode eligibility")
    source_slots = resolve_source_mode_slots(
        completed_manifest["modes"], rigid_manifest["modes"]
    )
    basis_components = np.asarray(completed_arrays["basis_component_index"])
    active = np.asarray(completed_arrays["basis_active_mask"])
    if (
        basis_components.ndim != 1
        or not np.issubdtype(basis_components.dtype, np.integer)
        or not len(basis_components)
        or basis_components[-1] != -1
        or np.any(basis_components[:-1] < 0)
        or np.any(basis_components[:-1] >= len(graph_ids))
    ):
        raise ValueError("Per-mode graph basis component indices are invalid")
    if active.dtype != np.bool_ or active.shape != (len(source_slots), len(basis_components)):
        raise ValueError("Per-mode graph basis activity must be boolean [K,B]")
    expected_active = np.column_stack((
        retained[source_slots][:, basis_components[:-1]],
        np.ones(len(source_slots), dtype=bool),
    ))
    if not np.array_equal(active, expected_active):
        raise ValueError("Per-mode graph basis activity differs from source trust")
    basis_graph_ids = graph_ids[basis_components[:-1]]
    colors = np.stack([
        _component_colors(edge_components, trusted_component_index=basis_graph_ids[row[:-1]])
        for row in active
    ])
    return colors[0], colors


@dataclass(frozen=True)
class ViewerCamera:
    """Expose one calibrated result camera in the Viser convention."""

    label: str
    camera: Camera
    c2w: np.ndarray
    fov: float
    aspect: float


def _motion_fill_display_classes(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Map the final sequential-fill state to the three useful Viewer roles."""

    # Both trusted seeds and promoted components are fixed anchors. All other
    # completed points are filled, while incomplete points remain unobserved.
    anchors = arrays["trusted_seed_mask"] | arrays["component_anchor_point_mask"]
    classes = np.full(anchors.shape, 2, dtype=np.int8)  # Unobserved.
    classes[arrays["completion_mask"] & ~anchors] = 1  # Filled.
    classes[anchors] = 0  # Anchor, including promoted single-view components.
    return classes


def _basis_display_classes(
    manifest: dict[str, Any], arrays: dict[str, np.ndarray]
) -> np.ndarray:
    """Map shared, per-frequency, or green-refined basis support to Viewer roles."""

    version = manifest.get("version")
    methods = {
        3: "shared_motion_basis_blend",
        4: "per_frequency_motion_basis_blend",
        5: "fixed_observation_green_basis_refinement",
        6: "per_frequency_motion_basis_blend",
        7: "fixed_observation_green_basis_refinement",
    }
    if version not in methods or manifest.get("completion_method") != methods[version]:
        raise ValueError("Unsupported motion-basis completion role contract")
    names = (
        "measurement_supported_mask",
        "graph_propagated_mask",
        "zero_fallback_mask",
    )
    missing = [name for name in (*names, "phi") if name not in arrays]
    if missing:
        raise ValueError(
            "Motion-basis Viewer arrays are missing: " + ", ".join(missing)
        )
    masks = tuple(np.asarray(arrays[name]) for name in names)
    phi = np.asarray(arrays["phi"])
    if phi.ndim != 3 or phi.shape[2] != 3:
        raise ValueError("Motion-basis phi must have shape [K,G,3]")
    gaussian_count = phi.shape[1]
    support_shape = (gaussian_count,) if version == 3 else phi.shape[:2]
    if any(
        mask.dtype != np.bool_ or mask.shape != support_shape
        for mask in masks
    ):
        shape_label = "[G]" if version == 3 else "[K,G]"
        raise ValueError(f"Motion-basis support masks must be boolean {shape_label}")
    membership = np.stack(masks, axis=0).sum(axis=0)
    if not np.all(membership == 1):
        raise ValueError(
            "Motion-basis support masks must be mutually exclusive and exhaustive"
        )
    classes = np.full(support_shape, 2, dtype=np.int8)
    classes[masks[1]] = 1
    classes[masks[0]] = 0
    if version != 3:
        return classes
    return np.repeat(classes[None], phi.shape[0], axis=0)


def _completed_mode_display_roles(
    manifest: dict[str, Any], arrays: dict[str, np.ndarray]
) -> tuple[np.ndarray, tuple[str, ...], str]:
    """Resolve exact debug-role labels for sequential fill or basis blending."""

    if manifest.get("version") in (8, 9, 10, 11, 12, 13, 14, 15, 16):
        derived = manifest["version"] in (9, 10, 11, 12, 13, 14, 15, 16)
        method = {8: "neural_complex_displacement_field", 9: "neural_fragment_motion_propagation",
                  10: "neural_field_with_training_fragment_fill", 11: "neural_field_with_surface_attachments",
                  12: "neural_field_with_pointwise_displacement_fill", 13: "neural_pointwise_observation_refinement",
                  14: "neural_field_with_guarded_neighbor_residuals", 15: "neural_guarded_observation_refinement",
                  16: "neural_component_field_with_stable_donors"}[manifest["version"]]
        if manifest.get("completion_method") != method:
            raise ValueError("Unsupported neural completion role contract")
        phi = np.asarray(arrays["phi"])
        support = np.asarray(arrays["support_class"])
        if (
            phi.ndim != 3 or phi.shape[2] != 3
            or support.shape != phi.shape[:2]
            or not np.issubdtype(support.dtype, np.integer)
            or np.any((support < 0) | (support > (3 if derived else 2)))
        ):
            raise ValueError("Neural support classes must match the method's role range")
        classes = np.asarray([2, 0, 1, 3] if derived else [2, 0, 1], dtype=np.int8)[support]
        legend = (
            "**Role colors:** directly image-supervised = blue | "
            "structure-inferred = green | unresolved = purple"
        )
        if derived:
            legend += " | fragment-propagated = yellow (original image support recorded separately)"
        if manifest["version"] == 13:
            legend += " | yellow includes observation-refined propagated points"
        if manifest["version"] in (14, 15):
            legend += " | blue includes small-component residuals constrained by stable neighbors; yellow stays propagated"
        names = NEURAL_SUPPORT_DISPLAY_NAMES + (("fragment-propagated",) if derived else ())
        if manifest["version"] == 16:
            legend += " | blue/green use their own component field; donor eligibility is recorded separately"
        return classes, names, legend
    if manifest.get("version") in (3, 4, 5, 6, 7):
        classes = _basis_display_classes(manifest, arrays)
        legend = (
            "**Role colors:** measurement-supported = blue | "
            "graph-propagated = green | zero-fallback = purple"
        )
        return classes, BASIS_SUPPORT_DISPLAY_NAMES, legend
    classes = _motion_fill_display_classes(arrays)
    legend = "**Role colors:** anchor = blue | filled = green | unobserved = purple"
    return classes, LEGACY_SUPPORT_DISPLAY_NAMES, legend


def _hsv_to_rgb(hue: Tensor, value: Tensor) -> Tensor:
    """Map phase hue and amplitude value to differentiable RGB tensors."""

    hue = torch.remainder(hue, 1.0)
    h6 = hue * 6.0
    sector = torch.floor(h6).long()
    fraction = h6 - sector.to(dtype=hue.dtype)
    zero = torch.zeros_like(value)
    q = value * (1.0 - fraction)
    t = value * fraction
    candidates = (
        torch.stack((value, t, zero), dim=-1),
        torch.stack((q, value, zero), dim=-1),
        torch.stack((zero, value, t), dim=-1),
        torch.stack((zero, q, value), dim=-1),
        torch.stack((t, zero, value), dim=-1),
        torch.stack((value, zero, q), dim=-1),
    )
    rgb = torch.zeros(hue.shape + (3,), dtype=hue.dtype, device=hue.device)
    for index, candidate in enumerate(candidates):
        rgb = torch.where((sector == index)[..., None], candidate, rgb)
    return rgb


def _viewer_cameras(result: ModalResultArtifact) -> tuple[ViewerCamera, ...]:
    """Resolve the ordered result views to exact static-scene cameras."""

    if result.scene.manifest is None:
        raise ValueError("Modal-result static scene has no manifest")
    by_name = {
        camera.name: camera
        for camera in cameras_from_scene_manifest(result.scene.manifest)
    }
    cameras: list[ViewerCamera] = []
    for view in result.manifest["views"]:
        name = str(view["camera_name"])
        camera = by_name.get(name)
        if camera is None:
            raise ValueError(f"Result camera {name!r} is absent from static scene")
        c2w = np.linalg.inv(camera.world_to_camera.detach().cpu().numpy())
        fy = float(camera.K[1, 1].item())
        fov = 2.0 * math.atan(0.5 * float(camera.height) / fy)
        cameras.append(
            ViewerCamera(
                label=str(view["label"]),
                camera=camera,
                c2w=c2w.astype(np.float64),
                fov=fov,
                aspect=float(camera.width) / float(camera.height),
            )
        )
    if len({camera.label for camera in cameras}) != len(cameras):
        raise ValueError("Modal-result Viewer camera labels are not unique")
    return tuple(cameras)


def _observation_counts(result: ModalResultArtifact) -> np.ndarray:
    """Count unique topology views contributing to each foreground Gaussian."""

    completed = result.completed_modes
    if completed.manifest.get("version") in (8, 9, 10, 11, 12, 13, 14, 15, 16):
        observed = np.asarray(completed.arrays["observation_view_mask"])
        expected = (
            len(result.manifest["modes"]), result.scene.foreground.count,
            len(result.manifest["views"]),
        )
        if observed.dtype != np.bool_ or observed.shape != expected:
            raise ValueError("Neural observation view mask must be boolean [K,G,V]")
        return observed.sum(axis=2, dtype=np.int16)
    topology = load_observation_topology(completed.manifest["topology"])
    offsets = topology.arrays.sample_offsets
    contributor_views = np.repeat(
        topology.arrays.sample_view_index,
        np.diff(offsets),
    )
    contributors = topology.arrays.contributor_gaussian_index
    view_count = len(result.manifest["views"])
    gaussian_count = result.scene.foreground.count
    observed = np.zeros((gaussian_count, view_count), dtype=bool)
    observed[contributors, contributor_views] = True
    per_gaussian = np.asarray(observed.sum(axis=1), dtype=np.int16)
    return np.repeat(
        per_gaussian[None], len(result.manifest["modes"]), axis=0
    )


def _neural_graph_display(
    arrays: dict[str, np.ndarray], gaussian_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Display neural geometry connectivity without interpreting rigid trust."""

    edges = np.asarray(arrays["g_edge_index"])
    components = np.asarray(arrays["g_component_index"])
    weights = np.asarray(arrays["g_edge_weight"])
    if (
        edges.ndim != 2 or edges.shape[1] != 2
        or not np.issubdtype(edges.dtype, np.integer)
        or np.any(edges < 0) or np.any(edges >= gaussian_count)
        or components.shape != (gaussian_count,)
        or not np.issubdtype(components.dtype, np.integer)
        or np.any(components < 0)
        or weights.shape != (len(edges),)
        or not np.isfinite(weights).all() or np.any(weights <= 0)
    ):
        raise ValueError("Neural Gaussian geometry graph arrays are invalid")
    if np.any(components[edges[:, 0]] != components[edges[:, 1]]):
        raise ValueError("Neural geometry edges cross graph components")
    return edges, _component_colors(components[edges[:, 0]])


def _control_point_gaussian_indices(arrays: dict[str, np.ndarray], gaussian_count: int) -> np.ndarray:
    """Map saved controls from a possible host-local domain to foreground order."""
    if "c_control_point_index" not in arrays:
        return np.empty(0, dtype=np.int64)
    indices = np.asarray(arrays["c_control_point_index"])
    hosts = np.asarray(arrays.get("t_host_gaussian_index", np.arange(gaussian_count)))
    if (indices.ndim != 1 or hosts.ndim != 1
            or not np.issubdtype(indices.dtype, np.integer)
            or not np.issubdtype(hosts.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= len(hosts))
            or np.any(hosts < 0) or np.any(hosts >= gaussian_count)):
        raise ValueError("Control point indices differ from their foreground/host domain")
    return hosts[indices].astype(np.int64, copy=False)


class ModalViewerData:
    """Hold strict result data and implement all scientific display transforms."""

    def __init__(self, result_dir: str | Path, device: str = "cuda", *, preview: bool = False) -> None:
        requested_device = torch.device(device)
        if requested_device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Modal Gaussian Viser requires a CUDA device")
        prepared = None
        self.is_preview = preview
        if preview:
            from modal_gaussians.motion.neural.preview import load_preview
            self.result = load_preview(result_dir)
            prepared = self.result.prepared
        else:
            self.result = load_modal_result(result_dir)
            contract_path = Path(result_dir).resolve().parent / "iteration.json"
            if contract_path.is_file():
                import json
                from modal_gaussians.motion.neural.prepared import load_prepared
                from modal_gaussians.motion.neural.neural_modes import _source_identity
                contract = json.loads(contract_path.read_text())
                prepared = load_prepared(contract["prepared"])
                if (prepared.manifest["prepared_identity"] != contract["prepared_identity"]
                        or _source_identity(self.result.completed_modes.manifest) != prepared.manifest["source_identity"]):
                    raise ValueError("Viewer preparation differs from result sources")
        self.device = requested_device
        self.scene = self.result.scene.to(self.device)
        self.cameras = _viewer_cameras(self.result)
        self.camera_by_label = {camera.label: camera for camera in self.cameras}
        self.frequencies_hz = np.asarray(
            [mode["frequency_hz"] for mode in self.result.manifest["modes"]],
            dtype=np.float64,
        )
        self.frequency_order = tuple(
            int(index)
            for index in np.argsort(self.frequencies_hz, kind="stable")
        )
        display_indices = np.empty(len(self.frequency_order), dtype=np.int64)
        display_indices[np.asarray(self.frequency_order)] = np.arange(
            len(self.frequency_order)
        )
        self.display_indices = tuple(int(value) for value in display_indices)
        self.phi = torch.from_numpy(
            np.asarray(self.result.completed_modes.arrays["phi"])
        ).to(self.device)
        self.coordinates = None if preview else np.asarray(
            self.result.coordinates.coordinates, dtype=np.complex64
        )
        (
            self.display_class,
            self.support_display_names,
            self.support_legend,
        ) = _completed_mode_display_roles(
            self.result.completed_modes.manifest,
            self.result.completed_modes.arrays,
        )
        self.observation_counts = _observation_counts(self.result)
        self.control_point_gaussian_index = _control_point_gaussian_indices(
            self.result.completed_modes.arrays, self.scene.foreground.count,
        )
        self.control_point_colors = (
            _component_colors(self.result.completed_modes.arrays["g_component_index"][self.control_point_gaussian_index])
            if len(self.control_point_gaussian_index) else np.empty((0, 3), dtype=np.float32)
        )
        self._load_graph_display()
        self.spectrum = SpectrumComparisonController(self.result, prepared=prepared)

    def _load_graph_display(self) -> None:
        """Select geometry diagnostics belonging to the completed-mode method."""

        if self.result.completed_modes.manifest.get("version") in (8, 9, 10, 11, 12, 13, 14, 15, 16):
            self.structure_graph = None
            self.graph_edge_gaussian_index, self.graph_edge_colors = _neural_graph_display(
                self.result.completed_modes.arrays, self.scene.foreground.count,
            )
            self.graph_edge_colors_by_mode = None
            self.graph_legend = (
                "**Geometry graph:** distinct colors = connected geometry components. "
                "Colors indicate connectivity, not rigid trust or observation support."
            )
            return
        graph_path = self.result.completed_modes.manifest.get(
            "observed_structure_graph"
        )
        graph_identity = self.result.completed_modes.manifest.get(
            "observed_structure_graph_identity"
        )
        if not isinstance(graph_path, str) or not graph_path:
            raise ValueError("Completed modes do not identify their observed graph")
        self.structure_graph = (
            load_observed_structure_graph(graph_path)
        )
        if (
            self.structure_graph.manifest["observed_structure_graph_identity"]
            != graph_identity
        ):
            raise ValueError("Completed modes identify a different observed graph")
        if (
            self.structure_graph.manifest["static_scene_identity"]
            != self.result.manifest["static_scene_identity"]
            or self.structure_graph.manifest["foreground_identity"]
            != self.result.manifest["foreground_identity"]
        ):
            raise ValueError("Observed graph does not belong to the Viewer scene")
        graph_arrays = self.structure_graph.arrays
        self.graph_edge_gaussian_index = graph_arrays.node_gaussian_index[
            graph_arrays.edge_index
        ]
        edge_components = graph_arrays.component_index[
            graph_arrays.edge_index[:, 0]
        ]
        completed_manifest = self.result.completed_modes.manifest
        rigid = load_rigid_modes(completed_manifest["rigid_modes"])
        for identity in (
            "rigid_modes_identity",
            "observed_structure_graph_identity",
            "static_scene_identity",
            "foreground_identity",
        ):
            if rigid.manifest[identity] != completed_manifest[identity]:
                raise ValueError(f"Component graph rigid source differs: {identity}")
        self.graph_edge_colors, self.graph_edge_colors_by_mode = _component_graph_color_frames(
            edge_components, completed_manifest, self.result.completed_modes.arrays,
            rigid.manifest, rigid.arrays,
        )
        self.graph_legend = (
            "**Component graph:** distinct colors = active trusted components at "
            "Selected frequency (Hz); gray = inactive components. Select a frequency "
            "in Gaussian color or Spectrum."
            if self.graph_edge_colors_by_mode is not None else
            "**Component graph:** distinct colors = components trusted at "
            "every source frequency; gray = other components."
        )

    def coordinate(self, view_index: int, local_frame: int) -> np.ndarray:
        """Return one stored flow-derived complex coordinate vector."""

        if self.coordinates is None:
            raise ValueError("This preview has no video coordinates; use the manual oscillator")
        view = self.result.manifest["views"][view_index]
        frame_count = int(view["frame_count"])
        if not 0 <= local_frame < frame_count:
            raise ValueError("Viewer local frame index is outside its view")
        global_index = int(view["frame_offset"]) + local_frame
        return np.asarray(self.coordinates[global_index], dtype=np.complex64)

    def deformed_means(self, q: np.ndarray, scale: float = 1.0) -> Tensor:
        """Apply the result's exact real(q*phi) foreground deformation."""

        values = np.array(q, dtype=np.complex64, copy=True)
        if values.shape != (len(self.frequencies_hz),):
            raise ValueError("Viewer modal coordinate has the wrong mode count")
        q_tensor = torch.from_numpy(values).to(self.device)
        offset = torch.einsum("k,kgc->gc", q_tensor.real, self.phi.real)
        offset -= torch.einsum("k,kgc->gc", q_tensor.imag, self.phi.imag)
        return self.scene.foreground.active()["means"] + float(scale) * offset

    def phase_colors(
        self,
        mode_index: int,
        component_index: int,
        normalization: str,
    ) -> Tensor:
        """Color foreground Gaussians by their calibrated projected modal phase."""

        label, alpha, identifiable, magnitude_hi = (
            self.spectrum.current_modal_phase_display_context(
                mode_index, component_index, normalization
            )
        )
        means = self.scene.foreground.active()["means"]
        if not identifiable:
            return torch.zeros_like(means)
        camera = self.camera_by_label[label].camera.to(self.device)
        w2c = camera.world_to_camera
        rotation, translation = w2c[:3, :3], w2c[:3, 3]
        camera_points = means @ rotation.T + translation[None]
        x, y, z = camera_points.unbind(dim=1)
        epsilon = torch.as_tensor(1.0e-6, device=z.device, dtype=z.dtype)
        z_safe = torch.where(
            z.abs() < epsilon,
            torch.where(z >= 0, epsilon, -epsilon),
            z,
        )
        jacobian_camera = torch.zeros(
            (len(means), 2, 3), device=self.device, dtype=means.dtype
        )
        fx, fy = camera.K[0, 0], camera.K[1, 1]
        jacobian_camera[:, 0, 0] = fx / z_safe
        jacobian_camera[:, 0, 2] = -fx * x / z_safe.square()
        jacobian_camera[:, 1, 1] = fy / z_safe
        jacobian_camera[:, 1, 2] = -fy * y / z_safe.square()
        if camera.radial_distortion:
            k = camera.radial_distortion
            qx, qy = x / z_safe, y / z_safe
            s = 1 + k * (qx.square() + qy.square())
            distortion = torch.empty((len(means), 2, 2), device=self.device, dtype=means.dtype)
            distortion[:, 0, 0] = s + 2 * k * qx.square()
            distortion[:, 0, 1] = (fx / fy) * 2 * k * qx * qy
            distortion[:, 1, 0] = (fy / fx) * 2 * k * qx * qy
            distortion[:, 1, 1] = s + 2 * k * qy.square()
            jacobian_camera = distortion @ jacobian_camera
        jacobian = torch.einsum("nij,jk->nik", jacobian_camera, rotation)
        projected = torch.einsum(
            "nij,nj->ni", jacobian, self.phi[mode_index]
        )[:, component_index]
        alpha_tensor = torch.as_tensor(alpha, device=self.device)
        projected = alpha_tensor * projected
        amplitude = torch.abs(projected)
        phase = torch.angle(projected)
        finite = torch.isfinite(amplitude) & torch.isfinite(phase)
        high = torch.as_tensor(
            float(magnitude_hi), device=self.device, dtype=amplitude.dtype
        )
        value = torch.zeros_like(amplitude)
        value[finite] = (amplitude[finite] / high).clamp(0.0, 1.0)
        hue = (torch.where(finite, phase, torch.zeros_like(phase)) + torch.pi) / (
            2.0 * torch.pi
        )
        return _hsv_to_rgb(hue, value)

    def observation_colors(self, mode_index: int) -> Tensor:
        """Apply the old one/two/three-plus calibrated-view color legend."""

        counts = torch.from_numpy(self.observation_counts[mode_index]).to(self.device)
        colors = torch.full(
            (len(counts), 3), 0.15, device=self.device, dtype=torch.float32
        )
        colors[counts <= 1] = torch.tensor((1.0, 0.43, 0.16), device=self.device)
        colors[counts == 2] = torch.tensor((0.27, 0.55, 1.0), device=self.device)
        colors[counts >= 3] = torch.tensor((0.27, 0.82, 0.47), device=self.device)
        return colors


class ModalViserViewer:
    """Recreate the accepted old modal Viewer against the new artifact chain."""

    def __init__(
        self,
        data: ModalViewerData,
        *,
        work_dir: str | Path,
        host: str = "0.0.0.0",
        port: int = 8080,
        viewer_resolution: int = 2048,
    ) -> None:
        self.data = data
        self.work_dir = Path(work_dir).expanduser().resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.server = viser.ViserServer(host=host, port=port, label="Modal Gaussians")
        self.server.gui.configure_theme(
            control_width="medium",
            dark_mode=True,
            brand_color=(255, 211, 105),
        )
        self.server.gui.main_panel.dock_right()
        self._viewer_resolution = int(viewer_resolution)
        self._last_client: viser.ClientHandle | None = None
        self._render_lock = threading.Lock()
        self._render_running = False
        self._render_again = False
        self._selection_sync = False
        self._canonical_disabled_cache: list[bool] = []
        self._point_cloud: Any | None = None
        self._control_point_cloud: Any | None = None
        self._component_graph_handle: Any | None = None
        self._component_graph_edge_indices = np.empty((0,), dtype=np.int64)
        self._component_graph_color_mode: int | None = None
        self._frustums: dict[str, Any] = {}
        all_means = torch.cat(
            (
                self.data.scene.foreground.active()["means"],
                self.data.scene.background.active()["means"],
            ),
            dim=0,
        ).detach()
        finite = torch.isfinite(all_means).all(dim=1)
        if bool(finite.any().item()):
            finite_means = all_means[finite]
            center = torch.median(finite_means, dim=0).values
            distances = torch.linalg.norm(finite_means - center[None], dim=1)
            scene_scale = max(float(torch.quantile(distances, 0.9).item()), 1.0e-3)
            self._orbit_center = center.cpu().numpy()
        else:
            scene_scale = 1.0
            self._orbit_center = np.zeros(3, dtype=np.float32)
        self._camera_frustum_scale = 0.08 * scene_scale
        self._build_gui()
        self._register_clients()

    @property
    def views(self) -> list[dict[str, Any]]:
        """Return the validated ordered playback-view records."""

        return self.data.result.manifest["views"]

    def _active_view_index(self) -> int:
        """Resolve the current playback label to its result view index."""

        label = str(self.playback_view.value)
        labels = [str(view["label"]) for view in self.views]
        if label not in labels:
            raise ValueError(f"Unknown playback view: {label!r}")
        return labels.index(label)

    def _active_frame_count(self) -> int:
        """Return the frame count of the selected playback view."""

        return 1800 if self.data.is_preview else int(self.views[self._active_view_index()]["frame_count"])

    def _build_gui(self) -> None:
        """Build every accepted control group from the old modal Viewer."""

        with self.server.gui.add_folder("Rendering"):
            self.viewer_resolution = self.server.gui.add_slider(
                "Viewer Res",
                min=64,
                max=2048,
                step=1,
                initial_value=self._viewer_resolution,
                hint="Maximum resolution of the viewer rendered image.",
            )
            self.viewer_resolution.on_update(self.request_render)

        with self.server.gui.add_folder("Time"):
            if self.data.is_preview:
                self.server.gui.add_markdown("**Mode preview — manual oscillation. Video coordinates have not been fitted.**")
            self.playback_view = self.server.gui.add_dropdown(
                "Playback view",
                options=tuple(str(view["label"]) for view in self.views),
                initial_value=str(self.views[0]["label"]),
            )
            self.playback = add_gui_playback_group(
                self.server,
                num_frames=self._active_frame_count(),
                initial_fps=30.0 if self.data.is_preview else 15.0,
                num_frames_getter=self._active_frame_count,
            )
            self.timestep = self.playback[0]
            self.canonical = self.server.gui.add_checkbox("Canonical", False)
            self.timestep.on_update(self.request_render)
            self.playback_view.on_update(self._on_playback_view)
            self.canonical.on_update(self._on_canonical)

        self._build_camera_controls()
        self._build_color_controls()
        self._build_modal_controls()
        self._build_debug_controls()

        tabs = self.server.gui.add_tab_group()
        with tabs.add_tab("Render", viser.Icon.CAMERA):
            self.render_tab_state = populate_render_tab(
                self.server,
                self.work_dir / "camera_paths",
                self.timestep,
            )
        self.spectrum_window = self.server.gui.add_panel()
        with self.spectrum_window.add_tab("Spectrum"):
            self.spectrum_panel = ModalSpectrumPanel(
                self.server,
                self.data.spectrum,
                on_view_selected=self._on_spectrum_view,
                on_mode_selected=self._set_selected_mode,
                on_component_selected=self._set_selected_component,
                on_normalization_selected=self._set_selected_normalization,
                on_solo_selected=self._solo_mode,
                on_enable_all=self._enable_all_modes,
            )
        self.spectrum_window.float(x=16.0, y=16.0, width=720.0, height=800.0)

    def _on_playback_view(self, event: Any) -> None:
        """Clamp the local frame slider after changing playback view."""

        maximum = self._active_frame_count() - 1
        if hasattr(self.timestep, "max"):
            self.timestep.max = maximum
        if int(self.timestep.value) > maximum:
            self.timestep.value = maximum
        self.request_render(event)

    def _on_canonical(self, event: Any) -> None:
        """Freeze or restore all time/playback controls in canonical mode."""

        disabled = bool(self.canonical.value)
        handles = (self.playback_view, *self.playback)
        if disabled:
            self._canonical_disabled_cache = [
                bool(handle.disabled) for handle in handles
            ]
            for handle in handles:
                handle.disabled = True
        else:
            cached = self._canonical_disabled_cache
            if len(cached) != len(handles):
                cached = [False] * len(handles)
            for handle, was_disabled in zip(handles, cached):
                handle.disabled = was_disabled
        self.request_render(event)

    def _build_modal_controls(self) -> None:
        """Build flow-driven and manual-oscillator modal playback controls."""

        with self.server.gui.add_folder("Modal playback"):
            self.drive = self.server.gui.add_dropdown(
                "Drive",
                options=(DRIVE_MANUAL,) if self.data.is_preview else (DRIVE_FLOW, DRIVE_MANUAL),
                initial_value=DRIVE_MANUAL if self.data.is_preview else DRIVE_FLOW,
            )
            self.motion_scale = self.server.gui.add_slider(
                "Motion scale",
                min=0.0,
                max=1.0,
                step=0.001,
                initial_value=0.04,
            )
            disable_all = self.server.gui.add_button("Turn off all modes")
            modes_by_index: dict[int, dict[str, Any]] = {}
            for display_index, mode_index in enumerate(self.data.frequency_order):
                frequency = float(self.data.frequencies_hz[mode_index])
                enabled = self.server.gui.add_checkbox(
                    f"Mode {display_index} ({frequency:.3f} Hz) enable", True
                )
                gain = self.server.gui.add_slider(
                    f"Mode {display_index} gain",
                    min=0.0,
                    max=5.0,
                    step=0.01,
                    initial_value=1.0,
                )
                phase = self.server.gui.add_slider(
                    f"Mode {display_index} phase",
                    min=-math.pi,
                    max=math.pi,
                    step=0.01,
                    initial_value=0.0,
                )
                modes_by_index[mode_index] = {
                    "enabled": enabled,
                    "gain": gain,
                    "phase": phase,
                    "frequency": frequency,
                }
                enabled.on_update(self._on_mode_enabled)
                gain.on_update(self.request_render)
                phase.on_update(self.request_render)
            self.mode_controls = tuple(
                modes_by_index[index]
                for index in range(len(self.data.frequencies_hz))
            )
            disable_all.on_click(self._disable_all_modes)
            self.drive.on_update(self.request_render)
            self.motion_scale.on_update(self.request_render)

    def _disable_all_modes(self, event: Any) -> None:
        """Disable every manual oscillator mode."""

        for mode in self.mode_controls:
            mode["enabled"].value = False
        self.request_render(event)

    def _solo_mode(self, mode_index: int) -> None:
        """Enable only one greedy mode slot."""

        for index, mode in enumerate(self.mode_controls):
            mode["enabled"].value = index == int(mode_index)
        self._set_selected_mode(int(mode_index))

    def _on_mode_enabled(self, event: Any) -> None:
        """Single-mode playback always selects matching phase and role colors."""
        active = [i for i, mode in enumerate(getattr(self, "mode_controls", ())) if mode["enabled"].value]
        if len(active) == 1:
            self._set_selected_mode(active[0], event)
        else:
            self.request_render(event)

    def _enable_all_modes(self) -> None:
        """Enable every manual oscillator mode."""

        for mode in self.mode_controls:
            mode["enabled"].value = True
        self.request_render(None)

    def _manual_coordinate(self) -> tuple[np.ndarray, float]:
        """Evaluate the old per-mode gain/phase oscillator at Viewer time."""

        time_seconds = float(self.timestep.value) / max(
            float(self.playback[5].value), 1.0e-6
        )
        values: list[complex] = []
        for mode in self.mode_controls:
            amplitude = float(mode["gain"].value) if mode["enabled"].value else 0.0
            phase = (
                2.0 * math.pi * float(mode["frequency"]) * time_seconds
                + float(mode["phase"].value)
            )
            values.append(amplitude * np.exp(1j * phase))
        return np.asarray(values, dtype=np.complex64), float(self.motion_scale.value)

    def _current_coordinate(self) -> tuple[np.ndarray, float]:
        """Select canonical, stored flow-derived, or manual oscillator coordinates."""

        if bool(self.canonical.value):
            return np.zeros(len(self.data.frequencies_hz), dtype=np.complex64), 1.0
        if str(self.drive.value) == DRIVE_MANUAL:
            return self._manual_coordinate()
        return (
            self.data.coordinate(self._active_view_index(), int(self.timestep.value)),
            1.0,
        )

    def _build_color_controls(self) -> None:
        """Build RGB, projected phase, and observation-support color controls."""

        with self.server.gui.add_folder("Gaussian color"):
            self.color_mode = self.server.gui.add_dropdown(
                "Render color mode",
                options=(COLOR_RGB, COLOR_PHASE, COLOR_OBSERVATIONS),
                initial_value=COLOR_RGB,
            )
            self.phase_component = self.server.gui.add_dropdown(
                "Projection direction", options=("u", "v"), initial_value="u"
            )
            self.phase_normalization = self.server.gui.add_dropdown(
                "Amplitude normalization",
                options=("per mode", "entire spectrum"),
                initial_value="per mode",
            )
            self.phase_mode = self.server.gui.add_slider(
                "Phase frequency index",
                min=0,
                max=len(self.data.frequencies_hz) - 1,
                step=1,
                initial_value=self.data.display_indices[0],
            )
            self.phase_frequency = self.server.gui.add_number(
                "Selected frequency (Hz)",
                initial_value=float(self.data.frequencies_hz[0]),
                disabled=True,
            )
            self.observation_mode = self.server.gui.add_slider(
                "Obs count mode index",
                min=0,
                max=len(self.data.frequencies_hz) - 1,
                step=1,
                initial_value=0,
            )
        self.color_mode.on_update(self.request_render)
        self.phase_mode.on_update(self._on_phase_mode)
        self.phase_component.on_update(self._on_phase_component)
        self.phase_normalization.on_update(self._on_phase_normalization)
        self.observation_mode.on_update(self.request_render)

    def _on_phase_mode(self, event: Any) -> None:
        """Translate sorted display index back to immutable greedy mode slot."""

        mode_index = self.data.frequency_order[int(self.phase_mode.value)]
        self._set_selected_mode(mode_index, event)

    def _on_phase_component(self, event: Any) -> None:
        """Synchronize the 3D and spectrum-panel U/V selections."""

        self._set_selected_component(str(self.phase_component.value), event)

    def _on_phase_normalization(self, event: Any) -> None:
        """Synchronize 3D and spectrum-panel amplitude normalization."""

        self._set_selected_normalization(str(self.phase_normalization.value), event)

    def _set_selected_mode(self, mode_index: int, event: Any = None) -> None:
        """Synchronize one greedy mode between all phase displays."""

        index = int(mode_index)
        if not 0 <= index < len(self.data.frequencies_hz):
            raise ValueError("Selected Viewer mode index is outside the result")
        if self._selection_sync:
            return
        self._selection_sync = True
        try:
            display_index = self.data.display_indices[index]
            if int(self.phase_mode.value) != display_index:
                self.phase_mode.value = display_index
            self.phase_frequency.value = float(self.data.frequencies_hz[index])
            if hasattr(self, "spectrum_panel"):
                self.spectrum_panel.set_mode_index(index)
            if getattr(self, "support_mode", None) is not None:
                self.support_mode.value = self.support_mode_labels[index]
        finally:
            self._selection_sync = False
        self.request_render(event)

    def _set_selected_component(self, component: str, event: Any = None) -> None:
        """Synchronize U/V projection selection between phase displays."""

        value = str(component).lower()
        if value not in ("u", "v"):
            raise ValueError(f"Unknown Viewer phase component: {component!r}")
        if self._selection_sync:
            return
        self._selection_sync = True
        try:
            if str(self.phase_component.value) != value:
                self.phase_component.value = value
            if hasattr(self, "spectrum_panel"):
                self.spectrum_panel.set_component(value.upper())
        finally:
            self._selection_sync = False
        self.request_render(event)

    def _set_selected_normalization(
        self, normalization: str, event: Any = None
    ) -> None:
        """Synchronize phase-image brightness policy between both panels."""

        value = str(normalization)
        if value == "entire spectrum" and not self.data.spectrum.full_spectrum_ready:
            value = "per mode"
        if value not in ("per mode", "entire spectrum"):
            raise ValueError(f"Unknown Viewer normalization: {value!r}")
        if self._selection_sync:
            return
        self._selection_sync = True
        try:
            if str(self.phase_normalization.value) != value:
                self.phase_normalization.value = value
            if hasattr(self, "spectrum_panel"):
                self.spectrum_panel.set_amplitude_normalization(value)
        finally:
            self._selection_sync = False
        self.request_render(event)

    def _on_spectrum_view(self, label: str) -> None:
        """Rerender 3D phase colors when the spectrum reference view changes."""

        if self.data.spectrum.view_id != str(label):
            raise ValueError("Spectrum panel did not apply its selected view")
        self._set_selected_normalization(self.data.spectrum.amplitude_normalization)
        self.request_render(None)

    def _current_foreground_colors(self) -> Tensor | None:
        """Return a foreground display-color override, if one is active."""

        mode = str(self.color_mode.value)
        if mode == COLOR_RGB:
            return None
        if mode == COLOR_OBSERVATIONS:
            return self.data.observation_colors(int(self.observation_mode.value))
        if mode != COLOR_PHASE:
            raise ValueError(f"Unknown Viewer color mode: {mode!r}")
        display_index = int(self.phase_mode.value)
        mode_index = self.data.frequency_order[display_index]
        component = 0 if str(self.phase_component.value) == "u" else 1
        return self.data.phase_colors(
            mode_index, component, str(self.phase_normalization.value)
        )

    def _build_debug_controls(self) -> None:
        """Build Gaussian visibility and motion-fill support point controls."""

        gaussian_count = self.data.scene.foreground.count
        step = max(gaussian_count // 200, 1)
        graph_edge_count = len(self.data.graph_edge_gaussian_index)
        graph_edge_step = max(graph_edge_count // 200, 1)
        with self.server.gui.add_folder("Debug points"):
            control_count = len(self.data.control_point_gaussian_index)
            self.show_controls_only = (
                self.server.gui.add_checkbox("Show control points only", False)
                if control_count else None
            )
            self.control_point_size = (
                self.server.gui.add_slider("Control point size", min=0.0002, max=0.008,
                                           step=0.0001, initial_value=0.002)
                if control_count else None
            )
            if control_count:
                self.server.gui.add_markdown(
                    f"**Control points:** {control_count:,}. All controls are shown, colored by geometry component. "
                    "Use **Canonical** to inspect their static distribution."
                )
            self.hide_render = self.server.gui.add_checkbox(
                "Hide Gaussian render", False
            )
            self.hide_background = (
                self.server.gui.add_checkbox("Hide background", False)
                if self.data.scene.background.count > 0
                else None
            )
            self.show_support = self.server.gui.add_checkbox(
                "Show modal points by role", False
            )
            self.support_count = self.server.gui.add_slider(
                "Modal point visible count",
                min=0,
                max=gaussian_count,
                step=step,
                initial_value=min(5000, gaussian_count),
            )
            self.support_size = self.server.gui.add_slider(
                "Modal point size",
                min=0.0002,
                max=0.008,
                step=0.0001,
                initial_value=0.002,
            )
            self.server.gui.add_markdown(self.data.support_legend)
            self.server.gui.add_markdown(
                "Role colors describe **Modal role mode** only. Solo playback selects the same role frequency; "
                "when multiple modes are enabled, displayed motion is their sum."
            )
            self.support_mode_labels = tuple(
                f"Mode {index}: {frequency:.3f} Hz"
                for index, frequency in enumerate(self.data.frequencies_hz)
            )
            self.support_mode = (
                self.server.gui.add_dropdown(
                    "Modal role mode",
                    options=self.support_mode_labels,
                    initial_value=self.support_mode_labels[self.data.frequency_order[int(self.phase_mode.value)]],
                )
                if len(self.support_mode_labels) > 1
                else None
            )
            self.support_filters = tuple(
                self.server.gui.add_checkbox(name.capitalize(), True)
                for name in self.data.support_display_names
            )
            self.server.gui.add_markdown(self.data.graph_legend)
            self.show_component_graph = self.server.gui.add_checkbox(
                "Show component graph", False
            )
            self.component_graph_edge_count = self.server.gui.add_slider(
                "Max visible graph edges",
                min=0,
                max=max(graph_edge_count, 1),
                step=graph_edge_step,
                initial_value=min(20_000, graph_edge_count),
            )
            self.component_graph_line_width = self.server.gui.add_slider(
                "Graph line width",
                min=0.1,
                max=10.0,
                step=0.1,
                initial_value=1.0,
            )
        debug_handles = (
            self.show_controls_only,
            self.control_point_size,
            self.hide_render,
            self.hide_background,
            self.show_support,
            self.support_count,
            self.support_size,
            self.support_mode,
            *self.support_filters,
            self.show_component_graph,
            self.component_graph_edge_count,
            self.component_graph_line_width,
        )
        for handle in debug_handles:
            if handle is not None:
                handle.on_update(self._on_debug_update)

    def _on_debug_update(self, event: Any) -> None:
        """Remove a stale cloud when hidden and refresh the Viewer."""

        only_controls = self._controls_only_enabled()
        if only_controls or not bool(self.show_support.value):
            self._remove_support_cloud()
        if only_controls or not bool(self.show_component_graph.value):
            self._remove_component_graph()
        if not only_controls:
            self._remove_control_cloud()
        self.request_render(event)

    def _controls_only_enabled(self) -> bool:
        handle = getattr(self, "show_controls_only", None)
        return handle is not None and bool(handle.value)

    def _remove_control_cloud(self) -> None:
        if getattr(self, "_control_point_cloud", None) is not None:
            self._control_point_cloud.remove()
            self._control_point_cloud = None

    def _update_control_cloud(self, means: Tensor) -> None:
        indices = torch.as_tensor(self.data.control_point_gaussian_index, device=means.device)
        points = means[indices].detach().cpu().numpy()
        if self._control_point_cloud is None:
            self._control_point_cloud = self.server.scene.add_point_cloud(
                "/debug/control_points", points=points, colors=self.data.control_point_colors,
                point_size=float(self.control_point_size.value),
            )
        else:
            self._control_point_cloud.points = points
            self._control_point_cloud.point_size = float(self.control_point_size.value)

    def _remove_support_cloud(self) -> None:
        """Remove the current Viser support-role point cloud."""

        if self._point_cloud is not None:
            self._point_cloud.remove()
            self._point_cloud = None

    def _remove_component_graph(self) -> None:
        """Remove the component-edge overlay and its cached edge subset."""

        if self._component_graph_handle is not None:
            self._component_graph_handle.remove()
            self._component_graph_handle = None
        self._component_graph_edge_indices = np.empty((0,), dtype=np.int64)
        self._component_graph_color_mode = None

    def _update_component_graph(self, means: Tensor) -> None:
        """Display observed graph edges with trusted colors and untrusted gray."""

        if not bool(self.show_component_graph.value):
            return
        edge_count = len(self.data.graph_edge_gaussian_index)
        selected = _stable_uniform_indices(
            edge_count, max(int(self.component_graph_edge_count.value), 0)
        )
        if len(selected) == 0:
            self._remove_component_graph()
            return
        gaussian_edges = self.data.graph_edge_gaussian_index[selected]
        points = means[torch.as_tensor(gaussian_edges, device=means.device)]
        points_numpy = points.detach().cpu().numpy()
        mode_index = None
        colors = self.data.graph_edge_colors
        if self.data.graph_edge_colors_by_mode is not None:
            mode_index = self.data.frequency_order[int(self.phase_mode.value)]
            colors = self.data.graph_edge_colors_by_mode[mode_index]
        if (
            self._component_graph_handle is None
            or not np.array_equal(selected, self._component_graph_edge_indices)
        ):
            self._remove_component_graph()
            edge_colors = colors[selected]
            self._component_graph_handle = self.server.scene.add_line_segments(
                "/debug/component_graph",
                points=points_numpy,
                colors=np.repeat(edge_colors[:, None, :], 2, axis=1),
                thickness=float(self.component_graph_line_width.value),
                thickness_units="screen",
            )
            self._component_graph_edge_indices = selected
            self._component_graph_color_mode = mode_index
            return
        if mode_index != self._component_graph_color_mode:
            self._component_graph_handle.colors = np.repeat(colors[selected, None, :], 2, axis=1)
            self._component_graph_color_mode = mode_index
        self._component_graph_handle.points = points_numpy
        self._component_graph_handle.thickness = float(
            self.component_graph_line_width.value
        )

    def _update_support_cloud(self, means: Tensor) -> None:
        """Display filtered completed-mode support roles at deformed positions."""

        if not bool(self.show_support.value):
            return
        mode_index = (
            self.support_mode_labels.index(str(self.support_mode.value))
            if self.support_mode is not None
            else 0
        )
        classes = self.data.display_class[mode_index]
        enabled = np.asarray(
            [bool(handle.value) for handle in self.support_filters], dtype=bool
        )
        selected = np.flatnonzero(enabled[classes])
        requested = max(int(self.support_count.value), 0)
        selected = selected[:requested]
        self._remove_support_cloud()
        if len(selected) == 0:
            return
        points = means.detach().cpu().numpy()[selected]
        colors = SUPPORT_COLORS[classes[selected]]
        self._point_cloud = self.server.scene.add_point_cloud(
            "/debug/modal_anchors",
            points=points,
            colors=colors,
            point_size=float(self.support_size.value),
        )

    @staticmethod
    def _camera_pose(camera: ViewerCamera) -> tuple[np.ndarray, np.ndarray]:
        """Return Viser quaternion and position for one calibrated camera."""

        return (
            vtf.SO3.from_matrix(camera.c2w[:3, :3]).wxyz,
            camera.c2w[:3, 3],
        )

    def _apply_camera(self, client: viser.ClientHandle, camera: ViewerCamera) -> None:
        """Jump one connected client to an exact calibrated result camera."""

        wxyz, position = self._camera_pose(camera)
        with client.atomic():
            client.camera.position = position
            client.camera.look_at = self._orbit_center
            client.camera.wxyz = wxyz
            client.camera.fov = camera.fov

    def _build_camera_controls(self) -> None:
        """Build calibrated frustums, camera-jump buttons, and orbit reset."""

        palette = (
            (80, 150, 255),
            (255, 130, 70),
            (95, 200, 120),
            (210, 120, 255),
            (255, 210, 80),
            (80, 220, 220),
        )
        with self.server.gui.add_folder("Cameras"):
            show = self.server.gui.add_checkbox("Show cameras", True)
            reset = self.server.gui.add_button("Reset orbit center")
            for index, camera in enumerate(self.data.cameras):
                wxyz, position = self._camera_pose(camera)
                self._frustums[camera.label] = self.server.scene.add_camera_frustum(
                    f"/result_cameras/{camera.label}",
                    fov=camera.fov,
                    aspect=camera.aspect,
                    scale=self._camera_frustum_scale,
                    color=palette[index % len(palette)],
                    wxyz=wxyz,
                    position=position,
                )
                button = self.server.gui.add_button(f"Go to {camera.label}")

                @button.on_click
                def _go_to(event: Any, target: ViewerCamera = camera) -> None:
                    if event.client is not None:
                        self._apply_camera(event.client, target)
                        self.request_render(event)

            @show.on_update
            def _toggle(event: Any) -> None:
                for handle in self._frustums.values():
                    handle.visible = bool(event.target.value)

            @reset.on_click
            def _reset(event: Any) -> None:
                if event.client is not None:
                    event.client.camera.look_at = self._orbit_center
                    self.request_render(event)

    def _register_clients(self) -> None:
        """Initialize connected clients and render after every camera change."""

        @self.server.on_client_connect
        def _connect(client: viser.ClientHandle) -> None:
            self._last_client = client
            self._apply_camera(client, self.data.cameras[0])

            @client.camera.on_update
            def _camera_update(_: Any) -> None:
                self._last_client = client
                self.request_render(None)

            self.request_render(None)

    def _render_size(self, aspect: float) -> tuple[int, int]:
        """Preserve client aspect while bounding the longest image dimension."""

        maximum = max(64, int(self.viewer_resolution.value))
        if not math.isfinite(aspect) or aspect <= 0.0:
            aspect = 1.0
        if aspect >= 1.0:
            width, height = maximum, max(1, int(round(maximum / aspect)))
        else:
            height, width = maximum, max(1, int(round(maximum * aspect)))
        return width, height

    def _render_camera(self, client: viser.ClientHandle) -> Camera:
        """Convert the current Viser orbit camera to the static renderer contract."""

        width, height = self._render_size(float(client.camera.aspect))
        fov = float(client.camera.fov)
        focal = 0.5 * float(height) / math.tan(0.5 * fov)
        K = torch.tensor(
            ((focal, 0.0, width / 2.0), (0.0, focal, height / 2.0), (0.0, 0.0, 1.0)),
            dtype=torch.float32,
            device=self.data.device,
        )
        pose = vtf.SE3.from_rotation_and_translation(
            vtf.SO3(np.asarray(client.camera.wxyz)),
            np.asarray(client.camera.position),
        ).as_matrix()
        w2c = torch.from_numpy(np.linalg.inv(pose).astype(np.float32)).to(
            self.data.device
        )
        return Camera(
            name="viser",
            role="reference",
            label=None,
            width=width,
            height=height,
            K=K,
            raw_world_to_camera=w2c,
            world_to_camera=w2c,
            camera_model="PINHOLE",
            camera_parameters=(),
            image_relative_path="",
            mask_relative_path="",
            image_sha256="",
            mask_sha256="",
        )

    def request_render(self, event: Any = None) -> None:
        """Coalesce rapid GUI/camera updates into a single render worker."""

        if event is not None and getattr(event, "client", None) is not None:
            self._last_client = event.client
        with self._render_lock:
            self._render_again = True
            if self._render_running:
                return
            self._render_running = True
        threading.Thread(target=self._render_worker, daemon=True).start()

    def _render_worker(self) -> None:
        """Render until all updates that arrived during rasterization are consumed."""

        while True:
            with self._render_lock:
                if not self._render_again:
                    self._render_running = False
                    return
                self._render_again = False
                client = self._last_client
            if client is None:
                time.sleep(0.01)
                continue
            try:
                image = self._render(client)
                self.server.scene.set_background_image(
                    image, format="jpeg", jpeg_quality=90
                )
            except Exception as error:
                print(f"Viewer render failed: {error}")

    @torch.inference_mode()
    def _render(self, client: viser.ClientHandle) -> np.ndarray:
        """Render one frame and refresh deformed point and graph overlays."""

        q, scale = self._current_coordinate()
        means = self.data.deformed_means(q, scale)
        only_controls = self._controls_only_enabled()
        if only_controls:
            self._remove_support_cloud()
            self._remove_component_graph()
            self._update_control_cloud(means)
        else:
            self._remove_control_cloud()
            self._update_support_cloud(means)
            self._update_component_graph(means)
        camera = self._render_camera(client)
        if only_controls or bool(self.hide_render.value):
            return np.full((camera.height, camera.width, 3), 255, dtype=np.uint8)
        colors = self._current_foreground_colors()
        rendered = self.data.scene.render_deformed(
            camera,
            means,
            foreground_colors=colors,
            include_background=(
                self.hide_background is None or not bool(self.hide_background.value)
            ),
        )["rgb"]
        return (
            rendered.clamp(0.0, 1.0).mul(255.0).round().byte().cpu().numpy()
        )

    def wait(self) -> None:
        """Keep the Viewer server alive until the user interrupts the command."""

        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            self.server.stop()


def run_modal_viewer(
    *,
    result_dir: str | Path,
    work_dir: str | Path,
    host: str = "0.0.0.0",
    port: int = 8080,
    viewer_resolution: int = 2048,
    preview: bool = False,
) -> None:
    """Load one complete modal result and run its full Viser interface."""

    # Load the CUDA backend synchronously. Deferring this to a render worker
    # leaves a connected but blank Viewer when compiler activation fails.
    _load_gsplat_rasterization()
    data = ModalViewerData(result_dir, preview=preview)
    viewer = ModalViserViewer(
        data,
        work_dir=work_dir,
        host=host,
        port=port,
        viewer_resolution=viewer_resolution,
    )
    viewer.wait()


__all__ = [
    "ModalViewerData",
    "ModalViserViewer",
    "ViewerCamera",
    "run_modal_viewer",
]
