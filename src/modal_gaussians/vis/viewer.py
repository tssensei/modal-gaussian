"""Artifact-native Viser playback for a completed modal Gaussian result."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from modal_gaussians.common.scene_store import resolve_path
import threading
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor
import viser
import viser.transforms as vtf

from modal_gaussians.motion.common.projection import projection_jacobian
from modal_gaussians.geometry.scene import Camera, _load_gsplat_rasterization, cameras_from_scene_manifest
from modal_gaussians.vis.playback_panel import add_gui_playback_group
from modal_gaussians.vis.inputs import ViewerInput, load_viewer_input
from modal_gaussians.vis.projections import ViewerProjections
from modal_gaussians.vis.render_panel import populate_render_tab
from modal_gaussians.vis.spectrum import ModalSpectrumPanel, SpectrumComparisonController, _hsv_rgb


DRIVE_FLOW = "flow-derived coordinates"
DRIVE_MANUAL = "manual oscillator"
COLOR_RGB = "rgb"
COLOR_PHASE = "modal phase"
COLOR_OBSERVATIONS = "obs count"

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


def _rotate_gaussian_quaternions(base: Tensor, rotation_vectors: Tensor) -> Tensor:
    """Left-compose world rotation vectors with static wxyz orientations."""
    angles = torch.linalg.vector_norm(rotation_vectors, dim=-1, keepdim=True)
    scalar = torch.cos(angles / 2)
    vector = 0.5 * torch.sinc(angles / (2 * math.pi)) * rotation_vectors
    real, imaginary = base[:, :1], base[:, 1:]
    rotated = torch.cat((scalar * real - (vector * imaginary).sum(dim=-1, keepdim=True),
                         scalar * imaginary + real * vector
                         + torch.linalg.cross(vector, imaginary, dim=-1)), dim=-1)
    rotated = torch.nn.functional.normalize(rotated, dim=-1)
    return torch.where(angles == 0, base, rotated)


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


@dataclass(frozen=True)
class ViewerCamera:
    """Expose one calibrated result camera in the Viser convention."""

    label: str
    camera: Camera
    c2w: np.ndarray
    fov: float
    aspect: float

    @classmethod
    def from_camera(cls, camera: Camera, label: str | None = None) -> ViewerCamera:
        return cls(
            label=camera.name if label is None else label, camera=camera,
            c2w=np.linalg.inv(camera.world_to_camera.detach().cpu().numpy()).astype(np.float64),
            fov=2.0 * math.atan(0.5 * camera.height / float(camera.K[1, 1])),
            aspect=camera.width / camera.height,
        )


def _completed_mode_display_roles(manifest, arrays):
    if manifest.get('version') not in (18, 17, 20):
        raise ValueError('Unsupported model for Viewer')
    support, phi = np.asarray(arrays['support_class']), np.asarray(arrays['phi'])
    if (phi.ndim != 3 or phi.shape[2] != 3 or support.shape != phi.shape[:2]
            or support.dtype.kind not in 'iu' or np.any((support < 0) | (support > 3))):
        raise ValueError('Invalid support classes')
    if manifest.get('version') == 20:
        return np.asarray([2,0,1,3], np.int8)[support], tuple('inherited ' + name for name in NEURAL_SUPPORT_DISPLAY_NAMES) + ('inherited propagated',), (
            '**Inherited mode sources:** blue = supervised source | green = inferred source | '
            'purple = unresolved | yellow = donor source. Control fields were refined by RGB; modal observations are inherited.')
    return np.asarray([2,0,1,3], np.int8)[support], NEURAL_SUPPORT_DISPLAY_NAMES + ('propagated',), (
        '**Role colors:** directly supervised = blue | structure inferred = green | '
        'unresolved = purple | propagated = yellow')


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


def _viewer_cameras(result: ViewerInput) -> tuple[ViewerCamera, ...]:
    """Resolve the ordered result views to exact static-scene cameras."""

    if result.scene.manifest is None:
        raise ValueError("Modal-result static scene has no manifest")
    by_name = {
        camera.name: camera
        for camera in cameras_from_scene_manifest(result.scene.manifest)
    }
    cameras: list[ViewerCamera] = []
    for view in result.views:
        name = str(view["camera_name"])
        camera = by_name.get(name)
        if camera is None:
            raise ValueError(f"Result camera {name!r} is absent from static scene")
        cameras.append(ViewerCamera.from_camera(camera, str(view["label"])))
    if len({camera.label for camera in cameras}) != len(cameras):
        raise ValueError("Modal-result Viewer camera labels are not unique")
    return tuple(cameras)


def _observation_counts(completed, gaussian_count):
    observed = np.asarray(completed.arrays['observation_view_mask'])
    if observed.dtype != bool or observed.shape != (len(completed.manifest['modes']), gaussian_count, len(completed.manifest['views'])):
        raise ValueError('Observation mask must be boolean [K,G,V]')
    return observed.sum(axis=2, dtype=np.int16)


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

    def __init__(self, result_dir: str | Path, device: str = "cuda", *,
                 coordinates=None, work_dir=None, with_spectrum=True) -> None:
        requested_device = torch.device(device)
        if requested_device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Modal Gaussian Viser requires a CUDA device")
        self.result = load_viewer_input(result_dir, coordinates=coordinates)
        self.device = requested_device
        self.scene = self.result.scene.to(self.device)
        self.result.scene = self.scene
        self.cameras = _viewer_cameras(self.result)
        self.camera_by_label = {camera.label: camera for camera in self.cameras}
        modes = self.result.modes
        self.frequencies_hz = np.asarray([m.frequency for m in modes], dtype=np.float64)
        self.phi = torch.from_numpy(np.stack([m.artifact.arrays["phi"][m.slot] for m in modes])).to(self.device)
        self.rotation = None
        if all(m.artifact.rotation is not None for m in modes):
            rotation = np.stack([m.artifact.rotation[m.slot] for m in modes])
            if rotation.dtype != np.complex64 or rotation.shape != tuple(self.phi.shape):
                raise ValueError("Viewer rotation mode domain differs")
            self.rotation = torch.from_numpy(rotation).to(self.device)
        elif any(m.artifact.rotation is not None for m in modes):
            raise ValueError("Viewer modes have incompatible rotation capabilities")
        fitted = self.result.coordinates
        self.coordinates = None if fitted is None else np.asarray(
            fitted.coordinates[:, self.result.coordinate_columns], dtype=np.complex64)
        roles, counts = [], []
        for mode in modes:
            artifact = mode.artifact
            display, names, legend = _completed_mode_display_roles(artifact.manifest, artifact.arrays)
            if roles and names != self.support_display_names:
                raise ValueError("Viewer support roles differ across modes")
            self.support_display_names, self.support_legend = names, legend
            roles.append(display[mode.slot])
            counts.append(_observation_counts(artifact, self.scene.foreground.count)[mode.slot])
        self.display_class, self.observation_counts = np.stack(roles), np.stack(counts)
        self.select_mode(0)
        self._load_graph_display(0)
        self.projections = None
        self.spectrum = None
        if with_spectrum and all("selected_modal_supervision" in m.artifact.manifest or m.artifact.manifest.get("version") in (17, 20) for m in modes):
            self.projections = ViewerProjections(self.result, work_dir)
            self.spectrum = SpectrumComparisonController(self.result, projections=self.projections)
        self.gpu_lock = self.projections.gpu_lock if self.projections else threading.RLock()

    def select_mode(self, index):
        """Select diagnostics without assuming shared control-point layouts."""
        mode = self.result.modes[index]
        arrays = mode.artifact.arrays
        self.control_point_gaussian_index = _control_point_gaussian_indices(arrays, self.scene.foreground.count)
        self.control_point_colors = (_component_colors(arrays.get("g_component_index", np.zeros(self.scene.foreground.count, np.int64))[self.control_point_gaussian_index])
            if len(self.control_point_gaussian_index) else np.empty((0, 3), np.float32))
        self.control_positions = np.asarray(arrays.get("c_positions", np.empty((0, 3), np.float32)))
        self.control_displacement = mode.artifact.control_displacement
        self._control_slot = mode.slot
        if self.control_displacement is not None and (
                self.control_displacement[mode.slot].shape != self.control_positions.shape
                or self.control_positions.shape != (len(self.control_point_gaussian_index), 3)):
            raise ValueError("Viewer control displacement domain differs")

    def has_coordinates(self, view_index):
        return self.result.views[view_index]["label"] in self.result.coordinate_views

    def _load_graph_display(self, mode_index: int | None = None) -> None:
        """Select geometry diagnostics belonging to the completed-mode method."""

        mode_index = 0 if mode_index is None else mode_index
        completed = self.result.modes[mode_index].artifact
        self.structure_graph = None
        self.reference_graph_points = None
        if completed.manifest.get("version") == 20:
            self.reference_graph_points = completed.arrays["reference_points"]
            self.graph_edge_gaussian_index = completed.arrays["reference_edges"]
            self.graph_edge_colors = np.tile(np.array([[.3, .7, .9]], np.float32), (len(self.graph_edge_gaussian_index), 1))
            self.graph_mode_index = mode_index
            self.graph_edge_colors_by_mode = None
            self.graph_legend = '**Fixed motion reference graph:** original canonical nodes and unchanged Gaussian order.'
            return
        self.graph_edge_gaussian_index, self.graph_edge_colors = _neural_graph_display(
            completed.arrays, self.scene.foreground.count)
        self.graph_mode_index = mode_index
        self.graph_edge_colors_by_mode = None
        self.graph_legend = '**Geometry graph:** distinct colors = connected components.'

    def coordinate(self, view_index: int, local_frame: int) -> np.ndarray:
        """Return one stored flow-derived complex coordinate vector."""

        if self.coordinates is None:
            raise ValueError("This model has no video coordinates; use the manual oscillator")
        label = self.result.views[view_index]["label"]
        if label not in self.result.coordinate_views:
            raise ValueError(f"View {label!r} has no fitted coordinates")
        view = self.result.coordinate_views[label]
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

    def deformed_quaternions(self, q: np.ndarray, scale: float = 1.0) -> Tensor | None:
        """Derive orientations from the same modal coefficients as the centers."""
        if self.rotation is None:
            return None
        values = np.array(q, dtype=np.complex64, copy=True)
        if values.shape != (len(self.frequencies_hz),):
            raise ValueError("Viewer modal coordinate has the wrong mode count")
        if not np.isfinite(values).all() or not math.isfinite(scale):
            raise ValueError("Viewer modal coordinate and scale must be finite")
        if scale == 0 or not np.any(values):
            return None
        coefficients = torch.from_numpy(values).to(self.device)
        angles = torch.einsum("k,kgc->gc", coefficients.real, self.rotation.real)
        angles -= torch.einsum("k,kgc->gc", coefficients.imag, self.rotation.imag)
        # ponytail: blended control rotation is a kinematic prior; infer local
        # rotations from center motion if this approximation fails visually.
        return _rotate_gaussian_quaternions(
            self.scene.foreground.active()["quaternions"], float(scale) * angles)

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
            "nij,nj->ni", jacobian.to(self.phi.dtype), self.phi[mode_index]
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

    def control_phase_colors(self, mode_index: int, component_index: int,
                             normalization: str, camera: Camera | None = None) -> np.ndarray:
        """Project raw control translations at rest, before Gaussian interpolation.

        An orbit camera uses the artifact's phase and control-amplitude p99.
        Without a camera, use the Spectrum reference's alpha and brightness.
        Neither path depends on oscillator time, gain or deformed positions.
        """
        if self.control_displacement is None:
            raise ValueError("This artifact has no runtime control displacement")
        slot = getattr(self, "_control_slot", mode_index)
        if not 0 <= slot < len(self.control_displacement) or component_index not in (0, 1):
            raise ValueError("Control modal mode/component is out of range")
        alpha, magnitude_hi = 1.0 + 0.0j, None
        if camera is None:
            label, alpha, identifiable, magnitude_hi = self.spectrum.current_modal_phase_display_context(
                mode_index, component_index, normalization)
            if not identifiable:
                return np.zeros((len(self.control_positions), 3), dtype=np.float32)
            camera = self.camera_by_label[label].camera
        jacobian, projectable = projection_jacobian(
            self.control_positions, camera.K.detach().cpu().numpy(),
            camera.world_to_camera.detach().cpu().numpy(), camera.radial_distortion)
        projected = alpha * np.einsum(
            "cj,cj->c", jacobian[:, component_index], self.control_displacement[slot])
        if magnitude_hi is None:
            magnitude_hi = float(np.percentile(np.abs(projected[projectable]), 99)) if projectable.any() else 1.0
        return _hsv_rgb(projected, magnitude_hi)

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
        self.work_dir = resolve_path(work_dir)
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

        return self.data.result.views

    def _active_view_index(self) -> int:
        """Resolve the current playback label to its result view index."""

        label = str(self.playback_view.value)
        labels = [str(view["label"]) for view in self.views]
        if label not in labels:
            raise ValueError(f"Unknown playback view: {label!r}")
        return labels.index(label)

    def _active_frame_count(self) -> int:
        """Return the frame count of the selected playback view."""

        label = self.views[self._active_view_index()]["label"]
        view = self.data.result.coordinate_views.get(label)
        return 1800 if view is None else int(view["frame_count"])

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
            if self.data.coordinates is None:
                self.server.gui.add_markdown("**Mode preview — manual oscillation. Video coordinates have not been fitted.**")
            self.playback_view = self.server.gui.add_dropdown(
                "Playback view",
                options=tuple(str(view["label"]) for view in self.views),
                initial_value=next(iter(self.data.result.coordinate_views), str(self.views[0]["label"])),
            )
            self.playback = add_gui_playback_group(
                self.server,
                num_frames=self._active_frame_count(),
                initial_fps=self.data.result.coordinate_views.get(str(self.playback_view.value), {}).get('fps_hz', 30.),
                max_fps=max(60., max((v.get('fps_hz', 30.) for v in self.data.result.coordinate_views.values()), default=30.)),
                num_frames_getter=self._active_frame_count,
            )
            self.timestep = self.playback[0]
            self.follow_camera = self.server.gui.add_checkbox("Follow recorded frame camera", False)
            self.follow_camera.on_update(self._follow_frame_camera)
            self.timestep.on_update(self._follow_frame_camera)
            self.playback_view.on_update(self._follow_frame_camera)
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
        if self.data.spectrum is None:
            return
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

        available = self.data.has_coordinates(self._active_view_index())
        self.drive.options = (DRIVE_FLOW, DRIVE_MANUAL) if available else (DRIVE_MANUAL,)
        if not available:
            self.drive.value = DRIVE_MANUAL
        maximum = self._active_frame_count() - 1
        if hasattr(self.timestep, "max"):
            self.timestep.max = maximum
        if int(self.timestep.value) > maximum:
            self.timestep.value = maximum
        if hasattr(self, 'playback'):
            self.playback[-1].value = self.data.result.coordinate_views.get(str(self.playback_view.value), {}).get('fps_hz', 30.)
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
                options=(DRIVE_FLOW, DRIVE_MANUAL) if self.data.has_coordinates(self._active_view_index()) else (DRIVE_MANUAL,),
                initial_value=DRIVE_FLOW if self.data.has_coordinates(self._active_view_index()) else DRIVE_MANUAL,
            )
            self.motion_scale = self.server.gui.add_slider(
                "Motion scale",
                min=0.0,
                max=1.0,
                step=0.001,
                initial_value=0.04,
            )
            self.rotate_ellipsoids = self.server.gui.add_checkbox(
                "Rotate Gaussian ellipsoids", self.data.rotation is not None,
                disabled=self.data.rotation is None,
            )
            self.rotate_ellipsoids.on_update(self.request_render)
            disable_all = self.server.gui.add_button("Turn off all modes")
            self.mode_controls = []
            for mode_index, frequency in enumerate(self.data.frequencies_hz):
                enabled = self.server.gui.add_checkbox(
                    f"Mode {mode_index} ({frequency:.3f} Hz) enable", True
                )
                gain = self.server.gui.add_slider(
                    f"Mode {mode_index} gain",
                    min=0.0,
                    max=5.0,
                    step=0.01,
                    initial_value=1.0,
                )
                phase = self.server.gui.add_slider(
                    f"Mode {mode_index} phase",
                    min=-math.pi,
                    max=math.pi,
                    step=0.01,
                    initial_value=0.0,
                )
                self.mode_controls.append({
                    "enabled": enabled,
                    "gain": gain,
                    "phase": phase,
                    "frequency": float(frequency),
                })
                enabled.on_update(self._on_mode_enabled)
                gain.on_update(self.request_render)
                phase.on_update(self.request_render)
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
            if any(m.artifact.manifest.get("version") == 20 for m in self.data.result.modes):
                self.server.gui.add_markdown("Observation counts and roles are inherited from the original mode sources; control fields have RGB refinement, with inherited modal observations.")
            self.color_mode = self.server.gui.add_dropdown(
                "Render color mode",
                options=((COLOR_RGB, COLOR_PHASE, COLOR_OBSERVATIONS) if self.data.spectrum is not None
                         else (COLOR_RGB, COLOR_OBSERVATIONS)),
                initial_value=COLOR_RGB,
            )
            self.phase_component = self.server.gui.add_dropdown(
                "Projection direction", options=("u", "v"), initial_value="u"
            )
            self.phase_normalization = self.server.gui.add_dropdown(
                "Amplitude normalization",
                options=("per mode", "all saved modes"),
                initial_value="per mode",
            )
            self.phase_mode = self.server.gui.add_slider(
                "Phase frequency index",
                min=0,
                max=len(self.data.frequencies_hz) - 1,
                step=1,
                initial_value=0,
            )
            self.phase_frequency = self.server.gui.add_number(
                "Selected frequency (Hz)",
                initial_value=float(self.data.frequencies_hz[0]),
                disabled=True,
            )
        self.color_mode.on_update(self.request_render)
        self.phase_mode.on_update(self._on_phase_mode)
        self.phase_component.on_update(self._on_phase_component)
        self.phase_normalization.on_update(self._on_phase_normalization)

    def _on_phase_mode(self, event: Any) -> None:
        """Use the selected input mode for every diagnostic panel."""

        mode_index = int(self.phase_mode.value)
        self._set_selected_mode(mode_index, event)

    def _on_phase_component(self, event: Any) -> None:
        """Synchronize the 3D and spectrum-panel U/V selections."""

        self._set_selected_component(str(self.phase_component.value), event)

    def _on_phase_normalization(self, event: Any) -> None:
        """Synchronize 3D and spectrum-panel amplitude normalization."""

        self._set_selected_normalization(str(self.phase_normalization.value), event)

    def _set_selected_mode(self, mode_index: int, event: Any = None) -> None:
        """Synchronize the selected mode between all displays."""

        index = int(mode_index)
        if not 0 <= index < len(self.data.frequencies_hz):
            raise ValueError("Selected Viewer mode index is outside the result")
        if self._selection_sync:
            return
        self._selection_sync = True
        try:
            with self.data.gpu_lock:
                self.data.select_mode(index)
            self._remove_control_cloud()
            if getattr(self, "show_controls_only", None) is not None:
                self.show_controls_only.disabled = not len(self.data.control_point_gaussian_index)
                if self.show_controls_only.disabled:
                    self.show_controls_only.value = False
                self.control_color_mode.disabled = self.data.control_displacement is None
                self.control_projection.disabled = self.data.control_displacement is None
            if int(self.phase_mode.value) != index:
                self.phase_mode.value = index
            self.phase_frequency.value = float(self.data.frequencies_hz[index])
            if hasattr(self, "spectrum_panel"):
                self.spectrum_panel.set_mode_index(index)
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
        if value not in ("per mode", "all saved modes"):
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
        self.request_render(None)

    def _current_foreground_colors(self) -> Tensor | None:
        """Return a foreground display-color override, if one is active."""

        mode = str(self.color_mode.value)
        if mode == COLOR_RGB:
            return None
        if mode == COLOR_OBSERVATIONS:
            return self.data.observation_colors(int(self.phase_mode.value))
        if mode != COLOR_PHASE:
            raise ValueError(f"Unknown Viewer color mode: {mode!r}")
        mode_index = int(self.phase_mode.value)
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
            has_control_modes = self.data.control_displacement is not None
            self.show_controls_only = self.server.gui.add_checkbox(
                "Show control points only", False, disabled=not control_count)
            self.control_point_size = self.server.gui.add_slider(
                "Control point size", min=0.0002, max=0.008, step=0.0001, initial_value=0.002)
            self.control_color_mode = self.server.gui.add_dropdown(
                "Control point color", options=("geometry component", "modal shape"),
                initial_value="modal shape" if has_control_modes else "geometry component",
                disabled=not has_control_modes)
            self.control_projection = self.server.gui.add_dropdown(
                "Control projection", options=("current camera", "spectrum view"),
                initial_value="current camera", disabled=not has_control_modes)
            self.server.gui.add_markdown(
                "Control points and their colors follow the selected frequency's original model. "
                "Current camera uses control-amplitude p99; Spectrum view uses saved alpha and image brightness.")
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
                "All diagnostics use **Selected frequency (Hz)**. Solo playback selects the same frequency; "
                "when multiple modes are enabled, displayed motion is their sum."
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
            self.control_color_mode,
            self.control_projection,
            self.hide_render,
            self.hide_background,
            self.show_support,
            self.support_count,
            self.support_size,
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

    def _update_control_cloud(self, means: Tensor, camera: Camera) -> None:
        indices = torch.as_tensor(self.data.control_point_gaussian_index, device=means.device)
        points = means[indices].detach().cpu().numpy()
        colors = self.data.control_point_colors
        if self.control_color_mode.value == "modal shape":
            colors = self.data.control_phase_colors(
                int(self.phase_mode.value),
                0 if self.phase_component.value == "u" else 1,
                str(self.phase_normalization.value),
                camera if self.control_projection.value == "current camera" else None)
        if self._control_point_cloud is None:
            self._control_point_cloud = self.server.scene.add_point_cloud(
                "/debug/control_points", points=points, colors=colors,
                point_size=float(self.control_point_size.value),
            )
        else:
            self._control_point_cloud.points = points
            self._control_point_cloud.colors = colors
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
        index = int(self.phase_mode.value)
        if index != self.data.graph_mode_index:
            self.data._load_graph_display(index)
            edge_count = len(self.data.graph_edge_gaussian_index)
            self.component_graph_edge_count.max = max(edge_count, 1)
            self.component_graph_edge_count.step = max(edge_count // 200, 1)
            self.component_graph_edge_count.value = min(int(self.component_graph_edge_count.value), edge_count)
            self._remove_component_graph()
        edge_count = len(self.data.graph_edge_gaussian_index)
        selected = _stable_uniform_indices(
            edge_count, max(int(self.component_graph_edge_count.value), 0)
        )
        mode_index = None
        colors = self.data.graph_edge_colors
        if self.data.graph_edge_colors_by_mode is not None:
            mode_index = int(self.phase_mode.value)
            source_slot = self.data.result.modes[mode_index].slot
            colors = self.data.graph_edge_colors_by_mode[source_slot]
        self._display_graph_edges(means, selected, colors, mode_index)

    def _display_graph_edges(self, means: Tensor, selected: np.ndarray,
                             colors: np.ndarray, mode_index: int | None = None) -> None:
        """Update the shared edge overlay after each viewer selects its edges/colors."""
        if len(selected) == 0:
            self._remove_component_graph()
            return
        gaussian_edges = self.data.graph_edge_gaussian_index[selected]
        reference = getattr(self.data, "reference_graph_points", None)
        if reference is None:
            points = means[torch.as_tensor(gaussian_edges, device=means.device)]
            points_numpy = points.detach().cpu().numpy()
        else:
            points_numpy = reference[gaussian_edges]
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
        mode_index = int(self.phase_mode.value)
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

    def _recorded_frame_camera(self):
        from modal_gaussians.coordinates.sequences import frame_camera
        view = self.data.result.coordinate_views.get(str(self.playback_view.value))
        if view is None:
            return self.data.camera_by_label[str(self.playback_view.value)].camera
        cameras = {c.name: c for c in cameras_from_scene_manifest(self.data.scene.manifest)}
        return frame_camera(cameras, view, min(int(self.timestep.value), view['frame_count'] - 1))

    def _follow_frame_camera(self, event=None):
        if self.follow_camera.value:
            camera = ViewerCamera.from_camera(self._recorded_frame_camera())
            for client in self.server.get_clients().values():
                self._apply_camera(client, camera)
        self.request_render(event)

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
                with getattr(self.data, "gpu_lock", self._render_lock):
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
        camera = self._render_camera(client)
        if self.follow_camera.value:
            from modal_gaussians.coordinates.rendering import scaled_camera
            saved = self._recorded_frame_camera()
            camera = scaled_camera(saved, min(1., int(self.viewer_resolution.value) / max(saved.width, saved.height))).to(self.data.device)
        only_controls = self._controls_only_enabled()
        if only_controls:
            self._remove_support_cloud()
            self._remove_component_graph()
            self._update_control_cloud(means, camera)
        else:
            self._remove_control_cloud()
            self._update_support_cloud(means)
            self._update_component_graph(means)
        if only_controls or bool(self.hide_render.value):
            return np.full((camera.height, camera.width, 3), 255, dtype=np.uint8)
        colors = self._current_foreground_colors()
        rendered = self.data.scene.render_deformed(
            camera,
            means,
            foreground_quaternions=(
                self.data.deformed_quaternions(q, scale) if self.rotate_ellipsoids.value else None
            ),
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
            pass
        finally:
            if getattr(self.data, "spectrum", None) is not None:
                self.data.spectrum.close()
            self.server.stop()


def run_modal_viewer(
    *,
    result_dir: str | Path,
    work_dir: str | Path,
    host: str = "0.0.0.0",
    port: int = 8080,
    viewer_resolution: int = 2048,
    coordinates: str | Path | None = None,
    with_spectrum: bool = True,
) -> None:
    """Load one complete modal result and run its full Viser interface."""

    # Load the CUDA backend synchronously. Deferring this to a render worker
    # leaves a connected but blank Viewer when compiler activation fails.
    _load_gsplat_rasterization()
    data = ModalViewerData(result_dir, coordinates=coordinates, work_dir=work_dir,
                           with_spectrum=with_spectrum)
    viewer = ModalViserViewer(
        data,
        work_dir=work_dir,
        host=host,
        port=port,
        viewer_resolution=viewer_resolution,
    )
    if data.spectrum is not None:
        def projection_updated():
            viewer.spectrum_panel._refresh()
            viewer.request_render()
        data.spectrum.start(projection_updated)
    viewer.wait()


__all__ = [
    "ModalViewerData",
    "ModalViserViewer",
    "ViewerCamera",
    "run_modal_viewer",
]
