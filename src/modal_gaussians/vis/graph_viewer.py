"""Static geometry inspection without frequencies, learned modes or network replay."""
from __future__ import annotations

import json
import math
from pathlib import Path
from modal_gaussians.scene_store import resolve_path

import numpy as np
import torch

from modal_gaussians.iteration_cache import identity, load_entry
from modal_gaussians.motion.neural.geometry_graph import GeometryGraph
from modal_gaussians.static import load_static_scene, cameras_from_scene_manifest, _load_gsplat_rasterization
from modal_gaussians.vis.viewer import ModalViserViewer, ViewerCamera, _component_colors, _stable_uniform_indices


def _candidate_graph_display(graph, status=None):
    """Keep pruned candidates visible in white, with retained component colors."""
    edges = np.asarray(graph.candidate_edge_index)
    if (edges.ndim != 2 or edges.shape[1] != 2 or not np.issubdtype(edges.dtype, np.integer)
            or np.any(edges < 0) or np.any(edges >= len(graph.points))):
        raise ValueError("Invalid candidate edge indices")
    count = len(graph.points)
    candidates = np.sort(edges, axis=1).astype(np.int64)
    retained = np.sort(graph.edge_index, axis=1).astype(np.int64)
    removed = ~np.isin(candidates[:, 0] * count + candidates[:, 1],
                       retained[:, 0] * count + retained[:, 1])
    colors = np.round(255 * _component_colors(graph.component_index[edges[:, 0]])).astype(np.uint8)
    colors[removed] = 255
    if status is not None:
        status = np.asarray(status)
        if (status.shape != (len(edges),) or status.dtype != np.uint8
                or np.any(status > 2) or not np.array_equal(status != 0, removed)):
            raise ValueError("Candidate status does not match retained graph edges")
        colors[status == 2] = 90
    return edges, colors, removed


def _candidate_edge_subset(removed, maximum, show_retained=True, show_removed=True):
    """Balance the display budget so rare removed edges remain visible."""
    kept = np.flatnonzero(~removed) if show_retained else np.empty(0, dtype=np.int64)
    cut = np.flatnonzero(removed) if show_removed else np.empty(0, dtype=np.int64)
    maximum = max(int(maximum), 0)
    cut_count = min(len(cut), (maximum + 1) // 2)
    kept_count = min(len(kept), maximum - cut_count)
    cut_count = min(len(cut), maximum - kept_count)
    return np.sort(np.concatenate((kept[_stable_uniform_indices(len(kept), kept_count)],
                                   cut[_stable_uniform_indices(len(cut), cut_count)])))


def _soft_weight_colors(colors, factors):
    factors = np.asarray(factors)
    if (factors.shape != (len(colors),) or not np.isfinite(factors).all()
            or np.any(factors <= 0) or np.any(factors > 1)):
        raise ValueError("Soft edge factors must be finite in (0,1]")
    result = colors.copy()
    weak = factors < 1
    result[weak] = np.rint(255 * (.15 + .70 * factors[weak, None])).astype(np.uint8)
    return result


def _similarity_edge_subset(status, maximum, show_retained, show_rejected, show_unsupported):
    groups = [np.flatnonzero(status == value) for value, show in
              enumerate((show_retained, show_rejected, show_unsupported)) if show]
    groups = [group for group in groups if len(group)]
    if not groups:
        return np.empty(0, dtype=np.int64)
    budget = max(int(maximum), 0)
    selected = []
    # Allocate small groups first, then share the remaining display budget.
    groups.sort(key=len)
    for index, group in enumerate(groups):
        count = min(len(group), (budget + len(groups) - index - 1) // (len(groups) - index))
        selected.append(group[_stable_uniform_indices(len(group), count)])
        budget -= count
    return np.sort(np.concatenate(selected))


class GraphViewerData:
    def __init__(self, scene_dir, graph_dir, device="cuda"):
        self.device = torch.device(device)
        self.scene = load_static_scene(scene_dir, self.device).eval()
        self.scene.requires_grad_(False)
        path = resolve_path(graph_dir, strict=True)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        self.is_similarity_graph = manifest.get("format") == "modal_gaussians.modal_similarity_graph"
        self.is_soft_graph = self.is_similarity_graph and manifest["config"]["modal_similarity"].get("soft_weights", False)
        self.graph_edge_factor = None
        status = None
        if manifest.get("format") in ("modal_gaussians.modal_gradient_graph", "modal_gaussians.modal_similarity_graph"):
            if manifest.get("version") != 1 or manifest.get("graph_file") != "graph.npz":
                raise ValueError("Unsupported modal graph artifact")
            if manifest.get("foreground_identity") != self.scene.manifest["foreground_identity"]:
                raise ValueError("Modal graph does not belong to this scene's foreground")
            with np.load(path / "graph.npz", allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            if self.is_similarity_graph:
                if manifest.get("evidence_file") != "edge_evidence.npz":
                    raise ValueError("Unsupported modal-similarity evidence file")
                with np.load(path / "edge_evidence.npz", allow_pickle=False) as archive:
                    status = archive["candidate_status"]
                    if self.is_soft_graph:
                        self.graph_edge_factor = archive["candidate_edge_factor"]
            self.graph_config = manifest["config"]
        else:
            contract = manifest.get("contract", {})
            if (contract.get("implementation") != "neural_geometry_cache_v1"
                    or contract.get("foreground") != self.scene.manifest["foreground_identity"]):
                raise ValueError("Geometry cache does not belong to this scene's foreground")
            if path.name != identity(contract):
                raise ValueError("Geometry cache directory differs from its contract identity")
            arrays = load_entry(path.parent, contract)
            if arrays is None:
                raise FileNotFoundError(f"Geometry cache is missing: {path}")
            self.graph_config = contract["config"]
        self.graph = GeometryGraph.from_dict(arrays)
        means = self.scene.foreground.active()["means"].detach().cpu().numpy()
        if not np.array_equal(self.graph.points, means):
            raise ValueError("Geometry points differ from the scene's foreground order or positions")
        self.graph_edge_gaussian_index, self.graph_edge_colors, self.graph_edge_removed = _candidate_graph_display(self.graph, status)
        if self.is_soft_graph:
            self.graph_edge_colors = _soft_weight_colors(self.graph_edge_colors, self.graph_edge_factor)
        self.graph_edge_status = status
        self.graph_edge_colors_by_mode = None
        self.point_colors = _component_colors(self.graph.component_index)
        cameras = cameras_from_scene_manifest(self.scene.manifest)
        selected = [camera for camera in cameras if camera.role == "reference"] or list(cameras[:1])
        if not selected:
            raise ValueError("Static graph viewer requires a calibrated scene camera")
        self.cameras = tuple(ViewerCamera(
            label=camera.name, camera=camera,
            c2w=np.linalg.inv(camera.world_to_camera.detach().cpu().numpy()).astype(np.float64),
            fov=2 * math.atan(0.5 * camera.height / float(camera.K[1, 1])),
            aspect=camera.width / camera.height,
        ) for camera in selected)


class GraphViserViewer(ModalViserViewer):
    """Reuse camera navigation, render scheduling and graph overlays only."""

    def _build_gui(self):
        gui = self.server.gui
        graph = self.data.graph
        config = self.data.graph_config
        similarity = self.data.is_similarity_graph
        soft = self.data.is_soft_graph
        description = "Removed candidate edges are white. The display budget samples both groups."
        if soft:
            factor = self.data.graph_edge_factor
            description = (f"All candidate connections remain. {(factor < 1).sum():,} downweighted edges "
                "are gray; darker means a smaller fraction of the original weight. "
                "Unchanged edges keep component colors. Edges without reliable support are also downweighted. "
                "Control placement and coverage use geometric paths; soft weights attenuate interpolation.")
        elif similarity:
            counts = np.bincount(self.data.graph_edge_status, minlength=3)
            parameters = config["modal_similarity"]
            thresholds = (f"Relative complex-motion distance: similar ≤ {parameters['similarity_threshold']:g}, "
                f"different ≥ {parameters['difference_threshold']:g}. "
                f"Maximum projected separation: {parameters['max_pixel_distance']:g} pixels.")
            description = (f"Motion difference: {counts[1]:,} candidates (white); "
                f"insufficient evidence: {counts[2]:,} candidates (gray). "
                "Only trusted retained edges are shown initially.\n\n" + thresholds)
        gui.add_markdown(
            f"**Static geometry graph** — {len(graph.points):,} Gaussians, "
            f"{len(graph.edge_index):,} retained edges, {int(self.data.graph_edge_removed.sum()):,} removed, "
            f"{len(graph.component_size):,} components.\n\n"
            f"K = {config['graph_neighbors']}, radius = {config['graph_max_distance']:g} "
            f"(scene units), filter = {config['graph_edge_filter']}. "
            "Colors show retained connected components, including isolated points. "
            + description)
        self.viewer_resolution = gui.add_slider(
            "Viewer Res", min=64, max=2048, step=1, initial_value=self._viewer_resolution)
        self.hide_render = gui.add_checkbox("Hide Gaussian render", False)
        self.hide_background = gui.add_checkbox(
            "Hide background", True, disabled=self.data.scene.background.count == 0)
        self.show_points = gui.add_checkbox("Show Gaussian centers", True)
        self.point_size = gui.add_slider("Point size", min=0.0002, max=0.008, step=0.0001, initial_value=0.001)
        self.show_component_graph = gui.add_checkbox("Show component graph", True)
        self.show_retained_edges = gui.add_checkbox("Show unchanged edges" if soft else "Show retained edges", True)
        self.show_removed_edges = gui.add_checkbox(
            "Show downweighted edges" if soft else "Show motion-difference edges" if similarity else "Show removed edges",
            soft or not similarity)
        self.show_unsupported_edges = None
        if similarity and not soft:
            self.show_unsupported_edges = gui.add_checkbox("Show unsupported edges", False)
            self.show_unsupported_edges.on_update(self.request_render)
        candidate_count = len(self.data.graph_edge_gaussian_index)
        self.component_graph_edge_count = gui.add_slider(
            "Max visible graph edges", min=0, max=max(candidate_count, 1),
            step=1, initial_value=min(20_000, candidate_count))
        self.component_graph_line_width = gui.add_slider(
            "Graph line width", min=0.1, max=10.0, step=0.1, initial_value=1.0)
        for handle in (self.viewer_resolution, self.hide_render, self.hide_background,
                       self.show_points, self.point_size, self.show_component_graph,
                       self.show_retained_edges, self.show_removed_edges,
                       self.component_graph_edge_count, self.component_graph_line_width):
            handle.on_update(self.request_render)
        self._build_camera_controls()

    def _update_component_graph(self, means):
        if self.data.is_soft_graph:
            selected = _candidate_edge_subset(self.data.graph_edge_factor < 1,
                self.component_graph_edge_count.value, self.show_retained_edges.value, self.show_removed_edges.value)
        elif getattr(self.data, "graph_edge_status", None) is not None:
            selected = _similarity_edge_subset(self.data.graph_edge_status,
                self.component_graph_edge_count.value, self.show_retained_edges.value,
                self.show_removed_edges.value, self.show_unsupported_edges.value)
        else:
            selected = _candidate_edge_subset(self.data.graph_edge_removed,
                self.component_graph_edge_count.value, self.show_retained_edges.value, self.show_removed_edges.value)
        if not self.show_component_graph.value or not len(selected):
            self._remove_component_graph()
            return
        edges = self.data.graph_edge_gaussian_index[selected]
        points = means[torch.as_tensor(edges, device=means.device)].detach().cpu().numpy()
        if self._component_graph_handle is None or not np.array_equal(selected, self._component_graph_edge_indices):
            self._remove_component_graph()
            self._component_graph_handle = self.server.scene.add_line_segments(
                "/debug/component_graph", points=points,
                colors=np.repeat(self.data.graph_edge_colors[selected, None, :], 2, axis=1),
                thickness=float(self.component_graph_line_width.value), thickness_units="screen")
            self._component_graph_edge_indices = selected
        else:
            self._component_graph_handle.points = points
            self._component_graph_handle.thickness = float(self.component_graph_line_width.value)

    @torch.inference_mode()
    def _render(self, client):
        means = self.data.scene.foreground.active()["means"]
        if self.show_component_graph.value:
            self._update_component_graph(means)
        else:
            self._remove_component_graph()
        if self.show_points.value and self._point_cloud is None:
            self._point_cloud = self.server.scene.add_point_cloud(
                "/debug/geometry_points", points=self.data.graph.points,
                colors=self.data.point_colors, point_size=float(self.point_size.value))
        if self._point_cloud is not None:
            self._point_cloud.visible = bool(self.show_points.value)
            self._point_cloud.point_size = float(self.point_size.value)
        camera = self._render_camera(client)
        if self.hide_render.value:
            dark = (self.show_component_graph.value and self.show_removed_edges.value
                    and self.data.graph_edge_removed.any())
            return np.full((camera.height, camera.width, 3), 32 if dark else 255, dtype=np.uint8)
        rendered = self.data.scene.render_deformed(
            camera, means, include_background=not self.hide_background.value)["rgb"]
        return rendered.clamp(0, 1).mul(255).round().byte().cpu().numpy()


def run_graph_viewer(*, scene_dir, graph_dir, work_dir, host="127.0.0.1", port=8080,
                     viewer_resolution=2048):
    _load_gsplat_rasterization()
    data = GraphViewerData(scene_dir, graph_dir)
    viewer = GraphViserViewer(data, work_dir=work_dir, host=host, port=port,
                              viewer_resolution=viewer_resolution)
    viewer.wait()
