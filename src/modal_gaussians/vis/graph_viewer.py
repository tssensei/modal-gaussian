"""Static geometry inspection without frequencies, learned modes or network replay."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from modal_gaussians.iteration_cache import identity, load_entry
from modal_gaussians.motion.neural.geometry_graph import GeometryGraph
from modal_gaussians.static import load_static_scene, cameras_from_scene_manifest, _load_gsplat_rasterization
from modal_gaussians.vis.viewer import (
    ModalViserViewer, ViewerCamera, _component_colors, _neural_graph_display, _stable_uniform_indices,
)


def _candidate_graph_display(graph):
    """Keep candidate order; white marks edges absent from the filtered graph."""
    count = len(graph.points)
    retained, _ = _neural_graph_display({"g_edge_index": graph.edge_index,
        "g_component_index": graph.component_index, "g_edge_weight": graph.edge_weight}, count)
    candidates = np.asarray(graph.candidate_edge_index)
    if (candidates.ndim != 2 or candidates.shape[1] != 2
            or not np.issubdtype(candidates.dtype, np.integer)
            or np.any(candidates < 0) or np.any(candidates >= count)):
        raise ValueError("Geometry candidate edge indices are invalid")

    def keys(edges):
        ordered = np.sort(edges.astype(np.int64, copy=False), axis=1)
        return ordered[:, 0] * count + ordered[:, 1]

    removed = ~np.isin(keys(candidates), keys(retained))
    if np.count_nonzero(~removed) != len(retained):
        raise ValueError("Geometry retained edges do not match the candidate edges")
    colors = np.round(255 * _component_colors(graph.component_index[candidates[:, 0]])).astype(np.uint8)
    colors[removed] = 255
    return candidates, colors, removed


def _candidate_edge_subset(removed, maximum, show_retained=True, show_removed=True):
    """Uniformly sample each visible class, reserving room for rare removed edges."""
    retained = np.flatnonzero(~removed) if show_retained else np.empty(0, dtype=np.int64)
    rejected = np.flatnonzero(removed) if show_removed else np.empty(0, dtype=np.int64)
    budget = min(max(int(maximum), 0), len(retained) + len(rejected))
    if not budget:
        return np.empty(0, dtype=np.int64)
    if len(retained) and len(rejected):
        rejected_budget = max(1, round(budget * len(rejected) / (len(retained) + len(rejected))))
        if budget > 1:
            rejected_budget = min(rejected_budget, budget - 1)
        rejected_budget = min(len(rejected), rejected_budget)
        retained_budget = min(len(retained), budget - rejected_budget)
        rejected_budget = budget - retained_budget
    else:
        rejected_budget = min(len(rejected), budget)
        retained_budget = budget - rejected_budget
    return np.sort(np.concatenate((retained[_stable_uniform_indices(len(retained), retained_budget)],
                                   rejected[_stable_uniform_indices(len(rejected), rejected_budget)])))


class GraphViewerData:
    def __init__(self, scene_dir, graph_dir, device="cuda"):
        self.device = torch.device(device)
        self.scene = load_static_scene(scene_dir, self.device).eval()
        self.scene.requires_grad_(False)
        path = Path(graph_dir).expanduser().resolve(strict=True)
        contract = json.loads((path / "manifest.json").read_text(encoding="utf-8")).get("contract", {})
        if (contract.get("implementation") != "neural_geometry_cache_v1"
                or contract.get("foreground") != self.scene.manifest["foreground_identity"]):
            raise ValueError("Geometry cache does not belong to this scene's foreground")
        if path.name != identity(contract):
            raise ValueError("Geometry cache directory differs from its contract identity")
        arrays = load_entry(path.parent, contract)
        if arrays is None:
            raise FileNotFoundError(f"Geometry cache is missing: {path}")
        self.graph = GeometryGraph.from_dict(arrays)
        means = self.scene.foreground.active()["means"].detach().cpu().numpy()
        if not np.array_equal(self.graph.points, means):
            raise ValueError("Geometry points differ from the scene's foreground order or positions")
        self.graph_config = contract["config"]
        self.graph_edge_gaussian_index, self.graph_edge_colors, self.graph_edge_removed = _candidate_graph_display(self.graph)
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
        removed_count = int(self.data.graph_edge_removed.sum())
        candidate_count = len(self.data.graph_edge_gaussian_index)
        gui.add_markdown(
            f"**Static geometry graph** — {len(graph.points):,} Gaussians, "
            f"{len(graph.edge_index):,} retained edges, {removed_count:,} removed edges, "
            f"{len(graph.component_size):,} components.\n\n"
            f"K = {config['graph_neighbors']}, radius = {config['graph_max_distance']:g} "
            f"(scene units), filter = {config['graph_edge_filter']}. "
            "Colors show filtered connected components, including isolated points. "
            "White edges were removed by the depth filter rules, including unsupported long edges; "
            "white does not mean downweighted. Edge sampling reserves space for removed edges.")
        self.viewer_resolution = gui.add_slider(
            "Viewer Res", min=64, max=2048, step=1, initial_value=self._viewer_resolution)
        self.hide_render = gui.add_checkbox("Hide Gaussian render", False)
        self.hide_background = gui.add_checkbox(
            "Hide background", True, disabled=self.data.scene.background.count == 0)
        self.show_points = gui.add_checkbox("Show Gaussian centers", True)
        self.point_size = gui.add_slider("Point size", min=0.0002, max=0.008, step=0.0001, initial_value=0.001)
        self.show_component_graph = gui.add_checkbox("Show component graph", True)
        self.show_retained_edges = gui.add_checkbox("Show retained edges", True)
        self.show_removed_edges = gui.add_checkbox("Show removed edges (white)", True, disabled=removed_count == 0)
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
        selected = _candidate_edge_subset(self.data.graph_edge_removed, self.component_graph_edge_count.value,
                                         self.show_retained_edges.value, self.show_removed_edges.value)
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
            background = 32 if self.data.graph_edge_removed.any() else 255
            return np.full((camera.height, camera.width, 3), background, dtype=np.uint8)
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
