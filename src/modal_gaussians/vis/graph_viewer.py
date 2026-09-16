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
from modal_gaussians.vis.viewer import ModalViserViewer, ViewerCamera, _component_colors, _neural_graph_display


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
        self.graph_edge_gaussian_index, self.graph_edge_colors = _neural_graph_display(
            {"g_" + name: value for name, value in arrays.items()}, len(means))
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
        gui.add_markdown(
            f"**Static geometry graph** — {len(graph.points):,} Gaussians, "
            f"{len(graph.edge_index):,} edges, {len(graph.component_size):,} components.\n\n"
            f"K = {config['graph_neighbors']}, radius = {config['graph_max_distance']:g} "
            f"(scene units), filter = {config['graph_edge_filter']}. "
            "Colors show connected components, including isolated points.")
        self.viewer_resolution = gui.add_slider(
            "Viewer Res", min=64, max=2048, step=1, initial_value=self._viewer_resolution)
        self.hide_render = gui.add_checkbox("Hide Gaussian render", False)
        self.hide_background = gui.add_checkbox(
            "Hide background", True, disabled=self.data.scene.background.count == 0)
        self.show_points = gui.add_checkbox("Show Gaussian centers", True)
        self.point_size = gui.add_slider("Point size", min=0.0002, max=0.008, step=0.0001, initial_value=0.001)
        self.show_component_graph = gui.add_checkbox("Show component graph", True)
        self.component_graph_edge_count = gui.add_slider(
            "Max visible graph edges", min=0, max=max(len(graph.edge_index), 1),
            step=1, initial_value=min(20_000, len(graph.edge_index)))
        self.component_graph_line_width = gui.add_slider(
            "Graph line width", min=0.1, max=10.0, step=0.1, initial_value=1.0)
        for handle in (self.viewer_resolution, self.hide_render, self.hide_background,
                       self.show_points, self.point_size, self.show_component_graph,
                       self.component_graph_edge_count, self.component_graph_line_width):
            handle.on_update(self.request_render)
        self._build_camera_controls()

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
            return np.full((camera.height, camera.width, 3), 255, dtype=np.uint8)
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
