"""Graph-only entry checks without starting a Viser server or loading modes."""
from contextlib import ExitStack, redirect_stderr
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from modal_gaussians import cli
from modal_gaussians.iteration_cache import identity, put_entry
from modal_gaussians.vis import graph_viewer, viewer
from test_geometry_graph import synthetic_graph
from test_neural_rendering import tiny_cuda_scene


class GraphViewerTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.scene, self.camera = tiny_cuda_scene("cpu")
        self.scene.manifest = {"foreground_identity": "f" * 64}
        points = self.scene.foreground.active()["means"].detach().numpy()
        self.graph = synthetic_graph(points, [[0, 1]])
        self.contract = {"implementation": "neural_geometry_cache_v1", "foreground": "f" * 64,
                         "code": "a" * 64, "config": {"graph_neighbors": 8,
                         "graph_max_distance": 0.008, "graph_edge_filter": "none"}}
        put_entry(self.root, self.contract, self.graph.as_dict())
        self.path = self.root / identity(self.contract)
        stack = self.enterContext(ExitStack())
        stack.enter_context(patch.object(graph_viewer, "load_static_scene", return_value=self.scene))
        stack.enter_context(patch.object(graph_viewer, "cameras_from_scene_manifest", return_value=(self.camera,)))
        stack.enter_context(patch.object(viewer, "ModalViewerData", side_effect=AssertionError("modal load")))
        stack.enter_context(patch.object(viewer, "load_viewer_input", side_effect=AssertionError("modal input load")))

    def test_cache_binding_and_points_without_modes(self):
        data = graph_viewer.GraphViewerData("scene", self.path, device="cpu")
        np.testing.assert_array_equal(data.graph_edge_gaussian_index, [[0, 1]])
        self.assertEqual(len(data.point_colors), 4)  # Includes both isolated Gaussians.
        self.assertFalse(hasattr(data, "phi"))
        self.assertFalse(hasattr(data, "frequencies_hz"))
        self.scene.manifest["foreground_identity"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "foreground"):
            graph_viewer.GraphViewerData("scene", self.path, device="cpu")
        self.scene.manifest["foreground_identity"] = "f" * 64
        with torch.no_grad():
            self.scene.foreground.params["means"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "positions"):
            graph_viewer.GraphViewerData("scene", self.path, device="cpu")

    def test_soft_graph_keeps_edges_and_exposes_weight_overlays(self):
        artifact = self.root / "soft-graph"
        artifact.mkdir()
        np.savez(artifact / "graph.npz", **self.graph.as_dict())
        np.savez(artifact / "edge_evidence.npz", candidate_status=np.zeros(1, dtype=np.uint8),
                 candidate_edge_factor=np.array([.05]))
        manifest = {"format": "modal_gaussians.modal_similarity_graph", "version": 1,
            "foreground_identity": "f" * 64, "graph_file": "graph.npz", "evidence_file": "edge_evidence.npz",
            "config": {**self.contract["config"], "graph_edge_filter": "modal-similarity",
                       "modal_similarity": {"soft_weights": True}}}
        (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        data = graph_viewer.GraphViewerData("scene", artifact, device="cpu")
        self.assertTrue(data.is_soft_graph)
        self.assertFalse(data.graph_edge_removed.any())
        np.testing.assert_array_equal(data.graph_edge_gaussian_index, self.graph.edge_index)
        self.assertTrue(np.all(data.graph_edge_colors == data.graph_edge_colors[:, :1]))
        controls = {}
        server = MagicMock()
        def control(label, initial_value=None, **kwargs):
            handle = SimpleNamespace(value=initial_value, on_update=MagicMock())
            controls[label] = handle
            return handle
        server.gui.add_checkbox.side_effect = control
        server.gui.add_slider.side_effect = control
        with patch.object(viewer.viser, "ViserServer", return_value=server):
            instance = graph_viewer.GraphViserViewer(data, work_dir=self.root / "viewer")
        self.assertTrue(controls["Show unchanged edges"].value)
        self.assertTrue(controls["Show downweighted edges"].value)
        self.assertNotIn("Show unsupported edges", controls)
        instance._update_component_graph(self.scene.foreground.active()["means"])
        server.scene.add_line_segments.assert_called_once()
        colors = graph_viewer._soft_weight_colors(np.full((3, 3), 200, dtype=np.uint8), np.array([1., .5, .05]))
        np.testing.assert_array_equal(colors[0], [200, 200, 200])
        self.assertGreater(colors[1, 0], colors[2, 0])

    def test_gui_and_render_reuse_static_centers_without_modal_controls(self):
        data = graph_viewer.GraphViewerData("scene", self.path, device="cpu")
        server = MagicMock()
        controls = {}

        def control(label, initial_value=None, **kwargs):
            handle = SimpleNamespace(value=initial_value, on_update=MagicMock())
            controls[label] = handle
            return handle

        server.gui.add_checkbox.side_effect = control
        server.gui.add_slider.side_effect = control
        with patch.object(viewer.viser, "ViserServer", return_value=server):
            instance = graph_viewer.GraphViserViewer(data, work_dir=self.root / "viewer")
        self.assertTrue(controls["Show component graph"].value)
        self.assertNotIn("Motion scale", controls)
        server.gui.add_tab_group.assert_not_called()
        before = {name: tensor.clone() for name, tensor in self.scene.state_dict().items()}
        with patch.object(instance, "_render_camera", return_value=self.camera), \
             patch.object(self.scene, "render_deformed", return_value={"rgb": torch.zeros(32, 32, 3)}) as render:
            image = instance._render(MagicMock())
            self.assertEqual(image.shape, (32, 32, 3))
            self.assertEqual(image.dtype, np.uint8)
            torch.testing.assert_close(render.call_args.args[1], self.scene.foreground.active()["means"])
            self.assertEqual(render.call_args.kwargs, {"include_background": False})
            np.testing.assert_array_equal(server.scene.add_line_segments.call_args.kwargs["points"],
                                          self.graph.points[self.graph.edge_index])
            controls["Show component graph"].value = False
            controls["Hide Gaussian render"].value = True
            render.reset_mock()
            self.assertTrue(np.all(instance._render(MagicMock()) == 255))
            render.assert_not_called()
            server.scene.add_line_segments.return_value.remove.assert_called_once()
        for name, value in self.scene.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    def test_modal_gradient_artifact_keeps_removed_candidates_visible(self):
        artifact = self.root / "modal-gradient"
        artifact.mkdir()
        graph = replace(self.graph, candidate_edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64))
        np.savez(artifact / "graph.npz", **graph.as_dict())
        manifest = {"format": "modal_gaussians.modal_gradient_graph", "version": 1,
            "scene_dir": str(self.root / "scene"), "foreground_identity": "f" * 64,
            "graph_file": "graph.npz", "evidence_file": "edge_evidence.npz",
            "config": {**self.contract["config"], "graph_edge_filter": "modal-gradient",
                       "modal_gradient": {"gradient_threshold": 0.1}},
            "summary": {"candidate_edges": 2, "retained_edges": 1, "removed_edges": 1}}
        manifest_path = artifact / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with patch.object(graph_viewer, "load_entry", side_effect=AssertionError("cache replay")):
            data = graph_viewer.GraphViewerData("scene", artifact, device="cpu")
        np.testing.assert_array_equal(data.graph_edge_gaussian_index, [[0, 1], [1, 2]])
        np.testing.assert_array_equal(data.graph_edge_removed, [False, True])
        np.testing.assert_array_equal(data.graph_edge_colors[1], [255, 255, 255])
        self.assertEqual(data.graph_config["graph_edge_filter"], "modal-gradient")
        manifest["foreground_identity"] = "b" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "foreground"):
            graph_viewer.GraphViewerData("scene", artifact, device="cpu")
        manifest["foreground_identity"] = "f" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with torch.no_grad():
            self.scene.foreground.params["means"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "positions"):
            graph_viewer.GraphViewerData("scene", artifact, device="cpu")

    def test_cli_routes_graph_input_and_rejects_incomplete_pairs(self):
        base = ["viewer", "--work-dir", str(self.root / "viewer")]
        with patch.object(graph_viewer, "run_graph_viewer") as graph, \
             patch.object(viewer, "run_modal_viewer") as modal:
            self.assertEqual(cli.main(base + ["--scene", "scene", "--geometry-graph", str(self.path)]), 0)
            graph.assert_called_once()
            modal.assert_not_called()
            self.assertEqual(graph.call_args.kwargs["graph_dir"], self.path)
            for invalid in (["--scene", "scene"], ["--preview", "preview", "--geometry-graph", str(self.path)],
                            ["--scene", "scene", "--preview", "preview"]):
                with self.subTest(args=invalid), redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
                    cli.main(base + invalid)
                self.assertEqual(raised.exception.code, 2)

    def test_modal_similarity_artifact_categories_and_default_overlays(self):
        artifact = self.root / "modal-similarity"
        artifact.mkdir()
        graph = replace(self.graph, candidate_edge_index=np.array([[1, 2], [0, 1], [2, 3]], dtype=np.int64))
        status = np.array([2, 0, 1], dtype=np.uint8)
        np.savez(artifact / "graph.npz", **graph.as_dict())
        np.savez(artifact / "edge_evidence.npz", candidate_status=status)
        manifest = {"format": "modal_gaussians.modal_similarity_graph", "version": 1,
            "foreground_identity": "f" * 64, "graph_file": "graph.npz", "evidence_file": "edge_evidence.npz",
            "config": {**self.contract["config"], "graph_edge_filter": "modal-similarity",
                "modal_similarity": {"similarity_threshold": 0.15, "difference_threshold": 0.3,
                                     "max_pixel_distance": 32.0}}}
        (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        data = graph_viewer.GraphViewerData("scene", artifact, device="cpu")
        np.testing.assert_array_equal(data.graph_edge_gaussian_index, graph.candidate_edge_index)
        np.testing.assert_array_equal(data.graph_edge_status, status)
        np.testing.assert_array_equal(data.graph_edge_colors[[0, 2]], [[90, 90, 90], [255, 255, 255]])
        controls = {}
        server = MagicMock()

        def control(label, initial_value=None, **kwargs):
            handle = SimpleNamespace(value=initial_value, on_update=MagicMock())
            controls[label] = handle
            return handle

        server.gui.add_checkbox.side_effect = control
        server.gui.add_slider.side_effect = control
        with patch.object(viewer.viser, "ViserServer", return_value=server):
            instance = graph_viewer.GraphViserViewer(data, work_dir=self.root / "viewer")
        self.assertTrue(controls["Show retained edges"].value)
        self.assertFalse(controls["Show motion-difference edges"].value)
        self.assertFalse(controls["Show unsupported edges"].value)
        means = self.scene.foreground.active()["means"]
        instance._update_component_graph(means)
        np.testing.assert_array_equal(instance._component_graph_edge_indices, [1])
        controls["Show motion-difference edges"].value = True
        controls["Show unsupported edges"].value = True
        instance._update_component_graph(means)
        np.testing.assert_array_equal(instance._component_graph_edge_indices, [0, 1, 2])
        np.testing.assert_array_equal(graph_viewer._similarity_edge_subset(status, 10, False, False, True), [0])
        self.assertEqual(len(graph_viewer._similarity_edge_subset(status, 0, True, True, True)), 0)
        self.assertEqual(len(graph_viewer._similarity_edge_subset(status, 10, False, False, False)), 0)
        status[0] = 0
        np.savez(artifact / "edge_evidence.npz", candidate_status=status)
        with self.assertRaisesRegex(ValueError, "Candidate status"):
            graph_viewer.GraphViewerData("scene", artifact, device="cpu")


class CandidateEdgeDisplayTest(unittest.TestCase):
    def test_retained_removed_mapping_sampling_and_static_overlay(self):
        graph = synthetic_graph([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], [[2, 3], [0, 1]])
        graph = replace(graph, candidate_edge_index=np.array([[1, 2], [3, 2], [0, 1], [0, 3]], dtype=np.int64))
        edges, colors, removed = graph_viewer._candidate_graph_display(graph)
        np.testing.assert_array_equal(edges, graph.candidate_edge_index)
        np.testing.assert_array_equal(removed, [True, False, False, True])
        np.testing.assert_array_equal(colors[removed], np.full((2, 3), 255, dtype=np.uint8))
        expected = np.round(255 * viewer._component_colors(graph.component_index[[2, 0]])).astype(np.uint8)
        np.testing.assert_array_equal(colors[~removed], expected)
        sparse_removed = np.zeros(101, dtype=bool)
        sparse_removed[-1] = True
        np.testing.assert_array_equal(graph_viewer._candidate_edge_subset(sparse_removed, 2), [0, 100])
        np.testing.assert_array_equal(graph_viewer._candidate_edge_subset(sparse_removed, 1), [100])
        np.testing.assert_array_equal(graph_viewer._candidate_edge_subset(removed, 10, False), [0, 3])
        np.testing.assert_array_equal(graph_viewer._candidate_edge_subset(removed, 10, True, False), [1, 2])
        self.assertEqual(len(graph_viewer._candidate_edge_subset(removed, 10, False, False)), 0)
        self.assertEqual(len(graph_viewer._candidate_edge_subset(removed, 0)), 0)

        means = torch.from_numpy(graph.points)
        instance = graph_viewer.GraphViserViewer.__new__(graph_viewer.GraphViserViewer)
        instance.data = SimpleNamespace(is_soft_graph=False, graph_edge_gaussian_index=edges, graph_edge_colors=colors,
            graph_edge_removed=removed, scene=SimpleNamespace(foreground=SimpleNamespace(active=lambda: {"means": means})))
        instance.server = MagicMock()
        instance._component_graph_handle = None
        instance._component_graph_edge_indices = np.empty(0, dtype=np.int64)
        instance._component_graph_color_mode = None
        instance._point_cloud = None
        for name, value in (("show_component_graph", True), ("show_retained_edges", True),
                            ("show_removed_edges", True), ("component_graph_edge_count", 10),
                            ("component_graph_line_width", 1), ("show_points", False), ("hide_render", True)):
            setattr(instance, name, SimpleNamespace(value=value))
        instance._render_camera = lambda client: SimpleNamespace(height=2, width=3)
        self.assertTrue(np.all(instance._render(None) == 32))
        drawn = instance.server.scene.add_line_segments.call_args.kwargs
        np.testing.assert_array_equal(drawn["points"], graph.points[edges])
        np.testing.assert_array_equal(drawn["colors"], np.repeat(colors[:, None, :], 2, axis=1))
        instance.show_retained_edges.value = False
        instance._update_component_graph(means)
        np.testing.assert_array_equal(instance._component_graph_edge_indices, [0, 3])
        self.assertTrue(np.all(instance.server.scene.add_line_segments.call_args.kwargs["colors"] == 255))
        instance.show_removed_edges.value = False
        instance._update_component_graph(means)
        self.assertIsNone(instance._component_graph_handle)

        graph = replace(graph, candidate_edge_index=graph.edge_index.copy())  # Existing KNN-only cache.
        old_edges, old_colors = viewer._neural_graph_display(
            {"g_" + key: value for key, value in graph.as_dict().items()}, len(graph.points))
        edges, colors, removed = graph_viewer._candidate_graph_display(graph)
        np.testing.assert_array_equal(edges, old_edges)
        np.testing.assert_array_equal(colors, np.round(old_colors * 255).astype(np.uint8))
        self.assertFalse(removed.any())
        np.testing.assert_array_equal(graph_viewer._candidate_edge_subset(removed, 1),
                                      viewer._stable_uniform_indices(len(old_edges), 1))
        instance.data.graph_edge_removed = removed
        self.assertTrue(np.all(instance._render(None) == 255))


if __name__ == "__main__":
    unittest.main()
