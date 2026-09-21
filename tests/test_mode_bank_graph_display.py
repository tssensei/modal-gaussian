"""Synthetic mode-bank KNN display checks; no datasets, CUDA or Viser server."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from modal_gaussians.motion.common import completed_modes
from modal_gaussians.vis import viewer
from modal_gaussians.vis.inputs import ViewerInput, ViewerMode


class ModeBankGraphDisplayTest(unittest.TestCase):
    def test_mode_bank_graph_switches_sources_without_loading_modes(self):
        graphs = [np.array([[0, 1], [2, 3]]), np.array([[0, 2]])]
        modes = [ViewerMode(SimpleNamespace(manifest={"version": 16}, arrays={
            "g_edge_index": edges, "g_edge_weight": np.ones(len(edges)),
            "g_component_index": np.array([0, 0, 1, 1]) if k == 0 else np.array([0, 1, 0, 1])}), 0)
            for k, edges in enumerate(graphs)]
        data = viewer.ModalViewerData.__new__(viewer.ModalViewerData)
        data.result = ViewerInput(Path("."), None, list(reversed(modes)), [])
        data.scene = SimpleNamespace(foreground=SimpleNamespace(count=4))
        with patch.object(completed_modes, "load_completed_modes", side_effect=AssertionError("mode replay")), \
                patch.object(Path, "read_text", side_effect=AssertionError("source reread")):
            data._load_graph_display()
            np.testing.assert_array_equal(data.graph_edge_gaussian_index, graphs[1])
            instance = viewer.ModalViserViewer.__new__(viewer.ModalViserViewer)
            instance.data = data
            instance.server = SimpleNamespace(scene=Mock())
            instance.show_component_graph = SimpleNamespace(value=True)
            instance.component_graph_edge_count = SimpleNamespace(value=2, max=2, step=1)
            instance.component_graph_line_width = SimpleNamespace(value=1.)
            instance.phase_mode = SimpleNamespace(value=0)
            instance._component_graph_handle = None
            instance._component_graph_edge_indices = np.empty(0, np.int64)
            instance._component_graph_color_mode = None
            means = torch.arange(12, dtype=torch.float32).reshape(4, 3)
            instance._update_component_graph(means)
            old_handle = instance._component_graph_handle
            instance.server.scene.add_line_segments.return_value = Mock()
            instance.phase_mode.value = 1
            with patch.object(data, "_load_graph_display", wraps=data._load_graph_display) as load:
                instance._update_component_graph(means)
                instance._update_component_graph(means)
                load.assert_called_once_with(1)
            old_handle.remove.assert_called_once()
            call = instance.server.scene.add_line_segments.call_args.kwargs
            np.testing.assert_array_equal(call["points"], means.numpy()[graphs[0]])
            np.testing.assert_array_equal(call["colors"], np.repeat(data.graph_edge_colors[:, None], 2, axis=1))
            self.assertEqual(instance.server.scene.add_line_segments.call_count, 2)
            self.assertEqual(instance.component_graph_edge_count.max, 2)
            instance.phase_mode.value = 0
            instance._update_component_graph(means)
            self.assertEqual(instance.component_graph_edge_count.max, 1)
            self.assertEqual(instance.component_graph_edge_count.value, 1)
            instance.show_component_graph.value = False
            instance.phase_mode.value = 1
            with patch.object(data, "_load_graph_display", side_effect=AssertionError("hidden graph load")):
                instance._update_component_graph(means)

            # Observation colors and support roles use that same selected mode.
            instance.color_mode = SimpleNamespace(value=viewer.COLOR_OBSERVATIONS)
            data.observation_colors = Mock(return_value="colors")
            instance.show_support = SimpleNamespace(value=True)
            instance.support_filters = [SimpleNamespace(value=False), SimpleNamespace(value=True)]
            instance.support_count = SimpleNamespace(value=4)
            instance.support_size = SimpleNamespace(value=.01)
            instance._point_cloud = None
            data.display_class = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
            for index, point in ((0, 0), (1, 2)):
                instance.phase_mode.value = index
                self.assertEqual(instance._current_foreground_colors(), "colors")
                data.observation_colors.assert_called_with(index)
                instance._update_support_cloud(means)
                np.testing.assert_array_equal(instance.server.scene.add_point_cloud.call_args.kwargs["points"],
                                              means.numpy()[[point]])


if __name__ == "__main__":
    unittest.main()
