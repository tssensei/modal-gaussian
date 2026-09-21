"""Shared-cache panel: saved alignment, exact bins, lazy views and GUI synchronization."""
from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from modal_gaussians.iteration_cache import identity
from modal_gaussians.vis import spectrum
from modal_gaussians.vis.inputs import ViewerInput, ViewerMode
from modal_gaussians.vis.viewer import ModalViewerData


class SpectrumTest(unittest.TestCase):
    def test_saved_sources_and_panel(self):
        with TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary).resolve()
            cache_path = root / "cache"
            cache_path.mkdir()
            labels, frequencies = ["view1", "view2"], [1.0, 3.0]
            views = [{"label": label, "camera_identity": label, "shape_hw": [4, 6],
                      "flow_reference_frame_name": "001", "flow_reference_frame_index": 0,
                      "fps_hz": 8.0} for label in labels]
            cached_views = []
            for label in labels:
                cv2.imwrite(str(cache_path / f"{label}.png"), np.full((4, 6, 3), 80, np.uint8))
                cached_views.append({"label": label, "source_path": str(root / label),
                                     "source_manifest": {"frames": ["001", "002"]},
                                     "reference_image": f"{label}.png", "shape_hw": [4, 6]})
            cache = SimpleNamespace(path=cache_path, frequencies=np.arange(9) * .5,
                manifest={"spectrum_identity": "cache-id", "fft_length": 16, "views": cached_views})
            modes = []
            for k, frequency in enumerate(frequencies):
                model_path = root / f"mode{k}"
                model_path.mkdir()
                dense_path = model_path / "dense"
                dense_path.mkdir()
                dense_views = []
                for view in views:
                    label = view["label"]
                    exported_path = model_path / label
                    exported_path.mkdir()
                    np.save(exported_path / "modal_image.npy", np.full((1, 4, 6, 2), 50+10j, np.complex64))
                    exported = {"format": "modal_gaussians.spectrum_selected_frequency", "status": "complete",
                        "spectrum_source": {"path": str(cache_path), "identity": "cache-id",
                                            "fft_length": 16, "bin_index": int(frequency * 2)},
                        "frequency_hz": frequency, "source_flow_path": str(root / label),
                        "reference_frame_name": "001", "reference_frame_index": 0,
                        "frames": ["001", "002"], "fps_hz": 8.0,
                        "modes_shape": [1, 4, 6, 2], "modes_file": "modal_image.npy"}
                    (exported_path / "manifest.json").write_text(json.dumps(exported))
                    dense_views.append({**view, "modes_file": str(exported_path / "modal_image.npy"),
                        "selected_source": {"path": str(exported_path), "manifest_identity": identity(exported)}})
                dense = {"complex_2d_modes_identity": f"dense{k}", "views": dense_views}
                (dense_path / "manifest.json").write_text(json.dumps(dense))
                model = {"version": 16, "completed_modes_identity": f"mode{k}", "static_scene_identity": "scene",
                    "foreground_identity": "fg", "modes": [{"frequency_hz": frequency}],
                    "complex_2d_modes": str(dense_path), "complex_2d_modes_identity": f"dense{k}",
                    "arrays_file": "completed_modes.npz", "views": [{"label": label} for label in labels]}
                (model_path / "manifest.json").write_text(json.dumps(model))
                np.savez(model_path / "completed_modes.npz", alphas=np.array([[1, 2j]], np.complex64),
                         alpha_identifiable_mask=np.array([[True, k == 0]], bool))
                with np.load(model_path / "completed_modes.npz", allow_pickle=False) as archive:
                    modes.append(ViewerMode(SimpleNamespace(path=model_path, manifest=model,
                        arrays=dict(archive)), 0, design_slot=k))
            # Projected = 1+2j; alpha of view2 is saved 2j, deliberately not fitted to raw=50+10j.
            matrix = np.tile(np.array([1, -2, 1, -2], np.float32), (4, 2, 1))
            design = SimpleNamespace(manifest={"views": views}, design=matrix,
                samples={"view_sample_offsets": np.array([0, 2, 4]),
                         "sample_pixels_xy": np.array([[1, 1], [2, 2]] * 2)})
            for mode in modes:
                mode.design = design
            result = ViewerInput(root, SimpleNamespace(manifest={}), modes, views)
            stack.enter_context(patch.object(spectrum, "load_spectrum", return_value=cache))
            curves = stack.enter_context(patch.object(spectrum, "read_curves",
                return_value={"selected_box": np.arange(9, dtype=float)}))
            stack.enter_context(patch.object(spectrum, "read_region", return_value=np.ones((4, 6), bool)))
            controller = spectrum.SpectrumComparisonController(result)
            self.assertEqual(curves.call_count, 1)  # Other recording's numerical data stays lazy.
            controller.select_view("view2")
            self.assertEqual(curves.call_count, 2)
            np.testing.assert_allclose(controller._projection(1, 0), -4+2j)
            np.testing.assert_array_equal(controller._projection(1, 1), 0)
            self.assertEqual(controller.raw_modal_image.shape, (4, 6, 3))
            controller.select_component("V")
            controller.select_amplitude_normalization("all saved modes")
            self.assertEqual(controller.current_modal_phase_display_context(0, 1, "all saved modes")[:3],
                             ("view2", 2j, True))
            # Newly enabled 3D phase colors must use the same complex alignment as the images.
            data = ModalViewerData.__new__(ModalViewerData)
            data.device, data.spectrum = "cpu", controller
            data.phi = torch.full((2, 1, 3), 1+2j, dtype=torch.complex64)
            data.scene = SimpleNamespace(foreground=SimpleNamespace(
                active=lambda: {"means": torch.tensor([[0., 0., 1.]])}))
            camera = SimpleNamespace(world_to_camera=torch.eye(4), K=torch.eye(3), radial_distortion=0)
            camera.to = lambda device: camera
            data.camera_by_label = {"view2": SimpleNamespace(camera=camera)}
            colors = data.phase_colors(0, 1, "all saved modes").numpy()
            np.testing.assert_allclose(colors, spectrum._hsv_rgb(np.array([-4+2j]),
                                       controller.modal_image_magnitude_hi), atol=1e-6)

            class GUI:
                def __getattr__(self, name):
                    return lambda *args, **kw: SimpleNamespace(
                        value=kw.get("initial_value"), on_update=lambda f: f, on_click=lambda f: f, **kw)

            panel = spectrum.ModalSpectrumPanel(SimpleNamespace(gui=GUI()), controller,
                **{name: lambda *args: None for name in ("on_view_selected", "on_mode_selected",
                   "on_component_selected", "on_normalization_selected", "on_solo_selected", "on_enable_all")})
            panel.set_mode_index(1)
            self.assertEqual(panel.mode.value, 1)  # Every panel uses the normalized input order.
            self.assertIn("unidentifiable", controller.status)
            controller.select_view("view1")
            self.assertEqual(curves.call_count, 2)
            with self.assertRaisesRegex(ValueError, "Unknown spectrum view"):
                controller.select_view("view3")
            cache.frequencies[6] += .1
            with self.assertRaisesRegex(ValueError, "exact shared FFT bin"):
                spectrum.SpectrumComparisonController(result)


if __name__ == "__main__":
    unittest.main()
