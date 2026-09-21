"""Single-frequency previews display saved inputs without loading a full FFT."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from modal_gaussians.iteration_cache import identity
from modal_gaussians.vis import spectrum
from modal_gaussians.vis.inputs import ViewerInput, ViewerMode


class SelectedModalPanelTest(unittest.TestCase):
    def test_selected_images_alignment_and_single_frequency_panel(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            views = [{"label": label, "camera_identity": label, "shape_hw": [4, 6],
                      "flow_reference_frame_name": "001", "flow_reference_frame_index": 0,
                      "fps_hz": 30.} for label in ("view1", "view2")]
            dense_views = []
            for view in views:
                export = root / view["label"]
                export.mkdir()
                reference = export / "reference.png"
                cv2.imwrite(str(reference), np.full((4, 6, 3), 80, np.uint8))
                field = np.full((1, 4, 6, 2), 5+2j, np.complex64)
                field[0, 1, 2] = [1j, 0j]  # Unsampled pixel: retain its own phase/amplitude.
                np.save(export / "modal_image.npy", field)
                manifest = {"format": "modal_gaussians.sea_raft_selected_frequency_experiment",
                    "status": "complete", "frequency_hz": .744, "reference_frame_name": "001",
                    "reference_frame_index": 0, "fps_hz": 30., "modes_shape": [1, 4, 6, 2],
                    "modes_file": "modal_image.npy", "reference_image": str(reference)}
                (export / "manifest.json").write_text(json.dumps(manifest))
                dense_views.append({**view, "modes_file": str(export / "modal_image.npy"),
                    "selected_source": {"path": str(export), "manifest_identity": identity(manifest)}})
            dense = root / "dense"
            dense.mkdir()
            (dense / "manifest.json").write_text(json.dumps({"complex_2d_modes_identity": "dense",
                                                               "views": dense_views}))
            model = {"version": 16, "completed_modes_identity": "model", "static_scene_identity": "scene",
                "foreground_identity": "fg", "modes": [{"mode_slot": 0, "frequency_hz": .744}],
                "complex_2d_modes": str(dense), "complex_2d_modes_identity": "dense", "views": views,
                "selected_modal_supervision": {}}
            result = NS(completed_modes=NS(path=root, manifest=model, arrays={
                "alphas": np.array([[1., 2j]], np.complex64),
                "alpha_identifiable_mask": np.array([[True, False]])}),
                rendered_design=NS(manifest={"views": views},
                    samples={"view_sample_offsets": np.array([0, 2, 4]),
                             "sample_pixels_xy": np.array([[1, 1], [2, 2]] * 2)},
                    design=np.tile(np.array([1., -2.], np.float32), (4, 2, 1))))
            def viewer_input():
                return ViewerInput(root, NS(manifest={}),
                    [ViewerMode(result.completed_modes, 0, result.rendered_design)], views)

            with patch.object(spectrum, "load_spectrum", side_effect=AssertionError("No FFT cache")), \
                 patch.object(spectrum, "read_curves", side_effect=AssertionError("No spectrum read")), \
                 patch.object(spectrum, "read_region", side_effect=AssertionError("No old region read")):
                controller = spectrum.SpectrumComparisonController(viewer_input())
                self.assertFalse(controller.full_spectrum_available)
                np.testing.assert_array_equal(controller.raw_frequencies_hz, [.744])
                np.testing.assert_allclose(controller.raw_power, [np.sqrt(58.)])
                np.testing.assert_allclose(controller._projection(0, 0), 1+2j)
                self.assertEqual(len(controller._states), 1)
                image = controller.reconstructed_modal_image.copy()
                brightness = controller.modal_image_magnitude_hi
                expected_support = np.zeros((4, 6), bool)
                expected_support[:3, :3] = expected_support[1:4, 1:4] = True
                for overlay in (controller.raw_modal_image, image):
                    np.testing.assert_array_equal(np.any(overlay != 28, axis=-1), expected_support)
                np.testing.assert_array_equal(controller.raw_modal_image[1, 2],
                    np.rint(np.array([.5, 0., 1.]) * 255 / np.sqrt(29.)).astype(np.uint8))
                np.testing.assert_allclose(brightness, np.sqrt(29.))
                controller.select_component("V")
                np.testing.assert_array_equal(controller.raw_modal_image[1, 2], [0, 0, 0])
                controller.select_component("U")
                controller.select_original_image_region("Full image")
                self.assertFalse(np.array_equal(controller.raw_modal_image[0, 5], [28, 28, 28]))
                np.testing.assert_array_equal(controller.reconstructed_modal_image, image)
                self.assertEqual(controller.modal_image_magnitude_hi, brightness)
                controller.select_view("view2")
                np.testing.assert_array_equal(controller._projection(1, 0), 0.)
                self.assertIn("unidentifiable", controller.status)
                result.completed_modes.arrays["alpha_identifiable_mask"][0, 1] = True
                aligned = spectrum.SpectrumComparisonController(viewer_input())
                np.testing.assert_allclose(aligned._projection(1, 0), -4+2j)

                class GUI:
                    def __getattr__(self, name):
                        return lambda *args, **kw: NS(value=kw.get("initial_value"),
                            on_update=lambda f: f, on_click=lambda f: f, **kw)

                panel = spectrum.ModalSpectrumPanel(NS(gui=GUI()), controller,
                    **{name: lambda *args: None for name in ("on_view_selected", "on_mode_selected",
                       "on_component_selected", "on_normalization_selected", "on_solo_selected", "on_enable_all")})
                self.assertEqual(panel.frequency_range.options, ("selected modes",))
                lo, hi = panel._frequency_limits()
                self.assertLess(lo, .744)
                self.assertGreater(hi, .744)
                panel.set_component("V")
                panel.set_mode_index(0)
                self.assertEqual(panel.frequency.value, .744)
                self.assertIn("Saved input frequencies", panel.raw_plot.title)
                with self.assertRaisesRegex(ValueError, "region"):
                    controller.select_original_image_region("Cached region")

                model["modes"][0]["frequency_hz"] = .75
                with self.assertRaisesRegex(ValueError, "frequency"):
                    spectrum.SpectrumComparisonController(viewer_input())

                # Motion reference may differ from the unchanged geometry reference.
                model["modes"][0]["frequency_hz"] = .744
                for design_view, dense_view in zip(views, dense_views):
                    design_view["motion_reference"] = {"reference_frame_name": "002",
                        "reference_frame_index": 1, "selection_identity": "selection"}
                    path = Path(dense_view["selected_source"]["path"]) / "manifest.json"
                    exported = json.loads(path.read_text())
                    exported.update(reference_frame_name="002", reference_frame_index=1,
                                    reference_selection={"identity":"selection"})
                    path.write_text(json.dumps(exported))
                    dense_view["selected_source"]["manifest_identity"] = identity(exported)
                (dense / "manifest.json").write_text(json.dumps({
                    "complex_2d_modes_identity": "dense", "views": dense_views}))
                spectrum.SpectrumComparisonController(viewer_input())
                views[0]["motion_reference"]["selection_identity"] = "wrong"
                with self.assertRaisesRegex(ValueError, "reference"):
                    spectrum.SpectrumComparisonController(viewer_input())
                views[0]["motion_reference"]["selection_identity"] = "selection"

                # Corn stores the same frequency with different float representations.
                for view in dense_views:
                    path = Path(view["selected_source"]["path"]) / "manifest.json"
                    exported = json.loads(path.read_text())
                    exported["frequency_hz"] = .225
                    path.write_text(json.dumps(exported))
                    view["selected_source"]["manifest_identity"] = identity(exported)
                (dense / "manifest.json").write_text(json.dumps({
                    "complex_2d_modes_identity": "dense", "views": dense_views}))
                model["modes"][0]["frequency_hz"] = .22500000000000003
                corn = spectrum.SpectrumComparisonController(viewer_input())
                np.testing.assert_array_equal(corn.frequencies_hz, [.22500000000000003])
                model["modes"][0]["frequency_hz"] = .225 + 2e-9
                with self.assertRaisesRegex(ValueError, "frequency"):
                    spectrum.SpectrumComparisonController(viewer_input())


if __name__ == "__main__":
    unittest.main()
