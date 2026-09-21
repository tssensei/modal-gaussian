"""Synthetic Viewer checks: no datasets, renderer, server or background threads."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
import cv2
import torch
import viser.transforms as tf

from modal_gaussians.vis.render_panel import CameraPath, Keyframe, populate_render_tab
from modal_gaussians.vis.viewer import ViewerCamera


class Handle(NS):
    def __enter__(self): return self
    def __exit__(self, *_): pass
    def on_update(self, callback): self.updates.append(callback); return callback
    def on_click(self, callback): self.clicks.append(callback); return callback
    def close(self): pass
    def emit(self, kind, event):
        for callback in getattr(self, kind): callback(event)


class GUI:
    def __init__(self): self.handles = {}; self.created = []
    def __getattr__(self, method):
        def create(label, *args, **kwargs):
            handle = Handle(value=kwargs.pop("initial_value", args[0] if args else None),
                            updates=[], clicks=[], order=kwargs.pop("order", 0), **kwargs)
            self.handles[label] = handle
            self.created.append((method, label, handle))
            return handle
        return create


class VisRefactorTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("viser.ViserServer", side_effect=AssertionError("No server")))
        self.enterContext(patch("threading.Thread", side_effect=AssertionError("No thread")))

    def test_geometry_reference_reader_keeps_rgb_order_and_shape_guard(self):
        from modal_gaussians.motion.neural.prepared import _read_reference_rgb
        with TemporaryDirectory() as directory:
            image = np.zeros((3, 4, 3), np.uint8)
            image[:] = [10, 30, 90]
            cv2.imwrite(str(Path(directory) / "001.png"), image)
            flow = NS(manifest={"inputs": {"sequence": {"image_directory": directory}},
                               "reference_frame_name": "001"}, arrays=NS(mask_union=np.zeros((3, 4), bool)))
            np.testing.assert_array_equal(_read_reference_rgb(flow), image[..., ::-1])
            flow.arrays.mask_union = np.zeros((4, 4), bool)
            with self.assertRaisesRegex(ValueError, "shape"):
                _read_reference_rgb(flow)

    def test_camera_conversion_and_keyframe_roundtrip(self):
        pose = tf.SE3.from_rotation_and_translation(tf.SO3.from_z_radians(.7), np.array([2., 3., 4.]))
        camera = NS(name="camera", height=120, width=200, K=torch.diag(torch.tensor([100., 150., 1.])),
                    world_to_camera=torch.tensor(pose.inverse().as_matrix()))
        converted = ViewerCamera.from_camera(camera, "view2")
        self.assertEqual(converted.label, "view2")
        self.assertEqual(ViewerCamera.from_camera(camera).label, "camera")
        np.testing.assert_allclose(converted.c2w, pose.as_matrix(), atol=1e-14)
        self.assertAlmostEqual(converted.fov, 2 * np.arctan(.4))
        self.assertAlmostEqual(converted.aspect, 200/120)
        frame = Keyframe(17., pose.translation(), pose.rotation().wxyz, True, .9, 200/120, True, 1.25)
        record = frame.to_dict(75.)
        restored = Keyframe.from_dict(json.loads(json.dumps(record)), 75.)
        np.testing.assert_allclose(restored.to_dict(75.)["matrix"], record["matrix"], atol=1e-14)
        self.assertEqual(restored.time, 17.)
        self.assertAlmostEqual(restored.override_fov_rad, .9)
        self.assertEqual(restored.override_transition_sec, 1.25)
        record.pop("time")  # Previous exports omitted it.
        self.assertEqual(Keyframe.from_dict(record, 75.).time, 0.)

    def test_transition_duration_uses_same_open_and_closed_segments(self):
        path = CameraPath(Mock(), NS(value=0))
        path.default_transition_sec = 2.
        for loop in (False, True):
            path.loop = loop
            self.assertEqual(path.compute_duration(), 0.)
        first = Keyframe(0., np.zeros(3), np.array([1., 0, 0, 0]), False, 1., 1., True, .5)
        second = Keyframe(8., np.ones(3), np.array([1., 0, 0, 0]), False, 1., 1., False, None)
        path._keyframes = {0: (first, Mock()), 1: (second, Mock())}
        for loop, times in ((False, [0., 2.]), (True, [0., 2., 2.5])):
            path.loop = loop
            np.testing.assert_array_equal(path.compute_transition_times_cumsum(), times)
            self.assertEqual(path.compute_duration(), times[-1])

    def test_render_panel_reuses_slider_and_loads_its_saved_path(self):
        with TemporaryDirectory() as directory:
            gui = GUI()
            camera = NS(position=np.zeros(3), wxyz=np.array([1., 0, 0, 0]), fov=1.)
            client = NS(camera=camera, gui=gui)
            server = NS(gui=gui, scene=Mock(), get_clients=lambda: {1: client})
            timestep = NS(value=0)
            root = Path(directory) / "camera_paths"
            populate_render_tab(server, root, timestep)
            event = NS(client=client, client_id=1)
            controls = gui.handles
            self.assertNotIn(":", controls["Camera path name"].value)
            controls["Add Keyframe"].emit("clicks", event)
            camera.position = np.array([1., 2., 3.])
            camera.wxyz = tf.SO3.from_y_radians(.3).wxyz
            timestep.value = 9
            controls["Add Keyframe"].emit("clicks", event)
            slider = controls["Preview frame"]
            controls["FPS"].value = 2.
            controls["FPS"].emit("updates", event)
            self.assertIs(controls["Preview frame"], slider)
            self.assertEqual(slider.max, 3)
            self.assertEqual(sum(label == "Preview frame" for _, label, _ in gui.created), 1)
            controls["Save Camera Path"].emit("clicks", event)
            saved_path, = root.glob("*.json")
            before = json.loads(saved_path.read_text())
            self.assertEqual([frame["time"] for frame in before["keyframes"]], [0, 9])
            controls["Default FOV"].value = 100.
            controls["FPS"].value = 5.
            controls["Load Path"].emit("clicks", event)
            self.assertEqual(controls["Camera Path"].value, saved_path.name)
            controls["Load"].emit("clicks", event)
            controls["Save Camera Path"].emit("clicks", event)
            after = json.loads(saved_path.read_text())
            for key in ("default_fov", "fps", "seconds", "is_cycle", "smoothness_value"):
                self.assertEqual(before[key], after[key])
            for a, b in zip(before["keyframes"], after["keyframes"]):
                self.assertEqual(a["time"], b["time"])
                np.testing.assert_allclose(a["matrix"], b["matrix"], atol=1e-14)
            for a, b in zip(before["camera_path"], after["camera_path"]):
                np.testing.assert_allclose(a["w2c"], b["w2c"], atol=1e-14)
                np.testing.assert_allclose(a["K"], b["K"], atol=1e-14)

if __name__ == "__main__":
    unittest.main()
