"""Synthetic viewer inputs and projections; no server or real scene is loaded."""
from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from modal_gaussians.iteration_cache import identity
from modal_gaussians import cli
from modal_gaussians.motion.common import projection
from modal_gaussians.vis import inputs, projections, spectrum, viewer


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class SyntheticInputs:
    def __init__(self, root):
        self.root = root
        self.views = [{"label": label, "camera_identity": label, "shape_hw": [4, 6],
            "index": i, "flow_identity": label + "-flow", "fps_hz": 20.,
            "motion_reference": {"reference_frame_name": "002", "reference_frame_index": 1,
                                 "selection_identity": "new-ref"}}
            for i, label in enumerate(("view1", "view2"))]
        self.points = torch.tensor([[0., 0., 1.], [1., 0., 2.], [0., 1., 2.]])
        self.scene = NS(manifest={"static_scene_identity": "scene", "foreground_identity": "fg"},
            foreground=NS(count=3, active=lambda: {"means": self.points,
                "quaternions": torch.tensor([[1., 0., 0., 0.]] * 3)}))
        self.scene.to = lambda device: self.scene
        self.cameras = [NS(label=v["label"], role="reference", name=v["label"],
            K=torch.eye(3), world_to_camera=torch.eye(4), width=6, height=4,
            to_manifest_record=lambda v=v: {"camera_identity": v["camera_identity"]}) for v in self.views]
        self.models, self.designs, self.records = {}, {}, []
        for i, frequency in enumerate((1., .5)):
            path = root / f"model{i}"
            dense = root / f"dense{i}"
            dense_views = []
            for view in self.views:
                export = root / f"export{i}-{view['label']}"
                export.mkdir()
                cv2.imwrite(str(export / "rgb.png"), np.full((4, 6, 3), 80, np.uint8))
                np.save(export / "modal.npy", np.full((1, 4, 6, 2), 2+3j, np.complex64))
                m = {"format": "modal_gaussians.sea_raft_selected_frequency_experiment", "status": "complete",
                    "frequency_hz": frequency, "reference_frame_name": "002", "reference_frame_index": 1,
                    "reference_selection": {"identity": "new-ref"}, "fps_hz": 20.,
                    "modes_shape": [1, 4, 6, 2], "modes_file": "modal.npy",
                    "reference_image": str(root / f"export0-{view['label']}" / "rgb.png")}
                write(export / "manifest.json", m)
                dense_views.append({**view, "modes_file": str(export / "modal.npy"),
                    "selected_source": {"path": str(export), "manifest_identity": identity(m)}})
            write(dense / "manifest.json", {"complex_2d_modes_identity": str(i), "views": dense_views})
            manifest = {"format": "modal_gaussians.completed_modes", "version": 16,
                "completion_method": "neural_component_field_with_stable_donors",
                "completed_modes_identity": f"model{i}", "static_scene": str(root / "scene"),
                "static_scene_identity": "scene", "foreground_identity": "fg", "views": self.views,
                "modes": [{"mode_slot": 0, "frequency_hz": frequency}],
                "complex_2d_modes": str(dense), "complex_2d_modes_identity": str(i), "selected_modal_supervision": {}}
            write(path / "manifest.json", manifest)
            arrays = {"phi": np.full((1, 3, 3), i+1j, np.complex64), "g_points": self.points.numpy().copy(),
                "support_class": np.ones((1, 3), np.int8), "observation_view_mask": np.ones((1, 3, 2), bool),
                "alphas": np.array([[1., 2j]], np.complex64), "alpha_identifiable_mask": np.ones((1, 2), bool),
                "g_edge_index": np.array([[0, i+1]]), "g_edge_weight": np.ones(1), "g_component_index": np.zeros(3, int),
                "c_control_point_index": np.arange(i+1), "c_positions": self.points.numpy()[:i+1].copy()}
            self.models[path] = NS(path=path, manifest=manifest, arrays=arrays,
                rotation=np.full((1, 3, 3), 2+i+1j, np.complex64),
                control_displacement=np.full((1, i+1, 3), i+2j, np.complex64))
            experiment = root / f"experiment{i}"
            design_path = experiment / "rendered_design"
            self.designs[design_path] = self.make_design(design_path, manifest, f"design{i}")
            self.records.append({"completed_modes": str(path), "completed_modes_identity": f"model{i}",
                "frequency_hz": frequency, "experiment": str(experiment), "status": "complete"})
        self.index = root / "results_index.json"
        write(self.index, self.records + [{"status": "training", "frequency_hz": 3.}])
        self.bank = root / "bank"
        bank_manifest = {**self.models[root / "model0"].manifest, "version": 17, "completed_modes_identity": "bank",
            "modes": [{"mode_slot": k, "frequency_hz": r["frequency_hz"]} for k, r in enumerate(self.records)],
            "sources": [{"path": r["completed_modes"], "identity": r["completed_modes_identity"], "slot": 0}
                        for r in self.records]}
        write(self.bank / "manifest.json", bank_manifest)
        self.models[self.bank] = NS(path=self.bank, manifest=bank_manifest)
        self.bank_design = self.make_design(root / "bank_design", bank_manifest, "bank-design")
        self.designs[self.bank_design.path] = self.bank_design
        cm = {"rendered_design": str(self.bank_design.path), "rendered_design_identity": "bank-design",
            "completed_modes_identity": "bank", "modes": bank_manifest["modes"], "rgb_coordinates_identity": "coords",
            "views": [{**self.views[0], "reference_frame_name": "002", "reference_frame_index": 1,
                       "frame_count": 3, "frame_offset": 0}]}
        self.coordinates = NS(manifest=cm, coordinates=np.array([[10+1j, 20+2j]] * 3, np.complex64))

    def make_design(self, path, model, name):
        views = [{**v, "camera_name": v["label"], "flow_reference_frame_name": "001",
                  "flow_reference_frame_index": 0} for v in self.views]
        manifest = {"rendered_design_identity": name, "completed_modes": str(self.root / model["completed_modes_identity"]),
            "completed_modes_identity": model["completed_modes_identity"], "static_scene_identity": "scene",
            "foreground_identity": "fg", "modes": model["modes"], "views": views,
            "settings": projection.RenderedDesignConfig().to_dict()}
        write(path / "manifest.json", manifest)
        return NS(path=path, manifest=manifest,
            samples={"view_sample_offsets": np.array([0, 2, 4]),
                     "sample_pixels_xy": np.array([[1, 1], [2, 2]] * 2)},
            design=np.tile(np.array([1., -2.] * len(model["modes"]), np.float32), (4, 2, 1)))

    def patches(self):
        stack = ExitStack()
        for module, name, value in ((inputs, "load_completed_modes", lambda p: self.models[Path(p)]),
            (inputs, "load_rendered_modal_design", lambda p: self.designs[Path(p)]),
            (inputs, "load_static_scene", lambda *args: self.scene),
            (inputs, "cameras_from_scene_manifest", lambda m: self.cameras),
            (viewer, "cameras_from_scene_manifest", lambda m: self.cameras),
            (inputs, "_load_coordinate_artifact", lambda p: ("rgb", self.coordinates))):
            stack.enter_context(patch.object(module, name, side_effect=value))
        return stack


class ViewerInputTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(viewer.viser, 'ViserServer', side_effect=AssertionError('No server in synthetic tests')))

    def test_cli_common_entry_and_unfitted_view_playback(self):
        parser = cli.build_parser()
        for option in ('--input', '--preview', '--result'):
            args = parser.parse_args(['viewer', option, 'source', '--work-dir', 'work'])
            with patch.object(viewer, 'run_modal_viewer') as run:
                self.assertEqual(cli._dispatch(parser, args, []), 0)
                self.assertEqual(run.call_args.kwargs['result_dir'], Path('source').resolve())
        args = parser.parse_args(['viewer', '--input', 'models', '--coordinates', 'q', '--work-dir', 'work'])
        with patch.object(viewer, 'run_modal_viewer') as run:
            cli._dispatch(parser, args, [])
            self.assertEqual(run.call_args.kwargs['coordinates'], Path('q').resolve())
        # Changing to an unfitted view disables stored playback and retains all modes.
        instance = viewer.ModalViserViewer.__new__(viewer.ModalViserViewer)
        fitted = NS(manifest={'views': [{'label': 'view1', 'frame_count': 4}]})
        normalized = inputs.ViewerInput(Path('source'), None, [], [{'label': 'view1'}, {'label': 'view2'}], fitted)
        data = viewer.ModalViewerData.__new__(viewer.ModalViewerData)
        data.result = normalized
        instance.data = data
        instance.playback_view = NS(value='view2')
        instance.drive = NS(value=viewer.DRIVE_FLOW)
        instance.timestep = NS(value=100, max=100)
        with patch.object(instance, 'request_render'):
            instance._on_playback_view(None)
            self.assertEqual(instance.drive.options, (viewer.DRIVE_MANUAL,))
            self.assertEqual(instance.drive.value, viewer.DRIVE_MANUAL)
            self.assertEqual(instance.timestep.max, 1799)
            instance.playback_view.value = 'view1'
            instance._on_playback_view(None)
            self.assertEqual(instance.drive.options, (viewer.DRIVE_FLOW, viewer.DRIVE_MANUAL))
            self.assertEqual(instance.timestep.value, 3)

    def test_async_selection_deduplication_failure_retry_and_shared_brightness(self):
        class Queue:
            def __init__(self, **kw): self.jobs = []
            def submit(self, fn): self.jobs.append(fn)
            def shutdown(self, **kw): self.jobs.clear()
            def drain(self):
                while self.jobs: self.jobs.pop(0)()
        with TemporaryDirectory() as tmp:
            f = SyntheticInputs(Path(tmp).resolve())
            with f.patches(), patch.object(projections, 'scene_cache', side_effect=lambda m, p: p), \
                    patch.object(spectrum, 'ThreadPoolExecutor', Queue):
                loaded = inputs.load_viewer_input(f.index)
                for mode in loaded.modes: mode.design = None
                store = projections.ViewerProjections(loaded, f.root / 'viewer')
                failed = {(0, 'view2')}
                calls = []
                def compute(k, label, settings):
                    calls.append((k, label))
                    if (k, label) in failed: raise RuntimeError('synthetic GPU failure')
                    return projections.ModalProjection(np.array([[1+k, 1]]),
                        np.full((1, 2), 1+k+2j, np.complex64))
                with patch.object(store, '_compute', side_effect=compute):
                    controller = spectrum.SpectrumComparisonController(loaded, projections=store)
                    self.assertEqual(calls, [])
                    self.assertTrue(np.isnan(controller.reconstructed_power).all())
                    notifications = []
                    controller.start(lambda: notifications.append((controller.view_id, controller.reconstructed_index)))
                    controller.select_mode(1)
                    controller.select_mode(1)
                    self.assertEqual(len(controller._executor.jobs), 2)
                    controller._executor.jobs.pop(0)()
                    self.assertEqual(controller.reconstructed_index, 1)
                    self.assertIn('1.000000', controller.status)
                    controller.select_view('view2')
                    controller.select_amplitude_normalization('all saved modes')
                    controller._executor.drain()
                    self.assertEqual((controller.view_id, controller.reconstructed_index), ('view2', 1))
                    self.assertEqual(len(calls), 4)
                    controller.select_mode(0)
                    self.assertIn('synthetic GPU failure', controller.status)
                    self.assertTrue(np.isnan(controller.reconstructed_power[0]))
                    self.assertIn('Shared brightness pending', controller.status)
                    failed.clear()
                    controller.complete_view()
                    controller._executor.drain()
                    self.assertTrue(np.isfinite(controller.reconstructed_power).all())
                    self.assertNotIn('pending', controller.status)
                    high = controller.modal_image_magnitude_hi
                    controller.select_mode(1)
                    self.assertEqual(controller.modal_image_magnitude_hi, high)
                    self.assertEqual(len(calls), 5)
                    controller.close()
                    self.assertEqual(controller._pending, set())
                x, y, marker = spectrum._spectrum_plot_data(np.array([.5, 1., 1.5]),
                    np.array([1., np.nan, 2.]), 1., 3.)
                self.assertTrue(np.isnan(y[(x > .5) & (x < 1.5)]).all())

    def test_models_index_old_wrappers_and_coordinate_columns(self):
        with TemporaryDirectory() as tmp:
            f = SyntheticInputs(Path(tmp).resolve())
            before = {p: p.read_bytes() for p in f.root.rglob('*') if p.is_file()}
            with f.patches():
                single = inputs.load_viewer_input(f.root / "model0")
                self.assertEqual(len(single.modes), 1)
                loaded = inputs.load_viewer_input(f.index, coordinates="coefficients")
                self.assertEqual([m.frequency for m in loaded.modes], [.5, 1.])
                self.assertEqual(loaded.coordinate_columns, (1, 0))
                self.assertEqual(set(loaded.coordinate_views), {"view1"})
                data = viewer.ModalViewerData.__new__(viewer.ModalViewerData)
                data.result, data.scene = loaded, f.scene
                data.coordinates = f.coordinates.coordinates[:, loaded.coordinate_columns]
                np.testing.assert_equal(data.coordinate(0, 0), [20+2j, 10+1j])
                with self.assertRaisesRegex(ValueError, 'no fitted'):
                    data.coordinate(1, 0)
                self.assertFalse(data.has_coordinates(1))
                for k, expected in enumerate((2, 1)):
                    data.select_mode(k)
                    data._load_graph_display(k)
                    self.assertEqual(len(data.control_positions), expected)
                    np.testing.assert_equal(data.graph_edge_gaussian_index, [[0, expected]])
                bank = inputs.load_viewer_input(f.bank)
                self.assertEqual([m.key for m in bank.modes], [m.key for m in loaded.modes])
                preview = f.root / 'preview'
                write(preview / 'manifest.json', {'format': 'modal_gaussians.modal_preview', 'version': 2,
                    'completed_modes': str(f.bank), 'completed_modes_identity': 'bank',
                    'rendered_design': str(f.bank_design.path), 'rendered_design_identity': 'bank-design'})
                old = inputs.load_viewer_input(preview)
                for a, b in zip(old.modes, loaded.modes):
                    np.testing.assert_equal(a.artifact.arrays['phi'][a.slot], b.artifact.arrays['phi'][b.slot])
                    np.testing.assert_equal(a.artifact.rotation[a.slot], b.artifact.rotation[b.slot])
                result = f.root / 'result'
                write(result / 'manifest.json', {'format': 'modal_gaussians.modal_result', 'version': 1, 'sources': {
                    'completed_modes': {'path': str(f.bank), 'identity_name': 'completed_modes_identity', 'identity': 'bank'},
                    'rendered_design': {'path': str(f.bank_design.path), 'identity_name': 'rendered_design_identity', 'identity': 'bank-design'},
                    'coordinates': {'path': 'coefficients', 'identity_name': 'rgb_coordinates_identity', 'identity': 'coords'}}})
                old = inputs.load_viewer_input(result)
                self.assertEqual(old.coordinate_columns, loaded.coordinate_columns)
                # Exercise the full data constructor on synthetic CPU tensors; never initialize CUDA or Viser.
                with patch.object(viewer.torch.cuda, 'is_available', return_value=True), \
                        patch.object(torch.Tensor, 'to', lambda tensor, *a, **kw: tensor):
                    index_data = viewer.ModalViewerData(f.index, coordinates='coefficients', work_dir=f.root / 'viewer')
                    old_data = viewer.ModalViewerData(result, work_dir=f.root / 'viewer')
                    q = index_data.coordinate(0, 1)
                    torch.testing.assert_close(index_data.deformed_means(q), old_data.deformed_means(q))
                    torch.testing.assert_close(index_data.deformed_quaternions(q), old_data.deformed_quaternions(q))
                    self.assertEqual(index_data.spectrum.available_view_ids, ('view1', 'view2'))
                with self.assertRaisesRegex(ValueError, 'override'):
                    inputs.load_viewer_input(result, coordinates='other')
                # The same loader supports a one-mode coefficient binding too.
                f.coordinates.manifest.update(completed_modes_identity='model0',
                    rendered_design=str(f.root / 'experiment0/rendered_design'), rendered_design_identity='design0',
                    modes=f.models[f.root / 'model0'].manifest['modes'])
                f.coordinates.coordinates = f.coordinates.coordinates[:, :1]
                self.assertEqual(inputs.load_viewer_input(f.root / 'model0', coordinates='coefficients').coordinate_columns, (0,))
            for p, value in before.items():
                self.assertEqual(p.read_bytes(), value)
            self.assertFalse((f.root / 'combined').exists())

    def test_reject_mixed_identity_order_reference_and_duplicate_frequency(self):
        with TemporaryDirectory() as tmp:
            f = SyntheticInputs(Path(tmp).resolve())
            with f.patches():
                for key in ('static_scene_identity', 'foreground_identity'):
                    m = f.models[f.root / 'model1'].manifest
                    old = m[key]
                    m[key] = 'wrong'
                    with self.assertRaises(ValueError): inputs.load_viewer_input(f.index)
                    m[key] = old
                bank = f.models[f.bank].manifest
                for key, wrong in (("identity", "other"), ("slot", 5)):
                    source = bank["sources"][0]
                    old = source[key]
                    source[key] = wrong
                    with self.assertRaisesRegex(ValueError, 'source identity'):
                        inputs.load_viewer_input(f.bank)
                    source[key] = old
                points = f.models[f.root / 'model1'].arrays['g_points']
                points[0, 0] += 1
                with self.assertRaisesRegex(ValueError, 'Gaussian order'): inputs.load_viewer_input(f.index)
                points[0, 0] -= 1
                write(f.index, [f.records[0], f.records[0]])
                with self.assertRaisesRegex(ValueError, 'Duplicate'): inputs.load_viewer_input(f.index)
                write(f.index, f.records)
                f.coordinates.manifest['completed_modes_identity'] = 'other-model'
                with self.assertRaisesRegex(ValueError, 'binding'): inputs.load_viewer_input(f.index, coordinates='q')
                f.coordinates.manifest['completed_modes_identity'] = 'bank'
                f.coordinates.manifest['views'][0]['reference_frame_index'] = 0
                with self.assertRaisesRegex(ValueError, 'reference'): inputs.load_viewer_input(f.index, coordinates='q')
                f.records[0]['completed_modes_identity'] = 'wrong'
                write(f.index, f.records)
                with self.assertRaisesRegex(ValueError, 'identity'): inputs.load_viewer_input(f.index)

    def test_projection_reuse_disk_cache_failure_and_invalidation(self):
        with TemporaryDirectory() as tmp:
            f = SyntheticInputs(Path(tmp).resolve())
            # Reuse the same reference image for a view across frequencies.
            for i in (0, 1):
                for label in ('view1', 'view2'):
                    path = f.root / f'export{i}-{label}' / 'manifest.json'
                    m = json.loads(path.read_text())
                    m['reference_image'] = str(f.root / f'export0-{label}/rgb.png')
                    write(path, m)
                    dpath = f.root / f'dense{i}/manifest.json'
                    dense = json.loads(dpath.read_text())
                    next(v for v in dense['views'] if v['label'] == label)['selected_source']['manifest_identity'] = identity(m)
                    write(dpath, dense)
            with f.patches(), patch.object(projections, 'scene_cache', side_effect=lambda m, p: p):
                loaded = inputs.load_viewer_input(f.index)
                store = projections.ViewerProjections(loaded, f.root / 'viewer')
                with patch.object(store, '_compute', side_effect=AssertionError('Do not render a cached design')):
                    sample = store.get(0, 'view2', compute=True)
                    np.testing.assert_equal(sample.values, 1+2j)
                loaded.modes[0].design = None
                store = projections.ViewerProjections(loaded, f.root / 'viewer')
                self.assertIsNone(store.get(0, 'view1'))
                value = projections.ModalProjection(np.array([[1, 1]]), np.array([[3+4j, 5+6j]], np.complex64))
                with patch.object(store, '_compute', return_value=value) as compute:
                    np.testing.assert_equal(store.get(0, 'view1', compute=True).values, value.values)
                    store.get(0, 'view1', compute=True)
                    compute.assert_called_once()
                reloaded = projections.ViewerProjections(loaded, f.root / 'viewer')
                with patch.object(reloaded, '_compute', side_effect=AssertionError('Disk cache hit')):
                    np.testing.assert_equal(reloaded.get(0, 'view1', compute=True).values, value.values)
                loaded.modes[0].settings['pixel_sample_stride'] = 3
                changed = projections.ViewerProjections(loaded, f.root / 'viewer')
                self.assertIsNone(changed.get(0, 'view1'))
                before = set(changed.cache_dir.iterdir())
                with patch.object(changed, '_compute', side_effect=RuntimeError('GPU failure')):
                    with self.assertRaisesRegex(RuntimeError, 'GPU failure'): changed.get(0, 'view1', compute=True)
                self.assertEqual(set(changed.cache_dir.iterdir()), before)
                with patch.object(changed, '_compute', return_value=value), patch.object(projections, 'publish_directory', side_effect=InterruptedError):
                    with self.assertRaises(InterruptedError): changed.get(0, 'view1', compute=True)
                self.assertEqual(set(changed.cache_dir.iterdir()), before)

    def test_shared_projection_packing_and_sampling_conventions(self):
        # A tiny feature rasterizer makes the exact old packing independently calculable.
        scene = NS(foreground=NS(active=lambda: {'means': torch.tensor([[0., 0., 1.], [.2, .1, 1.]])}, count=2),
                   manifest={})
        camera = NS(K=torch.tensor([[2., 0., 0.], [0., 3., 0.], [0., 0., 1.]]),
            world_to_camera=torch.eye(4), radial_distortion=.1, label='v')
        alpha = np.full((4, 6), .4, np.float32)
        mask = np.ones((4, 6), bool)
        cfg = projection.RenderedDesignConfig(mask_erosion_iterations=0)
        with patch.object(projection, 'render_motion_features', return_value=(None, torch.from_numpy(alpha))):
            pixels, sampled, jacobian, visible = projection.prepare_modal_projection(scene, camera, mask, cfg)
        expected = projection.projection_jacobian(scene.foreground.active()['means'].numpy(),
            camera.K.numpy(), camera.world_to_camera.numpy(), .1)[0]
        np.testing.assert_equal(jacobian, expected)
        self.assertEqual(visible, 2)
        np.testing.assert_equal((pixels, sampled)[0], projection.candidate_observation_pixels(scene, mask, alpha, cfg)[0])
        phi = np.array([[[1+2j, 3+4j, .1j], [2+1j, 1+3j, .2j]],
                        [[2+4j, 6+8j, .2j], [4+2j, 2+6j, .4j]]], np.complex64)
        weights = np.array([[.25, .75]] * len(pixels), np.float32)
        with patch.object(projection, 'sample_feature_render', side_effect=lambda s, c, features, p, a: weights @ features.numpy()):
            packed = projection.project_modal_features(scene, camera, phi, pixels, sampled, jacobian)
        expected = np.einsum('pg,gij,kgj->pik', weights, jacobian, phi)
        np.testing.assert_allclose(packed[:, :, 0::2], expected.real, rtol=1e-6)
        np.testing.assert_allclose(packed[:, :, 1::2], -expected.imag, rtol=1e-6)
        # Visible subjects use rendered alpha even when the inherited flow mask is empty.
        scene.manifest = {'partition': {'method': 'manual_subject_selection_v1'}}
        self.assertGreater(len(projection.candidate_observation_pixels(scene, None, alpha, cfg)[0]), 0)


if __name__ == '__main__':
    unittest.main()
