"""Synthetic CUDA rendering only; never loads scene-library data."""
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from modal_gaussians.static import Camera, GaussianSet, ForegroundBackgroundScene
from modal_gaussians.motion.neural.neural_modes import NeuralModesConfig, FrozenModalProjector
from modal_gaussians.motion.neural.modal_projection import CachedModalRasterizer
from modal_gaussians.motion.common.projection import render_motion_features


def scene_camera(subject=False, radial=False):
    device = 'cuda'
    def part(z):
        means = torch.tensor([[-.2, 0, z], [.2, .1, z+.15]], device=device)
        return GaussianSet(means, torch.tensor([[1., 0, 0, 0]]*2, device=device),
            torch.full((2, 3), -1.9, device=device), torch.zeros((2, 3), device=device),
            torch.full((2,), 1., device=device))
    scene = ForegroundBackgroundScene(part(2.), part(1.8), manifest={
        'partition': {'method': 'manual_subject_selection_v1' if subject else 'synthetic'}})
    scene.requires_grad_(False)
    K = torch.tensor([[25., 0, 12], [0, 25, 10], [0, 0, 1]], device=device)
    camera = Camera('synthetic', 'reference', 'view1', 24, 20, K,
        torch.eye(4, device=device), torch.eye(4, device=device), 'SIMPLE_RADIAL',
        (25., 12., 10., -.06 if radial else 0.), '', '', '', '', radial)
    pixels = np.array([[10, 10], [12, 10], [14, 11], [1, 1], [23, 19]], dtype=np.int64)
    return scene, camera, pixels


class ProjectionConfigTests(unittest.TestCase):
    def test_old_config_and_backend_contract(self):
        old = NeuralModesConfig().to_dict()
        self.assertNotIn('modal_projection_backend', old)
        self.assertEqual(NeuralModesConfig.from_dict(old).modal_projection_backend, 'dynamic')
        self.assertEqual(NeuralModesConfig.from_dict(dict(old, modal_projection_backend='dynamic')).to_dict(), old)
        cached = replace(NeuralModesConfig(), modal_projection_backend='cached').to_dict()
        self.assertEqual(NeuralModesConfig.from_dict(cached).to_dict(), cached)
        from modal_gaussians.motion.neural.iteration import resolve_config
        from modal_gaussians.motion.neural.baseline import baseline_overrides
        from modal_gaussians.motion.common.projection import RenderedDesignConfig
        from modal_gaussians.motion.neural.continuation import increased_limit
        defaults = dict(neural=old, design=RenderedDesignConfig().to_dict(),
                        fragment=baseline_overrides()['fragment'])
        resolved = resolve_config(defaults, {'neural': {'modal_projection_backend': 'cached'}})
        self.assertEqual(resolved['neural']['modal_projection_backend'], 'cached')
        with self.assertRaisesRegex(ValueError, 'only increase'):
            increased_limit(old, dict(cached, max_iterations=old['max_iterations']+1))
        with self.assertRaisesRegex(ValueError, 'modal_projection_backend'):
            replace(NeuralModesConfig(), modal_projection_backend='invalid').validate()


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA rendering check')
class CachedProjectionTests(unittest.TestCase):
    def test_prepared_training_selects_cached_backend(self):
        from modal_gaussians.motion.neural.prepared import PreparedNeuralInputs
        from modal_gaussians.motion.neural.neural_modes import _source_identity
        from modal_gaussians.iteration_cache import Timings
        scene, camera, pixels = scene_camera()
        record = camera.to_manifest_record()
        scene.manifest = {'cameras': [record]}
        source = {name: 'synthetic' for name in ('static_scene_identity', 'foreground_identity',
            'topology_identity', 'gaussian_measurements_identity', 'observed_structure_graph_identity',
            'alignment_identity', 'complex_2d_modes_identity')}
        source.update(modes=[{'frequency_hz': 1.}], views=[{'label': 'view1', 'camera_identity': record['camera_identity']}])
        fixed = dict(view_sample_offsets=np.array([0, len(pixels)]), sample_pixels_xy=pixels,
                     sample_confidence=np.ones(len(pixels), np.float32))
        prepared = PreparedNeuralInputs(Path('synthetic'),
            dict(source_identity=_source_identity(source), defaults={'neural': NeuralModesConfig().to_dict()}),
            dict(v0_jacobian=np.ones((2, 2, 3), np.float32), v0_depth=np.ones((20, 24)), v0_alpha=np.ones((20, 24))))
        result = prepared.training_inputs(source, scene, None,
            replace(NeuralModesConfig(), modal_projection_backend='cached'), 'cuda', Timings(), frozen_arrays=fixed)
        self.assertIs(result[0], fixed)
        self.assertIsNotNone(result[1][0].cached)

    def test_render_and_feature_gradient_match_without_reprojection(self):
        torch.manual_seed(123)
        for subject in (False, True):
            for radial in (False, True):
                with self.subTest(subject=subject, radial=radial):
                    scene, camera, pixels = scene_camera(subject, radial)
                    cached = CachedModalRasterizer(scene, camera, pixels)
                    for channels in (1, 4):
                        value = torch.randn((2, channels), device='cuda', requires_grad=True)
                        image, _ = render_motion_features(scene, camera, value)
                        expected = image[pixels[:, 1], pixels[:, 0]]
                        cotangent = torch.randn_like(expected)
                        expected_grad, = torch.autograd.grad((expected*cotangent).sum(), value)
                        with patch('gsplat.rendering.fully_fused_projection', side_effect=AssertionError('reprojection')):
                            actual = cached(value)
                            actual_grad, = torch.autograd.grad((actual*cotangent).sum(), value)
                        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                        torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=2e-6)

    def test_complex_projector_normalization_and_backward(self):
        scene, camera, pixels = scene_camera(True, True)
        jacobian = torch.randn((2, 2, 3), device='cuda')
        alpha = torch.tensor([.4, .5, .6, .2, .3], device='cuda')
        dynamic = FrozenModalProjector(scene, camera, jacobian, pixels, alpha)
        cached = FrozenModalProjector(scene, camera, jacobian, pixels, alpha, backend='cached')
        for _ in range(2):
            phi = torch.randn((2, 3), device='cuda', dtype=torch.complex64, requires_grad=True)
            a, b = dynamic(phi), cached(phi)
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
            da, = torch.autograd.grad(a.abs().square().sum(), phi)
            db, = torch.autograd.grad(b.abs().square().sum(), phi)
            torch.testing.assert_close(da, db, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(dynamic.contribution_mass(), cached.contribution_mass(), rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(dynamic.projection_sensitivity(), cached.projection_sensitivity(), rtol=2e-5, atol=2e-6)

    def test_geometry_changes_and_geometry_gradients_are_rejected(self):
        for name in ('means', 'quaternions', 'log_scales', 'opacity_logits'):
            scene, camera, pixels = scene_camera()
            cached = CachedModalRasterizer(scene, camera, pixels)
            with torch.no_grad():
                scene.foreground.params[name].add_(.01)
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                cached(torch.ones((2, 4), device='cuda'))
        for field in ('K', 'world_to_camera'):
            scene, camera, pixels = scene_camera()
            cached = CachedModalRasterizer(scene, camera, pixels)
            getattr(camera, field).add_(.001)
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                cached(torch.ones((2, 4), device='cuda'))
        scene, camera, pixels = scene_camera(True)
        cached = CachedModalRasterizer(scene, camera, pixels)
        scene.background.replace_parameter('means', torch.zeros((3, 3), device='cuda')).requires_grad_(False)
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            cached(torch.ones((2, 4), device='cuda'))
        scene, camera, pixels = scene_camera()
        cached = CachedModalRasterizer(scene, camera, pixels)
        scene.foreground.params['means'].requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, 'frozen geometry'):
            cached(torch.ones((2, 4), device='cuda'))
        # The original dynamic path still carries geometry gradients.
        image, _ = render_motion_features(scene, camera, torch.ones((2, 4), device='cuda'))
        grad, = torch.autograd.grad(image.square().sum(), scene.foreground.params['means'])
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
