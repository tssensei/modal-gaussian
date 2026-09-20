"""Synthetic only: no scene artifacts or real-data validation."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import cupy as cp
from scipy.optimize import least_squares
from scipy.optimize._numdiff import approx_derivative

from modal_gaussians import synchronization as s
from modal_gaussians.synchronization_gpu import Workspace
from modal_gaussians import _alpha_trf as trf


def observations(views=3, count=30, noise=0.):
    rng = np.random.default_rng(312)
    j = rng.normal(size=(count, views, 2, 3)).astype(np.float32)
    phi = rng.normal(size=(count, 3)) + 1j*rng.normal(size=(count, 3))
    alpha = np.array([1, 1.3*np.exp(.4j), .7*np.exp(-.3j)])[:views]
    y = np.einsum('gvki,gi->gvk', j, phi)*alpha[None, :, None]
    y += noise*(rng.normal(size=y.shape)+1j*rng.normal(size=y.shape))
    rows = count*views
    p = s.PreparedObservations(np.zeros((count, 3), np.float32), np.repeat(np.arange(count), views),
        np.tile(np.arange(views), count), y.reshape(rows, 2).astype(np.complex64),
        j.reshape(rows, 2, 3), np.ones(rows), tuple(np.arange(g*views, (g+1)*views) for g in range(count)),
        tuple(f'view{v}' for v in range(views)))
    return p, alpha


class AlphaGPUTests(unittest.TestCase):
    def test_geometry_cache_and_view_subsets(self):
        p, _ = observations()
        size = len(p.obs_y)
        topology = SimpleNamespace(sample_view_index=p.obs_view_index, sample_offsets=np.arange(size+1),
            contributor_gaussian_index=p.obs_point_index, contributor_weight=p.obs_weights,
            contributor_jacobian=p.obs_jacobian)
        with tempfile.TemporaryDirectory() as folder:
            ws = Workspace(max_blocks=7)
            try:
                q = ws.prepare(p.points, topology, p.obs_y, p.view_labels, topology_identity='synthetic', cache_dir=folder)
                first = ws.constraints(q)
                expected = first.information.get()
                ws.close()
                with patch.object(cp.linalg, 'svd', side_effect=AssertionError('cached geometry decomposed again')):
                    q = ws.prepare(p.points, topology, .5*p.obs_y, p.view_labels, topology_identity='synthetic', cache_dir=folder)
                    np.testing.assert_allclose(ws.constraints(q).information.get(), expected, rtol=1e-9, atol=1e-11)
                allowed = np.array([True, False, True])
                subset = ws.constraints(q, allowed)
                reference = s._information_blocks(s._build_constraints(q, allowed), np.arange(3))
                np.testing.assert_allclose(subset.information.get(), reference, rtol=1e-9, atol=1e-11)
                self.assertEqual(len(list((Path(folder)/'alpha_geometry').glob('*/manifest.json'))), 3)
                q = ws.prepare(p.points, topology, p.obs_y, p.view_labels, topology_identity='changed', cache_dir=folder)
                ws.constraints(q)
                self.assertEqual(len(list((Path(folder)/'alpha_geometry').glob('*/manifest.json'))), 5)
            finally:
                ws.close()

    def test_allocation_retry_and_reservation(self):
        ws = Workspace()
        try:
            attempts = []
            def operation(lo, hi):
                attempts.append(hi-lo)
                if hi-lo > 2:
                    raise cp.cuda.memory.OutOfMemoryError(100, 100)
                return list(range(lo, hi))
            values = [v for _, _, chunk in ws._chunks(7, 1, operation) for v in chunk]
            self.assertEqual(values, list(range(7)))
            self.assertGreater(ws.stats['allocation_retries'], 0)
            ws.reserve_bytes = 2**60
            with self.assertRaises(MemoryError): ws._budget()
        finally:
            ws.close()

    def test_no_host_arrays_in_optimizer_and_failure_states(self):
        def gpu(x): return cp.stack((x[:, 0]-3., 2*x[:, 0]-6.), axis=1)
        bounds = (cp.array([-1.]), cp.array([1.]))
        # cupy.ndarray.get is C-defined; prohibit public conversion and assert device payloads.
        with patch.object(cp, 'asnumpy', side_effect=AssertionError('host transfer')):
            result = trf.least_squares(gpu, cp.array([0.]), bounds)
            self.assertIsInstance(result.jac, cp.ndarray)
            self.assertIsInstance(result.fun, cp.ndarray)
            self.assertEqual(int(result.active_mask[0]), 1)
            limited = trf.least_squares(gpu, cp.array([0.]), bounds, max_nfev=1)
            self.assertEqual(limited.status, 0)
            self.assertEqual(limited.nfev, 1)
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            trf.least_squares(lambda x:cp.full((len(x), 2), cp.nan), cp.array([0.]), bounds)
        def broken_trial(x): return cp.where(x < .2, x-1., cp.nan)
        cpu = least_squares(lambda x:np.where(x < .2, x-1., np.nan), [0.], bounds=([-1.], [1.]), max_nfev=20)
        gpu_result = trf.least_squares(broken_trial, cp.array([0.]), bounds, max_nfev=20)
        np.testing.assert_allclose(gpu_result.x.get(), cpu.x, rtol=1e-5, atol=1e-7)

    def test_constraints_residuals_and_blocks(self):
        p, _ = observations(noise=.01)
        cpu = s._build_constraints(p)
        views = np.arange(3)
        batch = s._build_profiled_batch(p, cpu, views)
        x = np.array([.2, -.1, .3, -.2])
        expected = s._profiled_residuals(batch, x, 0)
        for block in (None, 3):
            ws = Workspace(max_blocks=block)
            try:
                c = ws.constraints(p)
                np.testing.assert_allclose(c.information.get(), s._information_blocks(cpu, views), rtol=1e-9, atol=1e-11)
                actual = ws.residuals(c, cp.asarray(x)[None], cp.asarray(views))
                for a, b in zip(actual, expected):
                    np.testing.assert_allclose(a[0].get(), b, rtol=1e-9, atol=1e-11)
                for a, b in zip(ws.edge_statistics(c), s._edge_statistics(cpu, 3)):
                    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-11)
            finally:
                ws.close()

    def test_trf_and_difference(self):
        for initial in ([0., 0.], [1., -1.], [.7, .7]):
            x = np.array(initial)
            lb, ub = np.full(2, -1.), np.ones(2)
            def cpu(x): return np.array([10*(x[1]-x[0]**2), 1-x[0], .1*x[1]])
            def gpu(x): return cp.stack((10*(x[:, 1]-x[:, 0]**2), 1-x[:, 0], .1*x[:, 1]), axis=1)
            actual = trf.difference(gpu, cp.asarray(x), cp.asarray(cpu(x)), cp.asarray(lb), cp.asarray(ub))
            np.testing.assert_allclose(actual.get(), approx_derivative(cpu, x, method='2-point', bounds=(lb, ub)), rtol=1e-4, atol=1e-7)
            a, b = trf.least_squares(gpu, cp.asarray(x), (lb, ub)), least_squares(cpu, x, bounds=(lb, ub), max_nfev=500)
            np.testing.assert_allclose(a.x.get(), b.x, rtol=1e-5, atol=1e-7)
            np.testing.assert_array_equal(a.active_mask.get(), b.active_mask)

    def test_full_alpha_and_exclusion(self):
        for views, noise in ((2, 0), (3, 0), (3, .01)):
            p, expected = observations(views, noise=noise)
            cpu = s.solve_alpha_sync(p, backend='cpu')
            gpu = s.solve_alpha_sync(p)
            np.testing.assert_allclose(gpu.alphas, cpu.alphas, rtol=1e-5, atol=1e-7)
            np.testing.assert_array_equal(gpu.identifiable_mask, cpu.identifiable_mask)
            np.testing.assert_array_equal(gpu.exclusion_reason, cpu.exclusion_reason)
        p, _ = observations()
        for broken in (replace(p, obs_y=np.zeros_like(p.obs_y)), replace(p, obs_jacobian=np.zeros_like(p.obs_jacobian))):
            a, b = s.solve_alpha_sync(broken), s.solve_alpha_sync(broken, backend='cpu')
            np.testing.assert_array_equal(a.identifiable_mask, b.identifiable_mask)
            np.testing.assert_array_equal(a.exclusion_reason, b.exclusion_reason)

    def test_outliers_degenerate_and_gain_bounds(self):
        p, _ = observations()
        outlier = p.obs_y.copy(); outlier[:5] *= 10
        oversized = p.obs_y.copy(); oversized[p.obs_view_index == 1] *= 10
        rank_two = p.obs_jacobian.copy(); rank_two[:, :, 2] = 0
        weak = p.obs_weights.copy(); weak[p.obs_view_index == 2] = 0
        for case in (replace(p, obs_y=outlier), replace(p, obs_y=oversized),
                     replace(p, obs_jacobian=rank_two), replace(p, obs_weights=weak)):
            a, b = s.solve_alpha_sync(case), s.solve_alpha_sync(case, backend='cpu')
            np.testing.assert_allclose(a.alphas, b.alphas, rtol=1e-5, atol=1e-7)
            np.testing.assert_array_equal(a.identifiable_mask, b.identifiable_mask)
            np.testing.assert_array_equal(a.exclusion_reason, b.exclusion_reason)
            np.testing.assert_array_equal(a.gain_bound_active_mask, b.gain_bound_active_mask)

    def test_defaults_no_fallback_and_old_metadata(self):
        from modal_gaussians.cli import build_parser
        from modal_gaussians.motion.neural import batch
        from modal_gaussians.iteration_cache import atomic_json
        parser = build_parser()
        for args in (['motion','prepare-selected-modal','--prepared','p','--view','a','v',
                      '--frequency-hz','1','--output','o'],
                     ['motion','batch-neural','--modal-images','m','--prepared','p','--geometry-graph','g',
                      '--config','c','--output','o']):
            self.assertEqual(parser.parse_args(args).alpha_backend, 'cupy')
            self.assertEqual(parser.parse_args(args+['--alpha-backend','cpu']).alpha_backend, 'cpu')
        p, _ = observations()
        with patch('modal_gaussians.synchronization_gpu.Workspace', side_effect=RuntimeError('CUDA unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'CUDA unavailable'): s.solve_alpha_sync(p)
            self.assertTrue(s.solve_alpha_sync(p, backend='cpu').identifiable_mask.all())
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder)/'manifest.json'
            atomic_json(marker, dict(format='modal_gaussians.neural_prepared', source=dict(
                modes=[dict(frequency_hz=1)], selected_modal_supervision=dict(parent_prepared_identity='parent'))))
            job = dict(frequency_hz=1, alpha_contract={'backend':'cpu'})
            self.assertTrue(batch._published(job, ('prepare',marker,[]), {'prepared_identity':'parent'}))
            job['alpha_contract'] = s.backend_identity('cupy')
            with self.assertRaisesRegex(ValueError, 'alpha preparation'):
                batch._published(job, ('prepare',marker,[]), {'prepared_identity':'parent'})


if __name__ == '__main__':
    unittest.main()
