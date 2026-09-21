"""CPU-only absolute/relative modal loss and artifact configuration contracts."""
import copy
from dataclasses import replace
import unittest

import torch

from modal_gaussians.motion.neural import neural_field as nf, neural_modes as nm


class ModalLossNormalizationTests(unittest.TestCase):
    def test_configuration_and_historical_defaults(self):
        for cls in (nf.NeuralFieldConfig, nm.NeuralModesConfig):
            historical = cls().to_dict()
            self.assertNotIn("data_loss_normalization", historical)
            self.assertEqual(cls.from_dict(historical).data_loss_normalization, "view_rms")
            explicit = dict(historical, data_loss_normalization="view_rms")
            self.assertEqual(cls.from_dict(explicit).to_dict(), historical)
            absolute = dict(historical, data_loss_normalization="none")
            self.assertEqual(cls.from_dict(absolute).to_dict(), absolute)
            with self.assertRaises(ValueError):
                replace(cls(), data_loss_normalization="invalid").validate()
        self.assertEqual(nm._field_config(nm.NeuralModesConfig(
            data_loss_normalization="none")).data_loss_normalization, "none")

    def test_two_view_loss_and_gradients(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        points = torch.tensor([[0., 0., 0.], [.3, .1, 0.], [.5, -.2, .1]], dtype=torch.float64)
        geometry = nf.NeuralFieldGeometry.from_arrays(dict(
            gaussian_positions=points, control_positions=points,
            interpolation_indptr=[0, 1, 2, 3], interpolation_indices=[0, 1, 2],
            interpolation_weights=[1., 1., 1.], gaussian_edges=[[0, 1], [1, 2]],
            control_edges=[[0, 1], [1, 2]]), dtype=torch.float64)
        target = torch.tensor([[.05 + .1j, .2 - .2j], [2. + .3j, -1j], [0., 0.]], dtype=torch.complex128)
        weights = torch.tensor([.1, .2, .7], dtype=torch.float64)
        observations = [nf.ModalObservation(target, lambda x: x[:, :2], alpha=.8 + .3j),
                        nf.ModalObservation(target.flip(0), lambda x: x[:, 1:], alpha=1.2 - .1j)]
        prepared = [(o, o.target, weights, o.alpha) for o in observations]
        scales = [.5, 2.]
        runtime = nf._prepare_runtime(geometry, 1., compute_rotation=False)
        config = nf.NeuralFieldConfig(hidden_dim=8, message_layers=1, rotation_weight=0., deformation_weight=.03)
        with torch.random.fork_rng():
            torch.manual_seed(17)
            initial = nf.PerFrequencyModalGNN(config, control_count=3).double()
        with torch.no_grad():
            initial.head.weight.fill_(.03)
        results = {}
        for normalization in ("view_rms", "none"):
            settings = replace(config, data_loss_normalization=normalization)
            actual, expected = copy.deepcopy(initial), copy.deepcopy(initial)
            field = nf.model_field(expected, geometry, length_scale=1., amplitude_scale=1.)
            terms = []
            for observation, scale in zip(observations, scales):
                error = observation.alpha * observation.project(field.field) - observation.target
                radius = torch.linalg.vector_norm(error, dim=-1)
                if normalization == "view_rms":
                    radius = radius / scale
                terms.append((weights * torch.where(radius <= 1., radius.square() / 2., radius - .5)).sum() / 2.)
            edge, _ = nf.structural_losses(geometry, field, length_scale=1., amplitude_scale=1., compute_rotation=False)
            loss = sum(terms) + .03 * edge
            loss.backward()
            result = nf._objective(actual, geometry, prepared, scales, settings, runtime,
                                   length=1., amplitude=1., backward=True)
            self.assertAlmostEqual(result["loss"], float(loss.detach()), places=12)
            for a, e in zip(actual.parameters(), expected.parameters()):
                if e.grad is None:
                    self.assertIsNone(a.grad)
                else:
                    torch.testing.assert_close(a.grad, e.grad, rtol=1e-9, atol=1e-11)
            if normalization == "none":
                changed = nf._objective(actual, geometry, prepared, [17., .01], settings, runtime,
                                        length=1., amplitude=1., backward=False)
                self.assertEqual(result["data_loss"], changed["data_loss"])
            results[normalization] = result
        self.assertEqual(results["none"]["edge_loss"], results["view_rms"]["edge_loss"])
        self.assertNotEqual(results["none"]["data_loss"], results["view_rms"]["data_loss"])


if __name__ == "__main__":
    unittest.main()
