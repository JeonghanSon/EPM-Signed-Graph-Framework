import unittest

import numpy as np
import pandas as pd

from signed_epm.experiments.measurement_sensitivity import (
    pca_coordinates,
    raw_coordinates,
)
from signed_epm.polarization.measure import (
    build_weighted_laplacian,
    opinion_coordinates,
    polarization,
    polarization_from_laplacian,
    signed_energy_components,
)


class PolarizationTests(unittest.TestCase):
    def setUp(self):
        self.graph = pd.DataFrame({
            "source": [0, 1, 2, 3],
            "target": [1, 2, 3, 0],
            "sign": [1, -1, 1, 1],
        })

    def test_sparse_solver_matches_explicit_pseudoinverse(self):
        coordinates = np.array([
            [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],
        ])
        laplacian = build_weighted_laplacian(self.graph, 4, 1.0, 0.1)
        actual = polarization_from_laplacian(laplacian, coordinates)
        pinv = np.linalg.pinv(laplacian.toarray())
        expected = np.sqrt(np.mean([
            axis @ pinv @ axis for axis in coordinates.T
        ]))
        self.assertAlmostEqual(actual, expected, places=8)

    def test_opinion_coordinates_are_nodewise_normalized(self):
        state = np.array([
            [2.0, 0.0, 1.0], [0.0, 2.0, 1.0],
            [-2.0, 0.0, 1.0], [0.0, -2.0, 1.0],
        ])
        coordinates, singular = opinion_coordinates(state, 2)
        np.testing.assert_allclose(np.linalg.norm(coordinates, axis=1), 1.0)
        self.assertEqual(singular.shape, (2,))

    def test_measurement_sensitivity_coordinate_variants(self):
        state = np.asarray([
            [1., 0., 2.], [0., 1., 1.], [-1., 0., 0.], [0., -1., -1.],
        ])
        projected = pca_coordinates(state, 2, True)
        unnormalized = pca_coordinates(state, 2, False)
        raw = raw_coordinates(state)
        self.assertEqual(projected.shape, (4, 2))
        self.assertEqual(unnormalized.shape, (4, 2))
        self.assertEqual(raw.shape, state.shape)
        np.testing.assert_allclose(np.linalg.norm(projected, axis=1), 1.0)
        np.testing.assert_allclose(np.linalg.norm(raw, axis=1), 1.0)

    def test_disconnected_graph_is_rejected(self):
        disconnected = pd.DataFrame({
            "source": [0, 2], "target": [1, 3], "sign": [1, -1],
        })
        with self.assertRaisesRegex(ValueError, "connected"):
            build_weighted_laplacian(disconnected, 4)

    def test_canonical_score_combines_structural_and_antagonistic_energies(self):
        coordinates = np.array([
            [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],
        ])
        components = signed_energy_components(self.graph, coordinates, 1.0, 0.1)
        self.assertAlmostEqual(
            polarization(self.graph, coordinates, 1.0, 0.1, 0.05) ** 2,
            components["structural_energy"] + 0.05 * components["antagonistic_energy"],
            places=8,
        )
        self.assertAlmostEqual(
            polarization(self.graph, coordinates, 1.0, 0.1, 0.0) ** 2,
            components["structural_energy"],
            places=8,
        )


if __name__ == "__main__":
    unittest.main()
