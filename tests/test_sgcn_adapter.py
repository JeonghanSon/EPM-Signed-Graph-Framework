import unittest

import numpy as np
import pandas as pd

from signed_epm.models.sgcn import signed_unit_adjacency, signed_unit_tsvd


class SGCNAdapterTests(unittest.TestCase):
    def setUp(self):
        self.graph = pd.DataFrame({
            "source": [0, 1],
            "target": [1, 2],
            "sign": [1, -1],
        })

    def test_tsvd_adjacency_has_unit_signed_weights(self):
        actual = signed_unit_adjacency(self.graph, 3).toarray()
        expected = np.asarray([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, -1.0],
            [0.0, -1.0, 0.0],
        ])
        np.testing.assert_array_equal(actual, expected)

    def test_tsvd_is_independent_of_input_direction_and_order(self):
        reversed_graph = self.graph.iloc[::-1].copy()
        reversed_graph[["source", "target"]] = reversed_graph[["target", "source"]]
        first = signed_unit_tsvd(self.graph, 3, 2, random_state=0).numpy()
        second = signed_unit_tsvd(reversed_graph, 3, 2, random_state=0).numpy()
        np.testing.assert_allclose(first, second, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
