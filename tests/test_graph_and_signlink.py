import unittest

import pandas as pd

from signed_epm.graph import graph_fingerprint
from signed_epm.tasks.signlink import SignLinkProtocol, SignPredictionProtocol, pair_set, protocol_for


class GraphAndSignLinkTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({"source": [0, 1, 2], "target": [1, 2, 3], "sign": [1, -1, 1]})
        self.validation = pd.DataFrame({"source": [0], "target": [2], "sign": [-1]})
        self.test = pd.DataFrame({"source": [0], "target": [3], "sign": [1]})

    def test_fingerprint_is_order_independent_but_graph_specific(self):
        first = graph_fingerprint(self.train, directed=False)
        second = graph_fingerprint(self.train.iloc[::-1], directed=False)
        changed = self.train.copy(); changed.loc[0, "sign"] = -1
        self.assertEqual(first, second)
        self.assertNotEqual(first, graph_fingerprint(changed, directed=False))

    def test_strict_history_forbidden_sets(self):
        protocol = SignLinkProtocol(seed=0, directed=False)
        examples = protocol.examples(self.train, self.validation, self.test, self.train, 6)
        train_non_edges = examples["train"][examples["train"].label == 2]
        validation_non_edges = examples["validation"][examples["validation"].label == 2]
        test_non_edges = examples["test"][examples["test"].label == 2]
        self.assertFalse(pair_set(train_non_edges) & pair_set(self.train))
        self.assertFalse(pair_set(validation_non_edges) & (pair_set(self.train) | pair_set(self.validation)))
        self.assertFalse(pair_set(test_non_edges) &
                         (pair_set(self.train) | pair_set(self.validation) | pair_set(self.test)))

    def test_binary_sign_prediction_has_no_non_edges(self):
        protocol = SignPredictionProtocol(seed=0, directed=False)
        examples = protocol.examples(self.train, self.validation, self.test, self.train, 6)
        self.assertEqual(examples["train"].label.tolist(), [1, 0, 1])
        self.assertEqual(set(examples["train"].label), {0, 1})
        self.assertEqual(len(examples["validation"]), len(self.validation))
        self.assertIsInstance(protocol_for("sign_prediction_2class", 0, False),
                              SignPredictionProtocol)

    def test_augmented_train_edges_are_not_sampled_as_non_edges(self):
        augmented = pd.concat([
            self.train,
            pd.DataFrame({"source": [4], "target": [5], "sign": [1]}),
        ], ignore_index=True)
        protocol = SignLinkProtocol(seed=0, directed=False)
        examples = protocol.examples(augmented, self.validation, self.test, self.train, 6)
        added = {(4, 5)}
        for split in ("train", "validation", "test"):
            non_edges = examples[split][examples[split].label == 2]
            self.assertFalse(pair_set(non_edges) & added)


if __name__ == "__main__":
    unittest.main()
