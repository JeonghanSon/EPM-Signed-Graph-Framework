import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from signed_epm.data.preprocess import connected, load_json, sha256, write_dataset


class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]

    def test_raw_manifest_matches_local_files(self):
        manifest = json.loads((self.root / "data/metadata/raw_files.json").read_text())
        for item in manifest["files"].values():
            path = self.root / "data/raw" / item["filename"]
            self.assertEqual(path.stat().st_size, item["bytes"])
            self.assertEqual(sha256(path), item["sha256"])

    def test_connected_rejects_isolated_node(self):
        frame = pd.DataFrame({"source": [0], "target": [1]})
        self.assertTrue(connected(frame, 2))
        self.assertFalse(connected(frame, 3))

    def test_bitcoinalpha_preprocessing_statistics(self):
        spec = load_json(self.root / "configs/datasets.json")["bitcoinalpha"]
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_dataset(
                "bitcoinalpha", spec, self.root / "data/raw" / spec["raw_file"],
                Path(directory) / "bitcoinalpha", 42, (0.7, 0.15, 0.15), False,
            )
        self.assertEqual(manifest["counts"]["num_nodes"], 3775)
        self.assertEqual(manifest["counts"]["splits"],
                         {"train": 9884, "val": 2118, "test": 2118})
        self.assertEqual(manifest["counts"]["positive"],
                         {"train": 8976, "val": 1896, "test": 1874})
        self.assertEqual(manifest["counts"]["negative"],
                         {"train": 908, "val": 222, "test": 244})
        self.assertEqual(manifest["validation"], {
            "train_connected": True, "pair_disjoint": True, "transductive": True,
        })

    def test_blank_endpoints_are_not_nodes(self):
        from signed_epm.data.preprocess import build_events
        raw = pd.DataFrame({"source": ["", "alice"], "target": ["bob", "bob"],
                            "weight": [1, -1], "timestamp": [1, 2]})
        events, mapping = build_events(raw)
        self.assertEqual(len(events), 1)
        self.assertNotIn("", mapping.values())


if __name__ == "__main__":
    unittest.main()
