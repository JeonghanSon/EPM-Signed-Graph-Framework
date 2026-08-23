import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from signed_epm.data.preprocess import ROOT, load_json
from signed_epm.training.tune import run_candidate, select_common_configuration


class TuningSelectionTests(unittest.TestCase):
    def test_sgcn_search_matches_common_protocol(self):
        config = load_json(ROOT / "configs/models/sgcn.json")
        self.assertEqual(config["input_dimensions"], [64])
        self.assertEqual(config["output_dimensions"], [64])
        self.assertEqual(config["layers"], [2])
        self.assertEqual(config["learning_rates"], [0.01, 0.001, 0.0001])
        self.assertEqual(config["epochs"], [50, 100, 200])
        self.assertEqual(config["search_profile"], "semba_common")

    def test_selects_mean_validation_not_best_single_seed(self):
        rows = []
        for seed, score_a, score_b in [(0, 0.9, 0.7), (1, 0.4, 0.7), (2, 0.4, 0.7)]:
            for lr, score in [(0.01, score_a), (0.001, score_b)]:
                rows.append({"seed": seed, "input_dimension": 64, "output_dimension": 64,
                             "layers": 2, "learning_rate": lr, "epochs": 50,
                             "class_weight": "none", "validation_macro_f1": score})
        common, selected, _ = select_common_configuration(pd.DataFrame(rows))
        self.assertEqual(common["learning_rate"], 0.001)
        self.assertEqual(selected.seed.tolist(), [0, 1, 2])

    def test_rejects_configuration_missing_a_seed(self):
        frame = pd.DataFrame([
            {"seed": seed, "input_dimension": 64, "output_dimension": 64, "layers": 2,
             "learning_rate": lr, "epochs": 50, "class_weight": "none",
             "validation_macro_f1": score}
            for seed, lr, score in [(0, 0.01, 0.8), (0, 0.001, 0.7), (1, 0.001, 0.7)]
        ])
        common, _, _ = select_common_configuration(frame)
        self.assertEqual(common["learning_rate"], 0.001)

    def test_seed_specific_training_graph_template(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            graph = root / "graphs" / "seed_3" / "train.csv"
            graph.parent.mkdir(parents=True)
            graph.write_text("source,target,sign\n0,1,1\n")
            output = root / "runs"
            args = Namespace(
                output_root=output, overwrite_candidates=False,
                model="sgcn", task="signlink_3class", data_dir=root,
                cache_root=root / "cache", device="cpu",
                train_path=None,
                train_path_template=str(root / "graphs" / "seed_{seed}" / "train.csv"),
            )
            run_dir = output / "seed_3" / "candidates" / (
                "in_64__out_64__layers_2__lr_0p01__epochs_50__weight_none"
            )
            run_dir.mkdir(parents=True)
            (run_dir / "metrics.json").write_text(
                '{"test": null, "validation": {"macro_f1": 0.5, '
                '"weighted_f1": 0.6, "accuracy": 0.7, "auc": null}, '
                '"graph_fingerprint": "abc"}'
            )
            # Existing metrics make this a resume-path test; removing them lets
            # the command construction and seed substitution be observed.
            (run_dir / "metrics.json").unlink()
            with patch("signed_epm.training.tune.subprocess.run") as called:
                def write_metrics(command, check):
                    self.assertIn(str(graph), command)
                    (run_dir / "metrics.json").write_text(
                        '{"test": null, "validation": {"macro_f1": 0.5, '
                        '"weighted_f1": 0.6, "accuracy": 0.7, "auc": null}, '
                        '"graph_fingerprint": "abc"}'
                    )
                called.side_effect = write_metrics
                row = run_candidate(args, 3, 64, 64, 2, 0.01, 50, "none")
            self.assertEqual(row["seed"], 3)


if __name__ == "__main__":
    unittest.main()
