# EPM: Reviewer Artifact

This repository provides a compact implementation of embedding-aware
polarization measurement and mitigation (EPM) for signed networks. It is
intended for reviewers who want to inspect and run the core method described in
the paper. The artifact includes SGCN and SDGNN integrations and a lightweight
end-to-end example using BTC-Alpha.

The repository is intentionally limited to the code and data needed to verify
the main pipeline. Additional backbone integrations and complete experimental
artifacts will be released upon publication.

## Installation

Python 3.10 is recommended. Install a CUDA-compatible PyTorch build first when
using a GPU, then run:

```bash
pip install -e '.[test]'
pytest -q
```

## Quick reproduction

The following command runs preprocessing, SGCN training, polarization
measurement, 5% EPM augmentation, retraining, and final evaluation on
BTC-Alpha:

```bash
bash experiments/quickstart.sh cuda artifacts/quickstart
```

Use `cpu` instead of `cuda` on a CPU-only machine. Choose a different output
directory when running the command more than once.

The command prints the original and mitigated test Macro-F1 and polarization
scores. Detailed outputs are written to:

```text
artifacts/quickstart/base/metrics.json
artifacts/quickstart/base_measurement/measurement.json
artifacts/quickstart/selection/seed_0/selected_edges.csv
artifacts/quickstart/mitigated/metrics.json
artifacts/quickstart/mitigated_measurement/measurement.json
```

This single-seed run is a functional reproduction of the complete pipeline,
not a reproduction of the paper's aggregated five-seed tables.

## Data and paper settings

BTC-Alpha is included for the quick reproduction. Instructions and source links
for the other five datasets are provided in [`data/README.md`](data/README.md).

The paper uses preprocessing split seed 0 and reports five-seed results. Model
hyperparameters are selected by validation Macro-F1 and then fixed across the
base graph, EPM, and matched baselines. The small-network intervention rates
are 2.5%, 5%, 7.5%, and 10% of the physical training-edge count. Measurement,
sampling, and selection settings are recorded in
[`configs/paper/mitigation.json`](configs/paper/mitigation.json).

## Running another dataset

After placing a downloaded raw file under `data/raw/`, the individual stages
can be run as follows. Replace the example dataset, `k`, minimum community size,
and node-state path using [`configs/datasets.json`](configs/datasets.json) and
the selected run recorded in `selected_runs.csv`.

```bash
DATASET=bitcoinotc
MODEL=sgcn
K=17
MIN_COMMUNITY_SIZE=30
ROOT=artifacts/manual/$MODEL/$DATASET

bash experiments/preprocess.sh "$DATASET"
bash experiments/train_base.sh "$MODEL" "$DATASET" cuda

# Use node_embeddings.pt from one of the runs listed in:
# artifacts/runs/$MODEL/$DATASET/signlink_3class/base/selected_runs.csv
NODE_STATE=artifacts/path/to/selected/run/node_embeddings.pt

bash experiments/measure.sh \
  "$NODE_STATE" "$DATASET" "$K" "$ROOT/measurement/seed_0"
bash experiments/prepare_mitigation.sh \
  "$NODE_STATE" "$DATASET" "$K" "$ROOT/preparation/seed_0" \
  "$MIN_COMMUNITY_SIZE"

TRAIN_EDGES=$(($(wc -l < "data/processed/$DATASET/train_snapshot_undirected.csv") - 1))
MAX_BUDGET=$((TRAIN_EDGES * 10 / 100))
bash experiments/select_mitigation.sh \
  "$DATASET" "$ROOT/preparation/seed_0" \
  "$ROOT/measurement/seed_0/opinion_coordinates.npy" \
  "$MAX_BUDGET" 0 "$ROOT/selection/seed_0"
bash experiments/materialize_mitigation.sh \
  "$ROOT/selection" "$DATASET" "$ROOT/graphs" .025 .05 .075 .10
```

Set `DIRECTED_BACKBONE=1` for SDGNN materialization. The generated augmented
training snapshots can be passed to `signed-epm-train` using the selected base
configuration; run `signed-epm-train --help` for its arguments.

Random and Graph-Wide Greedy baselines are exposed through
`signed-epm-random` and `signed-epm-graph-wide`; run either command with
`--help` for its arguments.

## Repository layout

```text
configs/       model, task, dataset, and paper settings
data/          dataset instructions, metadata, and BTC-Alpha input
experiments/   quick reproduction and individual stage scripts
src/           EPM implementation
tests/         unit and regression tests
```

Generated datasets, checkpoints, logs, and result artifacts are excluded from
version control.

## License

The implementation is released under the [MIT License](LICENSE). Third-party
datasets remain subject to their original source terms and attribution
requirements.
