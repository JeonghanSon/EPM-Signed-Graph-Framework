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

To run the same workflow with model selection, all five seeds, and every paper
budget on another downloaded dataset, use:

```bash
bash experiments/run_full.sh bitcoinotc
```

This defaults to SGCN and CUDA. The script is resumable and reuses completed
preprocessing, training, and evaluation outputs.

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

After placing a downloaded raw file under `data/raw/`, run the desired stages
individually.

Preprocess the dataset and estimate `k` with signed Louvain:

```bash
bash experiments/preprocess.sh [dataset]
```

Train the backbone and select its common five-seed configuration:

```bash
bash experiments/train_base.sh [model] [dataset]
```

This also trains and evaluates the selected configuration on all five seeds.

Measure polarization for the selected run:

```bash
bash experiments/measure.sh [model] [dataset] [seed]
```

Construct the mitigated graphs:

```bash
bash experiments/mitigate.sh [model] [dataset] [seed]
```

Retrain and evaluate the backbone on those graphs:

```bash
bash experiments/retrain.sh [model] [dataset] [seed] [device]
```

All arguments are optional; the defaults are SGCN, BTC-Alpha, seed 0, and CUDA.
The scripts locate the validation-selected node representation automatically
and use either the newly computed `k` or the stored paper value. Dataset-specific
community-size and small/large-network settings are also loaded automatically.

Candidate selection, graph materialization, matched baselines, and optional
arguments are available through the remaining scripts in `experiments/` and
the installed `signed-epm-*` commands.

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
