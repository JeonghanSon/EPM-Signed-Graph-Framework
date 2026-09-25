# EPM Signed Graph Framework

This repository contains the reproducible implementation of embedding-aware
polarization measurement and mitigation for signed graphs. It supports SGCN
and SDGNN backbones while keeping representation learning separate from EPM.

The public pipeline contains preprocessing, backbone training, the revised
signed polarization measure, cached mitigation preparation, uniformly sampled
gray-zone candidates, pooled greedy edge selection, graph materialization,
retraining, and synthetic validation. Exploratory notebooks, plotting scripts,
server launchers, logs, and manuscript working files are intentionally absent.

## Installation

Python 3.10 is recommended. Install a PyTorch build appropriate for your
machine first, then install this package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest
```

## Data and preprocessing

The raw BTC-Alpha file is included for a lightweight reproduction. Other
datasets can be downloaded as described in `data/README.md`.

```bash
bash experiments/preprocess.sh bitcoinalpha
```

This creates the fixed train/validation/test split, directed model view,
undirected physical-edge view, and signed-Louvain metadata under ignored
artifact directories.

## Backbone training

The paper model grid is stored in `configs/paper/model_search.json`. Selection
uses mean validation Macro-F1 over five seeds; test labels are never used for
model or intervention selection.

```bash
bash experiments/train_base.sh sgcn bitcoinalpha cuda
```

SGCN and SDGNN use the same preprocessing and evaluation protocol. Model
hyperparameters can also be supplied directly to `signed-epm-train`.

## Measurement

Measurement consumes a trained node-state file and the undirected training
graph. The paper uses negative conductance `eta=0.1`, antagonistic weight
`alpha=0.05`, and the dataset-specific PCA dimension recorded by preprocessing.

```bash
bash experiments/measure.sh \
  artifacts/path/to/node_embeddings.pt bitcoinalpha 14 \
  artifacts/measurement/seed_0
```

## Mitigation

Preparation is cached separately so that candidate sampling and multiple edge
budgets reuse the same communities, pair scores, and gray-node rankings:

```bash
bash experiments/prepare_mitigation.sh \
  artifacts/path/to/node_embeddings.pt bitcoinalpha 14 \
  artifacts/mitigation/seed_0/preparation 30
```

The final method has no pair-score threshold (`tau`), pair-degree constraint
(`dmax`), or legacy per-pair strength (`gamma`). It retains all eligible
community pairs, samples candidates uniformly without replacement, and applies
one pooled greedy budget. The public runner uses a candidate cap of
`20 * maximum_budget`:

```bash
bash experiments/select_mitigation.sh \
  bitcoinalpha \
  artifacts/mitigation/seed_0/preparation \
  artifacts/measurement/seed_0/opinion_coordinates.npy \
  1000 0 artifacts/mitigation/seed_0/selection
```

The selected positive physical edges are stored in selection order. Use
`signed-epm-materialize` to create augmented train snapshots at the desired
budget rate, then pass each generated
`train_snapshot_directed_augmented.csv` to `signed-epm-tune` or
`signed-epm-train`. The representation and downstream classifier must be
trained again on every augmented graph; base representations are not reused
for final evaluation.

The complete method settings are recorded in
`configs/paper/mitigation.json`. Large-graph runs use the same objective and
candidate policy with batched greedy selection for scalability.

## Synthetic validation

Exact five-seed synthetic graph bundles and the generation code are included.
To regenerate the networks and train a fresh SGCN for each condition:

```bash
bash experiments/run_synthetic.sh cuda
```

To use the exact bundled graphs:

```bash
bash experiments/run_synthetic_bundled.sh cuda
```

The configuration in `configs/synthetic.json` records graph size, SBM
conditions, signed-edge construction, opinions, seeds, and model settings.

## Repository layout

```text
configs/          dataset, model, task, measurement, and mitigation settings
data/             dataset documentation, BTC-Alpha raw data, synthetic bundles
experiments/      concise stage-by-stage reproduction commands
src/signed_epm/   preprocessing, models, measurement, mitigation, evaluation
tests/            unit tests for the released pipeline
```

Generated datasets, checkpoints, NumPy arrays, logs, figures, and result
artifacts are ignored by Git. No machine-specific paths or credentials are
required by the released commands.

For manuscript tables, create a CSV manifest with columns `dataset`, `seed`,
`base_metrics`, `mitigated_metrics`, `base_measurement`, and
`mitigated_measurement`, then run:

```bash
signed-epm-collect-results \
  --manifest artifacts/result_manifest.csv \
  --output artifacts/main_results.csv \
  --split test
```
