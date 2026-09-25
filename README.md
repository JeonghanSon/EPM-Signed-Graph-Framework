# EPM Signed Graph Framework

This repository implements embedding-aware polarization measurement and
mitigation for signed graphs with SGCN and SDGNN backbones.

The canonical mitigation pipeline is deliberately stage-oriented:

1. preprocess a connected transductive train/validation/test split;
2. train a signed graph encoder and a separate multinomial logistic probe;
3. measure signed polarization from PCA/L2-normalized node coordinates;
4. cache KMeans communities and score-free gray-zone rankings;
5. uniformly sample unique physical candidates from the union of every
   retained community-pair gray space;
6. select one pooled greedy edge order and materialize physical-budget
   prefixes;
7. recompute graph-derived features, retrain, and remeasure every augmented
   graph.

There is no pair-score threshold, maximum community-pair degree, per-pair
allocation, or legacy strength multiplier in the released method. Intervention
strength is the number/rate of positive physical edges added.

## Installation

Python 3.10 is recommended. Install the CUDA-compatible PyTorch build for your
machine if needed, then run:

```bash
pip install -e '.[test]'
pytest -q
```

## Data

BTC-Alpha is included for a lightweight reproduction. Download the other SNAP
signed-network files as described in `data/README.md`.

```bash
bash experiments/preprocess.sh bitcoinalpha
```

## Base model and measurement

```bash
bash experiments/train_base.sh sgcn bitcoinalpha cuda

bash experiments/measure.sh \
  artifacts/path/to/node_embeddings.pt bitcoinalpha 14 \
  artifacts/measurement/seed_0
```

Model hyperparameters are selected once per dataset/backbone using mean
validation Macro-F1 across five seeds and then held fixed across interventions
and baselines. Test labels are reporting-only.

## Mitigation

```bash
bash experiments/prepare_mitigation.sh \
  artifacts/path/to/node_embeddings.pt bitcoinalpha 14 \
  artifacts/mitigation/preparation/seed_0 30

bash experiments/select_mitigation.sh \
  bitcoinalpha artifacts/mitigation/preparation/seed_0 \
  artifacts/measurement/seed_0/opinion_coordinates.npy \
  988 0 artifacts/mitigation/selection/seed_0
```

For multiple seeds, place each `selection` output at
`<selection-parent>/seed_N/selected_edges.csv`, then materialize prefixes:

```bash
bash experiments/materialize_mitigation.sh \
  artifacts/mitigation/selection bitcoinalpha \
  artifacts/mitigation/graphs .025 .05 .075 .10
```

For SDGNN, preserve directed model edges during materialization:

```bash
DIRECTED_BACKBONE=1 bash experiments/materialize_mitigation.sh \
  artifacts/mitigation/selection bitcoinalpha \
  artifacts/mitigation/graphs .025 .05 .075 .10
```

Pass each generated `train_snapshot_directed_augmented.csv` to
`signed-epm-train` with the base-selected model hyperparameters. SGCN
recomputes TSVD from that exact augmented graph; SDGNN receives the directed
augmented view. The encoder and probe are freshly trained for every graph.

## Synthetic validation

Deterministic signed-SBM generators and validators are included under
`signed_epm.synthetic`. Synthetic graphs, seeds, and generated metadata can be
saved verbatim for figure reproduction. The released measurement is

```text
sqrt(structural_energy + alpha * antagonistic_energy)
```

with `Y = L^- Z`, negative conductance `eta=0.1`, and antagonistic weight
`alpha=0.05` in the paper configuration.

## Layout

```text
configs/       model, task, dataset, and paper settings
data/          public-data instructions and BTC-Alpha raw input
experiments/   concise stage commands
src/           audited implementation
tests/         unit/regression tests
```

Generated data, checkpoints, logs, figures, and full result directories are
ignored. No server address, username, absolute path, or credential is required.
