# Signed EPM

This repository implements embedding-aware polarization measurement and
mitigation (EPM) for signed graphs. SGCN and SDGNN are used through model
adapters; EPM consumes their learned node states without changing either
backbone's objective.

The pipeline uses a connected, static, undirected training snapshot for the
structural operator. Positive conductance is 1, negative conductance is a small
positive value (0.1 in the paper), and non-edges have conductance 0. Directed
model training still uses the corresponding directed view.

## Installation

Python 3.10 and a PyTorch build compatible with your CUDA device are
recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For an existing CUDA environment, install its matching PyTorch build first and
then run `pip install -e .`.

## Lightweight reproduction

BTC-Alpha raw data is included. Run preprocessing and signed Louvain first:

```bash
signed-epm-preprocess --dataset bitcoinalpha
signed-epm-communities --dataset bitcoinalpha
```

Tune the SGCN baseline. Model selection uses only the mean validation Macro-F1
across five seeds; test labels do not participate in selection.

```bash
bash experiments/train_base.sh sgcn bitcoinalpha cuda
```

Measurement and mitigation are deliberately separate. Preparation stores the
base measurement, K-means memberships, polarized community-pair scores, and
gray-node rankings so multiple intervention settings can reuse them.

```bash
bash experiments/measure.sh sgcn bitcoinalpha
bash experiments/evaluate_base.sh sgcn bitcoinalpha
bash experiments/prepare_mitigation.sh sgcn bitcoinalpha
bash experiments/generate_mitigation.sh sgcn bitcoinalpha 0.5 3 2.0
bash experiments/train_mitigated.sh sgcn bitcoinalpha 0.5 3 2.0 cuda
```

Paper grids and the selected representative intervention settings are recorded
under `configs/paper/`. Every script accepts explicit arguments, so alternative
settings do not require source changes.

## Synthetic validation

The public package contains the generators for the signed structural-separation
and antagonistic-alignment experiments. Run:

```bash
bash experiments/run_synthetic.sh cuda
```

The command regenerates the synthetic networks from fixed seeds and trains a
fresh SGCN representation for every graph condition. Generated artifacts are
written outside the tracked source tree.

To reuse the exact graph files bundled with this repository instead of
regenerating them, run:

```bash
bash experiments/run_synthetic_bundled.sh cuda
```

## Repository layout

```text
configs/          model, dataset, task, and paper experiment settings
data/             BTC-Alpha raw data, metadata, and synthetic bundles
experiments/      stage-by-stage reproduction commands
paper_results/    compact main mitigation result table
src/signed_epm/   preprocessing, adapters, measurement, and mitigation
tests/            unit tests for public pipeline components
```

Training checkpoints, caches, logs, exploratory analyses, and manuscript
figures are intentionally excluded.
