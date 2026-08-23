#!/usr/bin/env bash
set -euo pipefail

device="${1:-cuda}"
root="data/synthetic/generated"

python -m signed_epm.synthetic.generate_signed_structural_sbm \
  --output-root "$root/structural" \
  --graph-seeds 0 1 2 3 4 \
  --opinion-seeds 0 1 2 3 4 \
  --negative-share 0.2 \
  --negative-inter-levels 0.6 0.7 0.8 0.9 1.0 \
  --overwrite

python -m signed_epm.synthetic.run_sgcn \
  --data-root "$root/structural" \
  --output-root "$root/structural_sgcn" \
  --graph-seeds 0 1 2 3 4 \
  --device "$device"

python -m signed_epm.synthetic.generate_signed_sbm \
  --positive-root "$root/structural" \
  --output-root "$root/antagonistic" \
  --graph-seeds 0 1 2 3 4 \
  --negative-share 0.2 \
  --levels 0.6 0.7 0.8 0.9 1.0 \
  --overwrite

python -m signed_epm.synthetic.run_sgcn \
  --data-root "$root/antagonistic" \
  --output-root "$root/antagonistic_sgcn" \
  --graph-seeds 0 1 2 3 4 \
  --device "$device"
