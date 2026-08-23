#!/usr/bin/env bash
set -euo pipefail

device="${1:-cuda}"
root="data/synthetic/generated/bundled"
structural="$root/eq1_signed_structural20_n1000_k8"
antagonistic="$root/eq1_antagonistic20_n1000_k8"

rm -rf "$root"
mkdir -p "$root"
tar -xzf data/synthetic/structural_hm_sbm.tar.gz -C "$root"
tar -xzf data/synthetic/antagonistic_alignment.tar.gz -C "$root"

python -m signed_epm.synthetic.run_sgcn \
  --data-root "$structural" \
  --output-root "$root/results/structural_sgcn" \
  --graph-seeds 0 1 2 3 4 \
  --device "$device"

python -m signed_epm.synthetic.validate \
  --data-root "$structural" \
  --output-dir "$root/results/structural_legacy_aligned" \
  --opinion-set primary

python -m signed_epm.synthetic.validate \
  --data-root "$structural" \
  --output-dir "$root/results/structural_legacy_random" \
  --opinion-set additional_random

python -m signed_epm.synthetic.run_sgcn \
  --data-root "$antagonistic" \
  --output-root "$root/results/antagonistic_sgcn" \
  --graph-seeds 0 1 2 3 4 \
  --device "$device"

python -m signed_epm.synthetic.validate \
  --data-root "$antagonistic" \
  --output-dir "$root/results/antagonistic_legacy_aligned" \
  --opinion-set primary

python -m signed_epm.synthetic.validate \
  --data-root "$antagonistic" \
  --output-dir "$root/results/antagonistic_legacy_random" \
  --opinion-set additional_random
