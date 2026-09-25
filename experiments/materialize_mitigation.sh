#!/usr/bin/env bash
set -euo pipefail
selection_parent="${1:?directory containing seed_N/selected_edges.csv required}"
dataset="${2:-bitcoinalpha}"
output="${3:-artifacts/mitigation/graphs}"
shift_count=3
if (( $# > shift_count )); then shift "$shift_count"; rates=("$@"); else rates=(.025 .05 .075 .10); fi
directed_args=()
if [[ "${DIRECTED_BACKBONE:-0}" == "1" ]]; then
  directed_args+=(--directed-backbone)
fi
signed-epm-materialize --selection-root "$selection_parent" \
  --data-dir "data/processed/$dataset" --output-root "$output" \
  --selection-kind pooled --methods global --rates "${rates[@]}" \
  "${directed_args[@]}"
