#!/usr/bin/env bash
set -euo pipefail

dataset="${1:-bitcoinalpha}"
signed-epm-preprocess --dataset "$dataset"
signed-epm-communities --dataset "$dataset" --fallback
