#!/usr/bin/env bash
set -euo pipefail
dataset="${1:-bitcoinalpha}"
signed-epm-preprocess --dataset "$dataset" --seed 0
signed-epm-communities --dataset "$dataset"
