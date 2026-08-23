#!/usr/bin/env bash
set -euo pipefail

device="${1:-cuda}"
python -m signed_epm.synthetic.paper_runner generated --device "$device"
