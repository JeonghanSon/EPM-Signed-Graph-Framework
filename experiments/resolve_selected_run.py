#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve a validation-selected encoder run")
    parser.add_argument("model", nargs="?", default="sgcn")
    parser.add_argument("dataset", nargs="?", default="bitcoinalpha")
    parser.add_argument("seed", nargs="?", type=int, default=0)
    args = parser.parse_args()

    final_node_state = (Path("artifacts/base") / args.model / args.dataset
                        / f"seed_{args.seed}" / "node_embeddings.pt")
    if final_node_state.exists():
        print(final_node_state)
        return

    root = (Path("artifacts/runs") / args.model / args.dataset
            / "signlink_3class" / "base")
    selected_runs = root / "selected_runs.csv"
    if not selected_runs.exists():
        raise SystemExit(
            f"selected runs not found: {selected_runs}; run train_base.sh first"
        )
    with selected_runs.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle)
                if int(row["seed"]) == args.seed]
    if len(rows) != 1:
        raise SystemExit(
            f"expected one selected run for seed {args.seed}, found {len(rows)}"
        )
    node_state = Path(rows[0]["run_dir"]) / "node_embeddings.pt"
    if not node_state.exists():
        raise SystemExit(f"selected node state not found: {node_state}")
    print(node_state)


if __name__ == "__main__":
    main()
