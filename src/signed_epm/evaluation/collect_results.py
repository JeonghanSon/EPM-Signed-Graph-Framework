from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_metric(path: Path, split: str, name: str) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    block = payload.get(split)
    if not isinstance(block, dict) or name not in block:
        raise KeyError(f"{path}: missing {split}.{name}")
    return float(block[name])


def read_polarization(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["polarization"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate base/mitigated utility and polarization from an explicit manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    required = {
        "dataset", "seed", "base_metrics", "mitigated_metrics",
        "base_measurement", "mitigated_measurement",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")

    rows = []
    for item in manifest.itertuples(index=False):
        base_f1 = read_metric(Path(item.base_metrics), args.split, "macro_f1")
        mitigated_f1 = read_metric(Path(item.mitigated_metrics), args.split, "macro_f1")
        base_p = read_polarization(Path(item.base_measurement))
        mitigated_p = read_polarization(Path(item.mitigated_measurement))
        rows.append({
            "dataset": item.dataset,
            "seed": int(item.seed),
            "base_macro_f1": base_f1,
            "mitigated_macro_f1": mitigated_f1,
            "macro_f1_change_pct": 100.0 * (mitigated_f1 - base_f1) / base_f1,
            "base_polarization": base_p,
            "mitigated_polarization": mitigated_p,
            "polarization_reduction_pct": 100.0 * (base_p - mitigated_p) / base_p,
        })
    raw = pd.DataFrame(rows)
    summary = raw.groupby("dataset", sort=False).agg(
        seeds=("seed", "count"),
        base_macro_f1_mean=("base_macro_f1", "mean"),
        mitigated_macro_f1_mean=("mitigated_macro_f1", "mean"),
        macro_f1_change_pct_mean=("macro_f1_change_pct", "mean"),
        macro_f1_change_pct_std=("macro_f1_change_pct", "std"),
        polarization_reduction_pct_mean=("polarization_reduction_pct", "mean"),
        polarization_reduction_pct_std=("polarization_reduction_pct", "std"),
    ).reset_index()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    raw.to_csv(args.output.with_name(f"{args.output.stem}_per_seed.csv"), index=False)


if __name__ == "__main__":
    main()
