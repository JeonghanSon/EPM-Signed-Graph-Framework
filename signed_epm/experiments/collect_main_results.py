from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


DATASETS = {
    "bitcoinalpha": "BTC-Alpha",
    "bitcoinotc": "BTC-OTC",
    "wiki_elec": "Wiki-Elec",
    "wiki_rfa": "Wiki-RfA",
    "slashdot": "Slashdot",
    "epinions": "Epinions",
}

SGCN_SETTINGS = {
    "bitcoinalpha": "tau0.5_dmax3_gamma2.0",
    "bitcoinotc": "tau0.5_dmax3_gamma3.0",
    "wiki_elec": "tau0.5_dmax3_gamma2.0",
    "wiki_rfa": "tau0.5_dmax3_gamma2.5",
    "slashdot": "tau0.2_dmax4_gamma1.5",
    "epinions": "tau0.2_dmax3_gamma1.5",
}

# Model-agnostic comparison: SDGNN uses exactly the intervention operating
# point selected with SGCN for each dataset; only model training is retuned.
SDGNN_EXISTING_SETTINGS = dict(SGCN_SETTINGS)


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def accuracy(path: Path) -> tuple[float | None, float | None]:
    if not path.exists():
        return None, None
    frame = pd.read_csv(path)
    if "test_accuracy" not in frame:
        return None, None
    values = frame["test_accuracy"].dropna()
    if values.empty:
        return None, None
    return float(values.mean()), float(values.std(ddof=1))


def row(backbone: str, dataset: str, setting: str | None, status: str) -> dict:
    root = Path("artifacts/runs") / backbone / dataset / "signlink_3class"
    base = load(root / "base" / "final_summary.json")
    epm_root = root / "epm" / setting if setting else None
    epm = load(epm_root / "final_summary.json") if epm_root else None
    base_accuracy, base_accuracy_std = accuracy(root / "base" / "final_evaluation.csv")
    epm_accuracy, epm_accuracy_std = (
        accuracy(epm_root / "final_evaluation.csv") if epm_root else (None, None)
    )
    result = {
        "dataset": DATASETS[dataset],
        "backbone": backbone.upper(),
        "setting": setting,
        "status": status,
        "base_test_macro_f1_mean": None,
        "base_test_macro_f1_std": None,
        "epm_test_macro_f1_mean": None,
        "epm_test_macro_f1_std": None,
        "relative_macro_f1_change_pct": None,
        "base_test_accuracy_mean": base_accuracy,
        "base_test_accuracy_std": base_accuracy_std,
        "epm_test_accuracy_mean": epm_accuracy,
        "epm_test_accuracy_std": epm_accuracy_std,
        "relative_accuracy_change_pct": None,
        "base_polarization_mean": None,
        "base_polarization_std": None,
        "epm_polarization_mean": None,
        "epm_polarization_std": None,
        "polarization_reduction_pct": None,
    }
    if base:
        result.update({
            "base_test_macro_f1_mean": base["test_macro_f1_mean"],
            "base_test_macro_f1_std": base["test_macro_f1_std"],
            "base_polarization_mean": base["polarization_mean"],
            "base_polarization_std": base["polarization_std"],
        })
    if epm:
        result.update({
            "epm_test_macro_f1_mean": epm["test_macro_f1_mean"],
            "epm_test_macro_f1_std": epm["test_macro_f1_std"],
            "epm_polarization_mean": epm["polarization_mean"],
            "epm_polarization_std": epm["polarization_std"],
        })
    if base and epm:
        result.update({
            "relative_macro_f1_change_pct": 100.0 * (
                epm["test_macro_f1_mean"] / base["test_macro_f1_mean"] - 1.0
            ),
            "polarization_reduction_pct": 100.0 * (
                1.0 - epm["polarization_mean"] / base["polarization_mean"]
            ),
        })
        if base_accuracy is not None and epm_accuracy is not None:
            result["relative_accuracy_change_pct"] = 100.0 * (
                epm_accuracy / base_accuracy - 1.0
            )
    return result


def main() -> None:
    rows = [
        row("sgcn", dataset, setting, "frozen")
        for dataset, setting in SGCN_SETTINGS.items()
    ]
    rows.extend(
        row("sdgnn", dataset, SDGNN_EXISTING_SETTINGS.get(dataset),
            "provisional_grid_in_progress" if dataset in SDGNN_EXISTING_SETTINGS else "pending")
        for dataset in DATASETS
    )
    output = Path("artifacts/paper/main/main_mitigation_results.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep manuscript-facing means and their five-seed sample standard
    # deviations together so the paper table can report mean +/- std.
    main_columns = [
        "dataset",
        "backbone",
        "base_test_macro_f1_mean",
        "base_test_macro_f1_std",
        "epm_test_macro_f1_mean",
        "epm_test_macro_f1_std",
        "base_test_accuracy_mean",
        "base_test_accuracy_std",
        "epm_test_accuracy_mean",
        "epm_test_accuracy_std",
        "base_polarization_mean",
        "base_polarization_std",
        "epm_polarization_mean",
        "epm_polarization_std",
        "polarization_reduction_pct",
    ]
    pd.DataFrame(rows)[main_columns].to_csv(output, index=False)
    print(f"saved={output}")


if __name__ == "__main__":
    main()
