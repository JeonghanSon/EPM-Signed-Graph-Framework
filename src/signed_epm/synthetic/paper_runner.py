from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from signed_epm.data.preprocess import ROOT
from signed_epm.synthetic.generate_signed_structural_sbm import HOHMANN_P_IN_LEVELS
from signed_epm.synthetic.generate_unsigned_sbm import P_OUT_LEVELS, SBMConfig


def run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def values(items: list) -> list[str]:
    return [str(item) for item in items]


def validate_config(config: dict) -> None:
    expected = SBMConfig()
    checks = {
        "num_nodes": (config["num_nodes"], expected.num_nodes),
        "num_communities": (config["num_communities"], expected.num_communities),
        "baseline_probability": (
            config["structural"]["baseline_probability"], expected.baseline_probability,
        ),
        "positive_p_in": (
            tuple(config["structural"]["positive_p_in"]), HOHMANN_P_IN_LEVELS,
        ),
        "positive_p_out": (
            tuple(config["structural"]["positive_p_out"]), P_OUT_LEVELS,
        ),
        "positive_source_level": (
            config["antagonistic"]["positive_source_level"], 5,
        ),
        "positive_conductance": (
            config["measurement"]["positive_conductance"], 1.0,
        ),
        "pca_dimension": (
            config["measurement"]["pca_dimension"], config["num_communities"],
        ),
    }
    mismatches = {
        name: {"configured": configured, "supported": supported}
        for name, (configured, supported) in checks.items()
        if configured != supported
    }
    if mismatches:
        raise ValueError(f"synthetic config is incompatible with the paper generator: {mismatches}")


def sgcn_command(data_root: Path, output_root: Path, config: dict, device: str) -> list[str]:
    model = config["sgcn"]
    return [
        sys.executable, "-m", "signed_epm.synthetic.run_sgcn",
        "--data-root", str(data_root), "--output-root", str(output_root),
        "--graph-seeds", *values(config["graph_seeds"]),
        "--input-dimension", str(model["input_dimension"]),
        "--output-dimension", str(model["output_dimension"]),
        "--layers", str(model["layers"]),
        "--learning-rate", str(model["learning_rate"]),
        "--epochs", str(model["epochs"]),
        "--negative-conductance", str(config["measurement"]["negative_conductance"]),
        "--device", device,
    ]


def controls(data_root: Path, result_root: Path) -> None:
    for opinion_set, suffix in (("primary", "aligned"), ("additional_random", "random")):
        run([
            sys.executable, "-m", "signed_epm.synthetic.validate",
            "--data-root", str(data_root),
            "--output-dir", str(result_root / f"legacy_{suffix}"),
            "--opinion-set", opinion_set,
        ])


def generated(root: Path, config: dict, device: str) -> None:
    structural = root / "structural"
    antagonistic = root / "antagonistic"
    run([
        sys.executable, "-m", "signed_epm.synthetic.generate_signed_structural_sbm",
        "--output-root", str(structural),
        "--graph-seeds", *values(config["graph_seeds"]),
        "--opinion-seeds", *values(config["opinion_seeds"]),
        "--negative-share", str(config["structural"]["negative_edge_share"]),
        "--negative-inter-levels", *values(config["structural"]["negative_inter_fraction"]),
        "--overwrite",
    ])
    run(sgcn_command(structural, root / "results" / "structural_sgcn", config, device))
    controls(structural, root / "results" / "structural")
    run([
        sys.executable, "-m", "signed_epm.synthetic.generate_signed_sbm",
        "--positive-root", str(structural), "--output-root", str(antagonistic),
        "--graph-seeds", *values(config["graph_seeds"]),
        "--negative-share", str(config["antagonistic"]["negative_edge_share"]),
        "--levels", *values(config["antagonistic"]["negative_inter_fraction"]),
        "--overwrite",
    ])
    run(sgcn_command(antagonistic, root / "results" / "antagonistic_sgcn", config, device))
    controls(antagonistic, root / "results" / "antagonistic")


def extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as handle:
        destination = destination.resolve()
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"unsafe archive member: {member.name}")
        handle.extractall(destination)


def bundled(root: Path, config: dict, device: str) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    extract(ROOT / "data/synthetic/structural_hm_sbm.tar.gz", root)
    extract(ROOT / "data/synthetic/antagonistic_alignment.tar.gz", root)
    structural = root / "eq1_signed_structural20_n1000_k8"
    antagonistic = root / "eq1_antagonistic20_n1000_k8"
    run(sgcn_command(structural, root / "results" / "structural_sgcn", config, device))
    controls(structural, root / "results" / "structural")
    run(sgcn_command(antagonistic, root / "results" / "antagonistic_sgcn", config, device))
    controls(antagonistic, root / "results" / "antagonistic")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the paper synthetic experiments")
    parser.add_argument("mode", choices=["generated", "bundled"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/synthetic.json")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    output = args.output_root or ROOT / "data/synthetic/generated" / args.mode
    (generated if args.mode == "generated" else bundled)(output, config, args.device)


if __name__ == "__main__":
    main()
