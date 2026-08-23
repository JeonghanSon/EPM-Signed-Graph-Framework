from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import networkx as nx
import numpy as np
import pandas as pd


def relative(frame: pd.DataFrame, value: str, keys: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    first = frame.level.min()
    baseline = frame[frame.level == first].set_index(keys)[value]
    frame["key"] = list(map(tuple, frame[keys].to_numpy()))
    if len(keys) == 1:
        baseline.index = [(item,) for item in baseline.index]
    frame["relative"] = frame[value] / frame.key.map(baseline)
    return frame.groupby("level", as_index=False).relative.mean()


def normalized_positions(positions: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    nodes = sorted(positions)
    values = np.asarray([positions[node] for node in nodes], dtype=float)
    values -= values.mean(axis=0, keepdims=True)
    values /= max(float(np.abs(values).max()), 1e-12)
    return dict(zip(nodes, values))


def structural_panel(figure: plt.Figure, slot, data_root: Path,
                     result_root: Path, graph_seed: int, opinion_seed: int,
                     iterations: int) -> None:
    grid = slot.subgridspec(
        2, 6, width_ratios=(1.0, 1, 1, 1, 1, 1),
        height_ratios=(2.25, 2.05), hspace=0.01, wspace=0.14,
    )
    # Keep the two score rows close, then separate the graph-generation
    # condition with a dedicated vertical spacer.
    measure_grid = grid[1, 1:].subgridspec(
        4, 5, height_ratios=(1.0, 1.0, 0.50, 1.0),
        hspace=-0.18, wspace=0.14,
    )
    epm = pd.read_csv(result_root / "epm" / "epm_raw.csv").groupby(
        "level",
    ).epm_polarization.agg(["mean", "std"])
    random_er = pd.read_csv(
        result_root / "legacy_random" / "legacy_er_raw.csv",
    ).groupby("level").legacy_er.agg(["mean", "std"])
    conditions: dict[int, tuple[float, float]] = {}
    for column, level in enumerate(range(1, 6), start=1):
        axis = figure.add_subplot(grid[0, column])
        graph_root = data_root / "structural" / f"graph_seed_{graph_seed}" / f"level_{level}"
        edges = pd.read_csv(graph_root / "train_snapshot_undirected.csv")
        positive = edges[edges.sign > 0]
        negative = edges[edges.sign < 0]
        nodes = pd.read_csv(graph_root / "nodes.csv")
        metadata = json.loads((graph_root / "manifest.json").read_text(encoding="utf-8"))
        synthetic = metadata["synthetic"]
        community_sizes = nodes.groupby("community").size().to_numpy(dtype=np.int64)
        intra_pairs = float(np.sum(community_sizes * (community_sizes - 1) // 2))
        total_pairs = float(len(nodes) * (len(nodes) - 1) // 2)
        inter_pairs = total_pairs - intra_pairs
        expected_positive_intra = intra_pairs * float(synthetic["p_positive_in"])
        expected_positive_inter = inter_pairs * float(synthetic["p_positive_out"])
        conditions[level] = (
            expected_positive_intra / (expected_positive_intra + expected_positive_inter),
            float(synthetic["expected_inter_negative_fraction"]),
        )
        graph = nx.Graph()
        graph.add_nodes_from(nodes.node_id.astype(int))
        graph.add_edges_from(positive[["source", "target"]].itertuples(index=False, name=None))
        positions = normalized_positions(nx.spring_layout(
            graph, k=0.045, iterations=iterations,
            seed=40_000 + graph_seed * 10 + level,
        ))
        positions_array = np.asarray([positions[int(node)] for node in nodes.node_id])
        positive_segments = np.asarray([
            [positions[int(s)], positions[int(t)]]
            for s, t in positive[["source", "target"]].itertuples(index=False)
        ])
        negative_segments = np.asarray([
            [positions[int(s)], positions[int(t)]]
            for s, t in negative[["source", "target"]].itertuples(index=False)
        ])
        axis.add_collection(LineCollection(
            positive_segments, colors="#2f6690", linewidths=0.22,
            alpha=0.52, rasterized=True, zorder=1,
        ))
        axis.add_collection(LineCollection(
            negative_segments, colors="#d1495b", linewidths=0.26,
            alpha=0.40, rasterized=True, zorder=2,
        ))
        axis.scatter(positions_array[:, 0], positions_array[:, 1],
                     color="#999999", s=2.5, alpha=0.32,
                     edgecolors="none", rasterized=True, zorder=3)
        axis.set_title("ABCDE"[level - 1], fontsize=14.6, fontweight="bold", pad=0.8)
        axis.set_xlim(-1.03, 1.03); axis.set_ylim(-1.03, 1.03)
        axis.set_aspect("equal"); axis.set_anchor("S"); axis.axis("off")
        position = axis.get_position()
        axis.set_position([
            position.x0, position.y0 - 0.018, position.width, position.height,
        ])

    rows = (
        (0, r"$P_{G,\hat{Z}}$", epm, "score"),
        (1, r"$\delta_{G,o}$", random_er, "score"),
        (3, r"$r^+_{\mathrm{intra}} / r^-_{\mathrm{inter}}$", None, "condition"),
    )
    for row, label, values, kind in rows:
        for column, level in enumerate(range(1, 6), start=1):
            value_axis = figure.add_subplot(measure_grid[row, column - 1])
            if column == 1:
                label_x = -0.43 if kind == "condition" else -0.58
                value_axis.text(
                    label_x, 0.5, label, transform=value_axis.transAxes,
                    ha="right", va="center", fontsize=14.7,
                )
            if kind == "score":
                text = rf"${values.loc[level, 'mean']:.2f}\,\pm\,{values.loc[level, 'std']:.2f}$"
            else:
                text = rf"${conditions[level][0]:.2f}\,/\,{conditions[level][1]:.2f}$"
            value_axis.text(0.5, 0.5, text, ha="center", va="center", fontsize=13.5)
            value_axis.axis("off")


def antagonistic_panel(axis: plt.Axes, result_root: Path) -> None:
    epm_raw = pd.read_csv(result_root / "epm" / "epm_raw.csv")
    er_raw = pd.read_csv(result_root / "legacy_random" / "legacy_er_raw.csv")
    # Match panel (a): use r_inter^- = 0.6 as the first condition and
    # normalize the displayed response to that condition.
    epm_raw = epm_raw[epm_raw.level >= 0.6]
    er_raw = er_raw[er_raw.level >= 0.6]
    epm = relative(epm_raw,
                   "epm_polarization", ["graph_seed"])
    er = relative(er_raw,
                  "legacy_er", ["graph_seed", "opinion_seed"])
    axis.plot(epm.level, epm.relative, "s-", color="#c44536", lw=2.4,
              markersize=5.8, label=r"$P_{G,\hat{Z}}$")
    axis.plot(er.level, er.relative, "o--", color="#2f6690", lw=2.0,
              markersize=5.2, label=r"$\delta_{G,o}$")
    axis.axhline(1.0, color="black", lw=0.7, alpha=0.25)
    axis.set_xlabel(r"$r^-_{\mathrm{inter}}$", labelpad=5)
    axis.set_ylabel("Relative score", labelpad=2)
    axis.set_xticks(epm.level)
    axis.set_ylim(0.96, max(1.72, float(epm.relative.max() + 0.06)))
    axis.set_box_aspect(0.52)
    axis.grid(alpha=0.16)
    axis.legend(frameon=False, fontsize=12.1, loc="upper left",
                handlelength=1.5, borderaxespad=0.25)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the two-panel signed synthetic validation")
    parser.add_argument("--structural-data", type=Path, required=True)
    parser.add_argument("--structural-results", type=Path, required=True)
    parser.add_argument("--antagonistic-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph-seed", type=int, default=0)
    parser.add_argument("--opinion-seed", type=int, default=0)
    parser.add_argument("--layout-iterations", type=int, default=80)
    args = parser.parse_args()

    with plt.rc_context({"font.size": 14.8, "axes.labelsize": 16.0,
                         "xtick.labelsize": 13.2, "ytick.labelsize": 13.2}):
        figure = plt.figure(figsize=(12.2, 3.58))
        outer = figure.add_gridspec(1, 2, width_ratios=(2.50, 1.0), wspace=0.27)
        structural_panel(figure, outer[0], args.structural_data,
                         args.structural_results, args.graph_seed,
                         args.opinion_seed, args.layout_iterations)
        antagonistic_panel(figure.add_subplot(outer[1]), args.antagonistic_results)
        figure.text(0.340, 0.100, "(a) Increasing Signed Structural Polarization",
                    ha="center", va="bottom", fontsize=15.2)
        figure.text(0.845, 0.100, "(b) Effect of Negative Edges",
                    ha="center", va="bottom", fontsize=15.2)
        figure.subplots_adjust(left=0.006, right=0.997, top=0.992, bottom=0.17)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=300, bbox_inches="tight", pad_inches=0.035)
        figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.035)
        plt.close(figure)


if __name__ == "__main__":
    main()
