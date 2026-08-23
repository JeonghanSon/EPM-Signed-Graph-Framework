from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "datasets.json"
DEFAULT_CHECKSUMS = ROOT / "data" / "metadata" / "raw_files.json"
ALIASES = {
    "bitcoinalpha": "bitcoinalpha",
    "bitcoin-alpha": "bitcoinalpha",
    "bitcoinotc": "bitcoinotc",
    "bitcoin-otc": "bitcoinotc",
    "wiki-elec": "wiki_elec",
    "wiki_elec": "wiki_elec",
    "wikielec": "wiki_elec",
    "wiki-rfa": "wiki_rfa",
    "wiki_rfa": "wiki_rfa",
    "wikirfa": "wiki_rfa",
    "slashdot": "slashdot",
    "epinions": "epinions",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_bitcoin(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, header=None, names=["source", "target", "weight", "timestamp"])
    frame = frame.dropna(subset=["source", "target", "weight", "timestamp"])
    frame = frame[frame["weight"] != 0]
    # Preserve the historical EPM ordering, including pandas' default handling
    # of tied timestamps. This is required for exact keep-last split parity.
    return frame.sort_values("timestamp").reset_index(drop=True)


def parse_wiki_elec(path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    current_target = None
    with path.open("r", encoding="latin-1") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("U\t"):
                fields = line.split("\t")
                current_target = fields[1] if len(fields) >= 2 else None
            elif line.startswith("V\t") and current_target is not None:
                fields = line.split("\t")
                if len(fields) < 4:
                    continue
                try:
                    weight = int(fields[1])
                except ValueError:
                    continue
                if weight != 0 and fields[2] and fields[3]:
                    rows.append({"source": fields[2], "target": current_target,
                                 "weight": weight, "timestamp": fields[3]})
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    return frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def parse_wiki_rfa(path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    current: dict = {}
    last_timestamp: int | None = None

    def parse_date(value: str) -> int | None:
        try:
            return int(datetime.strptime(value.strip(), "%H:%M, %d %B %Y").timestamp())
        except (TypeError, ValueError):
            return None

    def flush() -> None:
        nonlocal current, last_timestamp
        if not current or not all(key in current for key in ("source", "target", "weight")):
            current = {}
            return
        if int(current["weight"]) == 0:
            current = {}
            return
        candidate = parse_date(str(current.get("date", "")))
        if candidate is None:
            candidate = (int(datetime(int(current["year"]), 1, 1).timestamp())
                         if last_timestamp is None and current.get("year") is not None
                         else 0 if last_timestamp is None else last_timestamp + 1)
        timestamp = candidate if last_timestamp is None or candidate > last_timestamp else last_timestamp + 1
        last_timestamp = int(timestamp)
        rows.append({"source": current["source"], "target": current["target"],
                     "weight": int(current["weight"]), "timestamp": int(timestamp)})
        current = {}

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("SRC:"):
                current["source"] = line[4:].strip()
            elif line.startswith("TGT:"):
                current["target"] = line[4:].strip()
            elif line.startswith("VOT:"):
                try:
                    current["weight"] = int(line[4:].strip())
                except ValueError:
                    current["weight"] = 0
            elif line.startswith("YEA:"):
                try:
                    current["year"] = int(line[4:].strip())
                except ValueError:
                    current["year"] = None
            elif line.startswith("DAT:"):
                current["date"] = line[4:].strip()
            elif not line:
                flush()
    flush()
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def parse_snap_static(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, comment="#", header=None, sep=r"\s+",
                        names=["source", "target", "weight"], engine="python")
    frame = frame.dropna(subset=["source", "target", "weight"])
    for column in ("source", "target", "weight"):
        frame[column] = frame[column].astype(int)
    frame = frame[(frame["weight"] != 0) & (frame["source"] != frame["target"])].copy()
    source, target = frame["source"].to_numpy(), frame["target"].to_numpy()
    frame["source"], frame["target"] = np.minimum(source, target), np.maximum(source, target)
    return frame.reset_index(drop=True)


def parse_raw(path: Path, format_name: str) -> pd.DataFrame:
    parsers = {"bitcoin": parse_bitcoin, "wiki_elec": parse_wiki_elec,
               "wiki_rfa": parse_wiki_rfa, "snap_static": parse_snap_static}
    return parsers[format_name](path)


def remap_first_seen(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, str]]:
    mapping: dict[str, int] = {}
    reverse: dict[int, str] = {}
    source_ids, target_ids = [], []
    for source, target in frame[["source", "target"]].itertuples(index=False):
        for node in (str(source), str(target)):
            if node not in mapping:
                index = len(mapping)
                mapping[node] = index
                reverse[index] = node
        source_ids.append(mapping[str(source)])
        target_ids.append(mapping[str(target)])
    result = frame.copy()
    result["source"], result["target"] = source_ids, target_ids
    return result, reverse


def build_events(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, str]]:
    frame = raw.dropna(subset=["source", "target", "weight"]).copy()
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")
    frame = frame.dropna(subset=["weight"])
    frame["weight"] = frame["weight"].astype(int)
    frame["source"], frame["target"] = frame["source"].astype(str), frame["target"].astype(str)
    # Wiki-RfA contains records with an empty voter name. The historical
    # two-stage CSV pipeline discarded these when pandas decoded them as NaN;
    # reject them explicitly in the in-memory implementation.
    frame = frame[(frame["source"].str.strip() != "") &
                  (frame["target"].str.strip() != "")].copy()
    frame = frame[(frame["weight"] != 0) & (frame["source"] != frame["target"])].copy()
    temporal = "timestamp" in frame.columns
    if temporal:
        frame = frame.sort_values("timestamp").reset_index(drop=True)
    else:
        source, target = frame["source"].astype(int).to_numpy(), frame["target"].astype(int).to_numpy()
        frame["source"], frame["target"] = np.minimum(source, target), np.maximum(source, target)
        frame["_order"] = np.arange(len(frame), dtype=np.int64)
        frame = frame.drop_duplicates(["source", "target"], keep="last")
        frame = frame.sort_values("_order", kind="mergesort").drop(columns="_order").reset_index(drop=True)
        frame["source"], frame["target"] = frame["source"].astype(str), frame["target"].astype(str)
    frame["event_id"] = np.arange(len(frame), dtype=np.int64)
    return remap_first_seen(frame)


def canonical_keep_last(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    source, target = frame["source"].to_numpy(), frame["target"].to_numpy()
    frame["source"], frame["target"] = np.minimum(source, target), np.maximum(source, target)
    frame["sign"] = np.sign(frame["weight"]).astype(np.int8)
    frame = frame.drop_duplicates(["source", "target"], keep="last")
    return frame.reset_index(drop=True)


def largest_component_nodes(frame: pd.DataFrame) -> np.ndarray:
    nodes = np.unique(frame[["source", "target"]].to_numpy())
    position = {int(node): index for index, node in enumerate(nodes)}
    parent, size = np.arange(len(nodes)), np.ones(len(nodes), dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for source, target in frame[["source", "target"]].itertuples(index=False):
        left, right = find(position[int(source)]), find(position[int(target)])
        if left != right:
            if size[left] < size[right]:
                left, right = right, left
            parent[right] = left
            size[left] += size[right]
    roots = np.asarray([find(i) for i in range(len(nodes))])
    labels, counts = np.unique(roots, return_counts=True)
    return np.sort(nodes[roots == labels[np.argmax(counts)]])


def remap_lcc(frame: pd.DataFrame, nodes: np.ndarray) -> tuple[pd.DataFrame, dict[int, int]]:
    mapping = {int(node): index for index, node in enumerate(np.sort(nodes))}
    result = frame[frame["source"].isin(mapping) & frame["target"].isin(mapping)].copy()
    result["source"] = result["source"].map(mapping).astype(int)
    result["target"] = result["target"].map(mapping).astype(int)
    return result.reset_index(drop=True), mapping


def spanning_tree_indices(frame: pd.DataFrame, num_nodes: int) -> np.ndarray:
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(num_nodes)]
    for edge_id, (source, target) in enumerate(frame[["source", "target"]].itertuples(index=False)):
        adjacency[int(source)].append((int(target), edge_id))
        adjacency[int(target)].append((int(source), edge_id))
    visited, queue, selected = np.zeros(num_nodes, dtype=bool), deque([0]), []
    visited[0] = True
    while queue:
        node = queue.popleft()
        for neighbor, edge_id in adjacency[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
                selected.append(edge_id)
    if not visited.all():
        raise RuntimeError("LCC graph unexpectedly disconnected")
    return np.asarray(selected, dtype=np.int64)


def split_snapshot(frame: pd.DataFrame, seed: int, ratios: tuple[float, float, float]) -> dict[str, pd.DataFrame]:
    tree = spanning_tree_indices(frame, int(frame[["source", "target"]].to_numpy().max()) + 1)
    edge_count = len(frame)
    train_count = int(round(edge_count * ratios[0]))
    validation_count = int(round(edge_count * ratios[1]))
    if train_count < len(tree):
        raise RuntimeError("train split is too small to contain a spanning tree")
    tree_set = set(tree.tolist())
    remaining = np.asarray([i for i in range(edge_count) if i not in tree_set], dtype=np.int64)
    np.random.default_rng(seed).shuffle(remaining)
    extra = train_count - len(tree)
    indices = {
        "train": np.concatenate([tree, remaining[:extra]]),
        "val": remaining[extra:extra + validation_count],
        "test": remaining[extra + validation_count:],
    }
    return {name: frame.iloc[index].reset_index(drop=True) for name, index in indices.items()}


def connected(frame: pd.DataFrame, num_nodes: int) -> bool:
    if num_nodes == 0:
        return False
    adjacency = [[] for _ in range(num_nodes)]
    for source, target in frame[["source", "target"]].itertuples(index=False):
        adjacency[int(source)].append(int(target)); adjacency[int(target)].append(int(source))
    seen, stack = {0}, [0]
    while stack:
        for neighbor in adjacency[stack.pop()]:
            if neighbor not in seen:
                seen.add(neighbor); stack.append(neighbor)
    return len(seen) == num_nodes


def write_dataset(dataset: str, spec: dict, raw_path: Path, output: Path,
                  seed: int, ratios: tuple[float, float, float], overwrite: bool) -> dict:
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output exists: {output} (use --overwrite)")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    raw = parse_raw(raw_path, spec["format"])
    events, build_to_raw = build_events(raw)
    final = canonical_keep_last(events)
    lcc_nodes = largest_component_nodes(final)
    final_lcc, build_to_final = remap_lcc(final, lcc_nodes)
    final_lcc = final_lcc[["source", "target", "sign", "weight", "event_id"] +
                          (["timestamp"] if "timestamp" in final_lcc else [])]
    splits = split_snapshot(final_lcc, seed, ratios)
    num_nodes = len(lcc_nodes)

    directed_final = events.copy()
    directed_final["sign"] = np.sign(directed_final["weight"]).astype(np.int8)
    directed_final = directed_final[
        directed_final["source"].isin(build_to_final) & directed_final["target"].isin(build_to_final)
    ].copy()
    directed_final["source"] = directed_final["source"].map(build_to_final).astype(int)
    directed_final["target"] = directed_final["target"].map(build_to_final).astype(int)
    directed_final["pair_source"] = np.minimum(directed_final["source"], directed_final["target"])
    directed_final["pair_target"] = np.maximum(directed_final["source"], directed_final["target"])
    directed_final = directed_final.drop_duplicates(["pair_source", "pair_target"], keep="last")
    directed_splits = {}
    for name, physical in splits.items():
        keys = physical[["source", "target"]].rename(
            columns={"source": "pair_source", "target": "pair_target"}).copy()
        keys["_split_order"] = np.arange(len(keys), dtype=np.int64)
        event_columns = ["pair_source", "pair_target", "event_id", "source", "target", "sign", "weight"]
        if "timestamp" in directed_final:
            event_columns.append("timestamp")
        directed = keys.merge(
            directed_final[event_columns], on=["pair_source", "pair_target"],
            how="left", validate="one_to_one", sort=False,
        ).sort_values("_split_order", kind="mergesort")
        directed = directed.drop(columns=["pair_source", "pair_target", "_split_order"])
        if directed[["event_id", "source", "target", "sign", "weight"]].isna().any().any():
            raise RuntimeError(f"failed to restore directed events for {dataset}/{name}")
        for column in ("event_id", "source", "target", "sign"):
            directed[column] = directed[column].astype(int)
        directed["weight"] = directed["weight"].astype(float)
        if "timestamp" in directed:
            if pd.api.types.is_datetime64_any_dtype(directed["timestamp"]):
                directed["timestamp"] = directed["timestamp"].map(lambda value: int(value.timestamp()))
            else:
                directed["timestamp"] = directed["timestamp"].astype(int)
        directed.to_csv(output / f"{name}_events.csv", index=False)
        directed_splits[name] = directed

    train_undirected = splits["train"][["source", "target", "sign", "weight"]].copy()
    # The structural snapshot is an unweighted signed graph. Preserve the
    # original rating only in the directed event view used by the model.
    train_undirected["weight"] = train_undirected["sign"].astype(float)
    train_undirected.to_csv(output / "train_snapshot_undirected.csv", index=False)
    directed_splits["train"].to_csv(output / "train_snapshot_directed.csv", index=False)
    pd.DataFrame({
        "node_id": range(num_nodes),
        "build_node_id": np.sort(lcc_nodes),
        "raw_node_id": [build_to_raw[int(node)] for node in np.sort(lcc_nodes)],
    }).to_csv(output / "node_mapping.csv", index=False)

    pair_sets = {name: set(map(tuple, frame[["source", "target"]].to_numpy()))
                 for name, frame in splits.items()}
    disjoint = not (pair_sets["train"] & pair_sets["val"] or
                    pair_sets["train"] & pair_sets["test"] or pair_sets["val"] & pair_sets["test"])
    manifest = {
        "schema_version": 1,
        "dataset": dataset,
        "raw_file": spec["raw_file"],
        "raw_sha256": sha256(raw_path),
        "preprocessing": {
            "split_seed": seed,
            "split_ratios": list(ratios),
            "snapshot": "undirected physical pairs; keep final event",
            "component": "largest connected component of full final snapshot",
            "train_connectivity": "spanning tree reserved before random split",
            "direction": "final observed event direction restored for model views",
        },
        "counts": {
            "raw_parsed_edges": len(raw), "built_events": len(events),
            "final_physical_edges": len(final), "lcc_physical_edges": len(final_lcc),
            "num_nodes": num_nodes,
            "splits": {name: len(frame) for name, frame in splits.items()},
            "positive": {name: int((frame["sign"] > 0).sum()) for name, frame in splits.items()},
            "negative": {name: int((frame["sign"] < 0).sum()) for name, frame in splits.items()},
        },
        "validation": {"train_connected": connected(splits["train"], num_nodes),
                       "pair_disjoint": disjoint, "transductive": True},
        "communities": {
            "result": "communities/signed_louvain_summary.json",
            "fallback_k": spec["fallback_k"],
            "minimum_community_size": spec["minimum_community_size"],
        },
    }
    if not all(manifest["validation"].values()):
        raise RuntimeError(f"preprocessing validation failed: {manifest['validation']}")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-contained signed-EPM preprocessing")
    parser.add_argument("--dataset", nargs="+", default=["bitcoinalpha"])
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checksums", type=Path, default=DEFAULT_CHECKSUMS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", nargs=3, type=float, default=(0.7, 0.15, 0.15))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    configs, checksums = load_json(args.config), load_json(args.checksums)["files"]
    requested = list(configs) if args.dataset == ["all"] else [ALIASES.get(name.lower(), name) for name in args.dataset]
    if not np.isclose(sum(args.ratios), 1.0):
        raise ValueError("split ratios must sum to one")
    for dataset in requested:
        if dataset not in configs:
            raise ValueError(f"unknown dataset: {dataset}")
        spec = configs[dataset]
        raw_path = args.raw_dir / spec["raw_file"]
        expected = checksums[dataset]
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        actual = sha256(raw_path)
        if actual != expected["sha256"]:
            raise RuntimeError(f"checksum mismatch for {raw_path}: {actual}")
        manifest = write_dataset(dataset, spec, raw_path, args.output_root / dataset,
                                 args.seed, tuple(args.ratios), args.overwrite)
        print(json.dumps({"dataset": dataset, "output": str(args.output_root / dataset),
                          **manifest["counts"], **manifest["validation"]}, indent=2))


if __name__ == "__main__":
    main()
