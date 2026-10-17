"""
data_loader.py -- Phase 1: dataset loading and preprocessing.

Dataset: SNAP "ego-Facebook" (facebook_combined.txt).
  * Nodes = anonymised Facebook users (integer IDs, no real identities).
  * Edges = undirected friendship connections.
It is a purely STRUCTURAL graph: it has no real campaign cascades or influencer
labels, so all diffusion in this project is SIMULATED.
"""
from __future__ import annotations

import gzip
import shutil
import urllib.request
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

SNAP_URL = "https://snap.stanford.edu/data/facebook_combined.txt.gz"


def download_dataset(dest: Path) -> Path:
    """Download + unzip the SNAP Facebook edge list to `dest` (relative path is fine)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    gz_path = dest.with_suffix(dest.suffix + ".gz")
    print(f"Downloading {SNAP_URL} ...")
    urllib.request.urlretrieve(SNAP_URL, gz_path)
    with gzip.open(gz_path, "rb") as fin, open(dest, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    gz_path.unlink(missing_ok=True)
    print(f"Saved dataset to {dest}")
    return dest


def load_edge_list(path: Path) -> pd.DataFrame:
    """Read a whitespace separated edge list (lines starting with '#' are ignored)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Fix: run `python download_dataset.py` (or see README 'Dataset Setup') "
            "to place facebook_combined.txt inside the data/ folder.\n"
            "For a pipeline smoke-test only you may run `python main.py --synthetic`."
        )
    edges = pd.read_csv(path, sep=r"\s+", comment="#", header=None,
                        names=["u", "v"], dtype=np.int64, engine="python")
    if edges.empty:
        raise ValueError(f"Edge list '{path}' is empty.")
    return edges


def build_graph(edges: pd.DataFrame) -> tuple[nx.Graph, dict]:
    """
    Build a simple undirected NetworkX graph.
      * self-loops are removed (counted),
      * duplicate / reversed duplicate edges collapse (counted),
      * nodes are relabelled to consecutive integers 0..n-1 (sorted by original ID).
    For the Facebook file the IDs are already 0..4038, so labels are unchanged.
    """
    n_lines = len(edges)
    n_self = int((edges.u == edges.v).sum())
    e = edges[edges.u != edges.v]
    canon = pd.DataFrame({"a": np.minimum(e.u, e.v), "b": np.maximum(e.u, e.v)})
    n_dup = int(canon.duplicated().sum())
    canon = canon.drop_duplicates()

    G = nx.Graph()
    G.add_edges_from(canon.itertuples(index=False, name=None))
    if G.number_of_nodes() == 0:
        raise ValueError("Graph is empty after preprocessing.")

    orig_ids = np.array(sorted(G.nodes()))
    mapping_changed = not np.array_equal(orig_ids, np.arange(len(orig_ids)))
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    report = {"raw_edge_lines": n_lines, "self_loops_removed": n_self,
              "duplicate_edges_removed": n_dup, "ids_relabelled": mapping_changed,
              "orig_ids": orig_ids}
    return G, report


def graph_report(G: nx.Graph, report: dict | None = None, verbose: bool = True) -> dict:
    """Connectedness report. NOTHING is discarded here -- we only identify the LCC."""
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    lcc = comps[0]
    info = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "graph_type": "Undirected, unweighted, simple graph",
        "n_isolated_nodes": sum(1 for _, d in G.degree() if d == 0),
        "n_components": len(comps),
        "lcc_size": len(lcc),
        "is_connected": len(comps) == 1,
        "lcc_nodes": lcc,
    }
    if verbose:
        print("=" * 60)
        print("GRAPH / PREPROCESSING REPORT")
        if report and "raw_edge_lines" in report:
            print(f"  Raw edge lines read        : {report['raw_edge_lines']}")
            print(f"  Self-loops removed         : {report['self_loops_removed']}")
            print(f"  Duplicate edges removed    : {report['duplicate_edges_removed']}")
        print(f"  Number of nodes            : {info['n_nodes']}")
        print(f"  Number of edges            : {info['n_edges']}")
        print(f"  Graph type                 : {info['graph_type']}")
        print(f"  Isolated nodes             : {info['n_isolated_nodes']}")
        print(f"  Connected components       : {info['n_components']}")
        print(f"  Largest component size     : {info['lcc_size']}")
        print(f"  Graph connected?           : {info['is_connected']}")
        if not info["is_connected"]:
            print("  NOTE: path-length metrics use the largest component only; "
                  "diffusion and centralities use the FULL graph.")
        print("=" * 60)
    return info


def load_graph(path: Path) -> tuple[nx.Graph, dict]:
    edges = load_edge_list(path)
    G, rep = build_graph(edges)
    info = graph_report(G, rep)
    info["prep"] = rep
    return G, info


def generate_synthetic_graph(n: int = 2000, seed: int = 42) -> tuple[nx.Graph, dict]:
    """
    SMOKE-TEST ONLY. A scale-free, clustered synthetic graph used to verify that the
    pipeline runs when the real dataset is unavailable. Results from this graph must
    NEVER be reported as results of the real-world dataset.
    """
    G = nx.powerlaw_cluster_graph(n, max(4, int(round(n * 0.0055))), 0.5, seed=seed)
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    info = graph_report(G, None)
    info["prep"] = {"synthetic": True}
    return G, info
