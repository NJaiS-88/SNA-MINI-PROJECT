"""
feature_engineering.py -- Phase 3: one row of structural (SNA) features per node.

Features (all computed from graph structure only -- NO diffusion/target information):
  degree                  number of friends
  degree_centrality       degree / (n-1)
  betweenness             share of shortest paths passing through the node (bridges/brokers)
  closeness               inverse average distance to all other nodes (central position)
  pagerank                importance by being linked from important nodes
  eigenvector             like PageRank but without damping/teleport
  clustering              fraction of the node's friends that are friends of each other
  avg_neighbor_degree     how popular the node's friends are (2-hop reach proxy)
  core_number             k-core: deepest dense core the node belongs to
  community_id            Louvain label (kept for analysis ONLY, never a regression input)
  community_size          size of the node's community (numeric, meaningful)
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

# Columns the ML model is allowed to see. community_id is deliberately excluded
# (an arbitrary label would be wrongly treated as an ordered number).
ML_FEATURES = ["degree", "degree_centrality", "betweenness", "closeness", "pagerank",
               "eigenvector", "clustering", "avg_neighbor_degree", "core_number",
               "community_size"]


def _eigenvector(G: nx.Graph) -> dict:
    """Eigenvector centrality with fallbacks for convergence problems."""
    try:
        return nx.eigenvector_centrality(G, max_iter=1000, tol=1e-8)
    except (nx.PowerIterationFailedConvergence, nx.NetworkXException):
        pass
    try:
        return nx.eigenvector_centrality_numpy(G)
    except Exception:
        print("  [warn] eigenvector centrality failed -> using zeros (feature carries no information).")
        return {n: 0.0 for n in G}


def compute_features(G: nx.Graph, part: dict, betweenness_k: int | None = None,
                     seed: int = 42, cache_dir: Path | None = None,
                     force: bool = False) -> pd.DataFrame:
    """
    Compute (or load from cache) the node feature table.
    betweenness_k=None -> exact betweenness (slow on 4k nodes / 88k edges, minutes);
    betweenness_k=int  -> approximate using that many sampled source nodes (quick mode).
    """
    n, m = G.number_of_nodes(), G.number_of_edges()
    if n == 0:
        raise ValueError("Empty graph.")
    sig = {"n": n, "m": m, "betweenness_k": betweenness_k, "seed": seed}
    csv_path = meta_path = None
    if cache_dir is not None:
        csv_path, meta_path = Path(cache_dir) / "node_features.csv", Path(cache_dir) / "node_features_meta.json"
        if not force and csv_path.exists() and meta_path.exists():
            if json.loads(meta_path.read_text()) == sig:
                print("  Loaded cached node features.")
                return pd.read_csv(csv_path)

    print("  Computing centralities (betweenness/closeness can take a while) ...")
    deg = dict(G.degree())
    k = None if betweenness_k is None or betweenness_k >= n else betweenness_k
    bet = nx.betweenness_centrality(G, k=k, seed=seed)   # brokers lying on shortest paths
    clo = nx.closeness_centrality(G)                     # wf_improved handles disconnected graphs
    pr = nx.pagerank(G, alpha=0.85)                      # random-surfer importance
    eig = _eigenvector(G)
    clus = nx.clustering(G)
    and_ = nx.average_neighbor_degree(G)
    core = nx.core_number(G)
    comm_size = pd.Series(part).value_counts().to_dict()

    df = pd.DataFrame({
        "node_id": list(G.nodes()),
        "degree": [deg[v] for v in G],
        "degree_centrality": [deg[v] / max(n - 1, 1) for v in G],
        "betweenness": [bet[v] for v in G],
        "closeness": [clo[v] for v in G],
        "pagerank": [pr[v] for v in G],
        "eigenvector": [eig[v] for v in G],
        "clustering": [clus[v] for v in G],
        "avg_neighbor_degree": [and_[v] for v in G],      # 0 for isolated nodes (no neighbours)
        "core_number": [core[v] for v in G],
        "community_id": [part[v] for v in G],
        "community_size": [comm_size[part[v]] for v in G],
    }).sort_values("node_id").reset_index(drop=True)

    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        meta_path.write_text(json.dumps(sig))
    return df


def assert_no_leakage(feature_cols: list[str]) -> None:
    """Guard: the target (or anything derived from it) must never be an input feature."""
    banned = {"mean_spread", "std_spread", "mean_coverage", "influence_score",
              "predicted_influence_score", "node_id", "community_id"}
    bad = banned.intersection(feature_cols)
    if bad:
        raise AssertionError(f"Leakage / invalid features in X: {sorted(bad)}")
