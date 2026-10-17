"""network_analysis.py -- Phase 2: basic Social Network Analysis statistics."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def _path_stats(G_lcc: nx.Graph, sample: int | None, seed: int) -> tuple[float, int, bool]:
    """
    Average shortest path length + diameter via BFS from every node (exact) or from
    `sample` random sources (estimate; the diameter is then a LOWER BOUND).
    """
    nodes = list(G_lcc.nodes())
    exact = sample is None or sample >= len(nodes)
    if not exact:
        rng = np.random.default_rng(seed)
        nodes = list(rng.choice(nodes, size=sample, replace=False))
    total, pairs, ecc_max = 0, 0, 0
    for s in nodes:
        d = nx.single_source_shortest_path_length(G_lcc, int(s))
        total += sum(d.values())
        pairs += len(d) - 1
        ecc_max = max(ecc_max, max(d.values()))
    return total / max(pairs, 1), ecc_max, exact


def compute_network_statistics(G: nx.Graph, path_sample: int | None = None,
                               seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the 'Network Statistics' table (Metric | Value) and the degree array."""
    n, m = G.number_of_nodes(), G.number_of_edges()
    deg = np.array([d for _, d in G.degree()])
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    lcc = G.subgraph(comps[0])
    avg_path, diam, exact = _path_stats(lcc, path_sample, seed)
    rows = [
        ("Nodes", n),
        ("Edges", m),
        ("Network density", round(nx.density(G), 6)),
        ("Average degree", round(deg.mean(), 3)),
        ("Minimum degree", int(deg.min())),
        ("Maximum degree", int(deg.max())),
        ("Median degree", float(np.median(deg))),
        ("Average clustering coefficient", round(nx.average_clustering(G), 4)),
        ("Global transitivity", round(nx.transitivity(G), 4)),
        ("Degree assortativity", round(nx.degree_assortativity_coefficient(G), 4)),
        ("Connected components", len(comps)),
        ("Largest component size", len(comps[0])),
        ("Average shortest path (largest component)" + ("" if exact else " [sampled estimate]"),
         round(avg_path, 3)),
        ("Diameter (largest component)" + ("" if exact else " [lower bound, sampled]"), diam),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"]), deg


def interpret_statistics(table: pd.DataFrame) -> list[str]:
    """Plain-language interpretation generated from the actual numbers."""
    v = dict(zip(table.Metric, table.Value))
    path_key = [k for k in v if k.startswith("Average shortest path")][0]
    out = [f"Density {v['Network density']} -> the graph is sparse: each user is linked to a small "
           f"fraction of all users (average degree {v['Average degree']})."]
    if v["Maximum degree"] > 5 * v["Average degree"]:
        out.append(f"Maximum degree ({v['Maximum degree']}) is far above the average: a heavy-tailed "
                   "degree distribution with a few hub users.")
    c = v["Average clustering coefficient"]
    out.append(f"Average clustering {c} -> " + ("friends of a user are very often friends with each "
               "other (dense local clusters)." if c > 0.3 else "moderate local clustering."))
    out.append(f"Average path length {v[path_key]} -> " + ("short paths ('small-world' property): "
               "information can reach distant users in few hops." if v[path_key] < 8 else
               "relatively long paths."))
    return out
