"""
community_detection.py -- Phase 5.

COMMUNITY DETECTION groups users so that there are many edges inside a group and
few between groups. We use the LOUVAIN algorithm: greedily move nodes between groups
to maximise MODULARITY Q, then collapse groups into super-nodes and repeat.
Fallback (if python-louvain is not installed): NetworkX greedy modularity.
"""
from __future__ import annotations

import networkx as nx
import pandas as pd
from networkx.algorithms.community import modularity


def detect_communities(G: nx.Graph, random_state: int = 42) -> tuple[dict, str, float]:
    """Return (node -> community_id, method name, modularity)."""
    try:
        import community as community_louvain  # python-louvain
        part = community_louvain.best_partition(G, random_state=random_state)
        method = "Louvain"
    except ImportError:
        print("  [warn] python-louvain not installed -> using greedy modularity fallback.")
        comms = nx.algorithms.community.greedy_modularity_communities(G)
        part = {n: i for i, c in enumerate(comms) for n in c}
        method = "Greedy modularity (fallback)"
    groups: dict[int, set] = {}
    for n, c in part.items():
        groups.setdefault(c, set()).add(n)
    q = modularity(G, list(groups.values()))
    return part, method, float(q)


def community_statistics(G: nx.Graph, part: dict, pagerank: dict) -> pd.DataFrame:
    """Per-community size, top-degree node and top-PageRank node."""
    rows = []
    groups: dict[int, list] = {}
    for n, c in part.items():
        groups.setdefault(c, []).append(n)
    for c, nodes in groups.items():
        top_deg = max(nodes, key=lambda x: (G.degree(x), -x))
        top_pr = max(nodes, key=lambda x: (pagerank[x], -x))
        rows.append({"community_id": c, "size": len(nodes),
                     "top_degree_node": top_deg, "top_degree": G.degree(top_deg),
                     "top_pagerank_node": top_pr})
    df = pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)
    return df


def add_ml_top_node(stats: pd.DataFrame, part: dict, predicted: pd.Series) -> pd.DataFrame:
    """Add the top ML-predicted influencer of each community (called after ML is trained)."""
    best = {}
    for n, c in part.items():
        s = predicted.get(n, float("-inf"))
        if c not in best or s > best[c][1] or (s == best[c][1] and n < best[c][0]):
            best[c] = (n, s)
    out = stats.copy()
    out["top_ml_node"] = out.community_id.map(lambda c: best[c][0])
    out["top_ml_predicted_influence"] = out.community_id.map(lambda c: round(best[c][1], 3))
    return out
