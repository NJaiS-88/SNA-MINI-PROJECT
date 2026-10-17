"""
campaign_optimizer.py -- Phase 12: greedy community-aware campaign seed optimisation.

Picks a SET of k seeds one at a time. At each step every remaining candidate v gets

    score(v) = alpha * predicted_influence_norm(v)       (be influential)
             + beta  * community_diversity_bonus(v)      (reach a new community)
             - gamma * overlap_penalty(v)                (avoid redundant audience)

and the best-scoring candidate is added. This is a heuristic ("greedy community-aware
campaign seed optimisation"); it does NOT guarantee a global optimum.

  predicted_influence_norm : ML prediction min-max scaled to [0,1] over the candidate pool
  community_diversity_bonus: 1 if v's community has no seed yet, else 0
  overlap_penalty          : max Jaccard similarity between v's closed neighbourhood and
                             that of any already-selected seed (0 for the first pick)
"""
from __future__ import annotations

import networkx as nx
import pandas as pd


def jaccard(a: set, b: set) -> float:
    """J(A,B) = |A ∩ B| / |A ∪ B|, returning 0.0 if the union is empty."""
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def greedy_community_aware_selection(G: nx.Graph, predictions: pd.DataFrame, part: dict, k: int,
                                     alpha: float = 0.6, beta: float = 0.25, gamma: float = 0.5,
                                     pool_size: int = 300) -> tuple[list[int], pd.DataFrame]:
    """Return (seed list, step-by-step trace table)."""
    n = G.number_of_nodes()
    if k < 1 or k > n:
        raise ValueError(f"Invalid seed budget k={k} for a graph with {n} nodes.")
    pool_size = min(n, max(pool_size, 3 * k))
    pool = predictions.sort_values(["predicted_influence_score", "node_id"],
                                   ascending=[False, True]).head(pool_size)
    pred = dict(zip(pool.node_id.astype(int), pool.predicted_influence_score))
    lo, hi = min(pred.values()), max(pred.values())
    norm = {v: (s - lo) / (hi - lo) if hi > lo else 1.0 for v, s in pred.items()}
    nbhd = {v: set(G.neighbors(v)) | {v} for v in pred}      # closed neighbourhoods

    seeds, covered, trace = [], set(), []
    remaining = set(pred)
    while len(seeds) < k:
        best = None
        for v in sorted(remaining):
            bonus = 0.0 if part[v] in covered else 1.0
            overlap = max((jaccard(nbhd[v], nbhd[s]) for s in seeds), default=0.0)
            score = alpha * norm[v] + beta * bonus - gamma * overlap
            if best is None or score > best[0] + 1e-12:
                best = (score, v, norm[v], bonus, overlap)
        score, v, nv, bonus, overlap = best
        seeds.append(int(v))
        covered.add(part[v])
        remaining.discard(v)
        trace.append({"step": len(seeds), "node_id": int(v), "community_id": part[v],
                      "predicted_norm": round(nv, 4), "diversity_bonus": bonus,
                      "overlap_penalty": round(overlap, 4), "score": round(score, 4)})
    return seeds, pd.DataFrame(trace)
