"""
influencer_selection.py -- Phases 4 & 11: traditional rankings and seed strategies A-F.

Every strategy returns exactly k distinct nodes (same campaign budget for all).
Ties are broken deterministically by node id.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .campaign_optimizer import greedy_community_aware_selection

CENTRALITIES = ["degree_centrality", "betweenness", "closeness", "pagerank", "eigenvector"]


def check_budget(k: int, n: int) -> None:
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"Seed budget k must be a positive integer (got {k}).")
    if k > n:
        raise ValueError(f"Seed budget k={k} exceeds number of nodes n={n}.")


def top_k_nodes(scores: pd.Series | np.ndarray, k: int, node_ids=None) -> list[int]:
    """Top-k by score (descending), ties -> smaller node id first."""
    scores = np.asarray(scores, dtype=float)
    ids = np.arange(len(scores)) if node_ids is None else np.asarray(node_ids)
    check_budget(k, len(scores))
    order = np.lexsort((ids, -scores))
    return [int(ids[i]) for i in order[:k]]


def traditional_rankings(features: pd.DataFrame, top_n: int = 10):
    """Top-N nodes per classical centrality + Spearman matrix + top-N overlap matrix."""
    rows, tops = [], {}
    for c in CENTRALITIES:
        top = top_k_nodes(features[c].to_numpy(), top_n, features.node_id.to_numpy())
        tops[c] = top
        for r, v in enumerate(top, 1):
            rows.append({"measure": c, "rank": r, "node_id": v, "score": float(features.loc[features.node_id == v, c].iloc[0])})
    trad = pd.DataFrame(rows)
    spear = features[CENTRALITIES].corr(method="spearman")
    overlap = pd.DataFrame({a: {b: len(set(tops[a]) & set(tops[b])) / top_n for b in CENTRALITIES}
                            for a in CENTRALITIES})
    return trad, tops, spear, overlap


def traditional_table(features: pd.DataFrame, tops: dict) -> pd.DataFrame:
    """'Traditional Influencers' table: Rank | Node | Degree | PageRank | Betweenness (by degree rank)."""
    f = features.set_index("node_id")
    rows = [{"Rank": r, "Node": v, "Degree": int(f.loc[v, "degree"]), "PageRank": f.loc[v, "pagerank"],
             "Betweenness": f.loc[v, "betweenness"]} for r, v in enumerate(tops["degree_centrality"], 1)]
    return pd.DataFrame(rows)


# ---- Strategy A: random --------------------------------------------------------------
def random_seed_sets(n: int, k: int, n_sets: int, base_seed: int) -> list[list[int]]:
    """A NEW random seed set for every simulation, so the baseline is averaged over draws."""
    check_budget(k, n)
    return [sorted(int(x) for x in np.random.default_rng([base_seed, i]).choice(n, size=k, replace=False))
            for i in range(n_sets)]


# ---- Strategies B, C, D: single-metric top-k ---------------------------------------------
def degree_seeds(features, k):      return top_k_nodes(features.degree.to_numpy(), k, features.node_id.to_numpy())
def pagerank_seeds(features, k):    return top_k_nodes(features.pagerank.to_numpy(), k, features.node_id.to_numpy())
def betweenness_seeds(features, k): return top_k_nodes(features.betweenness.to_numpy(), k, features.node_id.to_numpy())


# ---- Strategy E: AI top-k --------------------------------------------------------------
def ai_seeds(predictions: pd.DataFrame, k: int) -> list[int]:
    return top_k_nodes(predictions.predicted_influence_score.to_numpy(), k, predictions.node_id.to_numpy())


# ---- Strategy F: AI + community-aware greedy optimisation -------------------------------
def ai_community_seeds(G, predictions, part, k, alpha, beta, gamma, pool_size):
    return greedy_community_aware_selection(G, predictions, part, k, alpha, beta, gamma, pool_size)


def build_all_strategies(G, features, predictions, part, k, n_random_sets, rand_seed,
                         alpha, beta, gamma, pool_size) -> tuple[dict, pd.DataFrame]:
    """Return {strategy name: list of seed sets} (1 set = fixed, many = per-simulation)."""
    n = G.number_of_nodes()
    comm_seeds, trace = ai_community_seeds(G, predictions, part, k, alpha, beta, gamma, pool_size)
    strategies = {
        "Random": random_seed_sets(n, k, n_random_sets, rand_seed),
        "Degree": [degree_seeds(features, k)],
        "PageRank": [pagerank_seeds(features, k)],
        "Betweenness": [betweenness_seeds(features, k)],
        "AI": [ai_seeds(predictions, k)],
        "AI + Community": [comm_seeds],
    }
    return strategies, trace
