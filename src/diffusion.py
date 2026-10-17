"""
diffusion.py -- Phase 6 & 13: Independent Cascade (IC) and Linear Threshold (LT),
implemented FROM SCRATCH (no diffusion library).

Graph representation: CSR arrays (indptr, indices). The neighbours of node u are
`indices[indptr[u]:indptr[u+1]]`. This is only a fast adjacency list; the algorithms
below are the textbook ones and are easy to explain.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

CSR = tuple[np.ndarray, np.ndarray]


def build_csr(G: nx.Graph) -> CSR:
    """Convert a graph with nodes 0..n-1 into CSR adjacency arrays."""
    n = G.number_of_nodes()
    if n == 0:
        raise ValueError("Cannot simulate diffusion on an empty graph.")
    if set(G.nodes()) != set(range(n)):
        raise ValueError("Nodes must be labelled 0..n-1 (data_loader does this).")
    deg = np.fromiter((G.degree(i) for i in range(n)), dtype=np.int64, count=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(deg)
    indices = np.empty(indptr[-1], dtype=np.int32)
    for i in range(n):
        indices[indptr[i]:indptr[i + 1]] = sorted(G.neighbors(i))
    return indptr, indices


def _prepare(csr: CSR, seeds, random_seed, rng):
    indptr, indices = csr
    n = len(indptr) - 1
    if rng is None:
        rng = np.random.default_rng(random_seed)
    seeds = np.unique(np.asarray(list(seeds), dtype=np.int64))
    if seeds.size == 0:
        raise ValueError("At least one seed node is required.")
    if seeds.min() < 0 or seeds.max() >= n:
        raise ValueError("Seed node outside graph.")
    return indptr, indices, n, rng, seeds


def _package(active, act_time, new_counts, n):
    new_per_step = np.asarray(new_counts, dtype=np.int64)
    total = int(active.sum())
    return {
        "activated_nodes": np.flatnonzero(active),          # final activated set
        "total_activated": total,                            # includes the seeds
        "activation_time": act_time,                         # -1 = never activated
        "new_per_step": new_per_step,                        # index 0 = seeds (t=0)
        "cumulative_per_step": np.cumsum(new_per_step),
        "n_steps": len(new_counts) - 1,                      # cascade duration
        "coverage": 100.0 * total / n,                       # % of network
    }


def independent_cascade(csr: CSR, seeds, p: float, random_seed: int | None = None,
                        rng: np.random.Generator | None = None) -> dict:
    """
    INDEPENDENT CASCADE (Kempe, Kleinberg, Tardos 2003).
      1. Seeds are active at t=0.
      2. Every node that became active at step t-1 gets exactly ONE chance to
         activate each of its inactive neighbours.
      3. Each attempt succeeds independently with probability p.
      4. Newly activated nodes become the frontier of step t (they try at t+1).
      5. Nodes never retry (they only live in the frontier once).
      6. Stop when a step produces no new activation.
    (Marking a node active immediately when ANY attempt succeeds does not change the
    outcome distribution: a second success on the same node would be irrelevant.)
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("Propagation probability p must be in [0, 1].")
    indptr, indices, n, rng, seeds = _prepare(csr, seeds, random_seed, rng)

    active = np.zeros(n, dtype=bool)
    act_time = np.full(n, -1, dtype=np.int32)
    active[seeds] = True
    act_time[seeds] = 0
    frontier = seeds
    new_counts = [int(seeds.size)]
    t = 0
    while frontier.size > 0:
        t += 1
        newly = []
        for u in frontier:
            nbrs = indices[indptr[u]:indptr[u + 1]]
            cand = nbrs[~active[nbrs]]                   # inactive neighbours only
            if cand.size == 0:
                continue
            hit = cand[rng.random(cand.size) < p]        # each edge succeeds w.p. p
            if hit.size:
                active[hit] = True
                act_time[hit] = t
                newly.append(hit)
        if not newly:
            break
        frontier = np.concatenate(newly)
        new_counts.append(int(frontier.size))
    return _package(active, act_time, new_counts, n)


def linear_threshold(csr: CSR, seeds, random_seed: int | None = None,
                     rng: np.random.Generator | None = None) -> dict:
    """
    LINEAR THRESHOLD (Granovetter 1978 / Kempe et al. 2003), uniform edge weights.
      * Every node v draws a threshold theta_v ~ U(0,1).
      * Every neighbour of v has weight 1/deg(v) (weights of v sum to 1).
      * v activates when (#active neighbours)/deg(v) >= theta_v.
      * Synchronous rounds until no new activation.
    LT has no propagation probability; randomness comes only from thresholds.
    """
    indptr, indices, n, rng, seeds = _prepare(csr, seeds, random_seed, rng)
    deg = np.diff(indptr)
    theta = rng.random(n)
    n_active_nbrs = np.zeros(n, dtype=np.int64)

    active = np.zeros(n, dtype=bool)
    act_time = np.full(n, -1, dtype=np.int32)
    active[seeds] = True
    act_time[seeds] = 0
    frontier = seeds
    new_counts = [int(seeds.size)]
    t = 0
    while frontier.size > 0:
        t += 1
        touched = []
        for u in frontier:
            nbrs = indices[indptr[u]:indptr[u + 1]]
            n_active_nbrs[nbrs] += 1                     # u is now an active neighbour
            touched.append(nbrs)
        cand = np.unique(np.concatenate(touched))
        cand = cand[~active[cand]]
        if cand.size == 0:
            break
        ok = n_active_nbrs[cand] >= theta[cand] * deg[cand] - 1e-12
        newly = cand[ok]
        if newly.size == 0:
            break
        active[newly] = True
        act_time[newly] = t
        frontier = newly
        new_counts.append(int(newly.size))
    return _package(active, act_time, new_counts, n)


def cascade_metrics(res: dict, n_seeds: int) -> dict:
    """Scalar descriptors of one simulated campaign cascade (viral-campaign analysis)."""
    new = res["new_per_step"]
    spread = res["total_activated"]
    if len(new) > 1:                                     # ignore t=0 (the seeds themselves)
        peak = int(new[1:].max())
        t_peak = int(np.argmax(new[1:]) + 1)
    else:
        peak, t_peak = 0, 0
    return {
        "spread": spread,
        "coverage": res["coverage"],
        "duration": res["n_steps"],
        "peak_new": peak,
        "time_to_peak": t_peak,
        # average number of NEW activations per timestep after t=0
        "growth_rate": (spread - n_seeds) / max(res["n_steps"], 1),
    }
