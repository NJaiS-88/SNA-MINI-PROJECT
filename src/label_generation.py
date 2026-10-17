"""
label_generation.py -- Phase 7: diffusion-based influence labels (the SNA <-> AI bridge).

There are no real influencer labels, so we ESTIMATE each candidate node's diffusion
potential by MONTE CARLO simulation:

    Influence(v) = E[ number of nodes activated by Independent Cascade when v is the
                      ONLY seed ]   ~=  mean over R simulated cascades.

This is "simulation-based influencer potential", not an observed real-world quantity.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .diffusion import CSR, independent_cascade


def select_candidates(degrees: np.ndarray, n_candidates: int, seed: int,
                      n_bins: int = 10) -> np.ndarray:
    """
    Choose the nodes for which labels are simulated (simulating all nodes can be slow).
    Sampling is STRATIFIED over degree quantile bins so that hubs (rare but important)
    are represented in both train and test. Documented in README: the labelled sample is
    therefore not a uniform sample of the population.
    """
    n = len(degrees)
    if n_candidates >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    order = np.argsort(degrees, kind="stable")
    bins = np.array_split(order, n_bins)
    per_bin = int(np.ceil(n_candidates / n_bins))
    chosen = [rng.choice(b, size=min(per_bin, len(b)), replace=False) for b in bins]
    cand = np.concatenate(chosen)
    if len(cand) > n_candidates:
        cand = rng.choice(cand, size=n_candidates, replace=False)
    return np.sort(cand)


def _simulate_nodes(csr: CSR, nodes, p: float, n_sims: int, seed_base: int) -> list[tuple]:
    out = []
    n_total = len(csr[0]) - 1
    for v in nodes:
        spreads = np.empty(n_sims)
        for s in range(n_sims):
            rng = np.random.default_rng([seed_base, int(v), s])   # unique, reproducible stream
            spreads[s] = independent_cascade(csr, [int(v)], p, rng=rng)["total_activated"]
        out.append((int(v), spreads.mean(), spreads.std(ddof=1) if n_sims > 1 else 0.0,
                    100.0 * spreads.mean() / n_total))
    return out


def monte_carlo_spread(csr: CSR, nodes, p: float, n_sims: int, seed_base: int,
                       n_jobs: int = 1) -> pd.DataFrame:
    """Single-seed Monte Carlo spread for every node in `nodes` (parallel over chunks)."""
    nodes = np.asarray(nodes)
    n_jobs = max(1, n_jobs if n_jobs > 0 else (__import__("os").cpu_count() or 1))
    chunks = np.array_split(nodes, max(1, min(len(nodes), n_jobs * 4)))
    if n_jobs == 1:
        parts = [_simulate_nodes(csr, c, p, n_sims, seed_base) for c in chunks]
    else:
        parts = Parallel(n_jobs=n_jobs)(delayed(_simulate_nodes)(csr, c, p, n_sims, seed_base)
                                        for c in chunks)
    rows = [r for part in parts for r in part]
    return pd.DataFrame(rows, columns=["node_id", "mean_spread", "std_spread", "mean_coverage"])


def calibrate_probability(csr: CSR, p_grid: list[float], n_nodes: int, n_sims: int,
                          seed: int, seed_base: int, n_jobs: int = 1) -> tuple[float, pd.DataFrame]:
    """
    Pick the label probability p_label from a small pilot experiment.
    A good p gives (a) average coverage that is neither ~0 nor ~everything, (b) not too many
    cascades that die at once, (c) few giant cascades, and then (d) among those the HIGHEST
    label reliability = share of the between-node variance of mean spread that is real signal
    rather than Monte Carlo noise (most learnable target for the ML model).
    """
    n_total = len(csr[0]) - 1
    rng = np.random.default_rng(seed)
    sample = rng.choice(n_total, size=min(n_nodes, n_total), replace=False)
    rows = []
    for p in p_grid:
        sp, node_mean, node_var = [], [], []
        for v in sample:
            s = np.array([independent_cascade(csr, [int(v)], p, rng=np.random.default_rng([seed_base, int(v), i]))
                          ["total_activated"] for i in range(n_sims)], dtype=float)
            sp.extend(s)
            node_mean.append(s.mean())
            node_var.append(s.var(ddof=1))
        sp, node_mean = np.array(sp), np.array(node_mean)
        var_obs = node_mean.var(ddof=1)                         # variance of the Monte Carlo node means
        noise = np.mean(node_var) / n_sims                      # part of it that is pure simulation noise
        reliability = max(0.0, (var_obs - noise) / var_obs) if var_obs > 0 else 0.0
        rows.append({"p": p,
                     "mean_coverage_pct": 100 * sp.mean() / n_total,
                     "frac_dead(<=3 nodes)": float((sp <= 3).mean()),
                     "frac_giant(>=60%)": float((sp >= 0.6 * n_total).mean()),
                     "between_node_std_of_mean_spread": float(np.sqrt(var_obs)),
                     "label_reliability": reliability})
    cal = pd.DataFrame(rows)
    ok = cal[(cal.mean_coverage_pct.between(2, 25)) & (cal["frac_dead(<=3 nodes)"] <= 0.6)
             & (cal["frac_giant(>=60%)"] <= 0.2)]
    if len(ok):
        p_sel = float(ok.loc[ok.label_reliability.idxmax(), "p"])
        reason = "highest label reliability (signal / signal+noise) among eligible probabilities"
    else:
        p_sel = float(cal.loc[(cal.mean_coverage_pct - 10).abs().idxmin(), "p"])
        reason = "no probability met all criteria; closest to 10% mean coverage"
    cal["selected"] = cal.p == p_sel
    print(f"  Calibration -> p_label = {p_sel} ({reason})")
    return p_sel, cal


def generate_influence_labels(csr: CSR, candidates: np.ndarray, p: float, n_sims: int,
                              seed_base: int, results_dir: Path, n_jobs: int = 1,
                              force: bool = False, extra_meta: dict | None = None) -> pd.DataFrame:
    """Generate (or load cached) influence labels -> outputs/results/influence_labels.csv."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path = results_dir / "influence_labels.csv", results_dir / "influence_labels_meta.json"
    meta = {"p_label": p, "n_sims": n_sims, "n_candidates": int(len(candidates)),
            "candidates_hash": int(np.sum(candidates * 31 + 7)), "seed_base": seed_base,
            "n_nodes": len(csr[0]) - 1, "n_edge_entries": int(len(csr[1])), **(extra_meta or {})}
    if not force and csv_path.exists() and meta_path.exists():
        if json.loads(meta_path.read_text()) == meta:
            print("  Loaded cached influence labels.")
            return pd.read_csv(csv_path)
    print(f"  Simulating {len(candidates)} nodes x {n_sims} IC runs (p={p}) ...")
    df = monte_carlo_spread(csr, candidates, p, n_sims, seed_base, n_jobs)
    df.to_csv(csv_path, index=False)
    meta_path.write_text(json.dumps(meta))
    return df
