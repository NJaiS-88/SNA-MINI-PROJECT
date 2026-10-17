"""
visualization.py -- all plots. Large graphs are drawn with small nodes, a SAMPLE of edges
and labels only for highlighted nodes. The layout is computed once and reused.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from .evaluation import mean_curves

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "axes.grid": True, "grid.alpha": 0.25})
STRAT_COLORS = {"Random": "#9e9e9e", "Degree": "#1f77b4", "PageRank": "#2ca02c", "Betweenness": "#9467bd",
                "AI": "#ff7f0e", "AI + Community": "#d62728"}


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- network drawing helpers
def compute_layout(G: nx.Graph, seed: int, cache: Path | None, quick: bool) -> dict:
    """Layout computed ONCE (and cached) then reused for every network figure."""
    n = G.number_of_nodes()
    if cache is not None and Path(cache).exists():
        arr = np.load(cache)
        if arr.shape == (n, 2):
            return {i: arr[i] for i in range(n)}
    print("  Computing network layout (once) ...")
    pos = nx.spring_layout(G, seed=seed, iterations=25 if quick else 50)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, np.array([pos[i] for i in range(n)]))
    return pos


def _draw_edges(ax, G, pos, max_edges=12000, seed=0):
    edges = list(G.edges())
    if len(edges) > max_edges:
        idx = np.random.default_rng(seed).choice(len(edges), max_edges, replace=False)
        edges = [edges[i] for i in idx]
    segs = [(pos[u], pos[v]) for u, v in edges]
    ax.add_collection(LineCollection(segs, colors="#888888", linewidths=0.2, alpha=0.12))


def _xy(pos, nodes):
    nodes = list(nodes)
    return np.array([pos[v] for v in nodes]) if nodes else np.zeros((0, 2))


def _base(ax, G, pos, title):
    _draw_edges(ax, G, pos)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.autoscale()


def plot_network_overview(G, pos, path):
    fig, ax = plt.subplots(figsize=(8, 8))
    _base(ax, G, pos, f"Social network overview ({G.number_of_nodes()} users, {G.number_of_edges()} ties)")
    xy = _xy(pos, G.nodes()); deg = np.array([G.degree(v) for v in G])
    ax.scatter(xy[:, 0], xy[:, 1], s=2 + 40 * deg / deg.max(), c=np.log1p(deg), cmap="viridis", alpha=0.8, linewidths=0)
    _save(fig, path)


def plot_degree_distribution(deg, path_hist, path_loglog):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(deg, bins=50, color="#4c72b0", edgecolor="white")
    ax.axvline(deg.mean(), color="r", ls="--", label=f"mean = {deg.mean():.1f}")
    ax.axvline(np.median(deg), color="k", ls=":", label=f"median = {np.median(deg):.0f}")
    ax.set(xlabel="Degree", ylabel="Number of users", title="Degree distribution"); ax.legend()
    _save(fig, path_hist)
    vals, counts = np.unique(deg[deg > 0], return_counts=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(vals, counts / counts.sum(), "o", ms=4, color="#c44e52")
    ax.set(xlabel="Degree (log)", ylabel="P(degree) (log)", title="Degree distribution (log-log)")
    _save(fig, path_loglog)


def plot_centrality_comparison(spear, overlap, path):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mat, title, fmt in ((axs[0], spear, "Spearman rank correlation of centralities", "{:.2f}"),
                                (axs[1], overlap, "Top-10 overlap between centralities", "{:.1f}")):
        im = ax.imshow(mat.values, cmap="coolwarm" if ax is axs[0] else "Blues", vmin=0 if ax is axs[1] else -1, vmax=1)
        ax.set_xticks(range(len(mat))); ax.set_yticks(range(len(mat)))
        ax.set_xticklabels(mat.columns, rotation=45, ha="right"); ax.set_yticklabels(mat.index)
        for i in range(len(mat)):
            for j in range(len(mat)):
                ax.text(j, i, fmt.format(mat.values[i, j]), ha="center", va="center", fontsize=8)
        ax.set_title(title); ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, path)


def plot_top_nodes(G, pos, nodes, title, path, color="#d62728"):
    fig, ax = plt.subplots(figsize=(8, 8))
    _base(ax, G, pos, title)
    xy = _xy(pos, G.nodes()); ax.scatter(xy[:, 0], xy[:, 1], s=2, c="#b0b0b0", linewidths=0)
    h = _xy(pos, nodes); ax.scatter(h[:, 0], h[:, 1], s=140, c=color, edgecolors="k", zorder=5)
    for r, v in enumerate(nodes, 1):
        ax.annotate(f"{r}:{v}", pos[v], fontsize=8, xytext=(4, 4), textcoords="offset points", zorder=6)
    _save(fig, path)


def _community_colors(part, top=12):
    sizes = pd.Series(part).value_counts()
    cmap = plt.get_cmap("tab20"); lut = {c: cmap(i % 20) for i, c in enumerate(sizes.index[:top])}
    return lambda v: lut.get(part[v], (0.75, 0.75, 0.75, 1))


def plot_communities(G, pos, part, method, q, path):
    fig, ax = plt.subplots(figsize=(8, 8))
    _base(ax, G, pos, f"{method} communities (n={len(set(part.values()))}, Q={q:.3f}); 12 largest coloured")
    col = _community_colors(part); xy = _xy(pos, G.nodes())
    ax.scatter(xy[:, 0], xy[:, 1], s=4, c=[col(v) for v in G], linewidths=0)
    _save(fig, path)


def plot_corr_matrix(features, cols, path):
    corr = features[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=60, ha="right"); ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("SNA feature correlation matrix (Spearman)"); ax.grid(False); fig.colorbar(im, fraction=0.046)
    _save(fig, path)


# ---------------------------------------------------------------- ML plots
def plot_actual_vs_pred(y, yp, name, r2, rho, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, yp, s=14, alpha=0.6)
    lim = [min(y.min(), yp.min()), max(y.max(), yp.max())]
    ax.plot(lim, lim, "r--", label="perfect prediction")
    ax.set(xlabel="Actual influence (simulated mean spread)", ylabel="Predicted influence",
           title=f"{name}: held-out test nodes\nR²={r2:.3f}, Spearman={rho:.3f}"); ax.legend()
    _save(fig, path)


def plot_model_comparison(table, path):
    t = table[~table.Model.str.startswith("[baseline]")]
    fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    for ax, m in zip(axs, ["MAE", "RMSE", "R2", "Spearman"]):
        ax.bar(t.Model, t[m], color=["#4c72b0", "#55a868", "#c44e52"][:len(t)])
        ax.set_title(m); ax.tick_params(axis="x", rotation=25)
    _save(fig, path)


def plot_feature_importance(fi, best_name, path):
    imp = fi[fi.type == "impurity"]; models = list(imp.model.unique())
    perm = fi[fi.type == "permutation(test)"]
    panels = [(m, imp[imp.model == m], "impurity importance") for m in models] + [(best_name, perm, "permutation importance (test)")]
    fig, axs = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.5))
    axs = np.atleast_1d(axs)
    for ax, (m, d, lab) in zip(axs, panels):
        d = d.sort_values("importance")
        ax.barh(d.feature, d.importance, color="#4c72b0"); ax.set_title(f"{m}\n{lab}"); ax.set_xlabel("Importance")
    _save(fig, path)


def plot_top_ai(ai_top, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([str(n) for n in ai_top.node_id], ai_top.predicted_influence_score, color="#ff7f0e")
    ax.set(xlabel="Node ID", ylabel="Predicted influence (expected spread)", title="Top AI-predicted influencers")
    _save(fig, path)


def plot_ai_locations(G, pos, part, ai_nodes, path):
    fig, axs = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [2, 1]})
    _base(axs[0], G, pos, "Top AI influencers (stars) over communities")
    col = _community_colors(part); xy = _xy(pos, G.nodes())
    axs[0].scatter(xy[:, 0], xy[:, 1], s=3, c=[col(v) for v in G], linewidths=0)
    h = _xy(pos, ai_nodes); axs[0].scatter(h[:, 0], h[:, 1], s=220, marker="*", c="gold", edgecolors="k", zorder=5)
    cnt = pd.Series([part[v] for v in ai_nodes]).value_counts()
    axs[1].bar([str(c) for c in cnt.index], cnt.values, color="gold", edgecolor="k")
    axs[1].set(xlabel="Community ID", ylabel="# AI influencers", title="AI influencers per community")
    _save(fig, path)


# ---------------------------------------------------------------- campaign plots
def plot_seed_strategies(G, pos, strategies, path):
    fig, axs = plt.subplots(2, 3, figsize=(16, 10.5)); xy = _xy(pos, G.nodes())
    for ax, (name, sets) in zip(axs.ravel(), strategies.items()):
        _base(ax, G, pos, name + (" (one example draw)" if len(sets) > 1 else ""))
        ax.scatter(xy[:, 0], xy[:, 1], s=2, c="#c8c8c8", linewidths=0)
        h = _xy(pos, sets[0]); ax.scatter(h[:, 0], h[:, 1], s=130, marker="*", c=STRAT_COLORS[name], edgecolors="k", zorder=5)
    _save(fig, path)


def plot_diffusion_state(G, pos, seeds, act_time, t, path, title):
    fig, ax = plt.subplots(figsize=(8, 8))
    _base(ax, G, pos, f"{title}: t = {t}  (active = {int((act_time >= 0)[act_time <= t].sum())})")
    xy = _xy(pos, G.nodes()); ax.scatter(xy[:, 0], xy[:, 1], s=2, c="#d0d0d0", linewidths=0)
    act = np.flatnonzero((act_time >= 0) & (act_time <= t))
    ax.scatter(xy[act, 0], xy[act, 1], s=5, c="#e34a33", linewidths=0, zorder=3)
    h = _xy(pos, seeds); ax.scatter(h[:, 0], h[:, 1], s=160, marker="*", c="blue", edgecolors="k", zorder=6)
    _save(fig, path)


def plot_cascade_curves(curves_by_strategy, path_cum, path_new, n_nodes):
    fig1, a1 = plt.subplots(figsize=(8, 5)); fig2, a2 = plt.subplots(figsize=(8, 5))
    for name, cur in curves_by_strategy.items():
        if not cur:
            continue
        new, cum = mean_curves(cur); t = np.arange(len(new))
        a1.plot(t, cum, label=name, color=STRAT_COLORS[name], lw=2); a2.plot(t, new, label=name, color=STRAT_COLORS[name], lw=2)
    a1.set(xlabel="Timestep", ylabel="Cumulative activated nodes (mean over runs)", title="Cumulative spread of simulated campaign cascades")
    a2.set(xlabel="Timestep", ylabel="New activations (mean over runs)", title="New activations per timestep (simulated campaign cascades)")
    a1.legend(); a2.legend(); _save(fig1, path_cum); _save(fig2, path_new)


def plot_strategy_bar(res, path, title):
    order = res.Strategy.tolist()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(order, res.mean_coverage, yerr=res.ci95_coverage, capsize=5, color=[STRAT_COLORS[s] for s in order])
    ax.set(ylabel="Mean campaign coverage (% of network)", title=title); ax.tick_params(axis="x", rotation=20)
    _save(fig, path)


def plot_lines(df, xcol, xlabel, path, title, logx=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    for s, d in df.groupby("Strategy"):
        d = d.sort_values(xcol)
        ax.errorbar(d[xcol], d.mean_coverage, yerr=d.ci95_coverage, marker="o", capsize=3, label=s, color=STRAT_COLORS[s])
    if logx: ax.set_xscale("log")
    ax.set(xlabel=xlabel, ylabel="Mean network coverage (%)", title=title); ax.legend()
    _save(fig, path)


def plot_ai_vs_comm(res, path):
    d = res[res.Strategy.isin(["AI", "AI + Community"])]
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for ax, col, lab in zip(axs, ["mean_coverage", "std_coverage", "communities_represented", "mean_duration"],
                            ["Mean coverage (%)", "Std of coverage (pp)", "Communities represented", "Mean cascade duration"]):
        ax.bar(d.Strategy, d[col], color=[STRAT_COLORS[s] for s in d.Strategy]); ax.set_title(lab)
    _save(fig, path)


def plot_ic_vs_lt(df, path):
    p = df.pivot(index="Strategy", columns="model", values="mean_coverage")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, m in zip(axs, ["IC", "LT"]):
        s = p[m].sort_values(ascending=False)
        ax.bar(s.index, s.values, color=[STRAT_COLORS[x] for x in s.index]); ax.set_title(f"{m}: strategy ranking (best → worst)")
        ax.set_ylabel("Mean coverage (%)"); ax.tick_params(axis="x", rotation=20)
    _save(fig, path)


def plot_viral(viral_df, thr, path):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    axs[0].bar(viral_df.Strategy, viral_df.viral_rate, color=[STRAT_COLORS[s] for s in viral_df.Strategy])
    axs[0].set(ylabel=f"Viral rate (coverage ≥ {thr}%)", ylim=(0, 1.05)); axs[0].tick_params(axis="x", rotation=20)
    axs[1].bar(viral_df.Strategy, viral_df.mean_time_to_peak, color=[STRAT_COLORS[s] for s in viral_df.Strategy])
    axs[1].set(ylabel="Mean time to peak (timesteps)"); axs[1].tick_params(axis="x", rotation=20)
    _save(fig, path)


def plot_calibration(cal, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(cal.p, cal.mean_coverage_pct, "o-", label="mean coverage (%)")
    ax.plot(cal.p, 100 * cal["frac_dead(<=3 nodes)"], "s--", label="% cascades dying (≤3 nodes)")
    ax.plot(cal.p, 100 * cal["frac_giant(>=60%)"], "^--", label="% giant cascades (≥60%)")
    sel = cal[cal.selected]; ax.axvline(sel.p.iloc[0], color="r", ls=":", label=f"selected p = {sel.p.iloc[0]}")
    ax.set(xlabel="Propagation probability p", ylabel="%", title="Calibration of p for label generation"); ax.legend()
    _save(fig, path)
