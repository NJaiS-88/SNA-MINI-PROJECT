"""
evaluation.py -- Phases 14, 34-36, 39 and the final-findings generator.

FAIR COMPARISON RULES implemented here
  * identical k, p, graph and number of Monte Carlo runs for every strategy;
  * COMMON RANDOM SEEDS: simulation i uses RNG stream [EVAL_SEED_BASE, i] for every
    strategy (paired design);
  * these evaluation streams are DIFFERENT from the label-generation streams, so
    the ML target simulations are never reused for campaign evaluation (no circularity).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from .diffusion import CSR, cascade_metrics, independent_cascade, linear_threshold


def _md(df: pd.DataFrame, index: bool = False) -> str:
    """Tiny markdown-table writer (avoids an extra 'tabulate' dependency)."""
    if index:
        df = df.reset_index()
    cols = [str(c) for c in df.columns]
    fmt = lambda v: f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _eval_chunk(csr, idxs, seed_sets, p, base_seed, model, keep_curves):
    rows, curves = [], {}
    for i in idxs:
        seeds = seed_sets[i] if len(seed_sets) > 1 else seed_sets[0]
        rng = np.random.default_rng([base_seed, int(i)])
        res = (independent_cascade(csr, seeds, p, rng=rng) if model == "IC"
               else linear_threshold(csr, seeds, rng=rng))
        m = cascade_metrics(res, len(seeds))
        m["sim"] = int(i)
        rows.append(m)
        if i < keep_curves:
            curves[int(i)] = res["new_per_step"]
    return rows, curves


def run_batch(csr: CSR, seed_sets: list[list[int]], p: float | None, n_sims: int, base_seed: int,
              model: str = "IC", n_jobs: int = 1, keep_curves: int = 0):
    """Run n_sims Monte Carlo cascades. Returns (DataFrame of per-run metrics, curves dict)."""
    if model not in ("IC", "LT"):
        raise ValueError("model must be 'IC' or 'LT'")
    if len(seed_sets) not in (1, n_sims):
        raise ValueError("seed_sets must contain 1 set (fixed) or n_sims sets (per-run).")
    n_jobs = n_jobs if n_jobs > 0 else (os.cpu_count() or 1)
    idx = np.arange(n_sims)
    chunks = np.array_split(idx, max(1, min(n_sims, n_jobs * 4)))
    if n_jobs == 1:
        parts = [_eval_chunk(csr, c, seed_sets, p, base_seed, model, keep_curves) for c in chunks]
    else:
        parts = Parallel(n_jobs=n_jobs)(delayed(_eval_chunk)(csr, c, seed_sets, p, base_seed, model,
                                                             keep_curves) for c in chunks)
    rows = [r for pr, _ in parts for r in pr]
    curves = {k: v for _, c in parts for k, v in c.items()}
    return pd.DataFrame(rows).sort_values("sim").reset_index(drop=True), curves


def summarize(df: pd.DataFrame, viral_threshold: float, extra_thresholds=(5, 10, 20)) -> dict:
    """Aggregate per-run metrics of one strategy."""
    s = {
        "mean_spread": df.spread.mean(), "std_spread": df.spread.std(ddof=1),
        "median_spread": df.spread.median(), "min_spread": df.spread.min(), "max_spread": df.spread.max(),
        "mean_coverage": df.coverage.mean(), "std_coverage": df.coverage.std(ddof=1),
        "ci95_coverage": 1.96 * df.coverage.std(ddof=1) / np.sqrt(len(df)),
        "mean_duration": df.duration.mean(),
        "viral_rate": float((df.coverage >= viral_threshold).mean()),
        "mean_peak_new": df.peak_new.mean(), "mean_time_to_peak": df.time_to_peak.mean(),
        "mean_growth_rate": df.growth_rate.mean(),
    }
    for t in extra_thresholds:
        s[f"viral_rate_{t}pct"] = float((df.coverage >= t).mean())
    return s


def evaluate_strategies(csr, strategies: dict, p, n_sims, base_seed, model, n_jobs, viral_threshold,
                        part: dict | None = None, keep_curves: int = 0, k: int | None = None):
    """Evaluate every strategy under identical conditions. Returns (summary DF, per-run dict, curves dict)."""
    rows, runs, curves = [], {}, {}
    for name, sets in strategies.items():
        df, cur = run_batch(csr, sets, p, n_sims, base_seed, model, n_jobs, keep_curves)
        runs[name], curves[name] = df, cur
        row = {"Strategy": name, "k": k if k is not None else len(sets[0]), "model": model,
               "p": p if model == "IC" else np.nan, "n_sims": n_sims,
               **summarize(df, viral_threshold)}
        if part is not None:
            row["communities_represented"] = float(np.mean([len({part[v] for v in s}) for s in sets]))
        rows.append(row)
    return pd.DataFrame(rows), runs, curves


# ------------------------------------------------------------------ improvements & tests
def pct_improvement(new: float, base: float) -> float:
    """Relative improvement in %, NaN if the baseline is zero (no division by zero)."""
    return float("nan") if base == 0 else 100.0 * (new - base) / base


def improvement_table(res: pd.DataFrame) -> pd.DataFrame:
    """Relative (%) and absolute (percentage-point) differences for the key comparisons."""
    c = res.set_index("Strategy").mean_coverage
    pairs = [("AI", "Random"), ("AI", "Degree"), ("AI", "PageRank"), ("AI", "Betweenness"),
             ("AI + Community", "Random"), ("AI + Community", "PageRank"), ("AI + Community", "AI")]
    rows = []
    for a, b in pairs:
        if a in c.index and b in c.index:
            rows.append({"Strategy": a, "Baseline": b, "Strategy_coverage": c[a], "Baseline_coverage": c[b],
                         "Improvement_pct": pct_improvement(c[a], c[b]),
                         "Difference_percentage_points": c[a] - c[b]})
    return pd.DataFrame(rows)


def paired_tests(runs: dict, pairs=None) -> pd.DataFrame:
    """
    Paired comparison of per-run coverage (same RNG stream index => paired design).
    Paired t-test + Wilcoxon signed-rank + 95% CI of the mean difference.
    Note: pairing is by shared random numbers, not an exact coupling of cascades, so the
    p-values are approximate guidance, not proof. 'Random' is excluded from pairing
    (it uses a different seed set every run).
    """
    pairs = pairs or [("AI", "Degree"), ("AI", "PageRank"), ("AI", "Betweenness"), ("AI", "Random"),
                      ("AI + Community", "AI"), ("AI + Community", "PageRank"),
                      ("AI + Community", "Degree"), ("AI + Community", "Random")]
    rows = []
    for a, b in pairs:
        if a not in runs or b not in runs:
            continue
        x, y = runs[a].coverage.to_numpy(), runs[b].coverage.to_numpy()
        d = x - y
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
        if len(d) > 1 and np.allclose(d, 0):
            t_p = w_p = 1.0
        else:
            t_p = float(stats.ttest_rel(x, y).pvalue)
            try:
                w_p = float(stats.wilcoxon(x, y).pvalue)
            except ValueError:
                w_p = 1.0
        rows.append({"A": a, "B": b, "mean_diff_pp": d.mean(), "ci95_low": d.mean() - 1.96 * se,
                     "ci95_high": d.mean() + 1.96 * se, "paired_t_p": t_p, "wilcoxon_p": w_p})
    return pd.DataFrame(rows)


def mean_curves(curves: dict) -> tuple[np.ndarray, np.ndarray]:
    """Average new-activation / cumulative curves over runs (padded with 0 new / last cumulative)."""
    arrs = list(curves.values())
    T = max(len(a) for a in arrs)
    new = np.zeros((len(arrs), T))
    for i, a in enumerate(arrs):
        new[i, :len(a)] = a
    return new.mean(axis=0), np.cumsum(new, axis=1).mean(axis=0)


# ------------------------------------------------------------------ final findings text
def _verdict(diff: float, pval: float | None, name_a: str, name_b: str) -> str:
    sig = pval is not None and pval < 0.05
    if diff > 0:
        return f"{name_a} outperformed {name_b}" + (" (statistically significant)" if sig else " (difference NOT statistically significant)")
    if diff < 0:
        return f"{name_a} performed WORSE than {name_b}" + (" (statistically significant)" if sig else " (difference NOT statistically significant)")
    return f"{name_a} and {name_b} performed identically"


def write_final_findings(path, ctx: dict) -> str:
    """Generate final_findings.md from the ACTUAL results (nothing hard-coded as positive)."""
    ns, trad_tops, ml, best, topk = ctx["net_stats"], ctx["trad_tops"], ctx["ml_table"], ctx["best_model"], ctx["topk"]
    fi, camp, tests = ctx["feat_imp"], ctx["campaign"], ctx["tests"]
    bud, prob, ltic, cfg = ctx["budget"], ctx["prob"], ctx["ltic"], ctx["cfg"]
    c = camp.set_index("Strategy")
    nv = dict(zip(ns.Metric, ns.Value))
    L = [f"# Final Findings (auto-generated from experiment outputs)\n",
         f"Dataset: {ctx['dataset_name']}. All diffusion is SIMULATED; labels are simulation-based influencer "
         f"potential, not real campaign outcomes. Config: k={cfg['SEED_BUDGET']}, p_eval={ctx['p_eval']}, "
         f"p_label={ctx['p_label']}, eval simulations={cfg['N_EVAL_SIMULATIONS']}, label simulations={cfg['N_LABEL_SIMULATIONS']}.\n"]

    L.append("## 1. Network structure")
    L += [f"- {m}: {v}" for m, v in nv.items()] + [f"- {t}" for t in ctx["interpretation"]]
    L.append(f"- Communities ({ctx['comm_method']}): {ctx['n_comm']}, modularity Q = {ctx['modularity']:.3f}.\n")

    L.append("## 2. Influential users per traditional SNA")
    for m, top in trad_tops.items():
        L.append(f"- Top-5 by {m}: {top[:5]}")
    L.append("")

    L.append("## 3. ML prediction accuracy (held-out test nodes)")
    L.append(_md(ml.round(4), index=False))
    bm = ml.set_index("Model").loc[best]
    L.append(f"\nBest model by average rank across metrics: **{best}** (R2={bm.R2:.3f}, Spearman={bm.Spearman:.3f}).")
    tk = topk[topk.Method == best]
    L.append("Top-K overlap of predicted vs simulated influencers (test nodes): " +
             ", ".join(f"K={int(r.K)}: {r.TopK_overlap:.2f}" for r in tk.itertuples()) + ".")
    base = ml[ml.Model.str.startswith("[baseline]")]
    best_base = base.loc[base.Spearman.idxmax()]
    L.append(f"Best single-feature ranking baseline: {best_base.Model} (Spearman={best_base.Spearman:.3f}) "
             f"vs {best} (Spearman={bm.Spearman:.3f}).\n")

    L.append("## 4. Most useful SNA features")
    pm = fi[(fi.type == "permutation(test)")].sort_values("importance", ascending=False).head(3)
    L.append("Permutation importance (test set, best model): " + ", ".join(f"{r.feature} ({r.importance:.3f})" for r in pm.itertuples()))
    imp = fi[(fi.type == "impurity") & (fi.model == "Random Forest")].sort_values("importance", ascending=False).head(3)
    if len(imp):
        L.append("Random Forest impurity importance (top 3): " + ", ".join(f"{r.feature} ({r.importance:.3f})" for r in imp.itertuples()))
    L.append("")

    L.append("## 5. Did AI-selected influencers outperform traditional centralities? (IC, equal budget)")
    L.append(_md(camp[["Strategy", "mean_coverage", "std_coverage", "viral_rate", "mean_duration"]].round(3), index=False))
    tt = tests.set_index(["A", "B"]) if len(tests) else None
    for b in ("Degree", "PageRank", "Betweenness", "Random"):
        if b in c.index and "AI" in c.index:
            pv = tt.loc[("AI", b), "paired_t_p"] if tt is not None and ("AI", b) in tt.index else None
            L.append(f"- {_verdict(c.loc['AI','mean_coverage']-c.loc[b,'mean_coverage'], pv, 'AI', b)}: "
                     f"{c.loc['AI','mean_coverage']:.2f}% vs {c.loc[b,'mean_coverage']:.2f}% coverage.")
    L.append("")

    L.append("## 6. Did community-aware optimisation help?")
    for b in ("AI", "PageRank"):
        if b in c.index and "AI + Community" in c.index:
            pv = tt.loc[("AI + Community", b), "paired_t_p"] if tt is not None and ("AI + Community", b) in tt.index else None
            L.append(f"- {_verdict(c.loc['AI + Community','mean_coverage']-c.loc[b,'mean_coverage'], pv, 'AI + Community', b)}: "
                     f"{c.loc['AI + Community','mean_coverage']:.2f}% vs {c.loc[b,'mean_coverage']:.2f}%.")
    if "communities_represented" in c:
        L.append(f"- Communities represented by seeds: AI={c.loc['AI','communities_represented']:.0f}, "
                 f"AI + Community={c.loc['AI + Community','communities_represented']:.0f}, "
                 f"PageRank={c.loc['PageRank','communities_represented']:.0f}.")
    L.append("")

    L.append("## 7. Effect of campaign budget (coverage % by k)")
    L.append(_md(bud.pivot(index="k", columns="Strategy", values="mean_coverage").round(2), index=True))
    gains = bud[bud.Strategy == "AI + Community"].sort_values("k")
    if len(gains) > 1:
        L.append(f"\nMarginal gain of AI + Community: " + ", ".join(
            f"k={int(a.k)}->{int(b.k)}: {b.mean_coverage-a.mean_coverage:+.2f} pp"
            for a, b in zip(gains.itertuples(), gains.iloc[1:].itertuples())) + ".")
    L.append("")

    L.append("## 8. Effect of propagation probability (coverage % by p)")
    L.append(_md(prob.pivot(index="p", columns="Strategy", values="mean_coverage").round(2), index=True))
    spread = prob.groupby("p").mean_coverage.agg(lambda s: s.max() - s.min())
    L.append("\nGap between best and worst strategy (pp) per p: " + ", ".join(f"p={p}: {g:.2f}" for p, g in spread.items()) + ".\n")

    L.append("## 9. IC vs LT (strategy ranking)")
    L.append(_md(ltic.pivot(index="Strategy", columns="model", values="mean_coverage").round(2), index=True))
    if ctx.get("rank_corr") is not None:
        L.append(f"\nSpearman correlation between IC and LT strategy rankings: {ctx['rank_corr']:.2f}.\n")

    L.append("## 10. Recommended seed-selection strategy")
    best_s = c.mean_coverage.idxmax()
    rec = f"Highest mean IC coverage at k={cfg['SEED_BUDGET']}: **{best_s}** ({c.loc[best_s,'mean_coverage']:.2f}%)."
    if best_s in ("AI", "AI + Community") and tt is not None and ("AI + Community", "PageRank") in tt.index:
        pv = tt.loc[("AI + Community", "PageRank"), "paired_t_p"] if best_s == "AI + Community" else tt.loc[("AI", "PageRank"), "paired_t_p"]
        if pv >= 0.05:
            rec += " However the advantage over PageRank is not statistically significant, so the cheaper PageRank heuristic is a defensible alternative."
    elif best_s not in ("AI", "AI + Community"):
        rec += " A traditional baseline beat the proposed methods in this experiment; the hypothesis is NOT supported for this setting."
    L.append(rec)
    L.append("\n## Hypothesis check")
    def state(a, others):
        out = []
        for b in others:
            if b not in c.index or a not in c.index:
                continue
            d = c.loc[a, "mean_coverage"] - c.loc[b, "mean_coverage"]
            pv = tt.loc[(a, b), "paired_t_p"] if tt is not None and (a, b) in tt.index else 1.0
            out.append("sig. better" if (d > 0 and pv < 0.05) else "higher mean, not significant" if d > 0
                       else "sig. worse" if (d < 0 and pv < 0.05) else "lower/equal mean")
        return out
    s1 = state("AI", ["Degree", "PageRank", "Betweenness"])
    L.append(f"- Hypothesis part 1 (ML combination > single centralities; Degree/PageRank/Betweenness): {s1}")
    L.append(f"- Hypothesis part 2 (community-aware > plain AI top-k): {state('AI + Community', ['AI'])}")
    supported = all(x == "sig. better" for x in s1) and state("AI + Community", ["AI"]) == ["sig. better"]
    L.append(f"- Hypothesis fully supported in this experiment (needs significant improvements): **{supported}**")
    L.append("- Interpret together with the significance tests; results are for simulated cascades on one static network.")
    text = "\n".join(L)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text
