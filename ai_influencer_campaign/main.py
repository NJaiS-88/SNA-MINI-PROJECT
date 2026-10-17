"""
main.py -- orchestrates the whole project.

    python main.py                 quick demo (fewer simulations)      ~minutes
    python main.py --full          final-report settings (more simulations)
    python main.py --synthetic     smoke-test on a synthetic graph (NOT real results)
    python main.py --force         ignore cached features / labels

WHAT HAPPENS WHEN YOU RUN IT (2-minute version is in README, section "Viva"):
 load graph -> SNA stats -> communities -> features -> traditional rankings ->
 calibrate p -> Monte-Carlo IC influence labels -> train/test ML -> AI ranking ->
 greedy community-aware seeds -> fair IC/LT campaign experiments -> plots, CSVs, findings.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src import (campaign_optimizer, community_detection, data_loader, diffusion, evaluation,
                 feature_engineering, influencer_selection, label_generation, ml_model,
                 network_analysis, visualization as viz)

# =============================== USER CONFIGURATION ===============================
QUICK_MODE = True                 # True: fast demo. False (or --full): final-report quality
RANDOM_STATE = 42                 # master seed (splits, Louvain, sampling, layout)
SEED_BUDGET = 10                  # k: number of seed influencers in the central experiment
PROPAGATION_PROBABILITY = None    # None -> calibrated automatically; or fix e.g. 0.02
TEST_SIZE = 0.2                   # 80/20 train/test split of labelled nodes
N_LABEL_SIMULATIONS = None        # IC runs per candidate node (None -> by mode: 20 quick / 50 full)
N_EVAL_SIMULATIONS = None         # IC/LT runs per strategy      (None -> by mode: 60 quick / 200 full)
VIRAL_THRESHOLD_PCT = 10.0        # "viral" = cascade reaches >= this % of the network
K_LIST = [1, 3, 5, 10]            # budget experiment
ALPHA, BETA, GAMMA = 0.6, 0.25, 0.5   # optimiser weights: influence / community bonus / overlap penalty
POOL_SIZE = 300                   # optimiser considers the top-POOL_SIZE AI-ranked nodes
# Independent random-number streams (label generation and evaluation NEVER share streams)
CAL_SEED_BASE, LABEL_SEED_BASE, RANDOM_SEED_BASE = 3_000_017, 1_000_003, 7_000_019
TUNE_SEED_BASE, EVAL_SEED_BASE = 5_000_011, 9_000_001
# ==================================================================================

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "facebook_combined.txt"
STRATS_BUDGET = ["Random", "Degree", "PageRank", "AI", "AI + Community"]
STRATS_PROB = ["Random", "Degree", "PageRank", "AI", "AI + Community"]


def mode_settings(quick: bool) -> dict:
    return dict(
        N_CANDIDATES=500 if quick else 1500,
        N_LABEL_SIMULATIONS=N_LABEL_SIMULATIONS or (20 if quick else 50),
        N_EVAL_SIMULATIONS=N_EVAL_SIMULATIONS or (60 if quick else 200),
        BETWEENNESS_K=300 if quick else None,            # None = exact betweenness
        PATH_SAMPLE=300 if quick else None,              # None = exact path statistics
        P_GRID_EXP=[0.01, 0.02, 0.05] if quick else [0.005, 0.01, 0.02, 0.03, 0.05, 0.10],
        LABEL_P_GRID=[0.005, 0.01, 0.02, 0.03, 0.05, 0.08],
        N_CAL_NODES=40 if quick else 80, N_CAL_SIMS=10 if quick else 20,
        CURVE_RUNS=20 if quick else 50,
    )


def banner(msg: str) -> None:
    print(f"\n{'#' * 70}\n# {msg}\n{'#' * 70}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="final-report simulation counts")
    ap.add_argument("--quick", action="store_true", help="force quick mode")
    ap.add_argument("--synthetic", action="store_true", help="smoke-test graph (NOT real results)")
    ap.add_argument("--force", action="store_true", help="recompute cached features/labels")
    ap.add_argument("--synthetic-n", type=int, default=2000, help="size of the synthetic smoke-test graph")
    ap.add_argument("--n-jobs", type=int, default=-1, help="parallel workers (-1 = all cores)")
    args = ap.parse_args()
    quick = (QUICK_MODE and not args.full) or args.quick
    S = mode_settings(quick)
    cfg = dict(QUICK_MODE=quick, RANDOM_STATE=RANDOM_STATE, SEED_BUDGET=SEED_BUDGET, TEST_SIZE=TEST_SIZE,
               VIRAL_THRESHOLD_PCT=VIRAL_THRESHOLD_PCT, K_LIST=K_LIST, ALPHA=ALPHA, BETA=BETA, GAMMA=GAMMA,
               POOL_SIZE=POOL_SIZE, synthetic=args.synthetic, **S)
    t_start = time.time()

    out = ROOT / ("outputs_synthetic_demo" if args.synthetic else "outputs")
    res_dir, plot_dir, model_dir = out / "results", out / "plots", out / "models"
    for d in (res_dir, plot_dir, model_dir):
        d.mkdir(parents=True, exist_ok=True)
    P = lambda name: plot_dir / name
    R = lambda name: res_dir / name
    jobs = args.n_jobs
    print(f"Mode: {'QUICK' if quick else 'FULL'} | synthetic={args.synthetic} | outputs -> {out.name}/")
    if args.synthetic:
        print("!! SYNTHETIC GRAPH: pipeline smoke-test only. Do NOT report these numbers. !!")

    # ------------------------------------------------------------ 1. load + preprocess
    banner("1. DATA LOADING & PREPROCESSING")
    if args.synthetic:
        G, info = data_loader.generate_synthetic_graph(args.synthetic_n, RANDOM_STATE)
        dataset_name = "SYNTHETIC smoke-test graph (not real data)"
    else:
        G, info = data_loader.load_graph(DATA_PATH)
        dataset_name = "SNAP ego-Facebook (facebook_combined.txt), undirected"
        if info["prep"]["ids_relabelled"]:
            pd.DataFrame({"node_id": range(G.number_of_nodes()), "original_id": info["prep"]["orig_ids"]}
                         ).to_csv(R("node_id_mapping.csv"), index=False)
    n = G.number_of_nodes()
    K_USED = [k for k in K_LIST if k <= n]
    if SEED_BUDGET > n:
        raise ValueError(f"SEED_BUDGET={SEED_BUDGET} exceeds number of nodes {n}")
    csr = diffusion.build_csr(G)

    # ------------------------------------------------------------ 2. network analysis
    banner("2. NETWORK STRUCTURE ANALYSIS")
    net_stats, deg = network_analysis.compute_network_statistics(G, S["PATH_SAMPLE"], RANDOM_STATE)
    print(net_stats.to_string(index=False))
    interp = network_analysis.interpret_statistics(net_stats)
    print("Interpretation:"); [print("  -", t) for t in interp]
    net_stats.to_csv(R("network_statistics.csv"), index=False)
    viz.plot_degree_distribution(deg, P("02_degree_distribution.png"), P("02b_degree_distribution_loglog.png"))

    # ------------------------------------------------------------ 3. communities
    banner("3. COMMUNITY DETECTION")
    part, comm_method, modularity = community_detection.detect_communities(G, RANDOM_STATE)
    n_comm = len(set(part.values()))
    print(f"  {comm_method}: {n_comm} communities, modularity Q = {modularity:.4f}")

    # ------------------------------------------------------------ 4. features
    banner("4. SNA FEATURE ENGINEERING")
    feats = feature_engineering.compute_features(G, part, S["BETWEENNESS_K"], RANDOM_STATE, res_dir, args.force)
    feature_engineering.assert_no_leakage(feature_engineering.ML_FEATURES)
    feats.to_csv(R("node_features.csv"), index=False)
    pr_dict = dict(zip(feats.node_id, feats.pagerank))
    comm_stats = community_detection.community_statistics(G, part, pr_dict)
    print(comm_stats.head(8).to_string(index=False))

    # ------------------------------------------------------------ 5. traditional rankings
    banner("5. TRADITIONAL INFLUENCER DETECTION")
    trad, tops, spear, overlap = influencer_selection.traditional_rankings(feats, 10)
    trad.to_csv(R("traditional_influencers.csv"), index=False)
    spear.to_csv(R("centrality_spearman.csv")); overlap.to_csv(R("centrality_top10_overlap.csv"))
    trad_table = influencer_selection.traditional_table(feats, tops)
    print(trad_table.round(5).to_string(index=False))
    print("Spearman correlation between centralities:\n", spear.round(2).to_string())

    # ------------------------------------------------------------ 6. calibration + labels
    banner("6. DIFFUSION-BASED INFLUENCE LABELS (Independent Cascade, Monte Carlo)")
    if PROPAGATION_PROBABILITY is None:
        p_label, cal = label_generation.calibrate_probability(
            csr, S["LABEL_P_GRID"], S["N_CAL_NODES"], S["N_CAL_SIMS"], RANDOM_STATE, CAL_SEED_BASE, jobs)
        cal.to_csv(R("probability_calibration.csv"), index=False)
        print(cal.round(4).to_string(index=False))
        viz.plot_calibration(cal, P("00_probability_calibration.png"))
    else:
        p_label = float(PROPAGATION_PROBABILITY)
    p_eval = p_label if PROPAGATION_PROBABILITY is None else float(PROPAGATION_PROBABILITY)
    cands = label_generation.select_candidates(deg, S["N_CANDIDATES"], RANDOM_STATE)
    labels = label_generation.generate_influence_labels(
        csr, cands, p_label, S["N_LABEL_SIMULATIONS"], LABEL_SEED_BASE, res_dir, jobs, args.force,
        extra_meta={"random_state": RANDOM_STATE, "synthetic": args.synthetic})
    print(labels.describe().round(3).to_string())

    # ------------------------------------------------------------ 7. ML
    banner("7. MACHINE LEARNING: SNA FEATURES -> DIFFUSION POTENTIAL")
    ds = ml_model.build_ml_dataset(feats, labels)
    train, test = ml_model.split_data(ds, TEST_SIZE, RANDOM_STATE)
    print(f"  labelled nodes: {len(ds)} | train: {len(train)} | held-out test: {len(test)}")
    pd.concat([train.assign(split="train"), test.assign(split="test")])[["node_id", "split", "mean_spread"]
        ].to_csv(R("train_test_split.csv"), index=False)
    ml_table, fitted, preds, topk = ml_model.train_and_evaluate(train, test, RANDOM_STATE, quick)
    best = ml_model.select_best_model(ml_table)
    print(ml_table.round(4).to_string(index=False)); print(f"  Best model (avg rank over metrics): {best}")
    ml_table.to_csv(R("ml_model_evaluation.csv"), index=False); topk.to_csv(R("ml_topk_overlap.csv"), index=False)
    pd.DataFrame({"node_id": test.node_id, "actual": test.mean_spread, "predicted": preds[best]}
                 ).to_csv(R("ml_test_predictions.csv"), index=False)
    fi = ml_model.feature_importance(fitted, test, best, RANDOM_STATE)
    fi.to_csv(R("feature_importance.csv"), index=False)
    print(fi[fi.type == "permutation(test)"].sort_values("importance", ascending=False).head(5).to_string(index=False))
    ml_model.save_model(fitted[best], best, model_dir / "best_model.joblib")
    bundle = ml_model.load_model(model_dir / "best_model.joblib")     # round-trip proves the saved model works
    predictions = ml_model.predict_all_nodes(bundle, feats)
    predictions.to_csv(R("predicted_influence_all_nodes.csv"), index=False)
    ai_ranked = predictions.sort_values(["predicted_influence_score", "node_id"], ascending=[False, True]).reset_index(drop=True)
    ai_top = ai_ranked.head(10).copy()
    ai_top.insert(0, "rank", range(1, 11)); ai_top["community_id"] = ai_top.node_id.map(part)
    ai_top.to_csv(R("ai_influencers.csv"), index=False)
    print(ai_top.round(3).to_string(index=False))
    ai10 = set(ai_top.node_id)
    ov = pd.DataFrame([{"AI_vs": m, "overlap_top10": len(ai10 & set(tops[c])) / 10}
                       for m, c in (("Degree", "degree_centrality"), ("PageRank", "pagerank"),
                                    ("Betweenness", "betweenness"), ("Closeness", "closeness"),
                                    ("Eigenvector", "eigenvector"))])
    ov.to_csv(R("ai_vs_traditional_overlap.csv"), index=False); print(ov.to_string(index=False))
    comm_stats = community_detection.add_ml_top_node(comm_stats, part, predictions.set_index("node_id").predicted_influence_score)
    comm_stats.to_csv(R("community_statistics.csv"), index=False)

    # ------------------------------------------------------------ 8. campaign experiments
    banner("8. CAMPAIGN EXPERIMENTS (fresh Monte Carlo streams, equal budget)")
    NS = S["N_EVAL_SIMULATIONS"]
    strategies, trace = influencer_selection.build_all_strategies(
        G, feats, predictions, part, SEED_BUDGET, NS, RANDOM_SEED_BASE + SEED_BUDGET, ALPHA, BETA, GAMMA, POOL_SIZE)
    trace.to_csv(R("optimizer_trace.csv"), index=False)
    pd.DataFrame([{"strategy": s, "seeds": " ".join(map(str, sets[0])) if len(sets) == 1 else "(new random set per run)"}
                  for s, sets in strategies.items()]).to_csv(R("seed_sets.csv"), index=False)
    print(trace.to_string(index=False))

    # Experiment 1 (+2): all six strategies, IC, k = SEED_BUDGET
    print(f"\n[Exp 1/2] six strategies, IC, k={SEED_BUDGET}, p={p_eval}, {NS} runs each")
    camp, runs, curves = evaluation.evaluate_strategies(csr, strategies, p_eval, NS, EVAL_SEED_BASE, "IC", jobs,
                                                       VIRAL_THRESHOLD_PCT, part, S["CURVE_RUNS"], SEED_BUDGET)
    camp.to_csv(R("campaign_results.csv"), index=False)
    show = ["Strategy", "k", "p", "mean_spread", "std_spread", "mean_coverage", "mean_duration", "communities_represented"]
    print(camp[show].round(3).to_string(index=False))
    impr = evaluation.improvement_table(camp); impr.to_csv(R("improvement_over_baseline.csv"), index=False)
    print(impr.round(3).to_string(index=False))
    tests = evaluation.paired_tests(runs); tests.to_csv(R("statistical_tests.csv"), index=False)
    print(tests.round(4).to_string(index=False))
    opt = camp[camp.Strategy.isin(["AI", "AI + Community"])][["Strategy", "mean_coverage", "std_coverage", "mean_spread",
                                                              "mean_duration", "communities_represented"]].copy()
    opt["Improvement_over_PageRank_pct"] = [evaluation.pct_improvement(v, camp.set_index("Strategy").mean_coverage["PageRank"]) for v in opt.mean_coverage]
    opt.to_csv(R("optimization_comparison.csv"), index=False)
    viral = camp[["Strategy", "viral_rate", "mean_peak_new", "mean_time_to_peak", "mean_coverage", "mean_growth_rate",
                  "mean_duration", "viral_rate_5pct", "viral_rate_10pct", "viral_rate_20pct"]]
    viral.to_csv(R("viral_campaign_results.csv"), index=False)
    pooled = pd.concat(runs.values()).coverage
    print(f"\nViral threshold = {VIRAL_THRESHOLD_PCT}% coverage. Pooled coverage percentiles (25/50/75/90): "
          f"{np.percentile(pooled, [25, 50, 75, 90]).round(2)}")
    if viral.viral_rate.max() == 0 or viral.viral_rate.min() == 1:
        print("  [warn] threshold does not discriminate between strategies -> adjust VIRAL_THRESHOLD_PCT.")

    # Experiment 3: budget
    print("\n[Exp 3] campaign budget")
    bud_rows = []
    for k in K_USED:
        st, _ = influencer_selection.build_all_strategies(G, feats, predictions, part, k, NS, RANDOM_SEED_BASE + k,
                                                          ALPHA, BETA, GAMMA, POOL_SIZE)
        st = {s: st[s] for s in STRATS_BUDGET}
        r, _, _ = evaluation.evaluate_strategies(csr, st, p_eval, NS, EVAL_SEED_BASE, "IC", jobs, VIRAL_THRESHOLD_PCT, part, 0, k)
        bud_rows.append(r)
    bud = pd.concat(bud_rows, ignore_index=True); bud.to_csv(R("budget_experiment.csv"), index=False)
    print(bud.pivot(index="k", columns="Strategy", values="mean_coverage").round(2).to_string())

    # Experiment 4: propagation probability
    print("\n[Exp 4] propagation probability")
    st4 = {s: strategies[s] for s in STRATS_PROB}
    pr_rows = [evaluation.evaluate_strategies(csr, st4, p, NS, EVAL_SEED_BASE, "IC", jobs, VIRAL_THRESHOLD_PCT, part, 0, SEED_BUDGET)[0]
               for p in S["P_GRID_EXP"]]
    prob = pd.concat(pr_rows, ignore_index=True); prob.to_csv(R("probability_experiment.csv"), index=False)
    print(prob.pivot(index="p", columns="Strategy", values="mean_coverage").round(2).to_string())

    # Experiment 5: IC vs LT
    print("\n[Exp 5] IC vs LT")
    lt, lt_runs, _ = evaluation.evaluate_strategies(csr, strategies, None, NS, EVAL_SEED_BASE, "LT", jobs,
                                                    VIRAL_THRESHOLD_PCT, part, 0, SEED_BUDGET)
    ltic = pd.concat([camp.assign(model="IC"), lt], ignore_index=True); ltic.to_csv(R("ic_vs_lt.csv"), index=False)
    piv = ltic.pivot(index="Strategy", columns="model", values="mean_coverage")
    rho = spearmanr(piv.IC, piv.LT)[0]
    print(piv.round(2).to_string(), f"\n  Spearman(IC ranking, LT ranking) = {rho:.2f}")
    lt_tests = evaluation.paired_tests(lt_runs); lt_tests.to_csv(R("statistical_tests_LT.csv"), index=False)

    # Optimiser parameter sensitivity (separate tuning streams -> no tuning on the evaluation runs)
    print("\n[Extra] optimiser weight sensitivity (independent streams; defaults were fixed a priori)")
    sens = []
    for a, b, g in [(ALPHA, BETA, GAMMA), (1.0, 0.0, 0.0), (0.6, 0.4, 0.5), (0.6, 0.25, 1.0), (0.6, 0.25, 0.0), (0.4, 0.4, 0.5)]:
        seeds_, _ = influencer_selection.ai_community_seeds(G, predictions, part, SEED_BUDGET, a, b, g, POOL_SIZE)
        r, _, _ = evaluation.evaluate_strategies(csr, {"AI + Community": [seeds_]}, p_eval, max(20, NS // 2), TUNE_SEED_BASE,
                                                 "IC", jobs, VIRAL_THRESHOLD_PCT, part, 0, SEED_BUDGET)
        sens.append({"alpha": a, "beta": b, "gamma": g, "mean_coverage": r.mean_coverage[0],
                     "communities_represented": r.communities_represented[0]})
    sens = pd.DataFrame(sens); sens.to_csv(R("parameter_sensitivity.csv"), index=False); print(sens.round(3).to_string(index=False))

    # ------------------------------------------------------------ 9. plots
    banner("9. VISUALISATIONS")
    pos = viz.compute_layout(G, RANDOM_STATE, res_dir / "layout.npy", quick)
    viz.plot_network_overview(G, pos, P("01_network_overview.png"))
    viz.plot_centrality_comparison(spear, overlap, P("03_centrality_comparison.png"))
    viz.plot_top_nodes(G, pos, tops["degree_centrality"], "Top-10 Degree influencers", P("04_top_degree.png"), "#1f77b4")
    viz.plot_top_nodes(G, pos, tops["pagerank"], "Top-10 PageRank influencers", P("05_top_pagerank.png"), "#2ca02c")
    viz.plot_top_nodes(G, pos, tops["betweenness"], "Top-10 Betweenness influencers", P("06_top_betweenness.png"), "#9467bd")
    viz.plot_communities(G, pos, part, comm_method, modularity, P("07_communities.png"))
    viz.plot_corr_matrix(feats, feature_engineering.ML_FEATURES, P("08_feature_correlation.png"))
    bm = ml_table.set_index("Model").loc[best]
    viz.plot_actual_vs_pred(test.mean_spread.to_numpy(), preds[best], best, bm.R2, bm.Spearman, P("09_actual_vs_predicted.png"))
    viz.plot_model_comparison(ml_table, P("10_model_comparison.png"))
    viz.plot_feature_importance(fi, best, P("11_feature_importance.png"))
    viz.plot_top_ai(ai_top, P("12_top_ai_influencers.png"))
    viz.plot_ai_locations(G, pos, part, list(ai_top.node_id), P("13_ai_influencer_communities.png"))
    viz.plot_seed_strategies(G, pos, strategies, P("14_campaign_seed_strategies.png"))
    # one simulated cascade of the proposed strategy, same RNG stream as evaluation run 0
    sd = strategies["AI + Community"][0]
    res0 = diffusion.independent_cascade(csr, sd, p_eval, rng=np.random.default_rng([EVAL_SEED_BASE, 0]))
    T = res0["n_steps"]; at = res0["activation_time"]
    viz.plot_diffusion_state(G, pos, sd, at, 0, P("15_ic_t0.png"), "IC diffusion (AI + Community seeds)")
    viz.plot_diffusion_state(G, pos, sd, at, max(1, T // 2), P("16_ic_intermediate.png"), "IC diffusion (AI + Community seeds)")
    viz.plot_diffusion_state(G, pos, sd, at, T, P("17_ic_final.png"), "IC diffusion (AI + Community seeds)")
    viz.plot_cascade_curves(curves, P("18_cumulative_spread.png"), P("19_new_activations.png"), n)
    viz.plot_strategy_bar(camp, P("20_strategy_vs_coverage.png"), f"Strategy vs mean coverage (IC, k={SEED_BUDGET}, p={p_eval}; 95% CI)")
    viz.plot_lines(bud, "k", "Number of seed influencers (campaign budget)", P("21_budget_vs_coverage.png"), "Campaign budget vs coverage")
    viz.plot_lines(prob, "p", "Propagation probability p", P("22_probability_vs_coverage.png"), "Propagation probability vs coverage")
    viz.plot_ai_vs_comm(camp, P("23_ai_vs_ai_community.png"))
    viz.plot_ic_vs_lt(ltic, P("24_ic_vs_lt.png"))
    viz.plot_viral(viral, VIRAL_THRESHOLD_PCT, P("25_viral_analysis.png"))
    print(f"  plots saved in {plot_dir}")

    # ------------------------------------------------------------ 10. findings + config
    banner("10. FINAL FINDINGS")
    text = evaluation.write_final_findings(R("final_findings.md"), dict(
        net_stats=net_stats, interpretation=interp, trad_tops=tops, ml_table=ml_table, best_model=best, topk=topk,
        feat_imp=fi, campaign=camp, tests=tests, budget=bud, prob=prob, ltic=ltic, rank_corr=rho, cfg=cfg,
        p_eval=p_eval, p_label=p_label, dataset_name=dataset_name, comm_method=comm_method, n_comm=n_comm,
        modularity=modularity))
    print(text)
    cfg.update(p_label=p_label, p_eval=p_eval, n_nodes=n, n_edges=G.number_of_edges(), dataset=dataset_name,
               runtime_seconds=round(time.time() - t_start, 1))
    R("run_config.json").write_text(json.dumps(cfg, indent=2, default=str))
    print(f"\nDone in {time.time() - t_start:.0f}s. Results: {res_dir}  Plots: {plot_dir}")


if __name__ == "__main__":
    main()
