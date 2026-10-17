# AI-Based Influencer Detection and Viral Campaign Optimization Using Social Network Analysis

## Overview
A Social Network Analysis (SNA) mini-project. It builds a graph from a public social network, finds influential users with
classical centralities, **simulates** information diffusion (Independent Cascade and Linear Threshold), trains an ML model that
predicts a user's *simulation-based influencer potential* from SNA features, selects campaign seeds with a
**greedy community-aware campaign seed optimization**, and compares everything under equal budget.

> **Important:** the dataset is a structural graph only. It contains **no real campaign cascades, reshares or influencer labels**.
> All diffusion is simulated; all "influence" values are *simulation-based influencer potential*; all campaigns are
> *simulated viral campaign optimization*. Nothing here claims to be real campaign data.

## Problem Statement
Highly central users are not automatically the users who generate the greatest diffusion, and picking several top-ranked users
can waste budget on overlapping audiences.

## Objective
Estimate diffusion potential per user, rank influencers with ML, choose a non-redundant seed set, and test whether this beats
traditional centrality-based selection.

## Research Questions
1. Can ML on SNA features identify high-diffusion influencers and beat centrality-based campaigns? 2. Which features matter?
3. Does ML beat Degree/PageRank? 4. Do multiple top-ranked users cause redundant reach? 5. Does community-aware selection help?
6. Effect of propagation probability. 7. Effect of budget k. 8. Which strategy has the highest coverage at equal budget?

**Hypothesis (tested, not assumed):** combining SNA features with ML identifies users with more diffusion potential than single
centralities, and community-aware seed optimization further improves coverage by reducing redundancy.

## Dataset
SNAP **ego-Facebook** (`facebook_combined.txt`), https://snap.stanford.edu/data/egonets-Facebook.html
4,039 nodes, 88,234 edges, **undirected**. Node = anonymised user (integer ID, no identities); edge = friendship.
Suitable because it has realistic hubs, dense clusters and short paths, which drive diffusion.

## Dataset Setup
```
python download_dataset.py
```
(or download `facebook_combined.txt.gz` from the SNAP page above, un-gzip it, and put it at `data/facebook_combined.txt`).
Paths are relative to the project folder.

## Architecture
Graph -> network analysis -> communities -> SNA features -> (traditional rankings | IC Monte Carlo labels) -> ML ->
AI influencers -> community-aware optimisation -> IC/LT campaign evaluation -> viral analysis -> plots/CSVs/findings.

## Technologies Used
Python 3.10+, NetworkX, Pandas, NumPy, Scikit-learn, SciPy, Matplotlib, python-louvain, joblib (parallel Monte Carlo).

## Project Structure
```
main.py  download_dataset.py  requirements.txt  README.md
data/    outputs/{plots,results,models}/
src/ data_loader, network_analysis, community_detection, feature_engineering, diffusion, label_generation,
     ml_model, influencer_selection, campaign_optimizer, evaluation, visualization
```

## Installation
```
python -m venv venv
venv\Scripts\activate        # Windows   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```

## Running the Project
```
python main.py            # QUICK mode (demo, a few minutes)
python main.py --full     # final-report settings (more simulations, exact betweenness; slower)
python main.py --force    # ignore cached features/labels
python main.py --synthetic  # smoke test on a synthetic graph; writes to outputs_synthetic_demo/. NOT real results
```
Settings (`QUICK_MODE`, `RANDOM_STATE`, `SEED_BUDGET`, `PROPAGATION_PROBABILITY`, `N_LABEL_SIMULATIONS`, `N_EVAL_SIMULATIONS`,
viral threshold, optimiser weights) are at the top of `main.py`. Use `--full` for the numbers in your report.

## Social Network Analysis
Nodes, edges, density, degree stats, clustering, components, average path length and diameter (largest component; sampled in
quick mode and labelled as estimate), assortativity, degree histogram + log-log plot. Interpretation text is generated from the numbers.

## SNA Feature Engineering
degree, degree_centrality, betweenness, closeness, pagerank, eigenvector, clustering, avg_neighbor_degree, core_number,
community_size (community_id kept in the CSV for analysis but **not** used as an ML input). Quick mode uses sampled betweenness
(300 sources); full mode is exact.

## Information Diffusion
`diffusion.py`, from scratch. **IC**: seeds active at t=0; each newly active node gets one try per inactive neighbour, success
probability p; stop when nothing new activates. **LT**: each node has a random threshold in [0,1]; each neighbour weighs 1/deg;
a node activates when active-neighbour weight >= threshold. IC is the primary model; LT is a robustness check.

## Influence Label Generation
Label(v) = mean spread over R IC runs with v as the only seed (Monte Carlo). p_label is **calibrated** (pilot runs avoid
"everything dies" and "everything activates" and maximise label reliability); results in `probability_calibration.csv`.
Labels are computed for a degree-stratified sample of candidate nodes and cached.

## Machine Learning
Linear Regression, Random Forest, Gradient Boosting. 80/20 node split, fixed seed, trained on train labels only, scored on
held-out test labels (MAE, RMSE, R², Spearman, Top-K overlap for K=5,10,20), plus single-feature (degree/PageRank/betweenness)
ranking baselines on the same test nodes. The best model is chosen by average rank over metrics (not assumed). Feature
importance: impurity (tree models) and permutation importance on the test set.

## AI Influencer Detection
The saved model (`outputs/models/best_model.joblib`) predicts influence for all nodes; top 10 -> `ai_influencers.csv`.
(Nodes used for training receive in-sample predictions; the honest accuracy numbers are the held-out ones.)

## Community Detection
Louvain (fallback: greedy modularity). Reports number of communities, modularity, sizes, top-degree / top-PageRank / top-ML node per community.

## Campaign Optimization
**Greedy community-aware campaign seed optimization**: score(v) = alpha*norm_predicted_influence + beta*new_community_bonus
- gamma*max Jaccard overlap with chosen seeds. Defaults alpha=0.6, beta=0.25, gamma=0.5, candidate pool = top 300 AI nodes. Heuristic only; no
claim of global optimality. `parameter_sensitivity.csv` tests other weights on separate random streams.

## Experiments
Fair rules: same k, p, graph and runs for all strategies; paired random streams (run i uses the same stream for each strategy);
Random draws a fresh seed set every run. **Evaluation streams are different from label streams** (no circular evaluation).
1. Random/Degree/PageRank/Betweenness/AI/AI+Community (IC). 2. AI vs AI+Community (+communities represented).
3. Budget k in {1,3,5,10}. 4. Propagation probability grid. 5. IC vs LT (compare *rankings*, not raw parameters).
6. ML quality (actual vs predicted, Top-K overlap). Plus viral analysis and paired t-test / Wilcoxon (approximate guidance).

## Evaluation Metrics
Mean/std/median/min/max spread, mean coverage % (+95% CI), cascade duration, peak new activations, time to peak, growth rate,
viral rate (coverage >= `VIRAL_THRESHOLD_PCT`, default 10%; the program prints coverage percentiles and warns if the threshold
does not discriminate - adjust it after looking at your results), improvement over baseline (% and percentage points).

## Generated Visualizations
`outputs/plots/`: 00 calibration, 01 network, 02/02b degree distribution, 03 centrality comparison, 04-06 top Degree/PageRank/Betweenness,
07 communities, 08 feature correlation, 09 actual vs predicted, 10 model comparison, 11 feature importance, 12 top AI influencers,
13 AI influencer communities, 14 seed strategies, 15-17 IC at t=0/mid/final, 18 cumulative spread, 19 new activations,
20 strategy vs coverage, 21 budget vs coverage, 22 probability vs coverage, 23 AI vs AI+Community, 24 IC vs LT, 25 viral analysis.

## Results
Nothing is pre-filled. After running, see `outputs/results/`: `network_statistics.csv`, `node_features.csv`, `traditional_influencers.csv`,
`centrality_spearman.csv`, `centrality_top10_overlap.csv`, `community_statistics.csv`, `probability_calibration.csv`, `influence_labels.csv`,
`train_test_split.csv`, `ml_model_evaluation.csv`, `ml_topk_overlap.csv`, `ml_test_predictions.csv`, `feature_importance.csv`,
`predicted_influence_all_nodes.csv`, `ai_influencers.csv`, `ai_vs_traditional_overlap.csv`, `optimizer_trace.csv`, `seed_sets.csv`,
`campaign_results.csv`, `improvement_over_baseline.csv`, `statistical_tests.csv`, `statistical_tests_LT.csv`, `optimization_comparison.csv`,
`viral_campaign_results.csv`, `budget_experiment.csv`, `probability_experiment.csv`, `ic_vs_lt.csv`, `parameter_sensitivity.csv`,
`run_config.json`, and **`final_findings.md`** (auto-written from the actual numbers; reports negative results too).

## Contribution
The project integrates Social Network Analysis, simulation-derived influence estimation, machine-learning-based influencer ranking,
and community-aware seed optimization into a unified campaign optimization pipeline. It does not claim a novel algorithm.

## Practical Use
A brand with budget for 5 initial promoters analyses structure, predicts diffusion potential, avoids redundant influencers, spreads seeds
across communities and estimates reach - useful for awareness campaigns, product launches, public-information campaigns and content promotion.
Similar analysis could help *study* misinformation spread; this project makes no claim of controlling it.

## Limitations
1. Dataset has structure but no real campaign cascades. 2. Diffusion is simulated. 3. IC/LT simplify human behaviour.
4. A connection does not guarantee exposure. 5. Propagation probabilities are assumptions. 6. Labels come from simulation, not real outcomes.
7. The network is static. 8. The optimiser is greedy, not globally optimal. 9. One network may not generalise to other platforms.
10. Labels are computed on a degree-stratified sample; tree models cannot extrapolate beyond the labelled range. 11. In near-critical regimes
cascades are bimodal (die or go giant), so large k can saturate coverage and hide strategy differences.

## Future Work
Real repost-cascade datasets, temporal networks, dynamic probabilities, user/content features, GNNs, reinforcement learning,
classical influence-maximisation (CELF/greedy), real campaign data, cross-platform analysis.

## Viva: What happens when I run main.py? (2 minutes)
It loads the Facebook graph and cleans it, prints network statistics, detects Louvain communities and computes SNA features per node.
It ranks users by classical centralities. Because there are no real influencer labels, it calibrates a spread probability, then for a
sample of users runs many Independent Cascade simulations with that user as the only seed; the average spread is the user's influence label.
A regression model learns features -> influence on 80% of those users and is tested on the held-out 20%. The best model predicts influence
for everyone, giving AI influencers. A greedy optimiser then picks k seeds balancing predicted influence, new communities and low neighbourhood
overlap. Finally six strategies (Random, Degree, PageRank, Betweenness, AI, AI+Community) are run with the same k, p and fresh random streams;
budget, probability and IC-vs-LT experiments follow. Plots, CSVs and `final_findings.md` are written, reporting whatever actually happened.

## Likely Viva Questions
- **Why simulate labels?** No real influencer ground truth exists; expected spread under IC is a standard definition of influence.
- **Why not train and test on the same nodes?** That inflates accuracy; we hold out test nodes and evaluate campaigns on fresh random streams.
- **Why Spearman?** Influencer selection is a ranking problem.
- **Why exclude community_id?** It is an arbitrary label; treating it as a number would be meaningless. community_size is used instead.
- **IC vs LT?** IC: each edge succeeds independently with p. LT: a node activates once enough of its neighbours (weighted) are active.
- **What if PageRank wins?** We report it; the hypothesis is tested, not assumed.
- **Is the optimiser optimal?** No, it is a greedy heuristic.
- **Betweenness vs PageRank?** Betweenness = bridge on shortest paths; PageRank = linked to by important nodes.
- **Why calibrate p?** Too low and every cascade dies; too high and everything is activated, so labels carry no information.
