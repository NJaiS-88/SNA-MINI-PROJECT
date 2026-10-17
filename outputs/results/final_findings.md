# Final Findings (auto-generated from experiment outputs)

Dataset: SNAP ego-Facebook (facebook_combined.txt), undirected. All diffusion is SIMULATED; labels are simulation-based influencer potential, not real campaign outcomes. Config: k=10, p_eval=0.05, p_label=0.05, eval simulations=200, label simulations=50.

## 1. Network structure
- Nodes: 4039.0
- Edges: 88234.0
- Network density: 0.01082
- Average degree: 43.691
- Minimum degree: 1.0
- Maximum degree: 1045.0
- Median degree: 25.0
- Average clustering coefficient: 0.6055
- Global transitivity: 0.5192
- Degree assortativity: 0.0636
- Connected components: 1.0
- Largest component size: 4039.0
- Average shortest path (largest component): 3.693
- Diameter (largest component): 8.0
- Density 0.01082 -> the graph is sparse: each user is linked to a small fraction of all users (average degree 43.691).
- Maximum degree (1045.0) is far above the average: a heavy-tailed degree distribution with a few hub users.
- Average clustering 0.6055 -> friends of a user are very often friends with each other (dense local clusters).
- Average path length 3.693 -> short paths ('small-world' property): information can reach distant users in few hops.
- Communities (Louvain): 15, modularity Q = 0.835.

## 2. Influential users per traditional SNA
- Top-5 by degree_centrality: [107, 1684, 1912, 3437, 0]
- Top-5 by betweenness: [107, 1684, 3437, 1912, 1085]
- Top-5 by closeness: [107, 58, 428, 563, 1684]
- Top-5 by pagerank: [3437, 107, 1684, 0, 1912]
- Top-5 by eigenvector: [1912, 2266, 2206, 2233, 2464]

## 3. ML prediction accuracy (held-out test nodes)
| Model | MAE | RMSE | R2 | Spearman |
|---|---|---|---|---|
| Linear Regression | 240.5 | 291.6 | 0.8143 | 0.944 |
| Random Forest | 85.45 | 116.2 | 0.9705 | 0.9788 |
| Gradient Boosting | 88.9 | 120.2 | 0.9685 | 0.9786 |
| [baseline] degree only | nan | nan | nan | 0.8846 |
| [baseline] pagerank only | nan | nan | nan | 0.5502 |
| [baseline] betweenness only | nan | nan | nan | 0.6493 |

Best model by average rank across metrics: **Random Forest** (R2=0.971, Spearman=0.979).
Top-K overlap of predicted vs simulated influencers (test nodes): K=5: 0.00, K=10: 0.10, K=20: 0.45.
Best single-feature ranking baseline: [baseline] degree only (Spearman=0.885) vs Random Forest (Spearman=0.979).

## 4. Most useful SNA features
Permutation importance (test set, best model): core_number (1.161), eigenvector (0.064), closeness (0.040)
Random Forest impurity importance (top 3): core_number (0.873), eigenvector (0.042), closeness (0.035)

## 5. Did AI-selected influencers outperform traditional centralities? (IC, equal budget)
| Strategy | mean_coverage | std_coverage | viral_rate | mean_duration |
|---|---|---|---|---|
| Random | 49.82 | 5.585 | 1 | 17.59 |
| Degree | 53.4 | 1.102 | 1 | 12.89 |
| PageRank | 55.16 | 1.165 | 1 | 12.9 |
| Betweenness | 54.51 | 1.327 | 1 | 13.03 |
| AI | 47.62 | 5.271 | 1 | 17.17 |
| AI + Community | 49.51 | 2.443 | 1 | 15.45 |
- AI performed WORSE than Degree (statistically significant): 47.62% vs 53.40% coverage.
- AI performed WORSE than PageRank (statistically significant): 47.62% vs 55.16% coverage.
- AI performed WORSE than Betweenness (statistically significant): 47.62% vs 54.51% coverage.
- AI performed WORSE than Random (statistically significant): 47.62% vs 49.82% coverage.

## 6. Did community-aware optimisation help?
- AI + Community outperformed AI (statistically significant): 49.51% vs 47.62%.
- AI + Community performed WORSE than PageRank (statistically significant): 49.51% vs 55.16%.
- Communities represented by seeds: AI=3, AI + Community=6, PageRank=8.

## 7. Effect of campaign budget (coverage % by k)
| k | AI | AI + Community | Degree | PageRank | Random |
|---|---|---|---|---|---|
| 1 | 46.94 | 46.94 | 46.31 | 19.96 | 22.72 |
| 3 | 46.92 | 47.57 | 49.65 | 51.32 | 41.32 |
| 5 | 48.15 | 47.71 | 53.4 | 53.4 | 46.94 |
| 10 | 47.62 | 49.51 | 53.4 | 55.16 | 49.82 |

Marginal gain of AI + Community: k=1->3: +0.63 pp, k=3->5: +0.14 pp, k=5->10: +1.80 pp.

## 8. Effect of propagation probability (coverage % by p)
| p | AI | AI + Community | Degree | PageRank | Random |
|---|---|---|---|---|---|
| 0.005 | 0.65 | 0.69 | 1.34 | 1.03 | 0.4 |
| 0.01 | 5.73 | 6.03 | 7.63 | 6.4 | 2.62 |
| 0.02 | 18.94 | 21.82 | 22.39 | 22.94 | 15.28 |
| 0.03 | 28.93 | 34.53 | 36.31 | 37.03 | 29.79 |
| 0.05 | 47.62 | 49.51 | 53.4 | 55.16 | 49.82 |
| 0.1 | 71.63 | 72.09 | 72.9 | 75.71 | 72.7 |

Gap between best and worst strategy (pp) per p: p=0.005: 0.95, p=0.01: 5.01, p=0.02: 7.65, p=0.03: 8.10, p=0.05: 7.53, p=0.1: 4.08.

## 9. IC vs LT (strategy ranking)
| Strategy | IC | LT |
|---|---|---|
| AI | 47.62 | 7.47 |
| AI + Community | 49.51 | 12.92 |
| Betweenness | 54.51 | 31.36 |
| Degree | 53.4 | 33.8 |
| PageRank | 55.16 | 35.36 |
| Random | 49.82 | 3.09 |

Spearman correlation between IC and LT strategy rankings: 0.77.

## 10. Recommended seed-selection strategy
Highest mean IC coverage at k=10: **PageRank** (55.16%). A traditional baseline beat the proposed methods in this experiment; the hypothesis is NOT supported for this setting.

## Hypothesis check
- Hypothesis part 1 (ML combination > single centralities; Degree/PageRank/Betweenness): ['sig. worse', 'sig. worse', 'sig. worse']
- Hypothesis part 2 (community-aware > plain AI top-k): ['sig. better']
- Hypothesis fully supported in this experiment (needs significant improvements): **False**
- Interpret together with the significance tests; results are for simulated cascades on one static network.