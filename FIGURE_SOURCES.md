# Figure Sources and Data Provenance

All figures were generated from actual experimental result files using Python and Matplotlib.
No data were fabricated, interpolated, or reconstructed from summary statistics where trial-level data exist.

Generation scripts: `Experiment_5/figures/generate_figures.py` and `Experiment_7/figures/generate_figures.py`

---

## Experiment 5 Figures

### E5_01 — Choice Proportion Over Trials

| Field | Value |
|-------|-------|
| File | `Experiment_5/figures/E5_01_choice_by_trial.png` |
| Data source (0.6B) | `/Users/amelia/Desktop/llm experiment/exp5_0.6b/trials.jsonl` |
| Data source (1.7B) | `/Users/amelia/Desktop/llm experiment/exp1-4/results/exp5/behavior_reflexion_1.7b_20x20/trials.jsonl` |
| Fields used | `trial`, `chosen_arm`, `episode_id`, `reversal_trial` |
| Sample size | 0.6B: 1,500 trials across 50 episodes; 1.7B: 400 trials across 20 episodes |
| Calculation | For each trial position, mean fraction of episodes where `chosen_arm == 'A'`; ±1 SE across episodes |
| Reversal line | Median `reversal_trial` across episodes (computed from data, not assumed) |
| Fabrication check | ✅ No invented values — all points are episode-level averages from raw trial data |

---

### E5_02 — Pre- and Post-Reversal Performance

| Field | Value |
|-------|-------|
| File | `Experiment_5/figures/E5_02_reversal_performance.png` |
| Data source | `Experiment_5/Qwen3_0.6B/behavior_summary.json`, `Experiment_5/Qwen3_1.7B_20x20/behavior_summary.json` |
| Fields used | `optimal_choice_accuracy_pre_reversal`, `optimal_choice_accuracy_post_reversal` |
| Verified values | 0.6B pre = 0.5451, post = 0.4651; 1.7B pre = 0.5792, post = 0.3788 |
| Fabrication check | ✅ Direct reads from saved summary files; no aggregation performed |

---

## Experiment 7 Figures

### E7_01 — Layer-wise Linear Probe Performance

| Field | Value |
|-------|-------|
| File | `Experiment_7/figures/E7_01_probe_layerwise.png` |
| Data source | `/Users/amelia/Desktop/llmneuroscientist/results/exp7b/probe_results.json` |
| Fields used | `position`, `layer`, `target`, `r_test` |
| Filter | `position == "choice"`, layers 1–28 (layer 0 is embedding output; all NaN in this dataset) |
| Variables shown | `choice_tendency`, `reward_history`, `recent_reward`, `delta_q` |
| Fabrication check | ✅ One `r_test` value per (layer, target) combination read directly from file; NaN values excluded |

---

### E7_02 — Best Linear Probe Performance per Target Variable

| Field | Value |
|-------|-------|
| File | `Experiment_7/figures/E7_02_probe_summary.png` |
| Data source | `/Users/amelia/Desktop/llmneuroscientist/results/exp7b/probe_summary.json` |
| Fields used | `best_r["choice|<target>"]["r"]` and `["layer"]` for each target |
| Verified values | choice_tendency: r=0.9687 layer 27; reward_history: r=0.9381 layer 5; recent_reward: r=0.8133 layer 7; delta_q: r=0.5428 layer 22; post_reversal: r=0.3610 layer 15; rpe_proxy: r=0.1294 layer 17 |
| Fabrication check | ✅ Direct reads from saved summary file |

---

### E7_03 — DAS V2 Confirmatory Result

| Field | Value |
|-------|-------|
| File | `Experiment_7/figures/E7_03_das_primary_result.png` |
| Data sources | `/Users/amelia/Desktop/llmneuroscientist/VALIDATION_EXP7_V2/results/confirmatory_run/summary.json` (CI and cluster values); `confirmatory_run/records.jsonl` (mean absolute change, SE) |
| Fields used | `primary_cluster_bootstrap.mean`, `primary_cluster_bootstrap.interval_95_percentile`, `raw_margin_change_absolute`, `condition` |
| Verified values | Mean paired diff = 0.11195; CI = [0.07784, 0.15055]; DAS mean abs = 0.138; random mean abs = 0.026 |
| SE calculation | SE across 10 cluster means (not across individual records) |
| Fabrication check | ✅ Mean and CI read directly from summary.json; per-record absolute changes read from records.jsonl |

---

### E7_04 — DAS Cluster-Level Effects

| Field | Value |
|-------|-------|
| File | `Experiment_7/figures/E7_04_das_cluster_effects.png` |
| Data source | `/Users/amelia/Desktop/llmneuroscientist/VALIDATION_EXP7_V2/results/confirmatory_run/summary.json` |
| Fields used | `primary_cluster_bootstrap.clusters` (10 values), `primary_cluster_bootstrap.mean`, `primary_cluster_bootstrap.interval_95_percentile` |
| Verified values | Clusters: [0.2234, 0.0618, 0.1394, 0.0628, 0.1536, 0.0684, 0.0881, 0.1987, 0.0603, 0.0630] |
| Fabrication check | ✅ All 10 cluster values read directly from summary.json; none constructed |

---

### E7_05 — DAS Intervention Distribution

| Field | Value |
|-------|-------|
| File | `Experiment_7/figures/E7_05_das_intervention_dist.png` |
| Data source | `/Users/amelia/Desktop/llmneuroscientist/VALIDATION_EXP7_V2/results/confirmatory_run/records.jsonl` |
| Fields used | `condition`, `raw_margin_change_absolute`, `restricted_ab_argmax_flip`, `directional_agreement` |
| Sample sizes | DAS: 200 records; random: 2,000 records (10 norm-matched bases × 200 trials) |
| Verified values | DAS argmax flips: 0/200; random argmax flips: 0/2,000; DAS DA: 76.0% (152/200); random DA: 50.85% (1,017/2,000) |
| Contamination note | These records are from the independent V2 confirmatory run only. The contaminated historical validation result (IIA=0.995 from validate_02_random_iia.py) is not used or referenced anywhere in these figures. |
| Fabrication check | ✅ All records read line-by-line from records.jsonl; histogram bins computed from data |

---

## Notes

**Reversal trial annotation (E5_01):** The reversal trial shown in the figure is the median `reversal_trial` value computed from the actual trial records, not a fixed assumption. For Qwen3-0.6B this is trial 16; for Qwen3-1.7B this is trial 11.

**Layer 0 exclusion (E7_01):** Layer 0 corresponds to the token embedding output before any transformer blocks. All `r_test` values at layer 0 are NaN in `probe_results.json` and are excluded from the layer-wise plot.

**DAS internal metric terminology (E7_02, E7_03, E7_04, E7_05):** The field `iia_val` in `das_results.json` measures directional agreement, not standard interchange intervention accuracy. See `Experiment_7/README.md` section 8 for the full terminology note.
