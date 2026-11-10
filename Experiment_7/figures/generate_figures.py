"""
Experiment 7 — Figure Generation Script
========================================
Generates publication-quality figures from actual experimental data.

Data sources
------------
Probe layer profile   : /Users/amelia/Desktop/llmneuroscientist/results/exp7b/probe_results.json
Probe summary         : /Users/amelia/Desktop/llmneuroscientist/results/exp7b/probe_summary.json
DAS training results  : /Users/amelia/Desktop/llmneuroscientist/results/exp7h/das_results.json
V2 summary            : /Users/amelia/Desktop/llmneuroscientist/VALIDATION_EXP7_V2/results/confirmatory_run/summary.json
V2 intervention recs  : /Users/amelia/Desktop/llmneuroscientist/VALIDATION_EXP7_V2/results/confirmatory_run/records.jsonl

Figures generated
-----------------
E7_01_probe_layerwise.png           Layer-wise Pearson r, choice position, key variables
E7_02_probe_summary.png             Best-layer r per target variable (bar chart)
E7_03_das_primary_result.png        V2 paired difference with 95% CI and cluster values
E7_04_das_cluster_effects.png       Per-cluster paired differences (10 clusters)
E7_05_das_intervention_dist.png     Distribution of absolute logit change, DAS vs random
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE     = Path(__file__).resolve().parent
OUT      = HERE

PROBE_RESULTS = Path("/Users/amelia/Desktop/llmneuroscientist/results/exp7b/probe_results.json")
PROBE_SUMMARY = Path("/Users/amelia/Desktop/llmneuroscientist/results/exp7b/probe_summary.json")
DAS_RESULTS   = Path("/Users/amelia/Desktop/llmneuroscientist/results/exp7h/das_results.json")
V2_SUMMARY    = Path("/Users/amelia/Desktop/llmneuroscientist/VALIDATION_EXP7_V2/results/confirmatory_run/summary.json")
V2_RECORDS    = Path("/Users/amelia/Desktop/llmneuroscientist/VALIDATION_EXP7_V2/results/confirmatory_run/records.jsonl")

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family":         "DejaVu Sans",
    "font.size":           11,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.linewidth":      0.8,
    "xtick.major.width":   0.8,
    "ytick.major.width":   0.8,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "figure.facecolor":    "white",
    "axes.facecolor":      "white",
    "grid.color":          "#e0e0e0",
    "grid.linewidth":      0.5,
    "legend.frameon":      False,
    "legend.fontsize":     10,
    "axes.titlesize":      12,
    "axes.titleweight":    "bold",
    "axes.labelsize":      11,
})

# Colour palette
C_CHOICE   = "#2166ac"   # choice_tendency — blue
C_REWARD   = "#1b7837"   # reward_history  — dark green
C_RECENT   = "#d6604d"   # recent_reward   — red-orange
C_DELTA    = "#762a83"   # delta_q         — purple
C_DAS      = "#b2182b"   # DAS             — dark red
C_RANDOM   = "#878787"   # random baseline — grey
C_CLUSTER  = "#4393c3"   # cluster dots    — light blue

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_probe_results() -> dict[tuple[str, int, str], float]:
    """Returns {(position, layer, target): r_test}."""
    records = json.loads(PROBE_RESULTS.read_text())
    out = {}
    for rec in records:
        r = rec["r_test"]
        if r is not None and not (isinstance(r, float) and np.isnan(r)):
            out[(rec["position"], rec["layer"], rec["target"])] = r
    return out


def load_v2_records() -> list[dict]:
    return [json.loads(line) for line in V2_RECORDS.read_text().splitlines() if line.strip()]

# ---------------------------------------------------------------------------
# Figure E7-01 — Layer-wise probing performance
# ---------------------------------------------------------------------------

def figure_probe_layerwise() -> None:
    data = load_probe_results()
    layers = list(range(1, 29))   # skip layer 0 (embedding; all NaN in this dataset)

    # Key variables at the choice token position
    variables = [
        ("choice_tendency", "Choice tendency",  C_CHOICE,  2.0),
        ("reward_history",  "Reward history",   C_REWARD,  1.6),
        ("recent_reward",   "Recent reward",    C_RECENT,  1.4),
        ("delta_q",         "Delta Q (stated)", C_DELTA,   1.2),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    for target, label, color, lw in variables:
        rs = [data.get(("choice", l, target), np.nan) for l in layers]
        ax.plot(layers, rs, color=color, linewidth=lw, label=label, marker="o",
                markersize=3.5, markerfacecolor=color, markeredgewidth=0)

    # Annotate peak for choice_tendency
    peak_layer = 27
    peak_r = data.get(("choice", peak_layer, "choice_tendency"), np.nan)
    if not np.isnan(peak_r):
        ax.annotate(f"r = {peak_r:.3f}\n(Layer {peak_layer})",
                    xy=(peak_layer, peak_r),
                    xytext=(peak_layer - 5, peak_r - 0.12),
                    fontsize=9,
                    color=C_CHOICE,
                    arrowprops=dict(arrowstyle="->", color=C_CHOICE, lw=0.8))

    ax.axhline(0, color="#aaaaaa", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Pearson r (held-out test set)")
    ax.set_title("Experiment 7 — Layer-wise Linear Probe Performance\n(choice token position)", fontweight="bold")
    ax.set_xlim(0.5, 28.5)
    ax.set_ylim(-0.15, 1.05)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 28])
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
    ax.legend(loc="upper left", ncol=2)
    ax.yaxis.grid(True, zorder=0)

    ax.text(0.5, -0.14,
            "OLS probe fitted on 35 training episodes (1,050 trials); r measured on 15 held-out test episodes (450 trials).\n"
            "Source: results/exp7b/probe_results.json",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#555555")

    fig.tight_layout()
    out_path = OUT / "E7_01_probe_layerwise.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure E7-02 — Probe summary bar chart
# ---------------------------------------------------------------------------

def figure_probe_summary() -> None:
    summary = json.loads(PROBE_SUMMARY.read_text())

    # Choice-position results, sorted by r descending
    choice_vars = [
        ("choice_tendency", "Choice tendency",  C_CHOICE),
        ("reward_history",  "Reward history",   C_REWARD),
        ("recent_reward",   "Recent reward",    C_RECENT),
        ("delta_q",         "Delta Q (stated)", C_DELTA),
        ("post_reversal",   "Post-reversal",    "#b35806"),
        ("rpe_proxy",       "RPE proxy",        "#878787"),
    ]

    labels, rs, colors, best_layers = [], [], [], []
    for key, label, color in choice_vars:
        entry = summary["best_r"].get(f"choice|{key}")
        if entry:
            labels.append(label)
            rs.append(entry["r"])
            colors.append(color)
            best_layers.append(entry["layer"])

    # Sort by r descending
    order = np.argsort(rs)[::-1]
    labels      = [labels[i] for i in order]
    rs          = [rs[i] for i in order]
    colors      = [colors[i] for i in order]
    best_layers = [best_layers[i] for i in order]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    y = np.arange(len(labels))
    bars = ax.barh(y, rs, color=colors, alpha=0.85, height=0.55)

    for i, (bar, r, layer) in enumerate(zip(bars, rs, best_layers)):
        ax.text(r + 0.01, bar.get_y() + bar.get_height() / 2,
                f"r = {r:.3f}  (layer {layer})",
                va="center", fontsize=9.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Pearson r (held-out test set, 450 trials)")
    ax.set_xlim(0, 1.22)
    ax.axvline(0, color="#aaaaaa", linewidth=0.6)
    ax.set_title("Experiment 7 — Best Linear Probe Performance per Target Variable\n(choice token position)", fontweight="bold")

    ax.text(0.5, -0.14,
            "Best-layer r for each target variable at the choice token position.\n"
            "Source: results/exp7b/probe_summary.json",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#555555")

    fig.tight_layout()
    out_path = OUT / "E7_02_probe_summary.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure E7-03 — DAS V2 primary result
# ---------------------------------------------------------------------------

def figure_das_primary_result() -> None:
    v2 = json.loads(V2_SUMMARY.read_text())
    bootstrap = v2["primary_cluster_bootstrap"]

    mean_diff   = bootstrap["mean"]
    ci_lo, ci_hi = bootstrap["interval_95_percentile"]

    # Mean absolute changes from records
    records = load_v2_records()
    das_abs    = [r["raw_margin_change_absolute"] for r in records if r["condition"] == "das"]
    random_abs = [r["raw_margin_change_absolute"] for r in records if r["condition"] == "random"]
    das_mean    = float(np.mean(das_abs))
    random_mean = float(np.mean(random_abs))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5),
                              gridspec_kw={"width_ratios": [1.4, 1]})

    # --- Left panel: mean absolute change ---
    ax = axes[0]
    labels = ["DAS\n(n = 200)", "Norm-matched\nrandom\n(n = 2,000)"]
    means  = [das_mean, random_mean]
    colors = [C_DAS, C_RANDOM]

    # SE across clusters for DAS
    clusters = bootstrap["clusters"]
    # Cluster means for DAS absolute change (20 DAS pairs per cluster)
    das_by_cluster = {}
    for r in records:
        if r["condition"] == "das":
            c = r["cluster"]
            das_by_cluster.setdefault(c, []).append(r["raw_margin_change_absolute"])
    cluster_das_means  = [np.mean(v) for v in das_by_cluster.values()]
    das_se = float(np.std(cluster_das_means, ddof=1) / np.sqrt(len(cluster_das_means)))

    random_by_cluster = {}
    for r in records:
        if r["condition"] == "random":
            c = r["cluster"]
            random_by_cluster.setdefault(c, []).append(r["raw_margin_change_absolute"])
    cluster_random_means = [np.mean(v) for v in random_by_cluster.values()]
    random_se = float(np.std(cluster_random_means, ddof=1) / np.sqrt(len(cluster_random_means)))

    bars = ax.bar([0, 1], means, color=colors, alpha=0.85, width=0.45,
                  yerr=[das_se, random_se], capsize=5,
                  error_kw=dict(ecolor="#444444", elinewidth=1.2, capthick=1.2))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean absolute A/B logit change")
    ax.set_ylim(0, max(means) * 1.55)
    ax.set_title("Mean Intervention Effect", fontweight="bold")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f"{m:.3f}", ha="center", va="bottom", fontsize=10)

    # --- Right panel: paired difference + CI ---
    ax2 = axes[1]
    ax2.errorbar(0, mean_diff, yerr=[[mean_diff - ci_lo], [ci_hi - mean_diff]],
                 fmt="o", color=C_DAS, markersize=9, capsize=7,
                 elinewidth=2.0, capthick=2.0, markeredgewidth=0, zorder=3)
    ax2.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_ylim(-0.02, max(ci_hi * 1.5, 0.22))
    ax2.set_xticks([0])
    ax2.set_xticklabels(["DAS − random"])
    ax2.set_ylabel("Paired difference in absolute logit change")
    ax2.set_title("Paired Difference\n(10-cluster bootstrap)", fontweight="bold")
    ax2.text(0, ci_hi + 0.008, f"{mean_diff:.4f}\n[{ci_lo:.4f}, {ci_hi:.4f}]",
             ha="center", va="bottom", fontsize=9, color=C_DAS)

    fig.suptitle("Experiment 7 — DAS V2 Confirmatory Result\n(pre-registered, 20 independent episodes)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.text(0.5, -0.04,
             "Left: mean absolute A/B logit change ± SE across clusters. Right: mean paired difference (DAS − random)\n"
             "with 95% bootstrap CI (10 clusters, 10,000 resamples, seed 8128). Source: confirmatory_run/summary.json and records.jsonl",
             ha="center", fontsize=8.5, color="#555555")

    fig.tight_layout()
    out_path = OUT / "E7_03_das_primary_result.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure E7-04 — Per-cluster paired differences
# ---------------------------------------------------------------------------

def figure_das_cluster_effects() -> None:
    v2 = json.loads(V2_SUMMARY.read_text())
    bootstrap = v2["primary_cluster_bootstrap"]
    clusters  = bootstrap["clusters"]
    mean_diff = bootstrap["mean"]
    ci_lo, ci_hi = bootstrap["interval_95_percentile"]

    n = len(clusters)
    x = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.bar(x, clusters, color=C_CLUSTER, alpha=0.75, width=0.6, zorder=2)
    ax.axhline(mean_diff, color=C_DAS, linewidth=1.8, linestyle="-",
               label=f"Mean = {mean_diff:.4f}", zorder=3)
    ax.axhline(0, color="#aaaaaa", linewidth=0.7, linestyle="--", zorder=1)

    # CI band on mean
    ax.axhspan(ci_lo, ci_hi, color=C_DAS, alpha=0.08,
               label=f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]", zorder=0)

    ax.set_xlabel("Cluster (20 DAS pairs each)")
    ax.set_ylabel("Paired difference\n(DAS − random) in absolute logit change")
    ax.set_title("Experiment 7 — DAS Cluster-Level Effects\n(10 clusters, V2 confirmatory run)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{i}" for i in x])
    ax.yaxis.grid(True, zorder=0, alpha=0.6)
    ax.set_ylim(-0.02, max(clusters) * 1.3)
    ax.legend(loc="upper right")

    for xi, v in zip(x, clusters):
        ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, color="#333333")

    ax.text(0.5, -0.14,
            "Each cluster contains 20 DAS intervention pairs and 200 norm-matched random intervention pairs.\n"
            "All 10 clusters show DAS > random. Source: confirmatory_run/summary.json",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#555555")

    fig.tight_layout()
    out_path = OUT / "E7_04_das_cluster_effects.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure E7-05 — Distribution of absolute logit change
# ---------------------------------------------------------------------------

def figure_das_intervention_dist() -> None:
    records    = load_v2_records()
    das_abs    = np.array([r["raw_margin_change_absolute"] for r in records if r["condition"] == "das"])
    random_abs = np.array([r["raw_margin_change_absolute"] for r in records if r["condition"] == "random"])

    n_das_flips    = sum(r["restricted_ab_argmax_flip"] for r in records if r["condition"] == "das")
    n_random_flips = sum(r["restricted_ab_argmax_flip"] for r in records if r["condition"] == "random")
    n_das_da       = sum(r["directional_agreement"] for r in records if r["condition"] == "das")
    n_random_da    = sum(r["directional_agreement"] for r in records if r["condition"] == "random")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Left: distributions ---
    ax = axes[0]
    bins = np.linspace(0, max(das_abs.max(), random_abs.max()) * 1.05, 35)

    ax.hist(random_abs, bins=bins, color=C_RANDOM, alpha=0.65, density=True,
            label=f"Random (n={len(random_abs):,})", zorder=2)
    ax.hist(das_abs,    bins=bins, color=C_DAS,    alpha=0.75, density=True,
            label=f"DAS (n={len(das_abs):,})", zorder=3)

    ax.axvline(das_abs.mean(),    color=C_DAS,    linewidth=1.5, linestyle="--",
               label=f"DAS mean = {das_abs.mean():.3f}")
    ax.axvline(random_abs.mean(), color=C_RANDOM, linewidth=1.5, linestyle="--",
               label=f"Random mean = {random_abs.mean():.3f}")

    ax.set_xlabel("Absolute A/B logit change")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Absolute Logit Change", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)

    # --- Right: outcome summary ---
    ax2 = axes[1]
    categories = ["Argmax\nflips", "Directional\nagreement"]
    das_rates    = [n_das_flips / len(das_abs), n_das_da / len(das_abs)]
    random_rates = [n_random_flips / len(random_abs), n_random_da / len(random_abs)]

    x = np.arange(len(categories))
    w = 0.3
    bars_das    = ax2.bar(x - w / 2, das_rates,    w, label="DAS",    color=C_DAS,    alpha=0.85)
    bars_random = ax2.bar(x + w / 2, random_rates, w, label="Random", color=C_RANDOM, alpha=0.85)

    for bars in [bars_das, bars_random]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                     f"{h:.1%}", ha="center", va="bottom", fontsize=9)

    ax2.axhline(0.5, color="#aaaaaa", linewidth=0.6, linestyle=":", label="Chance (0.5)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_ylabel("Rate")
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    ax2.set_title("Intervention Outcomes", fontweight="bold")
    ax2.legend(loc="upper right")

    fig.suptitle("Experiment 7 — DAS Intervention Analysis\n(V2 confirmatory run, 20 independent episodes)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.text(0.5, -0.04,
             f"Left: distribution of absolute A/B logit change. Right: argmax flip rate and directional agreement rate.\n"
             f"DAS: {n_das_flips}/200 argmax flips (0%). Random: {n_random_flips}/2,000 argmax flips ({n_random_flips/len(random_abs):.1%}).\n"
             "Source: confirmatory_run/records.jsonl",
             ha="center", fontsize=8.5, color="#555555")

    fig.tight_layout()
    out_path = OUT / "E7_05_das_intervention_dist.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating Experiment 7 figures...")
    figure_probe_layerwise()
    figure_probe_summary()
    figure_das_primary_result()
    figure_das_cluster_effects()
    figure_das_intervention_dist()
    print("Done.")
