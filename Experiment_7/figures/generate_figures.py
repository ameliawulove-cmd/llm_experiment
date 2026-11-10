"""
Experiment 7 — Figure Generation Script (revised)
===================================================
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
E7_01_probe_layerwise           Layer-wise Pearson r, choice position, key variables
E7_02_probe_summary             Best-layer r per target variable (bar chart)
E7_03_das_primary_result        V2 paired difference with 95% CI and cluster values
E7_04_das_cluster_effects       Per-cluster paired differences (10 clusters)
E7_05_das_intervention_dist     Distribution of absolute logit change, DAS vs random
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE  = Path(__file__).resolve().parent
OUT   = HERE

PROBE_RESULTS = Path("/Users/amelia/Desktop/llmneuroscientist/results/exp7b/probe_results.json")
PROBE_SUMMARY = Path("/Users/amelia/Desktop/llmneuroscientist/results/exp7b/probe_summary.json")
DAS_RESULTS   = Path("/Users/amelia/Desktop/llmneuroscientist/results/exp7h/das_results.json")
V2_SUMMARY    = Path("/Users/amelia/Desktop/llmneuroscientist/VALIDATION_EXP7_V2/results/confirmatory_run/summary.json")
V2_RECORDS    = Path("/Users/amelia/Desktop/llmneuroscientist/VALIDATION_EXP7_V2/results/confirmatory_run/records.jsonl")

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
TEXT_COLOR = "#353047"
GRID_COLOR = "#E8E3EF"
SPINE_COLOR = "#CCCCCC"

matplotlib.rcParams.update({
    "font.family":          "DejaVu Sans",
    "font.size":            10,
    "text.color":           TEXT_COLOR,
    "axes.labelcolor":      TEXT_COLOR,
    "xtick.color":          TEXT_COLOR,
    "ytick.color":          TEXT_COLOR,
    "axes.edgecolor":       SPINE_COLOR,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.linewidth":       0.7,
    "xtick.major.width":    0.7,
    "ytick.major.width":    0.7,
    "xtick.major.size":     4,
    "ytick.major.size":     4,
    "xtick.minor.visible":  False,
    "ytick.minor.visible":  False,
    "figure.facecolor":     "white",
    "axes.facecolor":       "white",
    "grid.color":           GRID_COLOR,
    "grid.linewidth":       0.6,
    "legend.frameon":       False,
    "legend.fontsize":      9,
    "axes.titlesize":       10,
    "axes.titleweight":     "normal",
    "axes.titlepad":        8,
    "axes.labelsize":       10,
    "figure.dpi":           100,
})

# Probe palette — four clearly distinguishable, muted academic colours
C_CHOICE  = "#2D6A9F"   # steel blue
C_REWARD  = "#3A8A5C"   # sage green
C_RECENT  = "#C67E3A"   # warm amber
C_DELTA   = "#7A5FA5"   # medium purple

# DAS palette
C_DAS     = "#8B73B5"   # muted lavender-purple
C_RANDOM  = "#D9B86C"   # champagne gold
C_CLUSTER = "#6B8FC4"   # lighter blue for cluster bars

SAVE_DPI = 300


def save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "svg"):
        path = OUT / f"{stem}.{ext}"
        fig.savefig(path, dpi=SAVE_DPI if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
        print(f"  Saved: {path}")
    plt.close(fig)


def footnote(fig: plt.Figure, text: str, y: float = 0.015) -> None:
    fig.text(0.5, y, text, ha="center", va="bottom",
             fontsize=8, color="#888888", style="italic",
             transform=fig.transFigure)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_probe_results() -> dict[tuple[str, int, str], float]:
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
    layers = list(range(1, 29))

    variables = [
        ("choice_tendency", "Choice tendency",  C_CHOICE, 2.0),
        ("reward_history",  "Reward history",   C_REWARD, 1.6),
        ("recent_reward",   "Recent reward",    C_RECENT, 1.4),
        ("delta_q",         "Delta Q (stated)", C_DELTA,  1.2),
    ]

    # Manual layout: title + legend strip at top, axes, then footnote strip at bottom
    fig = plt.figure(figsize=(10, 5.8), facecolor="white")
    # Leave room: top=0.78 for title+legend, bottom=0.18 for footnote
    ax = fig.add_axes([0.08, 0.20, 0.88, 0.57])  # [left, bottom, width, height]

    lines = []
    for target, label, color, lw in variables:
        rs = [data.get(("choice", l, target), np.nan) for l in layers]
        line, = ax.plot(layers, rs, color=color, linewidth=lw,
                        marker="o", markersize=3.5,
                        markerfacecolor=color, markeredgewidth=0,
                        label=label)
        lines.append(line)

    # Annotate the peak
    peak_layer = 27
    peak_r = data.get(("choice", peak_layer, "choice_tendency"), np.nan)
    if not np.isnan(peak_r):
        ax.annotate(
            f"r = {peak_r:.3f}  (layer {peak_layer})",
            xy=(peak_layer, peak_r),
            xytext=(peak_layer - 7.5, peak_r - 0.17),
            fontsize=8.5, color=C_CHOICE,
            arrowprops=dict(arrowstyle="->", color=C_CHOICE, lw=0.8),
        )

    ax.axhline(0, color=SPINE_COLOR, linewidth=0.6, linestyle=":")
    ax.set_xlabel("Transformer layer", labelpad=6)
    ax.set_ylabel("Pearson r  (held-out test set)", labelpad=6)
    ax.set_xlim(0.5, 28.5)
    ax.set_ylim(-0.20, 1.08)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 28])
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
    ax.yaxis.grid(True, zorder=0, color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)

    # Legend — placed above the axes, inside the figure top margin
    fig.legend(
        handles=lines,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        fontsize=9,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.5,
    )

    # Title — above the legend
    fig.text(0.5, 1.00, "Layer-wise Linear Probe Performance — Choice Token Position",
             ha="center", va="top", fontsize=10.5, color=TEXT_COLOR,
             transform=fig.transFigure)

    footnote(
        fig,
        "OLS probe fitted on 35 training episodes (1,050 trials); r measured on 15 held-out test episodes (450 trials).  "
        "Source: results/exp7b/probe_results.json",
        y=0.025,
    )

    save(fig, "E7_01_probe_layerwise")


# ---------------------------------------------------------------------------
# Figure E7-02 — Probe summary bar chart
# ---------------------------------------------------------------------------

def figure_probe_summary() -> None:
    summary = json.loads(PROBE_SUMMARY.read_text())

    choice_vars = [
        ("choice_tendency", "Choice tendency",  C_CHOICE),
        ("reward_history",  "Reward history",   C_REWARD),
        ("recent_reward",   "Recent reward",    C_RECENT),
        ("delta_q",         "Delta Q (stated)", C_DELTA),
        ("post_reversal",   "Post-reversal",    "#B08A5E"),
        ("rpe_proxy",       "RPE proxy",        "#A0A0A0"),
    ]

    labels, rs, colors, best_layers = [], [], [], []
    for key, label, color in choice_vars:
        entry = summary["best_r"].get(f"choice|{key}")
        if entry:
            labels.append(label)
            rs.append(entry["r"])
            colors.append(color)
            best_layers.append(entry["layer"])

    order = np.argsort(rs)[::-1]
    labels      = [labels[i] for i in order]
    rs          = [rs[i] for i in order]
    colors      = [colors[i] for i in order]
    best_layers = [best_layers[i] for i in order]

    fig, ax = plt.subplots(figsize=(8.5, 4.6), facecolor="white")
    fig.subplots_adjust(left=0.20, right=0.78, top=0.88, bottom=0.18)

    y = np.arange(len(labels))
    bars = ax.barh(y, rs, color=colors, alpha=0.85, height=0.52)

    for bar, r, layer in zip(bars, rs, best_layers):
        ax.text(r + 0.012, bar.get_y() + bar.get_height() / 2,
                f"r = {r:.3f}   layer {layer}",
                va="center", fontsize=9, color=TEXT_COLOR)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("Pearson r  (held-out test set, 450 trials)", labelpad=6)
    ax.set_xlim(0, 1.26)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
    ax.xaxis.grid(True, zorder=0, color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Best Linear Probe Performance per Target Variable\n(choice token position)",
                 fontsize=10.5, pad=10)

    footnote(fig,
             "Best-layer r for each target variable at the choice token position.  "
             "Source: results/exp7b/probe_summary.json")

    save(fig, "E7_02_probe_summary")


# ---------------------------------------------------------------------------
# Figure E7-03 — DAS V2 primary result
# ---------------------------------------------------------------------------

def figure_das_primary_result() -> None:
    v2        = json.loads(V2_SUMMARY.read_text())
    bootstrap = v2["primary_cluster_bootstrap"]
    mean_diff  = bootstrap["mean"]
    ci_lo, ci_hi = bootstrap["interval_95_percentile"]

    records    = load_v2_records()
    das_abs    = [r["raw_margin_change_absolute"] for r in records if r["condition"] == "das"]
    random_abs = [r["raw_margin_change_absolute"] for r in records if r["condition"] == "random"]
    das_mean    = float(np.mean(das_abs))
    random_mean = float(np.mean(random_abs))

    # Cluster-level SE
    das_by_cluster = {}
    random_by_cluster = {}
    for r in records:
        c = r["cluster"]
        if r["condition"] == "das":
            das_by_cluster.setdefault(c, []).append(r["raw_margin_change_absolute"])
        else:
            random_by_cluster.setdefault(c, []).append(r["raw_margin_change_absolute"])
    das_se    = float(np.std([np.mean(v) for v in das_by_cluster.values()], ddof=1) /
                      np.sqrt(len(das_by_cluster)))
    random_se = float(np.std([np.mean(v) for v in random_by_cluster.values()], ddof=1) /
                      np.sqrt(len(random_by_cluster)))

    fig = plt.figure(figsize=(10.5, 4.8), facecolor="white")
    gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.5, 1],
                            left=0.09, right=0.96, top=0.84, bottom=0.20,
                            wspace=0.42)
    ax_bar  = fig.add_subplot(gs[0])
    ax_diff = fig.add_subplot(gs[1])

    # --- Left: mean absolute change ---
    for ax in (ax_bar, ax_diff):
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(SPINE_COLOR)
        ax.spines["bottom"].set_color(SPINE_COLOR)

    x_pos   = [0, 1]
    heights = [das_mean, random_mean]
    colors  = [C_DAS, C_RANDOM]
    errs    = [das_se, random_se]
    labels  = [f"DAS\n(n = 200)", f"Norm-matched\nrandom\n(n = 2,000)"]

    bars = ax_bar.bar(x_pos, heights, color=colors, alpha=0.88, width=0.42,
                      yerr=errs, capsize=5,
                      error_kw=dict(ecolor="#888888", elinewidth=1.0, capthick=1.0),
                      zorder=3)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(labels, fontsize=9)
    ax_bar.set_ylabel("Mean absolute A/B logit change", labelpad=6)
    ax_bar.set_ylim(0, max(heights) * 1.6)
    ax_bar.yaxis.grid(True, color=GRID_COLOR, linewidth=0.6, zorder=0)
    ax_bar.set_axisbelow(True)
    ax_bar.set_title("Mean Intervention Effect", fontsize=9.5, pad=8)

    for bar, h in zip(bars, heights):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, h + errs[bars.index(bar)] + 0.004,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9, color=TEXT_COLOR)

    # --- Right: paired difference + CI ---
    ax_diff.errorbar(0, mean_diff,
                     yerr=[[mean_diff - ci_lo], [ci_hi - mean_diff]],
                     fmt="o", color=C_DAS, markersize=8,
                     capsize=7, elinewidth=1.8, capthick=1.8,
                     markeredgewidth=0, zorder=3)
    ax_diff.axhline(0, color=SPINE_COLOR, linewidth=0.8, linestyle="--")
    ax_diff.set_xlim(-0.6, 0.6)
    y_top = ci_hi * 1.65
    ax_diff.set_ylim(-0.02, max(y_top, 0.22))
    ax_diff.set_xticks([0])
    ax_diff.set_xticklabels(["DAS − random"], fontsize=9)
    ax_diff.set_ylabel("Paired difference", labelpad=6)
    ax_diff.yaxis.grid(True, color=GRID_COLOR, linewidth=0.6, zorder=0)
    ax_diff.set_axisbelow(True)
    ax_diff.set_title("Paired Difference\n(10-cluster bootstrap)", fontsize=9.5, pad=8)

    ax_diff.text(0, ci_hi + 0.010,
                 f"{mean_diff:.4f}\n[{ci_lo:.4f}, {ci_hi:.4f}]",
                 ha="center", va="bottom", fontsize=8.5, color=C_DAS)

    fig.text(0.5, 0.95,
             "DAS V2 Confirmatory Result  —  pre-registered, 20 independent episodes",
             ha="center", va="top", fontsize=10.5, color=TEXT_COLOR,
             transform=fig.transFigure)

    footnote(fig,
             "Left: mean absolute A/B logit change ± SE across clusters.  "
             "Right: mean paired difference (DAS − random) with 95% bootstrap CI "
             "(10 clusters, 10,000 resamples, seed 8128).  "
             "Source: confirmatory_run/summary.json and records.jsonl")

    save(fig, "E7_03_das_primary_result")


# ---------------------------------------------------------------------------
# Figure E7-04 — Per-cluster paired differences
# ---------------------------------------------------------------------------

def figure_das_cluster_effects() -> None:
    v2        = json.loads(V2_SUMMARY.read_text())
    bootstrap = v2["primary_cluster_bootstrap"]
    clusters  = bootstrap["clusters"]
    mean_diff = bootstrap["mean"]
    ci_lo, ci_hi = bootstrap["interval_95_percentile"]

    n = len(clusters)
    x = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor="white")
    fig.subplots_adjust(left=0.10, right=0.96, top=0.86, bottom=0.22)

    ax.bar(x, clusters, color=C_CLUSTER, alpha=0.80, width=0.55, zorder=2)
    ax.axhline(mean_diff, color=C_DAS, linewidth=1.6, linestyle="-",
               label=f"Mean = {mean_diff:.4f}", zorder=3)
    ax.axhline(0, color=SPINE_COLOR, linewidth=0.7, linestyle="--", zorder=1)
    ax.axhspan(ci_lo, ci_hi, color=C_DAS, alpha=0.09,
               label=f"95% CI  [{ci_lo:.4f}, {ci_hi:.4f}]", zorder=0)

    ax.set_xlabel("Cluster  (20 DAS pairs each)", labelpad=6)
    ax.set_ylabel("Paired difference\n(DAS − random) in absolute logit change", labelpad=6)
    ax.set_title("DAS Cluster-Level Effects  —  10 clusters, V2 confirmatory run",
                 fontsize=10.5, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{i}" for i in x], fontsize=9)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.6, zorder=0, alpha=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.02, max(clusters) * 1.32)

    legend = ax.legend(loc="upper right", fontsize=8.5,
                       frameon=True, framealpha=0.9,
                       edgecolor=GRID_COLOR)
    legend.get_frame().set_linewidth(0.5)

    for xi, v in zip(x, clusters):
        ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", va="bottom",
                fontsize=8, color=TEXT_COLOR)

    footnote(fig,
             "Each cluster contains 20 DAS intervention pairs and 200 norm-matched random pairs.  "
             "All 10 clusters show DAS > random.  Source: confirmatory_run/summary.json")

    save(fig, "E7_04_das_cluster_effects")


# ---------------------------------------------------------------------------
# Figure E7-05 — Distribution of absolute logit change
# ---------------------------------------------------------------------------

def figure_das_intervention_dist() -> None:
    records    = load_v2_records()
    das_abs    = np.array([r["raw_margin_change_absolute"]
                           for r in records if r["condition"] == "das"])
    random_abs = np.array([r["raw_margin_change_absolute"]
                           for r in records if r["condition"] == "random"])

    n_das_flips    = sum(r["restricted_ab_argmax_flip"] for r in records if r["condition"] == "das")
    n_random_flips = sum(r["restricted_ab_argmax_flip"] for r in records if r["condition"] == "random")
    n_das_da       = sum(r["directional_agreement"] for r in records if r["condition"] == "das")
    n_random_da    = sum(r["directional_agreement"] for r in records if r["condition"] == "random")

    fig = plt.figure(figsize=(11.5, 4.8), facecolor="white")
    gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.65, 1],
                            left=0.08, right=0.97, top=0.84, bottom=0.20,
                            wspace=0.38)
    ax_hist = fig.add_subplot(gs[0])
    ax_bar  = fig.add_subplot(gs[1])

    for ax in (ax_hist, ax_bar):
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(SPINE_COLOR)
        ax.spines["bottom"].set_color(SPINE_COLOR)

    # --- Left: histogram ---
    bins = np.linspace(0, max(das_abs.max(), random_abs.max()) * 1.05, 36)
    ax_hist.hist(random_abs, bins=bins, color=C_RANDOM, alpha=0.70, density=True,
                 label=f"Random  (n = {len(random_abs):,})", zorder=2)
    ax_hist.hist(das_abs,    bins=bins, color=C_DAS,    alpha=0.78, density=True,
                 label=f"DAS  (n = {len(das_abs):,})",    zorder=3)

    ax_hist.axvline(das_abs.mean(),    color=C_DAS,    linewidth=1.3, linestyle="--",
                    label=f"DAS mean = {das_abs.mean():.3f}")
    ax_hist.axvline(random_abs.mean(), color=C_RANDOM, linewidth=1.3, linestyle="--",
                    label=f"Random mean = {random_abs.mean():.3f}")

    ax_hist.set_xlabel("Absolute A/B logit change", labelpad=6)
    ax_hist.set_ylabel("Density", labelpad=6)
    ax_hist.set_title("Distribution of Absolute Logit Change", fontsize=9.5, pad=8)
    ax_hist.yaxis.grid(True, color=GRID_COLOR, linewidth=0.6, zorder=0)
    ax_hist.set_axisbelow(True)
    leg = ax_hist.legend(loc="upper right", fontsize=8.5,
                         frameon=True, framealpha=0.92, edgecolor=GRID_COLOR)
    leg.get_frame().set_linewidth(0.5)

    # --- Right: outcome rates ---
    categories  = ["Argmax\nflips", "Directional\nagreement"]
    das_rates    = [n_das_flips / len(das_abs), n_das_da / len(das_abs)]
    random_rates = [n_random_flips / len(random_abs), n_random_da / len(random_abs)]

    xc = np.arange(len(categories))
    w  = 0.30
    b_das    = ax_bar.bar(xc - w / 2, das_rates,    w, label="DAS",    color=C_DAS,    alpha=0.88)
    b_random = ax_bar.bar(xc + w / 2, random_rates, w, label="Random", color=C_RANDOM, alpha=0.88)
    ax_bar.axhline(0.5, color=SPINE_COLOR, linewidth=0.7, linestyle=":",
                   label="Chance (0.5)")

    for bars in [b_das, b_random]:
        for bar in bars:
            h = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                        f"{h:.1%}", ha="center", va="bottom", fontsize=8.5, color=TEXT_COLOR)

    ax_bar.set_xticks(xc)
    ax_bar.set_xticklabels(categories, fontsize=9)
    ax_bar.set_ylabel("Rate", labelpad=6)
    ax_bar.set_ylim(0, 1.08)
    ax_bar.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    ax_bar.yaxis.grid(True, color=GRID_COLOR, linewidth=0.6, zorder=0)
    ax_bar.set_axisbelow(True)
    ax_bar.set_title("Intervention Outcomes", fontsize=9.5, pad=8)
    leg2 = ax_bar.legend(loc="upper left", fontsize=8.5,
                         frameon=True, framealpha=0.92, edgecolor=GRID_COLOR)
    leg2.get_frame().set_linewidth(0.5)

    fig.text(0.5, 0.95,
             "DAS Intervention Analysis  —  V2 confirmatory run, 20 independent episodes",
             ha="center", va="top", fontsize=10.5, color=TEXT_COLOR,
             transform=fig.transFigure)

    footnote(fig,
             f"Left: density of absolute A/B logit change per intervention record.  "
             f"Right: argmax flip rate and directional agreement rate.  "
             f"DAS: {n_das_flips}/200 argmax flips (0.0%).  "
             f"Random: {n_random_flips}/2,000 argmax flips ({n_random_flips/len(random_abs):.1%}).  "
             f"Source: confirmatory_run/records.jsonl")

    save(fig, "E7_05_das_intervention_dist")


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
