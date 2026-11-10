"""
Experiment 5 — Figure Generation Script
========================================
Generates publication-quality figures from actual experimental data.

Data sources
------------
Qwen3-0.6B trial data : /Users/amelia/Desktop/llm experiment/exp5_0.6b/trials.jsonl
Qwen3-1.7B trial data : /Users/amelia/Desktop/llm experiment/exp1-4/results/exp5/behavior_reflexion_1.7b_20x20/trials.jsonl
Behaviour summaries   : Experiment_5/Qwen3_0.6B/behavior_summary.json
                        Experiment_5/Qwen3_1.7B_20x20/behavior_summary.json

Figures generated
-----------------
E5_01_choice_by_trial.png       A-choice rate per trial position, both models
E5_02_reversal_performance.png  Pre/post-reversal optimal accuracy comparison
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
HERE       = Path(__file__).resolve().parent
REPO_ROOT  = HERE.parent.parent
OUT        = HERE

TRIALS_06  = Path("/Users/amelia/Desktop/llm experiment/exp5_0.6b/trials.jsonl")
TRIALS_17  = Path("/Users/amelia/Desktop/llm experiment/exp1-4/results/exp5/behavior_reflexion_1.7b_20x20/trials.jsonl")
SUMMARY_06 = REPO_ROOT / "Experiment_5/Qwen3_0.6B/behavior_summary.json"
SUMMARY_17 = REPO_ROOT / "Experiment_5/Qwen3_1.7B_20x20/behavior_summary.json"

# ---------------------------------------------------------------------------
# Colour palette — consistent with Experiment 7 figures
# ---------------------------------------------------------------------------
TEXT_COLOR  = "#353047"
GRID_COLOR  = "#E8E3EF"
SPINE_COLOR = "#CCCCCC"

C_06B = "#2D6A9F"   # steel blue  — Qwen3-0.6B
C_17B = "#C67E3A"   # warm amber  — Qwen3-1.7B
C_PRE  = "#2D6A9F"  # steel blue  — pre-reversal
C_POST = "#C67E3A"  # warm amber  — post-reversal

# ---------------------------------------------------------------------------
# Global rcParams
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family":          "DejaVu Sans",
    "font.size":            11,
    "text.color":           TEXT_COLOR,
    "axes.labelcolor":      TEXT_COLOR,
    "xtick.color":          TEXT_COLOR,
    "ytick.color":          TEXT_COLOR,
    "axes.edgecolor":       SPINE_COLOR,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.linewidth":       0.8,
    "xtick.major.width":    0.8,
    "ytick.major.width":    0.8,
    "xtick.minor.visible":  False,
    "ytick.minor.visible":  False,
    "figure.facecolor":     "white",
    "axes.facecolor":       "white",
    "grid.color":           GRID_COLOR,
    "grid.linewidth":       0.6,
    "legend.frameon":       False,
    "legend.fontsize":      10,
    "axes.titlesize":       11,
    "axes.titleweight":     "bold",
    "axes.labelsize":       11,
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "svg"):
        path = OUT / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {path}")


def footnote(fig: plt.Figure, text: str, y: float = 0.025) -> None:
    fig.text(0.5, y, text, ha="center", fontsize=8.5,
             style="italic", color="#888888", wrap=True)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trials(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def choice_rate_by_trial(rows: list[dict]) -> dict:
    """Returns {trial_num: (mean_a_rate, se)} across all episodes."""
    from collections import defaultdict
    by_ep: dict[str, dict[int, int]] = defaultdict(dict)
    for r in rows:
        by_ep[r["episode_id"]][int(r["trial"])] = 1 if r["chosen_arm"] == "A" else 0

    trials_set = sorted({int(r["trial"]) for r in rows})
    result = {}
    for t in trials_set:
        vals = [ep[t] for ep in by_ep.values() if t in ep]
        arr = np.array(vals, dtype=float)
        result[t] = (arr.mean(), arr.std() / np.sqrt(len(arr)))
    return result


def median_reversal(rows: list[dict]) -> float:
    return float(np.median([r["reversal_trial"] for r in rows]))

# ---------------------------------------------------------------------------
# Figure E5-01 — A-choice rate by trial position
# ---------------------------------------------------------------------------

def figure_choice_by_trial() -> None:
    rows_06 = load_trials(TRIALS_06)
    rows_17 = load_trials(TRIALS_17)

    rate_06 = choice_rate_by_trial(rows_06)
    rate_17 = choice_rate_by_trial(rows_17)

    rev_06 = median_reversal(rows_06)
    rev_17 = median_reversal(rows_17)

    n_ep_06 = len({r["episode_id"] for r in rows_06})
    n_ep_17 = len({r["episode_id"] for r in rows_17})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.84, bottom=0.20, wspace=0.10)

    for ax, rate, rev, model, n_ep, color in [
        (axes[0], rate_06, rev_06, "Qwen3-0.6B", n_ep_06, C_06B),
        (axes[1], rate_17, rev_17, "Qwen3-1.7B", n_ep_17, C_17B),
    ]:
        trials = sorted(rate.keys())
        means  = np.array([rate[t][0] for t in trials])
        ses    = np.array([rate[t][1] for t in trials])

        ax.fill_between(trials, means - ses, means + ses, color=color, alpha=0.15)
        ax.plot(trials, means, color=color, linewidth=1.8)
        ax.axvline(rev, color="#888888", linewidth=1.0, linestyle="--",
                   label=f"Reversal (trial {int(rev)})")
        ax.axhline(0.5, color=SPINE_COLOR, linewidth=0.7, linestyle=":")

        ax.set_title(model, color=TEXT_COLOR)
        ax.set_xlabel("Trial", color=TEXT_COLOR)
        ax.set_xlim(min(trials) - 0.5, max(trials) + 0.5)
        ax.set_ylim(-0.03, 1.03)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=6))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
        ax.grid(axis="y", zorder=0)
        ax.legend(loc="upper right", fontsize=9)

        n_trials = len(trials) * n_ep
        ax.text(0.03, 0.04, f"n = {n_ep} episodes, {n_trials} trials",
                transform=ax.transAxes, fontsize=9, color="#888888")

    axes[0].set_ylabel("A-choice rate", color=TEXT_COLOR)

    # Shared A-choice rate colour swatch in figure legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=C_06B, linewidth=1.8, label="Qwen3-0.6B A-choice rate"),
        Line2D([0], [0], color=C_17B, linewidth=1.8, label="Qwen3-1.7B A-choice rate"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.97),
               ncol=2, fontsize=9, frameon=False)

    fig.text(0.5, 0.95,
             "Experiment 5 — Choice Proportion Over Trials",
             ha="center", fontsize=13, fontweight="bold", color=TEXT_COLOR)

    footnote(fig,
             "A-choice rate per trial position, averaged across episodes (shading ±1 SE).  "
             "Source: trials.jsonl",
             y=0.025)

    save(fig, "E5_01_choice_by_trial")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure E5-02 — Pre/post-reversal performance comparison
# ---------------------------------------------------------------------------

def figure_reversal_performance() -> None:
    s06 = json.loads(SUMMARY_06.read_text())
    s17 = json.loads(SUMMARY_17.read_text())

    models   = ["Qwen3-0.6B\n(50 ep × 30 trials)", "Qwen3-1.7B\n(20 ep × 20 trials)"]
    pre_acc  = [s06["optimal_choice_accuracy_pre_reversal"],
                s17["optimal_choice_accuracy_pre_reversal"]]
    post_acc = [s06["optimal_choice_accuracy_post_reversal"],
                s17["optimal_choice_accuracy_post_reversal"]]

    x     = np.arange(len(models))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7, 4.8))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.86, bottom=0.26)

    bars_pre  = ax.bar(x - width / 2, pre_acc,  width, label="Pre-reversal",  color=C_PRE,  alpha=0.88)
    bars_post = ax.bar(x + width / 2, post_acc, width, label="Post-reversal", color=C_POST, alpha=0.88)

    ax.axhline(0.5, color=SPINE_COLOR, linewidth=0.9, linestyle="--", label="Chance (0.5)")
    ax.grid(axis="y", zorder=0)

    for bar in list(bars_pre) + list(bars_post):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                f"{h:.1%}", ha="center", va="bottom", fontsize=9, color=TEXT_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Optimal choice accuracy")
    ax.set_ylim(0, 0.80)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title("Experiment 5 — Pre- and Post-Reversal Performance",
                 color=TEXT_COLOR, pad=8)
    ax.legend(loc="upper right")

    footnote(fig,
             "Optimal choice accuracy: fraction of trials where the model chose the currently higher-reward arm.  "
             "Source: Qwen3_0.6B/behavior_summary.json and Qwen3_1.7B_20x20/behavior_summary.json",
             y=0.025)

    save(fig, "E5_02_reversal_performance")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating Experiment 5 figures...")
    figure_choice_by_trial()
    figure_reversal_performance()
    print("Done.")
