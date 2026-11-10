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
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE       = Path(__file__).resolve().parent
REPO_ROOT  = HERE.parent.parent
OUT        = HERE          # figures saved alongside this script

TRIALS_06  = Path("/Users/amelia/Desktop/llm experiment/exp5_0.6b/trials.jsonl")
TRIALS_17  = Path("/Users/amelia/Desktop/llm experiment/exp1-4/results/exp5/behavior_reflexion_1.7b_20x20/trials.jsonl")
SUMMARY_06 = REPO_ROOT / "Experiment_5/Qwen3_0.6B/behavior_summary.json"
SUMMARY_17 = REPO_ROOT / "Experiment_5/Qwen3_1.7B_20x20/behavior_summary.json"

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "grid.color":         "#e0e0e0",
    "grid.linewidth":     0.5,
    "legend.frameon":     False,
    "legend.fontsize":    10,
    "axes.titlesize":     12,
    "axes.titleweight":   "bold",
    "axes.labelsize":     11,
})

BLUE   = "#2166ac"
ORANGE = "#d6604d"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trials(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def choice_rate_by_trial(rows: list[dict]) -> dict:
    """
    Returns {trial_num: (mean_a_rate, se)} across all episodes.
    trial_num is 1-indexed.
    """
    from collections import defaultdict
    by_ep: dict[str, dict[int, int]] = defaultdict(dict)
    for r in rows:
        by_ep[r["episode_id"]][r["trial"]] = 1 if r["chosen_arm"] == "A" else 0

    trials_set = sorted({r["trial"] for r in rows})
    result = {}
    for t in trials_set:
        vals = [ep[t] for ep in by_ep.values() if t in ep]
        arr = np.array(vals, dtype=float)
        result[t] = (arr.mean(), arr.std() / np.sqrt(len(arr)))
    return result


def median_reversal(rows: list[dict]) -> float:
    revs = [r["reversal_trial"] for r in rows]
    return float(np.median(revs))

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

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, rate, rev, model, n_ep, color in [
        (axes[0], rate_06, rev_06, "Qwen3-0.6B", n_ep_06, BLUE),
        (axes[1], rate_17, rev_17, "Qwen3-1.7B", n_ep_17, ORANGE),
    ]:
        trials  = sorted(rate.keys())
        means   = np.array([rate[t][0] for t in trials])
        ses     = np.array([rate[t][1] for t in trials])

        ax.fill_between(trials, means - ses, means + ses, color=color, alpha=0.15)
        ax.plot(trials, means, color=color, linewidth=1.8, label="A-choice rate")
        ax.axvline(rev, color="#555555", linewidth=1.0, linestyle="--", alpha=0.8,
                   label=f"Reversal (trial {int(rev)})")
        ax.axhline(0.5, color="#aaaaaa", linewidth=0.6, linestyle=":")

        ax.set_title(model, fontweight="bold")
        ax.set_xlabel("Trial")
        ax.set_xlim(min(trials) - 0.5, max(trials) + 0.5)
        ax.set_ylim(-0.03, 1.03)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
        ax.legend(loc="upper right")

        n_trials = len(trials) * n_ep
        ax.text(0.02, 0.04, f"n = {n_ep} episodes, {n_trials} trials",
                transform=ax.transAxes, fontsize=9, color="#555555")

    axes[0].set_ylabel("A-choice rate")
    axes[0].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))

    fig.suptitle("Experiment 5 — Choice Proportion Over Trials", y=1.02, fontsize=13, fontweight="bold")
    fig.text(0.5, -0.02, "Choice proportion per trial position, averaged across episodes (shading ±1 SE).",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout()
    out_path = OUT / "E5_01_choice_by_trial.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure E5-02 — Pre/post-reversal performance comparison
# ---------------------------------------------------------------------------

def figure_reversal_performance() -> None:
    s06 = json.loads(SUMMARY_06.read_text())
    s17 = json.loads(SUMMARY_17.read_text())

    models  = ["Qwen3-0.6B\n(50 ep × 30 trials)", "Qwen3-1.7B\n(20 ep × 20 trials)"]
    pre_acc = [s06["optimal_choice_accuracy_pre_reversal"],
               s17["optimal_choice_accuracy_pre_reversal"]]
    post_acc = [s06["optimal_choice_accuracy_post_reversal"],
                s17["optimal_choice_accuracy_post_reversal"]]

    x     = np.arange(len(models))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7, 4.5))

    bars_pre  = ax.bar(x - width / 2, pre_acc,  width, label="Pre-reversal",  color=BLUE,   alpha=0.85)
    bars_post = ax.bar(x + width / 2, post_acc, width, label="Post-reversal", color=ORANGE, alpha=0.85)

    ax.axhline(0.5, color="#aaaaaa", linewidth=0.8, linestyle="--", label="Chance (0.5)")

    for bar in list(bars_pre) + list(bars_post):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                f"{h:.1%}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Optimal choice accuracy")
    ax.set_ylim(0, 0.80)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title("Experiment 5 — Pre- and Post-Reversal Performance", fontweight="bold")
    ax.legend(loc="upper right")

    ax.text(0.5, -0.14,
            "Optimal choice accuracy: fraction of trials where the model chose the currently higher-reward arm.\n"
            "Source: Experiment_5/Qwen3_0.6B/behavior_summary.json and Experiment_5/Qwen3_1.7B_20x20/behavior_summary.json",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#555555")

    fig.tight_layout()
    out_path = OUT / "E5_02_reversal_performance.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating Experiment 5 figures...")
    figure_choice_by_trial()
    figure_reversal_performance()
    print("Done.")
