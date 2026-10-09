#!/usr/bin/env python3
"""Exp 5 — Reflexion-Assisted Sequential Bandit (Phase 1: behavioural feasibility).

Inherits the full Exp 4 environment and measurement pipeline unchanged.
The only experimental manipulation is trial-by-trial self-reflection.

Command: behavior_reflexion
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from causal_rpe_bandit import load_model, model_device, seed_everything
from sequential_bandit_rpe import (
    BINS, BIN_LABELS, SYSTEM,
    calibrated_logits, corr, dump, estimate_value,
    feedback_readout_raw, history_text, choice_raw,
    logits_for, prepared, save_jsonl, softmax, value_raw,
)


# ---------------------------------------------------------------------------
# New functions
# ---------------------------------------------------------------------------

def choice_raw_with_memory(
    history: list[tuple[str, int]],
    reflection_memory: list[str],
    choice_order: str = "AB",
) -> str:
    """Extend choice_raw() with an optional reflection-memory block.

    Returns exactly the same string as choice_raw() when memory is empty,
    so the Exp 4 baseline is recovered by passing an empty list.
    """
    shown = "A/B" if choice_order == "AB" else "B/A"
    base = f"{SYSTEM}\n\nHistory (trial option outcome):\n{history_text(history)}"
    if not reflection_memory:
        return f"{base}\n\nNEXT_CHOICE ({shown}):"
    reflections = "\n".join(f"[{i + 1}] {r}" for i, r in enumerate(reflection_memory))
    return f"{base}\n\nPrevious reflections:\n{reflections}\n\nNEXT_CHOICE ({shown}):"


def reflection_raw(
    history_after: list[tuple[str, int]],
    chosen: str,
    reward: int,
) -> str:
    """Open-ended post-trial reflection prompt."""
    return (
        f"{SYSTEM}\n\nTrial history so far:\n{history_text(history_after)}\n\n"
        f"You chose {chosen} and received a reward of {reward}.\n"
        "Briefly reflect on what, if anything, this outcome suggests for your next decision.\n"
        "REFLECTION:"
    )


def generate_reflection(
    torch: Any,
    model: Any,
    tokenizer: Any,
    raw: str,
    max_new_tokens: int = 60,
) -> str:
    """Greedy generation using the same chat-template wrapper as logits_for()."""
    prompt = prepared(tokenizer, raw)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {k: v.to(model_device(model)) for k, v in encoded.items()}
    input_len = encoded["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, input_len:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Phase 1 command
# ---------------------------------------------------------------------------

def command_behavior_reflexion(args: argparse.Namespace) -> None:
    env = json.loads((Path(args.environment_dir) / "environments.json").read_text())
    reversal_lookup = {ep["episode_id"]: ep["reversal_trial"] for ep in env["episodes"]}
    torch, model, tokenizer = load_model(args.model, args.device, args.dtype)
    seed_everything(args.seed, torch)
    rng = np.random.default_rng(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl_path = out / "trials.jsonl"

    # Resume from any previously completed episodes so an interrupted run can continue.
    completed_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    if jsonl_path.exists():
        rows = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l]
        completed_ids = {r["episode_id"] for r in rows}
        print(f"Resuming: {len(completed_ids)} episodes already in {jsonl_path}")

    for ei, episode in enumerate(env["episodes"], 1):
        if episode["episode_id"] in completed_ids:
            print(f"behavior_reflexion episode {ei}/{len(env['episodes'])} [skipped — already done]")
            continue
        episode_rows: list[dict[str, Any]] = []
        history: list[tuple[str, int]] = []
        reflection_memory: list[str] = []
        reversal_trial = reversal_lookup[episode["episode_id"]]
        bin_values = rng.permutation(BINS)
        baseline_logits = {
            arm: logits_for(torch, model, tokenizer, value_raw([], arm, bin_values), BIN_LABELS)
            for arm in ("A", "B")
        }
        choice_baseline_logits = (
            logits_for(torch, model, tokenizer, choice_raw([], args.choice_order), ("A", "B"))
            if args.calibrate_choice_prior else None
        )

        for item in episode["schedule"]:
            memory_snapshot = list(reflection_memory)
            choice_logits_raw = logits_for(
                torch, model, tokenizer,
                choice_raw_with_memory(history, reflection_memory, args.choice_order),
                ("A", "B"),
            )
            choice_logits = calibrated_logits(choice_logits_raw, choice_baseline_logits)
            choice_probs = softmax(choice_logits / args.choice_temperature)
            exploratory = bool(rng.random() < args.epsilon)
            if exploratory:
                chosen = ("A", "B")[int(rng.integers(2))]
            elif args.sample_choices:
                chosen = ("A", "B")[int(rng.choice(2, p=choice_probs))]
            else:
                chosen = ("A", "B")[int(np.argmax(choice_logits))]

            prior = baseline_logits[chosen] if args.calibrate_label_prior else None
            q_before, q_before_probs, q_before_raw, q_before_raw_probs = estimate_value(
                torch, model, tokenizer, history, chosen, bin_values, prior
            )
            reward = int(item[f"potential_{chosen}"])
            rpe_model = reward - q_before
            true_p = float(item[f"p_{chosen}"])
            history_after = history + [(chosen, reward)]
            q_after, q_after_probs, q_after_raw, q_after_raw_probs = estimate_value(
                torch, model, tokenizer, history_after, chosen, bin_values, prior
            )

            reflection_text = generate_reflection(
                torch, model, tokenizer,
                reflection_raw(history_after, chosen, reward),
                max_new_tokens=args.max_reflection_tokens,
            )
            if not args.discard_reflection:
                reflection_memory.append(reflection_text)

            episode_rows.append({
                "episode_id": episode["episode_id"],
                "trial": item["trial"],
                "chosen_arm": chosen,
                "reward": reward,
                "true_p": true_p,
                "choice_logit_A": float(choice_logits[0]),
                "choice_logit_B": float(choice_logits[1]),
                "choice_prob_A": float(choice_probs[0]),
                "choice_prob_B": float(choice_probs[1]),
                "choice_logit_A_raw": float(choice_logits_raw[0]),
                "choice_logit_B_raw": float(choice_logits_raw[1]),
                "choice_baseline_logit_A": (float(choice_baseline_logits[0])
                                             if choice_baseline_logits is not None else None),
                "choice_baseline_logit_B": (float(choice_baseline_logits[1])
                                             if choice_baseline_logits is not None else None),
                "exploratory_choice": exploratory,
                "bin_values_by_label": bin_values.tolist(),
                "q_model_before": q_before,
                "q_model_after": q_after,
                "model_update": q_after - q_before,
                "rpe_model": rpe_model,
                "q_model_before_raw": q_before_raw,
                "q_model_after_raw": q_after_raw,
                "raw_model_update": q_after_raw - q_before_raw,
                "q_before_distribution": q_before_probs,
                "q_after_distribution": q_after_probs,
                "q_before_distribution_raw": q_before_raw_probs,
                "q_after_distribution_raw": q_after_raw_probs,
                "history": history,
                "feedback_prompt": feedback_readout_raw(history, chosen, reward, bin_values)[0],
                "reversal_trial": reversal_trial,
                "is_post_reversal": item["trial"] >= reversal_trial,
                "reflection_text": reflection_text,
                "reflection_memory_before": memory_snapshot,
                "n_reflections_used": len(memory_snapshot),
            })
            history = history_after

        # Flush completed episode to disk immediately so progress survives interruption.
        with jsonl_path.open("a", encoding="utf-8") as f:
            for row in episode_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows.extend(episode_rows)
        print(f"behavior_reflexion episode {ei}/{len(env['episodes'])}", flush=True)

    # Gate computation — identical to command_behavior in sequential_bandit_rpe.py
    rpe = np.array([r["rpe_model"] for r in rows])
    update = np.array([r["model_update"] for r in rows])
    post_first = np.array([r["trial"] > 1 for r in rows])
    gate_rpe, gate_update = rpe[post_first], update[post_first]
    positive = rpe > 0; negative = rpe < 0
    gate_positive = gate_rpe > 0; gate_negative = gate_rpe < 0
    direction_accuracy = float(np.mean(np.where(gate_positive, gate_update > 0, gate_update < 0)))
    chosen_a = np.array([r["chosen_arm"] == "A" for r in rows])
    optimal = np.array([(r["chosen_arm"] == "A") == (r["true_p"] > 0.5) for r in rows])
    gate_corr = corr(gate_rpe, gate_update)

    # Reversal-phase breakdown
    pre_mask = np.array([not r["is_post_reversal"] for r in rows])
    post_mask = np.array([r["is_post_reversal"] for r in rows])
    rpe_pre, update_pre = rpe[pre_mask], update[pre_mask]
    rpe_post, update_post = rpe[post_mask], update[post_mask]
    optimal_pre, optimal_post = optimal[pre_mask], optimal[post_mask]
    chosen_a_pre, chosen_a_post = chosen_a[pre_mask], chosen_a[post_mask]

    summary: dict[str, Any] = {
        "model": args.model,
        "n_trials": len(rows),
        "episodes": len(env["episodes"]),
        "choice_order": args.choice_order,
        "label_prior_calibration": args.calibrate_label_prior,
        "choice_prior_calibration": args.calibrate_choice_prior,
        "discard_reflection": args.discard_reflection,
        "choice_a_count": int(chosen_a.sum()),
        "choice_b_count": int((~chosen_a).sum()),
        "optimal_choice_accuracy": float(optimal.mean()),
        "rpe_update_correlation": corr(rpe, update),
        "rpe_update_correlation_excluding_trial_1": gate_corr,
        "mean_positive_rpe_update": float(update[positive].mean()) if np.any(positive) else None,
        "median_positive_rpe_update": float(np.median(update[positive])) if np.any(positive) else None,
        "mean_negative_rpe_update": float(update[negative].mean()) if np.any(negative) else None,
        "median_negative_rpe_update": float(np.median(update[negative])) if np.any(negative) else None,
        "signed_update_accuracy": float(np.mean(np.where(positive, update > 0, update < 0))),
        "signed_update_accuracy_excluding_trial_1": direction_accuracy,
        "positive_update_accuracy": float(np.mean(update[positive] > 0)) if np.any(positive) else None,
        "negative_update_accuracy": float(np.mean(update[negative] < 0)) if np.any(negative) else None,
        "behavioral_readout_passed": bool(
            gate_corr >= args.min_rpe_update_correlation
            and gate_update[gate_positive].mean() > 0
            and gate_update[gate_negative].mean() < 0
            and direction_accuracy >= args.min_direction_accuracy
        ),
        "gate_thresholds": {
            "min_rpe_update_correlation": args.min_rpe_update_correlation,
            "min_direction_accuracy": args.min_direction_accuracy,
            "evaluation_subset": "trials after trial 1",
        },
        "optimal_choice_accuracy_pre_reversal": float(optimal_pre.mean()) if pre_mask.any() else None,
        "optimal_choice_accuracy_post_reversal": float(optimal_post.mean()) if post_mask.any() else None,
        "rpe_update_correlation_pre_reversal": corr(rpe_pre, update_pre),
        "rpe_update_correlation_post_reversal": corr(rpe_post, update_post),
        "choice_a_rate_pre_reversal": float(chosen_a_pre.mean()) if pre_mask.any() else None,
        "choice_a_rate_post_reversal": float(chosen_a_post.mean()) if post_mask.any() else None,
        "behavior_gate": "Continue to activation collection only if behavioral_readout_passed is true.",
    }
    dump(out / "behavior_summary.json", summary)
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="command", required=True)
    x = s.add_parser("behavior_reflexion")
    x.add_argument("--environment-dir", default="results/sequential/environments")
    x.add_argument("--model", default="Qwen/Qwen3-1.7B")
    x.add_argument("--output-dir", default="results/exp5/behavior_reflexion")
    x.add_argument("--choice-order", choices=["AB", "BA"], default="AB",
                   help="Display NEXT_CHOICE as A/B or B/A while always scoring the same A and B tokens")
    x.add_argument("--sample-choices", action="store_true")
    x.add_argument("--choice-temperature", type=float, default=1.0)
    x.add_argument("--epsilon", type=float, default=0.1,
                   help="Forced exploration probability")
    x.add_argument("--calibrate-choice-prior", action=argparse.BooleanOptionalAction, default=True,
                   help="Subtract the identical no-history A/B logits before choosing")
    x.add_argument("--calibrate-label-prior", action=argparse.BooleanOptionalAction, default=True,
                   help="Subtract no-history logits for the same randomized label map")
    x.add_argument("--discard-reflection", action="store_true",
                   help="Generate reflection but do not store in memory (Phase 2 C2 discard control)")
    x.add_argument("--max-reflection-tokens", type=int, default=60,
                   help="Maximum new tokens per reflection generation")
    x.add_argument("--min-rpe-update-correlation", type=float, default=0.30)
    x.add_argument("--min-direction-accuracy", type=float, default=0.60)
    x.add_argument("--device", default="auto")
    x.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    x.add_argument("--seed", type=int, default=42)
    x.set_defaults(func=command_behavior_reflexion)
    return p


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
