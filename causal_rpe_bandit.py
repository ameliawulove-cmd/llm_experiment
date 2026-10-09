#!/usr/bin/env python3
"""Causal search for reward-prediction-error representations in an LLM.

This script has three subcommands:

  collect  Create controlled expectation/outcome examples and save residual
           stream activations plus forced-choice behavioral logits.
  analyze  Discover layer-wise RPE directions with grouped cross-validation,
           test nuisance variables and held-out template generalization.
  patch    Remove, inject, or swap a discovered direction and measure its
           causal effect on the model's update/next-choice logits.

The experiment studies an inference-time, in-context computation. It does not
update model weights and does not imply biological equivalence.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TEMPLATES = {
    "numeric": (
        "You are tracking a two-option reward task. The chosen option is {arm}. "
        "Before observing the result, its estimated reward probability was {p:.2f}. "
        "The observed outcome is {reward}, where 1 means rewarded and 0 means not rewarded.\n"
        "Based only on prediction error, should the estimated value of {arm} increase or decrease?\n"
        "Answer exactly {positive_label} for increase or {negative_label} for decrease.\nAnswer:"
    ),
    "symbolic": (
        "In a repeated choice task, option {arm} was expected to succeed with probability {p:.2f}. "
        "For this block, X means rewarded and Y means unrewarded. The result symbol is {symbol}.\n"
        "Should the value assigned to option {arm} go up or down?\n"
        "Answer exactly {positive_label} for up or {negative_label} for down.\nAnswer:"
    ),
    "verbal": (
        "A decision maker selected {arm}. Its prior chance of paying off was {p:.2f}. "
        "It {verbal_outcome}.\n"
        "Should the estimated value of {arm} rise or fall?\n"
        "Answer exactly {positive_label} for rise or {negative_label} for fall.\nAnswer:"
    ),
}

BEHAVIOR_TEMPLATES = {
    "numeric": (
        "The chosen option is {arm}. Its estimated reward probability before feedback was {p:.2f}. "
        "The observed outcome is {reward}, where 1 means rewarded and 0 means unrewarded."
    ),
    "symbolic": (
        "The chosen option is {arm}. Its estimated reward probability before feedback was {p:.2f}. "
        "X means rewarded and Y means unrewarded. The observed result symbol is {symbol}."
    ),
    "verbal": (
        "The chosen option is {arm}. Its estimated reward probability before feedback was {p:.2f}. "
        "The option {verbal_outcome}."
    ),
}

BEHAVIOR_INSTRUCTIONS = """
Compute signed reward prediction error as:
RPE = numerical outcome - expected reward probability.

If RPE is greater than zero, its sign is POSITIVE and the selected option's value should INCREASE.
If RPE is less than zero, its sign is NEGATIVE and the selected option's value should DECREASE.

Examples:
- Expected probability 0.80, outcome 1: RPE = +0.20, so POSITIVE and INCREASE.
- Expected probability 0.80, outcome 0: RPE = -0.80, so NEGATIVE and DECREASE.

Solve the new case below. You may reason first, but finish with exactly these two lines:
RPE_SIGN: POSITIVE or NEGATIVE
UPDATE: INCREASE or DECREASE
""".strip()


@dataclass(frozen=True)
class Example:
    example_id: str
    episode_id: str
    template: str
    arm: str
    history: str
    expected_p: float
    reward: int
    rpe: float
    abs_rpe: float
    prompt: str
    readout_prefix: str
    positive_word: str
    negative_word: str


def require_ml() -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing model dependencies. Install with: "
            "pip install -r requirements-causal-rpe.txt"
        ) from exc
    return torch, (AutoModelForCausalLM, AutoTokenizer)


def seed_everything(seed: int, torch: Any | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def build_prompt(
    template: str,
    arm: str,
    p: float,
    reward: int,
    positive_label: str,
    negative_label: str,
    history: str = "",
) -> str:
    task = TEMPLATES[template].format(
        arm=arm,
        p=p,
        reward=reward,
        symbol="X" if reward else "Y",
        verbal_outcome="produced a reward" if reward else "produced no reward",
        positive_label=positive_label,
        negative_label=negative_label,
    )
    return f"Recent task history: {history}\n{task}" if history else task


def build_examples(
    episodes: int,
    templates: list[str],
    probabilities: list[float],
    seed: int,
    structured_readout: bool = False,
) -> list[Example]:
    """Create balanced cells with episode groups for leakage-safe CV."""
    rng = random.Random(seed)
    examples: list[Example] = []
    for episode in range(episodes):
        arm_map = ["K", "M"]
        rng.shuffle(arm_map)
        history_items = []
        for h in range(rng.randint(3, 8)):
            h_arm = rng.choice(arm_map)
            h_reward = rng.randint(0, 1)
            history_items.append(f"t{h + 1}:{h_arm}->{h_reward}")
        history = ", ".join(history_items)
        cells = [(t, p, r) for t in templates for p in probabilities for r in (0, 1)]
        rng.shuffle(cells)
        for j, (template, p, reward) in enumerate(cells):
            arm = arm_map[j % 2]
            if structured_readout:
                positive_label, negative_label = "A", "B"  # unused by this prompt
                pos, neg = " P", " N"
                prompt = (
                    f"Recent task history: {history}\n\n"
                    f"{build_behavior_prompt(template, arm, p, reward)}"
                )
                readout_prefix = "RPE_SIGN (P=POSITIVE, N=NEGATIVE):"
            else:
                # Counterbalance labels so an A/B preference cannot masquerade
                # as correct value updating.
                if rng.random() < 0.5:
                    positive_label, negative_label = "A", "B"
                else:
                    positive_label, negative_label = "B", "A"
                pos, neg = f" {positive_label}", f" {negative_label}"
                prompt = build_prompt(
                    template, arm, p, reward,
                    positive_label, negative_label, history,
                )
                readout_prefix = ""
            examples.append(
                Example(
                    example_id=f"e{episode:04d}_{j:03d}",
                    episode_id=f"e{episode:04d}",
                    template=template,
                    arm=arm,
                    history=history,
                    expected_p=float(p),
                    reward=reward,
                    rpe=float(reward - p),
                    abs_rpe=float(abs(reward - p)),
                    prompt=prompt,
                    readout_prefix=readout_prefix,
                    positive_word=pos,
                    negative_word=neg,
                )
            )
    return examples


def build_behavior_prompt(template: str, arm: str, p: float, reward: int) -> str:
    case = BEHAVIOR_TEMPLATES[template].format(
        arm=arm,
        p=p,
        reward=reward,
        symbol="X" if reward else "Y",
        verbal_outcome="produced a reward" if reward else "produced no reward",
    )
    return f"{BEHAVIOR_INSTRUCTIONS}\n\nNew case:\n{case}"


def model_device(model: Any) -> Any:
    return next(model.parameters()).device


def load_model(model_id: str, device: str, dtype: str) -> tuple[Any, Any, Any]:
    torch, classes = require_ml()
    AutoModelForCausalLM, AutoTokenizer = classes
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if dtype == "auto":
        torch_dtype = "auto"
    else:
        torch_dtype = getattr(torch, dtype)

    kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
    if device == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if device != "auto":
        model.to(device)
    model.eval()
    return torch, model, tokenizer


def token_id_for_completion(tokenizer: Any, prompt: str, completion: str) -> int:
    """Require a one-token readout after the exact prompt context."""
    base = tokenizer(prompt, add_special_tokens=False).input_ids
    full = tokenizer(prompt + completion, add_special_tokens=False).input_ids
    suffix = full[len(base):]
    if len(suffix) != 1:
        standalone = tokenizer(completion, add_special_tokens=False).input_ids
        if len(standalone) == 1:
            return int(standalone[0])
        raise ValueError(
            f"Readout {completion!r} is not one token for this tokenizer; got {suffix or standalone}. "
            "Choose different readout words or model."
        )
    return int(suffix[0])


def apply_chat_template(tokenizer: Any, text: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            # Qwen3 otherwise begins with a reasoning sequence; first-token
            # A/B logits would not represent its answer decision.
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                enable_thinking=False,
                **kwargs,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                **kwargs,
            )
    return text


def apply_generation_chat_template(tokenizer: Any, text: str, enable_thinking: bool) -> str:
    if not getattr(tokenizer, "chat_template", None):
        return text
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            enable_thinking=enable_thinking,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            **kwargs,
        )


def parse_behavior_answer(text: str) -> tuple[str | None, str | None]:
    sign_matches = re.findall(r"RPE[_ ]?SIGN\s*[:=]\s*(POSITIVE|NEGATIVE)", text, re.I)
    update_matches = re.findall(r"UPDATE\s*[:=]\s*(INCREASE|DECREASE)", text, re.I)
    sign = sign_matches[-1].upper() if sign_matches else None
    update = update_matches[-1].upper() if update_matches else None
    return sign, update


def command_behavior_test(args: argparse.Namespace) -> None:
    torch, model, tokenizer = load_model(args.model, args.device, args.dtype)
    seed_everything(args.seed, torch)
    rng = random.Random(args.seed)
    conditions = [
        (template, p, reward, repeat)
        for repeat in range(args.repeats)
        for template in args.templates
        for p in args.probabilities
        for reward in (0, 1)
    ]
    rng.shuffle(conditions)
    rows: list[dict[str, Any]] = []

    for i, (template, p, reward, repeat) in enumerate(conditions, 1):
        arm = rng.choice(["K", "M"])
        raw_prompt = build_behavior_prompt(template, arm, p, reward)
        prompt = apply_generation_chat_template(tokenizer, raw_prompt, args.enable_thinking)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {k: v.to(model_device(model)) for k, v in encoded.items()}
        prompt_len = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True)
        predicted_sign, predicted_update = parse_behavior_answer(answer)
        expected_sign = "POSITIVE" if reward - p > 0 else "NEGATIVE"
        expected_update = "INCREASE" if reward - p > 0 else "DECREASE"
        parse_ok = predicted_sign is not None and predicted_update is not None
        sign_correct = predicted_sign == expected_sign
        update_correct = predicted_update == expected_update
        rows.append({
            "example_id": f"behavior_{i:03d}",
            "repeat": repeat,
            "template": template,
            "arm": arm,
            "expected_p": p,
            "reward": reward,
            "rpe": reward - p,
            "expected_sign": expected_sign,
            "expected_update": expected_update,
            "predicted_sign": predicted_sign,
            "predicted_update": predicted_update,
            "parse_ok": parse_ok,
            "sign_correct": sign_correct,
            "update_correct": update_correct,
            "both_correct": sign_correct and update_correct,
            "prompt": raw_prompt,
            "answer": answer,
        })
        status = "PASS" if rows[-1]["both_correct"] else "FAIL"
        print(
            f"[{i:02d}/{len(conditions)}] {status} {template:8s} "
            f"p={p:.1f} r={reward} expected={expected_sign}/{expected_update} "
            f"got={predicted_sign}/{predicted_update}"
        )

    def summarize(subset: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(subset),
            "parse_rate": float(np.mean([r["parse_ok"] for r in subset])),
            "sign_accuracy": float(np.mean([r["sign_correct"] for r in subset])),
            "update_accuracy": float(np.mean([r["update_correct"] for r in subset])),
            "both_accuracy": float(np.mean([r["both_correct"] for r in subset])),
        }

    summary = {
        "model": args.model,
        "enable_thinking": args.enable_thinking,
        "max_new_tokens": args.max_new_tokens,
        "overall": summarize(rows),
        "by_template": {
            template: summarize([r for r in rows if r["template"] == template])
            for template in args.templates
        },
        "recommended_for_activation_collection": bool(
            summarize(rows)["both_accuracy"] >= args.min_accuracy
            and all(
                summarize([r for r in rows if r["template"] == template])["both_accuracy"]
                >= args.min_accuracy
                for template in args.templates
            )
        ),
        "minimum_accuracy": args.min_accuracy,
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "behavior_examples.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "behavior_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n" + json.dumps(summary, indent=2))
    print(f"Saved behavior-only results to {out}")


def capture_example(torch: Any, model: Any, tokenizer: Any, ex: Example) -> tuple[np.ndarray, dict[str, float]]:
    prompt = apply_chat_template(tokenizer, ex.prompt) + ex.readout_prefix
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {k: v.to(model_device(model)) for k, v in encoded.items()}
    pos_id = token_id_for_completion(tokenizer, prompt, ex.positive_word)
    neg_id = token_id_for_completion(tokenizer, prompt, ex.negative_word)
    with torch.inference_mode():
        out = model(**encoded, output_hidden_states=True, return_dict=True)
    residual = np.stack(
        [h[0, -1].detach().float().cpu().numpy() for h in out.hidden_states], axis=0
    ).astype(np.float32)
    logits = out.logits[0, -1].detach().float()
    pos_logit = float(logits[pos_id].cpu())
    neg_logit = float(logits[neg_id].cpu())
    return residual, {
        "positive_token_id": pos_id,
        "negative_token_id": neg_id,
        "positive_logit": pos_logit,
        "negative_logit": neg_logit,
        "update_logit_diff": pos_logit - neg_logit,
    }


def command_collect(args: argparse.Namespace) -> None:
    torch, model, tokenizer = load_model(args.model, args.device, args.dtype)
    seed_everything(args.seed, torch)
    examples = build_examples(
        args.episodes, args.templates, args.probabilities, args.seed,
        structured_readout=args.structured_readout,
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    activations: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for i, ex in enumerate(examples, 1):
        act, readout = capture_example(torch, model, tokenizer, ex)
        activations.append(act)
        rows.append({**asdict(ex), **readout})
        if i % 20 == 0 or i == len(examples):
            print(f"Captured {i}/{len(examples)}")

    arr = np.stack(activations, axis=0)
    np.save(out / "residual_stream.npy", arr)
    with (out / "examples.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    metadata = {
        "experiment_version": 3 if args.structured_readout else 2,
        "readout_mode": "structured_rpe_sign" if args.structured_readout else "counterbalanced_ab",
        "model": args.model,
        "seed": args.seed,
        "shape": list(arr.shape),
        "dimensions": ["example", "hidden_state_index", "residual_dimension"],
        "note": "hidden state index 0 is embedding output; subsequent indices follow transformer blocks",
        "templates": args.templates,
        "probabilities": args.probabilities,
        "episodes": args.episodes,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved dataset to {out} with shape {arr.shape}")


def load_dataset(data_dir: str) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    root = Path(data_dir)
    acts = np.load(root / "residual_stream.npy", mmap_mode="r")
    rows = [json.loads(line) for line in (root / "examples.jsonl").read_text().splitlines() if line]
    meta = json.loads((root / "metadata.json").read_text())
    if len(rows) != acts.shape[0]:
        raise ValueError("examples.jsonl and residual_stream.npy have different example counts")
    return acts, rows, meta


def grouped_folds(groups: np.ndarray, n_splits: int, seed: int) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    unique = np.unique(groups)
    if len(unique) < n_splits:
        raise ValueError(f"Need at least {n_splits} episodes; found {len(unique)}")
    rng = np.random.default_rng(seed)
    unique = rng.permutation(unique)
    chunks = np.array_split(unique, n_splits)
    for test_groups in chunks:
        test = np.isin(groups, test_groups)
        yield np.flatnonzero(~test), np.flatnonzero(test)


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    scale = X.std(axis=0) + 1e-6
    Z = (X - mean) / scale
    yc = y - y.mean()
    # Use the dual form when residual width exceeds the number of examples.
    # This is dramatically cheaper for small-model activations (often 1k+
    # dimensions) and bandit datasets with only hundreds of observations.
    if Z.shape[1] > Z.shape[0]:
        kernel = Z @ Z.T
        dual = np.linalg.solve(kernel + alpha * np.eye(kernel.shape[0]), yc)
        w_std = Z.T @ dual
    else:
        gram = Z.T @ Z
        w_std = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), Z.T @ yc)
    w = w_std / scale
    intercept = float(y.mean() - mean @ w)
    return w.astype(np.float32), intercept, mean.astype(np.float32), scale.astype(np.float32)


def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - np.sum((y - pred) ** 2) / denom) if denom else float("nan")


def pearson(y: np.ndarray, pred: np.ndarray) -> float:
    if np.std(y) == 0 or np.std(pred) == 0:
        return float("nan")
    return float(np.corrcoef(y, pred)[0, 1])


def behavioral_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate whether the model's update readout follows signed RPE."""
    result: dict[str, Any] = {}
    templates = ["all", *sorted({r["template"] for r in rows})]
    for template in templates:
        subset = rows if template == "all" else [r for r in rows if r["template"] == template]
        rpe = np.array([r["rpe"] for r in subset], dtype=float)
        readout = np.array([r["update_logit_diff"] for r in subset], dtype=float)
        # update_logit_diff is always logit(correct-positive-label) minus
        # logit(correct-negative-label), independent of whether A or B was
        # assigned to the positive response in this example.
        result[template] = {
            "n": len(subset),
            "raw_zero_threshold_accuracy": float(np.mean((readout > 0) == (rpe > 0))),
            # P and N can have different unconditional token priors. In the
            # balanced design, subtracting the mean removes this fixed lexical
            # offset while preserving trial-dependent changes.
            "direction_accuracy": float(np.mean(((readout - readout.mean()) > 0) == (rpe > 0))),
            "readout_center": float(readout.mean()),
            "rpe_logit_correlation": pearson(rpe, readout),
            "mean_positive_rpe_logit_diff": float(readout[rpe > 0].mean()),
            "mean_negative_rpe_logit_diff": float(readout[rpe < 0].mean()),
        }
    return result


def cv_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int, alpha: float, seed: int) -> dict[str, float]:
    pred = np.full(len(y), np.nan)
    for train, test in grouped_folds(groups, folds, seed):
        w, b, _, _ = ridge_fit(X[train], y[train], alpha)
        pred[test] = X[test] @ w + b
    return {"r2": r2_score(y, pred), "pearson_r": pearson(y, pred)}


def surface_residual(rows: list[dict[str, Any]], y: np.ndarray) -> np.ndarray:
    """Residualize RPE against surface labels, not its mathematical inputs.

    RPE is exactly reward minus expectation, so controlling for both would
    remove the target by definition. Outcome and expectation are instead
    reported as separate decoding targets below.
    """
    templates = sorted({r["template"] for r in rows})
    cols = [np.ones(len(rows))]
    cols.append(np.array([r["arm"] == "M" for r in rows], dtype=float))
    for template in templates[1:]:
        cols.append(np.array([r["template"] == template for r in rows], dtype=float))
    design = np.stack(cols, axis=1)
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return y - design @ beta


def contrast_direction(X: np.ndarray, rows: list[dict[str, Any]]) -> tuple[np.ndarray, float]:
    p = np.array([r["expected_p"] for r in rows])
    reward = np.array([r["reward"] for r in rows])
    high = p >= np.median(p)
    surprising_win = (reward == 1) & ~high
    expected_win = (reward == 1) & high
    expected_loss = (reward == 0) & ~high
    surprising_loss = (reward == 0) & high
    d_pos = X[surprising_win].mean(0) - X[expected_win].mean(0)
    d_neg = X[expected_loss].mean(0) - X[surprising_loss].mean(0)
    alignment = float(d_pos @ d_neg / ((np.linalg.norm(d_pos) * np.linalg.norm(d_neg)) + 1e-12))
    direction = d_pos + d_neg
    direction /= np.linalg.norm(direction) + 1e-12
    return direction.astype(np.float32), alignment


def cross_template_score(X: np.ndarray, y: np.ndarray, rows: list[dict[str, Any]], alpha: float) -> float:
    templates = sorted({r["template"] for r in rows})
    scores = []
    labels = np.array([r["template"] for r in rows])
    for held_out in templates:
        train, test = labels != held_out, labels == held_out
        if train.sum() == 0 or test.sum() == 0:
            continue
        w, b, _, _ = ridge_fit(X[train], y[train], alpha)
        scores.append(pearson(y[test], X[test] @ w + b))
    return float(np.nanmean(scores)) if scores else float("nan")


def command_analyze(args: argparse.Namespace) -> None:
    acts, rows, meta = load_dataset(args.data_dir)
    if meta.get("experiment_version", 1) < 2 and not args.allow_legacy:
        raise SystemExit(
            "This dataset was collected with the old behavioral readout. "
            "Run collect again into a new directory, or pass --allow-legacy "
            "only to reproduce the old diagnostic analysis."
        )
    y = np.array([r["rpe"] for r in rows], dtype=np.float32)
    y_abs = np.abs(y)
    y_reward = np.array([r["reward"] for r in rows], dtype=np.float32)
    y_expect = np.array([r["expected_p"] for r in rows], dtype=np.float32)
    y_surface = surface_residual(rows, y).astype(np.float32)
    groups = np.array([r["episode_id"] for r in rows])
    behavior = behavioral_metrics(rows)
    print("Behavioral readout:")
    for name, values in behavior.items():
        print(
            f"  {name:8s}: accuracy={values['direction_accuracy']:.3f}, "
            f"RPE/logit r={values['rpe_logit_correlation']:+.3f}"
        )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    directions = []
    for layer in range(acts.shape[1]):
        X = np.asarray(acts[:, layer, :], dtype=np.float64)
        rpe = cv_probe(X, y, groups, args.folds, args.alpha, args.seed)
        surface = cv_probe(X, y_surface, groups, args.folds, args.alpha, args.seed)
        reward = cv_probe(X, y_reward, groups, args.folds, args.alpha, args.seed)
        expectation = cv_probe(X, y_expect, groups, args.folds, args.alpha, args.seed)
        surprise = cv_probe(X, y_abs, groups, args.folds, args.alpha, args.seed)
        direction, alignment = contrast_direction(X, rows)
        directions.append(direction)
        records.append({
            "layer": layer,
            "rpe_r2": rpe["r2"], "rpe_r": rpe["pearson_r"],
            "surface_adjusted_rpe_r2": surface["r2"],
            "surface_adjusted_rpe_r": surface["pearson_r"],
            "reward_r": reward["pearson_r"],
            "expectation_r": expectation["pearson_r"],
            "surprise_r": surprise["pearson_r"],
            "cross_template_r": cross_template_score(X, y, rows, args.alpha),
            "contrast_alignment": alignment,
        })
        print(f"Layer {layer:02d}: RPE r={rpe['pearson_r']:+.3f}, held-template r={records[-1]['cross_template_r']:+.3f}")

    np.save(out / "contrast_directions.npy", np.stack(directions))
    with (out / "layer_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    eligible = [r for r in records if math.isfinite(r["cross_template_r"])]
    if eligible:
        best = max(eligible, key=lambda r: (r["cross_template_r"], r["contrast_alignment"]))
        selection_rule = "maximum held-out-template RPE correlation; alignment breaks ties"
    else:
        best = max(records[1:], key=lambda r: (r["rpe_r"], r["contrast_alignment"]))
        selection_rule = "single-template fallback: maximum grouped-CV RPE correlation; alignment breaks ties"
    summary = {
        "source_data": str(Path(args.data_dir).resolve()),
        "model": meta["model"],
        "best_layer": best["layer"],
        "selection_rule": selection_rule,
        "best_metrics": best,
        "behavioral_readout": behavior,
        "behavioral_readout_passed": bool(
            behavior["all"]["direction_accuracy"] >= args.min_behavior_accuracy
            and behavior["all"]["rpe_logit_correlation"] >= args.min_behavior_correlation
            and all(
                behavior[t]["direction_accuracy"] >= args.min_behavior_accuracy
                and behavior[t]["rpe_logit_correlation"] >= args.min_behavior_correlation
                for t in behavior if t != "all"
            )
        ),
        "minimum_behavior_accuracy": args.min_behavior_accuracy,
        "minimum_behavior_correlation": args.min_behavior_correlation,
        "warning": (
            "Probe/contrast evidence is correlational. Run patching only after "
            "the behavioral readout passes in every template."
        ),
    }
    (out / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Best candidate hidden-state index: {best['layer']}; outputs saved to {out}")


def resolve_transformer_blocks(model: Any) -> Any:
    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("model", "decoder", "layers"),
    ]
    for path in candidates:
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            if len(obj):
                return obj
        except (AttributeError, TypeError):
            pass
    raise ValueError("Unsupported model architecture: could not locate transformer block list")


@contextlib.contextmanager
def patch_block_output(block: Any, direction: Any, coefficient: float, mode: str):
    """Patch last-token block output. Handles tensor and tuple block outputs."""
    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        tensor = output[0] if isinstance(output, tuple) else output
        d = direction.to(device=tensor.device, dtype=tensor.dtype)
        current = (tensor[:, -1, :] * d).sum(dim=-1, keepdim=True)
        if mode == "remove":
            delta = -current
        elif mode in {"inject", "swap"}:
            delta = coefficient - current
        else:
            raise ValueError(mode)
        patched = tensor.clone()
        patched[:, -1, :] = patched[:, -1, :] + delta * d
        if isinstance(output, tuple):
            return (patched, *output[1:])
        return patched

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def forward_readout(torch: Any, model: Any, tokenizer: Any, row: dict[str, Any], patch: tuple[Any, Any, float, str] | None = None) -> float:
    prompt = apply_chat_template(tokenizer, row["prompt"]) + row.get("readout_prefix", "")
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {k: v.to(model_device(model)) for k, v in encoded.items()}
    pos = token_id_for_completion(tokenizer, prompt, row["positive_word"])
    neg = token_id_for_completion(tokenizer, prompt, row["negative_word"])
    manager = contextlib.nullcontext()
    if patch is not None:
        block, direction, coefficient, mode = patch
        manager = patch_block_output(block, direction, coefficient, mode)
    with manager, torch.inference_mode():
        logits = model(**encoded, return_dict=True).logits[0, -1].float()
    return float((logits[pos] - logits[neg]).cpu())


def command_patch(args: argparse.Namespace) -> None:
    acts, rows, meta = load_dataset(args.data_dir)
    analysis = json.loads((Path(args.analysis_dir) / "analysis_summary.json").read_text())
    if not analysis.get("behavioral_readout_passed", False) and not args.allow_failed_behavior:
        raise SystemExit(
            "Behavioral readout did not pass (or this is an older analysis). "
            "Re-collect with experiment version 2 and re-run analyze. Use "
            "--allow-failed-behavior only for debugging, not for causal claims."
        )
    directions = np.load(Path(args.analysis_dir) / "contrast_directions.npy")
    hidden_index = args.layer if args.layer is not None else int(analysis["best_layer"])
    if hidden_index == 0:
        raise ValueError("Embedding output cannot be patched through a transformer block; select layer >= 1")
    block_index = hidden_index - 1

    torch, model, tokenizer = load_model(meta["model"], args.device, args.dtype)
    blocks = resolve_transformer_blocks(model)
    if block_index >= len(blocks):
        raise ValueError(f"Hidden-state index {hidden_index} maps beyond {len(blocks)} blocks")
    direction = torch.tensor(directions[hidden_index], dtype=torch.float32)
    control_directions = []
    rng = np.random.default_rng(args.seed)
    candidate_np = directions[hidden_index]
    for _ in range(args.random_controls):
        control = rng.normal(size=candidate_np.shape).astype(np.float32)
        control -= float(control @ candidate_np) * candidate_np
        control /= np.linalg.norm(control) + 1e-12
        control_directions.append(torch.tensor(control, dtype=torch.float32))
    coefficients = np.asarray(acts[:, hidden_index, :]) @ directions[hidden_index]
    source_mask = np.array([r["rpe"] >= args.source_rpe for r in rows])
    if not source_mask.any():
        raise ValueError("No examples satisfy --source-rpe")
    source_coefficient = float(np.mean(coefficients[source_mask]))

    candidates = np.flatnonzero(np.array([r["rpe"] <= args.target_rpe for r in rows]))
    rng.shuffle(candidates)
    candidates = candidates[: args.max_examples]
    output_rows = []
    for i, idx in enumerate(candidates, 1):
        row = rows[int(idx)]
        baseline = forward_readout(torch, model, tokenizer, row)
        removed = forward_readout(
            torch, model, tokenizer, row,
            (blocks[block_index], direction, 0.0, "remove"),
        )
        injected = forward_readout(
            torch, model, tokenizer, row,
            (blocks[block_index], direction, source_coefficient * args.strength, "inject"),
        )
        control_effects = []
        for control_direction in control_directions:
            control_value = forward_readout(
                torch, model, tokenizer, row,
                (blocks[block_index], control_direction, source_coefficient * args.strength, "inject"),
            )
            control_effects.append(control_value - baseline)
        output_rows.append({
            "example_id": row["example_id"], "rpe": row["rpe"],
            "baseline_logit_diff": baseline,
            "removed_logit_diff": removed,
            "injected_logit_diff": injected,
            "removal_effect": removed - baseline,
            "injection_effect": injected - baseline,
            "random_control_mean_effect": float(np.mean(control_effects)) if control_effects else float("nan"),
        })
        print(f"Patched {i}/{len(candidates)}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "patch_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    effects = np.array([r["injection_effect"] for r in output_rows])
    removal = np.array([r["removal_effect"] for r in output_rows])
    controls = np.array([r["random_control_mean_effect"] for r in output_rows])
    summary = {
        "hidden_state_index": hidden_index,
        "transformer_block_index": block_index,
        "n": len(output_rows),
        "source_coefficient": source_coefficient,
        "strength": args.strength,
        "mean_injection_effect": float(effects.mean()),
        "injection_effect_sem": float(effects.std(ddof=1) / math.sqrt(len(effects))) if len(effects) > 1 else None,
        "mean_removal_effect": float(removal.mean()),
        "random_controls_per_example": args.random_controls,
        "mean_random_control_effect": float(np.nanmean(controls)) if args.random_controls else None,
        "candidate_minus_random_control": float(np.nanmean(effects - controls)) if args.random_controls else None,
        "interpretation": "Positive effects favor the update-increase word over the update-decrease word.",
    }
    (out / "patch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def command_matched_patch(args: argparse.Namespace) -> None:
    """Patch expectation contrasts while holding outcome identity fixed."""
    acts, rows, meta = load_dataset(args.data_dir)
    analysis = json.loads((Path(args.analysis_dir) / "analysis_summary.json").read_text())
    if not analysis.get("behavioral_readout_passed", False) and not args.allow_failed_behavior:
        raise SystemExit("Behavioral readout did not pass; matched patching is not interpretable.")
    directions = np.load(Path(args.analysis_dir) / "contrast_directions.npy")
    hidden_index = args.layer
    if hidden_index == 0:
        raise ValueError("Select a hidden-state index >= 1")
    block_index = hidden_index - 1

    torch, model, tokenizer = load_model(meta["model"], args.device, args.dtype)
    blocks = resolve_transformer_blocks(model)
    direction_np = directions[hidden_index]
    direction = torch.tensor(direction_np, dtype=torch.float32)
    coefficients = np.asarray(acts[:, hidden_index, :]) @ direction_np
    rng = np.random.default_rng(args.seed)
    control_directions = []
    for _ in range(args.random_controls):
        control = rng.normal(size=direction_np.shape).astype(np.float32)
        control -= float(control @ direction_np) * direction_np
        control /= np.linalg.norm(control) + 1e-12
        control_directions.append(torch.tensor(control, dtype=torch.float32))

    # Each pair changes expected probability while holding observed outcome.
    # Both directions are included to test whether the intervention reverses.
    comparisons = [
        ("win_toward_more_positive_rpe", 1, args.low_expectation, args.high_expectation),
        ("win_toward_less_positive_rpe", 1, args.high_expectation, args.low_expectation),
        ("loss_toward_less_negative_rpe", 0, args.low_expectation, args.high_expectation),
        ("loss_toward_more_negative_rpe", 0, args.high_expectation, args.low_expectation),
    ]
    all_output: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    reward_arr = np.array([r["reward"] for r in rows])
    p_arr = np.array([r["expected_p"] for r in rows], dtype=float)

    for name, reward, source_p, target_p in comparisons:
        source_mask = (reward_arr == reward) & np.isclose(p_arr, source_p)
        target_indices = np.flatnonzero((reward_arr == reward) & np.isclose(p_arr, target_p))
        if not source_mask.any() or not len(target_indices):
            raise ValueError(f"Missing source or target cells for {name}")
        rng.shuffle(target_indices)
        target_indices = target_indices[: args.max_examples]
        source_coefficient = float(coefficients[source_mask].mean())
        effects = []
        controls = []
        for j, idx in enumerate(target_indices, 1):
            row = rows[int(idx)]
            baseline = forward_readout(torch, model, tokenizer, row)
            patched = forward_readout(
                torch, model, tokenizer, row,
                (blocks[block_index], direction, source_coefficient * args.strength, "inject"),
            )
            control_effects = []
            for control_direction in control_directions:
                control_value = forward_readout(
                    torch, model, tokenizer, row,
                    (blocks[block_index], control_direction, source_coefficient * args.strength, "inject"),
                )
                control_effects.append(control_value - baseline)
            effect = patched - baseline
            control_mean = float(np.mean(control_effects)) if control_effects else float("nan")
            effects.append(effect)
            controls.append(control_mean)
            all_output.append({
                "comparison": name,
                "example_id": row["example_id"],
                "reward": reward,
                "source_expected_p": source_p,
                "target_expected_p": target_p,
                "source_rpe": reward - source_p,
                "target_rpe": reward - target_p,
                "source_coefficient": source_coefficient,
                "target_original_coefficient": float(coefficients[int(idx)]),
                "baseline_logit_diff": baseline,
                "patched_logit_diff": patched,
                "injection_effect": effect,
                "random_control_mean_effect": control_mean,
                "candidate_minus_control": effect - control_mean,
            })
            print(f"{name}: {j}/{len(target_indices)}")
        e = np.array(effects)
        c = np.array(controls)
        summaries[name] = {
            "n": len(e),
            "reward": reward,
            "source_expected_p": source_p,
            "target_expected_p": target_p,
            "source_rpe": reward - source_p,
            "target_rpe": reward - target_p,
            "mean_injection_effect": float(e.mean()),
            "injection_effect_sem": float(e.std(ddof=1) / math.sqrt(len(e))) if len(e) > 1 else None,
            "mean_random_control_effect": float(np.nanmean(c)) if args.random_controls else None,
            "candidate_minus_random_control": float(np.nanmean(e - c)) if args.random_controls else None,
            "positive_effect_fraction": float(np.mean(e > 0)),
        }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "matched_patch_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_output[0]))
        writer.writeheader()
        writer.writerows(all_output)
    result = {
        "hidden_state_index": hidden_index,
        "transformer_block_index": block_index,
        "strength": args.strength,
        "random_controls_per_example": args.random_controls,
        "comparisons": summaries,
        "warning": (
            "The P-vs-N readout measures RPE sign, not calibrated magnitude. "
            "Matched-outcome effects are exploratory and should be judged for "
            "bidirectional consistency, not merely a positive sign."
        ),
    }
    (out / "matched_patch_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    behavior = sub.add_parser("behavior-test", help="Run a small generation-only task sanity check")
    behavior.add_argument("--model", default="Qwen/Qwen3-0.6B")
    behavior.add_argument("--templates", nargs="+", choices=sorted(BEHAVIOR_TEMPLATES), default=sorted(BEHAVIOR_TEMPLATES))
    behavior.add_argument("--probabilities", nargs="+", type=float, default=[0.1, 0.3, 0.7, 0.9])
    behavior.add_argument("--repeats", type=int, default=1)
    behavior.add_argument("--max-new-tokens", type=int, default=64)
    behavior.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    behavior.add_argument("--min-accuracy", type=float, default=0.90)
    behavior.add_argument("--output-dir", default="analysis/behavior_pilot")
    behavior.add_argument("--device", default="auto")
    behavior.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    behavior.add_argument("--seed", type=int, default=42)
    behavior.set_defaults(func=command_behavior_test)

    collect = sub.add_parser("collect", help="Collect controlled residual-stream activations")
    collect.add_argument("--model", default="Qwen/Qwen3-0.6B")
    collect.add_argument("--episodes", type=int, default=30)
    collect.add_argument("--templates", nargs="+", choices=sorted(TEMPLATES), default=sorted(TEMPLATES))
    collect.add_argument("--probabilities", nargs="+", type=float, default=[0.1, 0.3, 0.7, 0.9])
    collect.add_argument("--output-dir", default="results/causal_rpe")
    collect.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    collect.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    collect.add_argument("--seed", type=int, default=42)
    collect.add_argument(
        "--structured-readout",
        action="store_true",
        help="Capture at the RPE_SIGN decision position and compare POSITIVE vs NEGATIVE",
    )
    collect.set_defaults(func=command_collect)

    analyze = sub.add_parser("analyze", help="Discover and evaluate candidate RPE directions")
    analyze.add_argument("--data-dir", default="results/causal_rpe")
    analyze.add_argument("--output-dir", default="analysis/causal_rpe")
    analyze.add_argument("--folds", type=int, default=5)
    analyze.add_argument("--alpha", type=float, default=100.0)
    analyze.add_argument("--seed", type=int, default=42)
    analyze.add_argument("--min-behavior-accuracy", type=float, default=0.80)
    analyze.add_argument("--min-behavior-correlation", type=float, default=0.50)
    analyze.add_argument("--allow-legacy", action="store_true")
    analyze.set_defaults(func=command_analyze)

    patch = sub.add_parser("patch", help="Causally patch a discovered RPE direction")
    patch.add_argument("--data-dir", default="results/causal_rpe")
    patch.add_argument("--analysis-dir", default="analysis/causal_rpe")
    patch.add_argument("--output-dir", default="analysis/causal_rpe/patching")
    patch.add_argument("--layer", type=int, default=None, help="Hidden-state index; defaults to selected candidate")
    patch.add_argument("--source-rpe", type=float, default=0.7)
    patch.add_argument("--target-rpe", type=float, default=-0.7)
    patch.add_argument("--strength", type=float, default=1.0)
    patch.add_argument("--max-examples", type=int, default=100)
    patch.add_argument("--random-controls", type=int, default=3)
    patch.add_argument("--allow-failed-behavior", action="store_true")
    patch.add_argument("--device", default="auto")
    patch.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    patch.add_argument("--seed", type=int, default=42)
    patch.set_defaults(func=command_patch)

    matched = sub.add_parser("matched-patch", help="Patch RPE contrasts while holding outcome fixed")
    matched.add_argument("--data-dir", default="results/causal_rpe_numeric_v3")
    matched.add_argument("--analysis-dir", default="analysis/causal_rpe_numeric_v3")
    matched.add_argument("--output-dir", default="analysis/causal_rpe_numeric_v3/matched_patch")
    matched.add_argument("--layer", type=int, default=18)
    matched.add_argument("--low-expectation", type=float, default=0.1)
    matched.add_argument("--high-expectation", type=float, default=0.9)
    matched.add_argument("--strength", type=float, default=1.0)
    matched.add_argument("--max-examples", type=int, default=20)
    matched.add_argument("--random-controls", type=int, default=1)
    matched.add_argument("--device", default="auto")
    matched.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    matched.add_argument("--seed", type=int, default=42)
    matched.add_argument("--allow-failed-behavior", action="store_true")
    matched.set_defaults(func=command_matched_patch)
    return p


def main() -> None:
    args = parser().parse_args()
    if hasattr(args, "probabilities") and any(not 0 <= p <= 1 for p in args.probabilities):
        raise SystemExit("All probabilities must be between 0 and 1")
    args.func(args)


if __name__ == "__main__":
    main()
