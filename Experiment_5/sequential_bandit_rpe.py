#!/usr/bin/env python3
"""Sequential-bandit search for causal prediction-error representations.

The model sees canonical trial history but is never told reward probabilities or
the RPE formula.  Model weights and KV state are reset on every forward call;
the episode state is carried only by the reconstructed text history.

Commands: simulate -> behavior -> collect -> extract -> monitor -> steer -> validate
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from causal_rpe_bandit import (
    apply_chat_template,
    load_model,
    model_device,
    resolve_transformer_blocks,
    seed_everything,
    token_id_for_completion,
)


BINS = np.arange(0.05, 1.0, 0.10)
BIN_LABELS = tuple("0123456789")
VECTOR_NAMES = ("expectation", "outcome", "signed_rpe", "within_outcome_rpe", "rpe_orthogonal")
SYSTEM = (
    "You are playing a repeated two-armed reward task. Reward tendencies can change. "
    "Infer each option's current value from the observed history."
)


def dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def history_text(history: list[tuple[str, int]]) -> str:
    if not history:
        return "(no previous trials)"
    return "\n".join(f"{i:02d} {a} {r}" for i, (a, r) in enumerate(history, 1))


def choice_raw(history: list[tuple[str, int]], choice_order: str = "AB") -> str:
    shown = "A/B" if choice_order == "AB" else "B/A"
    return f"{SYSTEM}\n\nHistory (trial option outcome):\n{history_text(history)}\n\nNEXT_CHOICE ({shown}):"


def value_raw(history: list[tuple[str, int]], arm: str, bin_values: np.ndarray) -> str:
    labels = ", ".join(f"{k}={p:.2f}" for k, p in zip(BIN_LABELS, bin_values))
    return (
        f"{SYSTEM}\n\nHistory (trial option outcome):\n{history_text(history)}\n\n"
        f"Estimate the current reward probability of option {arm}.\n"
        f"Probability labels: {labels}\nVALUE_LABEL:"
    )


def feedback_readout_raw(history: list[tuple[str, int]], arm: str, reward: int,
                         bin_values: np.ndarray) -> tuple[str, str]:
    """Return full prompt and prefix ending exactly at the causal feedback site."""
    labels = ", ".join(f"{k}={p:.2f}" for k, p in zip(BIN_LABELS, bin_values))
    prefix = (
        f"{SYSTEM}\n\nHistory (trial option outcome):\n{history_text(history)}\n\n"
        f"Current trial:\nCHOSEN {arm}\nOUTCOME {reward}\nFEEDBACK_END"
    )
    full = (
        prefix + f"\n\nAfter incorporating that feedback, estimate option {arm}'s current reward probability.\n"
        f"Probability labels: {labels}\nUPDATED_VALUE_LABEL:"
    )
    return full, prefix


def prepared(tokenizer: Any, raw: str) -> str:
    return apply_chat_template(tokenizer, raw)


def label_ids(tokenizer: Any, prompt: str, labels: tuple[str, ...]) -> list[int]:
    ids = []
    for label in labels:
        # Qwen tokenizes bare digits as one token but splits space+digit; letter
        # readouts may work either way depending on the chat-template boundary.
        try:
            ids.append(token_id_for_completion(tokenizer, prompt, label))
        except ValueError:
            ids.append(token_id_for_completion(tokenizer, prompt, " " + label))
    return ids


def logits_for(torch: Any, model: Any, tokenizer: Any, raw: str, labels: tuple[str, ...]) -> np.ndarray:
    prompt = prepared(tokenizer, raw)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {k: v.to(model_device(model)) for k, v in encoded.items()}
    ids = label_ids(tokenizer, prompt, labels)
    with torch.inference_mode():
        logits = model(**encoded, return_dict=True).logits[0, -1].float()
    return logits[ids].cpu().numpy()


def softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - np.max(x)); return z / z.sum()


def calibrated_logits(raw_logits: np.ndarray, baseline_logits: np.ndarray | None) -> np.ndarray:
    """Remove stable answer-token preferences from an otherwise identical readout."""
    return raw_logits - baseline_logits if baseline_logits is not None else raw_logits


def estimate_value(torch: Any, model: Any, tokenizer: Any, history: list[tuple[str, int]], arm: str,
                   bin_values: np.ndarray, baseline_logits: np.ndarray | None = None
                   ) -> tuple[float, list[float], float, list[float]]:
    """Return calibrated and raw value estimates.

    Subtracting logits from the identical no-history question removes stable
    preferences for particular label tokens.  The random label-to-value map is
    held fixed within an episode, so the subtraction is label-wise valid.
    """
    raw_logits = logits_for(torch, model, tokenizer, value_raw(history, arm, bin_values), BIN_LABELS)
    raw_probs = softmax(raw_logits)
    adjusted_logits = calibrated_logits(raw_logits, baseline_logits)
    calibrated_probs = softmax(adjusted_logits)
    return (float(calibrated_probs @ bin_values), calibrated_probs.tolist(),
            float(raw_probs @ bin_values), raw_probs.tolist())


def command_simulate(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    episodes = []
    for e in range(args.episodes):
        high_arm = "A" if rng.random() < 0.5 else "B"
        reversal = args.trials // 2 + int(rng.integers(-args.reversal_jitter, args.reversal_jitter + 1))
        uniforms = rng.random((args.trials, 2))
        schedule = []
        for t in range(args.trials):
            current_high = high_arm if t < reversal else ("B" if high_arm == "A" else "A")
            pa = args.high_p if current_high == "A" else args.low_p
            pb = args.high_p if current_high == "B" else args.low_p
            schedule.append({"trial": t + 1, "p_A": pa, "p_B": pb,
                             "potential_A": int(uniforms[t, 0] < pa), "potential_B": int(uniforms[t, 1] < pb)})
        episodes.append({"episode_id": f"seq_{e:04d}", "initial_high_arm": high_arm,
                         "reversal_trial": reversal + 1, "schedule": schedule})
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dump(out / "environments.json", {"seed": args.seed, "episodes": episodes,
                                      "low_p": args.low_p, "high_p": args.high_p})
    print(f"Saved {args.episodes} environments x {args.trials} trials to {out}")


def command_behavior(args: argparse.Namespace) -> None:
    env = json.loads((Path(args.environment_dir) / "environments.json").read_text())
    torch, model, tokenizer = load_model(args.model, args.device, args.dtype)
    seed_everything(args.seed, torch); rng = np.random.default_rng(args.seed)
    rows = []
    for ei, episode in enumerate(env["episodes"], 1):
        history: list[tuple[str, int]] = []
        # Counterbalance token priors: labels 0..9 denote a new random ordering
        # of probability bins in every episode.
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
            choice_logits_raw = logits_for(
                torch, model, tokenizer, choice_raw(history, args.choice_order), ("A", "B")
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
            rows.append({"episode_id": episode["episode_id"], "trial": item["trial"],
                         "chosen_arm": chosen, "reward": reward, "true_p": true_p,
                         "choice_logit_A": float(choice_logits[0]), "choice_logit_B": float(choice_logits[1]),
                         "choice_prob_A": float(choice_probs[0]), "choice_prob_B": float(choice_probs[1]),
                         "choice_logit_A_raw": float(choice_logits_raw[0]),
                         "choice_logit_B_raw": float(choice_logits_raw[1]),
                         "choice_baseline_logit_A": (float(choice_baseline_logits[0])
                                                       if choice_baseline_logits is not None else None),
                         "choice_baseline_logit_B": (float(choice_baseline_logits[1])
                                                       if choice_baseline_logits is not None else None),
                         "exploratory_choice": exploratory, "bin_values_by_label": bin_values.tolist(),
                         "q_model_before": q_before, "q_model_after": q_after,
                         "model_update": q_after - q_before, "rpe_model": rpe_model,
                         "q_model_before_raw": q_before_raw, "q_model_after_raw": q_after_raw,
                         "raw_model_update": q_after_raw - q_before_raw,
                         "q_before_distribution": q_before_probs, "q_after_distribution": q_after_probs,
                         "q_before_distribution_raw": q_before_raw_probs,
                         "q_after_distribution_raw": q_after_raw_probs,
                         "history": history, "feedback_prompt": feedback_readout_raw(history, chosen, reward, bin_values)[0]})
            history = history_after
        print(f"Behavior episode {ei}/{len(env['episodes'])}")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    save_jsonl(out / "trials.jsonl", rows)
    # Trial 1 has no history and can dominate a short pilot. Gate on later trials,
    # while retaining all-trial metrics as diagnostics.
    rpe = np.array([r["rpe_model"] for r in rows]); update = np.array([r["model_update"] for r in rows])
    post_first = np.array([r["trial"] > 1 for r in rows])
    gate_rpe, gate_update = rpe[post_first], update[post_first]
    positive = rpe > 0; negative = rpe < 0
    gate_positive = gate_rpe > 0; gate_negative = gate_rpe < 0
    direction_accuracy = float(np.mean(np.where(gate_positive, gate_update > 0, gate_update < 0)))
    chosen_a = np.array([r["chosen_arm"] == "A" for r in rows])
    optimal = np.array([(r["chosen_arm"] == "A") == (r["true_p"] > .5) for r in rows])
    gate_corr = corr(gate_rpe, gate_update)
    summary = {"model": args.model, "n_trials": len(rows), "episodes": len(env["episodes"]),
               "choice_order": args.choice_order,
               "label_prior_calibration": args.calibrate_label_prior,
               "choice_prior_calibration": args.calibrate_choice_prior,
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
               "behavioral_readout_passed": bool(gate_corr >= args.min_rpe_update_correlation
                                                  and gate_update[gate_positive].mean() > 0
                                                  and gate_update[gate_negative].mean() < 0
                                                  and direction_accuracy >= args.min_direction_accuracy),
               "gate_thresholds": {"min_rpe_update_correlation": args.min_rpe_update_correlation,
                                   "min_direction_accuracy": args.min_direction_accuracy,
                                   "evaluation_subset": "trials after trial 1"},
               "behavior_gate": "Continue to activation collection only if behavioral_readout_passed is true."}
    dump(out / "behavior_summary.json", summary); print(json.dumps(summary, indent=2))


def feedback_position(tokenizer: Any, raw_full: str, raw_prefix: str) -> tuple[str, int]:
    full = prepared(tokenizer, raw_full)
    # Chat templates append generation markers, so locate the tokenized raw-prefix
    # content inside a separately templated prefix and use its last content token.
    prefix_chat = prepared(tokenizer, raw_prefix)
    prefix_ids = tokenizer(prefix_chat, add_special_tokens=False).input_ids
    full_ids = tokenizer(full, add_special_tokens=False).input_ids
    # Remove the generation suffix by finding the longest common prefix.
    n = 0
    while n < min(len(prefix_ids), len(full_ids)) and prefix_ids[n] == full_ids[n]: n += 1
    if n == 0: raise ValueError("Could not locate feedback prefix in templated prompt")
    return full, n - 1


def command_collect(args: argparse.Namespace) -> None:
    rows = jsonl(Path(args.behavior_dir) / "trials.jsonl")
    meta = json.loads((Path(args.behavior_dir) / "behavior_summary.json").read_text())
    if not meta.get("behavioral_readout_passed", False) and not args.allow_failed_behavior:
        raise SystemExit(
            "Behavioral readout did not pass. Re-run behavior after revising the readout/model, "
            "or use --allow-failed-behavior only for pipeline debugging."
        )
    torch, model, tokenizer = load_model(meta["model"], args.device, args.dtype)
    activations = []
    for i, row in enumerate(rows, 1):
        history = [(str(a), int(r)) for a, r in row["history"]]
        bins = np.asarray(row["bin_values_by_label"], float)
        raw, prefix = feedback_readout_raw(history, row["chosen_arm"], row["reward"], bins)
        prompt, position = feedback_position(tokenizer, raw, prefix)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {k: v.to(model_device(model)) for k, v in encoded.items()}
        with torch.inference_mode(): out = model(**encoded, output_hidden_states=True, return_dict=True)
        activations.append(np.stack([h[0, position].detach().float().cpu().numpy() for h in out.hidden_states]))
        row["feedback_token_position"] = position
        if i % 25 == 0 or i == len(rows): print(f"Collected {i}/{len(rows)}")
    arr = np.asarray(activations, np.float32)
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "feedback_activations.npy", arr); save_jsonl(outdir / "trials.jsonl", rows)
    dump(outdir / "metadata.json", {"model": meta["model"], "shape": list(arr.shape),
                                     "dimensions": ["trial", "hidden_state_index", "residual_dimension"],
                                     "site": "last token of FEEDBACK_END before updated-value question"})


def split_episodes(rows: list[dict[str, Any]], seed: int) -> dict[str, list[str]]:
    eps = np.array(sorted({r["episode_id"] for r in rows})); rng = np.random.default_rng(seed); rng.shuffle(eps)
    a, b = int(.6 * len(eps)), int(.8 * len(eps))
    return {"discovery": eps[:a].tolist(), "validation": eps[a:b].tolist(), "test": eps[b:].tolist()}


def unit(x: np.ndarray) -> np.ndarray: return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)
def corr(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and np.std(x) > 1e-10 and np.std(y) > 1e-10 else float("nan")
def cosine(x: np.ndarray, y: np.ndarray) -> float: return float(x @ y / (np.linalg.norm(x)*np.linalg.norm(y)+1e-12))


def contrasts(X: np.ndarray, q: np.ndarray, reward: np.ndarray, rpe: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    def diff_by_quantile(target: np.ndarray, mask: np.ndarray) -> np.ndarray:
        lo, hi = np.quantile(target[mask], [.25, .75])
        return X[mask & (target >= hi)].mean(0) - X[mask & (target <= lo)].mean(0)
    expectation = .5 * (diff_by_quantile(q, reward == 0) + diff_by_quantile(q, reward == 1))
    outcome = X[reward == 1].mean(0) - X[reward == 0].mean(0)
    signed = diff_by_quantile(rpe, np.ones(len(rpe), bool))
    win = diff_by_quantile(rpe, reward == 1); loss = diff_by_quantile(rpe, reward == 0)
    within = .5 * (win + loss)
    basis = []
    for v in (expectation, outcome):
        z = v.copy()
        for b in basis: z -= (z @ b) * b
        if np.linalg.norm(z) > 1e-10: basis.append(z / np.linalg.norm(z))
    orth = signed.copy()
    for b in basis: orth -= (orth @ b) * b
    vec = np.stack([expectation, outcome, signed, within, orth]).astype(np.float32)
    return vec, {"cos_rpe_outcome": cosine(signed, outcome), "cos_rpe_expectation": cosine(signed, expectation),
                 "cos_win_loss": cosine(win, loss), "orthogonal_fraction": float(np.linalg.norm(orth)/(np.linalg.norm(signed)+1e-12))}


def load_collected(root: str) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    p=Path(root); return np.load(p/"feedback_activations.npy", mmap_mode="r"), jsonl(p/"trials.jsonl"), json.loads((p/"metadata.json").read_text())


def command_extract(args: argparse.Namespace) -> None:
    acts, rows, meta = load_collected(args.data_dir); split = split_episodes(rows, args.seed)
    m=np.array([r["episode_id"] in set(split["discovery"]) for r in rows])
    q=np.array([r["q_model_before"] for r in rows])[m]; reward=np.array([r["reward"] for r in rows])[m]; rpe=np.array([r["rpe_model"] for r in rows])[m]
    vectors=[]; geometry=[]
    for layer in range(acts.shape[1]):
        v,g=contrasts(np.asarray(acts[m,layer],float),q,reward,rpe); vectors.append(v); geometry.append({"layer":layer,**g})
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); raw=np.stack(vectors)
    np.save(out/"vectors_raw.npy",raw); np.save(out/"vectors_unit.npy",unit(raw).astype(np.float32)); dump(out/"split.json",split)
    dump(out/"metadata.json",{"model":meta["model"],"vector_names":list(VECTOR_NAMES),"shape":list(raw.shape),"seed":args.seed})
    with (out/"geometry.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(geometry[0]));w.writeheader();w.writerows(geometry)


def command_monitor(args: argparse.Namespace) -> None:
    acts,rows,_=load_collected(args.data_dir); root=Path(args.vectors_dir); split=json.loads((root/"split.json").read_text()); vectors=np.load(root/"vectors_unit.npy")
    records=[]
    # Deliberately do not compute test projections here. The test episodes stay
    # sealed until the locked causal configuration is evaluated by validate.
    for phase in ("validation",):
        m=np.array([r["episode_id"] in set(split[phase]) for r in rows]); rpe=np.array([r["rpe_model"] for r in rows])[m]; reward=np.array([r["reward"] for r in rows])[m]
        # Exclude the final block output: a patch there cannot influence later
        # readout tokens because no downstream attention layer remains.
        for layer in range(1,acts.shape[1]-1):
            for j,name in enumerate(VECTOR_NAMES):
                z=np.asarray(acts[m,layer])@vectors[layer,j]; win=reward==1; loss=reward==0
                records.append({"phase":phase,"layer":layer,"vector":name,"rpe_r":corr(z,rpe),"win_r":corr(z[win],rpe[win]),"loss_r":corr(z[loss],rpe[loss]),"update_r":corr(z,np.array([r["model_update"] for r in rows])[m])})
    cand=[r for r in records if r["phase"]=="validation" and r["vector"]==args.vector]
    best=max(cand,key=lambda r:(min(r["win_r"],r["loss_r"]),r["update_r"]))
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    with (out/"monitoring.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    dump(out/"monitor_summary.json",{"selected_layer":best["layer"],"selected_vector":args.vector,"validation_metrics":best,"selection_phase":"validation"});print(json.dumps(best,indent=2))


@contextlib.contextmanager
def patch_at(block: Any, vector: Any, alpha: float, position: int):
    def hook(_m:Any,_i:Any,out:Any)->Any:
        x=out[0] if isinstance(out,tuple) else out; y=x.clone(); v=vector.to(y.device,y.dtype); y[:,position,:]+=alpha*v
        return (y,*out[1:]) if isinstance(out,tuple) else y
    h=block.register_forward_hook(hook)
    try:yield
    finally:h.remove()


def updated_value(torch:Any,model:Any,tokenizer:Any,row:dict[str,Any],block:Any|None=None,vector:Any|None=None,alpha:float=0)->float:
    history=[(str(a),int(r)) for a,r in row["history"]]; bins=np.asarray(row["bin_values_by_label"],float); raw,prefix=feedback_readout_raw(history,row["chosen_arm"],row["reward"],bins); prompt,pos=feedback_position(tokenizer,raw,prefix)
    enc=tokenizer(prompt,return_tensors="pt",add_special_tokens=False);enc={k:v.to(model_device(model)) for k,v in enc.items()};ids=label_ids(tokenizer,prompt,BIN_LABELS)
    manager=patch_at(block,vector,alpha,pos) if block is not None and alpha!=0 else contextlib.nullcontext()
    with manager,torch.inference_mode(): logits=model(**enc,return_dict=True).logits[0,-1].float()[ids].cpu().numpy()
    return float(softmax(logits)@BINS)


def steering_run(args:argparse.Namespace,phase:str,layer:int,vector_name:str,alphas:list[float],out:Path)->dict[str,Any]:
    _,rows,meta=load_collected(args.data_dir);root=Path(args.vectors_dir);split=json.loads((root/"split.json").read_text());raw=np.load(root/"vectors_raw.npy");v=raw[layer,VECTOR_NAMES.index(vector_name)]
    torch,model,tokenizer=load_model(meta["model"],args.device,args.dtype);blocks=resolve_transformer_blocks(model);block=blocks[layer-1];candidate=torch.tensor(v,dtype=torch.float32)
    rng=np.random.default_rng(args.seed); controls=[]
    for _ in range(args.random_controls):
        z=rng.normal(size=v.shape);z-=z@v/(v@v+1e-12)*v;z*=np.linalg.norm(v)/(np.linalg.norm(z)+1e-12);controls.append(torch.tensor(z,dtype=torch.float32))
    idx=np.flatnonzero([r["episode_id"] in set(split[phase]) for r in rows]);rng.shuffle(idx);idx=idx[:args.max_examples];output=[]
    for n,i in enumerate(idx,1):
        row=rows[int(i)];base=updated_value(torch,model,tokenizer,row)
        for a in alphas:
            val=updated_value(torch,model,tokenizer,row,block,candidate,a);cs=[updated_value(torch,model,tokenizer,row,block,c,a) for c in controls] if a else [base]*len(controls)
            output.append({"phase":phase,"example_id":row["episode_id"]+f"_{row['trial']:03d}","reward":row["reward"],"rpe_model":row["rpe_model"],"alpha":a,"baseline_value":base,"steered_value":val,"effect":val-base,"control_effect":float(np.mean(cs)-base) if cs else None})
        if n%10==0 or n==len(idx):print(f"Steering {phase} {n}/{len(idx)}")
    out.mkdir(parents=True,exist_ok=True)
    with (out/f"steering_{phase}.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    dose=[]
    for a in alphas:
        s=[x for x in output if x["alpha"]==a];e=np.array([x["effect"] for x in s]);c=np.array([x["control_effect"] for x in s if x["control_effect"] is not None])
        dose.append({"alpha":a,"n":len(e),"mean_effect":float(e.mean()),"sem":float(e.std(ddof=1)/math.sqrt(len(e))) if len(e)>1 else None,"candidate_minus_control":float(np.mean(e-c)) if len(c) else None})
    summary={"phase":phase,"layer":layer,"vector":vector_name,"vector_norm":float(np.linalg.norm(v)),"dose_response":dose};dump(out/f"steering_{phase}_summary.json",summary);return summary


def command_steer(args:argparse.Namespace)->None:
    m=json.loads((Path(args.monitor_dir)/"monitor_summary.json").read_text());layer=args.layer or int(m["selected_layer"]);name=args.vector or m["selected_vector"]
    s=steering_run(args,"validation",layer,name,args.alphas,Path(args.output_dir));positive=[d for d in s["dose_response"] if d["alpha"]>0];chosen=max(positive,key=lambda d:d["candidate_minus_control"] if d["candidate_minus_control"] is not None else d["mean_effect"])
    lock={"layer":layer,"vector":name,"alpha":chosen["alpha"],"validation_passed":bool(chosen["mean_effect"]>0 and (chosen["candidate_minus_control"] or 0)>0)};dump(Path(args.output_dir)/"locked_config.json",lock);print(json.dumps(lock,indent=2))


def command_validate(args:argparse.Namespace)->None:
    lock=json.loads((Path(args.steering_dir)/"locked_config.json").read_text());a=float(lock["alpha"]);s=steering_run(args,"test",int(lock["layer"]),lock["vector"],[0,a,-a],Path(args.output_dir));d={float(x["alpha"]):x for x in s["dose_response"]}
    result={"locked_config":lock,"positive":d[a],"negative":d[-a],"bidirectional":bool(d[a]["mean_effect"]>0>d[-a]["mean_effect"]),"claim_boundary":"Causal influence on updated-value readout is not by itself proof of biological reinforcement learning."};dump(Path(args.output_dir)/"final_validation.json",result);print(json.dumps(result,indent=2))


def steering_args(p:argparse.ArgumentParser,default:str)->None:
    p.add_argument("--data-dir",default="results/sequential/activations");p.add_argument("--vectors-dir",default="analysis/sequential/vectors");p.add_argument("--output-dir",default=default);p.add_argument("--max-examples",type=int,default=200);p.add_argument("--random-controls",type=int,default=3);p.add_argument("--device",default="auto");p.add_argument("--dtype",default="auto",choices=["auto","float32","float16","bfloat16"]);p.add_argument("--seed",type=int,default=42)


def parser()->argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest="command",required=True)
    x=s.add_parser("simulate");x.add_argument("--episodes",type=int,default=50);x.add_argument("--trials",type=int,default=30);x.add_argument("--low-p",type=float,default=.2);x.add_argument("--high-p",type=float,default=.8);x.add_argument("--reversal-jitter",type=int,default=3);x.add_argument("--seed",type=int,default=20260902);x.add_argument("--output-dir",default="results/sequential/environments");x.set_defaults(func=command_simulate)
    x=s.add_parser("behavior");x.add_argument("--environment-dir",default="results/sequential/environments");x.add_argument("--model",default="Qwen/Qwen3-1.7B");x.add_argument("--output-dir",default="results/sequential/behavior");x.add_argument("--choice-order",choices=["AB","BA"],default="AB",help="Display NEXT_CHOICE as A/B or B/A while always scoring the same A and B tokens");x.add_argument("--sample-choices",action="store_true");x.add_argument("--choice-temperature",type=float,default=1.0);x.add_argument("--epsilon",type=float,default=.1,help="Forced exploration probability");x.add_argument("--calibrate-choice-prior",action=argparse.BooleanOptionalAction,default=True,help="Subtract the identical no-history A/B logits before choosing");x.add_argument("--calibrate-label-prior",action=argparse.BooleanOptionalAction,default=True,help="Subtract no-history logits for the same randomized label map");x.add_argument("--min-rpe-update-correlation",type=float,default=.30);x.add_argument("--min-direction-accuracy",type=float,default=.60);x.add_argument("--device",default="auto");x.add_argument("--dtype",default="auto",choices=["auto","float32","float16","bfloat16"]);x.add_argument("--seed",type=int,default=42);x.set_defaults(func=command_behavior)
    x=s.add_parser("collect");x.add_argument("--behavior-dir",default="results/sequential/behavior");x.add_argument("--output-dir",default="results/sequential/activations");x.add_argument("--allow-failed-behavior",action="store_true");x.add_argument("--device",default="auto");x.add_argument("--dtype",default="auto",choices=["auto","float32","float16","bfloat16"]);x.set_defaults(func=command_collect)
    x=s.add_parser("extract");x.add_argument("--data-dir",default="results/sequential/activations");x.add_argument("--output-dir",default="analysis/sequential/vectors");x.add_argument("--seed",type=int,default=20260902);x.set_defaults(func=command_extract)
    x=s.add_parser("monitor");x.add_argument("--data-dir",default="results/sequential/activations");x.add_argument("--vectors-dir",default="analysis/sequential/vectors");x.add_argument("--output-dir",default="analysis/sequential/monitor");x.add_argument("--vector",choices=VECTOR_NAMES,default="signed_rpe");x.set_defaults(func=command_monitor)
    x=s.add_parser("steer");steering_args(x,"analysis/sequential/steering");x.add_argument("--monitor-dir",default="analysis/sequential/monitor");x.add_argument("--layer",type=int);x.add_argument("--vector",choices=VECTOR_NAMES);x.add_argument("--alphas",nargs="+",type=float,default=[-.5,-.25,0,.25,.5]);x.set_defaults(func=command_steer)
    x=s.add_parser("validate");steering_args(x,"analysis/sequential/test");x.add_argument("--steering-dir",default="analysis/sequential/steering");x.set_defaults(func=command_validate)
    return p


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
