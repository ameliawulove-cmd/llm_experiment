#!/Users/amelia/qwen_activation/.venv/bin/python3
"""
Exp 7D — Decision Formation Dynamics

How do perturbations propagate through Transformer computations,
and how does that propagation relate to the final A/B decision?

Three metrics per (trial, inject_layer, direction, downstream_layer l'):
  R_h          ‖Δh_{l'}‖ / ‖δ‖
               Overall perturbation propagation (normalized by injection norm)
  q_align      |w_{l'} · Δh_{l'}| / (‖w_{l'}‖ ‖Δh_{l'}‖ + ε)
               Normalized probe alignment [0,1]; exploratory — measures
               how aligned the perturbation is with the probe direction,
               NOT a causal claim.
  cos_dir      δ · Δh_{l'} / (‖δ‖ ‖Δh_{l'}‖ + ε)
               How much has the perturbation direction rotated?

Per-layer baseline for cos_dir interpretation:
  cos_probe    w_{l} · w_{l'} / (‖w_l‖ ‖w_{l'}‖)  [precomputed from probe weights]

Per-perturbation scalar:
  delta_M      Δ[logit(A)−logit(B)] at model output

Null model: random directions — fit ρ̂ from mean R_h decay, then
  predict exponential baseline Rec(l,l') = 1 − ρ̂^(28−l_inj) for comparison.

Commands:
  fit_probes   Refit OLS probes at all 29 layers (choice position).
               Output: results/exp7d/probe_weights.npz
  pilot        50-trial pilot: 3 inject layers, 6 directions each.
               Output: results/exp7d/pilot_results.json + timing summary
  full         All test-set trials: 7 inject layers, 22 directions each.
               Output: results/exp7d/full_results.npz
  analyze      Load results, generate plots.
               Output: figures/exp7d/
"""
# ============================================================
# 实验7D — 激活动态与有向扰动分析
# ============================================================
# 研究目的：
#   通过在 Transformer 中间层注入扰动（additive hook），
#   研究扰动如何传播到后续各层，以及对最终 A/B 选择 logit 的影响。
#   比较 DAS 子空间方向与随机方向的因果效应。
#
# 关键设计：
#   additive hook（加法干预）：h_new = h_original + epsilon * direction
#   注意：V2 使用的是 replacement hook（替换干预），不是加法！
#
# 主要命令：
#   python exp7d_dynamics.py fit_probes   重新拟合所有层的 probe
#   python exp7d_dynamics.py full         全量实验（所有层×方向）
#   python exp7d_dynamics.py directed     有向 DAS 扰动
#   python exp7d_dynamics.py rand_subspace 随机子空间基线
#
# 主要输出：
#   results/exp7d/probe_weights.npz       所有层的 probe 权重
#   results/exp7d/full_results.npz        完整扰动结果
# ============================================================
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ── Paths (match exp7_neuro.py) ───────────────────────────────────────────────
EXP6_ACTS   = Path("~/Desktop/llmneuroscientist/data/exp6_activations").expanduser()
EXP6_FULL   = Path("~/Desktop/llmneuroscientist/data/exp6_full").expanduser()
EXP1_CODE   = Path("~/Desktop/llm experiment/exp1-4").expanduser()
FMR_CODE    = Path("~/Desktop/llmneuroscientist/data/causal_rpe_bandit_code").expanduser()
OUT_BASE    = Path("~/Desktop/llmneuroscientist/results").expanduser()
FIG_DIR     = Path("~/Desktop/llmneuroscientist/figures").expanduser()
EXP7B_DIR   = OUT_BASE / "exp7b"
EXP7H_DIR   = OUT_BASE / "exp7h"
EXP7D_DIR   = OUT_BASE / "exp7d"

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
N_LAYERS      = 29    # 0=embedding + 28 block outputs
N_BLOCKS      = 28
HIDDEN_SIZE   = 1024
EPS           = 1e-8

# Inject layers (stored-activation indices, 1-indexed block outputs)
INJECT_LAYERS_PILOT = [5, 14, 27]
INJECT_LAYERS_FULL  = [5, 10, 14, 18, 22, 25, 27]

N_RANDOM_PILOT = 4    # + 1 probe = 5 dirs per non-DAS layer; + DAS dirs at L27
N_RANDOM_FULL  = 19   # + 1 probe + 2 DAS @L27 = 22 dirs total at L27

DAS_LAYER = 27
DAS_K     = 16

# ── Utilities ─────────────────────────────────────────────────────────────────

def load_trials() -> list[dict]:
    return [json.loads(l) for l in (EXP6_FULL / "trials.jsonl").read_text().splitlines() if l]


def episode_split(rows: list[dict], seed: int = 42, train_frac: float = 0.7
                  ) -> tuple[np.ndarray, np.ndarray]:
    all_eps = np.array(sorted(set(r["episode_id"] for r in rows)))
    rng = np.random.default_rng(seed)
    rng.shuffle(all_eps)
    train_eps = set(all_eps[:int(train_frac * len(all_eps))])
    tr = np.array([r["episode_id"] in train_eps for r in rows])
    return tr, ~tr


def _import_model():
    if str(EXP1_CODE) not in sys.path:
        sys.path.insert(0, str(EXP1_CODE))
    from causal_rpe_bandit import (load_model, model_device,
                                    apply_chat_template, resolve_transformer_blocks)
    return load_model, model_device, apply_chat_template, resolve_transformer_blocks


def _import_fvd():
    for p in [str(EXP1_CODE), str(FMR_CODE)]:
        if p not in sys.path:
            sys.path.insert(0, p)
    from exp6_fvd_bandit import fvd_choice_raw
    return fvd_choice_raw


def _parse_field(v):
    """Parse trial field: handles string-encoded floats, lists, bools."""
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return v
    return v


def build_prompt(row: dict, fvd_choice_raw) -> str:
    history  = _parse_field(row["history"]) or []
    bins     = _parse_field(row["bin_values_by_label"])
    la, lb   = row["q_stated_A_label"], row["q_stated_B_label"]
    return fvd_choice_raw(history, bins, la, lb)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 4 or np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def ols_probe(X_tr: np.ndarray, y_tr: np.ndarray,
              X_te: np.ndarray, y_te: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """OLS probe. Returns (r_test, w_unit, mu_train)."""
    mu = X_tr.mean(0)
    Xc_tr = X_tr - mu; Xc_te = X_te - mu
    if np.std(y_tr) < 1e-10 or len(y_tr) < 4:
        return float("nan"), np.zeros(X_tr.shape[1]), mu
    w, _, _, _ = np.linalg.lstsq(Xc_tr, y_tr - y_tr.mean(), rcond=None)
    r = corr(Xc_te @ w, y_te - y_te.mean())
    return r, (w / (np.linalg.norm(w) + EPS)), mu


# ── Probe fitting ─────────────────────────────────────────────────────────────

def command_fit_probes(args: argparse.Namespace) -> None:
    """Fit OLS probes for choice_tendency at all 29 layers (position=choice)."""
    EXP7D_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_trials()
    tr, te = episode_split(rows)
    y = np.array([float(_parse_field(r["choice_logit_diff"])) for r in rows])

    A = np.load(EXP6_ACTS / "acts_choice.npy", mmap_mode="r")  # (1500, 29, 1024)

    weights = np.zeros((N_LAYERS, HIDDEN_SIZE), dtype=np.float32)
    means   = np.zeros((N_LAYERS, HIDDEN_SIZE), dtype=np.float32)
    r_vals  = np.full(N_LAYERS, np.nan)

    for l in range(N_LAYERS):
        X = np.asarray(A[:, l, :], dtype=np.float64)
        r, w, mu = ols_probe(X[tr], y[tr], X[te], y[te])
        weights[l] = w.astype(np.float32)
        means[l]   = mu.astype(np.float32)
        r_vals[l]  = r
        print(f"  L{l:02d}  r_test={r:.3f}" if np.isfinite(r) else f"  L{l:02d}  r_test=NaN")

    out = EXP7D_DIR / "probe_weights.npz"
    np.savez(out, weights=weights, means=means, r_vals=r_vals)
    print(f"\nSaved {out}")
    print(f"Peak: L{int(np.nanargmax(r_vals))}  r={np.nanmax(r_vals):.3f}")

    # Also compute and save cross-layer probe cosine similarities
    norms   = np.linalg.norm(weights, axis=1, keepdims=True) + EPS
    w_norm  = weights / norms
    cos_probe_pairs = w_norm @ w_norm.T  # (29, 29)
    np.save(EXP7D_DIR / "cos_probe_pairs.npy", cos_probe_pairs)
    print("Saved cos_probe_pairs.npy  (29×29 cross-layer probe cosine similarity)")


# ── Forward pass helpers ───────────────────────────────────────────────────────

def _tokenize(prompt: str, tokenizer, device, apply_chat_template):
    full = apply_chat_template(tokenizer, prompt)
    enc  = tokenizer(full, return_tensors="pt", add_special_tokens=False)
    return {k: v.to(device) for k, v in enc.items()}


def _logit_margin(logits, tokenizer, torch_mod) -> float:
    A_id = tokenizer.encode("A", add_special_tokens=False)[-1]
    B_id = tokenizer.encode("B", add_special_tokens=False)[-1]
    return float(logits[0, -1, A_id].cpu()) - float(logits[0, -1, B_id].cpu())


def run_baseline(enc: dict, model, torch_mod) -> tuple[np.ndarray, float]:
    """Returns h_base (29, 1024) float32 and logit_margin float."""
    with torch_mod.inference_mode():
        out = model(**enc, output_hidden_states=True, return_dict=True)
    h = np.stack([hs[0, -1].detach().float().cpu().numpy()
                  for hs in out.hidden_states], axis=0)  # (29, 1024)
    margin = float(out.logits[0, -1, :].cpu().numpy()[
        # compute margin directly from logits tensor
        0]) - 0.0  # placeholder, recompute below
    # proper margin computation
    A_id = None; B_id = None
    return h, out.logits  # return logits for margin computation outside


def run_perturbed(enc: dict, model, torch_mod, blocks,
                  l_inj: int, delta: np.ndarray, torch) -> tuple[np.ndarray, object]:
    """在指定层注入扰动向量，返回扰动后的 hidden states 和 logits。

    参数：
        enc: tokenized 输入，包含 input_ids 等
        model: Qwen 模型（不更新权重）
        l_inj: int，注入层的 hidden_states 索引（如 27 → blocks[26]）
        delta: shape (1024,)，float32，要加到最后一个 token 上的扰动向量
    返回：
        h_perturbed: shape (29, 1024)，扰动后各层最后一个 token 的 hidden state
        logits: 模型输出的词表 logits，用于计算 A/B margin
    实现细节：
        在 blocks[l_inj-1] 上注册 forward hook（加法干预）
        hook: h_new[0, -1, :] = h_old[0, -1, :] + delta
        注意：这是加法干预（additive），不是替换干预（replacement）
              V2 验证中使用的是替换干预，更精确地控制干预大小
    原始实现说明：
        Inject delta (float32, shape 1024) at stored-activation layer l_inj.
        Implementation: register_forward_hook on blocks[l_inj-1].
        Blocks are 0-indexed (blocks[0]..blocks[27]).
        blocks[l_inj-1] output = hidden_states[l_inj].
    """
    delta_t = torch.tensor(delta, dtype=model.dtype,
                           device=next(model.parameters()).device)

    def _hook(module, inp, out_h):
        tensor = out_h[0] if isinstance(out_h, tuple) else out_h
        new_tensor = tensor.clone()
        new_tensor[0, -1, :] = new_tensor[0, -1, :] + delta_t
        if isinstance(out_h, tuple):
            return (new_tensor,) + out_h[1:]
        return new_tensor

    # blocks[l_inj-1] post-hook modifies output → hidden_states[l_inj]
    # For l_inj=0 this would need special handling; we only inject at l_inj>=1
    block_idx = l_inj - 1
    handle = blocks[block_idx].register_forward_hook(_hook)
    try:
        with torch_mod.inference_mode():
            out = model(**enc, output_hidden_states=True, return_dict=True)
    finally:
        handle.remove()

    h_perturbed = np.stack([hs[0, -1].detach().float().cpu().numpy()
                            for hs in out.hidden_states], axis=0)  # (29, 1024)
    return h_perturbed, out.logits


def compute_metrics(h_base: np.ndarray, h_pert: np.ndarray,
                    delta_norm: float, probe_weights: np.ndarray,
                    l_inj: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns R_h, q_align, cos_dir — each shape (29,), with NaN for l' <= l_inj.
    h_base, h_pert: (29, 1024); probe_weights: (29, 1024); delta_norm: ‖δ‖
    """
    R_h     = np.full(N_LAYERS, np.nan, dtype=np.float32)
    q_align = np.full(N_LAYERS, np.nan, dtype=np.float32)
    cos_dir = np.full(N_LAYERS, np.nan, dtype=np.float32)

    delta_inj = h_pert[l_inj] - h_base[l_inj]  # actual injection ≈ delta (plus any nonlin)

    for lp in range(l_inj + 1, N_LAYERS):
        dh = (h_pert[lp] - h_base[lp]).astype(np.float64)
        dh_norm = np.linalg.norm(dh)

        # R_h: normalized magnitude propagation
        R_h[lp] = float(dh_norm / (delta_norm + EPS))

        # q_align: cosine between Δh_{l'} and probe direction w_{l'}
        w = probe_weights[lp].astype(np.float64)
        w_norm = np.linalg.norm(w)
        if dh_norm > EPS and w_norm > EPS:
            q_align[lp] = float(abs(w @ dh) / (w_norm * dh_norm + EPS))

        # cos_dir: cosine between initial Δh_{l_inj} and downstream Δh_{l'}
        d_inj_norm = np.linalg.norm(delta_inj)
        if dh_norm > EPS and d_inj_norm > EPS:
            cos_dir[lp] = float(delta_inj @ dh / (d_inj_norm * dh_norm + EPS))

    return R_h, q_align, cos_dir


def build_directions(l_inj: int, probe_weights: np.ndarray, R_das: np.ndarray | None,
                     n_random: int, rng: np.random.Generator, eps_scale: float = 0.01,
                     h_norm: float = 1.0) -> list[tuple[str, np.ndarray]]:
    """
    Build perturbation directions for injection at layer l_inj.
    eps = eps_scale × h_norm so perturbation is 1% of the residual stream norm.
    Returns list of (label, delta_vector_float32).
    """
    eps = eps_scale * h_norm
    dirs = []

    # Probe direction (unit-norm probe weight × eps)
    w = probe_weights[l_inj].astype(np.float64)
    w_norm = np.linalg.norm(w)
    if w_norm > EPS:
        dirs.append(("probe", (eps * w / w_norm).astype(np.float32)))

    # DAS causal subspace and orthogonal complement (only at DAS_LAYER)
    if l_inj == DAS_LAYER and R_das is not None:
        P     = R_das @ R_das.T   # (1024, 1024) causal projection
        v_rnd = rng.standard_normal(HIDDEN_SIZE).astype(np.float64)
        # causal: project into causal subspace
        v_causal = P @ v_rnd
        vc_norm  = np.linalg.norm(v_causal)
        if vc_norm > EPS:
            dirs.append(("das_causal", (eps * v_causal / vc_norm).astype(np.float32)))
        # orthogonal complement
        v_orth = v_rnd - v_causal
        vo_norm = np.linalg.norm(v_orth)
        if vo_norm > EPS:
            dirs.append(("das_orth", (eps * v_orth / vo_norm).astype(np.float32)))

    # Random directions
    for i in range(n_random):
        v = rng.standard_normal(HIDDEN_SIZE).astype(np.float64)
        v /= np.linalg.norm(v) + EPS
        dirs.append((f"random_{i:02d}", (eps * v).astype(np.float32)))

    return dirs


# ── Pilot command ─────────────────────────────────────────────────────────────

def command_pilot(args: argparse.Namespace) -> None:
    """50-trial pilot: 3 inject layers, (4 random + probe + DAS@L27) directions."""
    EXP7D_DIR.mkdir(parents=True, exist_ok=True)

    # Load probe weights
    probe_path = EXP7D_DIR / "probe_weights.npz"
    if not probe_path.exists():
        sys.exit("ERROR: run fit_probes first.")
    probe_npz      = np.load(probe_path)
    probe_weights  = probe_npz["weights"]   # (29, 1024) float32
    probe_r        = probe_npz["r_vals"]    # (29,)

    # Load DAS rotation for L27
    das_path = EXP7H_DIR / f"R_choice_tendency_L{DAS_LAYER}_k{DAS_K}.npy"
    R_das = np.load(das_path).astype(np.float64) if das_path.exists() else None
    if R_das is None:
        print(f"WARNING: DAS matrix not found at {das_path}, skipping das directions")

    # Load model
    import torch as torch_lib
    load_model, model_device, apply_chat_template, resolve_transformer_blocks = _import_model()
    fvd_choice_raw = _import_fvd()
    torch_mod, model, tokenizer = load_model(args.model, args.device, args.dtype)
    dev    = model_device(model)
    blocks = resolve_transformer_blocks(model)
    print(f"Model loaded. Blocks: {len(blocks)}  Device: {dev}")

    # Decode A/B token ids once
    A_id = tokenizer.encode("A", add_special_tokens=False)[-1]
    B_id = tokenizer.encode("B", add_special_tokens=False)[-1]

    # Episode split → test set
    rows = load_trials()
    _, te_mask = episode_split(rows)
    test_rows  = [r for r, m in zip(rows, te_mask) if m]
    pilot_rows = test_rows[:args.n_pilot]
    print(f"Pilot: {len(pilot_rows)} trials from {len(test_rows)} test-set trials")

    # Cross-layer probe cosine pairs (precomputed)
    cos_probe_pairs_path = EXP7D_DIR / "cos_probe_pairs.npy"
    cos_probe_pairs = np.load(cos_probe_pairs_path) if cos_probe_pairs_path.exists() else None

    rng     = np.random.default_rng(args.seed)
    records = []
    t_start = time.time()

    for ti, row in enumerate(pilot_rows):
        prompt = build_prompt(row, fvd_choice_raw)
        enc    = _tokenize(prompt, tokenizer, dev, apply_chat_template)

        # Baseline forward pass
        with torch_lib.inference_mode():
            out_base = model(**enc, output_hidden_states=True, return_dict=True)
        h_base     = np.stack([hs[0, -1].detach().float().cpu().numpy()
                               for hs in out_base.hidden_states], axis=0)
        logits_base = out_base.logits[0, -1].detach().float().cpu().numpy()
        M_base      = float(logits_base[A_id] - logits_base[B_id])
        M_base_cal  = float(_parse_field(row["choice_logit_diff"]))

        for l_inj in args.inject_layers:
            h_norm = float(np.linalg.norm(h_base[l_inj]))
            dirs   = build_directions(l_inj, probe_weights, R_das,
                                      n_random=args.n_random, rng=rng,
                                      eps_scale=args.eps_scale, h_norm=h_norm)

            for dir_label, delta in dirs:
                delta_norm = float(np.linalg.norm(delta))

                h_pert, logits_pert = run_perturbed(
                    enc, model, torch_lib, blocks, l_inj, delta, torch_lib)
                lp_arr = logits_pert[0, -1].detach().float().cpu().numpy()
                M_pert = float(lp_arr[A_id] - lp_arr[B_id])

                R_h, q_align, cos_dir = compute_metrics(
                    h_base, h_pert, delta_norm, probe_weights, l_inj)

                records.append({
                    "trial_idx":    ti,
                    "episode_id":   row["episode_id"],
                    "l_inj":        l_inj,
                    "dir_label":    dir_label,
                    "delta_norm":   delta_norm,
                    "M_base":       M_base,
                    "M_base_cal":   M_base_cal,
                    "M_pert":       M_pert,
                    "delta_M":      M_pert - M_base,
                    "chosen_arm":   row["chosen_arm"],
                    "b_collapse":   (row["chosen_arm"] == "B" and
                                     float(_parse_field(row["q_stated_A"])) >
                                     float(_parse_field(row["q_stated_B"]))),
                    "R_h":          R_h.tolist(),
                    "q_align":      q_align.tolist(),
                    "cos_dir":      cos_dir.tolist(),
                })

        elapsed = time.time() - t_start
        rate    = (ti + 1) / elapsed
        eta     = (len(pilot_rows) - ti - 1) / rate if rate > 0 else 0
        print(f"  trial {ti+1:02d}/{len(pilot_rows)}  "
              f"elapsed={elapsed:.1f}s  rate={rate:.2f} trial/s  ETA={eta:.0f}s",
              flush=True)

    elapsed_total = time.time() - t_start
    print(f"\nPilot complete: {len(records)} records in {elapsed_total:.1f}s")
    print(f"Rate: {len(pilot_rows)/elapsed_total:.2f} trials/s  "
          f"(est full run {len(test_rows)}/{len(pilot_rows)}×{elapsed_total:.0f}s ≈ "
          f"{len(test_rows)/len(pilot_rows)*elapsed_total/60:.1f} min)")

    # Save
    out_path = EXP7D_DIR / "pilot_results.json"
    with open(out_path, "w") as f:
        json.dump({"records": records,
                   "probe_r": probe_r.tolist(),
                   "inject_layers": args.inject_layers,
                   "n_pilot": len(pilot_rows),
                   "n_random": args.n_random,
                   "eps_scale": args.eps_scale,
                   "das_layer": DAS_LAYER,
                   "das_k": DAS_K,
                   "elapsed_s": elapsed_total}, f, indent=2)
    print(f"Saved {out_path}  ({out_path.stat().st_size // 1024} KB)")

    # Spot-check summary
    import math
    _pilot_summary(records, probe_r)


def _pilot_summary(records: list[dict], probe_r: np.ndarray) -> None:
    """Print a quick sanity-check table after the pilot."""
    print("\n── Pilot Spot-Check ─────────────────────────────────────────────────────")
    print(f"{'l_inj':>6} {'dir_label':>14} {'mean|ΔM|':>10} "
          f"{'R_h@l_inj+1':>12} {'q_align@27':>12}")
    print("─" * 60)
    from collections import defaultdict
    by_key = defaultdict(list)
    for rec in records:
        by_key[(rec["l_inj"], rec["dir_label"])].append(rec)
    for (l_inj, dir_label), recs in sorted(by_key.items()):
        mean_absdM = float(np.mean([abs(r["delta_M"]) for r in recs]))
        rh_lp1     = float(np.nanmean([r["R_h"][l_inj + 1] for r in recs]))
        qa_27_vals = [r["q_align"][27] for r in recs
                      if l_inj < 27 and not np.isnan(r["q_align"][27])]
        qa_27      = float(np.mean(qa_27_vals)) if qa_27_vals else float("nan")
        print(f"{l_inj:>6} {dir_label:>14} {mean_absdM:>10.4f} "
              f"{rh_lp1:>12.4f} {qa_27:>12.4f}")
    print("─" * 60)
    print(f"Probe r@L27 = {probe_r[27]:.3f}  (sanity: should be ~0.97)")


# ── Full experiment ────────────────────────────────────────────────────────────

def command_full(args: argparse.Namespace) -> None:
    """Full experiment: all test-set trials, 7 inject layers, 22 directions."""
    EXP7D_DIR.mkdir(parents=True, exist_ok=True)
    probe_npz     = np.load(EXP7D_DIR / "probe_weights.npz")
    probe_weights = probe_npz["weights"]
    probe_r       = probe_npz["r_vals"]

    das_path = EXP7H_DIR / f"R_choice_tendency_L{DAS_LAYER}_k{DAS_K}.npy"
    R_das    = np.load(das_path).astype(np.float64) if das_path.exists() else None

    import torch as torch_lib
    load_model, model_device, apply_chat_template, resolve_transformer_blocks = _import_model()
    fvd_choice_raw = _import_fvd()
    torch_mod, model, tokenizer = load_model(args.model, args.device, args.dtype)
    dev    = model_device(model)
    blocks = resolve_transformer_blocks(model)

    A_id = tokenizer.encode("A", add_special_tokens=False)[-1]
    B_id = tokenizer.encode("B", add_special_tokens=False)[-1]

    rows = load_trials()
    _, te_mask = episode_split(rows)
    test_rows  = [r for r, m in zip(rows, te_mask) if m]
    print(f"Full experiment: {len(test_rows)} test-set trials")

    rng = np.random.default_rng(args.seed)

    # Pre-allocate storage
    # We'll store flat arrays: one entry per (trial, inject, direction)
    # then numpy-compress at the end
    all_records = []
    t_start = time.time()

    for ti, row in enumerate(test_rows):
        prompt = build_prompt(row, fvd_choice_raw)
        enc    = _tokenize(prompt, tokenizer, dev, apply_chat_template)

        with torch_lib.inference_mode():
            out_base = model(**enc, output_hidden_states=True, return_dict=True)
        h_base   = np.stack([hs[0, -1].detach().float().cpu().numpy()
                             for hs in out_base.hidden_states], axis=0)
        lg_base    = out_base.logits[0, -1].detach().float().cpu().numpy()
        M_base     = float(lg_base[A_id] - lg_base[B_id])
        M_base_cal = float(_parse_field(row["choice_logit_diff"]))

        for l_inj in args.inject_layers:
            h_norm = float(np.linalg.norm(h_base[l_inj]))
            dirs   = build_directions(l_inj, probe_weights, R_das,
                                      n_random=args.n_random, rng=rng,
                                      eps_scale=args.eps_scale, h_norm=h_norm)

            for dir_label, delta in dirs:
                delta_norm = float(np.linalg.norm(delta))
                h_pert, logits_pert = run_perturbed(
                    enc, model, torch_lib, blocks, l_inj, delta, torch_lib)
                lp_arr = logits_pert[0, -1].detach().float().cpu().numpy()
                M_pert = float(lp_arr[A_id] - lp_arr[B_id])

                R_h, q_align, cos_dir = compute_metrics(
                    h_base, h_pert, delta_norm, probe_weights, l_inj)

                all_records.append({
                    "trial_idx": ti, "l_inj": l_inj,
                    "dir_label": dir_label, "delta_norm": delta_norm,
                    "M_base": M_base, "M_base_cal": M_base_cal,
                    "delta_M": M_pert - M_base,
                    "b_collapse": (row["chosen_arm"] == "B" and
                                   float(_parse_field(row["q_stated_A"])) >
                                   float(_parse_field(row["q_stated_B"]))),
                    "R_h": R_h, "q_align": q_align, "cos_dir": cos_dir,
                })

        if (ti + 1) % 50 == 0:
            el = time.time() - t_start
            print(f"  {ti+1}/{len(test_rows)}  {el:.0f}s  "
                  f"ETA {(len(test_rows)-ti-1)/(ti+1)*el:.0f}s", flush=True)

        # Checkpoint every 100 trials
        if (ti + 1) % 100 == 0:
            _save_full(all_records, EXP7D_DIR / "full_results_checkpoint.npz")

    _save_full(all_records, EXP7D_DIR / "full_results.npz")
    print(f"\nFull experiment done: {len(all_records)} records  "
          f"({(time.time()-t_start)/60:.1f} min)")


def _save_full(records: list[dict], path: Path) -> None:
    """Pack list-of-dicts into compact npz arrays."""
    n = len(records)
    R_h_arr    = np.array([r["R_h"]    for r in records], dtype=np.float32)
    qa_arr     = np.array([r["q_align"] for r in records], dtype=np.float32)
    cd_arr     = np.array([r["cos_dir"] for r in records], dtype=np.float32)
    meta_arr   = np.array([[r["trial_idx"], r["l_inj"],
                            r["delta_norm"], r["M_base"], r["delta_M"],
                            float(r["b_collapse"]), r.get("M_base_cal", np.nan)]
                           for r in records], dtype=np.float32)
    dir_labels = [r["dir_label"] for r in records]
    np.savez_compressed(path,
                        R_h=R_h_arr, q_align=qa_arr, cos_dir=cd_arr,
                        meta=meta_arr,
                        dir_labels=np.array(dir_labels))
    print(f"  Saved {path}  ({path.stat().st_size // 1024} KB)")


# ── Analyze command ────────────────────────────────────────────────────────────

def command_analyze(args: argparse.Namespace) -> None:
    """Load pilot or full results, generate summary plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; printing text summary only")
        plt = None

    # Load
    fname_map = {
        "pilot":      "pilot_results.json",
        "full":        "full_results.npz",
        "calibrated": "full_results_calibrated.npz",
    }
    data_path = EXP7D_DIR / fname_map[args.source]
    if not data_path.exists():
        sys.exit(f"ERROR: {data_path} not found. "
                 f"Run 'full' (then 'recalibrate' for calibrated) first.")

    if args.source == "pilot":
        data    = json.load(open(data_path))
        recs    = data["records"]
        probe_r = np.array(data["probe_r"])
        _analyze_pilot(recs, probe_r, plt, args)
    else:
        npz = np.load(data_path, allow_pickle=True)
        _analyze_full(npz, plt, args)


def _analyze_pilot(recs: list[dict], probe_r: np.ndarray, plt, args) -> None:
    from collections import defaultdict

    out_dir = FIG_DIR / "exp7d"
    out_dir.mkdir(parents=True, exist_ok=True)

    inject_layers = sorted(set(r["l_inj"] for r in recs))
    all_dirs      = sorted(set(r["dir_label"] for r in recs))

    print("\n══ PILOT ANALYSIS ══════════════════════════════════════════════════")
    print(f"Inject layers: {inject_layers}")
    print(f"Direction types: {all_dirs}")
    print(f"Total records: {len(recs)}")

    # Summary: mean |ΔM| per inject layer × direction
    print(f"\n{'l_inj':>6} {'direction':>14} {'|ΔM| mean':>10} {'|ΔM| std':>10} "
          f"{'R_h@28 mean':>12}")
    print("─" * 58)
    by_key = defaultdict(list)
    for r in recs:
        by_key[(r["l_inj"], r["dir_label"])].append(r)

    for (l_inj, dlabel), group in sorted(by_key.items()):
        dMs  = [abs(r["delta_M"]) for r in group]
        rh28 = [r["R_h"][28] for r in group if not np.isnan(r["R_h"][28])]
        print(f"{l_inj:>6} {dlabel:>14} "
              f"{np.mean(dMs):>10.4f} {np.std(dMs):>10.4f} "
              f"{np.nanmean(rh28) if rh28 else float('nan'):>12.4f}")

    # R_h propagation curves by inject layer
    if plt is not None:
        fig, axes = plt.subplots(1, len(inject_layers), figsize=(5*len(inject_layers), 4),
                                 sharey=True)
        layers_plot = np.arange(N_LAYERS)
        for ax, l_inj in zip(axes, inject_layers):
            for dlabel in all_dirs:
                grp   = [r for r in by_key[(l_inj, dlabel)]]
                if not grp:
                    continue
                R_h_mat = np.array([r["R_h"] for r in grp])  # (n, 29)
                mean_Rh = np.nanmean(R_h_mat, axis=0)
                color   = "blue" if "probe" in dlabel else \
                          "red" if "das_causal" in dlabel else \
                          "green" if "das_orth" in dlabel else "gray"
                alpha   = 0.8 if "random" not in dlabel else 0.3
                lw      = 2.0 if "random" not in dlabel else 0.8
                ax.plot(layers_plot, mean_Rh, color=color, alpha=alpha, lw=lw,
                        label=dlabel)
            ax.axvline(l_inj, color="black", ls="--", lw=1, label=f"inject @{l_inj}")
            ax.set_title(f"Inject L{l_inj}")
            ax.set_xlabel("Downstream layer l'")
            ax.set_ylabel("R_h = ‖Δh‖/‖δ‖")
            ax.legend(fontsize=7, loc="upper left")
            ax.set_xlim(0, 28)
        plt.suptitle("Perturbation Propagation: R_h across layers")
        plt.tight_layout()
        fig.savefig(out_dir / "pilot_R_h_propagation.png", dpi=150)
        plt.close(fig)
        print(f"\nSaved {out_dir / 'pilot_R_h_propagation.png'}")

        # q_align propagation
        fig, axes = plt.subplots(1, len(inject_layers), figsize=(5*len(inject_layers), 4),
                                 sharey=True)
        for ax, l_inj in zip(axes, inject_layers):
            for dlabel in all_dirs:
                grp   = [r for r in by_key[(l_inj, dlabel)]]
                if not grp:
                    continue
                qa_mat  = np.array([r["q_align"] for r in grp])
                mean_qa = np.nanmean(qa_mat, axis=0)
                color   = "blue" if "probe" in dlabel else \
                          "red" if "das_causal" in dlabel else \
                          "green" if "das_orth" in dlabel else "gray"
                alpha   = 0.8 if "random" not in dlabel else 0.3
                lw      = 2.0 if "random" not in dlabel else 0.8
                ax.plot(layers_plot, mean_qa, color=color, alpha=alpha, lw=lw,
                        label=dlabel)
            ax.axvline(l_inj, color="black", ls="--", lw=1)
            ax.set_title(f"Inject L{l_inj}")
            ax.set_xlabel("Downstream layer l'")
            ax.set_ylabel("q_align (probe alignment)")
            ax.legend(fontsize=7)
            ax.set_xlim(0, 28); ax.set_ylim(0, 1)
        plt.suptitle("Probe Alignment: q_align = |w·Δh|/(‖w‖‖Δh‖)")
        plt.tight_layout()
        fig.savefig(out_dir / "pilot_q_align.png", dpi=150)
        plt.close(fig)
        print(f"Saved {out_dir / 'pilot_q_align.png'}")

        # |ΔM| vs inject layer by direction type
        fig, ax = plt.subplots(figsize=(7, 4))
        width   = 0.15
        x_pos   = np.arange(len(inject_layers))
        for j, dlabel in enumerate(all_dirs):
            vals = [np.mean([abs(r["delta_M"]) for r in by_key[(l, dlabel)]])
                    for l in inject_layers]
            ax.bar(x_pos + j*width - width*len(all_dirs)/2, vals, width,
                   label=dlabel, alpha=0.8)
        ax.set_xticks(x_pos); ax.set_xticklabels([f"L{l}" for l in inject_layers])
        ax.set_xlabel("Injection layer"); ax.set_ylabel("|ΔM|  (logit margin change)")
        ax.legend(fontsize=8); ax.set_title("Decision Sensitivity by Injection Layer")
        plt.tight_layout()
        fig.savefig(out_dir / "pilot_delta_M.png", dpi=150)
        plt.close(fig)
        print(f"Saved {out_dir / 'pilot_delta_M.png'}")

    print("\nNote: q_align is representational alignment, not causal decision specificity.")
    print("      |ΔM| is the direct decision impact measure.")
    print("      Compare probe/das_causal against random baseline before drawing conclusions.")


def _analyze_full(npz, plt, args) -> None:
    """Full analysis: perturbation propagation, decision sensitivity, DAS validation."""
    meta       = npz["meta"]       # (N,6): trial_idx, l_inj, delta_norm, M_base, delta_M, b_collapse
    R_h        = npz["R_h"]        # (N,29)
    q_align    = npz["q_align"]    # (N,29)
    cos_dir    = npz["cos_dir"]    # (N,29)
    dir_labels = npz["dir_labels"] # (N,)

    col = {"trial_idx":0,"l_inj":1,"delta_norm":2,"M_base":3,"delta_M":4,"b_collapse":5,"M_base_cal":6}
    has_cal = meta.shape[1] > 6
    inject_layers = sorted(set(meta[:, col["l_inj"]].astype(int).tolist()))
    all_dirs      = sorted(set(dir_labels.tolist()))
    N             = meta.shape[0]

    print(f"\n══ FULL ANALYSIS ═══════════════════════════════════════════════════")
    print(f"Records: {N}  Inject layers: {inject_layers}")
    print(f"Direction types: {all_dirs}\n")

    out_dir = FIG_DIR / "exp7d"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load layer norms for absolute perturbation comparison
    layer_norm_path = EXP6_ACTS / "acts_choice.npy"
    if layer_norm_path.exists():
        A = np.load(layer_norm_path, mmap_mode="r")
        mean_layer_norms = np.linalg.norm(A, axis=2).mean(axis=0)  # (29,)
        print("Layer norms (mean ‖h_l‖) and absolute injection magnitude (eps=1%):")
        for l in inject_layers:
            eps_abs = 0.01 * mean_layer_norms[l]
            print(f"  L{l:02d}: ‖h_l‖={mean_layer_norms[l]:.1f}  ‖δ‖={eps_abs:.3f}  "
                  f"abs‖Δh_28‖≈R_h@28×{eps_abs:.3f}")
        print(f"  L28: ‖h_28‖={mean_layer_norms[28]:.1f}  "
              f"(all injections ≈ {0.01*mean_layer_norms[5]/mean_layer_norms[28]*100:.2f}% "
              f"to {0.01*mean_layer_norms[27]/mean_layer_norms[28]*100:.2f}% of ‖h_28‖ at destination)")
        print("  NOTE: R_h is not directly comparable across layers due to norm growth.\n"
              "  Use abs_Dh28 = R_h × ‖δ_l‖ for fair comparison.\n")
    else:
        mean_layer_norms = None

    # ── 1. |ΔM| summary table ────────────────────────────────────────────────
    print(f"{'l_inj':>6} {'direction':>14} {'|ΔM| mean':>10} {'|ΔM| std':>9} "
          f"{'ΔM mean':>9} {'flip%':>7} "
          f"{'vs rand':>8} {'abs‖Δh28‖':>11}")
    print("─" * 82)

    # Compute random-direction baseline per inject layer
    rand_dM_by_linj = {}
    for l_inj in inject_layers:
        mask = ((meta[:, col["l_inj"]] == l_inj) &
                np.array(["random" in d for d in dir_labels]))
        rand_dM_by_linj[l_inj] = float(np.mean(np.abs(meta[mask, col["delta_M"]]))) if mask.any() else np.nan

    for l_inj in inject_layers:
        for dlabel in all_dirs:
            mask = ((meta[:, col["l_inj"]] == l_inj) &
                    (dir_labels == dlabel))
            if not mask.any():
                continue
            dMs_signed = meta[mask, col["delta_M"]]
            dMs        = np.abs(dMs_signed)
            M_base     = meta[mask, col["M_base"]]
            # Use calibrated M_base for flip (raw is biased ~+4 by letter prior)
            M_for_flip = (meta[mask, col["M_base_cal"]] if has_cal
                          else M_base)
            flipped    = ((M_for_flip > 0) & (M_for_flip + dMs_signed < 0)) | \
                         ((M_for_flip < 0) & (M_for_flip + dMs_signed > 0))
            flip_pct   = float(100 * flipped.mean())
            rh28       = R_h[mask, 28]
            # absolute ‖Δh_{28}‖ = R_h × ‖δ_l‖ = R_h × 0.01 × ‖h_l‖
            abs_dh28   = np.nanmean(rh28) * (0.01 * mean_layer_norms[l_inj]) \
                         if mean_layer_norms is not None else np.nan
            rand_base  = rand_dM_by_linj.get(l_inj, np.nan)
            ratio      = float(np.mean(dMs) / rand_base) if rand_base > 0 else np.nan
            print(f"{l_inj:>6} {dlabel:>14} "
                  f"{np.mean(dMs):>10.4f} {np.std(dMs):>9.4f} "
                  f"{np.mean(dMs_signed):>9.4f} {flip_pct:>7.1f}% "
                  f"{ratio:>8.2f}x "
                  f"{abs_dh28:>11.4f}")
        print()

    if plt is None:
        return

    layers_arr = np.arange(N_LAYERS)

    # ── 2. R_h propagation curves ────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(inject_layers), figsize=(4*len(inject_layers), 4), sharey=False)
    if len(inject_layers) == 1:
        axes = [axes]
    for ax, l_inj in zip(axes, inject_layers):
        for dlabel in all_dirs:
            mask = ((meta[:, col["l_inj"]] == l_inj) & (dir_labels == dlabel))
            if not mask.any():
                continue
            mean_Rh = np.nanmean(R_h[mask], axis=0)
            color = ("royalblue" if dlabel == "probe" else
                     "crimson"   if dlabel == "das_causal" else
                     "seagreen"  if dlabel == "das_orth" else "lightgray")
            lw    = 2.2 if dlabel in ("probe","das_causal","das_orth") else 0.7
            alpha = 0.9 if dlabel in ("probe","das_causal","das_orth") else 0.4
            ax.semilogy(layers_arr, mean_Rh, color=color, lw=lw, alpha=alpha,
                        label=dlabel)
        ax.axvline(l_inj, color="black", ls="--", lw=1)
        ax.set_title(f"Inject L{l_inj}")
        ax.set_xlabel("Downstream layer l'")
        ax.set_ylabel("R_h = ‖Δh‖/‖δ‖  (log scale)")
        ax.legend(fontsize=7)
    plt.suptitle("Perturbation Propagation  (n=450 trials)", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_dir / "full_R_h_propagation.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir}/full_R_h_propagation.png")

    # ── 3. |ΔM| bar chart: probe & DAS vs random baseline ───────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    special_dirs = [d for d in all_dirs if d in ("probe","das_causal","das_orth")]
    rand_dirs    = [d for d in all_dirs if "random" in d]
    x_pos  = np.arange(len(inject_layers))
    width  = 0.22
    colors = {"probe":"royalblue","das_causal":"crimson","das_orth":"seagreen"}

    # random baseline band
    rand_means = []
    rand_stds  = []
    for l_inj in inject_layers:
        mask = ((meta[:, col["l_inj"]] == l_inj) &
                np.array(["random" in d for d in dir_labels]))
        vals = np.abs(meta[mask, col["delta_M"]]) if mask.any() else np.array([0.])
        rand_means.append(np.mean(vals)); rand_stds.append(np.std(vals))
    rand_means = np.array(rand_means); rand_stds = np.array(rand_stds)
    ax.bar(x_pos, rand_means, width*len(special_dirs)+0.05, color="lightgray",
           alpha=0.5, label="random baseline", zorder=1)
    ax.errorbar(x_pos, rand_means, rand_stds, fmt="none", color="gray", capsize=3, zorder=2)

    for j, dlabel in enumerate(special_dirs):
        vals_per_l = []
        stds_per_l = []
        for l_inj in inject_layers:
            mask = ((meta[:, col["l_inj"]] == l_inj) & (dir_labels == dlabel))
            v = np.abs(meta[mask, col["delta_M"]]) if mask.any() else np.array([0.])
            vals_per_l.append(np.mean(v)); stds_per_l.append(np.std(v))
        offset = (j - len(special_dirs)/2 + 0.5) * width
        ax.bar(x_pos + offset, vals_per_l, width, color=colors[dlabel], alpha=0.9,
               label=dlabel, zorder=3)
        ax.errorbar(x_pos + offset, vals_per_l, stds_per_l,
                    fmt="none", color="black", capsize=3, zorder=4)

    ax.set_xticks(x_pos); ax.set_xticklabels([f"L{l}" for l in inject_layers])
    ax.set_xlabel("Injection layer")
    ax.set_ylabel("|ΔM|  (logit margin change)")
    ax.legend(fontsize=9)
    ax.set_title("|ΔM| by injection layer and direction  (n=450 trials)\n"
                 "Gray = random baseline  ·  Bars = special directions")
    plt.tight_layout()
    fig.savefig(out_dir / "full_delta_M_bars.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir}/full_delta_M_bars.png")

    # ── 4. q_align across layers: probe vs random ────────────────────────────
    fig, axes = plt.subplots(1, len(inject_layers), figsize=(4*len(inject_layers), 4), sharey=True)
    if len(inject_layers) == 1:
        axes = [axes]
    for ax, l_inj in zip(axes, inject_layers):
        for dlabel in ("probe",) + tuple(d for d in rand_dirs[:2]):
            mask = ((meta[:, col["l_inj"]] == l_inj) & (dir_labels == dlabel))
            if not mask.any():
                continue
            mean_qa = np.nanmean(q_align[mask], axis=0)
            color = "royalblue" if dlabel == "probe" else "lightgray"
            lw    = 2.0 if dlabel == "probe" else 0.8
            ax.plot(layers_arr, mean_qa, color=color, lw=lw, label=dlabel)
        ax.axvline(l_inj, color="black", ls="--", lw=1)
        ax.set_title(f"Inject L{l_inj}")
        ax.set_xlabel("Downstream layer l'")
        ax.set_ylabel("q_align (probe alignment)")
        ax.set_xlim(0, 28); ax.set_ylim(0, 0.3)
        ax.legend(fontsize=7)
    plt.suptitle("Probe Alignment q_align across layers  [exploratory]", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_dir / "full_q_align.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir}/full_q_align.png")

    # ── 5. B-collapse stratification ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (label, collapse_val) in enumerate(
            [("B-collapse (n≈784)", 1.0), ("non-B-collapse (n≈716)", 0.0)]):
        collapse_mask = (meta[:, col["b_collapse"]] == collapse_val)
        vals_per_l = []
        for l_inj in inject_layers:
            for dlabel in ["das_causal"] if 27 in inject_layers else []:
                mask = (collapse_mask &
                        (meta[:, col["l_inj"]] == l_inj) &
                        (dir_labels == dlabel))
                v = np.abs(meta[mask, col["delta_M"]]) if mask.any() else np.array([np.nan])
                vals_per_l.append((l_inj, dlabel, float(np.nanmean(v))))
        if not vals_per_l:
            continue
        ls_list = sorted(set(v[0] for v in vals_per_l))
        for dlabel in set(v[1] for v in vals_per_l):
            ys = [v[2] for v in vals_per_l if v[1] == dlabel]
            ax.plot(ls_list[:len(ys)], ys, "o-",
                    label=f"{label} | {dlabel}", alpha=0.8)
    ax.set_xlabel("Injection layer")
    ax.set_ylabel("|ΔM|")
    ax.legend(fontsize=8)
    ax.set_title("B-collapse vs non-B-collapse: DAS causal direction |ΔM|")
    plt.tight_layout()
    fig.savefig(out_dir / "full_bcollapse_stratification.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir}/full_bcollapse_stratification.png")

    print("\n── Interpretation notes ─────────────────────────────────────────────")
    print("R_h@28: amplification ratio at final layer (>1 = grows, <1 = shrinks)")
    print("'vs rand mean': how many times larger than random baseline direction")
    print("q_align: exploratory probe alignment (not causal; see |ΔM| for causality)")
    print("das_causal vs das_orth at L27: key DAS cross-validation comparison")


# ── Stats command (no model required) ────────────────────────────────────────

def command_stats(args: argparse.Namespace) -> None:
    """Episode-level bootstrap CI + paired significance tests on saved .npz data.
    No model required — reads full_results_calibrated.npz (or full_results.npz).
    """
    EXP7D_DIR.mkdir(parents=True, exist_ok=True)
    fname = ("full_results_calibrated.npz" if args.source == "calibrated"
             else "full_results.npz")
    data_path = EXP7D_DIR / fname
    if not data_path.exists():
        sys.exit(f"ERROR: {data_path} not found.")

    npz        = np.load(data_path, allow_pickle=True)
    meta       = npz["meta"]        # (N, 6 or 7)
    dir_labels = npz["dir_labels"]  # (N,) str

    COL = {"trial_idx": 0, "l_inj": 1, "delta_M": 4}

    # Focus on L27
    l27_sel = meta[:, COL["l_inj"]] == 27
    m27     = meta[l27_sel]
    dl27    = dir_labels[l27_sel]

    # Rebuild test-set episode map (same seed/split as command_full)
    rows      = load_trials()
    _, te_m   = episode_split(rows)
    test_rows = [r for r, m in zip(rows, te_m) if m]
    trial_ep  = {i: r["episode_id"] for i, r in enumerate(test_rows)}

    trial_ids = np.unique(m27[:, COL["trial_idx"]].astype(int))
    n_tr      = len(trial_ids)

    # Per-trial mean |ΔM| for each direction category ─────────────────────────
    def pt(select):
        """select: str label or callable(label)->bool."""
        match = (dl27 == select) if isinstance(select, str) \
                else np.array([select(d) for d in dl27])
        out = np.full(n_tr, np.nan)
        for j, ti in enumerate(trial_ids):
            m = match & (m27[:, COL["trial_idx"]].astype(int) == ti)
            if m.any():
                out[j] = float(np.mean(np.abs(m27[m, COL["delta_M"]])))
        return out

    das  = pt("das_causal")
    orth = pt("das_orth")
    prb  = pt("probe")
    rnd  = pt(lambda d: d.startswith("random"))

    ok   = np.isfinite(das) & np.isfinite(orth) & np.isfinite(prb) & np.isfinite(rnd)
    das, orth, prb, rnd = das[ok], orth[ok], prb[ok], rnd[ok]
    tids_ok = trial_ids[ok]
    n_ok    = int(ok.sum())

    ep_arr  = np.array([trial_ep[ti] for ti in tids_ok])
    eps_u   = np.unique(ep_arr)
    n_ep    = len(eps_u)
    print(f"\nL27 — valid trials: {n_ok}  episodes: {n_ep}")

    def ep_mean(arr):
        return np.array([arr[ep_arr == ep].mean() for ep in eps_u])

    ed, eo, ep_, er = ep_mean(das), ep_mean(orth), ep_mean(prb), ep_mean(rnd)

    # Observed statistics ─────────────────────────────────────────────────────
    print(f"\n{'Direction':>12}  {'ep mean |ΔM|':>14}  {'se':>8}")
    print("─" * 40)
    for lbl, arr in [("das_causal", ed), ("das_orth", eo),
                     ("probe", ep_), ("random", er)]:
        print(f"  {lbl:>10}  {arr.mean():>14.5f}  {arr.std()/np.sqrt(n_ep):>8.5f}")

    obs_dr = ed.mean() / er.mean()
    obs_do = ed.mean() / eo.mean()
    obs_pr = ep_.mean() / er.mean()
    obs_dd = (ed - er).mean()
    obs_od = (ed - eo).mean()
    obs_pd = (ep_ - er).mean()

    # Bootstrap 95% CI (vectorised) ───────────────────────────────────────────
    rng_b  = np.random.default_rng(0)
    n_boot = args.n_boot
    bidx   = rng_b.integers(0, n_ep, (n_boot, n_ep))   # (n_boot, n_ep)

    # resample episode-level means
    bd = ed[bidx].mean(1); bo = eo[bidx].mean(1)
    bp = ep_[bidx].mean(1); br = er[bidx].mean(1)

    boot = {
        "das/rand ratio":   bd / br,
        "das/orth ratio":   bd / bo,
        "probe/rand ratio": bp / br,
        "das-rand diff":    bd - br,
        "das-orth diff":    bd - bo,
        "probe-rand diff":  bp - br,
    }
    obs_vals = {
        "das/rand ratio":   obs_dr,
        "das/orth ratio":   obs_do,
        "probe/rand ratio": obs_pr,
        "das-rand diff":    obs_dd,
        "das-orth diff":    obs_od,
        "probe-rand diff":  obs_pd,
    }

    print(f"\n── Bootstrap 95% CI  (n_boot={n_boot}, n_ep={n_ep}) "
          f"─────────────────────────────")
    print(f"{'Metric':>22}  {'Observed':>10}  {'95% CI lower':>14}  "
          f"{'95% CI upper':>14}  {'p(>0)':>8}")
    print("─" * 80)
    boot_lines = []
    for k in boot:
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        p_pos  = float((boot[k] > 0).mean())
        line   = (f"  {k:>20}  {obs_vals[k]:>10.4f}  "
                  f"{lo:>14.4f}  {hi:>14.4f}  {p_pos:>8.4f}")
        print(line); boot_lines.append(line)

    # Permutation test (sign-flip over episodes, vectorised) ──────────────────
    n_perm = n_boot
    signs  = rng_b.choice(np.array([-1., 1.]), size=(n_perm, n_ep))

    perm_pairs = {
        "das vs rand": (ed - er, obs_dd),
        "das vs orth": (ed - eo, obs_od),
        "probe vs rand": (ep_ - er, obs_pd),
    }
    print(f"\n── Permutation test  (sign-flip, n_perm={n_perm}) "
          f"────────────────────────────")
    perm_lines = []
    for name, (diffs, obs_d) in perm_pairs.items():
        perm_dist = (signs * diffs[None, :]).mean(axis=1)
        p_val     = float((perm_dist >= obs_d).mean())
        line      = f"  {name:>16}  obs={obs_d:.5f}  p={p_val:.5f}"
        if p_val == 0:
            line += f"  (< {1/n_perm:.0e})"
        print(line); perm_lines.append(line)

    # Paired Wilcoxon (trial-level) ───────────────────────────────────────────
    print(f"\n── Paired Wilcoxon signed-rank  (trial-level, n={n_ok}) "
          f"───────────────────")
    wilcox_lines = []
    try:
        from scipy.stats import wilcoxon
        for name, (a, b) in [("das > rand",   (das, rnd)),
                              ("das > orth",   (das, orth)),
                              ("probe > rand", (prb, rnd))]:
            stat, p = wilcoxon(a, b, alternative="greater")
            line = f"  {name:>14}  W={stat:.0f}  p={p:.2e}"
            print(line); wilcox_lines.append(line)
        scipy_used = True
    except ImportError:
        scipy_used = False
        for name, (a, b) in [("das > rand",   (das, rnd)),
                              ("das > orth",   (das, orth)),
                              ("probe > rand", (prb, rnd))]:
            n_pos = int((a > b).sum())
            # exact binomial p (one-sided sign test)
            log_p  = float(np.sum([np.log(k) - np.log(n_ok - k + 1)
                                   for k in range(n_pos + 1, n_ok + 1)]) -
                           n_ok * np.log(2))  # rough; use for large n
            line = (f"  {name:>14}  n_pos={n_pos}/{n_ok} "
                    f"({'scipy unavailable — sign test'})")
            print(line); wilcox_lines.append(line)

    # Save output ─────────────────────────────────────────────────────────────
    out_txt = EXP7D_DIR / "stats_output.txt"
    with open(out_txt, "w") as f:
        f.write(f"Exp 7D — Statistical Tests\n")
        f.write(f"source={args.source}  trials={n_ok}  episodes={n_ep}  "
                f"n_boot={n_boot}  scipy={'yes' if scipy_used else 'no'}\n\n")
        f.write("Bootstrap CIs:\n")
        for ln in boot_lines: f.write(ln.strip() + "\n")
        f.write("\nPermutation tests:\n")
        for ln in perm_lines: f.write(ln.strip() + "\n")
        f.write("\nWilcoxon tests:\n")
        for ln in wilcox_lines: f.write(ln.strip() + "\n")
    print(f"\nSaved {out_txt}")


# ── Directed intervention ─────────────────────────────────────────────────────

def _directed_summary(records: list[dict], eps_list: list[float]) -> None:
    from collections import defaultdict
    print(f"\n── Directed Intervention Summary ─────────────────────────────────────────")
    print(f"{'eps%':>5} {'direction':>18} {'sign':>5} "
          f"{'mean ΔM':>10} {'mean|ΔM|':>10} {'flip%':>7} {'n':>5}")
    print("─" * 65)
    by_key: dict = defaultdict(list)
    for r in records:
        by_key[(r["eps_frac"], r["direction"], r["sign"])].append(r)
    for eps in sorted(eps_list):
        for dname in ["probe", "das_proj", "das_orth_probe"]:
            for sign in [+1, -1]:
                recs = by_key.get((eps, dname, sign), [])
                if not recs:
                    continue
                dMs   = np.array([r["delta_M"] for r in recs])
                flips = np.array([r["flip"]    for r in recs])
                slabel = ("+" if sign > 0 else "-") + dname
                print(f"{eps*100:>4.0f}%  {slabel:>18}  {sign:>+5d}  "
                      f"{dMs.mean():>10.4f}  {np.abs(dMs).mean():>10.4f}  "
                      f"{100*flips.mean():>7.1f}%  {len(recs):>5d}")
        print()


def command_directed(args: argparse.Namespace) -> None:
    """Bidirectional directed perturbation at L27 with eps sweep.

    Three deterministic unit directions:
      probe          — OLS probe at L27 (unit vector)
      das_proj       — P_DAS @ w_probe / ||P_DAS @ w_probe||
      das_orth_probe — (w_probe - P_DAS @ w_probe) / ||...||

    Each applied ±eps for eps in eps_list (as fractions of ||h_L27||).
    Records delta_M (actual model output) and flip indicator (calibrated margin sign change).
    """
    EXP7D_DIR.mkdir(parents=True, exist_ok=True)

    # Load DAS subspace
    das_path = EXP7H_DIR / f"R_choice_tendency_L{DAS_LAYER}_k{DAS_K}.npy"
    if not das_path.exists():
        sys.exit(f"DAS matrix not found: {das_path}")
    R_das = np.load(das_path).astype(np.float64)   # (1024, k)
    P_das = R_das @ R_das.T                         # (1024, 1024) projection

    # Load probe weights
    probe_path = EXP7D_DIR / "probe_weights.npz"
    if not probe_path.exists():
        sys.exit("Run fit_probes first.")
    w = np.load(probe_path)["weights"][DAS_LAYER].astype(np.float64)  # unit vector

    # Build the three directions
    w_proj   = P_das @ w
    w_orth   = w - w_proj
    proj_n   = float(np.linalg.norm(w_proj))
    orth_n   = float(np.linalg.norm(w_orth))
    rand_exp = float(np.sqrt(DAS_K / HIDDEN_SIZE))

    d_probe      = w.astype(np.float32)
    d_das_proj   = (w_proj / (proj_n + EPS)).astype(np.float32)
    d_orth_probe = (w_orth / (orth_n + EPS)).astype(np.float32)

    print(f"Direction diagnostics (L27):")
    print(f"  ||P_DAS w_probe||            = {proj_n:.4f}  "
          f"(expected random: {rand_exp:.4f}  ratio: {proj_n/rand_exp:.2f}x)")
    print(f"  ||(I-P_DAS) w_probe||        = {orth_n:.4f}")
    print(f"  dot(d_das_proj, d_probe)      = "
          f"{float(np.dot(d_das_proj.astype(np.float64), w)):.4f}")
    print(f"  dot(d_orth_probe, d_probe)   = "
          f"{float(np.dot(d_orth_probe.astype(np.float64), w)):.4f}")

    # Load model
    import torch as torch_lib
    load_model, model_device, apply_chat_template, resolve_transformer_blocks = _import_model()
    fvd_choice_raw = _import_fvd()
    torch_mod, model, tokenizer = load_model(args.model, args.device, args.dtype)
    dev    = model_device(model)
    blocks = resolve_transformer_blocks(model)
    A_id   = tokenizer.encode("A", add_special_tokens=False)[-1]
    B_id   = tokenizer.encode("B", add_special_tokens=False)[-1]

    rows = load_trials()
    _, te_mask = episode_split(rows)
    test_rows  = [r for r, m in zip(rows, te_mask) if m]
    n_max      = min(args.n_trials, len(test_rows))
    n_fwd      = n_max * (1 + 2 * 3 * len(args.eps_list))
    print(f"\nTrials: {n_max}  Forward passes: {n_fwd}  "
          f"(1 baseline + {2*3*len(args.eps_list)} perturbed per trial)")

    named_dirs = [("probe",         d_probe),
                  ("das_proj",      d_das_proj),
                  ("das_orth_probe", d_orth_probe)]

    records: list[dict] = []
    t0 = time.time()

    for ti, row in enumerate(test_rows[:n_max]):
        prompt = build_prompt(row, fvd_choice_raw)
        enc    = _tokenize(prompt, tokenizer, dev, apply_chat_template)

        # Baseline forward pass
        with torch_lib.inference_mode():
            out_b = model(**enc, output_hidden_states=True, return_dict=True)
        h_l27  = out_b.hidden_states[DAS_LAYER][0, -1].detach().float().cpu().numpy()
        h_norm = float(np.linalg.norm(h_l27))
        lg_b   = out_b.logits[0, -1].detach().float().cpu().numpy()
        M_base = float(lg_b[A_id] - lg_b[B_id])
        M_cal  = float(_parse_field(row["choice_logit_diff"]))
        b_col  = bool(row["chosen_arm"] == "B" and
                      float(_parse_field(row["q_stated_A"])) >
                      float(_parse_field(row["q_stated_B"])))

        for eps_frac in args.eps_list:
            eps = eps_frac * h_norm
            for sign in [+1, -1]:
                for dname, d_unit in named_dirs:
                    delta     = (sign * eps * d_unit).astype(np.float32)
                    _, lgt_p  = run_perturbed(enc, model, torch_lib, blocks,
                                              DAS_LAYER, delta, torch_lib)
                    M_pert    = float(lgt_p[0, -1, A_id].cpu()
                                      - lgt_p[0, -1, B_id].cpu())
                    dM        = M_pert - M_base
                    flip      = bool(M_cal * (M_cal + dM) < 0)
                    records.append({
                        "ti": ti, "eps_frac": eps_frac, "sign": sign,
                        "direction": dname, "h_norm": h_norm,
                        "M_base": M_base, "M_cal": M_cal,
                        "delta_M": dM, "flip": flip, "b_collapse": b_col,
                    })

        if (ti + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {ti+1}/{n_max}  {el:.0f}s  "
                  f"ETA {(n_max - ti - 1) / (ti + 1) * el:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {len(records)} records  {elapsed/60:.1f} min")

    # Save compact npz
    out_npz = EXP7D_DIR / "directed_results.npz"
    np.savez_compressed(
        out_npz,
        meta=np.array(
            [[r["ti"], r["eps_frac"], r["sign"], r["h_norm"],
              r["M_base"], r["M_cal"], r["delta_M"],
              float(r["flip"]), float(r["b_collapse"])]
             for r in records], dtype=np.float32),
        directions=np.array([r["direction"] for r in records]),
        meta_cols=np.array(["ti","eps_frac","sign","h_norm",
                             "M_base","M_cal","delta_M","flip","b_collapse"]),
        proj_norm=np.float32(proj_n),
        orth_norm=np.float32(orth_n),
    )
    print(f"Saved {out_npz}  ({out_npz.stat().st_size // 1024} KB)")

    _directed_summary(records, args.eps_list)


# ── Random subspace control ───────────────────────────────────────────────────

def command_rand_subspace(args: argparse.Namespace) -> None:
    """Random subspace control for directed intervention (Q3).

    Generates K random 16-dim orthonormal subspaces (QR of Gaussian matrices).
    For each subspace R_rand: d_rand_proj = R_rand @ R_rand.T @ w_probe / ||...||
    Applies +eps × d_rand_proj at L27 and records flip% for comparison to das_proj.

    Reuses per-trial M_cal / h_norm / M_base from directed_results.npz to avoid
    redundant baseline forward passes.
    """
    EXP7D_DIR.mkdir(parents=True, exist_ok=True)

    # Load directed baselines (M_cal, h_norm, M_base) per trial index
    dir_path = EXP7D_DIR / "directed_results.npz"
    if not dir_path.exists():
        sys.exit("Run 'directed' first — need directed_results.npz for per-trial baselines.")
    dnpz   = np.load(dir_path, allow_pickle=True)
    meta_d = dnpz["meta"]       # (N, 9): ti, eps_frac, sign, h_norm, M_base, M_cal, ...
    dirs_d = dnpz["directions"]

    # Build per-trial lookup from any record (M_cal/h_norm don't vary by direction)
    trial_info: dict[int, dict] = {}
    for i in range(len(meta_d)):
        ti = int(meta_d[i, 0])
        if ti not in trial_info:
            trial_info[ti] = {
                "h_norm": float(meta_d[i, 3]),
                "M_base": float(meta_d[i, 4]),
                "M_cal":  float(meta_d[i, 5]),
            }

    # Benchmark: das_proj flip% at each eps (sign=+1)
    eps_list = args.eps_list
    das_flip_by_eps: dict[float, float] = {}
    for eps_frac in eps_list:
        mask = (
            (meta_d[:, 1] == eps_frac) &
            (meta_d[:, 2] == 1.0) &
            (np.array(dirs_d) == "das_proj")
        )
        das_flip_by_eps[eps_frac] = float(meta_d[mask, 7].mean()) if mask.any() else float("nan")

    # Load DAS subspace and probe weights
    das_path = EXP7H_DIR / f"R_choice_tendency_L{DAS_LAYER}_k{DAS_K}.npy"
    if not das_path.exists():
        sys.exit(f"DAS matrix not found: {das_path}")
    R_das = np.load(das_path).astype(np.float64)
    P_das = R_das @ R_das.T
    probe_path = EXP7D_DIR / "probe_weights.npz"
    if not probe_path.exists():
        sys.exit("Run fit_probes first.")
    w = np.load(probe_path)["weights"][DAS_LAYER].astype(np.float64)

    # DAS projection for reference diagnostics
    w_das = P_das @ w
    das_proj_n  = float(np.linalg.norm(w_das))
    d_das_proj  = (w_das / (das_proj_n + EPS)).astype(np.float32)
    rand_exp    = float(np.sqrt(DAS_K / HIDDEN_SIZE))

    # Generate K random orthonormal 16-dim subspaces
    K = args.k_subspaces
    rng = np.random.default_rng(args.seed)
    rand_proj_dirs: list[tuple[int, np.ndarray, float]] = []
    print(f"Random subspace diagnostics (K={K}, k={DAS_K}, expected proj norm ≈ {rand_exp:.4f}):")
    for k in range(K):
        G    = rng.standard_normal((HIDDEN_SIZE, DAS_K))
        Q, _ = np.linalg.qr(G)
        Q    = Q[:, :DAS_K]          # (1024, 16) orthonormal
        P_rk = Q @ Q.T
        w_rk = P_rk @ w
        n_rk = float(np.linalg.norm(w_rk))
        d_rk = (w_rk / (n_rk + EPS)).astype(np.float32)
        rand_proj_dirs.append((k, d_rk, n_rk))
        dot_das = float(np.dot(d_rk.astype(np.float64), d_das_proj.astype(np.float64)))
        print(f"  rand[{k}]  proj_norm={n_rk:.4f}  dot(d_rand, d_das)={dot_das:+.4f}")
    print(f"  das_proj  proj_norm={das_proj_n:.4f}  (ratio vs expected: {das_proj_n/rand_exp:.2f}x)")

    # Load model
    import torch as torch_lib
    load_model, model_device, apply_chat_template, resolve_transformer_blocks = _import_model()
    fvd_choice_raw = _import_fvd()
    torch_mod, model, tokenizer = load_model(args.model, args.device, args.dtype)
    dev    = model_device(model)
    blocks = resolve_transformer_blocks(model)
    A_id   = tokenizer.encode("A", add_special_tokens=False)[-1]
    B_id   = tokenizer.encode("B", add_special_tokens=False)[-1]

    rows = load_trials()
    _, te_mask = episode_split(rows)
    test_rows = [r for r, m in zip(rows, te_mask) if m]
    n_max = min(args.n_trials, len(test_rows))
    n_fwd = n_max * len(eps_list) * K
    print(f"\nTrials: {n_max}  eps: {eps_list}  K: {K}  "
          f"Forward passes: {n_fwd}  (baselines reused from directed_results.npz)")

    records: list[dict] = []
    t0 = time.time()

    for ti, row in enumerate(test_rows[:n_max]):
        # Reuse stored baselines; fall back to live baseline only if missing
        if ti in trial_info:
            h_norm = trial_info[ti]["h_norm"]
            M_base = trial_info[ti]["M_base"]
            M_cal  = trial_info[ti]["M_cal"]
        else:
            prompt = build_prompt(row, fvd_choice_raw)
            enc_b  = _tokenize(prompt, tokenizer, dev, apply_chat_template)
            with torch_lib.inference_mode():
                out_b = model(**enc_b, output_hidden_states=True, return_dict=True)
            h_l27  = out_b.hidden_states[DAS_LAYER][0, -1].detach().float().cpu().numpy()
            h_norm = float(np.linalg.norm(h_l27))
            lg_b   = out_b.logits[0, -1].detach().float().cpu().numpy()
            M_base = float(lg_b[A_id] - lg_b[B_id])
            M_cal  = float(_parse_field(row["choice_logit_diff"]))

        prompt = build_prompt(row, fvd_choice_raw)
        enc    = _tokenize(prompt, tokenizer, dev, apply_chat_template)

        for eps_frac in eps_list:
            eps = eps_frac * h_norm
            for k, d_rk, n_rk in rand_proj_dirs:
                delta    = (eps * d_rk).astype(np.float32)
                _, lgt_p = run_perturbed(enc, model, torch_lib, blocks,
                                         DAS_LAYER, delta, torch_lib)
                M_pert   = float(lgt_p[0, -1, A_id].cpu() - lgt_p[0, -1, B_id].cpu())
                dM       = M_pert - M_base
                flip     = bool(M_cal * (M_cal + dM) < 0)
                records.append({
                    "ti": ti, "eps_frac": eps_frac, "subspace_idx": k,
                    "h_norm": h_norm, "M_base": M_base, "M_cal": M_cal,
                    "delta_M": dM, "flip": flip,
                })

        if (ti + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {ti+1}/{n_max}  {el:.0f}s  "
                  f"ETA {(n_max - ti - 1) / (ti + 1) * el:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {len(records)} records  {elapsed/60:.1f} min")

    # Save
    out_npz = EXP7D_DIR / "rand_subspace_results.npz"
    np.savez_compressed(
        out_npz,
        meta=np.array(
            [[r["ti"], r["eps_frac"], r["subspace_idx"], r["h_norm"],
              r["M_base"], r["M_cal"], r["delta_M"], float(r["flip"])]
             for r in records], dtype=np.float32),
        meta_cols=np.array(["ti", "eps_frac", "subspace_idx", "h_norm",
                             "M_base", "M_cal", "delta_M", "flip"]),
        k_subspaces=np.int32(K),
        rand_proj_norms=np.array([n_rk for _, _, n_rk in rand_proj_dirs]),
        das_proj_norm=np.float32(das_proj_n),
        rand_exp=np.float32(rand_exp),
    )
    print(f"Saved {out_npz}  ({out_npz.stat().st_size // 1024} KB)")

    # Summary table
    print(f"\n── Random Subspace Control (Q3) ─────────────────────────────────────────")
    print(f"{'eps%':>5}  {'subspace':>14}  {'flip%':>7}  {'mean ΔM':>10}")
    print("─" * 48)
    for eps_frac in sorted(eps_list):
        das_flip_pct = das_flip_by_eps.get(eps_frac, float("nan")) * 100
        print(f"{eps_frac*100:>4.0f}%  {'das_proj':>14}  {das_flip_pct:>7.1f}%  (benchmark from directed)")
        rand_flip_pcts: list[float] = []
        for k, d_rk, n_rk in rand_proj_dirs:
            recs_k  = [r for r in records if r["eps_frac"] == eps_frac and r["subspace_idx"] == k]
            if recs_k:
                dMs_k   = np.array([r["delta_M"] for r in recs_k])
                flips_k = np.array([r["flip"]    for r in recs_k])
                fp_k    = 100 * flips_k.mean()
                rand_flip_pcts.append(fp_k)
                print(f"{eps_frac*100:>4.0f}%  {'rand['+str(k)+']':>14}  "
                      f"{fp_k:>7.1f}%  {dMs_k.mean():>10.4f}")
        if rand_flip_pcts:
            mu_r    = float(np.mean(rand_flip_pcts))
            sd_r    = float(np.std(rand_flip_pcts))
            ratio   = das_flip_pct / (mu_r + 1e-8)
            print(f"\n  rand mean ± std = {mu_r:.1f}% ± {sd_r:.1f}%  |  "
                  f"das_proj / rand_mean = {ratio:.2f}×\n")


# ── Recalibrate existing results ──────────────────────────────────────────────

def command_recalibrate(args: argparse.Namespace) -> None:
    """Post-process full_results.npz — add calibrated M_base_cal column (col 6).

    Reads choice_logit_diff from trials.jsonl (already baseline-subtracted) and
    appends it as column 6 of the meta array.  No model needed.
    Output: results/exp7d/full_results_calibrated.npz
    """
    src = EXP7D_DIR / "full_results.npz"
    if not src.exists():
        sys.exit(f"ERROR: {src} not found. Run 'full' first.")

    rows = load_trials()
    _, te_mask = episode_split(rows)
    test_rows  = [r for r, m in zip(rows, te_mask) if m]

    npz  = np.load(src, allow_pickle=True)
    meta = npz["meta"].copy()   # (N, 6) float32

    if meta.shape[1] >= 7:
        print("Meta already has 7+ columns (M_base_cal present). Nothing to do.")
        return

    # Build calibrated margin per test-set trial index
    trial_cal = np.array([float(_parse_field(r["choice_logit_diff"]))
                          for r in test_rows], dtype=np.float32)

    trial_idx_col  = meta[:, 0].astype(int)
    m_base_cal_col = trial_cal[trial_idx_col]
    new_meta = np.concatenate([meta, m_base_cal_col[:, None]], axis=1)  # (N, 7)

    dst = EXP7D_DIR / "full_results_calibrated.npz"
    np.savez_compressed(dst,
                        R_h=npz["R_h"], q_align=npz["q_align"], cos_dir=npz["cos_dir"],
                        meta=new_meta, dir_labels=npz["dir_labels"])
    print(f"Saved {dst}  ({dst.stat().st_size // 1024} KB)")

    # Sanity summary
    raw_vals = new_meta[:, 3]
    cal_vals = new_meta[:, 6]
    bc_mask  = new_meta[:, 5] > 0.5
    print(f"\nRaw M_base:       mean={raw_vals.mean():.3f}  (biased ~+4 by letter prior)")
    print(f"Calibrated M_base: mean={cal_vals.mean():.3f}  std={cal_vals.std():.3f}")
    print(f"  B-collapse  cal mean: {cal_vals[bc_mask].mean():.3f}  (expect <0)")
    print(f"  Normal      cal mean: {cal_vals[~bc_mask].mean():.3f}  (expect >0)")
    n_bc_neg = (cal_vals[bc_mask] < 0).mean() * 100
    print(f"  B-collapse with cal<0: {n_bc_neg:.1f}%  (should be ~99.7%)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Exp 7D — Decision Formation Dynamics")
    p.add_argument("--model",  default=DEFAULT_MODEL)
    p.add_argument("--device", default="mps")
    p.add_argument("--dtype",  default="bfloat16")
    p.add_argument("--seed",   type=int, default=42)
    sub = p.add_subparsers(dest="cmd")

    # fit_probes
    sub.add_parser("fit_probes", help="Fit layer-specific OLS probes (numpy only)")

    # pilot
    pp = sub.add_parser("pilot", help="50-trial pilot run")
    pp.add_argument("--n_pilot",       type=int,   default=50)
    pp.add_argument("--n_random",      type=int,   default=N_RANDOM_PILOT)
    pp.add_argument("--eps_scale",     type=float, default=0.01,
                    help="Perturbation size as fraction of ‖h_l‖ (default 1%%)")
    pp.add_argument("--inject_layers", type=int,   nargs="+",
                    default=INJECT_LAYERS_PILOT)

    # full
    fp = sub.add_parser("full", help="Full experiment on all test-set trials")
    fp.add_argument("--n_random",      type=int,   default=N_RANDOM_FULL)
    fp.add_argument("--eps_scale",     type=float, default=0.01)
    fp.add_argument("--inject_layers", type=int,   nargs="+",
                    default=INJECT_LAYERS_FULL)

    # analyze
    ap = sub.add_parser("analyze", help="Generate plots from saved results")
    ap.add_argument("--source", choices=["pilot", "full", "calibrated"], default="pilot")

    # recalibrate
    sub.add_parser("recalibrate",
                   help="Add calibrated M_base_cal to full_results.npz (no model needed)")

    # stats — no model required
    sp = sub.add_parser("stats",
                        help="Bootstrap CI + significance tests on saved .npz (no model)")
    sp.add_argument("--source",  choices=["full", "calibrated"], default="calibrated")
    sp.add_argument("--n_boot",  type=int, default=10000)

    # directed — requires model
    dp = sub.add_parser("directed",
                        help="±probe / ±DAS-proj-probe / ±DAS-orth-probe at L27, eps sweep")
    dp.add_argument("--n_trials",  type=int,   default=450,
                    help="Number of test trials to run (default: all 450)")
    dp.add_argument("--eps_list",  type=float, nargs="+",
                    default=[0.01, 0.02, 0.05],
                    help="Perturbation magnitudes as fractions of ||h_L27||")

    # rand_subspace — requires model; Q3 control
    rp = sub.add_parser("rand_subspace",
                        help="Random 16-dim subspace control vs DAS projection at L27 (Q3)")
    rp.add_argument("--n_trials",    type=int,   default=450)
    rp.add_argument("--k_subspaces", type=int,   default=5,
                    help="Number of random orthonormal subspaces to test (default: 5)")
    rp.add_argument("--eps_list",    type=float, nargs="+", default=[0.02, 0.05],
                    help="Perturbation magnitudes as fractions of ||h_L27||")

    args = p.parse_args()
    if args.cmd is None:
        p.print_help(); return

    dispatch = {
        "fit_probes":    command_fit_probes,
        "pilot":         command_pilot,
        "full":          command_full,
        "analyze":       command_analyze,
        "recalibrate":   command_recalibrate,
        "stats":         command_stats,
        "directed":      command_directed,
        "rand_subspace": command_rand_subspace,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
