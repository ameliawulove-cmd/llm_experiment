#!/usr/bin/env python3
"""Exp 7H — Distributed Alignment Search (DAS).

Finds, for each (variable, layer, k), the k-dimensional subspace R ∈ R^{1024×k}
along which an *interchange intervention* produces the behavioural change
predicted by that variable.

Interchange
-----------
Given a *base* trial b and a *source* trial s with v_s > v_b:
    h_intervened = h_b + R R^⊤ (h_s − h_b)
Predicted behaviour: s(h_intervened) > s(h_b), where s(h) = w · (h − μ_tr)
is the same linear decision probe used in exp7g.

Loss
----
We use the sigmoid of the signed shift and maximise correctness:
    L = − log σ( sign(v_s − v_b) · (s(h_intervened) − s(h_b)) )
R is parametrised as the Q from QR(A) on a learned A ∈ R^{1024×k}, so columns
are orthonormal by construction.

CLI
---
    python exp7h_das.py train     # fit R for each (var, layer, k)
    python exp7h_das.py evaluate  # recompute IIA on held-out pairs
"""
# ============================================================
# 实验7H — 分布式对齐搜索（DAS, Distributed Alignment Search）
# ============================================================
# 研究目的：
#   在 Qwen3-0.6B 的 L27 层（hidden_states[27]）学习一个
#   16维正交子空间 R ∈ R^{1024×16}，使得"交换干预"
#   （interchange intervention）能预测性地移动模型的选择输出。
#
# 核心公式：
#   h_int = h_b + R · R^T · (h_s - h_b)
#   将 source 的子空间分量替换到 base 的激活中
#
# 训练数据：
#   前1200行（episodes seq_0000 到 seq_0039）→ 训练集
#   后300行（episodes seq_0040 到 seq_0049）→ 验证集
#   注意：这是行索引分割，不是随机 episode 分割
#
# 主要输出：
#   results/exp7h/R_choice_tendency_L27_k16.npy  冻结的基矩阵 (1024, 16)
#   results/exp7h/das_results.json               训练历史
#
# 命令：
#   python exp7h_das.py train    训练R（会覆盖已有结果，注意！）
#   python exp7h_das.py evaluate 在新的配对上重新评估
# ============================================================
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent.parent  # experiment_7 repo root
EXP6_FULL = _THIS_DIR / "data/exp6_full"
EXP6_ACTS = _THIS_DIR / "data/exp6_activations"
OUT_BASE  = _THIS_DIR / "results"
OUT_DIR   = OUT_BASE / "exp7h"

HIDDEN_SIZE = 1024

VARIABLES = ["reward_history", "choice_tendency", "delta_q", "rw_delta_q"]
LAYERS    = [5, 14, 22, 27]
K_VALUES  = [8, 16, 32]
POSITION  = "choice"


# ---------------------------------------------------------------------------
# Data loading (replicates exp7_neuro.py)
# ---------------------------------------------------------------------------

def load_trials() -> list[dict]:
    path = EXP6_FULL / "trials.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing exp6 trials: {path}")
    return [json.loads(l) for l in path.read_text().splitlines() if l]


def build_targets(rows: list[dict]) -> dict[str, np.ndarray]:
    n = len(rows)
    reward_history = np.zeros(n)
    ep_cumsum: dict[str, float] = {}
    ep_count:  dict[str, int]   = {}
    for i, r in enumerate(rows):
        eid = r["episode_id"]
        if eid not in ep_cumsum:
            ep_cumsum[eid] = 0.0
            ep_count[eid]  = 0
        cnt = ep_count[eid]
        reward_history[i] = ep_cumsum[eid] / cnt if cnt > 0 else 0.5
        ep_cumsum[eid] += r["reward"]
        ep_count[eid]  += 1
    delta_q = np.array([r["q_stated_A"] - r["q_stated_B"] for r in rows])
    rpe_proxy = np.array([
        r["reward"] - (r["q_stated_A"] if r["chosen_arm"] == "A" else r["q_stated_B"])
        for r in rows
    ])
    choice_tendency = np.array([r["choice_logit_diff"] for r in rows])
    recent_reward = np.zeros(n)
    ep_rewards: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        eid = r["episode_id"]
        if eid not in ep_rewards:
            ep_rewards[eid] = []
        hist = ep_rewards[eid]
        recent_reward[i] = np.mean(hist[-3:]) if hist else 0.5
        ep_rewards[eid].append(r["reward"])
    post_reversal = np.array([float(r["is_post_reversal"]) for r in rows])
    choice_a = np.array([float(r["chosen_arm"] == "A") for r in rows])
    targets = {
        "reward_history":  reward_history,
        "recent_reward":   recent_reward,
        "delta_q":         delta_q,
        "rpe_proxy":       rpe_proxy,
        "choice_tendency": choice_tendency,
        "post_reversal":   post_reversal,
        "choice_a":        choice_a,
    }
    mf_path = OUT_BASE / "exp7a" / "model_fits.json"
    if mf_path.exists():
        try:
            mf = json.loads(mf_path.read_text())
            lookup = {(d["episode_id"], d["trial"]): d for d in mf.get("per_trial", [])}
            rw_delta_q  = np.full(n, np.nan)
            rw_rpe      = np.full(n, np.nan)
            rw_q_chosen = np.full(n, np.nan)
            for i, r in enumerate(rows):
                d = lookup.get((r["episode_id"], r["trial"]))
                if d is None:
                    continue
                qa = d.get("rw_Q_A", np.nan)
                qb = d.get("rw_Q_B", np.nan)
                rw_delta_q[i]  = qa - qb
                rw_rpe[i]      = d.get("rw_rpe", np.nan)
                rw_q_chosen[i] = qa if r["chosen_arm"] == "A" else qb
            if np.isfinite(rw_delta_q).any():
                targets["rw_delta_q"]  = np.nan_to_num(rw_delta_q,  nan=0.0)
                targets["rw_rpe"]      = np.nan_to_num(rw_rpe,      nan=0.0)
                targets["rw_q_chosen"] = np.nan_to_num(rw_q_chosen, nan=0.5)
        except Exception as e:
            print(f"NOTE: could not load model fits: {e}")
    return targets


def load_exp6_data() -> dict[str, Any]:
    rows = load_trials()
    choices  = np.array([1 if r["chosen_arm"] == "A" else 0 for r in rows], dtype=int)
    rewards  = np.array([int(r["reward"]) for r in rows], dtype=int)
    declared_a = np.array([r["q_stated_A"] for r in rows])
    declared_b = np.array([r["q_stated_B"] for r in rows])
    episode_ids = np.array([r["episode_id"] for r in rows])
    return {
        "rows": rows, "choices": choices, "rewards": rewards,
        "declared_a": declared_a, "declared_b": declared_b,
        "episode_ids": episode_ids, "targets": build_targets(rows),
    }


def load_activations(position: str) -> np.ndarray:
    path = EXP6_ACTS / f"acts_{position}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing activation file: {path}")
    return np.load(path, mmap_mode="r")


def episode_aware_split(episode_ids: np.ndarray,
                        n_train: int = 1200) -> tuple[np.ndarray, np.ndarray]:
    # Row-index split: first 1200 rows = training, last 300 rows = validation.
    # With the standard 50-episode × 30-trial layout this corresponds exactly to
    # episodes seq_0000–seq_0039 (train) and seq_0040–seq_0049 (val).
    # There is NO episode overlap between the two sets (verified by test suite).
    #
    # NOTE: this is NOT a random episode shuffle.  An earlier validation script
    # (results/validation/validate_02_random_iia.py) used seed=42 random shuffle
    # of episodes and happened to place 9/10 "validation" episodes inside the DAS
    # training rows — producing a contaminated full-forward IIA of 0.995.
    # That script's result should NOT be reported as independent evidence.
    # The clean held-out evidence is in results/exp7h/das_evaluation.json
    # (from command_evaluate, using val_mask) and VALIDATION_EXP7_V2 (new episodes).
    n = len(episode_ids)
    train = np.zeros(n, dtype=bool)
    train[:n_train] = True
    return train, ~train


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Decision probe (same as 7G)
# ---------------------------------------------------------------------------

def fit_ridge_probe(X_tr: np.ndarray, y_tr: np.ndarray,
                    alpha: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    mu = X_tr.mean(0)
    Xc = X_tr - mu
    yc = y_tr - y_tr.mean()
    d = Xc.shape[1]
    A = Xc.T @ Xc + alpha * np.eye(d)
    w = np.linalg.solve(A, Xc.T @ yc)
    return w.astype(np.float32), mu.astype(np.float32)


# ---------------------------------------------------------------------------
# Pair sampling
# ---------------------------------------------------------------------------

def sample_pairs(v: np.ndarray,
                 mask: np.ndarray,
                 n_pairs: int,
                 threshold_frac: float = 0.25,
                 rng: np.random.Generator | None = None,
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Sample (base, source) index pairs with |v_s − v_b| > threshold.

    Threshold = `threshold_frac` × std(v on mask).
    Returns int arrays `base_idx`, `src_idx`.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    pool = np.flatnonzero(mask)
    v_pool = v[pool]
    thresh = threshold_frac * (np.std(v_pool) + 1e-12)
    bases, srcs = [], []
    # Over-sample attempts and filter — simple rejection sampling.
    attempts = 0
    max_attempts = n_pairs * 50
    while len(bases) < n_pairs and attempts < max_attempts:
        b = rng.integers(0, len(pool))
        s = rng.integers(0, len(pool))
        attempts += 1
        if abs(v_pool[s] - v_pool[b]) < thresh:
            continue
        bases.append(pool[b])
        srcs.append(pool[s])
    if len(bases) < n_pairs:
        print(f"  WARNING: only sampled {len(bases)}/{n_pairs} pairs "
              f"(variable range too narrow? thresh={thresh:.3g})")
    return np.array(bases, dtype=int), np.array(srcs, dtype=int)


# ---------------------------------------------------------------------------
# Rotation parametrisation (R = QR(A).Q)
# ---------------------------------------------------------------------------

class RotationBasis(nn.Module):
    """可学习的正交归一基矩阵 R ∈ R^{d × k}，通过可微分 QR 分解实现。

    工作原理：
        存储可训练参数 self.A，shape (d, k)
        每次 forward 时对 self.A 做 QR 分解，取 Q 作为 R
        QR 分解保证 R 的列向量是正交归一的（orthonormal）
    参数：
        d: int，hidden size，Qwen3-0.6B 中为 1024
        k: int，子空间维度，实验中为 16
    输出（调用 self.R()）：
        Q: shape (1024, 16)，列正交归一的基矩阵
    注意：
        这是唯一被优化的参数。Qwen 的权重完全不更新。
    """

    def __init__(self, d: int, k: int):
        super().__init__()
        self.d = d
        self.k = k
        # Init: random Gaussian then orthonormalise once for a sane starting R
        A0 = torch.randn(d, k)
        Q0, _ = torch.linalg.qr(A0)
        self.A = nn.Parameter(Q0.clone())

    def R(self) -> torch.Tensor:
        Q, _ = torch.linalg.qr(self.A)
        return Q


# ---------------------------------------------------------------------------
# DAS training for one (var, layer, k)
# ---------------------------------------------------------------------------

def train_das(h_all: np.ndarray,
              v_all: np.ndarray,
              w: np.ndarray,
              mu: np.ndarray,
              train_mask: np.ndarray,
              val_mask: np.ndarray,
              k: int,
              n_train_pairs: int = 2000,
              n_val_pairs:   int = 500,
              steps: int = 500,
              lr: float = 1e-3,
              batch_size: int = 128,
              device: torch.device | None = None,
              verbose: bool = False,
              ) -> dict:
    """训练 DAS 子空间基矩阵 R，返回训练/验证 IIA 和最优 R。

    参数：
        h_all: shape (1500, 1024)，某一层所有 trial 的激活（float32）
        v_all: shape (1500,)，目标变量（如 choice_tendency）
        w: shape (1024,)，Ridge probe 权重向量
        mu: shape (1024,)，训练集激活的均值
        train_mask: shape (1500,) 布尔数组，前1200行为 True
        val_mask: shape (1500,) 布尔数组，后300行为 True
        k: int，子空间维度（如16）
    返回：
        dict，包含 'R' (1024, k)、'iia_val'、'history' 等
    核心操作：
        1. 采样训练/验证 (base, source) 配对
        2. 对每个配对：h_int = h_b + R R^T (h_s - h_b)
        3. 计算 probe 分数变化：shift = w·h_int - w·h_b
        4. 最大化方向性：sign(v_s - v_b) · shift > 0
        5. 保存验证集上最佳的 R
    """
    if device is None:
        device = pick_device()

    rng = np.random.default_rng(42)
    b_tr, s_tr = sample_pairs(v_all, train_mask, n_train_pairs, rng=rng)
    b_va, s_va = sample_pairs(v_all, val_mask,   n_val_pairs,   rng=rng)

    # Prepare tensors
    H  = torch.from_numpy(h_all.astype(np.float32)).to(device)
    v  = torch.from_numpy(v_all.astype(np.float32)).to(device)
    w_t = torch.from_numpy(w.astype(np.float32)).to(device)
    mu_t = torch.from_numpy(mu.astype(np.float32)).to(device)

    b_tr_t = torch.from_numpy(b_tr).long().to(device)
    s_tr_t = torch.from_numpy(s_tr).long().to(device)
    b_va_t = torch.from_numpy(b_va).long().to(device)
    s_va_t = torch.from_numpy(s_va).long().to(device)

    basis = RotationBasis(HIDDEN_SIZE, k).to(device)
    opt = torch.optim.Adam(basis.parameters(), lr=lr)

    def score_pair(idx_base: torch.Tensor, idx_src: torch.Tensor,
                   R: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_b = H[idx_base]
        h_s = H[idx_src]
        v_b = v[idx_base]
        v_s = v[idx_src]
        delta = h_s - h_b                       # (B, d)
        proj  = (delta @ R) @ R.T               # (B, d)
        h_int = h_b + proj
        s_b   = (h_b   - mu_t) @ w_t
        s_int = (h_int - mu_t) @ w_t
        shift = s_int - s_b
        signed = torch.sign(v_s - v_b) * shift
        return signed, shift

    pbar = tqdm(range(steps), disable=not verbose, desc=f"DAS k={k}")
    history: list[dict] = []
    # "IIA" here means directional agreement (DA): fraction of held-out pairs where
    # sign(v_s − v_b) × (probe_score(h_int) − probe_score(h_b)) > 0.
    # This is NOT standard interchange intervention accuracy (IIA), which requires
    # specifying a high-level causal model and comparing against counterfactual outputs.
    best_val_iia = -1.0
    best_R: np.ndarray | None = None
    for step in pbar:
        basis.train()
        idx = torch.randperm(len(b_tr_t), device=device)[:batch_size]
        R = basis.R()
        signed, _ = score_pair(b_tr_t[idx], s_tr_t[idx], R)
        loss = F.binary_cross_entropy_with_logits(
            signed, torch.ones_like(signed)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % 25 == 0 or step == steps - 1:
            basis.eval()
            with torch.no_grad():
                R = basis.R()
                signed_tr, _ = score_pair(b_tr_t, s_tr_t, R)
                signed_va, _ = score_pair(b_va_t, s_va_t, R)
                iia_tr = float((signed_tr > 0).float().mean().item())
                iia_va = float((signed_va > 0).float().mean().item())
            history.append({
                "step": step, "loss": float(loss.item()),
                "iia_train": iia_tr, "iia_val": iia_va,
            })
            if iia_va > best_val_iia:
                best_val_iia = iia_va
                best_R = R.detach().cpu().numpy()
            pbar.set_postfix(iia_tr=f"{iia_tr:.3f}", iia_va=f"{iia_va:.3f}",
                             best=f"{best_val_iia:.3f}")

    return {
        "k": k,
        "iia_val":  best_val_iia,
        "iia_train": history[-1]["iia_train"] if history else float("nan"),
        "iia_val_final": history[-1]["iia_val"] if history else float("nan"),
        "history":  history,
        "R":        best_R,
    }


# ---------------------------------------------------------------------------
# Command: train
# ---------------------------------------------------------------------------

def command_train(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_exp6_data()
    tgts = data["targets"]
    episode_ids = data["episode_ids"]
    train_mask, val_mask = episode_aware_split(episode_ids, n_train=1200)
    device = pick_device()
    print(f"Using device: {device}")

    acts = load_activations(POSITION)

    # Fit decision probe once, on L27
    h27 = np.asarray(acts[:, 27, :], dtype=np.float32)
    y = (data["choices"] == 0).astype(np.float32)   # 1 if B
    w_np, mu_np = fit_ridge_probe(h27[train_mask], y[train_mask], alpha=0.01)
    print(f"Decision probe |w|={np.linalg.norm(w_np):.3f}")

    results: dict[str, Any] = {
        "probe": {"w_norm": float(np.linalg.norm(w_np)),
                   "position": POSITION, "layer": 27,
                   "n_train": int(train_mask.sum()),
                   "n_val":   int(val_mask.sum())},
        "runs":  [],
    }

    only_vars   = set(args.only_vars)   if args.only_vars   else None
    only_layers = set(args.only_layers) if args.only_layers else None
    only_k      = set(args.only_k)      if args.only_k      else None

    for var in VARIABLES:
        if only_vars and var not in only_vars:
            print(f"skip variable {var}: not in --only_vars")
            continue
        if var not in tgts:
            print(f"skip variable {var}: not in targets")
            continue
        v_arr = tgts[var]
        for layer in LAYERS:
            if only_layers and layer not in only_layers:
                print(f"skip layer {layer}: not in --only_layers")
                continue
            h = np.asarray(acts[:, layer, :], dtype=np.float32)
            for k in K_VALUES:
                if only_k and k not in only_k:
                    print(f"skip k={k}: not in --only_k")
                    continue
                tag = f"{var}_L{layer}_k{k}"
                print(f"\n=== DAS {tag} ===")
                run = train_das(
                    h, v_arr, w_np, mu_np, train_mask, val_mask, k,
                    n_train_pairs=args.n_train_pairs,
                    n_val_pairs=args.n_val_pairs,
                    steps=args.steps, lr=args.lr,
                    batch_size=args.batch_size, device=device,
                    verbose=True,
                )
                # Save R
                R_path = OUT_DIR / f"R_{tag}.npy"
                if run["R"] is not None:
                    np.save(R_path, run["R"])
                results["runs"].append({
                    "variable": var,
                    "layer":    layer,
                    "k":        k,
                    "iia_val":        run["iia_val"],
                    "iia_val_final":  run["iia_val_final"],
                    "iia_train":      run["iia_train"],
                    "R_path":         str(R_path),
                    "history_last":   run["history"][-1] if run["history"] else None,
                })
                print(f"  {tag}: best val IIA = {run['iia_val']:.3f}")
                # Save incrementally so partial runs are usable
                (OUT_DIR / "das_results.json").write_text(json.dumps(results, indent=2))

    print(f"\nSaved {OUT_DIR / 'das_results.json'}")


# ---------------------------------------------------------------------------
# Command: evaluate — recompute IIA on held-out pairs using stored R
# ---------------------------------------------------------------------------

def command_evaluate(args: argparse.Namespace) -> None:
    # IMPORTANT: This evaluation uses val_mask (episodes seq_0040–seq_0049), which are
    # the SAME episodes used for checkpoint selection during training (best R is selected
    # by maximising val directional agreement).  Therefore the iia_heldout values in
    # das_evaluation.json reflect episodes held out from training but NOT from model
    # selection.  The genuinely independent held-out evidence is in VALIDATION_EXP7_V2
    # (20 entirely new episodes, pre-registered protocol).
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res_path = OUT_DIR / "das_results.json"
    if not res_path.exists():
        raise FileNotFoundError(
            f"{res_path} not found. Run `python exp7h_das.py train` first."
        )
    results = json.loads(res_path.read_text())

    data = load_exp6_data()
    train_mask, val_mask = episode_aware_split(data["episode_ids"], n_train=1200)
    acts = load_activations(POSITION)

    h27 = np.asarray(acts[:, 27, :], dtype=np.float32)
    y = (data["choices"] == 0).astype(np.float32)
    w_np, mu_np = fit_ridge_probe(h27[train_mask], y[train_mask], alpha=0.01)

    rng = np.random.default_rng(123)   # fresh seed for held-out eval
    table = []
    for run in results["runs"]:
        var = run["variable"]
        layer = run["layer"]
        k = run["k"]
        R_path = Path(run["R_path"])
        if not R_path.exists():
            continue
        R = np.load(R_path)
        v_arr = data["targets"][var]
        h = np.asarray(acts[:, layer, :], dtype=np.float32)

        b_idx, s_idx = sample_pairs(v_arr, val_mask, n_pairs=1000,
                                    threshold_frac=0.5, rng=rng)
        if len(b_idx) == 0:
            continue
        h_b = h[b_idx]
        h_s = h[s_idx]
        delta = h_s - h_b
        proj = (delta @ R) @ R.T
        h_int = h_b + proj
        s_b = (h_b   - mu_np) @ w_np
        s_i = (h_int - mu_np) @ w_np
        shift = s_i - s_b
        signed = np.sign(v_arr[s_idx] - v_arr[b_idx]) * shift
        iia = float((signed > 0).mean())
        table.append({
            "variable": var, "layer": layer, "k": k,
            "iia_heldout": iia, "iia_val_train": run["iia_val"],
            "n_pairs": int(len(b_idx)),
        })
        print(f"  {var:>16s}  L{layer:2d}  k={k:2d}  "
              f"iia_heldout={iia:.3f}  (train-time val={run['iia_val']:.3f})")

    (OUT_DIR / "das_evaluation.json").write_text(json.dumps(table, indent=2))
    print(f"\nSaved {OUT_DIR / 'das_evaluation.json'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    p_tr = sub.add_parser("train", help="fit R for all (var, layer, k)")
    p_tr.add_argument("--n_train_pairs", type=int, default=2000)
    p_tr.add_argument("--n_val_pairs",   type=int, default=500)
    p_tr.add_argument("--steps",         type=int, default=500)
    p_tr.add_argument("--lr",            type=float, default=1e-3)
    p_tr.add_argument("--batch_size",    type=int, default=128)
    p_tr.add_argument("--only_vars",     nargs="+", default=None,
                      help="restrict to these variable names")
    p_tr.add_argument("--only_layers",   nargs="+", type=int, default=None,
                      help="restrict to these layer indices")
    p_tr.add_argument("--only_k",        nargs="+", type=int, default=None,
                      help="restrict to these k values")
    p_tr.set_defaults(func=command_train)

    p_ev = sub.add_parser("evaluate", help="re-score IIA on fresh pairs")
    p_ev.set_defaults(func=command_evaluate)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
