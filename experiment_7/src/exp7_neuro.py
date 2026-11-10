#!/usr/bin/env python3
"""Experiment 7 — Mapping the Neural Dynamics of Decision-Making in Language Models

From Reward History to Choice — A Causal Investigation of Internal Computation

Builds on Exp 6 (FVD-Bandit, 1500 trials, Qwen3-0.6B).
Requires existing data at:
  ~/Desktop/llmneuroscientist/data/exp6_full/        (trials.jsonl)
  ~/Desktop/llmneuroscientist/data/exp6_activations/ (acts_declare_a/b.npy, acts_choice.npy)

Sub-experiments and commands
──────────────────────────────
  collect_t2      7A  Extract residual-stream activations at the reward-feedback
                      position (T2) for all 1500 trials.
                      Output: results/exp7a/acts_feedback.npy  (1500, 29, 1024)

  collect_comps   7C  Extract sub-block activations x_mid (post-attention, pre-MLP)
                      using forward hooks on post_attention_layernorm.
                      Output: results/exp7c/acts_x_mid.npy    (1500, 28, 1024)

  probe_7b        7B  Layer-wise linear probing of four computational variables:
                        reward_history  (cumulative mean reward)
                        delta_q         (q_stated_A − q_stated_B)
                        rpe_proxy       (reward − q_stated_chosen)
                        choice_tendency (choice_logit_diff)
                      across four activation positions.
                      Output: results/exp7b/probe_results.json

  analyze_7c      7C  Attention vs MLP contribution analysis.
                      Probes both attn_contribution (x_mid − x_in) and
                      mlp_contribution (x_out − x_mid) for all 7B variables.
                      Output: results/exp7c/component_analysis.json

  intervene_7d    7D  Causal activation steering at the best probe layer.
                      Applies ±α * v_direction and measures ΔP(A).
                      Controls: random direction, orthogonal direction.
                      Output: results/exp7d/intervention_results.json

  discover_7e     7E  Data-driven discovery of choice-predictive low-dim states.
                      PCA on residual stream → logistic regression → future-choice
                      prediction with held-out episodes.
                      Output: results/exp7e/discovery_results.json

  report          All Generate all figures (PNG) and a summary JSON.
                      Output: figures/
"""

# ============================================================
# 实验7主脚本 — 神经动态与决策机制分析
# ============================================================
# 研究目的：
#   检测 Qwen3-0.6B 在二臂反转赌博机任务中，
#   不同 Transformer 层的 hidden states 是否包含
#   可线性解码的决策相关变量（尤其是 choice_tendency）。
#
# 本脚本涵盖：
#   7A — 计算模型拟合（Rescorla-Wagner, WSLS, Recency）
#   7B — 层级 Linear Probe 训练和评估
#   7C — 注意力/MLP 组件分析（需要先运行 collect_comps，当前输出缺失）
#   7E — 数据驱动发现（PCA + logistic regression）
#
# 主要输入：
#   data/exp6_full/trials.jsonl          行为数据（1500 trials）
#   data/exp6_activations/acts_*.npy     保存的 hidden states [1500, 29, 1024]
#
# 主要输出：
#   results/exp7a/model_fits.json        行为模型拟合结果
#   results/exp7b/probe_results.json     各层 Probe 结果
#   results/exp7b/probe_summary.json     最佳层汇总（L27 r=0.969）
#
# 注意：本脚本不更新 Qwen 模型权重。
# ============================================================

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent.parent  # experiment_7 repo root
EXP6_FULL       = _THIS_DIR / "data/exp6_full"
EXP6_ACTS       = _THIS_DIR / "data/exp6_activations"
EXP1_CODE       = _THIS_DIR / "data/exp1-4"
OUT_BASE        = _THIS_DIR / "results"
FIG_DIR         = _THIS_DIR / "figures"

DEFAULT_MODEL   = "Qwen/Qwen3-0.6B"
N_BLOCKS        = 28   # Qwen3-0.6B transformer blocks
N_LAYERS        = 29   # hidden_states index: 0=embedding, 1..28=block outputs
HIDDEN_SIZE     = 1024


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_trials() -> list[dict]:
    return [json.loads(l) for l in (EXP6_FULL / "trials.jsonl").read_text().splitlines() if l]


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 4 or np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def ols_probe(X_tr: np.ndarray, y_tr: np.ndarray,
              X_te: np.ndarray, y_te: np.ndarray) -> tuple[float, np.ndarray]:
    """OLS probe 线性探测器。在训练集上训练，在测试集上用 Pearson r 评估。

    参数：
        X_tr: 训练集激活矩阵，shape (N_train, 1024)
        y_tr: 训练集目标变量，shape (N_train,)，如 choice_tendency
        X_te: 测试集激活矩阵，shape (N_test, 1024)
        y_te: 测试集目标变量，shape (N_test,)
    返回：
        r_test: float，测试集上的 Pearson r（相关系数）
        w_unit: shape (1024,)，单位化的 probe 方向向量
    数学：
        最小化 ||X_c · w - y_c||²（普通最小二乘法 OLS）
        解析解：w = (X_c^T X_c)^{-1} X_c^T y_c（通过 lstsq 实现）
    注意：
        不改变 Qwen 参数；probe 只是在激活空间里找一个线性方向。
    """
    mu = X_tr.mean(0)
    Xc_tr = X_tr - mu
    Xc_te = X_te - mu
    if np.std(y_tr) < 1e-10 or len(y_tr) < 4:
        return float("nan"), np.zeros(X_tr.shape[1])
    w, _, _, _ = np.linalg.lstsq(Xc_tr, y_tr - y_tr.mean(), rcond=None)
    r = corr(Xc_te @ w, y_te - y_te.mean())
    w_unit = w / (np.linalg.norm(w) + 1e-12)
    return r, w_unit


def episode_split(rows: list[dict], seed: int = 42, train_frac: float = 0.7
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Episode 级别的 70/30 训练/测试分割，避免 within-episode 数据泄露。

    参数：
        rows: trial 记录列表，每条记录含 "episode_id" 字段
        seed: 随机种子，默认42（保证可复现）
        train_frac: 训练比例，默认0.7（50个episode → 35训练 + 15测试）
    返回：
        tr: 布尔数组 shape (N,)，True 表示该 trial 在训练集
        te: 布尔数组 shape (N,)，True 表示该 trial 在测试集
    重要：
        按 episode 分割，同一 episode 的所有 trial 不会被分开。
        这样可以避免同一个 episode 内相邻 trial 之间的信息泄露。
    """
    all_eps = np.array(sorted(set(r["episode_id"] for r in rows)))
    rng = np.random.default_rng(seed)
    rng.shuffle(all_eps)
    train_eps = set(all_eps[:int(train_frac * len(all_eps))])
    tr = np.array([r["episode_id"] in train_eps for r in rows])
    return tr, ~tr


def _import_exp1() -> tuple[Any, Any, Any, Any]:
    """Import load_model, model_device, apply_chat_template, resolve_transformer_blocks."""
    if str(EXP1_CODE) not in sys.path:
        sys.path.insert(0, str(EXP1_CODE))
    from causal_rpe_bandit import (
        load_model, model_device,
        apply_chat_template, resolve_transformer_blocks,
    )
    return load_model, model_device, apply_chat_template, resolve_transformer_blocks


def _import_seq() -> Any:
    if str(EXP1_CODE) not in sys.path:
        sys.path.insert(0, str(EXP1_CODE))
    from sequential_bandit_rpe import (
        SYSTEM, BIN_LABELS, BINS,
        history_text, feedback_readout_raw, prepared,
    )
    return SYSTEM, BIN_LABELS, BINS, history_text, feedback_readout_raw, prepared


# ---------------------------------------------------------------------------
# Computational variables for probing (7B targets)
# ---------------------------------------------------------------------------

def build_targets(rows: list[dict]) -> dict[str, np.ndarray]:
    """Build four computational variable arrays from trial records."""
    n = len(rows)

    # reward_history: cumulative mean reward up to (but not including) current trial
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

    # delta_q: q_stated_A - q_stated_B (relative value estimate)
    delta_q = np.array([r["q_stated_A"] - r["q_stated_B"] for r in rows])

    # rpe_proxy: reward - q_stated_chosen (prediction error proxy)
    rpe_proxy = np.array([
        r["reward"] - (r["q_stated_A"] if r["chosen_arm"] == "A" else r["q_stated_B"])
        for r in rows
    ])

    # choice_tendency: calibrated choice logit difference (A - B)
    choice_tendency = np.array([r["choice_logit_diff"] for r in rows])

    # recent_reward: mean reward in last 3 trials of same episode
    recent_reward = np.zeros(n)
    ep_rewards: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        eid = r["episode_id"]
        if eid not in ep_rewards:
            ep_rewards[eid] = []
        hist = ep_rewards[eid]
        recent_reward[i] = np.mean(hist[-3:]) if hist else 0.5
        ep_rewards[eid].append(r["reward"])

    # is_post_reversal: binary phase label
    post_reversal = np.array([float(r["is_post_reversal"]) for r in rows])

    # choice_a: binary outcome
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

    # ── Optional: enrich with behavioral-model-fitted variables (Stage 7A) ──
    model_fits_path = OUT_BASE / "exp7a" / "model_fits.json"
    if model_fits_path.exists():
        try:
            mf = json.loads(model_fits_path.read_text())
            per_trial = mf.get("per_trial", [])
            # Index by (episode_id, trial) for safe alignment
            lookup = {(d["episode_id"], d["trial"]): d for d in per_trial}
            rw_delta_q  = np.full(n, np.nan)
            rw_rpe      = np.full(n, np.nan)
            rw_q_chosen = np.full(n, np.nan)
            for i, r in enumerate(rows):
                key = (r["episode_id"], r["trial"])
                d = lookup.get(key)
                if d is None:
                    continue
                qa = d.get("rw_Q_A", np.nan)
                qb = d.get("rw_Q_B", np.nan)
                rw_delta_q[i] = qa - qb
                rw_rpe[i]     = d.get("rw_rpe", np.nan)
                rw_q_chosen[i] = qa if r["chosen_arm"] == "A" else qb
            # Only add if we have some non-NaN values
            if np.isfinite(rw_delta_q).any():
                # Replace NaN with 0 so downstream probing is robust; the mask is implicit
                targets["rw_delta_q"]  = np.nan_to_num(rw_delta_q,  nan=0.0)
                targets["rw_rpe"]      = np.nan_to_num(rw_rpe,      nan=0.0)
                targets["rw_q_chosen"] = np.nan_to_num(rw_q_chosen, nan=0.5)
        except Exception as e:
            print(f"NOTE: could not load model-fitted targets from {model_fits_path}: {e}")

    return targets


# ---------------------------------------------------------------------------
# 7A behavioral — fit_models (RW, Recency, WSLS)
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    # numerically stable sigmoid
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def _rw_nll(params, choices_a, rewards):
    """Rescorla-Wagner negative log-likelihood for a single episode.
    choices_a: 1 if A chosen on trial t, else 0.  rewards: 0/1 outcome.
    """
    alpha, beta = params
    T = len(choices_a)
    QA, QB = 0.5, 0.5
    nll = 0.0
    eps = 1e-9
    for t in range(T):
        dq = QA - QB
        p_a = 1.0 / (1.0 + np.exp(-beta * dq))
        p_a = min(max(p_a, eps), 1.0 - eps)
        if choices_a[t] == 1:
            nll -= np.log(p_a)
        else:
            nll -= np.log(1.0 - p_a)
        # Update Q of chosen arm
        if choices_a[t] == 1:
            delta = rewards[t] - QA
            QA = QA + alpha * delta
        else:
            delta = rewards[t] - QB
            QB = QB + alpha * delta
    return nll


def _rw_trajectory(alpha, beta, choices_a, rewards):
    """Return per-trial Q_A, Q_B, RPE (δ_t = r_t - Q(chosen)), P(A) sequences."""
    T = len(choices_a)
    QA, QB = 0.5, 0.5
    Qa_arr  = np.zeros(T)
    Qb_arr  = np.zeros(T)
    rpe_arr = np.zeros(T)
    pa_arr  = np.zeros(T)
    for t in range(T):
        Qa_arr[t] = QA
        Qb_arr[t] = QB
        dq = QA - QB
        pa_arr[t] = 1.0 / (1.0 + np.exp(-beta * dq))
        if choices_a[t] == 1:
            delta = rewards[t] - QA
            rpe_arr[t] = delta
            QA = QA + alpha * delta
        else:
            delta = rewards[t] - QB
            rpe_arr[t] = delta
            QB = QB + alpha * delta
    return Qa_arr, Qb_arr, rpe_arr, pa_arr


def _recency_nll(params, choices_a, rewards):
    """Recency-weighted (per-arm exponential moving average) NLL."""
    gamma, beta = params
    T = len(choices_a)
    QA, QB = 0.5, 0.5
    nll = 0.0
    eps = 1e-9
    for t in range(T):
        dq = QA - QB
        p_a = 1.0 / (1.0 + np.exp(-beta * dq))
        p_a = min(max(p_a, eps), 1.0 - eps)
        if choices_a[t] == 1:
            nll -= np.log(p_a)
            # EMA update for arm A only
            QA = gamma * QA + (1.0 - gamma) * rewards[t]
        else:
            nll -= np.log(1.0 - p_a)
            QB = gamma * QB + (1.0 - gamma) * rewards[t]
    return nll


def _recency_trajectory(gamma, beta, choices_a, rewards):
    T = len(choices_a)
    QA, QB = 0.5, 0.5
    Qa_arr = np.zeros(T)
    Qb_arr = np.zeros(T)
    pa_arr = np.zeros(T)
    for t in range(T):
        Qa_arr[t] = QA
        Qb_arr[t] = QB
        dq = QA - QB
        pa_arr[t] = 1.0 / (1.0 + np.exp(-beta * dq))
        if choices_a[t] == 1:
            QA = gamma * QA + (1.0 - gamma) * rewards[t]
        else:
            QB = gamma * QB + (1.0 - gamma) * rewards[t]
    return Qa_arr, Qb_arr, pa_arr


def _wsls_nll(params, choices_a, rewards):
    """Win-Stay-Lose-Shift NLL. Trial 0 is uniform."""
    (eps_noise,) = params
    T = len(choices_a)
    nll = 0.0
    eps = 1e-9
    for t in range(T):
        if t == 0:
            # No prior → uniform
            p_a = 0.5
        else:
            prev_a = choices_a[t - 1] == 1
            prev_r = rewards[t - 1] == 1
            if prev_r:
                # Stay: P(stay) = 1 - eps_noise
                p_stay = 1.0 - eps_noise
            else:
                # Shift: P(stay) = eps_noise
                p_stay = eps_noise
            # P(A) depends on prev choice
            if prev_a:
                p_a = p_stay
            else:
                p_a = 1.0 - p_stay
        p_a = min(max(p_a, eps), 1.0 - eps)
        if choices_a[t] == 1:
            nll -= np.log(p_a)
        else:
            nll -= np.log(1.0 - p_a)
    return nll


def _wsls_trajectory(eps_noise, choices_a, rewards):
    T = len(choices_a)
    p_stay_arr = np.zeros(T)
    for t in range(T):
        if t == 0:
            p_stay_arr[t] = 0.5
            continue
        prev_r = rewards[t - 1] == 1
        p_stay_arr[t] = (1.0 - eps_noise) if prev_r else eps_noise
    return p_stay_arr


def command_fit_models(args: argparse.Namespace) -> None:
    """Stage 7A behavioral — fit RW, Recency, WSLS models per episode."""
    from scipy.optimize import minimize

    rows = load_trials()

    # Group trials by episode (preserving within-episode order)
    ep_to_rows: dict[str, list[dict]] = {}
    for r in rows:
        ep_to_rows.setdefault(r["episode_id"], []).append(r)
    for eid in ep_to_rows:
        ep_to_rows[eid].sort(key=lambda d: d["trial"])

    rw_fits:      list[dict] = []
    rec_fits:     list[dict] = []
    wsls_fits:    list[dict] = []
    per_trial:    list[dict] = []

    # NLL/BIC totals across episodes
    tot = {"rw": {"nll": 0.0, "bic": 0.0, "k": 2},
           "recency": {"nll": 0.0, "bic": 0.0, "k": 2},
           "wsls": {"nll": 0.0, "bic": 0.0, "k": 1}}

    rng_init = np.random.default_rng(42)

    for ep_i, (eid, ep_rows) in enumerate(sorted(ep_to_rows.items())):
        T = len(ep_rows)
        choices_a = np.array([1 if r["chosen_arm"] == "A" else 0 for r in ep_rows], dtype=int)
        rewards   = np.array([int(r["reward"]) for r in ep_rows], dtype=int)

        # ── Fit RW ──
        best_rw = None
        for _ in range(4):
            a0 = float(rng_init.uniform(0.05, 0.95))
            b0 = float(rng_init.uniform(1.0, 10.0))
            try:
                res = minimize(_rw_nll, [a0, b0],
                               args=(choices_a, rewards),
                               method="L-BFGS-B",
                               bounds=[(1e-4, 1.0), (0.0, 20.0)])
                if best_rw is None or res.fun < best_rw.fun:
                    best_rw = res
            except Exception:
                continue
        if best_rw is None:
            alpha_hat, beta_hat_rw, nll_rw = 0.1, 1.0, _rw_nll([0.1, 1.0], choices_a, rewards)
        else:
            alpha_hat, beta_hat_rw = float(best_rw.x[0]), float(best_rw.x[1])
            nll_rw = float(best_rw.fun)
        bic_rw = 2 * np.log(T) + 2 * nll_rw

        # ── Fit Recency ──
        best_rec = None
        for _ in range(4):
            g0 = float(rng_init.uniform(0.55, 0.95))
            b0 = float(rng_init.uniform(1.0, 10.0))
            try:
                res = minimize(_recency_nll, [g0, b0],
                               args=(choices_a, rewards),
                               method="L-BFGS-B",
                               bounds=[(0.5, 1.0 - 1e-4), (0.0, 20.0)])
                if best_rec is None or res.fun < best_rec.fun:
                    best_rec = res
            except Exception:
                continue
        if best_rec is None:
            gamma_hat, beta_hat_rec, nll_rec = 0.8, 1.0, _recency_nll([0.8, 1.0], choices_a, rewards)
        else:
            gamma_hat, beta_hat_rec = float(best_rec.x[0]), float(best_rec.x[1])
            nll_rec = float(best_rec.fun)
        bic_rec = 2 * np.log(T) + 2 * nll_rec

        # ── Fit WSLS ──
        best_w = None
        for _ in range(4):
            e0 = float(rng_init.uniform(0.05, 0.45))
            try:
                res = minimize(_wsls_nll, [e0],
                               args=(choices_a, rewards),
                               method="L-BFGS-B",
                               bounds=[(1e-4, 0.5)])
                if best_w is None or res.fun < best_w.fun:
                    best_w = res
            except Exception:
                continue
        if best_w is None:
            eps_hat, nll_w = 0.25, _wsls_nll([0.25], choices_a, rewards)
        else:
            eps_hat = float(best_w.x[0])
            nll_w = float(best_w.fun)
        bic_w = 1 * np.log(T) + 2 * nll_w

        # Accumulate totals
        tot["rw"]["nll"]      += nll_rw
        tot["rw"]["bic"]      += bic_rw
        tot["recency"]["nll"] += nll_rec
        tot["recency"]["bic"] += bic_rec
        tot["wsls"]["nll"]    += nll_w
        tot["wsls"]["bic"]    += bic_w

        rw_fits.append({"episode_id": eid, "alpha": alpha_hat, "beta": beta_hat_rw,
                        "nll": nll_rw, "bic": bic_rw, "T": T})
        rec_fits.append({"episode_id": eid, "gamma": gamma_hat, "beta": beta_hat_rec,
                         "nll": nll_rec, "bic": bic_rec, "T": T})
        wsls_fits.append({"episode_id": eid, "epsilon": eps_hat,
                          "nll": nll_w, "bic": bic_w, "T": T})

        # Reconstruct per-trial trajectories
        rw_QA, rw_QB, rw_rpe, rw_PA   = _rw_trajectory(alpha_hat, beta_hat_rw, choices_a, rewards)
        rec_QA, rec_QB, rec_PA        = _recency_trajectory(gamma_hat, beta_hat_rec, choices_a, rewards)
        wsls_p_stay                   = _wsls_trajectory(eps_hat, choices_a, rewards)

        for t, r in enumerate(ep_rows):
            per_trial.append({
                "episode_id":     eid,
                "trial":          int(r["trial"]),
                "chosen_arm":     r["chosen_arm"],
                "reward":         int(r["reward"]),
                "rw_Q_A":         float(rw_QA[t]),
                "rw_Q_B":         float(rw_QB[t]),
                "rw_rpe":         float(rw_rpe[t]),
                "rw_P_A":         float(rw_PA[t]),
                "recency_Q_A":    float(rec_QA[t]),
                "recency_Q_B":    float(rec_QB[t]),
                "recency_P_A":    float(rec_PA[t]),
                "wsls_p_stay":    float(wsls_p_stay[t]),
            })

        if (ep_i + 1) % 10 == 0 or ep_i + 1 == len(ep_to_rows):
            print(f"fit_models  ep {ep_i+1}/{len(ep_to_rows)}  "
                  f"RW α={alpha_hat:.2f} β={beta_hat_rw:.2f}  "
                  f"Rec γ={gamma_hat:.2f}  WSLS ε={eps_hat:.2f}", flush=True)

    # Winner
    totals_sorted = sorted(
        [("rw", tot["rw"]["bic"]), ("recency", tot["recency"]["bic"]), ("wsls", tot["wsls"]["bic"])],
        key=lambda x: x[1])
    winner = totals_sorted[0][0]

    print("\n=== Behavioral model comparison ===")
    print(f"  RW       total NLL={tot['rw']['nll']:.1f}   total BIC={tot['rw']['bic']:.1f}")
    print(f"  Recency  total NLL={tot['recency']['nll']:.1f}   total BIC={tot['recency']['bic']:.1f}")
    print(f"  WSLS     total NLL={tot['wsls']['nll']:.1f}   total BIC={tot['wsls']['bic']:.1f}")
    print(f"  Winner (lowest BIC): {winner}")

    out = OUT_BASE / "exp7a"
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "model_fits.json", {
        "totals":   tot,
        "winner":   winner,
        "rw_fits":  rw_fits,
        "recency_fits": rec_fits,
        "wsls_fits":    wsls_fits,
        "per_trial":    per_trial,
    })
    print(f"\nSaved {out / 'model_fits.json'}")

    # Optional: companion figure for immediate inspection
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle("Exp 7A — Behavioral Model Fits", fontsize=13, fontweight="bold")

        # Panel 1: mean BIC per model
        ax = axes[0, 0]
        bars = ax.bar(["RW", "Recency", "WSLS"],
                      [tot["rw"]["bic"], tot["recency"]["bic"], tot["wsls"]["bic"]],
                      color=["#2196F3", "#FF9800", "#9E9E9E"])
        ax.set_ylabel("Total BIC (lower = better)")
        ax.set_title("Model comparison (BIC)")
        for b, v in zip(bars, [tot["rw"]["bic"], tot["recency"]["bic"], tot["wsls"]["bic"]]):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=9)

        # Panel 2: alpha histogram
        ax = axes[0, 1]
        alphas = [f["alpha"] for f in rw_fits]
        ax.hist(alphas, bins=20, color="#2196F3", alpha=0.8)
        ax.set_xlabel("α (RW learning rate)")
        ax.set_ylabel("Episode count")
        ax.set_title(f"RW α distribution  (mean={np.mean(alphas):.2f})")

        # Panel 3: fitted P(A) vs actual, averaged across episodes
        ax = axes[1, 0]
        # group per_trial by trial index within episode
        max_T = max(f["T"] for f in rw_fits)
        actual = np.full((len(rw_fits), max_T), np.nan)
        rw_pa  = np.full((len(rw_fits), max_T), np.nan)
        rec_pa = np.full((len(rw_fits), max_T), np.nan)
        # Need trial-within-episode index — reconstruct
        ep_order: dict[str, int] = {}
        for d in per_trial:
            idx = ep_order.get(d["episode_id"], 0)
            ep_order[d["episode_id"]] = idx + 1
        # Rebuild with within-ep index
        ep_counter: dict[str, int] = {}
        ep_id_to_row: dict[str, int] = {f["episode_id"]: i for i, f in enumerate(rw_fits)}
        for d in per_trial:
            eid = d["episode_id"]
            t_in_ep = ep_counter.get(eid, 0)
            row = ep_id_to_row[eid]
            if t_in_ep < max_T:
                actual[row, t_in_ep] = 1.0 if d["chosen_arm"] == "A" else 0.0
                rw_pa[row,  t_in_ep] = d["rw_P_A"]
                rec_pa[row, t_in_ep] = d["recency_P_A"]
            ep_counter[eid] = t_in_ep + 1

        trial_idx = np.arange(max_T)
        ax.plot(trial_idx, np.nanmean(actual, axis=0), color="black",
                linewidth=2, label="Actual P(A)")
        ax.plot(trial_idx, np.nanmean(rw_pa,  axis=0), color="#2196F3",
                linewidth=1.6, label="RW fitted")
        ax.plot(trial_idx, np.nanmean(rec_pa, axis=0), color="#FF9800",
                linewidth=1.6, label="Recency fitted")
        ax.set_xlabel("Trial within episode")
        ax.set_ylabel("P(A)")
        ax.set_title("Mean P(A) trajectory")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Panel 4: RW δ_t vs q_stated_diff
        ax = axes[1, 1]
        # match per_trial with rows by (episode_id, trial)
        trial_lookup = {(r["episode_id"], r["trial"]): r for r in rows}
        xs, ys = [], []
        for d in per_trial:
            r = trial_lookup.get((d["episode_id"], d["trial"]))
            if r is None:
                continue
            xs.append(d["rw_rpe"])
            ys.append(r["q_stated_A"] - r["q_stated_B"])
        xs, ys = np.array(xs), np.array(ys)
        ax.scatter(xs, ys, alpha=0.25, s=10, color="#2196F3")
        c = corr(xs, ys)
        ax.set_xlabel("RW δ_t (RPE)")
        ax.set_ylabel("q_stated_A − q_stated_B")
        ax.set_title(f"RW RPE vs declared ΔQ   r={c:+.3f}")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.axvline(0, color="gray", linewidth=0.8)

        plt.tight_layout()
        fig.savefig(FIG_DIR / "exp7a_model_comparison.png", dpi=150)
        plt.close(fig)
        print(f"Saved {FIG_DIR / 'exp7a_model_comparison.png'}")
    except Exception as e:
        print(f"NOTE: skipped companion figure: {e}")


# ---------------------------------------------------------------------------
# 7A / collect_t2 — feedback-position activations
# ---------------------------------------------------------------------------

def command_collect_t2(args: argparse.Namespace) -> None:
    """Extract residual-stream activations at the reward-feedback position (T2)."""
    load_model, model_device, apply_chat_template, _ = _import_exp1()
    SYSTEM, BIN_LABELS, BINS, history_text, feedback_readout_raw, prepared = _import_seq()
    import torch

    rows = load_trials()
    out  = (OUT_BASE / "exp7a")
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "acts_feedback.npy"

    # Pre-allocate memmap
    arr = np.lib.format.open_memmap(out_path, mode="w+",
                                    dtype=np.float32, shape=(len(rows), N_LAYERS, HIDDEN_SIZE))

    torch_mod, model, tokenizer = load_model(args.model, args.device, args.dtype)
    dev = model_device(model)

    for i, row in enumerate(rows):
        history = [(str(a), int(r)) for a, r in row["history"]]
        bins    = np.asarray(row["bin_values_by_label"], float)
        _, prefix = feedback_readout_raw(history, row["chosen_arm"], row["reward"], bins)
        prompt = apply_chat_template(tokenizer, prefix)
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch_mod.inference_mode():
            out_m = model(**enc, output_hidden_states=True, return_dict=True)
        arr[i] = np.stack([h[0, -1].detach().float().cpu().numpy()
                           for h in out_m.hidden_states])
        if (i + 1) % 100 == 0 or i + 1 == len(rows):
            print(f"collect_t2  {i+1}/{len(rows)}", flush=True)

    dump(out / "metadata.json", {
        "model": args.model, "n_trials": len(rows),
        "shape": [len(rows), N_LAYERS, HIDDEN_SIZE],
        "position": "FEEDBACK_END token — model has just seen reward outcome",
    })
    print(f"Saved {out_path}  shape={arr.shape}")


# ---------------------------------------------------------------------------
# 7C / collect_comps — x_mid via post_attention_layernorm hooks
# ---------------------------------------------------------------------------

def command_collect_comps(args: argparse.Namespace) -> None:
    """Extract x_mid (post-attention, pre-MLP) for every block at the choice position."""
    load_model, model_device, apply_chat_template, resolve_transformer_blocks = _import_exp1()
    SYSTEM, BIN_LABELS, BINS, history_text, _, prepared = _import_seq()
    from exp6_fvd_bandit import fvd_choice_raw  # reuse from exp6 code

    import torch
    sys.path.insert(0, str(_THIS_DIR / "data/causal_rpe_bandit_code"))
    try:
        from exp6_fvd_bandit import fvd_choice_raw
    except ImportError:
        sys.path.insert(0, str(_THIS_DIR / "data/causal_rpe_bandit_code"))
        from exp6_fvd_bandit import fvd_choice_raw

    rows = load_trials()
    out  = (OUT_BASE / "exp7c")
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "acts_x_mid.npy"

    arr = np.lib.format.open_memmap(out_path, mode="w+",
                                    dtype=np.float32, shape=(len(rows), N_BLOCKS, HIDDEN_SIZE))

    torch_mod, model, tokenizer = load_model(args.model, args.device, args.dtype)
    dev = model_device(model)
    blocks = resolve_transformer_blocks(model)

    # Storage for this forward pass
    x_mid_this: dict[int, np.ndarray] = {}
    hooks = []

    for l_idx in range(N_BLOCKS):
        # post_attention_layernorm input = x_mid
        norm_module = _get_post_attn_norm(blocks[l_idx])
        if norm_module is None:
            raise RuntimeError(
                f"Cannot find post_attention_layernorm in block {l_idx}. "
                "Check model architecture."
            )
        def make_hook(li):
            def _hook(module, inp, out_h):
                # inp[0] shape: (batch, seq, hidden) — x_mid before layernorm
                x_mid_this[li] = inp[0][0, -1].detach().float().cpu().numpy()
            return _hook
        hooks.append(norm_module.register_forward_hook(make_hook(l_idx)))

    for i, row in enumerate(rows):
        x_mid_this.clear()
        history = [(str(a), int(r)) for a, r in row["history"]]
        bins    = np.asarray(row["bin_values_by_label"], float)
        la, lb  = row["q_stated_A_label"], row["q_stated_B_label"]
        raw     = fvd_choice_raw(history, bins, la, lb)
        prompt  = apply_chat_template(tokenizer, raw)
        enc     = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        enc     = {k: v.to(dev) for k, v in enc.items()}
        with torch_mod.inference_mode():
            model(**enc, return_dict=True)
        for l_idx in range(N_BLOCKS):
            if l_idx in x_mid_this:
                arr[i, l_idx] = x_mid_this[l_idx]
        if (i + 1) % 100 == 0 or i + 1 == len(rows):
            print(f"collect_comps  {i+1}/{len(rows)}", flush=True)

    for h in hooks:
        h.remove()

    dump(out / "metadata.json", {
        "model": args.model, "n_trials": len(rows),
        "shape": [len(rows), N_BLOCKS, HIDDEN_SIZE],
        "position": "choice token — x_mid captured via post_attention_layernorm hook",
        "note": "x_in[l]=acts_choice[:,l,:], x_out[l]=acts_choice[:,l+1,:] from exp6_activations",
    })
    print(f"Saved {out_path}  shape={arr.shape}")


def _get_post_attn_norm(block: Any) -> Any:
    """Find the post-attention layernorm in a decoder block."""
    for name in ("post_attention_layernorm", "ln2", "ln_2", "post_attn_layernorm"):
        m = getattr(block, name, None)
        if m is not None:
            return m
    return None


# ---------------------------------------------------------------------------
# 7B / probe_7b — four-variable layer-wise probing
# ---------------------------------------------------------------------------

def command_probe_7b(args: argparse.Namespace) -> None:
    """Layer-wise OLS probing of four computational variables across all positions."""
    rows = load_trials()
    tgts = build_targets(rows)
    tr, te = episode_split(rows, seed=args.seed)

    # Available activation arrays
    acts_map: dict[str, np.ndarray] = {}
    for name, path in [
        ("declare_a", EXP6_ACTS / "acts_declare_a.npy"),
        ("declare_b", EXP6_ACTS / "acts_declare_b.npy"),
        ("choice",    EXP6_ACTS / "acts_choice.npy"),
    ]:
        if path.exists():
            acts_map[name] = np.load(path, mmap_mode="r")
        else:
            print(f"WARNING: {path} not found — skipping position {name}")

    t2_path = OUT_BASE / "exp7a" / "acts_feedback.npy"
    if t2_path.exists():
        acts_map["feedback"] = np.load(t2_path, mmap_mode="r")
    else:
        print(f"NOTE: {t2_path} not found — run collect_t2 first to include T2 position")

    # Variables to probe at each position
    target_names = ["reward_history", "recent_reward", "delta_q", "rpe_proxy",
                    "choice_tendency", "post_reversal"]
    # Model-fitted targets (only probed if available in tgts)
    for extra in ("rw_delta_q", "rw_rpe", "rw_q_chosen"):
        if extra in tgts:
            target_names.append(extra)

    records: list[dict] = []
    probe_dirs: dict[tuple, np.ndarray] = {}

    for pos, A in acts_map.items():
        n_lay = A.shape[1]
        for layer in range(n_lay):
            X = np.asarray(A[:, layer, :], dtype=np.float64)
            for tname in target_names:
                y = tgts[tname]
                r, w = ols_probe(X[tr], y[tr], X[te], y[te])
                records.append({
                    "position": pos, "layer": layer,
                    "target": tname, "r_test": r,
                })
                probe_dirs[(pos, layer, tname)] = w
        print(f"probe_7b  probed position: {pos}", flush=True)

    # Cross-position cosine similarities (same variable, different positions)
    cos_records: list[dict] = []
    positions = list(acts_map.keys())
    for tname in target_names:
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                pa, pb = positions[i], positions[j]
                for layer in range(min(acts_map[pa].shape[1], acts_map[pb].shape[1])):
                    wa = probe_dirs.get((pa, layer, tname))
                    wb = probe_dirs.get((pb, layer, tname))
                    if wa is not None and wb is not None:
                        cos_records.append({
                            "target": tname, "layer": layer,
                            "pos_a": pa, "pos_b": pb,
                            "cosine": float(np.dot(wa, wb)),
                        })

    # Summary: best layer per (position, target)
    from collections import defaultdict
    by_pt: dict[tuple, list] = defaultdict(list)
    for rec in records:
        by_pt[(rec["position"], rec["target"])].append((rec["layer"], rec["r_test"]))

    best_r: dict[str, dict] = {}
    print("\n=== Best test-set r per (position, target) ===")
    for (pos, tgt), vals in sorted(by_pt.items()):
        valid = [(l, r) for l, r in vals if r == r]  # filter NaN
        if not valid:
            continue
        best_l, best_r_val = max(valid, key=lambda x: abs(x[1]))
        key = f"{pos}|{tgt}"
        best_r[key] = {"r": best_r_val, "layer": best_l}
        print(f"  {pos:12s}  {tgt:20s}  r={best_r_val:+.3f} @ layer {best_l}")

    # ── Theory-derived vs model-fitted comparison (per position) ──
    theory_vs_model_pairs = [
        ("delta_q",   "rw_delta_q"),
        ("rpe_proxy", "rw_rpe"),
    ]
    tvm_summary: list[dict] = []
    print("\n=== Theory-derived vs model-fitted targets ===")
    for pos in acts_map.keys():
        for theory_tgt, model_tgt in theory_vs_model_pairs:
            k_theory = f"{pos}|{theory_tgt}"
            k_model  = f"{pos}|{model_tgt}"
            theory_info = best_r.get(k_theory)
            model_info  = best_r.get(k_model)
            if theory_info is None and model_info is None:
                continue
            r_theory = theory_info["r"] if theory_info else float("nan")
            r_model  = model_info["r"]  if model_info  else float("nan")
            which = ("model" if model_info and
                     (theory_info is None or abs(r_model) > abs(r_theory))
                     else "theory")
            tvm_summary.append({
                "position": pos,
                "theory_target": theory_tgt,
                "model_target":  model_tgt,
                "r_theory":  r_theory,
                "r_model":   r_model,
                "best_layer_theory": theory_info["layer"] if theory_info else None,
                "best_layer_model":  model_info["layer"]  if model_info  else None,
                "winner": which,
            })
            print(f"  {pos:12s}  theory({theory_tgt:12s}) r={r_theory:+.3f}  "
                  f"vs model({model_tgt:12s}) r={r_model:+.3f}  → {which}")

    # Best variable per position (across all probed targets)
    best_per_pos: dict[str, dict] = {}
    for key, info in best_r.items():
        pos, tgt = key.split("|", 1)
        cur = best_per_pos.get(pos)
        if cur is None or abs(info["r"]) > abs(cur["r"]):
            best_per_pos[pos] = {"target": tgt, "r": info["r"], "layer": info["layer"]}
    print("\n=== Best variable per position (any target) ===")
    for pos, info in sorted(best_per_pos.items()):
        print(f"  {pos:12s}  best={info['target']:20s}  r={info['r']:+.3f} @ layer {info['layer']}")

    out = OUT_BASE / "exp7b"
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "probe_results.json", records)
    dump(out / "cosine_records.json", cos_records)
    dump(out / "probe_summary.json", {
        "best_r": best_r,
        "theory_vs_model": tvm_summary,
        "best_per_position": best_per_pos,
    })
    print(f"\nSaved to {out}")


# ---------------------------------------------------------------------------
# 7C / analyze_7c — attention vs MLP contribution
# ---------------------------------------------------------------------------

def command_analyze_7c(args: argparse.Namespace) -> None:
    """Probe attention and MLP contributions separately for all 7B variables."""
    x_mid_path = OUT_BASE / "exp7c" / "acts_x_mid.npy"
    if not x_mid_path.exists():
        sys.exit(f"ERROR: {x_mid_path} not found. Run collect_comps first.")

    rows  = load_trials()
    tgts  = build_targets(rows)
    tr, te = episode_split(rows, seed=42)

    # x_in[l] = acts_choice[:, l, :]  (residual entering block l)
    # x_mid[l] = acts_x_mid[:, l, :]  (after attention residual add)
    # x_out[l] = acts_choice[:, l+1, :] (after MLP residual add)
    acts_choice = np.load(EXP6_ACTS / "acts_choice.npy", mmap_mode="r")
    acts_x_mid  = np.load(x_mid_path, mmap_mode="r")

    target_names = ["reward_history", "delta_q", "rpe_proxy", "choice_tendency", "post_reversal"]
    records: list[dict] = []

    for l_idx in range(N_BLOCKS):
        x_in  = np.asarray(acts_choice[:, l_idx,     :], dtype=np.float64)
        x_mid = np.asarray(acts_x_mid[:,  l_idx,     :], dtype=np.float64)
        x_out = np.asarray(acts_choice[:, l_idx + 1, :], dtype=np.float64)
        d_attn = x_mid - x_in   # attention contribution
        d_mlp  = x_out - x_mid  # MLP contribution

        for tname in target_names:
            y = tgts[tname]
            r_full,  _ = ols_probe(x_out[tr], y[tr], x_out[te], y[te])
            r_attn,  _ = ols_probe(d_attn[tr], y[tr], d_attn[te], y[te])
            r_mlp,   _ = ols_probe(d_mlp[tr], y[tr], d_mlp[te], y[te])
            r_x_mid, _ = ols_probe(x_mid[tr], y[tr], x_mid[te], y[te])
            records.append({
                "layer": l_idx,
                "target": tname,
                "r_full_block": r_full,
                "r_attn_contribution": r_attn,
                "r_mlp_contribution": r_mlp,
                "r_x_mid": r_x_mid,
            })
        if (l_idx + 1) % 7 == 0 or l_idx + 1 == N_BLOCKS:
            print(f"analyze_7c  layer {l_idx + 1}/{N_BLOCKS}", flush=True)

    # Summary: per target, which component contributes more?
    from collections import defaultdict
    by_tgt: dict[str, list] = defaultdict(list)
    for rec in records:
        by_tgt[rec["target"]].append(rec)

    summary: dict[str, dict] = {}
    print("\n=== Component analysis summary ===")
    for tname, recs in sorted(by_tgt.items()):
        best_attn = max(recs, key=lambda x: abs(x["r_attn_contribution"] or 0))
        best_mlp  = max(recs, key=lambda x: abs(x["r_mlp_contribution"]  or 0))
        best_full = max(recs, key=lambda x: abs(x["r_full_block"]        or 0))
        summary[tname] = {
            "best_attn_layer": best_attn["layer"],
            "best_attn_r": best_attn["r_attn_contribution"],
            "best_mlp_layer": best_mlp["layer"],
            "best_mlp_r": best_mlp["r_mlp_contribution"],
            "best_full_layer": best_full["layer"],
            "best_full_r": best_full["r_full_block"],
        }
        print(f"  {tname:20s}  attn r={best_attn['r_attn_contribution']:+.3f}@L{best_attn['layer']}"
              f"  mlp r={best_mlp['r_mlp_contribution']:+.3f}@L{best_mlp['layer']}"
              f"  full r={best_full['r_full_block']:+.3f}@L{best_full['layer']}")

    out = OUT_BASE / "exp7c"
    dump(out / "component_analysis.json", records)
    dump(out / "component_summary.json", summary)
    print(f"\nSaved to {out}")


# ---------------------------------------------------------------------------
# 7D / intervene_7d — causal activation steering
# ---------------------------------------------------------------------------

def command_intervene_7d(args: argparse.Namespace) -> None:
    """Causal intervention: steer residual stream at best probe layer, measure ΔP(A)."""
    # Load probe summary to find best layer for choice_tendency
    probe_summary_path = OUT_BASE / "exp7b" / "probe_summary.json"
    if not probe_summary_path.exists():
        sys.exit("ERROR: run probe_7b first.")

    summary = json.loads(probe_summary_path.read_text())
    best_r = summary["best_r"]

    # Pick target variable and position for intervention
    target = args.intervention_target
    pos    = args.intervention_position
    key    = f"{pos}|{target}"
    if key not in best_r:
        available = [k for k in best_r if pos in k]
        sys.exit(f"ERROR: '{key}' not found. Available with pos={pos}: {available}")

    best_layer = best_r[key]["layer"]
    best_r_val = best_r[key]["r"]
    print(f"Intervention target: {target} @ position {pos}, layer {best_layer}  (r={best_r_val:.3f})")

    # Re-extract probe direction on full dataset (training set only)
    rows  = load_trials()
    tgts  = build_targets(rows)
    tr, te = episode_split(rows, seed=42)

    acts_map = {
        "declare_a": EXP6_ACTS / "acts_declare_a.npy",
        "declare_b": EXP6_ACTS / "acts_declare_b.npy",
        "choice":    EXP6_ACTS / "acts_choice.npy",
    }
    t2_path = OUT_BASE / "exp7a" / "acts_feedback.npy"
    if t2_path.exists():
        acts_map["feedback"] = t2_path

    if pos not in acts_map:
        sys.exit(f"ERROR: position '{pos}' not available. Options: {list(acts_map.keys())}")

    A = np.load(acts_map[pos], mmap_mode="r")
    X = np.asarray(A[:, best_layer, :], dtype=np.float64)
    y = tgts[target]
    _, v_dir = ols_probe(X[tr], y[tr], X[te], y[te])
    v_dir = v_dir.astype(np.float32)

    # Orthogonal control direction (random unit vector orthogonalized against v_dir)
    rng_ctrl = np.random.default_rng(args.seed)
    rand_vec = rng_ctrl.standard_normal(HIDDEN_SIZE).astype(np.float32)
    rand_vec -= rand_vec.dot(v_dir) * v_dir
    rand_vec /= np.linalg.norm(rand_vec) + 1e-12

    load_model, model_device, apply_chat_template, resolve_transformer_blocks = _import_exp1()
    import torch

    # Reuse fvd_choice_raw from exp6
    sys.path.insert(0, str(_THIS_DIR / "data/causal_rpe_bandit_code"))
    from exp6_fvd_bandit import fvd_choice_raw
    from sequential_bandit_rpe import BINS, BIN_LABELS, calibrated_logits, softmax, logits_for

    torch_mod, model, tokenizer = load_model(args.model, args.device, args.dtype)
    dev = model_device(model)
    blocks = resolve_transformer_blocks(model)

    # Calibration baseline (no-history, neutral labels)
    neutral_label = BIN_LABELS[len(BIN_LABELS) // 2]
    rng_ep = np.random.default_rng(42)
    rand_bins = rng_ep.permutation(BINS)

    def get_choice_probs(prompt_raw: str, patch_vec: np.ndarray | None,
                         alpha: float, layer_idx: int) -> np.ndarray:
        prompt = apply_chat_template(tokenizer, prompt_raw)
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(dev) for k, v in enc.items()}

        handles = []
        if patch_vec is not None:
            v_t = torch_mod.tensor(patch_vec, device=dev, dtype=model.dtype)
            def _hook(module, inp, out_h):
                tensor = out_h[0] if isinstance(out_h, tuple) else out_h
                tensor[:, -1, :] += alpha * v_t
                return (tensor,) + out_h[1:] if isinstance(out_h, tuple) else tensor
            handles.append(blocks[layer_idx].register_forward_hook(_hook))

        with torch_mod.inference_mode():
            out_m = model(**enc, return_dict=True)

        for h in handles:
            h.remove()

        logits = out_m.logits[0, -1]
        A_id = tokenizer.encode("A", add_special_tokens=False)[-1]
        B_id = tokenizer.encode("B", add_special_tokens=False)[-1]
        log_A = float(logits[A_id].cpu())
        log_B = float(logits[B_id].cpu())
        return np.array([log_A, log_B])

    # Run intervention on held-out trials
    # We patch at layer best_layer in the choice position
    # (note: the intervention happens at block best_layer output)
    # But if best position is 'choice', we patch at the choice forward pass.
    # For simplicity, we always patch at the choice prompt regardless of pos.

    test_rows = [r for r, is_te in zip(rows, te) if is_te]
    n_test = min(len(test_rows), args.n_test)
    test_rows = test_rows[:n_test]

    print(f"Running intervention on {n_test} held-out trials, α={args.alpha}")
    results: list[dict] = []

    for i, row in enumerate(test_rows):
        history = [(str(a), int(r)) for a, r in row["history"]]
        bins    = np.asarray(row["bin_values_by_label"], float)
        la, lb  = row["q_stated_A_label"], row["q_stated_B_label"]
        raw     = fvd_choice_raw(history, bins, la, lb)

        logits_base    = get_choice_probs(raw, None,      0.0, best_layer)
        logits_pos     = get_choice_probs(raw, v_dir,    +args.alpha, best_layer)
        logits_neg     = get_choice_probs(raw, v_dir,    -args.alpha, best_layer)
        logits_rand    = get_choice_probs(raw, rand_vec, +args.alpha, best_layer)

        pa_base  = float(np.exp(logits_base[0])  / (np.exp(logits_base[0])  + np.exp(logits_base[1])))
        pa_pos   = float(np.exp(logits_pos[0])   / (np.exp(logits_pos[0])   + np.exp(logits_pos[1])))
        pa_neg   = float(np.exp(logits_neg[0])   / (np.exp(logits_neg[0])   + np.exp(logits_neg[1])))
        pa_rand  = float(np.exp(logits_rand[0])  / (np.exp(logits_rand[0])  + np.exp(logits_rand[1])))

        results.append({
            "episode_id":  row["episode_id"],
            "trial":       row["trial"],
            "chosen_arm":  row["chosen_arm"],
            "reward":      row["reward"],
            "is_post_reversal": row["is_post_reversal"],
            "delta_q":     row["q_stated_A"] - row["q_stated_B"],
            "pa_base":     pa_base,
            "pa_pos":      pa_pos,
            "pa_neg":      pa_neg,
            "pa_rand":     pa_rand,
            "delta_pa_pos":  pa_pos  - pa_base,
            "delta_pa_neg":  pa_neg  - pa_base,
            "delta_pa_rand": pa_rand - pa_base,
        })
        if (i + 1) % 50 == 0 or i + 1 == n_test:
            print(f"intervene_7d  {i+1}/{n_test}", flush=True)

    # Summary statistics
    delta_pos  = np.array([r["delta_pa_pos"]  for r in results])
    delta_neg  = np.array([r["delta_pa_neg"]  for r in results])
    delta_rand = np.array([r["delta_pa_rand"] for r in results])

    summary = {
        "target": target, "position": pos, "layer": best_layer,
        "probe_r": best_r_val, "alpha": args.alpha, "n_test": n_test,
        "mean_delta_pa_positive_intervention": float(delta_pos.mean()),
        "mean_delta_pa_negative_intervention": float(delta_neg.mean()),
        "mean_delta_pa_random_control":         float(delta_rand.mean()),
        "std_delta_pa_positive": float(delta_pos.std()),
        "std_delta_pa_negative": float(delta_neg.std()),
        "std_delta_pa_random":   float(delta_rand.std()),
        "direction_consistent_positive": float(np.mean(delta_pos > 0)),
        "direction_consistent_negative": float(np.mean(delta_neg < 0)),
        "corr_delta_q_vs_delta_pa_pos": corr(
            np.array([r["delta_q"] for r in results]), delta_pos
        ),
    }
    print("\n=== Intervention summary ===")
    print(json.dumps(summary, indent=2))

    out = OUT_BASE / "exp7d"
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "intervention_results.json", results)
    dump(out / "intervention_summary.json", summary)
    print(f"\nSaved to {out}")


# ---------------------------------------------------------------------------
# 7E / discover_7e — data-driven PCA state discovery
# ---------------------------------------------------------------------------

def command_discover_7e(args: argparse.Namespace) -> None:
    """PCA-based discovery of choice-predictive low-dimensional hidden states."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    rows = load_trials()
    acts_choice = np.load(EXP6_ACTS / "acts_choice.npy", mmap_mode="r")

    # Use best-performing layer range from exp7b if available
    probe_summary_path = OUT_BASE / "exp7b" / "probe_summary.json"
    if probe_summary_path.exists():
        ps = json.loads(probe_summary_path.read_text())
        # Find layer with highest r for choice_tendency at choice position
        best_info = ps["best_r"].get("choice|choice_tendency", {})
        best_layer = int(best_info.get("layer", 20))
    else:
        best_layer = 20  # fallback
    print(f"Using layer {best_layer} for discovery (best choice_tendency probe layer)")

    # Layer range for multi-layer analysis
    layers_to_test = [5, 10, 15, best_layer, 25, 27]
    layers_to_test = sorted(set(l for l in layers_to_test if 0 <= l < N_LAYERS))

    # Build next-choice target: does the model choose A on the NEXT trial?
    # (within the same episode)
    ep_to_rows: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        ep_to_rows.setdefault(r["episode_id"], []).append(i)

    # next_choice_a[i] = 1 if model chooses A on trial i+1, else 0 (-1 = no next trial)
    next_choice_a = np.full(len(rows), -1, dtype=int)
    for ep_indices in ep_to_rows.values():
        for j in range(len(ep_indices) - 1):
            cur_idx  = ep_indices[j]
            next_idx = ep_indices[j + 1]
            next_choice_a[cur_idx] = int(rows[next_idx]["chosen_arm"] == "A")

    has_next = next_choice_a >= 0
    y_next = next_choice_a[has_next]

    # Episode-level split
    tr_mask, te_mask = episode_split(rows, seed=42)
    tr_with_next = tr_mask & has_next
    te_with_next = te_mask & has_next

    results: list[dict] = []
    print(f"n_train={tr_with_next.sum()}, n_test={te_with_next.sum()} (trials with next choice)")

    for layer in layers_to_test:
        X_full = np.asarray(acts_choice[:, layer, :], dtype=np.float64)

        # PCA on training set
        pca = PCA(n_components=args.n_components, random_state=42)
        sc  = StandardScaler()

        X_tr = X_full[tr_with_next]
        X_te = X_full[te_with_next]
        y_tr = next_choice_a[tr_with_next]
        y_te = next_choice_a[te_with_next]

        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)
        Z_tr = pca.fit_transform(X_tr_sc)
        Z_te = pca.transform(X_te_sc)

        # Logistic regression on PCA scores
        lr = LogisticRegression(max_iter=500, random_state=42, C=1.0)
        lr.fit(Z_tr, y_tr)
        y_pred_te = lr.predict_proba(Z_te)[:, 1]
        try:
            auc = float(roc_auc_score(y_te, y_pred_te))
        except Exception:
            auc = float("nan")
        acc = float(lr.score(Z_te, y_te))

        # Baseline: shuffle labels
        rng = np.random.default_rng(42)
        y_shuf = rng.permutation(y_tr)
        lr_shuf = LogisticRegression(max_iter=500, random_state=42, C=1.0)
        lr_shuf.fit(Z_tr, y_shuf)
        y_pred_shuf = lr_shuf.predict_proba(Z_te)[:, 1]
        try:
            auc_shuf = float(roc_auc_score(y_te, y_pred_shuf))
        except Exception:
            auc_shuf = float("nan")

        # Variance explained
        var_explained = float(pca.explained_variance_ratio_.sum())

        # How much does each PC individually predict next choice?
        pc_aucs = []
        for k in range(min(args.n_components, 5)):
            try:
                pc_auc = float(roc_auc_score(y_te, Z_te[:, k]))
            except Exception:
                pc_auc = float("nan")
            pc_aucs.append(pc_auc)

        results.append({
            "layer": layer,
            "n_components": args.n_components,
            "var_explained": var_explained,
            "auc_next_choice": auc,
            "acc_next_choice": acc,
            "auc_shuffled_baseline": auc_shuf,
            "auc_lift_over_shuffle": auc - auc_shuf if auc == auc else float("nan"),
            "top5_pc_individual_aucs": pc_aucs,
        })
        print(f"  layer {layer:2d}  AUC={auc:.3f}  shuffle={auc_shuf:.3f}  "
              f"var={var_explained:.3f}", flush=True)

    # Also test prediction 2 and 3 trials ahead using best layer
    ahead_results: list[dict] = []
    X_full = np.asarray(acts_choice[:, best_layer, :], dtype=np.float64)
    pca_best = PCA(n_components=args.n_components, random_state=42)
    sc_best  = StandardScaler()
    X_tr_sc  = sc_best.fit_transform(X_full[tr_with_next])
    X_te_sc  = sc_best.transform(X_full[te_with_next])
    Z_tr_best = pca_best.fit_transform(X_tr_sc)
    Z_te_best = pca_best.transform(X_te_sc)

    for k_ahead in [1, 2, 3]:
        k_target = np.full(len(rows), -1, dtype=int)
        for ep_indices in ep_to_rows.values():
            for j in range(len(ep_indices) - k_ahead):
                cur_idx = ep_indices[j]
                fut_idx = ep_indices[j + k_ahead]
                k_target[cur_idx] = int(rows[fut_idx]["chosen_arm"] == "A")
        has_k = k_target >= 0
        tr_k  = tr_mask & has_k
        te_k  = te_mask & has_k

        Z_tr_k = pca_best.transform(sc_best.transform(X_full[tr_k]))
        Z_te_k = pca_best.transform(sc_best.transform(X_full[te_k]))
        y_tr_k, y_te_k = k_target[tr_k], k_target[te_k]

        lr_k = LogisticRegression(max_iter=500, random_state=42, C=1.0)
        lr_k.fit(Z_tr_k, y_tr_k)
        try:
            auc_k = float(roc_auc_score(y_te_k, lr_k.predict_proba(Z_te_k)[:, 1]))
        except Exception:
            auc_k = float("nan")
        ahead_results.append({"k_ahead": k_ahead, "layer": best_layer, "auc": auc_k})
        print(f"  prediction {k_ahead}-trial-ahead AUC={auc_k:.3f}")

    out = OUT_BASE / "exp7e"
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "discovery_results.json", results)
    dump(out / "ahead_prediction.json", ahead_results)
    print(f"\nSaved to {out}")


# ---------------------------------------------------------------------------
# report — generate all figures
# ---------------------------------------------------------------------------

def command_report(args: argparse.Namespace) -> None:
    """Generate figures for all sub-experiments."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        sys.exit("ERROR: matplotlib not installed. pip install matplotlib")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_trials()

    # ── Figure 0: 7A behavioral model comparison ──
    model_fits_path = OUT_BASE / "exp7a" / "model_fits.json"
    if model_fits_path.exists():
        mf = json.loads(model_fits_path.read_text())
        tot      = mf["totals"]
        rw_fits  = mf["rw_fits"]
        per_trial = mf["per_trial"]

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle("Exp 7A — Behavioral Model Comparison\n(RW vs Recency vs WSLS)",
                     fontsize=13, fontweight="bold")

        # Panel 1: total BIC per model
        ax = axes[0, 0]
        bic_vals = [tot["rw"]["bic"], tot["recency"]["bic"], tot["wsls"]["bic"]]
        bars = ax.bar(["RW", "Recency", "WSLS"], bic_vals,
                      color=["#2196F3", "#FF9800", "#9E9E9E"])
        ax.set_ylabel("Total BIC (lower = better)")
        ax.set_title(f"Model comparison (winner={mf.get('winner', '?')})")
        for b, v in zip(bars, bic_vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=9)

        # Panel 2: RW α histogram
        ax = axes[0, 1]
        alphas = [f["alpha"] for f in rw_fits]
        ax.hist(alphas, bins=20, color="#2196F3", alpha=0.8)
        ax.axvline(float(np.mean(alphas)), color="black", linestyle="--",
                   linewidth=1, label=f"mean={np.mean(alphas):.2f}")
        ax.set_xlabel("α (RW learning rate)")
        ax.set_ylabel("Episode count")
        ax.set_title("RW α distribution across episodes")
        ax.legend(fontsize=9)

        # Panel 3: mean P(A) actual vs fitted
        ax = axes[1, 0]
        # Build per-trial-in-episode arrays
        ep_counter: dict[str, int] = {}
        ep_T: dict[str, int] = {}
        for f in rw_fits:
            ep_T[f["episode_id"]] = f["T"]
        max_T = max(ep_T.values()) if ep_T else 30
        n_eps = len(ep_T)
        ep_id_to_row = {eid: i for i, eid in enumerate(sorted(ep_T.keys()))}
        actual = np.full((n_eps, max_T), np.nan)
        rw_pa  = np.full((n_eps, max_T), np.nan)
        rec_pa = np.full((n_eps, max_T), np.nan)
        for d in per_trial:
            eid = d["episode_id"]
            t_in_ep = ep_counter.get(eid, 0)
            row = ep_id_to_row[eid]
            if t_in_ep < max_T:
                actual[row, t_in_ep] = 1.0 if d["chosen_arm"] == "A" else 0.0
                rw_pa[row,  t_in_ep] = d["rw_P_A"]
                rec_pa[row, t_in_ep] = d["recency_P_A"]
            ep_counter[eid] = t_in_ep + 1

        trial_idx = np.arange(max_T)
        ax.plot(trial_idx, np.nanmean(actual, axis=0), color="black",
                linewidth=2, label="Actual P(A)")
        ax.plot(trial_idx, np.nanmean(rw_pa,  axis=0), color="#2196F3",
                linewidth=1.6, label="RW fitted")
        ax.plot(trial_idx, np.nanmean(rec_pa, axis=0), color="#FF9800",
                linewidth=1.6, label="Recency fitted")
        ax.set_xlabel("Trial within episode")
        ax.set_ylabel("P(A)")
        ax.set_title("Mean P(A) trajectory")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Panel 4: scatter RW δ_t vs q_stated_diff
        ax = axes[1, 1]
        trial_lookup = {(r["episode_id"], r["trial"]): r for r in rows}
        xs, ys = [], []
        for d in per_trial:
            r = trial_lookup.get((d["episode_id"], d["trial"]))
            if r is None:
                continue
            xs.append(d["rw_rpe"])
            ys.append(r["q_stated_A"] - r["q_stated_B"])
        xs, ys = np.array(xs), np.array(ys)
        ax.scatter(xs, ys, alpha=0.25, s=10, color="#2196F3")
        c = corr(xs, ys)
        ax.set_xlabel("RW δ_t (RPE)")
        ax.set_ylabel("q_stated_A − q_stated_B")
        ax.set_title(f"Trial-level RW RPE vs declared ΔQ   r={c:+.3f}")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.axvline(0, color="gray", linewidth=0.8)

        plt.tight_layout()
        fig.savefig(FIG_DIR / "exp7a_model_comparison.png", dpi=150)
        plt.close(fig)
        print(f"Saved {FIG_DIR / 'exp7a_model_comparison.png'}")

    # ── Figure 1: 7B probe results — layer curves for all targets × positions ──
    probe_path = OUT_BASE / "exp7b" / "probe_results.json"
    if probe_path.exists():
        probe_records = json.loads(probe_path.read_text())
        from collections import defaultdict
        by_pos_tgt: dict[tuple, dict[int, float]] = defaultdict(dict)
        for rec in probe_records:
            by_pos_tgt[(rec["position"], rec["target"])][rec["layer"]] = rec["r_test"]

        targets_main = ["reward_history", "delta_q", "rpe_proxy", "choice_tendency"]
        positions_all = sorted(set(p for p, _ in by_pos_tgt.keys()))
        colors = {"declare_a": "#2196F3", "declare_b": "#FF9800",
                  "choice": "#4CAF50", "feedback": "#9C27B0"}
        linestyles = {"declare_a": "-", "declare_b": "--", "choice": "-.", "feedback": ":"}

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Exp 7B — Layer-wise Probe R (test set)\nFour Computational Variables × Positions",
                     fontsize=13, fontweight="bold")

        for ax, tname in zip(axes.flat, targets_main):
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            for pos in positions_all:
                key = (pos, tname)
                if key not in by_pos_tgt:
                    continue
                layer_r = by_pos_tgt[key]
                layers  = sorted(layer_r.keys())
                rs      = [layer_r[l] for l in layers]
                col = colors.get(pos, "black")
                ls  = linestyles.get(pos, "-")
                ax.plot(layers, rs, label=pos, color=col, linestyle=ls, linewidth=1.8)
            ax.set_title(tname.replace("_", " ").title(), fontsize=11)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Pearson r (test)")
            ax.legend(fontsize=8)
            ax.set_xlim(0, N_LAYERS - 1)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(FIG_DIR / "exp7b_probe_curves.png", dpi=150)
        plt.close(fig)
        print(f"Saved {FIG_DIR / 'exp7b_probe_curves.png'}")

    # ── Figure 2: 7C — attention vs MLP contribution ──
    comp_path = OUT_BASE / "exp7c" / "component_analysis.json"
    if comp_path.exists():
        comp_records = json.loads(comp_path.read_text())
        targets_c = ["reward_history", "delta_q", "rpe_proxy", "choice_tendency"]

        from collections import defaultdict
        by_tgt_layer: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(dict))
        for rec in comp_records:
            by_tgt_layer[rec["target"]][rec["layer"]] = rec

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Exp 7C — Attention vs MLP Contribution\n(probe r on Δattn vs ΔMLP activations)",
                     fontsize=13, fontweight="bold")

        for ax, tname in zip(axes.flat, targets_c):
            if tname not in by_tgt_layer:
                continue
            layer_data = by_tgt_layer[tname]
            layers = sorted(layer_data.keys())
            r_attn = [layer_data[l].get("r_attn_contribution", float("nan")) or 0 for l in layers]
            r_mlp  = [layer_data[l].get("r_mlp_contribution",  float("nan")) or 0 for l in layers]
            r_full = [layer_data[l].get("r_full_block",         float("nan")) or 0 for l in layers]

            ax.plot(layers, r_attn, label="Attn contribution", color="#2196F3", linewidth=1.8)
            ax.plot(layers, r_mlp,  label="MLP contribution",  color="#F44336", linewidth=1.8)
            ax.plot(layers, r_full, label="Full block",         color="#333333", linewidth=1.4,
                    linestyle="--", alpha=0.7)
            ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
            ax.set_title(tname.replace("_", " ").title(), fontsize=11)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Pearson r (test)")
            ax.legend(fontsize=8)
            ax.set_xlim(0, N_BLOCKS - 1)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(FIG_DIR / "exp7c_component_analysis.png", dpi=150)
        plt.close(fig)
        print(f"Saved {FIG_DIR / 'exp7c_component_analysis.png'}")

    # ── Figure 3: 7D — intervention results ──
    intv_path = OUT_BASE / "exp7d" / "intervention_results.json"
    intv_summ_path = OUT_BASE / "exp7d" / "intervention_summary.json"
    if intv_path.exists() and intv_summ_path.exists():
        intv_results = json.loads(intv_path.read_text())
        intv_summary = json.loads(intv_summ_path.read_text())

        delta_q_arr   = np.array([r["delta_q"]     for r in intv_results])
        delta_pos_arr = np.array([r["delta_pa_pos"] for r in intv_results])
        delta_neg_arr = np.array([r["delta_pa_neg"] for r in intv_results])
        delta_rnd_arr = np.array([r["delta_pa_rand"] for r in intv_results])

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f"Exp 7D — Causal Intervention\n"
            f"target={intv_summary['target']}  position={intv_summary['position']}  "
            f"layer={intv_summary['layer']}  α={intv_summary['alpha']}",
            fontsize=12, fontweight="bold"
        )

        # Subplot A: ΔP(A) distribution for pos/neg/random
        ax = axes[0]
        ax.hist(delta_pos_arr, bins=30, alpha=0.6, label="+ direction", color="#4CAF50")
        ax.hist(delta_neg_arr, bins=30, alpha=0.6, label="− direction", color="#F44336")
        ax.hist(delta_rnd_arr, bins=30, alpha=0.6, label="random ctrl", color="#9E9E9E")
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("ΔP(A)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of ΔP(A) per condition")
        ax.legend(fontsize=9)

        # Subplot B: bar chart of mean ΔP(A)
        ax = axes[1]
        means  = [intv_summary["mean_delta_pa_positive_intervention"],
                  intv_summary["mean_delta_pa_negative_intervention"],
                  intv_summary["mean_delta_pa_random_control"]]
        stds   = [intv_summary["std_delta_pa_positive"],
                  intv_summary["std_delta_pa_negative"],
                  intv_summary["std_delta_pa_random"]]
        labels = ["+α·v", "−α·v", "rand"]
        colors_b = ["#4CAF50", "#F44336", "#9E9E9E"]
        bars = ax.bar(labels, means, color=colors_b, alpha=0.8)
        ax.errorbar(range(len(labels)), means, yerr=stds,
                    fmt="none", capsize=5, color="black", linewidth=1.5)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_ylabel("Mean ΔP(A)")
        ax.set_title("Mean causal effect per condition")
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, m + 0.001 * np.sign(m),
                    f"{m:+.4f}", ha="center", fontsize=9)

        # Subplot C: scatter ΔQ vs ΔP(A) for positive intervention
        ax = axes[2]
        ax.scatter(delta_q_arr, delta_pos_arr, alpha=0.4, s=20, color="#2196F3")
        c = corr(delta_q_arr, delta_pos_arr)
        ax.set_xlabel("δQ = q_stated_A − q_stated_B")
        ax.set_ylabel("ΔP(A) after +α·v")
        ax.set_title(f"ΔQ vs ΔP(A)  r={c:+.3f}")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.axvline(0, color="gray", linewidth=0.8)

        plt.tight_layout()
        fig.savefig(FIG_DIR / "exp7d_intervention.png", dpi=150)
        plt.close(fig)
        print(f"Saved {FIG_DIR / 'exp7d_intervention.png'}")

    # ── Figure 4: 7E — discovery results ──
    disc_path  = OUT_BASE / "exp7e" / "discovery_results.json"
    ahead_path = OUT_BASE / "exp7e" / "ahead_prediction.json"
    if disc_path.exists():
        disc_results  = json.loads(disc_path.read_text())
        ahead_results = json.loads(ahead_path.read_text()) if ahead_path.exists() else []

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Exp 7E — Data-Driven Discovery\nPCA States → Future Choice Prediction",
                     fontsize=12, fontweight="bold")

        # AUC across layers
        ax = axes[0]
        layers  = [r["layer"] for r in disc_results]
        aucs    = [r["auc_next_choice"] for r in disc_results]
        shuf    = [r["auc_shuffled_baseline"] for r in disc_results]
        ax.plot(layers, aucs, "o-", color="#2196F3", label="PCA features", linewidth=2)
        ax.plot(layers, shuf, "s--", color="#9E9E9E", label="shuffle baseline", linewidth=1.5)
        ax.axhline(0.5, color="black", linewidth=0.8, linestyle=":")
        ax.set_xlabel("Layer")
        ax.set_ylabel("AUC (next-trial choice)")
        ax.set_title("Prediction AUC by layer")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # AUC decay over k-ahead
        ax = axes[1]
        if ahead_results:
            ks   = [r["k_ahead"] for r in ahead_results]
            aucs_k = [r["auc"] for r in ahead_results]
            ax.plot(ks, aucs_k, "o-", color="#FF9800", linewidth=2)
            ax.axhline(0.5, color="black", linewidth=0.8, linestyle=":")
            ax.set_xlabel("Prediction horizon (trials ahead)")
            ax.set_ylabel("AUC")
            ax.set_title("Temporal decay of predictability")
            ax.set_xticks(ks)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "ahead_prediction.json not found",
                    ha="center", va="center", transform=ax.transAxes)

        plt.tight_layout()
        fig.savefig(FIG_DIR / "exp7e_discovery.png", dpi=150)
        plt.close(fig)
        print(f"Saved {FIG_DIR / 'exp7e_discovery.png'}")

    # ── Figure 5: combined summary heatmap ──
    if probe_path.exists():
        probe_records = json.loads(probe_path.read_text())
        from collections import defaultdict

        # Build matrix: target × layer for 'choice' position
        by_tl: dict[str, dict[int, float]] = defaultdict(dict)
        for rec in probe_records:
            if rec["position"] == "choice":
                by_tl[rec["target"]][rec["layer"]] = rec["r_test"]

        targets_h = ["reward_history", "recent_reward", "delta_q",
                     "rpe_proxy", "choice_tendency", "post_reversal"]
        targets_h = [t for t in targets_h if t in by_tl]
        if targets_h:
            mat = np.zeros((len(targets_h), N_LAYERS))
            for ti, tname in enumerate(targets_h):
                for l in range(N_LAYERS):
                    mat[ti, l] = by_tl[tname].get(l, float("nan")) or 0.0

            fig, ax = plt.subplots(figsize=(14, 5))
            im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                           origin="upper")
            plt.colorbar(im, ax=ax, label="Pearson r (test)")
            ax.set_xticks(range(N_LAYERS))
            ax.set_xticklabels(range(N_LAYERS), fontsize=7)
            ax.set_yticks(range(len(targets_h)))
            ax.set_yticklabels([t.replace("_", " ") for t in targets_h], fontsize=10)
            ax.set_xlabel("Layer")
            ax.set_title("Exp 7B — Probe R Heatmap (choice position)", fontsize=12,
                         fontweight="bold")
            plt.tight_layout()
            fig.savefig(FIG_DIR / "exp7b_probe_heatmap.png", dpi=150)
            plt.close(fig)
            print(f"Saved {FIG_DIR / 'exp7b_probe_heatmap.png'}")

    # ── Summary JSON ──
    summary_out = {
        "exp7a_available": model_fits_path.exists(),
        "exp7b_available": probe_path.exists(),
        "exp7c_available": comp_path.exists() if "comp_path" in dir() else False,
        "exp7d_available": intv_path.exists(),
        "exp7e_available": disc_path.exists(),
        "figures_dir": str(FIG_DIR),
    }
    if model_fits_path.exists():
        mf = json.loads(model_fits_path.read_text())
        summary_out["exp7a_summary"] = {
            "totals": mf.get("totals"),
            "winner": mf.get("winner"),
        }
    if intv_summ_path.exists():
        summary_out["exp7d_summary"] = json.loads(intv_summ_path.read_text())

    dump(FIG_DIR / "report_summary.json", summary_out)
    print(f"\nReport complete. Figures at {FIG_DIR}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model",  default=DEFAULT_MODEL)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype",  default="auto",
                   choices=["auto", "float32", "float16", "bfloat16"])


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = root.add_subparsers(dest="command", required=True)

    # fit_models (behavioral, Stage 7A)
    p = sub.add_parser("fit_models",
                       help="7A behavioral: fit RW, Recency, WSLS models to choices")
    p.set_defaults(func=command_fit_models)

    # collect_t2
    p = sub.add_parser("collect_t2", help="7A: extract feedback-position activations")
    _add_model_args(p)
    p.set_defaults(func=command_collect_t2)

    # collect_comps
    p = sub.add_parser("collect_comps", help="7C: extract x_mid via post_attention_layernorm hooks")
    _add_model_args(p)
    p.set_defaults(func=command_collect_comps)

    # probe_7b
    p = sub.add_parser("probe_7b", help="7B: layer-wise probing of four computational variables")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=command_probe_7b)

    # analyze_7c
    p = sub.add_parser("analyze_7c", help="7C: attention vs MLP contribution analysis")
    p.set_defaults(func=command_analyze_7c)

    # intervene_7d
    p = sub.add_parser("intervene_7d", help="7D: causal activation steering")
    _add_model_args(p)
    p.add_argument("--intervention-target",   default="choice_tendency",
                   choices=["reward_history", "delta_q", "rpe_proxy", "choice_tendency"])
    p.add_argument("--intervention-position", default="choice",
                   choices=["declare_a", "declare_b", "choice", "feedback"])
    p.add_argument("--alpha",  type=float, default=30.0,
                   help="Steering coefficient (scale of activation std units)")
    p.add_argument("--n-test", type=int,   default=150,
                   help="Number of held-out trials to steer")
    p.add_argument("--seed",   type=int,   default=42)
    p.set_defaults(func=command_intervene_7d)

    # discover_7e
    p = sub.add_parser("discover_7e", help="7E: data-driven PCA state discovery")
    p.add_argument("--n-components", type=int, default=20,
                   help="Number of PCA components")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=command_discover_7e)

    # report
    p = sub.add_parser("report", help="Generate all figures and summary")
    p.set_defaults(func=command_report)

    return root


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
