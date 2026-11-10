# Research Update — Supervisor Brief

**Project:** Mechanistic interpretability of Qwen3-0.6B in a repeated-choice task  
**Prepared for:** Lili  
**Date:** 2026-11-04 (updated after validation)
**Status:** Core pipeline + independent validation complete

---

## Research Question

Can we identify and causally test internal representations associated with
decision-making in a language model performing a repeated-choice task?

We distinguish four levels of evidence, which are not interchangeable:

1. **Behavioral pattern** — what choices the model makes
2. **Decodable representation** — what information can be read out from hidden states
3. **Causally aligned subspace** — whether a subspace supports causal interchange interventions
4. **Controlled behavioral effect** — whether intervening on that subspace reliably changes choices

---

## Experimental Summary

### 1. Behavioral Characterization (Exp 7A)

Qwen3-0.6B was evaluated on 1,500 trials of a two-armed reversal bandit (50 episodes × 30 trials).
Three behavioral models were fit to its choice sequences.

**Findings:**
- Rescorla-Wagner (RW) and Recency models fit the data equivalently (ΔBIC = 0.047).
- Estimated learning rate α ≈ 0.019 (slow updater).
- Win-Stay-Lose-Shift performs substantially worse (ΔBIC = +149).

**Interpretation:** The model's aggregate behavior is consistent with slow value learning.
RW and Recency cannot be distinguished behaviorally at this sample size.

---

### 2. Representation Analysis (Exp 7B)

Layer-wise linear probes were trained to decode task-relevant variables from
hidden states at each Transformer layer.

**Findings:**

| Variable | Best layer | r (test) |
|----------|-----------|---------|
| choice_tendency | L27 | 0.97 |
| reward_history | L5–L7 | 0.94–0.95 |
| delta_q | L14 | 0.67 |
| rpe_proxy | — | ~0.13 (unreliable) |

**Interpretation:** Decision-relevant information is decodable, with the
strongest choice-tendency signal concentrated at layer 27 (of 28).

**Important:** Decodability does not establish that these representations
causally influence choices. They are candidate locations for causal testing.

---

### 3. Causal Subspace Analysis (Exp 7H)

Distributed Alignment Search (DAS) was applied to identify a subspace at L27
whose interchange interventions produce outputs consistent with a predefined
high-level variable (choice_tendency) being directly intervened upon.

**Findings:**
- IIA (Interchange Intervention Accuracy) = **0.992** on validation set;
  **0.998** on independent held-out trials.
- Comparison: IIA for delta_q at the same layer = 0.63 (lower specificity).

**Interpretation:** A 16-dimensional subspace at L27 supports interchange
interventions that are highly consistent with choice_tendency being causally
controlled. This is the strongest mechanistic lead from this study.

**New controls added in validation (2026-11-04):**

| Metric | DAS | Random (5 seeds, 16-dim) | Gap |
|--------|-----|--------------------------|-----|
| Probe-score IIA (n=500 pairs) | **0.990** | 0.524 ± 0.111 | 0.47 |
| **Full-forward-pass IIA (NEW, n=200 pairs)** | **0.995** | 0.633 ± 0.084 | **0.36** |

The random-subspace full-FP IIA is above chance (0.63 rather than 0.5) because the base–source hidden-state difference itself carries the choice_tendency signal (r=0.97 decodable from L27), so any random projection preserves some of it. The DAS–random gap of 0.36 absolute is still a large effect. **This validates the DAS subspace at the model-output level, not just probe-score level.**

**Current status:** ✅ **PASS** — this is the strongest mechanistic result of the project.

---

### 4. Directed Perturbation Analysis (Exp 7D)

Two sets of experiments were run at L27.

**4a. Sensitivity test (eps = 1%, original)**

Perturbations in three directions: DAS causal direction, orthogonal complement, random.

- DAS causal direction: **2.74×** the absolute logit-margin change of random (p < 4e-47, Wilcoxon)
- Orthogonal complement: 0.93× — no significant effect; confirms on-manifold specificity
- Bootstrap 95% CI for das/rand ratio: [2.57, 2.92]

At eps = 1%, choice flip rate = 0–0.9% (perturbation too small to cross the logit-margin gap).

**4b. Behavioral control test (eps = 5%, directed) — updated after validation**

| Condition | ΔM mean | Greedy flip | Calibrated flip | P(A) mean | A-sample % (n=30) |
|-----------|---------|-------------|-----------------|-----------|-------------------|
| Baseline | 0.000 | — | — | 0.9486 | 94.4% |
| A/B-swap baseline | 0.000 | — | — | 0.9485 | 94.5% |
| das_proj (+eps=5%) | +1.343 | **0.0%** | **83.0%** | 0.9860 | 98.6% |
| das_orth_probe | +0.301 | 0.0% | 4.0% | 0.9614 | 96.2% |
| random subspace (pooled, 3 seeds) | −0.156 | 0.0% | 0.0% | 0.9405 | 94.4% |

**Corrected interpretation:**

At eps=5%, das_proj produces a calibrated logit-margin threshold crossing in 83% of trials (37.86× more often than random 16-dim subspaces). This does not translate to a change in generated behaviour: greedy generation remains unchanged across all conditions (0% flip), and stochastic sampling shifts P(A) from 0.949 to 0.986 — a shift smaller than one standard deviation of sampling noise at n=30. The DAS direction is a specific intervention direction with a reliable directional effect on the output margin, but the overall logit distribution remains dominated by a +4.02 token-level prior favouring 'A'.

**Letter-prior audit:** The +4.02 raw-logit bias toward 'A' is a pure token-level prior, not content-driven. Swapping A/B labels in the prompt changes raw M by mean −0.0045 and P(A) by −0.0001. This prior prevents greedy-generation flips under any content-level activation intervention.

**4c. Random subspace control**

5 random 16-dim orthonormal subspaces (QR of Gaussian matrices) with matched projection norms:

| eps | das_proj calibrated flip | rand mean ± std | ratio |
|-----|--------------------------|-----------------|-------|
| 2% | 7.1% | 0.9% ± 0.2% | **7.62×** |
| 5% | 83.0% | 0.0% ± 0.0% | **∞** |

|ΔM| ratio (eps=5%): das_proj 1.343 / random_pooled 0.156 = **8.61×**.

The DAS subspace is not interchangeable with a random subspace of the same dimensionality.

---

## Overall Assessment

**What has been established (after validation, 2026-11-04):**
- Behavioral baseline: Qwen3-0.6B behaves like a slow value learner (α ≈ 0.019); RW tied with Recency (ΔBIC=0.047)
- Strong decodability of choice_tendency from L27 (r = 0.97)
- **DAS full-forward-pass IIA = 0.995** (DAS vs random 16-dim: 0.995 vs 0.633, gap = 0.36 absolute)
- Directional perturbation sensitivity at L27 (2.74× random at eps=1%; 8.61× at eps=5%)
- DAS direction is specific: 83% calibrated flip vs 0% for random subspaces at eps=5%
- **Letter prior (+4.02 raw logits for 'A') is content-insensitive** (A/B swap changes raw M by −0.0045)

**What has not been established:**
- Behavioral control at the generation level (greedy flip = 0% across all conditions)
- A complete, causally validated decision circuit
- Whether the DAS subspace is the unique or sufficient locus of choice computation
- Generalization beyond Qwen3-0.6B

**What was explored but not yet validated:**
- Sparse autoencoder features (Exp 7F): candidate features identified, causal role unknown
- Lagrangian minimal perturbation (Exp 7G): optimization converges (174/174 trials),
  but energy allocation does not directly quantify causal weight
- Direct Logit Attribution (Exp 7J): Block 27 MLP shows strong negative contribution;
  component-level causal role not yet established
- Activation patching (Exp 7J): analysis has unresolved corruption design and
  restoration-metric issues; results not currently interpretable

---

## Next Steps

**Immediate (before Exp 8):**
1. Discuss letter-prior limitation with Lili — recommend adding it as an explicit scope caveat in the paper.
2. Update slides (`REPORTS/EXP7_Presentation.pptx`) to reflect: (a) full-FP IIA = 0.995 as headline causal result; (b) behavioural flip is calibrated-margin-only, not generation-level.

**Exp 8 design:**
Consider a task variant that removes the letter prior — either constrained-logit decoding, or randomised A/B label assignment across trials — so that generation-level behavioural control can be directly tested.

**Optional:**
Cross-model replication: repeat 7B/7H on a second LLM (Qwen2.5-1.5B or Llama-3.2-1B) to test generalisation.

---

## What This Study Is

An exploratory mechanistic interpretability study combining behavioral fitting,
linear probing, DAS causal subspace analysis, and perturbation experiments.
It has produced a coherent set of preliminary evidence pointing toward L27 as
a candidate locus for choice-tendency representation and causal influence.

It has not yet produced a fully validated computational mechanism.

This is expected at this stage of the investigation.
