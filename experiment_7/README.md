# Experiment 7 — What Information Does Qwen Store About Its Own Choices?

> **Model:** Qwen3-0.6B &nbsp;|&nbsp; **Task:** Two-armed reversal bandit &nbsp;|&nbsp; **Status:** Complete, pre-registered confirmatory validation

We ran a language model on a bandit task and asked two questions:

1. **Can we read decision-related information from its hidden layers?** (Linear Probing)
2. **If we directly modify a hidden layer, does the model's output change as expected?** (DAS)

These are different questions. The first is about correlation. The second is about causal sensitivity.

---

## The Task

Qwen3-0.6B plays a two-armed bandit. On each trial it sees a history of past choices and rewards for arm A and arm B, then picks an arm.

- 50 episodes, 30 trials each = **1,500 trials** in the training dataset
- Arms swap reward probabilities around trial 15 of each episode
- Reward probabilities: 0.8 (good arm) / 0.2 (bad arm)

---

## Method 1 — Linear Probing

### What it tests
Can a single linear function predict a specific variable from one layer's hidden state vector?

If yes → that information is **linearly decodable** from that layer.  
This tells us the information *exists* in the layer. It does not tell us the model *uses* it.

### Input → Process → Output

**Input**

For each of the 1,500 trials, we have:
- The model's hidden state at one transformer layer (1,024 numbers per trial)
- A target variable for that trial — e.g., the model's A−B logit difference

**Process**

We train a linear probe: find a 1,024-dimensional weight vector W such that  
`ŷ = W · h` predicts the target as closely as possible.

Steps:
1. Split 50 episodes into **35 training** (1,050 trials) and **15 test** (450 trials) — split by episode, so all 30 trials from one episode stay together
2. Subtract the training-set mean from all hidden states (mean-centring)
3. Fit W using Ordinary Least Squares (`np.linalg.lstsq`) — one analytic solve, no gradient descent
4. Apply W to the 450 test trials
5. Measure Pearson r between predicted and actual values on the test set

We repeat this for all 29 layers and 7 target variables.

**Train / test split in detail**

| Set | Episodes | Trials | Use |
|-----|----------|--------|-----|
| Training | 35 (randomly selected, seed=42) | 1,050 | Fit probe weights W |
| Test | 15 (held out, never seen during fitting) | 450 | Measure Pearson r |

No validation set is needed — OLS has no hyperparameters to tune.  
No preprocessing is applied to the test set before first fitting on the training set.

**Output**

One Pearson r value per (layer, target variable). The best layer for each variable.

### Verified Results

Source: `results/probing/probe_summary.json`, test set only (450 trials, 15 unseen episodes).

| Target variable | What it measures | Best layer | Pearson r |
|----------------|-----------------|-----------|-----------|
| **choice_tendency** | Qwen's calibrated A−B logit margin | **27** | **0.969** |
| reward_history | Cumulative mean reward so far | 5 | 0.938 |
| recent_reward | Mean reward over last 3 trials | 7 | 0.813 |
| delta_q (stated) | Model's stated Q(A) − Q(B) | 22 | 0.543 |
| post_reversal | Binary: before/after arm swap | 15 | 0.361 |
| rpe_proxy | Reward − Q(chosen) | 17 | 0.129 |

**What this means:** Layer 27 (the last transformer block output) encodes the model's upcoming choice tendency with r = 0.969 on held-out episodes. Early layers (5–7) encode reward history more strongly than choice tendency.

**What this does not mean:** Finding that a variable is decodable from a layer does not prove the model uses that layer to make its choice. It is a correlation result.

---

## Method 2 — DAS (Distributed Alignment Search)

### What it tests
If we directly replace part of a layer's hidden state, does the model's output change in the expected direction more than a random replacement would?

### Input → Process → Output

**Input**

- The same 1,500-trial activation matrix, one layer at a time
- Pairs of trials: a **base** trial and a **source** trial, with clearly different target values
- A probe direction W (fitted on the training set) that defines "correct direction"

**Process**

We learn a 16-dimensional subspace R (a 1,024×16 matrix with orthonormal columns).  
For a base/source pair, the intervention is:

```
h_intervened = h_base + R · R^T · (h_source − h_base)
```

This copies the component of (h_source − h_base) that lies within R's subspace into h_base.

Training steps:
1. Use the first 1,200 rows of data (episodes seq_0000–seq_0039) as training
2. Use the last 300 rows (episodes seq_0040–seq_0049) as validation
3. Sample 2,000 training pairs and 500 validation pairs
4. Train R to maximise directional agreement: the probe score should shift toward the source
5. Save the R with the highest validation directional agreement

Parameterisation: R is computed as the Q factor of a QR decomposition, so it is always exactly orthonormal during training.

**Train / validation / test split in detail**

| Set | Rows | Episodes | Use |
|-----|------|----------|-----|
| Training | 0–1199 | seq_0000–seq_0039 | Fit R |
| Validation | 1200–1499 | seq_0040–seq_0049 | Select best R checkpoint |
| Confirmatory test (V2) | 600 trials from 20 entirely new episodes | Never seen in any prior step | Measure causal effect |

The training and validation sets have **zero episode overlap** — verified by automated test.

The confirmatory test (V2) uses **20 completely new episodes** generated after the protocol was frozen. These are independent of both the training and validation sets.

**Output**

- A frozen R matrix: `results/das/R_choice_tendency_L27_k16.npy`, shape (1,024, 16)
- After V2 intervention: for each trial, we compare the A/B logit difference before and after replacing L27's last-token activation

### Verified Results

Source: `VALIDATION_EXP7_V2/results/confirmatory_run/summary.json` and `VALIDATION_RESULTS.md`.

**Primary result** (pre-specified outcome):

| Measure | DAS | Norm-matched random (mean of 10 bases) |
|---------|-----|---------------------------------------|
| Mean absolute A/B logit change | **0.138** | 0.026 |
| Paired difference (DAS − random) | **0.112** | — |
| 95% bootstrap CI | **[0.078, 0.151]** | — |
| Clusters with positive difference | **10/10** | — |

**Secondary results** (exploratory, not pre-specified):

| Measure | DAS | Random |
|---------|-----|--------|
| Directional agreement | **76.0%** (152/200) | 50.85% (1,017/2,000) |
| A/B restricted argmax flip | **0 / 200** | 0 / 2,000 |
| Calibrated threshold crossing | 1/200 | 7/2,000 |

**What this means:** The trained 16D subspace at Layer 27 produces significantly larger and more directionally consistent logit changes than equally sized random subspaces, on completely new episodes with a pre-registered protocol.

---

## What the Results Do Not Show

**No choice flip occurred.**  
Zero out of 200 DAS interventions changed which arm had the higher logit. The model has a persistent +3–4 logit bias toward arm A (from pretraining), which the 2.9-norm intervention cannot overcome.

**Probe r = 0.969 is not proof of a causal mechanism.**  
It shows that Layer 27 activations can predict the choice tendency. It does not show that Layer 27 *causes* the choice.

**DAS sensitivity ≠ DAS necessity.**  
The DAS result shows that the 16D subspace responds causally. It does not show this subspace is the only or necessary mediator of decisions.

**The model's decision algorithm is not identified.**  
Behaviour fits Rescorla-Wagner and Recency equally (ΔBIC = 0.05). We cannot distinguish them from behavioural data alone.

---

## Repository Structure

```
experiment_7/
├── README.md
├── requirements.txt
│
├── src/
│   ├── exp7_neuro.py         # linear probing (7B) and behaviour models (7A)
│   ├── exp7h_das.py          # DAS training (7H)
│   └── exp7d_dynamics.py     # activation perturbation dynamics (7D)
│
├── validation/
│   ├── FINAL_PROTOCOL.md     # pre-registered protocol (frozen before V2 run)
│   ├── REPRODUCIBILITY_MANIFEST.json
│   ├── README.md
│   └── scripts/
│       ├── confirm_v2.py     # V2 confirmatory experiment
│       ├── core.py           # metric definitions
│       └── model_adapter.py  # model loading utility
│
├── results/
│   ├── probing/
│   │   ├── probe_summary.json        # best layer and r per variable ← key result
│   │   └── probe_results.json        # full 29-layer profile
│   ├── das/
│   │   ├── R_choice_tendency_L27_k16.npy  # frozen DAS subspace (64 KB)
│   │   └── das_results.json               # training metrics
│   └── validation/
│       ├── VALIDATION_RESULTS.md     # V2 results summary ← key result
│       └── summary.json              # machine-readable V2 metrics
│
├── tests/
│   └── test_das_split_integrity.py   # 25 automated tests (all pass)
│
└── docs/
    ├── SCIENTIFIC_SIGNOFF.md
    ├── SUPERVISOR_BRIEF.md
    └── CODEBASE_MAP.md
```

---

## How to Run

### Requirements

```bash
pip install -r requirements.txt
# Key packages: torch, numpy, transformers, tqdm
```

### Verify saved results (no model needed)

```bash
python -m pytest tests/test_das_split_integrity.py -v
# 25/25 tests should pass
```

### Re-run linear probing from saved activations (no model needed)

Requires `data/exp6_activations/acts_choice.npy` (not included — ~180 MB per file).

```bash
python src/exp7_neuro.py probe_7b
```

### Re-run DAS training from saved activations (no model needed)

```bash
python src/exp7h_das.py train --only_vars choice_tendency --only_layers 27 --only_k 16
```

### Re-run V2 confirmatory experiment (requires Qwen3-0.6B + Apple MPS)

```bash
# Step 1: development test
python validation/scripts/confirm_v2.py --mode preflight

# Step 2: full confirmatory run (~14 minutes on Apple MPS)
python validation/scripts/confirm_v2.py --mode final
```

The confirmatory run will refuse to overwrite the existing `validation/results/confirmatory_run/` directory. The existing directory contains the original pre-registered results.

---

## Reproducibility at a Glance

| Component | Needs model? | Included? |
|-----------|-------------|-----------|
| Read and verify probe_summary.json | No | ✅ |
| Read and verify V2 summary.json | No | ✅ |
| Run the 25 automated tests | No | ✅ |
| Re-run linear probing (7B) | No | Needs activation files (not included) |
| Re-run DAS training (7H) | No | Needs activation files (not included) |
| Re-run V2 confirmatory | Yes (Qwen3-0.6B) | Needs model + ~14 min on MPS |

**Activation files** (`acts_choice.npy`, etc.) are ~180 MB each and are not included. The model must be downloaded from HuggingFace: `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`.

---

## A Note on Metric Terminology

The metric called `iia_val` in `results/das/das_results.json` is **directional agreement (DA)** — the fraction of held-out pairs where the probe score shifted in the correct direction. It is not standard Interchange Intervention Accuracy (IIA) as defined in causal representation learning, which requires specifying a full high-level causal model. We use "directional agreement" throughout to avoid this confusion.

