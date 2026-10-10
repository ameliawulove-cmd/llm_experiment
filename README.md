# Reflexion-Assisted Sequential Bandit — LLM Reward Learning

Can a language model learn to adapt its choices in a two-armed bandit task through trial-by-trial self-reflection? This experiment tests whether Reflexion — verbal self-reflection stored in episodic memory — enables reward-sensitive adaptation and reversal learning in Qwen3.

---

## Background

Prior experiments (not in this repo) established that RPE-like information is decodable from Qwen3 residual-stream activations but is not causally used to drive behaviour. Qwen3-0.6B and 1.7B both failed a sequential reversal bandit without reflection. This experiment adds open-ended self-reflection after each trial to test whether verbal memory bridges the gap.

---

## Task

- **Two-armed bandit** with arms A and B
- **50 episodes × 30 trials** per episode
- **Reversal** at mid-episode: the high-reward arm (p = 0.80) swaps with the low-reward arm (p = 0.20)
- The model is never told reward probabilities or that a reversal will occur
- After each trial the model generates a short reflection; all reflections are prepended to the next choice prompt
- Memory resets at each episode boundary

```
Trial t:
  choice_t  ←  model reads history + reflections[1..t-1]
  reward_t  ←  environment
  reflection_t  ←  same model, open-ended prompt
  reflections  ←  append reflection_t

Episode end: reflections = []
```

---

## Results — Qwen3-0.6B (Phase 1 complete)

### Qwen3-0.6B — 50 episodes × 30 trials

| Metric | Value | Threshold | Pass |
|--------|-------|-----------|------|
| RPE–update correlation (excl. trial 1) | 0.019 | ≥ 0.30 | ✗ |
| Signed update accuracy (excl. trial 1) | 48.3% | ≥ 60% | ✗ |
| Mean update after positive RPE | +0.00037 | > 0 | ✓ |
| Mean update after negative RPE | +0.00119 | < 0 | ✗ |
| **Gate passed** | **No** | | |

| Phase | Optimal accuracy | A-choice rate |
|-------|-----------------|---------------|
| Pre-reversal | 54.5% | 23.5% |
| Post-reversal | 46.5% | 8.0% |

Choice collapse: 84.2% B. No reversal adaptation.

---

### Qwen3-1.7B — 20 episodes × 20 trials

| Metric | Value | Threshold | Pass |
|--------|-------|-----------|------|
| RPE–update correlation (excl. trial 1) | 0.102 | ≥ 0.30 | ✗ |
| Signed update accuracy (excl. trial 1) | 51.8% | ≥ 60% | ✗ |
| Mean update after positive RPE | +0.0023 | > 0 | ✓ |
| Mean update after negative RPE | −0.0025 | < 0 | ✓ |
| **Gate passed** | **No** | | |

| Phase | Optimal accuracy | A-choice rate |
|-------|-----------------|---------------|
| Pre-reversal | 57.9% | 13.4% |
| Post-reversal | 37.9% | 5.6% |

Choice collapse: 90.5% B — stronger than 0.6B. Reversal adaptation absent; post-reversal accuracy (37.9%) falls below chance.

---

### Cross-model comparison

| | Qwen3-0.6B | Qwen3-1.7B |
|-|-----------|-----------|
| Scale | 50×30 | 20×20 |
| B-collapse | 84.2% | 90.5% |
| RPE–update corr (excl. T1) | 0.019 | 0.102 |
| Negative RPE → negative update | ✗ | ✓ |
| Post-reversal optimal accuracy | 46.5% | 37.9% |
| Gate passed | No | No |

1.7B shows marginal improvement in value-update directionality (RPE–update correlation higher; negative RPE updates now go in the correct direction) but produces a stronger B-collapse and worse post-reversal adaptation.

---

### Verbalization–behaviour gap

Both models generate correct verbal diagnoses that are not reflected in subsequent choices:

```
# Qwen3-0.6B — trials 16–18 (post-reversal)
Reflection: "B is negative, indicating it is less valuable than option A."
Next choice: B

# Qwen3-1.7B — trial 11 (pre-reversal)
Reflection: "choosing B may not be the optimal strategy in this particular trial."
Next choice: B
```

**Why B-collapse:** The reflection memory accumulates B-focused text (because B is already being chosen), creating a self-reinforcing loop. Choice-prior calibration removes the static positional A-token bias but cannot cancel in-context B-language introduced dynamically by the model's own reflections.

---

## Repository

```
├── exp5_sequential_reflexion.py   # Main experiment script
├── sequential_bandit_rpe.py       # Environment and measurement utilities (Exp 4, unchanged)
├── causal_rpe_bandit.py           # Shared utilities: load_model, seed_everything, chat template
├── experiment_summaries/
│   └── exp5_0.6b/
│       └── behavior_summary.json  # Gate metrics, reversal breakdown (0.6B)
└── results/                       # Trial logs — gitignored
```

---

## Setup

```bash
git clone https://github.com/ameliawulove-cmd/llm_experiment.git
cd llm_experiment

python -m venv .venv
source .venv/bin/activate
pip install torch transformers accelerate numpy
```

Tested on Apple Silicon (M4), Python 3.11, torch 2.14, transformers 5.17.

---

## How to Run

Generate environments (required before first run):

```bash
python sequential_bandit_rpe.py simulate \
    --episodes 50 --trials 30 \
    --output-dir results/sequential/environments
```

Run Phase 1 (behaviour only):

```bash
python exp5_sequential_reflexion.py behavior_reflexion \
    --model Qwen/Qwen3-0.6B \
    --environment-dir results/sequential/environments \
    --output-dir results/exp5/behavior_reflexion \
    --device mps --dtype float32
```

Run is resumable if interrupted. Completed episodes are detected automatically.

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen/Qwen3-1.7B` | HuggingFace model ID |
| `--device` | `auto` | `mps` for Apple Silicon |
| `--dtype` | `auto` | `float32` recommended for MPS |
| `--max-reflection-tokens` | `60` | Max tokens per reflection |
| `--discard-reflection` | off | Generate but do not store (ablation control) |
| `--epsilon` | `0.10` | Forced exploration probability |

---

## Claim Boundary

This project studies inference-time, in-context computation only. Model weights are never updated. A decodable direction in activations is evidence that information is present, not that it is causally used. These results do not claim LLMs learn like biological agents or use dopamine-like mechanisms.
