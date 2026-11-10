# Investigating Decision-Making in Large Language Models

This repository contains experiments on how large language models process reward information and make decisions in structured tasks. All experiments use Qwen3 models on variants of the two-armed reversal bandit. No model weights are updated — the work studies inference-time, in-context computation only.

---

## Research Overview

A core question in LLM interpretability is whether a model's internal representations causally drive its behaviour, or merely correlate with it. These experiments approach that question from two directions:

**Experiment 5** asks whether providing a model with its own verbal reflections on past trials improves reward-sensitive choice. It tests whether in-context memory can bridge the gap between what the model knows and what it does.

**Experiment 7** asks what information is encoded in Qwen3-0.6B's hidden layers during the same task, and whether directly modifying a hidden layer changes the model's output in a predictable direction.

Together, the experiments characterise a failure mode: the model encodes decision-relevant information in its activations, verbalises correct diagnoses of its situation, but does not translate either into adaptive behaviour.

---

## Experiments

| Experiment | Model(s) | Method | Key finding |
|------------|---------|--------|-------------|
| [Experiment 5](Experiment_5/README.md) | Qwen3-0.6B, Qwen3-1.7B | Reflexion-assisted sequential bandit | Both models fail the reversal gate. Reflection generates a self-reinforcing B-collapse. |
| [Experiment 7](Experiment_7/README.md) | Qwen3-0.6B | Linear probing + DAS | Layer 27 encodes choice tendency with r = 0.969. Pre-registered causal intervention confirms directional sensitivity. |

---

## Repository Structure

```
llm_experiment/
├── README.md                        ← this file
├── .gitignore
│
├── Experiment_5/                    ← Reflexion-assisted bandit (Exp 5)
│   ├── README.md
│   ├── exp5_sequential_reflexion.py
│   ├── sequential_bandit_rpe.py
│   ├── causal_rpe_bandit.py
│   ├── Qwen3_0.6B/
│   │   └── behavior_summary.json
│   └── Qwen3_1.7B_20x20/
│       └── behavior_summary.json
│
├── Experiment_7/                    ← Linear probing and DAS (Exp 7)
│   ├── README.md
│   ├── requirements.txt
│   ├── src/
│   ├── validation/
│   ├── results/
│   ├── tests/
│   └── docs/
│
└── experiment_summaries/            ← Cross-experiment result summaries
```

---

## Task

Both experiments use the same two-armed reversal bandit:

- Arms A and B with reward probabilities 0.80 (good) / 0.20 (bad)
- Reversal at approximately trial 15 of each episode
- The model is never told reward probabilities or that a reversal will occur
- Model weights are never updated; all adaptation must occur in context

---

## Claim Boundary

This work studies inference-time computation only. A linearly decodable direction in activations is evidence that information is present — not that the model uses it to make its choice. The DAS result shows that a 16-dimensional subspace at Layer 27 is causally sensitive to interventions; it does not identify this subspace as the sole or necessary mediator of decisions. These results do not claim LLMs learn like biological agents or use dopamine-like mechanisms.
