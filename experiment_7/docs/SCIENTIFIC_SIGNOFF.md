# Experiment 7 — Final Scientific Sign-Off

**Sign-off author:** Claude (claude-sonnet-4-6), independent reviewer  
**Date:** 2026-11-07  
**Basis:** EXP7_COMPLETE_RESEARCH_AND_AUDIT_SYNTHESIS.md; VALIDATION_EXP7_V2/VALIDATION_RESULTS.md; AUDIT_CODEX/EXP7_SCIENTIFIC_AUDIT.md; AUDIT_CODEX/SECOND_PASS_VERIFICATION.md; AUDIT_CODEX/CLAUDE_CROSSREVIEW.md; independent numerical verification of all saved records.

**File status:**  
- `REPORTS/EXP7_COMPLETE_RESEARCH_AND_AUDIT_SYNTHESIS.md` — ✅ read  
- `AUDIT_CODEX/EXP7_FINAL_AUDIT_CLOSURE.md` — ❌ **file does not exist**; sign-off proceeds from the synthesis and the two audit documents, which are sufficient  
- `VALIDATION_EXP7_V2/VALIDATION_RESULTS.md` — ✅ read; all statistics independently recomputed from `records.jsonl`

---

## Question 1 — What has Experiment 7 actually demonstrated?

**Three things, with clearly different evidentiary weights.**

### (a) Decodability (high confidence)

A linear probe trained on L27 hidden states predicts the model's own calibrated A-minus-B output logit margin with Pearson r = 0.969 on a held-out episode split. This is a strong and reproducible representational result. Its correct interpretation is narrow: the calibrated output margin is linearly accessible from L27 activations in this prompt family. Because the probe target is derived from the same model's output, the high correlation partly reflects proximity to the output readout; it does not independently establish that L27 encodes a task-relevant variable in any sense stronger than "the model's output preference is predictable from its penultimate layer."

### (b) Intervention sensitivity (moderate confidence, V2-confirmed)

Replacing the L27 residual stream component along a frozen 16-dimensional trained subspace changes the A-B output logit margin more than 10 norm-matched random 16-dimensional subspace interventions. In Validation V2 on 20 independently generated episodes, the primary cluster-mean paired difference is 0.112 logits (95% percentile bootstrap CI [0.078, 0.151], ten disjoint episode-pair clusters, all ten positive). Directional agreement is 76.0% for DAS versus 50.9% for matched random controls. These results hold on data not used to train the basis, with model activations computed live at float32, norms explicitly matched, and propagation verified. This is a real and controlled causal finding at the level of model outputs.

### (c) Behavioral control (not demonstrated)

Restricted A-B argmax flips are zero in both the original steering experiments (0/100) and Validation V2 (0/200 DAS, 0/2000 random). Calibrated threshold crossings in V2 are 1/200 for DAS and 7/2000 for random — not meaningfully different. The model's choice readout is not changed by any tested intervention. The original 83% calibrated-flip figure describes threshold crossings in a different intervention design (probe-projected direction at 5% residual norm) on training-contaminated episodes; it has no V2 counterpart and cannot be treated as a replication of behavioral control.

---

## Question 2 — Which original claims were weakened or rejected?

**Four claims require explicit correction in any reporting.**

### (i) "Full-forward-pass IIA = 0.995 (gold-standard)" — rejected as stated

The 0.995 figure is a directional sign agreement rate, not standard interchange intervention accuracy. Standard IIA requires a predefined high-level causal model and comparison of the intervened output to its counterfactual prediction; neither is present. Additionally, the validation episodes used to compute 0.995 have 9/10 overlap with DAS training episodes under current-code reconstruction — they do not constitute an independent evaluation. The number is arithmetically correct from the saved file but is scientifically misdescribed in the original reports.

### (ii) "83% behavioral flip rate" — weakened to threshold crossing in a contaminated sample

The 83/100 figure is a calibrated-margin zero crossing in a steering experiment where the 100 trials come from four episode clusters, all within DAS training episodes. It is not a generated-choice flip. V2 found 1/200 calibrated crossings under a different but comparably controlled intervention design. The two figures measure related but non-identical estimands and should not be directly compared; neither establishes generated behavioral control.

### (iii) "Letter prior is a pure token-level prior, content-insensitive" — not established

The A/B label swap examined only the declared value labels; 84/100 test rows had identical labels making the swap a no-op. The historical arm labels in the prompt history were not remapped. The +4.02 mean calibration offset is real and nearly constant (std 0.067), but its attribution to tokenization or pretraining rather than to the specific baseline calibration design is not supported by the available evidence.

### (iv) "Mechanism confirmed; L27 choice tendency drives B-collapse" — rejected

This language (from `results/exp7_FINAL_SUMMARY.txt`) exceeds every piece of evidence in this study. Neither necessity, sufficiency, nor unique mediation has been tested. The intervention sensitivity result establishes that this subspace is causally connected to the output margin; it does not establish that it drives B-collapse, implements value computation, or constitutes the model's decision mechanism.

---

## Question 3 — What is the strongest defensible causal conclusion from Validation V2?

> **In Qwen3-0.6B, on the recorded bandit prompt template and episode distribution, replacing the L27 residual stream component along the frozen trained 16-dimensional DAS subspace — using the difference between source and base live activations, with the intervention norm matched to ten fixed random subspace controls — produces a reliably larger mean absolute A-B output logit margin change than those controls. The primary paired difference is 0.112 logits (95% CI [0.078, 0.151]), positive in all ten episode-pair clusters, with 76% directional agreement versus 51% for matched random interventions.**

This is causal evidence at the level of model outputs: the trained subspace is more causally sensitive than same-norm random subspaces at this layer and intervention type. It is not standard IIA, it does not control behavior, and it does not establish what computation the subspace performs. The directional metric is derived from the model's own output margin and does not compare against an independently specified high-level model prediction.

The honest strength of this conclusion is Pearl's Level 2 (interventional), limited to output-margin effects, within the tested model snapshot, layer, prompt family, and episode distribution.

---

## Question 4 — What remains unknown about the model's internal decision-making mechanism?

**Five substantive open questions, in approximate order of scientific importance.**

1. **Causal role of L27.** Intervention sensitivity shows the subspace is connected to the output; it does not show it is necessary. No ablation, lesion, or circuit-level analysis has established that removing or corrupting L27 impairs choice behavior. The relationship between early reward-history encoding (L5–L7, r ≈ 0.94) and late-layer choice tendency (L27) is undescribed causally.

2. **What the trained subspace represents.** The DAS probe target is the model's own calibrated output margin — a downstream quantity. Whether the trained subspace encodes a computational intermediate (a value estimate, an action policy) or simply correlates with the output readout because of proximity cannot be resolved from these data. The read-out direction (probe) and intervention direction (DAS) are nearly orthogonal, which is consistent with either interpretation.

3. **Generalization.** All results are from one model snapshot (revision c1899de), one prompt template, and one parameterization of the two-armed bandit. No cross-model, cross-task, or cross-template evidence exists.

4. **The letter prior.** The +4.02 calibration offset preventing generated-choice control has not been cleanly attributed. This matters scientifically because the baseline calibration design and any tokenization or pretraining effects are conflated. Whether a task redesign (e.g. randomized A/B labeling) would yield generated behavioral control is an open empirical question.

5. **Intrinsic dimensionality and uniqueness.** The 16-dimensional subspace is a design choice, not an estimate of the mechanism's intrinsic dimension. Whether a smaller or differently oriented subspace would yield the same or stronger effect, and whether the learned subspace is the unique location of this sensitivity, are unknown.

---

## Question 5 — Is the research ready to be reported to the PI or research group?

### READY WITH QUALIFICATIONS

**What can be reported without qualification:**
- L27 linear probe r = 0.969 (probe target is model's calibrated output margin; layer selected on test performance)
- Validation V2: DAS subspace produces larger absolute output-margin shifts than 10 norm-matched random subspaces on 20 new episodes, primary difference 0.112 logits CI [0.078, 0.151], all 10 clusters positive, 0 behavioral flips
- Intervention propagation technically verified (float32, next-block input confirmed)
- 0% restricted A-B argmax flip in both original and V2

**What must be explicitly corrected before or during reporting:**
- The 0.995 figure must be described as directional sign agreement on training-contaminated episodes, not as independent IIA
- The 83% figure must be described as a calibrated threshold crossing in a training-contaminated sample, not as behavioral control; V2 found near-zero calibrated crossings
- The letter prior claim must be qualified: the offset is real and stable, its mechanistic attribution is not established
- "Mechanism confirmed" language must be removed

---

## Decision

**READY WITH QUALIFICATIONS**

The core empirical finding — that the frozen DAS subspace is causally more sensitive than norm-matched random subspaces at the output-margin level, confirmed on new episodes with matched norms and verified propagation — is real, reproducible, and worth reporting. However, three original claims (IIA 0.995, 83% flip, mechanism confirmed) require explicit correction before they appear in any PI-facing or group-facing document, because the gap between what those claims say and what the evidence supports is large enough to mislead a reader. The letter prior limitation must be stated as an open question, not a resolved finding. With those corrections made clearly, the research represents a methodologically honest contribution: a documented example of output-margin intervention sensitivity with a negative behavioral control result and a transparent audit trail.

---

*Sign-off complete. No further experiments or validation phases are proposed.*
