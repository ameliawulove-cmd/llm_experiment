# Experiment 7 Validation V2 — Confirmatory results

## Primary result

Mean paired difference in absolute A/B margin shift (DAS minus norm-matched random): **0.111947**.
95% percentile cluster bootstrap interval: **[0.077842, 0.150549]**.
Ten disjoint episode-pair clusters; 20 ordered pairs per cluster; ten fixed random bases per pair. The interval captures cluster resampling uncertainty, not random-basis sampling uncertainty.

## Secondary metrics

| Metric | DAS | Mean random |
|---|---:|---:|
| Absolute margin change | 0.137844 | 0.025896 |
| Signed margin change | 0.000011 | 0.000044 |
| Directional agreement | 0.760000 | 0.508500 |
| Restricted A/B flip | 0.000000 | 0.000000 |
| Calibrated threshold crossing | 0.005000 | 0.003500 |
| Intervention norm | 2.903969 | 2.903969 |

Equal-target pairs excluded from directional agreement: 0/200.
Maximum next-block propagation error: 1.525039e-05; maximum norm difference: 1.4045912e-06.

As an **exploratory, post-run secondary analysis**, the episode-pair-cluster mean DAS-minus-random directional-agreement difference is 0.2515, with a percentile bootstrap interval [0.1605, 0.3560] using 10,000 resamples and seed 8128. This interval uses the same ten clusters and excludes uncertainty in the ten fixed random bases; it was not the prespecified primary outcome. DAS agrees in 152/200 pairs; random controls agree in 1,017/2,000 pair-condition records, which are not independent observations. Restricted A/B flips are 0/200 for DAS and 0/2,000 control records; calibrated crossings are 1/200 and 7/2,000 respectively.

## Episode-pair robustness

| Cluster | DAS absolute | Random absolute | Paired difference | DAS directional | Random directional |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.24240 | 0.01900 | 0.22340 | 1.00000 | 0.61000 |
| 1 | 0.08813 | 0.02637 | 0.06176 | 0.80000 | 0.56000 |
| 2 | 0.18301 | 0.04360 | 0.13941 | 0.70000 | 0.60500 |
| 3 | 0.08620 | 0.02344 | 0.06276 | 0.80000 | 0.56000 |
| 4 | 0.17996 | 0.02634 | 0.15362 | 0.70000 | 0.54000 |
| 5 | 0.08339 | 0.01501 | 0.06839 | 0.70000 | 0.56000 |
| 6 | 0.11473 | 0.02661 | 0.08812 | 0.60000 | 0.50000 |
| 7 | 0.22617 | 0.02745 | 0.19872 | 1.00000 | 0.37000 |
| 8 | 0.08154 | 0.02127 | 0.06026 | 0.80000 | 0.41000 |
| 9 | 0.09290 | 0.02988 | 0.06302 | 0.50000 | 0.37000 |

## Provenance and limits

- Model: Qwen3-0.6B, revision c1899de289a04d12100db370d81485cdf75e47ca, mps:0, torch.float32; historical checkpoint identity is unverified.
- 4060 Qwen forwards; wall time 814.3 seconds.
- Exact historical overlaps: histories 102/600, reconstructed raw prompts 0/600; empty new histories 20.
- Among the 200 base trials selected for intervention, 16 have an exact history that occurs in the historical trial file. None of the 600 reconstructed raw choice prompts exactly matches a historical reconstructed raw prompt. The new episodes and exact reward schedules are separate under the checked candidate environment, but the histories and shared template are not all novel.
- Candidate historical environment matches observed rewards and reversals but its unchosen potential-outcome provenance is not uniquely established; episode seeds and exact schedule comparison cannot establish full distributional independence.
- Episode collection used exploration probability 0.1, matching the current historical collector's default; the actual historical invocation did not record its exploration setting, so exact behavioural-collection equivalence remains unverified.
- Directional agreement is an output-derived sign metric, not standard IIA. Matched controls are scaled interventions. Neither an output shift nor a significant contrast alone establishes a decision mechanism.
- Only ten independent episode-pair clusters and ten fixed random bases; cluster bootstrap may be imprecise, and random-basis variability is not included.
- Frozen protocol and hashes: FINAL_PROTOCOL.md and REPRODUCIBILITY_MANIFEST.json; record hash 5866c07689a582a223a1abfb25e44d179eef2058512b49d0aa0d0ca3f3cb25c0.
- Analysis script SHA-256: 74527db6849827acbb239ce4511bdef0e806b910efc9e5fee38d971a9d11c186.

## Reproducible analysis

~~~bash
PYTHONDONTWRITEBYTECODE=1 python3 VALIDATION_EXP7_V2/scripts/report_results.py
~~~

The script refuses to overwrite this existing report; preserve or relocate the report before rerunning. Its primary statistic is independently checked against the saved run summary before writing.
