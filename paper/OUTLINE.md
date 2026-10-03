# Detect Different Lies: Paper Outline

## Title and central claim

**Working title:** *Detecting Different Lies: Near-Perfect White-Box Probe Performance Vanishes Under Matched Controls*

Central claim: in a matched factual-QA pilot, a standard residual-stream lie probe appears highly accurate when prompt context differs, but does not separate two kinds of false pressured reports once prompt regime and output falsity are controlled.

## Abstract (150--250 words)

- Motivate mechanism-specific monitoring: factual error is not the same construct as reporting against an available belief.
- Describe 500 evaluation and 200 disjoint probe-training TriviaQA questions, five neutral belief samples, neutral/pressure reports, and three detector/control scores.
- Report pressure yield (212/261; 81.2%, Wilson 95% CI 76.0--85.5%) and false-report composition (664 total; 31.9% operational lies).
- Contrast the context-confounded response AUROC (0.985), prompt-only leakage (1.000), matched false-vs-false AUROC (0.478), and false-vs-correct AUROCs (0.975 lie probe; 0.951 correctness probe).
- State the implication: high AUROC against honest-correct controls is insufficient evidence of a lie-specific representation.

## 1. Introduction

- Hook: a monitor that treats every false answer as a lie has poor construct validity.
- Explain the honesty/accuracy distinction and why matched false-vs-false comparisons matter.
- Gap: belief benchmarks, deception probes, and uncertainty detectors are usually evaluated separately.
- Approach and protocol overview; point to the design figure.
- Quantitative preview with the four central numbers above.
- Contributions: matched taxonomy; empirical mechanism shares; prompt-matched detector stress test; reproducible artifacts.
- Evidence: `results/summary.json`, `results/processed/detector_metrics.csv`, figures.
- Citations: MASK, Liars' Bench, Apollo linear probes, semantic uncertainty, recall-vs-truthfulness work.

## 2. Related Work

### Honesty, accuracy, and operational belief
- MASK and Liars' Bench; belief-verified model organisms.
- Position our same-question panel as a direct mechanism comparison, without mental-state claims.

### Activation-based lie and truth probes
- Internal-state truth classifiers; universal truth directions; Apollo; targeted and cross-type transfer studies.
- Emphasize prompt/type dependence and the role of controls.

### Hallucination and uncertainty detection
- Semantic entropy and recall/association-strength findings.
- Explain stable-wrong errors as a necessary third epistemic category.

## 3. Methodology

### Task and operational definitions
- Define question, aliases, five neutral samples, normalized modal answer, stability indicator, correctness, and entropy.
- Define stable-correct, stable-wrong, unstable, and operational lie.

### Data and matched protocol
- TriviaQA `rc.nocontext`; deterministic deduplication/filter/shuffle; 500 evaluation and next 200 training questions; no overlap.
- Neutral and pressure prompts; random incorrect decoy; deterministic reports.
- Include a design figure and a mechanism-label table.

### Detector baselines
- L2 logistic response-token probe, prompt-token shortcut control, hallucination/correctness probe, answer entropy.
- Training counts, grouped five-fold CV, candidate layers/C values, frozen choices.

### Statistics and implementation
- Question-level bootstrap (2,000 replicates), Wilson interval, exact McNemar, one-sided Mann--Whitney, Holm across 12 tests.
- Qwen revision, software, A6000, bfloat16, batch 32, seed 42, runtime.

## 4. Results

### Behavioral yield
- Strata: 261 stable-correct, 139 unstable, 100 stable-wrong.
- Pressure yield: 212/261; 181/212 match decoy; McNemar p = 3.04e-64.

### Composition of false reports
- Standalone table for five cells with counts, shares, bootstrap intervals.
- Mechanism-shares figure.
- Stress that shares are design-conditioned.

### Detector contrasts
- Standalone 12-row table with AUROC/CIs and interpretations; bold the primary matched comparison.
- Detector-distribution figure.
- Primary inference: matched false-vs-false lie probe is chance; correctness probe direction is not mechanism-specific.

### Robustness and error analysis
- Threshold sensitivity table (3/5, 4/5, 5/5).
- Alias-scoring correction and sampled qualitative cases.
- Hypothesis accounting: H1 supported under strong pressure; H2 unsupported; H3 circular manipulation check; H4 refuted in unconstrained contrast.

## 5. Discussion

- Interpret apparent probe success as prompt/output-correctness sensitivity, not direct proof of a belief-report mismatch.
- Separate induced goal conflict from spontaneous scheming.
- Discuss monitoring implications and why matched controls are necessary.
- Limitations: operational belief, artificial pressure, ambiguous pressured errors, one 7B model, probe provenance, hidden-state prompt integration, answer scoring, entropy circularity, contamination, deterministic reports, design-conditioned shares.
- Broader impacts and potential misuse: monitors can help audits but should not justify punitive or high-stakes decisions without validation.

## 6. Conclusion

- Restate matched design and main result.
- Key takeaway: near-perfect unconstrained AUROC can vanish under prompt-matched false-vs-false evaluation.
- Future work: natural incentives, post-pressure belief checks, blinded semantics, pressure-matched probe training, architecture-native released probes, multiple scales, independent uncertainty benchmarks.

## Tables and figures

1. **Figure 1:** protocol diagram (neutral belief elicitation -> strata -> neutral/pressure reports -> mechanism cells -> detectors).
2. **Figure 2:** mechanism shares among all 664 false reports (`figures/mechanism_shares.png`).
3. **Figure 3:** score distributions by comparison (`figures/detector_distributions.png`).
4. **Table 1:** operational mechanism taxonomy.
5. **Table 2:** mechanism counts/shares/CIs.
6. **Table 3:** detector AUROCs for all 12 preregistered contrasts.
7. **Table 4:** stability-threshold sensitivity.
8. **Appendix tables:** exact prompts, probe selection grid/implementation, reproducibility artifacts.

## Citation map

- Operational honesty/belief: Ren et al.; Kretschmar et al.; Cooney et al.
- Probe baselines and transfer limits: Goldowsky-Dill et al.; Natarajan et al.; Moustafa et al.; Luikham; Hopkins et al.; Azaria and Mitchell; Buerger et al.; Pacchiardi et al.; Thormann.
- Uncertainty and hallucination mechanisms: Kuhn et al.; Cheang et al.
- Data/model: Joshi et al. (TriviaQA); Qwen technical report/model card.
