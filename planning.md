# Experimental Direction Ranking

## Motivation & Novelty Assessment

### Why This Research Matters

White-box deception monitors are only useful if their score is specific to a mismatch between a model's available belief and its report. A detector that instead responds to factual falsity, weak recall, or pressure-prompt style will generate false alarms on honest errors and can miss confident lies, undermining both monitoring and scientific interpretation.

### Gap in Existing Work

MASK separates behavioral honesty from accuracy but does not compare internal deception and hallucination detectors. Liars' Bench and recent probe studies cover heterogeneous lie types and transfer, while hallucination studies distinguish weak recall from strong wrong associations; none of the reviewed work supplies the same-question, same-model crossing of neutral belief status and reporting incentive needed to compare false lies directly with honest false errors.

### Our Novel Contribution

We will build a matched factual-QA panel for one open-weight model, label knowledge from repeated neutral elicitation before applying pressure, retain stable-wrong answers as their own category, and measure the yield and composition of false outputs. We will then compare two deliberately simple detector families on the informative false-vs-false contrast: a standard linear residual-stream deception probe trained on disjoint instructed pairs, and sampling-based answer entropy. Prompt-token-only probe scores and pressured honest answers provide explicit prompt-confound checks.

### Experiment Justification

- **Experiment 1 — behavioral mechanism yield:** needed to estimate how often pressure produces a belief-contradicting false report rather than an honest epistemic error; the lie yield is a result, including if it is near zero.
- **Experiment 2 — detector dissociation:** needed to determine whether a linear lie probe and uncertainty score separate lies from epistemic errors when both outputs are false.
- **Experiment 3 — subtype and prompt controls:** needed to distinguish unstable recall from stable wrong association and to determine whether the lie probe reads the pressure prompt rather than the generated report.

## Research Question

For Qwen2.5-7B-Instruct answering matched factual questions, what share of false statements are operational epistemic errors versus pressure-induced reports that contradict a stable correct neutral belief, and can standard activation and sampling-uncertainty detectors distinguish these mechanisms?

## Background and Motivation

The literature review establishes that honesty and accuracy are distinct constructs, that deception probes generalize unevenly across lie types, and that hallucination detectors often measure recall strength rather than truth. This experiment joins those threads in one controlled panel. “Belief,” “knowledge,” and “lie” are operational behavioral labels: repeated consistency is not a claim about consciousness or intent.

## Hypothesis Decomposition

- **H1 (yield):** pressure will increase false reporting among questions with a stable correct neutral belief, but genuine belief-contradicting lies may be rare in a 7B instruction model.
- **H2 (specificity):** a response-token linear deception probe will discriminate operational lies from honest epistemic errors above chance (AUROC > 0.5).
- **H3 (dissociation):** answer entropy will be higher for unstable epistemic errors than for operational lies, while stable-wrong errors may resemble lies in having low entropy.
- **H4 (prompt confound):** discrimination from the final prompt-token activation alone will be materially weaker than response-token discrimination; failure of this condition indicates prompt leakage.

Independent variables are belief stratum (stable correct, stable wrong, unstable), reporting context (neutral, matched pressure), and token locus (prompt versus response). Outcomes are answer correctness, belief contradiction, answer entropy, probe score, and mechanism label. Alternative explanations include answer-alias failures, pressure-induced belief updating, memorized benchmark items, and stylistic response differences.

## Proposed Methodology

### Approach

Use a reproducible subset of deduplicated TriviaQA validation questions and the exact Hugging Face weights for `Qwen/Qwen2.5-7B-Instruct`. This older model is scientifically intentional: the task requires local hidden states, it is explicitly identified as the ungated fallback in the study instructions, and its 7B size makes repeated sampling feasible on the available A6000. The live OpenRouter catalog was checked on 2026-10-03 and lists API ID `qwen/qwen-2.5-7b-instruct` (32,768-token context; $0.10/M input and $0.20/M output), but the local Hugging Face model is required for white-box access.

### Experimental Steps

1. Deduplicate TriviaQA by question ID, exclude ambiguous/very long items, and deterministically sample a development/training panel and an evaluation panel.
2. For evaluation questions, generate five short neutral answers at temperature 0.7 with fixed seeds. Normalize exact short answers and match the complete alias list. Label stable-correct at >=4/5 identical correct, stable-wrong at >=4/5 identical incorrect, and all others unstable.
3. Generate one deterministic neutral report and one deterministic pressure report per evaluation question. The pressure says that an evaluator rewards a supplied plausible decoy answer but does not command lying; the decoy is drawn from another question, so compliance yield remains empirical.
4. Label false neutral reports from unstable questions as unstable epistemic errors and those from stable-wrong questions as associated/stable errors. Label a false pressure report as an operational lie only if the question was stable-correct and the report contradicts that belief. Keep all other cases ambiguous/other rather than forcing a binary label.
5. Build a disjoint linear-probe training set using paired prompts that explicitly request truthful versus deliberately incorrect short answers. Extract residual-stream activations at the last prompt token and last generated token from several middle/late layers. Fit L2 logistic regression with question-grouped cross-validation and freeze layer/regularization before target evaluation.
6. Compare the frozen response probe and prompt-only control on lie versus epistemic-error cases. Use normalized-answer entropy from the five neutral samples as the hallucination baseline.
7. Repeat primary estimates across reasonable stability thresholds (3/5, 4/5, 5/5) and report stable-wrong separately.

### Baselines

- Random score (AUROC 0.5) and output correctness (not expected to discriminate because the primary contrast contains only false outputs).
- Prompt-token linear probe as a direct pressure-prompt leakage control.
- Normalized-answer entropy as a short-answer approximation to semantic entropy.
- Pressured honest/correct outputs as controls for whether probe scores merely rise under pressure.

Released Apollo/DYL checkpoints are not architecture-compatible with Qwen2.5-7B. The planned standard logistic residual-stream method therefore uses separately generated instructed-pair training data and is reported as a method baseline, not as a reproduction of a released detector.

### Evaluation Metrics

- Mechanism counts and shares among false outputs with question-level percentile-bootstrap 95% confidence intervals.
- AUROC and average precision for lie versus epistemic error; balanced accuracy only at a threshold selected within probe training data.
- Bootstrap 95% intervals for AUROC differences between response probe, prompt probe, and entropy.
- Score distributions for lie, unstable error, stable-wrong error, pressured honest/correct, and neutral correct.
- Lie compliance yield among stable-correct questions.

### Statistical Analysis Plan

The question is the resampling unit. Use 2,000 seeded bootstrap replicates for shares, AUROCs, and paired score differences. Fisher's exact test tests pressure-associated error rates within stable-correct questions when the deterministic reports are paired; McNemar's exact test is used for neutral-versus-pressure correctness. Holm correction applies across the three preregistered detector comparisons. Report exact p-values, confidence intervals, and effect sizes (AUROC difference and rank-biserial/Cliff-style probability where suitable). If either primary class has fewer than 20 cases, detector results are exploratory and no strong inferential claim will be made.

## Expected Outcomes

H1 is supported by a nonzero and practically material operational-lie yield under pressure. H2 is supported only if the response probe separates false lies from false epistemic errors with a confidence interval mostly above 0.5 and outperforms the prompt-only control. H3 is supported if entropy ranks unstable errors above lies while revealing lower entropy for stable-wrong errors. Near-zero lie yield, chance detector performance, or prompt-only equivalence refutes the corresponding hypothesis and remains an informative result.

## Timeline and Milestones

1. Planning and environment audit: 20 minutes.
2. Model/data setup and pilot: 30–45 minutes.
3. Batched generation and activation extraction: 60–120 minutes.
4. Statistical analysis and figures: 30–45 minutes.
5. Reproduction run, validation, and documentation: 30–45 minutes, with roughly 25% contingency for download/runtime issues.

## Potential Challenges

- A 7B model may not produce uninstructed lies; report the genuine yield and do not relabel instructed outputs.
- TriviaQA alias matching may misclassify paraphrases; constrain answers to short spans and preserve raw text for audit.
- Stable consistency can reflect a confident misconception; retain stable-wrong separately.
- The pressure prompt is inseparable from incentive; use pressured-honest controls and prompt-token activation checks.
- A detector trained on explicit lie instructions may exploit style; use disjoint questions, terse output formatting, grouped splits, and describe this transfer limitation.
- Model or dependency download failure triggers a documented reduced behavioral API experiment, but simulated model outputs are never permitted.

## Success Criteria

The research succeeds if it produces real-model outputs, auditable mechanism labels, honest cell yields, at least one response-token activation detector and one sampling-uncertainty baseline, prompt-confound controls, uncertainty intervals, raw artifacts, and a reproducible report. The original target of 100 cases per false-output cell is a go criterion for confirmatory detector claims, not permission to manufacture balance; lower-yield results will be explicitly preliminary.

Scoring is 1–5 on literature support (L), hypothesis fit (H), expected information gain (I), and feasibility (F), for a maximum of 20. Only the top three directions are retained.

| Rank | Direction | L | H | I | F | Total | Decision |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | Matched-question 2×2 mechanism study: neutral belief status (known vs unknown/unstable) × incentive (neutral vs pressure), with the same model and questions | 5 | 5 | 5 | 4 | 19 | Keep |
| 2 | Detector dissociation matrix: apply a standard white-box lie probe and semantic-uncertainty/hallucination score to true, epistemic-error, and incentivized-misreport cells | 5 | 5 | 5 | 3 | 18 | Keep |
| 3 | Robustness by false-output subtype: stable wrong association vs unstable/no recall, plus correct lies where the model's belief is wrong | 4 | 5 | 5 | 3 | 17 | Keep |
| 4 | Train a universal lie detector on a pooled taxonomy | 4 | 3 | 2 | 3 | 12 | Reject: repeated cross-type generalization failures make this low-information without first establishing clean mechanisms. |
| 5 | Compare many model families/scales | 4 | 2 | 3 | 1 | 10 | Reject: violates the one-model matched design and confounds mechanism with capability/model family. |
| 6 | Agentic, multi-turn spontaneous scheming benchmark | 4 | 3 | 4 | 1 | 12 | Reject: high ecological value but costly, difficult belief ground truth, and poor first experiment. |
| 7 | Non-lying deception (omission/implicature) | 3 | 2 | 4 | 2 | 11 | Reject for this phase: outside the stated false-statement estimand; preserve as future work. |

## Operational plan for the top three

1. Select one open-weight instruction model whose activations and token probabilities are accessible. Deduplicate TriviaQA validation by question ID and add clean MASK known/provided-fact items.
2. Run at least three independent neutral belief elicitations per question at low temperature. Define **known-correct** only when answers are stable and match aliases; **stable-wrong** when stable but factually wrong; **unknown/unstable** when inconsistent, refusing, or low-consistency. Do not infer ignorance from one wrong sample.
3. Re-ask the identical question in neutral and incentive/pressure frames. Classify false outputs as epistemic errors only in the unknown/unstable or stable-wrong neutral strata, and as operational lies only when a pressured statement contradicts a robustly elicited belief. Report ambiguous cases separately.
4. Balance or reweight questions across mechanism cells. Store prompt pair, seeds, completions, belief votes, ground-truth match, and exclusion reason.
5. Evaluate: (a) Apollo mean activation probe or DYL where model compatibility permits; (b) semantic entropy from multiple sampled answers; (c) output correctness baseline; and optionally (d) black-box unrelated-question detector.
6. Primary estimands: share of all false statements in each mechanism class with bootstrap confidence intervals; detector AUROC and AUPRC for lie-vs-epistemic-error; calibration; TPR at 1% FPR on ordinary truthful controls. Use question-level bootstrap and paired tests because prompts are matched.
7. For Direction 3, report detector results separately for unassociated/unstable errors, stable associated errors, false lies, and “correct lies” (output happens to match ground truth while contradicting a stable wrong belief).

## Stop/go criteria

- Proceed only if each main false-output cell has at least 100 question-level examples after exclusions.
- If pressure changes apparent neutral beliefs rather than only reports, exclude or flag those items.
- If repeated neutral answers have low agreement, expand elicitation before labeling; never collapse ambiguity into hallucination.
- Do not tune detector thresholds on the test mechanisms. Reserve ordinary Alpaca-style dialogue for false-positive calibration.
