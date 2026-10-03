# Resources Catalog

## Summary

The workspace contains 13 validated PDFs, two locally usable datasets, four shallow-cloned repositories, chunk manifests for paper review, and a ranked experiment plan. The collection is optimized for a one-model, matched-question mechanism study rather than a broad multi-model benchmark.

## Papers

Total PDFs: **13**. See `papers/README.md` and `literature_review.md`. The most actionable papers are MASK, Liars' Bench, Detecting Strategic Deception Using Linear Probes, “Did you lie?”, and the recall-vs-truthfulness hallucination analysis.

## Datasets

| Name | Source | Local size | Task | Location | Notes |
|---|---|---:|---|---|---|
| MASK | Hugging Face `cais/MASK` | 1.52 MB / 1,000 rows | Honesty under pressure with belief elicitation | `datasets/MASK/` | All six configs downloaded and validated; no nulls. |
| TriviaQA validation (`rc.nocontext`) | Hugging Face `mandarjoshi/trivia_qa` | 14.1 MB / 17,944 rows | Open-domain QA | `datasets/trivia_qa_rc_nocontext_validation/` | 9,960 unique IDs; deduplicate before sampling. Full 34.8 GB dataset intentionally not downloaded. |

Liars' Bench is highly relevant but its canonical Hugging Face dataset requires accepting access conditions. Its public code and schema were cloned; no credentials or license acceptance were assumed. The 2026 Anthropic `Noddybear/lies` corpus is a useful external robustness set but was pruned because it does not supply the matched knowledge/pressure design needed for the primary estimand.

## Code repositories

| Name | URL | Purpose | Location |
|---|---|---|---|
| MASK | https://github.com/centerforaisafety/mask | Honesty/accuracy generation and scoring | `code/mask/` |
| deception-detection | https://github.com/ApolloResearch/deception-detection | Activation-probe baseline | `code/deception-detection/` |
| Liars' Bench | https://github.com/Cadenza-Labs/liars-bench | Multi-type generation and detector evaluation | `code/liars-bench/` |
| lie_detectors | https://github.com/UKGovernmentBEIS/lie_detectors | Released DYL/probe checkpoint loader | `code/lie-detectors/` |

See `code/README.md` for commits, entry points, and blockers. All Python sources compile; full runs require model weights, GPU capacity, and in some cases API keys.

## Search strategy and selection

The primary diligent paper-finder call failed with an HTTP 500 and recorded a fallback result under `paper_search_results/`. Manual discovery then resolved all specified arXiv IDs, searched foundational method names, followed citation/project links, and checked official arXiv, Hugging Face, and GitHub pages. Inclusion emphasized direct mechanism separation and reusable detectors; adjacent work was retained only when it supplies a baseline or exposes a validity failure.

## Challenges and workarounds

- The paper-finder service was unavailable; source-first manual search replaced it.
- The 112-page Liars' Bench PDF contains a long prompt/example appendix; it was still processed across all 38 three-page chunks, with extraction focused on methods, labels, metrics, and examples.
- TriviaQA's full repository is too large for this phase; a standard held-out no-context split was saved instead.
- Liars' Bench data is gated; the canonical repository and documentation are present, but data access was not bypassed.
- White-box experiments cannot be smoke-tested without downloading a compatible multi-billion-parameter model; static compilation and code-path inspection were used instead.

## Experiment recommendation

Use a compatible open-weight instruction model and build a matched dataset from deduplicated TriviaQA plus MASK known/provided facts. Establish neutral belief using repeated low-temperature paraphrases, retain stable-wrong and ambiguous cases, then apply paired incentive prompts. Evaluate a frozen Apollo or DYL probe beside semantic entropy. The primary output is the mechanism share among false statements; the primary detector test is lie versus epistemic-error discrimination under question-level paired resampling. Full direction scores and exclusions are in `planning.md`.

## Executed experiment (2026-10-03)

The experiment runner selected `Qwen/Qwen2.5-7B-Instruct` revision `a09a35458c702b33eeacc393d103063234e8bc28`, generated five belief samples plus neutral/pressure reports on 500 deduplicated TriviaQA questions, and built disjoint linear-probe training data on 200 further questions. The released Apollo/DYL checkpoints were architecture-incompatible, so their standard residual-stream logistic method was implemented with explicit provenance and prompt-token controls. Raw outputs, activations, statistics, and figures are under `results/` and `figures/`; full findings are in `REPORT.md`. MASK informed the belief/accuracy operationalization but was not mixed into the primary factual-QA panel because its scenario formats and proposition labels would introduce a task confound.
