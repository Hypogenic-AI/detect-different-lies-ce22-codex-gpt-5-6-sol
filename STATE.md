# Research State

- Current phase: `None`
- Pipeline completed: `True`

## Previous phases

resource_finder (succeeded), experiment_runner (succeeded)

## Current phase context

- Phase: `experiment_runner`
- Status: `completed`
- Started: `2026-10-03T07:35:04.738579Z`
- Next steps:
  - Validate the report and experimental artifacts before finalizing.

## Workspace check

- Root: `/workspaces/detect-different-lies-ce22-codex-gpt-5-6-sol`
- Directory usable: `True`

## Output validation

- Valid: `True`
- Expected: `REPORT.md`
- Missing: None
- Outside workspace: None

## Agent notes

<!-- NEURICO_AGENT_NOTES_START -->
### resource_finder
<!-- NEURICO_AGENT_NOTES_START:resource_finder -->
Phase `resource_finder` completed 2026-10-03. Created a fresh uv environment; gathered 13 validated PDFs, all six MASK configs (1,000 rows), TriviaQA `rc.nocontext` validation (17,944 rows; 9,960 unique IDs), and four detector/evaluation repositories. Deep-read all chunks of the nine specified arXiv papers; synthesis and source links are in `literature_review.md`, resource inventory in `resources.md`, and ranked directions/rejections in `planning.md`.

Key decision: run a one-model matched belief-status × incentive study. Require repeated neutral belief elicitation; keep stable-wrong and unknown/unstable strata separate; compare a frozen Apollo/DYL white-box score with semantic entropy. Do not equate a single wrong answer with ignorance or a detector score with deceptive intent.

Next phase: `experiment_runner`. Select a probe-compatible open-weight model; deduplicate TriviaQA by `question_id`; construct paired neutral/pressure prompts using MASK schemas; generate repeated neutral and pressured answers with seeds; label true report, epistemic error, operational lie, stable associated error, and ambiguity; estimate mechanism shares and detector dissociation with question-level bootstrap intervals.

Unresolved: model/GPU availability and checkpoint compatibility; Liars' Bench canonical data is gated; pressure may update beliefs; behavioral consistency is only an operational belief proxy. The paper-finder service failed with HTTP 500, so manual source-first discovery was used. Evidence paths: `papers/pages/*_manifest.txt`, `datasets/README.md`, and `code/README.md`.
<!-- NEURICO_AGENT_NOTES_END:resource_finder -->

### experiment_runner
<!-- NEURICO_AGENT_NOTES_START:experiment_runner -->
Phases 1–6 complete. Used local `Qwen/Qwen2.5-7B-Instruct` revision `a09a35458c702b33eeacc393d103063234e8bc28` on an RTX A6000: 500 evaluation questions with five neutral belief samples plus matched neutral/pressure reports, and 200 disjoint probe-training questions. Final strata: 261 stable-correct, 139 unstable, 100 stable-wrong. Of 664 false reports, 212 (31.9%) were operational lies; pressure lie yield among stable-correct questions was 81.2% [76.0%, 85.5%]. The response lie probe was strong in the context-confounded contrast (AUROC 0.985) but chance for same-pressure lie vs unknown false output (0.478 [0.414, 0.538]); prompt-only AUROC was 1.000 in the former, confirming leakage. Both lie and hallucination probes separated false lies from pressured correct answers, supporting falsehood/recall rather than lie-specific sensitivity.

Artifacts: `REPORT.md`, `README.md`, `src/run_experiment.py`, `src/analyze_results.py`, `results/raw/`, `results/processed/`, and `figures/`. A scoring audit replaced exact-only aliases with complete-token containment and all labels were recomputed from immutable raw text. Decoy selection now rejects semantic alias matches. Two independent final full runs produced byte-identical raw outputs and activation archives (SHA-256 recorded in the execution transcript); final analysis validation passed 13 artifact/content assertions and source compilation. Remaining uncertainty is scientific rather than operational: behavioral belief proxies, artificial incentive strength, one 7B model, circular entropy baseline, and unresolved pressure errors without stable correct knowledge.
<!-- NEURICO_AGENT_NOTES_END:experiment_runner -->

<!-- NEURICO_AGENT_NOTES_END -->
