# Cloned Repositories

All repositories were shallow-cloned on 2026-10-03 and their Python sources passed `compileall`. Full experiments were not run because they require large open-weight models, GPU memory, and/or provider credentials.

## MASK

- **URL:** https://github.com/centerforaisafety/mask
- **Location:** `code/mask/`
- **Commit:** `25e0b12`
- **Purpose:** generate pressure and belief-elicitation responses, map responses to propositions, and compute separate honesty and accuracy metrics
- **Key files:** `mask/generate_responses.py`, `mask/evaluate.py`, `mask/metric.py`, `mask/process_metrics.py`
- **Use here:** reuse prompt schemas and robust-belief checks; adapt evaluation to preserve an unknown/unstable category.

## Apollo deception-detection

- **URL:** https://github.com/ApolloResearch/deception-detection
- **Location:** `code/deception-detection/`
- **Commit:** `f8ec401`
- **Purpose:** train and evaluate residual-stream linear probes on instructed pairs, roleplay, insider trading, and sandbagging
- **Key files:** `deception_detection/experiment.py`, `deception_detection/scripts/experiment.py`, configs under `deception_detection/scripts/configs/`, and example probe weights under `example_results/`
- **Requirements/blockers:** local model activations and substantial accelerator memory; some generation paths require API/Hugging Face keys.
- **Use here:** standard white-box lie/deception baseline with frozen training data and a threshold calibrated on unrelated honest chat.

## Liars' Bench

- **URL:** https://github.com/Cadenza-Labs/liars-bench
- **Location:** `code/liars-bench/`
- **Commit:** `ba10de1`
- **Purpose:** generation and evaluation harness for seven lie types, black-box judges, self-evaluation, unrelated-question classification, and probe adapters
- **Key files:** scenario code under `src/`, `src/blackbox/blackbox.py`, `src/blackbox/self_eval.py`, and `src/blackbox/pacchiardi.py`
- **Requirements/blockers:** generation needs provider keys; white-box submodules were intentionally not initialized because review mirrors are static and model experiments are heavyweight.
- **Use here:** metric aggregation and cross-mechanism evaluation patterns, especially balanced accuracy and TPR at calibrated low FPR.

## Did-You-Lie probe loader

- **URL:** https://github.com/UKGovernmentBEIS/lie_detectors
- **Location:** `code/lie-detectors/`
- **Commit:** `8804308`
- **Purpose:** load released DYL/Apollo/unrelated-question probe checkpoints from Hugging Face and expose layer and calibrated threshold metadata
- **Key files:** `src/lie_detectors/`
- **Requirements/blockers:** checkpoint download and model-compatible activations; no weights were downloaded during resource finding.
- **Use here:** preferred newer probe if the selected subject model has a compatible released checkpoint; otherwise use Apollo.
