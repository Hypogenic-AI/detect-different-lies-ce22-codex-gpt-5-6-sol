# Downloaded Papers

All PDFs were validated with `pypdf`. The nine user-specified arXiv papers were split into three-page files under `papers/pages/` and reviewed across every chunk; four discovered foundations were screened from abstracts and their first chunks.

| Paper | Authors | Year | File | Relevance |
|---|---|---:|---|---|
| Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation | Kuhn, Gal, Farquhar | 2023 | `2302.09664_semantic_uncertainty.pdf` | Standard uncertainty/hallucination baseline; uses TriviaQA. |
| The Internal State of an LLM Knows When It's Lying | Azaria, Mitchell | 2023 | `2304.13734_internal_state_truthfulness.pdf` | Foundational hidden-state truth classifier; conflates factuality and lying terminology. |
| How to Catch an AI Liar | Pacchiardi et al. | 2023 | `2309.15840_black_box_unrelated_questions.pdf` | Standard black-box unrelated-question detector. |
| Truth is Universal | Bürger, Hamprecht, Nadler | 2024 | `2407.12831_truth_is_universal.pdf` | Two-dimensional truth subspace and polarity-robust detector. |
| Detecting Strategic Deception Using Linear Probes | Goldowsky-Dill et al. | 2025 | `2502.03407_strategic_deception_linear_probes.pdf` | Primary standard white-box deception probe. |
| The MASK Benchmark | Ren et al. | 2025/2026 | `2503.03750_MASK_benchmark.pdf` | Directly separates elicited belief, pressured statement, and ground truth. |
| Do LLMs Really Know What They Don't Know? | Cheang et al. | 2025/2026 | `2510.09033_internal_states_knowledge_recall.pdf` | Separates associated and unassociated hallucinations; tests detector specificity. |
| Liars' Bench | Kretschmar et al. | 2025/2026 | `2511.16035_liars_bench.pdf` | Diverse, large evaluation of three black/white-box detector families. |
| One Probe Won't Catch Them All | Natarajan et al. | 2026 | `2602.01425_targeted_deception_detection.pdf` | Demonstrates type-matched probe advantage and prompt dominance. |
| Probing the Limits of the Lie Detector Approach | Thormann | 2026 | `2603.10003_limits_lie_detector.pdf` | Shows truth probes miss deception without false statements. |
| “Did you lie?” | Cooney, Africa, Irving | 2026 | `2606.12618_did_you_lie.pdf` | Belief-verified model organisms and DYL detector; strong construct-validity evidence. |
| Beyond Liars' Bench | Moustafa, Feser, Mai | 2026 | `2607.20479_beyond_liars_bench.pdf` | Lie typology, layer, nonlinearity, and sparse-feature ablations. |
| Asymmetries in Spontaneous and Instructed Deception | Luikham | 2026 | `2609.00180_spontaneous_instructed_deception.pdf` | Tests geometric, classifier, and steering transfer across elicitation settings. |

Also reviewed online: Hopkins, Khullar, Wang, and Roger, **Fine-Tuned Lie Detectors Failed to Generalize** (Anthropic Alignment Science Blog, 2026). No PDF was published at the specified URL, so it is cataloged as a web report rather than placed in this PDF-only directory.
