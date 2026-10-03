# Detect Different Lies

A matched white-box study of false reports from `Qwen/Qwen2.5-7B-Instruct`. It separates unstable epistemic errors, stable wrong associations, and pressure-induced contradictions of stable correct neutral beliefs, then tests linear lie and hallucination/correctness probes.

## Key findings

- 212/261 stable-correct questions became false under incentive: 81.2% (95% CI 76.0–85.5%).
- Operational lies were 31.9% of all 664 false reports; 35.5% of false reports under pressure had no stable correct neutral belief and remained mechanism-ambiguous.
- A lie probe scored AUROC 0.985 on lie versus neutral unstable error, but a prompt-only probe scored 1.000—complete context leakage.
- With pressure and falsity controlled, the lie probe was at chance for lie versus unknown false output: AUROC 0.478 [0.414, 0.538].
- The lie probe and hallucination probe both strongly separated pressured false from pressured correct outputs, consistent with falsehood/recall sensitivity rather than lie-specific detection.

See [REPORT.md](REPORT.md) for the full methodology, results, caveats, and references.

## Reproduce

Requirements: an NVIDIA GPU with roughly 20 GiB or more, Hugging Face model access/network, and Python 3.10+.

```bash
uv sync
source .venv/bin/activate
python src/run_experiment.py --eval-n 500 --train-n 200 --batch-size 32
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python src/analyze_results.py
```

The exact model revision, prompts, sampling parameters, software versions, and hardware are recorded in `results/config.json`.

## Structure

- `src/`: generation, activation extraction, probe fitting, and analysis
- `results/raw/`: immutable real-model completions
- `results/processed/`: activations, scored reports, mechanism shares, and detector metrics
- `figures/`: publication-ready plots
- `planning.md`: preregistered hypotheses, direction ranking, and analysis plan
- `literature_review.md`: systematic synthesis of the relevant work
- `REPORT.md`: complete research report
