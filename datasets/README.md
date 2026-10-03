# Downloaded Datasets

Large data files are ignored by Git. Run `bash datasets/download.sh` from the repository root after creating the project environment with `uv sync`.

## MASK

- **Source:** `cais/MASK` on Hugging Face
- **Local path:** `datasets/MASK/<config>/`
- **Size:** 1,000 examples, 1.52 MB on disk
- **Splits:** six test-only configurations: continuations (176), disinformation (125), doubling-down known facts (120), known facts (209), provided facts (274), and statistics (96)
- **Schema:** pressure prompt, proposition, ground truth, and one or more neutral belief-elicitation prompts; some configurations add dialogue fields
- **License:** consult the upstream dataset card/repository
- **Validation:** all configurations load with `datasets.load_from_disk`; no null values were found across their schemas
- **Use:** direct source of incentive/pressure conditions and repeated belief questions. Prefer known/provided facts for clean matched comparisons; analyze archetypes separately.

```python
from datasets import load_from_disk
known_facts = load_from_disk("datasets/MASK/known_facts")["test"]
```

Sample: `datasets/samples/mask_known_facts.json`.

## TriviaQA (`rc.nocontext`, validation)

- **Source:** `mandarjoshi/trivia_qa` on Hugging Face
- **Local path:** `datasets/trivia_qa_rc_nocontext_validation/`
- **Size:** 17,944 rows, 14.1 MB saved on disk
- **Format:** Hugging Face Arrow dataset; question, question ID, source metadata, evidence metadata, and answer aliases
- **Task:** open-domain factual question answering
- **License:** upstream card says the University of Washington does not own copyright in included questions/documents; review terms before redistribution
- **Validation:** loads successfully; no empty questions; 9,960 unique question IDs
- **Use:** source of matched factual questions. Deduplicate by `question_id` before sampling. The full repository is about 34.8 GB, so only this held-out, no-context split was retained.

```python
from datasets import load_from_disk
trivia = load_from_disk("datasets/trivia_qa_rc_nocontext_validation")
```

Sample: `datasets/samples/trivia_qa_validation.json`.

## Known limitations

MASK labels a contradiction between pressured statement and elicited belief; this operationalizes lying but does not prove conscious intent. TriviaQA answer aliases provide factual ground truth but not model belief. Repeated neutral elicitation is therefore required before assigning a question to “known,” “unknown/unstable,” or “incorrect stable belief.”
