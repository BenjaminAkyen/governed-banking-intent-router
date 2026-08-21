# Frozen RoBERTa Embedding Baseline Card

## Purpose

This module tests whether generic contextual representations improve BANKING77 intent separation
without updating the Transformer. It is a controlled baseline for later LoRA adaptation, not a
production routing model.

## Encoder contract

- Model: [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base)
- Revision: `e2da8e2f811d1448a5b465c236feacd80ffbac7b`
- Licence recorded by the model repository: MIT
- Base encoder parameters: 124,055,040
- Trainable encoder parameters: 0
- Execution: float32 inference on Apple MPS
- Maximum sequence length: 96 tokens
- Observed truncation: 0 rows across train, validation and test
- Snapshot allowlist and SHA-256 hashes: `reports/frozen-roberta/selection.json`

The masked-language-model head in the upstream checkpoint is deliberately ignored. The unused
pooler is not instantiated. CLS embeddings use the final hidden state of the first token. Mean
embeddings average only non-special, non-padding final-layer token states, followed by L2
normalisation.

## Leakage controls

- Exact rows are loaded from the hash-verified Module 2 manifest.
- Train and validation embeddings are created before test access.
- Test records and embeddings remain inaccessible until the selection lock validates.
- The selection lock binds dataset, configuration, implementation and model-file hashes.
- Local caches bind source-index hashes, array hashes, device metadata and extraction policy.
- Reports and prediction records contain no message text.

## Validation search history

The initial search ended at its regularisation boundary. The search was expanded using validation
results only. Each amendment recorded that no test access had occurred. Round 4 was declared the
final plateau check.

| Candidate | Pooling | C | Validation macro-F1 |
|---|---|---:|---:|
| `cls_c1` | CLS | 1 | 0.0162 |
| `mean_c1` | Mean | 1 | 0.4458 |
| `mean_c4` | Mean | 4 | 0.6557 |
| `mean_c16` | Mean | 16 | 0.7967 |
| `mean_c64` | Mean | 64 | 0.8584 |
| `mean_c256` | Mean | 256 | 0.8818 |
| `mean_c1024` | Mean | 1,024 | **0.8823** |
| `mean_c4096` | Mean | 4,096 | 0.8811 |

`mean_c1024` was the interior winner and every classifier fit converged.

## Locked test result

| Measure | Frozen RoBERTa | TF-IDF reference |
|---|---:|---:|
| Accuracy | 0.8961 | 0.9052 |
| Macro-F1 | 0.8964 | 0.9053 |
| Weighted-F1 | 0.8964 | 0.9053 |
| Log-loss | 0.3990 | 0.4970 |
| Top-3 accuracy | 0.9718 | 0.9705 |

Lower log-loss is not treated as proof of calibration. Calibration is evaluated separately later.

## Paired comparison

| Outcome | Rows |
|---|---:|
| Both correct | 2,625 |
| TF-IDF only correct | 163 |
| Frozen RoBERTa only correct | 135 |
| Both wrong | 157 |
| Different predicted labels | 368 |

The exact two-sided McNemar p-value is 0.1177. The observed paired correctness difference is not
statistically significant at 0.05; this does not prove that the models are equivalent. The 135
RoBERTa-only successes justify analysing complementary errors during later policy work.

## Material failure modes

The weakest intent is `balance_not_updated_after_bank_transfer` at 0.6988 F1. Recurring confusions
include both directions between `pending_transfer` and `balance_not_updated_after_bank_transfer`,
unrecognised card versus direct-debit payments, and failed versus reverted top-ups.

## Limitations

- This is a single-seed classifier evaluation.
- RoBERTa pretraining is generic and English-only for this checkpoint; no banking adaptation occurs.
- The public benchmark is not representative production bank traffic.
- CLS performance demonstrates that a token representation is not automatically a sentence embedding.
- Validation-guided search was amended three times; all amendments occurred before test access.
- Confidence is uncalibrated and cannot yet drive automatic routing.
- MPS timings are local observations, not portable performance guarantees.

## Evidence

- Configuration: `configs/frozen_roberta.yaml`
- Selection lock: `reports/frozen-roberta/selection.json`
- Extraction evidence: `reports/frozen-roberta/embedding-extraction.json`
- Locked test: `reports/frozen-roberta/test.json`
- Predictions: `reports/frozen-roberta/test-predictions.jsonl`
- Paired comparison: `reports/frozen-roberta/paired-vs-tfidf.json`
- Guided audit: `output/jupyter-notebook/04-frozen-roberta-embedding-baseline.ipynb`
