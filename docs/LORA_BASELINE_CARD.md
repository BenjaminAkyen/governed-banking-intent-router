# LoRA-Adapted RoBERTa Baseline Card

## Purpose

Module 5 tests whether parameter-efficient domain adaptation improves fine-grained BANKING77
intent routing over the Module 3 lexical and Module 4 frozen-encoder baselines. This is a
single-seed engineering checkpoint, not a production model or the final multi-seed comparison.

## Architecture

- Base model: [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base)
- Pinned revision: `e2da8e2f811d1448a5b465c236feacd80ffbac7b`
- Adapter method: LoRA on every attention `query` and `value` projection
- Classification head: newly initialised for 77 intents, trained and saved with the adapter
- Frozen parameters: all other RoBERTa parameters
- Execution: float32 training and inference on Apple MPS
- Maximum sequence length: 96 tokens
- Observed truncation: zero train, validation or test rows

LoRA represents the weight update to a frozen projection with two low-rank matrices. The adapter is
initialised as a no-op and learns the task-specific update while the underlying RoBERTa weights stay
fixed. The sequence-classification head is not present in the upstream masked-language-model
checkpoint, so it must be trained and included in the adapter artifact.

## Registered candidates

Both candidates used AdamW, three epochs, 0.01 weight decay, linear decay after 10% warmup, a
physical batch of 32, two gradient-accumulation steps and gradient clipping at 1.0.

| Candidate | Rank / alpha | Learning rate | Trainable parameters | Best epoch | Validation macro-F1 |
|---|---:|---:|---:|---:|---:|
| `lora_r8_lr2e4` | 8 / 16 | 2e-4 | 944,717 (0.7519%) | 3 | **0.8241** |
| `lora_r16_lr1e4` | 16 / 32 | 1e-4 | 1,239,629 (0.9843%) | 3 | 0.6774 |

Validation macro-F1 by epoch was 0.5814, 0.7749 and 0.8241 for rank 8; and 0.3713, 0.6369 and
0.6774 for rank 16. Both curves were still improving at the registered three-epoch boundary. The
rank-8 checkpoint was selected before test access.

## Leakage and privacy controls

- Exact rows come from the hash-verified Module 2 manifest.
- Candidate fitting uses `train`; checkpoint selection uses `validation` only.
- The selection report records `test_split_loaded: false`.
- Dataset, configuration, implementation, base-model files and adapter files are SHA-256 bound.
- The official test split is loaded only after the selection lock and selected adapter validate.
- Only SafeTensors adapter weights are accepted; pickle-based weight files are rejected.
- Reports and prediction rows contain indices, labels and scores but no customer message text.

## Locked test result

All methods below used the same 3,080 official test rows.

| Measure | TF-IDF | Frozen RoBERTa | LoRA RoBERTa |
|---|---:|---:|---:|
| Accuracy | **0.9052** | 0.8961 | 0.8273 |
| Macro-F1 | **0.9053** | 0.8964 | 0.8202 |
| Weighted-F1 | **0.9053** | 0.8964 | 0.8202 |
| Log-loss | 0.4970 | **0.3990** | 0.7746 |
| Top-3 accuracy | 0.9705 | **0.9718** | 0.9497 |

This registered LoRA run did **not** outperform either baseline. Relative to TF-IDF, LoRA was 0.0851
lower in macro-F1; TF-IDF alone was correct on 350 paired rows while LoRA alone was correct on 110.
Relative to frozen RoBERTa, LoRA was 0.0762 lower; frozen RoBERTa alone was correct on 345 rows and
LoRA alone on 133. Both exact paired tests have two-sided p-values below 1e-10 at the report's
precision.

## Failure analysis

The weakest test intents were `contactless_not_working` (0.2609 F1),
`virtual_card_not_working` (0.3673), `topping_up_by_card` (0.5079), `card_acceptance` (0.6557) and
`card_swallowed` (0.6667). The two largest confusion groups were:

- `contactless_not_working` predicted as `lost_or_stolen_phone`: 18 rows;
- `virtual_card_not_working` predicted as `get_disposable_virtual_card`: 17 rows.

The closely matched validation (0.8241) and test (0.8202) scores do not suggest a test-only collapse.
The monotonic validation curves instead indicate that the registered three-epoch budget stopped too
early, particularly with a newly initialised 77-class head.

## Claim boundary and next experiment

This result supports the claim that the **registered Module 5 protocol underfit**. It does not support
the broader claim that LoRA is inferior to lexical models or frozen embeddings.

The official test result is now known. Extending the epochs and reporting a new score on the same
test as if it were untouched would be post-test tuning. A follow-up must therefore be explicitly
labelled exploratory, select its stopping rule on validation only, and obtain genuinely independent
confirmation—such as a separately sourced banking-intent evaluation set—before supporting a new
confirmatory performance claim. Module 6 will also run the registered seeds 17, 42 and 73.

## Other limitations

- BANKING77 is a public English benchmark, not representative bank traffic.
- Only two adapter configurations and one seed were evaluated here.
- MPS randomness is seeded but exact cross-version bitwise reproducibility is not claimed.
- Confidence values are uncalibrated and must not drive automatic routing.
- No possible-OOD, robustness, privacy-redaction or risk-policy gate is implemented in this module.

## Evidence

- Configuration: `configs/lora_roberta.yaml`
- Selection lock: `reports/lora-roberta/selection.json`
- Locked test: `reports/lora-roberta/test.json`
- Text-free predictions: `reports/lora-roberta/test-predictions.jsonl`
- Paired comparisons: `reports/lora-roberta/paired-comparisons.json`
- Local adapters: ignored under `artifacts/lora-roberta/`; hashes are recorded in the selection lock
- Guided audit: `output/jupyter-notebook/05-lora-adapted-roberta.ipynb`
