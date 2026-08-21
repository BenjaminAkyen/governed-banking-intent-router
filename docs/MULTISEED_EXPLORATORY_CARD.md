# Multi-Seed Validation-Only LoRA Card

## Purpose and claim boundary

Module 6 revises the Module 5 training budget and measures sensitivity across seeds 17, 42 and 73.
It is explicitly **post-test exploratory**: BANKING77's official test result was already observed in
Module 5. This module does not load that split for modelling, compute new test metrics or support a
confirmatory improvement claim.

## Immutable manifests

Each seed derives a group-stratified train/validation split from the hash-pinned official training
source. Exact normalized-text duplicate groups remain together. The official test source is read by
the separate integrity-preparation stage only to quarantine exact training duplicates; the training
runner cannot load or score it.

| Seed | Train rows | Validation rows | Validation-index identity |
|---:|---:|---:|---|
| 17 | 8,497 | 1,499 | `20b7e3f41c8d70de…` |
| 42 | 8,495 | 1,501 | `9c66b4850a185873…` |
| 73 | 8,497 | 1,499 | `7a9141d6134fb5c…` |

The small count difference is intentional: duplicate groups are indivisible, so exact stratified
row counts are less important than preventing group leakage. The text-free registry binds every
manifest, configuration and implementation hash.

## Revised protocol

The architecture is Module 5's validation-selected rank-8 LoRA candidate:

- pinned `FacebookAI/roberta-base` revision;
- LoRA rank 8, alpha 16 and dropout 0.1;
- attention `query` and `value` projections adapted;
- 77-class head trained and saved with the adapter;
- 944,717 trainable parameters, or 0.7519%;
- AdamW at 2e-4, linear schedule, 10% warmup and 0.01 weight decay;
- physical batch 32 with two gradient-accumulation steps;
- float32 Apple MPS training.

The stopping rule was registered before these runs:

- minimum 4 epochs;
- maximum 8 epochs;
- monitor validation macro-F1;
- an improvement must exceed 0.002 to reset patience;
- stop after 2 consecutive insufficient-improvement epochs;
- retain the best validation checkpoint even if it differs from the stopping epoch.

## Validation results

| Seed | Best epoch | Epochs completed | Best validation macro-F1 | Stop reason |
|---:|---:|---:|---:|---|
| 17 | 8 | 8 | 0.9006 | Maximum epochs reached |
| 42 | 8 | 8 | 0.9026 | Patience reached on final epoch |
| 73 | 6 | 8 | 0.8890 | Patience reached |
| **Mean** | — | — | **0.8974** | — |

Sample standard deviation is 0.0073. The descriptive 95% t-interval is `[0.8792, 0.9156]`; with
only three seeds, it is wide and is not a production or confirmatory uncertainty estimate.

## Interpretation

The revised budget materially changes validation behaviour relative to Module 5's three-epoch
protocol. However, these validation values cannot be compared with Module 5's official-test score
as proof of improved generalisation. They come from different data roles and the revised protocol
was designed after observing the earlier test outcome.

Seeds 17 and 42 achieved their best values at epoch 8. The maximum epoch remains an active search
boundary for two of three runs, so this module does not establish full convergence. Seed 73 peaked
at epoch 6 and declined slightly, demonstrating why restoring the best validation checkpoint matters.

## Safety and evidence controls

- The runner has no test-evaluation phase, test-report argument or prediction output.
- A fail-closed split gate permits only `train` and `validation`; `test` and unknown names raise.
- Every report records `test_split_loaded: false` and `official_test_metrics_computed: false`.
- Reports bind the manifest, configuration, implementation, base-model and adapter hashes.
- Resume accepts a run only if its report and local checkpoint fully validate.
- Adapter weights use SafeTensors; pickle-based weight files are rejected.
- Reports contain no source message text.

## Next decision

Do not spend another cycle extending BANKING77 epochs and comparing against the already observed
test. The next credible performance step is an independently sourced confirmation set with a frozen
protocol. Before that external evaluation, Module 7 can fit temperature scaling on held-out
validation data, provided calibration development and assessment subsets are separated.

## Evidence

- Configuration: `configs/multiseed_lora.yaml`
- Manifest registry: `data/manifests/banking77-multiseed-index.json`
- Seed reports: `reports/multiseed-lora/seed-*-validation.json`
- Aggregate: `reports/multiseed-lora/validation-aggregate.json`
- Local adapters: ignored under `artifacts/multiseed-lora/`; hashes appear in seed reports
- Guided audit: `output/jupyter-notebook/06-multiseed-validation-only-lora.ipynb`
