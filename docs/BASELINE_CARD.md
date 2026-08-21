# TF-IDF Logistic-Regression Baseline Card

## Purpose

This baseline establishes the performance and failure profile that later RoBERTa experiments must
beat or materially complement. It is intentionally transparent, inexpensive and CPU-compatible.
It is not a customer-facing routing system.

## Leakage controls

- Exact rows come from the hash-verified Module 2 manifest.
- TF-IDF feature vocabularies and inverse-document-frequency weights fit on training messages only.
- Three candidates were declared in `configs/baseline_tfidf.yaml` before fitting.
- Candidate ranking uses validation macro-F1, then accuracy, lower log-loss and candidate name.
- The selection report states that the test split was not loaded and is protected by a content hash.
- Test loading occurs only after the selection artifact matches the dataset, configuration and code.
- Reports and predictions contain no message text.

## Candidate selection

| Rank | Candidate | Features | Validation macro-F1 | Validation log-loss |
|---:|---|---:|---:|---:|
| 1 | `word_char_c4` | 22,089 | 0.8920 | 0.5376 |
| 2 | `word_char_c2` | 22,089 | 0.8857 | 0.6425 |
| 3 | `word_12_c2` | 9,263 | 0.8691 | 0.9847 |

All three fits converged. `word_char_c4` was locked before test evaluation.

## Locked test result

| Measure | Result |
|---|---:|
| Test rows | 3,080 |
| Intents | 77 |
| Accuracy | 0.9052 |
| Macro-F1 | 0.9053 |
| Weighted-F1 | 0.9053 |
| Log-loss | 0.4970 |
| Top-3 accuracy | 0.9705 |
| Features | 22,089 |

The fitted local model is approximately 14.4 MB. Timing is machine- and run-specific and is
recorded in the evidence report rather than presented as a general performance claim.

## Material failure modes

The lowest test F1 scores include:

| Intent | F1 | Recall |
|---|---:|---:|
| `balance_not_updated_after_bank_transfer` | 0.7556 | 0.8500 |
| `top_up_failed` | 0.7952 | 0.8250 |
| `card_not_working` | 0.7955 | 0.8750 |
| `pending_transfer` | 0.8000 | 0.7500 |
| `failed_transfer` | 0.8046 | 0.8750 |

Recurring directional errors include `pending_top_up` predicted as `top_up_failed`,
`verify_my_identity` predicted as `why_verify_identity`, and `virtual_card_not_working` predicted
as `get_disposable_virtual_card`. These distinctions can change the appropriate support workflow.

## Limitations

- This is one seed; seeds 17 and 73 remain required for the primary comparison.
- The public English benchmark is not representative production bank traffic.
- Exact and normalised-text leakage checks do not detect every semantic near-duplicate.
- Logistic-regression probabilities are not calibrated merely because `predict_proba` exists.
- Top-3 accuracy does not justify automated action.
- Operational cost, class risk and abstention are not evaluated in this module.

## Evidence

- Selection: `reports/baseline/tfidf-logreg-selection.json`
- Locked test: `reports/baseline/tfidf-logreg-test.json`
- Predictions: `reports/baseline/tfidf-logreg-test-predictions.jsonl`
- Guided audit: `output/jupyter-notebook/03-tfidf-logreg-baseline.ipynb`

The next baseline should use the same manifest and reporting contract so comparisons remain fair.
