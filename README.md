# Governed Banking Intent Router

A Mac-first research implementation of a privacy-aware banking support router. The system uses
LoRA-adapted RoBERTa as one component in a larger decision pipeline that includes calibration,
uncertainty handling, deterministic escalation and metadata-only audit events.

> **Status: first baseline evaluated; no model is approved for production use.**

![System architecture](docs/images/system-architecture.svg)

## Problem

Customer-support teams must route short and often ambiguous messages. A closed-set classifier will
always choose one of its known labels, even when a request is unfamiliar, contains multiple issues
or requires urgent security review. Optimising average accuracy alone does not make that behaviour
safe.

This project evaluates a more defensible design:

1. redact structured personal information before model inference;
2. classify the redacted request with LoRA-RoBERTa;
3. calibrate probabilities on validation data;
4. compute uncertainty and possible out-of-distribution signals;
5. apply a versioned, deterministic routing policy;
6. record decision metadata without retaining the customer message.

The service recommends a queue. It never moves money, freezes an account, approves a payment,
authenticates a customer or provides financial advice.

## Current module

Module 3 establishes a serious sparse-text baseline. TF-IDF vocabularies are fitted on training
messages only, candidate selection uses validation macro-F1, and test access requires a valid
selection lock. This is a single-seed engineering result, not yet the three-seed primary comparison.

Run the local checks:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev,notebook]"
ruff check .
pytest --cov=governed_banking --cov-report=term-missing
```

Prepare the dataset and reproduce the baseline:

```bash
python scripts/prepare_banking77.py --offline
python scripts/run_tfidf_baseline.py run
```

Then open
[`03-tfidf-logreg-baseline.ipynb`](output/jupyter-notebook/03-tfidf-logreg-baseline.ipynb) in VS
Code and run it with the `Governed Banking AI (MPS)` kernel.

## Verified Module 3 result

The locked winner combines word (1,2)-grams and character-within-word (3,5)-grams with logistic
regression at `C=4.0`. On all 3,080 official test rows it achieved:

| Metric | Result |
|---|---:|
| Accuracy | 0.9052 |
| Macro-F1 | 0.9053 |
| Weighted-F1 | 0.9053 |
| Log-loss | 0.4970 |
| Top-3 accuracy | 0.9705 |

The weakest intent F1 was 0.7556 for `balance_not_updated_after_bank_transfer`. Identity,
transfer-state and top-up-state confusions remain important. Probabilities are uncalibrated and
must not yet drive automated routing. See the [baseline card](docs/BASELINE_CARD.md).

## Evidence contract

- The official test split is not used for training, checkpoint selection, calibration or threshold
  selection.
- Model comparisons use the same data policy and report seeds 17, 42 and 73.
- Public results include macro-F1, per-intent failures, calibration and risk-coverage behaviour.
- Synthetic unknown-request tests are identified as synthetic and not presented as production
  evidence.
- Unverified results are never promoted from notebooks into the README.

See the [data card](docs/DATA_CARD.md), [system boundary](docs/SYSTEM_BOUNDARY.md) and
[evaluation protocol](docs/EVALUATION_PROTOCOL.md).

## Planned modules

1. Mac/MPS foundation and evaluation contract - **complete**
2. Immutable BANKING77 loader and split-integrity tests - **complete**
3. TF-IDF logistic-regression baseline - **complete**
4. Frozen RoBERTa embedding baseline - **next**
5. LoRA-RoBERTa and optional full fine-tuning
6. Multi-seed manifests and aggregation
7. Calibration and selective prediction
8. Possible-OOD evaluation and robustness suites
9. PII controls, risk policy and metadata-only auditing
10. FastAPI service, latency evaluation and release evidence

## Responsible-use limitations

- BANKING77 is a public benchmark, not representative bank traffic.
- Confidence and entropy do not prove that a request is in distribution.
- Regex-based PII redaction cannot identify every contextual identifier.
- Banking operations, fraud, privacy and compliance specialists must review any real deployment.

## Licence

Project code is released under the [MIT License](LICENSE). Dataset and model licences must be
reviewed and documented separately before redistribution.
