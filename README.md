# Governed Banking Intent Router

A Mac-first research implementation of a privacy-aware banking support router. The system uses
LoRA-adapted RoBERTa as one component in a larger decision pipeline that includes calibration,
uncertainty handling, deterministic escalation and metadata-only audit events.

> **Status: data pipeline validated; models not yet trained or approved for production use.**

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

Module 2 adds a commit- and hash-pinned BANKING77 loader, group-aware validation split, leakage
quarantine and text-free reproducibility manifest. No model result is claimed at this stage.

Run the local checks:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev,notebook]"
ruff check .
pytest --cov=governed_banking --cov-report=term-missing
```

Prepare and audit the dataset:

```bash
python scripts/prepare_banking77.py
```

Then open
[`02-build-banking77-manifest.ipynb`](output/jupyter-notebook/02-build-banking77-manifest.ipynb)
in VS Code and run it with the `Governed Banking AI (MPS)` kernel. The verified seed-42 manifest
contains 8,495 training rows, 1,501 validation rows and all 3,080 official test rows. Seven
training rows that duplicate normalized official-test messages are quarantined.

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
3. TF-IDF logistic-regression baseline - **next**
4. Frozen RoBERTa embedding baseline
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
