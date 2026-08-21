# Governed Banking Intent Router

A Mac-first research implementation of a privacy-aware banking support router. The system uses
LoRA-adapted RoBERTa as one component in a larger decision pipeline that includes calibration,
uncertainty handling, deterministic escalation and metadata-only audit events.

> **Status: lexical and frozen-encoder baselines evaluated; no model is production approved.**

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

Module 4 evaluates a pinned, fully frozen RoBERTa-base encoder against Module 3. Embeddings are
created on MPS, while validation-only model selection and locked-test evidence use the same row
manifest and artifact controls. These remain single-seed engineering results.

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
python scripts/run_frozen_roberta_baseline.py run --offline
python scripts/compare_module4_baselines.py
```

Then open
[`04-frozen-roberta-embedding-baseline.ipynb`](output/jupyter-notebook/04-frozen-roberta-embedding-baseline.ipynb) in VS
Code and run it with the `Governed Banking AI (MPS)` kernel.

## Verified baseline comparison

Both models were evaluated on the same 3,080 official test rows:

| Metric | TF-IDF | Frozen RoBERTa |
|---|---:|---:|
| Accuracy | 0.9052 | 0.8961 |
| Macro-F1 | 0.9053 | 0.8964 |
| Weighted-F1 | 0.9053 | 0.8964 |
| Log-loss | 0.4970 | 0.3990 |
| Top-3 accuracy | 0.9705 | 0.9718 |

Frozen RoBERTa corrected 135 rows missed by TF-IDF, while TF-IDF corrected 163 rows missed by
RoBERTa. Their paired exact McNemar p-value is 0.1177, which does not establish equivalence. The
results suggest complementarity, but not that generic frozen embeddings beat the lexical baseline.
Probabilities remain uncalibrated. See the [frozen baseline card](docs/FROZEN_BASELINE_CARD.md).

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
4. Frozen RoBERTa embedding baseline - **complete**
5. LoRA-RoBERTa and optional full fine-tuning - **next**
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
