# Governed Banking Intent Router

A Mac-first research implementation of a privacy-aware banking support router. The system uses
LoRA-adapted RoBERTa as one component in a larger decision pipeline that includes calibration,
uncertainty handling, deterministic escalation and metadata-only audit events.

> **Status: lexical, frozen-encoder and LoRA baselines evaluated; no model is production approved.**

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

Module 6 evaluates the revised rank-8 LoRA training protocol across seeds 17, 42 and 73 using only
train and validation data. It is explicitly post-test exploratory because Module 5's BANKING77 test
result is already known. The runner has no test-evaluation mode and cannot support a new test
improvement claim.

Run the local checks:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev,notebook]"
ruff check .
pytest --cov=governed_banking --cov-report=term-missing
```

Prepare the multi-seed manifests and reproduce Module 6 (about 85 minutes on the documented Mac):

```bash
python scripts/prepare_banking77.py --offline
python scripts/prepare_multiseed_manifests.py
python scripts/run_multiseed_lora.py --offline --resume
```

Then open
[`06-multiseed-validation-only-lora.ipynb`](output/jupyter-notebook/06-multiseed-validation-only-lora.ipynb) in VS
Code and run it with the project virtual environment's `Python 3` kernel.

## Verified Module 6 validation evidence

| Seed | Best epoch | Best validation macro-F1 |
|---:|---:|---:|
| 17 | 8 | 0.9006 |
| 42 | 8 | 0.9026 |
| 73 | 6 | 0.8890 |
| **Mean ± sample SD** | — | **0.8974 ± 0.0073** |

These values describe validation behaviour only. They cannot be compared with Module 5's official
test result as evidence of improved generalisation. Seeds 17 and 42 still peaked at the eight-epoch
boundary, so full convergence is not established. See the
[multi-seed exploratory card](docs/MULTISEED_EXPLORATORY_CARD.md).

## Verified Module 5 comparison

Both models were evaluated on the same 3,080 official test rows:

| Metric | TF-IDF | Frozen RoBERTa | LoRA RoBERTa |
|---|---:|---:|---:|
| Accuracy | **0.9052** | 0.8961 | 0.8273 |
| Macro-F1 | **0.9053** | 0.8964 | 0.8202 |
| Weighted-F1 | **0.9053** | 0.8964 | 0.8202 |
| Log-loss | 0.4970 | **0.3990** | 0.7746 |
| Top-3 accuracy | 0.9705 | **0.9718** | 0.9497 |

Frozen RoBERTa corrected 135 rows missed by TF-IDF, while TF-IDF corrected 163 rows missed by
RoBERTa. Their paired exact McNemar p-value is 0.1177, which does not establish equivalence. The
results suggest complementarity, but not that generic frozen embeddings beat the lexical baseline.

The rank-8 LoRA candidate trained 944,717 parameters (0.7519%) and won validation, but finished
0.0851 macro-F1 below TF-IDF and 0.0762 below frozen RoBERTa. Its validation curve improved in every
epoch, so the defensible conclusion is that this registered protocol underfit—not that LoRA is
generally inferior. Probabilities remain uncalibrated. See the
[LoRA baseline card](docs/LORA_BASELINE_CARD.md).

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
5. LoRA-RoBERTa registered baseline - **complete; negative result preserved**
6. Multi-seed manifests, revised development protocol and aggregation - **complete; exploratory**
7. Calibration and selective prediction - **next**
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
