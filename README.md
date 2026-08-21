<div align="center">

# Governed Banking Intent Router

**An evidence-first reference implementation for privacy-aware, risk-controlled banking support routing.**

[![Quality and evidence](https://github.com/BenjaminAkyen/governed-banking-intent-router/actions/workflows/quality.yml/badge.svg)](https://github.com/BenjaminAkyen/governed-banking-intent-router/actions/workflows/quality.yml)
[![CodeQL](https://github.com/BenjaminAkyen/governed-banking-intent-router/actions/workflows/codeql.yml/badge.svg)](https://github.com/BenjaminAkyen/governed-banking-intent-router/actions/workflows/codeql.yml)
[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12%2B-EE4C2C?logo=pytorch&logoColor=white)](pyproject.toml)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers%205-FFD21E)](pyproject.toml)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-65A30D)](pyproject.toml)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116%2B-009688?logo=fastapi&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-2EA44F)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Preview-6F42C1)](#project-status)

</div>

> [!IMPORTANT]
> This repository is a **research preview**, not production banking software. The service operates
> in `shadow_review_only` mode and can route only to human review or a security queue. It cannot
> authorize transactions, freeze accounts, authenticate customers, make fraud decisions or provide
> financial advice.

## Why this project exists

Banking support messages are short, ambiguous and sometimes security-critical. A conventional
closed-set classifier always returns a known intent—even when the request is unfamiliar,
multi-intent or unsafe to automate.

This project treats intent classification as one component of a governed decision system. It
combines model evaluation with privacy controls, calibrated confidence, uncertainty signals,
deterministic escalation, metadata-only auditing and explicit human oversight.

![Governed Banking Intent Router architecture](docs/images/system-architecture.svg)

## What the system demonstrates

| Capability | Implementation |
|---|---|
| Model comparison | TF-IDF logistic regression, frozen RoBERTa and LoRA-adapted RoBERTa under a shared evaluation contract |
| Champion–challenger control | TF-IDF remains champion until a challenger passes registered validation gates |
| Privacy before inference | Structured PII is redacted before classification; raw or redacted message text is excluded from audit and telemetry records |
| Calibrated uncertainty | Temperature scaling, entropy, confidence and selective-risk analysis use separated development and assessment roles |
| Deterministic routing | Versioned policy overrides model confidence for security-sensitive and failed-control cases |
| Governed service | Authenticated FastAPI endpoints, readiness checks, concurrency limits, rate limiting and rollback metadata |
| Operational evidence | Reproducible manifests, hash-bound artifacts, privacy-safe OpenTelemetry signals and cross-platform CI |

The request path is deliberately layered:

1. validate the request and authentication boundary;
2. redact registered PII patterns;
3. classify the redacted message;
4. apply temperature scaling and compute uncertainty signals;
5. execute deterministic risk-routing rules;
6. return a bounded recommendation for human handling; and
7. emit metadata-only audit and observability events.

## Verified evidence

### Model comparison

The official BANKING77 test split was evaluated before the later development work. It is therefore
historical evidence, not a fresh confirmation set.

| Model | Test macro-F1 | Position |
|---|---:|---|
| TF-IDF word + character logistic regression | **0.9053** | Champion |
| Frozen RoBERTa embeddings + classifier | 0.8964 | Challenger |
| LoRA-adapted RoBERTa | 0.8202 | Challenger; registered protocol underfit |

The revised three-seed LoRA study reached **0.8974 ± 0.0073 validation macro-F1**. That result is
post-test development evidence and is not presented as proof that LoRA surpassed the champion.

### Calibration, uncertainty and robustness

| Assessment | Result | Decision |
|---|---:|---|
| Mean expected calibration error, raw → temperature-scaled | 0.0470 → 0.0280 | Improvement on held-out calibration assessment roles |
| Mean selective risk | 0.0643 | Failed the registered 0.05 ceiling |
| Mean synthetic possible-OOD recall | 0.8889 | Failed the registered 0.90 target |
| Synthetic robustness acceptable-intent rate | 68.52% | Failed promotion gate |
| Expected-security routing recall | 78.57% | Failed promotion gate |
| PII expectation agreement | 100% (60/60) | Passed on the versioned synthetic pack |

These negative results are intentional project outputs: strong ranking metrics do not automatically
produce a safe operating threshold. TF-IDF remains the champion, automatic model promotion is
prohibited and production deployment is not approved.

### Engineering assurance

- **260 tests pass locally**, with a green hosted matrix across Ubuntu, macOS and Windows on Python
  3.12 and 3.13.
- Quality gates cover formatting, linting, static typing, safety-critical coverage, package builds,
  container contracts, dependency review, vulnerability scanning, SBOM generation, secret scanning
  and CodeQL analysis.
- Native Apple MPS evidence completed 36/36 synthetic API requests with 39/39 metadata-only audit
  events validated and zero message-value matches.
- Runtime profiles support explicit `cpu`, `mps` and `cuda` selection. Unsupported explicit device
  requests fail closed; real CUDA prediction-parity evidence remains pending.

See the [claims register](docs/CLAIMS_REGISTER.md) for the evidence boundary behind public claims.

## Quick start

### Requirements

- Python 3.12 or 3.13
- Git

### Install and validate

```bash
git clone https://github.com/BenjaminAkyen/governed-banking-intent-router.git
cd governed-banking-intent-router

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

pytest -q
```

On Windows PowerShell, activate the environment with
`.\.venv\Scripts\Activate.ps1` before installing the package.

To inspect the privacy, policy and API safety boundary without a model checkpoint:

```bash
pytest -q tests/test_privacy.py tests/test_policy.py tests/test_audit.py tests/test_api.py
```

### Run the research notebooks

```bash
python -m pip install -e ".[dev,notebook]"
jupyter lab output/jupyter-notebook
```

The repository does not redistribute trained adapters, cached BANKING77 data or model snapshots.
Model-backed experiments require the exact registered artifacts and hashes described in the
[model card](docs/MODEL_CARD.md) and [data card](docs/DATA_CARD.md).

### Run the governed API

After the registered LoRA artifacts are available locally:

```bash
export GOVERNED_BANKING_API_TOKEN="$(openssl rand -hex 32)"
python scripts/run_api.py
```

The development service binds to `127.0.0.1:8000` and exposes:

- `GET /health/live`
- `GET /health/ready`
- `POST /v1/route`

Deployment-specific authentication and runtime profiles are documented in
[Deployable service profiles](docs/DEPLOYMENT_PROFILES.md).

## Runtime profiles

| Profile | Purpose | Status |
|---|---|---|
| Native macOS MPS | Local Apple Silicon development and evidence generation | Verified on Apple M4 |
| Linux CPU container | Portable CPU deployment contract | Dockerfile contract verified; runtime pending |
| Linux NVIDIA CUDA container | GPU deployment and Colab parity workflow | Dockerfile contract verified; real CUDA runtime and parity pending |
| Auto selection | Select CUDA → MPS → CPU | Supported for convenience; registered evidence uses explicit devices |

MPS runs natively on macOS; it is not presented as available inside a standard Linux container.

## Repository structure

```text
.
├── src/governed_banking/     # Models, controls, service and evidence logic
├── configs/                  # Versioned model, policy, runtime and deployment contracts
├── tests/                    # Unit, integration, privacy, policy and evidence tests
├── data/                     # Manifests and explicitly labelled synthetic fixtures
├── output/jupyter-notebook/  # Reproducible research notebooks
├── reports/                  # Registered metadata-only evidence artifacts
├── deploy/                   # CPU/CUDA containers, gateway and Kubernetes templates
├── docs/                     # Technical, assurance and operational documentation
└── scripts/                  # Reproducible experiment and verification entry points
```

## Documentation

| Area | Start here |
|---|---|
| System and model | [System card](docs/SYSTEM_CARD.md) · [Model card](docs/MODEL_CARD.md) · [System boundary](docs/SYSTEM_BOUNDARY.md) |
| Data and evaluation | [Data card](docs/DATA_CARD.md) · [Evaluation protocol](docs/EVALUATION_PROTOCOL.md) · [Champion–challenger card](docs/CHAMPION_CHALLENGER_CARD.md) |
| Safety evidence | [Calibration](docs/CALIBRATION_CARD.md) · [Uncertainty and OOD](docs/UNCERTAINTY_OOD_CARD.md) · [Robustness](docs/ROBUSTNESS_EVALUATION_CARD.md) |
| Privacy and service | [Privacy and audit](docs/PRIVACY_AUDIT_CARD.md) · [Routing policy](docs/RISK_ROUTING_POLICY.md) · [Service boundary](docs/SERVICE_BOUNDARY.md) |
| Operations | [Deployment profiles](docs/DEPLOYMENT_PROFILES.md) · [Observability](docs/OBSERVABILITY.md) · [Continuous integration](docs/CONTINUOUS_INTEGRATION.md) |
| Governance | [Governance inventory](docs/governance/README.md) · [Threat model](docs/governance/governed-banking-intent-router-threat-model.md) · [NIST AI RMF mapping](docs/governance/NIST_AI_RMF_MAPPING.md) |

## Project status

The target release is **v0.2.0 research preview**. It is not labelled v1.0 because:

- uncertainty and robustness promotion gates have not passed;
- BANKING77 and authored synthetic cases are not representative production banking data;
- the served LoRA research model is not the retained TF-IDF champion;
- real CUDA prediction parity remains to be established; and
- accountable release approval has not been recorded.

This limitation is a governance feature, not a missing disclaimer: the repository is designed to
show how evidence can stop an unsafe promotion.

## Responsible use

Use this repository for research, education, reproducibility studies and development of governed
AI-routing patterns. Do not use it as a substitute for banking operations, fraud investigation,
identity verification, regulated advice or autonomous customer decision-making.

Known limitations include benchmark-only training data, regex-based PII recognition, synthetic
robustness cases and unapproved uncertainty thresholds. Any real deployment requires representative
data, privacy and security review, domain-expert validation, accountable approval and continuous
human oversight.

## Contributing

Evidence-changing contributions should preserve data separation, deterministic policies, artifact
hashes and the claims register. Open a
[GitHub issue](https://github.com/BenjaminAkyen/governed-banking-intent-router/issues) before a
substantial model, policy or governance change.

## Licence

Project code is released under the [MIT License](LICENSE). BANKING77, RoBERTa and other upstream
assets retain their own licences and redistribution terms. Review
[Third-party notices](THIRD_PARTY_NOTICES.md) before distributing datasets, checkpoints, adapters or
derived artifacts.
