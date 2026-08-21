# Claims register

This register defines what may be said publicly about the project. A claim must be no broader than
its cited evidence. Where this register and promotional copy conflict, this register controls.

## Status definitions

| Status | Meaning |
|---|---|
| Verified | The exact claim is supported by committed evidence and executable checks |
| Verified, bounded | Supported only for the named fixture, device or environment |
| Unverified | Required evidence has not been produced |
| Refuted | Existing evidence contradicts the claim |
| Not supported | Available evidence is insufficient to decide |
| Prohibited | The claim materially exceeds the research or approval boundary |

## Data and model performance

| Claim | Status | Evidence or constraint |
|---|---|---|
| The prepared dataset retains all 3,080 official BANKING77 test rows and removes normalized train/test overlap from development data | Verified | `data/manifests/banking77-seed-42.json` and split-integrity tests |
| Seed-42 TF-IDF reaches 0.9053 official-test macro-F1 | Verified, single seed | `reports/baseline/tfidf-logreg-test.json` |
| Seed-42 frozen RoBERTa reaches 0.8964 official-test macro-F1 | Verified, single seed | `reports/frozen-roberta/test.json` |
| Frozen RoBERTa outperforms TF-IDF | Refuted for the registered comparison | Observed macro-F1 is 0.0089 lower |
| Frozen RoBERTa and TF-IDF are equivalent | Not supported | McNemar p=0.1177 does not establish equivalence |
| The original seed-42 LoRA protocol reaches 0.8202 official-test macro-F1 | Verified, single seed | `reports/lora-roberta/test.json` |
| The original LoRA protocol improves on either baseline | Refuted | Macro-F1 is 0.0851 below TF-IDF and 0.0762 below frozen RoBERTa |
| LoRA is generally inferior to lexical or frozen-embedding models | Prohibited | One underfit registered protocol cannot support an architecture-wide claim |
| Revised LoRA reaches 0.8974 mean validation macro-F1 across seeds 17, 42 and 73 | Verified, post-test development | `reports/multiseed-lora/validation-aggregate.json` |
| Revised LoRA improves official-test performance | Prohibited | No fresh test result exists; the official test was already observed |
| Revised LoRA convergence is established | Not supported | Two seeds peaked at the eight-epoch boundary |
| TF-IDF is the current champion | Verified | `configs/champion_challenger.yaml` and generated registry |
| The shadow API serves the champion | Refuted | The API serves revised LoRA while TF-IDF remains champion |
| A challenger can be promoted using another BANKING77 test run | Prohibited | Promotion requires a new, locked external evaluation source |

## Calibration, uncertainty and robustness

| Claim | Status | Evidence or constraint |
|---|---|---|
| Scalar temperature scaling lowers mean assessment ECE from 0.0470 to 0.0280 | Verified, post-selection | Disjoint fit/assessment roles and `reports/calibration/temperature-scaling-aggregate.json` |
| Temperature scaling changes predicted labels | Refuted | Zero assessment predictions changed across all seeds |
| The model is calibrated for production traffic | Prohibited | Assessment rows came from development validation pools; no representative external data exists |
| Uncertainty scores rank synthetic possible-OOD below known requests with mean AUROC 0.9611 | Verified, synthetic and post-selection | `reports/uncertainty/selective-ood-aggregate.json` |
| The locked uncertainty operating points meet the safety gates | Refuted | All seeds exceed 5% selective risk; two miss 90% possible-OOD recall |
| High possible-OOD AUROC makes automated routing safe | Refuted | Strong ranking coexists with failed operating-point gates |
| The system detects real-world OOD requests | Prohibited | Only authored synthetic possible-OOD cases were evaluated |
| The synthetic robustness pack is lexically independent of pinned BANKING77 at the registered threshold | Verified, bounded | Zero exact or character-five-gram matches across 13,083 rows at Jaccard ≥0.85 |
| The LoRA service meets the acceptable-intent robustness gate | Refuted | 68.52% observed against an 80% minimum |
| The LoRA service meets the expected-security routing gate | Refuted | 78.57% observed against a 100% requirement |
| The robustness pack establishes representative real-world quality | Prohibited | The pack contains 60 project-authored synthetic cases |

## Privacy, policy and service controls

| Claim | Status | Evidence or constraint |
|---|---|---|
| Structured-PII controls match the registered 23-case fixture | Verified, bounded | 23/23 expectations and 11/11 detector classes exercised |
| Routing controls match the registered eight-case fixture | Verified, bounded | 8/8 expected actions and queues |
| Registered audit round trips contain no fixture source or redacted values | Verified, bounded | 24/24 events validated with zero prohibited-value matches |
| Audit and telemetry exclude all personal data in production | Prohibited | Representative adversarial evaluation and production sink inspection do not exist |
| Uncertainty can authorize `suggest_queue` | Refuted by design | The policy is review-only and the API schema cannot return that action |
| The shadow API loads the registered seed-42 LoRA adapter on MPS | Verified, bounded | Hash-bound Apple M4 integration report |
| The API completed 36/36 measured synthetic requests with 13.34 ms p50 and 299.25 ms p95 latency | Verified on the documented Mac only | Sequential in-process run after warm-up |
| The local latency result establishes throughput, capacity or an SLO | Prohibited | No networked concurrency, sustained load or failure-recovery study exists |
| Application telemetry emits only registered attributes in the Apple M4 smoke test | Verified, bounded | 14 metrics, two span names and synthetic privacy canaries |
| A real Collector, Prometheus or trace backend is operationally validated | Unverified | The recorded observability run used an in-memory exporter |

## Runtime, deployment and governance

| Claim | Status | Evidence or constraint |
|---|---|---|
| Explicit CPU and MPS profiles execute on observed local hardware | Verified, bounded | Self-hashing runtime reports; Apple M4 MPS and arm64 CPU |
| Explicit CUDA execution works on a real NVIDIA GPU | Unverified | Registered CUDA runtime and prediction reports are absent |
| MPS and CUDA predictions meet registered parity tolerances | Unverified | Real CUDA and comparison reports are absent |
| Linux CPU model serving is verified | Unverified | Dockerfile validation is not runtime evidence |
| Linux CUDA model serving is verified | Unverified | A digest-pinned image and real NVIDIA runtime smoke are required |
| The gateway template proves identity-provider configuration | Refuted | The template contains replacement values and `identity_provider_configured: false` |
| The in-process rate limiter is a fleet-wide quota | Prohibited | A gateway or shared limiter is required |
| Cross-platform CPU CI and CodeQL pass for the current public branch | Verified | Hosted workflow results and live README badges |
| The project is production ready | Prohibited | Representative data, passed safety gates, aligned champion service, target-runtime evidence and accountable approval are missing |
| The governance documents approve v0.2.0 | Refuted | The release record remains pending with no recorded approvals |
| The NIST AI RMF mapping is certification or full conformance | Prohibited | It is a voluntary-framework current-state crosswalk with explicit gaps |
| Human oversight is operationally validated | Unverified | Named assignments, training, capacity, response targets and exercises are absent |

## Maintenance

Update this register whenever a result, system boundary, deployment status or approval changes.
Preserve refuted and prohibited claims when they remain relevant; deleting an inconvenient result
does not change the evidence.
