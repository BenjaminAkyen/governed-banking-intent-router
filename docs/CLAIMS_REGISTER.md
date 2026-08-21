# Claims Register

| Claim | Status | Evidence required |
|---|---|---|
| The project implements an MPS-aware runtime | Testable now | Device module and tests |
| The Module 2 pipeline leaves all official test rows untouched | Verified | Seed-42 manifest and split-integrity tests |
| Seed-42 word+character TF-IDF reaches 0.9053 test macro-F1 | Verified, single seed | Locked selection, test report and index-addressed predictions |
| Seed-42 frozen RoBERTa reaches 0.8964 test macro-F1 | Verified, single seed | Pinned encoder, MPS extraction evidence, locked report and predictions |
| Frozen RoBERTa outperforms TF-IDF | Not supported | Paired report; observed macro-F1 is 0.0089 lower |
| Frozen RoBERTa and TF-IDF are equivalent | Prohibited | McNemar p=0.1177 does not establish equivalence |
| Seed-42 registered LoRA reaches 0.8202 test macro-F1 | Verified, single seed | Pinned adapter selection, locked report and predictions |
| The registered Module 5 LoRA improves on either baseline | Refuted for this protocol | It is 0.0851 below TF-IDF and 0.0762 below frozen RoBERTa in macro-F1 |
| LoRA is generally worse than the baselines | Prohibited | The registered training curves were still improving; broader candidate and independent evaluation are required |
| A revised LoRA protocol improves on the baseline | Prohibited until independently evaluated | Validation-only development, three seeds and a new independent confirmation set |
| Revised LoRA reaches 0.8974 mean validation macro-F1 across seeds 17/42/73 | Verified, post-test exploratory | Immutable manifests and validation-only seed reports |
| Module 6 improves official-test performance | Prohibited | No Module 6 official-test metrics were computed; Module 5 test was already observed |
| Module 6 establishes LoRA convergence | Not supported | Seeds 17 and 42 peaked at the eight-epoch boundary |
| Module 7 scalar temperature scaling lowers mean assessment ECE from 0.0470 to 0.0280 | Verified, post-selection/post-test exploratory | Disjoint fit/assessment registry, three seed reports and aggregate |
| Module 7 changes predicted intent labels | Refuted | Positive scalar temperature changed 0 assessment predictions across all seeds |
| Temperature scaling improves independent or production calibration | Prohibited | The assessment roles came from validation pools already used for checkpoint selection; representative untouched data are required |
| Temperature scaling improves every calibration statistic | Refuted | Seed-17 maximum calibration error increased and ECE improvement uncertainty includes zero for seed 73 |
| Module 8 uncertainty ranking separates known from synthetic possible-OOD requests with mean AUROC 0.9611 | Verified, synthetic/post-selection exploratory | Disjoint assessment roles and three seed reports |
| Module 8 meets the registered selective-routing safety gates | Refuted | All seeds exceeded 5% selective risk; seeds 17 and 42 also missed 90% possible-OOD recall |
| High possible-OOD AUROC establishes a safe operating threshold | Refuted | Strong ranking coexisted with failed locked-threshold gates |
| The selected uncertainty signal is stable across seeds | Refuted | Entropy was selected for seeds 17/73 and maximum probability for seed 42 |
| The system detects OOD requests | Overstated | Representative external evaluation; use "possible-OOD" meanwhile |
| Module 9 synthetic controls match 23 redaction and 8 routing expectations | Verified, bounded synthetic evidence | Versioned fixtures, executable report and tests |
| Module 9 audit serialization excludes fixture source and redacted values | Verified for registered fixtures | 24-event validated round trip and zero prohibited-value matches |
| Audit events exclude all personal data in production | Prohibited | Representative adversarial evaluation, production DLP and operational log inspection |
| Module 8 uncertainty can authorize an automated suggestion | Refuted by current policy | Failed gates are hash-bound as `review_signal_only`; Module 9 emits zero suggestions |
| Module 10 integrates the registered seed-42 adapter through the shadow API on MPS | Verified, local synthetic integration | Hash-bound configuration, real-MPS report and executed notebook |
| Module 10 measured 13.34 ms p50 and 299.25 ms p95 in-process sequential latency | Verified on the documented Mac only | 36-request registered run after three warm-ups; higher percentile method |
| Module 10 API and audit outputs exclude registered fixture source and redacted values | Verified for the synthetic fixture | 39-event audit round trip and zero prohibited-value matches |
| Module 10 supports production throughput or availability targets | Prohibited | Sustained concurrency, network, capacity, failure-recovery and representative workload evidence |
| Module 11 explicit CPU and MPS profiles execute on observed local hardware | Verified | Self-hashing tensor-probe reports; Apple M4 MPS and arm64 CPU metadata |
| Module 11 MPS inference completes the 12 registered synthetic cases without persisting text | Verified, bounded synthetic evidence | Hash-bound MPS prediction report and privacy-boundary validation |
| Module 11 CUDA execution works on a real NVIDIA GPU | Unverified | Registered CUDA runtime and prediction reports from Notebook 11B |
| Module 11 MPS and CUDA predictions satisfy the registered parity gates | Unverified | Independent backend reports and passing comparison; no tolerance changes after observation |
| Cross-device parity establishes model quality or production fitness | Prohibited | Parity evaluates numerical and routing consistency only; representative independent evaluation is required |
| TF-IDF is the current Module 12 champion | Verified registry decision | Best historical like-for-like test macro-F1 and hash-bound champion registry |
| Revised LoRA is eligible to replace TF-IDF | Not supported | Existing validation/calibration evidence is post-test development evidence and uncertainty gates failed |
| A challenger may be promoted using another BANKING77 test run | Prohibited | The official test is already observed; a new locked external evaluation source is mandatory |
| Module 12 automatically promotes a model that passes numeric gates | Refuted by design | Passing creates eligibility for human review; automatic registry mutation is prohibited |
| Module 10 serves the approved champion | Refuted | The shadow service uses revised LoRA while TF-IDF remains champion |
| The Module 13 synthetic pack covers all ten registered robustness families | Verified, authored synthetic evidence | 60 hash-locked cases, six per family, with complete case-level routing, risk, escalation, provenance and licence annotations |
| The Module 13 pack copies or closely reproduces a pinned BANKING77 message | Refuted by the registered lexical check | Zero exact or character-five-gram near matches across 13,083 pinned train/test rows at the locked 0.85 Jaccard threshold |
| The Module 10 LoRA service meets the Module 13 acceptable-intent gate | Refuted | 37/54 in-scope cases were acceptable (68.52%) versus the preregistered 80% minimum |
| The Module 10 LoRA service meets the Module 13 expected-security routing gate | Refuted | 11/14 expected-security cases reached security (78.57%) versus the preregistered 100% requirement |
| Module 13 PII and no-suggestion control checks passed on the synthetic pack | Verified, bounded synthetic evidence | 60/60 PII expectations matched, zero suggestions, and no input/redacted value or message hash was persisted |
| Module 13 establishes representative real-world robustness | Prohibited | Authored synthetic cases support failure discovery only; governed real-world data and independent assessment are required |
| Module 14 exposes separate liveness, readiness and versioned routing endpoints | Verified in native MPS smoke and focused tests | Hash-bound native report plus lifecycle/service tests |
| Module 14 native service loads the registered adapter on real MPS without fallback | Verified on the documented Apple M4 only | Native profile and metadata-only 12-check smoke report |
| Module 14 Linux CPU container executes successfully | Unverified | Digest-pinned image build and runtime smoke evidence on Linux CPU |
| Module 14 Linux CUDA container executes successfully | Unverified | Digest-pinned CUDA image and real NVIDIA runtime smoke evidence |
| Standard Linux Docker on a Mac provides normal PyTorch MPS acceleration | Refuted by design | MPS is restricted to native macOS; Linux profiles allow only CPU or CUDA |
| Module 14 in-process rate limiting is a distributed fleet quota | Prohibited | An organisation gateway or shared limiter is required for fleet-wide enforcement |
| Module 14 gateway template proves an identity provider is configured | Refuted | The contract contains explicit replacement values and `identity_provider_configured: false` |
| Module 14 provides rollback by hot-swapping model files | Refuted by design | Rollback replaces an immutable process/container revision and versioned model bundle |
| The system is production ready | Prohibited | Representative deployment evidence and organisational approval |
