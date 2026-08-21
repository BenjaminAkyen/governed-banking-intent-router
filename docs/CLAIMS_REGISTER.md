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
| Calibration improves confidence quality | Prohibited until evaluated | Validation-fitted temperature and test ECE |
| The system detects OOD requests | Overstated | Representative external evaluation; use "possible-OOD" meanwhile |
| Audit events exclude message text | Pending control implementation | Schema, privacy tests and log inspection |
| The system is production ready | Prohibited | Representative deployment evidence and organisational approval |
