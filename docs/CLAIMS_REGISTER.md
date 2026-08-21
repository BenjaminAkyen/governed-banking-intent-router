# Claims Register

| Claim | Status | Evidence required |
|---|---|---|
| The project implements an MPS-aware runtime | Testable now | Device module and tests |
| The Module 2 pipeline leaves all official test rows untouched | Verified | Seed-42 manifest and split-integrity tests |
| Seed-42 word+character TF-IDF reaches 0.9053 test macro-F1 | Verified, single seed | Locked selection, test report and index-addressed predictions |
| Seed-42 frozen RoBERTa reaches 0.8964 test macro-F1 | Verified, single seed | Pinned encoder, MPS extraction evidence, locked report and predictions |
| Frozen RoBERTa outperforms TF-IDF | Not supported | Paired report; observed macro-F1 is 0.0089 lower |
| Frozen RoBERTa and TF-IDF are equivalent | Prohibited | McNemar p=0.1177 does not establish equivalence |
| LoRA improves on the baseline | Prohibited until evaluated | Three-seed locked-test results |
| Calibration improves confidence quality | Prohibited until evaluated | Validation-fitted temperature and test ECE |
| The system detects OOD requests | Overstated | Representative external evaluation; use "possible-OOD" meanwhile |
| Audit events exclude message text | Pending control implementation | Schema, privacy tests and log inspection |
| The system is production ready | Prohibited | Representative deployment evidence and organisational approval |
