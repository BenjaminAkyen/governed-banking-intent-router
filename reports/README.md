# Reports

Verified aggregate tables and publication figures are generated here. Reports and prediction
records must not contain source message text.

Module 3 commits the validation-selection report, locked-test report and 3,080 index-addressed test
predictions under `baseline/`. The fitted joblib model stays in ignored local storage under
`artifacts/`; its SHA-256 digest is recorded in the test report. Joblib files are executable
serialization and must never be loaded from an untrusted source.

Module 4 stores frozen-RoBERTa selection, extraction, locked-test, prediction and paired-comparison
evidence under `frozen-roberta/`. The model snapshot and embedding arrays remain in ignored local
storage; reports retain their hashes and text-free metadata.

Module 5 stores LoRA validation selection, locked test, predictions and paired comparisons under
`lora-roberta/`. Adapter weights stay in ignored local storage under `artifacts/lora-roberta/`;
their SafeTensors hashes and sizes are preserved in the committed selection lock. The result is a
documented negative finding: the registered three-epoch protocol underfit and did not beat either
baseline.

Module 6 stores three validation-only seed reports and their aggregate under `multiseed-lora/`.
These are explicitly post-test exploratory artifacts: they contain no official-test metrics or
predictions and cannot support an improvement claim. Local adapters remain ignored under
`artifacts/multiseed-lora/`; their hashes are preserved in each seed report.

Module 7 stores three temperature-scaling reports and their aggregate under `calibration/`. Each
report fits temperature on `temperature_fit` and calculates the published metrics only on disjoint
`calibration_assessment` rows. Reports include fixed-width reliability bins and paired bootstrap
intervals but no message text, logits or official-test metrics. These artifacts are post-selection,
post-test exploratory evidence—not independent model evaluation.

Module 8 stores three locked-threshold assessments and their aggregate under `uncertainty/`.
Development evidence records signal and threshold selection; assessment evidence records
risk-coverage, selective risk, possible-OOD recall, ranking metrics, per-domain failures and
bootstrap intervals. Reports contain no source message text or official-test metrics. The failed
acceptance gates are preserved as evidence rather than relaxed after assessment.

Module 9 stores `governance/module9-controls.json`. It summarizes 23 synthetic redaction cases,
eight metadata-only routing cases and a 24-event audit round trip. The report contains fixture and
implementation hashes, aggregate counts, permission checks and limitations—but no source message,
redacted message, message hash, exact input length, model inference or official-test metric.
