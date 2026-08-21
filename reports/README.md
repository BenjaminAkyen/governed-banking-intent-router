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
