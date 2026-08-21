# Module 13 synthetic robustness evaluation card

## Decision

The registered Module 10 LoRA research service **failed** the locked Module 13 synthetic robustness
assessment. It remains a challenger and is not approved for production routing. This result does
not change Module 12's retained TF-IDF champion.

The test exposed both forms of safety error:

- three expected-security cases were under-routed to ordinary human review;
- three non-banking cases were over-routed to security after the closed-set classifier forced them
  into security labels.

The correct response is to improve the system and evaluate it on new locked evidence—not to relax
the gates or tune against this observed pack.

## Evaluated system and boundary

The run used the hash-bound seed-42 LoRA-RoBERTa adapter, scalar temperature, structured-PII
redactor and deterministic Module 9 routing policy already integrated by Module 10. Inference ran
on a real Apple MPS backend with CPU fallback prohibited.

The evaluated LoRA service is not the approved champion. The pack is project-authored synthetic
data, not a sample of customer traffic. BANKING77 train and test messages were read only during the
separate leakage check; the official test was not accessed or scored during model assessment.

## Versioned evaluation pack

`data/robustness/v1/cases.jsonl` contains 60 cases—six in each primary family:

1. typographical errors;
2. speech-transcription errors;
3. paraphrases;
4. multi-intent requests;
5. short and ambiguous requests;
6. code-switching;
7. PII-bearing requests;
8. prompt-like manipulation;
9. non-banking and adversarial requests;
10. high-risk fraud, lost-device and compromised-account requests.

Every record contains a stable ID, one primary family, overlapping robustness tags, a semantic
group, an acceptable intent set, an out-of-scope flag, expected routing action, risk severity,
mandatory escalation decision and reasons, expected PII types, provenance and licence metadata.
Ambiguous requests may have multiple acceptable labels; non-banking requests have none.

All cases declare:

- origin: `project_authored_synthetic`;
- creation: `human_directed_ai_assisted_scenario_authoring`;
- customer data: false;
- production derivation: false;
- BANKING77-text derivation: false;
- licence: MIT, with the repository copyright reference.

## Pack integrity and leakage result

Before inference, the pack was hash-locked and checked with the project's Unicode NFKC,
case-folding and whitespace normalization. Exact matches and character five-gram Jaccard overlap
at or above 0.85 were prohibited.

| Construction check | Result |
|---|---:|
| Synthetic cases | 60 |
| Primary families | 10; six cases each |
| Ambiguous-label cases | 23 |
| Out-of-scope cases | 6 |
| Structured-PII detector types exercised | 11/11 |
| Internal exact / near duplicates | 0 / 0 |
| Pinned BANKING77 rows scanned | 13,083 |
| BANKING77 exact / near matches | 0 / 0 |

The lexical near-duplicate method cannot prove the absence of every semantic derivation. It is a
specific reproducible leakage control, not an originality guarantee.

## Preregistered assessment gates

The following gates were committed before real-model inference:

| Gate | Required | Observed | Result |
|---|---:|---:|---|
| In-scope acceptable-intent rate | ≥ 80% | 68.52% (37/54) | Fail |
| Expected-security routing recall | 100% | 78.57% (11/14) | Fail |
| Overall routing-action agreement | ≥ 95% | 90.00% (54/60) | Fail |
| PII expectation agreement | 100% | 100% (60/60) | Pass |
| Suggestion actions | 0 | 0 | Pass |
| Original/redacted values or message hashes in report | 0 | 0 | Pass |

Thirty-one of 60 cases were at or above the seed-42 experimental uncertainty threshold. This is
reported descriptively only. Module 8 failed its registered gates, so the threshold remains a
review signal and cannot authorize automation.

## Performance by primary family

Acceptable-intent rate applies only to in-scope cases. The six non-banking cases intentionally
have no acceptable BANKING77 label.

| Primary family | Acceptable intent | Routing action |
|---|---:|---:|
| Multi-intent | 100.00% | 100.00% |
| High-risk security | 83.33% | 83.33% |
| Short and ambiguous | 83.33% | 100.00% |
| Speech-transcription error | 83.33% | 100.00% |
| Code-switching | 66.67% | 83.33% |
| Paraphrase | 66.67% | 100.00% |
| Prompt-like manipulation | 66.67% | 100.00% |
| PII-bearing | 50.00% | 100.00% |
| Typographical error | 16.67% | 83.33% |
| Non-banking/adversarial | Not applicable | 50.00% |

These percentages summarize six authored cases per family. They are failure-discovery evidence,
not stable estimates for a user population.

## Safety interpretation

The under-routing failures show that a security policy driven mainly by the predicted intent can
inherit classifier errors. Lost-card and lost-device language needs an independent, conservative
security trigger rather than relying only on the 77-way top-1 label.

The over-routing failures show the opposite problem: a closed-set classifier has no native
non-banking class and can force unrelated input into a sensitive banking label. Shadow mode
prevents automatic customer action, but unnecessary security escalations would still waste scarce
operations capacity.

## Engineering response

The next challenger iteration should:

1. add training-only typographical and speech-noise augmentation with source-group isolation;
2. add an explicit out-of-scope/abstention component rather than treating maximum probability as a
   production-approved detector;
3. add deterministic lexical or separately validated security triggers for lost card, lost phone
   and unauthorized-transaction language;
4. score transformation consistency across clean/noisy semantic groups;
5. keep this pack frozen as observed development evidence;
6. use a new locked synthetic version for regression and appropriately governed real-world data
   before representative claims.

## Reproduction

From the repository root, with the verified local data and private adapter already available:

```bash
python scripts/build_robustness_pack.py
python scripts/run_robustness_evaluation.py
pytest -q tests/test_robustness.py
```

The second command explicitly requires real MPS and fails if that backend is unavailable. The
metadata-only notebook is
`output/jupyter-notebook/13-synthetic-robustness-evaluation.ipynb`.

## Claims that remain prohibited

This module does not establish production accuracy, privacy recall, fairness, resilience to real
attacks, operational security performance or regulatory compliance. Representative claims require
lawfully obtained, appropriately governed real-world data, independent evaluation and operational
approval.
