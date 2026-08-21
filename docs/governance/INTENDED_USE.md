# Intended and prohibited use

## Release boundary

Version v0.2.0 is a research preview. It may be cloned, inspected, tested and adapted using public
benchmark data and labelled synthetic fixtures. It is not a production banking product, a fraud
system or a customer-decision system.

## Permitted uses

- Reproduce the registered BANKING77 experiments.
- Compare TF-IDF, frozen RoBERTa and LoRA research baselines.
- Study calibration, uncertainty, selective prediction and possible-OOD evaluation.
- Test structured-PII minimisation and metadata-only audit design with synthetic values.
- Exercise deterministic escalation and human-oversight workflows in a local or private sandbox.
- Evaluate MPS, CPU or CUDA portability without making production-quality claims.
- Teach and review AI engineering, MLOps, security and governance techniques.

All demonstrations must use the pinned public benchmark or labelled synthetic requests. Any
service must remain local or private behind an organisation-managed gateway.

## Prohibited uses

The research preview must not be used to:

- process real customer or bank data;
- expose the origin service directly to the public internet;
- route customer requests without human review;
- execute, approve, decline, reverse or recommend a financial transaction;
- authenticate a person, determine account ownership or recover credentials;
- freeze, close, restrict or otherwise change an account;
- make a fraud, credit, eligibility, complaints, vulnerability or safeguarding decision;
- provide financial, legal, regulatory or investment advice;
- treat a probability or uncertainty score as proof of correctness or safety;
- automatically promote, retrain or deploy a model from feedback;
- claim fairness, regulatory compliance, production readiness or representative performance; or
- deploy the shadow LoRA model as though it were the registered TF-IDF champion.

## Scope-change rule

Real data, public exposure, multi-tenancy, autonomous routing, customer-impacting decisions or a
production pilot are material scope changes. Work must stop until the risk register, threat model,
privacy/legal assessment, validation plan, human-oversight capacity and change approval are updated.

## Enforcement and reporting

The accountable system owner, model risk reviewer and security/privacy reviewer each have release
veto authority. Suspected misuse is handled under `docs/governance/INCIDENT_RESPONSE.md`. The MIT
licence permits reuse, but it does not convert prohibited project claims into approved or safe use.
