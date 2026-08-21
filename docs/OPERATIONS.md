# Operations

| Attribute | Value |
|---|---|
| Status | Research preview; private or loopback operation only |
| Service mode | `shadow_review_only` |
| Production identity provider | Not configured |
| Production audit/telemetry backend | Not configured |
| Production approval | No |

## Runtime profiles

| Profile | Intended environment | Device behaviour | Evidence status |
|---|---|---|---|
| `configs/deployment/native-mps.yaml` | Native Apple Silicon macOS | Explicit MPS; fail if unavailable | Apple M4 smoke verified |
| `configs/deployment/linux-cpu.yaml` | Linux CPU container | Explicit CPU | Dockerfile contract verified; serving runtime pending |
| `configs/deployment/linux-cuda.yaml` | Linux NVIDIA container | Explicit CUDA; fail if unavailable | Dockerfile contract verified; real runtime pending |

Standard Linux containers on macOS do not expose PyTorch MPS. Use the native profile for Apple
Silicon acceleration. Explicit device requests never silently fall back.

## Native development service

Install the project from the repository root and make the registered, unredistributed model
artifacts available locally. Then create an ephemeral development token:

```bash
export GOVERNED_BANKING_DEV_API_TOKEN="$(openssl rand -hex 32)"
python scripts/run_deployment_api.py \
  --profile configs/deployment/native-mps.yaml
```

The native profile binds only to `127.0.0.1:8000`. Check readiness before routing synthetic input:

```bash
curl --fail http://127.0.0.1:8000/health/ready

curl --request POST http://127.0.0.1:8000/v1/route \
  --header "Authorization: Bearer $GOVERNED_BANKING_DEV_API_TOKEN" \
  --header "Content-Type: application/json" \
  --data '{"message":"When will my replacement card arrive?"}'
```

Use only synthetic messages. Do not paste customer or account information into the research
service.

## Service lifecycle and admission control

`/health/live` reports process liveness. `/health/ready` returns success only after the registered
model, device, policy and audit store are ready. The versioned routing endpoint is `/v1/route`;
there is no unversioned route.

Admission is ordered:

1. authenticate the caller;
2. apply the per-principal, per-process fixed-window rate limit;
3. reserve bounded inference and queue capacity;
4. enforce queue-wait and request deadlines; and
5. retain the capacity slot after a caller timeout until background inference actually finishes.

On shutdown, the process stops admission, drains in-flight work within the configured deadline,
closes the audit store and releases device cache. One worker is used per process to avoid loading
multiple model copies; scale-out requires separately governed replicas.

## Authentication boundary

Native development accepts only `GOVERNED_BANKING_DEV_API_TOKEN`. Container profiles require an
organisation-managed gateway that validates the caller's OIDC token, strips caller-supplied
identity headers and injects a verified subject, issuer and rotated origin assertion.

The contract in `deploy/gateway/identity-gateway-contract.yaml` contains deliberate replacement
values. It does not prove that an identity provider, private origin, TLS policy or secret rotation
has been configured. Never expose the origin directly to the public internet.

## Container contracts

CPU and CUDA Dockerfiles require an explicitly approved base image. Use digest-pinned images in a
reviewed environment:

```bash
docker build \
  --build-arg PYTHON_CPU_IMAGE='REPLACE_WITH_APPROVED_PYTHON_IMAGE_DIGEST' \
  --file deploy/docker/cpu.Dockerfile \
  --tag governed-router-cpu:research .

docker build \
  --build-arg PYTORCH_CUDA_IMAGE='REPLACE_WITH_APPROVED_PYTORCH_CUDA_IMAGE_DIGEST' \
  --file deploy/docker/cuda.Dockerfile \
  --tag governed-router-cuda:research .
```

Model weights and Hugging Face snapshots are excluded from Git and the image context. Mount the
registered artifacts read-only and provide durable writable audit storage separately. CUDA
requires an NVIDIA container runtime, compatible driver and real GPU. Missing or incompatible
CUDA makes the service non-ready.

The Kubernetes templates use a private `ClusterIP`, non-root/read-only container settings, network
policy, health probes, persistent mounts and retained rollout history. Replace every
`REPLACE_WITH_...` value and validate the resulting manifests before use. The templates are not
safe to apply unchanged.

## Audit storage

The audit-store protocol accepts only validated metadata events. Customer text, redacted text,
tokens, names, account data, headers and message hashes remain prohibited regardless of backend.

The bundled JSONL store is a local research implementation. A deployment backend requires managed
authentication, authorisation, encryption, append integrity, retention/deletion controls,
monitoring and tested failure recovery. Audit append failure rejects the route response.

## Observability

Manual OpenTelemetry instrumentation records:

- request, error and latency metrics;
- model-loading duration and selected device;
- model, policy and service versions;
- human-review and security-escalation rates;
- categorical redaction counts;
- uncertainty distribution; and
- routing-distribution change.

Attributes use fixed allowlists. The application excludes message text, redacted text, matched PII,
headers, identities, request/correlation IDs, baggage, exception bodies and message hashes. Exact
intent labels are also excluded from online telemetry because they can reveal sensitive issue
types.

The application exports only to a credential-free HTTP(S) origin. Plaintext OTLP is restricted to
a loopback Collector; backend credentials belong to the Collector. The Collector's redaction
processor is defence in depth and does not replace application allowlisting.

Local Collector example:

```bash
otelcol-contrib --config deploy/observability/collector-local.yaml
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317
```

The verified Apple M4 run emitted all 14 registered metrics and both span names for 20 synthetic
routes without prohibited values. It used an in-memory exporter; no real Collector, Prometheus
deployment, trace backend, alert or retention policy has been validated.

## Continuous integration

The `quality-and-evidence` workflow runs:

- CPU tests on Ubuntu, macOS and Windows with Python 3.12 and 3.13;
- Ruff linting and prospective formatting for changed Python files;
- scoped strict type checking over safety-critical boundaries;
- separate privacy, policy, integration and unit reports;
- registered module-level coverage floors;
- source and wheel builds with isolated installation;
- a lightweight restricted container smoke test;
- structural validation of CPU and CUDA deployment Dockerfiles;
- dependency and vulnerability audits, CycloneDX SBOM generation and secret scanning; and
- a metadata-only control-path benchmark.

CodeQL runs separately. External GitHub Actions are pinned to commit SHAs, checkout credentials are
not persisted, `pull_request_target` is not used and workflow permissions are least-privilege.

Coverage floors are regression controls, not proof of safety. The CPU/CUDA Dockerfile checks are
not model-serving runtime evidence. GitHub repository settings—branch protection, push protection,
Dependabot and administrator policy—must be verified independently of workflow files.

## Publication and release hygiene

Run the hygiene gate from any directory inside the checkout:

```bash
python scripts/check_publication_hygiene.py --execute-notebook-setup
```

The check scans notebooks and the public tree for machine-specific paths, credential signatures,
tracked environments, data, checkpoints, logs and missing ignore rules. It executes notebook setup
cells from a nested directory to verify repository discovery without rerunning experiments.

Before release:

1. run the full clean-clone test suite and hygiene gate;
2. confirm required hosted workflows and security settings;
3. preserve failed gates and unverified environments in the claims register;
4. verify artifact and configuration hashes;
5. review third-party terms and the SBOM;
6. attach durable checksums and approved evidence to the release; and
7. record accountable approval without fabricating absent reviewers.

## Rollback and incident handling

Rollback replaces the complete immutable process or container revision and its versioned model
bundle. Hot mutation is prohibited. Re-run readiness and synthetic safety checks after rollback.

Follow the [rollback procedure](governance/ROLLBACK_PROCEDURE.md) and
[incident-response process](governance/INCIDENT_RESPONSE.md). A suspected privacy leak, unapproved
action, artifact mismatch or missed mandatory security route requires containment and review.

## Evidence and remaining gaps

Verified native evidence is recorded under `reports/service/`, `reports/deployment/` and
`reports/observability/`. Current gaps include:

- real CUDA prediction and MPS/CUDA parity;
- Linux CPU and CUDA serving-runtime smoke tests;
- a configured identity gateway and private deployment origin;
- durable audit and telemetry backends;
- representative capacity, recovery and monitoring baselines;
- trained and staffed human-review operations; and
- production approval.
