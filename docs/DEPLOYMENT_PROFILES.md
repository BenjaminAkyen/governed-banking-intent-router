# Module 14 Deployable Service Profiles

## Decision and evidence status

Module 14 packages the existing Module 10 LoRA research service behind a stricter lifecycle and
deployment boundary. It does **not** promote that model, make it the champion, or approve production
use. TF-IDF remains the registered champion, the external evaluation lock is missing, Module 13's
classification and safety-routing gates failed, and real CUDA prediction evidence is still pending.

The only executed Module 14 profile is native macOS MPS. The CPU and CUDA container definitions are
portable deployment templates whose schema and safety properties are tested locally; they have not
been built or executed in the recorded environment.

| Profile | Execution boundary | Device rule | Module 14 evidence |
|---|---|---|---|
| `native-mps.yaml` | Native macOS process, loopback only | Explicit MPS; fail if unavailable; CPU fallback prohibited | Real Apple M4 smoke test passed |
| `linux-cpu.yaml` | Standard Linux container | Explicit CPU | Template validated; container execution pending |
| `linux-cuda.yaml` | Linux NVIDIA container | Explicit CUDA; fail if unavailable | Template validated; real CUDA execution pending |

Normal PyTorch MPS acceleration is not offered inside either Linux container. On a Mac, use the
native MPS profile. Docker Desktop can run the Linux CPU image, but that does not expose Apple MPS
to the container.

## Service contract

The deployment API exposes only:

- `GET /health/live`: process liveness; it does not assert model readiness.
- `GET /health/ready`: returns HTTP 200 only after the model and audit store load on the registered
  device; otherwise it returns HTTP 503.
- `POST /v1/route`: authenticated, versioned governed routing.

OpenAPI and interactive documentation remain disabled. The unversioned `/route` path is absent.
Every HTTP response receives fresh canonical UUID request and correlation identifiers. A caller may
provide only a canonical UUID correlation identifier; arbitrary strings are rejected so customer
data cannot be smuggled into that metadata field.

Admission controls are intentionally layered:

1. authenticate the caller;
2. apply a per-principal, per-process fixed-window rate limit;
3. cap concurrent inference and queue depth;
4. bound queue wait and request wait;
5. retain the capacity slot if a request times out until its background inference really ends.

The in-process limiter is a protective last line, not a distributed fleet-wide quota. A deployed
gateway must enforce the organisation's global rate and authorization policy.

On shutdown the service stops accepting work, waits for in-flight work up to the profile deadline,
releases accelerator cache, closes the audit store and stops. Uvicorn runs one worker per process;
horizontal replicas provide scale without loading multiple large models into one process.

## Authentication boundaries

Native development uses only `GOVERNED_BANKING_DEV_API_TOKEN`. The legacy Module 10 token variable
is deliberately not accepted by the Module 14 route.

Container profiles do not accept end-user bearer tokens. An organisation-managed API gateway must
validate the user's OIDC token, strip all client-supplied identity and origin-authentication
headers, then inject:

- `X-Authenticated-Subject` from the verified `sub` claim;
- `X-Authenticated-Issuer` from the verified `iss` claim;
- `X-Governed-Gateway-Assertion` from a rotated secret unavailable to end users.

The origin service must remain private and reachable only from the gateway. The vendor-neutral
contract is in `deploy/gateway/identity-gateway-contract.yaml`; it is not a claim that an identity
provider has already been configured. Replace the example issuer in a reviewed environment-specific
deployment profile before using a real identity provider.

## Native MPS development

Create a fresh development token and launch from the repository root:

```bash
export GOVERNED_BANKING_DEV_API_TOKEN="$(openssl rand -hex 32)"
python scripts/run_deployment_api.py \
  --profile configs/deployment/native-mps.yaml
```

The process binds only to `127.0.0.1:8000`. Verify readiness before sending synthetic input:

```bash
curl --fail http://127.0.0.1:8000/health/ready
curl --request POST http://127.0.0.1:8000/v1/route \
  --header "Authorization: Bearer $GOVERNED_BANKING_DEV_API_TOKEN" \
  --header "Content-Type: application/json" \
  --data '{"message":"When will my replacement card arrive?"}'
```

If MPS is unavailable, startup remains non-ready. The profile never silently selects CPU.

## Linux container builds

Both Dockerfiles require an explicitly supplied, organisation-approved base image. Use a digest,
not a mutable tag. The CUDA base must already contain a CUDA-enabled PyTorch build compatible with
the NVIDIA driver available at runtime.

```bash
docker build \
  --build-arg PYTHON_CPU_IMAGE='REPLACE_WITH_APPROVED_PYTHON_IMAGE_DIGEST' \
  --file deploy/docker/cpu.Dockerfile \
  --tag governed-router-cpu:module14 .

docker build \
  --build-arg PYTORCH_CUDA_IMAGE='REPLACE_WITH_APPROVED_PYTORCH_CUDA_IMAGE_DIGEST' \
  --file deploy/docker/cuda.Dockerfile \
  --tag governed-router-cuda:module14 .
```

Model weights and the offline Hugging Face snapshot are intentionally absent from the Git
repository and image build context. Mount the exact hash-registered adapter and snapshot read-only
at `/app/artifacts/multiseed-lora/seed-42` and `/app/artifacts/huggingface`. Mount writable,
durable audit storage at `/app/artifacts/audit`. Startup validates the registered files before the
readiness endpoint can pass.

The CUDA workload additionally requires an NVIDIA container runtime and a real GPU. A missing CUDA
runtime, incompatible build, or unavailable device makes startup non-ready; there is no mocked
selection or CPU fallback.

## Audit-store boundary

`AuditStore` is a minimal `append(event)` and `close()` protocol. The built-in implementation
retains the Module 9 metadata-only, allowlisted JSONL event and restrictive local permissions.
Deployments may inject a durable implementation through `create_deployment_app(...,
audit_store_factory=...)`; the factory must pass protocol validation before readiness.

Pluggability does not relax the event contract. Customer text, redacted text, tokens, names,
account data and message hashes remain prohibited. A real backend still needs retention,
encryption, integrity, access-control, failure-recovery and operational review.

## Gateway, Kubernetes and rollback templates

`deploy/kubernetes/router.yaml.template` provides a private `ClusterIP`, gateway-only ingress policy,
non-root/read-only container security, health probes, persistent model and audit mounts, rolling
updates and five retained revisions. `cuda-patch.yaml.template` changes the profile and requests one
real NVIDIA GPU.

Replace every `REPLACE_WITH_...` value, validate the resulting manifest in the target cluster and
bind the gateway assertion from a secret manager. These templates are not ready to apply unchanged.

Rollback replaces the complete immutable process or container revision; the running process never
hot-swaps model files. A container must receive both the current registered release ID and a
distinct previous immutable image reference before it can become ready. In Kubernetes, retain the
previous image and versioned model bundle together, then use the platform's rollout history and
rollback mechanism. Re-run readiness and synthetic safety checks after rollback.

## Verified native smoke result

The registered smoke script loaded the real hash-bound adapter on Apple M4 MPS, exercised liveness,
readiness, authentication, versioning, identifiers, one metadata-only audit append and graceful
shutdown.

| Observation | Recorded result |
|---|---:|
| Selected backend | Real MPS; no fallback |
| Startup | 1.2368 seconds |
| One synthetic route | 0.1194 seconds |
| Registered checks | 12/12 passed |
| Audit events | 1 |
| Input present in response or audit serialization | No |

This is a single in-process request on one Mac. It is not a load test, availability result,
container verification, production privacy assessment or production approval. Reproduce it on a
real MPS host with:

```bash
python scripts/run_deployment_smoke.py
pytest -q tests/test_deployment_config.py \
  tests/test_deployment_service.py \
  tests/test_deployment_artifacts.py
```
