# Module 10 Shadow API Boundary

## Current authorization

The FastAPI service is a local research interface for advisory routing. It binds to
`127.0.0.1`, requires a bearer token on `/v1/route`, disables OpenAPI documentation and CORS, and
accepts one bounded JSON field: `message`.

It is not approved for internet exposure, production customer data or automated banking actions.
The current policy can return only `human_review` or `security_queue`; those values are queue
recommendations, not executed actions.

## Request path

1. Reject an untrusted host, missing or invalid bearer token, oversized body, unexpected JSON field
   or invalid message.
2. Redact registered structured PII while the original message remains transient in memory.
3. Send only the redacted representation to the hash-bound offline seed-42 LoRA-RoBERTa adapter.
4. Apply the registered scalar temperature and compute seed 42's `max_probability` observation.
5. Apply the Module 9 deterministic shadow policy. Module 8 thresholds remain experimental and
   cannot authorize an automated suggestion.
6. Persist the allowlisted metadata-only audit event before returning the advisory response.

If redaction or model inference fails, the service produces a metadata-only failure event and
returns `human_review`. If the audit event cannot be persisted, the route response is rejected with
a generic service-unavailable error.

## Public response boundary

The response includes an opaque UUID4 request identifier, service and policy modes, processing
status, predicted intent when inference succeeds, advisory action and queue, ordered reason codes,
a redaction flag and a categorical uncertainty observation.

It excludes original message text, redacted message text, PII values, PII-type details, exact input
length and uncertainty score. Responses use `Cache-Control: no-store`.

## Security defaults

- Bearer tokens come only from `GOVERNED_BANKING_API_TOKEN`; no token value is committed.
- Comparison uses a constant-time primitive and tokens must contain at least 32 characters.
- Trusted hosts are limited to loopback names plus `testserver` for the in-process test client.
- Request bodies are capped at 8,192 bytes before FastAPI parses JSON.
- Unknown request fields are rejected and validation errors do not echo the submitted value.
- Security headers are added centrally; no CORS, cookies, uploads, outbound HTTP or file-serving
  endpoints are enabled.
- The server starts without debug mode, reload, proxy-header trust or access logging.
- Model and calibration resolution is offline and hash-bound to versioned evidence.

## Production gaps

A future production service needs a new configuration and review covering managed identity, token
rotation, TLS termination, proxy trust, rate limiting, centralized audit storage, retention,
encryption, integrity monitoring, availability objectives, multi-worker/MPS behaviour, operational
ownership and representative privacy and routing validation. None is implied by this local module.
