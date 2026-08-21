# Continuous integration and supply-chain controls

## Status

Module 16 introduces reproducible CI evidence and preventive supply-chain controls. It does
not change the registered model, policy, calibration, uncertainty or runtime evidence. The
workflow is intentionally CPU-only: accelerator parity remains governed by Module 11 evidence.

## Enforced controls

The `quality-and-evidence` workflow provides:

- complete CPU tests on Ubuntu 24.04, macOS 15 and Windows 2025 with Python 3.12 and 3.13;
- repository linting plus prospective formatting for Python files changed by a commit or pull
  request;
- strict static typing over the privacy, policy, audit, API, deployment configuration and
  observability safety boundary;
- separate JUnit reports for privacy, policy, integration and remaining unit tests;
- registered per-module coverage floors in `configs/ci/coverage-gates.yaml`;
- source-distribution and wheel building, followed by isolated wheel installation;
- a real lightweight container build and restricted container smoke execution;
- structural validation of the CPU and CUDA deployment Dockerfiles;
- installed-environment vulnerability auditing, a CycloneDX SBOM and provider-aware secret
  detection; and
- metadata-only control-path benchmark evidence with no message text or message-derived hashes.

All external GitHub Actions are pinned to complete commit SHAs. Workflows use least-privilege
`GITHUB_TOKEN` permissions, disable checkout credential persistence, avoid
`pull_request_target`, do not consume repository secrets and cancel superseded executions.

## Historical formatting boundary

Some pre-Module 16 Python files and notebooks are hash-bound by published evidence. Bulk
formatting those files would invalidate the evidence chain without changing behaviour. CI
therefore lints the maintained source, scripts and tests globally, but applies the formatting
gate prospectively to Python files changed in the proposed commit range. Any future edit to a
historical file brings that file under the formatting gate and must be accompanied by regenerated
evidence where its hash is registered.

Static typing is currently enforced on nine governed boundary modules. Two historical service
composition modules are excluded because they contain pre-existing typing debt and are bound to
Module 14 or Module 15 evidence. This is an explicit transitional boundary, not a claim that the
entire repository is strictly typed.

## Coverage interpretation

Coverage floors are regression controls, not proof of correctness or safety. They are fixed at
or slightly below the observed Module 15 coverage for safety-critical modules so that a change
cannot silently reduce exercised behaviour. Percentages are compared without rounding. Raising a
floor requires tests and review; lowering one requires a documented risk acceptance.

## GitHub repository settings required

Committed files cannot activate every GitHub security control. A repository administrator must:

1. enable the dependency graph, Dependabot alerts and Dependabot security updates;
2. enable CodeQL default availability for the repository so the committed advanced workflow can
   publish results;
3. enable secret scanning and push protection, including protection for contributors;
4. protect `main`, require pull requests, require approval and require the Module 16 checks;
5. prevent force pushes and branch deletion on `main`;
6. restrict workflow permissions to read-only by default and prevent unapproved third-party
   actions; and
7. review uploaded JUnit, coverage, SBOM, audit and benchmark artifacts before release.

For a public repository, GitHub may enable secret scanning automatically. Push protection and
branch rules must still be verified in repository settings rather than inferred from a green
workflow.

## Known limitations

- The project declares compatible dependency ranges but does not yet publish platform-specific,
  hash-locked environment files. The SBOM describes the resolved CI environment for that run;
  it is not a universal lockfile.
- The provider-aware scan disables generic entropy and keyword detectors because immutable
  research evidence contains many checksums and test-only token labels. GitHub secret scanning
  and push protection remain necessary compensating controls.
- The lightweight CI container verifies the package and container contract, not a model-serving
  image. Deployment images require organisation-approved digest-pinned CPU or CUDA bases and
  separately governed model artifacts.
- Benchmark timings are informational and runner-dependent. Module 16 applies no latency
  promotion or production-SLO gate.
- Local Docker execution was not available on the development Mac; the actual container build is
  delegated to the Linux GitHub-hosted runner.

## Evidence handling

Generated artifacts expire after 14 or 30 days. They must not be treated as permanent release
evidence. A release process should download required artifacts, verify the workflow commit SHA,
record checksums and attach the approved evidence to the corresponding versioned release.
