# ADR 0002: Additive cross-platform accelerator runtime

- Status: accepted
- Date: 2026-08-21

## Context

Modules 1–10 were developed and evaluated on one Apple Silicon Mac. Their configurations, reports
and implementation hashes form historical MPS evidence. Editing the registered runtime files would
make those reports unverifiable and would incorrectly present new code as the code that produced
old results.

The open-source project also needs a real NVIDIA CUDA path, an explicit CPU path and an automatic
selection policy. CUDA is not available on the development Mac and must not be simulated.

## Decision

### Version boundary

- Keep `src/governed_banking/device.py`, `src/governed_banking/inference.py` and all Module 1–10
  configurations and reports unchanged.
- Introduce a separate Module 11 accelerator runtime and portable inference evidence contract.
- Label Module 10 results `legacy_mps_evidence`; do not regenerate or reinterpret them as
  cross-platform results.

### Device selection

The accepted preferences are `cuda`, `mps`, `cpu` and `auto`.

- `auto` selects the first available backend in this fixed order: CUDA, MPS, CPU.
- Explicit `cuda` fails if `torch.cuda.is_available()` is false.
- Explicit `mps` fails if `torch.backends.mps.is_available()` is false.
- Explicit `cpu` always selects CPU.
- An explicit accelerator request never falls back to another device.
- CUDA support is unverified until the registered CUDA checks run on a real NVIDIA device.

### Runtime operations

- Seed Python, NumPy and PyTorch for every run.
- Seed every CUDA device when CUDA is selected.
- Seed MPS when MPS is selected.
- Dispatch synchronization and cache cleanup to the selected backend only.
- Do not enable unsupported-operation CPU fallback in the registered MPS profile.

### Runtime evidence

Every runtime report records:

- requested and selected backend;
- PyTorch, Python, operating-system and machine architecture versions;
- CUDA build version, cuDNN version and CUDA device count;
- accelerator name, device index and compute capability where available;
- CUDA total/free memory or the MPS recommended maximum working-set size;
- memory-field semantics so unified MPS memory is not described as dedicated VRAM;
- runtime-profile hash and implementation hashes; and
- whether the report was produced on real hardware.

Unavailable values are represented as `null`, not inferred or mocked.

### Cross-device parity

Parity uses the same hash-bound adapter, calibration temperature, label ordering and versioned
synthetic API fixture. It does not access the official BANKING77 test split or estimate model
accuracy.

The preregistered comparison gates are:

- identical checkpoint-file hashes;
- identical fixture, label-order and inference-configuration hashes;
- identical predicted intent for every case;
- maximum absolute probability delta no greater than `0.001`; and
- identical deterministic routing action for every case.

The reports retain full-precision probabilities for comparison but never retain input or redacted
message text. Failure is reported rather than hidden by relaxing the registered tolerance.

## Consequences

The Mac can implement and verify the CPU and MPS paths. Selecting CUDA on the Mac must fail. A
Google Colab notebook will run the same registered check on a real CUDA device after the Module 11
code is available from GitHub and the hash-bound adapter is supplied through private artifact
storage.

PyTorch does not guarantee bitwise-identical floating-point results across platforms. Module 11
therefore evaluates declared numerical tolerances and decision agreement instead of exact tensor
equality.
