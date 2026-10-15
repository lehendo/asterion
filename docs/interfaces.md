# CLI and Python API

The installed `asterion` command exposes passive and executable workflows. Use `asterion COMMAND --help` for exact arguments. Commands operate on declared supported formats, not arbitrary applications.

| Command | Input and behavior |
|---|---|
| `validate CASE` | Passive external contract/source identity, native regression specification or portfolio validation. No GPU correctness test. |
| `diagnose CASE --output DIR` | Execute a supported comparison; external Python additionally requires `--trusted-code`. |
| `explain RESULTS --output DIR` | Regenerate a saved diagnosis/reduction or analyzer report without executing kernels. |
| `reduce CASE --output DIR` | Try constrained shape reduction; external Python requires `--trusted-code`. Accepts `--max-trials` and `--budget-seconds`. |
| `reproduce CASE --export-only --output DIR` | Passive standalone export. Omitting `--export-only` runs the checked diagnosis workflow instead. |
| `analyze TRACE --output DIR` | CPU analysis of supported version 1 workload/trace data. Unmeasured specs remain unmeasured. |
| `compare BASELINE CANDIDATE --output DIR` | Saved trace/analyzer comparison, with optional variant names; not arbitrary external diagnosis pairs. |
| `doctor` | Capture environment details. Review paths, device identities and host metadata before sharing output. |

The CLI also retains native-IR `benchmark`, `optimize`, `certify` and `cache stats`. `optimize` is the existing training-only portfolio path, distinct from diagnostic reversions. No risk-aware search advantage is established. Certificates cover explicit empirical domains and are not universal proofs.

## Existing-kernel API

```python
from asterion import diagnose, export_reproducer, reduce_regression

# Passive; no Triton module import or GPU execution.
export_reproducer("examples/external_add.json", output="standalone")

# Execute only after reviewing source and reference in a trusted environment.
result = diagnose(
    "examples/external_add.json", output="diagnosis", trusted_code=True
)
reduced = reduce_regression(
    "diagnosis/results.json", output="reduction", trusted_code=True,
    max_trials=6, budget_seconds=120,
)
```

`spec` can be a path or supported specification object. In-memory external contracts resolve the module relative to the current working directory; a specification path resolves it relative to that file. Paths are simplest for this interface. External execution authorization comes from the caller, never from input files. Outputs must be new or empty. Invalid or unsupported contracts raise `asterion.io.ValidationError`; saved GPU execution failures are distinct result statuses.

The returned external diagnosis includes `status`, `numerical_status`, `comparison`, `variants`, `recommendations`, compiler identities and raw samples. A recommendation is evidence at the stated boundary, not automatic deployment. Inspect admission and result status before interpreting timings. Reduction can return a confirmed smaller witness, a confirmed unchanged capture or failure; do not assume the witness directory exists.

## Native IR and runtime

The existing operation interface remains available:

```python
from asterion.backend import Kernel
from asterion.ir import Schedule, make_program, transform

program = make_program("softmax", (113, 577))
schedule, trail = transform(program, Schedule(), rows=4, row_tile=1024)
kernel = Kernel(program, schedule)
# y = kernel(x)  # caller supplies supported contiguous finite FP32 CUDA input
```

Native operations cover pointwise expressions, row sum/max, stable softmax and tiled FP32 GEMM. `asterion.runtime.Runtime` uses portfolio metadata and empirical numerical authorization, with a conservative fallback. `prepare(*inputs).launch()` reuses bound output storage. Callers preserve storage/module lifetimes, establish cross-stream dependencies and serialize access to reused outputs; graph replay bypasses Python checks. No autograd or arbitrary layouts are promised.

The optional C++ Driver launcher supports the tested legacy Triton 3.3.1 ABI, rejecting unsupported modern configurations. The default current GPU dependency set is Triton 3.8.0. Host launch support is not an independent GPU compiler.
