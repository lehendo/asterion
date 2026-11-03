# Asterion

Repeatable GPU regression experiments for supported Triton kernels.

A kernel gets slower after a configuration change. Is the difference repeatable? Do both versions still meet your numerical requirements? Which declared change reverses the slowdown?

Asterion runs that investigation from an explicit kernel contract. It saves numerical checks, randomized measurements, compiler provenance and controlled interventions, then attempts to export a smaller standalone reproducer. It works alongside Triton and existing profiling tools.

## Try the included example

From a source checkout, install the CPU package with Python 3.11 or newer:

```sh
python -m pip install .
asterion validate examples/external_add.json
asterion reproduce examples/external_add.json --export-only --output standalone
```

These commands validate the contract and export source without importing or executing its Triton module. The example is independently written vector addition with an **injected small-block configuration regression**, not a production incident.

For actual measurements, use Linux, a compatible NVIDIA CUDA driver and GPU, and the optional dependencies:

```sh
python -m pip install '.[gpu]'
asterion diagnose examples/external_add.json --trusted-code --output diagnosis
asterion explain diagnosis/results.json --output report-copy
asterion reduce diagnosis/results.json --trusted-code --max-trials 6 --budget-seconds 120 --output reduction
```

Open `diagnosis/report.html`. It shows numerical admission, the measured baseline/candidate contrast, tested reversions, generated compiler artifacts and the limits of each explanation. Use new output directories for each run.

If reduction confirms a witness, replay it without Asterion installed:

```sh
python reduction/witness/reproduce.py --trusted-code --output replay
```

Imported Python is **trusted executable code**. Review the module and reference before running it. Environment filtering and worker processes are not a security sandbox. See [getting started](docs/getting-started.md) for failed or inconclusive outcomes.

## What you get

- Baseline/candidate comparisons using repeated randomized paired measurements.
- Numerical checks under an explicit application contract, before timing.
- Experiments reverting declared parameters, with whole-package changes labeled separately.
- Source, launch settings, available compiler IR, PTX and compiled-binary identities.
- Local HTML reports and retained evidence, including failures and abstentions.
- Bounded shape reduction with fresh confirmation and standalone export.

Reduction is an attempt, not a promise. A slower configuration can become indistinguishable at a smaller shape. Numerical failure prevents a correctness-implying recommendation; inconclusive timing does not prove equivalence.

## Supported scope

The existing-kernel adapter accepts a reviewed, self-contained plain `triton.jit` module and a version 1 `triton_case` specification. You supply tensor metadata, deterministic inputs, grid, baseline/candidate settings and a numerical reference. It supports contiguous nonaliasing inputs and one output; arbitrary applications, hidden autotuners, host wrappers and automatic source rewriting are outside this interface.

GPU execution requires SM80 or newer; that minimum is not validation of every device. The tested external suite uses a full H100 with PyTorch 2.14.1, CUDA 12.6 and Triton 3.8.0. Earlier Asterion IR/runtime tests used a full RTX A6000. CPU analysis, contract validation, report regeneration and export need no GPU packages.

Asterion also retains its typed operation IR, verified scheduling transformations and correctness-aware runtime. Triton performs GPU machine compilation. The optional native component provides host functions and a legacy CUDA Driver launcher, not an independent device compiler.

## How it fits

[Triton](https://github.com/triton-lang/triton) compiles kernels. [TritonParse](https://github.com/meta-pytorch/tritonparse) provides compiler capture, inspection and reproduction workflows. [TileLens](https://github.com/Deep-Learning-Profiling-Tools/tilelens) provides visualization and analysis. Asterion focuses on declared, numerically gated regression experiments and confirmed witnesses. Use a hardware profiler when the question requires hardware counters; timing and compiler metadata alone do not establish a physical bottleneck.

This is an early tool with bounded empirical coverage. General speedups, novel search algorithms, broad diagnostic accuracy and developer-time savings have not been demonstrated.

## Documentation and feedback

- [Getting started](docs/getting-started.md)
- [Integrating a Triton kernel](docs/triton-kernels.md)
- [CLI and Python API](docs/interfaces.md)
- [Numerical and measurement semantics](docs/measurement.md)
- [Limitations](docs/limitations.md)
- [Contributing and reporting issues](docs/contributing.md)

Feedback on installation, unsupported kernel patterns, numerical discrepancies and reproducer reliability is welcome. Share only code and metadata you have permission to disclose.

Asterion is [MIT licensed](LICENSE). Optional dependencies retain their own terms; see [third-party notices](THIRD_PARTY_NOTICES.txt).
