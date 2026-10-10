# Getting started

Install from a reviewed source checkout with Python 3.11 or newer:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install .
asterion --help
```

The base package uses PyYAML and does not install PyTorch, Triton, CUDA or a C++ compiler. These passive operations work without a GPU:

```sh
asterion validate examples/external_add.json
asterion reproduce examples/external_add.json --export-only --output standalone
```

`validate` checks the specification and source identity, not GPU numerical correctness. Export copies the declared source and standalone support without importing the kernel. The exported bundle needs compatible GPU dependencies when executed.

## Measure the injected example

Actual diagnosis requires Linux, a compatible CUDA driver, an NVIDIA SM80-or-newer GPU and the optional packages. The tested external combination is PyTorch 2.14.1 with CUDA 12.6 and Triton 3.8.0 on a full H100. Other compatible devices are not automatically validated.

```sh
python -m pip install '.[gpu]'
asterion diagnose examples/external_add.json --trusted-code --output diagnosis
asterion explain diagnosis/results.json --output report-copy
asterion reduce diagnosis/results.json --trusted-code --max-trials 6 --budget-seconds 120 --output reduction
```

The independently owned example compares the same addition kernel at BLOCK 1024 and BLOCK 32. This is an injected configuration contrast. It makes no claim about application acceleration or a production incident. You do not need a particular scheduler, institutional account or internal environment path.

Review imported Python and the application reference before passing `--trusted-code`. They run with your filesystem and network authority. Workers filter environment variables and own their child processes, but do not provide comprehensive isolation. Use an appropriate trusted environment without exposed credentials.

## Read and reproduce the result

Open `diagnosis/report.html` in a browser. The canonical `results.json` records the workload, target, numerical checks, measurements, interventions and limitations. `source/` preserves the declared source; compiler artifacts are captured when available. Reports are generated from saved evidence and can be regenerated without executing the module.

If `reduction/reduction.json` records `CONFIRMED_SMALLER_REPRODUCER` or `CONFIRMED_UNCHANGED_CAPTURE`, a confirmed bundle is available:

```sh
python reduction/witness/reproduce.py --trusted-code --output replay
```

The bundle does not import an installed Asterion package. It vendors the required MIT support and keeps the kernel's own license. Review all code before running it. A confirmed unchanged capture is useful for replay but is not a smaller reproducer.

## Common outcomes

| Outcome | What to do |
|---|---|
| Existing output directory is nonempty | Choose a new directory; previous evidence is preserved. |
| Source hash mismatch | Review the new source, then explicitly update the contract's SHA256. |
| Missing GPU dependency or unsupported target | Use the passive commands, or supply a compatible trusted GPU environment. |
| `NUMERICAL_FAILURE` | Inspect the per-variant checks and application reference. Do not treat it as a timing negative or silently relax the required tolerance. |
| `EXECUTION_FAILURE` | Inspect retained preparation/compiler/worker errors. No valid timing recommendation follows. |
| `REGRESSION_INCONCLUSIVE` | The declared practical effect was not established. This is not evidence of equivalence. |
| No confirmed witness | Inspect rejected trials or original admission; no successful reduction should be claimed. |

Exit status 3 marks external execution/numerical failure and specified reduction failures; timing inconclusiveness can still exit successfully. Read the saved status, not only the shell exit code. Reduction's nominal time budget excludes a separate bounded fresh-confirmation allowance and some filesystem/report overhead.

Next: [supply your own contract](triton-kernels.md), [understand measurements](measurement.md), or [use the API](interfaces.md).
