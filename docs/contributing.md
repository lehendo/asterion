# Contributing and reporting problems

Useful early feedback includes installation failures, unsupported kernel patterns, numerical discrepancies, measurement instability and whether a standalone witness reproduces on another compatible environment. Proposals for capture adapters or simpler contract preparation are welcome. General performance or adoption claims require evidence.

Before sharing a report, remove credentials, private paths/accounts, device UUIDs, proprietary source and sensitive inputs. Share only material you own or have permission to disclose. A generated artifact is not automatically suitable for publication. A small independently written example is preferable to disclosing private workload data.

For a useful issue, include:

- The command, package/Python versions and operating system.
- GPU model, compute capability and CUDA/PyTorch/Triton versions when relevant; avoid unique hardware identifiers.
- A sanitized minimal module/contract when redistribution is permitted.
- The expected behavior, actual result status and numerical policy.
- The measurement boundary, repeatability and dispersion for performance issues.
- Whether passive validation, diagnosis, reduction or standalone replay failed.

Do not paste full environment dumps or unreviewed raw logs. If code cannot be shared, describe the unsupported pattern without revealing it. Never include credentials in an issue or executable reproducer. Imported modules/references and bundles are trusted code; review them before running other contributors' examples.

## Development checks

Install the optional development dependencies from a source checkout:

```sh
python -m pip install '.[dev]'
python -m ruff check python tests examples tools setup.py
python -m ruff format --check python tests examples tools setup.py
python -m pytest tests --ignore=tests/gpu --ignore=tests/native -q
python tools/check_public_tree.py
python -m build
python tools/check_public_tree.py --archives dist
```

Optional native host tests need CMake 3.24+, a C++20 compiler and the pinned Catch2 dependency fetched by CMake:

```sh
cmake --preset host
cmake --build build/host -j2
ctest --test-dir build/host --output-on-failure
ASTERION_NATIVE_LIBRARY="$PWD/build/host/libasterion_native.so" python -m pytest tests/native -q
```

GPU tests run only in an authorized compatible CUDA environment; CPU CI does not claim to run them. Make small changes, preserve numerical contracts and negative results, and include focused regression tests for genuine fixes. Do not tune an example merely to force a favorable result.

`public-files.txt` is the explicit prospective source allowlist. New public files need deliberate inclusion there, in the scoped ignore exceptions and in the source-distribution manifest where appropriate. Keep internal notes, raw results and local infrastructure outside the public set. The helper can export a clean non-Git preview for validation; it never stages files or changes Git history.
