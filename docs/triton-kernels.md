# Integrating an existing Triton kernel

The supported input is a reviewed, self-contained Python module with plain `triton.jit` entrypoints and an explicit `triton_case` version 1 contract. No Asterion IR rewrite is required. Asterion does not infer an application's reference, layout or invocation from arbitrary Python.

The complete working example consists of [external_add.py](../examples/external_add.py) and [external_add.json](../examples/external_add.json). Copy both when learning the interface. The selected kernel is ordinary masked vector addition:

```python
@triton.jit
def add_kernel(X, Y, Z, N, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(X + index, index < N, other=0)
    b = tl.load(Y + index, index < N, other=0)
    tl.store(Z + index, a + b, index < N)
```

The complete module also imports its dependencies and defines the application reference. Reference functions receive an input dictionary and dimension dictionary. Addition explicitly rounds its result to FP32; do not substitute a reference with different intended numerical semantics.

## Contract fields

| Field | Meaning in the complete example |
|---|---|
| `schema_version`, `artifact_type` | `1`, `triton_case`. Unknown/unsupported fields are rejected. |
| `module`, `source_sha256` | Relative reviewed `.py` path and exact SHA256 of its bytes. |
| `dimensions` | `N: 262149`; positive bounded dimensions used in expressions. |
| `tensors` | Contiguous FP32 inputs `x`, `y` and one output `z`, each shape `N`; roles are explicit. |
| `arguments` | Maps kernel arguments X/Y/Z to tensors and N to the dimension. |
| `parameters` | Declares legal constexpr/compiler-option domains: BLOCK 32/1024, four warps and explicit floating-point fusion setting. |
| `baseline`, `candidate` | Both select `add_kernel`; only BLOCK differs. Parameters cannot silently escape their declared domains. |
| `grid` | Bounded expression `ceildiv(N, BLOCK)` gives the launch grid. No arbitrary expression evaluation. |
| `reference` | Reviewed `reference` entrypoint and description of intended semantics. |
| `numerical` | Exact numeric equality with a justification specific to this one-addition operation. |
| `measurement` | 20 paired rounds, 100 launches per batch, deterministic seed and 10% practical threshold. |
| `origin` | Identifies this independently written MIT example as an injected contrast. It is not execution authorization. |

Inputs use deterministic normal generation by default. The optional `generator` tensor field can declare `ones`; it changes the tested domain. Numerical checks use two seeds and normal, zero, alternating and scaled-normal patterns; declared constant-one inputs remain constant rather than becoming independent distributions. Reference inputs are provided as trusted FP64 copies; the reference must explicitly implement the intended output rounding and return the supported reference representation.

Supported schema dtypes are float32, float16, bfloat16, int8 and int32, subject to actual device/compiler capability. Inputs and output must be contiguous and nonaliasing; there is exactly one output, at most four dimensions per tensor and a 2 GiB declared-storage cap. Reference and diagnostic workspace are additional. Hardware support is narrower than a schema declaration; see [limitations](limitations.md).

After editing your module, inspect it and calculate its identity:

```sh
python -c "import hashlib; print(hashlib.sha256(open('examples/external_add.py', 'rb').read()).hexdigest())"
asterion validate examples/external_add.json
```

Do not automatically approve changed source just to bypass a hash mismatch. Hashes check consistency, not authenticity. Imported JSON cannot grant the trusted-code flag.

## Interventions and reduction

The adapter tests reversions within the declared parameter space. A one-parameter source change can affect several downstream compiler decisions; its measured effect does not isolate one hardware mechanism. Different entrypoints/source configurations can yield a useful whole-package rollback, which is labeled as such. Arbitrary AST edits, hidden autotuners, arbitrary launch wrappers, relative helper-module imports, arbitrary strides/aliasing and multiple outputs are unsupported.

Optional constraints use the contract's bounded dimension/parameter expressions. The reducer halves declared dimensions, rejects illegal or ineffective trials and requires fresh confirmation. The storage objective counts declared input plus output bytes. It does not reduce arbitrary source graphs, prove global minimality, or guarantee preservation of the physical bottleneck. Compiler-version environment management and automatic TritonParse capture import are outside the public adapter.

Use [the CLI or installed Python API](interfaces.md) to run the contract. Numerical failure, execution failure and inconclusive timing are separate outcomes; see [measurement semantics](measurement.md).
