# Numerical checks and performance evidence

Numerical admission, timing detection and intervention usefulness are separate questions. Passing a finite check does not prove a kernel correct for every input. A performance improvement does not authorize a numerically invalid implementation.

## Numerical policies

| Policy | Contract |
|---|---|
| `exact` | Exact numeric equality in the tested domain; signed-zero bit patterns are not distinguished. |
| `allclose` | Application-declared `rtol`, `atol` and justification. The application owns the acceptable error. |
| `matmul_forward_bound`, `model_version: 2` | Conditional FP32 accumulation envelope with FP64 reference/absolute-product scale and final-output rounding allowance. Requires the declared arithmetic assumptions; not a universal tolerance or proof. |

All policies reject nonfinite results and incompatible references. Each variant receives two deterministic seeds across four finite patterns before measurement. Constant generators retain their stated domain. Input mutation is rejected. References are reviewed executable application code, not authenticated ground truth.

For the forward model, with `gamma(n,p)=n*2^-p/(1-n*2^-p)`, `g32=gamma(2K,24)`, `g64=gamma(2K,53)`, computed absolute-product sum S and FP64 reference R, the allowance is `(1+u_out)*(g32+g64)*S/(1-g64) + u_out*abs(R) + half_minimum_output_subnormal`. It assumes standard rounding and the declared accumulation length, without intermediate overflow/underflow or lossy flush-to-zero behavior. A broad envelope may admit an error unacceptable to a particular application; use an explicit application contract in that case.

Failures preserve residuals, samples and available provenance. `NUMERICAL_FAILURE` is not a "no regression" result. `EXECUTION_FAILURE` covers preparation/compiler/runtime failure and likewise supplies no correctness-implying recommendation.

## Measurement boundaries

The external adapter records fixed-buffer CUDA Graph device time separately from checked wrapper-plus-completion time. First launch can use existing caches and is not claimed as cold compilation. These measurements do not establish full application throughput. Do not compare graph replay against an allocating eager call as an equivalent execution boundary.

Repeated seeded randomized paired blocks compare baseline/candidate. The rule uses a declared practical threshold and guarded paired sign exceedance tests. Independent stationary blocks are assumptions: clock drift, thermal effects, host scheduling, shared-resource changes and correlated noise can still invalidate interpretation. Inconclusive timing is neither equivalence nor proof of no regression. No universal false-positive guarantee or general diagnostic accuracy is claimed.

Controlled reversions get separate valid measurements. Reports distinguish measured observations, controlled interventions, compiler evidence, model estimates, hypotheses and unavailable evidence. Registers/shared-memory metadata and PTX differences are compiler observations. Without suitable hardware evidence they do not prove occupancy, bandwidth or another physical bottleneck.

## Reduction and safe sharing

Accepted shape reductions must preserve numerical admission, the declared effect and a useful intervention where required. Fresh-seed confirmation precedes witness export. Greedy halving does not prove global minimality; preserving slowdown does not guarantee preserving a hardware cause. A confirmed unchanged capture is not a smaller reproducer.

The external budget bounds original/proposal workers; fresh confirmation has an additional bounded allowance. Compilation/checks and filesystem/report overhead must be considered when scheduling work. A standalone repeats the vendored procedure; it is independent of the installed package, not an independently designed measurement algorithm.

Artifacts may contain source, input descriptions, paths, hardware/software identifiers and compiler details. Review and sanitize them before sharing. Do not assume a generated bundle is safe to publish or execute merely because its checksums match.
