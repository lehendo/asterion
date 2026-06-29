#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ABI version 2. All pointers are borrowed for call duration; no ownership transfer.
uint32_t asterion_abi_version(void);
// fields: block, warps, rows, tile_m, tile_n, tile_k, unroll, fusion, schema_version, row_tile.
// operation: pointwise=0, sum=1, max=2, softmax=3, gemm=4. Returns 0 if valid.
int asterion_verify_schedule(uint32_t operation, const uint64_t* shape, size_t rank,
                            const uint64_t* fields, size_t field_count);
// Output is written only on success; returns nonzero for invalid/nonfinite input.
int asterion_cvar(const double* values, const double* weights, size_t count,
                  double quantile, double* output);
int asterion_objective(const double* ratios, const double* weights, size_t count,
                       double risk, double quantile, double* output);

#ifdef __cplusplus
}
#endif
